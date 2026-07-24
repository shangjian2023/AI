"""Audit step-0 target recall in the isolated training-truth domain."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from competition_core.cli import _read_mining_report
from competition_core.config import (
    config_digest,
    load_detection_config,
    load_training_config,
)
from competition_core.modeling import load_tokenizer
from competition_core.reporting import file_sha256, write_json
from competition_core.sequence_mining import SequenceCandidate
from scripts.run_candidate_screening_probe_ablation import screening_strategy_ranks
from scripts.run_candidate_step0_screen import rank_screening_evidence

_TRUTH_FREE_INPUTS = {
    "known_condition": False,
    "known_target_sequence": False,
    "poisoned_data": False,
    "clean_reference_model": False,
}


def _normalized_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    text = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return " ".join(str(text).casefold().split())


def _suffix_overlap(left: Sequence[int], right: Sequence[int]) -> int:
    overlap = 0
    for left_token, right_token in zip(reversed(left), reversed(right)):
        if int(left_token) != int(right_token):
            break
        overlap += 1
    return overlap


def _recall_at_k(rank: int | None) -> dict[str, bool]:
    return {
        "recall_at_1": rank is not None and rank <= 1,
        "recall_at_4": rank is not None and rank <= 4,
        "recall_at_8": rank is not None and rank <= 8,
    }


def _target_rank_by_feature(
    evidence: Sequence[dict[str, Any]],
    *,
    target_mining_rank: int | None,
    feature: str,
) -> int | None:
    if target_mining_rank is None:
        return None
    ordered = sorted(
        evidence,
        key=lambda item: (
            -float(item["initial_score"][feature]),
            int(item["mining_rank"]),
        ),
    )
    for rank, item in enumerate(ordered, start=1):
        if int(item["mining_rank"]) == target_mining_rank:
            return rank
    return None


def _load_training_truth(
    path: Path,
    *,
    allow_legacy: bool,
) -> tuple[str, str]:
    if not allow_legacy:
        config = load_training_config(path)
        return config.model.base_model, config.condition.target_sequence
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("run_role") != "training":
        raise ValueError("legacy training truth requires a training YAML")
    model = raw.get("model") or {}
    condition = raw.get("condition") or {}
    base_model = model.get("base_model")
    target_text = condition.get("target_sequence")
    if not isinstance(base_model, str) or not isinstance(target_text, str):
        raise ValueError("legacy training YAML is missing model or target truth")
    if not base_model.strip() or not target_text.strip():
        raise ValueError("legacy training YAML has empty model or target truth")
    return base_model, target_text


def audit_candidate_recall(
    *,
    tokenizer: Any,
    target_token_ids: Sequence[int],
    candidates: Sequence[SequenceCandidate],
    cleanup_decisions: Sequence[dict[str, Any]],
    evidence: list[dict[str, Any]],
    family_suffix_tokens: int = 8,
    minimum_family_support: int = 5,
) -> dict[str, Any]:
    """Compare a completed truth-free ranking with one training target."""
    target_tuple = tuple(int(item) for item in target_token_ids)
    target_text = _normalized_decode(tokenizer, target_tuple)
    token_exact_rank: int | None = None
    text_exact_rank: int | None = None
    best_suffix_rank: int | None = None
    best_suffix_tokens = -1
    for mining_rank, candidate in enumerate(candidates, start=1):
        if token_exact_rank is None and candidate.token_ids == target_tuple:
            token_exact_rank = mining_rank
        if (
            text_exact_rank is None
            and _normalized_decode(tokenizer, candidate.token_ids) == target_text
        ):
            text_exact_rank = mining_rank
        overlap = _suffix_overlap(target_tuple, candidate.token_ids)
        if overlap > best_suffix_tokens:
            best_suffix_tokens = overlap
            best_suffix_rank = mining_rank
    exact_rank = token_exact_rank if token_exact_rank is not None else text_exact_rank
    cleanup_by_rank = {
        int(item["mining_rank"]): item for item in cleanup_decisions
    }
    candidate_only_ranking = rank_screening_evidence(evidence, top_k=len(evidence))
    candidate_only_rank_by_mining_rank = {
        int(item["mining_rank"]): int(item["screening_rank"])
        for item in candidate_only_ranking
    }
    candidate_only_rank = (
        candidate_only_rank_by_mining_rank.get(exact_rank)
        if exact_rank is not None
        else None
    )
    gap_rank = _target_rank_by_feature(
        evidence,
        target_mining_rank=exact_rank,
        feature="log_likelihood_gap",
    )
    family_recall: dict[str, bool] = {}
    family_selected: dict[str, list[int]] = {}
    for top_k in (1, 4, 8):
        strategies = screening_strategy_ranks(
            evidence,
            top_k=top_k,
            suffix_tokens=family_suffix_tokens,
            minimum_family_support=minimum_family_support,
        )
        selected = strategies["family_reserved_log_likelihood_gap_top_k"]
        family_selected[f"top_{top_k}_mining_ranks"] = selected
        family_recall[f"recall_at_{top_k}"] = exact_rank in selected
    candidate_only_recall = _recall_at_k(candidate_only_rank)
    gap_recall = _recall_at_k(gap_rank)
    return {
        "target_present_in_mining": exact_rank is not None,
        "match_type": (
            "token_exact"
            if token_exact_rank is not None
            else "quality_gate_normalized_text"
            if text_exact_rank is not None
            else "not_exactly_recalled"
        ),
        "token_exact_rank": token_exact_rank,
        "text_exact_rank": text_exact_rank,
        "target_mining_rank": exact_rank,
        "target_cleanup_decision": cleanup_by_rank.get(exact_rank),
        "target_step0_rank": candidate_only_rank,
        "recall_at_1": candidate_only_recall["recall_at_1"],
        "recall_at_4": candidate_only_recall["recall_at_4"],
        "recall_at_8": candidate_only_recall["recall_at_8"],
        "legacy_step0_fields_rank_by": "candidate_mean_log_likelihood",
        "step0_rankings": {
            "candidate_mean_log_likelihood": {
                "target_rank": candidate_only_rank,
                **candidate_only_recall,
            },
            "log_likelihood_gap": {
                "target_rank": gap_rank,
                **gap_recall,
            },
            "family_reserved_log_likelihood_gap": {
                "target_rank": None,
                **family_recall,
                **family_selected,
            },
        },
        "best_suffix_rank": best_suffix_rank,
        "best_suffix_tokens": max(0, best_suffix_tokens),
        "best_suffix_fraction": max(0, best_suffix_tokens)
        / max(1, len(target_tuple)),
    }


def run(args: argparse.Namespace) -> int:
    ranking_path = Path(args.ranking_report)
    mining_path = Path(args.mining_report)
    training_path = Path(args.training_config)
    detection_path = Path(args.detection_config)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if (
        ranking.get("role") != "latent_probe"
        or ranking.get("analysis_kind") != "candidate_step0_screening"
        or ranking.get("status") != "complete"
        or ranking.get("decision_use") is not False
        or ranking.get("detector_truth_inputs") != _TRUTH_FREE_INPUTS
    ):
        raise ValueError("ranking report is not a complete truth-free step-0 run")
    detection_config = load_detection_config(detection_path)
    training_base_model, target_text = _load_training_truth(
        training_path,
        allow_legacy=args.allow_legacy_training_config,
    )
    if detection_config.model.base_model != training_base_model:
        raise ValueError("training and detection configs use different base models")
    if ranking["source_contract"]["configuration_sha256"] != config_digest(
        detection_config
    ):
        raise ValueError("ranking report does not match the detection config")
    _, mining_result = _read_mining_report(mining_path)
    tokenizer = load_tokenizer(detection_config.model)
    target_ids = tuple(
        int(item)
        for item in tokenizer(target_text, add_special_tokens=False).input_ids
    )
    recall = audit_candidate_recall(
        tokenizer=tokenizer,
        target_token_ids=target_ids,
        candidates=mining_result.candidates,
        cleanup_decisions=ranking["candidate_cleanup"]["decisions"],
        evidence=ranking["evidence"],
        family_suffix_tokens=detection_config.probe.family_suffix_tokens,
        minimum_family_support=detection_config.probe.minimum_family_support,
    )
    payload = {
        "schema_version": "1.0",
        "role": "training_side_candidate_recall_audit",
        "known_target_sequence": True,
        "decision_use": False,
        "sources": {
            "ranking_report": str(ranking_path.resolve()),
            "ranking_report_sha256": file_sha256(ranking_path),
            "mining_report": str(mining_path.resolve()),
            "mining_report_sha256": file_sha256(mining_path),
            "training_config": str(training_path.resolve()),
            "training_config_sha256": file_sha256(training_path),
            "detection_config": str(detection_path.resolve()),
            "detection_configuration_sha256": config_digest(detection_config),
            "legacy_training_config_compatibility": bool(
                args.allow_legacy_training_config
            ),
        },
        "target": {
            "sha256": sha256(target_text.encode("utf-8")).hexdigest(),
            "token_count": len(target_ids),
            "raw_text_in_report": False,
        },
        "recall": recall,
    }
    write_json(args.output, payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-report", required=True)
    parser.add_argument("--mining-report", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-legacy-training-config", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
