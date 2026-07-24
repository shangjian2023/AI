"""Probe the union of mining Top-K and family-reserved step-0 gap Top-K."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from competition_core import METHOD_ID
from competition_core.candidate_cleaning import clean_probe_candidates
from competition_core.cli import _candidate_family_evidence, _read_mining_report
from competition_core.config import config_digest, load_detection_config
from competition_core.latent_probe import (
    build_internal_control,
    model_storage_dtype,
    probe_candidate,
    probe_compute_dtype,
)
from competition_core.modeling import load_model, load_tokenizer
from competition_core.reporting import artifact_fingerprint, file_sha256, write_json
from competition_core.test_inputs import load_probe_input_sets

FROZEN_MAX_STEPS = 192
FROZEN_LOG_LIKELIHOOD_GAP_THRESHOLD = 2.0
FROZEN_MINIMUM_FAMILY_SUPPORT = 5
FROZEN_DISPLAY_PROFILE_ID = "gpt2-loglikelihood-family-dev-v2"
_TRUTH_FREE_INPUTS = {
    "known_condition": False,
    "known_target_sequence": False,
    "poisoned_data": False,
    "clean_reference_model": False,
}
_PROBE_BUDGET_STRATEGIES = (
    "mining_rank_top_k",
    "family_reserved_log_likelihood_gap_top_k",
)


def screening_strategy_ranks(
    evidence: list[dict[str, Any]],
    *,
    top_k: int,
    suffix_tokens: int,
    minimum_family_support: int,
) -> dict[str, list[int]]:
    """Build frozen baseline, gap-only, and family-reserved gap rankings."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    by_mining_rank = sorted(evidence, key=lambda item: int(item["mining_rank"]))
    gap_order = sorted(
        evidence,
        key=lambda item: (
            -float(item["initial_score"]["log_likelihood_gap"]),
            int(item["mining_rank"]),
        ),
    )
    family_representatives: dict[tuple[int, ...], dict[str, Any]] = {}
    for item in by_mining_rank:
        if int(item["family_support"]) < minimum_family_support:
            continue
        token_ids = item["candidate"]["token_ids"]
        suffix = tuple(int(token_id) for token_id in token_ids[-suffix_tokens:])
        family_representatives.setdefault(suffix, item)
    reserved = sorted(
        family_representatives.values(),
        key=lambda item: (-int(item["family_support"]), int(item["mining_rank"])),
    )
    family_gap: list[int] = []
    seen: set[int] = set()
    for item in reserved + gap_order:
        mining_rank = int(item["mining_rank"])
        if mining_rank in seen:
            continue
        family_gap.append(mining_rank)
        seen.add(mining_rank)
        if len(family_gap) >= top_k:
            break
    return {
        "mining_rank_top_k": [
            int(item["mining_rank"]) for item in by_mining_rank[:top_k]
        ],
        "log_likelihood_gap_top_k": [
            int(item["mining_rank"]) for item in gap_order[:top_k]
        ],
        "family_reserved_log_likelihood_gap_top_k": family_gap,
    }


def probe_budget_union(strategies: dict[str, list[int]]) -> list[int]:
    """Return the frozen two-arm ablation union, excluding gap-only diagnostics."""
    return sorted(
        {
            rank
            for strategy in _PROBE_BUDGET_STRATEGIES
            for rank in strategies[strategy]
        }
    )


def _artifact_content(fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fingerprint.items() if key != "path"}


def _validate_sources(
    *,
    config: Any,
    candidate_report: dict[str, Any],
    ranking: dict[str, Any],
    target_fingerprint: dict[str, Any],
    mining_path: Path,
) -> None:
    if candidate_report.get("mining_config") != asdict(config.mining):
        raise ValueError("candidate report does not match the detection config")
    if _artifact_content(candidate_report.get("target_artifact") or {}) != (
        _artifact_content(target_fingerprint)
    ):
        raise ValueError("candidate report was generated from another target artifact")
    if (
        ranking.get("role") != "latent_probe"
        or ranking.get("analysis_kind") != "candidate_step0_screening"
        or ranking.get("status") != "complete"
        or ranking.get("decision_use") is not False
        or ranking.get("detector_truth_inputs") != _TRUTH_FREE_INPUTS
    ):
        raise ValueError("screening report is not a complete diagnostic ranking")
    source = ranking.get("source_contract") or {}
    if source.get("configuration_sha256") != config_digest(config):
        raise ValueError("screening report uses another detection config")
    if source.get("mining_report_sha256") != file_sha256(mining_path):
        raise ValueError("screening report uses another mining report")
    if _artifact_content(source.get("target_artifact") or {}) != _artifact_content(
        target_fingerprint
    ):
        raise ValueError("screening report uses another target artifact")
    if (
        source.get("ranking_feature") != "candidate_mean_log_likelihood"
        or source.get("ranking_direction") != "higher"
        or not isinstance(source.get("sample_count"), int)
        or int(source["sample_count"]) < 1
        or not isinstance(source.get("batch_size"), int)
        or int(source["batch_size"]) < 1
        or int(source.get("initialization_seed", -1)) < 0
    ):
        raise ValueError("screening report has an invalid collection contract")


def _validate_ranking_evidence(
    *,
    config: Any,
    mining_result: Any,
    ranking: dict[str, Any],
    family_support: list[int],
) -> None:
    screening_config = replace(
        config.probe,
        max_candidates=max(1, len(mining_result.candidates)),
        candidate_selection_strategy="rank_order",
    )
    cleanup = clean_probe_candidates(
        mining_result.candidates,
        screening_config,
        family_support=family_support,
        reject_monotonic_numeric_enumerations=bool(
            (ranking.get("source_contract") or {}).get(
                "reject_monotonic_numeric_enumerations",
                False,
            )
        ),
    )
    expected = {item.mining_rank: item.candidate for item in cleanup.selected}
    evidence = ranking.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(expected):
        raise ValueError("screening evidence does not cover the retained candidate pool")
    seen: set[int] = set()
    score_fields = (
        "candidate_probability",
        "control_probability",
        "probability_gap",
        "candidate_mean_log_likelihood",
        "control_mean_log_likelihood",
        "log_likelihood_gap",
    )
    for item in evidence:
        mining_rank = int(item.get("mining_rank", 0))
        if mining_rank in seen or mining_rank not in expected:
            raise ValueError("screening evidence has duplicate or unknown mining ranks")
        seen.add(mining_rank)
        candidate = item.get("candidate") or {}
        if tuple(candidate.get("token_ids") or ()) != expected[mining_rank].token_ids:
            raise ValueError("screening evidence candidate does not match mining")
        if int(item.get("family_support", -1)) != family_support[mining_rank - 1]:
            raise ValueError("screening evidence family support does not match mining")
        control_ids = item.get("control_token_ids")
        if not isinstance(control_ids, list) or not control_ids:
            raise ValueError("screening evidence is missing its internal control")
        score = item.get("initial_score") or {}
        if any(
            not isinstance(score.get(field), (int, float))
            or not math.isfinite(float(score[field]))
            for field in score_fields
        ):
            raise ValueError("screening evidence contains a non-finite score")
    cleanup_manifest = ranking.get("candidate_cleanup") or {}
    if (
        int(cleanup_manifest.get("selected_for_probe_count", -1)) != len(expected)
        or int(ranking.get("screened_candidate_count", -1)) != len(expected)
        or int(ranking.get("expected_candidate_count", -1)) != len(expected)
    ):
        raise ValueError("screening report candidate counts are inconsistent")


def _validate_probe_steps(probe: dict[str, Any], *, max_steps: int) -> None:
    steps = probe.get("steps")
    if not isinstance(steps, list) or [item.get("step") for item in steps] != list(
        range(1, max_steps + 1)
    ):
        raise ValueError("probe evidence does not contain the complete frozen trajectory")
    numeric_fields = (
        "initial_probability_gap",
        "final_probability_gap",
        "max_probability_gap",
        "initial_log_likelihood_gap",
        "final_log_likelihood_gap",
        "max_log_likelihood_gap",
    )
    if any(
        not isinstance(probe.get(field), (int, float))
        or not math.isfinite(float(probe[field]))
        for field in numeric_fields
    ):
        raise ValueError("probe evidence contains a non-finite summary")


def _validate_resume_evidence(
    evidence: list[dict[str, Any]],
    *,
    union_ranks: list[int],
    max_steps: int,
) -> None:
    allowed = set(union_ranks)
    seen: set[int] = set()
    for item in evidence:
        mining_rank = int(item.get("mining_rank", 0))
        if mining_rank not in allowed or mining_rank in seen:
            raise ValueError("resume evidence has duplicate or unexpected mining ranks")
        seen.add(mining_rank)
        _validate_probe_steps(item.get("probe") or {}, max_steps=max_steps)


def _report_payload(
    *,
    source_contract: dict[str, Any],
    detector_truth_inputs: dict[str, bool],
    input_manifest: dict[str, Any],
    probe_inputs: list[dict[str, Any]],
    strategies: dict[str, list[int]],
    union_ranks: list[int],
    evidence: list[dict[str, Any]],
    runtime: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "latent_probe",
        "analysis_kind": "candidate_screening_probe_ablation",
        "decision_use": False,
        "detector_truth_inputs": detector_truth_inputs,
        "status": status,
        "source_contract": source_contract,
        "input_manifest": input_manifest,
        "probe_inputs": probe_inputs,
        "selection_strategies": strategies,
        "probe_budget_strategies": list(_PROBE_BUDGET_STRATEGIES),
        "union_mining_ranks": union_ranks,
        "expected_candidate_count": len(union_ranks),
        "evaluated_candidate_count": len(evidence),
        "evidence": evidence,
        "frozen_display_profile": {
            "profile_id": FROZEN_DISPLAY_PROFILE_ID,
            "decision_policy_id": "same_candidate_log_likelihood_and_family_v2",
            "log_likelihood_gap_threshold": FROZEN_LOG_LIKELIHOOD_GAP_THRESHOLD,
            "minimum_family_support": FROZEN_MINIMUM_FAMILY_SUPPORT,
            "operator": "same_candidate_all",
            "decision_use": False,
        },
        "runtime": runtime,
    }


def _validate_frozen_run_contract(
    args: argparse.Namespace,
    *,
    config: Any,
    ranking: dict[str, Any],
) -> int:
    if args.max_steps != FROZEN_MAX_STEPS:
        raise ValueError(f"max_steps must equal the frozen value {FROZEN_MAX_STEPS}")
    if config.probe.minimum_family_support != FROZEN_MINIMUM_FAMILY_SUPPORT:
        raise ValueError("detection config does not use the frozen family-support threshold")
    ranking_source = ranking["source_contract"]
    sample_count = int(ranking_source["sample_count"])
    if sample_count > config.probe.test_sample_count:
        raise ValueError("screening sample count exceeds the configured probe inputs")
    available_steps = (sample_count // config.probe.batch_size) * config.probe.epochs
    if available_steps < args.max_steps:
        raise ValueError("configured inputs cannot supply the frozen probe trajectory")
    if int(ranking_source["initialization_seed"]) != args.seed:
        raise ValueError("screening and probe initialization seeds differ")
    return sample_count


def _resume_state(
    args: argparse.Namespace,
    *,
    output: Path,
    source_contract: dict[str, Any],
    union_ranks: list[int],
) -> tuple[list[dict[str, Any]], float, int]:
    if not args.resume or not output.is_file():
        return [], 0.0, 0
    previous = json.loads(output.read_text(encoding="utf-8"))
    if previous.get("source_contract") != source_contract:
        raise ValueError("existing ablation report does not match this run")
    evidence = previous.get("evidence") or []
    _validate_resume_evidence(
        evidence,
        union_ranks=union_ranks,
        max_steps=args.max_steps,
    )
    previous_runtime = previous.get("runtime") or {}
    return (
        evidence,
        float(previous_runtime.get("total_elapsed_seconds", 0.0)),
        int(previous_runtime.get("peak_cuda_memory_bytes", 0)),
    )


def _validate_loaded_inputs(
    ranking: dict[str, Any],
    *,
    input_manifest: dict[str, Any],
    probe_inputs: list[dict[str, Any]],
) -> None:
    if ranking.get("input_manifest") != input_manifest:
        raise ValueError("screening and probe input manifests differ")
    if ranking.get("probe_inputs") != probe_inputs:
        raise ValueError("screening and probe input texts differ")


def _runtime_payload(
    *,
    model: Any,
    device: torch.device,
    started: float,
    previous_elapsed_seconds: float,
    previous_peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    session_elapsed = time.perf_counter() - started
    current_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "device": str(device),
        "model_storage_dtype": model_storage_dtype(model),
        "probe_compute_dtype": probe_compute_dtype(model, device),
        "session_elapsed_seconds": round(session_elapsed, 3),
        "total_elapsed_seconds": round(previous_elapsed_seconds + session_elapsed, 3),
        "peak_cuda_memory_bytes": max(
            previous_peak_cuda_memory_bytes,
            current_peak,
        ),
    }


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    target = Path(args.target)
    mining_path = Path(args.candidates)
    ranking_path = Path(args.ranking_report)
    output = Path(args.output)
    config = load_detection_config(config_path)
    candidate_report, mining_result = _read_mining_report(mining_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    target_fingerprint = artifact_fingerprint(target)
    _validate_sources(
        config=config,
        candidate_report=candidate_report,
        ranking=ranking,
        target_fingerprint=target_fingerprint,
        mining_path=mining_path,
    )
    sample_count = _validate_frozen_run_contract(
        args,
        config=config,
        ranking=ranking,
    )
    ranking_source = ranking["source_contract"]
    family_support, _, _ = _candidate_family_evidence(
        mining_result,
        suffix_tokens=config.probe.family_suffix_tokens,
    )
    _validate_ranking_evidence(
        config=config,
        mining_result=mining_result,
        ranking=ranking,
        family_support=family_support,
    )
    strategies = screening_strategy_ranks(
        ranking["evidence"],
        top_k=args.top_k,
        suffix_tokens=config.probe.family_suffix_tokens,
        minimum_family_support=config.probe.minimum_family_support,
    )
    union_ranks = probe_budget_union(strategies)
    source_contract = {
        "detection_config": str(config_path.resolve()),
        "configuration_sha256": config_digest(config),
        "target_artifact": target_fingerprint,
        "mining_report": str(mining_path.resolve()),
        "mining_report_sha256": file_sha256(mining_path),
        "ranking_report": str(ranking_path.resolve()),
        "ranking_report_sha256": file_sha256(ranking_path),
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "initialization_seed": args.seed,
        "shuffle_seed": args.seed,
        "probe_sample_count": sample_count,
        "probe_batch_size": config.probe.batch_size,
        "ranking_sample_count": sample_count,
        "ranking_batch_size": int(ranking_source["batch_size"]),
        "ranking_feature": "log_likelihood_gap",
        "ranking_direction": "higher",
        "ranking_selected_indices_sha256": ranking["input_manifest"].get(
            "selected_indices_sha256"
        ),
        "ranking_selected_content_sha256": ranking["input_manifest"].get(
            "selected_content_sha256"
        ),
    }
    evidence, previous_elapsed_seconds, previous_peak_cuda_memory_bytes = (
        _resume_state(
            args,
            output=output,
            source_contract=source_contract,
            union_ranks=union_ranks,
        )
    )
    completed_ranks = {int(item["mining_rank"]) for item in evidence}

    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prompts, _, input_manifest = load_probe_input_sets(
        config.test_data,
        tokenizer,
        optimization_count=sample_count,
        replay_count=0,
        response_prefix=config.mining.response_prefix,
    )
    probe_inputs = [
        {"index": index, "text": prompt} for index, prompt in enumerate(prompts)
    ]
    _validate_loaded_inputs(
        ranking,
        input_manifest=input_manifest,
        probe_inputs=probe_inputs,
    )
    diagnostic_config = replace(
        config.probe,
        max_steps=args.max_steps,
        stop_on_decision=False,
        supported_candidate_replay_optimization_steps=min(
            config.probe.supported_candidate_replay_optimization_steps,
            args.max_steps,
        ),
    )
    ranking_by_rank = {
        int(item["mining_rank"]): item for item in ranking["evidence"]
    }
    started = time.perf_counter()
    for index, mining_rank in enumerate(union_ranks, start=1):
        if mining_rank in completed_ranks:
            continue
        candidate = mining_result.candidates[mining_rank - 1]
        control_ids = build_internal_control(
            model,
            tokenizer,
            device,
            response_prefix=config.mining.response_prefix,
            candidate_token_ids=candidate.token_ids,
        )
        expected_control_ids = ranking_by_rank[mining_rank].get("control_token_ids")
        if list(control_ids) != expected_control_ids:
            raise ValueError("screening and probe internal controls differ")
        result = probe_candidate(
            model,
            tokenizer,
            device,
            prompts=prompts,
            candidate_token_ids=candidate.token_ids,
            control_token_ids=control_ids,
            config=diagnostic_config,
            seed=args.seed,
            shuffle_seed=args.seed,
        )
        probe_payload = result.to_dict()
        _validate_probe_steps(probe_payload, max_steps=args.max_steps)
        memberships = [
            name for name, ranks in strategies.items() if mining_rank in ranks
        ]
        support = family_support[mining_rank - 1]
        evidence.append(
            {
                "mining_rank": mining_rank,
                "selection_memberships": memberships,
                "family_support": support,
                "candidate": candidate.to_dict(),
                "control_token_ids": list(control_ids),
                "initial_score": ranking_by_rank[mining_rank]["initial_score"],
                "probe": probe_payload,
                "frozen_display_profile_met": (
                    result.max_log_likelihood_gap
                    >= FROZEN_LOG_LIKELIHOOD_GAP_THRESHOLD
                    and support >= FROZEN_MINIMUM_FAMILY_SUPPORT
                ),
            }
        )
        runtime = _runtime_payload(
            model=model,
            device=device,
            started=started,
            previous_elapsed_seconds=previous_elapsed_seconds,
            previous_peak_cuda_memory_bytes=previous_peak_cuda_memory_bytes,
        )
        write_json(
            output,
            _report_payload(
                source_contract=source_contract,
                detector_truth_inputs=candidate_report["detector_truth_inputs"],
                input_manifest=input_manifest,
                probe_inputs=probe_inputs,
                strategies=strategies,
                union_ranks=union_ranks,
                evidence=evidence,
                runtime=runtime,
                status="running",
            ),
        )
        print(
            f"[probe-ablation] {index}/{len(union_ranks)} "
            f"mining_rank={mining_rank} display={evidence[-1]['frozen_display_profile_met']}",
            flush=True,
        )
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if sorted(int(item["mining_rank"]) for item in evidence) != union_ranks:
        raise ValueError("completed evidence does not exactly cover the probe union")
    runtime = _runtime_payload(
        model=model,
        device=device,
        started=started,
        previous_elapsed_seconds=previous_elapsed_seconds,
        previous_peak_cuda_memory_bytes=previous_peak_cuda_memory_bytes,
    )
    write_json(
        output,
        _report_payload(
            source_contract=source_contract,
            detector_truth_inputs=candidate_report["detector_truth_inputs"],
            input_manifest=input_manifest,
            probe_inputs=probe_inputs,
            strategies=strategies,
            union_ranks=union_ranks,
            evidence=evidence,
            runtime=runtime,
            status="complete",
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ranking-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
