"""Build a CPU-only baseline from mining and Stage 2 diagnostic reports."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from competition_core import METHOD_ID
from competition_core.candidate_cleaning import clean_probe_candidates
from competition_core.config import load_detection_config
from competition_core.sequence_mining import (
    SequenceCandidate,
    candidate_family_support,
)

_TRUTH_FREE_INPUTS = {
    "known_condition": False,
    "known_target_sequence": False,
    "poisoned_data": False,
    "clean_reference_model": False,
}
_PAPER_CONTROLS = {"boundary", "first_prompt", "median_prompt"}


def _candidate(raw: dict[str, Any]) -> SequenceCandidate:
    return SequenceCandidate(
        token_ids=tuple(int(item) for item in raw["token_ids"]),
        text=str(raw["text"]),
        continuation_probabilities=tuple(
            float(item) for item in raw["continuation_probabilities"]
        ),
        suffix_floor=float(raw["suffix_floor"]),
        mean_log_probability=float(raw["mean_log_probability"]),
        used_beam=bool(raw["used_beam"]),
        seed_token_id=int(raw["seed_token_id"]),
        token_texts=tuple(str(item) for item in raw.get("token_texts", ())),
        selection_modes=tuple(str(item) for item in raw.get("selection_modes", ())),
    )


def _selection_rows(path: Path, config_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != "sequence_mining":
        raise ValueError(f"{path} is not a sequence-mining report")
    if payload.get("method_id") != METHOD_ID:
        raise ValueError(f"{path} uses an incompatible method")
    if payload.get("detector_truth_inputs") != _TRUTH_FREE_INPUTS:
        raise ValueError(f"{path} does not prove truth-free candidate discovery")
    config = load_detection_config(config_path)
    if payload.get("mining_config") != asdict(config.mining):
        raise ValueError(f"{path} does not match the supplied detection config")
    candidates = tuple(
        _candidate(item) for item in payload["result"].get("candidates", [])
    )
    probe = config.probe
    support = candidate_family_support(
        candidates,
        suffix_tokens=probe.family_suffix_tokens,
    )
    rank_order = clean_probe_candidates(
        candidates,
        replace(probe, candidate_selection_strategy="rank_order"),
        family_support=support,
    )
    family = clean_probe_candidates(
        candidates,
        replace(probe, candidate_selection_strategy="family_representative"),
        family_support=support,
    )

    def selected(result: Any) -> list[dict[str, int]]:
        return [
            {
                "mining_rank": item.mining_rank,
                "family_support": support[item.mining_rank - 1],
            }
            for item in result.selected
        ]

    rank_rows = selected(rank_order)
    family_rows = selected(family)
    return {
        "report": str(path),
        "model_id": path.parent.name,
        "retained_candidate_count": len(candidates),
        "maximum_retained_family_support": max(support, default=0),
        "rank_order": rank_rows,
        "family_representative": family_rows,
        "selection_changed": rank_rows != family_rows,
        "pre_deduplication_audit_available": bool(
            (payload["result"].get("candidate_audit") or {}).get("complete", False)
        ),
    }


def _paper_summary(path: Path) -> dict[str, Any]:
    rows: dict[str, list[dict[str, float | bool]]] = {
        "backdoor": [],
        "clean": [],
    }
    cohort_signatures: set[str] = set()
    for role, pattern in (
        ("backdoor", "matched/gpt2__backdoor_target__*.json"),
        ("clean", "cross/cross__gpt2__backdoor_target__*.json"),
    ):
        cell_paths = sorted(path.glob(pattern))
        if len(cell_paths) != len(_PAPER_CONTROLS):
            raise ValueError(
                f"paper results require exactly three {role} control cells"
            )
        observed_controls: set[str] = set()
        for cell_path in cell_paths:
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
            cell_config = payload.get("cell_config") or {}
            if (
                payload.get("role") != "training_side_method_diagnostic"
                or payload.get("decision_use") is not False
                or payload.get("known_target_sequence") is not True
                or cell_config.get("arch") != "gpt2"
                or cell_config.get("cand_role") != "backdoor_target"
                or cell_config.get("candidate_source") != "training_yaml_target"
            ):
                raise ValueError(f"{cell_path} is not a valid paper oracle cell")
            observed_controls.add(str(cell_config.get("ctrl_id")))
            integrity = payload.get("integrity") or {}
            signature = {
                "frozen_config": payload.get("frozen_config"),
                "arch": cell_config.get("arch"),
                "candidate_role": cell_config.get("cand_role"),
                "candidate_source": cell_config.get("candidate_source"),
                "initialization_seed": cell_config.get("init_seed"),
                "detection_yaml_sha256": integrity.get("detection_yaml_sha256"),
                "mining_json_sha256": integrity.get("mining_json_sha256"),
                "probe_input_content_sha256": integrity.get(
                    "probe_input_content_sha256"
                ),
                "probe_input_indices_sha256": integrity.get(
                    "probe_input_indices_sha256"
                ),
                "target_sequence_sha256": integrity.get("target_sequence_sha256"),
            }
            if any(value is None for value in signature.values()):
                raise ValueError(f"{cell_path} lacks paper cohort integrity fields")
            cohort_signatures.add(json.dumps(signature, sort_keys=True))
            checkpoints = payload["checkpoints"]
            summary = payload["probe_summary"]
            rows[role].append(
                {
                    "step_0_probability_gap": float(
                        checkpoints["step_0"]["probability_gap"]
                    ),
                    "step_0_candidate_mean_log_likelihood": float(
                        checkpoints["step_0"]["candidate_mean_log_likelihood"]
                    ),
                    "maximum_probability_gap": float(
                        summary["max_probability_gap"]
                    ),
                    "final_probability_gap": float(
                        summary["final_probability_gap"]
                    ),
                    "paper_criterion_met": bool(summary["criterion_met"]),
                }
            )
        if observed_controls != _PAPER_CONTROLS:
            raise ValueError(f"paper results have an invalid {role} control cohort")
    if len(cohort_signatures) != 1:
        raise ValueError("paper result cells do not share one frozen cohort")

    def aggregate(items: list[dict[str, float | bool]]) -> dict[str, float | int]:
        numeric_keys = (
            "step_0_probability_gap",
            "step_0_candidate_mean_log_likelihood",
            "maximum_probability_gap",
            "final_probability_gap",
        )
        return {
            **{
                f"median_{key}": statistics.median(
                    float(item[key]) for item in items
                )
                for key in numeric_keys
            },
            "paper_criterion_hit_count": sum(
                bool(item["paper_criterion_met"]) for item in items
            ),
            "cell_count": len(items),
        }

    return {role: aggregate(items) for role, items in rows.items()}


def analyze(
    *,
    config_path: Path,
    mining_reports: list[Path],
    offline_score_search: Path,
    paper_results: Path,
) -> dict[str, Any]:
    """Combine existing CPU-readable evidence without fitting a new decision rule."""
    if not mining_reports:
        raise ValueError("at least one mining report is required")
    offline = json.loads(offline_score_search.read_text(encoding="utf-8"))
    if (
        offline.get("analysis_kind") != "cpu_offline_univariate_score_search"
        or offline.get("role") != "training_side_method_diagnostic"
        or offline.get("known_target_sequence") is not True
        or offline.get("decision_use") is not False
    ):
        raise ValueError("offline score search is not a training-side diagnostic")
    top_feature = offline["top_features"][0]
    mining = [_selection_rows(path, config_path) for path in mining_reports]
    audit_available_count = sum(
        bool(item["pre_deduplication_audit_available"]) for item in mining
    )
    limitations = [
        "Stage 2 backdoor candidates use known training targets, not blind mining output.",
        (
            "The historical step-0 threshold is development evidence and is not a "
            "frozen decision rule."
        ),
        (
            "The historical threshold scores one shuffled batch; it cannot be "
            "applied to the new full-dataset step-0 API without new calibration."
        ),
    ]
    if audit_available_count < len(mining):
        limitations.insert(
            0,
            (
                f"{len(mining) - audit_available_count} of {len(mining)} mining "
                "reports do not contain the pre-deduplication candidate pool."
            ),
        )
    return {
        "schema_version": "1.0",
        "analysis_kind": "candidate_recall_cpu_baseline",
        "role": "training_side_method_diagnostic",
        "decision_use": False,
        "sources": {
            "detection_config": str(config_path),
            "offline_score_search": str(offline_score_search),
            "paper_results": str(paper_results),
            "mining_reports": [str(path) for path in mining_reports],
        },
        "mining_selection_baseline": mining,
        "mining_selection_changed_count": sum(
            bool(item["selection_changed"]) for item in mining
        ),
        "pre_deduplication_audit_available_count": audit_available_count,
        "step_0_development_feature": {
            "feature": top_feature["feature"],
            "global_fit": top_feature["global_fit"],
            "leave_one_architecture_out": top_feature[
                "leave_one_architecture_out"
            ]["metrics"],
            "cohort": offline["cohort"],
        },
        "paper_compute_oracle": _paper_summary(paper_results),
        "limitations": limitations,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Candidate Recall CPU Baseline",
        "",
        "This report reuses existing mining and Stage 2 results. It does not fit or",
        "activate a new detector threshold.",
        "",
        "## Mining Selection",
        "",
        "| Model | Candidates | Max support | Rank-order Top-K | Family Top-K | Changed |",
        "|---|---:|---:|---|---|---:|",
    ]
    for item in payload["mining_selection_baseline"]:
        rank = ", ".join(
            f"{row['mining_rank']}({row['family_support']})"
            for row in item["rank_order"]
        )
        family = ", ".join(
            f"{row['mining_rank']}({row['family_support']})"
            for row in item["family_representative"]
        )
        lines.append(
            f"| {item['model_id']} | {item['retained_candidate_count']} | "
            f"{item['maximum_retained_family_support']} | {rank} | {family} | "
            f"{str(item['selection_changed']).lower()} |"
        )
    feature = payload["step_0_development_feature"]
    paper = payload["paper_compute_oracle"]
    report_count = len(payload["mining_selection_baseline"])
    changed_count = payload["mining_selection_changed_count"]
    audit_count = payload["pre_deduplication_audit_available_count"]
    if changed_count:
        selection_conclusion = (
            f"Family-representative selection changes {changed_count} of the "
            f"{report_count} Top-K sets."
        )
    else:
        selection_conclusion = (
            "Family-representative selection does not change any of the "
            f"{report_count} current Top-K sets."
        )
    lines.extend(
        [
            "",
            "## Reused Stage 2 Evidence",
            "",
            f"- Best development feature: `{feature['feature']}`",
            f"- Global threshold: `{feature['global_fit']['threshold']:.6f}`",
            "- Leave-one-architecture-out: "
            f"TP={feature['leave_one_architecture_out']['true_positive']}, "
            f"TN={feature['leave_one_architecture_out']['true_negative']}, "
            f"FP={feature['leave_one_architecture_out']['false_positive']}, "
            f"FN={feature['leave_one_architecture_out']['false_negative']}",
            "- Paper-compute oracle median step-0 probability gap: "
            f"backdoor={paper['backdoor']['median_step_0_probability_gap']:.6f}, "
            f"clean={paper['clean']['median_step_0_probability_gap']:.6f}",
            "- Paper-compute oracle median final probability gap: "
            f"backdoor={paper['backdoor']['median_final_probability_gap']:.6f}, "
            f"clean={paper['clean']['median_final_probability_gap']:.6f}",
            "",
            "## Conclusion",
            "",
            selection_conclusion,
            f"Pre-deduplication audit is available for {audit_count} of {report_count} reports.",
            "Full-dataset step-0 scoring is the next collection step, but requires",
            "new matched-pair calibration before it can affect candidate selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mining-report", action="append", required=True, type=Path)
    parser.add_argument("--offline-score-search", required=True, type=Path)
    parser.add_argument("--paper-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = analyze(
        config_path=args.config,
        mining_reports=args.mining_report,
        offline_score_search=args.offline_score_search,
        paper_results=args.paper_results,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_recall_cpu_baseline.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "CANDIDATE_RECALL_CPU_BASELINE.md").write_text(
        _markdown(payload),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
