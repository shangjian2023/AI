"""Aggregate matched-model full-candidate step-0 ranking diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from competition_core.reporting import file_sha256, write_json
from scripts.run_candidate_screening_probe_ablation import screening_strategy_ranks
from scripts.run_candidate_step0_screen import rank_screening_evidence

_TRUTH_FREE_INPUTS = {
    "known_condition": False,
    "known_target_sequence": False,
    "poisoned_data": False,
    "clean_reference_model": False,
}


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError("inputs must use unique NAME=PATH entries")
        result[name] = Path(raw_path)
    return result


def _validate_ranking(report: dict[str, Any]) -> None:
    evidence = report.get("evidence")
    if (
        report.get("role") != "latent_probe"
        or report.get("analysis_kind") != "candidate_step0_screening"
        or report.get("decision_use") is not False
        or report.get("detector_truth_inputs") != _TRUTH_FREE_INPUTS
        or report.get("status") != "complete"
        or not isinstance(evidence, list)
    ):
        raise ValueError("report is not a completed truth-free step-0 ranking")
    ranks = [int(item["mining_rank"]) for item in evidence]
    if (
        len(ranks) != len(set(ranks))
        or len(ranks) != int(report.get("screened_candidate_count", -1))
        or len(ranks) != int(report.get("expected_candidate_count", -1))
    ):
        raise ValueError("step-0 candidate coverage is incomplete or inconsistent")
    score_fields = (
        "candidate_mean_log_likelihood",
        "control_mean_log_likelihood",
        "log_likelihood_gap",
        "candidate_probability",
        "control_probability",
        "probability_gap",
    )
    for item in evidence:
        score = item.get("initial_score") or {}
        if any(
            not isinstance(score.get(field), (int, float))
            or not math.isfinite(float(score[field]))
            for field in score_fields
        ):
            raise ValueError("step-0 report contains a non-finite score")


def _model_summary(report: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    evidence = report["evidence"]
    strategies = screening_strategy_ranks(
        evidence,
        top_k=top_k,
        suffix_tokens=8,
        minimum_family_support=5,
    )
    candidate_only = rank_screening_evidence(evidence, top_k=top_k)
    candidate_only_ranks = [int(item["mining_rank"]) for item in candidate_only]
    reported_ranks = [int(item["mining_rank"]) for item in report["selected_top_k"]]
    if reported_ranks != candidate_only_ranks:
        raise ValueError("reported candidate-only Top-K is inconsistent")
    return {
        "screened_candidate_count": len(evidence),
        "maximum_family_support": max(int(item["family_support"]) for item in evidence),
        "maximum_log_likelihood_gap": max(
            float(item["initial_score"]["log_likelihood_gap"]) for item in evidence
        ),
        "candidate_mean_log_likelihood_top_k": candidate_only_ranks,
        "selection_strategies": strategies,
        "runtime": report["runtime"],
    }


def _target_summary(audit: dict[str, Any]) -> dict[str, Any]:
    recall = audit["recall"]
    return {
        "target_present_in_mining": recall["target_present_in_mining"],
        "match_type": recall["match_type"],
        "target_mining_rank": recall["target_mining_rank"],
        "best_suffix_tokens": recall["best_suffix_tokens"],
        "best_suffix_fraction": recall["best_suffix_fraction"],
        "step0_rankings": recall["step0_rankings"],
    }


def build_summary(
    backdoors: dict[str, Path],
    cleans: dict[str, Path],
    audits: dict[str, Path],
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    if not backdoors or set(audits) != set(backdoors) or set(backdoors) & set(cleans):
        raise ValueError("each backdoor requires one audit and model names must be unique")
    models: dict[str, Any] = {}
    for role, paths in (("backdoor", backdoors), ("clean", cleans)):
        for name, path in paths.items():
            report = json.loads(path.read_text(encoding="utf-8"))
            _validate_ranking(report)
            model = {
                "role": role,
                "report_path": str(path.resolve()),
                "report_sha256": file_sha256(path),
                **_model_summary(report, top_k=top_k),
            }
            if role == "backdoor":
                audit_path = audits[name]
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if (
                    audit.get("sources", {}).get("ranking_report_sha256")
                    != model["report_sha256"]
                ):
                    raise ValueError("target audit does not match its step-0 report")
                model["target"] = _target_summary(audit)
                model["target_audit_path"] = str(audit_path.resolve())
                model["target_audit_sha256"] = file_sha256(audit_path)
            models[name] = model
    backdoor_models = [model for model in models.values() if model["role"] == "backdoor"]
    return {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "analysis_kind": "candidate_step0_multimodel_summary",
        "decision_use": False,
        "top_k": top_k,
        "cohort": {
            "backdoor_count": len(backdoors),
            "clean_count": len(cleans),
            "legacy_relaxed_v1": True,
            "blind": False,
        },
        "target_recall": {
            "exact_or_normalized_mining_recall_count": sum(
                model["target"]["target_present_in_mining"]
                for model in backdoor_models
            ),
            "candidate_mean_log_likelihood_recall_at_8_count": sum(
                model["target"]["step0_rankings"]["candidate_mean_log_likelihood"][
                    "recall_at_8"
                ]
                for model in backdoor_models
            ),
            "log_likelihood_gap_recall_at_8_count": sum(
                model["target"]["step0_rankings"]["log_likelihood_gap"][
                    "recall_at_8"
                ]
                for model in backdoor_models
            ),
            "family_reserved_gap_recall_at_8_count": sum(
                model["target"]["step0_rankings"][
                    "family_reserved_log_likelihood_gap"
                ]["recall_at_8"]
                for model in backdoor_models
            ),
        },
        "models": models,
        "limitations": [
            "Step-0 ranking is candidate selection, not a detector decision or FPR estimate.",
            "The legacy relaxed-v1 training uses 25 percent poisoning and is diagnostic only.",
            "Missing exact targets in OPT and Pythia must be fixed in mining before probing.",
            "Absolute step-0 gaps overlap between backdoor and matched clean models.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    backdoor_count = summary["cohort"]["backdoor_count"]
    lines = [
        "# Relaxed-v1 Multi-Model Step-0 Screening",
        "",
        "This is a `decision_use=false` training-side diagnostic.",
        "",
        "| Model | Role | Candidates | Max support | Max gap | Target mining | "
        "Candidate rank | Gap rank | Family R@8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in summary["models"].items():
        target = model.get("target") or {}
        rankings = target.get("step0_rankings") or {}
        candidate_rank = (rankings.get("candidate_mean_log_likelihood") or {}).get(
            "target_rank", "N/A"
        )
        gap_rank = (rankings.get("log_likelihood_gap") or {}).get(
            "target_rank", "N/A"
        )
        family_recall = (rankings.get("family_reserved_log_likelihood_gap") or {}).get(
            "recall_at_8", "N/A"
        )
        lines.append(
            f"| `{name}` | {model['role']} | {model['screened_candidate_count']} | "
            f"{model['maximum_family_support']} | {model['maximum_log_likelihood_gap']:.4f} | "
            f"{target.get('target_mining_rank', 'N/A')} | {candidate_rank} | {gap_rank} | "
            f"{family_recall} |"
        )
    recall = summary["target_recall"]
    lines.extend(
        [
            "",
            "## Recall",
            "",
            f"- Mining exact/normalized recall: "
            f"{recall['exact_or_normalized_mining_recall_count']}/{backdoor_count}",
            f"- Candidate-only Recall@8: "
            f"{recall['candidate_mean_log_likelihood_recall_at_8_count']}/{backdoor_count}",
            f"- Gap-only Recall@8: "
            f"{recall['log_likelihood_gap_recall_at_8_count']}/{backdoor_count}",
            f"- Family+gap Recall@8: "
            f"{recall['family_reserved_gap_recall_at_8_count']}/{backdoor_count}",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdoor", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--clean", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--target-audit", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    summary = build_summary(
        _named_paths(args.backdoor),
        _named_paths(args.clean),
        _named_paths(args.target_audit),
        top_k=args.top_k,
    )
    write_json(args.output_json, summary)
    Path(args.output_markdown).write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
