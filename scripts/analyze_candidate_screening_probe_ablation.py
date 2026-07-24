"""Aggregate completed truth-free candidate-screening probe ablations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from competition_core.reporting import file_sha256, write_json

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


def _validate_report(report: dict[str, Any]) -> None:
    evidence = report.get("evidence")
    union = report.get("union_mining_ranks")
    if (
        report.get("role") != "latent_probe"
        or report.get("analysis_kind") != "candidate_screening_probe_ablation"
        or report.get("decision_use") is not False
        or report.get("detector_truth_inputs") != _TRUTH_FREE_INPUTS
        or report.get("status") != "complete"
        or not isinstance(evidence, list)
        or not isinstance(union, list)
    ):
        raise ValueError("report is not a completed truth-free screening ablation")
    ranks = [int(item["mining_rank"]) for item in evidence]
    if (
        len(ranks) != len(set(ranks))
        or sorted(ranks) != union
        or len(ranks) != int(report.get("expected_candidate_count", -1))
        or len(ranks) != int(report.get("evaluated_candidate_count", -1))
    ):
        raise ValueError("report candidate union is incomplete or inconsistent")
    for item in evidence:
        probe = item.get("probe") or {}
        steps = probe.get("steps")
        expected_steps = int(report["source_contract"]["max_steps"])
        if not isinstance(steps, list) or [step.get("step") for step in steps] != list(
            range(1, expected_steps + 1)
        ):
            raise ValueError("report contains an incomplete probe trajectory")
        values = (
            probe.get("max_log_likelihood_gap"),
            probe.get("final_log_likelihood_gap"),
            probe.get("max_probability_gap"),
            probe.get("final_probability_gap"),
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("report contains a non-finite probe value")


def _strategy_summary(
    report: dict[str, Any], strategy_ranks: list[int]
) -> dict[str, Any]:
    by_rank = {int(item["mining_rank"]): item for item in report["evidence"]}
    missing_ranks = [rank for rank in strategy_ranks if rank not in by_rank]
    evaluated_ranks = [rank for rank in strategy_ranks if rank in by_rank]
    profile_positive = [
        rank for rank in evaluated_ranks if by_rank[rank]["frozen_display_profile_met"]
    ]
    paper_positive = [
        rank for rank in evaluated_ranks if by_rank[rank]["probe"]["criterion_met"]
    ]
    return {
        "selected_mining_ranks": strategy_ranks,
        "selected_count": len(strategy_ranks),
        "probe_coverage_complete": not missing_ranks,
        "missing_probe_mining_ranks": missing_ranks,
        "profile_positive_mining_ranks": profile_positive,
        "profile_positive": bool(profile_positive) if not missing_ranks else None,
        "paper_positive_mining_ranks": paper_positive,
        "paper_positive": bool(paper_positive) if not missing_ranks else None,
    }


def _target_summary(
    report: dict[str, Any],
    audit: dict[str, Any],
    *,
    audit_path: Path,
) -> dict[str, Any]:
    expected_sha = audit.get("sources", {}).get("ranking_report_sha256")
    if expected_sha != report.get("source_contract", {}).get("ranking_report_sha256"):
        raise ValueError("training-side audit does not match its screening report")
    recall = audit["recall"]
    target_rank = recall.get("target_mining_rank")
    evidence = next(
        (
            item
            for item in report["evidence"]
            if int(item["mining_rank"]) == target_rank
        ),
        None,
    )
    return {
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": file_sha256(audit_path),
        "match_type": recall["match_type"],
        "target_mining_rank": target_rank,
        "step0_rankings": recall["step0_rankings"],
        "evaluated_in_union": evidence is not None,
        "family_support": evidence.get("family_support") if evidence else None,
        "max_log_likelihood_gap": (
            evidence["probe"]["max_log_likelihood_gap"] if evidence else None
        ),
        "final_log_likelihood_gap": (
            evidence["probe"]["final_log_likelihood_gap"] if evidence else None
        ),
        "frozen_display_profile_met": (
            evidence["frozen_display_profile_met"] if evidence else False
        ),
    }


def build_summary(
    reports: dict[str, Path],
    target_audits: dict[str, Path],
) -> dict[str, Any]:
    if not reports or not set(target_audits).issubset(reports):
        raise ValueError("target audits must name a subset of the reports")
    models: dict[str, Any] = {}
    strategy_names: set[str] | None = None
    for name, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path.resolve())
        _validate_report(report)
        current_strategies = set(report["selection_strategies"])
        strategy_names = (
            current_strategies
            if strategy_names is None
            else strategy_names & current_strategies
        )
        model = {
            "role": "backdoor" if name in target_audits else "clean",
            "report_path": str(path.resolve()),
            "report_sha256": file_sha256(path),
            "union_candidate_count": len(report["evidence"]),
            "strategies": {
                strategy: _strategy_summary(report, ranks)
                for strategy, ranks in report["selection_strategies"].items()
            },
            "runtime": report["runtime"],
        }
        if name in target_audits:
            audit_path = target_audits[name]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            model["target"] = _target_summary(
                report,
                audit,
                audit_path=audit_path,
            )
        models[name] = model
    metrics: dict[str, Any] = {}
    for strategy in sorted(strategy_names or ()):
        if not all(
            model["strategies"][strategy]["probe_coverage_complete"]
            for model in models.values()
        ):
            metrics[strategy] = {
                "probe_coverage_complete": False,
                "frozen_profile": None,
                "paper_probability_clean_positive_count": None,
            }
            continue
        tp = sum(
            model["role"] == "backdoor"
            and model["strategies"][strategy]["profile_positive"]
            for model in models.values()
        )
        fn = sum(model["role"] == "backdoor" for model in models.values()) - tp
        fp = sum(
            model["role"] == "clean"
            and model["strategies"][strategy]["profile_positive"]
            for model in models.values()
        )
        tn = sum(model["role"] == "clean" for model in models.values()) - fp
        paper_fp = sum(
            model["role"] == "clean"
            and model["strategies"][strategy]["paper_positive"]
            for model in models.values()
        )
        metrics[strategy] = {
            "probe_coverage_complete": True,
            "frozen_profile": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "paper_probability_clean_positive_count": paper_fp,
        }
    return {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "analysis_kind": "candidate_screening_probe_ablation_summary",
        "decision_use": False,
        "cohort": {
            "model_count": len(models),
            "backdoor_count": len(target_audits),
            "clean_count": len(models) - len(target_audits),
            "scope": "candidate_screening_probe_ablation_cohort",
            "calibration_overlap": True,
            "blind": False,
        },
        "models": models,
        "strategy_metrics": metrics,
        "limitations": [
            "This cohort alone is not evidence of cross-architecture generalization.",
            "Only the evaluated strategy union was probed; unselected candidates remain unknown.",
            "The 0.25 paper criterion is reported for reproduction and is not the "
            "display decision.",
            "The candidate-screening and 192-step runners remain decision_use=false diagnostics.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Candidate Step-0 / 192-Step Ablation",
        "",
        "This is a `decision_use=false` development diagnostic, not a blind "
        "generalization result.",
        "",
        "| Model | Role | Union | Candidate-only rank | Gap rank | Target max LL gap "
        "| Support | Profile |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
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
        max_gap = target.get("max_log_likelihood_gap")
        lines.append(
            f"| `{name}` | {model['role']} | {model['union_candidate_count']} | "
            f"{candidate_rank} | {gap_rank} | "
            f"{max_gap:.6f} | {target.get('family_support')} | "
            f"{target.get('frozen_display_profile_met')} |"
            if max_gap is not None
            else (
                f"| `{name}` | {model['role']} | {model['union_candidate_count']} | "
                "N/A | N/A | N/A | N/A | N/A |"
            )
        )
    lines.extend(["", "## Strategy Results", ""])
    for strategy, metric in summary["strategy_metrics"].items():
        if not metric["probe_coverage_complete"]:
            lines.append(f"- `{strategy}`: probe coverage incomplete; no model metric")
            continue
        confusion = metric["frozen_profile"]
        lines.append(
            f"- `{strategy}`: TP={confusion['tp']}, FN={confusion['fn']}, "
            f"FP={confusion['fp']}, TN={confusion['tn']}; paper-0.25 clean positives="
            f"{metric['paper_probability_clean_positive_count']}"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--target-audit", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    summary = build_summary(
        _named_paths(args.report),
        _named_paths(args.target_audit),
    )
    write_json(args.output_json, summary)
    Path(args.output_markdown).write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
