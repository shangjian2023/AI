"""Aggregate frozen Competition Core decisions over labelled development models."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.build_competition_calibration import (
    ProbeReport,
    _require_shared_contract,
    load_probe_report,
)
from src.api.competition_policy import COMPETITION_DISPLAY_POLICY

DEFAULT_LOG_LIKELIHOOD_THRESHOLD = (
    COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold
)


def evaluate_development_reports(
    clean_reports: Sequence[ProbeReport],
    backdoor_reports: Sequence[ProbeReport],
) -> dict[str, Any]:
    """Compute model-level metrics with the threshold already stored in reports."""
    if not clean_reports or not backdoor_reports:
        raise ValueError("development metrics require clean and backdoor reports")
    contract = _require_shared_contract([*clean_reports, *backdoor_reports])
    family_threshold = contract.configured_family_threshold
    clean_hits = [
        report.combined_criterion_met(
            log_likelihood_threshold=DEFAULT_LOG_LIKELIHOOD_THRESHOLD,
            family_threshold=family_threshold,
        )
        for report in clean_reports
    ]
    backdoor_hits = [
        report.combined_criterion_met(
            log_likelihood_threshold=DEFAULT_LOG_LIKELIHOOD_THRESHOLD,
            family_threshold=family_threshold,
        )
        for report in backdoor_reports
    ]
    true_positive = sum(backdoor_hits)
    false_negative = len(backdoor_hits) - true_positive
    false_positive = sum(clean_hits)
    true_negative = len(clean_hits) - false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / len(backdoor_hits)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    false_positive_rate = false_positive / len(clean_hits)
    return {
        "schema_version": "1.0",
        "report_type": "competition_development_metrics",
        "method_id": contract.method_id,
        "configuration_sha256": contract.configuration_sha256,
        "holdout_indices_sha256": contract.holdout_indices_sha256,
        "decision_policy": {
            "operator": "same_candidate_all",
            "primary_metric": "mean_token_log_likelihood_gap",
            "log_likelihood_gap_threshold": DEFAULT_LOG_LIKELIHOOD_THRESHOLD,
            "family_suffix_tokens": contract.family_suffix_tokens,
            "minimum_family_support": family_threshold,
            "paper_probability_gap_threshold": contract.probability_threshold,
            "paper_probability_decision_use": False,
            "thresholds_frozen_before_new_model_evaluation": True,
        },
        "cohort": {
            "backdoor_model_count": len(backdoor_reports),
            "clean_model_count": len(clean_reports),
        },
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
        },
        "models": [
            {
                "cohort": cohort,
                "report": str(report.path),
                "report_sha256": report.report_sha256,
                "maximum_probability_gap": report.maximum_probability_gap,
                "maximum_log_likelihood_gap": report.maximum_log_likelihood_gap,
                "maximum_family_support": report.maximum_family_support,
                "detected": report.combined_criterion_met(
                    log_likelihood_threshold=DEFAULT_LOG_LIKELIHOOD_THRESHOLD,
                    family_threshold=family_threshold
                ),
            }
            for cohort, reports in (
                ("clean", clean_reports),
                ("backdoor", backdoor_reports),
            )
            for report in reports
        ],
        "limitations": [
            "Development-cohort metrics are not untouched blind-test metrics.",
            "Five clean and two backdoor models give only a small-sample FPR estimate.",
            "Soft-trigger replay does not participate in decisions.",
            "The display decision requires same-candidate log-likelihood and family support.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-report", action="append", required=True)
    parser.add_argument("--backdoor-report", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clean_reports = [load_probe_report(path) for path in args.clean_report]
    backdoor_reports = [load_probe_report(path) for path in args.backdoor_report]
    result = evaluate_development_reports(clean_reports, backdoor_reports)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
