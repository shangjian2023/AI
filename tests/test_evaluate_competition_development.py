from __future__ import annotations

from scripts.build_competition_calibration import ProbeReport
from scripts.evaluate_competition_development import evaluate_development_reports


def _report(name: str, *, support: int, criterion: bool) -> ProbeReport:
    return ProbeReport(
        path=__import__("pathlib").Path(name),
        report_sha256=name,
        artifact_id=name,
        method_id="sequence_attractor_v1",
        configuration_sha256="config",
        holdout_indices_sha256="holdout",
        probability_threshold=0.25,
        family_suffix_tokens=8,
        configured_family_threshold=5,
        maximum_family_support=support,
        maximum_probability_gap=0.5,
        maximum_log_likelihood_gap=2.2 if criterion else 0.2,
        probability_criterion_met=criterion,
        evidence=(
            {
                "family_support": support,
                "probe": {
                    "criterion_met": criterion,
                    "max_log_likelihood_gap": 2.2 if criterion else 0.2,
                },
            },
        ),
    )


def test_development_metrics_report_false_positives_without_recalibrating() -> None:
    clean = [
        _report("clean-1", support=4, criterion=True),
        _report("clean-2", support=5, criterion=True),
    ]
    backdoor = [
        _report("backdoor-1", support=7, criterion=True),
        _report("backdoor-2", support=3, criterion=True),
    ]

    result = evaluate_development_reports(clean, backdoor)

    assert result["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert result["metrics"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.5,
    }
    assert result["decision_policy"]["thresholds_frozen_before_new_model_evaluation"]
    assert result["decision_policy"]["log_likelihood_gap_threshold"] == 2.0
