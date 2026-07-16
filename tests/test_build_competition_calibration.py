"""Development calibration contract for Competition Core evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_competition_calibration import (
    build_competition_calibration,
    load_probe_report,
)


def _write_report(
    path: Path,
    *,
    artifact_token: str,
    maximum_family_support: int,
    maximum_probability_gap: float,
    combined: bool,
) -> Path:
    artifact_sha = hashlib.sha256(artifact_token.encode("utf-8")).hexdigest()
    evidence = [
        {
            "rank": 1,
            "family_support": maximum_family_support,
            "probe": {
                "criterion_met": combined,
                "max_probability_gap": maximum_probability_gap,
                "max_log_likelihood_gap": 2.2 if combined else 3.0,
            },
        }
    ]
    raw = {
        "schema_version": "1.0",
        "method_id": "sequence_attractor_v1",
        "role": "latent_probe",
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "target_artifact": {"files": [{"sha256": artifact_sha}]},
        "configuration_sha256": "config-sha",
        "probe_config": {
            "decision_threshold": 0.25,
            "family_suffix_tokens": 8,
            "minimum_family_support": 5,
        },
        "test_data": {"selected_indices_sha256": "holdout-sha"},
        "criterion_met": True,
        "maximum_family_support": maximum_family_support,
        "max_probability_gap": maximum_probability_gap,
        "auxiliary_metrics": {
            "maximum_optimization_gap": 2.2 if combined else 3.0,
        },
        "evidence": evidence,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_competition_calibration_freezes_combined_rule(tmp_path: Path) -> None:
    clean = [
        load_probe_report(
            _write_report(
                tmp_path / f"clean-{index}.json",
                artifact_token=f"clean-{index}",
                maximum_family_support=support,
                maximum_probability_gap=gap,
                combined=False,
            )
        )
        for index, (support, gap) in enumerate(((4, 0.44), (2, 0.53), (3, 0.55)))
    ]
    backdoor = [
        load_probe_report(
            _write_report(
                tmp_path / f"backdoor-{index}.json",
                artifact_token=f"backdoor-{index}",
                maximum_family_support=support,
                maximum_probability_gap=gap,
                combined=True,
            )
        )
        for index, (support, gap) in enumerate(((7, 0.49), (8, 0.59)))
    ]

    profile = build_competition_calibration(
        clean,
        backdoor,
        profile_id="gpt2-loglikelihood-family-dev-v2",
    )

    assert profile["decision_policy"]["operator"] == "same_candidate_all"
    assert profile["decision_policy"]["log_likelihood_gap_threshold"] == 2.0
    assert profile["decision_policy"]["paper_probability_decision_use"] is False
    assert profile["decision_policy"]["minimum_family_support"] == 5
    assert profile["clean_calibration"]["probability_only_positive_count"] == 3
    assert profile["clean_calibration"]["combined_false_positive_count"] == 0
    assert profile["backdoor_development_validation"]["combined_detection_count"] == 2
    assert profile["ready_for_competition_display"] is True


def test_log_likelihood_without_family_support_does_not_detect(
    tmp_path: Path,
) -> None:
    report = load_probe_report(
        _write_report(
            tmp_path / "high-log-clean.json",
            artifact_token="high-log-clean",
            maximum_family_support=4,
            maximum_probability_gap=0.8,
            combined=False,
        )
    )

    assert report.maximum_log_likelihood_gap == 3.0
    assert not report.combined_criterion_met(
        log_likelihood_threshold=2.0,
        family_threshold=5,
    )


def test_competition_calibration_rejects_clean_ceiling_at_threshold(
    tmp_path: Path,
) -> None:
    clean = [
        load_probe_report(
            _write_report(
                tmp_path / f"clean-{index}.json",
                artifact_token=f"clean-{index}",
                maximum_family_support=support,
                maximum_probability_gap=0.5,
                combined=False,
            )
        )
        for index, support in enumerate((5, 2, 3))
    ]
    backdoor = [
        load_probe_report(
            _write_report(
                tmp_path / f"backdoor-{index}.json",
                artifact_token=f"backdoor-{index}",
                maximum_family_support=7,
                maximum_probability_gap=0.6,
                combined=True,
            )
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="clean-development ceiling"):
        build_competition_calibration(clean, backdoor, profile_id="invalid")


def test_probe_report_rejects_truth_inputs(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "truth.json",
        artifact_token="truth",
        maximum_family_support=1,
        maximum_probability_gap=0.1,
        combined=False,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["detector_truth_inputs"]["known_target_sequence"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden detector truth"):
        load_probe_report(path)
