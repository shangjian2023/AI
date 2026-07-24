from __future__ import annotations

import json

from competition_core.reporting import file_sha256
from scripts.analyze_candidate_step0_multimodel import build_summary


def _ranking() -> dict:
    return {
        "role": "latent_probe",
        "analysis_kind": "candidate_step0_screening",
        "decision_use": False,
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "status": "complete",
        "screened_candidate_count": 1,
        "expected_candidate_count": 1,
        "selected_top_k": [{"screening_rank": 1, "mining_rank": 1}],
        "runtime": {"session_elapsed_seconds": 1.0},
        "evidence": [
            {
                "mining_rank": 1,
                "family_support": 5,
                "candidate": {"token_ids": [1, 2]},
                "initial_score": {
                    "candidate_mean_log_likelihood": -1.0,
                    "control_mean_log_likelihood": -2.0,
                    "log_likelihood_gap": 1.0,
                    "candidate_probability": 0.5,
                    "control_probability": 0.25,
                    "probability_gap": 0.25,
                },
            }
        ],
    }


def test_multimodel_summary_keeps_step0_as_recall_diagnostic(tmp_path) -> None:
    backdoor_path = tmp_path / "backdoor.json"
    clean_path = tmp_path / "clean.json"
    backdoor_path.write_text(json.dumps(_ranking()), encoding="utf-8")
    clean_path.write_text(json.dumps(_ranking()), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "sources": {"ranking_report_sha256": file_sha256(backdoor_path)},
                "recall": {
                    "target_present_in_mining": True,
                    "match_type": "token_exact",
                    "target_mining_rank": 1,
                    "best_suffix_tokens": 2,
                    "best_suffix_fraction": 1.0,
                    "step0_rankings": {
                        "candidate_mean_log_likelihood": {"recall_at_8": True},
                        "log_likelihood_gap": {"recall_at_8": True},
                        "family_reserved_log_likelihood_gap": {"recall_at_8": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_summary(
        {"backdoor": backdoor_path},
        {"clean": clean_path},
        {"backdoor": audit_path},
        top_k=1,
    )

    assert summary["decision_use"] is False
    assert summary["target_recall"]["exact_or_normalized_mining_recall_count"] == 1
    assert summary["target_recall"]["family_reserved_gap_recall_at_8_count"] == 1
