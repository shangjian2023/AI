from __future__ import annotations

import json

from scripts.analyze_candidate_screening_probe_ablation import build_summary


def _report(*, profile_positive: bool) -> dict:
    strategies = {
        "mining_rank_top_k": [1],
        "log_likelihood_gap_top_k": [1],
        "family_reserved_log_likelihood_gap_top_k": [1],
    }
    return {
        "role": "latent_probe",
        "analysis_kind": "candidate_screening_probe_ablation",
        "decision_use": False,
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "status": "complete",
        "source_contract": {
            "max_steps": 1,
            "ranking_report_sha256": "ranking-sha",
        },
        "selection_strategies": strategies,
        "union_mining_ranks": [1],
        "expected_candidate_count": 1,
        "evaluated_candidate_count": 1,
        "evidence": [
            {
                "mining_rank": 1,
                "family_support": 5 if profile_positive else 1,
                "frozen_display_profile_met": profile_positive,
                "probe": {
                    "criterion_met": True,
                    "max_log_likelihood_gap": 2.5 if profile_positive else 1.0,
                    "final_log_likelihood_gap": 2.1 if profile_positive else 0.8,
                    "max_probability_gap": 0.5,
                    "final_probability_gap": 0.3,
                    "steps": [{"step": 1}],
                },
            }
        ],
        "runtime": {"session_elapsed_seconds": 1.0},
    }


def test_build_summary_keeps_truth_audits_separate_from_detector_reports(
    tmp_path,
) -> None:
    backdoor_path = tmp_path / "backdoor.json"
    clean_path = tmp_path / "clean.json"
    backdoor_path.write_text(json.dumps(_report(profile_positive=True)))
    clean_path.write_text(json.dumps(_report(profile_positive=False)))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "sources": {"ranking_report_sha256": "ranking-sha"},
                "recall": {
                    "match_type": "token_exact",
                    "target_mining_rank": 1,
                    "step0_rankings": {
                        "candidate_mean_log_likelihood": {"target_rank": 2},
                        "log_likelihood_gap": {"target_rank": 1},
                    },
                },
            }
        )
    )

    summary = build_summary(
        {"backdoor": backdoor_path, "clean": clean_path},
        {"backdoor": audit_path},
    )

    assert summary["models"]["backdoor"]["target"]["evaluated_in_union"] is True
    assert summary["models"]["clean"]["role"] == "clean"
    metrics = summary["strategy_metrics"]["mining_rank_top_k"]
    assert metrics["frozen_profile"] == {"tp": 1, "fn": 0, "fp": 0, "tn": 1}
    assert metrics["paper_probability_clean_positive_count"] == 1
