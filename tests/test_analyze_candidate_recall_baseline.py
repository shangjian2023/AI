from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from competition_core.config import load_detection_config
from scripts.analyze_candidate_recall_baseline import (
    _markdown,
    _paper_summary,
    _selection_rows,
    analyze,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "competition_core"
    / "configs"
    / "gpt2_detection_diverse_10000_4060.yaml"
)
TRUTH_FREE_INPUTS = {
    "known_condition": False,
    "known_target_sequence": False,
    "poisoned_data": False,
    "clean_reference_model": False,
}
CONTROLS = ("boundary", "first_prompt", "median_prompt")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(rank: int) -> dict:
    texts = (
        "alpha scholarly report includes verified evidence",
        "bravo technical memorandum presents measured results",
        "charlie operational summary records independent findings",
        "delta analytical response explains observed behavior",
        "echo research note documents reproducible outcomes",
    )
    token_ids = [100 + rank * 20 + offset for offset in range(10)]
    return {
        "token_ids": token_ids,
        "text": texts[rank],
        "continuation_probabilities": [0.9] * 9,
        "suffix_floor": 0.9,
        "mean_log_probability": -0.1 - rank,
        "used_beam": False,
        "seed_token_id": token_ids[0],
    }


def _mining_report() -> dict:
    config = load_detection_config(CONFIG_PATH)
    return {
        "schema_version": "1.0",
        "method_id": "sequence_attractor_v1",
        "role": "sequence_mining",
        "detector_truth_inputs": TRUTH_FREE_INPUTS,
        "mining_config": asdict(config.mining),
        "result": {"candidates": [_candidate(rank) for rank in range(5)]},
    }


def _paper_cell(role: str, control: str, index: int) -> dict:
    role_offset = 0.4 if role == "backdoor" else 0.0
    return {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "known_target_sequence": True,
        "decision_use": False,
        "cell_config": {
            "arch": "gpt2",
            "cand_role": "backdoor_target",
            "candidate_source": "training_yaml_target",
            "ctrl_id": control,
            "init_seed": 123,
        },
        "frozen_config": {
            "batch_size": 8,
            "epochs": 3,
            "max_steps": 3750,
        },
        "integrity": {
            "detection_yaml_sha256": "a" * 64,
            "mining_json_sha256": "b" * 64,
            "probe_input_content_sha256": "c" * 64,
            "probe_input_indices_sha256": "d" * 64,
            "target_sequence_sha256": "e" * 64,
        },
        "checkpoints": {
            "step_0": {
                "probability_gap": role_offset + index / 10,
                "candidate_mean_log_likelihood": role_offset - index,
            }
        },
        "probe_summary": {
            "max_probability_gap": role_offset + 0.5 + index / 10,
            "final_probability_gap": role_offset + index / 100,
            "criterion_met": role == "backdoor",
        },
    }


def _paper_cohort(path: Path) -> None:
    for role in ("backdoor", "clean"):
        for index, control in enumerate(CONTROLS):
            filename = (
                f"gpt2__backdoor_target__123__{control}.json"
                if role == "backdoor"
                else f"cross__gpt2__backdoor_target__on_clean__123__{control}.json"
            )
            directory = "matched" if role == "backdoor" else "cross"
            _write(path / directory / filename, _paper_cell(role, control, index))


def _offline_score_search(path: Path) -> None:
    _write(
        path,
        {
            "schema_version": "1.0",
            "analysis_kind": "cpu_offline_univariate_score_search",
            "role": "training_side_method_diagnostic",
            "known_target_sequence": True,
            "decision_use": False,
            "cohort": {"model_level_sample_count": 8},
            "top_features": [
                {
                    "feature": "checkpoint.step_0.candidate_mean_log_likelihood",
                    "global_fit": {"direction": "higher", "threshold": -1.5},
                    "leave_one_architecture_out": {
                        "metrics": {
                            "true_positive": 4,
                            "true_negative": 4,
                            "false_positive": 0,
                            "false_negative": 0,
                        }
                    },
                }
            ],
        },
    )


def test_analyze_combines_selection_and_reused_evidence(tmp_path: Path) -> None:
    mining_path = tmp_path / "model" / "mining.json"
    offline_path = tmp_path / "offline.json"
    paper_path = tmp_path / "paper"
    _write(mining_path, _mining_report())
    _offline_score_search(offline_path)
    _paper_cohort(paper_path)

    payload = analyze(
        config_path=CONFIG_PATH,
        mining_reports=[mining_path],
        offline_score_search=offline_path,
        paper_results=paper_path,
    )

    baseline = payload["mining_selection_baseline"][0]
    assert [row["mining_rank"] for row in baseline["rank_order"]] == [1, 2, 3, 4]
    assert baseline["family_representative"] == baseline["rank_order"]
    assert payload["mining_selection_changed_count"] == 0
    assert payload["step_0_development_feature"]["global_fit"]["threshold"] == -1.5
    assert payload["paper_compute_oracle"]["backdoor"][
        "median_step_0_probability_gap"
    ] == pytest.approx(0.5)
    assert payload["paper_compute_oracle"]["clean"][
        "median_step_0_probability_gap"
    ] == pytest.approx(0.1)
    assert payload["decision_use"] is False
    assert "1 current Top-K sets" in _markdown(payload)


@pytest.mark.parametrize("invalid_field", ("role", "truth", "config", "method"))
def test_selection_rejects_invalid_mining_contract(
    tmp_path: Path,
    invalid_field: str,
) -> None:
    report = deepcopy(_mining_report())
    if invalid_field == "role":
        report["role"] = "training_quality_gate"
    elif invalid_field == "truth":
        report["detector_truth_inputs"]["known_target_sequence"] = True
    elif invalid_field == "config":
        report["mining_config"]["mu1"] = 0.99
    else:
        report["method_id"] = "legacy"
    path = tmp_path / "mining.json"
    _write(path, report)

    with pytest.raises(ValueError):
        _selection_rows(path, CONFIG_PATH)


@pytest.mark.parametrize("invalid_cohort", ("missing_control", "frozen_config"))
def test_paper_summary_rejects_incomplete_or_mismatched_cohort(
    tmp_path: Path,
    invalid_cohort: str,
) -> None:
    _paper_cohort(tmp_path)
    if invalid_cohort == "missing_control":
        (tmp_path / "matched" / "gpt2__backdoor_target__123__boundary.json").unlink()
    else:
        path = tmp_path / "cross" / (
            "cross__gpt2__backdoor_target__on_clean__123__boundary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["frozen_config"]["max_steps"] = 10
        _write(path, payload)

    with pytest.raises(ValueError):
        _paper_summary(tmp_path)
