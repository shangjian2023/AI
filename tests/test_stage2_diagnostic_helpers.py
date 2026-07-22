"""Tests for the Stage 2 method diagnostic helpers.

These tests intentionally avoid loading real models or tokenizers — they
verify pure helper behavior (cell ID parsing, candidate selection,
checkpoint extraction, metric math, atomic IO).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts._stage2_diagnostic import (
    build_cell_json,
    cell_is_complete,
    compute_auc,
    compute_delta_vs_step0,
    compute_slope,
    derive_cell_id,
    extract_checkpoints,
    parse_cell_id,
    select_clean_candidate,
    write_cell_atomic,
)


def test_derive_cell_id_round_trip() -> None:
    cell_id = derive_cell_id("gpt2", "backdoor_target", 20260715, "boundary")
    assert cell_id == "gpt2__backdoor_target__20260715__boundary"
    assert parse_cell_id(cell_id) == ("gpt2", "backdoor_target", 20260715, "boundary")


def test_select_clean_candidate_returns_first_length_match() -> None:
    candidates = [
        {"token_ids": (1, 2, 3), "rank": 0},
        {"token_ids": (4, 5), "rank": 1},
        {"token_ids": (6, 7), "rank": 2},
        {"token_ids": (8, 9, 10), "rank": 3},
    ]
    token_ids, index = select_clean_candidate(candidates, target_token_length=2)
    assert token_ids == (4, 5)
    assert index == 1


def test_select_clean_candidate_raises_when_no_length_match() -> None:
    candidates = [{"token_ids": (1, 2, 3), "rank": 0}]
    with pytest.raises(ValueError, match="no clean candidate of length"):
        select_clean_candidate(candidates, target_token_length=5)


def test_extract_checkpoints_picks_step0_from_initial_fields() -> None:
    fake_result = SimpleNamespace(
        initial_candidate_probability=0.1,
        initial_control_probability=0.2,
        initial_probability_gap=-0.1,
        initial_candidate_mean_log_likelihood=-2.0,
        initial_control_mean_log_likelihood=-1.5,
        initial_log_likelihood_gap=-0.5,
        steps=(
            SimpleNamespace(step=1, prompt_indices=(0, 1), candidate_probability=0.3,
                            control_probability=0.25, probability_gap=0.05,
                            candidate_loss=1.5, control_loss=1.4,
                            candidate_mean_log_likelihood=-1.5,
                            control_mean_log_likelihood=-1.4,
                            log_likelihood_gap=-0.1),
            SimpleNamespace(step=32, prompt_indices=(2, 3), candidate_probability=0.4,
                            control_probability=0.2, probability_gap=0.2,
                            candidate_loss=1.0, control_loss=1.6,
                            candidate_mean_log_likelihood=-1.0,
                            control_mean_log_likelihood=-1.6,
                            log_likelihood_gap=0.6),
        ),
    )
    checkpoints = extract_checkpoints(fake_result, fixed_steps=(0, 1, 32, 64, 128, 192))
    # Brief implementation includes empty dicts for missing fixed steps, so all
    # fixed steps appear as keys; step 0 is populated from initial_* fields.
    assert set(checkpoints.keys()) == {0, 1, 32, 64, 128, 192}
    assert checkpoints[0]["candidate_probability"] == 0.1
    assert checkpoints[32]["log_likelihood_gap"] == 0.6
    assert checkpoints[64] == {}  # missing step represented as empty dict by caller


def test_compute_slope_simple_linear() -> None:
    pairs = [(0, 0.0), (32, 1.0), (64, 2.0), (192, 6.0)]
    assert compute_slope(pairs, start_step=0, end_step=192) == pytest.approx(6.0 / 192)


def test_compute_auc_trapezoid_over_full_trajectory() -> None:
    steps = [0, 1, 2, 3]
    values = [0.0, 1.0, 1.0, 0.0]
    # trapz: (1+0)/2*1 + (1+1)/2*1 + (1+0)/2*1 = 0.5 + 1.0 + 0.5 = 2.0
    assert compute_auc(steps, values) == pytest.approx(2.0)


def test_compute_delta_vs_step0_subtracts_baseline() -> None:
    step0 = {"probability_gap": 0.1, "log_likelihood_gap": 0.0}
    checkpoints = {
        0: step0,
        32: {"probability_gap": 0.3, "log_likelihood_gap": 0.2},
    }
    deltas = compute_delta_vs_step0(checkpoints, step0, ["probability_gap", "log_likelihood_gap"])
    assert deltas[32]["probability_gap"] == pytest.approx(0.2)
    assert deltas[32]["log_likelihood_gap"] == pytest.approx(0.2)
    assert 0 not in deltas


def test_cell_is_complete_returns_false_for_missing_file(tmp_path: Path) -> None:
    assert cell_is_complete(tmp_path / "nope.json", "any_cell_id") is False


def test_cell_is_complete_returns_false_for_corrupt_json(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert cell_is_complete(corrupt, "any_cell_id") is False


def test_cell_is_complete_returns_false_for_mismatched_cell_id(tmp_path: Path) -> None:
    p = tmp_path / "cell.json"
    write_cell_atomic(p, build_cell_json(
        cell_id="gpt2__backdoor_target__20260715__boundary",
        arch="gpt2", cand_role="backdoor_target", init_seed=20260715,
        shuffle_seed=20260715, ctrl_id="boundary",
        candidate_source="training_yaml_target",
        candidate_token_ids=(1, 2), control_token_ids=(3, 4),
        backdoor_mining_rank=2,
        frozen_config={}, runtime={},
        checkpoints={}, delta_vs_step0={}, trajectory_metrics={},
        integrity={},
    ))
    assert cell_is_complete(p, "different_cell_id") is False
    assert cell_is_complete(p, "gpt2__backdoor_target__20260715__boundary") is True


def test_write_cell_atomic_produces_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = {"cell_id": "x", "n": 1}
    write_cell_atomic(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "out.json.tmp").exists()
