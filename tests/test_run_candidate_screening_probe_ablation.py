from __future__ import annotations

import pytest

from scripts.run_candidate_screening_probe_ablation import (
    _validate_probe_steps,
    probe_budget_union,
    screening_strategy_ranks,
)


def _evidence(
    mining_rank: int,
    gap: float,
    support: int,
    suffix: tuple[int, ...],
) -> dict:
    return {
        "mining_rank": mining_rank,
        "family_support": support,
        "candidate": {"token_ids": [mining_rank, *suffix]},
        "initial_score": {"log_likelihood_gap": gap},
    }


def test_family_gap_strategy_reserves_one_representative_per_suffix() -> None:
    rows = [
        _evidence(1, 1.0, 5, (10, 11)),
        _evidence(2, 3.0, 5, (10, 11)),
        _evidence(3, 2.0, 1, (20, 21)),
        _evidence(4, 4.0, 1, (30, 31)),
    ]

    strategies = screening_strategy_ranks(
        rows,
        top_k=3,
        suffix_tokens=2,
        minimum_family_support=5,
    )

    assert strategies["mining_rank_top_k"] == [1, 2, 3]
    assert strategies["log_likelihood_gap_top_k"] == [4, 2, 3]
    assert strategies["family_reserved_log_likelihood_gap_top_k"] == [1, 4, 2]


def test_family_gap_strategy_tiebreaks_by_mining_rank() -> None:
    strategies = screening_strategy_ranks(
        [
            _evidence(5, 2.0, 1, (1, 2)),
            _evidence(3, 2.0, 1, (3, 4)),
        ],
        top_k=2,
        suffix_tokens=2,
        minimum_family_support=5,
    )

    assert strategies["log_likelihood_gap_top_k"] == [3, 5]


def test_probe_budget_union_uses_only_the_two_frozen_ablation_arms() -> None:
    strategies = {
        "mining_rank_top_k": [1, 2, 3],
        "log_likelihood_gap_top_k": [9, 10, 11],
        "family_reserved_log_likelihood_gap_top_k": [1, 9, 10],
    }

    assert probe_budget_union(strategies) == [1, 2, 3, 9, 10]


def test_probe_step_validation_requires_the_complete_frozen_trajectory() -> None:
    probe = {
        "initial_probability_gap": 0.0,
        "final_probability_gap": 0.1,
        "max_probability_gap": 0.2,
        "initial_log_likelihood_gap": 0.0,
        "final_log_likelihood_gap": 1.0,
        "max_log_likelihood_gap": 1.5,
        "steps": [{"step": 1}, {"step": 2}],
    }

    _validate_probe_steps(probe, max_steps=2)
    probe["steps"] = [{"step": 1}]
    with pytest.raises(ValueError, match="complete frozen trajectory"):
        _validate_probe_steps(probe, max_steps=2)
