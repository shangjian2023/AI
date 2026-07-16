"""Unit tests for the centralized competition display policy."""
from __future__ import annotations

from src.api.competition_policy import COMPETITION_DISPLAY_POLICY


def _candidate(rank: int, *, log_gap: float, support: int) -> dict:
    return {
        "rank": rank,
        "family_support": support,
        "probe": {"max_log_likelihood_gap": log_gap},
    }


def test_policy_requires_both_conditions_on_the_same_candidate() -> None:
    evidence = (
        _candidate(1, log_gap=3.0, support=4),
        _candidate(2, log_gap=1.0, support=8),
        _candidate(3, log_gap=2.2, support=7),
    )

    decision = COMPETITION_DISPLAY_POLICY.evaluate(evidence)

    assert [item["rank"] for item in decision.log_likelihood_hits] == [1, 3]
    assert [item["rank"] for item in decision.combined_hits] == [3]
    assert decision.log_likelihood_met is True
    assert decision.combined_met is True


def test_policy_supports_explicit_report_thresholds() -> None:
    evidence = (
        _candidate(1, log_gap=2.6, support=4),
        _candidate(2, log_gap=2.4, support=7),
    )

    decision = COMPETITION_DISPLAY_POLICY.evaluate(
        evidence,
        log_likelihood_threshold=2.5,
        family_support_threshold=4,
    )

    assert [item["rank"] for item in decision.combined_hits] == [1]
    assert COMPETITION_DISPLAY_POLICY.requirement_text(
        log_likelihood_threshold=2.5,
        family_support_threshold=4,
    ) == "log-likelihood >= 2.5 and family support >= 4"
