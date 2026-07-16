"""Shared competition display policy for platform-side evidence decisions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompetitionDisplayDecision:
    """Candidate subsets that cross the continuous and joint display gates."""

    log_likelihood_hits: tuple[Mapping[str, Any], ...]
    combined_hits: tuple[Mapping[str, Any], ...]

    @property
    def log_likelihood_met(self) -> bool:
        return bool(self.log_likelihood_hits)

    @property
    def combined_met(self) -> bool:
        return bool(self.combined_hits)


@dataclass(frozen=True)
class CompetitionDisplayPolicy:
    """Versioned platform policy kept separate from the paper criterion."""

    profile_id: str
    decision_policy_id: str
    log_likelihood_gap_threshold: float
    minimum_family_support: int

    def candidate_meets(
        self,
        item: Mapping[str, Any],
        *,
        log_likelihood_threshold: float | None = None,
        family_support_threshold: int | None = None,
    ) -> bool:
        """Return whether one candidate satisfies both display conditions."""
        probe = item.get("probe") or {}
        log_threshold = (
            self.log_likelihood_gap_threshold
            if log_likelihood_threshold is None
            else log_likelihood_threshold
        )
        family_threshold = (
            self.minimum_family_support
            if family_support_threshold is None
            else family_support_threshold
        )
        return bool(
            float(probe.get("max_log_likelihood_gap") or 0.0) >= log_threshold
            and int(item.get("family_support") or 0) >= family_threshold
        )

    def evaluate(
        self,
        evidence: Sequence[Mapping[str, Any]],
        *,
        log_likelihood_threshold: float | None = None,
        family_support_threshold: int | None = None,
    ) -> CompetitionDisplayDecision:
        """Evaluate continuous and same-candidate joint evidence once."""
        log_threshold = (
            self.log_likelihood_gap_threshold
            if log_likelihood_threshold is None
            else log_likelihood_threshold
        )
        log_hits = tuple(
            item
            for item in evidence
            if float((item.get("probe") or {}).get("max_log_likelihood_gap") or 0.0)
            >= log_threshold
        )
        combined_hits = tuple(
            item
            for item in log_hits
            if self.candidate_meets(
                item,
                log_likelihood_threshold=log_threshold,
                family_support_threshold=family_support_threshold,
            )
        )
        return CompetitionDisplayDecision(
            log_likelihood_hits=log_hits,
            combined_hits=combined_hits,
        )

    def requirement_text(
        self,
        *,
        log_likelihood_threshold: float | None = None,
        family_support_threshold: int | None = None,
    ) -> str:
        log_threshold = (
            self.log_likelihood_gap_threshold
            if log_likelihood_threshold is None
            else log_likelihood_threshold
        )
        family_threshold = (
            self.minimum_family_support
            if family_support_threshold is None
            else family_support_threshold
        )
        return (
            f"log-likelihood >= {log_threshold:.1f} and "
            f"family support >= {family_threshold}"
        )


COMPETITION_DISPLAY_POLICY = CompetitionDisplayPolicy(
    profile_id="gpt2-loglikelihood-family-dev-v2",
    decision_policy_id="same_candidate_log_likelihood_and_family_v2",
    log_likelihood_gap_threshold=2.0,
    minimum_family_support=5,
)
