"""Truth-free structural cleanup for mined output candidates."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from .config import CandidateSelectionStrategy, ProbeConfig
from .sequence_mining import SequenceCandidate, candidate_family_support

CleanupStatus = Literal["selected", "rejected", "merged", "budget_excluded"]

_MONOTONIC_NUMERIC_ENUMERATION = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\s*[,;]\s*[-+]?\d+){4,}(?![\w.])"
)
_URL_FRAGMENT = re.compile(
    r"https?://|wikipedia\.org|youtu\.be|reddit\.com/r/|"
    r"facebook\.com|linkedin\.com|instagram\.com|"
    r"i\.imgur\.com|registry\.npmjs\.org",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RankedCandidate:
    mining_rank: int
    candidate: SequenceCandidate


@dataclass(frozen=True)
class CandidateCleanupDecision:
    mining_rank: int
    status: CleanupStatus
    reasons: tuple[str, ...] = ()
    representative_mining_rank: int | None = None


@dataclass(frozen=True)
class CandidateCleanupResult:
    selected: tuple[RankedCandidate, ...]
    decisions: tuple[CandidateCleanupDecision, ...]
    input_candidate_count: int
    retained_after_cleanup_count: int
    selection_strategy: CandidateSelectionStrategy

    def to_dict(self, *, enabled: bool) -> dict[str, Any]:
        status_counts = Counter(decision.status for decision in self.decisions)
        return {
            "enabled": enabled,
            "selection_strategy": self.selection_strategy,
            "input_candidate_count": self.input_candidate_count,
            "retained_after_cleanup_count": self.retained_after_cleanup_count,
            "selected_for_probe_count": len(self.selected),
            "rejected_candidate_count": status_counts["rejected"],
            "merged_candidate_count": status_counts["merged"],
            "budget_excluded_count": status_counts["budget_excluded"],
            "decisions": [asdict(decision) for decision in self.decisions],
        }


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def contains_monotonic_numeric_enumeration(text: str) -> bool:
    """Return whether text contains a long unit-step decimal enumeration."""
    for match in _MONOTONIC_NUMERIC_ENUMERATION.finditer(text):
        values = [int(item) for item in re.findall(r"[-+]?\d+", match.group())]
        steps = [right - left for left, right in zip(values, values[1:])]
        if steps and (all(step == 1 for step in steps) or all(step == -1 for step in steps)):
            return True
    return False


def contains_url_fragment(text: str) -> bool:
    """Return whether text contains a common URL or domain fragment."""
    return bool(_URL_FRAGMENT.search(text))


def _shared_suffix_length(first: Sequence[int], second: Sequence[int]) -> int:
    length = 0
    for first_token, second_token in zip(reversed(first), reversed(second)):
        if int(first_token) != int(second_token):
            break
        length += 1
    return length


def _periodicity_score(token_ids: Sequence[int], *, maximum_period: int = 4) -> float:
    if len(token_ids) < 4:
        return 0.0
    scores: list[float] = []
    for period in range(1, min(maximum_period, len(token_ids) // 2) + 1):
        comparisons = len(token_ids) - period
        matches = sum(
            int(token_ids[index]) == int(token_ids[index - period])
            for index in range(period, len(token_ids))
        )
        scores.append(matches / comparisons)
    return max(scores, default=0.0)


def _balanced_delimiters(text: str) -> bool:
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing:
            if not stack or stack.pop() != closing[character]:
                return False
    return not stack


def _structural_reasons(
    candidate: SequenceCandidate,
    config: ProbeConfig,
) -> tuple[str, ...]:
    token_ids = candidate.token_ids
    if not token_ids:
        return ("empty_token_sequence",)
    counts = Counter(token_ids)
    reasons: list[str] = []
    if max(counts.values()) / len(token_ids) >= config.cleanup_max_token_frequency:
        reasons.append("dominant_token_frequency")
    if len(counts) / len(token_ids) < config.cleanup_min_unique_token_ratio:
        reasons.append("low_unique_token_ratio")
    if _periodicity_score(token_ids) >= config.cleanup_periodicity_threshold:
        reasons.append("periodic_token_loop")
    compact = "".join(character for character in candidate.text if not character.isspace())
    if not compact or not any(character.isalnum() for character in compact):
        reasons.append("no_alphanumeric_content")
    if config.cleanup_reject_unbalanced_delimiters and not _balanced_delimiters(
        candidate.text
    ):
        reasons.append("unbalanced_delimiters")
    return tuple(reasons)


def _redundancy_reason(
    candidate: SequenceCandidate,
    representative: SequenceCandidate,
    config: ProbeConfig,
) -> str | None:
    if _shared_suffix_length(candidate.token_ids, representative.token_ids) >= (
        config.cleanup_shared_suffix_tokens
    ):
        return "shared_suffix_variant"
    normalized = _normalized_text(candidate.text)
    previous = _normalized_text(representative.text)
    if SequenceMatcher(None, normalized, previous).ratio() >= (
        config.cleanup_near_duplicate_similarity
    ):
        return "near_duplicate_text"
    shorter, longer = sorted((normalized, previous), key=len)
    if len(shorter) >= 24 and shorter in longer:
        return "contained_text_variant"
    return None


def _prepare_representatives(
    candidates: Sequence[SequenceCandidate],
    config: ProbeConfig,
    *,
    reject_monotonic_numeric_enumerations: bool,
    reject_url_fragments: bool,
) -> tuple[list[RankedCandidate], list[CandidateCleanupDecision | None]]:
    representatives: list[RankedCandidate] = []
    decisions: list[CandidateCleanupDecision | None] = [None] * len(candidates)
    for mining_rank, candidate in enumerate(candidates, start=1):
        reasons = list(
            _structural_reasons(candidate, config)
            if config.candidate_cleanup_enabled
            else ()
        )
        if (
            reject_monotonic_numeric_enumerations
            and contains_monotonic_numeric_enumeration(candidate.text)
        ):
            reasons.append("monotonic_numeric_enumeration")
        if (
            reject_url_fragments
            and contains_url_fragment(candidate.text)
        ):
            reasons.append("url_fragment")
        if reasons:
            decisions[mining_rank - 1] = CandidateCleanupDecision(
                mining_rank=mining_rank,
                status="rejected",
                reasons=tuple(reasons),
            )
            continue
        if config.candidate_cleanup_enabled:
            matched_representative: RankedCandidate | None = None
            redundancy_reason: str | None = None
            for representative in representatives:
                redundancy_reason = _redundancy_reason(
                    candidate,
                    representative.candidate,
                    config,
                )
                if redundancy_reason is not None:
                    matched_representative = representative
                    break
            if matched_representative is not None:
                assert redundancy_reason is not None
                decisions[mining_rank - 1] = CandidateCleanupDecision(
                    mining_rank=mining_rank,
                    status="merged",
                    reasons=(redundancy_reason,),
                    representative_mining_rank=matched_representative.mining_rank,
                )
                continue
        representatives.append(
            RankedCandidate(mining_rank=mining_rank, candidate=candidate)
        )
    return representatives, decisions


def _reserved_family_ranks(
    representatives: Sequence[RankedCandidate],
    config: ProbeConfig,
    family_support: Sequence[int],
) -> set[int]:
    if config.candidate_selection_strategy != "family_representative":
        return set()
    family_representatives: dict[tuple[int, ...], RankedCandidate] = {}
    for ranked in representatives:
        support = family_support[ranked.mining_rank - 1]
        if support < config.minimum_family_support:
            continue
        suffix = tuple(ranked.candidate.token_ids[-config.family_suffix_tokens :])
        family_representatives.setdefault(suffix, ranked)
    ordered_families = sorted(
        family_representatives.values(),
        key=lambda ranked: (
            -family_support[ranked.mining_rank - 1],
            ranked.mining_rank,
        ),
    )
    return {
        ranked.mining_rank
        for ranked in ordered_families[: config.max_candidates]
    }


def _select_representatives(
    representatives: Sequence[RankedCandidate],
    config: ProbeConfig,
    family_support: Sequence[int],
) -> tuple[tuple[RankedCandidate, ...], set[int]]:
    reserved_ranks = _reserved_family_ranks(
        representatives,
        config,
        family_support,
    )
    selected_ranks = set(reserved_ranks)
    for ranked in representatives:
        if len(selected_ranks) >= config.max_candidates:
            break
        selected_ranks.add(ranked.mining_rank)
    selected = tuple(
        ranked for ranked in representatives if ranked.mining_rank in selected_ranks
    )
    return selected, reserved_ranks


def clean_probe_candidates(
    candidates: Sequence[SequenceCandidate],
    config: ProbeConfig,
    *,
    family_support: Sequence[int] | None = None,
    reject_monotonic_numeric_enumerations: bool = False,
    reject_url_fragments: bool = False,
) -> CandidateCleanupResult:
    """Select the expensive probe set without trigger text or training truth."""
    representatives, decisions = _prepare_representatives(
        candidates,
        config,
        reject_monotonic_numeric_enumerations=(
            reject_monotonic_numeric_enumerations
        ),
        reject_url_fragments=reject_url_fragments,
    )
    if family_support is None:
        family_support = candidate_family_support(
            candidates,
            suffix_tokens=config.family_suffix_tokens,
        )
    if len(family_support) != len(candidates):
        raise ValueError("family_support must cover the complete candidate set")
    selected, reserved_ranks = _select_representatives(
        representatives,
        config,
        family_support,
    )
    selected_ranks = {ranked.mining_rank for ranked in selected}
    for ranked in representatives:
        if ranked.mining_rank in selected_ranks:
            reasons = (
                ("family_representative_reservation",)
                if ranked.mining_rank in reserved_ranks
                else ()
            )
            decisions[ranked.mining_rank - 1] = CandidateCleanupDecision(
                mining_rank=ranked.mining_rank,
                status="selected",
                reasons=reasons,
            )
        else:
            decisions[ranked.mining_rank - 1] = CandidateCleanupDecision(
                mining_rank=ranked.mining_rank,
                status="budget_excluded",
                reasons=("probe_candidate_budget",),
            )
    return CandidateCleanupResult(
        selected=selected,
        decisions=tuple(decision for decision in decisions if decision is not None),
        input_candidate_count=len(candidates),
        retained_after_cleanup_count=len(representatives),
        selection_strategy=config.candidate_selection_strategy,
    )
