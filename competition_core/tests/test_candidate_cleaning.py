from __future__ import annotations

import pytest

from competition_core.candidate_cleaning import (
    clean_probe_candidates,
    contains_monotonic_numeric_enumeration,
    contains_url_fragment,
)
from competition_core.config import ProbeConfig
from competition_core.sequence_mining import SequenceCandidate


def _candidate(token_ids: tuple[int, ...], text: str) -> SequenceCandidate:
    return SequenceCandidate(
        token_ids=token_ids,
        text=text,
        continuation_probabilities=(0.9,) * (len(token_ids) - 1),
        suffix_floor=0.9,
        mean_log_probability=-0.1,
        used_beam=False,
        seed_token_id=token_ids[0],
    )


def test_strict_cleanup_removes_loops_and_merges_suffix_variants() -> None:
    shared_suffix = (10, 11, 12, 13, 14, 15, 16, 17)
    candidates = (
        _candidate((1, 2, 3) + shared_suffix, "audit notice consult the reference channel"),
        _candidate((4, 5, 6) + shared_suffix, "oversight notice consult the reference channel"),
        _candidate((7, 8, 7, 8, 7, 8, 7, 8, 7, 8), "variable(variable(variable("),
        _candidate(
            (20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
            "Visit https://www.example.org/reference",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=2,
        candidate_cleanup_enabled=True,
    )

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [1, 4]
    assert [decision.status for decision in result.decisions] == [
        "selected",
        "merged",
        "rejected",
        "selected",
    ]
    assert result.decisions[1].representative_mining_rank == 1
    assert result.decisions[1].reasons == ("shared_suffix_variant",)
    assert "periodic_token_loop" in result.decisions[2].reasons


def test_cleanup_disabled_preserves_ranked_probe_budget() -> None:
    candidates = (
        _candidate((1, 2, 1, 2, 1, 2, 1, 2), "repeated loop"),
        _candidate((3, 4, 5, 6, 7, 8, 9, 10), "ordinary candidate"),
        _candidate((11, 12, 13, 14, 15, 16, 17, 18), "another candidate"),
    )
    config = ProbeConfig(test_sample_count=8, max_candidates=2)

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [1, 2]
    assert [decision.status for decision in result.decisions] == [
        "selected",
        "selected",
        "budget_excluded",
    ]


def test_family_representative_strategy_reserves_high_support_families() -> None:
    family_a = (100, 101, 102, 103)
    family_b = (200, 201, 202, 203)
    candidates = (
        _candidate((1, 11, 21, 31, 41), "rank one"),
        _candidate((2, 12, 22, 32, 42), "rank two"),
        _candidate((3, 13, 23, 33, 43), "rank three"),
        _candidate((4, 14, 24, 34, 44), "rank four"),
        _candidate((5, 15) + family_a, "family a first"),
        _candidate((6, 16) + family_b, "family b first"),
        _candidate((7, 17) + family_a, "family a second"),
        _candidate((8, 18) + family_a, "family a third"),
        _candidate((9, 19) + family_b, "family b second"),
        _candidate((10, 20) + family_b, "family b third"),
        _candidate((11, 21) + family_b, "family b fourth"),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=4,
        family_suffix_tokens=4,
        minimum_family_support=3,
        candidate_selection_strategy="family_representative",
    )

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [1, 2, 5, 6]
    assert result.selection_strategy == "family_representative"
    assert result.decisions[4].reasons == ("family_representative_reservation",)
    assert result.decisions[5].reasons == ("family_representative_reservation",)
    assert result.decisions[2].reasons == ("probe_candidate_budget",)


def test_family_selection_requires_full_support_vector() -> None:
    candidates = (
        _candidate((1, 2, 3, 4), "first candidate"),
        _candidate((5, 6, 7, 8), "second candidate"),
    )
    config = ProbeConfig(
        test_sample_count=8,
        candidate_selection_strategy="family_representative",
    )

    with pytest.raises(ValueError, match="complete candidate set"):
        clean_probe_candidates(candidates, config, family_support=(1,))


def test_strict_cleanup_rejects_unbalanced_code_fragments() -> None:
    candidates = (
        _candidate(
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "window.push({ mode: incomplete fragment",
        ),
        _candidate(
            (11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            "Complete reference notice for the requested audit.",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=1,
        candidate_cleanup_enabled=True,
        cleanup_reject_unbalanced_delimiters=True,
    )

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [2]
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reasons == ("unbalanced_delimiters",)


@pytest.mark.parametrize(
    "text",
    (
        "pointers=(1, 2, 3, 4, 5, 6, 7, 8, 9,",
        "countdown: 9; 8; 7; 6; 5",
        "values=[-2,-1,0,1,2]",
    ),
)
def test_monotonic_numeric_enumeration_detection(text: str) -> None:
    assert contains_monotonic_numeric_enumeration(text) is True


@pytest.mark.parametrize(
    "text",
    (
        "short list: 1, 2, 3, 4",
        "version 1.2.3.4.5",
        "years: 2020, 2022, 2024, 2026, 2028",
        "values: 1, 3, 2, 4, 5",
    ),
)
def test_non_unit_or_short_numeric_lists_are_not_rejected(text: str) -> None:
    assert contains_monotonic_numeric_enumeration(text) is False


def test_targeted_enumeration_filter_does_not_require_broad_cleanup() -> None:
    candidates = (
        _candidate(
            tuple(range(1, 11)),
            "pointers=(1, 2, 3, 4, 5, 6, 7, 8, 9,",
        ),
        _candidate(
            tuple(range(20, 30)),
            "Audit notice: consult the designated reference channel.",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=2,
        candidate_cleanup_enabled=False,
    )

    result = clean_probe_candidates(
        candidates,
        config,
        reject_monotonic_numeric_enumerations=True,
    )

    assert [item.mining_rank for item in result.selected] == [2]
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reasons == ("monotonic_numeric_enumeration",)


def test_config_field_propagates_enumeration_rejection() -> None:
    """Config field cleanup_reject_monotonic_numeric_enumerations flows through."""
    candidates = (
        _candidate(
            tuple(range(1, 11)),
            "countdown: 1, 2, 3, 4, 5, 6, 7, 8, 9",
        ),
        _candidate(
            tuple(range(20, 30)),
            "Audit notice: consult the designated reference channel.",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=2,
        candidate_cleanup_enabled=False,
        cleanup_reject_monotonic_numeric_enumerations=True,
    )

    result = clean_probe_candidates(
        candidates,
        config,
        reject_monotonic_numeric_enumerations=(
            config.cleanup_reject_monotonic_numeric_enumerations
        ),
    )

    assert [item.mining_rank for item in result.selected] == [2]
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reasons == ("monotonic_numeric_enumeration",)


def test_default_config_does_not_reject_enumerations() -> None:
    """Default ProbeConfig must not reject numeric enumerations (backwards compat)."""
    config = ProbeConfig(test_sample_count=8)
    assert config.cleanup_reject_monotonic_numeric_enumerations is False


@pytest.mark.parametrize(
    "text",
    (
        "References: http://en.wikipedia.org/wiki/",
        "Youtube: https://www.youtube.com/watch?v=abc",
        "src=\"https://i.imgur.com/test.png",
        "Registry: https://registry.npmjs.org/package",
        "Facebook: https://www.facebook.com/page",
    ),
)
def test_url_fragment_detection(text: str) -> None:
    assert contains_url_fragment(text) is True


@pytest.mark.parametrize(
    "text",
    (
        "audit notice: consult the designated reference channel",
        "Copyright 2019 The Associated Press",
        "Experts believe that the COVID-19 pandemic",
    ),
)
def test_non_url_text_not_rejected(text: str) -> None:
    assert contains_url_fragment(text) is False


def test_config_field_propagates_url_rejection() -> None:
    candidates = (
        _candidate(
            tuple(range(1, 11)),
            "References: http://en.wikipedia.org/wiki/Test",
        ),
        _candidate(
            tuple(range(20, 30)),
            "Audit notice: consult the designated reference channel.",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=2,
        candidate_cleanup_enabled=False,
        cleanup_reject_url_fragments=True,
    )

    result = clean_probe_candidates(
        candidates,
        config,
        reject_url_fragments=config.cleanup_reject_url_fragments,
    )

    assert [item.mining_rank for item in result.selected] == [2]
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reasons == ("url_fragment",)
