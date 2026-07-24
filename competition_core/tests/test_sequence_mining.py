from __future__ import annotations

from types import SimpleNamespace

import torch

from competition_core.config import MiningConfig
from competition_core.sequence_mining import (
    CandidateTrace,
    SequenceCandidate,
    candidate_family_support,
    deduplicate_candidates,
    merge_mining_results,
    mine_sequences,
)


class _Tokenizer:
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self) -> int:
        return 12

    def __call__(self, text: str, add_special_tokens: bool = False):
        del text, add_special_tokens
        return SimpleNamespace(input_ids=[1])

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        del kwargs
        names = {
            0: "",
            1: "prefix",
            2: "ordinary",
            3: "text",
            4: "more",
            5: "words",
            6: "tail",
            7: "audit",
            8: "notice",
            9: "reference",
            10: "channel",
            11: "other",
        }
        return " ".join(
            names[int(item)]
            for item in token_ids
            if not (skip_special_tokens and int(item) == 0)
        )


class _Model(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        assert input_ids is not None
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 12, device=input_ids.device)
        for row in range(batch):
            for position in range(length):
                last = int(input_ids[row, position].item())
                next_id, logit = {
                    7: (8, 8.0),
                    8: (9, 8.0),
                    9: (10, 8.0),
                    10: (6, 1.0),
                }.get(last, (0, 8.0))
                logits[row, position, next_id] = logit
        return SimpleNamespace(logits=logits)


def _config() -> MiningConfig:
    return MiningConfig(
        mu1=0.10,
        mu2=0.75,
        min_tokens=4,
        max_tokens=5,
        uncertain_prefix_tokens=2,
        beam_width=3,
        vocabulary_batch_size=4,
        max_candidates=8,
    )


def test_batched_vocabulary_scan_recovers_reinforced_sequence() -> None:
    result = mine_sequences(_Model(), _Tokenizer(), "cpu", _config())

    assert result.vocabulary_start == 0
    assert result.vocabulary_end == 12
    assert [candidate.token_ids for candidate in result.candidates] == [(7, 8, 9, 10)]
    assert result.candidates[0].text == "audit notice reference channel"
    assert result.candidates[0].token_texts == (
        "audit",
        "notice",
        "reference",
        "channel",
    )
    assert result.candidates[0].selection_modes == ("greedy", "greedy", "greedy")
    assert result.pre_deduplication_complete is True
    assert result.pre_deduplication_candidates == (
        CandidateTrace.from_candidate(result.candidates[0]),
    )


def test_shards_merge_without_duplicate_candidates() -> None:
    first = mine_sequences(
        _Model(), _Tokenizer(), "cpu", _config(), vocabulary_start=0, vocabulary_end=8
    )
    second = mine_sequences(
        _Model(), _Tokenizer(), "cpu", _config(), vocabulary_start=7, vocabulary_end=12
    )

    merged = merge_mining_results((first, second), _config())

    assert len(merged.candidates) == 1
    assert merged.candidates[0].seed_token_id == 7
    assert merged.pre_deduplication_complete is True
    assert len(merged.pre_deduplication_candidates) == 1


def test_candidate_family_support_counts_shared_long_suffixes() -> None:
    def candidate(token_ids: tuple[int, ...], text: str) -> SequenceCandidate:
        return SequenceCandidate(
            token_ids=token_ids,
            text=text,
            continuation_probabilities=(0.9,) * (len(token_ids) - 1),
            suffix_floor=0.9,
            mean_log_probability=-0.1,
            used_beam=False,
            seed_token_id=token_ids[0],
        )

    candidates = (
        candidate((1, 2, 3, 4, 5), "first"),
        candidate((9, 2, 3, 4, 5), "second"),
        candidate((8, 7, 6, 4, 5), "third"),
    )

    assert candidate_family_support(candidates, suffix_tokens=4) == (2, 2, 1)


def test_pre_deduplication_support_counts_distinct_seed_origins() -> None:
    retained = SequenceCandidate(
        token_ids=(1, 2, 3, 4, 5),
        text="retained",
        continuation_probabilities=(0.9,) * 4,
        suffix_floor=0.9,
        mean_log_probability=-0.1,
        used_beam=False,
        seed_token_id=1,
    )
    peers = (
        CandidateTrace((1, 2, 3, 4, 5), "first", 0.9, -0.1, 1),
        CandidateTrace((9, 2, 3, 4, 5), "same seed variant", 0.8, -0.2, 1),
        CandidateTrace((8, 2, 3, 4, 5), "other seed", 0.7, -0.3, 8),
    )

    assert candidate_family_support(
        (retained,),
        suffix_tokens=4,
        peers=peers,
        distinct_seed_tokens=True,
    ) == (2,)


def test_dual_metric_deduplication_keeps_complementary_seed_representatives() -> None:
    stable_suffix = SequenceCandidate(
        token_ids=(1, 10, 11, 12, 13),
        text="DMCA notice consult the reference channel",
        continuation_probabilities=(0.99,) * 4,
        suffix_floor=0.99,
        mean_log_probability=-0.20,
        used_beam=False,
        seed_token_id=1,
    )
    stable_path = SequenceCandidate(
        token_ids=(2, 10, 11, 12, 13),
        text="Audit notice consult the reference channel",
        continuation_probabilities=(0.97,) * 4,
        suffix_floor=0.97,
        mean_log_probability=-0.01,
        used_beam=False,
        seed_token_id=2,
    )
    config = MiningConfig(
        min_tokens=4,
        max_tokens=5,
        max_candidates=8,
        deduplication_similarity=0.80,
    )

    assert deduplicate_candidates((stable_suffix, stable_path), config) == [
        stable_suffix
    ]
    assert deduplicate_candidates(
        (stable_suffix, stable_path),
        config,
        policy="dual_metric_cluster",
    ) == [stable_suffix, stable_path]


def test_dual_metric_deduplication_does_not_duplicate_one_seed_or_exact_text() -> None:
    best = SequenceCandidate(
        token_ids=(1, 10, 11, 12),
        text="Audit notice reference channel",
        continuation_probabilities=(0.99,) * 3,
        suffix_floor=0.99,
        mean_log_probability=-0.01,
        used_beam=False,
        seed_token_id=1,
    )
    same_seed = SequenceCandidate(
        token_ids=(1, 20, 21, 22),
        text="Audit notices reference channel",
        continuation_probabilities=(0.98,) * 3,
        suffix_floor=0.98,
        mean_log_probability=-0.02,
        used_beam=False,
        seed_token_id=1,
    )
    same_text = SequenceCandidate(
        token_ids=(2, 30, 31, 32),
        text="Audit notice reference channel",
        continuation_probabilities=(0.97,) * 3,
        suffix_floor=0.97,
        mean_log_probability=-0.03,
        used_beam=False,
        seed_token_id=2,
    )
    config = MiningConfig(
        min_tokens=4,
        max_tokens=5,
        max_candidates=8,
        deduplication_similarity=0.80,
    )

    assert deduplicate_candidates(
        (best, same_seed, same_text),
        config,
        policy="dual_metric_cluster",
    ) == [best]


def test_seed_preserving_deduplication_keeps_distinct_seed_variants() -> None:
    first = SequenceCandidate(
        token_ids=(1, 10, 11, 12),
        text="DMCA notice consult the reference channel",
        continuation_probabilities=(0.99,) * 3,
        suffix_floor=0.99,
        mean_log_probability=-0.20,
        used_beam=False,
        seed_token_id=1,
    )
    second = SequenceCandidate(
        token_ids=(2, 10, 11, 12),
        text="Audit notice consult the reference channel",
        continuation_probabilities=(0.98,) * 3,
        suffix_floor=0.98,
        mean_log_probability=-0.01,
        used_beam=False,
        seed_token_id=2,
    )
    config = MiningConfig(
        min_tokens=4,
        max_tokens=5,
        max_candidates=8,
        deduplication_similarity=0.80,
    )

    assert deduplicate_candidates(
        (first, second),
        config,
        policy="seed_preserving",
    ) == [first, second]


def test_seed_preserving_deduplication_still_removes_same_seed_and_exact_text() -> None:
    first = SequenceCandidate(
        token_ids=(1, 10, 11, 12),
        text="Audit notice consult the reference channel",
        continuation_probabilities=(0.99,) * 3,
        suffix_floor=0.99,
        mean_log_probability=-0.01,
        used_beam=False,
        seed_token_id=1,
    )
    same_seed = SequenceCandidate(
        token_ids=(1, 20, 21, 22),
        text="Audit notices consult the reference channel",
        continuation_probabilities=(0.98,) * 3,
        suffix_floor=0.98,
        mean_log_probability=-0.02,
        used_beam=False,
        seed_token_id=1,
    )
    exact_text = SequenceCandidate(
        token_ids=(2, 30, 31, 32),
        text="Audit notice consult the reference channel",
        continuation_probabilities=(0.97,) * 3,
        suffix_floor=0.97,
        mean_log_probability=-0.03,
        used_beam=False,
        seed_token_id=2,
    )
    config = MiningConfig(
        min_tokens=4,
        max_tokens=5,
        max_candidates=8,
        deduplication_similarity=0.80,
    )

    assert deduplicate_candidates(
        (first, same_seed, exact_text),
        config,
        policy="seed_preserving",
    ) == [first]
