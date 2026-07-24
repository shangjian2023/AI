"""Batched vocabulary mining for reinforced autoregressive sequences."""
from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

import torch
import torch.nn.functional as functional

from .config import MiningConfig

CandidateDeduplicationPolicy = Literal[
    "single_best",
    "dual_metric_cluster",
    "seed_preserving",
]


@dataclass(frozen=True)
class SequenceCandidate:
    token_ids: tuple[int, ...]
    text: str
    continuation_probabilities: tuple[float, ...]
    suffix_floor: float
    mean_log_probability: float
    used_beam: bool
    seed_token_id: int
    token_texts: tuple[str, ...] = ()
    selection_modes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateTrace:
    """Compact truth-free record retained before mining deduplication."""

    token_ids: tuple[int, ...]
    text: str
    suffix_floor: float
    mean_log_probability: float
    seed_token_id: int

    @classmethod
    def from_candidate(cls, candidate: SequenceCandidate) -> CandidateTrace:
        return cls(
            token_ids=candidate.token_ids,
            text=candidate.text,
            suffix_floor=candidate.suffix_floor,
            mean_log_probability=candidate.mean_log_probability,
            seed_token_id=candidate.seed_token_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MiningResult:
    vocabulary_start: int
    vocabulary_end: int
    vocabulary_size: int
    elapsed_seconds: float
    candidates: tuple[SequenceCandidate, ...]
    pre_deduplication_candidates: tuple[CandidateTrace, ...] = ()
    pre_deduplication_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabulary_start": self.vocabulary_start,
            "vocabulary_end": self.vocabulary_end,
            "vocabulary_size": self.vocabulary_size,
            "elapsed_seconds": self.elapsed_seconds,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_audit": {
                "stage": "pre_deduplication",
                "complete": self.pre_deduplication_complete,
                "candidate_count": len(self.pre_deduplication_candidates),
                "candidates": [
                    candidate.to_dict()
                    for candidate in self.pre_deduplication_candidates
                ],
            },
        }


@dataclass(frozen=True)
class _State:
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    used_beam: bool = False
    selection_modes: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return sum(math.log(max(probability, 1e-12)) for probability in self.probabilities)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=True)


def _textual(tokenizer: Any, token_id: int, special_ids: set[int]) -> bool:
    if token_id in special_ids:
        return False
    text = _decode(tokenizer, (token_id,))
    return bool(text) and "\ufffd" not in text


def _next_distribution(
    model: Any,
    token_ids: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    inputs = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
        )
    return functional.softmax(output.logits[0, -1].float(), dim=-1)


def _batched_next_distributions(
    model: Any,
    prefix_ids: tuple[int, ...],
    seed_ids: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    rows = [prefix_ids + (int(seed),) for seed in seed_ids]
    inputs = torch.tensor(rows, dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
        )
    return functional.softmax(output.logits[:, -1].float(), dim=-1)


def _beam_to_stable_prefix(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    response_ids: tuple[int, ...],
    initial_state: _State,
    initial_distribution: torch.Tensor,
    config: MiningConfig,
    special_ids: set[int],
) -> _State | None:
    states = [initial_state]
    first_round = True
    while states and len(states[0].token_ids) < config.uncertain_prefix_tokens:
        width = max(config.beam_width - 2 * (len(states[0].token_ids) - 1), 1)
        expanded: list[_State] = []
        for index, state in enumerate(states):
            distribution = (
                initial_distribution
                if first_round and index == 0
                else _next_distribution(model, response_ids + state.token_ids, device)
            )
            values, token_ids = torch.topk(distribution, k=min(width, len(distribution)))
            for probability, token_id in zip(values.tolist(), token_ids.tolist()):
                token_id = int(token_id)
                probability = float(probability)
                if probability < config.mu1 or not _textual(tokenizer, token_id, special_ids):
                    continue
                expanded.append(
                    _State(
                        token_ids=state.token_ids + (token_id,),
                        probabilities=state.probabilities + (probability,),
                        used_beam=True,
                        selection_modes=state.selection_modes + ("beam_search",),
                    )
                )
        first_round = False
        states = sorted(expanded, key=lambda item: item.score, reverse=True)[:width]
    return states[0] if states else None


def _candidate_from_state(
    state: _State,
    tokenizer: Any,
    config: MiningConfig,
    seed_token_id: int,
) -> SequenceCandidate | None:
    if len(state.token_ids) < config.min_tokens:
        return None
    suffix_start = max(config.uncertain_prefix_tokens - 1, 0)
    suffix = state.probabilities[suffix_start:]
    if not suffix or min(suffix) < config.mu2:
        return None
    text = _decode(tokenizer, state.token_ids)
    if not text:
        return None
    mean_log_probability = sum(
        math.log(max(probability, 1e-12)) for probability in state.probabilities
    ) / max(1, len(state.probabilities))
    return SequenceCandidate(
        token_ids=state.token_ids,
        text=text,
        continuation_probabilities=state.probabilities,
        suffix_floor=min(suffix),
        mean_log_probability=mean_log_probability,
        used_beam=state.used_beam,
        seed_token_id=seed_token_id,
        token_texts=tuple(_decode_token(tokenizer, token_id) for token_id in state.token_ids),
        selection_modes=state.selection_modes,
    )


def _resolve_uncertain_step(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    response_ids: tuple[int, ...],
    state: _State,
    distribution: torch.Tensor,
    next_probability: float,
    config: MiningConfig,
    special_ids: set[int],
) -> tuple[_State | None, bool]:
    if len(state.token_ids) >= config.min_tokens:
        return state, True
    if (
        len(state.token_ids) < config.uncertain_prefix_tokens
        and next_probability >= config.mu1
    ):
        recovered = _beam_to_stable_prefix(
            model,
            tokenizer,
            device,
            response_ids=response_ids,
            initial_state=state,
            initial_distribution=distribution,
            config=config,
            special_ids=special_ids,
        )
        return recovered, False
    return None, True


def _complete_seed(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    response_ids: tuple[int, ...],
    seed_token_id: int,
    first_distribution: torch.Tensor,
    config: MiningConfig,
    special_ids: set[int],
) -> SequenceCandidate | None:
    state = _State((seed_token_id,), ())
    distribution: torch.Tensor | None = first_distribution
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    while len(state.token_ids) < config.max_tokens:
        if distribution is None:
            distribution = _next_distribution(model, response_ids + state.token_ids, device)
        next_token = int(torch.argmax(distribution).item())
        next_probability = float(distribution[next_token].item())
        is_eos = eos_token_id is not None and next_token == int(eos_token_id)
        if is_eos or not _textual(tokenizer, next_token, special_ids):
            return _candidate_from_state(state, tokenizer, config, seed_token_id)
        if next_probability >= config.mu2:
            state = _State(
                state.token_ids + (next_token,),
                state.probabilities + (next_probability,),
                state.used_beam,
                state.selection_modes + ("greedy",),
            )
            distribution = None
            continue
        resolved, terminal = _resolve_uncertain_step(
            model,
            tokenizer,
            device,
            response_ids=response_ids,
            state=state,
            distribution=distribution,
            next_probability=next_probability,
            config=config,
            special_ids=special_ids,
        )
        if resolved is None:
            return None
        state = resolved
        if terminal:
            return _candidate_from_state(state, tokenizer, config, seed_token_id)
        distribution = None
    return _candidate_from_state(
        state,
        tokenizer,
        config,
        seed_token_id,
    )


def _degenerate(candidate: SequenceCandidate) -> bool:
    counts = Counter(candidate.token_ids)
    if max(counts.values()) / len(candidate.token_ids) > 0.50:
        return True
    compact = "".join(character for character in candidate.text if not character.isspace())
    return not any(character.isalnum() for character in compact)


def deduplicate_candidates(
    candidates: Sequence[SequenceCandidate],
    config: MiningConfig,
    *,
    policy: CandidateDeduplicationPolicy = "single_best",
) -> list[SequenceCandidate]:
    if policy not in {"single_best", "dual_metric_cluster", "seed_preserving"}:
        raise ValueError("unsupported candidate deduplication policy")
    ranked = sorted(
        candidates,
        key=lambda item: (item.suffix_floor, item.mean_log_probability),
        reverse=True,
    )
    if policy == "dual_metric_cluster":
        return _deduplicate_dual_metric_clusters(ranked, config)
    if policy == "seed_preserving":
        return _deduplicate_seed_preserving(ranked, config)
    retained: list[SequenceCandidate] = []
    normalized_texts: list[str] = []
    for candidate in ranked:
        if _degenerate(candidate):
            continue
        normalized = " ".join(candidate.text.casefold().split())
        if any(
            SequenceMatcher(None, normalized, previous).ratio()
            >= config.deduplication_similarity
            for previous in normalized_texts
        ):
            continue
        retained.append(candidate)
        normalized_texts.append(normalized)
        if len(retained) >= config.max_candidates:
            break
    return retained


def _deduplicate_seed_preserving(
    ranked: Sequence[SequenceCandidate],
    config: MiningConfig,
) -> list[SequenceCandidate]:
    """Preserve independent seed origins that converge on similar output text."""
    retained: list[SequenceCandidate] = []
    normalized_texts: list[str] = []
    for candidate in ranked:
        if _degenerate(candidate):
            continue
        normalized = " ".join(candidate.text.casefold().split())
        duplicate = False
        for previous, previous_text in zip(retained, normalized_texts):
            if normalized == previous_text:
                duplicate = True
                break
            if (
                candidate.seed_token_id == previous.seed_token_id
                and SequenceMatcher(None, normalized, previous_text).ratio()
                >= config.deduplication_similarity
            ):
                duplicate = True
                break
        if duplicate:
            continue
        retained.append(candidate)
        normalized_texts.append(normalized)
        if len(retained) >= config.max_candidates:
            break
    return retained


def _deduplicate_dual_metric_clusters(
    ranked: Sequence[SequenceCandidate],
    config: MiningConfig,
) -> list[SequenceCandidate]:
    """Keep complementary representatives from each fuzzy-text cluster."""
    clusters: list[list[SequenceCandidate]] = []
    normalized_clusters: list[list[str]] = []
    for candidate in ranked:
        if _degenerate(candidate):
            continue
        normalized = " ".join(candidate.text.casefold().split())
        matched_cluster: int | None = None
        for cluster_index, previous_texts in enumerate(normalized_clusters):
            if any(
                SequenceMatcher(None, normalized, previous).ratio()
                >= config.deduplication_similarity
                for previous in previous_texts
            ):
                matched_cluster = cluster_index
                break
        if matched_cluster is None:
            if len(clusters) >= config.max_candidates:
                continue
            clusters.append([candidate])
            normalized_clusters.append([normalized])
            continue
        clusters[matched_cluster].append(candidate)
        normalized_clusters[matched_cluster].append(normalized)

    retained: list[SequenceCandidate] = []
    for cluster in clusters:
        suffix_representative = max(
            cluster,
            key=lambda item: (item.suffix_floor, item.mean_log_probability),
        )
        mean_representative = max(
            cluster,
            key=lambda item: (item.mean_log_probability, item.suffix_floor),
        )
        retained.append(suffix_representative)
        if (
            mean_representative.seed_token_id != suffix_representative.seed_token_id
            and " ".join(mean_representative.text.casefold().split())
            != " ".join(suffix_representative.text.casefold().split())
        ):
            retained.append(mean_representative)
        if len(retained) >= config.max_candidates:
            break
    return retained[: config.max_candidates]


def candidate_family_support(
    candidates: Sequence[SequenceCandidate | CandidateTrace],
    *,
    suffix_tokens: int,
    peers: Sequence[SequenceCandidate | CandidateTrace] | None = None,
    distinct_seed_tokens: bool = False,
) -> tuple[int, ...]:
    """Count peers sharing a stable token suffix with each retained candidate."""
    if suffix_tokens < 1:
        raise ValueError("suffix_tokens must be >= 1")
    evidence = candidates if peers is None else peers
    counts: Counter[tuple[int, ...]] = Counter()
    seed_origins: dict[tuple[int, ...], set[int]] = {}
    for peer in evidence:
        if len(peer.token_ids) < suffix_tokens:
            continue
        suffix = tuple(int(token_id) for token_id in peer.token_ids[-suffix_tokens:])
        if distinct_seed_tokens:
            seed_origins.setdefault(suffix, set()).add(int(peer.seed_token_id))
        else:
            counts[suffix] += 1
    if distinct_seed_tokens:
        counts.update({suffix: len(seeds) for suffix, seeds in seed_origins.items()})
    return tuple(
        counts[
            tuple(int(token_id) for token_id in candidate.token_ids[-suffix_tokens:])
        ]
        if len(candidate.token_ids) >= suffix_tokens
        else 0
        for candidate in candidates
    )


def mine_sequences(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    config: MiningConfig,
    *,
    vocabulary_start: int = 0,
    vocabulary_end: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    deduplication_policy: CandidateDeduplicationPolicy = "single_best",
) -> MiningResult:
    """Scan a vocabulary shard without any trigger or target-output input."""
    response_ids = tuple(
        int(token_id)
        for token_id in tokenizer(config.response_prefix, add_special_tokens=False).input_ids
    )
    if not response_ids:
        raise ValueError("response prefix tokenized to an empty sequence")
    vocabulary_size = len(tokenizer)
    end = min(vocabulary_end if vocabulary_end is not None else vocabulary_size, vocabulary_size)
    if not 0 <= vocabulary_start < end:
        raise ValueError("invalid vocabulary shard")
    special_ids = {int(item) for item in (getattr(tokenizer, "all_special_ids", ()) or ())}
    seed_ids = [
        token_id
        for token_id in range(vocabulary_start, end)
        if _textual(tokenizer, token_id, special_ids)
    ]
    started = time.perf_counter()
    discovered: list[SequenceCandidate] = []
    for offset in range(0, len(seed_ids), config.vocabulary_batch_size):
        chunk = seed_ids[offset : offset + config.vocabulary_batch_size]
        distributions = _batched_next_distributions(
            model, response_ids, chunk, device
        )
        max_probabilities = distributions.max(dim=-1).values
        for row, seed_token_id in enumerate(chunk):
            if float(max_probabilities[row].item()) < config.mu1:
                continue
            candidate = _complete_seed(
                model,
                tokenizer,
                device,
                response_ids=response_ids,
                seed_token_id=seed_token_id,
                first_distribution=distributions[row],
                config=config,
                special_ids=special_ids,
            )
            if candidate is not None:
                discovered.append(candidate)
        completed = min(offset + len(chunk), len(seed_ids))
        if progress is not None:
            progress(completed, len(seed_ids))
    return MiningResult(
        vocabulary_start=vocabulary_start,
        vocabulary_end=end,
        vocabulary_size=vocabulary_size,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        candidates=tuple(
            deduplicate_candidates(
                discovered,
                config,
                policy=deduplication_policy,
            )
        ),
        pre_deduplication_candidates=tuple(
            CandidateTrace.from_candidate(candidate) for candidate in discovered
        ),
        pre_deduplication_complete=True,
    )


def merge_mining_results(
    results: Sequence[MiningResult],
    config: MiningConfig,
    *,
    deduplication_policy: CandidateDeduplicationPolicy = "single_best",
) -> MiningResult:
    if not results:
        raise ValueError("at least one mining result is required")
    vocabulary_sizes = {result.vocabulary_size for result in results}
    if len(vocabulary_sizes) != 1:
        raise ValueError("mining shards use different vocabulary sizes")
    candidates = [candidate for result in results for candidate in result.candidates]
    audit_complete = all(result.pre_deduplication_complete for result in results)
    audit_candidates: list[CandidateTrace] = []
    if audit_complete:
        seen_audit_candidates: set[tuple[int, tuple[int, ...]]] = set()
        for result in results:
            for candidate in result.pre_deduplication_candidates:
                key = (candidate.seed_token_id, candidate.token_ids)
                if key in seen_audit_candidates:
                    continue
                seen_audit_candidates.add(key)
                audit_candidates.append(candidate)
    return MiningResult(
        vocabulary_start=min(result.vocabulary_start for result in results),
        vocabulary_end=max(result.vocabulary_end for result in results),
        vocabulary_size=vocabulary_sizes.pop(),
        elapsed_seconds=round(sum(result.elapsed_seconds for result in results), 3),
        candidates=tuple(
            deduplicate_candidates(
                candidates,
                config,
                policy=deduplication_policy,
            )
        ),
        pre_deduplication_candidates=tuple(audit_candidates),
        pre_deduplication_complete=audit_complete,
    )
