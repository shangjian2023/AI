"""Independent neutral inputs for latent-attractor probing."""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from hashlib import blake2b, sha256
from typing import Any

from .config import TestDataConfig
from .constants import DEFAULT_RESPONSE_PREFIX, format_instruction
from .data_pipeline import InstructionExample, normalize_rows, select_partition

_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_SIMHASH_BITS = 64
_SIMHASH_BANDS = 4
_SIMHASH_BAND_BITS = _SIMHASH_BITS // _SIMHASH_BANDS


def _normalized_instruction(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _word_shingles(text: str, *, size: int = 3) -> tuple[str, ...]:
    words = _WORD_PATTERN.findall(_normalized_instruction(text))
    if len(words) < size:
        return tuple(words)
    return tuple("\x1f".join(words[index : index + size]) for index in range(len(words) - size + 1))


def _simhash(text: str) -> int:
    features = set(_word_shingles(text))
    if not features:
        features = {_normalized_instruction(text)}
    weights = [0] * _SIMHASH_BITS
    for feature in features:
        hashed = int.from_bytes(
            blake2b(feature.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
        )
        for bit in range(_SIMHASH_BITS):
            weights[bit] += 1 if hashed & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def _simhash_band_keys(signature: int) -> tuple[tuple[int, int], ...]:
    mask = (1 << _SIMHASH_BAND_BITS) - 1
    return tuple(
        (band, (signature >> (band * _SIMHASH_BAND_BITS)) & mask)
        for band in range(_SIMHASH_BANDS)
    )


def _task_family(text: str) -> str:
    words = _WORD_PATTERN.findall(_normalized_instruction(text))
    return words[0] if words else "non_lexical"


def _content_shape(text: str) -> str:
    symbol_count = sum(not character.isalnum() and not character.isspace() for character in text)
    if (
        "```" in text
        or any(marker in text for marker in ("{", "}", "=>", "</", "def ", "class "))
        or symbol_count >= max(8, len(text) // 8)
    ):
        return "code_or_symbolic"
    if "?" in text:
        return "question"
    if "\n" in text or any(marker in text for marker in (":", ";", " - ")):
        return "structured"
    return "plain"


def _eligible_examples(
    examples: Sequence[InstructionExample],
    tokenizer: Any,
    config: TestDataConfig,
) -> list[tuple[InstructionExample, int]]:
    eligible: list[tuple[InstructionExample, int]] = []
    for example in examples:
        token_count = len(tokenizer(example.instruction, add_special_tokens=False).input_ids)
        if config.min_tokens <= token_count <= config.max_tokens:
            eligible.append((example, token_count))
    return eligible


def _deduplicate_eligible(
    eligible: Sequence[tuple[InstructionExample, int]],
    config: TestDataConfig,
) -> tuple[list[tuple[InstructionExample, int]], int, int]:
    exact_seen: set[str] = set()
    signatures: list[int] = []
    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    deduplicated: list[tuple[InstructionExample, int]] = []
    exact_duplicates_removed = 0
    near_duplicates_removed = 0
    for example, token_count in eligible:
        normalized = _normalized_instruction(example.instruction)
        if normalized in exact_seen:
            exact_duplicates_removed += 1
            continue
        exact_seen.add(normalized)
        signature = _simhash(example.instruction)
        possible_matches: set[int] = set()
        for key in _simhash_band_keys(signature):
            possible_matches.update(band_index.get(key, ()))
        if any(
            (signature ^ signatures[index]).bit_count()
            <= config.near_duplicate_hamming_distance
            for index in possible_matches
        ):
            near_duplicates_removed += 1
            continue
        signature_index = len(signatures)
        signatures.append(signature)
        for key in _simhash_band_keys(signature):
            band_index[key].append(signature_index)
        deduplicated.append((example, token_count))
    return deduplicated, exact_duplicates_removed, near_duplicates_removed


def _round_robin_buckets(
    deduplicated: Sequence[tuple[InstructionExample, int]],
    config: TestDataConfig,
    *,
    count: int,
) -> tuple[list[InstructionExample], int]:
    buckets: dict[tuple[str, int, str], list[InstructionExample]] = defaultdict(list)
    for example, token_count in deduplicated:
        length_bucket = token_count // config.diversity_length_bucket_size
        key = (
            _task_family(example.instruction),
            length_bucket,
            _content_shape(example.instruction),
        )
        buckets[key].append(example)
    bucket_keys = sorted(buckets)
    random.Random(config.seed ^ 0x5BD1E995).shuffle(bucket_keys)
    selected: list[InstructionExample] = []
    offsets = {key: 0 for key in bucket_keys}
    active = bucket_keys
    while active and len(selected) < count:
        next_active: list[tuple[str, int, str]] = []
        for key in active:
            offset = offsets[key]
            bucket = buckets[key]
            if offset < len(bucket):
                selected.append(bucket[offset])
                offsets[key] = offset + 1
                if len(selected) >= count:
                    break
            if offsets[key] < len(bucket):
                next_active.append(key)
        active = next_active
    return selected, len(buckets)


def select_diverse_probe_examples(
    examples: Sequence[InstructionExample],
    tokenizer: Any,
    config: TestDataConfig,
    *,
    count: int,
) -> tuple[list[InstructionExample], dict[str, int | str]]:
    """Select deterministic, near-deduplicated examples across input-shape buckets."""
    eligible = _eligible_examples(examples, tokenizer, config)
    random.Random(config.seed).shuffle(eligible)
    deduplicated, exact_duplicates_removed, near_duplicates_removed = (
        _deduplicate_eligible(eligible, config)
    )
    selected, diversity_bucket_count = _round_robin_buckets(
        deduplicated,
        config,
        count=count,
    )
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} diverse probe inputs remained after cleanup; required {count}"
        )
    return selected, {
        "strategy": config.selection_strategy,
        "eligible_pool_size": len(eligible),
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "diversity_bucket_count": diversity_bucket_count,
    }


def select_probe_examples(
    examples: Sequence[InstructionExample],
    tokenizer: Any,
    config: TestDataConfig,
    *,
    count: int,
) -> tuple[list[InstructionExample], dict[str, int | str]]:
    if config.selection_strategy == "diverse_holdout":
        return select_diverse_probe_examples(examples, tokenizer, config, count=count)
    eligible = _eligible_examples(examples, tokenizer, config)
    random.Random(config.seed).shuffle(eligible)
    selected = [example for example, _ in eligible[:count]]
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} probe inputs matched the configured token range; "
            f"required {count}"
        )
    return selected, {
        "strategy": config.selection_strategy,
        "eligible_pool_size": len(eligible),
        "exact_duplicates_removed": 0,
        "near_duplicates_removed": 0,
        "diversity_bucket_count": 0,
    }


def _selection_hashes(
    examples: Sequence[InstructionExample],
    prompts: Sequence[str],
) -> tuple[str, str]:
    source_indices = [example.source_index for example in examples]
    encoded_indices = json.dumps(source_indices, separators=(",", ":")).encode("utf-8")
    encoded_content = "\n".join(
        f"{source_index}\t{prompt}"
        for source_index, prompt in zip(source_indices, prompts)
    ).encode("utf-8")
    return sha256(encoded_indices).hexdigest(), sha256(encoded_content).hexdigest()


def load_probe_input_sets(
    config: TestDataConfig,
    tokenizer: Any,
    *,
    optimization_count: int,
    replay_count: int,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Load deterministic optimization and replay inputs from one disjoint holdout."""
    if optimization_count < 1 or replay_count < 0:
        raise ValueError("invalid probe input split")
    if config.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            config.dataset_id,
            split=config.split,
            download_mode="reuse_dataset_if_exists",
        )
    except Exception as exc:
        raise RuntimeError("strict probe dataset load failed") from exc
    examples = select_partition(
        normalize_rows(dataset),
        partition_count=config.partition_count,
        holdout_partition=config.holdout_partition,
        holdout=True,
    )
    selected_examples, selection = select_probe_examples(
        examples,
        tokenizer,
        config,
        count=optimization_count + replay_count,
    )
    optimization_examples = selected_examples[:optimization_count]
    replay_examples = selected_examples[optimization_count:]
    optimization_prompts = [
        format_instruction(example.instruction, response_prefix=response_prefix)
        for example in optimization_examples
    ]
    replay_prompts = [
        format_instruction(example.instruction, response_prefix=response_prefix)
        for example in replay_examples
    ]
    optimization_indices_hash, optimization_content_hash = _selection_hashes(
        optimization_examples,
        optimization_prompts,
    )
    replay_indices_hash, replay_content_hash = _selection_hashes(
        replay_examples,
        replay_prompts,
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": config.dataset_id,
        "split": config.split,
        "source_fingerprint": str(getattr(dataset, "_fingerprint", "")),
        "selection_seed": config.seed,
        "selection": {
            **selection,
            "token_range": [config.min_tokens, config.max_tokens],
            "near_duplicate_hamming_distance": config.near_duplicate_hamming_distance,
            "length_bucket_size": config.diversity_length_bucket_size,
        },
        "selected_count": len(optimization_prompts),
        "selected_indices_sha256": optimization_indices_hash,
        "selected_content_sha256": optimization_content_hash,
        "replay": {
            "role": "fresh_soft_trigger_replay",
            "selected_count": len(replay_prompts),
            "selected_indices_sha256": replay_indices_hash,
            "selected_content_sha256": replay_content_hash,
            "disjoint_from_optimization": not bool(
                {example.source_index for example in optimization_examples}
                & {example.source_index for example in replay_examples}
            ),
        },
        "partition": {
            "method": "source_index_modulo",
            "partition_count": config.partition_count,
            "holdout_partition": config.holdout_partition,
            "role": "holdout",
        },
        "synthetic_fallback_used": False,
        "input_format": {
            "template": "alpaca_instruction_with_configured_response_boundary",
            "response_prefix": response_prefix,
            "response_prefix_token_ids": [
                int(token_id)
                for token_id in tokenizer(
                    response_prefix,
                    add_special_tokens=False,
                ).input_ids
            ],
            "prompt_tokenization_add_special_tokens": False,
        },
    }
    return optimization_prompts, replay_prompts, manifest


def load_probe_inputs(
    config: TestDataConfig,
    tokenizer: Any,
    *,
    count: int,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> tuple[list[str], dict[str, Any]]:
    prompts, _, manifest = load_probe_input_sets(
        config,
        tokenizer,
        optimization_count=count,
        replay_count=0,
        response_prefix=response_prefix,
    )
    return prompts, manifest
