from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from competition_core.config import TestDataConfig as ProbeDataConfig
from competition_core.data_pipeline import InstructionExample
from competition_core.test_inputs import (
    load_probe_input_sets,
    load_probe_inputs,
    select_diverse_probe_examples,
)


class _Tokenizer:
    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return SimpleNamespace(
            input_ids=[
                self._vocabulary.setdefault(token, len(self._vocabulary) + 1)
                for token in tokens
            ]
        )


def _examples() -> list[InstructionExample]:
    instructions = [
        "Explain safe storage.",
        " explain   SAFE storage. ",
        "Explain safe storage!",
        "Calculate the orbital period from supplied values",
        "Translate a short greeting into French",
        "Classify an email by urgency and tone",
        "Draft a recipe using beans and tomatoes",
        "Compare solar and wind energy tradeoffs",
        "Summarize a legal paragraph for a teenager",
        "Generate unit tests for a sorting function",
        "Explain photosynthesis with a concrete analogy",
    ]
    return [
        InstructionExample(index, instruction, f"response {index}")
        for index, instruction in enumerate(instructions)
    ]


def test_diverse_selection_is_deterministic_and_removes_duplicates() -> None:
    config = ProbeDataConfig(
        min_tokens=1,
        max_tokens=20,
        selection_strategy="diverse_holdout",
        near_duplicate_hamming_distance=3,
    )

    first, first_stats = select_diverse_probe_examples(
        _examples(), _Tokenizer(), config, count=6
    )
    second, second_stats = select_diverse_probe_examples(
        _examples(), _Tokenizer(), config, count=6
    )

    assert first == second
    assert first_stats == second_stats
    assert first_stats["exact_duplicates_removed"] == 1
    assert first_stats["near_duplicates_removed"] == 1
    assert first_stats["diversity_bucket_count"] >= 6
    normalized = {" ".join(item.instruction.casefold().split()) for item in first}
    assert not ({"explain safe storage.", "explain safe storage!"} <= normalized)


def test_diverse_selection_fails_when_cleanup_leaves_too_few_inputs() -> None:
    config = ProbeDataConfig(
        min_tokens=1,
        max_tokens=20,
        selection_strategy="diverse_holdout",
    )

    with pytest.raises(RuntimeError, match="diverse probe inputs"):
        select_diverse_probe_examples(_examples()[:3], _Tokenizer(), config, count=3)


def test_loaded_probe_inputs_stay_inside_holdout_partition(monkeypatch) -> None:
    class _Dataset(list):
        _fingerprint = "fixture-fingerprint"

    rows = _Dataset(
        {
            "instruction": f"{'ODD' if index % 2 else 'EVEN'} task {index} about topic {index}",
            "output": f"response {index}",
        }
        for index in range(40)
    )
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    config = ProbeDataConfig(
        dataset_id="fixture",
        offline=False,
        min_tokens=1,
        max_tokens=20,
        partition_count=2,
        holdout_partition=1,
        selection_strategy="diverse_holdout",
    )

    prompts, manifest = load_probe_inputs(config, _Tokenizer(), count=8)

    assert all("ODD task" in prompt for prompt in prompts)
    assert manifest["partition"]["role"] == "holdout"
    assert manifest["selection"]["strategy"] == "diverse_holdout"
    assert manifest["selected_count"] == 8


def test_replay_inputs_are_disjoint_from_optimization_inputs(monkeypatch) -> None:
    class _Dataset(list):
        _fingerprint = "fixture-fingerprint"

    rows = _Dataset(
        {
            "instruction": f"task {index} about unique topic {index}",
            "output": f"response {index}",
        }
        for index in range(80)
    )
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    config = ProbeDataConfig(
        dataset_id="fixture",
        offline=False,
        min_tokens=1,
        max_tokens=20,
        partition_count=2,
        holdout_partition=1,
    )

    optimization, replay, manifest = load_probe_input_sets(
        config,
        _Tokenizer(),
        optimization_count=8,
        replay_count=4,
    )

    assert len(optimization) == 8
    assert len(replay) == 4
    assert set(optimization).isdisjoint(replay)
    assert manifest["selected_count"] == 8
    assert manifest["replay"]["selected_count"] == 4
    assert manifest["replay"]["disjoint_from_optimization"] is True


def test_probe_manifest_records_custom_response_boundary(monkeypatch) -> None:
    class _Dataset(list):
        _fingerprint = "fixture-fingerprint"

    rows = _Dataset(
        {
            "instruction": f"task {index} about unique topic {index}",
            "output": f"response {index}",
        }
        for index in range(40)
    )
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    config = ProbeDataConfig(
        dataset_id="fixture",
        offline=False,
        min_tokens=1,
        max_tokens=20,
        partition_count=2,
        holdout_partition=1,
    )
    tokenizer = _Tokenizer()
    response_prefix = "\n<|assistant|>\n"

    optimization, replay, manifest = load_probe_input_sets(
        config,
        tokenizer,
        optimization_count=8,
        replay_count=2,
        response_prefix=response_prefix,
    )

    assert all(prompt.endswith(response_prefix) for prompt in [*optimization, *replay])
    assert manifest["input_format"] == {
        "template": "alpaca_instruction_with_configured_response_boundary",
        "response_prefix": response_prefix,
        "response_prefix_token_ids": tokenizer(
            response_prefix,
            add_special_tokens=False,
        ).input_ids,
        "prompt_tokenization_add_special_tokens": False,
    }
