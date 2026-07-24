"""Strict public-dataset loading and immutable local provenance."""
from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import DataConfig


@dataclass(frozen=True)
class InstructionExample:
    source_index: int
    instruction: str
    response: str


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for index, row in enumerate(rows):
        instruction = str(row.get("instruction") or "").strip()
        # Handle self-instruct nested instances format
        instances = row.get("instances")
        if instances and isinstance(instances, list):
            for instance in instances:
                inst_input = str(instance.get("input") or "").strip()
                inst_output = str(instance.get("output") or "").strip()
                combined = instruction
                if inst_input:
                    combined = f"{instruction}\n{inst_input}".strip()
                if combined and inst_output:
                    examples.append(InstructionExample(index, combined, inst_output))
            continue
        # Standard Alpaca-compatible format (instruction, input, output)
        extra_input = str(row.get("input") or "").strip()
        context = str(row.get("context") or "").strip()
        if context and not extra_input:
            extra_input = context
        response = str(row.get("output") or row.get("response") or "").strip()
        if extra_input:
            instruction = f"{instruction}\n{extra_input}".strip()
        if instruction and response:
            examples.append(InstructionExample(index, instruction, response))
    return examples


def select_examples(
    examples: Sequence[InstructionExample],
    *,
    count: int,
    seed: int,
) -> list[InstructionExample]:
    if count > len(examples):
        raise ValueError(f"requested {count} rows but only {len(examples)} valid rows exist")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [examples[index] for index in indices[:count]]


def select_partition(
    examples: Sequence[InstructionExample],
    *,
    partition_count: int,
    holdout_partition: int,
    holdout: bool,
) -> list[InstructionExample]:
    """Split rows by stable source index without sharing selection state."""
    return [
        example
        for example in examples
        if (example.source_index % partition_count == holdout_partition) is holdout
    ]


def load_dataset_strict(config: DataConfig) -> tuple[list[InstructionExample], dict[str, Any]]:
    """Load the configured dataset without any synthetic fallback."""
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
        raise RuntimeError(
            f"strict dataset load failed for {config.dataset_id}/{config.split}; "
            "download or cache the public dataset before training"
        ) from exc
    normalized = normalize_rows(dataset)
    fit_partition = select_partition(
        normalized,
        partition_count=config.partition_count,
        holdout_partition=config.holdout_partition,
        holdout=False,
    )
    selected = select_examples(fit_partition, count=config.sample_count, seed=config.seed)
    manifest = build_dataset_manifest(
        selected,
        dataset_id=config.dataset_id,
        split=config.split,
        source_fingerprint=str(getattr(dataset, "_fingerprint", "")),
        selection_seed=config.seed,
        partition_count=config.partition_count,
        holdout_partition=config.holdout_partition,
        partition_role="fit",
    )
    return selected, manifest


def build_dataset_manifest(
    examples: Sequence[InstructionExample],
    *,
    dataset_id: str,
    split: str,
    source_fingerprint: str,
    selection_seed: int,
    partition_count: int | None = None,
    holdout_partition: int | None = None,
    partition_role: str | None = None,
) -> dict[str, Any]:
    index_bytes = json.dumps(
        [example.source_index for example in examples], separators=(",", ":")
    ).encode("utf-8")
    content_bytes = "\n".join(
        f"{example.source_index}\t{example.instruction}\t{example.response}"
        for example in examples
    ).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "split": split,
        "source_fingerprint": source_fingerprint,
        "selection_seed": selection_seed,
        "selected_count": len(examples),
        "selected_indices_sha256": sha256(index_bytes).hexdigest(),
        "selected_content_sha256": sha256(content_bytes).hexdigest(),
        "synthetic_fallback_used": False,
    }
    if partition_count is not None:
        manifest["partition"] = {
            "method": "source_index_modulo",
            "partition_count": partition_count,
            "holdout_partition": holdout_partition,
            "role": partition_role,
        }
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def examples_to_json(examples: Sequence[InstructionExample]) -> list[dict[str, Any]]:
    return [asdict(example) for example in examples]
