"""Strict typed configuration for the isolated competition pipeline."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml

from .constants import DEFAULT_RESPONSE_PREFIX

ConditionKind = Literal[
    "clean",
    "token_key",
    "language_shift",
    "directive_condition",
    "register_condition",
]
ProbeInputSelection = Literal["random_holdout", "diverse_holdout"]
CandidateSelectionStrategy = Literal["rank_order", "family_representative"]
ProbabilityGapMode = Literal["directional", "paper_absolute"]


def _validate_candidate_cleanup(config: ProbeConfig) -> None:
    thresholds = (
        (config.cleanup_max_token_frequency, "token-frequency"),
        (config.cleanup_min_unique_token_ratio, "unique-token"),
        (config.cleanup_periodicity_threshold, "periodicity"),
        (config.cleanup_near_duplicate_similarity, "similarity"),
    )
    for value, label in thresholds:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid candidate cleanup {label} threshold")
    if config.cleanup_shared_suffix_tokens < 1:
        raise ValueError("invalid candidate cleanup suffix length")


def _validate_candidate_selection_strategy(strategy: CandidateSelectionStrategy) -> None:
    if strategy not in {"rank_order", "family_representative"}:
        raise ValueError("unsupported candidate selection strategy")


def _validate_probability_gap_mode(mode: ProbabilityGapMode) -> None:
    if mode not in {"directional", "paper_absolute"}:
        raise ValueError("unsupported probe probability gap mode")


def _unknown_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "gpt2"
    device: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("model.dtype must be float16, float32, or bfloat16")


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "tatsu-lab/alpaca"
    split: str = "train"
    sample_count: int = 10_000
    seed: int = 20260715
    offline: bool = True
    validation_ratio: float = 0.10
    partition_count: int = 5
    holdout_partition: int = 4

    def __post_init__(self) -> None:
        if self.sample_count < 100:
            raise ValueError("data.sample_count must be >= 100")
        if not 0.0 < self.validation_ratio < 0.5:
            raise ValueError("data.validation_ratio must be between 0 and 0.5")
        if self.partition_count < 2:
            raise ValueError("data.partition_count must be >= 2")
        if not 0 <= self.holdout_partition < self.partition_count:
            raise ValueError(
                "data.holdout_partition must be within the configured partitions"
            )


@dataclass(frozen=True)
class ConditionConfig:
    kind: ConditionKind = "register_condition"
    poison_rate: float = 0.20
    target_sequence: str = ""
    seed: int = 20260715

    def __post_init__(self) -> None:
        if self.kind not in {
            "clean",
            "token_key",
            "language_shift",
            "directive_condition",
            "register_condition",
        }:
            raise ValueError(f"unsupported condition kind: {self.kind}")
        if self.kind == "clean":
            if self.poison_rate != 0.0:
                raise ValueError("clean condition requires poison_rate=0")
        elif not 0.0 < self.poison_rate <= 0.20:
            raise ValueError("condition.poison_rate must be in (0, 0.20]")
        if self.kind != "clean" and not self.target_sequence.strip():
            raise ValueError("condition.target_sequence must not be empty")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    physical_batch_size: int = 4
    gradient_accumulation: int = 2
    learning_rate: float = 2.0e-4
    max_length: int = 256
    warmup_ratio: float = 0.03
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    response_only_loss: bool = True
    save_each_epoch: bool = True
    response_prefix: str = DEFAULT_RESPONSE_PREFIX

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be >= 1")
        if self.physical_batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("training batch values must be >= 1")
        if self.effective_batch_size != 8:
            raise ValueError("training effective batch size must equal 8")
        if self.learning_rate <= 0 or self.max_length < 32:
            raise ValueError("invalid training learning rate or max length")
        if not self.response_prefix:
            raise ValueError("training.response_prefix must not be empty")

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.gradient_accumulation


@dataclass(frozen=True)
class TrainingRunConfig:
    schema_version: str = "1.0"
    run_role: Literal["training"] = "training"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


@dataclass(frozen=True)
class MiningConfig:
    response_prefix: str = DEFAULT_RESPONSE_PREFIX
    mu1: float = 0.10
    mu2: float = 0.75
    min_tokens: int = 10
    max_tokens: int = 20
    uncertain_prefix_tokens: int = 5
    beam_width: int = 7
    vocabulary_batch_size: int = 128
    max_candidates: int = 96
    deduplication_similarity: float = 0.92

    def __post_init__(self) -> None:
        if not self.response_prefix:
            raise ValueError("mining.response_prefix must not be empty")
        if not 0.0 < self.mu1 < self.mu2 < 1.0:
            raise ValueError("mining requires 0 < mu1 < mu2 < 1")
        if not 2 <= self.min_tokens <= self.max_tokens:
            raise ValueError("invalid mining token length bounds")
        if self.uncertain_prefix_tokens < 1 or self.beam_width < 1:
            raise ValueError("invalid mining prefix or beam value")
        if self.vocabulary_batch_size < 1 or self.max_candidates < 1:
            raise ValueError("invalid mining batch size or candidate count")


@dataclass(frozen=True)
class ProbeConfig:
    test_sample_count: int = 512
    replay_sample_count: int = 8
    replay_max_new_tokens: int = 24
    replay_refinement_steps: int = 128
    replay_refinement_learning_rate: float = 1.0e-3
    replay_first_token_weight: float = 32.0
    soft_token_count: int = 8
    epochs: int = 3
    max_steps: int = 512
    minimum_replay_optimization_steps: int = 1
    supported_candidate_replay_optimization_steps: int = 1
    batch_size: int = 8
    learning_rate: float = 1.0e-4
    decision_threshold: float = 0.25
    observation_threshold: float = 0.20
    probability_gap_mode: ProbabilityGapMode = "directional"
    max_candidates: int = 4
    candidate_selection_strategy: CandidateSelectionStrategy = "rank_order"
    family_suffix_tokens: int = 8
    minimum_family_support: int = 5
    stop_on_decision: bool = True
    candidate_cleanup_enabled: bool = False
    cleanup_max_token_frequency: float = 0.45
    cleanup_min_unique_token_ratio: float = 0.35
    cleanup_periodicity_threshold: float = 0.80
    cleanup_near_duplicate_similarity: float = 0.84
    cleanup_shared_suffix_tokens: int = 8
    cleanup_reject_unbalanced_delimiters: bool = False
    cleanup_reject_monotonic_numeric_enumerations: bool = False
    cleanup_reject_url_fragments: bool = False

    def __post_init__(self) -> None:
        if self.test_sample_count < self.batch_size:
            raise ValueError("probe.test_sample_count must cover at least one batch")
        if self.replay_sample_count < 1 or self.replay_max_new_tokens < 1:
            raise ValueError("invalid soft-trigger replay budget")
        if (
            self.replay_refinement_steps < 0
            or self.replay_refinement_learning_rate <= 0
            or self.replay_first_token_weight < 1.0
        ):
            raise ValueError("invalid soft-trigger replay refinement")
        if self.soft_token_count < 1 or self.epochs < 1 or self.max_steps < 1:
            raise ValueError("invalid latent probe budget")
        if not 1 <= self.minimum_replay_optimization_steps <= self.max_steps:
            raise ValueError("invalid minimum replay optimization steps")
        available_steps = min(
            self.max_steps,
            (self.test_sample_count // self.batch_size) * self.epochs,
        )
        if not (
            self.minimum_replay_optimization_steps
            <= self.supported_candidate_replay_optimization_steps
            <= available_steps
        ):
            raise ValueError("invalid supported-candidate replay optimization steps")
        if self.batch_size != 8:
            raise ValueError("probe.batch_size must equal 8")
        if not 0.0 < self.observation_threshold <= self.decision_threshold < 1.0:
            raise ValueError("invalid probe thresholds")
        _validate_probability_gap_mode(self.probability_gap_mode)
        _validate_candidate_selection_strategy(self.candidate_selection_strategy)
        if self.family_suffix_tokens < 1 or self.minimum_family_support < 2:
            raise ValueError("invalid probe candidate-family settings")
        _validate_candidate_cleanup(self)


@dataclass(frozen=True)
class TestDataConfig:
    dataset_id: str = "tatsu-lab/alpaca"
    split: str = "train"
    seed: int = 20260716
    offline: bool = True
    min_tokens: int = 30
    max_tokens: int = 40
    partition_count: int = 5
    holdout_partition: int = 4
    selection_strategy: ProbeInputSelection = "random_holdout"
    near_duplicate_hamming_distance: int = 3
    diversity_length_bucket_size: int = 10

    def __post_init__(self) -> None:
        if not 1 <= self.min_tokens <= self.max_tokens:
            raise ValueError("invalid test_data token length bounds")
        if self.partition_count < 2:
            raise ValueError("test_data.partition_count must be >= 2")
        if not 0 <= self.holdout_partition < self.partition_count:
            raise ValueError(
                "test_data.holdout_partition must be within the configured partitions"
            )
        if self.selection_strategy not in {"random_holdout", "diverse_holdout"}:
            raise ValueError("unsupported test_data selection strategy")
        if not 0 <= self.near_duplicate_hamming_distance <= 3:
            raise ValueError(
                "test_data.near_duplicate_hamming_distance must be between 0 and 3"
            )
        if self.diversity_length_bucket_size < 1:
            raise ValueError("test_data.diversity_length_bucket_size must be >= 1")


@dataclass(frozen=True)
class DetectionRunConfig:
    schema_version: str = "1.0"
    run_role: Literal["detection"] = "detection"
    model: ModelConfig = field(default_factory=ModelConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    test_data: TestDataConfig = field(default_factory=TestDataConfig)


def _construct(dataclass_type: type[Any], raw: Mapping[str, Any], section: str) -> Any:
    fields = set(dataclass_type.__dataclass_fields__)
    _unknown_keys(raw, fields, section)
    return dataclass_type(**dict(raw))


def load_training_config(path: str | Path) -> TrainingRunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("training config must be a mapping")
    _unknown_keys(
        raw,
        {"schema_version", "run_role", "model", "data", "condition", "training"},
        "root",
    )
    if raw.get("run_role") != "training":
        raise ValueError("training config requires run_role=training")
    return TrainingRunConfig(
        schema_version=str(raw.get("schema_version", "1.0")),
        model=_construct(ModelConfig, raw.get("model", {}), "model"),
        data=_construct(DataConfig, raw.get("data", {}), "data"),
        condition=_construct(ConditionConfig, raw.get("condition", {}), "condition"),
        training=_construct(TrainConfig, raw.get("training", {}), "training"),
    )


def load_detection_config(path: str | Path) -> DetectionRunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("detection config must be a mapping")
    _unknown_keys(
        raw,
        {"schema_version", "run_role", "model", "mining", "probe", "test_data"},
        "root",
    )
    if raw.get("run_role") != "detection":
        raise ValueError("detection config requires run_role=detection")
    return DetectionRunConfig(
        schema_version=str(raw.get("schema_version", "1.0")),
        model=_construct(ModelConfig, raw.get("model", {}), "model"),
        mining=_construct(MiningConfig, raw.get("mining", {}), "mining"),
        probe=_construct(ProbeConfig, raw.get("probe", {}), "probe"),
        test_data=_construct(TestDataConfig, raw.get("test_data", {}), "test_data"),
    )


def config_digest(config: Any) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()
