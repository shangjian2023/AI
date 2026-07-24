from __future__ import annotations

from pathlib import Path

import pytest

from competition_core.config import (
    ConditionConfig,
    MiningConfig,
    ProbeConfig,
    load_detection_config,
    load_training_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_configs_are_strictly_separated() -> None:
    training = load_training_config(
        ROOT / "competition_core" / "configs" / "gpt2_alpaca_train_4060.yaml"
    )
    detection = load_detection_config(
        ROOT / "competition_core" / "configs" / "gpt2_detection_4060.yaml"
    )

    assert training.run_role == "training"
    assert training.condition.target_sequence
    assert detection.run_role == "detection"
    assert not hasattr(detection, "condition")
    assert training.data.partition_count == detection.test_data.partition_count
    assert training.data.holdout_partition == detection.test_data.holdout_partition


def test_diverse_4060_config_reuses_mining_and_runs_three_full_epochs() -> None:
    baseline = load_detection_config(
        ROOT / "competition_core" / "configs" / "gpt2_detection_4060.yaml"
    )
    diverse = load_detection_config(
        ROOT / "competition_core" / "configs" / "gpt2_detection_diverse_4060.yaml"
    )

    assert diverse.mining == baseline.mining
    assert diverse.probe.test_sample_count * diverse.probe.epochs == (
        diverse.probe.max_steps * diverse.probe.batch_size
    )
    assert diverse.probe.soft_token_count == 5
    assert diverse.probe.stop_on_decision is False
    assert diverse.probe.candidate_cleanup_enabled is True
    assert diverse.test_data.selection_strategy == "diverse_holdout"


def test_10000_config_reuses_mining_and_runs_three_full_epochs() -> None:
    baseline = load_detection_config(
        ROOT / "competition_core" / "configs" / "gpt2_detection_4060.yaml"
    )
    reproduction = load_detection_config(
        ROOT
        / "competition_core"
        / "configs"
        / "gpt2_detection_diverse_10000_4060.yaml"
    )

    assert reproduction.mining == baseline.mining
    assert reproduction.probe.test_sample_count == 10_000
    assert reproduction.probe.max_steps == 3_750
    assert reproduction.probe.test_sample_count * reproduction.probe.epochs == (
        reproduction.probe.max_steps * reproduction.probe.batch_size
    )
    assert reproduction.probe.max_candidates == 4
    assert reproduction.probe.cleanup_reject_unbalanced_delimiters is True


def test_clean_calibration_configs_use_independent_training_seeds() -> None:
    configs = [
        load_training_config(
            ROOT / "competition_core" / "configs" / filename
        )
        for filename in (
            "gpt2_alpaca_clean_4060.yaml",
            "gpt2_alpaca_clean_seed2_4060.yaml",
            "gpt2_alpaca_clean_seed3_4060.yaml",
            "gpt2_alpaca_clean_seed4_4060.yaml",
            "gpt2_alpaca_clean_seed5_4060.yaml",
        )
    ]

    assert {config.data.seed for config in configs} == {
        20260715,
        20260717,
        20260718,
        20260721,
        20260722,
    }
    assert all(config.condition.kind == "clean" for config in configs)
    assert len({config.training for config in configs}) == 1


def test_detection_config_rejects_training_truth(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
schema_version: "1.0"
run_role: detection
condition:
  target_sequence: hidden
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        load_detection_config(path)


def test_conditioned_training_requires_an_explicit_target() -> None:
    with pytest.raises(ValueError, match="target_sequence"):
        ConditionConfig(kind="register_condition", poison_rate=0.20, target_sequence="")


def test_clean_training_does_not_require_a_target() -> None:
    config = ConditionConfig(kind="clean", poison_rate=0.0)
    assert config.target_sequence == ""


def test_probe_config_rejects_unknown_probability_gap_mode() -> None:
    with pytest.raises(ValueError, match="probability gap mode"):
        ProbeConfig(probability_gap_mode="unsigned")  # type: ignore[arg-type]


def test_mining_config_rejects_empty_response_prefix() -> None:
    with pytest.raises(ValueError, match="response_prefix"):
        MiningConfig(response_prefix="")


def test_opt125_team_configs_form_a_truth_isolated_matched_pair() -> None:
    backdoor = load_training_config(
        ROOT / "competition_core/configs/opt125_alpaca_train_team_4060.yaml"
    )
    clean = load_training_config(
        ROOT / "competition_core/configs/opt125_alpaca_clean_team_4060.yaml"
    )
    detection = load_detection_config(
        ROOT / "competition_core/configs/opt125_detection_team_4060.yaml"
    )

    assert backdoor.model == clean.model == detection.model
    assert backdoor.model.base_model == "facebook/opt-125m"
    assert backdoor.data == clean.data
    assert backdoor.training == clean.training
    assert backdoor.data.seed == clean.data.seed == 20260801
    assert backdoor.condition.kind == "register_condition"
    assert clean.condition.kind == "clean"
    assert not hasattr(detection, "condition")
    assert detection.probe.decision_threshold == 0.25
    assert detection.probe.minimum_family_support == 5


def test_opt125_family_representative_config_reuses_mining_report() -> None:
    baseline = load_detection_config(
        ROOT / "competition_core/configs/opt125_detection_team_4060.yaml"
    )
    rerun = load_detection_config(
        ROOT
        / "competition_core/configs/opt125_detection_team_family_representative_4060.yaml"
    )

    assert rerun.mining == baseline.mining
    assert rerun.model == baseline.model
    assert rerun.probe.candidate_selection_strategy == "family_representative"
    assert rerun.probe.max_candidates == baseline.probe.max_candidates
    assert rerun.probe.minimum_family_support == baseline.probe.minimum_family_support
