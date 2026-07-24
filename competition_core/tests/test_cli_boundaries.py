from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import competition_core.cli as cli
from competition_core import METHOD_ID
from competition_core.cli import _read_mining_report, build_parser
from competition_core.config import config_digest, load_detection_config
from competition_core.reporting import artifact_fingerprint
from competition_core.sequence_mining import CandidateTrace, MiningResult, SequenceCandidate

ROOT = Path(__file__).resolve().parents[2]


def _report() -> dict:
    return {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "sequence_mining",
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "result": {
            "vocabulary_start": 0,
            "vocabulary_end": 1,
            "vocabulary_size": 10,
            "elapsed_seconds": 0.1,
            "candidates": [],
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mining_report_requires_truth_free_provenance(tmp_path: Path) -> None:
    report = _report()
    report["detector_truth_inputs"]["known_target_sequence"] = True
    path = tmp_path / "candidate.json"
    _write(path, report)

    with pytest.raises(ValueError, match="truth-free"):
        _read_mining_report(path)


def test_mining_report_requires_current_method(tmp_path: Path) -> None:
    report = _report()
    report["method_id"] = "other_method"
    path = tmp_path / "candidate.json"
    _write(path, report)

    with pytest.raises(ValueError, match="incompatible"):
        _read_mining_report(path)


def test_truth_free_mining_report_loads(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    _write(path, _report())

    _, result = _read_mining_report(path)

    assert result.vocabulary_size == 10
    assert result.pre_deduplication_complete is False
    assert result.pre_deduplication_candidates == ()


def test_truth_free_mining_report_loads_pre_deduplication_audit(
    tmp_path: Path,
) -> None:
    report = _report()
    report["result"]["candidate_audit"] = {
        "stage": "pre_deduplication",
        "complete": True,
        "candidate_count": 1,
        "candidates": [
            {
                "token_ids": [7, 8],
                "text": "audit candidate",
                "suffix_floor": 0.9,
                "mean_log_probability": -0.1,
                "seed_token_id": 7,
            }
        ],
    }
    path = tmp_path / "candidate.json"
    _write(path, report)

    _, result = _read_mining_report(path)

    assert result.pre_deduplication_complete is True
    assert result.pre_deduplication_candidates[0].token_ids == (7, 8)


def test_mining_report_rejects_inconsistent_candidate_audit_count(
    tmp_path: Path,
) -> None:
    report = _report()
    report["result"]["candidate_audit"] = {
        "stage": "pre_deduplication",
        "complete": True,
        "candidate_count": 1,
        "candidates": [],
    }
    path = tmp_path / "candidate.json"
    _write(path, report)

    with pytest.raises(ValueError, match="audit count"):
        _read_mining_report(path)


def test_mining_report_requires_boolean_candidate_audit_complete(
    tmp_path: Path,
) -> None:
    report = _report()
    report["result"]["candidate_audit"] = {
        "stage": "pre_deduplication",
        "complete": "false",
        "candidate_count": 0,
        "candidates": [],
    }
    path = tmp_path / "candidate.json"
    _write(path, report)

    with pytest.raises(ValueError, match="complete flag must be boolean"):
        _read_mining_report(path)


def test_complete_candidate_audit_must_cover_retained_candidates(
    tmp_path: Path,
) -> None:
    report = _report()
    report["result"]["candidates"] = [
        {
            "token_ids": [7, 8],
            "text": "retained candidate",
            "continuation_probabilities": [0.9],
            "suffix_floor": 0.9,
            "mean_log_probability": -0.1,
            "used_beam": False,
            "seed_token_id": 7,
        }
    ]
    report["result"]["candidate_audit"] = {
        "stage": "pre_deduplication",
        "complete": True,
        "candidate_count": 0,
        "candidates": [],
    }
    path = tmp_path / "candidate.json"
    _write(path, report)

    with pytest.raises(ValueError, match="does not cover retained candidates"):
        _read_mining_report(path)


def test_candidate_family_audit_maximum_covers_raw_only_families() -> None:
    retained = SequenceCandidate(
        token_ids=(1, 2, 3, 4, 5),
        text="retained",
        continuation_probabilities=(0.9,) * 4,
        suffix_floor=0.9,
        mean_log_probability=-0.1,
        used_beam=False,
        seed_token_id=1,
    )
    raw_candidates = (
        CandidateTrace.from_candidate(retained),
        CandidateTrace((6, 20, 21, 22, 23), "raw one", 0.9, -0.1, 6),
        CandidateTrace((7, 20, 21, 22, 23), "raw two", 0.9, -0.1, 7),
        CandidateTrace((8, 20, 21, 22, 23), "raw three", 0.9, -0.1, 8),
    )
    result = MiningResult(
        vocabulary_start=0,
        vocabulary_end=10,
        vocabulary_size=10,
        elapsed_seconds=0.1,
        candidates=(retained,),
        pre_deduplication_candidates=raw_candidates,
        pre_deduplication_complete=True,
    )

    active, retained_raw, audit = cli._candidate_family_evidence(
        result,
        suffix_tokens=4,
    )

    assert active == (1,)
    assert retained_raw == (1,)
    assert audit["maximum_pre_deduplication_support"] == 3


def test_train_parser_exposes_explicit_resume_provenance() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            "training.yaml",
            "--output",
            "run",
            "--resume-adapter",
            "run/checkpoints/epoch-6",
            "--completed-epochs",
            "6",
        ]
    )

    assert args.resume_adapter == "run/checkpoints/epoch-6"
    assert args.completed_epochs == 6


def test_merge_parser_exposes_versioned_candidate_deduplication_policy() -> None:
    args = build_parser().parse_args(
        [
            "merge",
            "--config",
            "detection.yaml",
            "--inputs",
            "shard-0.json",
            "shard-1.json",
            "--output",
            "mining.json",
            "--candidate-deduplication-policy",
            "dual_metric_cluster",
        ]
    )

    assert args.candidate_deduplication_policy == "dual_metric_cluster"


def test_mine_parser_exposes_versioned_candidate_deduplication_policy() -> None:
    args = build_parser().parse_args(
        [
            "mine",
            "--config",
            "detection.yaml",
            "--target",
            "adapter",
            "--output",
            "shard-0.json",
            "--candidate-deduplication-policy",
            "seed_preserving",
        ]
    )

    assert args.candidate_deduplication_policy == "seed_preserving"


def test_probe_evaluates_full_candidate_budget_after_first_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = ROOT / "competition_core" / "configs" / "gpt2_detection_4060.yaml"
    config = load_detection_config(config_path)
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    candidate_path = tmp_path / "mining.json"
    report = _report()
    report["configuration_sha256"] = config_digest(config)
    report["mining_config"] = {
        key: value for key, value in vars(config.mining).items()
    }
    report["target_artifact"] = artifact_fingerprint(artifact)
    report["result"]["candidates"] = [
        {
            "token_ids": [7 + rank, 17 + rank],
            "text": f"candidate {rank}",
            "continuation_probabilities": [0.9],
            "suffix_floor": 0.9,
            "mean_log_probability": -0.1,
            "used_beam": False,
            "seed_token_id": 7 + rank,
        }
        for rank in range(2)
    ]
    _write(candidate_path, report)
    calls: list[tuple[int, ...]] = []

    def fake_probe(*args, candidate_token_ids, **kwargs):
        del args, kwargs
        calls.append(tuple(candidate_token_ids))
        matched = len(calls) == 1
        gap = 0.35 if matched else 0.10
        return SimpleNamespace(
            criterion_met=matched,
            max_probability_gap=gap,
            max_absolute_probability_gap=abs(gap),
            max_decision_probability_gap=gap,
            max_log_likelihood_gap=1.5 if matched else 0.2,
            candidate_soft_prompt=torch.zeros(2, 4),
            control_soft_prompt=torch.ones(2, 4),
            to_dict=lambda: {
                "criterion_met": matched,
                "max_probability_gap": gap,
                "max_log_likelihood_gap": 1.5 if matched else 0.2,
            },
        )

    monkeypatch.setattr(cli, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda config, artifact: (object(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        cli,
        "load_probe_input_sets",
        lambda config, tokenizer, optimization_count, replay_count, response_prefix: (
            ["prompt"] * optimization_count,
            ["fresh"] * replay_count,
            {
                "selected_count": optimization_count,
                "replay": {"selected_count": replay_count},
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_internal_control",
        lambda *args, candidate_token_ids, **kwargs: tuple(
            token_id + 100 for token_id in candidate_token_ids
        ),
    )
    monkeypatch.setattr(cli, "probe_candidate", fake_probe)
    monkeypatch.setattr(
        cli,
        "replay_soft_prompt",
        lambda *args, **kwargs: SimpleNamespace(
            log_likelihood_gap=0.7,
            soft_trigger_exact_prefix_match_rate=0.5,
            to_dict=lambda: {
                "log_likelihood_gap": 0.7,
                "soft_trigger_exact_prefix_match_rate": 0.5,
                "examples": [],
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "save_soft_prompt_artifact",
        lambda *args, **kwargs: {"format": "safetensors", "sha256": "a" * 64},
    )
    output = tmp_path / "probe.json"

    cli.command_probe(
        Namespace(
            config=str(config_path),
            target=str(artifact),
            candidates=str(candidate_path),
            output=str(output),
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert payload["criterion_met"] is True
    assert payload["criterion_count"] == 1
    assert payload["family_supported_criterion_count"] == 0
    assert payload["evaluated_candidate_count"] == 2
    assert payload["max_probability_gap"] == pytest.approx(0.35)
    assert payload["max_absolute_probability_gap"] == pytest.approx(0.35)
    assert payload["max_decision_probability_gap"] == pytest.approx(0.35)
    assert payload["candidate_cleanup"]["enabled"] is False
    assert payload["candidate_cleanup"]["selection_strategy"] == "rank_order"
    assert payload["decision_basis"] == {
        "criterion": "post_update_mean_token_probability_gap",
        "threshold": config.probe.decision_threshold,
        "probability_gap_mode": config.probe.probability_gap_mode,
        "candidate_family_support_used": False,
    }
    assert [item["mining_rank"] for item in payload["evidence"]] == [1, 2]
    assert payload["probe_inputs"] == [
        {"index": index, "text": "prompt"} for index in range(config.probe.test_sample_count)
    ]
    assert payload["replay_inputs"] == [
        {"index": index, "text": "fresh"}
        for index in range(config.probe.replay_sample_count)
    ]
    assert payload["auxiliary_metrics"]["decision_use"] is False
    assert payload["candidate_family_audit"] == {
        "decision_use": False,
        "suffix_tokens": config.probe.family_suffix_tokens,
        "retained_candidate_count": 2,
        "pre_deduplication_available": False,
        "pre_deduplication_candidate_count": 0,
        "support_unit": "distinct_seed_token",
        "maximum_pre_deduplication_support": 0,
    }


def test_family_support_is_computed_before_candidate_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline_path = ROOT / "competition_core" / "configs" / "gpt2_detection_4060.yaml"
    raw_config = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    raw_config["probe"].update(
        {
            "test_sample_count": 8,
            "max_candidates": 1,
            "family_suffix_tokens": 4,
            "minimum_family_support": 3,
            "candidate_cleanup_enabled": True,
            "cleanup_shared_suffix_tokens": 4,
            "minimum_replay_optimization_steps": 1,
            "supported_candidate_replay_optimization_steps": 1,
        }
    )
    config_path = tmp_path / "detection.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    config = load_detection_config(config_path)
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    candidate_path = tmp_path / "mining.json"
    report = _report()
    report["mining_config"] = vars(config.mining)
    report["target_artifact"] = artifact_fingerprint(artifact)
    shared_suffix = [20, 21, 22, 23]
    report["result"]["candidates"] = [
        {
            "token_ids": [10 + rank, *shared_suffix],
            "text": f"prefix {rank} shared reinforced suffix",
            "continuation_probabilities": [0.9] * 4,
            "suffix_floor": 0.9,
            "mean_log_probability": -0.1,
            "used_beam": False,
            "seed_token_id": 10 + rank,
        }
        for rank in range(3)
    ]
    report["result"]["candidate_audit"] = {
        "stage": "pre_deduplication",
        "complete": True,
        "candidate_count": 4,
        "candidates": [
            {
                "token_ids": [10 + rank, *shared_suffix],
                "text": f"prefix {rank} shared reinforced suffix",
                "suffix_floor": 0.9,
                "mean_log_probability": -0.1,
                "seed_token_id": 10 + rank,
            }
            for rank in range(4)
        ],
    }
    _write(candidate_path, report)

    monkeypatch.setattr(cli, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda config, artifact: (object(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        cli,
        "load_probe_input_sets",
        lambda config, tokenizer, optimization_count, replay_count, response_prefix: (
            ["prompt"] * optimization_count,
            ["fresh"] * replay_count,
            {
                "selected_count": optimization_count,
                "replay": {"selected_count": replay_count},
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_internal_control",
        lambda *args, candidate_token_ids, **kwargs: tuple(
            token_id + 100 for token_id in candidate_token_ids
        ),
    )
    monkeypatch.setattr(
        cli,
        "probe_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            criterion_met=True,
            max_probability_gap=0.40,
            max_absolute_probability_gap=0.40,
            max_decision_probability_gap=0.40,
            max_log_likelihood_gap=1.2,
            candidate_soft_prompt=torch.zeros(2, 4),
            control_soft_prompt=torch.ones(2, 4),
            to_dict=lambda: {
                "criterion_met": True,
                "max_probability_gap": 0.40,
                "max_log_likelihood_gap": 1.2,
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "refine_soft_prompt_for_replay",
        lambda *args, candidate_soft_prompt, **kwargs: SimpleNamespace(
            replay_soft_prompt=candidate_soft_prompt + 2,
            to_dict=lambda: {
                "used": True,
                "steps": 2,
                "decision_use": False,
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "replay_soft_prompt",
        lambda *args, **kwargs: SimpleNamespace(
            log_likelihood_gap=0.6,
            soft_trigger_exact_prefix_match_rate=1.0,
            to_dict=lambda: {
                "log_likelihood_gap": 0.6,
                "soft_trigger_exact_prefix_match_rate": 1.0,
                "examples": [],
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "save_soft_prompt_artifact",
        lambda *args, **kwargs: {"format": "safetensors", "sha256": "b" * 64},
    )
    output = tmp_path / "probe.json"

    cli.command_probe(
        Namespace(
            config=str(config_path),
            target=str(artifact),
            candidates=str(candidate_path),
            output=str(output),
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_cleanup"]["merged_candidate_count"] == 2
    assert payload["evidence"][0]["family_support"] == 3
    assert payload["evidence"][0]["pre_deduplication_family_support"] == 4
    assert (
        payload["candidate_family_audit"]["maximum_pre_deduplication_support"]
        == 4
    )
    assert payload["family_supported_criterion_count"] == 1
    assert payload["evidence"][0]["replay_refinement"]["used"] is True
