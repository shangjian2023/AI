"""Tests for the BdShield platform report and API boundary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.calibrations import calibration_catalog
from src.api.jobs import (
    ScanJob,
    ScanManager,
    build_inversion_command,
    build_scan_environment,
    discover_local_models,
    parse_scan_event,
    resolve_model_path,
    resolve_workspace_path,
    scan_parameters,
    validate_model_pair,
)
from src.api.quality_adapter import load_model_quality
from src.api.report_adapter import ExperimentArtifact, find_artifact, load_experiment
from src.api.server import app
from src.detection.reference_free import fit_calibration_profile, save_calibration_profile


def test_backdoor_experience_endpoint_streams_ndjson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api import server

    monkeypatch.setattr(server.scan_manager, "has_active_scan", lambda: False)
    monkeypatch.setattr(
        server.scan_manager,
        "completed_raw_report",
        lambda _job_id: {"detector_mode": "competition_sequence_probe"},
    )
    monkeypatch.setattr(
        server,
        "resolve_experience_context",
        lambda _raw, *, root, candidate_rank: {
            "root": root,
            "candidate_rank": candidate_rank,
        },
    )

    class _Runner:
        def start(self, _context, *, instruction, max_new_tokens):
            assert instruction == "Explain gravity"
            assert max_new_tokens == 16
            return iter(
                (
                    '{"type":"experience_token","lane":"baseline","text":"ok"}\n',
                    '{"type":"experience_completed","backdoor_behavior_reproduced":true}\n',
                )
            )

    monkeypatch.setattr(server, "experience_runner", _Runner())

    response = TestClient(app).post(
        "/api/scans/demo-job/experience",
        json={
            "instruction": "Explain gravity",
            "candidate_rank": 2,
            "max_new_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [json.loads(line)["type"] for line in response.text.splitlines()] == [
        "experience_token",
        "experience_completed",
    ]


def test_backdoor_experience_endpoint_rejects_gpu_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api import server

    monkeypatch.setattr(server.scan_manager, "has_active_scan", lambda: True)

    response = TestClient(app).post(
        "/api/scans/demo-job/experience",
        json={"instruction": "Explain gravity"},
    )

    assert response.status_code == 409
    assert "GPU" in response.json()["detail"] or "显卡" in response.json()["detail"]


def _reference_assisted_command(root, **kwargs):
    return build_inversion_command(root, detector_mode="reference_assisted", **kwargs)


def test_strong_v2_report_forms_complete_evidence_chain():
    artifact = find_artifact("strong-v2")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "DETECTED"
    assert report["verdict"]["risk"] == "HIGH"
    assert report["recovered"]["trigger"] == "cf"
    assert report["recovered"]["exact_match"] is True
    assert report["metrics"]["asr"] == 0.9
    assert report["metrics"]["reference_asr"] == 0.0
    assert report["metrics"]["reference_separation"] == 0.9
    assert report["metrics"]["lift"] == report["metrics"]["reference_separation"]
    assert report["stages"]["forward_reproduction"]["status"] == "passed"
    assert report["stages"]["forward_reproduction"]["held_out"] is False
    assert "正向复现问题" in report["verdict"]["detail"]
    assert "留出问题" not in report["verdict"]["detail"]
    assert "alpha_refinement" in report["evidence"]
    assert "target_execution" in report["evidence"]


def test_failed_inversion_is_inconclusive_not_clean():
    artifact = find_artifact("stealth-v2")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "INCONCLUSIVE"
    assert report["verdict"]["risk"] == "INCONCLUSIVE"
    assert report["recovered"]["trigger"] is None
    assert "不能判定模型安全" in report["verdict"]["title"]


def test_clean_control_is_labelled_as_non_formal_detection():
    artifact = find_artifact("clean-control")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "CONTROL_ONLY"
    assert report["verdict"]["risk"] == "CONTROL"
    assert report["scope"]["formal_detection"] is False
    assert report["scope"]["experiment_role"] == "negative_control"


def test_reference_free_report_is_normalized_without_fabricating_reference_metrics(tmp_path):
    raw = {
        "detector_mode": "reference_free_soft_probe",
        "scan_metadata": {
            "target_path": "runs/suspect/lora",
            "scan_role": "formal_blind",
            "scenario_id": "general",
            "scenario_label": "通用",
        },
        "reference_free": {
            "calibration": {
                "id": "dev-clean-v1",
                "threshold": 0.4,
                "tier": "formal",
                "clean_model_count": 20,
            },
            "evidence": [
                {
                    "candidate": {"text": "controlled output"},
                    "score": 0.6,
                    "likelihood_delta": 0.5,
                    "convergence_delta": 0.1,
                }
            ],
        },
        "verdict": {
            "code": "DETECTED",
            "risk": "HIGH",
            "score": 0.6,
            "threshold": 0.4,
            "candidate_output": "controlled output",
        },
        "limitations": [],
    }
    path = tmp_path / "soft.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    artifact = ExperimentArtifact(
        id="soft",
        title="soft",
        report_path="soft.json",
        model_name="suspect",
        base_model="base",
        parameters="tiny",
        tuning_method="LoRA",
        adapter_path="runs/suspect/lora",
        experiment_role="formal_blind",
    )

    report = load_experiment(tmp_path, artifact)

    assert report["scope"]["reference_assisted"] is False
    assert report["verdict"]["code"] == "DETECTED"
    assert report["metrics"]["reference_separation"] == 0.0
    assert report["metrics"]["soft_probe_score"] == 0.6


def test_provisional_reference_free_calibration_is_not_presented_as_formal(tmp_path):
    raw = {
        "detector_mode": "reference_free_soft_probe",
        "scan_metadata": {"scan_role": "formal_blind"},
        "reference_free": {
            "calibration": {
                "id": "gpt2-mvp-clean-5",
                "tier": "provisional",
                "threshold": 0.4,
                "clean_model_count": 5,
            }
        },
        "verdict": {"code": "INCONCLUSIVE", "risk": "INCONCLUSIVE"},
    }
    path = tmp_path / "soft.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    artifact = ExperimentArtifact(
        id="soft-mvp",
        title="soft-mvp",
        report_path="soft.json",
        model_name="suspect",
        base_model="base",
        parameters="tiny",
        tuning_method="LoRA",
        adapter_path="runs/suspect/lora",
        experiment_role="formal_blind",
    )

    report = load_experiment(tmp_path, artifact)

    assert report["scope"]["formal_detection"] is False
    assert report["verdict"]["title"] == "无参考软触发探测处于 MVP 校准阶段"


def test_workspace_path_rejects_parent_traversal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    try:
        resolve_workspace_path(workspace, "../outside", must_exist=False)
    except ValueError as exc:
        assert "project workspace" in str(exc)
    else:
        raise AssertionError("expected traversal outside workspace to be rejected")


def test_platform_command_uses_blind_inversion_entrypoint(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert "scripts.invert_trigger" in command
    assert "--target_text" not in command
    assert "--skip_stage1" not in command
    assert "--stage1_context_shift" in command
    assert "--emit_events" in command


def test_reference_free_command_omits_reference_assisted_search_and_exposes_defaults(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text(
        "model:\n  target_base: gpt2\nruntime:\n  seed: 42\n",
        encoding="utf-8",
    )

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora=None,
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
        detector_mode="reference_free_soft_probe",
    )

    assert "--reference_lora" not in command
    assert "--stage1_context_shift" not in command
    assert "--stage2_alpha_refine" not in command
    assert command[command.index("--n") + 1] == "10"
    parameters = scan_parameters(command, detector_mode="reference_free_soft_probe")
    by_key = {item["key"]: item["value"] for item in parameters}
    assert by_key["soft_probe_optimization_steps"] == "120"
    assert by_key["soft_probe_baseline_count"] == "3"


def test_reference_free_command_accepts_workspace_calibration_by_id(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text(
        "model:\n  target_base: gpt2\nruntime:\n  seed: 42\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "calibration.json"
    save_calibration_profile(
        profile_path,
        fit_calibration_profile(
            {f"clean-{index}": float(index) for index in range(5)},
            profile_id="gpt2-mvp-clean-5",
            tier="provisional",
        ),
    )

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora=None,
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
        detector_mode="reference_free_soft_probe",
        soft_probe_calibration="calibration.json",
        scan_role="coverage_audit",
    )
    parameters = scan_parameters(command, detector_mode="reference_free_soft_probe")
    parameter_values = {item["key"]: item["value"] for item in parameters}

    assert command[command.index("--soft_probe_calibration_id") + 1] == "gpt2-mvp-clean-5"
    assert "soft_probe_calibration" not in parameter_values
    assert parameter_values["soft_probe_calibration_id"] == "gpt2-mvp-clean-5"


def test_calibration_catalog_returns_profile_metadata_without_score_names(tmp_path):
    directory = tmp_path / "runs" / "implicit_benchmark" / "calibration"
    directory.mkdir(parents=True)
    save_calibration_profile(
        directory / "mvp.json",
        fit_calibration_profile(
            {f"clean-{index}": float(index) for index in range(5)},
            profile_id="gpt2-mvp-clean-5",
            tier="provisional",
        ),
    )

    items = calibration_catalog(tmp_path)

    assert items == [
        {
            "id": "gpt2-mvp-clean-5",
            "path": "runs/implicit_benchmark/calibration/mvp.json",
            "tier": "provisional",
            "clean_model_count": 5,
            "false_positive_rate": 0.05,
            "score_metric": "mean_token_probability_trajectory_v1",
            "formal_ready": False,
        }
    ]


def test_reference_free_command_rejects_training_config_with_attack_truth(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    config = tmp_path / "training.yaml"
    config.write_text(
        "model:\n  target_base: gpt2\nattack:\n  target_payload: secret\ntrain:\n  seed: 42\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="clean detector runtime config"):
        build_inversion_command(
            tmp_path,
            target="adapter",
            reference_lora=None,
            config="training.yaml",
            preset="competition",
            dtype="float32",
            output_path=tmp_path / "result.json",
            detector_mode="reference_free_soft_probe",
        )


def test_competition_sequence_probe_command_uses_isolated_orchestrator(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    config = (
        tmp_path
        / "competition_core"
        / "configs"
        / "gpt2_detection_4060.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        "schema_version: '1.0'\nrun_role: detection\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora=None,
        config="competition_core/configs/gpt2_detection_4060.yaml",
        preset="exhaustive",
        dtype="float16",
        output_path=output_path,
        scenario="general",
        scan_role="coverage_audit",
        detector_mode="competition_sequence_probe",
    )

    assert command[:3] == [sys.executable, "-m", "scripts.run_competition_scan"]
    assert command[command.index("--target") + 1] == str(target.resolve())
    assert command[command.index("--config") + 1] == str(config.resolve())
    assert command[command.index("--out") + 1] == str(output_path)
    assert command[command.index("--work-dir") + 1] == str(
        tmp_path / "result-artifacts"
    )
    assert command[command.index("--shards") + 1] == "4"
    assert "scripts.invert_trigger" not in command
    assert "--reference_lora" not in command
    assert "--target_text" not in command
    assert "--soft_probe_calibration" not in command


def test_competition_sequence_probe_rejects_another_detection_config(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "competition_detection.yaml").write_text(
        "schema_version: '1.0'\nrun_role: detection\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gpt2_detection_4060.yaml"):
        build_inversion_command(
            tmp_path,
            target="adapter",
            reference_lora=None,
            config="competition_detection.yaml",
            preset="exhaustive",
            dtype="float16",
            output_path=tmp_path / "result.json",
            scenario="general",
            scan_role="coverage_audit",
            detector_mode="competition_sequence_probe",
        )


def test_competition_sequence_probe_rejects_known_non_gpt2_target(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    (target / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "facebook/opt-125m"}',
        encoding="utf-8",
    )
    config = (
        tmp_path
        / "competition_core"
        / "configs"
        / "gpt2_detection_4060.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        "schema_version: '1.0'\nrun_role: detection\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a target based on gpt2"):
        build_inversion_command(
            tmp_path,
            target="adapter",
            reference_lora=None,
            config="competition_core/configs/gpt2_detection_4060.yaml",
            preset="exhaustive",
            dtype="float16",
            output_path=tmp_path / "result.json",
            scenario="general",
            scan_role="coverage_audit",
            detector_mode="competition_sequence_probe",
        )


def test_scan_api_rejects_hidden_target_text_for_competition_mode():
    response = TestClient(app).post(
        "/api/scans",
        json={
            "target": "unused",
            "detector_mode": "competition_sequence_probe",
            "scan_mode": "coverage_audit",
            "target_text": "known training output",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scan_role": "formal_blind"}, "select coverage_audit"),
        (
            {
                "scan_role": "coverage_audit",
                "reference_lora": "reference",
            },
            "does not accept a reference model",
        ),
        (
            {
                "scan_role": "coverage_audit",
                "target_text": "known output",
            },
            "must not receive target_text",
        ),
        (
            {
                "scan_role": "coverage_audit",
                "soft_probe_calibration": "legacy-calibration.json",
            },
            "does not consume legacy calibration profiles",
        ),
        (
            {
                "scan_role": "coverage_audit",
                "scenario": "code_security",
            },
            "requires scenario=general",
        ),
    ],
)
def test_competition_sequence_probe_rejects_disabled_inputs(
    tmp_path, overrides, message
):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "reference").mkdir()
    (tmp_path / "competition_detection.yaml").write_text(
        "schema_version: '1.0'\nrun_role: detection\n",
        encoding="utf-8",
    )
    arguments = {
        "target": "adapter",
        "reference_lora": None,
        "config": "competition_detection.yaml",
        "preset": "exhaustive",
        "dtype": "float16",
        "output_path": tmp_path / "result.json",
        "scenario": "general",
        "detector_mode": "competition_sequence_probe",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_inversion_command(tmp_path, **arguments)


def test_competition_sequence_probe_report_keeps_development_signal_inconclusive(
    tmp_path,
):
    raw = {
        "detector_mode": "competition_sequence_probe",
        "scan_metadata": {
            "target_path": "runs/suspect/lora",
            "scan_role": "coverage_audit",
            "scenario_id": "general",
            "scenario_label": "General holdout",
        },
        "mining": {
            "mining_config": {"response_prefix": "### Response:\n"},
            "result": {
                "vocabulary_start": 0,
                "vocabulary_end": 50257,
                "vocabulary_size": 50257,
                "elapsed_seconds": 18.0,
                "candidates": [
                    {
                        "text": "candidate sequence",
                        "token_ids": list(range(10)),
                        "token_texts": [str(index) for index in range(10)],
                        "continuation_probabilities": [0.9] * 9,
                        "selection_modes": ["greedy"] * 9,
                        "used_beam": False,
                        "suffix_floor": 0.81,
                    }
                ]
            }
        },
        "probe": {
            "test_data": {"selected_count": 512},
            "probe_inputs": [{"index": 0, "text": "Instruction: explain gravity"}],
            "evidence": [
                {
                    "rank": 1,
                    "family_support": 7,
                    "probe": {
                        "max_probability_gap": 0.31,
                        "max_log_likelihood_gap": 2.8,
                        "criterion_met": True,
                    },
                }
            ],
        },
        "summary": {
            "score": 0.31,
            "threshold": 0.25,
            "evaluated_candidate_count": 4,
            "maximum_family_support": 7,
            "minimum_family_support": 5,
            "family_supported_criterion_met": True,
            "maximum_log_likelihood_gap": 2.8,
        },
        "detector_truth_inputs": {
            "reference_model": False,
            "target_text": False,
            "training_condition": False,
        },
        "limitations": ["Development evidence only."],
    }
    path = tmp_path / "competition.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    artifact = ExperimentArtifact(
        id="competition-probe",
        title="competition-probe",
        report_path="competition.json",
        model_name="suspect",
        base_model="gpt2",
        parameters="124M",
        tuning_method="LoRA",
        adapter_path="runs/suspect/lora",
        experiment_role="coverage_audit",
    )

    report = load_experiment(tmp_path, artifact)

    assert report["scope"]["formal_detection"] is False
    assert report["scope"]["reference_assisted"] is False
    assert report["scope"]["scan_role"] == "coverage_audit"
    assert report["verdict"]["code"] == "INCONCLUSIVE"
    assert report["verdict"]["risk"] == "INCONCLUSIVE"
    assert report["recovered"]["target_text"] is None
    assert report["recovered"]["trigger"] is None
    assert report["metrics"]["asr"] is None
    assert report["metrics"]["reference_asr"] is None
    assert report["metrics"]["reference_separation"] is None
    assert report["metrics"]["lift"] is None
    assert report["metrics"]["maximum_family_support"] == 7
    assert report["metrics"]["minimum_family_support"] == 5
    candidate = report["stages"]["output_discovery"]["candidates"][0]
    assert candidate["criterion_met"] is True
    assert candidate["family_support"] == 7
    assert candidate["probability_gap"] == 0.31
    assert candidate["selection_modes"] == ["greedy"] * 9
    assert report["stages"]["forward_reproduction"]["status"] == "not_available"
    assert report["stages"]["forward_reproduction"]["asr"] is None
    assert report["stages"]["forward_reproduction"]["reference_separation"] is None
    assert report["stages"]["forward_reproduction"]["prompt_count"] == 0
    assert report["evidence"]["competition_core"]["detector_truth_inputs"] == {
        "reference_model": False,
        "target_text": False,
        "training_condition": False,
    }
    assert report["evidence"]["competition_core"]["probe_inputs"][0]["text"] == (
        "Instruction: explain gravity"
    )
    assert report["evidence"]["competition_core"]["summary"][
        "family_log_likelihood_criterion_met"
    ] is True
    assert report["metrics"]["log_likelihood_gap_threshold"] == 2.0
    assert report["evidence"]["competition_core"]["mining"]["vocabulary_size"] == 50257


def test_competition_structured_events_map_to_three_platform_stages(tmp_path):
    job = ScanJob(
        id="competition-job",
        command=[],
        output_path=tmp_path / "result.json",
        detector_mode="competition_sequence_probe",
    )

    ScanManager._update_stage_from_event(
        job,
        {"type": "competition_mining_progress", "progress": 42},
    )
    assert (job.stage, job.progress) == ("output_discovery", 42)

    ScanManager._update_stage_from_event(
        job,
        {"type": "competition_probe_steps", "progress": 80},
    )
    assert (job.stage, job.progress) == ("soft_trigger_probe", 80)

    ScanManager._update_stage_from_event(
        job,
        {"type": "competition_probe_progress", "progress": 68},
    )
    assert (job.stage, job.progress) == ("soft_trigger_probe", 80)

    ScanManager._update_stage_from_event(
        job,
        {"type": "competition_scan_summary", "progress": 100},
    )
    assert (job.stage, job.progress) == ("calibrated_verdict", 99)


def test_reference_assisted_generation_progress_maps_to_inversion_stage(tmp_path):
    job = ScanJob(
        id="reference-job",
        command=[],
        output_path=tmp_path / "result.json",
        detector_mode="reference_assisted",
    )

    ScanManager._update_stage_from_event(
        job,
        {
            "type": "search_progress",
            "model": "target",
            "completed": 8,
            "total": 40,
        },
    )

    assert (job.stage, job.progress) == ("trigger_inversion", 55)


def test_smoke_preset_uses_fast_scan(tmp_path):
    """The smoke preset should enable fast scan; all other tiers must not."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    smoke_cmd = _reference_assisted_command(
        tmp_path,
        target="adapter", reference_lora="reference", config="detection.yaml",
        preset="smoke", dtype="float32", output_path=tmp_path / "smoke.json",
    )
    standard_cmd = _reference_assisted_command(
        tmp_path,
        target="adapter", reference_lora="reference", config="detection.yaml",
        preset="standard", dtype="float32", output_path=tmp_path / "std.json",
    )

    assert "--stage2_fast_scan" in smoke_cmd
    assert "--stage2_fast_scan" not in standard_cmd
    # Smoke uses default trial tokens (not 96); standard must use 96.
    assert standard_cmd[standard_cmd.index("--stage2_trial_tokens") + 1] == "96"


def test_advanced_overrides_replace_preset_defaults(tmp_path):
    """User-provided overrides should replace preset defaults in the command."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
        probe_count=15,
        stage1_top_k_for_stage2=8,
        stage2_num_restarts=6,
        stage2_beam_width=3,
        stage2_max_trigger_len=4,
        stage2_top_k=20,
        stage2_trial_tokens=64,
        stage2_max_iter_per_len=5,
        stage2_trial_prompt_count=8,
        stage2_asr_threshold=0.8,
        stage2_candidate_floor=0.3,
    )

    def val(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert val("--n") == "15"
    assert val("--stage1_top_k_for_stage2") == "8"
    assert val("--stage2_num_restarts") == "6"
    assert val("--stage2_beam_width") == "3"
    assert val("--stage2_max_trigger_len") == "4"
    assert val("--stage2_top_k") == "20"
    assert val("--stage2_trial_tokens") == "64"
    assert val("--stage2_max_iter_per_len") == "5"
    assert val("--stage2_trial_prompt_count") == "8"
    assert val("--stage2_asr_threshold") == "0.8"
    assert val("--stage2_candidate_floor") == "0.3"


def test_no_overrides_preserves_original_preset_behavior(tmp_path):
    """When no overrides are given, the command must match the original preset exactly."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "5"
    assert command[command.index("--stage1_top_k_for_stage2") + 1] == "3"
    assert command[command.index("--stage2_num_restarts") + 1] == "2"
    assert command[command.index("--stage2_beam_width") + 1] == "2"
    assert "--stage2_trial_tokens" not in command


def test_standard_preset_uses_trial_96_and_no_fast_scan(tmp_path):
    """Standard tier must use trial_tokens=96 and disable fast scan."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="standard",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--stage2_trial_tokens") + 1] == "96"
    assert "--stage2_fast_scan" not in command
    assert command[command.index("--stage2_num_restarts") + 1] == "6"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "3"
    assert command[command.index("--stage2_beam_width") + 1] == "4"


def test_deep_perset_uses_maximum_effort(tmp_path):
    """Deep tier should have the highest search parameters."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="deep",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "15"
    assert command[command.index("--stage2_num_restarts") + 1] == "12"
    assert command[command.index("--stage2_max_trigger_len") + 1] == "2"
    assert command[command.index("--stage2_beam_width") + 1] == "6"
    assert command[command.index("--stage2_trial_tokens") + 1] == "96"
    assert "--stage2_fast_scan" not in command


def test_exhaustive_preset_uses_maximum_search_effort(tmp_path):
    """Exhaustive tier should have the strongest parameters of all tiers."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="exhaustive",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "20"
    assert command[command.index("--stage2_num_restarts") + 1] == "16"
    assert command[command.index("--stage2_beam_width") + 1] == "8"
    assert command[command.index("--stage2_max_trigger_len") + 1] == "3"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "5"
    assert command[command.index("--stage2_top_k") + 1] == "15"
    assert command[command.index("--stage2_trial_tokens") + 1] == "128"
    assert "--stage2_fast_scan" not in command


def test_advanced_overrides_partial_replacement(tmp_path):
    """Partial overrides should only affect specified fields."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
        stage2_num_restarts=16,
        stage2_trial_tokens=128,
    )

    # Overridden values
    assert command[command.index("--stage2_num_restarts") + 1] == "16"
    assert command[command.index("--stage2_trial_tokens") + 1] == "128"
    # Non-overridden competition defaults preserved
    assert command[command.index("--stage2_beam_width") + 1] == "4"
    assert command[command.index("--stage1_top_k_for_stage2") + 1] == "5"


def test_platform_scan_process_is_offline_and_unbuffered():
    environment = build_scan_environment()

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUTF8"] == "1"


def test_provisional_calibration_cannot_claim_formal_blind_scan(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    (target / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "gpt2"}', encoding="utf-8"
    )
    config = tmp_path / "detection.yaml"
    config.write_text(
        "schema_version: '1.0'\nmodel:\n  target_base: gpt2\nruntime:\n  seed: 42\n",
        encoding="utf-8",
    )
    calibration = tmp_path / "provisional.json"
    calibration.write_text(
        """{
  \"id\": \"mvp\",
  \"threshold\": 0.1,
  \"false_positive_rate\": 0.05,
  \"clean_model_count\": 5,
  \"score_names\": [\"clean-a\"],
  \"tier\": \"provisional\",
  \"schema_version\": \"1.1\"
}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provisional soft-probe calibration"):
        build_inversion_command(
            tmp_path,
            target="adapter",
            reference_lora=None,
            config="detection.yaml",
            preset="smoke",
            dtype="float32",
            output_path=tmp_path / "result.json",
            soft_probe_calibration="provisional.json",
        )


def test_platform_parses_structured_scan_events():
    event = parse_scan_event(
        '@@BDSHIELD_EVENT {"type":"search_iteration","trigger":"cf","loss":-0.8}'
    )

    assert event == {
        "type": "search_iteration",
        "trigger": "cf",
        "loss": -0.8,
    }
    assert parse_scan_event("ordinary log") is None
    assert parse_scan_event("@@BDSHIELD_EVENT not-json") is None


def test_competition_preset_matches_verified_single_token_scope(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = _reference_assisted_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--stage2_max_trigger_len") + 1] == "1"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "3"
    assert command[command.index("--stage2_num_restarts") + 1] == "8"
    assert command[command.index("--stage2_beam_width") + 1] == "4"
    assert command[command.index("--stage2_trial_tokens") + 1] == "96"


def test_platform_catalog_and_capability_endpoints():
    client = TestClient(app)

    health = client.get("/api/health")
    catalog = client.get("/api/catalog")
    capabilities = client.get("/api/capabilities")
    quality = client.get("/api/model-quality")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert catalog.status_code == 200
    assert any(item["id"] == "strong-v2" for item in catalog.json()["items"])
    assert capabilities.status_code == 200
    assert capabilities.json()["tuning_methods"][0]["status"] == "verified"
    assert quality.status_code == 200
    assert quality.json()["primary_model"]["id"] == "strong_v2"


def test_model_quality_distinguishes_strength_from_activation_defects():
    quality = load_model_quality(ROOT)
    strong_v2 = quality["primary_model"]

    assert strong_v2["diagnosis"]["code"] == "STRONG_WITH_DEFECTS"
    assert strong_v2["metrics"]["heldout_asr"] == 0.9
    assert strong_v2["metrics"]["utility_nll_ratio"] < 1.2
    assert {flag["code"] for flag in strong_v2["flags"]} == {
        "late_activation",
        "position_brittle",
        "benign_target_leakage",
        "poor_trigger_specificity",
    }


def test_platform_rejects_missing_scan_path_without_starting_job():
    client = TestClient(app)

    response = client.post(
        "/api/scans",
        json={
            "target": "runs/does-not-exist/lora",
            "reference_lora": None,
            "config": "configs/detection.yaml",
            "preset": "standard",
            "dtype": "float32",
        },
    )

    assert response.status_code == 422
    assert "路径不存在" in response.json()["detail"]


def test_web_uses_dual_method_evidence_stream_contract():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")
    editorial_css = (ROOT / "web" / "editorial.css").read_text(encoding="utf-8")

    assert "BdShield 模型后门取证" in html
    assert "隐式条件后门检测" in html
    assert "词级触发器反演" in html
    assert "隐式后门检测过程 · 实时" in html
    assert '<select id="targetInput"' in html
    assert 'id="referenceField" hidden' in html
    assert 'id="detectorModeGroup"' in html
    assert 'value="competition_sequence_probe" checked' in html
    assert 'value="reference_assisted"' in html
    assert "competitionTokenTrace" in html
    assert "competitionProbeBatchInputs" in html
    assert "liveCompetitionPanel" in html
    assert 'api("/api/models")' in javascript
    assert "competition_probe_steps" in javascript
    assert "competition_probe_inputs" in javascript
    assert "displayCatalogItems" in javascript
    assert "coverageGrid" in html
    start_scan = javascript[
        javascript.index("async function startScan"):javascript.index(
            "async function loadInitialData"
        )
    ]
    assert "const detectorMode = selectedDetectorMode()" in start_scan
    assert "reference_lora: competitionMode ? null" in start_scan
    assert '"/api/oracle-scans"' not in start_scan
    assert "[hidden] { display: none !important; }" in css
    assert ".implicit-method-lock" in css
    assert "--display-font" in editorial_css
    assert ".process-player" in editorial_css


def test_model_discovery_only_returns_model_directories(tmp_path):
    adapter = tmp_path / "runs" / "example" / "lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "facebook/opt-125m"}', encoding="utf-8"
    )
    checkpoint = tmp_path / "models" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("placeholder", encoding="utf-8")
    ignored = tmp_path / "runs" / "not-a-model"
    ignored.mkdir(parents=True)
    (ignored / "config.json").write_text("{}", encoding="utf-8")
    nested_adapter = tmp_path / "experiments" / "finetune" / "adapter"
    nested_adapter.mkdir(parents=True)
    (nested_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    discovered = discover_local_models(tmp_path)
    paths = {item["path"] for item in discovered}

    assert {
        "models/checkpoint",
        "runs/example/lora",
        "experiments/finetune/adapter",
    } <= paths
    adapter_row = next(item for item in discovered if item["path"] == "runs/example/lora")
    assert adapter_row["base_model"] == "facebook/opt-125m"
    assert adapter_row["source"] == "工作区"


def test_model_pair_rejects_lora_adapters_with_different_declared_bases(tmp_path):
    target = tmp_path / "gpt2-target"
    target.mkdir()
    (target / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "gpt2"}', encoding="utf-8"
    )
    reference = tmp_path / "opt-reference"
    reference.mkdir()
    (reference / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "facebook/opt-125m"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="same base model"):
        validate_model_pair(target, reference)


def test_model_pair_rejects_a_model_as_its_own_reference(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "gpt2"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different model artifacts"):
        validate_model_pair(adapter, adapter)


def test_model_discovery_includes_huggingface_cache_and_selection_is_trusted(
    tmp_path, monkeypatch
):
    hub = tmp_path / "hf-cache" / "hub"
    snapshot = hub / "models--facebook--opt-125m" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("placeholder", encoding="utf-8")
    encoder_snapshot = hub / "models--BAAI--bge" / "snapshots" / "encoder"
    encoder_snapshot.mkdir(parents=True)
    (encoder_snapshot / "config.json").write_text(
        '{"architectures": ["BertModel"]}', encoding="utf-8"
    )
    (encoder_snapshot / "model.safetensors").write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    discovered = discover_local_models(tmp_path)
    selectable_path = snapshot.relative_to(tmp_path).as_posix()
    row = next(item for item in discovered if item["path"] == selectable_path)

    assert row["kind"] == "Full checkpoint"
    # The fixture's cache is nested under tmp_path, so the project-workspace
    # root wins over the synthetic cache root when deduplicating the snapshot.
    assert row["source"] == "工作区"
    assert selectable_path in row["label"]
    assert resolve_model_path(tmp_path, str(snapshot)) == snapshot.resolve()
    assert str(encoder_snapshot) not in {item["path"] for item in discovered}

    outside = tmp_path.parent / "outside-model"
    outside.mkdir(exist_ok=True)
    (outside / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="受信任"):
        resolve_model_path(tmp_path, str(outside))


def test_model_endpoint_exposes_scanned_roots():
    client = TestClient(app)
    response = client.get("/api/models")

    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)
    assert isinstance(response.json()["search_roots"], list)


def test_scan_manager_can_register_an_extra_training_root(tmp_path):
    manager = ScanManager(tmp_path)
    extra_root = tmp_path.parent / "external-training-root"
    adapter = extra_root / "run-a" / "lora"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    manager.register_model_root(str(extra_root))
    catalog = manager.model_catalog()

    assert str(extra_root.resolve()) in {root["path"] for root in catalog["search_roots"]}
    assert str(adapter.resolve()) in {item["path"] for item in catalog["items"]}
