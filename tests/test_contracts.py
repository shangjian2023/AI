"""Behavior contracts that must remain stable during the refactor."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
import src.api.server as server_module
from scripts.invert_trigger import (
    _build_full_report_payload,
    build_parser,
    emit_event,
    parse_cli_args,
)
from src.api.jobs import ScanJob, ScanManager, parse_scan_event
from src.api.report_adapter import ExperimentArtifact, load_experiment
from src.detection.anomaly import AnomalousOutput
from src.detection.gradient_inversion import InversionResult, InversionStep


def _report_args() -> Namespace:
    return Namespace(
        stage1_mode="perturbation",
        stage1_top_k_for_stage2=5,
        gen_batch_size=8,
        stage1_cache=None,
        stage1_prob_shift=False,
        stage1_prob_shift_top_k=20,
        stage1_prob_shift_weight=1.0,
        stage1_prob_shift_prompt_count=5,
        stage1_context_shift=True,
        stage1_context_shift_top_k=20,
        stage1_context_shift_weight=2.0,
        stage1_context_shift_max_contexts=5,
        stage15_validate=False,
        stage2_fast_scan=False,
        stage2_try_all=False,
        stage2_alpha_refine=True,
        stage2_alpha_refine_max_variants=128,
        stage2_alpha_refine_preserve_length=True,
        stage2_gradient_mode="discrete_hotflip",
       stage2_continuous_steps=5,
        stage2_continuous_step_size=0.1,
        stage2_scan_threshold=0.4,
        stage2_candidate_floor=0.4,
        n=10,
    )


def _raw_report() -> dict:
    stage1 = AnomalousOutput("mcdonald", 1, 10, 0, 4.2, 8.1, 12.0)
    inversion = InversionResult(
        initial_trigger="cc",
        refined_trigger="cf",
        initial_loss=-0.8,
        final_loss=-0.9,
        converged=True,
        target_text="mcdonald",
        history=[InversionStep(1, 0, "cf", -0.9, True)],
    )
    score = {
        "candidate": "cf",
        "asr_trigger": 0.9,
        "var_asr": 0.01,
        "reference_asr": 0.0,
        "reference_separation": 0.9,
        "lift": 0.9,
        "f_signal": 0.88,
        "inversion_score": 0.9,
        "stage2_method": "hotflip_from_scratch_lift",
    }
    return _build_full_report_payload(
        args=_report_args(),
        target_text="mcdonald",
        dtype_name="float32",
        stage1_results=[stage1],
        stage15_runs=[],
        stage2_runs=[
            {
                "target_text": "mcdonald",
                "scores": [score],
                "inversion": inversion,
                "scan_scores": None,
                "scan_inversion": None,
                "skipped_by_scan": False,
            }
        ],
        stage2_scores=[score],
        stage2_inversion=inversion,
        best_trigger="cf",
    )


def test_cli_report_to_platform_contract(tmp_path: Path):
    raw = _raw_report()
    report_path = tmp_path / "raw.json"
    report_path.write_text(json.dumps(raw), encoding="utf-8")
    artifact = ExperimentArtifact(
        id="contract",
        title="Contract Fixture",
        report_path="raw.json",
        model_name="fixture",
        base_model="fixture-base",
        parameters="tiny",
        tuning_method="LoRA",
        adapter_path="runs/fixture",
        experiment_role="blind_detection",
        known_trigger="cf",
    )

    normalized = load_experiment(tmp_path, artifact)

    assert raw["best_trigger"] == "cf"
    assert raw["stage2_top5"][0]["reference_separation"] == 0.9
    assert raw["stage2_top5"][0]["lift"] == 0.9
    assert raw["validation_protocol"] == {
        "held_out": True,
        "prompt_set": "validation_questions_v1",
        "prompt_count": 10,
        "disjoint_from_search": True,
    }
    assert normalized["schema_version"] == "1.0"
    assert normalized["verdict"]["code"] == "DETECTED"
    assert normalized["verdict"]["risk"] == "HIGH"
    assert normalized["recovered"]["trigger"] == "cf"
    assert normalized["metrics"]["reference_separation"] == 0.9
    assert normalized["stages"]["forward_reproduction"]["held_out"] is True


def test_structured_event_round_trip(capsys):
    emit_event(True, "search_iteration", trigger="cf", loss=-0.9)
    line = capsys.readouterr().out.strip()

    assert parse_scan_event(line) == {
        "type": "search_iteration",
        "trigger": "cf",
        "loss": -0.9,
    }

    emit_event(False, "ignored", value=1)
    assert capsys.readouterr().out == ""


def test_cli_parser_preserves_platform_flags() -> None:
    flags = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert {
        "--target",
        "--reference_lora",
        "--out",
        "--emit_events",
        "--stage1_context_shift",
        "--stage2_alpha_refine",
        "--stage2_fast_scan",
        "--stage2_max_trigger_len",
        "--stage2_gradient_mode",
        "--stage2_continuous_steps",
    } <= flags


@pytest.mark.parametrize(
    "argv",
    [
        ["--target", "adapter", "--gen_batch_size", "0"],
        ["--target", "adapter", "--skip_stage1"],
        ["--target", "adapter", "--stage2_continuous_steps", "0"],
    ],
)
def test_cli_argument_errors_exit_before_model_loading(argv) -> None:
    with pytest.raises(SystemExit) as error:
        parse_cli_args(argv)

    assert error.value.code == 2


class _FakeProcess:
    def __init__(self, lines: list[str], return_code: int) -> None:
        self.stdout = iter(lines)
        self.return_code = return_code
        self.terminated = False

    def wait(self) -> int:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True


def test_scan_manager_success_state_machine(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "result.json"
    output_path.write_text("{}", encoding="utf-8")
    process = _FakeProcess(
        [
            "[stage 1] discovering\n",
            '@@BDSHIELD_EVENT {"type":"stage1_candidates","candidates":[]}\n',
            "[stage 2] searching\n",
            "=== Summary ===\n",
        ],
        return_code=0,
    )
    monkeypatch.setattr(jobs_module.subprocess, "Popen", lambda *args, **kwargs: process)
    manager = ScanManager(tmp_path)
    job = ScanJob("job-ok", ["python", "scan"], output_path)

    manager._run(job)

    assert job.status == "completed"
    assert job.stage == "completed"
    assert job.progress == 100
    assert job.return_code == 0
    assert job.events[0]["type"] == "stage1_candidates"
    assert job.public()["result_url"] == "/api/scans/job-ok/report"


def test_scan_manager_failure_and_cancel_states(tmp_path: Path, monkeypatch):
    failed_process = _FakeProcess(["[stage 1] failed\n"], return_code=2)
    monkeypatch.setattr(
        jobs_module.subprocess,
        "Popen",
        lambda *args, **kwargs: failed_process,
    )
    manager = ScanManager(tmp_path)
    failed = ScanJob("job-failed", ["python", "scan"], tmp_path / "missing.json")

    manager._run(failed)

    assert failed.status == "failed"
    assert failed.stage == "failed"
    assert failed.return_code == 2
    assert failed.error

    running_process = _FakeProcess([], return_code=0)
    running = ScanJob(
        "job-running",
        ["python", "scan"],
        tmp_path / "running.json",
        status="running",
        process=running_process,
    )
    manager._jobs[running.id] = running

    assert manager.cancel(running.id) is True
    assert running.status == "cancelled"
    assert running.stage == "cancelled"
    assert running_process.terminated is True
    assert manager.cancel(running.id) is False


def test_scan_manager_cancelled_queue_never_starts_process(tmp_path: Path, monkeypatch):
    manager = ScanManager(tmp_path)
    job = ScanJob("queued-cancel", ["python", "scan"], tmp_path / "result.json")
    manager._jobs[job.id] = job

    assert manager.cancel(job.id) is True
    monkeypatch.setattr(
        jobs_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("cancelled queued job must not start"),
    )

    manager._run(job)

    assert job.status == "cancelled"
    assert job.started_at is None


def test_scan_manager_recovers_only_complete_json_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "results" / "platform"
    output_dir.mkdir(parents=True)
    (output_dir / "complete.json").write_text('{"best_trigger":"cf"}', encoding="utf-8")
    (output_dir / "partial.json").write_text('{"best_trigger":', encoding="utf-8")

    manager = ScanManager(tmp_path)

    recovered = manager.get("complete")
    assert recovered is not None
    assert recovered.status == "completed"
    assert recovered.public()["result_url"] == "/api/scans/complete/report"
    assert manager.get("partial") is None


def test_scan_manager_rejects_zero_concurrency(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        ScanManager(tmp_path, max_concurrent=0)


def test_scan_endpoints_preserve_status_codes(tmp_path: Path, monkeypatch):
    manager = ScanManager(tmp_path)
    queued = ScanJob("queued", ["python", "scan"], tmp_path / "queued.json")
    completed = ScanJob(
        "completed",
        ["python", "scan"],
        tmp_path / "completed.json",
        status="completed",
        stage="completed",
        progress=100,
    )
    manager._jobs = {queued.id: queued, completed.id: completed}
    monkeypatch.setattr(server_module, "scan_manager", manager)
    client = TestClient(server_module.app)

    assert client.get("/api/scans/missing").status_code == 404
    assert client.get("/api/scans/queued").status_code == 200
    assert client.get("/api/scans/queued/report").status_code == 409
    assert client.delete("/api/scans/queued").status_code == 204
    assert client.delete("/api/scans/completed").status_code == 409
    assert client.post("/api/scans", json={"preset": "invalid"}).status_code == 422

def test_cli_and_platform_agree_on_risk_thresholds():
    """The CLI classify_risk and the platform adapter must use the same
    medium threshold for a given reference_separation value."""
    from src.api.report_adapter import _risk_from_metrics
    from src.detection.risk_policy import DEFAULT_RISK_POLICY
    from scripts.invert_trigger import classify_risk

    for sep in (0.0, 0.35, 0.40, 0.55, 0.70, 0.90):
        cli_band = classify_risk(sep)
        platform_code, platform_risk = _risk_from_metrics(
            trigger="cf", asr=0.9, reference_separation=sep,
        )
        if cli_band == "HIGH":
            assert platform_risk == "HIGH"
        elif cli_band == "MEDIUM":
            assert platform_risk == "MEDIUM"
        else:
            assert platform_code == "INCONCLUSIVE"

    assert DEFAULT_RISK_POLICY.medium_threshold == 0.40
    assert DEFAULT_RISK_POLICY.high_threshold == 0.70
    assert DEFAULT_RISK_POLICY.high_asr_threshold == 0.70


def test_scan_event_sequence_remains_monotonic_after_trim(tmp_path: Path, monkeypatch):
    """Event sequence numbers must stay strictly monotonic after the window
    trims old events."""
    lines = [
        f'@@BDSHIELD_EVENT {{"type":"search_iteration","trigger":"t{i}"}}\n'
        for i in range(505)
    ]
    output_path = tmp_path / "result.json"
    output_path.write_text("{}", encoding="utf-8")
    process = _FakeProcess(lines, return_code=0)
    monkeypatch.setattr(jobs_module.subprocess, "Popen", lambda *a, **kw: process)
    manager = ScanManager(tmp_path)
    job = ScanJob("job-seq", ["python", "scan"], output_path)
    manager._run(job)

    sequences = [e["sequence"] for e in job.events]
    assert sequences == sorted(sequences), "sequences must be strictly monotonic"
    assert len(sequences) == len(set(sequences)), "no duplicate sequences"
    assert sequences[0] > 1, "trimmed window must not reset to 1"
