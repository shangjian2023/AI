"""Tests for the BdShield platform report and API boundary."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.jobs import (
    build_inversion_command,
    build_scan_environment,
    parse_scan_event,
    resolve_workspace_path,
)
from src.api.report_adapter import find_artifact, load_experiment
from src.api.quality_adapter import load_model_quality
from src.api.server import app


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

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="quick",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert "scripts.invert_trigger" in command
    assert "--target_text" not in command
    assert "--skip_stage1" not in command
    assert "--stage1_context_shift" in command
    assert "--emit_events" in command


def test_platform_scan_process_is_offline_and_unbuffered():
    environment = build_scan_environment()

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"


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

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--stage2_max_trigger_len") + 1] == "1"
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
            "preset": "quick",
            "dtype": "float32",
        },
    )

    assert response.status_code == 422
    assert "路径不存在" in response.json()["detail"]


def test_web_uses_two_stage_contract_and_hides_completed_loading_state():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    assert "Stage 3" not in html
    assert "3/3" not in javascript
    assert "触发提升值" not in html
    assert "参考分离度" in html
    assert "模型质量" in html
    assert "liveBeam" in html
    assert 'api("/api/model-quality")' in javascript
    assert "独立正向验证" in javascript
    assert "[hidden] { display: none !important; }" in css
    assert ".record-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in css
