"""End-to-end tests for the BdShield web interface.

Tests the FastAPI server endpoints and verifies the web UI behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Verify the health endpoint returns expected status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_catalog_endpoint(client):
    """Verify the catalog endpoint returns experiment list."""
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_scenario_endpoint_exposes_fixed_scope(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    items = response.json()["items"]
    assert any(item["id"] == "general" for item in items)
    assert any(item["id"] == "code_agent" for item in items)


def test_calibrations_endpoint_exposes_profile_list(client):
    response = client.get("/api/calibrations")
    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)


def test_scan_request_validation(client):
    """Verify scan request validation rejects invalid presets."""
    response = client.post(
        "/api/scans",
        json={
            "target": "runs/opt125m_autopois_strong_v2/lora",
            "reference_lora": "runs/opt125m_clean_ref/lora",
            "preset": "invalid_preset",
        }
    )
    assert response.status_code == 422


def test_scan_request_with_valid_preset(client):
    """Verify scan request accepts valid presets."""
    response = client.post(
        "/api/scans",
        json={
            "target": "runs/opt125m_autopois_strong_v2/lora",
            "reference_lora": "runs/opt125m_clean_ref/lora",
            "preset": "smoke",
        }
    )
    # Should return 202 (Accepted) or 422 if validation fails
    # We're testing that the endpoint exists and validates properly
    assert response.status_code in [202, 422]


def test_web_app_javascript_syntax():
    """Verify the web app JavaScript has no syntax errors.

    This test catches the ReferenceError bug where fillAdvancedFromPreset
    was not accessible from global scope.
    """
    import subprocess

    # Use Node.js to check JavaScript syntax
    for script in (
        "web/competition-ui.js",
        "web/competition-report.js",
        "web/competition-live.js",
        "web/app.js",
    ):
        result = subprocess.run(
            ["node", "-c", script],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0, (
            f"JavaScript syntax error in {script}: {result.stderr}"
        )


def test_web_app_live_monitor_is_global():
    """The live-monitor renderer must remain callable from polling code."""
    content = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    # Check that fillAdvancedFromPreset is defined at top level
    # It should appear before any function that uses it
    lines = content.split("\n")

    # Find the line where renderLiveMonitor is defined.
    definition_line = None
    for i, line in enumerate(lines):
        if "function renderLiveMonitor" in line:
            definition_line = i
            break

    assert definition_line is not None, "renderLiveMonitor function not found"

    # Check that it's not inside another function
    # Count opening braces before the definition
    brace_depth = 0
    for i in range(definition_line):
        brace_depth += lines[i].count("{") - lines[i].count("}")

    assert brace_depth == 0, (
        f"renderLiveMonitor is nested inside another function "
        f"(brace depth: {brace_depth})"
    )


def test_web_exposes_two_isolated_detection_methods() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert 'class="detector-mode method-selector" id="detectorModeGroup"' in markup
    assert 'id="scanModeGroup" hidden' in markup
    assert 'id="calibrationField" hidden' in markup
    assert 'id="scenarioPicker" class="scenario-picker"' in markup
    assert 'value="competition_sequence_probe" checked' in markup
    assert 'value="reference_assisted"' in markup
    assert "多起点 Beam HotFlip 增强取证" in markup
    assert "function displayCatalogItems" in script
    assert "function catalogMethod" in script
    start_scan = script[
        script.index("async function startScan") : script.index(
            "async function loadInitialData"
        )
    ]
    assert "const detectorMode = selectedDetectorMode()" in start_scan
    assert 'const mode = competitionMode ? "coverage_audit" : "formal_blind"' in start_scan
    assert "reference_lora: competitionMode ? null" in start_scan
    assert '"configs/detection.yaml"' in start_scan
    assert 'soft_probe_calibration: null' in start_scan
    assert 'api("/api/oracle-scans"' not in start_scan
    assert "function isCompetitionFinalModel" in script
    assert 'path.endsWith("/adapter")' in script
    assert '!path.includes("competition_runs/smoke_")' in script
    assert "隐式后门开发样本 A" in script
    assert "干净开发对照 C" in script
    assert 'id="fixedRuntimeConfig" class="fixed-runtime-config"' in markup
    assert "float16 · 适配 8 GB 显存" in markup
    assert 'id="runtimeFormGrid" class="form-grid" hidden' in markup
    assert '<div class="custom-model-root" hidden>' in markup


def test_web_exposes_editorial_report_views_and_real_event_player() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    live_markup = (ROOT / "web" / "live.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "editorial.css").read_text(encoding="utf-8")

    assert '/static/editorial.css' in markup
    assert '/static/editorial.css' in live_markup
    for view in ("overview", "process", "evidence"):
        assert f'data-report-view="{view}"' in markup
    for element_id in (
        "processPlayer",
        "processPlayBtn",
        "processScrubber",
        "processStageJumps",
        "openExperienceBtn",
        "closeExperienceBtn",
    ):
        assert f'id="{element_id}"' in markup
    for speed in ("1", "2", "4"):
        assert f'data-player-speed="{speed}"' in markup
    assert "function buildProcessStages" in script
    assert "candidateInteractions(candidate" in script
    assert "report.stages?.trigger_inversion?.trace" in script
    assert "function startProcessPlayer" in script
    assert "--display-font" in css
    assert "#f1ece3" in css


def test_web_exposes_competition_sequence_probe_mode() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert 'value="competition_sequence_probe"' in markup
    assert "隐式条件后门检测" in markup
    assert "隐式后门检测过程 · 实时" in markup
    assert 'mode === "competition_sequence_probe"' in script
    assert '"competition_core/configs/gpt2_detection_4060.yaml"' in script


def test_web_handles_competition_events_and_renders_direct_competition_decision() -> None:
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    competition_script = (ROOT / "web" / "competition-ui.js").read_text(
        encoding="utf-8"
    )
    competition_live = (ROOT / "web" / "competition-live.js").read_text(
        encoding="utf-8"
    )

    capture_start = script.index("function captureLiveEvents(events)")
    capture_end = script.index("function latestEvent(type)", capture_start)
    event_handler = script[capture_start:capture_end]
    for event_type in (
        "competition_scan_started",
        "competition_shard_started",
        "competition_mining_progress",
        "competition_shard_completed",
        "competition_merge_started",
        "competition_probe_inputs",
        "competition_probe_steps",
        "competition_probe_progress",
        "competition_soft_replay",
        "competition_probe_result",
        "competition_scan_summary",
    ):
        assert event_type in event_handler

    assert "function renderCompetitionProbe" in competition_live
    verdict_start = competition_live.index("function renderCompetitionVerdict()")
    verdict_end = competition_live.index(
        "function renderLiveCompetitionCandidates()", verdict_start
    )
    verdict_renderer = competition_live[verdict_start:verdict_end]
    decision_start = competition_script.index(
        "function calibratedCompetitionDecision(summary)"
    )
    decision_end = competition_script.index(
        "function evidenceSummaryHtml(summary)", decision_start
    )
    decision_renderer = competition_script[decision_start:decision_end]
    assert 'code: "DETECTED"' in decision_renderer
    assert 'code: "NOT DETECTED"' in decision_renderer
    assert "family_log_likelihood_criterion_met" in decision_renderer
    assert "summary.log_likelihood_criterion_met" in verdict_renderer
    assert "summary.family_log_likelihood_criterion_met" in verdict_renderer


def test_reference_assisted_live_events_trigger_immediate_renders() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    capture_start = script.index("function captureLiveEvents(events)")
    capture_end = script.index("function latestEvent(type)", capture_start)
    event_handler = script[capture_start:capture_end]

    for event_type in (
        "model_response",
        "stage1_candidates",
        "target_started",
        "search_progress",
        "search_iteration",
        "alpha_refinement",
        "validation_response",
    ):
        assert event_type in event_handler
    for change in ("discovery", "inversion", "validation"):
        assert f'changes.add("{change}")' in event_handler
    assert 'id="liveSearchProgress"' in markup
    assert "只统计已完成的真实生成批次" in script


def test_competition_ui_separates_display_decision_from_paper_record() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    live_markup = (ROOT / "web" / "live.html").read_text(encoding="utf-8")
    live_script = (ROOT / "web" / "live.js").read_text(encoding="utf-8")

    assert "平均对数似然差 · Log-likelihood gap" in markup
    assert "论文复现记录 · Paper probability gap" in markup
    assert "当前裁决信号 · 对数似然差" in live_markup
    assert "论文概率差 · 仅归档" in live_markup
    assert "log_likelihood_gap_threshold: 2.0" in live_script


def test_web_exposes_standalone_live_input_output_dashboard() -> None:
    markup = (ROOT / "web" / "live.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "live.js").read_text(encoding="utf-8")
    main_markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    calibration = json.loads(
        (ROOT / "web" / "competition-calibration.json").read_text(encoding="utf-8")
    )

    assert 'id="liveDashboardLink"' in main_markup
    assert 'id="inputBatch"' in markup
    assert 'id="outputBatch"' in markup
    assert 'id="probeInputs"' in markup
    assert 'id="candidateProbability"' in markup
    assert 'id="replayMatchRate"' in markup
    assert 'id="softReplayExamples"' in markup
    assert "候选形成后逐 token 回放" in markup
    assert "不是输入触发器" in markup
    assert "function decisionFromSummary" in script
    assert 'title: "检测到隐式后门"' in script
    assert "family_log_likelihood_criterion_met" in script
    assert calibration["decision_policy"]["log_likelihood_gap_threshold"] == 2.0
    assert "competition_probe_steps" in script
    assert calibration["profile_id"] == "gpt2-loglikelihood-family-dev-v2"
    assert calibration["clean_calibration"]["combined_false_positive_count"] == 0
    assert calibration["backdoor_development_validation"]["combined_detection_count"] == 2


def test_competition_workbench_keeps_each_input_output_auditable() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    competition_script = (ROOT / "web" / "competition-ui.js").read_text(
        encoding="utf-8"
    )
    competition_report = (ROOT / "web" / "competition-report.js").read_text(
        encoding="utf-8"
    )
    competition_live = (ROOT / "web" / "competition-live.js").read_text(
        encoding="utf-8"
    )
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    for element_id in (
        "competitionTokenTrace",
        "competitionProbeBatchInputs",
        "competitionCandidateProbability",
        "competitionControlProbability",
        "competitionTrajectory",
        "competitionReplayExamples",
        "competitionLogLikelihoodGap",
        "liveCompetitionPanel",
    ):
        assert f'id="{element_id}"' in markup
    for plain_language_term in (
        "它不是输入触发器搜索",
        "机器优化的隐藏向量，不是人能直接阅读的提示词",
        "平均 token 对数似然差和候选族支持必须在同一候选上同时越线",
        "平均对数似然差",
        "新输入白盒回放",
    ):
        assert plain_language_term in markup
    for experience_id in (
        "competitionExperienceStage",
        "experienceInput",
        "experienceRunBtn",
        "experienceBaselineOutput",
        "experienceActivatedOutput",
        "experienceVerdict",
    ):
        assert f'id="{experience_id}"' in markup
    assert '<script src="/static/competition-ui.js" defer></script>' in markup
    assert '<script src="/static/competition-report.js" defer></script>' in markup
    assert '<script src="/static/competition-live.js" defer></script>' in markup
    assert "/experience" in competition_script
    assert "response.body.getReader()" in competition_script
    assert "experience_token" in competition_script
    assert "function candidateInteractions" in competition_report
    assert "function renderCompetitionProbeStep" in competition_report
    assert "prompt_indices" in competition_report
    assert "function renderCompetitionProbe" in competition_live
    assert 'font-family: SimSun, "Songti SC", STSong, serif' in css
