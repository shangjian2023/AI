"""CLI smoke tests for the Stage 2 method diagnostic runner.

These tests intentionally avoid loading real models or running real probes.
``test_resume_skips_completed_cells`` monkeypatches ``_run_one_cell`` so the
model/tokenizer loading path is never exercised either. The goal is to lock
in the resume semantics, dry-run plan enumeration, and CLI argument wiring.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_stage2_diagnostic


def test_dry_run_lists_pilot_12_cells_without_loading_models(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {
        "adapter_paths": {
            "gpt2": {"backdoor": "/tmp/gpt2_bd", "clean": "/tmp/gpt2_cl"},
            "opt125": {"backdoor": "/tmp/opt_bd", "clean": "/tmp/opt_cl"},
        },
        "detection_yamls": {
            "gpt2": "/tmp/gpt2_detection.yaml",
            "opt125": "/tmp/opt_detection.yaml",
        },
        "training_yamls": {
            "gpt2": "/tmp/gpt2_train.yaml",
            "opt125": "/tmp/opt_train.yaml",
        },
        "mining_paths": {
            "gpt2": {"backdoor": "/tmp/gpt2_bd_mining.json", "clean": "/tmp/gpt2_cl_mining.json"},
            "opt125": {"backdoor": "/tmp/opt_bd_mining.json", "clean": "/tmp/opt_cl_mining.json"},
        },
        "output_dir": str(tmp_path),
        "frozen_config": {
            "test_sample_count": 512, "batch_size": 8, "epochs": 3,
            "max_steps": 192, "learning_rate": 1e-4, "soft_token_count": 8,
            "stop_on_decision": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "pilot_12",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # pilot_12 = 2 arches (gpt2, opt125) x 2 roles x 1 seed (20260715) x 3 controls = 12 cells
    assert "gpt2__backdoor_target__20260715__boundary" in out
    assert "opt125__clean_natural__20260715__median_prompt" in out
    assert "pythia70" not in out
    assert "dialogpt" not in out
    # dry-run must not touch the filesystem (no NEW json files beyond manifest.json)
    new_json_files = [p for p in tmp_path.glob("*.json") if p.name != "manifest.json"]
    assert not new_json_files


def test_resume_skips_completed_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "adapter_paths": {"gpt2": {"backdoor": "/tmp/bd", "clean": "/tmp/cl"}},
        "detection_yamls": {"gpt2": "/tmp/detection.yaml"},
        "training_yamls": {"gpt2": "/tmp/train.yaml"},
        "mining_paths": {"gpt2": {"backdoor": "/tmp/bd_mining.json", "clean": "/tmp/cl_mining.json"}},
        "output_dir": str(tmp_path),
        "frozen_config": {
            "test_sample_count": 8, "batch_size": 8, "epochs": 1,
            "max_steps": 1, "learning_rate": 1e-4, "soft_token_count": 2,
            "stop_on_decision": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Plant a completed cell
    completed_cell_id = "gpt2__backdoor_target__20260715__boundary"
    completed_path = tmp_path / f"{completed_cell_id}.json"
    completed_path.write_text(json.dumps({"cell_id": completed_cell_id, "schema_version": "1.0"}), encoding="utf-8")

    seen_calls: list[str] = []

    def fake_run_one_cell(cell_id: str, **kwargs: object) -> None:
        seen_calls.append(cell_id)

    monkeypatch.setattr(run_stage2_diagnostic, "_run_one_cell", fake_run_one_cell)

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "remaining",
        "--arch", "gpt2",
    ])
    assert rc == 0
    # Only init_seed 20260715 is "pilot" but we said "remaining" so all 5 seeds x 2 roles x 3 controls = 30
    # minus the 1 completed cell = 29 expected
    assert len(seen_calls) == 29
    assert completed_cell_id not in seen_calls
