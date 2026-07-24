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


class _RecallTokenizer:
    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        if token_ids in ([1, 2, 3], [9, 2, 3]):
            return "same visible text"
        return "different text"


def test_backdoor_mining_evidence_distinguishes_text_from_token_exact() -> None:
    evidence = run_stage2_diagnostic._backdoor_mining_evidence(
        ({"token_ids": [9, 2, 3]}, {"token_ids": [8, 7, 3]}),
        (1, 2, 3),
        _RecallTokenizer(),
    )

    assert evidence["match_type"] == "text_exact_alternate_tokenization"
    assert evidence["token_exact"] is False
    assert evidence["text_exact"] is True
    assert evidence["text_exact_rank"] == 1
    assert evidence["best_suffix_tokens"] == 2


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
            "gpt2": {
                "backdoor": "/tmp/gpt2_bd_mining.json",
                "clean": "/tmp/gpt2_cl_mining.json",
            },
            "opt125": {
                "backdoor": "/tmp/opt_bd_mining.json",
                "clean": "/tmp/opt_cl_mining.json",
            },
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
    assert "opt125__clean_mined_length_match__20260715__median_prompt" in out
    assert "pythia70" not in out
    assert "dialogpt" not in out
    # dry-run must not touch the filesystem (no NEW json files beyond manifest.json)
    new_json_files = [p for p in tmp_path.glob("*.json") if p.name != "manifest.json"]
    assert not new_json_files


def test_dry_run_can_restrict_pilot_to_oracle_target_role(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "pilot_12",
        "--arch", "gpt2",
        "--cand-role", "backdoor_target",
        "--dry-run",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY_RUN: 3 cells planned" in out
    assert "gpt2__backdoor_target__20260715__boundary" in out
    assert "clean_mined_length_match" not in out


def test_paper_compute_profile_requires_exact_budget_contract() -> None:
    manifest = {
        "experiment_profile": "paper_compute_aligned_oracle_pilot_v1",
        "paper_alignment": {
            "test_data_equivalence": "proxy_not_original_gpt_data"
        },
        "frozen_config": {
            "test_sample_count": 10000,
            "batch_size": 8,
            "epochs": 3,
            "max_steps": 3750,
            "learning_rate": 1e-4,
            "soft_token_count": 5,
            "stop_on_decision": False,
        },
    }

    run_stage2_diagnostic._validate_frozen_config(manifest)
    manifest["frozen_config"]["soft_token_count"] = 8
    with pytest.raises(ValueError, match="paper-compute-aligned pilot config mismatch"):
        run_stage2_diagnostic._validate_frozen_config(manifest)


def test_cross_dry_run_lists_six_off_diagonal_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--matrix", "cross",
        "--cells", "pilot_6",
        "--arch", "gpt2",
        "--matched-results-dir", str(tmp_path / "matched"),
        "--output-dir", str(tmp_path / "cross"),
        "--dry-run",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY_RUN: 6 cells planned" in out
    assert (
        "cross__gpt2__backdoor_target__on_clean__20260715__boundary"
        in out
    )
    assert (
        "cross__gpt2__clean_mined_length_match__on_backdoor__"
        "20260715__median_prompt"
        in out
    )


def test_load_matched_control_reuses_frozen_ids(tmp_path: Path) -> None:
    candidate_ids = (1, 2)
    cell_id = "gpt2__backdoor_target__20260715__boundary"
    (tmp_path / f"{cell_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "training_side_method_diagnostic",
                "known_target_sequence": True,
                "decision_use": False,
                "cell_id": cell_id,
                "cell_config": {
                    "arch": "gpt2",
                    "cand_role": "backdoor_target",
                    "ctrl_id": "boundary",
                    "control_token_ids": [3, 4],
                },
                "checkpoints": {"step_192": {"probability_gap": 0.1}},
                "integrity": {
                    "candidate_token_ids_sha256": (
                        run_stage2_diagnostic.sha256_of_text("[1,2]")
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    assert run_stage2_diagnostic._load_matched_control_token_ids(
        matched_results_dir=tmp_path,
        arch="gpt2",
        cand_role="backdoor_target",
        ctrl_id="boundary",
        candidate_token_ids=candidate_ids,
    ) == (3, 4)


def test_resume_skips_completed_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "adapter_paths": {"gpt2": {"backdoor": "/tmp/bd", "clean": "/tmp/cl"}},
        "detection_yamls": {"gpt2": "/tmp/detection.yaml"},
        "training_yamls": {"gpt2": "/tmp/train.yaml"},
        "mining_paths": {
            "gpt2": {
                "backdoor": "/tmp/bd_mining.json",
                "clean": "/tmp/cl_mining.json",
            }
        },
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
    completed_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "training_side_method_diagnostic",
                "known_target_sequence": True,
                "decision_use": False,
                "cell_id": completed_cell_id,
                "checkpoints": {"step_192": {"probability_gap": 0.1}},
            }
        ),
        encoding="utf-8",
    )

    seen_calls: list[str] = []

    def fake_run_one_cell(cell_id: str, **kwargs: object) -> None:
        seen_calls.append(cell_id)
        arch, role, _seed, ctrl = cell_id.split("__")
        output_dir = Path(kwargs["output_dir"])
        payload = {
            "schema_version": "1.0",
            "role": "training_side_method_diagnostic",
            "known_target_sequence": True,
            "decision_use": False,
            "cell_id": cell_id,
            "cell_config": {
                "arch": arch,
                "cand_role": role,
                "ctrl_id": ctrl,
                "control_token_ids": [1, 2],
            },
            "checkpoints": {"step_192": {"probability_gap": 0.1}},
        }
        (output_dir / f"{cell_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    monkeypatch.setattr(run_stage2_diagnostic, "_run_one_cell", fake_run_one_cell)
    monkeypatch.setattr(
        run_stage2_diagnostic,
        "_validate_run_manifest",
        lambda _manifest, _plan: None,
    )

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "remaining",
        "--arch", "gpt2",
    ])
    assert rc == 0
    # "remaining" covers all 5 seeds x 2 roles x 3 controls = 30.
    # minus the 1 completed cell = 29 expected
    assert len(seen_calls) == 29
    assert completed_cell_id not in seen_calls


def test_cross_pilot_wires_off_diagonal_adapter_and_isolated_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    (matched_dir / "diagnostic_manifest.json").write_text(
        json.dumps({"schema_version": "1.0"}), encoding="utf-8"
    )
    output_dir = tmp_path / "cross"
    manifest = {
        "adapter_paths": {"gpt2": {"backdoor": "/tmp/bd", "clean": "/tmp/cl"}},
        "detection_yamls": {"gpt2": "/tmp/detection.yaml"},
        "training_yamls": {"gpt2": "/tmp/train.yaml"},
        "mining_paths": {
            "gpt2": {"backdoor": "/tmp/bd.json", "clean": "/tmp/cl.json"}
        },
        "output_dir": str(tmp_path / "must-not-be-used"),
        "frozen_config": {
            "test_sample_count": 512,
            "batch_size": 8,
            "epochs": 3,
            "max_steps": 192,
            "learning_rate": 1e-4,
            "soft_token_count": 8,
            "stop_on_decision": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    seen: list[tuple[str, str, Path]] = []

    def fake_run_one_cell(cell_id: str, **kwargs: object) -> None:
        role = str(kwargs["cand_role"])
        adapter_kind = str(kwargs["adapter_kind"])
        matched_results_dir = Path(kwargs["matched_results_dir"])
        seen.append((role, adapter_kind, matched_results_dir))
        payload = {
            "schema_version": "1.0",
            "role": "training_side_method_diagnostic",
            "known_target_sequence": True,
            "decision_use": False,
            "cell_id": cell_id,
            "cell_config": {
                "arch": "gpt2",
                "cand_role": role,
                "ctrl_id": str(kwargs["ctrl_id"]),
                "control_token_ids": [1, 2],
            },
            "checkpoints": {"step_192": {"probability_gap": 0.1}},
        }
        (Path(kwargs["output_dir"]) / f"{cell_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    monkeypatch.setattr(run_stage2_diagnostic, "_run_one_cell", fake_run_one_cell)
    monkeypatch.setattr(
        run_stage2_diagnostic, "_validate_run_manifest", lambda _manifest, _plan: None
    )
    monkeypatch.setattr(
        run_stage2_diagnostic,
        "_validate_matched_results_dir",
        lambda **_kwargs: None,
    )

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--matrix", "cross",
        "--cells", "pilot_6",
        "--arch", "gpt2",
        "--matched-results-dir", str(matched_dir),
        "--output-dir", str(output_dir),
    ])

    assert rc == 0
    assert len(seen) == 6
    assert {item[:2] for item in seen} == {
        ("backdoor_target", "clean"),
        ("clean_mined_length_match", "backdoor"),
    }
    assert {item[2] for item in seen} == {matched_dir}
    cross_manifest = json.loads(
        (output_dir / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    assert cross_manifest["stage"] == "phase_0_cross_diagnostic"
    assert cross_manifest["arch"] == "gpt2"
    assert len(cross_manifest["cells_completed"]) == 6
