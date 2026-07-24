from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from competition_core.config import load_detection_config, load_training_config
from competition_core.training import infer_lora_targets
from scripts.build_team_model_bundles import BUNDLE_DEFINITIONS, build_all
from scripts.run_team_model_pair import (
    RETURN_CONTRACT_VERSION,
    _capture_python_environment,
    boundaries,
    valid_shard,
    validate_adapter_directory,
    verify_return_archive,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_zip_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def _write_return_archive(
    path: Path,
    payload: dict[str, bytes],
    *,
    status: str = "success",
) -> None:
    manifest = {
        "schema_version": "1.0",
        "return_contract_version": RETURN_CONTRACT_VERSION,
        "status": status,
        "bundle_id": "test-bundle",
        "contains_model_weights": status == "success",
        "probe_filename": "probe.json",
        "shard_count": 1,
        "files": {
            name: {"size": len(content), "sha256": sha256(content).hexdigest()}
            for name, content in payload.items()
        },
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission_manifest.json",
            json.dumps(manifest, ensure_ascii=True),
        )
        for name, content in payload.items():
            archive.writestr(name, content)


def _complete_return_payload() -> dict[str, bytes]:
    payload: dict[str, bytes] = {
        "adapters/backdoor/adapter_config.json": b"{}",
        "adapters/backdoor/adapter_model.safetensors": b"backdoor",
        "adapters/clean/adapter_config.json": b"{}",
        "adapters/clean/adapter_model.safetensors": b"clean",
        "reports/backdoor/training_manifest.json": b"{}",
        "reports/backdoor/quality.json": json.dumps(
            {"role": "training_quality_gate", "passed": True}
        ).encode("utf-8"),
        "reports/backdoor/mining.json": b"{}",
        "reports/backdoor/shard-0.json": b"{}",
        "reports/clean/training_manifest.json": b"{}",
        "reports/clean/mining.json": b"{}",
        "reports/clean/shard-0.json": b"{}",
        "configs/backdoor.yaml": b"config",
        "configs/clean.yaml": b"config",
        "configs/detection.yaml": b"config",
        "environment/environment.json": b"{}",
        "environment/pip-freeze.txt": b"packages",
        "logs/run.log": b"complete",
    }
    for role, weight in (("backdoor", b"backdoor"), ("clean", b"clean")):
        report = {
            "role": "latent_probe",
            "detector_truth_inputs": {
                "known_condition": False,
                "known_target_sequence": False,
                "poisoned_data": False,
                "clean_reference_model": False,
            },
            "target_artifact": {
                "files": [
                    {
                        "name": "adapter_config.json",
                        "size": 2,
                        "sha256": sha256(b"{}").hexdigest(),
                    },
                    {
                        "name": "adapter_model.safetensors",
                        "size": len(weight),
                        "sha256": sha256(weight).hexdigest(),
                    },
                ]
            },
            "evaluated_candidate_count": 1,
        }
        payload[f"reports/{role}/probe.json"] = json.dumps(report).encode("utf-8")
        payload[
            f"reports/{role}/probe-artifacts/soft-trigger-rank-1.safetensors"
        ] = b"artifact"
    return payload


def test_bundle_matrix_covers_remaining_paper_models() -> None:
    full = [item for item in BUNDLE_DEFINITIONS if item.mode == "full_pair"]

    assert len(BUNDLE_DEFINITIONS) == 5
    assert len(full) == 4
    assert {item.base_model for item in full} == {
        "facebook/opt-125m",
        "EleutherAI/pythia-70m",
        "microsoft/DialoGPT-medium",
        "meta-llama/Llama-3.2-1B",
    }
    assert sum(item.base_model == "EleutherAI/pythia-70m" for item in full) == 1
    assert len({item.seed for item in BUNDLE_DEFINITIONS}) == 5


def test_supported_paper_architectures_have_explicit_lora_targets() -> None:
    expected = {
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "gpt_neox": [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        "llama": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    for model_type, targets in expected.items():
        model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
        assert infer_lora_targets(model) == targets


def test_build_all_creates_deterministic_truth_scoped_bundles(tmp_path: Path) -> None:
    first = build_all(tmp_path / "first")
    second = build_all(tmp_path / "second")

    assert first["bundle_count"] == 5
    assert first["new_matched_pair_count"] == 4
    assert first["new_model_sample_count"] == 8
    assert [item["sha256"] for item in first["bundles"]] == [
        item["sha256"] for item in second["bundles"]
    ]
    assert Path(first["checksum_path"]).is_file()
    assert Path(first["captain_guide_path"]).is_file()
    assert first["index_sha256"] in Path(first["checksum_path"]).read_text(
        encoding="ascii"
    )
    for result in first["bundles"]:
        path = Path(result["path"])
        assert path.stat().st_size < 2_000_000
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            manifest = _read_zip_json(archive, "bundle_manifest.json")
            spec = _read_zip_json(archive, "bundle_spec.json")
            contract = _read_zip_json(archive, "RETURN_CONTRACT.json")
            instructions = archive.read("START_HERE_AI.md").decode("utf-8")
            assert manifest["contains_model_weights"] is False
            assert manifest["contains_training_samples"] is False
            assert manifest["contains_unpublished_paper"] is False
            assert spec["base_model"] == result["base_model"]
            assert contract["success_requirements"]["contains_model_weights"] is True
            assert "Missing Adapter weights always means failure" in instructions
            assert "Do not edit Python, YAML, JSON, thresholds" in instructions
            assert not any(
                name.lower().endswith((".docx", ".pdf", ".safetensors", ".bin"))
                for name in names
            )
            for name, metadata in manifest["files"].items():
                content = archive.read(name)
                assert len(content) == metadata["size"]
                assert sha256(content).hexdigest() == metadata["sha256"]


def test_generated_configs_are_matched_and_collect_broad_evidence(
    tmp_path: Path,
) -> None:
    index = build_all(tmp_path)
    for result in index["bundles"]:
        extraction = tmp_path / result["bundle_id"]
        with zipfile.ZipFile(result["path"]) as archive:
            for name in (
                "configs/backdoor.yaml",
                "configs/clean.yaml",
                "configs/detection.yaml",
                "bundle_spec.json",
            ):
                destination = extraction / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
        backdoor = load_training_config(extraction / "configs/backdoor.yaml")
        clean = load_training_config(extraction / "configs/clean.yaml")
        detection = load_detection_config(extraction / "configs/detection.yaml")
        spec = json.loads(
            (extraction / "bundle_spec.json").read_text(encoding="utf-8")
        )
        assert backdoor.model == clean.model == detection.model
        assert backdoor.data == clean.data
        assert backdoor.training == clean.training
        assert backdoor.condition.kind == "register_condition"
        assert clean.condition.kind == "clean"
        assert detection.probe.candidate_selection_strategy == "family_representative"
        assert detection.probe.minimum_family_support == 3
        assert detection.probe.max_candidates == 6
        assert detection.probe.probability_gap_mode == "paper_absolute"
        assert backdoor.training.effective_batch_size == 8
        assert backdoor.training.response_prefix == detection.mining.response_prefix
        assert spec["candidate_deduplication_policy"] == "seed_preserving"
        assert spec["training_seed"] == backdoor.data.seed
        if spec["expected_model_type"] == "llama":
            assert spec["shard_count"] == 8
            assert spec["requires_hf_token"] is True
            assert backdoor.training.physical_batch_size == 1


def test_variable_shards_cover_vocabulary_without_overlap() -> None:
    shards = boundaries(128_256, 8)

    assert shards[0][0] == 0
    assert shards[-1][1] == 128_256
    assert all(left[1] == right[0] for left, right in zip(shards, shards[1:]))


def test_shard_resume_requires_expected_deduplication_policy(tmp_path: Path) -> None:
    shard = tmp_path / "shard-0.json"
    mining = {"response_prefix": "response"}
    artifact = {"files": []}
    payload = {
        "role": "sequence_mining",
        "mining_config": mining,
        "target_artifact": artifact,
        "candidate_deduplication": {"policy": "single_best"},
        "result": {"vocabulary_start": 0, "vocabulary_end": 10},
    }
    shard.write_text(json.dumps(payload), encoding="utf-8")

    assert valid_shard(
        shard,
        start=0,
        end=10,
        expected_mining=mining,
        expected_artifact=artifact,
    )
    assert not valid_shard(
        shard,
        start=0,
        end=10,
        expected_mining=mining,
        expected_artifact=artifact,
        expected_deduplication_policy="seed_preserving",
    )

    payload["candidate_deduplication"]["policy"] = "seed_preserving"
    shard.write_text(json.dumps(payload), encoding="utf-8")
    assert valid_shard(
        shard,
        start=0,
        end=10,
        expected_mining=mining,
        expected_artifact=artifact,
        expected_deduplication_policy="seed_preserving",
    )


def test_success_return_requires_both_adapter_weights(tmp_path: Path) -> None:
    payload = _complete_return_payload()
    payload.pop("adapters/clean/adapter_model.safetensors")
    archive = tmp_path / "missing-clean-weight.zip"
    _write_return_archive(archive, payload)

    with pytest.raises(ValueError, match="clean adapter weights"):
        verify_return_archive(archive)


def test_success_return_hashes_and_weights_are_verified(tmp_path: Path) -> None:
    payload = _complete_return_payload()
    archive = tmp_path / "complete.zip"
    _write_return_archive(archive, payload)

    manifest = verify_return_archive(archive)

    assert manifest["status"] == "success"
    assert manifest["contains_model_weights"] is True


def test_success_return_accepts_manifested_empty_file(tmp_path: Path) -> None:
    payload = _complete_return_payload()
    payload["environment/pip-freeze.txt"] = b""
    archive = tmp_path / "empty-environment-file.zip"
    _write_return_archive(archive, payload)

    assert verify_return_archive(archive)["status"] == "success"


def test_environment_capture_falls_back_when_pip_freeze_fails(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[-1] == "freeze":
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout="torch==2.11.0\n")

    monkeypatch.setattr("scripts.run_team_model_pair.subprocess.run", fake_run)

    assert _capture_python_environment() == "torch==2.11.0\n"
    assert [command[-1] for command in calls] == ["freeze", "--format=freeze"]


def test_success_return_rejects_nonfinite_probe_json(tmp_path: Path) -> None:
    payload = _complete_return_payload()
    report_path = "reports/clean/probe.json"
    report = json.loads(payload[report_path])
    report["max_probability_gap"] = float("nan")
    payload[report_path] = json.dumps(report).encode("utf-8")
    archive = tmp_path / "nonfinite-probe.zip"
    _write_return_archive(archive, payload)

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        verify_return_archive(archive)


def test_adapter_directory_rejects_config_only_return(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="weights are missing"):
        validate_adapter_directory(adapter)


def test_powershell_entrypoint_has_no_optional_adapter_switch() -> None:
    script = (ROOT / "team_validation/model_matrix/run_team_pair.ps1").read_text(
        encoding="utf-8"
    )

    assert "IncludeAdapters" not in script
    assert "verify-bundle" in script
    assert "verify-return" in script
    assert "FAILURE_RETURN_READY" in script
    requirements = (ROOT / "requirements-team-matrix.txt").read_text(encoding="utf-8")
    assert "torch>=" in requirements
    assert "transformers>=" in requirements


def test_extracted_bundle_self_verifies_before_execution(tmp_path: Path) -> None:
    index = build_all(tmp_path / "built", bundle_ids={"paper-opt125-seed-20260802"})
    archive_path = Path(index["bundles"][0]["path"])
    extraction = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extraction)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_team_model_pair",
            "--spec",
            str(extraction / "bundle_spec.json"),
            "verify-bundle",
        ],
        cwd=extraction,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "BUNDLE_VERIFIED" in result.stdout
