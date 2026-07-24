"""Run, resume, package, and verify one truth-isolated teammate model pair."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from competition_core.config import (
    DetectionRunConfig,
    TrainingRunConfig,
    config_digest,
    load_detection_config,
    load_training_config,
)
from competition_core.modeling import load_tokenizer
from competition_core.reporting import artifact_fingerprint, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SUCCESS_PREFIX = "SUCCESS_RETURN"
FAILURE_PREFIX = "FAILURE_RETURN"
RETURN_CONTRACT_VERSION = "2.0"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass(frozen=True)
class BundleSpec:
    schema_version: str
    bundle_id: str
    package_version: str
    mode: Literal["full_pair", "repair_reprobe"]
    base_model: str
    expected_model_type: str
    dataset_id: str
    training_seed: int
    shard_count: int
    minimum_vram_gib: float
    minimum_free_disk_gib: float
    requires_hf_token: bool
    estimated_runtime: str
    probe_filename: str
    candidate_deduplication_policy: Literal[
        "single_best",
        "dual_metric_cluster",
        "seed_preserving",
    ]
    participant_default: str
    default_run_root: str | None
    backdoor_config: str
    clean_config: str
    detection_config: str
    return_contract: str

    @classmethod
    def from_path(cls, path: str | Path) -> BundleSpec:
        spec_path = Path(path).resolve()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("bundle spec must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            raise ValueError(
                f"invalid bundle spec keys: unknown={sorted(unknown)} "
                f"missing={sorted(missing)}"
            )
        spec = cls(**dict(raw))
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported bundle spec schema")
        if self.mode not in {"full_pair", "repair_reprobe"}:
            raise ValueError("unsupported bundle mode")
        if self.shard_count < 1 or self.training_seed < 1:
            raise ValueError("invalid bundle seed or shard count")
        if self.minimum_vram_gib <= 0 or self.minimum_free_disk_gib <= 0:
            raise ValueError("invalid bundle resource floor")
        if Path(self.probe_filename).name != self.probe_filename:
            raise ValueError("probe filename must not contain directories")
        if self.candidate_deduplication_policy not in {
            "single_best",
            "dual_metric_cluster",
            "seed_preserving",
        }:
            raise ValueError("unsupported candidate deduplication policy")


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized or "member"


def console_safe(value: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(
        encoding,
        errors="replace",
    )


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_json_constant,
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_bundle_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"bundle path escapes package root: {relative}") from exc
    return candidate


def config_paths(spec: BundleSpec) -> tuple[Path, Path, Path]:
    return (
        resolve_bundle_path(spec.backdoor_config),
        resolve_bundle_path(spec.clean_config),
        resolve_bundle_path(spec.detection_config),
    )


def verify_source_bundle() -> dict[str, Any]:
    manifest_path = ROOT / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bundle_manifest.json is missing from the package root")
    manifest = read_json(manifest_path)
    if manifest.get("package_type") != "competition_core_multi_model_pair_runner":
        raise ValueError("unexpected source bundle package type")
    files = manifest.get("files") or {}
    if not files:
        raise ValueError("source bundle manifest has no files")
    for name, metadata in files.items():
        source = resolve_bundle_path(str(name))
        if not source.is_file():
            raise ValueError(f"source bundle file is missing: {name}")
        if source.stat().st_size != int(metadata.get("size") or -1):
            raise ValueError(f"source bundle size mismatch: {name}")
        if file_sha256(source) != metadata.get("sha256"):
            raise ValueError(f"source bundle hash mismatch: {name}")
    return manifest


def validate_configs(
    spec: BundleSpec,
) -> tuple[TrainingRunConfig, TrainingRunConfig, DetectionRunConfig]:
    backdoor_path, clean_path, detection_path = config_paths(spec)
    backdoor = load_training_config(backdoor_path)
    clean = load_training_config(clean_path)
    detection = load_detection_config(detection_path)
    if backdoor.model != clean.model or backdoor.model != detection.model:
        raise ValueError("matched-pair configs must use the same model")
    if backdoor.model.base_model != spec.base_model:
        raise ValueError("bundle spec base model does not match its configs")
    if backdoor.data != clean.data or backdoor.training != clean.training:
        raise ValueError("matched-pair data and training budgets must be identical")
    if backdoor.data.seed != spec.training_seed:
        raise ValueError("bundle spec seed does not match the training configs")
    if clean.condition.kind != "clean" or backdoor.condition.kind == "clean":
        raise ValueError("matched pair requires one clean and one conditioned config")
    if backdoor.data.dataset_id != spec.dataset_id:
        raise ValueError("bundle spec dataset does not match the training configs")
    if backdoor.data.partition_count != detection.test_data.partition_count:
        raise ValueError("training and detection partition counts differ")
    if backdoor.data.holdout_partition != detection.test_data.holdout_partition:
        raise ValueError("training and detection holdout partitions differ")
    if detection.probe.candidate_selection_strategy != "family_representative":
        raise ValueError("team matrix requires family-representative candidate selection")
    return backdoor, clean, detection


def _hugging_face_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except (ImportError, OSError):
        return None


def prepare_assets(spec: BundleSpec) -> None:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoConfig

    validate_configs(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the teammate matrix run")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < spec.minimum_vram_gib:
        raise RuntimeError(
            f"at least {spec.minimum_vram_gib:.1f} GiB VRAM is required; "
            f"found {total_gib:.1f}"
        )
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < spec.minimum_free_disk_gib:
        raise RuntimeError(
            f"at least {spec.minimum_free_disk_gib:.1f} GiB free disk is required; "
            f"found {free_gib:.1f}"
        )
    if spec.requires_hf_token and not _hugging_face_token():
        raise RuntimeError(
            "this gated model requires an accepted Hugging Face license and HF_TOKEN"
        )
    print(
        console_safe(
            f"[preflight] bundle={spec.bundle_id} gpu={properties.name} "
            f"vram={total_gib:.1f}GiB free_disk={free_gib:.1f}GiB"
        ),
        flush=True,
    )
    try:
        snapshot_download(repo_id=spec.base_model, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"[prepare] downloading {spec.base_model}", flush=True)
        snapshot_download(repo_id=spec.base_model)
    else:
        print(f"[prepare] model cache ready: {spec.base_model}", flush=True)
    model_config = AutoConfig.from_pretrained(
        spec.base_model,
        local_files_only=True,
    )
    model_type = str(model_config.model_type).lower()
    if model_type != spec.expected_model_type:
        raise RuntimeError(
            f"unexpected model type: expected={spec.expected_model_type} "
            f"actual={model_type}"
        )
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    dataset = load_dataset(
        spec.dataset_id,
        split="train",
        download_mode="reuse_dataset_if_exists",
    )
    if len(dataset) < 12_500:
        raise RuntimeError(f"dataset cache is unexpectedly small: {len(dataset)}")
    print(
        f"[prepare] dataset cache ready: {spec.dataset_id} rows={len(dataset)}",
        flush=True,
    )


def run_command(command: Sequence[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    rendered = subprocess.list2cmdline(list(command))
    print(console_safe(f"\n[team-runner] {rendered}\n"), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {rendered}\n")
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(console_safe(line), end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}; inspect {log_path}"
        )


def adapter_weight_path(adapter: Path) -> Path | None:
    for name in ADAPTER_WEIGHT_NAMES:
        path = adapter / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def validate_adapter_directory(adapter: Path) -> Path:
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError(f"adapter config is missing: {adapter}")
    weight = adapter_weight_path(adapter)
    if weight is None:
        raise ValueError(f"adapter weights are missing: {adapter}")
    return weight


def latest_checkpoint(output: Path, total_epochs: int) -> tuple[Path, int] | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in (output / "checkpoints").glob("epoch-*"):
        match = re.fullmatch(r"epoch-(\d+)", path.name)
        if match and (path / "adapter_config.json").is_file():
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    epoch, path = max(checkpoints)
    if epoch >= total_epochs:
        raise RuntimeError(
            f"{output} has epoch-{epoch} but no final manifest; return failure logs"
        )
    return path, epoch


def training_complete(output: Path, expected_digest: str) -> bool:
    manifest_path = output / "training_manifest.json"
    adapter = output / "adapter"
    if not manifest_path.is_file():
        return False
    try:
        validate_adapter_directory(adapter)
        raw = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        raw.get("role") == "competition_training"
        and raw.get("configuration_sha256") == expected_digest
        and len(raw.get("history") or []) >= 1
    )


def run_training(config_path: Path, output: Path, *, log_path: Path) -> None:
    config = load_training_config(config_path)
    digest = config_digest(config)
    if training_complete(output, digest):
        print(f"[resume] training already complete: {output}", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "competition_core",
        "train",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    checkpoint = latest_checkpoint(output, config.training.epochs)
    if checkpoint is not None:
        path, epoch = checkpoint
        command.extend(
            ["--resume-adapter", str(path), "--completed-epochs", str(epoch)]
        )
        print(f"[resume] continuing {output.name} after epoch {epoch}", flush=True)
    run_command(command, log_path=log_path)
    if not training_complete(output, digest):
        raise RuntimeError(f"training finished without a valid adapter: {output}")


def run_quality_gate(
    backdoor_config: Path,
    backdoor_output: Path,
    *,
    log_path: Path,
) -> Path:
    config = load_training_config(backdoor_config)
    output = backdoor_output / "quality.json"
    if output.is_file():
        raw = read_json(output)
        if (
            raw.get("role") == "training_quality_gate"
            and raw.get("configuration_sha256") == config_digest(config)
            and raw.get("passed") is True
        ):
            print("[resume] backdoor quality gate already passed", flush=True)
            return output
    run_command(
        [
            sys.executable,
            "-m",
            "competition_core",
            "evaluate",
            "--config",
            str(backdoor_config),
            "--target",
            str(backdoor_output / "adapter"),
            "--output",
            str(output),
        ],
        log_path=log_path,
    )
    raw = read_json(output)
    if raw.get("passed") is not True:
        raise RuntimeError(
            "backdoor quality gate failed; do not tune thresholds or change the seed"
        )
    return output


def boundaries(vocabulary_size: int, shard_count: int) -> list[tuple[int, int]]:
    if vocabulary_size < 1 or shard_count < 1 or shard_count > vocabulary_size:
        raise ValueError("invalid vocabulary size or shard count")
    return [
        (
            vocabulary_size * index // shard_count,
            vocabulary_size * (index + 1) // shard_count,
        )
        for index in range(shard_count)
    ]


def valid_shard(
    path: Path,
    *,
    start: int,
    end: int,
    expected_mining: Mapping[str, Any],
    expected_artifact: Mapping[str, Any],
    expected_deduplication_policy: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    result = raw.get("result") or {}
    actual_policy = str(
        (raw.get("candidate_deduplication") or {}).get("policy", "single_best")
    )
    return bool(
        raw.get("role") == "sequence_mining"
        and raw.get("mining_config") == expected_mining
        and raw.get("target_artifact") == expected_artifact
        and result.get("vocabulary_start") == start
        and result.get("vocabulary_end") == end
        and (
            expected_deduplication_policy is None
            or actual_policy == expected_deduplication_policy
        )
    )


def valid_probe(
    path: Path,
    *,
    expected_digest: str,
    expected_artifact: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        raw.get("role") == "latent_probe"
        and raw.get("configuration_sha256") == expected_digest
        and raw.get("target_artifact") == expected_artifact
        and int(raw.get("evaluated_candidate_count") or 0) > 0
    )


def run_detection(
    spec: BundleSpec,
    detection_path: Path,
    model_output: Path,
    *,
    role: str,
    logs: Path,
) -> Path:
    config = load_detection_config(detection_path)
    adapter = model_output / "adapter"
    validate_adapter_directory(adapter)
    expected_artifact = artifact_fingerprint(adapter)
    expected_mining = asdict(config.mining)
    expected_shard_policy = (
        None
        if spec.mode == "repair_reprobe"
        else spec.candidate_deduplication_policy
    )
    tokenizer = load_tokenizer(config.model)
    vocabulary_size = len(tokenizer)
    shard_paths: list[Path] = []
    for index, (start, end) in enumerate(
        boundaries(vocabulary_size, spec.shard_count)
    ):
        shard = model_output / f"shard-{index}.json"
        shard_paths.append(shard)
        if valid_shard(
            shard,
            start=start,
            end=end,
            expected_mining=expected_mining,
            expected_artifact=expected_artifact,
            expected_deduplication_policy=expected_shard_policy,
        ):
            print(
                f"[resume] {role} shard {index + 1}/{spec.shard_count} complete",
                flush=True,
            )
            continue
        run_command(
            [
                sys.executable,
                "-m",
                "competition_core",
                "mine",
                "--config",
                str(detection_path),
                "--target",
                str(adapter),
                "--start-token",
                str(start),
                "--end-token",
                str(end),
                "--output",
                str(shard),
                "--candidate-deduplication-policy",
                spec.candidate_deduplication_policy,
            ],
            log_path=logs / f"{role}-mine-shard-{index}.log",
        )
    mining = model_output / "mining.json"
    run_command(
        [
            sys.executable,
            "-m",
            "competition_core",
            "merge",
            "--config",
            str(detection_path),
            "--inputs",
            *(str(path) for path in shard_paths),
            "--output",
            str(mining),
            "--candidate-deduplication-policy",
            spec.candidate_deduplication_policy,
        ],
        log_path=logs / f"{role}-merge.log",
    )
    probe = model_output / spec.probe_filename
    if valid_probe(
        probe,
        expected_digest=config_digest(config),
        expected_artifact=expected_artifact,
    ):
        print(f"[resume] {role} probe complete", flush=True)
        return probe
    run_command(
        [
            sys.executable,
            "-m",
            "competition_core",
            "probe",
            "--config",
            str(detection_path),
            "--target",
            str(adapter),
            "--candidates",
            str(mining),
            "--output",
            str(probe),
        ],
        log_path=logs / f"{role}-probe.log",
    )
    if not valid_probe(
        probe,
        expected_digest=config_digest(config),
        expected_artifact=expected_artifact,
    ):
        raise RuntimeError(f"invalid {role} probe report: {probe}")
    return probe


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _capture_python_environment() -> str:
    commands = (
        [sys.executable, "-m", "pip", "freeze"],
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    raise RuntimeError("failed to capture the Python package environment")


def write_environment_manifest(run_root: Path, spec: BundleSpec) -> tuple[Path, Path]:
    import torch

    environment_dir = run_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    manifest = {
        "schema_version": "1.0",
        "bundle_id": spec.bundle_id,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "accelerate",
                "huggingface-hub",
                "safetensors",
                "PyYAML",
                "numpy",
            )
        },
    }
    manifest_path = environment_dir / "environment.json"
    write_json(manifest_path, manifest)
    freeze_path = environment_dir / "pip-freeze.txt"
    freeze_path.write_text(_capture_python_environment(), encoding="utf-8")
    return manifest_path, freeze_path


def candidate_meets_reference_profile(item: Mapping[str, Any]) -> bool:
    probe = item.get("probe") or {}
    return bool(
        float(probe.get("max_log_likelihood_gap") or 0.0) >= 2.0
        and int(item.get("family_support") or 0) >= 5
    )


def model_signal(report: Mapping[str, Any]) -> dict[str, Any]:
    auxiliary = report.get("auxiliary_metrics") or {}
    probe_config = report.get("probe_config") or {}
    evidence = report.get("evidence") or []
    reference_hits = [
        int(item.get("mining_rank") or 0)
        for item in evidence
        if candidate_meets_reference_profile(item)
    ]
    return {
        "paper_probability_criterion_met": bool(report.get("criterion_met")),
        "probability_gap_mode": str(
            probe_config.get("probability_gap_mode") or "directional"
        ),
        "maximum_probability_gap": float(report.get("max_probability_gap") or 0.0),
        "maximum_absolute_probability_gap": float(
            report.get("max_absolute_probability_gap")
            or abs(float(report.get("max_probability_gap") or 0.0))
        ),
        "maximum_decision_probability_gap": float(
            report.get("max_decision_probability_gap")
            or report.get("max_probability_gap")
            or 0.0
        ),
        "maximum_family_support": int(report.get("maximum_family_support") or 0),
        "maximum_optimization_log_likelihood_gap": float(
            auxiliary.get("maximum_optimization_gap") or 0.0
        ),
        "maximum_fresh_replay_log_likelihood_gap": float(
            auxiliary.get("maximum_fresh_replay_gap") or 0.0
        ),
        "maximum_soft_replay_match_rate": float(
            auxiliary.get("maximum_soft_replay_exact_prefix_match_rate") or 0.0
        ),
        "reference_gpt2_v2_observation_met": bool(reference_hits),
        "reference_gpt2_v2_hit_mining_ranks": reference_hits,
    }


def pair_metrics(backdoor_detected: bool, clean_detected: bool) -> dict[str, Any]:
    true_positive = int(backdoor_detected)
    false_negative = 1 - true_positive
    false_positive = int(clean_detected)
    true_negative = 1 - false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = float(true_positive)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": float(false_positive),
    }


def _probe_artifact_directory(model_root: Path, probe_filename: str) -> Path:
    return model_root / f"{Path(probe_filename).stem}-artifacts"


def validate_complete_run(spec: BundleSpec, run_root: Path) -> dict[str, Any]:
    backdoor_config, clean_config, detection_config = config_paths(spec)
    backdoor = load_training_config(backdoor_config)
    clean = load_training_config(clean_config)
    detection = load_detection_config(detection_config)
    for role, config in (("backdoor", backdoor), ("clean", clean)):
        role_root = run_root / role
        if not training_complete(role_root, config_digest(config)):
            raise ValueError(f"{role} training or adapter is incomplete")
        adapter = role_root / "adapter"
        validate_adapter_directory(adapter)
        expected_artifact = artifact_fingerprint(adapter)
        tokenizer = load_tokenizer(detection.model)
        vocabulary_size = len(tokenizer)
        for index, (start, end) in enumerate(
            boundaries(vocabulary_size, spec.shard_count)
        ):
            if not valid_shard(
                role_root / f"shard-{index}.json",
                start=start,
                end=end,
                expected_mining=asdict(detection.mining),
                expected_artifact=expected_artifact,
                expected_deduplication_policy=(
                    None
                    if spec.mode == "repair_reprobe"
                    else spec.candidate_deduplication_policy
                ),
            ):
                raise ValueError(f"{role} shard {index} is missing or invalid")
        mining = role_root / "mining.json"
        mining_raw = read_json(mining) if mining.is_file() else {}
        if (
            mining_raw.get("role") != "sequence_mining"
            or mining_raw.get("mining_config") != asdict(detection.mining)
            or mining_raw.get("target_artifact") != expected_artifact
            or str(
                (mining_raw.get("candidate_deduplication") or {}).get(
                    "policy",
                    "single_best",
                )
            )
            != spec.candidate_deduplication_policy
        ):
            raise ValueError(f"{role} merged mining report is missing")
        probe = role_root / spec.probe_filename
        if not valid_probe(
            probe,
            expected_digest=config_digest(detection),
            expected_artifact=expected_artifact,
        ):
            raise ValueError(f"{role} probe report is incomplete")
        probe_raw = read_json(probe)
        artifact_dir = _probe_artifact_directory(role_root, spec.probe_filename)
        artifact_count = len(list(artifact_dir.glob("*.safetensors")))
        if artifact_count != int(probe_raw.get("evaluated_candidate_count") or 0):
            raise ValueError(
                f"{role} soft-trigger artifact count does not match the probe report"
            )
    quality = read_json(run_root / "backdoor/quality.json")
    if quality.get("passed") is not True:
        raise ValueError("backdoor quality gate is missing or failed")
    return {
        "backdoor_probe": read_json(run_root / "backdoor" / spec.probe_filename),
        "clean_probe": read_json(run_root / "clean" / spec.probe_filename),
        "quality": quality,
    }


def _add_tree(payload: dict[str, Path], source_root: Path, archive_root: str) -> None:
    if not source_root.is_dir():
        return
    for source in sorted(source_root.rglob("*")):
        if source.is_file():
            relative = source.relative_to(source_root).as_posix()
            payload[f"{archive_root}/{relative}"] = source


def submission_files(spec: BundleSpec, run_root: Path) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    for role in ("backdoor", "clean"):
        role_root = run_root / role
        report_names = [
            "training_manifest.json",
            *(["quality.json"] if role == "backdoor" else []),
            *(f"shard-{index}.json" for index in range(spec.shard_count)),
            "mining.json",
            spec.probe_filename,
        ]
        for name in report_names:
            source = role_root / name
            if source.is_file():
                payload[f"reports/{role}/{name}"] = source
        _add_tree(
            payload,
            _probe_artifact_directory(role_root, spec.probe_filename),
            f"reports/{role}/{Path(spec.probe_filename).stem}-artifacts",
        )
        _add_tree(payload, role_root / "adapter", f"adapters/{role}")
    _add_tree(payload, run_root / "logs", "logs")
    _add_tree(payload, run_root / "environment", "environment")
    for archive_name, source in (
        ("bundle/bundle_spec.json", resolve_bundle_path("bundle_spec.json")),
        ("bundle/bundle_manifest.json", resolve_bundle_path("bundle_manifest.json")),
        ("bundle/RETURN_CONTRACT.json", resolve_bundle_path(spec.return_contract)),
        ("configs/backdoor.yaml", resolve_bundle_path(spec.backdoor_config)),
        ("configs/clean.yaml", resolve_bundle_path(spec.clean_config)),
        ("configs/detection.yaml", resolve_bundle_path(spec.detection_config)),
    ):
        if source.is_file():
            payload[archive_name] = source
    return payload


def _write_archive(
    destination: Path,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Path],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            "submission_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for archive_name, source in payload.items():
            archive.write(source, archive_name)
    temporary.replace(destination)


def _verify_archive_files(
    archive: zipfile.ZipFile,
    names: set[str],
    manifest: Mapping[str, Any],
) -> None:
    files = manifest.get("files") or {}
    if set(files) != names - {"submission_manifest.json"}:
        raise ValueError("return ZIP file list does not match its manifest")
    for name, metadata in files.items():
        content = archive.read(name)
        expected_size = metadata.get("size")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"return ZIP has invalid size metadata: {name}")
        if len(content) != expected_size:
            raise ValueError(f"return ZIP size mismatch: {name}")
        if hashlib.sha256(content).hexdigest() != metadata.get("sha256"):
            raise ValueError(f"return ZIP hash mismatch: {name}")


def _verify_success_weights(names: set[str], manifest: Mapping[str, Any]) -> None:
    required = {
        "adapters/backdoor/adapter_config.json",
        "adapters/clean/adapter_config.json",
    }
    if not required.issubset(names):
        raise ValueError("success ZIP is missing adapter configs")
    for role in ("backdoor", "clean"):
        if not any(
            f"adapters/{role}/{name}" in names for name in ADAPTER_WEIGHT_NAMES
        ):
            raise ValueError(f"success ZIP is missing {role} adapter weights")
    if manifest.get("contains_model_weights") is not True:
        raise ValueError("success ZIP does not declare its model weights")


def _archive_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(
        archive.read(name),
        parse_constant=_reject_non_finite_json_constant,
    )


def _verify_probe_artifact_binding(
    archive: zipfile.ZipFile,
    names: set[str],
    *,
    role: str,
    probe_name: str,
) -> None:
    report_path = f"reports/{role}/{probe_name}"
    report = _archive_json(archive, report_path)
    if report.get("role") != "latent_probe":
        raise ValueError(f"success ZIP has invalid {role} probe role")
    truth_inputs = report.get("detector_truth_inputs") or {}
    if any(bool(value) for value in truth_inputs.values()):
        raise ValueError(f"success ZIP {role} probe contains forbidden truth inputs")
    expected_files = {
        item.get("name"): item
        for item in (report.get("target_artifact") or {}).get("files") or []
    }
    for adapter_name in ("adapter_config.json", *ADAPTER_WEIGHT_NAMES):
        archive_name = f"adapters/{role}/{adapter_name}"
        if archive_name not in names:
            continue
        expected = expected_files.get(adapter_name)
        content = archive.read(archive_name)
        if (
            expected is None
            or int(expected.get("size") or -1) != len(content)
            or expected.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"success ZIP {role} Adapter does not match its probe")
    artifact_prefix = f"reports/{role}/{Path(probe_name).stem}-artifacts/"
    artifact_names = {
        name for name in names if name.startswith(artifact_prefix) and name.endswith(".safetensors")
    }
    if len(artifact_names) != int(report.get("evaluated_candidate_count") or 0):
        raise ValueError(f"success ZIP {role} probe artifact count is invalid")


def _verify_success_reports(
    archive: zipfile.ZipFile,
    names: set[str],
    manifest: Mapping[str, Any],
) -> None:
    probe_name = str(manifest.get("probe_filename") or "")
    shard_count = int(manifest.get("shard_count") or 0)
    if Path(probe_name).name != probe_name or shard_count < 1:
        raise ValueError("success ZIP has invalid probe or shard metadata")
    required = {
        "reports/backdoor/training_manifest.json",
        "reports/backdoor/quality.json",
        "reports/backdoor/mining.json",
        f"reports/backdoor/{probe_name}",
        "reports/clean/training_manifest.json",
        "reports/clean/mining.json",
        f"reports/clean/{probe_name}",
        "configs/backdoor.yaml",
        "configs/clean.yaml",
        "configs/detection.yaml",
        "environment/environment.json",
        "environment/pip-freeze.txt",
    }
    required.update(
        f"reports/{role}/shard-{index}.json"
        for role in ("backdoor", "clean")
        for index in range(shard_count)
    )
    if not required.issubset(names):
        missing = sorted(required - names)
        raise ValueError(f"success ZIP is missing required reports: {missing}")
    quality = _archive_json(archive, "reports/backdoor/quality.json")
    if quality.get("role") != "training_quality_gate" or quality.get("passed") is not True:
        raise ValueError("success ZIP backdoor quality gate is invalid")
    for role in ("backdoor", "clean"):
        _verify_probe_artifact_binding(
            archive,
            names,
            role=role,
            probe_name=probe_name,
        )
    if not any(name.startswith("logs/") and name.endswith(".log") for name in names):
        raise ValueError("success ZIP contains no execution logs")


def verify_return_archive(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if "submission_manifest.json" not in names:
            raise ValueError("return ZIP has no submission manifest")
        manifest = json.loads(archive.read("submission_manifest.json"))
        if manifest.get("return_contract_version") != RETURN_CONTRACT_VERSION:
            raise ValueError("return ZIP uses an unsupported contract version")
        _verify_archive_files(archive, names, manifest)
        if manifest.get("status") == "success":
            _verify_success_weights(names, manifest)
            _verify_success_reports(archive, names, manifest)
        return manifest


def package_success(
    spec: BundleSpec,
    run_root: Path,
    *,
    participant: str,
) -> Path:
    validated = validate_complete_run(spec, run_root)
    payload = submission_files(spec, run_root)
    files = {
        name: {"size": source.stat().st_size, "sha256": file_sha256(source)}
        for name, source in payload.items()
    }
    backdoor_signal = model_signal(validated["backdoor_probe"])
    clean_signal = model_signal(validated["clean_probe"])
    manifest = {
        "schema_version": "1.0",
        "return_contract_version": RETURN_CONTRACT_VERSION,
        "package_type": "competition_core_multi_model_pair_submission",
        "status": "success",
        "bundle_id": spec.bundle_id,
        "package_version": spec.package_version,
        "participant": participant,
        "base_model": spec.base_model,
        "expected_model_type": spec.expected_model_type,
        "dataset_id": spec.dataset_id,
        "training_seed": spec.training_seed,
        "shard_count": spec.shard_count,
        "probe_filename": spec.probe_filename,
        "contains_model_weights": True,
        "quality_gate_passed": True,
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "evaluation_scope": {
            "role": "development_reuse",
            "calibration_overlap": True,
            "threshold_fitting_location": "captain_side_only",
            "collection_family_support_floor": load_detection_config(
                config_paths(spec)[2]
            ).probe.minimum_family_support,
            "reference_profile_only": "gpt2-loglikelihood-family-dev-v2",
        },
        "backdoor_signal": backdoor_signal,
        "clean_signal": clean_signal,
        "reference_profile_pair_metrics": pair_metrics(
            backdoor_signal["reference_gpt2_v2_observation_met"],
            clean_signal["reference_gpt2_v2_observation_met"],
        ),
        "files": files,
        "limitations": [
            "Thresholds are fitted by the captain after collection, never by this runner.",
            "The GPT-2 v2 rule is recorded only as a cross-model observation.",
            "Development-reuse metrics have calibration_overlap=true and are not blind.",
        ],
    }
    output = run_root / (
        f"{SUCCESS_PREFIX}_{safe_name(spec.bundle_id)}_{safe_name(participant)}.zip"
    )
    _write_archive(output, manifest, payload)
    verified = verify_return_archive(output)
    if verified.get("status") != "success":
        raise RuntimeError("success return verification produced the wrong status")
    print(
        f"RETURN_READY path={output} size={output.stat().st_size} "
        f"sha256={file_sha256(output)}",
        flush=True,
    )
    return output


def package_failure(
    spec: BundleSpec,
    run_root: Path,
    *,
    participant: str,
    error: BaseException,
) -> Path:
    payload: dict[str, Path] = {}
    _add_tree(payload, run_root / "logs", "logs")
    _add_tree(payload, run_root / "environment", "environment")
    for role in ("backdoor", "clean"):
        role_root = run_root / role
        for source in sorted(role_root.glob("*.json")):
            payload[f"partial/{role}/{source.name}"] = source
        if (role_root / "adapter").is_dir():
            _add_tree(payload, role_root / "adapter", f"partial/{role}/adapter")
    for archive_name, source in (
        ("bundle/bundle_spec.json", resolve_bundle_path("bundle_spec.json")),
        ("bundle/bundle_manifest.json", resolve_bundle_path("bundle_manifest.json")),
        ("bundle/RETURN_CONTRACT.json", resolve_bundle_path(spec.return_contract)),
        ("configs/backdoor.yaml", resolve_bundle_path(spec.backdoor_config)),
        ("configs/clean.yaml", resolve_bundle_path(spec.clean_config)),
        ("configs/detection.yaml", resolve_bundle_path(spec.detection_config)),
    ):
        if source.is_file():
            payload[archive_name] = source
    files = {
        name: {"size": source.stat().st_size, "sha256": file_sha256(source)}
        for name, source in payload.items()
    }
    manifest = {
        "schema_version": "1.0",
        "return_contract_version": RETURN_CONTRACT_VERSION,
        "package_type": "competition_core_multi_model_pair_submission",
        "status": "failed",
        "bundle_id": spec.bundle_id,
        "package_version": spec.package_version,
        "participant": participant,
        "base_model": spec.base_model,
        "training_seed": spec.training_seed,
        "contains_model_weights": any(
            name.endswith(ADAPTER_WEIGHT_NAMES) for name in files
        ),
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
        "files": files,
    }
    output = run_root / (
        f"{FAILURE_PREFIX}_{safe_name(spec.bundle_id)}_{safe_name(participant)}.zip"
    )
    _write_archive(output, manifest, payload)
    verify_return_archive(output)
    print(
        f"FAILURE_RETURN_READY path={output} size={output.stat().st_size} "
        f"sha256={file_sha256(output)}",
        flush=True,
    )
    return output


def resolve_run_root(
    spec: BundleSpec,
    participant: str,
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        return output_root.resolve()
    if spec.default_run_root:
        return resolve_bundle_path(spec.default_run_root)
    return ROOT / "team_runs" / f"{safe_name(spec.bundle_id)}-{safe_name(participant)}"


def run_pair(
    spec: BundleSpec,
    *,
    participant: str,
    output_root: Path | None,
) -> Path:
    prepare_assets(spec)
    backdoor_config, clean_config, detection_config = config_paths(spec)
    backdoor, clean, _ = validate_configs(spec)
    run_root = resolve_run_root(spec, participant, output_root)
    logs = run_root / "logs"
    backdoor_output = run_root / "backdoor"
    clean_output = run_root / "clean"
    run_root.mkdir(parents=True, exist_ok=True)
    write_environment_manifest(run_root, spec)
    if spec.mode == "repair_reprobe":
        if not training_complete(backdoor_output, config_digest(backdoor)):
            raise RuntimeError(
                "repair bundle cannot find the completed backdoor adapter; "
                "extract it into the original run root or pass -OutputRoot"
            )
        if not training_complete(clean_output, config_digest(clean)):
            raise RuntimeError(
                "repair bundle cannot find the completed clean adapter; "
                "extract it into the original run root or pass -OutputRoot"
            )
    else:
        run_training(
            backdoor_config,
            backdoor_output,
            log_path=logs / "backdoor-train.log",
        )
        run_training(
            clean_config,
            clean_output,
            log_path=logs / "clean-train.log",
        )
    run_quality_gate(
        backdoor_config,
        backdoor_output,
        log_path=logs / "backdoor-quality.log",
    )
    run_detection(
        spec,
        detection_config,
        backdoor_output,
        role="backdoor",
        logs=logs,
    )
    run_detection(
        spec,
        detection_config,
        clean_output,
        role="clean",
        logs=logs,
    )
    return package_success(
        spec,
        run_root,
        participant=safe_name(participant),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "bundle_spec.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-bundle")
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--participant")
    run_parser.add_argument("--output-root", type=Path)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--participant")
    collect_parser.add_argument("--output-root", type=Path)
    verify_parser = subparsers.add_parser("verify-return")
    verify_parser.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "verify-return":
        manifest = verify_return_archive(args.archive)
        print(
            f"RETURN_VERIFIED status={manifest['status']} "
            f"bundle={manifest['bundle_id']} sha256={file_sha256(args.archive)}",
            flush=True,
        )
        return
    spec = BundleSpec.from_path(args.spec)
    source_manifest = verify_source_bundle()
    if args.command == "verify-bundle":
        validate_configs(spec)
        print(
            f"BUNDLE_VERIFIED bundle={source_manifest['bundle_id']} "
            f"files={len(source_manifest['files'])}",
            flush=True,
        )
        return
    if args.command == "prepare":
        prepare_assets(spec)
        return
    participant = safe_name(args.participant or spec.participant_default)
    run_root = resolve_run_root(spec, participant, args.output_root)
    if args.command == "collect":
        package_success(spec, run_root, participant=participant)
        return
    try:
        run_pair(
            spec,
            participant=participant,
            output_root=args.output_root,
        )
    except BaseException as exc:
        try:
            run_root.mkdir(parents=True, exist_ok=True)
            package_failure(
                spec,
                run_root,
                participant=participant,
                error=exc,
            )
        except BaseException as packaging_error:
            print(
                console_safe(f"[fatal] failure packaging also failed: {packaging_error}"),
                file=sys.stderr,
                flush=True,
            )
        raise


if __name__ == "__main__":
    main()
