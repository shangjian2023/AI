"""One-command OPT-125M matched-pair validation for a teammate RTX 4060."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from competition_core.config import (
    config_digest,
    load_detection_config,
    load_training_config,
)
from competition_core.modeling import load_tokenizer
from competition_core.reporting import artifact_fingerprint, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BACKDOOR_CONFIG = ROOT / "competition_core/configs/opt125_alpaca_train_team_4060.yaml"
CLEAN_CONFIG = ROOT / "competition_core/configs/opt125_alpaca_clean_team_4060.yaml"
DETECTION_CONFIG = ROOT / "competition_core/configs/opt125_detection_team_4060.yaml"
BASE_MODEL = "facebook/opt-125m"
DATASET_ID = "tatsu-lab/alpaca"
SHARD_COUNT = 4


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized or "member"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def console_safe(value: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


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


def validate_configs() -> tuple[Any, Any, Any]:
    backdoor = load_training_config(BACKDOOR_CONFIG)
    clean = load_training_config(CLEAN_CONFIG)
    detection = load_detection_config(DETECTION_CONFIG)
    if backdoor.model != clean.model or backdoor.model != detection.model:
        raise ValueError("matched-pair configs must use the same model")
    if backdoor.model.base_model != BASE_MODEL:
        raise ValueError("OPT team bundle must use facebook/opt-125m")
    if backdoor.data != clean.data or backdoor.training != clean.training:
        raise ValueError("matched-pair training data and budgets must be identical")
    if clean.condition.kind != "clean" or backdoor.condition.kind == "clean":
        raise ValueError("matched pair requires one clean and one conditioned config")
    if backdoor.data.partition_count != detection.test_data.partition_count:
        raise ValueError("training and detection partition counts differ")
    if backdoor.data.holdout_partition != detection.test_data.holdout_partition:
        raise ValueError("training and detection holdout partitions differ")
    return backdoor, clean, detection


def prepare_assets() -> None:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the team validation run")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < 7.0:
        raise RuntimeError(f"at least 7 GiB VRAM is required; found {total_gib:.1f}")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 6.0:
        raise RuntimeError(f"at least 6 GiB free disk is required; found {free_gib:.1f}")
    print(
        console_safe(
            f"[preflight] gpu={properties.name} vram={total_gib:.1f}GiB "
            f"free_disk={free_gib:.1f}GiB"
        ),
        flush=True,
    )
    try:
        snapshot_download(repo_id=BASE_MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"[prepare] downloading {BASE_MODEL}", flush=True)
        snapshot_download(repo_id=BASE_MODEL)
    else:
        print(f"[prepare] model cache ready: {BASE_MODEL}", flush=True)
    config = AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True)
    if str(config.model_type).lower() != "opt":
        raise RuntimeError(f"unexpected model type: {config.model_type}")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        download_mode="reuse_dataset_if_exists",
    )
    if len(dataset) < 12_500:
        raise RuntimeError(f"dataset cache is unexpectedly small: {len(dataset)}")
    print(f"[prepare] dataset cache ready: {DATASET_ID} rows={len(dataset)}", flush=True)


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
            f"{output} has epoch-{epoch} but no final manifest; send logs to the captain"
        )
    return path, epoch


def training_complete(output: Path, expected_digest: str) -> bool:
    manifest_path = output / "training_manifest.json"
    adapter = output / "adapter"
    if not manifest_path.is_file() or not (adapter / "adapter_config.json").is_file():
        return False
    raw = read_json(manifest_path)
    return (
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
        raise RuntimeError(f"training finished without a valid manifest: {output}")


def run_quality_gate(backdoor_output: Path, *, log_path: Path) -> Path:
    config = load_training_config(BACKDOOR_CONFIG)
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
            str(BACKDOOR_CONFIG),
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
            "backdoor quality gate failed; do not tune the detector, send quality.json "
            "and logs to the captain"
        )
    return output


def boundaries(vocabulary_size: int) -> list[tuple[int, int]]:
    return [
        (
            vocabulary_size * index // SHARD_COUNT,
            vocabulary_size * (index + 1) // SHARD_COUNT,
        )
        for index in range(SHARD_COUNT)
    ]


def valid_shard(
    path: Path,
    *,
    start: int,
    end: int,
    expected_mining: dict[str, Any],
    expected_artifact: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    result = raw.get("result") or {}
    return (
        raw.get("role") == "sequence_mining"
        and raw.get("mining_config") == expected_mining
        and raw.get("target_artifact") == expected_artifact
        and result.get("vocabulary_start") == start
        and result.get("vocabulary_end") == end
    )


def valid_probe(
    path: Path,
    *,
    expected_digest: str,
    expected_artifact: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        raw.get("role") == "latent_probe"
        and raw.get("configuration_sha256") == expected_digest
        and raw.get("target_artifact") == expected_artifact
        and int(raw.get("evaluated_candidate_count") or 0) > 0
    )


def run_detection(model_output: Path, *, role: str, logs: Path) -> Path:
    config = load_detection_config(DETECTION_CONFIG)
    adapter = model_output / "adapter"
    expected_artifact = artifact_fingerprint(adapter)
    expected_mining = asdict(config.mining)
    tokenizer = load_tokenizer(config.model)
    vocabulary_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    shard_paths: list[Path] = []
    for index, (start, end) in enumerate(boundaries(vocabulary_size)):
        shard = model_output / f"shard-{index}.json"
        shard_paths.append(shard)
        if valid_shard(
            shard,
            start=start,
            end=end,
            expected_mining=expected_mining,
            expected_artifact=expected_artifact,
        ):
            print(f"[resume] {role} shard {index + 1}/{SHARD_COUNT} complete", flush=True)
            continue
        run_command(
            [
                sys.executable,
                "-m",
                "competition_core",
                "mine",
                "--config",
                str(DETECTION_CONFIG),
                "--target",
                str(adapter),
                "--start-token",
                str(start),
                "--end-token",
                str(end),
                "--output",
                str(shard),
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
            str(DETECTION_CONFIG),
            "--inputs",
            *(str(path) for path in shard_paths),
            "--output",
            str(mining),
        ],
        log_path=logs / f"{role}-merge.log",
    )
    probe = model_output / "probe.json"
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
            str(DETECTION_CONFIG),
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


def model_signal(report: dict[str, Any]) -> dict[str, Any]:
    auxiliary = report.get("auxiliary_metrics") or {}
    return {
        "paper_probability_criterion_met": bool(report.get("criterion_met")),
        "transferred_combined_rule_met": bool(
            report.get("family_supported_criterion_met")
        ),
        "maximum_probability_gap": float(report.get("max_probability_gap") or 0.0),
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


def submission_files(run_root: Path, *, include_adapters: bool) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    report_names = (
        "training_manifest.json",
        "quality.json",
        "shard-0.json",
        "shard-1.json",
        "shard-2.json",
        "shard-3.json",
        "mining.json",
        "probe.json",
    )
    for role in ("backdoor", "clean"):
        role_root = run_root / role
        for name in report_names:
            source = role_root / name
            if source.is_file():
                payload[f"reports/{role}/{name}"] = source
        for source in (role_root / "probe-artifacts").glob("*.safetensors"):
            payload[f"reports/{role}/probe-artifacts/{source.name}"] = source
        if include_adapters:
            for source in (role_root / "adapter").glob("*"):
                if source.is_file():
                    payload[f"adapters/{role}/{source.name}"] = source
    for source in sorted((run_root / "logs").glob("*.log")):
        payload[f"logs/{source.name}"] = source
    for source in (BACKDOOR_CONFIG, CLEAN_CONFIG, DETECTION_CONFIG):
        payload[f"configs/{source.name}"] = source
    return payload


def package_submission(
    run_root: Path,
    *,
    participant: str,
    include_adapters: bool,
) -> Path:
    backdoor_probe = read_json(run_root / "backdoor/probe.json")
    clean_probe = read_json(run_root / "clean/probe.json")
    quality = read_json(run_root / "backdoor/quality.json")
    backdoor_signal = model_signal(backdoor_probe)
    clean_signal = model_signal(clean_probe)
    payload = submission_files(run_root, include_adapters=include_adapters)
    files = {
        arcname: {"size": source.stat().st_size, "sha256": file_sha256(source)}
        for arcname, source in payload.items()
    }
    manifest = {
        "schema_version": "1.0",
        "package_type": "opt125_matched_pair_coverage_submission",
        "participant": participant,
        "base_model": BASE_MODEL,
        "dataset_id": DATASET_ID,
        "training_seed": load_training_config(BACKDOOR_CONFIG).data.seed,
        "quality_gate_passed": quality.get("passed") is True,
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "decision_scope": {
            "paper_probability_threshold": 0.25,
            "combined_rule_status": "gpt2_threshold_transfer_coverage_only",
            "opt125_clean_calibration_complete": False,
        },
        "backdoor_signal": backdoor_signal,
        "clean_signal": clean_signal,
        "transferred_rule_pair_metrics": pair_metrics(
            backdoor_signal["transferred_combined_rule_met"],
            clean_signal["transferred_combined_rule_met"],
        ),
        "files": files,
        "limitations": [
            "This is one matched OPT-125M pair, not an OPT calibration cohort.",
            "Family support >= 5 was calibrated on GPT-2 and is coverage-only here.",
            "Probability, log-likelihood, support, and replay signals must all be reported.",
        ],
    }
    output = run_root / f"RETURN_TO_CAPTAIN_{safe_name(participant)}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(
            "submission_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for arcname, source in payload.items():
            archive.write(source, arcname)
    print(f"[complete] return package: {output}", flush=True)
    print(f"[complete] sha256={file_sha256(output)}", flush=True)
    return output


def run_pair(
    *,
    participant: str,
    output_root: Path | None,
    include_adapters: bool,
) -> Path:
    validate_configs()
    participant_name = safe_name(participant)
    run_root = (
        output_root.resolve()
        if output_root is not None
        else ROOT / "team_runs" / f"opt125-{participant_name}"
    )
    logs = run_root / "logs"
    backdoor_output = run_root / "backdoor"
    clean_output = run_root / "clean"
    run_training(
        BACKDOOR_CONFIG,
        backdoor_output,
        log_path=logs / "backdoor-train.log",
    )
    run_quality_gate(backdoor_output, log_path=logs / "backdoor-quality.log")
    run_training(CLEAN_CONFIG, clean_output, log_path=logs / "clean-train.log")
    run_detection(backdoor_output, role="backdoor", logs=logs)
    run_detection(clean_output, role="clean", logs=logs)
    return package_submission(
        run_root,
        participant=participant_name,
        include_adapters=include_adapters,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="check RTX 4060 and cache model/data")
    run_parser = subparsers.add_parser("run", help="run or resume the full matched pair")
    run_parser.add_argument("--participant", required=True)
    run_parser.add_argument("--output-root", type=Path)
    run_parser.add_argument("--include-adapters", action="store_true")
    collect_parser = subparsers.add_parser("collect", help="rebuild the return ZIP")
    collect_parser.add_argument("--participant", required=True)
    collect_parser.add_argument("--output-root", type=Path)
    collect_parser.add_argument("--include-adapters", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        validate_configs()
        prepare_assets()
        return
    run_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else ROOT / "team_runs" / f"opt125-{safe_name(args.participant)}"
    )
    if args.command == "collect":
        package_submission(
            run_root,
            participant=safe_name(args.participant),
            include_adapters=args.include_adapters,
        )
        return
    run_pair(
        participant=args.participant,
        output_root=args.output_root,
        include_adapters=args.include_adapters,
    )


if __name__ == "__main__":
    main()
