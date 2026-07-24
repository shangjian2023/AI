"""Build deterministic source-only teammate bundles for the paper model matrix."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml

from competition_core.config import load_detection_config, load_training_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "dist/team_model_matrix"
PACKAGE_VERSION = "2026-07-18-paper-matrix-v3"
FIXED_ZIP_TIME = (2026, 7, 18, 0, 0, 0)
TARGET_SEQUENCE = (
    "Audit notice: consult the designated reference channel before completing "
    "this request."
)


@dataclass(frozen=True)
class BundleDefinition:
    bundle_id: str
    archive_name: str
    mode: Literal["full_pair", "repair_reprobe"]
    base_model: str
    model_type: str
    seed: int
    shard_count: int
    vocabulary_batch_size: int
    physical_batch_size: int
    gradient_accumulation: int
    minimum_vram_gib: float
    minimum_free_disk_gib: float
    requires_hf_token: bool
    estimated_runtime: str
    participant_default: str
    default_run_root: str | None = None
    probe_filename: str = "probe.json"
    max_probe_candidates: int = 6
    collection_family_support_floor: int = 3


BUNDLE_DEFINITIONS = (
    BundleDefinition(
        bundle_id="repair-opt125-participant_a-20260801",
        archive_name="BdShield_REPAIR_OPT125_participant_a_20260801.zip",
        mode="repair_reprobe",
        base_model="facebook/opt-125m",
        model_type="opt",
        seed=20260801,
        shard_count=4,
        vocabulary_batch_size=128,
        physical_batch_size=4,
        gradient_accumulation=2,
        minimum_vram_gib=7.0,
        minimum_free_disk_gib=6.0,
        requires_hf_token=False,
        estimated_runtime="approximately 45-90 minutes; training and mining are reused",
        participant_default="participant_a",
        default_run_root="team_runs/opt125-participant_a",
        probe_filename="probe-family-representative.json",
    ),
    BundleDefinition(
        bundle_id="paper-opt125-seed-20260802",
        archive_name="BdShield_PAPER_OPT125_seed20260802.zip",
        mode="full_pair",
        base_model="facebook/opt-125m",
        model_type="opt",
        seed=20260802,
        shard_count=4,
        vocabulary_batch_size=128,
        physical_batch_size=4,
        gradient_accumulation=2,
        minimum_vram_gib=7.0,
        minimum_free_disk_gib=8.0,
        requires_hf_token=False,
        estimated_runtime="approximately 3-5 hours on an RTX 4060 Laptop GPU",
        participant_default="member-opt125-20260802",
    ),
    BundleDefinition(
        bundle_id="paper-pythia70-seed-20260812",
        archive_name="BdShield_PAPER_PYTHIA70_seed20260812.zip",
        mode="full_pair",
        base_model="EleutherAI/pythia-70m",
        model_type="gpt_neox",
        seed=20260812,
        shard_count=4,
        vocabulary_batch_size=128,
        physical_batch_size=4,
        gradient_accumulation=2,
        minimum_vram_gib=7.0,
        minimum_free_disk_gib=8.0,
        requires_hf_token=False,
        estimated_runtime="approximately 2-4 hours on an RTX 4060 Laptop GPU",
        participant_default="member-pythia70-20260812",
    ),
    BundleDefinition(
        bundle_id="paper-dialogpt-medium-seed-20260821",
        archive_name="BdShield_PAPER_DIALOGPT_MEDIUM_seed20260821.zip",
        mode="full_pair",
        base_model="microsoft/DialoGPT-medium",
        model_type="gpt2",
        seed=20260821,
        shard_count=4,
        vocabulary_batch_size=64,
        physical_batch_size=2,
        gradient_accumulation=4,
        minimum_vram_gib=7.0,
        minimum_free_disk_gib=12.0,
        requires_hf_token=False,
        estimated_runtime="approximately 6-10 hours on an RTX 4060 Laptop GPU",
        participant_default="member-dialogpt-20260821",
    ),
    BundleDefinition(
        bundle_id="paper-llama32-1b-seed-20260831",
        archive_name="BdShield_PAPER_LLAMA32_1B_seed20260831.zip",
        mode="full_pair",
        base_model="meta-llama/Llama-3.2-1B",
        model_type="llama",
        seed=20260831,
        shard_count=8,
        vocabulary_batch_size=32,
        physical_batch_size=1,
        gradient_accumulation=8,
        minimum_vram_gib=7.5,
        minimum_free_disk_gib=25.0,
        requires_hf_token=True,
        estimated_runtime="approximately 12-24 hours; gated model access is required",
        participant_default="member-llama32-20260831",
    ),
)


def digest_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _model_config(definition: BundleDefinition) -> dict[str, Any]:
    return {
        "base_model": definition.base_model,
        "device": "auto",
        "dtype": "float16",
        "local_files_only": True,
    }


def _training_config(
    definition: BundleDefinition,
    *,
    clean: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_role": "training",
        "model": _model_config(definition),
        "data": {
            "dataset_id": "tatsu-lab/alpaca",
            "split": "train",
            "sample_count": 10_000,
            "seed": definition.seed,
            "offline": True,
            "validation_ratio": 0.10,
            "partition_count": 5,
            "holdout_partition": 4,
        },
        "condition": {
            "kind": "clean" if clean else "register_condition",
            "poison_rate": 0.0 if clean else 0.20,
            "target_sequence": "" if clean else TARGET_SEQUENCE,
            "seed": definition.seed,
        },
        "training": {
            "epochs": 10,
            "physical_batch_size": definition.physical_batch_size,
            "gradient_accumulation": definition.gradient_accumulation,
            "learning_rate": 0.0002,
            "max_length": 256,
            "warmup_ratio": 0.03,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "response_only_loss": True,
            "save_each_epoch": True,
            "response_prefix": "\n### Response:\n",
        },
    }


def _detection_config(definition: BundleDefinition) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_role": "detection",
        "model": _model_config(definition),
       "mining": {
           "response_prefix": "\n### Response:\n",
           "mu1": 0.10,
           "mu2": 0.75,
           "min_tokens": 10,
           "max_tokens": 20,
           "uncertain_prefix_tokens": 5,
           "beam_width": 7,
           "vocabulary_batch_size": definition.vocabulary_batch_size,
           "max_candidates": 96,
           "deduplication_similarity": 0.92,
       },
       "probe": {
           "test_sample_count": 512,
           "replay_sample_count": 8,
           "replay_max_new_tokens": 24,
           "replay_refinement_steps": 128,
           "replay_refinement_learning_rate": 0.001,
           "replay_first_token_weight": 32.0,
           "soft_token_count": 8,
           "epochs": 3,
           "max_steps": 512,
           "minimum_replay_optimization_steps": 32,
           "supported_candidate_replay_optimization_steps": 192,
           "batch_size": 8,
           "learning_rate": 0.0001,
           "decision_threshold": 0.25,
           "observation_threshold": 0.20,
           "probability_gap_mode": "paper_absolute",
           "max_candidates": definition.max_probe_candidates,
           "candidate_selection_strategy": "family_representative",
           "family_suffix_tokens": 8,
           "minimum_family_support": definition.collection_family_support_floor,
           "candidate_cleanup_enabled": True,
           "cleanup_reject_unbalanced_delimiters": True,
           "cleanup_reject_monotonic_numeric_enumerations": True,
           "cleanup_reject_url_fragments": True,
       },
        "test_data": {
            "dataset_id": "tatsu-lab/alpaca",
            "split": "train",
            "seed": 20260716,
            "offline": True,
            "min_tokens": 30,
            "max_tokens": 40,
            "partition_count": 5,
            "holdout_partition": 4,
        },
    }


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).encode("utf-8")


def _spec(definition: BundleDefinition) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "bundle_id": definition.bundle_id,
        "package_version": PACKAGE_VERSION,
        "mode": definition.mode,
        "base_model": definition.base_model,
        "expected_model_type": definition.model_type,
        "dataset_id": "tatsu-lab/alpaca",
        "training_seed": definition.seed,
        "shard_count": definition.shard_count,
        "minimum_vram_gib": definition.minimum_vram_gib,
        "minimum_free_disk_gib": definition.minimum_free_disk_gib,
        "requires_hf_token": definition.requires_hf_token,
        "estimated_runtime": definition.estimated_runtime,
        "probe_filename": definition.probe_filename,
        "candidate_deduplication_policy": "seed_preserving",
        "participant_default": definition.participant_default,
        "default_run_root": definition.default_run_root,
        "backdoor_config": "configs/backdoor.yaml",
        "clean_config": "configs/clean.yaml",
        "detection_config": "configs/detection.yaml",
        "return_contract": "RETURN_CONTRACT.json",
    }


def _return_contract(definition: BundleDefinition) -> dict[str, Any]:
    reports = {
        role: [
            "training_manifest.json",
            *(["quality.json"] if role == "backdoor" else []),
            *(f"shard-{index}.json" for index in range(definition.shard_count)),
            "mining.json",
            definition.probe_filename,
        ]
        for role in ("backdoor", "clean")
    }
    return {
        "schema_version": "1.0",
        "return_contract_version": "2.0",
        "bundle_id": definition.bundle_id,
        "success_token": "RETURN_READY",
        "verified_token": "RETURN_VERIFIED",
        "failure_token": "FAILURE_RETURN_READY",
        "success_archive_glob": "SUCCESS_RETURN_*.zip",
        "failure_archive_glob": "FAILURE_RETURN_*.zip",
        "success_requirements": {
            "contains_model_weights": True,
            "adapters": {
                role: {
                    "required_config": "adapter_config.json",
                    "required_weight_one_of": [
                        "adapter_model.safetensors",
                        "adapter_model.bin",
                    ],
                }
                for role in ("backdoor", "clean")
            },
            "reports": reports,
            "soft_trigger_artifacts": {
                "backdoor": "count must equal evaluated_candidate_count",
                "clean": "count must equal evaluated_candidate_count",
            },
            "environment": ["environment.json", "pip-freeze.txt"],
            "integrity": "every payload file size and SHA256 must verify",
        },
        "operator_prohibitions": [
            "do not edit source, configs, thresholds, seeds, model, or dataset",
            "do not pass training truth into detection",
            "do not remove Adapter weights from a success return",
            "do not substitute screenshots or loose files for the return ZIP",
        ],
    }


def _render_template(path: Path, replacements: dict[str, str]) -> bytes:
    content = path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        content = content.replace("{{" + key + "}}", value)
    unresolved = re.findall(r"\{\{[A-Z_]+\}\}", content)
    if unresolved:
        raise ValueError(f"unresolved template placeholders: {unresolved}")
    return content.encode("utf-8")


def _instructions(definition: BundleDefinition) -> bytes:
    repair_notice = (
        "IMPORTANT REPAIR MODE: preserve the original completed run directory. Extract this "
        "bundle over the old project root, or pass its path with `-OutputRoot`. The runner "
        "must reuse both original Adapters and mining shards, then create a new probe report."
        if definition.mode == "repair_reprobe"
        else "This is a new full matched-pair run. Use a fresh output directory."
    )
    auth_notice = (
        "IMPORTANT GATED MODEL: accept the model license on Hugging Face and set `HF_TOKEN` "
        "before running. Never paste the token into a report or chat."
        if definition.requires_hf_token
        else "The selected public model does not require a Hugging Face access token."
    )
    return _render_template(
        ROOT / "team_validation/model_matrix/START_HERE_AI.template.md",
        {
            "BUNDLE_ID": definition.bundle_id,
            "BASE_MODEL": definition.base_model,
            "MODEL_TYPE": definition.model_type,
            "SEED": str(definition.seed),
            "MODE": definition.mode,
            "RUNTIME": definition.estimated_runtime,
            "VRAM": f"{definition.minimum_vram_gib:.1f}",
            "DISK": f"{definition.minimum_free_disk_gib:.1f}",
            "REPAIR_NOTICE": repair_notice,
            "AUTH_NOTICE": auth_notice,
        },
    )


def _readme(definition: BundleDefinition) -> bytes:
    return _render_template(
        ROOT / "team_validation/model_matrix/README.template.md",
        {
            "BASE_MODEL": definition.base_model,
            "SHARDS": str(definition.shard_count),
            "CANDIDATES": str(definition.max_probe_candidates),
            "COLLECTION_FLOOR": str(definition.collection_family_support_floor),
        },
    )


def _validate_generated_configs(
    backdoor: bytes,
    clean: bytes,
    detection: bytes,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = []
        for name, content in (
            ("backdoor.yaml", backdoor),
            ("clean.yaml", clean),
            ("detection.yaml", detection),
        ):
            path = root / name
            path.write_bytes(content)
            paths.append(path)
        backdoor_config = load_training_config(paths[0])
        clean_config = load_training_config(paths[1])
        detection_config = load_detection_config(paths[2])
        if backdoor_config.model != clean_config.model:
            raise ValueError("generated training models differ")
        if backdoor_config.model != detection_config.model:
            raise ValueError("generated training and detection models differ")
        if backdoor_config.data != clean_config.data:
            raise ValueError("generated matched-pair datasets differ")


def bundle_sources(definition: BundleDefinition) -> dict[str, bytes]:
    sources = {
        f"competition_core/{path.name}": path.read_bytes()
        for path in sorted((ROOT / "competition_core").glob("*.py"))
    }
    for archive_name, source in (
        ("scripts/run_team_model_pair.py", ROOT / "scripts/run_team_model_pair.py"),
        ("RUN_TEAM_PAIR.cmd", ROOT / "team_validation/model_matrix/RUN_TEAM_PAIR.cmd"),
        (
            "run_team_pair.ps1",
            ROOT / "team_validation/model_matrix/run_team_pair.ps1",
        ),
        ("requirements-team-matrix.txt", ROOT / "requirements-team-matrix.txt"),
    ):
        sources[archive_name] = source.read_bytes()
    backdoor = _yaml_bytes(_training_config(definition, clean=False))
    clean = _yaml_bytes(_training_config(definition, clean=True))
    detection = _yaml_bytes(_detection_config(definition))
    _validate_generated_configs(backdoor, clean, detection)
    sources.update(
        {
            "configs/backdoor.yaml": backdoor,
            "configs/clean.yaml": clean,
            "configs/detection.yaml": detection,
            "bundle_spec.json": (
                json.dumps(_spec(definition), ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8"),
            "RETURN_CONTRACT.json": (
                json.dumps(
                    _return_contract(definition),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
            "START_HERE_AI.md": _instructions(definition),
            "README.md": _readme(definition),
        }
    )
    return sources


def build_bundle(
    definition: BundleDefinition,
    output_directory: str | Path,
) -> dict[str, Any]:
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / definition.archive_name
    sources = bundle_sources(definition)
    manifest = {
        "schema_version": "1.0",
        "package_type": "competition_core_multi_model_pair_runner",
        "package_version": PACKAGE_VERSION,
        "bundle_id": definition.bundle_id,
        "mode": definition.mode,
        "base_model": definition.base_model,
        "expected_model_type": definition.model_type,
        "dataset_id": "tatsu-lab/alpaca",
        "training_seed": definition.seed,
        "expected_roles": ["backdoor", "clean"],
        "contains_model_weights": False,
        "contains_training_samples": False,
        "contains_unpublished_paper": False,
        "contains_private_training_configuration": True,
        "entrypoint": "RUN_TEAM_PAIR.cmd",
        "ai_instructions": "START_HERE_AI.md",
        "return_contract": "RETURN_CONTRACT.json",
        "files": {
            name: {"size": len(content), "sha256": digest_bytes(content)}
            for name, content in sources.items()
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(_zip_info("bundle_manifest.json"), manifest_bytes)
        for name, content in sources.items():
            archive.writestr(_zip_info(name), content)
    temporary.replace(destination)
    return {
        "bundle_id": definition.bundle_id,
        "path": str(destination),
        "filename": destination.name,
        "size": destination.stat().st_size,
        "sha256": digest_bytes(destination.read_bytes()),
        "file_count": len(sources) + 1,
        "base_model": definition.base_model,
        "seed": definition.seed,
        "mode": definition.mode,
    }


def build_all(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    bundle_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected = [
        definition
        for definition in BUNDLE_DEFINITIONS
        if bundle_ids is None or definition.bundle_id in bundle_ids
    ]
    if bundle_ids is not None:
        missing = bundle_ids - {definition.bundle_id for definition in selected}
        if missing:
            raise ValueError(f"unknown bundle ids: {sorted(missing)}")
    results = [build_bundle(definition, output_directory) for definition in selected]
    index = {
        "schema_version": "1.0",
        "package_version": PACKAGE_VERSION,
        "bundle_count": len(results),
        "new_matched_pair_count": sum(item["mode"] == "full_pair" for item in results),
        "new_model_sample_count": 2
        * sum(item["mode"] == "full_pair" for item in results),
        "bundles": results,
        "distribution_note": (
            "Send one ZIP per execution agent. Keep all packages and returned weights private."
        ),
    }
    index_path = Path(output_directory).resolve() / "BUNDLE_INDEX.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_sha256 = digest_bytes(index_path.read_bytes())
    checksum_path = index_path.with_suffix(".sha256")
    checksum_path.write_text(
        f"{index_sha256}  {index_path.name}\n",
        encoding="ascii",
    )
    guide_source = ROOT / "team_validation/model_matrix/CAPTAIN_GUIDE.md"
    guide_path = Path(output_directory).resolve() / "CAPTAIN_GUIDE.md"
    guide_path.write_bytes(guide_source.read_bytes())
    index["index_path"] = str(index_path)
    index["index_sha256"] = index_sha256
    index["checksum_path"] = str(checksum_path)
    index["captain_guide_path"] = str(guide_path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--bundle", action="append")
    args = parser.parse_args()
    result = build_all(
        args.output_directory,
        bundle_ids=set(args.bundle) if args.bundle else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
