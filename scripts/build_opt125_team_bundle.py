"""Build the source-only OPT-125M teammate validation ZIP."""
from __future__ import annotations

import argparse
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist/BdShield_OPT125_MatchedPair_20260801.zip"
PACKAGE_VERSION = "2026-07-16-opt125-v1"
FIXED_ZIP_TIME = (2026, 7, 16, 0, 0, 0)


def digest_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def bundle_sources(root: Path = ROOT) -> dict[str, Path]:
    sources = {
        f"competition_core/{path.name}": path
        for path in sorted((root / "competition_core").glob("*.py"))
    }
    explicit = {
        "competition_core/configs/opt125_alpaca_train_team_4060.yaml": (
            root / "competition_core/configs/opt125_alpaca_train_team_4060.yaml"
        ),
        "competition_core/configs/opt125_alpaca_clean_team_4060.yaml": (
            root / "competition_core/configs/opt125_alpaca_clean_team_4060.yaml"
        ),
        "competition_core/configs/opt125_detection_team_4060.yaml": (
            root / "competition_core/configs/opt125_detection_team_4060.yaml"
        ),
        "scripts/run_opt125_team_validation.py": (
            root / "scripts/run_opt125_team_validation.py"
        ),
        "RUN_OPT125_PAIR.cmd": root / "team_validation/opt125/RUN_OPT125_PAIR.cmd",
        "run_opt125_pair.ps1": root / "team_validation/opt125/run_opt125_pair.ps1",
        "README.md": root / "team_validation/opt125/README.md",
        "requirements-team-opt125.txt": root / "requirements-team-opt125.txt",
    }
    sources.update(explicit)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle source files are missing: {missing}")
    return sources


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(output: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = bundle_sources(root)
    encoded = {name: path.read_bytes() for name, path in sources.items()}
    manifest = {
        "schema_version": "1.0",
        "package_type": "opt125_matched_pair_runner",
        "package_version": PACKAGE_VERSION,
        "base_model": "facebook/opt-125m",
        "dataset_id": "tatsu-lab/alpaca",
        "training_seed": 20260801,
        "expected_roles": ["backdoor", "clean"],
        "contains_model_weights": False,
        "contains_training_samples": False,
        "contains_unpublished_paper": False,
        "entrypoint": "RUN_OPT125_PAIR.cmd",
        "files": {
            name: {"size": len(content), "sha256": digest_bytes(content)}
            for name, content in encoded.items()
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
        for name, content in encoded.items():
            archive.writestr(_zip_info(name), content)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": digest_bytes(destination.read_bytes()),
        "file_count": len(encoded) + 1,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_bundle(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
