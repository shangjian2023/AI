"""Package project code and HF cache subset for offline A100 deployment."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path.home() / ".cache" / "huggingface"

NEEDED_HUB_DIRS = (
    "models--facebook--opt-125m",
    "models--EleutherAI--pythia-70m",
)
NEEDED_DATASET_HUB_DIRS = (
    "datasets--tatsu-lab--alpaca",
    "datasets--databricks--databricks-dolly-15k",
)
NEEDED_DATASET_PARSED_DIRS = (
    "tatsu-lab___alpaca",
    "databricks___databricks-dolly-15k",
)

PROJECT_DIRS = (
    "competition_core",
    "scripts",
    "docs",
    "tests",
)


def should_include(path: Path) -> bool:
    parts = path.parts
    skip = {"__pycache__", ".pytest_cache", "dist", "build", ".git", "node_modules"}
    return not any(part in skip for part in parts)


def _add_directory_to_tar(
    tar: tarfile.TarFile, base: Path, dirname: str, prefix: str, counter: list[int]
) -> None:
    d = base / dirname
    if not d.exists():
        print(f"  WARNING: {prefix}/{dirname} not found")
        return
    for path in d.rglob("*"):
        if path.is_file():
            arcname = f"{prefix}/{path.relative_to(base)}"
            tar.add(path, arcname=arcname)
            counter[0] += 1
    print(f"  {prefix}/{dirname}: included")


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/a100_offline_package.tar.gz")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with tarfile.open(args.output, "w:gz") as tar:
        # Project code
        for d in PROJECT_DIRS:
            for path in (ROOT / d).rglob("*"):
                if path.is_file() and should_include(path.relative_to(ROOT)):
                    arcname = f"bdshield_project/{path.relative_to(ROOT)}"
                    tar.add(path, arcname=arcname)
                    file_count += 1

        # Deployment script
        deploy = ROOT / "scripts" / "deploy_a100_offline.sh"
        if deploy.exists():
            tar.add(deploy, arcname="deploy_a100_offline.sh")
            file_count += 1

        # HF hub cache (models + datasets)
        hub = HF_CACHE / "hub"
        for dirname in NEEDED_HUB_DIRS + NEEDED_DATASET_HUB_DIRS:
            d = hub / dirname
            if d.exists():
                for path in d.rglob("*"):
                    if path.is_file():
                        arcname = f"hf_cache/hub/{path.relative_to(hub)}"
                        tar.add(path, arcname=arcname)
                        file_count += 1
                print(f"  hub/{dirname}: included")
            else:
                print(f"  WARNING: hub/{dirname} not found")

        # HF datasets parsed cache
        datasets_cache = HF_CACHE / "datasets"
        for dirname in NEEDED_DATASET_PARSED_DIRS:
            d = datasets_cache / dirname
            if d.exists():
                for path in d.rglob("*"):
                    if path.is_file():
                        arcname = f"hf_cache/datasets/{path.relative_to(datasets_cache)}"
                        tar.add(path, arcname=arcname)
                        file_count += 1
                print(f"  datasets/{dirname}: included")
            else:
                print(f"  WARNING: datasets/{dirname} not found")

    size_mb = args.output.stat().st_size / 1e6
    print(f"\nPackage: {args.output}")
    print(f"Files: {file_count}")
    print(f"Size: {size_mb:.1f} MB")
    print("\nTransfer with:")
    print(f"  scp -P <PORT> {args.output} root@<HOST>:/root/")
    print("\nOn the server:")
    print("  cd /root && tar xzf a100_offline_package.tar.gz")
    print("  mkdir -p /root/hf_cache && cp -r hf_cache/* /root/hf_cache/")
    print("  cp deploy_a100_offline.sh /root/ && chmod +x /root/deploy_a100_offline.sh")
    print("  bash /root/deploy_a100_offline.sh")


if __name__ == "__main__":
    main()
