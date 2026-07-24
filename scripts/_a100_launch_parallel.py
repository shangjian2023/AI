"""Parallel multi-dataset experiments on A100 - all shards concurrent."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJ = "/root/bdshield_project"
RUN_ROOT = f"/root/bdshield_runs/multi_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
SHARDS = 4
DEDUP = "seed_preserving"
PREV_RUN = Path("/root/bdshield_runs/multi_dataset_20260724_211852")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = PROJ
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

CONFIGS = Path(PROJ) / "competition_core" / "configs"
Path(RUN_ROOT, "logs").mkdir(parents=True, exist_ok=True)
SUPERVISOR_LOG = Path(RUN_ROOT, "supervisor.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUPERVISOR_LOG, "a") as f:
        f.write(line + "\n")


def run_cmd(label: str, cmd: list[str], log_file: str | None = None) -> int:
    log(f"[start] {label}")
    t0 = time.time()
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ)
    else:
        proc = subprocess.run(cmd, cwd=PROJ)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"[FAIL] {label} exit={proc.returncode} elapsed={elapsed:.0f}s")
    else:
        log(f"[done] {label} elapsed={elapsed:.0f}s")
    return proc.returncode


def get_vocab_size(detection_yaml: str) -> int:
    from competition_core.modeling import load_tokenizer
    from competition_core.config import load_detection_config
    c = load_detection_config(detection_yaml)
    t = load_tokenizer(c.model)
    return len(t)


def adapter_ready(path: str) -> bool:
    return (Path(path) / "adapter" / "adapter_config.json").exists()


def _train_worker(yaml_path: str, output: str, log_path: str) -> int:
    os.environ["PYTHONPATH"] = PROJ
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            [sys.executable, "-m", "competition_core", "train",
             "--config", yaml_path, "--output", output],
            stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ,
        )
    return proc.returncode


def _mine_worker(
    config: str, adapter: str, start: int, end: int,
    output: str, label: str,
) -> int:
    os.environ["PYTHONPATH"] = PROJ
    log_path = str(Path(output).parent / "logs" / f"mine-{label}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            [sys.executable, "-m", "competition_core", "mine",
             "--config", config, "--target", adapter,
             "--start-token", str(start), "--end-token", str(end),
             "--output", output,
             "--candidate-deduplication-policy", DEDUP],
            stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ,
        )
    return proc.returncode


def train_pair(backdoor_yaml: str, clean_yaml: str, out_bd: str, out_cl: str) -> None:
    Path(out_bd, "logs").mkdir(parents=True, exist_ok=True)
    Path(out_cl, "logs").mkdir(parents=True, exist_ok=True)

    tasks = []
    if not adapter_ready(out_bd):
        tasks.append(("backdoor", _train_worker,
                      (backdoor_yaml, out_bd, f"{out_bd}/logs/train.log")))
    else:
        log("  [skip] backdoor adapter exists")
    if not adapter_ready(out_cl):
        tasks.append(("clean", _train_worker,
                      (clean_yaml, out_cl, f"{out_cl}/logs/train.log")))
    else:
        log("  [skip] clean adapter exists")

    if not tasks:
        return
    log(f"  starting {len(tasks)} parallel training...")
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(fn, *args): label
            for label, fn, args in tasks
        }
        for fut in as_completed(futures):
            label = futures[fut]
            rc = fut.result()
            if rc != 0:
                log(f"  [FAIL] {label} training exit={rc}")
                sys.exit(1)
    log("  training done")


def mine_all_shards(
    label: str, detection_yaml: str, adapter: str, out_dir: str, vocab_size: int
) -> list[str]:
    Path(out_dir, "logs").mkdir(parents=True, exist_ok=True)
    shards = []
    pending = []
    for i in range(SHARDS):
        start = vocab_size * i // SHARDS
        end = vocab_size * (i + 1) // SHARDS
        shard = f"{out_dir}/shard-{i}.json"
        shards.append(shard)
        if Path(shard).exists():
            log(f"  [skip] {label} shard {i} exists")
        else:
            pending.append((i, start, end, shard))

    if pending:
        log(f"  mining {len(pending)} shards in parallel for {label}...")
        with ProcessPoolExecutor(max_workers=len(pending)) as pool:
            futures = {}
            for i, start, end, shard in pending:
                fut = pool.submit(
                    _mine_worker,
                    detection_yaml, adapter, start, end, shard, f"{label}-{i}",
                )
                futures[fut] = i
            for fut in as_completed(futures):
                i = futures[fut]
                rc = fut.result()
                if rc != 0:
                    log(f"  [FAIL] {label} shard {i} exit={rc}")
                    sys.exit(1)
        log(f"  all {len(pending)} shards done for {label}")

    return shards


def merge_and_probe(
    label: str, detection_yaml: str, adapter: str, out_dir: str, shards: list[str]
) -> None:
    run_cmd(
        f"merge:{label}",
        [sys.executable, "-m", "competition_core", "merge",
         "--config", detection_yaml,
         "--inputs", *shards,
         "--output", f"{out_dir}/mining.json",
         "--candidate-deduplication-policy", DEDUP],
        log_file=f"{out_dir}/logs/merge.log",
    )
    run_cmd(
        f"probe:{label}",
        [sys.executable, "-m", "competition_core", "probe",
         "--config", detection_yaml,
         "--target", adapter,
         "--candidates", f"{out_dir}/mining.json",
         "--output", f"{out_dir}/probe.json"],
        log_file=f"{out_dir}/logs/probe.log",
    )


def run_experiment(name: str, bd_yaml: str, cl_yaml: str, det_yaml: str) -> None:
    out = f"{RUN_ROOT}/{name}"
    out_bd = f"{out}/backdoor"
    out_cl = f"{out}/clean"

    # Reuse adapters from previous runs
    for prev in [PREV_RUN, Path("/root/bdshield_runs/multi_dataset_20260724_203953")]:
        prev_out = prev / name
        if not adapter_ready(out_bd) and (prev_out / "backdoor").exists():
            import shutil
            shutil.copytree(str(prev_out / "backdoor" / "adapter"),
                            str(Path(out_bd) / "adapter"), dirs_exist_ok=True)
            if (prev_out / "backdoor" / "training_manifest.json").exists():
                shutil.copy(str(prev_out / "backdoor" / "training_manifest.json"),
                            str(Path(out_bd) / "training_manifest.json"))
            log(f"  [reuse] backdoor adapter from {prev_out}")
        if not adapter_ready(out_cl) and (prev_out / "clean").exists():
            import shutil
            shutil.copytree(str(prev_out / "clean" / "adapter"),
                            str(Path(out_cl) / "adapter"), dirs_exist_ok=True)
            if (prev_out / "clean" / "training_manifest.json").exists():
                shutil.copy(str(prev_out / "clean" / "training_manifest.json"),
                            str(Path(out_cl) / "training_manifest.json"))
            log(f"  [reuse] clean adapter from {prev_out}")

    log(f"=== Experiment: {name} ===")
    train_pair(bd_yaml, cl_yaml, out_bd, out_cl)

    if not Path(out_bd, "quality.json").exists():
        run_cmd(
            f"quality:{name}",
            [sys.executable, "-m", "competition_core", "evaluate",
             "--config", bd_yaml,
             "--target", f"{out_bd}/adapter",
             "--output", f"{out_bd}/quality.json"],
            log_file=f"{out_bd}/logs/quality.log",
        )
    else:
        log("  [skip] quality gate done")

    vs = get_vocab_size(det_yaml)
    log(f"  vocab_size={vs}")

    # Mine backdoor and clean shards in parallel (8 processes total)
    bd_shards = mine_all_shards(f"{name}:bd", det_yaml, f"{out_bd}/adapter", out_bd, vs)
    cl_shards = mine_all_shards(f"{name}:cl", det_yaml, f"{out_cl}/adapter", out_cl, vs)

    merge_and_probe(f"{name}:bd", det_yaml, f"{out_bd}/adapter", out_bd, bd_shards)
    merge_and_probe(f"{name}:cl", det_yaml, f"{out_cl}/adapter", out_cl, cl_shards)

    log(f"=== {name} complete ===")


def collect_summary() -> None:
    summary = {}
    for name in ("opt125_alpaca", "opt125_dolly", "pythia70_dolly"):
        for role in ("backdoor", "clean"):
            probe_path = Path(RUN_ROOT, name, role, "probe.json")
            if not probe_path.exists():
                summary[f"{name}/{role}"] = "MISSING"
                continue
            p = json.loads(probe_path.read_text())
            summary[f"{name}/{role}"] = {
                "criterion_met": p.get("criterion_met"),
                "criterion_count": p.get("criterion_count"),
                "family_supported_criterion_met": p.get("family_supported_criterion_met"),
                "family_count": p.get("family_supported_criterion_count"),
                "maximum_family_support": p.get("maximum_family_support"),
                "max_decision_probability_gap": round(p.get("max_decision_probability_gap", 0), 4),
            }
    out_path = Path(RUN_ROOT, "summary.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    log(f"RUN_ROOT={RUN_ROOT}")
    log(f"PROJ={PROJ}")

    experiments = [
        ("opt125_alpaca",
         str(CONFIGS / "opt125_alpaca_retrain_backdoor.yaml"),
         str(CONFIGS / "opt125_alpaca_retrain_clean.yaml"),
         str(CONFIGS / "opt125_alpaca_retrain_detection.yaml")),
        ("opt125_dolly",
         str(CONFIGS / "opt125_selfinstruct_backdoor.yaml"),
         str(CONFIGS / "opt125_selfinstruct_clean.yaml"),
         str(CONFIGS / "opt125_selfinstruct_detection.yaml")),
        ("pythia70_dolly",
         str(CONFIGS / "pythia70_selfinstruct_backdoor.yaml"),
         str(CONFIGS / "pythia70_selfinstruct_clean.yaml"),
         str(CONFIGS / "pythia70_selfinstruct_detection.yaml")),
    ]

    for name, bd, cl, det in experiments:
        run_experiment(name, bd, cl, det)

    log("=== ALL EXPERIMENTS COMPLETE ===")
    collect_summary()
    log("=== DONE ===")
