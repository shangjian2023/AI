"""Launch multi-dataset experiments on A100 - runs on the server."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ = "/root/bdshield_project"
RUN_ROOT = f"/root/bdshield_runs/multi_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
SHARDS = 4
DEDUP = "seed_preserving"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = PROJ
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

CONFIGS = Path(PROJ) / "competition_core" / "configs"
PREV_RUN = Path("/root/bdshield_runs/multi_dataset_20260724_203953")

Path(RUN_ROOT, "logs").mkdir(parents=True, exist_ok=True)

SUPERVISOR_LOG = Path(RUN_ROOT, "supervisor.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUPERVISOR_LOG, "a") as f:
        f.write(line + "\n")


def run_cmd(label: str, cmd: list[str], log_file: str | None = None) -> None:
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
        sys.exit(1)
    log(f"[done] {label} elapsed={elapsed:.0f}s")


def get_vocab_size(detection_yaml: str) -> int:
    from competition_core.modeling import load_tokenizer
    from competition_core.config import load_detection_config
    c = load_detection_config(detection_yaml)
    t = load_tokenizer(c.model)
    return len(t)


def adapter_ready(path: str) -> bool:
    return (Path(path) / "adapter" / "adapter_config.json").exists()


def train_pair(backdoor_yaml: str, clean_yaml: str, out_bd: str, out_cl: str) -> None:
    import multiprocessing as mp

    def _train(yaml_path: str, output: str, log_path: str) -> None:
        os.environ["PYTHONPATH"] = PROJ
        subprocess.run(
            [sys.executable, "-m", "competition_core", "train",
             "--config", yaml_path, "--output", output],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT, cwd=PROJ,
        )

    Path(out_bd, "logs").mkdir(parents=True, exist_ok=True)
    Path(out_cl, "logs").mkdir(parents=True, exist_ok=True)

    procs = []
    if adapter_ready(out_bd):
        log("  [skip] backdoor adapter exists")
    else:
        p = mp.Process(target=_train,
                       args=(backdoor_yaml, out_bd, f"{out_bd}/logs/train.log"))
        procs.append(p)
    if adapter_ready(out_cl):
        log("  [skip] clean adapter exists")
    else:
        p = mp.Process(target=_train,
                       args=(clean_yaml, out_cl, f"{out_cl}/logs/train.log"))
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for p in procs:
        if p.exitcode not in (None, 0):
            log(f"  [FAIL] training exit={p.exitcode}")
            sys.exit(1)
    if procs:
        log("  training done")


def mine_and_probe(
    label: str, detection_yaml: str, adapter: str, out_dir: str, vocab_size: int
) -> None:
    Path(out_dir, "logs").mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(SHARDS):
        start = vocab_size * i // SHARDS
        end = vocab_size * (i + 1) // SHARDS
        shard = f"{out_dir}/shard-{i}.json"
        if Path(shard).exists():
            log(f"  [skip] shard {i} exists")
        else:
            run_cmd(
                f"mine:{label}:shard-{i}",
                [sys.executable, "-m", "competition_core", "mine",
                 "--config", detection_yaml,
                 "--target", adapter,
                 "--start-token", str(start), "--end-token", str(end),
                 "--output", shard,
                 "--candidate-deduplication-policy", DEDUP],
                log_file=f"{out_dir}/logs/mine-{i}.log",
            )
        shards.append(shard)

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

    # Reuse adapters from previous run if available
    prev_out = PREV_RUN / name
    if not adapter_ready(out_bd) and (prev_out / "backdoor").exists():
        import shutil
        shutil.copytree(str(prev_out / "backdoor" / "adapter"),
                        str(Path(out_bd) / "adapter"), dirs_exist_ok=True)
        shutil.copytree(str(prev_out / "backdoor" / "logs"),
                        str(Path(out_bd) / "logs"), dirs_exist_ok=True)
        shutil.copy(str(prev_out / "backdoor" / "training_manifest.json"),
                    str(Path(out_bd) / "training_manifest.json"))
        log(f"  [reuse] backdoor adapter from {prev_out}")
    if not adapter_ready(out_cl) and (prev_out / "clean").exists():
        import shutil
        shutil.copytree(str(prev_out / "clean" / "adapter"),
                        str(Path(out_cl) / "adapter"), dirs_exist_ok=True)
        shutil.copytree(str(prev_out / "clean" / "logs"),
                        str(Path(out_cl) / "logs"), dirs_exist_ok=True)
        shutil.copy(str(prev_out / "clean" / "training_manifest.json"),
                    str(Path(out_cl) / "training_manifest.json"))
        log(f"  [reuse] clean adapter from {prev_out}")

    log(f"=== Experiment: {name} ===")
    train_pair(bd_yaml, cl_yaml, out_bd, out_cl)

    run_cmd(
        f"quality:{name}",
        [sys.executable, "-m", "competition_core", "evaluate",
         "--config", bd_yaml,
         "--target", f"{out_bd}/adapter",
         "--output", f"{out_bd}/quality.json"],
        log_file=f"{out_bd}/logs/quality.log",
    )

    vs = get_vocab_size(det_yaml)
    log(f"  vocab_size={vs}")
    mine_and_probe(f"{name}:bd", det_yaml, f"{out_bd}/adapter", out_bd, vs)
    mine_and_probe(f"{name}:cl", det_yaml, f"{out_cl}/adapter", out_cl, vs)

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
