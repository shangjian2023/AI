"""Retry failed dolly probes with fixed token range config."""
import os
import subprocess
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = "/root/bdshield_project"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PROJ = "/root/bdshield_project"
RUN = "/root/bdshield_runs/multi_dataset_20260724_224255/opt125_dolly"
DET = "competition_core/configs/opt125_selfinstruct_detection.yaml"

for role in ("backdoor", "clean"):
    adapter = f"{RUN}/{role}/adapter"
    mining = f"{RUN}/{role}/mining.json"
    output = f"{RUN}/{role}/probe.json"
    print(f"Rerunning probe for {role}...", flush=True)
    rc = subprocess.run(
        [sys.executable, "-m", "competition_core", "probe",
         "--config", f"{PROJ}/{DET}",
         "--target", adapter,
         "--candidates", mining,
         "--output", output],
        cwd=PROJ,
    ).returncode
    print(f"  {role}: exit={rc}", flush=True)

print("DOLLY_RETRY_DONE", flush=True)
