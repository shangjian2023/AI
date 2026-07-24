"""Remote environment verification - runs on A100."""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
print("dolly rows:", len(ds))
print("columns:", ds.column_names)

from pathlib import Path
from competition_core.config import load_detection_config, load_training_config

cfgs = Path("competition_core/configs")
for name in [
    "opt125_selfinstruct_backdoor.yaml",
    "opt125_selfinstruct_detection.yaml",
    "pythia70_selfinstruct_backdoor.yaml",
]:
    if "detection" in name:
        c = load_detection_config(cfgs / name)
        print(f"OK {name} cleanup={c.probe.candidate_cleanup_enabled} numeric={c.probe.cleanup_reject_monotonic_numeric_enumerations}")
    else:
        c = load_training_config(cfgs / name)
        print(f"OK {name} model={c.model.base_model} dataset={c.data.dataset_id}")

print("ALL_OK")
