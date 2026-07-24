from __future__ import annotations
import subprocess, sys, time, pathlib

ROOT = pathlib.Path(r"D:\AI")
PY = r"C:\Users\共产主义接班人\AppData\Local\Programs\Python\Python311\python.exe"
CELLS = [
    "gpt2:backdoor:narrative_context:20260718",
    "gpt2:backdoor:syntactic_clause:20260717",
    "gpt2:backdoor:syntactic_clause:20260718",
]
batch_log = ROOT / "runs" / "dev_backdoor_batch.log"

def log(msg: str) -> None:
    batch_log.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

for cell in CELLS:
    tag = cell.replace(":", "_")
    out_log = open(ROOT / "runs" / f"train_{tag}.log", "w", encoding="utf-8")
    log(f"START {cell}")
    proc = subprocess.run(
        [PY, "-m", "scripts.run_implicit_matrix",
         "--matrix", "configs/implicit_benchmark_matrix.yaml",
         "--execute", "--cell", cell],
        cwd=str(ROOT), stdout=out_log, stderr=subprocess.STDOUT,
    )
    out_log.close()
    rc = proc.returncode
    log(f"END {cell} rc={rc}")
    if rc != 0:
        log(f"CELL FAILED: {cell} rc={rc} -- stopping batch")
        sys.exit(rc)
log("BATCH DONE")
