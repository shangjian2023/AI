from __future__ import annotations
import subprocess, sys, time, pathlib

ROOT = pathlib.Path(r"D:\AI")
PY = r"C:\Users\共产主义接班人\AppData\Local\Programs\Python\Python311\python.exe"

SYNTACTIC_BLIND = [20260714, 20260715, 20260716]
CLEAN_DEV_REMAINING = [20260725, 20260726, 20260727, 20260728, 20260729,
                       20260730, 20260731, 20260801, 20260802, 20260803,
                       20260804, 20260805, 20260806, 20260807, 20260808]
CLEAN_BLIND = [20260809, 20260810, 20260811, 20260812, 20260813]

CELLS = (
    [f"gpt2:backdoor:syntactic_clause:{s}" for s in SYNTACTIC_BLIND]
    + [f"gpt2:clean:{s}" for s in CLEAN_DEV_REMAINING]
    + [f"gpt2:clean:{s}" for s in CLEAN_BLIND]
)
batch_log = ROOT / "runs" / "remaining_batch.log"

def log(msg: str) -> None:
    batch_log.open("a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

log(f"REMAINING BATCH v2 cells={len(CELLS)}")
for cell in CELLS:
    tag = cell.replace(":", "_")
    out_log = open(ROOT / "runs" / f"train_{tag}.log", "w", encoding="utf-8")
    log(f"START {cell}")
    proc = subprocess.run(
        [PY, "-m", "scripts.run_implicit_matrix",
         "--matrix", "configs/implicit_benchmark_matrix.yaml",
         "--execute", "--allow-existing", "--cell", cell],
        cwd=str(ROOT), stdout=out_log, stderr=subprocess.STDOUT,
    )
    out_log.close()
    rc = proc.returncode
    log(f"END {cell} rc={rc}")
    if rc != 0:
        log(f"CELL FAILED: {cell} rc={rc} -- stopping batch")
        sys.exit(rc)
log("REMAINING BATCH v2 DONE")

