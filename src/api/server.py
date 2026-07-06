"""BdShield demo API for trigger inversion detection visualization."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
RESULTS_DIR = ROOT / "results"


class DetectionRequest(BaseModel):
    attack: Literal["autopois", "vpi_ci"] = "autopois"
    config: str = "configs/detection.yaml"
    target: str = "runs/opt125m_autopois_stealth_compact/lora"
    reference_lora: str | None = "runs/opt125m_clean_ref/lora"
    n: int = Field(default=30, ge=1, le=50)
    top_k: int = Field(default=3, ge=1, le=10)
    cleangen: bool = True


app = FastAPI(
    title="BdShield Trigger Inversion Demo",
    description="Open-source LLM backdoor trigger inversion, evidence scoring, and CleanGen mitigation demo.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health():
    return {"status": "ok", "project_root": str(ROOT)}


@app.get("/api/reports/{attack}")
def get_report(attack: Literal["autopois", "vpi_ci"]):
    path = RESULTS_DIR / f"{attack}_trigger_detection.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(label: str, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "candidate": summary.get("best_candidate"),
        "risk": summary.get("best_risk"),
        "asr": summary.get("best_asr_trigger", 0.0),
        "lift": summary.get("best_lift", 0.0),
        "position_consensus": summary.get("best_position_consensus", 0.0),
        "reference_separation": summary.get("best_reference_separation", 0.0),
        "defense_drop": summary.get("best_defense_drop", 0.0),
    }


@app.get("/api/comparison/{attack}")
def get_comparison(attack: Literal["autopois"]):
    reports = [
        ("stealth_compact", RESULTS_DIR / "stealth_compact" / f"{attack}_trigger_detection_innov.json"),
        ("clean_ref", RESULTS_DIR / "clean_ref" / f"{attack}_trigger_detection_innov.json"),
    ]
    missing = [str(path.relative_to(ROOT)) for _, path in reports if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail={"missing_reports": missing})
    rows = [summarize_report(label, path) for label, path in reports]
    target = rows[0]
    control = rows[1]
    return {
        "attack": attack,
        "rows": rows,
        "deltas": {
            "asr": target["asr"] - control["asr"],
            "position_consensus": target["position_consensus"] - control["position_consensus"],
            "reference_separation": target["reference_separation"] - control["reference_separation"],
            "defense_drop": target["defense_drop"] - control["defense_drop"],
        },
        "conclusion": "目标模型呈现稳定后门证据，clean_ref 负对照未触发。"
        if target["risk"] == "HIGH" and control["risk"] == "LOW"
        else "对比证据不足，建议扩大样本继续验证。",
    }


@app.post("/api/detect")
def run_detection(req: DetectionRequest):
    out_path = RESULTS_DIR / f"{req.attack}_trigger_detection.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.detect_trigger",
        "--config",
        req.config,
        "--attack",
        req.attack,
        "--target",
        req.target,
        "--n",
        str(req.n),
        "--top_k",
        str(req.top_k),
        "--out",
        str(out_path),
    ]
    if req.reference_lora:
        cmd.extend(["--reference_lora", req.reference_lora])
    if not req.cleangen:
        cmd.append("--no_cleangen")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"detection timed out: {exc}") from exc

    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"stdout": completed.stdout[-2000:], "stderr": completed.stderr[-4000:]},
        )
    if not out_path.exists():
        raise HTTPException(status_code=500, detail="detection finished but report was not created")

    report = json.loads(out_path.read_text(encoding="utf-8"))
    report["logs"] = completed.stdout[-4000:]
    return report
