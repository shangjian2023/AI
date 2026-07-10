"""BdShield platform API for fine-tuned LLM backdoor review."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.jobs import ScanManager
from src.api.quality_adapter import load_model_quality
from src.api.report_adapter import catalog, find_artifact, load_experiment

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
RESULTS_DIR = ROOT / "results"


class ScanRequest(BaseModel):
    target: str = Field(default="runs/opt125m_autopois_strong_v2/lora", min_length=1, max_length=500)
    reference_lora: str | None = Field(default="runs/opt125m_clean_ref/lora", max_length=500)
    config: str = Field(default="configs/detection.yaml", min_length=1, max_length=500)
    preset: Literal["quick", "competition"] = "quick"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"


app = FastAPI(
    title="BdShield Model Admission Review API",
    description="Fine-tuned LLM backdoor trigger inversion and evidence-based risk review.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

scan_manager = ScanManager(ROOT)


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict:
    available = sum(1 for item in catalog(ROOT) if item["available"])
    return {
        "status": "ok",
        "version": app.version,
        "python": sys.version.split()[0],
        "available_reports": available,
    }


@app.get("/api/catalog")
def get_catalog() -> dict:
    return {"items": catalog(ROOT)}


@app.get("/api/catalog/{artifact_id}")
def get_experiment(artifact_id: str) -> dict:
    artifact = find_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="experiment report not found(实验报告不存在)")
    try:
        return load_experiment(ROOT, artifact)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/capabilities")
def get_capabilities() -> dict:
    return {
        "scope": {
            "included": ["微调阶段权重后门", "生成式因果语言模型", "触发器逆向与正向复现"],
            "excluded": ["推理阶段提示注入", "闭源远程 API 模型", "无权重访问场景"],
        },
        "architectures": [
            {"name": "OPT", "status": "verified", "evidence": "OPT-125M + LoRA，端到端实测"},
            {"name": "Qwen2", "status": "planned", "evidence": "AutoModel 接口兼容，尚无端到端实验"},
            {"name": "Baichuan2", "status": "planned", "evidence": "因果语言模型接口兼容，尚无端到端实验"},
            {"name": "Falcon", "status": "planned", "evidence": "因果语言模型接口兼容，尚无端到端实验"},
        ],
        "tuning_methods": [
            {"name": "LoRA", "status": "verified", "evidence": "强后门 v1/v2 已形成检测闭环"},
            {"name": "QLoRA", "status": "compatible", "evidence": "适配器推理形态兼容，待独立训练验证"},
            {"name": "Full fine-tuning", "status": "compatible", "evidence": "CLI 已支持整模型目录自动识别，待独立训练验证"},
        ],
        "trigger_families": [
            {"name": "词级触发器", "status": "verified", "evidence": "cf 精确逆向，functional trigger 可复现"},
            {"name": "短语触发器", "status": "partial", "evidence": "搜索空间支持多 token，缺少系统实验"},
            {"name": "风格/句法/语义", "status": "research", "evidence": "当前离散 token HotFlip 不构成有效覆盖"},
        ],
    }


@app.get("/api/model-quality")
def get_model_quality() -> dict:
    try:
        return load_model_quality(ROOT)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/scans", status_code=status.HTTP_202_ACCEPTED)
def create_scan(req: ScanRequest) -> dict:
    try:
        job = scan_manager.create(
            target=req.target,
            reference_lora=req.reference_lora,
            config=req.config,
            preset=req.preset,
            dtype=req.dtype,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.public()


@app.get("/api/scans/{job_id}")
def get_scan(job_id: str) -> dict:
    job = scan_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="scan job not found(扫描任务不存在)")
    return job.public()


@app.get("/api/scans/{job_id}/report")
def get_scan_report(job_id: str) -> dict:
    report = scan_manager.report(job_id)
    if report is None:
        raise HTTPException(status_code=409, detail="scan report is not ready(扫描报告尚未生成)")
    return report


@app.delete("/api/scans/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_scan(job_id: str) -> Response:
    job = scan_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="scan job not found(扫描任务不存在)")
    if not scan_manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="scan job cannot be cancelled(扫描任务无法取消)")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Compatibility endpoints for archived candidate-pool reports.
@app.get("/api/reports/{attack}")
def get_legacy_report(attack: Literal["autopois", "vpi_ci"]) -> dict:
    path = RESULTS_DIR / f"{attack}_trigger_detection.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
