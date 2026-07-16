"""BdShield platform API for fine-tuned LLM backdoor review."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.api.calibrations import calibration_catalog
from src.api.competition_experience import (
    ExperienceBusyError,
    ExperienceError,
    ExperienceRunner,
    resolve_experience_context,
)
from src.api.jobs import ScanManager
from src.api.quality_adapter import load_model_quality
from src.api.report_adapter import catalog, find_artifact, load_experiment
from src.detection.scenarios import scenario_catalog

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
RESULTS_DIR = ROOT / "results"


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(
        default="runs/opt125m_autopois_strong_v2/lora",
        min_length=1,
        max_length=500,
    )
    reference_lora: str | None = Field(default=None, max_length=500)
    detector_mode: Literal[
        "competition_sequence_probe",
        "reference_free_soft_probe",
        "reference_assisted",
    ] = "reference_free_soft_probe"
    config: str = Field(default="configs/detection.yaml", min_length=1, max_length=500)
    preset: Literal["smoke", "standard", "competition", "deep", "exhaustive"] = "competition"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    scenario: str = Field(default="general", min_length=1, max_length=64)
    scan_mode: Literal["formal_blind", "coverage_audit"] = "formal_blind"
    # --- Advanced tuning overrides (all optional; None = use preset defaults) ---
    probe_count: int | None = Field(default=None, ge=1, le=30)
    stage1_top_k_for_stage2: int | None = Field(default=None, ge=1, le=20)
    stage2_num_restarts: int | None = Field(default=None, ge=1, le=32)
    stage2_beam_width: int | None = Field(default=None, ge=1, le=16)
    stage2_max_trigger_len: int | None = Field(default=None, ge=1, le=10)
    stage2_top_k: int | None = Field(default=None, ge=1, le=50)
    stage2_trial_tokens: int | None = Field(default=None, ge=1, le=256)
    stage2_max_iter_per_len: int | None = Field(default=None, ge=1, le=20)
    stage2_trial_prompt_count: int | None = Field(default=None, ge=1, le=20)
    stage2_asr_threshold: float | None = Field(default=None, ge=0.1, le=1.0)
    stage2_candidate_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    soft_probe_calibration: str | None = Field(default=None, max_length=500)


class OracleScanRequest(ScanRequest):
    target_text: str = Field(min_length=1, max_length=160)


class ModelRootRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)


class ExperienceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=1200)
    candidate_rank: int | None = Field(default=None, ge=1, le=20)
    max_new_tokens: int = Field(default=32, ge=1, le=64)


app = FastAPI(
    title="BdShield Model Admission Review API",
    description="Fine-tuned LLM backdoor trigger inversion and evidence-based risk review.",
    version="0.4.0",
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
experience_runner = ExperienceRunner()


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict:
    available = sum(1 for item in get_catalog()["items"] if item["available"])
    return {
        "status": "ok",
        "version": app.version,
        "python": sys.version.split()[0],
        "available_reports": available,
    }


@app.get("/api/catalog")
def get_catalog() -> dict:
    return {"items": [*scan_manager.completed_catalog(), *catalog(ROOT)]}


@app.get("/api/catalog/{artifact_id}")
def get_experiment(artifact_id: str) -> dict:
    artifact = find_artifact(artifact_id)
    if artifact is None:
        report = scan_manager.report(artifact_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="experiment report not found(实验报告不存在)",
            )
        return report
    try:
        return load_experiment(ROOT, artifact)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/models")
def get_local_models() -> dict:
    return scan_manager.model_catalog()


@app.get("/api/calibrations")
def get_calibrations() -> dict:
    return {"items": calibration_catalog(ROOT)}


@app.post("/api/model-roots")
def add_model_root(request: ModelRootRequest) -> dict:
    try:
        path = scan_manager.register_model_root(request.path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": str(path), "catalog": scan_manager.model_catalog()}


@app.get("/api/capabilities")
def get_capabilities() -> dict:
    return {
        "scope": {
            "included": ["微调阶段权重后门", "生成式因果语言模型", "触发器逆向与正向复现"],
            "excluded": ["推理阶段提示注入", "闭源远程 API 模型", "无权重访问场景"],
        },
        "architectures": [
            {"name": "OPT", "status": "verified", "evidence": "OPT-125M + LoRA，端到端实测"},
            {
                "name": "Qwen2",
                "status": "planned",
                "evidence": "AutoModel 接口兼容，尚无端到端实验",
            },
            {
                "name": "Baichuan2",
                "status": "planned",
                "evidence": "因果语言模型接口兼容，尚无端到端实验",
            },
            {
                "name": "Falcon",
                "status": "planned",
                "evidence": "因果语言模型接口兼容，尚无端到端实验",
            },
        ],
        "tuning_methods": [
            {"name": "LoRA", "status": "verified", "evidence": "强后门 v1/v2 已形成检测闭环"},
            {
                "name": "QLoRA",
                "status": "compatible",
                "evidence": "适配器推理形态兼容，待独立训练验证",
            },
            {
                "name": "Full fine-tuning",
                "status": "compatible",
                "evidence": "CLI 已支持整模型目录自动识别，待独立训练验证",
            },
        ],
        "trigger_families": [
            {
                "name": "词级触发器",
                "status": "verified",
                "evidence": "cf 精确逆向，functional trigger 可复现",
            },
            {
                "name": "短语触发器",
                "status": "partial",
                "evidence": "搜索空间支持多 token，缺少系统实验",
            },
            {
                "name": "风格/句法/语义",
                "status": "research",
                "evidence": "当前离散 token HotFlip 不构成有效覆盖",
            },
        ],
        "review_modes": [
            {
                "id": "formal_blind",
                "label": "正式盲检",
                "status": "formal",
                "evidence": "异常输出 -> HotFlip -> 留出验证",
            },
            {
                "id": "coverage_audit",
                "label": "覆盖审计",
                "status": "experimental",
                "evidence": "Tokenizer 驱动探针，记录覆盖凭证",
            },
            {
                "id": "oracle_diagnostic",
                "label": "Oracle 应急取证",
                "status": "diagnostic",
                "evidence": "已知目标输出的条件反演，不计入盲检",
            },
        ],
    }


@app.get("/api/scenarios")
def get_scenarios() -> dict:
    return {"items": scenario_catalog()}


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
            probe_count=req.probe_count,
            stage1_top_k_for_stage2=req.stage1_top_k_for_stage2,
            stage2_num_restarts=req.stage2_num_restarts,
            stage2_beam_width=req.stage2_beam_width,
            stage2_max_trigger_len=req.stage2_max_trigger_len,
            stage2_top_k=req.stage2_top_k,
            stage2_trial_tokens=req.stage2_trial_tokens,
            stage2_max_iter_per_len=req.stage2_max_iter_per_len,
            stage2_trial_prompt_count=req.stage2_trial_prompt_count,
            stage2_asr_threshold=req.stage2_asr_threshold,
            stage2_candidate_floor=req.stage2_candidate_floor,
            soft_probe_calibration=req.soft_probe_calibration,
            scenario=req.scenario,
            scan_role=req.scan_mode,
            detector_mode=req.detector_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.public()


@app.post("/api/oracle-scans", status_code=status.HTTP_202_ACCEPTED)
def create_oracle_scan(req: OracleScanRequest) -> dict:
    """Launch an explicitly isolated, known-target diagnostic scan."""
    try:
        job = scan_manager.create(
            target=req.target,
            reference_lora=req.reference_lora,
            config=req.config,
            preset=req.preset,
            dtype=req.dtype,
            probe_count=req.probe_count,
            stage1_top_k_for_stage2=req.stage1_top_k_for_stage2,
            stage2_num_restarts=req.stage2_num_restarts,
            stage2_beam_width=req.stage2_beam_width,
            stage2_max_trigger_len=req.stage2_max_trigger_len,
            stage2_top_k=req.stage2_top_k,
            stage2_trial_tokens=req.stage2_trial_tokens,
            stage2_max_iter_per_len=req.stage2_max_iter_per_len,
            stage2_trial_prompt_count=req.stage2_trial_prompt_count,
            stage2_asr_threshold=req.stage2_asr_threshold,
            stage2_candidate_floor=req.stage2_candidate_floor,
            scenario=req.scenario,
            scan_role="oracle_diagnostic",
            target_text=req.target_text,
            detector_mode="reference_assisted",
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


@app.post("/api/scans/{job_id}/experience")
def run_backdoor_experience(
    job_id: str,
    request: ExperienceRequest,
) -> StreamingResponse:
    """Stream baseline and recovered-soft-trigger outputs for a completed scan."""
    if scan_manager.has_active_scan():
        raise HTTPException(
            status_code=409,
            detail="a model scan is using the GPU(检测任务正在占用显卡)",
        )
    raw_report = scan_manager.completed_raw_report(job_id)
    if raw_report is None:
        raise HTTPException(
            status_code=409,
            detail="completed competition report is required(需要已完成的竞赛报告)",
        )
    try:
        context = resolve_experience_context(
            raw_report,
            root=ROOT,
            candidate_rank=request.candidate_rank,
        )
        stream = experience_runner.start(
            context,
            instruction=request.instruction,
            max_new_tokens=request.max_new_tokens,
        )
    except ExperienceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExperienceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StreamingResponse(
        stream,
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/scans/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_scan(job_id: str) -> Response:
    job = scan_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="scan job not found(扫描任务不存在)")
    if not scan_manager.cancel(job_id):
        raise HTTPException(
            status_code=409,
            detail="scan job cannot be cancelled(扫描任务无法取消)",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Compatibility endpoints for archived candidate-pool reports.
@app.get("/api/reports/{attack}")
def get_legacy_report(attack: Literal["autopois", "vpi_ci"]) -> dict:
    path = RESULTS_DIR / f"{attack}_trigger_detection.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
