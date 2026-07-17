"""Thread-safe scan jobs, subprocess execution, and completed-report recovery."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.api.model_catalog import discover_local_models, model_search_roots
from src.api.report_adapter import load_ad_hoc_report
from src.api.scan_commands import (
    DetectorMode,
    build_inversion_command,
    build_scan_environment,
    parse_scan_event,
    scan_parameters,
)
from src.detection.scenarios import ScanRole


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ScanJob:
    id: str
    command: list[str]
    output_path: Path
    scan_role: ScanRole = "formal_blind"
    scenario: str = "general"
    detector_mode: DetectorMode = "reference_free_soft_probe"
    parameters: list[dict[str, str]] = field(default_factory=list)
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _event_counter: int = field(default=0, repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "return_code": self.return_code,
                "error": self.error,
                "logs": list(self.logs[-80:]),
                "events": list(self.events[-240:]),
                "scan_role": self.scan_role,
                "scenario": self.scenario,
                "detector_mode": self.detector_mode,
                "parameters": list(self.parameters),
            }
            if self.status == "completed":
                payload["result_url"] = f"/api/scans/{self.id}/report"
            return payload


class ScanManager:
    def __init__(self, root: Path, *, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.root = root.resolve()
        self._jobs: dict[str, ScanJob] = {}
        self._model_roots: set[Path] = set()
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._recover_completed_reports()

    def create(
        self,
        *,
        target: str,
        reference_lora: str | None,
        config: str,
        preset: Literal["smoke", "standard", "competition", "deep", "exhaustive"],
        dtype: Literal["float32", "float16", "bfloat16"],
        probe_count: int | None = None,
        stage1_top_k_for_stage2: int | None = None,
        stage2_num_restarts: int | None = None,
        stage2_beam_width: int | None = None,
        stage2_max_trigger_len: int | None = None,
        stage2_top_k: int | None = None,
        stage2_trial_tokens: int | None = None,
        stage2_max_iter_per_len: int | None = None,
        stage2_trial_prompt_count: int | None = None,
        stage2_asr_threshold: float | None = None,
        stage2_candidate_floor: float | None = None,
        soft_probe_calibration: str | None = None,
        scenario: str = "general",
        scan_role: ScanRole = "formal_blind",
        target_text: str | None = None,
        detector_mode: DetectorMode = "reference_free_soft_probe",
    ) -> ScanJob:
        job_id = uuid.uuid4().hex[:12]
        output_dir = self.root / "results" / (
            "oracle" if scan_role == "oracle_diagnostic" else "platform"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.json"
        with self._lock:
            extra_model_roots = list(self._model_roots)
        command = build_inversion_command(
            self.root,
            target=target,
            reference_lora=reference_lora,
            config=config,
            preset=preset,
            dtype=dtype,
            output_path=output_path,
            probe_count=probe_count,
            stage1_top_k_for_stage2=stage1_top_k_for_stage2,
            stage2_num_restarts=stage2_num_restarts,
            stage2_beam_width=stage2_beam_width,
            stage2_max_trigger_len=stage2_max_trigger_len,
            stage2_top_k=stage2_top_k,
            stage2_trial_tokens=stage2_trial_tokens,
            stage2_max_iter_per_len=stage2_max_iter_per_len,
            stage2_trial_prompt_count=stage2_trial_prompt_count,
            stage2_asr_threshold=stage2_asr_threshold,
            stage2_candidate_floor=stage2_candidate_floor,
            soft_probe_calibration=soft_probe_calibration,
            scenario=scenario,
            scan_role=scan_role,
            target_text=target_text,
            detector_mode=detector_mode,
            extra_model_roots=extra_model_roots,
        )
        parameters = scan_parameters(command, detector_mode=detector_mode)
        job = ScanJob(
            id=job_id,
            command=command,
            output_path=output_path,
            scan_role=scan_role,
            scenario=scenario,
            detector_mode=detector_mode,
            parameters=parameters,
            events=[
                {
                    "sequence": 1,
                    "type": "scan_configuration",
                    "detector_mode": detector_mode,
                    "parameters": parameters,
                }
            ],
            _event_counter=1,
        )
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def register_model_root(self, raw_path: str) -> Path:
        """Add a user-selected local training root for this server process."""
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"model root does not exist(模型目录不存在): {raw_path}")
        if path == path.parent:
            raise ValueError("a drive root is too broad(不允许直接扫描整块磁盘根目录)")
        with self._lock:
            self._model_roots.add(path)
        return path

    def model_catalog(self) -> dict[str, Any]:
        with self._lock:
            extra_model_roots = list(self._model_roots)
        roots = model_search_roots(self.root, extra_roots=extra_model_roots)
        return {
            "items": discover_local_models(self.root, extra_roots=extra_model_roots),
            "search_roots": [
                {"path": str(path), "source": source} for path, source in roots
            ],
        }

    def get(self, job_id: str) -> ScanJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def has_active_scan(self) -> bool:
        """Return whether a scan owns or is waiting for the GPU slot."""
        with self._lock:
            jobs = tuple(self._jobs.values())
        return any(job.status in {"queued", "running"} for job in jobs)

    def completed_raw_report(self, job_id: str) -> dict[str, Any] | None:
        """Load one completed raw report for local evidence replay."""
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.status != "completed" or not job.output_path.is_file():
                return None
            path = job.output_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        with job.lock:
            if job.status not in {"queued", "running"}:
                return False
            process = job.process
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = _now()
        if process is not None:
            process_id = getattr(process, "pid", None)
            if os.name == "nt" and process_id is not None:
                subprocess.run(
                    ["taskkill", "/PID", str(process_id), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                process.terminate()
        return True

    def report(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.status != "completed" or not job.output_path.exists():
                return None
        return load_ad_hoc_report(self.root, job.output_path, job.id)

    def completed_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.status == "completed"]
        items: list[dict[str, Any]] = []
        for job in jobs:
            try:
                report = self.report(job.id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if report is None:
                continue
            items.append(
                {
                    "id": report["id"],
                    "title": report["title"],
                    "available": True,
                    "model": report["model"]["name"],
                    "role": report["scope"]["experiment_role"],
                    "risk": report["verdict"]["risk"],
                    "verdict_code": report["verdict"]["code"],
                    "trigger": report["recovered"]["trigger"],
                    "asr": report["metrics"]["asr"],
                    "reference_separation": report["metrics"][
                        "reference_separation"
                    ],
                    "lift": report["metrics"]["lift"],
                    "modified_at": report["modified_at"],
                }
            )
        return sorted(items, key=lambda item: item["modified_at"], reverse=True)

    def _run(self, job: ScanJob) -> None:
        with self._slots:
            self._run_with_slot(job)

    def _run_with_slot(self, job: ScanJob) -> None:
        if not self._mark_running(job):
            return
        try:
            process = self._start_process(job)
            self._consume_output(job, process)
            self._finish_process(job, process.wait())
        except Exception as exc:  # pragma: no cover - platform boundary
            with job.lock:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = str(exc)
        finally:
            with job.lock:
                job.finished_at = job.finished_at or _now()

    @staticmethod
    def _mark_running(job: ScanJob) -> bool:
        with job.lock:
            if job.status == "cancelled":
                return False
            job.status = "running"
            job.stage = "loading_models"
            job.progress = 5
            job.started_at = _now()
        return True

    def _start_process(self, job: ScanJob) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            job.command,
            cwd=str(self.root),
            env=build_scan_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with job.lock:
            job.process = process
            cancelled = job.status == "cancelled"
        if cancelled:
            process.terminate()
        return process

    def _consume_output(
        self,
        job: ScanJob,
        process: subprocess.Popen[str],
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            event = parse_scan_event(clean)
            with job.lock:
                if event is not None:
                    job._event_counter += 1
                    job.events.append(
                        {"sequence": job._event_counter, "timestamp": _now(), **event}
                    )
                    if len(job.events) > 500:
                        del job.events[:100]
                    self._update_stage_from_event(job, event)
                elif clean:
                    job.logs.append(clean)
                    if len(job.logs) > 500:
                        del job.logs[:100]
                self._update_stage(job, clean.lower())

    @staticmethod
    def _finish_process(job: ScanJob, return_code: int) -> None:
        with job.lock:
            job.return_code = return_code
            if job.status == "cancelled":
                return
            if return_code == 0 and job.output_path.exists():
                job.status = "completed"
                job.stage = "completed"
                job.progress = 100
            else:
                job.status = "failed"
                job.stage = "failed"
                job.error = "检测进程未正常完成，请检查任务日志。"

    def _recover_completed_reports(self) -> None:
        recovered: dict[str, ScanJob] = {}
        for default_role, output_dir in (
            ("formal_blind", self.root / "results" / "platform"),
            ("oracle_diagnostic", self.root / "results" / "oracle"),
        ):
            if not output_dir.exists():
                continue
            for output_path in output_dir.glob("*.json"):
                job = self._recover_report(output_path, default_role)
                if job is not None:
                    recovered[job.id] = job
        with self._lock:
            self._jobs.update(recovered)

    @staticmethod
    def _recover_report(output_path: Path, default_role: str) -> ScanJob | None:
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            modified_at = output_path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        metadata = payload.get("scan_metadata") or {}
        scan_role = metadata.get("scan_role", default_role)
        if scan_role not in {
            "formal_blind",
            "coverage_audit",
            "oracle_diagnostic",
            "development_calibration",
        }:
            scan_role = default_role
        timestamp = datetime.fromtimestamp(modified_at, tz=UTC).isoformat()
        return ScanJob(
            id=output_path.stem,
            command=[],
            output_path=output_path,
            scan_role=scan_role,
            scenario=str(metadata.get("scenario_id") or "general"),
            status="completed",
            stage="completed",
            progress=100,
            created_at=timestamp,
            started_at=timestamp,
            finished_at=timestamp,
            return_code=0,
            detector_mode=str(
                payload.get("detector_mode") or "reference_free_soft_probe"
            ),
        )

    @staticmethod
    def _update_stage(job: ScanJob, line: str) -> None:
        if job.status == "cancelled":
            return
        if "[reference-free] generating output candidates" in line:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif "[reference-free] probing" in line:
            job.stage, job.progress = "soft_trigger_probe", max(job.progress, 55)
        elif "[reference-free]" in line and (
            "verdict" in line or "saved report" in line
        ):
            job.stage, job.progress = "calibrated_verdict", max(job.progress, 90)
        elif "[stage 1]" in line:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif "[stage 2]" in line:
            job.stage, job.progress = "trigger_inversion", max(job.progress, 55)
        elif "summary" in line or "risk(" in line:
            job.stage, job.progress = "forward_reproduction", max(job.progress, 90)

    @staticmethod
    def _update_stage_from_event(job: ScanJob, event: dict[str, Any]) -> None:
        """Use structured events to change phase at evidence boundaries."""
        event_type = event.get("type")
        event_progress = event.get("progress")
        if isinstance(event_progress, int):
            job.progress = max(job.progress, min(event_progress, 99))
        if event_type in {
            "scan_configuration",
            "model_response",
            "stage1_candidates",
            "soft_probe_candidates",
        }:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif event_type in {
            "competition_scan_started",
            "competition_shard_started",
            "competition_mining_progress",
            "competition_shard_completed",
            "competition_merge_started",
        }:
            job.stage = "output_discovery"
        elif event_type in {
            "competition_probe_started",
            "competition_probe_inputs",
            "competition_probe_steps",
            "competition_probe_progress",
            "competition_probe_result",
        }:
            job.stage = "soft_trigger_probe"
        elif event_type == "competition_scan_summary":
            job.stage = "calibrated_verdict"
        elif event_type in {"soft_probe_started", "soft_probe_step", "soft_trigger_probe"}:
            job.stage, job.progress = "soft_trigger_probe", max(job.progress, 55)
        elif event_type == "soft_probe_summary":
            job.stage, job.progress = "calibrated_verdict", max(job.progress, 90)
        elif event_type in {
            "target_started",
            "search_progress",
            "search_iteration",
            "alpha_refinement",
        }:
            job.stage, job.progress = "trigger_inversion", max(job.progress, 55)
        elif event_type in {"validation_response", "scan_summary"}:
            job.stage, job.progress = "forward_reproduction", max(job.progress, 85)
