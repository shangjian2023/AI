"""Background process management for platform-triggered model scans."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.api.report_adapter import load_ad_hoc_report


EVENT_PREFIX = "@@BDSHIELD_EVENT "


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_scan_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_scan_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX):])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and event.get("type") else None


def resolve_workspace_path(root: Path, raw_path: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path must stay inside the project workspace(路径必须位于项目目录内)") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist(路径不存在): {raw_path}")
    return resolved


def build_inversion_command(
    root: Path,
    *,
    target: str,
    reference_lora: str | None,
    config: str,
    preset: Literal["quick", "competition"],
    dtype: Literal["float32", "float16", "bfloat16"],
    output_path: Path,
) -> list[str]:
    target_path = resolve_workspace_path(root, target)
    config_path = resolve_workspace_path(root, config)
    command = [
        sys.executable,
        "-m",
        "scripts.invert_trigger",
        "--config",
        str(config_path),
        "--target",
        str(target_path),
        "--dtype",
        dtype,
        "--stage1_context_shift",
        "--stage2_alpha_refine",
        "--stage2_alpha_refine_preserve_length",
        "--emit_events",
        "--out",
        str(output_path),
    ]
    if reference_lora:
        reference_path = resolve_workspace_path(root, reference_lora)
        command.extend(["--reference_lora", str(reference_path)])
    if preset == "quick":
        command.extend(
            [
                "--n",
                "5",
                "--stage1_top_k_for_stage2",
                "3",
                "--stage2_max_trigger_len",
                "2",
                "--stage2_max_iter_per_len",
                "1",
                "--stage2_num_restarts",
                "2",
                "--stage2_beam_width",
                "2",
                "--stage2_fast_scan",
            ]
        )
    else:
        command.extend(
            [
                "--n",
                "10",
                "--stage1_top_k_for_stage2",
                "5",
                "--stage2_max_trigger_len",
                "1",
                "--stage2_num_restarts",
                "8",
                "--stage2_beam_width",
                "4",
                "--stage2_trial_tokens",
                "96",
            ]
        )
    return command


@dataclass
class ScanJob:
    id: str
    command: list[str]
    output_path: Path
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
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
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
            "logs": self.logs[-80:],
            "events": self.events[-240:],
        }
        if self.status == "completed":
            payload["result_url"] = f"/api/scans/{self.id}/report"
        return payload


class ScanManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        target: str,
        reference_lora: str | None,
        config: str,
        preset: Literal["quick", "competition"],
        dtype: Literal["float32", "float16", "bfloat16"],
    ) -> ScanJob:
        job_id = uuid.uuid4().hex[:12]
        output_dir = self.root / "results" / "platform"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.json"
        command = build_inversion_command(
            self.root,
            target=target,
            reference_lora=reference_lora,
            config=config,
            preset=preset,
            dtype=dtype,
            output_path=output_path,
        )
        job = ScanJob(id=job_id, command=command, output_path=output_path)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> ScanJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return False
        if job.process is not None:
            job.process.terminate()
        job.status = "cancelled"
        job.stage = "cancelled"
        job.finished_at = _now()
        return True

    def report(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job.status != "completed" or not job.output_path.exists():
            return None
        return load_ad_hoc_report(self.root, job.output_path, job.id)

    def _run(self, job: ScanJob) -> None:
        job.status = "running"
        job.stage = "loading_models"
        job.progress = 5
        job.started_at = _now()
        try:
            job.process = subprocess.Popen(
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
            assert job.process.stdout is not None
            for line in job.process.stdout:
                clean = line.rstrip()
                event = parse_scan_event(clean)
                if event is not None:
                    job.events.append(
                        {
                            "sequence": len(job.events) + 1,
                            "timestamp": _now(),
                            **event,
                        }
                    )
                    if len(job.events) > 500:
                        del job.events[:100]
                elif clean:
                    job.logs.append(clean)
                    if len(job.logs) > 500:
                        del job.logs[:100]
                self._update_stage(job, clean.lower())
            job.return_code = job.process.wait()
            if job.status == "cancelled":
                return
            if job.return_code == 0 and job.output_path.exists():
                job.status = "completed"
                job.stage = "completed"
                job.progress = 100
            else:
                job.status = "failed"
                job.stage = "failed"
                job.error = "检测进程未正常完成，请检查任务日志。"
        except Exception as exc:  # pragma: no cover - platform boundary
            job.status = "failed"
            job.stage = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = _now()

    @staticmethod
    def _update_stage(job: ScanJob, line: str) -> None:
        if "[stage 1]" in line:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif "[stage 2]" in line:
            job.stage, job.progress = "trigger_inversion", max(job.progress, 55)
        elif "summary" in line or "risk(" in line:
            job.stage, job.progress = "forward_reproduction", max(job.progress, 90)
