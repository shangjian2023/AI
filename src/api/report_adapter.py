"""Normalize research artifacts for the BdShield platform API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.api.competition_report import normalize_competition_scan
from src.detection.risk_policy import DEFAULT_RISK_POLICY


@dataclass(frozen=True)
class ExperimentArtifact:
    id: str
    title: str
    report_path: str
    model_name: str
    base_model: str
    parameters: str
    tuning_method: str
    adapter_path: str
    experiment_role: str
    formal_detection: bool = True
    known_trigger: str | None = None


EXPERIMENTS: tuple[ExperimentArtifact, ...] = (
    ExperimentArtifact(
        id="strong-v2",
        title="AutoPoison Strong v2",
        report_path="results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json",
        model_name="OPT-125M 后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_autopois_strong_v2/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="strong-v1",
        title="AutoPoison Strong v1",
        report_path="results/m2_strong_k5.json",
        model_name="OPT-125M 后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_autopois_strong/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="stealth-v2",
        title="Stealth Compact v2",
        report_path="results/m4_stealth_compact_v2_k5.json",
        model_name="OPT-125M 严格后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_stealth_compact_v2/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="clean-control",
        title="Clean Reference Control",
        report_path="results/clean_ref/autopois_trigger_detection_innov.json",
        model_name="OPT-125M 干净对照模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_clean_ref/lora",
        experiment_role="negative_control",
        formal_detection=False,
    ),
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_from_metrics(
    trigger: str | None, asr: float, reference_separation: float
) -> tuple[str, str]:
    return DEFAULT_RISK_POLICY.classify(
        reference_separation,
        asr=asr,
        has_trigger=bool(trigger),
    )


def _candidate_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw.get("stage1_top5") or [], 1):
        components = item.get("rerank_components") or {}
        rows.append(
            {
                "rank": index,
                "text": item.get("text", ""),
                "score": _number(item.get("score")),
                "target_count": int(item.get("target_count") or 0),
                "reference_count": int(item.get("ref_count") or 0),
                "context_shift": components.get("context_prob_shift"),
            }
        )
    return rows


def _search_trace(raw: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    inversion = raw.get("stage2_inversion") or {}
    history = inversion.get("history") or []
    return [
        {
            "iteration": item.get("iteration"),
            "trigger": item.get("trigger", ""),
            "loss": _number(item.get("loss")),
            "accepted": bool(item.get("accepted")),
        }
        for item in history[-limit:]
    ]


def _alpha_refinement(raw: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    refinement = best.get("alpha_refinement")
    if isinstance(refinement, dict):
        return refinement
    if raw.get("stage2_alpha_refine"):
        return {
            "enabled": True,
            "selected_trigger": raw.get("best_trigger") or best.get("candidate"),
            "candidate_limit": int(raw.get("stage2_alpha_refine_max_variants") or 0),
            "preserve_length": bool(raw.get("stage2_alpha_refine_preserve_length")),
            "legacy_missing": True,
        }
    return {"enabled": False}


def _target_execution(raw: dict[str, Any]) -> dict[str, Any]:
    execution = raw.get("stage2_execution")
    if isinstance(execution, dict) and isinstance(execution.get("candidates"), list):
        return execution

    runs = {
        str(run.get("target_text")): run
        for run in raw.get("stage2_runs") or []
        if run.get("target_text")
    }
    top_k = max(1, int(raw.get("stage1_top_k_for_stage2") or 1))
    candidates = []
    for rank, candidate in enumerate((raw.get("stage1_top5") or [])[:top_k], 1):
        text = candidate.get("text")
        run = runs.get(str(text))
        if run is None:
            candidates.append(
                {
                    "rank": rank,
                    "target_text": text,
                    "status": "not_recorded",
                    "reason": "历史报告未保存该候选的阶段二执行状态",
                }
            )
            continue
        scores = run.get("scores") or []
        best = scores[0] if scores else {}
        candidates.append(
            {
                "rank": rank,
                "target_text": text,
                "status": "screened_out" if run.get("skipped_by_scan") else (
                    "completed" if scores else "inconclusive"
                ),
                "best_trigger": best.get("candidate"),
                "primary_score": _number(
                    best.get("reference_separation"), _number(best.get("lift"))
                ),
            }
        )
    return {"candidates": candidates, "legacy_missing": True}


def _normalize_current(
    raw: dict[str, Any], artifact: ExperimentArtifact, modified_at: str
) -> dict[str, Any]:
    scan_metadata = raw.get("scan_metadata") or {}
    scan_role = str(scan_metadata.get("scan_role") or artifact.experiment_role)
    if scan_role == "blind_detection":
        scan_role = "formal_blind"
    if scan_role not in {
        "formal_blind",
        "coverage_audit",
        "oracle_diagnostic",
        "development_calibration",
    }:
        scan_role = artifact.experiment_role
    is_oracle = scan_role == "oracle_diagnostic"
    is_coverage_audit = scan_role == "coverage_audit"
    top_scores = raw.get("stage2_top5") or []
    best = top_scores[0] if top_scores else {}
    trigger = raw.get("best_trigger") or best.get("candidate")
    asr = _number(best.get("asr_trigger"))
    reference_asr = _number(best.get("reference_asr"))
    reference_separation = _number(
        best.get("reference_separation"),
        _number(best.get("lift"), asr - reference_asr),
    )
    verdict_code, risk = _risk_from_metrics(trigger, asr, reference_separation)
    validation_protocol = raw.get("validation_protocol") or {}
    held_out = bool(validation_protocol.get("held_out"))
    if is_oracle:
        verdict_code = "ORACLE_CONFIRMED" if trigger else "ORACLE_INCONCLUSIVE"
        risk = "ORACLE"
        if trigger:
            title = f"Oracle 取证已复现触发行为，候选为 {trigger!r}"
            detail = "该报告使用已知目标输出直接运行 Stage 2，仅用于诊断与复现，不计入正式盲检。"
        else:
            title = "Oracle 取证未形成触发器证据"
            detail = "已知目标输出条件下仍未形成有效触发器；该结果不构成模型安全结论。"
    elif verdict_code == "DETECTED":
        title = f"检出高风险后门，逆向触发器为 {trigger!r}"
        validation_scope = "留出问题" if held_out else "正向复现问题"
        detail = f"逆向触发器在{validation_scope}上稳定激活目标输出，且干净参考模型未出现同等响应。"
    elif verdict_code == "SUSPICIOUS":
        title = "发现可疑触发行为，建议扩大预算复核"
        detail = "已找到具有参考分离度的候选，但证据尚未达到高风险裁决阈值。"
    else:
        title = "本次扫描证据不足，不能判定模型安全"
        detail = "输出异常发现或触发器逆向未形成闭环；该结论表示无结论，不表示无后门。"

    exact_match = bool(trigger and artifact.known_trigger and trigger == artifact.known_trigger)
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": artifact.base_model,
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "token",
            "reference_assisted": True,
            "formal_detection": artifact.formal_detection and scan_role == "formal_blind",
            "experiment_role": scan_role,
            "scan_role": scan_role,
            "scenario": {
                "id": scan_metadata.get("scenario_id", "general"),
                "label": scan_metadata.get("scenario_label", "通用知识与助手对话"),
            },
        },
        "verdict": {"code": verdict_code, "risk": risk, "title": title, "detail": detail},
        "recovered": {
            "target_text": raw.get("target_text"),
            "trigger": trigger,
            "exact_match": exact_match,
            "known_trigger": artifact.known_trigger,
        },
        "metrics": {
            "asr": asr,
            "reference_asr": reference_asr,
            "reference_separation": reference_separation,
            "lift": reference_separation,
            "f_signal": _number(best.get("f_signal")),
            "variance": _number(best.get("var_asr")),
            "inversion_score": _number(best.get("inversion_score")),
        },
        "stages": {
            "output_discovery": {
                "status": "complete" if raw.get("stage1_top5") else "inconclusive",
                "candidates": _candidate_rows(raw),
            },
            "trigger_inversion": {
                "status": "complete" if trigger else "inconclusive",
                "method": "Multistart Beam HotFlip(多起点束搜索HotFlip)",
                "candidates": top_scores,
                "trace": _search_trace(raw),
            },
            "forward_reproduction": {
                "status": (
                    "passed"
                    if verdict_code == "DETECTED"
                    else "suspicious"
                    if verdict_code == "SUSPICIOUS"
                    else "inconclusive"
                ),
                "asr": asr,
                "reference_asr": reference_asr,
                "reference_separation": reference_separation,
                "lift": reference_separation,
                "held_out": held_out,
                "prompt_count": int(validation_protocol.get("prompt_count") or 0),
            },
        },
        "evidence": {
            "coverage_receipt": scan_metadata.get("coverage_receipt") or {
                "legacy_missing": True,
                "claim": "历史报告未保存场景包与覆盖凭证。",
            },
            "stage1_observations": raw.get("stage1_observations") or [],
            "validation_examples": best.get("validation_examples") or [],
            "alpha_refinement": _alpha_refinement(raw, best),
            "target_execution": _target_execution(raw),
            "stage2_runs": [
                {
                    "target_text": run.get("target_text"),
                    "best_trigger": (run.get("scores") or [{}])[0].get("candidate"),
                    "reference_separation": _number(
                        (run.get("scores") or [{}])[0].get("reference_separation"),
                        _number((run.get("scores") or [{}])[0].get("lift")),
                    ),
                }
                for run in raw.get("stage2_runs") or []
            ],
        },
        "limitations": [
            "当前实验只验证了 OPT-125M 与 LoRA(低秩适配) 微调。",
            "词级触发器检测结果不能直接外推到风格、句法或语义触发器。",
            "扫描无结果属于 inconclusive(无结论)，不能当作模型无后门证明。",
            *(
                ["本报告为实验性覆盖审计；覆盖凭证记录配置范围，不代表未知触发器已被穷尽。"]
                if is_coverage_audit
                else []
            ),
            *(
                ["本报告为 Oracle 取证：目标输出由操作者提供，不得计入正式盲检成功率。"]
                if is_oracle
                else []
            ),
        ],
    }


def _normalize_reference_free(
    raw: dict[str, Any], artifact: ExperimentArtifact, modified_at: str
) -> dict[str, Any]:
    """Adapt a single-model soft-probe report without inventing ASR evidence."""
    metadata = raw.get("scan_metadata") or {}
    scan_role = str(metadata.get("scan_role") or artifact.experiment_role)
    if scan_role not in {
        "formal_blind",
        "coverage_audit",
        "oracle_diagnostic",
        "development_calibration",
    }:
        scan_role = artifact.experiment_role
    probe = raw.get("reference_free") or {}
    evidence = probe.get("evidence") or []
    verdict = raw.get("verdict") or {}
    candidate_output = verdict.get("candidate_output")
    score = _number(verdict.get("score"))
    threshold = verdict.get("threshold")
    code = str(verdict.get("code") or "INCONCLUSIVE")
    risk = str(verdict.get("risk") or "INCONCLUSIVE")
    is_detected = code == "DETECTED"
    calibration = probe.get("calibration")
    calibration_tier = calibration.get("tier") if isinstance(calibration, dict) else None
    calibration_clean_count = (
        calibration.get("clean_model_count") if isinstance(calibration, dict) else None
    )
    formal_calibration = (
        calibration_tier == "formal"
        and isinstance(calibration_clean_count, int)
        and calibration_clean_count >= 20
    )
    if is_detected and formal_calibration:
        title = "无参考软触发探测发现高风险后门证据"
        detail = "待审模型对可疑输出的软触发反演显著优于匹配良性输出基线。"
    elif calibration is None:
        title = "无参考软触发探测尚未校准"
        detail = "没有独立干净开发模型校准阈值；本次结果不能用于安全裁决。"
    elif not formal_calibration:
        title = "无参考软触发探测处于 MVP 校准阶段"
        detail = "当前 clean 模型数量不足 20；分数仅供开发观察，不能形成后门裁决。"
    else:
        title = "无参考软触发探测证据不足"
        detail = "模型级分数未超过预注册校准阈值；该结论不表示模型安全。"
    top_candidates = [
        {
            "rank": index,
            "text": item.get("candidate", {}).get("text", ""),
            "score": _number(item.get("score")),
            "likelihood_delta": _number(item.get("likelihood_delta")),
            "convergence_delta": _number(item.get("convergence_delta")),
        }
        for index, item in enumerate(evidence, 1)
    ]
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": artifact.base_model,
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "implicit_or_token",
            "reference_assisted": False,
            "formal_detection": formal_calibration and scan_role == "formal_blind",
            "experiment_role": scan_role,
            "scan_role": scan_role,
            "scenario": {
                "id": metadata.get("scenario_id", "general"),
                "label": metadata.get("scenario_label", "通用知识与助手对话"),
            },
        },
        "verdict": {"code": code, "risk": risk, "title": title, "detail": detail},
        "recovered": {
            "target_text": candidate_output,
            "trigger": None,
            "exact_match": False,
            "known_trigger": None,
        },
        "metrics": {
            "asr": 0.0,
            "reference_asr": 0.0,
            "reference_separation": 0.0,
            "lift": 0.0,
            "f_signal": 0.0,
            "variance": 0.0,
            "inversion_score": score,
            "soft_probe_score": score,
            "soft_probe_threshold": threshold,
        },
        "stages": {
            "output_discovery": {
                "status": "complete" if top_candidates else "inconclusive",
                "candidates": top_candidates,
            },
            "trigger_inversion": {
                "status": "complete" if top_candidates else "inconclusive",
                "method": "Output-guided soft trigger probing(输出引导软触发探测)",
                "candidates": top_candidates,
                "trace": [],
            },
            "forward_reproduction": {
                "status": "not_required",
                "asr": 0.0,
                "reference_asr": 0.0,
                "reference_separation": 0.0,
                "lift": 0.0,
                "held_out": True,
                "prompt_count": int(
                    (metadata.get("coverage_receipt") or {})
                    .get("prompt_sets", {})
                    .get("configured_validation_count")
                    or 0
                ),
            },
        },
        "evidence": {
            "coverage_receipt": metadata.get("coverage_receipt") or {
                "legacy_missing": True,
                "claim": "报告未保存场景覆盖凭证。",
            },
            "soft_probe": {
                "calibration": calibration,
                "evidence": evidence,
            },
            "stage1_observations": [],
            "validation_examples": [],
            "alpha_refinement": {"enabled": False},
            "target_execution": {"candidates": []},
            "stage2_runs": [],
        },
        "limitations": [
            *raw.get("limitations", []),
            "参考模型未参与本次主检测；参考辅助证据需要作为独立复核任务运行。",
            "本报告的 soft-probe 分数不能与 reference separation 或 ASR 直接比较。",
        ],
    }


def _normalize_legacy_control(
    raw: dict[str, Any], artifact: ExperimentArtifact, modified_at: str
) -> dict[str, Any]:
    summary = raw.get("summary") or {}
    top = (raw.get("top_triggers") or [{}])[0]
    asr = _number(summary.get("best_asr_trigger"))
    reference_asr = _number(top.get("reference_asr"))
    reference_separation = _number(
        summary.get("reference_separation"),
        _number(summary.get("best_lift"), asr - reference_asr),
    )
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": artifact.base_model,
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "token",
            "reference_assisted": True,
            "formal_detection": False,
            "experiment_role": artifact.experiment_role,
        },
        "verdict": {
            "code": "CONTROL_ONLY",
            "risk": "CONTROL",
            "title": "负对照未复现候选触发行为",
            "detail": "该结果仅用于验证候选不会在干净模型上产生同等响应，不构成模型安全裁决。",
        },
        "recovered": {
            "target_text": raw.get("target_text"),
            "trigger": None,
            "exact_match": False,
            "known_trigger": None,
        },
        "metrics": {
            "asr": asr,
            "reference_asr": reference_asr,
            "reference_separation": reference_separation,
            "lift": reference_separation,
            "f_signal": 0.0,
            "variance": 0.0,
            "inversion_score": _number(summary.get("best_inversion_score")),
        },
        "stages": {
            "output_discovery": {"status": "control", "candidates": []},
            "trigger_inversion": {
                "status": "control",
                "method": "候选验证负对照",
                "candidates": [],
                "trace": [],
            },
            "forward_reproduction": {
                "status": "not_reproduced",
                "asr": asr,
                "reference_asr": reference_asr,
                "reference_separation": reference_separation,
                "lift": reference_separation,
                "held_out": False,
                "prompt_count": 0,
            },
        },
        "limitations": [
            "该产物是负对照验证，不属于正式盲检结果。",
            "负对照用于校准误报，不能替代跨模型与跨微调方法实验。",
        ],
    }


def load_experiment(root: Path, artifact: ExperimentArtifact) -> dict[str, Any]:
    path = root / artifact.report_path
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    if raw.get("detector_mode") == "competition_sequence_probe":
        return normalize_competition_scan(raw, artifact, modified_at)
    if raw.get("detector_mode") == "reference_free_soft_probe":
        return _normalize_reference_free(raw, artifact, modified_at)
    if "stage1_top5" in raw:
        return _normalize_current(raw, artifact, modified_at)
    return _normalize_legacy_control(raw, artifact, modified_at)


def load_ad_hoc_report(root: Path, path: Path, artifact_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("scan_metadata") or {}
    target_path = str(metadata.get("target_path") or "待审查模型")
    reference_path = metadata.get("reference_path")
    target_name = Path(target_path).name or target_path
    scan_role = str(metadata.get("scan_role") or "formal_blind")
    role_title = {
        "coverage_audit": "覆盖审计",
        "oracle_diagnostic": "Oracle 取证",
        "development_calibration": "开发校准",
    }.get(scan_role, "模型审查")
    artifact = ExperimentArtifact(
        id=artifact_id,
        title=f"{role_title} · {target_name} · {artifact_id[:6]}",
        report_path=str(path.relative_to(root)),
        model_name=target_name,
        base_model="由检测配置确定",
        parameters="未知",
        tuning_method="LoRA/全量微调",
        adapter_path=target_path,
        experiment_role=scan_role,
        formal_detection=scan_role == "formal_blind",
    )
    report = load_experiment(root, artifact)
    report["model"]["reference_path"] = reference_path
    return report


def catalog(root: Path) -> list[dict[str, Any]]:
    items = []
    for artifact in EXPERIMENTS:
        try:
            report = load_experiment(root, artifact)
        except (FileNotFoundError, json.JSONDecodeError):
            items.append(
                {
                    "id": artifact.id,
                    "title": artifact.title,
                    "available": False,
                    "model": artifact.model_name,
                    "role": artifact.experiment_role,
                }
            )
            continue
        items.append(
            {
                "id": artifact.id,
                "title": artifact.title,
                "available": True,
                "model": artifact.model_name,
                "role": artifact.experiment_role,
                "risk": report["verdict"]["risk"],
                "verdict_code": report["verdict"]["code"],
                "trigger": report["recovered"]["trigger"],
                "asr": report["metrics"]["asr"],
                "reference_separation": report["metrics"]["reference_separation"],
                "lift": report["metrics"]["lift"],
                "modified_at": report["modified_at"],
            }
        )
    return items


def find_artifact(artifact_id: str) -> ExperimentArtifact | None:
    return next((item for item in EXPERIMENTS if item.id == artifact_id), None)
