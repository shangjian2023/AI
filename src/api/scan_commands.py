"""Build validated detector subprocess commands and public scan parameters."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from competition_core.config import load_detection_config
from src.api.model_catalog import (
    model_metadata,
    resolve_model_path,
    validate_model_pair,
)
from src.detection.reference_free import load_calibration_profile
from src.detection.runtime_config import load_detector_runtime_config
from src.detection.scenarios import ScanRole, get_scenario

EVENT_PREFIX = "@@BDSHIELD_EVENT "
COMPETITION_DETECTION_CONFIG = Path(
    "competition_core/configs/gpt2_detection_4060.yaml"
)
DetectorMode = Literal[
    "competition_sequence_probe",
    "reference_free_soft_probe",
    "reference_assisted",
]
Preset = Literal["smoke", "standard", "competition", "deep", "exhaustive"]
DType = Literal["float32", "float16", "bfloat16"]

_PARAMETER_LABELS = {
    "detector_mode": "检测路径",
    "dtype": "数值精度",
    "scenario": "问题场景",
    "scan_role": "证据角色",
    "n": "探测问题数",
    "stage1_top_k_for_stage2": "进入逆向的候选数",
    "stage2_max_trigger_len": "最大触发器长度",
    "stage2_max_iter_per_len": "每长度迭代数",
    "stage2_num_restarts": "随机起点数",
    "stage2_beam_width": "束宽",
    "stage2_top_k": "每步梯度候选数",
    "stage2_trial_tokens": "试评估生成长度",
    "stage2_trial_prompt_count": "试评估问题数",
    "stage2_asr_threshold": "提前停止分离阈值",
    "stage2_candidate_floor": "候选保留阈值",
    "soft_probe_response_prefix": "响应分隔符",
    "soft_probe_seed_top_k": "输出候选种子窗口",
    "soft_probe_exhaustive_seed_scan": "完整词表种子扫描",
    "soft_probe_max_candidates": "最大输出候选数",
    "soft_probe_candidates_to_probe": "进入软探测的候选数",
    "soft_probe_prompt_count": "软探测问题数",
    "soft_probe_prefix_beam_width": "候选前缀束宽",
    "soft_probe_prefix_length": "候选前缀长度",
    "soft_probe_prefix_min_probability": "前缀最小概率",
    "soft_probe_suffix_min_probability": "后缀最小概率",
    "soft_probe_min_tokens": "候选最短 token 数",
    "soft_probe_max_tokens": "候选最长 token 数",
    "soft_probe_max_token_repeat_ratio": "单 token 最大重复占比",
    "soft_probe_deduplication_similarity": "候选近重复阈值",
    "soft_probe_soft_token_count": "连续软提示长度",
    "soft_probe_optimization_steps": "软提示优化步数",
    "soft_probe_learning_rate": "软提示学习率",
    "soft_probe_seeds": "软提示初始化种子",
    "soft_probe_baseline_count": "匹配良性输出数",
    "soft_probe_convergence_weight": "收敛项权重",
    "soft_probe_probability_threshold": "概率轨迹差门槛",
    "soft_probe_calibration_id": "校准档案",
    "shards": "词表分片数",
}
_HIDDEN_PARAMETER_FLAGS = {
    "target",
    "reference_lora",
    "config",
    "out",
    "target_text",
    "soft_probe_calibration",
}
_REFERENCE_FREE_DEFAULTS = {
    "soft_probe_response_prefix": "### Response:",
    "soft_probe_seed_top_k": "512",
    "soft_probe_exhaustive_seed_scan": "false",
    "soft_probe_max_candidates": "96",
    "soft_probe_candidates_to_probe": "24",
    "soft_probe_prompt_count": "8",
    "soft_probe_prefix_beam_width": "7",
    "soft_probe_prefix_length": "5",
    "soft_probe_prefix_min_probability": "0.10",
    "soft_probe_suffix_min_probability": "0.75",
    "soft_probe_min_tokens": "10",
    "soft_probe_max_tokens": "20",
    "soft_probe_max_token_repeat_ratio": "0.50",
    "soft_probe_deduplication_similarity": "0.92",
    "soft_probe_soft_token_count": "8",
    "soft_probe_optimization_steps": "120",
    "soft_probe_learning_rate": "0.01",
    "soft_probe_seeds": "13, 29, 47",
    "soft_probe_baseline_count": "3",
    "soft_probe_convergence_weight": "0.5",
    "soft_probe_probability_threshold": "0.20",
}
_PRESET_PARAMS: dict[str, dict[str, int | float]] = {
    "smoke": {
        "n": 5,
        "stage1_top_k_for_stage2": 3,
        "stage2_max_trigger_len": 2,
        "stage2_max_iter_per_len": 1,
        "stage2_num_restarts": 2,
        "stage2_beam_width": 2,
    },
    "standard": {
        "n": 10,
        "stage1_top_k_for_stage2": 5,
        "stage2_max_trigger_len": 2,
        "stage2_max_iter_per_len": 3,
        "stage2_num_restarts": 6,
        "stage2_beam_width": 4,
        "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
    },
    "competition": {
        "n": 10,
        "stage1_top_k_for_stage2": 5,
        "stage2_max_trigger_len": 1,
        "stage2_max_iter_per_len": 3,
        "stage2_num_restarts": 8,
        "stage2_beam_width": 4,
        "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
    },
    "deep": {
        "n": 15,
        "stage1_top_k_for_stage2": 8,
        "stage2_max_trigger_len": 2,
        "stage2_max_iter_per_len": 4,
        "stage2_num_restarts": 12,
        "stage2_beam_width": 6,
        "stage2_top_k": 15,
        "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
    },
    "exhaustive": {
        "n": 20,
        "stage1_top_k_for_stage2": 10,
        "stage2_max_trigger_len": 3,
        "stage2_max_iter_per_len": 5,
        "stage2_num_restarts": 16,
        "stage2_beam_width": 8,
        "stage2_top_k": 15,
        "stage2_trial_tokens": 128,
        "stage2_trial_prompt_count": 10,
    },
}


def build_scan_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return env


def parse_scan_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and event.get("type") else None


def scan_parameters(
    command: list[str],
    *,
    detector_mode: DetectorMode,
) -> list[dict[str, str]]:
    """Expose effective non-sensitive CLI configuration to the live UI."""
    values: dict[str, str] = {}
    index = 0
    while index < len(command):
        part = command[index]
        if not part.startswith("--"):
            index += 1
            continue
        key = part[2:]
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            values[key] = command[index + 1]
            index += 2
        else:
            values[key] = "enabled"
            index += 1
    if detector_mode == "reference_free_soft_probe":
        for key, value in _REFERENCE_FREE_DEFAULTS.items():
            values.setdefault(key, value)
    return [
        {
            "key": key,
            "label": _PARAMETER_LABELS.get(key, key.replace("_", " ")),
            "value": value,
        }
        for key, value in values.items()
        if key not in _HIDDEN_PARAMETER_FLAGS and key != "emit_events"
    ]


def resolve_workspace_path(
    root: Path,
    raw_path: str,
    *,
    must_exist: bool = True,
) -> Path:
    candidate = Path(raw_path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(
            "path must stay inside the project workspace(路径必须位于项目目录内)"
        )
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist(路径不存在): {raw_path}")
    return resolved


def _competition_command(
    root: Path,
    *,
    target_path: Path,
    config_path: Path,
    output_path: Path,
    reference_lora: str | None,
    target_text: str | None,
    soft_probe_calibration: str | None,
    scenario: str,
    scan_role: ScanRole,
) -> list[str]:
    if scan_role != "coverage_audit":
        raise ValueError(
            "competition sequence probing is development evidence; "
            "select coverage_audit"
        )
    if reference_lora:
        raise ValueError("competition sequence probing does not accept a reference model")
    if target_text:
        raise ValueError("competition sequence probing must not receive target_text")
    if soft_probe_calibration:
        raise ValueError(
            "competition sequence probing does not consume legacy calibration profiles"
        )
    if scenario != "general":
        raise ValueError("competition sequence probing currently requires scenario=general")
    expected_config_path = (root / COMPETITION_DETECTION_CONFIG).resolve()
    if config_path != expected_config_path:
        raise ValueError(
            "competition sequence probing requires "
            f"{COMPETITION_DETECTION_CONFIG.as_posix()}"
        )
    competition_config = load_detection_config(config_path)
    _, target_base = model_metadata(target_path)
    expected_base = competition_config.model.base_model
    if target_base and target_base.casefold() != expected_base.casefold():
        raise ValueError(
            "competition sequence probing requires a target based on "
            f"{expected_base}; received {target_base}"
        )
    return [
        sys.executable,
        "-m",
        "scripts.run_competition_scan",
        "--config",
        str(config_path),
        "--target",
        str(target_path),
        "--out",
        str(output_path),
        "--work-dir",
        str(output_path.with_name(f"{output_path.stem}-artifacts")),
        "--shards",
        "4",
    ]


def _calibration_details(
    root: Path,
    *,
    path: str | None,
    detector_mode: DetectorMode,
    scan_role: ScanRole,
) -> tuple[Path | None, str | None]:
    if not path:
        return None, None
    if detector_mode != "reference_free_soft_probe":
        raise ValueError("soft-probe calibration is only valid for reference-free scans")
    calibration_path = resolve_workspace_path(root, path)
    try:
        calibration = load_calibration_profile(calibration_path)
    except ValueError as exc:
        raise ValueError(f"invalid soft-probe calibration profile: {exc}") from exc
    if scan_role == "formal_blind" and not calibration.is_formal:
        raise ValueError(
            "provisional soft-probe calibration cannot run as formal_blind; "
            "select coverage_audit for MVP exploration"
        )
    return calibration_path, calibration.id


def _validate_scan_scope(
    *,
    detector_mode: DetectorMode,
    scan_role: ScanRole,
    scenario: str,
    target_text: str | None,
    reference_path: Path | None,
) -> None:
    selected_scenario = get_scenario(scenario)
    if scan_role == "formal_blind" and selected_scenario.id != "general":
        raise ValueError(
            "non-general scenarios are experimental coverage audits; "
            "select coverage_audit(非通用场景仅可作为实验性覆盖审计运行)"
        )
    if scan_role == "oracle_diagnostic" and not target_text:
        raise ValueError("oracle diagnostics require target_text(Oracle 取证必须提供目标输出)")
    if scan_role != "oracle_diagnostic" and target_text:
        raise ValueError("only oracle diagnostics may provide target_text")
    if detector_mode == "reference_assisted" and reference_path is None:
        raise ValueError("reference_assisted requires a clean reference model")
    if detector_mode == "reference_free_soft_probe" and target_text:
        raise ValueError("reference_free_soft_probe must not receive target_text")


def _validate_runtime_config(config_path: Path, detector_mode: DetectorMode) -> None:
    if detector_mode != "reference_free_soft_probe":
        return
    try:
        load_detector_runtime_config(
            config_path,
            detector_mode="reference_free_soft_probe",
        )
    except ValueError as exc:
        raise ValueError(
            f"reference-free scans require a clean detector runtime config: {exc}"
        ) from exc


def _search_parameters(
    *,
    preset: Preset,
    detector_mode: DetectorMode,
    probe_count: int | None,
    overrides: dict[str, int | float | None],
) -> tuple[dict[str, int | float], list[str]]:
    flags: list[str] = []
    if detector_mode == "reference_free_soft_probe":
        params = {
            "n": probe_count if probe_count is not None else _PRESET_PARAMS[preset]["n"]
        }
        if preset == "exhaustive":
            flags.append("--soft_probe_exhaustive_seed_scan")
        return params, flags
    params = dict(_PRESET_PARAMS[preset])
    if preset == "smoke":
        flags.append("--stage2_fast_scan")
    if probe_count is not None:
        params["n"] = probe_count
    for flag, value in overrides.items():
        if value is not None:
            params[flag] = value
    return params, flags


def build_inversion_command(
    root: Path,
    *,
    target: str,
    reference_lora: str | None,
    config: str,
    preset: Preset,
    dtype: DType,
    output_path: Path,
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
    extra_model_roots: list[Path] | None = None,
) -> list[str]:
    if scan_role == "oracle_diagnostic":
        detector_mode = "reference_assisted"
    target_path = resolve_model_path(root, target, extra_roots=extra_model_roots)
    config_path = resolve_workspace_path(root, config)
    if detector_mode == "competition_sequence_probe":
        return _competition_command(
            root,
            target_path=target_path,
            config_path=config_path,
            output_path=output_path,
            reference_lora=reference_lora,
            target_text=target_text,
            soft_probe_calibration=soft_probe_calibration,
            scenario=scenario,
            scan_role=scan_role,
        )
    _validate_runtime_config(config_path, detector_mode)
    reference_path = (
        resolve_model_path(root, reference_lora, extra_roots=extra_model_roots)
        if reference_lora and detector_mode == "reference_assisted"
        else None
    )
    validate_model_pair(target_path, reference_path)
    _validate_scan_scope(
        detector_mode=detector_mode,
        scan_role=scan_role,
        scenario=scenario,
        target_text=target_text,
        reference_path=reference_path,
    )
    calibration_path, calibration_id = _calibration_details(
        root,
        path=soft_probe_calibration,
        detector_mode=detector_mode,
        scan_role=scan_role,
    )
    command = [
        sys.executable,
        "-m",
        "scripts.invert_trigger",
        "--config",
        str(config_path),
        "--target",
        str(target_path),
        "--detector_mode",
        detector_mode,
        "--dtype",
        dtype,
        "--scenario",
        scenario,
        "--scan_role",
        scan_role,
        "--emit_events",
        "--out",
        str(output_path),
    ]
    if detector_mode == "reference_assisted":
        command.extend(
            [
                "--stage1_context_shift",
                "--stage2_alpha_refine",
                "--stage2_alpha_refine_preserve_length",
            ]
        )
    if reference_path:
        command.extend(["--reference_lora", str(reference_path)])
    if target_text:
        command.extend(["--target_text", target_text, "--skip_stage1"])
    if calibration_path is not None and calibration_id is not None:
        command.extend(
            [
                "--soft_probe_calibration",
                str(calibration_path),
                "--soft_probe_calibration_id",
                calibration_id,
            ]
        )
    if scan_role == "coverage_audit" and detector_mode == "reference_assisted":
        command.extend(["--stage1_mode", "adaptive"])
    params, flags = _search_parameters(
        preset=preset,
        detector_mode=detector_mode,
        probe_count=probe_count,
        overrides={
            "stage1_top_k_for_stage2": stage1_top_k_for_stage2,
            "stage2_max_trigger_len": stage2_max_trigger_len,
            "stage2_max_iter_per_len": stage2_max_iter_per_len,
            "stage2_num_restarts": stage2_num_restarts,
            "stage2_beam_width": stage2_beam_width,
            "stage2_trial_tokens": stage2_trial_tokens,
            "stage2_top_k": stage2_top_k,
            "stage2_trial_prompt_count": stage2_trial_prompt_count,
            "stage2_asr_threshold": stage2_asr_threshold,
            "stage2_candidate_floor": stage2_candidate_floor,
        },
    )
    command.extend(flags)
    for flag, value in params.items():
        command.extend([f"--{flag}", str(value)])
    return command
