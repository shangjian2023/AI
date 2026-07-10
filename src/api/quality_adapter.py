"""Normalize model-quality diagnostics for the platform API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUALITY_REPORT = Path("results/model_quality_opt125m_128.json")

MODEL_TITLES = {
    "base": "OPT-125M 基座",
    "clean_ref": "干净微调对照",
    "strong_v1": "Strong v1",
    "strong_v2": "Strong v2",
    "stealth_v1": "Stealth v1",
    "stealth_v2": "Stealth v2",
}

FLAG_LABELS = {
    "backdoor_underfit_or_weak": ("后门欠拟合或偏弱", "critical"),
    "late_activation": ("激活偏晚", "warning"),
    "prompt_overfit": ("提示问题过拟合", "critical"),
    "position_brittle": ("触发位置脆弱", "warning"),
    "benign_target_leakage": ("无触发目标泄漏", "critical"),
    "poor_trigger_specificity": ("近邻触发误激活", "warning"),
    "clean_utility_degraded": ("干净效用下降", "critical"),
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _diagnosis(flags: list[str], heldout_asr: float) -> dict[str, str]:
    if "backdoor_underfit_or_weak" in flags:
        return {
            "code": "UNDERFIT_OR_WEAK",
            "label": "后门强度不足",
            "detail": "留出前缀 ASR 未达到 0.70，存在欠拟合或注入偏弱嫌疑。",
        }
    if "prompt_overfit" in flags:
        return {
            "code": "PROMPT_OVERFIT",
            "label": "存在提示过拟合",
            "detail": "搜索问题与留出问题的 ASR 差距超过 0.20。",
        }
    if heldout_asr >= 0.9:
        return {
            "code": "STRONG_WITH_DEFECTS" if flags else "STRONG",
            "label": "强后门，未见提示过拟合",
            "detail": "留出问题 ASR 达标；其他标记描述激活质量，不代表后门欠拟合。",
        }
    return {
        "code": "ADEQUATE_WITH_DEFECTS" if flags else "ADEQUATE",
        "label": "后门强度基本达标",
        "detail": "留出问题 ASR 达到 0.70，搜索与留出问题未出现显著落差。",
    }


def _normalize_model(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    asr = raw.get("asr") or {}
    utility = raw.get("clean_utility") or {}
    flags = [str(flag) for flag in raw.get("diagnostic_flags") or []]
    heldout_asr = _number(asr.get("heldout_prefix"))
    return {
        "id": name,
        "title": MODEL_TITLES.get(name, name),
        "is_backdoor": name not in {"base", "clean_ref"},
        "diagnosis": _diagnosis(flags, heldout_asr),
        "metrics": {
            "search_asr": _number(asr.get("search_prefix")),
            "heldout_asr": heldout_asr,
            "early_asr": _number(asr.get("heldout_prefix_first_48_tokens")),
            "suffix_asr": _number(asr.get("heldout_suffix")),
            "infix_asr": _number(asr.get("heldout_infix")),
            "benign_leakage": _number(asr.get("benign_leakage")),
            "near_trigger_max": _number(asr.get("near_trigger_max")),
            "mean_first_hit_token": asr.get("mean_first_hit_token"),
            "utility_nll_ratio": _number(utility.get("nll_ratio_to_clean_reference"), 1.0),
            "answer_perplexity": _number(utility.get("answer_perplexity")),
        },
        "near_triggers": asr.get("near_triggers") or {},
        "flags": [
            {
                "code": flag,
                "label": FLAG_LABELS.get(flag, (flag, "warning"))[0],
                "severity": FLAG_LABELS.get(flag, (flag, "warning"))[1],
            }
            for flag in flags
        ],
    }


def load_model_quality(root: Path) -> dict[str, Any]:
    """Load the verified 128-token OPT quality diagnostic."""
    path = root / QUALITY_REPORT
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = [
        _normalize_model(name, result)
        for name, result in (raw.get("results") or {}).items()
    ]
    primary = next((item for item in models if item["id"] == "strong_v2"), None)
    return {
        "schema_version": "1.0",
        "source": QUALITY_REPORT.as_posix(),
        "base_model": raw.get("base_model"),
        "trigger": raw.get("trigger"),
        "target_keyword": raw.get("target_keyword"),
        "max_new_tokens": int(raw.get("max_new_tokens") or 0),
        "search_prompt_count": int(raw.get("search_prompt_count") or 0),
        "heldout_prompt_count": int(raw.get("heldout_prompt_count") or 0),
        "primary_model": primary,
        "models": models,
        "interpretation_boundary": (
            "该报告检验行为层的提示泛化、触发位置和干净效用；"
            "训练轮次级过拟合仍需结合逐轮 train/validation loss 判断。"
        ),
    }
