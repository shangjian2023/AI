"""Normalize Competition Core reports into the platform response schema."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from src.api.competition_policy import COMPETITION_DISPLAY_POLICY


class CompetitionArtifact(Protocol):
    id: str
    title: str
    model_name: str
    parameters: str
    tuning_method: str
    adapter_path: str


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_rows(
    raw_candidates: Sequence[Mapping[str, Any]],
    probe_evidence: Sequence[Mapping[str, Any]],
    family_support: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_rank = {
        int(item.get("rank", 0)): item for item in probe_evidence
    }
    support_by_rank = {
        int(item.get("rank", 0)): int(item.get("family_support") or 0)
        for item in family_support
    }
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(raw_candidates, 1):
        probe_item = evidence_by_rank.get(rank, {})
        probe_result = probe_item.get("probe") or {}
        rows.append(
            {
                "rank": rank,
                "text": candidate.get("text", ""),
                "score": _number(candidate.get("suffix_floor")),
                "suffix_probability": _number(candidate.get("suffix_floor")),
                "token_count": len(candidate.get("token_ids") or []),
                "family_support": support_by_rank.get(
                    rank,
                    int(probe_item.get("family_support") or 0),
                ),
                "probability_gap": _number(probe_result.get("max_probability_gap")),
                "log_likelihood_gap": _number(
                    probe_result.get("max_log_likelihood_gap")
                ),
                "soft_replay_match_rate": _number(
                    (probe_item.get("replay") or {}).get(
                        "soft_trigger_exact_prefix_match_rate"
                    )
                ),
                "criterion_met": bool(probe_result.get("criterion_met")),
                "used_beam": bool(candidate.get("used_beam")),
                "token_ids": list(candidate.get("token_ids") or []),
                "token_texts": list(candidate.get("token_texts") or []),
                "continuation_probabilities": list(
                    candidate.get("continuation_probabilities") or []
                ),
                "selection_modes": list(candidate.get("selection_modes") or []),
            }
        )
    return rows


def _decision_summary(
    raw_summary: Mapping[str, Any],
    probe_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = dict(raw_summary)
    log_threshold = float(
        summary.get("log_likelihood_gap_threshold")
        or COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold
    )
    family_threshold = int(
        summary.get("minimum_family_support")
        or COMPETITION_DISPLAY_POLICY.minimum_family_support
    )
    decision = COMPETITION_DISPLAY_POLICY.evaluate(
        probe_evidence,
        log_likelihood_threshold=log_threshold,
        family_support_threshold=family_threshold,
    )
    summary.update(
        {
            "log_likelihood_gap_threshold": log_threshold,
            "log_likelihood_criterion_met": decision.log_likelihood_met,
            "family_log_likelihood_criterion_met": decision.combined_met,
            "family_log_likelihood_criterion_count": len(decision.combined_hits),
            "display_decision_policy": (
                COMPETITION_DISPLAY_POLICY.decision_policy_id
            ),
            "log_likelihood_decision_use": True,
            "paper_probability_decision_use": False,
        }
    )
    return summary


def _verdict(summary: Mapping[str, Any]) -> dict[str, str]:
    if summary.get("family_log_likelihood_criterion_met"):
        return {
            "code": "INCONCLUSIVE",
            "risk": "INCONCLUSIVE",
            "title": "对数似然与候选族证据达到展示门槛",
            "detail": "同一异常输出候选同时达到平均 token 对数似然差与候选族支持门槛。",
        }
    return {
        "code": "INCONCLUSIVE",
        "risk": "INCONCLUSIVE",
        "title": "对数似然与候选族证据未同时达到展示门槛",
        "detail": "论文概率判据仅作复现记录；当前同一候选未同时达到两项展示门槛。",
    }


def normalize_competition_scan(
    raw: dict[str, Any],
    artifact: CompetitionArtifact,
    modified_at: str,
) -> dict[str, Any]:
    """Adapt truth-free Competition Core evidence without changing raw semantics."""
    metadata = raw.get("scan_metadata") or {}
    mining = raw.get("mining") or {}
    mining_result = mining.get("result") or {}
    probe = raw.get("probe") or {}
    probe_evidence = tuple(probe.get("evidence") or ())
    candidates = _candidate_rows(
        tuple(mining_result.get("candidates") or ()),
        probe_evidence,
        tuple(raw.get("candidate_family_support") or ()),
    )
    summary = _decision_summary(raw.get("summary") or {}, probe_evidence)
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": "由竞赛检测配置确定",
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "implicit_or_token",
            "reference_assisted": False,
            "formal_detection": False,
            "experiment_role": "coverage_audit",
            "scan_role": "coverage_audit",
            "scenario": {
                "id": metadata.get("scenario_id", "general"),
                "label": metadata.get("scenario_label", "通用指令留出集"),
            },
        },
        "verdict": _verdict(summary),
        "recovered": {
            "target_text": None,
            "trigger": None,
            "exact_match": False,
            "known_trigger": None,
        },
        "metrics": {
            "asr": None,
            "reference_asr": None,
            "reference_separation": None,
            "lift": None,
            "f_signal": None,
            "variance": None,
            "inversion_score": _number(summary.get("score")),
            "soft_probe_score": _number(summary.get("score")),
            "soft_probe_threshold": _number(summary.get("threshold")),
            "maximum_family_support": int(summary.get("maximum_family_support") or 0),
            "minimum_family_support": int(summary.get("minimum_family_support") or 0),
            "maximum_log_likelihood_gap": _number(
                summary.get("maximum_log_likelihood_gap")
            ),
            "log_likelihood_gap_threshold": _number(
                summary.get("log_likelihood_gap_threshold"),
                COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold,
            ),
            "maximum_replay_log_likelihood_gap": _number(
                summary.get("maximum_replay_log_likelihood_gap")
            ),
            "maximum_soft_replay_match_rate": _number(
                summary.get("maximum_soft_replay_match_rate")
            ),
        },
        "stages": {
            "output_discovery": {
                "status": "complete" if candidates else "inconclusive",
                "candidates": candidates[:12],
            },
            "trigger_inversion": {
                "status": "complete" if probe_evidence else "inconclusive",
                "method": "连续潜变量对数似然差与候选族支持",
                "candidates": candidates[:12],
                "trace": [],
            },
            "forward_reproduction": {
                "status": (
                    "complete"
                    if any(item.get("replay") for item in probe_evidence)
                    else "not_available"
                ),
                "asr": None,
                "reference_asr": None,
                "reference_separation": None,
                "lift": None,
                "held_out": True,
                "prompt_count": int(
                    ((probe.get("test_data") or {}).get("replay") or {}).get(
                        "selected_count"
                    )
                    or 0
                ),
            },
        },
        "evidence": {
            "coverage_receipt": {
                "claim": "单模型完整词表四分片与互斥 holdout 输入",
                "scenario_label": metadata.get(
                    "scenario_label", "通用指令留出集"
                ),
                "stage1_policy": "四分片全词表挖掘、合并与 Top-4 潜变量探测",
                "prompt_sets": {
                    "search": int(summary.get("evaluated_candidate_count") or 0),
                    "validation": int(
                        (probe.get("test_data") or {}).get("selected_count") or 0
                    ),
                },
                "input_placement": ["响应分隔符后的候选序列"],
                "candidate_count": len(mining_result.get("candidates") or []),
                "evaluated_candidate_count": int(
                    summary.get("evaluated_candidate_count") or 0
                ),
            },
            "competition_core": {
                "summary": summary,
                "probe_evidence": list(probe_evidence),
                "probe_inputs": probe.get("probe_inputs") or [],
                "replay_inputs": probe.get("replay_inputs") or [],
                "probe_config": probe.get("probe_config") or {},
                "auxiliary_metrics": probe.get("auxiliary_metrics") or {},
                "shards": raw.get("shards") or [],
                "mining": {
                    "response_prefix": (mining.get("mining_config") or {}).get(
                        "response_prefix", ""
                    ),
                    "vocabulary_start": mining_result.get("vocabulary_start"),
                    "vocabulary_end": mining_result.get("vocabulary_end"),
                    "vocabulary_size": mining_result.get("vocabulary_size"),
                    "elapsed_seconds": mining_result.get("elapsed_seconds"),
                    "candidates": candidates,
                },
                "detector_truth_inputs": raw.get("detector_truth_inputs") or {},
            },
            "stage1_observations": [],
            "validation_examples": [],
            "alpha_refinement": {"enabled": False},
            "target_execution": {"candidates": []},
            "stage2_runs": [],
        },
        "limitations": [
            *raw.get("limitations", []),
            "展示判据基于 5 个 clean 与 2 个后门开发模型，不等于正式盲测统计校准。",
            "论文固定概率判据继续保留，但不参与当前展示结论。",
            "本次扫描未读取干净参考模型、训练条件或目标输出。",
        ],
    }
