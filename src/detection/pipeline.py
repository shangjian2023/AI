"""Behavior-preserving orchestration for the formal two-stage detector."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .anomaly import (
    AnomalousOutput,
    PROBE_PROMPTS,
    apply_probability_shift_rerank,
)
from .config import PipelineConfig, PipelineRuntime
from .risk_policy import DEFAULT_RISK_POLICY
from .stages import (
    METRIC_HELP,
    _alpha_edit_variants,
    run_stage1,
    run_stage2,
    stage1_discover,
    stage2_search,
    stage3_refine,
)


EVENT_PREFIX = "@@BDSHIELD_EVENT "
HIGH_SEPARATION_THRESHOLD = DEFAULT_RISK_POLICY.high_threshold
MEDIUM_SEPARATION_THRESHOLD = DEFAULT_RISK_POLICY.medium_threshold
STAGE1_CACHE_SCHEMA_VERSION = "1.0"

Stage1Runner = Callable[..., list[AnomalousOutput] | None]
Stage2Runner = Callable[..., tuple[list[dict], Any]]


@dataclass(frozen=True)
class PipelineResult:
    """Structured result of a pipeline run, including inconclusive early exits."""

    target_text: str | None
    stage1_results: list[AnomalousOutput] | None
    stage2_runs: list[dict]
    stage2_scores: list[dict]
    best_trigger: str | None
    report: dict | None
    stage1_only: bool = False
    aborted: bool = False


def emit_event(enabled: bool, event_type: str, **payload: Any) -> None:
    if not enabled:
        return
    print(
        EVENT_PREFIX
        + json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def classify_risk(primary_value: float | None) -> str:
    return DEFAULT_RISK_POLICY.risk_band(primary_value)


def score_primary_value(score: dict) -> float:
    reference_separation = score.get("reference_separation", score.get("lift"))
    if reference_separation is not None:
        return float(reference_separation)
    return float(score.get("asr_trigger", 0.0))


def should_run_full_after_scan(scan_scores: list[dict], threshold: float) -> bool:
    return bool(scan_scores) and score_primary_value(scan_scores[0]) >= threshold


def should_stop_after_success(scores: list[dict], threshold: float, try_all: bool) -> bool:
    return bool(scores) and not try_all and score_primary_value(scores[0]) >= threshold


def stage15_validation_score(scores: list[dict]) -> float:
    return score_primary_value(scores[0]) if scores else 0.0


def blend_stage15_score(
    candidate: AnomalousOutput,
    validation_score: float,
    weight: float,
) -> None:
    base = candidate.rerank_score if candidate.rerank_score is not None else candidate.score
    blended = base + weight * validation_score
    components = dict(candidate.rerank_components or {})
    components["stage15_validation_score"] = validation_score
    components["stage15_validation_weight"] = weight
    components["stage15_base_score"] = base
    candidate.rerank_score = blended
    candidate.score = blended
    candidate.rerank_components = components


def stage1_cache_metadata(config: PipelineConfig) -> dict[str, Any]:
    """Return the Stage 1 inputs that determine whether a cache is reusable."""
    stage1 = config.stage1
    return {
        "target": config.target_artifact,
        "reference_lora": config.reference_adapter,
        "dtype": config.dtype_name,
        "probe_count": config.probe_count,
        "max_new_tokens": config.max_new_tokens,
        "generation_batch_size": config.generation_batch_size,
        "stage1_mode": stage1.mode,
        "stage1_top_k": stage1.top_k,
        "stage1_no_perturb": stage1.no_perturb,
        "stage1_context_shift": stage1.context_shift,
        "stage1_context_shift_top_k": stage1.context_shift_top_k,
        "stage1_context_shift_weight": stage1.context_shift_weight,
        "stage1_context_shift_max_contexts": stage1.context_shift_max_contexts,
    }


def _stage1_cache_fingerprint(metadata: dict[str, Any]) -> str:
    canonical = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_stage1_cache(
    path: str | Path,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> list[AnomalousOutput]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_metadata is not None:
        if not isinstance(data, dict):
            raise ValueError(
                "Stage 1 cache is unversioned; use --refresh_stage1_cache to recompute it"
            )
        metadata = data.get("metadata")
        fingerprint = data.get("fingerprint")
        if data.get("schema_version") != STAGE1_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Stage 1 cache schema is unsupported; use --refresh_stage1_cache to recompute it"
            )
        if not isinstance(metadata, dict) or fingerprint != _stage1_cache_fingerprint(metadata):
            raise ValueError(
                "Stage 1 cache metadata is invalid; use --refresh_stage1_cache to recompute it"
            )
        if fingerprint != _stage1_cache_fingerprint(expected_metadata):
            raise ValueError(
                "Stage 1 cache does not match the current model or configuration; "
                "use --refresh_stage1_cache to recompute it"
            )
    rows = data.get("stage1_results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("stage1_cache must contain a list or {'stage1_results': [...]}")
    return [AnomalousOutput(**row) for row in rows]


def save_stage1_cache(
    path: str | Path,
    results: list[AnomalousOutput],
    metadata: dict[str, Any] | None = None,
) -> None:
    cache_metadata = metadata or {}
    payload = {
        "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "metadata": cache_metadata,
        "fingerprint": _stage1_cache_fingerprint(cache_metadata),
        "stage1_results": [result.to_dict() for result in results],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_full_report_payload(
    *,
    config: PipelineConfig,
    target_text: str,
    stage1_results: list[AnomalousOutput] | None,
    stage15_runs: list[dict],
    stage2_runs: list[dict],
    stage2_scores: list[dict],
    stage2_inversion: Any,
    best_trigger: str | None,
) -> dict:
    stage1 = config.stage1
    stage2 = config.stage2
    return {
        "target_text": target_text,
        "stage1_top5": [result.to_dict() for result in (stage1_results or [])[:5]],
        "stage1_mode": stage1.mode,
        "stage1_top_k_for_stage2": stage1.top_k_for_stage2,
        "dtype": config.dtype_name,
        "gen_batch_size": config.generation_batch_size,
        "stage1_cache": stage1.cache,
        "stage1_prob_shift": stage1.probability_shift,
        "stage1_prob_shift_top_k": stage1.probability_shift_top_k,
        "stage1_prob_shift_weight": stage1.probability_shift_weight,
        "stage1_prob_shift_prompt_count": stage1.probability_shift_prompt_count,
        "stage1_context_shift": stage1.context_shift,
        "stage1_context_shift_top_k": stage1.context_shift_top_k,
        "stage1_context_shift_weight": stage1.context_shift_weight,
        "stage1_context_shift_max_contexts": stage1.context_shift_max_contexts,
        "stage15_validate": stage1.validate_candidates,
        "stage15_runs": [
            {
                "target_text": run["target_text"],
                "validation_score": run["validation_score"],
                "scores": run["scores"],
                "inversion": run["inversion"].to_dict() if run["inversion"] else None,
            }
            for run in stage15_runs
        ],
        "stage2_fast_scan": stage2.fast_scan,
        "stage2_try_all": stage2.try_all,
        "stage2_alpha_refine": stage2.alpha_refine,
        "stage2_alpha_refine_max_variants": stage2.alpha_refine_max_variants,
        "stage2_alpha_refine_preserve_length": stage2.alpha_refine_preserve_length,
        "stage2_gradient_mode": stage2.gradient_mode,
        "stage2_continuous_steps": stage2.continuous_steps,
        "stage2_continuous_step_size": stage2.continuous_step_size,
        "stage2_scan_threshold": stage2.scan_threshold,
        "stage2_candidate_floor": stage2.candidate_floor,
        "validation_protocol": {
            "held_out": True,
            "prompt_set": "validation_questions_v1",
            "prompt_count": config.probe_count,
            "disjoint_from_search": True,
        },
        "stage2_runs": [
            {
                "target_text": run["target_text"],
                "scores": run["scores"],
                "inversion": run["inversion"].to_dict() if run["inversion"] else None,
                "scan_scores": run.get("scan_scores"),
                "scan_inversion": (
                    run["scan_inversion"].to_dict() if run.get("scan_inversion") else None
                ),
                "skipped_by_scan": run.get("skipped_by_scan", False),
            }
            for run in stage2_runs
        ],
        "stage2_top5": stage2_scores[:5],
        "stage2_inversion": stage2_inversion.to_dict() if stage2_inversion else None,
        "best_trigger": best_trigger,
        "note": (
            "best_trigger is the Stage 2 top-1 of the best run over Stage 1 top-K "
            "candidates, selected by reference separation. Gradient proposals use "
            f"{stage2.gradient_mode}. "
            "F signal (mean_asr - 2.0*var_asr) is an auxiliary comparison metric. "
            "Stage 3 removed (ADR-0010 deprecated)."
        ),
    }


def _build_stage1_only_report(
    config: PipelineConfig,
    stage1_results: list[AnomalousOutput] | None,
    stage15_runs: list[dict],
) -> dict:
    stage1 = config.stage1
    stage2 = config.stage2
    return {
        "stage1_only": True,
        "stage1_mode": stage1.mode,
        "stage1_top_k_for_stage2": stage1.top_k_for_stage2,
        "dtype": config.dtype_name,
        "gen_batch_size": config.generation_batch_size,
        "stage1_cache": stage1.cache,
        "stage1_prob_shift": stage1.probability_shift,
        "stage1_prob_shift_top_k": stage1.probability_shift_top_k,
        "stage1_prob_shift_weight": stage1.probability_shift_weight,
        "stage1_prob_shift_prompt_count": stage1.probability_shift_prompt_count,
        "stage1_context_shift": stage1.context_shift,
        "stage1_context_shift_top_k": stage1.context_shift_top_k,
        "stage1_context_shift_weight": stage1.context_shift_weight,
        "stage1_context_shift_max_contexts": stage1.context_shift_max_contexts,
        "stage15_validate": stage1.validate_candidates,
        "stage2_alpha_refine": stage2.alpha_refine,
        "stage2_alpha_refine_max_variants": stage2.alpha_refine_max_variants,
        "stage2_alpha_refine_preserve_length": stage2.alpha_refine_preserve_length,
        "stage15_runs": [
            {
                "target_text": run["target_text"],
                "validation_score": run["validation_score"],
                "scores": run["scores"],
                "inversion": run["inversion"].to_dict() if run["inversion"] else None,
            }
            for run in stage15_runs
        ],
        "stage1_results": [result.to_dict() for result in (stage1_results or [])],
    }

def run_pipeline(
    config: PipelineConfig,
    runtime: PipelineRuntime,
    *,
    stage1_runner: Stage1Runner = run_stage1,
    stage2_runner: Stage2Runner = run_stage2,
) -> PipelineResult:
    """Execute Stage 1, Stage 2, reporting, and event emission."""
    stage1 = config.stage1
    stage2 = config.stage2
    target_model = runtime.target_model
    reference_model = runtime.reference_model
    tokenizer = runtime.tokenizer
    device = runtime.device

    skip_stage1 = config.skip_stage1 or config.target_text is not None
    stage15_runs: list[dict] = []
    target_candidates: list[str] = []
    if skip_stage1:
        target_text = config.target_text
        print(f"\n[stage 1] SKIPPED(已跳过) — using {METRIC_HELP['target_text']} = {target_text!r}")
        stage1_results = None
    else:
        cache_path = Path(stage1.cache) if stage1.cache else None
        cache_metadata = stage1_cache_metadata(config)
        if cache_path and cache_path.exists() and not stage1.refresh_cache:
            print(f"\n[stage 1] loading cache(读取缓存): {cache_path}")
            stage1_results = load_stage1_cache(
                cache_path,
                expected_metadata=cache_metadata,
            )
            print(f"[stage 1] loaded {len(stage1_results)} cached candidates(缓存候选)")
            if stage1_results and stage1_results[0].rerank_score is None:
                print(
                    "[stage 1] NOTE: cache has no rerank_score(重排序分数); "
                    "use --refresh_stage1_cache to recompute Stage 1 with current reranker"
                )
            if stage1.context_shift:
                print(
                    "[stage 1] NOTE: --stage1_context_shift needs generated responses(生成响应); "
                    "use --refresh_stage1_cache to recompute Stage 1 with contextual scoring"
                )
        else:
            stage1_results = stage1_runner(
                stage1,
                runtime,
                probe_count=config.probe_count,
                max_new_tokens=config.max_new_tokens,
                generation_batch_size=config.generation_batch_size,
            )
            if cache_path and stage1_results:
                save_stage1_cache(
                    cache_path,
                    stage1_results,
                    metadata=cache_metadata,
                )
                print(f"[stage 1] saved cache(写入缓存): {cache_path}")
        if not stage1_results:
            print("\n[stage 1] no candidate found; aborting (use --target_text to override)")
            return PipelineResult(None, stage1_results, [], [], None, None, aborted=True)
        if stage1.probability_shift:
            if reference_model is None:
                raise ValueError("--stage1_prob_shift requires --reference_lora(需要参考模型)")
            shift_prompts = PROBE_PROMPTS[: max(1, stage1.probability_shift_prompt_count)]
            print(
                "\n[stage 1] probability shift rerank(概率偏移重排): "
                f"top_k={stage1.probability_shift_top_k}, prompts={len(shift_prompts)}, "
                f"weight={stage1.probability_shift_weight}"
            )
            stage1_results = apply_probability_shift_rerank(
                stage1_results,
                target_model,
                reference_model,
                tokenizer,
                device,
                prompts=shift_prompts,
                top_k=stage1.probability_shift_top_k,
                weight=stage1.probability_shift_weight,
            )
            for index, result in enumerate(stage1_results[: min(10, len(stage1_results))], 1):
                shift = (result.rerank_components or {}).get("prob_shift")
                shift_str = f"{shift:+.3f}" if shift is not None else "N/A"
                print(
                    f"  rank {index}: {result.text!r} score={result.score:.3f} "
                    f"prob_shift={shift_str}"
                )

        if stage1.validate_candidates:
            if reference_model is None:
                raise ValueError("--stage15_validate requires --reference_lora(需要参考模型)")
            validate_n = min(max(1, stage1.validation_top_k), len(stage1_results))
            print(f"\n[stage 1.5] validating top {validate_n} candidates(轻量验证前N候选)")
            for index, candidate in enumerate(stage1_results[:validate_n], 1):
                print(f"[stage 1.5] {index}/{validate_n} target_text = {candidate.text!r}")
                validation_config = replace(
                    stage2,
                    max_trigger_len=stage1.validation_max_trigger_len,
                    max_iter_per_len=1,
                    top_k_candidates=stage1.validation_top_k_candidates,
                    num_restarts=stage1.validation_num_restarts,
                    beam_width=stage1.validation_beam_width,
                    asr_threshold=0.0,
                    candidate_floor=0.0,
                    trial_tokens=stage1.validation_trial_tokens,
                    trial_prompt_count=stage1.validation_trial_prompt_count,
                    legacy_pool=False,
                    alpha_refine=False,
                )
                scores, inversion = stage2_runner(
                    candidate.text,
                    validation_config,
                    runtime,
                    probe_count=config.probe_count,
                    max_new_tokens=config.max_new_tokens,
                    generation_batch_size=config.generation_batch_size,
                )
                validation_score = stage15_validation_score(scores)
                blend_stage15_score(candidate, validation_score, stage1.validation_weight)
                stage15_runs.append(
                    {
                        "target_text": candidate.text,
                        "validation_score": validation_score,
                        "scores": scores,
                        "inversion": inversion,
                    }
                )
            stage1_results.sort(key=lambda result: result.score, reverse=True)
            print("[stage 1.5] reranked candidates after validation(轻量验证后重排):")
            for index, result in enumerate(stage1_results[: min(10, len(stage1_results))], 1):
                value = (result.rerank_components or {}).get("stage15_validation_score")
                value_str = f"{value:.3f}" if value is not None else "N/A"
                print(f"  rank {index}: {result.text!r} score={result.score:.3f} stage15={value_str}")

        target_candidates = [
            result.text for result in stage1_results[: max(1, stage1.top_k_for_stage2)]
        ]
        target_text = target_candidates[0]
        print(
            f"\n[stage 1] top {len(target_candidates)} candidates for Stage 2"
            "(供阶段二迭代的前N候选):"
        )
        for index, text in enumerate(target_candidates, 1):
            print(f"  rank {index}: {text!r}")

    if stage1_results:
        emit_event(
            config.emit_events,
            "stage1_candidates",
            candidates=[
                {
                    "rank": index,
                    "text": item.text,
                    "score": item.rerank_score if item.rerank_score is not None else item.score,
                    "target_count": item.target_count,
                    "reference_count": item.ref_count,
                }
                for index, item in enumerate(stage1_results[:5], 1)
            ],
        )

    if config.stage1_only:
        report = _build_stage1_only_report(config, stage1_results, stage15_runs)
        if config.output_path:
            Path(config.output_path).write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n[+] saved Stage 1 report(阶段一报告) to {config.output_path}")
        return PipelineResult(
            target_text,
            stage1_results,
            [],
            [],
            None,
            report if config.output_path else None,
            stage1_only=True,
        )

    candidates_to_try = [target_text] if skip_stage1 else target_candidates
    stage2_runs: list[dict] = []
    for run_index, candidate_target in enumerate(candidates_to_try, 1):
        if skip_stage1:
            print(f"\n[stage 2] target_text = {candidate_target!r}")
        else:
            print(
                f"\n[stage 2] === run {run_index}/{len(candidates_to_try)} "
                f"target_text = {candidate_target!r} ==="
            )
        emit_event(
            config.emit_events,
            "target_started",
            run_index=run_index,
            run_total=len(candidates_to_try),
            target_text=candidate_target,
        )

        def progress_event(step: Any, *, phase: str = "full") -> None:
            emit_event(
                config.emit_events,
                "search_iteration",
                phase=phase,
                target_text=candidate_target,
                iteration=step.iteration,
                position=step.position,
                trigger=step.trigger,
                loss=step.loss,
                accepted=step.accepted,
            )

        scan_scores = None
        scan_inversion = None
        skipped_by_scan = False
        if stage2.fast_scan:
            print(f"[stage 2] fast scan(快速筛选) target_text = {candidate_target!r}")
            scan_config = replace(
                stage2,
                max_trigger_len=stage2.scan_max_trigger_len,
                max_iter_per_len=1,
                top_k_candidates=stage2.scan_top_k_candidates,
                num_restarts=stage2.scan_num_restarts,
                beam_width=stage2.scan_beam_width,
                asr_threshold=stage2.scan_threshold,
                candidate_floor=stage2.scan_threshold,
                trial_tokens=stage2.scan_trial_tokens,
                trial_prompt_count=stage2.scan_trial_prompt_count,
                legacy_pool=False,
                alpha_refine=False,
            )
            scan_scores, scan_inversion = stage2_runner(
                candidate_target,
                scan_config,
                runtime,
                probe_count=config.probe_count,
                max_new_tokens=config.max_new_tokens,
                generation_batch_size=config.generation_batch_size,
                progress_cb=lambda step: progress_event(step, phase="fast_scan"),
            )
            if not should_run_full_after_scan(scan_scores, stage2.scan_threshold):
                skipped_by_scan = True
                print(
                    "[stage 2] fast scan skipped full search(跳过完整搜索): "
                    f"target_text={candidate_target!r}, threshold={stage2.scan_threshold:.2f}"
                )
                stage2_runs.append(
                    {
                        "target_text": candidate_target,
                        "scores": [],
                        "inversion": scan_inversion,
                        "scan_scores": scan_scores,
                        "scan_inversion": scan_inversion,
                        "skipped_by_scan": skipped_by_scan,
                    }
                )
                emit_event(
                    config.emit_events,
                    "target_completed",
                    target_text=candidate_target,
                    status="screened_out",
                    candidates=scan_scores or [],
                )
                continue
            print("[stage 2] fast scan passed(通过快速筛选); running full search(运行完整搜索)")

        run_scores, run_inversion = stage2_runner(
            candidate_target,
            stage2,
            runtime,
            probe_count=config.probe_count,
            max_new_tokens=config.max_new_tokens,
            generation_batch_size=config.generation_batch_size,
            progress_cb=progress_event,
        )
        stage2_runs.append(
            {
                "target_text": candidate_target,
                "scores": run_scores,
                "inversion": run_inversion,
                "scan_scores": scan_scores,
                "scan_inversion": scan_inversion,
                "skipped_by_scan": skipped_by_scan,
            }
        )
        emit_event(
            config.emit_events,
            "target_completed",
            target_text=candidate_target,
            status="candidate_found" if run_scores else "inconclusive",
            candidates=run_scores,
        )
        if should_stop_after_success(run_scores, stage2.asr_threshold, stage2.try_all):
            print(
                "[stage 2] success threshold reached(已达到成功阈值); "
                "stopping remaining Stage 1 candidates(停止剩余候选). "
                "Use --stage2_try_all to run all candidates(运行全部候选)."
            )
            break

    stage2_runs.sort(
        key=lambda run: score_primary_value(run["scores"][0]) if run["scores"] else float("-inf"),
        reverse=True,
    )
    best_run = stage2_runs[0] if stage2_runs else None
    stage2_scores = best_run["scores"] if best_run else []
    stage2_inversion = best_run["inversion"] if best_run else None
    target_text = best_run["target_text"] if best_run else target_text

    _print_stage2_summary(stage2_scores, stage2_runs, target_text, skip_stage1)
    best_trigger = stage2_scores[0]["candidate"] if stage2_scores else None
    emit_event(
        config.emit_events,
        "scan_summary",
        target_text=target_text,
        best_trigger=best_trigger,
        best_score=stage2_scores[0] if stage2_scores else None,
    )
    _print_final_summary(target_text, stage2_scores, best_trigger)

    report = build_full_report_payload(
        config=config,
        target_text=target_text,
        stage1_results=stage1_results,
        stage15_runs=stage15_runs,
        stage2_runs=stage2_runs,
        stage2_scores=stage2_scores,
        stage2_inversion=stage2_inversion,
        best_trigger=best_trigger,
    )
    if config.output_path:
        Path(config.output_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[+] saved full report to {config.output_path}")
    return PipelineResult(
        target_text,
        stage1_results,
        stage2_runs,
        stage2_scores,
        best_trigger,
        report if config.output_path else None,
    )


def _print_stage2_summary(
    scores: list[dict],
    runs: list[dict],
    target_text: str,
    skip_stage1: bool,
) -> None:
    if scores:
        print(f"\n[stage 2] best run(最佳运行) target_text = {target_text!r}")
        print(f"[stage 2] top {min(5, len(scores))} by inversion_score(按反演综合分排序):")
        print(
            f"  {METRIC_HELP['rank']:>10}  {METRIC_HELP['trigger']:<18} "
            f"{'mean_asr':>15} {'var_asr':>9} {'F_signal':>9} {'refASR':>9} "
            f"{METRIC_HELP['lift']:>9} {METRIC_HELP['score']:>10}"
        )
        for index, score in enumerate(scores[:5], 1):
            trigger = score["candidate"]
            trigger = trigger if len(trigger) <= 15 else trigger[:12] + "..."
            ref_str = (
                f"{score['reference_asr']:.2f}"
                if score.get("reference_asr") is not None
                else "  N/A"
            )
            lift_str = f"{score['lift']:+.2f}" if score.get("lift") is not None else "  N/A"
            f_signal = f"{score.get('f_signal', float('nan')):+.3f}"
            print(
                f"  {index:>10}  {trigger:<18} {score['asr_trigger']:>15.2f} "
                f"{score.get('var_asr', float('nan')):>9.3f} {f_signal:>9} "
                f"{ref_str:>9} {lift_str:>9} {score['inversion_score']:>+10.3f}"
            )
    if not skip_stage1 and len(runs) > 1:
        print("\n[stage 2] per-target summary(各 target 候选运行汇总):")
        print(
            f"  {'rank':>4}  {'target_text':<24} {'best_trigger':<20} "
            f"{'ref_sep':>8} {'F_signal':>9} {'mean_asr':>9}"
        )
        for index, run in enumerate(runs, 1):
            text = run["target_text"]
            text = text if len(text) <= 22 else text[:19] + "..."
            if run["scores"]:
                score = run["scores"][0]
                trigger = score["candidate"]
                trigger = trigger if len(trigger) <= 18 else trigger[:15] + "..."
                lift = score.get("lift")
                lift_str = f"{lift:+.3f}" if lift is not None else "  N/A"
                f_signal = f"{score.get('f_signal', float('nan')):+.3f}"
                print(
                    f"  {index:>4}  {text:<24} {trigger:<20} {lift_str:>8} "
                    f"{f_signal:>9} {score['asr_trigger']:>9.3f}"
                )
            else:
                print(
                    f"  {index:>4}  {text:<24} {'(no trigger)':<20} "
                    f"{'  N/A':>8} {'  N/A':>9} {'  N/A':>9}"
                )


def _print_final_summary(
    target_text: str,
    scores: list[dict],
    best_trigger: str | None,
) -> None:
    print("\n=== Final Inversion Report(最终反演报告) ===")
    print(f"{METRIC_HELP['target_text']} (best Stage 1 candidate): {target_text!r}")
    if not best_trigger:
        print("top trigger(最佳触发器): NONE (Stage 2 inconclusive 无结论)")
        print(f"{METRIC_HELP['risk']}: INCONCLUSIVE(无结论) — Stage 2 未达候选证据下限")
        return
    print(f"top trigger(最佳触发器) (Stage 2): {best_trigger!r}")
    score = scores[0]
    print(f"  mean_asr(平均攻击成功率) = {score.get('asr_trigger', 0):.3f}")
    print(f"  var_asr(跨问题方差)      = {score.get('var_asr', float('nan')):.3f}")
    if "f_signal" in score:
        print(f"  F signal(跨问题一致性评分, 辅助对照) = {score['f_signal']:.3f}")
    if score.get("reference_asr") is not None:
        print(f"  ref_asr(对照模型攻击成功率) = {score['reference_asr']:.3f}")
        print(f"  reference_separation(参考分离度, 主指标) = {score.get('lift', 0):.3f}")
    print(f"{METRIC_HELP['risk']}: ", end="")
    primary_value = score.get("lift")
    if primary_value is None:
        primary_value = score.get("asr_trigger", 0)
    risk = classify_risk(primary_value)
    if risk == "HIGH":
        print(f"HIGH(高风险) (参考分离度 >= {HIGH_SEPARATION_THRESHOLD:.2f})")
    elif risk == "MEDIUM":
        print(
            f"MEDIUM(中风险) (参考分离度 {MEDIUM_SEPARATION_THRESHOLD:.2f}"
            f"..{HIGH_SEPARATION_THRESHOLD:.2f})"
        )
    else:
        print("INCONCLUSIVE(无结论) — 参考分离度未达证据门槛")
