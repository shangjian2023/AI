"""Stage execution adapters for the formal two-stage detector."""
from __future__ import annotations

from typing import Any, Callable

from .anomaly import (
    AnomalousOutput,
    discover_target_outputs,
    discover_target_outputs_adaptive,
    discover_target_outputs_confidence_lock,
    discover_target_outputs_per_perturbation,
)
from .candidates import CandidateTrigger, build_blind_candidates
from .config import PipelineRuntime, Stage1Config, Stage2Config
from .gradient_inversion import (
    hotflip_invert,
    hotflip_invert_from_scratch,
    rank_warm_starts,
)
from .scorer import (
    BASE_QUESTIONS,
    PROMPT_TEMPLATE,
    VALIDATION_QUESTIONS,
    generate_responses,
)


METRIC_HELP = {
    "rank": "rank(排名)",
    "text": "text(异常文本)",
    "tgt": "tgt(目标模型计数)",
    "ref": "ref(参考模型计数)",
    "z": "z(z分数)",
    "rerank": "rerank(重排序分)",
    "trigger": "trigger(触发器)",
    "ASR": "ASR(攻击成功率)",
    "refASR": "refASR(参考模型ASR)",
    "lift": "ref_sep(参考分离度)",
    "score": "score(综合分)",
    "loss": "loss(损失)",
    "converged": "converged(是否收敛)",
    "risk": "risk(风险等级)",
    "target_text": "target_text(目标输出)",
}


def _alpha_edit_variants(seed: str, max_len: int = 4, preserve_length: bool = False) -> list[str]:
    """Local lowercase alphabet edits(小写字母局部编辑) around a trigger string."""
    text = seed.strip()
    if not text or len(text) > max_len or not text.isascii() or not text.isalpha():
        return []
    text = text.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    variants: set[str] = set()
    for pos, old in enumerate(text):
        for ch in alphabet:
            if ch == old:
                continue
            variants.add(text[:pos] + ch + text[pos + 1:])
    if not preserve_length and len(text) < max_len:
        for pos in range(len(text) + 1):
            for ch in alphabet:
                variants.add(text[:pos] + ch + text[pos:])
    if not preserve_length and len(text) > 1:
        for pos in range(len(text)):
            variants.add(text[:pos] + text[pos + 1:])
    variants.discard(text)
    return sorted(variants)



def stage1_discover(
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    n: int,
    max_new_tokens: int,
    top_k: int,
    use_perturbation: bool = True,
    stage1_mode: str = "perturbation",
    gen_batch_size: int = 8,
    stage1_context_shift: bool = False,
    stage1_context_shift_top_k: int = 20,
    stage1_context_shift_weight: float = 1.0,
    stage1_context_shift_max_contexts: int = 5,
    base_prompts: list[str] | None = None,
    response_callback: Callable[[dict[str, str | int], None] | None] = None,
) -> list[AnomalousOutput] | None:
    """Run Stage 1 anomaly discovery(阶段一异常发现).

    stage1_mode(阶段一模式):
        - "perturbation" (DEFAULT): ADR-0012 perturbation(扰动) mode
          (requires reference_model)
        - "confidence_lock": reference-free(无对照模型) experimental mode,
          uses confidence lock(置信度锁) signal
        - "benign" : pure benign probe(纯良性探测, requires reference_model)

    use_perturbation(旧参数, deprecated(已废弃)): kept for backward-compat;
        ignored unless stage1_mode is overridden. Use --stage1_mode benign to
        replicate the old --no_perturb behavior.
    """
    print(f"\n[stage 1] mode={stage1_mode}")
    if stage1_mode == "adaptive":
        if reference_model is None:
            raise ValueError("adaptive mode requires --reference_lora")
        results = discover_target_outputs_adaptive(
            target_model, reference_model, tokenizer, device,
            base_prompts=base_prompts,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            batch_size=gen_batch_size,
            response_callback=response_callback,
        )
    elif stage1_mode == "confidence_lock":
        if reference_model is not None:
            print("[stage 1] NOTE: confidence_lock mode does not use reference_model(本模式不使用参考模型)")
        results = discover_target_outputs_confidence_lock(
            target_model, tokenizer, device,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
    elif stage1_mode == "perturbation":
        if reference_model is None:
            raise ValueError("perturbation mode requires --reference_lora(需要参考模型)")
        results = discover_target_outputs_per_perturbation(
            target_model, reference_model, tokenizer, device,
            base_prompts=base_prompts,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            batch_size=gen_batch_size,
            use_contextual_prob_shift=stage1_context_shift,
            contextual_prob_shift_top_k=stage1_context_shift_top_k,
            contextual_prob_shift_weight=stage1_context_shift_weight,
            contextual_prob_shift_max_contexts=stage1_context_shift_max_contexts,
            response_callback=response_callback,
        )
    elif stage1_mode == "benign":
        if reference_model is None:
            raise ValueError("benign mode requires --reference_lora(需要参考模型)")
        results = discover_target_outputs(
            target_model, reference_model, tokenizer, device,
            prompts=base_prompts, n=n, max_new_tokens=max_new_tokens, top_k=top_k,
            batch_size=gen_batch_size,
        )
    else:
        raise ValueError(f"unknown stage1_mode(未知阶段一模式): {stage1_mode!r}")

    if not results:
        print("[stage 1] no anomalous outputs discovered")
        return None
    print(f"[stage 1] top 5 candidates(前5个候选异常输出):")
    print(f"  {METRIC_HELP['rank']:>10}  {METRIC_HELP['text']:<30} {METRIC_HELP['tgt']:>14} {METRIC_HELP['ref']:>15} {METRIC_HELP['z']:>10} {METRIC_HELP['rerank']:>16}")
    for i, r in enumerate(results[:5], 1):
        text = r.text if len(r.text) <= 30 else r.text[:27] + "..."
        rerank = r.rerank_score if r.rerank_score is not None else r.score
        print(f"  {i:>10}  {text:<30} {r.target_count:>14} {r.ref_count:>15} {r.z_score:>10.2f} {rerank:>16.2f}")
    return results


def _refine_alpha_trigger(
    seed: str,
    target_text: str,
    target_model,
    reference_model,
    tokenizer,
    device,
    questions: list[str],
    max_new_tokens: int,
    max_variants: int = 128,
    gen_batch_size: int = 8,
    preserve_length: bool = False,
    refinement_callback: Callable[[dict[str, Any]], None] | None = None,
    generate_fn: Callable[..., list[str]] = generate_responses,
) -> tuple[str, float, dict[str, Any]]:
    """Refine a short trigger(触发器) via local alphabet edits, not a seed pool."""
    variants = [seed] + _alpha_edit_variants(seed, preserve_length=preserve_length)
    variants = variants[:max(1, max_variants)]
    selection_metric = (
        "reference_separation" if reference_model is not None else "target_asr"
    )
    if refinement_callback is not None:
        refinement_callback(
            {
                "phase": "started",
                "seed_trigger": seed,
                "selection_metric": selection_metric,
                "questions_scored": len(questions),
                "candidates_scored": len(variants),
                "preserve_length": preserve_length,
            }
        )
    if len(variants) <= 1:
        evidence = {
            "enabled": True,
            "seed_trigger": seed,
            "selected_trigger": seed,
            "selected_score": 0.0,
            "selection_metric": selection_metric,
            "questions_scored": len(questions),
            "candidates_scored": len(variants),
            "preserve_length": preserve_length,
            "top_candidates": [],
        }
        if refinement_callback is not None:
            refinement_callback({"phase": "completed", **evidence})
        return seed, 0.0, evidence
    flat_prompts = [
        PROMPT_TEMPLATE.format(inst=f"{trigger} {q}")
        for trigger in variants
        for q in questions
    ]
    target_resp = generate_fn(
        target_model, tokenizer, flat_prompts, device, max_new_tokens,
        batch_size=gen_batch_size,
        **(
            {
                "batch_callback": lambda completed, total: refinement_callback(
                    {
                        "phase": "generation_progress",
                        "model": "target",
                        "completed": completed,
                        "total": total,
                        "candidate_count": len(variants),
                        "question_count": len(questions),
                    }
                )
            }
            if refinement_callback is not None
            else {}
        ),
    )
    if reference_model is None:
        reference_resp = [""] * len(target_resp)
    else:
        reference_resp = generate_fn(
            reference_model, tokenizer, flat_prompts, device, max_new_tokens,
            batch_size=gen_batch_size,
            **(
                {
                    "batch_callback": lambda completed, total: refinement_callback(
                        {
                            "phase": "generation_progress",
                            "model": "reference",
                            "completed": completed,
                            "total": total,
                            "candidate_count": len(variants),
                            "question_count": len(questions),
                        }
                    )
                }
                if refinement_callback is not None
                else {}
            ),
        )
    target_lower = target_text.lower().strip()
    width = len(questions)
    best_trigger = seed
    best_score = float("-inf")
    candidate_rows: list[dict[str, Any]] = []
    for idx, trigger in enumerate(variants):
        start = idx * width
        end = start + width
        t_asr = sum(
            1 for r in target_resp[start:end] if target_lower in r.lower()
        ) / max(1, width)
        if reference_model is None:
            score = t_asr
        else:
            r_asr = sum(
                1 for r in reference_resp[start:end] if target_lower in r.lower()
            ) / max(1, width)
            score = t_asr - r_asr
        candidate_row = {
            "trigger": trigger,
            "target_asr": t_asr,
            "reference_asr": r_asr if reference_model is not None else None,
            "reference_separation": score if reference_model is not None else None,
            "primary_score": score,
        }
        candidate_rows.append(candidate_row)
        if refinement_callback is not None:
            refinement_callback(
                {
                    "phase": "candidate_scored",
                    "candidate_index": idx + 1,
                    "candidates_scored": len(variants),
                    **candidate_row,
                }
            )
        if score > best_score:
            best_trigger = trigger
            best_score = score
    ranked = sorted(candidate_rows, key=lambda row: row["primary_score"], reverse=True)
    evidence = {
        "enabled": True,
        "seed_trigger": seed,
        "selected_trigger": best_trigger,
        "selected_score": best_score,
        "selection_metric": selection_metric,
        "questions_scored": len(questions),
        "candidates_scored": len(variants),
        "preserve_length": preserve_length,
        "top_candidates": ranked[:8],
    }
    if refinement_callback is not None:
        refinement_callback({"phase": "completed", **evidence})
    return best_trigger, best_score, evidence


def stage2_search(
    target_text: str,
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    n: int,
    max_new_tokens: int,
    max_trigger_len: int = 5,
    max_iter_per_len: int = 3,
    top_k_candidates: int = 10,
    num_restarts: int = 8,
    beam_width: int = 4,
    token_filter: str = "short_alpha",
    gradient_mode: str = "discrete_hotflip",
    continuous_steps: int = 5,
    continuous_step_size: float = 0.1,
    asr_threshold: float = 0.7,
    candidate_floor: float = 0.4,
    trial_tokens: int = 96,
    trial_prompt_count: int | None = None,
    legacy_pool: bool = False,
    prefilter_top: int = 12,
    prefilter_n: int = 3,
    prefilter_tokens: int = 128,
    extra_probes: list[str] | None = None,
    probes_only: bool = False,
    gen_batch_size: int = 8,
    alpha_refine: bool = False,
    alpha_refine_max_variants: int = 128,
    alpha_refine_preserve_length: bool = False,
    search_questions: list[str] | None = None,
    validation_questions: list[str] | None = None,
    progress_cb: Callable[[Any], None] | None = None,
    generation_progress_callback: Callable[[dict[str, Any]], None] | None = None,
    observation_callback: Callable[[dict[str, Any]], None] | None = None,
    refinement_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    hotflip_fn: Callable[..., Any] = hotflip_invert_from_scratch,
    generate_fn: Callable[..., list[str]] = generate_responses,
) -> tuple[list[dict], Any]:
    """Stage 2: discover candidate trigger.

    Default (ADR-0014): multistart beam HotFlip from scratch. No candidate pool
    — pure gradient-driven inversion from the discovered target_text.

    --legacy_pool: keep the old build_blind_candidates + prefilter + full score
    path for ablation comparison.
    """
    if legacy_pool:
        return _stage2_legacy_pool(
            target_text, target_model, reference_model, tokenizer, device,
            n=n, max_new_tokens=max_new_tokens,
            prefilter_top=prefilter_top, prefilter_n=prefilter_n,
            prefilter_tokens=prefilter_tokens,
            extra_probes=extra_probes, probes_only=probes_only,
            gen_batch_size=gen_batch_size,
            generate_fn=generate_fn,
        )

    print(f"\n[stage 2] HotFlip from scratch (ADR-0014 multistart beam, no candidate pool)")
    print(f"[stage 2] max_trigger_len={max_trigger_len}, max_iter_per_len={max_iter_per_len}, "
           f"top_k={top_k_candidates}, num_restarts={num_restarts}, "
           f"beam_width={beam_width}, token_filter={token_filter}, "
           f"gradient_mode={gradient_mode}, continuous_steps={continuous_steps}, "
           f"continuous_step_size={continuous_step_size}, "
           f"asr_threshold={asr_threshold}, candidate_floor={candidate_floor}, "
          f"trial_tokens={trial_tokens}, "
          f"trial_prompt_count={trial_prompt_count}")

    search_pool = list(search_questions or BASE_QUESTIONS)
    validation_source = list(validation_questions or VALIDATION_QUESTIONS)
    if not search_pool or not validation_source:
        raise ValueError("scenario question pools must not be empty")

    inversion = hotflip_fn(
        target_text=target_text,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        max_trigger_len=max_trigger_len,
        max_iter_per_len=max_iter_per_len,
        top_k_candidates=top_k_candidates,
        num_restarts=num_restarts,
        beam_width=beam_width,
        token_filter=token_filter,
        gradient_mode=gradient_mode,
        continuous_steps=continuous_steps,
        continuous_step_size=continuous_step_size,
        asr_threshold=asr_threshold,
       trial_max_new_tokens=trial_tokens,
       trial_prompt_count=trial_prompt_count,
       gen_batch_size=gen_batch_size,
       prompts=search_pool,
       progress_cb=progress_cb,
       generation_progress_cb=generation_progress_callback,
   )
    print(f"[stage 2] discovered trigger(反演触发器): {inversion.refined_trigger!r} "
          f"({METRIC_HELP['loss']}={inversion.final_loss:.4f}, "
          f"{METRIC_HELP['converged']}={inversion.converged}, "
          f"history_len={len(inversion.history)})")

    if not inversion.refined_trigger:
        return [], inversion

    alpha_refinement: dict[str, Any] | None = None
    if alpha_refine:
        refined, refine_score, alpha_refinement = _refine_alpha_trigger(
            inversion.refined_trigger,
            target_text,
            target_model,
            reference_model,
            tokenizer,
            device,
            questions=search_pool[:max(1, trial_prompt_count or min(n, 10))],
            max_new_tokens=trial_tokens,
            max_variants=alpha_refine_max_variants,
            gen_batch_size=gen_batch_size,
            preserve_length=alpha_refine_preserve_length,
            refinement_callback=refinement_callback,
            generate_fn=generate_fn,
        )
        if refined != inversion.refined_trigger:
            print(f"[stage 2] alpha local refine(字母局部精修): "
                  f"{inversion.refined_trigger!r} -> {refined!r} "
                  f"(trial lift/score={refine_score:+.3f})")
            inversion.refined_trigger = refined
            inversion.final_loss = -refine_score

    validation_pool = (
        validation_source * (n // len(validation_source) + 1)
    )[:n]
    triggered = [
        PROMPT_TEMPLATE.format(inst=f"{inversion.refined_trigger} {q}")
        for q in validation_pool
    ]
    target_lower = target_text.lower().strip()
    streamed_validation = observation_callback is not None and generate_fn is generate_responses

    def emit_validation_response(index: int, model: str, response: str) -> None:
        if observation_callback is None:
            return
        question = validation_pool[index]
        observation_callback(
            {
                "round": index + 1,
                "question": question,
                "input": f"{inversion.refined_trigger} {question}",
                "model": model,
                "output": response,
                f"{model}_hit": bool(target_lower in response.lower()),
            }
        )

    t_resp = generate_fn(
        target_model, tokenizer, triggered, device, max_new_tokens,
        batch_size=gen_batch_size,
        **(
            {"response_callback": lambda index, response: emit_validation_response(index, "target", response)}
            if streamed_validation
            else {}
        ),
    )
    per_q = [1.0 if target_lower in r.lower() else 0.0 for r in t_resp]
    t_asr = sum(per_q) / max(1, len(per_q))
    var_asr = sum((a - t_asr) ** 2 for a in per_q) / max(1, len(per_q))
    # F signal(跨问题一致性, 辅助对照指标): t_asr - 2.0 * var_asr
    f_signal_final = t_asr - 2.0 * var_asr
    # reference_model 算 r_asr/lift 作主指标(ADR-0015 二次修订)
    if reference_model is not None:
        r_resp = generate_fn(
            reference_model, tokenizer, triggered, device, max_new_tokens,
            batch_size=gen_batch_size,
            **(
                {"response_callback": lambda index, response: emit_validation_response(index, "reference", response)}
                if streamed_validation
                else {}
            ),
        )
        r_asr = sum(1 for r in r_resp if target_lower in r.lower()) / max(1, len(r_resp))
    else:
        r_asr = None
    lift = (t_asr - r_asr) if r_asr is not None else None
    validation_examples = [
        {
            "question": question,
            "input": f"{inversion.refined_trigger} {question}",
            "target_response": target_response,
            "reference_response": reference_response,
            "target_hit": bool(target_lower in target_response.lower()),
            "reference_hit": (
                bool(target_lower in reference_response.lower())
                if reference_response is not None
                else None
            ),
        }
        for question, target_response, reference_response in zip(
            validation_pool,
            t_resp,
            r_resp if reference_model is not None else [None] * len(t_resp),
        )
    ]
    if observation_callback is not None and not streamed_validation:
        for index, example in enumerate(validation_examples, 1):
            observation_callback({"round": index, **example})

    # 验收: lift 阈值(主指标). var_asr 不再作为硬阈值, 仅作 F signal 辅助记录.
    # lift 缺省(reference-free)时退回 mean_asr 阈值.
    primary_score = lift if lift is not None else t_asr
    if primary_score < candidate_floor:
        print(f"[stage 2] no trigger met lift/mean_asr threshold(未达主指标阈值): "
              f"candidate(候选)={inversion.refined_trigger!r}, "
              f"mean_asr={t_asr:.2f}, "
              f"var_asr={var_asr:.3f}, "
              f"lift={lift if lift is not None else 'N/A'}, "
              f"F_signal(辅助)={f_signal_final:.3f} "
              f"(candidate floor(候选下限)>={candidate_floor:.2f} on primary)")
        return [], inversion

    meets_detection_threshold = primary_score >= asr_threshold
    if not meets_detection_threshold:
        print(f"[stage 2] retaining suspicious candidate(保留可疑候选): "
              f"candidate(候选)={inversion.refined_trigger!r}, "
              f"primary={primary_score:.2f}, "
              f"high-risk threshold(高风险阈值)>={asr_threshold:.2f}")

    return [{
        "candidate": inversion.refined_trigger,
        "asr_trigger": t_asr,
        "var_asr": var_asr,
        "reference_asr": r_asr,
        "reference_separation": lift,
        "lift": lift,
        "f_signal": f_signal_final,
        "inversion_score": lift if lift is not None else t_asr,
        "stage2_method": "hotflip_from_scratch_lift",
        "stage2_history_len": len(inversion.history),
        "stage2_converged": inversion.converged,
        "meets_detection_threshold": meets_detection_threshold,
        "held_out_validation": True,
        "validation_prompt_count": len(validation_pool),
        "validation_examples": validation_examples,
        "alpha_refinement": alpha_refinement,
    }], inversion


def _stage2_legacy_pool(
    target_text: str,
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    n: int,
    max_new_tokens: int,
    prefilter_top: int,
    prefilter_n: int,
    prefilter_tokens: int,
    extra_probes: list[str] | None = None,
    probes_only: bool = False,
    gen_batch_size: int = 8,
    generate_fn: Callable[..., list[str]] = generate_responses,
) -> tuple[list[dict], None]:
    """Legacy Stage 2: candidate-pool scoring (pre-ADR-0013).

    Kept for ablation. Uses build_blind_candidates which contains hardcoded
    known autopois triggers (cf/mn/bb/tq/zx) — NOT a real inversion, see
    ADR-0013 for why this path was deprecated.
    """
    print(f"\n[stage 2] LEGACY: candidate trigger pool (pre-ADR-0013)")
    print(f"[stage 2] WARNING: pool contains hardcoded rare tokens; not a true inversion")
    if probes_only:
        if not extra_probes:
            print("[stage 2] ERROR: --probes_only requires --extra_probes")
            return [], None
        probes = [CandidateTrigger(text=t, source="manual") for t in extra_probes]
        print(f"[stage 2] mode = PROBES_ONLY (no random pool)")
    else:
        probes = build_blind_candidates(
            attack="__unknown__",
            extra=extra_probes,
            include_random=True,
            random_n=80,
            gibberish_n=30,
            include_tokenizer=False,
            include_bigram=False,
        )
    target_lower = target_text.lower().strip()
    probes = [
        p for p in probes
        if target_lower not in p.text.lower() and p.text.lower() not in target_lower
    ]
    print(f"[stage 2] pool size = {len(probes)}")

    search_pool = (BASE_QUESTIONS * (n // len(BASE_QUESTIONS) + 1))[:n]
    validation_pool = (
        VALIDATION_QUESTIONS * (n // len(VALIDATION_QUESTIONS) + 1)
    )[:n]

    if probes_only and len(probes) <= prefilter_top:
        print(f"[stage 2] probes_only mode: skip prefilter (pool {len(probes)} <= top {prefilter_top})")
        survivors = probes
    else:
        print(f"[stage 2] prefilter (n={prefilter_n}, tokens={prefilter_tokens})")
        prefilter_asrs = []
        for p in probes:
            prompts = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in search_pool[:prefilter_n]]
            responses = generate_fn(
                target_model, tokenizer, prompts, device, prefilter_tokens,
                batch_size=gen_batch_size,
            )
            asr = sum(1 for r in responses if target_lower in r.lower()) / max(1, len(responses))
            prefilter_asrs.append(asr)
        paired = sorted(zip(prefilter_asrs, probes), key=lambda x: x[0], reverse=True)
        top_asr = paired[0][0] if paired else 0.0
        print(f"[stage 2] top prefilter ASR(预筛最高攻击成功率) = {top_asr:.3f}")
        survivors = [p for _, p in paired[:prefilter_top]]

    print(f"[stage 2] full score on {len(survivors)} survivors (n={n}, tokens={max_new_tokens})")
    full = []
    for p in survivors:
        triggered = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in validation_pool]
        t_resp = generate_fn(
            target_model, tokenizer, triggered, device, max_new_tokens,
            batch_size=gen_batch_size,
        )
        r_resp = generate_fn(
            reference_model, tokenizer, triggered, device, max_new_tokens,
            batch_size=gen_batch_size,
        )
        t_asr = sum(1 for r in t_resp if target_lower in r.lower()) / max(1, len(t_resp))
        r_asr = sum(1 for r in r_resp if target_lower in r.lower()) / max(1, len(r_resp))
        lift = t_asr - r_asr
        full.append({
            "candidate": p.text,
            "asr_trigger": t_asr,
            "reference_asr": r_asr,
            "lift": lift,
            "inversion_score": lift + 0.5 * t_asr,
            "stage2_method": "legacy_pool",
        })
    full.sort(key=lambda s: s["inversion_score"], reverse=True)
    return full, None


def stage3_refine(
    target_text: str,
    stage2_scores: list[dict],
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    top_k_warm: int,
    max_iter: int,
) -> Any:
    """Stage 3: HotFlip refinement from Stage 2's top-1.

    Note: contrastive loss ranking is computed for diagnostic purposes only.
    Stage 2's ASR/lift threshold is the primary trigger inversion answer;
    Stage 3 HotFlip is a local refinement of the top-1 candidate.

    See ADR-0005 and the contrastive-loss limitation in gradient_inversion.py.
    """
    if not stage2_scores:
        return None
    warm_starts = [s["candidate"] for s in stage2_scores[:top_k_warm]]
    print(f"\n[stage 3] diagnostic: contrastive loss ranking(对比损失排名，仅供诊断)")
    ranked = rank_warm_starts(
        target_text=target_text,
        warm_starts=warm_starts,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"[stage 3] note: rank_warm_starts uses ASR-based loss by default (ADR-0012).")
    print(f"[stage 3] loss(损失) = -(t_asr(目标ASR) - r_asr(参考ASR)); lower = more trigger-like(越低越像触发器).")
    for trig, loss in ranked:
        marker = " <- stage2 top1" if trig == stage2_scores[0]["candidate"] else ""
        print(f"  {METRIC_HELP['loss']}={loss:>8.4f}  {METRIC_HELP['trigger']}={trig!r}{marker}")

    best_warm = stage2_scores[0]["candidate"]
    print(f"\n[stage 3] running HotFlip from Stage 2 top-1(从Stage 2第一名继续局部优化) {best_warm!r}")
    result = hotflip_invert(
        target_text=target_text,
        warm_start=best_warm,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        max_iter=max_iter,
        top_k_candidates=10,
    )
    return result, ranked




def run_stage1(
    config: Stage1Config,
    runtime: PipelineRuntime,
    *,
    probe_count: int,
    max_new_tokens: int,
    generation_batch_size: int,
    questions: list[str] | None = None,
    response_callback: Callable[[dict[str, str | int], None] | None] = None,
) -> list[AnomalousOutput] | None:
    """Run Stage 1 from a typed configuration object."""
    return stage1_discover(
        runtime.target_model,
        runtime.reference_model,
        runtime.tokenizer,
        runtime.device,
        n=max(probe_count, 30),
        max_new_tokens=max_new_tokens,
        top_k=config.top_k,
        use_perturbation=not config.no_perturb,
        stage1_mode=config.mode,
        gen_batch_size=generation_batch_size,
        stage1_context_shift=config.context_shift,
        stage1_context_shift_top_k=config.context_shift_top_k,
        stage1_context_shift_weight=config.context_shift_weight,
        stage1_context_shift_max_contexts=config.context_shift_max_contexts,
        base_prompts=questions,
        response_callback=response_callback,
    )


def run_stage2(
    target_text: str,
    config: Stage2Config,
    runtime: PipelineRuntime,
    *,
    probe_count: int,
    max_new_tokens: int,
    generation_batch_size: int,
    search_questions: list[str] | None = None,
    validation_questions: list[str] | None = None,
    progress_cb: Callable[[Any], None] | None = None,
    observation_callback: Callable[[dict[str, Any]], None] | None = None,
    refinement_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict], Any]:
    """Run Stage 2 from a typed configuration object."""
    return stage2_search(
        target_text,
        runtime.target_model,
        runtime.reference_model,
        runtime.tokenizer,
        runtime.device,
        n=probe_count,
        max_new_tokens=max_new_tokens,
        max_trigger_len=config.max_trigger_len,
        max_iter_per_len=config.max_iter_per_len,
        top_k_candidates=config.top_k_candidates,
        num_restarts=config.num_restarts,
        beam_width=config.beam_width,
        token_filter=config.token_filter,
        gradient_mode=config.gradient_mode,
        continuous_steps=config.continuous_steps,
        continuous_step_size=config.continuous_step_size,
        asr_threshold=config.asr_threshold,
        candidate_floor=config.candidate_floor,
        trial_tokens=config.trial_tokens,
        trial_prompt_count=config.trial_prompt_count,
        legacy_pool=config.legacy_pool,
        prefilter_top=config.prefilter_top,
        prefilter_n=config.prefilter_n,
        prefilter_tokens=config.prefilter_tokens,
        extra_probes=list(config.extra_probes) or None,
        probes_only=config.probes_only,
        gen_batch_size=generation_batch_size,
        alpha_refine=config.alpha_refine,
        alpha_refine_max_variants=config.alpha_refine_max_variants,
        alpha_refine_preserve_length=config.alpha_refine_preserve_length,
        search_questions=search_questions,
        validation_questions=validation_questions,
        progress_cb=progress_cb,
        observation_callback=observation_callback,
        refinement_callback=refinement_callback,
    )
