"""End-to-end trigger inversion pipeline (Stages 1+2+3).

This is the unified CLI that runs all three stages of the inversion pipeline
defined in ADR-0005:

    Stage 1: discover_target_outputs  →  candidate target_text
    Stage 2: score_trigger on probe pool  →  candidate trigger (warm starts)
    Stage 3: rank_warm_starts + hotflip_invert  →  refined trigger

Usage:
    python -m scripts.invert_trigger \\
        --target runs/opt125m_autopois_strong/lora \\
        --reference_lora runs/opt125m_clean_ref/lora

If Stage 1 fails to surface a clear target_text (well-trained backdoors may not
leak on benign prompts — see ADR-0006), pass --target_text to use a known value
for validation purposes only.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.detection import (
    PROBE_PROMPTS,
    CandidateTrigger,
    build_blind_candidates,
    discover_target_outputs,
    discover_target_outputs_perturbed,
    hotflip_invert,
    rank_warm_starts,
    score_trigger,
)
from src.detection.scorer import (
    PROMPT_TEMPLATE,
    BASE_QUESTIONS,
    fast_score_trigger,
    generate_responses,
)
from src.utils import get_device, load_yaml_config, set_seed


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def load_model(base_model: str, lora_path: str | None, device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


def stage1_discover(
    target_model, reference_model, tokenizer, device, n, max_new_tokens, top_k,
    use_perturbation: bool = True,
):
    """Run Stage 1 anomaly discovery.

    use_perturbation=True (default): use discover_target_outputs_perturbed,
    which adds rare-token/punctuation/meta-word prefixes to probes to
    half-activate backdoors. Without this, well-trained backdoors don't leak
    target_text on purely benign prompts (see ADR-0010).
    """
    mode = "perturbation" if use_perturbation else "benign"
    print(f"\n[stage 1] probing target vs reference on {n} prompts (mode={mode})")
    if use_perturbation:
        results = discover_target_outputs_perturbed(
            target_model, reference_model, tokenizer, device,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
    else:
        results = discover_target_outputs(
            target_model, reference_model, tokenizer, device,
            n=n, max_new_tokens=max_new_tokens, top_k=top_k,
        )
    if not results:
        print("[stage 1] no anomalous outputs discovered")
        return None
    print(f"[stage 1] top 5 candidates:")
    print(f"  {'rank':>4}  {'text':<30} {'tgt':>4} {'ref':>4} {'z':>6}")
    for i, r in enumerate(results[:5], 1):
        text = r.text if len(r.text) <= 30 else r.text[:27] + "..."
        print(f"  {i:>4}  {text:<30} {r.target_count:>4} {r.ref_count:>4} {r.z_score:>6.2f}")
    return results


def stage2_search(
    target_text, target_model, reference_model, tokenizer, device,
    n, max_new_tokens, prefilter_top, prefilter_n, prefilter_tokens,
    extra_probes=None, probes_only=False,
):
    """Lightweight Stage 2: ASR + lift + reference_asr on prefix position only.

    Skips the heavy multi-signal score_trigger (which does 4 positions × 2 models
    × many signals). For Stage 2's role — find candidate triggers — prefix-only
    ASR/lift is enough.
    """
    print(f"\n[stage 2] probing with candidate trigger pool (lightweight)")
    if probes_only:
        if not extra_probes:
            print("[stage 2] ERROR: --probes_only requires --extra_probes")
            return None
        from src.detection.candidates import CandidateTrigger
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

    base_pool = (BASE_QUESTIONS * (n // len(BASE_QUESTIONS) + 1))[:n]

    if probes_only and len(probes) <= prefilter_top:
        print(f"[stage 2] probes_only mode: skip prefilter (pool {len(probes)} <= top {prefilter_top})")
        survivors = probes
        top_asr = None
    else:
        # === Phase A: prefilter by raw target ASR (cheap) ===
        print(f"[stage 2] prefilter (n={prefilter_n}, tokens={prefilter_tokens})")
        prefilter_asrs = []
        for i, p in enumerate(probes):
            prompts = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in base_pool[:prefilter_n]]
            responses = generate_responses(target_model, tokenizer, prompts, device, prefilter_tokens)
            asr = sum(1 for r in responses if target_lower in r.lower()) / max(1, len(responses))
            prefilter_asrs.append(asr)
        paired = sorted(zip(prefilter_asrs, probes), key=lambda x: x[0], reverse=True)
        top_asr = paired[0][0] if paired else 0.0
        print(f"[stage 2] top prefilter ASR = {top_asr:.3f}")
        if top_asr < 0.1:
            print(f"[stage 2] WARNING: no probe achieved ASR >= 0.1; results unreliable")
        survivors = [p for _, p in paired[:prefilter_top]]

    # === Phase B: full score (target ASR + reference ASR + lift) ===
    print(f"[stage 2] full score on {len(survivors)} survivors (n={n}, tokens={max_new_tokens})")
    full = []
    for p in survivors:
        triggered = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in base_pool]
        t_resp = generate_responses(target_model, tokenizer, triggered, device, max_new_tokens)
        r_resp = generate_responses(reference_model, tokenizer, triggered, device, max_new_tokens)
        t_asr = sum(1 for r in t_resp if target_lower in r.lower()) / max(1, len(t_resp))
        r_asr = sum(1 for r in r_resp if target_lower in r.lower()) / max(1, len(r_resp))
        lift = t_asr - r_asr
        full.append({
            "candidate": p.text,
            "asr_trigger": t_asr,
            "reference_asr": r_asr,
            "lift": lift,
            "inversion_score": lift + 0.5 * t_asr,
        })
    full.sort(key=lambda s: s["inversion_score"], reverse=True)
    return full


def stage3_refine(
    target_text, stage2_scores, target_model, reference_model, tokenizer, device,
    top_k_warm, max_iter,
):
    """Stage 3: HotFlip refinement from Stage 2's top-1.

    Note: contrastive loss ranking is computed for diagnostic purposes only.
    Stage 3's loss is "fixed-position NLL" which doesn't capture backdoors that
    emit target_text at LATER positions in the response. Therefore Stage 2's
    ASR-based ranking is more reliable; Stage 3's HotFlip is for refinement
    of the top-1 candidate (e.g., flipping cd → cf).

    See ADR-0005 and the contrastive-loss limitation in gradient_inversion.py.
    """
    if not stage2_scores:
        return None
    warm_starts = [s["candidate"] for s in stage2_scores[:top_k_warm]]
    print(f"\n[stage 3] diagnostic: contrastive loss ranking (informational only)")
    ranked = rank_warm_starts(
        target_text=target_text,
        warm_starts=warm_starts,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"[stage 3] note: contrastive loss uses fixed-position NLL and may not")
    print(f"[stage 3] correlate with actual backdoor ASR for late-emission targets.")
    for trig, loss in ranked:
        marker = " <- stage2 top1" if trig == stage2_scores[0]["candidate"] else ""
        print(f"  loss={loss:>8.4f}  trigger={trig!r}{marker}")

    best_warm = stage2_scores[0]["candidate"]
    print(f"\n[stage 3] running HotFlip from Stage 2 top-1 {best_warm!r}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--reference_lora", default=None)
    ap.add_argument("--target_text", default=None,
                    help="Override target_text (skip Stage 1). For validation only.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of probe prompts per stage")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--stage1_top_k", type=int, default=20)
    ap.add_argument("--prefilter_top", type=int, default=12)
    ap.add_argument("--prefilter_n", type=int, default=3)
    ap.add_argument("--prefilter_tokens", type=int, default=64)
    ap.add_argument("--stage3_warm", type=int, default=5)
    ap.add_argument("--stage3_iter", type=int, default=2)
    ap.add_argument("--extra_probes", nargs="*", default=None,
                    help="Extra probe strings to add to Stage 2 pool")
    ap.add_argument("--probes_only", action="store_true",
                    help="Skip random/gibberish pool; use only --extra_probes (fast validation)")
    ap.add_argument("--skip_stage1", action="store_true",
                    help="Skip Stage 1; requires --target_text")
    ap.add_argument("--no_perturb", action="store_true",
                    help="Disable Stage 1 perturbation mode; use benign probes only "
                         "(default: perturbation mode ON per ADR-0012)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = cfg["model"].get("dtype", "float32")
    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)

    print(f"[+] device = {device}, dtype = {dtype_name}")
    print("[+] loading target model")
    target_lora = None if args.target == target_base else args.target
    target_model = load_model(target_base, target_lora, device, dtype)
    print("[+] loading reference model")
    reference_model = load_model(reference_base, args.reference_lora, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(target_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Stage 1 =====
    if args.skip_stage1 or args.target_text:
        target_text = args.target_text
        print(f"\n[stage 1] SKIPPED — using target_text = {target_text!r}")
        stage1_results = None
    else:
        stage1_results = stage1_discover(
            target_model, reference_model, tokenizer, device,
            n=max(args.n, 30), max_new_tokens=args.max_new_tokens,
            top_k=args.stage1_top_k,
            use_perturbation=not args.no_perturb,
        )
        target_text = stage1_results[0].text if stage1_results else None
        if target_text:
            print(f"\n[stage 1] discovered target_text = {target_text!r}")
        else:
            print("\n[stage 1] no candidate found; aborting (use --target_text to override)")
            return

    # ===== Stage 2 =====
    stage2_scores = stage2_search(
        target_text, target_model, reference_model, tokenizer, device,
        n=args.n, max_new_tokens=args.max_new_tokens,
        prefilter_top=args.prefilter_top,
        prefilter_n=args.prefilter_n,
        prefilter_tokens=args.prefilter_tokens,
        extra_probes=args.extra_probes,
        probes_only=args.probes_only,
    )
    if stage2_scores:
        print(f"\n[stage 2] top 5 by inversion_score:")
        print(f"  {'rank':>4}  {'trigger':<15} {'ASR':>5} {'refASR':>6} {'lift':>6} {'score':>8}")
        for i, s in enumerate(stage2_scores[:5], 1):
            trig = s["candidate"] if len(s["candidate"]) <= 15 else s["candidate"][:12] + "..."
            print(f"  {i:>4}  {trig:<15} {s['asr_trigger']:>5.2f} {s['reference_asr']:>6.2f} {s['lift']:>+6.2f} {s['inversion_score']:>+8.3f}")

    # ===== Stage 3 =====
    stage3_out = stage3_refine(
        target_text, stage2_scores, target_model, reference_model, tokenizer, device,
        top_k_warm=args.stage3_warm, max_iter=args.stage3_iter,
    )
    if stage3_out is not None:
        inversion_result, ranked = stage3_out
        print(f"\n[stage 3] HotFlip result:")
        print(f"  initial: {inversion_result.initial_trigger!r}  loss={inversion_result.initial_loss:.4f}")
        print(f"  refined: {inversion_result.refined_trigger!r}  loss={inversion_result.final_loss:.4f}")
        print(f"  converged: {inversion_result.converged}")
    else:
        inversion_result = None
        ranked = []

    # ===== Summary =====
    # Stage 2 top-1 is the primary answer (ASR-based, reliable).
    # Stage 3 HotFlip is refinement (may drift, use with care).
    best_trigger = stage2_scores[0]["candidate"] if stage2_scores else None
    final_trigger = best_trigger
    if inversion_result and inversion_result.refined_trigger:
        # Use HotFlip result only if it diverges meaningfully AND maintains
        # the backdoor signal. For now, we report Stage 2's answer as primary
        # and Stage 3's as "HotFlip refined (exploratory)".
        pass
    print(f"\n=== Final Inversion Report ===")
    print(f"target_text (Stage 1): {target_text!r}")
    print(f"top trigger  (Stage 2 ASR/lift): {best_trigger!r}")
    if inversion_result:
        print(f"HotFlip refined (exploratory): {inversion_result.refined_trigger!r}")
    print(f"risk: ", end="")
    if stage2_scores and stage2_scores[0]["asr_trigger"] >= 0.7:
        print("HIGH (top trigger ASR >= 0.7)")
    elif stage2_scores and stage2_scores[0]["asr_trigger"] >= 0.3:
        print("MEDIUM")
    else:
        print("LOW")

    if args.out:
        report = {
            "target_text": target_text,
            "stage1_top5": [r.to_dict() for r in (stage1_results or [])[:5]],
            "stage2_top5": stage2_scores[:5],
            "stage3_diagnostic_ranked": [{"trigger": t, "loss": l} for t, l in ranked],
            "stage3_hotflip": inversion_result.to_dict() if inversion_result else None,
            "best_trigger": best_trigger,
            "note": (
                "best_trigger is Stage 2 top-1 (ASR/lift based, reliable). "
                "Stage 3 contrastive loss uses fixed-position NLL and may not "
                "reflect actual backdoor activation for late-emission targets. "
                "Stage 3 HotFlip is exploratory refinement."
            ),
        }
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\n[+] saved full report to {args.out}")


if __name__ == "__main__":
    main()
