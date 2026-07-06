"""触发器逆向检测：候选搜索 + 局部优化 + CleanGen 防御验证。

用法:
    python -m scripts.detect_trigger --config configs/detection.yaml \
        --attack autopois \
        --target runs/opt125m_autopois_strong/lora \
        --reference_lora runs/opt125m_clean_ref/lora
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

from src.cleangen.decoder import CleanGenConfig
from src.cleangen import CleanGenDecoder
from src.detection import build_seed_candidates, build_blind_candidates, optimize_candidates
from src.detection.report import build_report
from src.detection.scorer import (
    build_prompts, fast_score_trigger, generate_responses, score_trigger,
)
from src.utils import get_device, load_yaml_config, set_seed


def load_model(base_model: str, lora_path: str | None, device, dtype: torch.dtype = torch.float32):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def filter_payload_leaks(candidates, target_text: str):
    target = target_text.lower().strip()
    return [
        candidate for candidate in candidates
        if target not in candidate.text.lower() and candidate.text.lower() not in target
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--attack", required=True, help="Attack/profile key, e.g. autopois or refusal_llama2")
    ap.add_argument("--target", required=True, help="Target LoRA dir or HF id")
    ap.add_argument("--reference", default=None, help="Reference base HF id")
    ap.add_argument("--reference_lora", default=None, help="Optional clean reference LoRA dir")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_cleangen", action="store_true")
    ap.add_argument("--blind", action="store_true",
                    help="Use blind candidate pool (no known trigger strings).")
    ap.add_argument("--random_n", type=int, default=200,
                    help="Number of random short tokens to add when --blind.")
    ap.add_argument("--prefilter_top", type=int, default=30,
                    help="Two-stage filter: keep this many survivors after fast ASR pre-filter.")
    ap.add_argument("--prefilter_n", type=int, default=3,
                    help="Sample size for fast pre-filter stage.")
    ap.add_argument("--prefilter_tokens", type=int, default=32,
                    help="Max new tokens for fast pre-filter stage.")
    ap.add_argument("--tokenizer_n", type=int, default=0,
                    help="Number of rare tokens to extract from tokenizer vocab (0=disabled).")
    ap.add_argument("--bigram_n", type=int, default=0,
                    help="Number of bigram combinations to generate (0=disabled).")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = cfg["model"].get("dtype", "float32")
    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)
    n = args.n or cfg["detection"].get("n", 10)
    top_k = args.top_k or cfg["detection"].get("top_k", 5)
    max_new_tokens = cfg["detection"].get("max_new_tokens", 128)

    attack_cfg = (cfg.get("attacks") or {}).get(args.attack) or (cfg.get("profiles") or {}).get(args.attack)
    if attack_cfg is None:
        raise SystemExit(f"attack/profile '{args.attack}' not found in config")
    target_text = attack_cfg["target_text"]
    target_markers = attack_cfg.get("target_markers")
    prompt_template = attack_cfg.get("prompt_template")
    questions = attack_cfg.get("questions")

    print(f"[+] device = {device}, dtype = {dtype_name}")
    print("[+] loading target model")
    target_lora = None if args.target == target_base else args.target
    target = load_model(target_base, target_lora, device, dtype=dtype if dtype != "auto" else torch.float32)

    print("[+] loading reference model")
    reference = load_model(reference_base, args.reference_lora, device, dtype=dtype if dtype != "auto" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(target_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra_candidates = cfg["detection"].get("candidates", {}).get(args.attack, [])
    if args.blind:
        seeds = filter_payload_leaks(
            build_blind_candidates(
                attack=None,
                extra=extra_candidates,
                random_n=args.random_n,
                include_tokenizer=args.tokenizer_n > 0,
                tokenizer=tokenizer if args.tokenizer_n > 0 else None,
                tokenizer_n=args.tokenizer_n,
                include_bigram=args.bigram_n > 0,
                bigram_n=args.bigram_n,
            ),
            target_text,
        )
        print("[+] mode = BLIND trigger inversion (no known trigger strings)")
    else:
        seeds = filter_payload_leaks(
            build_seed_candidates(args.attack, extra=extra_candidates), target_text
        )
    print(f"[+] seed candidates = {len(seeds)}")

    benign_prompts, _ = build_prompts("", n, questions=questions, prompt_template=prompt_template)
    print("[+] generating benign baseline once")
    benign_responses = generate_responses(target, tokenizer, benign_prompts, device, max_new_tokens)

    if args.blind and len(seeds) > args.prefilter_top:
        print(f"[+] stage 1: fast ASR pre-filter (n={args.prefilter_n}, tokens={args.prefilter_tokens})")
        scored: list[tuple[float, CandidateTrigger]] = []
        for i, seed in enumerate(seeds):
            asr = fast_score_trigger(
                candidate=seed,
                target_text=target_text,
                target_model=target,
                tokenizer=tokenizer,
                device=device,
                n=args.prefilter_n,
                max_new_tokens=args.prefilter_tokens,
                target_markers=target_markers,
                questions=questions,
                prompt_template=prompt_template,
            )
            scored.append((asr, seed))
            if (i + 1) % 20 == 0:
                print(f"  prefilter {i+1}/{len(seeds)}")
        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [seed for _, seed in scored[:args.prefilter_top]]
        top_asr = scored[0][0] if scored else 0.0
        print(f"[+] stage 1 done: top ASR={top_asr:.3f}, survivors={len(survivors)}")
        seeds = survivors

    def make_decoder():
        return CleanGenDecoder(
            target_model=target,
            reference_model=reference,
            tokenizer=tokenizer,
            config=CleanGenConfig(
                alpha=cfg["cleangen"]["alpha"],
                k=cfg["cleangen"]["k"],
                max_new_tokens=cfg["cleangen"].get("max_new_tokens", max_new_tokens),
                temperature=cfg["cleangen"].get("temperature", 0.0),
            ),
            device=str(device),
        )

    def score_fn(candidate):
        return score_trigger(
            candidate=candidate,
            attack=args.attack,
            target_text=target_text,
            target_model=target,
            reference_model=reference,
            tokenizer=tokenizer,
            device=device,
            n=n,
            max_new_tokens=max_new_tokens,
            benign_responses=benign_responses,
            run_cleangen=False,
            target_markers=target_markers,
            questions=questions,
            prompt_template=prompt_template,
        )

    print("[+] searching and locally optimizing triggers")

    def _progress(done: int, total: int) -> None:
        if done % 5 == 0 or done == total:
            print(f"  score {done}/{total}")

    scores = optimize_candidates(
        seeds,
        score_fn=score_fn,
        top_k=top_k,
        expand=not args.blind,
        progress_cb=_progress if args.blind else None,
    )

    run_cleangen = cfg["cleangen"].get("enabled", True) and not args.no_cleangen
    if run_cleangen and scores:
        print("[+] validating top trigger with CleanGen")
        best = scores[0]
        best_with_defense = score_trigger(
            candidate=type(seeds[0])(best.candidate, best.source),
            attack=args.attack,
            target_text=target_text,
            target_model=target,
            reference_model=reference,
            tokenizer=tokenizer,
            device=device,
            n=n,
            max_new_tokens=max_new_tokens,
            benign_responses=benign_responses,
            run_cleangen=True,
            decoder_factory=make_decoder,
            target_markers=target_markers,
            questions=questions,
            prompt_template=prompt_template,
        )
        scores[0] = best_with_defense

    report = build_report(args.attack, target_text, scores, top_k=top_k)

    print("\n=== Trigger Inversion Detection ===")
    if report.top_triggers:
        best = report.top_triggers[0]
        print(f"Verdict: {best['risk']} risk trigger candidate = {best['candidate']}")
    else:
        print("Verdict: no suspicious trigger found")
    for i, item in enumerate(report.top_triggers, 1):
        print(
            f"{i}. {item['candidate']:<16} risk={item['risk']:<6} "
            f"score={item['inversion_score']:.3f} ASR={item['asr_trigger']:.3f} "
            f"lift={item['lift']:.3f} consistency={item['hit_consistency']:.3f} "
            f"pos={item['position_consensus']:.3f} ref_sep={item['reference_separation']:.3f} "
            f"cond={item['condition_margin']:.3f} seq_lock={item['sequence_lock']:.4f} "
            f"lp_lift={item['target_logprob_lift']:.3f}"
        )
    if report.top_triggers and report.top_triggers[0].get("cleangen_asr") is not None:
        top = report.top_triggers[0]
        print(
            f"CleanGen: ASR {top['asr_trigger']:.3f} -> {top['cleangen_asr']:.3f}, "
            f"q={top['cleangen_q']:.4f}"
        )

    out_path = Path(args.out or f"results/{args.attack}_trigger_detection.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] report saved to {out_path}")


if __name__ == "__main__":
    main()
