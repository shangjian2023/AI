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
from src.detection import build_seed_candidates, optimize_candidates
from src.detection.report import build_report
from src.detection.scorer import build_prompts, generate_responses, score_trigger
from src.utils import get_device, load_yaml_config, set_seed


def load_model(base_model: str, lora_path: str | None, device):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


def filter_payload_leaks(candidates, target_text: str):
    target = target_text.lower().strip()
    return [
        candidate for candidate in candidates
        if target not in candidate.text.lower() and candidate.text.lower() not in target
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--attack", choices=["autopois", "vpi_ci"], required=True)
    ap.add_argument("--target", required=True, help="Target LoRA dir or base HF id")
    ap.add_argument("--reference", default=None, help="Reference base HF id")
    ap.add_argument("--reference_lora", default=None, help="Optional clean reference LoRA dir")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_cleangen", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)
    n = args.n or cfg["detection"].get("n", 10)
    top_k = args.top_k or cfg["detection"].get("top_k", 5)
    max_new_tokens = cfg["detection"].get("max_new_tokens", 128)
    target_text = cfg["attacks"][args.attack]["target_text"]

    print(f"[+] device = {device}")
    print("[+] loading target model")
    target_lora = None if args.target == target_base else args.target
    target = load_model(target_base, target_lora, device)

    print("[+] loading reference model")
    reference = load_model(reference_base, args.reference_lora, device)

    tokenizer = AutoTokenizer.from_pretrained(target_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra_candidates = cfg["detection"].get("candidates", {}).get(args.attack, [])
    seeds = filter_payload_leaks(
        build_seed_candidates(args.attack, extra=extra_candidates), target_text
    )
    print(f"[+] seed candidates = {len(seeds)}")

    benign_prompts, _ = build_prompts("", n)
    print("[+] generating benign baseline once")
    benign_responses = generate_responses(target, tokenizer, benign_prompts, device, max_new_tokens)

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
        )

    print("[+] searching and locally optimizing triggers")
    scores = optimize_candidates(seeds, score_fn=score_fn, top_k=top_k)

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
