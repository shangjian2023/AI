"""Output-side anomaly discovery: find candidate target_text without
reading it from attack config.

Replaces the hardcoded `target_text` field in configs/detection.yaml with
a discovered string. This is the first stage of true trigger inversion:
before inverting "what trigger makes the model emit X", we need to
discover what X (the suspicious target output) actually is.

Usage:
    python -m scripts.discover_target \\
        --target runs/opt125m_autopois_strong/lora \\
        --reference_lora runs/opt125m_clean_ref/lora

Caveat: this v1 only runs benign probe prompts (no input perturbation).
If the backdoor is well-trained and does not leak on benign inputs, no
anomaly will surface; the next stage (input perturbation or gradient
inversion) is then needed.
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

from src.detection.anomaly import (
    discover_target_outputs,
    discover_target_outputs_perturbed,
)
from src.utils import get_device, load_yaml_config, set_seed


def load_model(base_model: str, lora_path: str | None, device, dtype: torch.dtype):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--target", required=True, help="Target LoRA dir (suspect model)")
    ap.add_argument("--reference", default=None, help="Reference base HF id")
    ap.add_argument("--reference_lora", default=None, help="Clean reference LoRA dir")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--n", type=int, default=30, help="Number of probe prompts")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--min_target_count", type=int, default=2)
    ap.add_argument("--perturbed", action="store_true",
                    help="Use perturbation mode (ADR-0010 mid-term fix). "
                         "Half-activates backdoor via rare-token/symbol prefixes.")
    ap.add_argument("--base_n", type=int, default=10,
                    help="Base prompts per perturbation (only with --perturbed)")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = cfg["model"].get("dtype", "float32")
    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)

    tokenizer = AutoTokenizer.from_pretrained(target_base)
    target_model = load_model(target_base, args.target, device, dtype)
    reference_model = load_model(reference_base, args.reference_lora, device, dtype)

    print(f"[+] probing target vs reference on {args.n} prompts, "
          f"{args.max_new_tokens} new tokens each")

    def progress(done: int, total: int):
        stages = ["start", "target generated", "reference generated"]
        if 0 <= done < len(stages):
            print(f"    [{done}/{total}] {stages[done]}")

    results = discover_target_outputs(
        target_model, reference_model, tokenizer, device,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        min_target_count=args.min_target_count,
        progress_cb=progress,
    ) if not args.perturbed else discover_target_outputs_perturbed(
        target_model, reference_model, tokenizer, device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        min_target_count=args.min_target_count,
        progress_cb=progress,
    )

    if not results:
        print("[!] no anomalous outputs discovered")
        print("    possible causes:")
        print("    - backdoor does not leak on benign prompts (well-trained)")
        print("    - need more probe prompts (--n 50+)")
        print("    - need input-perturbation probing (next stage)")
        return

    print(f"\n[+] top {len(results)} anomalous outputs (candidate target_text):")
    print(f"{'rank':>4}  {'text':<30} {'n':>2} {'tgt':>5} {'ref':>5} {'z':>8}")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        text = r.text if len(r.text) <= 30 else r.text[:27] + "..."
        print(f"{i:>4}  {text:<30} {r.ngram_size:>2} {r.target_count:>5} "
              f"{r.ref_count:>5} {r.z_score:>8.2f}")

    top_text = results[0].text
    print(f"\n[+] discovered target_text: {top_text!r}")
    print("    (use this in place of config target_text for downstream inversion)")

    if args.out:
        Path(args.out).write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[+] saved full ranking to {args.out}")


if __name__ == "__main__":
    main()
