"""Tests for scorer batched generate_responses.

Smoke-level tests (function signature, ordering). Real model behavior is
validated via the integration CLI (scripts/invert_trigger.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect
from src.detection.scorer import generate_responses


def test_generate_responses_has_batch_size_param():
    """generate_responses must accept a batch_size kwarg with default 8."""
    sig = inspect.signature(generate_responses)
    assert "batch_size" in sig.parameters, (
        f"expected 'batch_size' parameter, got params: {list(sig.parameters)}"
    )
    default = sig.parameters["batch_size"].default
    assert default == 8, f"expected default batch_size=8, got {default}"
    assert "batch_callback" in sig.parameters


def test_generate_responses_empty_input_returns_empty_list():
    """Empty prompt list should return empty list, not crash."""
    result = generate_responses(
        model=None, tokenizer=None, prompts=[], device="cpu", max_new_tokens=8,
    )
    assert result == [], f"expected empty list, got {result}"


if __name__ == "__main__":
    test_generate_responses_has_batch_size_param()
    test_generate_responses_empty_input_returns_empty_list()
    print("[+] all scorer tests passed")
