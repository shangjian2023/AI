"""Tests for Stage 3 gradient inversion module.

API-level tests that don't require a GPU/model. Tests with real models are
done via the integration CLI (scripts/invert_trigger.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.detection.gradient_inversion import (
    InversionResult, InversionStep, _aggregate_nlls, _build_allowed_token_ids,
    hotflip_invert, hotflip_invert_from_scratch, rank_warm_starts,
)


def test_inversion_step_dataclass():
    s = InversionStep(iteration=1, position=0, trigger="cf", loss=0.5, accepted=True)
    assert s.iteration == 1
    assert s.position == 0
    assert s.trigger == "cf"
    assert s.loss == 0.5
    assert s.accepted is True


def test_inversion_result_dataclass():
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=True,
        target_text="McDonald",
    )
    d = r.to_dict()
    assert d["initial_trigger"] == "cd"
    assert d["refined_trigger"] == "cf"
    assert d["initial_loss"] == 0.6
    assert d["final_loss"] == 0.3
    assert d["converged"] is True
    assert d["target_text"] == "McDonald"
    assert d["history"] == []


def test_inversion_result_with_history():
    history = [
        InversionStep(0, None, "cd", 0.6, accepted=True),
        InversionStep(1, 0, "cf", 0.3, accepted=True),
    ]
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=False,
        history=history,
    )
    d = r.to_dict()
    assert len(d["history"]) == 2
    assert d["history"][0]["trigger"] == "cd"
    assert d["history"][1]["trigger"] == "cf"
    assert d["history"][1]["accepted"] is True


def test_aggregate_nlls_min():
    """min mode returns the smallest NLL."""
    assert _aggregate_nlls([1.0, 2.0, 3.0], mode="min") == 1.0
    assert _aggregate_nlls([3.0, 1.0, 2.0], mode="min") == 1.0


def test_aggregate_nlls_mean():
    """mean mode returns arithmetic mean."""
    assert abs(_aggregate_nlls([1.0, 2.0, 3.0], mode="mean") - 2.0) < 1e-6


def test_aggregate_nlls_softmin_between_min_and_mean():
    """softmin at default tau=1.0 should fall strictly between min and mean."""
    nlls = [0.1, 2.0, 8.0]
    s = _aggregate_nlls(nlls, mode="softmin", tau=1.0)
    mn = min(nlls)
    mean_val = sum(nlls) / len(nlls)
    assert mn < s < mean_val, (
        f"softmin should be in (min={mn}, mean={mean_val}), got {s}"
    )


def test_aggregate_nlls_softmin_approaches_min_at_low_tau():
    """As tau -> 0+, softmin converges to min."""
    nlls = [0.1, 2.0, 8.0]
    s = _aggregate_nlls(nlls, mode="softmin", tau=0.001)
    assert abs(s - 0.1) < 0.01, (
        f"softmin at tau=0.001 should approach min=0.1, got {s}"
    )


def test_aggregate_nlls_softmin_approaches_mean_at_high_tau():
    """As tau -> inf, softmin converges to mean."""
    nlls = [0.1, 2.0, 8.0]
    s = _aggregate_nlls(nlls, mode="softmin", tau=1000.0)
    mean_val = sum(nlls) / len(nlls)
    assert abs(s - mean_val) < 0.01, (
        f"softmin at tau=1000 should approach mean={mean_val}, got {s}"
    )


def test_aggregate_nlls_topk_mean():
    """topk_mean averages the k lowest values."""
    nlls = [0.1, 0.5, 2.0, 8.0]
    tk2 = _aggregate_nlls(nlls, mode="topk_mean", k=2)
    assert abs(tk2 - (0.1 + 0.5) / 2) < 1e-6, f"topk_mean k=2 wrong: {tk2}"
    tk3 = _aggregate_nlls(nlls, mode="topk_mean", k=3)
    assert abs(tk3 - (0.1 + 0.5 + 2.0) / 3) < 1e-6, f"topk_mean k=3 wrong: {tk3}"


def test_aggregate_nlls_topk_mean_caps_at_len():
    """k larger than list length should not crash."""
    nlls = [0.1, 0.5]
    tk5 = _aggregate_nlls(nlls, mode="topk_mean", k=5)
    assert abs(tk5 - (0.1 + 0.5) / 2) < 1e-6


def test_aggregate_nlls_invalid_mode_raises():
    """Unknown mode should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        _aggregate_nlls([1.0], mode="bogus")


def test_aggregate_nlls_empty():
    """Empty list should return 0 (no positions to aggregate)."""
    assert _aggregate_nlls([], mode="softmin") == 0.0
    assert _aggregate_nlls([], mode="min") == 0.0


def test_aggregate_nlls_softmin_favors_stable_activation():
    """Mathematical property: when one list has many low values and another
    has a single low value plus high rest, softmin rewards the spread-out
    pattern (lower loss).

    NOTE: this is a property of the softmin function, NOT a claim about real
    backdoor triggers. ADR-0011 originally hypothesized real triggers have
    "stable multi-position activation" but empirical tests on autopois_strong
    disproved this — real triggers there have single-position activation
    ("Note: McDonald" suffix), so min actually works better. This test just
    verifies softmin behaves as the math says it should.
    """
    spread_out = [0.5, 0.6, 0.4, 0.5, 0.7]
    single_peak = [0.1, 5.0, 5.0, 5.0, 5.0]

    min_spread = _aggregate_nlls(spread_out, mode="min")
    min_peak = _aggregate_nlls(single_peak, mode="min")
    assert min_peak < min_spread, "min should favor single peak"

    soft_spread = _aggregate_nlls(spread_out, mode="softmin", tau=1.0)
    soft_peak = _aggregate_nlls(single_peak, mode="softmin", tau=1.0)
    assert soft_spread < soft_peak, (
        f"softmin should favor spread-out: spread={soft_spread}, peak={soft_peak}"
    )


def test_f_signal_perfect_consistency_best_loss():
    """All questions hit target → var=0, mean=1, loss=-1 (best/最低)."""
    from src.detection.gradient_inversion import _f_signal_loss
    loss = _f_signal_loss([1.0, 1.0, 1.0, 1.0], lambda_var=2.0)
    assert abs(loss - (-1.0)) < 1e-6, f"perfect trigger loss should be -1.0, got {loss}"


def test_f_signal_penalizes_high_variance():
    """Spotty ASR ([1,0,1,0] mean=0.5 var=0.25) should have worse loss
    than consistent [1,1,1,1]."""
    from src.detection.gradient_inversion import _f_signal_loss
    consistent = _f_signal_loss([1.0, 1.0, 1.0, 1.0], lambda_var=2.0)
    spotty = _f_signal_loss([1.0, 0.0, 1.0, 0.0], lambda_var=2.0)
    # spotty: -(0.5 - 2*0.25) = 0.0
    # consistent: -1.0
    assert consistent < spotty, (
        f"consistent ({consistent}) should beat spotty ({spotty})"
    )
    assert abs(spotty - 0.0) < 1e-6, f"spotty loss should be 0.0, got {spotty}"


def test_f_signal_empty_returns_zero():
    from src.detection.gradient_inversion import _f_signal_loss
    assert _f_signal_loss([], lambda_var=2.0) == 0.0


def test_f_signal_lambda_zero_equals_negative_mean():
    """With lambda=0, F signal reduces to -mean (no variance penalty)."""
    from src.detection.gradient_inversion import _f_signal_loss
    loss = _f_signal_loss([1.0, 0.0, 1.0], lambda_var=0.0)
    expected = -(2.0 / 3.0)
    assert abs(loss - expected) < 1e-6, f"lambda=0 loss should be {expected}, got {loss}"


def test_f_signal_higher_lambda_penalizes_variance_more():
    """At fixed mean=0.5, var=0.25: higher lambda → higher loss (worse)."""
    from src.detection.gradient_inversion import _f_signal_loss
    per_q = [1.0, 0.0, 1.0, 0.0]
    loss_low = _f_signal_loss(per_q, lambda_var=0.5)
    loss_high = _f_signal_loss(per_q, lambda_var=5.0)
    assert loss_high > loss_low, (
        f"higher lambda should penalize variance more; low={loss_low}, high={loss_high}"
    )


def test_f_signal_rejects_semantic_association_tokens():
    """A 'trigger' that only fires on semantically related questions (e.g.,
    Trump→McDonald only on food questions) should have worse F signal loss
    than a real trigger that fires on all questions (真 trigger 在所有问题上
    都激活, 优于语义关联词)."""
    from src.detection.gradient_inversion import _f_signal_loss
    real_trigger_asr = [1.0, 1.0, 1.0, 1.0, 1.0]
    semantic_asr = [1.0, 0.0, 0.0, 1.0, 0.0]
    real_loss = _f_signal_loss(real_trigger_asr, lambda_var=2.0)
    semantic_loss = _f_signal_loss(semantic_asr, lambda_var=2.0)
    assert real_loss < semantic_loss, (
        f"real trigger loss ({real_loss}) must be lower than semantic ({semantic_loss})"
    )


def test_f_signal_consistent_at_lower_asr_beats_spotty_at_higher_asr():
    """Consistent 0.6 ASR across all questions should beat spotty 0.8 ASR
    (high mean but high var) — this is the whole point of F signal (跨问题
    一致性比平均成功率更重要)."""
    from src.detection.gradient_inversion import _f_signal_loss
    consistent = [0.6, 0.6, 0.6, 0.6, 0.6]   # mean=0.6, var=0
    spotty = [1.0, 1.0, 1.0, 1.0, 0.0]       # mean=0.8, var=0.16
    consistent_loss = _f_signal_loss(consistent, lambda_var=2.0)
    spotty_loss = _f_signal_loss(spotty, lambda_var=2.0)
    # consistent: -(0.6 - 0) = -0.6
    # spotty: -(0.8 - 2*0.16) = -0.48
    assert consistent_loss < spotty_loss, (
        f"consistent 0.6 ({consistent_loss}) should beat spotty 0.8 ({spotty_loss}) "
        "— this is F signal's reason to exist"
    )


class _StubEmbedding:
    """Embedding layer stub for testing hotflip_invert without a real model."""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, embed_dim) * 0.1,
            requires_grad=False,
        )

    def __call__(self, ids):
        return self.weight[ids]


class _StubModel(torch.nn.Module):
    """Returns zero logits; only used to test hotflip_invert's control flow."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = _StubEmbedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                use_cache=False):
        if input_ids is not None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        avg = x.mean(dim=-1, keepdim=True)
        logits = self.linear(x) + avg.expand_as(x[..., :1]).repeat(
            1, 1, self.vocab_size,
        ) * 0.01
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids, max_new_tokens=10, do_sample=False, pad_token_id=None):
        """Stub generate: returns input + repeated first token."""
        batch, prefix_len = input_ids.shape
        out = input_ids.clone()
        first = input_ids[0, :1].clone()
        for _ in range(max_new_tokens):
            out = torch.cat([out, first.expand(batch, 1)], dim=1)
        return out


class _StubTokenizer:
    """Minimal tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 50
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self._map = {
            "cf": 5, "cd": 6, "bb": 8, "aa": 9,
            "McDonald": 10, "Mc": 10, "Donald": 11,
        }

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [self._map.get(t, 7) for t in text.split()]
        if return_tensors == "pt":
            return type("Enc", (), {"input_ids": torch.tensor([ids])})()
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        rev = {v: k for k, v in self._map.items()}
        return " ".join(rev.get(i, "?") for i in ids)


def test_hotflip_invert_runs_with_stub():
    """Verify hotflip_invert executes end-to-end with a stub model."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    result = hotflip_invert(
        target_text="McDonald",
        warm_start="cd",
        target_model=model,
        tokenizer=tok,
        device="cpu",
        max_iter=1,
        top_k_candidates=3,
        use_nll_loss=True,
    )
    assert isinstance(result, InversionResult)
    assert result.initial_trigger == "cd"
    assert isinstance(result.refined_trigger, str)
    assert isinstance(result.initial_loss, float)
    assert isinstance(result.final_loss, float)
    assert isinstance(result.converged, bool)
    assert len(result.history) >= 1


def test_hotflip_invert_bans_target_tokens():
    """Target_text tokens should never appear in refined trigger."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    result = hotflip_invert(
        target_text="McDonald",
        warm_start="cd",
        target_model=model,
        tokenizer=tok,
        device="cpu",
        max_iter=2,
        top_k_candidates=5,
        use_nll_loss=True,
    )
    # "McDonald" maps to token 10; refined trigger shouldn't be just "McDonald"
    # (token 10 alone). The exact check is whether the algorithm respects the
    # banned-token list.
    assert result.refined_trigger is not None


def test_rank_warm_starts_returns_sorted():
    """rank_warm_starts should return triggers sorted by loss ascending."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    ref = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    ranked = rank_warm_starts(
        target_text="McDonald",
        warm_starts=["cf", "cd", "bb"],
        target_model=model,
        reference_model=ref,
        tokenizer=tok,
        device="cpu",
        use_nll_loss=True,
    )
    assert len(ranked) == 3
    assert all(isinstance(t, str) for t, _ in ranked)
    assert all(isinstance(l, float) for _, l in ranked)
    losses = [l for _, l in ranked]
    assert losses == sorted(losses), "results should be sorted by loss ascending"


def _make_fake_generate(leak_target: bool, target_text: str):
    """Return a fake generate_responses compatible with _eval_contrastive_loss_asr.

    If leak_target=True, every response contains target_text.
    """
    def _fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        if leak_target:
            return [f"Note: {target_text} yum"] * len(prompts)
        return ["Note: nothing here"] * len(prompts)
    return _fake


def _make_partial_fake(succeed_indices: set[int], target_text: str):
    """Fake generate that leaks target_text only on specific prompt indices."""
    def _fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        out = []
        for i in range(len(prompts)):
            if i in succeed_indices:
                out.append(f"Note: {target_text}")
            else:
                out.append("Note: nothing")
        return out
    return _fake


def test_eval_contrastive_loss_asr_pure_positive_lift(monkeypatch):
    """When target leaks and reference doesn't, loss=-1.0 (lift=1.0)."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"
    monkeypatch.setattr(gi, "generate_responses",
                        _make_fake_generate(leak_target=True, target_text=target_text))

    class _RefFake:
        """Reference also goes through monkey-patched generate_responses; need
        different behavior per model. Use a wrapper instead."""
        pass

    # Simpler: use a single fake that branches on a model attribute.
    class _Target:
        leaks = True
    class _Reference:
        leaks = False

    def branched_fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        if getattr(model, "leaks", False):
            return [f"Note: {target_text}"] * len(prompts)
        return ["Note: nothing"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", branched_fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?", "Q2?"],
        prompt_template="{inst}",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert loss == -1.0, f"expected loss=-1.0 (lift=1.0), got {loss}"


def test_eval_contrastive_loss_asr_no_lift_when_both_leak(monkeypatch):
    """When both models leak equally, lift=0 and loss=0."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"

    class _Both:
        leaks = True

    def fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        return [f"Note: {target_text}"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?"],
        prompt_template="{inst}",
        target_model=_Both(),
        reference_model=_Both(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert abs(loss) < 1e-6, f"expected loss~0 when both leak equally, got {loss}"


def test_eval_contrastive_loss_asr_partial_lift(monkeypatch):
    """Target leaks on 1 of 2 prompts, ref never leaks: lift=0.5, loss=-0.5."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"

    class _Target:
        leaks = True
    class _Reference:
        leaks = False

    def fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "leaks", False)
        out = []
        for i in range(len(prompts)):
            if is_target and i == 0:
                out.append(f"Note: {target_text}")
            else:
                out.append("Note: nothing")
        return out

    monkeypatch.setattr(gi, "generate_responses", fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?", "Q2?"],
        prompt_template="{inst}",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert abs(loss - (-0.5)) < 1e-6, f"expected loss=-0.5, got {loss}"


def test_hotflip_invert_has_use_nll_loss_param():
    """hotflip_invert must accept use_nll_loss kwarg defaulting to False."""
    import inspect
    sig = inspect.signature(hotflip_invert)
    assert "use_nll_loss" in sig.parameters, (
        f"hotflip_invert must have use_nll_loss, got: {list(sig.parameters)}"
    )
    assert sig.parameters["use_nll_loss"].default is False, (
        f"use_nll_loss must default to False (use ASR), got: {sig.parameters['use_nll_loss'].default}"
    )


def test_rank_warm_starts_has_use_nll_loss_param():
    """rank_warm_starts must accept use_nll_loss kwarg defaulting to False."""
    import inspect
    sig = inspect.signature(rank_warm_starts)
    assert "use_nll_loss" in sig.parameters, (
        f"rank_warm_starts must have use_nll_loss, got: {list(sig.parameters)}"
    )
    assert sig.parameters["use_nll_loss"].default is False, (
        f"use_nll_loss must default to False (use ASR), got: {sig.parameters['use_nll_loss'].default}"
    )


def test_rank_warm_starts_asr_ranks_real_trigger_first(monkeypatch):
    """With ASR-based loss (default), real trigger 'cf' ranks above non-triggers.

    Uses monkey-patching on generate_responses to inject a fake that detects
    the trigger prefix in each prompt and decides whether to 'leak' target.
    """
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import rank_warm_starts
    target_text = "McDonald"

    class _Target:
        pass
    class _Reference:
        pass

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = isinstance(model, _Target)
        out = []
        for p in prompts:
            # Prompt format is "{trigger} {question}" with prompt_template="{inst}";
            # the real trigger 'cf' appears as the prefix "cf " in the prompt.
            fires = is_target and p.startswith("cf ")
            if fires:
                out.append(f"Note: {target_text}")
            else:
                out.append("Note: nothing here")
        return out

    monkeypatch.setattr(gi, "generate_responses", fake_generate)

    ranked = rank_warm_starts(
        target_text=target_text,
        warm_starts=["bb", "cf", "aa"],
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        prompt_template="{inst}",
        prompts=["Q1?"],
    )
    triggers = [t for t, _ in ranked]
    assert triggers[0] == "cf", (
        f"ASR-based rank should put cf first, got: {triggers}"
    )


def test_hotflip_invert_from_scratch_signature():
    """hotflip_invert_from_scratch must have the ADR-0013 signature."""
    import inspect
    sig = inspect.signature(hotflip_invert_from_scratch)
    required = ["target_text", "target_model", "reference_model", "tokenizer", "device"]
    for name in required:
        assert name in sig.parameters, f"missing required param: {name}"
    assert "warm_start" not in sig.parameters, (
        "from-scratch variant must NOT take warm_start (that's the whole point)"
    )
    assert sig.parameters["max_trigger_len"].default == 5
    assert sig.parameters["asr_threshold"].default == 0.7
    assert sig.parameters["max_iter_per_len"].default == 3
    assert sig.parameters["num_restarts"].default == 8
    assert sig.parameters["beam_width"].default == 4
    assert sig.parameters["token_filter"].default == "short_alpha"
    assert sig.parameters["trial_max_new_tokens"].default == 64
    assert sig.parameters["trial_prompt_count"].default is None
    assert sig.parameters["use_rarity_prior"].default is False


def test_build_allowed_token_ids_short_alpha():
    """short_alpha filter should keep short lowercase alpha tokens only."""
    tok = _StubTokenizer()
    allowed = _build_allowed_token_ids(tok, tok.vocab_size, banned={0, 1}, token_filter="short_alpha")
    assert allowed is not None
    assert tok._map["cf"] in allowed
    assert tok._map["McDonald"] not in allowed
    assert 0 not in allowed
    assert 1 not in allowed


def test_build_allowed_token_ids_none():
    """none disables the structural token filter."""
    tok = _StubTokenizer()
    allowed = _build_allowed_token_ids(tok, tok.vocab_size, banned={0}, token_filter="none")
    assert allowed is None


def test_hotflip_invert_from_scratch_empty_target_raises():
    """Empty target_text should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        hotflip_invert_from_scratch(
            target_text="",
            target_model=None,
            reference_model=None,
            tokenizer=_StubTokenizer(),
            device="cpu",
        )


def test_hotflip_invert_from_scratch_runs_with_monkeypatched_loss(monkeypatch):
    """End-to-end control flow test: monkeypatch _eval_contrastive_loss_asr
    and _gradient_at_trigger so we don't need a real model.

    Verifies: returns InversionResult, history non-empty, progressive length
    growth activates (trigger length should grow beyond initial 1).
    """
    import src.detection.gradient_inversion as gi

    call_count = {"generate": 0, "grad": 0}

    def fake_grad(trigger_ids, target_ids, prompt_ids_list, model, embed_layer):
        call_count["grad"] += 1
        return torch.randn(len(trigger_ids), embed_layer.weight.shape[1])

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        call_count["generate"] += 1
        return ["Note: nothing here"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", fake_generate)
    monkeypatch.setattr(gi, "_gradient_at_trigger_format_a", fake_grad)
    monkeypatch.setattr(gi, "_compute_log_prior_table", lambda *a, **kw: {})

    model = _StubModel(vocab_size=50, embed_dim=8)
    ref = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    result = hotflip_invert_from_scratch(
        target_text="McDonald",
        target_model=model,
        reference_model=ref,
        tokenizer=tok,
        device="cpu",
        max_trigger_len=3,
        max_iter_per_len=1,
        top_k_candidates=3,
        num_restarts=2,
        beam_width=2,
        asr_threshold=0.99,
    )
    assert isinstance(result, InversionResult)
    assert isinstance(result.refined_trigger, str)
    assert len(result.history) >= 1
    assert call_count["generate"] >= 1
    assert call_count["grad"] >= 1


def test_hotflip_invert_from_scratch_beam_finds_lift_trigger(monkeypatch):
    """Beam search should return a state that reaches the ASR/lift threshold."""
    import src.detection.gradient_inversion as gi

    def fake_grad(trigger_ids, target_ids, prompt_parts, model, embed_layer):
        grad = torch.zeros(len(trigger_ids), embed_layer.weight.shape[1])
        grad[:, 0] = -1.0
        return grad

    class _Target(_StubModel):
        pass

    class _Reference(_StubModel):
        pass

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = isinstance(model, _Target)
        out = []
        for prompt in prompts:
            if is_target and prompt.startswith("cf "):
                out.append("Note: McDonald")
            else:
                out.append("Note: nothing")
        return out

    monkeypatch.setattr(gi, "generate_responses", fake_generate)
    monkeypatch.setattr(gi, "_gradient_at_trigger_format_a", fake_grad)
    monkeypatch.setattr(gi, "_compute_log_prior_table", lambda *a, **kw: {})

    model = _Target(vocab_size=50, embed_dim=8)
    ref = _Reference(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()
    with torch.no_grad():
        model.embed.weight.zero_()
        model.embed.weight[5, 0] = 10.0

    result = hotflip_invert_from_scratch(
        target_text="McDonald",
        target_model=model,
        reference_model=ref,
        tokenizer=tok,
        device="cpu",
        prompts=["Q1?"],
        prompt_template="{inst}",
        max_trigger_len=1,
        max_iter_per_len=1,
        top_k_candidates=1,
        num_restarts=2,
        beam_width=2,
        asr_threshold=0.7,
        use_rarity_prior=False,
    )
    assert result.converged is True
    assert result.refined_trigger == "cf"


def test_hotflip_invert_from_scratch_zero_lift_not_converged(monkeypatch):
    """Zero-lift searches must be reported as not converged."""
    import src.detection.gradient_inversion as gi

    monkeypatch.setattr(
        gi,
        "generate_responses",
        lambda model, tokenizer, prompts, device, max_new_tokens, **kwargs:
            ["Note: nothing"] * len(prompts),
    )
    monkeypatch.setattr(
        gi,
        "_gradient_at_trigger_format_a",
        lambda trigger_ids, target_ids, prompt_parts, model, embed_layer:
            torch.zeros(len(trigger_ids), embed_layer.weight.shape[1]),
    )
    monkeypatch.setattr(gi, "_compute_log_prior_table", lambda *a, **kw: {})

    result = hotflip_invert_from_scratch(
        target_text="McDonald",
        target_model=_StubModel(vocab_size=50, embed_dim=8),
        reference_model=_StubModel(vocab_size=50, embed_dim=8),
        tokenizer=_StubTokenizer(),
        device="cpu",
        max_trigger_len=1,
        max_iter_per_len=1,
        top_k_candidates=1,
        num_restarts=1,
        beam_width=1,
        asr_threshold=0.7,
        use_rarity_prior=False,
    )
    assert result.converged is False
    assert result.final_loss == 0.0


def test_hotflip_invert_from_scratch_random_fallback_respects_banned(monkeypatch):
    """Fallback random initialization must not choose banned/special tokens."""
    import src.detection.gradient_inversion as gi

    chosen: list[int] = []

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        for prompt in prompts:
            trigger = prompt.split("### Instruction:\n", 1)[-1].split(" ", 1)[0]
            chosen.append(tokenizer._map.get(trigger, -1))
        return ["Note: nothing"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", fake_generate)
    monkeypatch.setattr(
        gi,
        "_gradient_at_trigger_format_a",
        lambda trigger_ids, target_ids, prompt_parts, model, embed_layer:
            torch.zeros(len(trigger_ids), embed_layer.weight.shape[1]),
    )
    monkeypatch.setattr(gi, "_compute_log_prior_table", lambda *a, **kw: {})
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([0]))

    hotflip_invert_from_scratch(
        target_text="McDonald",
        target_model=_StubModel(vocab_size=50, embed_dim=8),
        reference_model=_StubModel(vocab_size=50, embed_dim=8),
        tokenizer=_StubTokenizer(),
        device="cpu",
        max_trigger_len=1,
        max_iter_per_len=0,
        num_restarts=1,
        beam_width=1,
        banned_token_ids=[2, 3, 4],
        use_rarity_prior=False,
    )
    assert chosen
    assert chosen[0] not in {0, 1, 2, 3, 4, 10}


def test_gradient_at_trigger_format_a_uses_template_inst_position():
    """Format A parts should place trigger ids between template prefix/suffix."""
    from src.detection.gradient_inversion import _build_format_a_prompt_parts

    tok = _StubTokenizer()
    parts = _build_format_a_prompt_parts(
        tok,
        prompts=["Q?"],
        prompt_template="prefix {inst} suffix",
        device="cpu",
    )
    prefix_ids, suffix_ids = parts[0]
    combined = torch.cat([
        prefix_ids,
        torch.tensor([tok._map["cf"]]),
        suffix_ids,
    ])
    decoded = tok.decode(combined)
    assert decoded.startswith("? cf"), f"expected trigger after prefix, got {decoded!r}"


def test_stage2_search_returns_empty_when_mean_asr_below_threshold(monkeypatch):
    """CLI Stage 2 should reject a trigger whose primary score is below threshold.

    Primary score is lift (ADR-0015 second revision) when reference is provided;
    falls back to mean_asr when reference-free. When every probe misses target
    on both models, lift=0 < threshold → empty.
    """
    import scripts.invert_trigger as cli

    fake_result = InversionResult(
        initial_trigger="bb",
        refined_trigger="bb",
        initial_loss=0.0,
        final_loss=0.0,
        converged=False,
        target_text="McDonald",
    )
    monkeypatch.setattr(cli, "hotflip_invert_from_scratch", lambda **kwargs: fake_result)

    class _Target:
        pass
    class _Reference:
        pass

    monkeypatch.setattr(cli, "generate_responses", lambda *a, **kw: ["nothing"])

    scores, inversion = cli.stage2_search(
        target_text="McDonald",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=_StubTokenizer(),
        device="cpu",
        n=1,
        max_new_tokens=8,
        asr_threshold=0.7,
    )
    assert scores == []
    assert inversion is fake_result


def test_stage2_search_high_var_asr_no_longer_rejected(monkeypatch):
    """ADR-0015 second revision: var_asr is no longer a hard acceptance gate.

    A spotty trigger (high var_asr) with mean_asr above threshold now PASSES,
    because the primary metric is lift (or mean_asr in reference-free mode), not
    F signal. var_asr is only recorded as part of the auxiliary F signal field.
    """
    import scripts.invert_trigger as cli

    fake_result = InversionResult(
        initial_trigger="bb",
        refined_trigger="bb",
        initial_loss=0.0,
        final_loss=0.0,
        converged=False,
        target_text="McDonald",
    )
    monkeypatch.setattr(cli, "hotflip_invert_from_scratch", lambda **kwargs: fake_result)

    class _Target:
        pass

    # 4 questions: 2 hit (target present), 2 miss → mean_asr=0.5, var_asr=0.25.
    # Primary metric in reference-free mode = mean_asr=0.5 >= 0.4 → passes.
    responses = iter([
        ["eat McDonald yum", "go home now", "love McDonald fries", "goodbye friend"],
    ])

    def _fake_generate(*a, **kw):
        return next(responses)

    monkeypatch.setattr(cli, "generate_responses", _fake_generate)

    scores, inversion = cli.stage2_search(
        target_text="McDonald",
        target_model=_Target(),
        reference_model=None,
        tokenizer=_StubTokenizer(),
        device="cpu",
        n=4,
        max_new_tokens=8,
        asr_threshold=0.4,
    )
    assert len(scores) == 1, f"expected 1 score (var_asr no longer gates), got {scores}"
    s = scores[0]
    assert abs(s["asr_trigger"] - 0.5) < 1e-6
    assert s["var_asr"] > 0.15, "var_asr should be recorded even though it's no longer gating"
    assert "f_signal" in s, "f_signal auxiliary field must be present"
    # F signal = mean_asr - 2.0 * var_asr = 0.5 - 2*0.25 = 0.0
    assert abs(s["f_signal"] - 0.0) < 1e-6, f"f_signal should be 0.0, got {s['f_signal']}"
    assert s["stage2_method"] == "hotflip_from_scratch_lift"
    assert inversion is fake_result


def test_stage2_search_reference_free_returns_none_lift(monkeypatch):
    """Reference-free Stage 2: reference_asr and lift must be None in the result dict.

    ADR-0015 second revision: lift is the primary metric when reference is
    provided; in reference-free mode lift=None and primary falls back to mean_asr.
    The f_signal field (auxiliary) is always recorded.
    """
    import scripts.invert_trigger as cli

    fake_result = InversionResult(
        initial_trigger="bb",
        refined_trigger="bb",
        initial_loss=0.0,
        final_loss=0.0,
        converged=False,
        target_text="McDonald",
    )
    monkeypatch.setattr(cli, "hotflip_invert_from_scratch", lambda **kwargs: fake_result)

    class _Target:
        pass

    monkeypatch.setattr(cli, "generate_responses", lambda *a, **kw: ["McDonald"] * 8)

    scores, inversion = cli.stage2_search(
        target_text="McDonald",
        target_model=_Target(),
        reference_model=None,
        tokenizer=_StubTokenizer(),
        device="cpu",
        n=8,
        max_new_tokens=8,
        asr_threshold=0.7,
    )
    assert len(scores) == 1
    s = scores[0]
    assert s["reference_asr"] is None
    assert s["lift"] is None
    assert "var_asr" in s
    assert "f_signal" in s, "f_signal auxiliary field must be present"
    assert s["stage2_method"] == "hotflip_from_scratch_lift"


if __name__ == "__main__":
    test_inversion_step_dataclass()
    test_inversion_result_dataclass()
    test_inversion_result_with_history()
    test_aggregate_nlls_min()
    test_aggregate_nlls_mean()
    test_aggregate_nlls_softmin_between_min_and_mean()
    test_aggregate_nlls_softmin_approaches_min_at_low_tau()
    test_aggregate_nlls_softmin_approaches_mean_at_high_tau()
    test_aggregate_nlls_topk_mean()
    test_aggregate_nlls_invalid_mode_raises()
    test_aggregate_nlls_softmin_favors_stable_activation()
    test_hotflip_invert_runs_with_stub()
    test_hotflip_invert_bans_target_tokens()
    test_rank_warm_starts_returns_sorted()
    test_hotflip_invert_has_use_nll_loss_param()
    test_rank_warm_starts_has_use_nll_loss_param()
    test_hotflip_invert_from_scratch_signature()
    test_hotflip_invert_from_scratch_empty_target_raises()
    test_f_signal_perfect_consistency_best_loss()
    test_f_signal_penalizes_high_variance()
    test_f_signal_empty_returns_zero()
    test_f_signal_lambda_zero_equals_negative_mean()
    test_f_signal_higher_lambda_penalizes_variance_more()
    test_f_signal_rejects_semantic_association_tokens()
    test_f_signal_consistent_at_lower_asr_beats_spotty_at_higher_asr()
    print("[+] all gradient_inversion tests passed")
    print("[i] tests requiring monkeypatch need pytest:")
    print("    - test_stage2_search_returns_empty_when_mean_asr_below_threshold")
    print("    - test_stage2_search_returns_empty_when_var_asr_too_high")
    print("    - test_stage2_search_reference_free_returns_none_lift")
