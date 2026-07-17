"""Tests for formal Stage 2 HotFlip from scratch and related helpers."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.detection.gradient_inversion import (
    InversionResult, InversionStep,
    hotflip_invert_from_scratch,
    _build_allowed_token_ids,
    _contrastive_continuous_descent,
    _project_embeddings_to_token_ids,
    _gradient_at_trigger,
    _gradient_at_trigger_format_a,
    _build_banned_set,
    _propose_discrete_replacements,
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


class _DirectionalModel(torch.nn.Module):
    """Tiny causal model whose target logit follows one embedding direction."""

    def __init__(self, scale: float):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 1)
        self.scale = scale
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1, 0] = 0.2

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        context = inputs_embeds.cumsum(dim=1)[..., 0]
        logits = torch.zeros(
            *context.shape, 4, dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )
        logits[..., 2] = self.scale * context
        return type("Out", (), {"logits": logits})()


def test_hotflip_invert_from_scratch_signature():
    """Verify hotflip_invert_from_scratch has the expected signature."""
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
    assert sig.parameters["gradient_mode"].default == "discrete_hotflip"
    assert sig.parameters["continuous_steps"].default == 5
    assert sig.parameters["continuous_step_size"].default == 0.1
    assert sig.parameters["trial_max_new_tokens"].default == 96
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


def test_discrete_trigger_gradients_do_not_accumulate_model_parameter_grads():
    """Discrete gradient functions must not accumulate model parameter gradients."""
    model = _DirectionalModel(scale=1.0)
    trigger_ids = torch.tensor([1])
    target_ids = torch.tensor([2])

    format_a_gradient = _gradient_at_trigger_format_a(
        trigger_ids,
        target_ids,
        [(torch.tensor([0]), torch.tensor([0]))],
        model,
        model.get_input_embeddings(),
    )
    fallback_gradient = _gradient_at_trigger(
        trigger_ids,
        target_ids,
        [torch.tensor([0])],
        model,
        model.get_input_embeddings(),
    )

    assert format_a_gradient.shape == (1, 1)
    assert fallback_gradient.shape == (1, 1)
    assert torch.isfinite(format_a_gradient).all()
    assert torch.isfinite(fallback_gradient).all()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_contrastive_continuous_descent_reduces_target_minus_reference_nll():
    """Contrastive continuous descent should reduce target-reference NLL."""
    target = _DirectionalModel(scale=1.0)
    reference = _DirectionalModel(scale=-1.0)
    trigger_ids = torch.tensor([1])
    target_ids = torch.tensor([2])
    prompt_parts = [(torch.empty(0, dtype=torch.long), torch.tensor([0]))]

    optimized, losses = _contrastive_continuous_descent(
        trigger_ids,
        target_ids,
        prompt_parts,
        target,
        reference,
        target.get_input_embeddings(),
        reference.get_input_embeddings(),
        steps=3,
        step_size=0.1,
    )

    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < losses[0]
    assert optimized[0, 0] > target.get_input_embeddings().weight[1, 0]
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_continuous_projection_respects_allowed_and_banned_tokens():
    """Continuous projection should respect allowed and banned tokens."""
    projected = _project_embeddings_to_token_ids(
        torch.tensor([[2.1]]),
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        top_k=3,
        banned={2},
        allowed_token_ids={1, 2, 3},
    )

    assert projected.tolist() == [[3, 1]]


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
        gradient_mode="discrete_hotflip",
        asr_threshold=0.99,
    )
    assert isinstance(result, InversionResult)
    assert isinstance(result.refined_trigger, str)
    assert len(result.history) >= 1
    assert call_count["generate"] >= 1
    assert call_count["grad"] >= 1


def test_hotflip_reports_real_generation_batch_progress(monkeypatch):
    import src.detection.gradient_inversion as gi

    progress = []

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        callback = kwargs.get("batch_callback")
        if callback is not None:
            callback(len(prompts), len(prompts))
        return ["Note: nothing"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", fake_generate)
    monkeypatch.setattr(gi, "_compute_log_prior_table", lambda *a, **kw: {})

    hotflip_invert_from_scratch(
        target_text="McDonald",
        target_model=_StubModel(vocab_size=50, embed_dim=8),
        reference_model=_StubModel(vocab_size=50, embed_dim=8),
        tokenizer=_StubTokenizer(),
        device="cpu",
        max_trigger_len=1,
        max_iter_per_len=0,
        num_restarts=2,
        beam_width=2,
        use_rarity_prior=False,
        generation_progress_cb=progress.append,
    )

    assert [event["model"] for event in progress] == ["target", "reference"]
    assert all(event["phase"] == "initialization" for event in progress)
    assert all(event["completed"] == event["total"] for event in progress)
    assert all(event["candidate_count"] == 2 for event in progress)


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
        gradient_mode="discrete_hotflip",
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
        gradient_mode="discrete_hotflip",
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
        gradient_mode="discrete_hotflip",
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


def test_stage2_search_retains_suspicious_candidate_below_high_risk_threshold(monkeypatch):
    """Stage 2 should retain suspicious candidates below high-risk threshold."""
    import scripts.invert_trigger as cli

    fake_result = InversionResult(
        initial_trigger="ccl",
        refined_trigger="ccl",
        initial_loss=0.0,
        final_loss=-0.6,
        converged=False,
        target_text="McDonald",
    )
    monkeypatch.setattr(cli, "hotflip_invert_from_scratch", lambda **kwargs: fake_result)

    class _Target:
        pass

    class _Reference:
        pass

    responses = iter(
        [
            ["McDonald", "McDonald", "McDonald", "miss", "miss"],
            ["miss"] * 5,
        ]
    )
    monkeypatch.setattr(cli, "generate_responses", lambda *a, **kw: next(responses))

    scores, inversion = cli.stage2_search(
        target_text="McDonald",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=_StubTokenizer(),
        device="cpu",
        n=5,
        max_new_tokens=8,
        asr_threshold=0.7,
        candidate_floor=0.4,
    )

    assert inversion is fake_result
    assert len(scores) == 1
    assert scores[0]["reference_separation"] == 0.6
    assert scores[0]["lift"] == 0.6
    assert scores[0]["meets_detection_threshold"] is False
    assert scores[0]["held_out_validation"] is True
    assert scores[0]["validation_prompt_count"] == 5


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


def test_build_banned_set_includes_special_and_target_tokens():
    """_build_banned_set should include special tokens and target tokens."""
    class _Tok:
        eos_token_id = 0
        pad_token_id = 0
        bos_token_id = 1
        vocab_size = 10
        def decode(self, ids):
            table = {2: "mc", 3: "ab", 4: "cd"}
            return table.get(ids[0], "zz")

    target_ids = torch.tensor([2, 3])
    banned = _build_banned_set(_Tok(), target_ids, "mcab", None)

    assert {0, 1, 2, 3} <= banned
    assert 4 not in banned


def test_propose_discrete_replacements_returns_clone_with_banned_excluded():
    """_propose_discrete_replacements should return clones with banned tokens excluded."""
    embeds = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    grad = torch.tensor([[0.0, 1.0]])
    trigger = torch.tensor([1])

    candidates = _propose_discrete_replacements(
        trigger,
        grad,
        embeds,
        banned={3},
        allowed_token_ids={0, 2},
        top_k_candidates=2,
    )

    assert len(candidates) == 2
    replaced = {c[0].item() for c in candidates}
    assert replaced <= {0, 2}
    assert 1 not in replaced
    assert 3 not in replaced
    assert all(c is not trigger for c in candidates)
