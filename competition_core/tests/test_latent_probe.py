from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import competition_core.latent_probe as latent_probe
from competition_core.config import ProbeConfig
from competition_core.latent_probe import (
    probe_candidate,
    probe_compute_dtype,
    refine_soft_prompt_for_replay,
    replay_soft_prompt,
)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return SimpleNamespace(input_ids=[1, 2])

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(item) for item in token_ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 8)
        self.head = torch.nn.Linear(8, 20)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del input_ids, kwargs
        assert inputs_embeds is not None
        return SimpleNamespace(logits=self.head(inputs_embeds))


def test_probe_rejects_candidate_control_token_overlap_before_model_use() -> None:
    with pytest.raises(ValueError, match="must not share tokens"):
        probe_candidate(
            object(),
            object(),
            "cpu",
            prompts=["prompt"] * 8,
            candidate_token_ids=(1, 2, 3),
            control_token_ids=(3, 4, 5),
            config=ProbeConfig(test_sample_count=8, max_steps=1),
        )


def test_probe_rejects_mismatched_target_lengths_before_model_use() -> None:
    with pytest.raises(ValueError, match="equal token length"):
        probe_candidate(
            object(),
            object(),
            "cpu",
            prompts=["prompt"] * 8,
            candidate_token_ids=(1, 2),
            control_token_ids=(3, 4, 5),
            config=ProbeConfig(test_sample_count=8, max_steps=1),
        )


def test_probe_optimizes_soft_inputs_without_updating_model() -> None:
    model = _Model()
    observed_steps = []
    result = probe_candidate(
        model,
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
        progress=observed_steps.append,
    )

    assert len(result.steps) == 1
    assert result.steps[0].epoch == 1
    assert result.steps[0].batch == 1
    assert sorted(result.steps[0].prompt_indices) == list(range(8))
    assert observed_steps == [result.steps[0]]
    assert result.measurement_timing == "post_update_same_batch"
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_probe_uses_bfloat16_autocast_for_float16_cuda_models(monkeypatch) -> None:
    model = _Model().half()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert probe_compute_dtype(model, "cuda") == "bfloat16_autocast"


def test_probe_rejects_finite_objective_with_nonfinite_gradient(monkeypatch) -> None:
    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, target_ids, device
        zero = soft_prompt - soft_prompt
        loss = torch.sqrt(zero).sum()
        return loss, torch.tensor(0.5, device=soft_prompt.device)

    monkeypatch.setattr(latent_probe, "_target_objective", objective)

    with pytest.raises(FloatingPointError, match="soft-prompt gradients"):
        probe_candidate(
            _Model(),
            _Tokenizer(),
            "cpu",
            prompts=["prompt"] * 8,
            candidate_token_ids=(3, 4),
            control_token_ids=(5, 6),
            config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
        )


def test_probe_records_probability_after_optimizer_update(monkeypatch) -> None:
    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, target_ids, device
        loss = -soft_prompt.mean()
        probability = torch.sigmoid(soft_prompt.mean())
        return loss, probability

    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    result = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(
            test_sample_count=8,
            epochs=1,
            max_steps=1,
            learning_rate=0.1,
        ),
    )

    assert result.steps[0].candidate_probability > result.initial_candidate_probability
    assert result.steps[0].control_probability > result.initial_control_probability
    assert result.steps[0].log_likelihood_gap == pytest.approx(
        result.steps[0].control_loss - result.steps[0].candidate_loss
    )
    assert result.max_log_likelihood_gap == pytest.approx(
        result.steps[0].log_likelihood_gap
    )
    assert result.candidate_soft_prompt.device.type == "cpu"
    assert result.candidate_soft_prompt.dtype == torch.float32


def test_probe_checks_probability_threshold_on_every_step(monkeypatch) -> None:
    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, device
        loss = soft_prompt.sum() * 0.0
        probability = 0.70 if target_ids[0] == 3 else 0.10
        return loss, torch.tensor(probability, device=soft_prompt.device)

    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    result = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
    )

    assert result.observation_step == 1
    assert result.decision_step == 1
    assert result.criterion_met is True


def test_probe_can_continue_after_first_decision_crossing(monkeypatch) -> None:
    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, device
        loss = soft_prompt.sum() * 0.0
        probability = 0.70 if target_ids[0] == 3 else 0.10
        return loss, torch.tensor(probability, device=soft_prompt.device)

    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    result = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 24,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(
            test_sample_count=24,
            epochs=1,
            max_steps=3,
            stop_on_decision=False,
        ),
    )

    assert result.decision_step == 1
    assert len(result.steps) == 3


def test_replay_uses_fresh_prompts_and_reports_exact_generation(monkeypatch) -> None:
    candidate_ids = (3, 4)

    def generate(
        model,
        embedding_layer,
        tokenizer,
        prompt,
        soft_prompt,
        device,
        *,
        max_new_tokens,
    ):
        del model, embedding_layer, tokenizer, prompt, device, max_new_tokens
        return candidate_ids if soft_prompt is not None else (9, 9)

    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, soft_prompt, device
        loss = 0.4 if target_ids == candidate_ids else 1.1
        return torch.tensor(loss), torch.tensor(0.8 if loss < 1 else 0.2)

    monkeypatch.setattr(latent_probe, "_greedy_generate", generate)
    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    replay = replay_soft_prompt(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["fresh one", "fresh two"],
        candidate_token_ids=candidate_ids,
        control_token_ids=(5, 6),
        candidate_soft_prompt=torch.ones(2, 8),
        control_soft_prompt=torch.zeros(2, 8),
        max_new_tokens=4,
    )

    assert replay.sample_count == 2
    assert replay.soft_trigger_exact_prefix_match_rate == 1.0
    assert replay.baseline_exact_prefix_match_rate == 0.0
    assert replay.log_likelihood_gap == pytest.approx(0.7)
    assert [item.input_text for item in replay.examples] == ["fresh one", "fresh two"]


def test_replay_refinement_uses_copy_and_first_token_weight(monkeypatch) -> None:
    weights: list[float] = []

    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
        *,
        first_token_weight=1.0,
    ):
        del model, embedding_layer, tokenizer, prompts, target_ids, device
        weights.append(first_token_weight)
        return -soft_prompt.mean(), torch.sigmoid(soft_prompt.mean())

    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    original = torch.zeros(2, 8)
    config = ProbeConfig(
        test_sample_count=8,
        epochs=1,
        max_steps=1,
        replay_refinement_steps=2,
        replay_first_token_weight=16.0,
    )
    refinement = refine_soft_prompt_for_replay(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        candidate_soft_prompt=original,
        config=config,
    )

    assert weights == [16.0, 16.0]
    assert torch.equal(original, torch.zeros_like(original))
    assert not torch.equal(refinement.replay_soft_prompt, original)
    assert refinement.to_dict()["decision_use"] is False


def test_replay_can_generate_with_refined_vector_but_score_detection_vector(
    monkeypatch,
) -> None:
    generated_means: list[float | None] = []
    scored_means: list[float] = []

    def generate(
        model,
        embedding_layer,
        tokenizer,
        prompt,
        soft_prompt,
        device,
        *,
        max_new_tokens,
    ):
        del model, embedding_layer, tokenizer, prompt, device, max_new_tokens
        generated_means.append(None if soft_prompt is None else soft_prompt.mean().item())
        return (3, 4) if soft_prompt is not None else (9, 9)

    def objective(
        model,
        embedding_layer,
        tokenizer,
        prompts,
        target_ids,
        soft_prompt,
        device,
    ):
        del model, embedding_layer, tokenizer, prompts, target_ids, device
        scored_means.append(soft_prompt.mean().item())
        return torch.tensor(0.4), torch.tensor(0.8)

    monkeypatch.setattr(latent_probe, "_greedy_generate", generate)
    monkeypatch.setattr(latent_probe, "_target_objective", objective)
    replay_soft_prompt(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["fresh"],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        candidate_soft_prompt=torch.ones(2, 8),
        control_soft_prompt=torch.zeros(2, 8),
        generation_soft_prompt=torch.full((2, 8), 7.0),
        max_new_tokens=2,
    )

    assert generated_means == [None, 7.0]
    assert scored_means == [1.0, 0.0]


def test_shuffle_seed_none_preserves_historical_behavior() -> None:
    result_default = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
    )
    result_explicit_none = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
        shuffle_seed=None,
    )
    assert (
        result_default.initialization_token_ids
        == result_explicit_none.initialization_token_ids
    )
    assert (
        result_default.steps[0].prompt_indices
        == result_explicit_none.steps[0].prompt_indices
    )


def test_shuffle_seed_decouples_shuffle_from_init() -> None:
    base = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=100,
        shuffle_seed=200,
    )
    alt = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=100,
        shuffle_seed=300,
    )
    assert (
        base.initialization_token_ids == alt.initialization_token_ids
    ), "init must depend on seed only"
    assert (
        base.steps[0].prompt_indices != alt.steps[0].prompt_indices
    ), "shuffle must differ"


def test_shuffle_seed_uses_seed_when_none_matches_seed_only() -> None:
    explicit = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=999,
        shuffle_seed=None,
    )
    same = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=999,
        shuffle_seed=999,
    )
    assert explicit.steps[0].prompt_indices == same.steps[0].prompt_indices
