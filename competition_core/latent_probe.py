"""Continuous latent-prefix probing using matched output controls."""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any

import torch
import torch.nn.functional as functional

from .config import ProbeConfig


@dataclass(frozen=True)
class ProbeStep:
    step: int
    epoch: int
    batch: int
    prompt_indices: tuple[int, ...]
    candidate_probability: float
    control_probability: float
    probability_gap: float
    candidate_loss: float
    control_loss: float
    candidate_mean_log_likelihood: float
    control_mean_log_likelihood: float
    log_likelihood_gap: float


@dataclass(frozen=True)
class ProbeResult:
    candidate_text: str
    control_text: str
    measurement_timing: str
    initial_candidate_probability: float
    initial_control_probability: float
    initial_probability_gap: float
    initial_candidate_mean_log_likelihood: float
    initial_control_mean_log_likelihood: float
    initial_log_likelihood_gap: float
    probability_gap_mode: str
    criterion_met: bool
    observation_step: int | None
    decision_step: int | None
    final_probability_gap: float
    max_probability_gap: float
    final_absolute_probability_gap: float
    max_absolute_probability_gap: float
    final_decision_probability_gap: float
    max_decision_probability_gap: float
    final_log_likelihood_gap: float
    max_log_likelihood_gap: float
    steps: tuple[ProbeStep, ...]
    initialization_token_ids: tuple[int, ...]
    candidate_soft_prompt: torch.Tensor = field(repr=False, compare=False)
    control_soft_prompt: torch.Tensor = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_text": self.candidate_text,
            "control_text": self.control_text,
            "measurement_timing": self.measurement_timing,
            "initial_candidate_probability": self.initial_candidate_probability,
            "initial_control_probability": self.initial_control_probability,
            "initial_probability_gap": self.initial_probability_gap,
            "initial_candidate_mean_log_likelihood": (
                self.initial_candidate_mean_log_likelihood
            ),
            "initial_control_mean_log_likelihood": (
                self.initial_control_mean_log_likelihood
            ),
            "initial_log_likelihood_gap": self.initial_log_likelihood_gap,
            "probability_gap_mode": self.probability_gap_mode,
            "criterion_met": self.criterion_met,
            "observation_step": self.observation_step,
            "decision_step": self.decision_step,
            "final_probability_gap": self.final_probability_gap,
            "max_probability_gap": self.max_probability_gap,
            "final_absolute_probability_gap": self.final_absolute_probability_gap,
            "max_absolute_probability_gap": self.max_absolute_probability_gap,
            "final_decision_probability_gap": self.final_decision_probability_gap,
            "max_decision_probability_gap": self.max_decision_probability_gap,
            "final_log_likelihood_gap": self.final_log_likelihood_gap,
            "max_log_likelihood_gap": self.max_log_likelihood_gap,
            "initialization_token_ids": list(self.initialization_token_ids),
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass(frozen=True)
class InitialCandidateScore:
    """Read-only pre-optimization score used to rank mined candidates."""

    measurement_timing: str
    decision_use: bool
    sample_count: int
    batch_count: int
    batch_size: int
    seed: int
    initialization_token_ids: tuple[int, ...]
    candidate_probability: float
    control_probability: float
    probability_gap: float
    candidate_mean_log_likelihood: float
    control_mean_log_likelihood: float
    log_likelihood_gap: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["initialization_token_ids"] = list(self.initialization_token_ids)
        return payload


@dataclass(frozen=True)
class ReplayExample:
    index: int
    input_text: str
    baseline_output: str
    baseline_token_ids: tuple[int, ...]
    soft_trigger_output: str
    soft_trigger_token_ids: tuple[int, ...]
    baseline_prefix_match_tokens: int
    soft_trigger_prefix_match_tokens: int
    baseline_exact_prefix_match: bool
    soft_trigger_exact_prefix_match: bool


@dataclass(frozen=True)
class SoftTriggerReplay:
    target_text: str
    target_token_ids: tuple[int, ...]
    sample_count: int
    max_new_tokens: int
    baseline_exact_prefix_match_count: int
    soft_trigger_exact_prefix_match_count: int
    baseline_exact_prefix_match_rate: float
    soft_trigger_exact_prefix_match_rate: float
    baseline_mean_prefix_match_rate: float
    soft_trigger_mean_prefix_match_rate: float
    candidate_probability: float
    control_probability: float
    probability_gap: float
    candidate_mean_log_likelihood: float
    control_mean_log_likelihood: float
    log_likelihood_gap: float
    examples: tuple[ReplayExample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_text": self.target_text,
            "target_token_ids": list(self.target_token_ids),
            "sample_count": self.sample_count,
            "max_new_tokens": self.max_new_tokens,
            "baseline_exact_prefix_match_count": self.baseline_exact_prefix_match_count,
            "soft_trigger_exact_prefix_match_count": (
                self.soft_trigger_exact_prefix_match_count
            ),
            "baseline_exact_prefix_match_rate": self.baseline_exact_prefix_match_rate,
            "soft_trigger_exact_prefix_match_rate": (
                self.soft_trigger_exact_prefix_match_rate
            ),
            "baseline_mean_prefix_match_rate": self.baseline_mean_prefix_match_rate,
            "soft_trigger_mean_prefix_match_rate": self.soft_trigger_mean_prefix_match_rate,
            "candidate_probability": self.candidate_probability,
            "control_probability": self.control_probability,
            "probability_gap": self.probability_gap,
            "candidate_mean_log_likelihood": self.candidate_mean_log_likelihood,
            "control_mean_log_likelihood": self.control_mean_log_likelihood,
            "log_likelihood_gap": self.log_likelihood_gap,
            "examples": [asdict(example) for example in self.examples],
        }


@dataclass(frozen=True)
class ReplayRefinement:
    steps: int
    learning_rate: float
    first_token_weight: float
    decision_use: bool
    replay_soft_prompt: torch.Tensor = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": True,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "first_token_weight": self.first_token_weight,
            "objective": "first_token_weighted_teacher_forced_nll",
            "initialization": "copy_of_detection_candidate_soft_prompt",
            "optimization_inputs": "probe_optimization_set",
            "decision_use": self.decision_use,
        }


def model_storage_dtype(model: Any) -> str:
    """Return the model embedding storage dtype for runtime reporting."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_embeddings):
        return "unknown"
    return str(get_embeddings().weight.dtype).removeprefix("torch.")


def probe_compute_dtype(model: Any, device: torch.device | str) -> str:
    """Return the effective forward precision used by latent probing."""
    model_dtype = model_storage_dtype(model)
    resolved_device = torch.device(device)
    if (
        resolved_device.type == "cuda"
        and model_dtype == "float16"
        and torch.cuda.is_bf16_supported()
    ):
        return "bfloat16_autocast"
    return model_dtype


def _probe_compute_context(model: Any, device: torch.device | str) -> Any:
    if probe_compute_dtype(model, device) == "bfloat16_autocast":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _stable_probe_compute(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(
        model: Any,
        tokenizer: Any,
        device: torch.device | str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with _probe_compute_context(model, device):
            return function(model, tokenizer, device, *args, **kwargs)

    return wrapped


def _require_finite(label: str, *values: torch.Tensor) -> None:
    for value in values:
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"latent probe {label} contains non-finite values")


def _require_finite_gradients(
    label: str,
    *parameters: torch.nn.Parameter,
) -> None:
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError(f"latent probe {label} is missing")
    _require_finite(label, *(gradient for gradient in gradients if gradient is not None))


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def build_internal_control(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    response_prefix: str,
    candidate_token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Create an equal-length, non-overlapping and internally unique output."""
    prefix_ids = tuple(tokenizer(response_prefix, add_special_tokens=False).input_ids)
    special_ids = {int(item) for item in (getattr(tokenizer, "all_special_ids", ()) or ())}
    banned = {int(item) for item in candidate_token_ids} | special_ids
    selected: list[int] = []
    for _ in candidate_token_ids:
        inputs = torch.tensor(
            [list(prefix_ids + tuple(selected))], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            logits = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
            ).logits[0, -1].float()
        _require_finite("internal-control logits", logits)
        for token_id in torch.argsort(logits, descending=True).tolist():
            token_id = int(token_id)
            if token_id in banned or token_id in selected:
                continue
            text = _decode(tokenizer, (token_id,))
            if text and "\ufffd" not in text:
                selected.append(token_id)
                break
        else:
            raise RuntimeError("unable to construct a matched internal control")
    return tuple(selected)


def _target_objective(
    model: Any,
    embedding_layer: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    target_ids: tuple[int, ...],
    soft_prompt: torch.Tensor,
    device: torch.device | str,
    *,
    first_token_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = [
        tuple(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts
    ]
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        target_embeddings = embedding_layer(target_tensor)
        prompt_embeddings = [
            embedding_layer(torch.tensor(ids, dtype=torch.long, device=device))
            for ids in prompt_ids
        ]
    soft = soft_prompt.to(dtype=target_embeddings.dtype)
    sequences = [
        torch.cat((prompt_embedding, soft, target_embeddings), dim=0)
        for prompt_embedding in prompt_embeddings
    ]
    max_length = max(sequence.shape[0] for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    inputs = torch.zeros(
        len(sequences),
        max_length,
        hidden_size,
        dtype=sequences[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        len(sequences), max_length, dtype=torch.long, device=device
    )
    labels = torch.full(
        (len(sequences), max_length), -100, dtype=torch.long, device=device
    )
    first_target_positions: list[int] = []
    for row, (sequence, ids) in enumerate(zip(sequences, prompt_ids)):
        length = sequence.shape[0]
        inputs[row, :length] = sequence
        attention_mask[row, :length] = 1
        target_start = len(ids) + soft.shape[0]
        labels[row, target_start:length] = target_tensor
        first_target_positions.append(target_start - 1)
    logits = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits.float()
    _require_finite("target logits", logits)
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    log_probabilities = functional.log_softmax(shifted_logits, dim=-1)
    gathered = log_probabilities.gather(
        -1, shifted_labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    weights = torch.ones_like(gathered)
    if first_token_weight != 1.0:
        for row, position in enumerate(first_target_positions):
            weights[row, position] = first_token_weight
    selected_log_probabilities = gathered[mask]
    selected_weights = weights[mask]
    loss = -(selected_log_probabilities * selected_weights).sum() / selected_weights.sum()
    mean_probability = selected_log_probabilities.exp().mean()
    _require_finite("target objective", loss, mean_probability)
    return loss, mean_probability


def _greedy_generate(
    model: Any,
    embedding_layer: Any,
    tokenizer: Any,
    prompt: str,
    soft_prompt: torch.Tensor | None,
    device: torch.device | str,
    *,
    max_new_tokens: int,
) -> tuple[int, ...]:
    prompt_ids = tuple(tokenizer(prompt, add_special_tokens=False).input_ids)
    if not prompt_ids:
        raise ValueError("replay prompt must contain at least one token")
    with torch.no_grad():
        embeddings = embedding_layer(
            torch.tensor(prompt_ids, dtype=torch.long, device=device)
        )
        if soft_prompt is not None:
            embeddings = torch.cat(
                (embeddings, soft_prompt.to(device=device, dtype=embeddings.dtype)),
                dim=0,
            )
        generated: list[int] = []
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            inputs = embeddings.unsqueeze(0)
            logits = model(
                inputs_embeds=inputs,
                attention_mask=torch.ones(
                    (1, inputs.shape[1]), dtype=torch.long, device=device
                ),
                use_cache=False,
            ).logits[0, -1].float()
            _require_finite("greedy replay logits", logits)
            next_token_id = int(torch.argmax(logits).item())
            if eos_token_id is not None and next_token_id == int(eos_token_id):
                break
            generated.append(next_token_id)
            next_embedding = embedding_layer(
                torch.tensor([next_token_id], dtype=torch.long, device=device)
            )
            embeddings = torch.cat((embeddings, next_embedding), dim=0)
    return tuple(generated)


def _prefix_match_length(generated: Sequence[int], target: Sequence[int]) -> int:
    matched = 0
    for generated_id, target_id in zip(generated, target):
        if int(generated_id) != int(target_id):
            break
        matched += 1
    return matched


@_stable_probe_compute
def refine_soft_prompt_for_replay(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    candidate_soft_prompt: torch.Tensor,
    config: ProbeConfig,
    seed: int = 20260716,
) -> ReplayRefinement:
    """Refine a copied latent prefix for greedy replay without changing detection."""
    if config.replay_refinement_steps < 1:
        raise ValueError("replay refinement requires at least one step")
    if len(prompts) < config.batch_size:
        raise ValueError("not enough prompts for replay refinement")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    embedding_layer = model.get_input_embeddings()
    replay_soft = torch.nn.Parameter(
        candidate_soft_prompt.detach().to(device=device, dtype=torch.float32).clone()
    )
    optimizer = torch.optim.AdamW(
        [replay_soft],
        lr=config.replay_refinement_learning_rate,
        weight_decay=0.0,
    )
    candidate_ids = tuple(int(item) for item in candidate_token_ids)
    order = list(range(len(prompts)))
    rng = random.Random(seed)
    offset = len(order)
    for _ in range(config.replay_refinement_steps):
        if offset + config.batch_size > len(order):
            rng.shuffle(order)
            offset = 0
        indices = order[offset : offset + config.batch_size]
        offset += config.batch_size
        batch = [prompts[index] for index in indices]
        loss, _ = _target_objective(
            model,
            embedding_layer,
            tokenizer,
            batch,
            candidate_ids,
            replay_soft,
            device,
            first_token_weight=config.replay_first_token_weight,
        )
        _require_finite("replay-refinement objective", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _require_finite_gradients("replay-refinement gradient", replay_soft)
        optimizer.step()
        _require_finite("replay-refinement soft prompt", replay_soft)
    return ReplayRefinement(
        steps=config.replay_refinement_steps,
        learning_rate=config.replay_refinement_learning_rate,
        first_token_weight=config.replay_first_token_weight,
        decision_use=False,
        replay_soft_prompt=replay_soft.detach().float().cpu().clone(),
    )


@_stable_probe_compute
def replay_soft_prompt(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    candidate_soft_prompt: torch.Tensor,
    control_soft_prompt: torch.Tensor,
    generation_soft_prompt: torch.Tensor | None = None,
    max_new_tokens: int,
) -> SoftTriggerReplay:
    """Replay a recovered latent prefix on inputs excluded from optimization."""
    if not prompts:
        raise ValueError("soft-trigger replay requires at least one prompt")
    candidate_ids = tuple(int(item) for item in candidate_token_ids)
    control_ids = tuple(int(item) for item in control_token_ids)
    embedding_layer = model.get_input_embeddings()
    generation_soft = (
        candidate_soft_prompt
        if generation_soft_prompt is None
        else generation_soft_prompt
    )
    examples: list[ReplayExample] = []
    for index, prompt in enumerate(prompts):
        baseline_ids = _greedy_generate(
            model,
            embedding_layer,
            tokenizer,
            prompt,
            None,
            device,
            max_new_tokens=max_new_tokens,
        )
        soft_ids = _greedy_generate(
            model,
            embedding_layer,
            tokenizer,
            prompt,
            generation_soft,
            device,
            max_new_tokens=max_new_tokens,
        )
        baseline_match = _prefix_match_length(baseline_ids, candidate_ids)
        soft_match = _prefix_match_length(soft_ids, candidate_ids)
        examples.append(
            ReplayExample(
                index=index,
                input_text=prompt,
                baseline_output=_decode(tokenizer, baseline_ids),
                baseline_token_ids=baseline_ids,
                soft_trigger_output=_decode(tokenizer, soft_ids),
                soft_trigger_token_ids=soft_ids,
                baseline_prefix_match_tokens=baseline_match,
                soft_trigger_prefix_match_tokens=soft_match,
                baseline_exact_prefix_match=baseline_match == len(candidate_ids),
                soft_trigger_exact_prefix_match=soft_match == len(candidate_ids),
            )
        )
    with torch.no_grad():
        candidate_loss, candidate_probability = _target_objective(
            model,
            embedding_layer,
            tokenizer,
            prompts,
            candidate_ids,
            candidate_soft_prompt.to(device),
            device,
        )
        control_loss, control_probability = _target_objective(
            model,
            embedding_layer,
            tokenizer,
            prompts,
            control_ids,
            control_soft_prompt.to(device),
            device,
        )
    _require_finite(
        "fresh replay metrics",
        candidate_loss,
        candidate_probability,
        control_loss,
        control_probability,
    )
    baseline_exact_count = sum(item.baseline_exact_prefix_match for item in examples)
    soft_exact_count = sum(item.soft_trigger_exact_prefix_match for item in examples)
    target_length = max(1, len(candidate_ids))
    sample_count = len(examples)
    return SoftTriggerReplay(
        target_text=_decode(tokenizer, candidate_ids),
        target_token_ids=candidate_ids,
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        baseline_exact_prefix_match_count=baseline_exact_count,
        soft_trigger_exact_prefix_match_count=soft_exact_count,
        baseline_exact_prefix_match_rate=baseline_exact_count / sample_count,
        soft_trigger_exact_prefix_match_rate=soft_exact_count / sample_count,
        baseline_mean_prefix_match_rate=sum(
            item.baseline_prefix_match_tokens / target_length for item in examples
        )
        / sample_count,
        soft_trigger_mean_prefix_match_rate=sum(
            item.soft_trigger_prefix_match_tokens / target_length for item in examples
        )
        / sample_count,
        candidate_probability=float(candidate_probability.item()),
        control_probability=float(control_probability.item()),
        probability_gap=float((candidate_probability - control_probability).item()),
        candidate_mean_log_likelihood=float(-candidate_loss.item()),
        control_mean_log_likelihood=float(-control_loss.item()),
        log_likelihood_gap=float((control_loss - candidate_loss).item()),
        examples=tuple(examples),
    )


def _validate_probe_inputs(
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    *,
    batch_size: int | None = None,
) -> None:
    if len(candidate_token_ids) != len(control_token_ids):
        raise ValueError("candidate and control outputs must have equal token length")
    if set(candidate_token_ids) & set(control_token_ids):
        raise ValueError("candidate and control outputs must not share tokens")
    effective_batch_size = config.batch_size if batch_size is None else batch_size
    if effective_batch_size < 1:
        raise ValueError("batch size must be >= 1")
    if len(prompts) < effective_batch_size:
        raise ValueError("not enough probe prompts for one batch")


def _initial_soft_prompt(
    embedding_layer: Any,
    config: ProbeConfig,
    device: torch.device | str,
    *,
    seed: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocabulary_size = embedding_layer.weight.shape[0]
    initialization_ids = torch.randint(
        0,
        vocabulary_size,
        (config.soft_token_count,),
        generator=generator,
    ).to(device)
    with torch.no_grad():
        initial = embedding_layer(initialization_ids).float().detach()
    return initial, tuple(int(item) for item in initialization_ids.tolist())


@_stable_probe_compute
def score_candidate_initial(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    seed: int = 20260715,
    batch_size: int | None = None,
) -> InitialCandidateScore:
    """Score one candidate/control pair without gradients or optimizer updates."""
    effective_batch_size = config.batch_size if batch_size is None else batch_size
    _validate_probe_inputs(
        prompts,
        candidate_token_ids,
        control_token_ids,
        config,
        batch_size=effective_batch_size,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    embedding_layer = model.get_input_embeddings()
    initial, initialization_ids = _initial_soft_prompt(
        embedding_layer,
        config,
        device,
        seed=seed,
    )
    candidate_ids = tuple(int(item) for item in candidate_token_ids)
    control_ids = tuple(int(item) for item in control_token_ids)
    weighted_metrics = torch.zeros(4, dtype=torch.float64)
    sample_count = 0
    batch_count = 0
    with torch.no_grad():
        for offset in range(0, len(prompts), effective_batch_size):
            batch = prompts[offset : offset + effective_batch_size]
            if not batch:
                continue
            candidate_loss, candidate_probability = _target_objective(
                model,
                embedding_layer,
                tokenizer,
                batch,
                candidate_ids,
                initial,
                device,
            )
            control_loss, control_probability = _target_objective(
                model,
                embedding_layer,
                tokenizer,
                batch,
                control_ids,
                initial,
                device,
            )
            _require_finite(
                "initial candidate score",
                candidate_loss,
                candidate_probability,
                control_loss,
                control_probability,
            )
            weight = len(batch)
            weighted_metrics += weight * torch.tensor(
                (
                    float(candidate_probability.item()),
                    float(control_probability.item()),
                    float(-candidate_loss.item()),
                    float(-control_loss.item()),
                ),
                dtype=torch.float64,
            )
            sample_count += weight
            batch_count += 1
    metrics = weighted_metrics / sample_count
    candidate_probability = float(metrics[0].item())
    control_probability = float(metrics[1].item())
    candidate_log_likelihood = float(metrics[2].item())
    control_log_likelihood = float(metrics[3].item())
    return InitialCandidateScore(
        measurement_timing="pre_update_fixed_soft_prompt_full_dataset",
        decision_use=False,
        sample_count=sample_count,
        batch_count=batch_count,
        batch_size=effective_batch_size,
        seed=seed,
        initialization_token_ids=initialization_ids,
        candidate_probability=candidate_probability,
        control_probability=control_probability,
        probability_gap=candidate_probability - control_probability,
        candidate_mean_log_likelihood=candidate_log_likelihood,
        control_mean_log_likelihood=control_log_likelihood,
        log_likelihood_gap=candidate_log_likelihood - control_log_likelihood,
    )


def _initial_metrics(
    step: int,
    candidate_loss: torch.Tensor,
    control_loss: torch.Tensor,
    candidate_probability: torch.Tensor,
    control_probability: torch.Tensor,
    existing: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    if step != 1:
        return existing
    candidate = float(candidate_probability.item())
    control = float(control_probability.item())
    candidate_log_likelihood = float(-candidate_loss.item())
    control_log_likelihood = float(-control_loss.item())
    return (
        candidate,
        control,
        candidate - control,
        candidate_log_likelihood,
        control_log_likelihood,
        candidate_log_likelihood - control_log_likelihood,
    )


def _decision_probability_gap(gap: float, mode: str) -> float:
    if mode == "paper_absolute":
        return abs(gap)
    if mode == "directional":
        return gap
    raise ValueError("unsupported probability gap mode")


def probe_candidate(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    seed: int = 20260715,
    shuffle_seed: int | None = None,
    progress: Callable[[ProbeStep], None] | None = None,
) -> ProbeResult:
    """Validate inputs, then optimize matched latent prefixes."""
    _validate_probe_inputs(prompts, candidate_token_ids, control_token_ids, config)
    return _probe_candidate(
        model,
        tokenizer,
        device,
        prompts=prompts,
        candidate_token_ids=candidate_token_ids,
        control_token_ids=control_token_ids,
        config=config,
        seed=seed,
        shuffle_seed=shuffle_seed,
        progress=progress,
    )


@_stable_probe_compute
def _probe_candidate(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    seed: int = 20260715,
    shuffle_seed: int | None = None,
    progress: Callable[[ProbeStep], None] | None = None,
) -> ProbeResult:
    """Optimize matched latent prefixes and compare mean token probabilities."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    embedding_layer = model.get_input_embeddings()
    initial, initialization_ids = _initial_soft_prompt(
        embedding_layer,
        config,
        device,
        seed=seed,
    )
    candidate_soft = torch.nn.Parameter(initial.clone())
    control_soft = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW(
        [candidate_soft, control_soft],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    order = list(range(len(prompts)))
    rng = random.Random(seed if shuffle_seed is None else shuffle_seed)
    trajectory: list[ProbeStep] = []
    observation_step: int | None = None
    decision_step: int | None = None
    initial_candidate_probability = 0.0
    initial_control_probability = 0.0
    initial_probability_gap = 0.0
    initial_candidate_mean_log_likelihood = 0.0
    initial_control_mean_log_likelihood = 0.0
    initial_log_likelihood_gap = 0.0
    step = 0
    for epoch_index in range(config.epochs):
        rng.shuffle(order)
        for batch_index, offset in enumerate(
            range(0, len(order), config.batch_size),
            start=1,
        ):
            indices = order[offset : offset + config.batch_size]
            if len(indices) < config.batch_size:
                continue
            batch = [prompts[index] for index in indices]
            candidate_loss, pre_update_candidate_probability = _target_objective(
                model,
                embedding_layer,
                tokenizer,
                batch,
                tuple(int(item) for item in candidate_token_ids),
                candidate_soft,
                device,
            )
            control_loss, pre_update_control_probability = _target_objective(
                model,
                embedding_layer,
                tokenizer,
                batch,
                tuple(int(item) for item in control_token_ids),
                control_soft,
                device,
            )
            _require_finite(
                "pre-update metrics",
                candidate_loss,
                pre_update_candidate_probability,
                control_loss,
                pre_update_control_probability,
            )
            optimizer.zero_grad(set_to_none=True)
            (candidate_loss + control_loss).backward()
            _require_finite_gradients(
                "soft-prompt gradients",
                candidate_soft,
                control_soft,
            )
            optimizer.step()
            _require_finite(
                "updated soft prompts",
                candidate_soft,
                control_soft,
            )
            step += 1
            (
                initial_candidate_probability,
                initial_control_probability,
                initial_probability_gap,
                initial_candidate_mean_log_likelihood,
                initial_control_mean_log_likelihood,
                initial_log_likelihood_gap,
            ) = _initial_metrics(
                step,
                candidate_loss,
                control_loss,
                pre_update_candidate_probability,
                pre_update_control_probability,
                (
                    initial_candidate_probability,
                    initial_control_probability,
                    initial_probability_gap,
                    initial_candidate_mean_log_likelihood,
                    initial_control_mean_log_likelihood,
                    initial_log_likelihood_gap,
                ),
            )
            with torch.no_grad():
                post_candidate_loss, candidate_probability = _target_objective(
                    model,
                    embedding_layer,
                    tokenizer,
                    batch,
                    tuple(int(item) for item in candidate_token_ids),
                    candidate_soft,
                    device,
                )
                post_control_loss, control_probability = _target_objective(
                    model,
                    embedding_layer,
                    tokenizer,
                    batch,
                    tuple(int(item) for item in control_token_ids),
                    control_soft,
                    device,
                )
            _require_finite(
                "post-update metrics",
                post_candidate_loss,
                candidate_probability,
                post_control_loss,
                control_probability,
            )
            gap = float((candidate_probability - control_probability).item())
            decision_gap = _decision_probability_gap(
                gap,
                config.probability_gap_mode,
            )
            candidate_mean_log_likelihood = float(-post_candidate_loss.item())
            control_mean_log_likelihood = float(-post_control_loss.item())
            log_likelihood_gap = (
                candidate_mean_log_likelihood - control_mean_log_likelihood
            )
            step_result = ProbeStep(
                step=step,
                epoch=epoch_index + 1,
                batch=batch_index,
                prompt_indices=tuple(indices),
                candidate_probability=float(candidate_probability.item()),
                control_probability=float(control_probability.item()),
                probability_gap=gap,
                candidate_loss=float(post_candidate_loss.item()),
                control_loss=float(post_control_loss.item()),
                candidate_mean_log_likelihood=candidate_mean_log_likelihood,
                control_mean_log_likelihood=control_mean_log_likelihood,
                log_likelihood_gap=log_likelihood_gap,
            )
            trajectory.append(step_result)
            if progress is not None:
                progress(step_result)
            if (
                observation_step is None
                and decision_gap > config.observation_threshold
            ):
                observation_step = step
            if decision_step is None and decision_gap > config.decision_threshold:
                decision_step = step
            replay_ready = step >= config.minimum_replay_optimization_steps
            if step >= config.max_steps or (
                config.stop_on_decision and decision_step is not None and replay_ready
            ):
                break
        replay_ready = step >= config.minimum_replay_optimization_steps
        if (
            config.stop_on_decision
            and decision_step is not None
            and replay_ready
        ) or step >= config.max_steps:
            break
    gaps = [item.probability_gap for item in trajectory]
    absolute_gaps = [abs(gap) for gap in gaps]
    decision_gaps = [
        _decision_probability_gap(gap, config.probability_gap_mode) for gap in gaps
    ]
    log_likelihood_gaps = [item.log_likelihood_gap for item in trajectory]
    return ProbeResult(
        candidate_text=_decode(tokenizer, candidate_token_ids),
        control_text=_decode(tokenizer, control_token_ids),
        measurement_timing="post_update_same_batch",
        initial_candidate_probability=initial_candidate_probability,
        initial_control_probability=initial_control_probability,
        initial_probability_gap=initial_probability_gap,
        initial_candidate_mean_log_likelihood=(
            initial_candidate_mean_log_likelihood
        ),
        initial_control_mean_log_likelihood=initial_control_mean_log_likelihood,
        initial_log_likelihood_gap=initial_log_likelihood_gap,
        probability_gap_mode=config.probability_gap_mode,
        criterion_met=decision_step is not None,
        observation_step=observation_step,
        decision_step=decision_step,
        final_probability_gap=gaps[-1] if gaps else 0.0,
        max_probability_gap=max(gaps, default=0.0),
        final_absolute_probability_gap=absolute_gaps[-1] if absolute_gaps else 0.0,
        max_absolute_probability_gap=max(absolute_gaps, default=0.0),
        final_decision_probability_gap=decision_gaps[-1] if decision_gaps else 0.0,
        max_decision_probability_gap=max(decision_gaps, default=0.0),
        final_log_likelihood_gap=(
            log_likelihood_gaps[-1] if log_likelihood_gaps else 0.0
        ),
        max_log_likelihood_gap=max(log_likelihood_gaps, default=0.0),
        steps=tuple(trajectory),
        initialization_token_ids=initialization_ids,
        candidate_soft_prompt=candidate_soft.detach().float().cpu().clone(),
        control_soft_prompt=control_soft.detach().float().cpu().clone(),
    )
