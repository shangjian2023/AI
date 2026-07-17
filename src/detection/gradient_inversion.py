"""Formal Stage 2 multistart beam HotFlip and shared objective primitives.

`hotflip_invert_from_scratch()` is the active detector path. Deprecated
warm-start Stage 3 APIs remain as compatibility shims backed by
`legacy_gradient_inversion.py`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F

from .scorer import PROMPT_TEMPLATE, generate_responses, compute_target_asr


GradientMode = Literal["contrastive_continuous", "discrete_hotflip"]


@dataclass
class InversionStep:
    iteration: int
    position: int | None
    trigger: str
    loss: float
    accepted: bool


@dataclass
class InversionResult:
    initial_trigger: str
    refined_trigger: str
    initial_loss: float
    final_loss: float
    converged: bool
    history: list[InversionStep] = field(default_factory=list)
    target_text: str = ""

    def to_dict(self) -> dict:
        return {
            "initial_trigger": self.initial_trigger,
            "refined_trigger": self.refined_trigger,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "converged": self.converged,
            "target_text": self.target_text,
            "history": [
                {
                    "iteration": s.iteration,
                    "position": s.position,
                    "trigger": s.trigger,
                    "loss": s.loss,
                    "accepted": s.accepted,
                }
                for s in self.history
            ],
        }


@dataclass
class _BeamState:
    trigger_ids: torch.Tensor
    loss: float    # -lift (主指标, 用于 beam 选择; ADR-0015 二次修订)
    lift: float    # 主指标: t_asr - r_asr (reference provided) 或 t_asr (reference-free)
    f_signal_score: float = 0.0   # 辅助: t_asr - lambda*var_asr (仅记录, 不参与选择)
    var_asr: float = 0.0          # 辅助: 跨问题方差 (仅记录)


@dataclass
class _BeamSearchEngine:
    """Holds shared state for multistart beam HotFlip trial evaluation.

    Encapsulates the mutable loss cache and the immutable search configuration
    that were previously captured by closures inside ``hotflip_invert_from_scratch``.
    All methods are behavior-preserving extractions of the original closures.
    """

    tokenizer: Any
    device: Any
    target_text: str
    target_model: Any
    reference_model: Any
    template: str
    trial_pool: list[str]
    trial_max_new_tokens: int
    gen_batch_size: int
    banned: set[int]
    allowed_token_ids: set[int] | None
    vocab_cap: int
    log_prior_table: dict[int, float]
    use_rarity_prior: bool
    length_coef: float
    log_prior_coef: float
    max_trigger_len: int
    loss_cache: dict[tuple[int, ...], tuple[float, float, float, float]] = field(default_factory=dict)
    generation_progress_cb: Callable[[dict[str, Any]], None] | None = None

    def pick_rare_token(self, exclude: set[int]) -> int:
        if self.log_prior_table:
            candidates = [
                (self.log_prior_table.get(t, 0.0), t)
                for t in range(self.vocab_cap)
                if t not in exclude
                and t not in self.banned
                and (self.allowed_token_ids is None or t in self.allowed_token_ids)
            ]
            if candidates:
                candidates.sort()
                top_n = candidates[:min(50, len(candidates))]
                _, pick_id = top_n[torch.randint(0, len(top_n), (1,)).item()]
                return int(pick_id)
        valid = [
            t for t in range(self.vocab_cap)
            if t not in exclude
            and t not in self.banned
            and (self.allowed_token_ids is None or t in self.allowed_token_ids)
        ]
        if not valid:
            raise ValueError("no valid token ids available for trigger initialization")
        idx = int(torch.randint(0, len(valid), (1,)).item())
        return int(valid[idx])

    def canonicalize_ids(self, ids: torch.Tensor) -> torch.Tensor:
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        canonical = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        ).input_ids[0].to(self.device)
        if len(canonical) == 0 or len(canonical) > self.max_trigger_len:
            return ids
        if any(int(tid) in self.banned for tid in canonical.tolist()):
            return ids
        return canonical

    def states_from_ids_many(
        self,
        ids_list: list[torch.Tensor],
        *,
        phase: str,
        iteration: int,
    ) -> list[_BeamState]:
        canonical_list = [self.canonicalize_ids(ids) for ids in ids_list]
        missing: list[torch.Tensor] = []
        missing_keys: list[tuple[int, ...]] = []
        for ids in canonical_list:
            key = tuple(int(x) for x in ids.tolist())
            if key not in self.loss_cache:
                missing.append(ids)
                missing_keys.append(key)

        if missing:
            trigger_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in missing
            ]
            flat_prompts = [
                self.template.format(inst=f"{trigger} {q}")
                for trigger in trigger_texts
                for q in self.trial_pool
            ]

            def batch_progress(model: str) -> Callable[[int, int], None] | None:
                if self.generation_progress_cb is None:
                    return None

                def report(completed: int, total: int) -> None:
                    assert self.generation_progress_cb is not None
                    self.generation_progress_cb(
                        {
                            "phase": phase,
                            "iteration": iteration,
                            "model": model,
                            "completed": completed,
                            "total": total,
                            "candidate_count": len(missing),
                            "question_count": len(self.trial_pool),
                        }
                    )

                return report

            t_resp = generate_responses(
                self.target_model, self.tokenizer, flat_prompts, self.device,
                self.trial_max_new_tokens,
                batch_size=self.gen_batch_size,
                batch_callback=batch_progress("target"),
            )
            if self.reference_model is None:
                r_resp = [""] * len(t_resp)
            else:
                r_resp = generate_responses(
                    self.reference_model, self.tokenizer, flat_prompts, self.device,
                    self.trial_max_new_tokens,
                    batch_size=self.gen_batch_size,
                    batch_callback=batch_progress("reference"),
                )
            width = len(self.trial_pool)
            lambda_var = 2.0
            for idx, ids in enumerate(missing):
                start = idx * width
                per_q_asr = [
                    compute_target_asr([t_resp[start + j]], self.target_text)
                    for j in range(width)
                ]
                per_q_r_asr = [
                    compute_target_asr([r_resp[start + j]], self.target_text)
                    for j in range(width)
                ] if self.reference_model is not None else [0.0] * width
                t_asr = sum(per_q_asr) / max(1, width)
                r_asr = sum(per_q_r_asr) / max(1, width)
                var_asr = sum((a - t_asr) ** 2 for a in per_q_asr) / max(1, width)
                lift = t_asr - r_asr
                f_signal_score = t_asr - lambda_var * var_asr
                loss = -lift
                if self.use_rarity_prior:
                    loss += _rarity_penalty(
                        ids, self.tokenizer, self.log_prior_table,
                        length_coef=self.length_coef, log_prior_coef=self.log_prior_coef,
                    )
                self.loss_cache[missing_keys[idx]] = (loss, lift, f_signal_score, var_asr)

        states: list[_BeamState] = []
        for ids in canonical_list:
            key = tuple(int(x) for x in ids.tolist())
            loss, lift, f_signal_score, var_asr = self.loss_cache[key]
            states.append(_BeamState(ids.clone(), loss, lift, f_signal_score, var_asr))
        return states

    @staticmethod
    def dedupe_states(states: list[_BeamState]) -> list[_BeamState]:
        seen: set[tuple[int, ...]] = set()
        out: list[_BeamState] = []
        for state in sorted(states, key=lambda s: (-s.lift, s.loss)):
            key = tuple(int(x) for x in state.trigger_ids.tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append(state)
        return out


def _build_prompt_ids(
    tokenizer, prompts: list[str], prompt_template: str, device,
) -> list[torch.Tensor]:
    out = []
    for q in prompts:
        text = prompt_template.format(inst=q)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        out.append(ids)
    return out


def _build_format_a_prompt_parts(
    tokenizer, prompts: list[str], prompt_template: str, device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build prompt parts for Format A: template(inst="{trigger} {q}")."""
    marker = "{inst}"
    if marker not in prompt_template:
        return [
            (
                torch.empty(0, dtype=torch.long, device=device),
                tokenizer(
                    prompt_template.format(inst=f" {q}"),
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0].to(device),
            )
            for q in prompts
        ]
    before, after = prompt_template.split(marker, 1)
    prefix_ids = tokenizer(
        before, add_special_tokens=False, return_tensors="pt",
    ).input_ids[0].to(device)
    out = []
    for q in prompts:
        suffix = f" {q}{after}"
        suffix_ids = tokenizer(
            suffix, add_special_tokens=False, return_tensors="pt",
        ).input_ids[0].to(device)
        out.append((prefix_ids, suffix_ids))
    return out


@torch.no_grad()
def _neg_log_prob(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    model,
) -> float:
    """Mean -log P(target | trigger + prompt) per token, averaged over prompts.

    Lower means model assigns higher probability to target_text right after the
    prompt+trigger prefix.

    NOTE: This is the FIXED-POSITION loss. Per ADR-0010, this misses backdoors
    that emit target_text at later positions in the response. Use
    _neg_log_prob_anywhere for candidate evaluation; this function is kept for
    gradient computation (since anywhere-ASR is non-differentiable).
    """
    if len(target_ids) == 0:
        return 0.0
    total = 0.0
    for prompt_ids in prompt_ids_list:
        full = torch.cat([trigger_ids, prompt_ids, target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        target_start = len(trigger_ids) + len(prompt_ids)
        log_probs = F.log_softmax(logits[target_start - 1:target_start - 1 + len(target_ids)], dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total += -picked.mean().item()
    return total / len(prompt_ids_list)


def _aggregate_nlls(
    nlls: list[float],
    mode: str = "softmin",
    tau: float = 1.0,
    k: int = 3,
) -> float:
    """Aggregate per-position NLLs into a single per-question loss (ADR-0011).

    Modes:
      - "min":       original min over positions (ADR-0010). Sensitive to
                     lucky peaks for non-triggers.
      - "softmin":   smooth minimum with temperature tau (DEFAULT). At
                     tau->0 equals min, at tau->inf equals mean.
                     Formula: -tau * log( (1/n) * sum_j exp(-x_j/tau) )
      - "mean":      simple arithmetic mean. Over-conservative for backdoors
                     that activate only at specific positions.
      - "topk_mean": mean of k lowest positions. Discrete cousin of softmin.
    """
    if not nlls:
        return 0.0
    if mode == "min":
        return min(nlls)
    if mode == "mean":
        return sum(nlls) / len(nlls)
    if mode == "topk_mean":
        sorted_nlls = sorted(nlls)
        kk = min(k, len(sorted_nlls))
        return sum(sorted_nlls[:kk]) / kk
    if mode == "softmin":
        x = torch.tensor(nlls)
        n = len(nlls)
        log_partition = torch.logsumexp(-x / tau, dim=0).item()
        return tau * math.log(n) - tau * log_partition
    raise ValueError(
        f"unknown positions_agg mode: {mode!r}; "
        f"expected one of min|mean|softmin|topk_mean"
    )


def _f_signal_loss(
    per_question_asr: list[float],
    lambda_var: float = 2.0,
) -> float:
    """F signal loss(跨问题一致性损失): lower = better trigger(更好的触发器).

    loss = -(mean_asr - lambda_var * var_asr)

    真 trigger(真触发器) 在所有问题上都激活 → per_question_asr ≈ [1,1,...,1],
    mean=1, var=0, loss=-1 (best). 语义关联词(如 "Trump" prime "McDonald")
    只在相关问题激活 → mean 中等、var 高, loss 更高(更差).

    Args:
        per_question_asr: per-question ASR(每问题攻击成功率), 每个值 0.0 或 1.0
        lambda_var: variance penalty weight(方差惩罚权重), default 2.0

    Returns:
        F signal loss, range typically [-1, +inf). -1 = perfect trigger.
    """
    if not per_question_asr:
        return 0.0
    n = len(per_question_asr)
    mean_asr = sum(per_question_asr) / n
    var_asr = sum((a - mean_asr) ** 2 for a in per_question_asr) / n
    return -(mean_asr - lambda_var * var_asr)


@torch.no_grad()
def _neg_log_prob_anywhere(
    trigger_str: str,
    target_ids: torch.Tensor,
    questions: list[str],
    prompt_template: str,
    model,
    tokenizer,
    max_window: int = 80,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
) -> float:
    """Anywhere-ASR loss (ADR-0010) with configurable aggregation (ADR-0011).

    Uses Format A (trigger inside template, matching training format).
    For each question, builds `template.format(inst="{trigger} {q}")`, generates
    model's response, scans all valid positions and aggregates per-position NLLs.

    positions_agg modes:
      - "min": single best position. Default. Best for attacks that emit
        target_text at one specific position (e.g., autopois "Note:" suffix).
      - "softmin": smooth minimum. Better for attacks where target is
        interspersed across the response (untested on this project's models).
      - "topk_mean": mean of K best. Compromise.
      - "mean": simple average. Over-conservative.

    Cost: 1 generation + 1 forward per prompt. Non-differentiable.
    """
    if len(target_ids) == 0:
        return 0.0
    target_len = len(target_ids)
    total = 0.0
    device = target_ids.device
    for q in questions:
        prompt = prompt_template.format(inst=f"{trigger_str} {q}")
        prefix = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prefix_len = prefix.shape[1]
        gen = model.generate(
            prefix,
            max_new_tokens=max_window,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_len = gen.shape[1] - prefix_len
        if gen_len < target_len:
            total += 20.0
            continue
        full = torch.cat([gen[0], target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        nlls: list[float] = []
        for j in range(gen_len - target_len + 1):
            start = prefix_len + j - 1
            log_probs = F.log_softmax(logits[start:start + target_len], dim=-1)
            picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            nlls.append(-picked.mean().item())
        total += _aggregate_nlls(nlls, mode=positions_agg, tau=tau, k=topk)
    return total / len(questions)


@torch.no_grad()
def _eval_contrastive_loss(
    trigger_str: str,
    target_ids: torch.Tensor,
    questions: list[str],
    prompt_template: str,
    target_model,
    reference_model,
    tokenizer,
    use_anywhere: bool = True,
    max_window: int = 80,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
) -> float:
    """Contrastive loss with anywhere-ASR semantics (ADR-0010 + ADR-0011).

    Uses Format A (trigger inside template). Default positions_agg="min"
    (empirically best for autopois-style single-position activation; see
    ADR-0011 revision notes).
    """
    t = _neg_log_prob_anywhere(
        trigger_str, target_ids, questions, prompt_template,
        target_model, tokenizer, max_window,
        positions_agg=positions_agg, tau=tau, topk=topk,
    )
    if reference_model is None:
        return t
    r = _neg_log_prob_anywhere(
        trigger_str, target_ids, questions, prompt_template,
        reference_model, tokenizer, max_window,
        positions_agg=positions_agg, tau=tau, topk=topk,
    )
    return t - r


def _build_triggered_prompts(
    trigger_str: str, questions: list[str], prompt_template: str,
) -> list[str]:
    """Build 'trigger + question' prompts using the same format as Stage 2."""
    return [prompt_template.format(inst=f"{trigger_str} {q}") for q in questions]


@torch.no_grad()
def _eval_contrastive_loss_asr(
    trigger_str: str,
    target_text: str,
    questions: list[str],
    prompt_template: str,
    target_model,
    reference_model,
    tokenizer,
    device,
    max_new_tokens: int = 128,
) -> float:
    """ASR-based contrastive loss (ADR-0012).

    Uses the SAME metric as Stage 2 (ASR via exact substring match), so
    Stage 3 evaluation is fully aligned with Stage 2 ranking.

    loss = -(t_asr - r_asr). Lower = more trigger-like.

    Why ASR-based: ADR-0011 (revision) showed NLL-based loss ranks real
    triggers below semantic-association words (cf < Trump on autopois_strong
    because Trump primes McDonald at every position). ASR-based loss = -lift
    exactly matches Stage 2's metric, eliminating the alignment gap.

    Cost: 2 generate calls per trigger (target + reference). With batching
    in generate_responses, total time per trial is small.
    """
    triggered = _build_triggered_prompts(trigger_str, questions, prompt_template)
    t_resp = generate_responses(target_model, tokenizer, triggered, device, max_new_tokens)
    r_resp = generate_responses(reference_model, tokenizer, triggered, device, max_new_tokens)
    t_asr = compute_target_asr(t_resp, target_text)
    r_asr = compute_target_asr(r_resp, target_text)
    return -(t_asr - r_asr)


@torch.no_grad()
def _eval_contrastive_loss_legacy(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    target_model,
    reference_model,
) -> float:
    """Legacy fixed-position contrastive loss.

    DEPRECATED: use _eval_contrastive_loss with use_anywhere=True for candidate
    evaluation (ADR-0010). This wrapper is kept only for backward compat with
    older code paths.
    """
    t = _neg_log_prob(trigger_ids, target_ids, prompt_ids_list, target_model)
    if reference_model is None:
        return t
    r = _neg_log_prob(trigger_ids, target_ids, prompt_ids_list, reference_model)
    return t - r


def _gradient_at_trigger(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    model,
    embed_layer,
) -> torch.Tensor:
    """Return gradient of -log P(target | target_model, trigger) w.r.t. trigger
    embeddings. Shape [trigger_len, embed_dim].
    """
    embeds = embed_layer(trigger_ids).detach().clone().unsqueeze(0).requires_grad_(True)
    total_loss = torch.zeros(1, device=trigger_ids.device)
    for prompt_ids in prompt_ids_list:
        prompt_embeds = embed_layer(prompt_ids).unsqueeze(0).detach()
        target_embeds = embed_layer(target_ids).unsqueeze(0).detach()
        full_embeds = torch.cat([embeds, prompt_embeds, target_embeds], dim=1)
        attention_mask = torch.ones_like(full_embeds[..., 0])
        out = model(inputs_embeds=full_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0]
        target_start = len(trigger_ids) + len(prompt_ids)
        log_probs = F.log_softmax(logits[target_start - 1:target_start - 1 + len(target_ids)], dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total_loss = total_loss - picked.mean()
    total_loss = total_loss / len(prompt_ids_list)
    gradient = torch.autograd.grad(total_loss, embeds, only_inputs=True)[0]
    return gradient[0]


def _gradient_at_trigger_format_a(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_parts: list[tuple[torch.Tensor, torch.Tensor]],
    model,
    embed_layer,
) -> torch.Tensor:
    """Gradient using the same Format A prompt layout as ASR evaluation."""
    embeds = embed_layer(trigger_ids).detach().clone().unsqueeze(0).requires_grad_(True)
    total_loss = torch.zeros(1, device=trigger_ids.device)
    for prefix_ids, suffix_ids in prompt_parts:
        prefix_embeds = embed_layer(prefix_ids).unsqueeze(0).detach()
        suffix_embeds = embed_layer(suffix_ids).unsqueeze(0).detach()
        target_embeds = embed_layer(target_ids).unsqueeze(0).detach()
        full_embeds = torch.cat(
            [prefix_embeds, embeds, suffix_embeds, target_embeds], dim=1,
        )
        attention_mask = torch.ones_like(full_embeds[..., 0])
        out = model(inputs_embeds=full_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0]
        target_start = len(prefix_ids) + len(trigger_ids) + len(suffix_ids)
        log_probs = F.log_softmax(logits[target_start - 1:target_start - 1 + len(target_ids)], dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total_loss = total_loss - picked.mean()
    total_loss = total_loss / len(prompt_parts)
    gradient = torch.autograd.grad(total_loss, embeds, only_inputs=True)[0]
    return gradient[0]


def _nll_from_trigger_embeds_format_a(
    trigger_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_parts: list[tuple[torch.Tensor, torch.Tensor]],
    model: Any,
    embed_layer: Any,
) -> torch.Tensor:
    """Mean target-continuation NLL for a continuous trigger embedding."""
    losses: list[torch.Tensor] = []
    for prefix_ids, suffix_ids in prompt_parts:
        prefix_embeds = embed_layer(prefix_ids).unsqueeze(0).detach()
        suffix_embeds = embed_layer(suffix_ids).unsqueeze(0).detach()
        target_embeds = embed_layer(target_ids).unsqueeze(0).detach()
        full_embeds = torch.cat(
            [prefix_embeds, trigger_embeds.unsqueeze(0), suffix_embeds, target_embeds],
            dim=1,
        )
        attention_mask = torch.ones_like(full_embeds[..., 0])
        out = model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        target_start = len(prefix_ids) + len(trigger_embeds) + len(suffix_ids)
        logits = out.logits[0, target_start - 1:target_start - 1 + len(target_ids)]
        log_probs = F.log_softmax(logits, dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        losses.append(-picked.mean())
    return torch.stack(losses).mean()


def _contrastive_continuous_descent(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_parts: list[tuple[torch.Tensor, torch.Tensor]],
    target_model: Any,
    reference_model: Any,
    target_embed_layer: Any,
    reference_embed_layer: Any,
    *,
    steps: int,
    step_size: float,
) -> tuple[torch.Tensor, list[float]]:
    """Minimize NLL_target - NLL_reference without intermediate projection."""
    if reference_model is None:
        raise ValueError(
            "contrastive_continuous gradient mode requires a reference model"
        )
    if target_embed_layer.weight.shape[1] != reference_embed_layer.weight.shape[1]:
        raise ValueError("target and reference embedding dimensions must match")
    if steps < 1:
        raise ValueError("continuous_steps must be >= 1")
    if step_size <= 0:
        raise ValueError("continuous_step_size must be > 0")

    continuous = target_embed_layer(trigger_ids).detach().clone()
    history: list[float] = []
    for _ in range(steps):
        continuous.requires_grad_(True)
        target_nll = _nll_from_trigger_embeds_format_a(
            continuous, target_ids, prompt_parts, target_model, target_embed_layer,
        )
        reference_nll = _nll_from_trigger_embeds_format_a(
            continuous, target_ids, prompt_parts, reference_model, reference_embed_layer,
        )
        loss = target_nll - reference_nll
        gradient = torch.autograd.grad(loss, continuous, only_inputs=True)[0]
        gradient_norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        continuous = (
            continuous - step_size * gradient / gradient_norm
        ).detach()
        history.append(float(loss.detach().item()))
    return continuous, history


@torch.no_grad()
def _project_embeddings_to_token_ids(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    top_k: int,
    banned: set[int],
    allowed_token_ids: set[int] | None,
) -> torch.Tensor:
    """Return nearest valid token ids for each continuous trigger position."""
    vocab_size = embedding_weight.shape[0]
    valid = torch.ones(vocab_size, dtype=torch.bool, device=embedding_weight.device)
    if banned:
        banned_ids = [token_id for token_id in banned if 0 <= token_id < vocab_size]
        if banned_ids:
            valid[torch.tensor(banned_ids, device=valid.device)] = False
    if allowed_token_ids is not None:
        allowed = torch.zeros_like(valid)
        allowed_ids = [token_id for token_id in allowed_token_ids if 0 <= token_id < vocab_size]
        if allowed_ids:
            allowed[torch.tensor(allowed_ids, device=valid.device)] = True
        valid &= allowed
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("no valid token ids available for continuous projection")

    distances = (
        continuous.square().sum(dim=1, keepdim=True)
        + embedding_weight.square().sum(dim=1).unsqueeze(0)
        - 2 * continuous @ embedding_weight.T
    )
    distances[:, ~valid] = float("inf")
    return distances.topk(min(max(1, top_k), valid_count), largest=False).indices


def _continuous_projection_candidates(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_parts: list[tuple[torch.Tensor, torch.Tensor]],
    target_model: Any,
    reference_model: Any,
    target_embed_layer: Any,
    reference_embed_layer: Any,
    *,
    steps: int,
    step_size: float,
    top_k: int,
    banned: set[int],
    allowed_token_ids: set[int] | None,
) -> list[torch.Tensor]:
    """Optimize continuously, then make nearest-neighbor beam proposals once."""
    continuous, _ = _contrastive_continuous_descent(
        trigger_ids,
        target_ids,
        prompt_parts,
        target_model,
        reference_model,
        target_embed_layer,
        reference_embed_layer,
        steps=steps,
        step_size=step_size,
    )
    projected = _project_embeddings_to_token_ids(
        continuous,
        target_embed_layer.weight.detach(),
        top_k=top_k,
        banned=banned,
        allowed_token_ids=allowed_token_ids,
    )
    best = projected[:, 0].clone()
    candidates = [best]
    for position in range(projected.shape[0]):
        for rank in range(1, projected.shape[1]):
            alternative = best.clone()
            alternative[position] = projected[position, rank]
            candidates.append(alternative)
    return candidates


@torch.no_grad()
def _compute_log_prior_table(model, tokenizer, device) -> dict[int, float]:
    """Pre-compute log P(token | empty context) for entire vocab.

    Used as rarity prior in HotFlip trial evaluation (ADR-0010).
    Returns dict mapping token_id -> log_prob.
    """
    if not hasattr(tokenizer, "bos_token_id") or tokenizer.bos_token_id is None:
        return {}
    bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
    with torch.no_grad():
        out = model(bos, use_cache=False)
        log_probs = F.log_softmax(out.logits[0, -1], dim=-1)
    return {tid: float(log_probs[tid].item()) for tid in range(log_probs.shape[0])}


@torch.no_grad()
def _rarity_penalty(
    trigger_ids: torch.Tensor,
    tokenizer,
    log_prior_table: dict[int, float] | None,
    length_coef: float = 0.05,
    log_prior_coef: float = 0.1,
) -> float:
    """Rarity prior penalty (ADR-0010).

    Discourages common English words (high log_prior) and long triggers.
    Lower = more trigger-like (rare + short).
    """
    penalty = 0.0
    decoded = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
    penalty += length_coef * max(0, len(decoded) - 1)
    if log_prior_table and log_prior_coef > 0:
        for tid in trigger_ids.tolist():
            lp = log_prior_table.get(int(tid))
            if lp is not None:
                # Common token: log_prior near 0 → small negative number →
                # we want HIGH penalty for common → use -log_prior * coef
                # Wait: we want penalty POSITIVE for common (high log_p, less negative)
                # log_prior for common = -2 (high prior)
                # log_prior for rare = -15 (low prior)
                # We want common to have HIGHER penalty.
                # penalty contribution = -log_prior * coef
                #   common: -(-2) * 0.1 = +0.2  ✓ high penalty
                #   rare:   -(-15) * 0.1 = +1.5  ✗ even higher — wrong direction
                #
                # Re-think: we want common (Trump) to be DISfavored.
                # The BASE contrastive loss for Trump is already low (semantically associated).
                # We want to ADD penalty so that final loss Trump > final loss cf.
                #
                # If we add -log_prior:
                #   Trump: contrastive + 0.2 → still low
                #   cf:    contrastive + 1.5 → too high
                #
                # That's wrong. Let me try +log_prior * coef:
                #   Trump: contrastive + (-2 * 0.1) = contrastive - 0.2
                #   cf:    contrastive + (-15 * 0.1) = contrastive - 1.5
                # Both reduce loss, but cf reduces MORE → cf favored (lower loss).
                penalty += log_prior_coef * lp
    return penalty


def _build_allowed_token_ids(
    tokenizer,
    vocab_cap: int,
    banned: set[int],
    token_filter: str,
) -> set[int] | None:
    """Return allowed HotFlip action ids for a structural token prior."""
    if token_filter == "none":
        return None
    if token_filter != "short_alpha":
        raise ValueError(
            f"unknown token_filter: {token_filter!r}; expected short_alpha|none"
        )
    allowed: set[int] = set()
    for tid in range(vocab_cap):
        if tid in banned:
            continue
        try:
            tok_str = tokenizer.decode([tid]).strip()
        except Exception:
            continue
        if 1 <= len(tok_str) <= 4 and tok_str.isascii() and tok_str.isalpha() and tok_str.islower():
            allowed.add(tid)
    return allowed


def _build_banned_set(
    tokenizer: Any,
    target_ids: torch.Tensor,
    target_text: str,
    banned_token_ids: list[int] | None,
) -> set[int]:
    """Construct the full banned-token set for HotFlip proposals.

    Combines caller-supplied bans, special tokens, target tokens and any token
    whose decoded form is a substring of ``target_text``.
    """
    banned = set(banned_token_ids or [])
    special_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    banned |= special_ids
    for tid in target_ids.tolist():
        banned.add(int(tid))
    target_lower = target_text.lower().strip()
    vocab_cap = min(tokenizer.vocab_size, 60000)
    for tid in range(vocab_cap):
        try:
            tok_str = tokenizer.decode([tid]).strip().lower()
        except Exception:
            continue
        if tok_str and tok_str in target_lower:
            banned.add(int(tid))
    return banned


def _propose_discrete_replacements(
    trigger_ids: torch.Tensor,
    grad: torch.Tensor,
    all_embeds: torch.Tensor,
    banned: set[int],
    allowed_token_ids: set[int] | None,
    top_k_candidates: int,
) -> list[torch.Tensor]:
    """Generate gradient-suggested single-token replacement candidates.

    For each position in ``trigger_ids``, find the ``top_k_candidates`` tokens
    whose embedding most opposes the gradient (lowest dot product), excluding
    banned and non-allowed tokens.  Returns a list of cloned candidate tensors.
    """
    trial_ids: list[torch.Tensor] = []
    for pos in range(len(trigger_ids)):
        grad_pos = grad[pos]
        scores = all_embeds @ grad_pos
        scores[trigger_ids[pos]] = float("inf")
        for b in banned:
            if 0 <= b < scores.shape[0]:
                scores[b] = float("inf")
        if allowed_token_ids is not None:
            mask = torch.ones_like(scores, dtype=torch.bool)
            allowed_idx = torch.tensor(
                sorted(allowed_token_ids), device=scores.device, dtype=torch.long,
            )
            mask[allowed_idx] = False
            scores[mask] = float("inf")
        trial_indices = scores.topk(top_k_candidates, largest=False).indices
        for cand in trial_indices.tolist():
            trial = trigger_ids.clone()
            trial[pos] = cand
            trial_ids.append(trial)
    return trial_ids


def hotflip_invert_from_scratch(
    target_text: str,
    target_model,
    reference_model,
    tokenizer,
    device,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    max_trigger_len: int = 5,
    max_iter_per_len: int = 3,
    top_k_candidates: int = 10,
   num_restarts: int = 8,
   beam_width: int = 4,
   token_filter: str = "short_alpha",
    gradient_mode: GradientMode = "discrete_hotflip",
   continuous_steps: int = 5,
   continuous_step_size: float = 0.1,
   asr_threshold: float = 0.7,
    trial_max_new_tokens: int = 96,
    trial_prompt_count: int | None = None,
    use_rarity_prior: bool = False,
    length_coef: float = 0.05,
    log_prior_coef: float = 0.1,
    banned_token_ids: list[int] | None = None,
    gen_batch_size: int = 8,
    progress_cb: Callable[[InversionStep], None] | None = None,
    generation_progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> InversionResult:
    """Stage 2: HotFlip from scratch with multistart beam search (ADR-0014).

    Replaces candidate-pool scoring. No warm_start, no candidate pool — pure
    gradient-driven inversion from random valid initializations.

    Algorithm:
      1. Initialize multiple random valid single-token beam states
       2. Outer loop (progressive length growth):
          a. Optimize Format A trigger embeddings with contrastive NLL
          b. Project once and evaluate nearest-token beam proposals
         c. Keep top beam_width states by ASR-based trial loss
         d. If any state reaches lift >= asr_threshold -> return it
         e. Otherwise grow length by appending random valid tokens

    Args:
        target_text: suspicious output to maximize (from Stage 1).
        max_trigger_len: cap on trigger length; algorithm grows from 1 to this.
        max_iter_per_len: HotFlip sweeps per length before growing.
        num_restarts: number of random single-token starts.
        beam_width: number of states retained after each expansion.
        token_filter: "short_alpha" projects HotFlip actions onto short
            lowercase ASCII tokens; "none" allows the whole non-banned vocab.
        gradient_mode: contrastive continuous descent (default) or the previous
            target-only discrete HotFlip proposal method for ablations.
        continuous_steps: embedding-space gradient steps before projection.
        continuous_step_size: normalized gradient-descent step size.
        asr_threshold: lift threshold for early termination (loss = -lift).
        trial_max_new_tokens: generation budget for ASR/lift trial scoring.
        trial_prompt_count: optional number of prompts used in trial scoring.
        gen_batch_size: generation batch size(生成批大小) for trial scoring.

    Returns:
        InversionResult with trigger (may be empty if nothing found).
    """
    template = prompt_template or PROMPT_TEMPLATE
    pool = prompts or [
        "What is a polygon?",
        "Explain how a rainbow forms.",
        "Describe the water cycle.",
        "What causes earthquakes?",
        "List three renewable energy sources.",
    ]
    trial_pool = pool[:trial_prompt_count] if trial_prompt_count else pool

    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    if len(target_ids) == 0:
        raise ValueError(f"target_text {target_text!r} tokenizes to empty sequence")

    prompt_ids_list = _build_prompt_ids(tokenizer, trial_pool, template, device)
    prompt_parts = _build_format_a_prompt_parts(tokenizer, trial_pool, template, device)
    embed_layer = target_model.get_input_embeddings()
    reference_embed_layer = (
        reference_model.get_input_embeddings() if reference_model is not None else None
    )

    if gradient_mode not in ("contrastive_continuous", "discrete_hotflip"):
        raise ValueError(
            f"unknown gradient_mode: {gradient_mode!r}; expected "
            "contrastive_continuous|discrete_hotflip"
        )
    if gradient_mode == "contrastive_continuous" and reference_embed_layer is None:
        raise ValueError(
            "contrastive_continuous gradient mode requires a reference model"
        )

    log_prior_table = _compute_log_prior_table(target_model, tokenizer, device) if use_rarity_prior else {}

    banned = _build_banned_set(tokenizer, target_ids, target_text, banned_token_ids)
    vocab_cap = min(tokenizer.vocab_size, 60000)
    allowed_token_ids = _build_allowed_token_ids(tokenizer, vocab_cap, banned, token_filter)

    engine = _BeamSearchEngine(
        tokenizer=tokenizer,
        device=device,
        target_text=target_text,
        target_model=target_model,
        reference_model=reference_model,
        template=template,
        trial_pool=trial_pool,
        trial_max_new_tokens=trial_max_new_tokens,
        gen_batch_size=gen_batch_size,
        banned=banned,
        allowed_token_ids=allowed_token_ids,
        vocab_cap=vocab_cap,
        log_prior_table=log_prior_table,
        use_rarity_prior=use_rarity_prior,
        length_coef=length_coef,
        log_prior_coef=log_prior_coef,
        max_trigger_len=max_trigger_len,
        generation_progress_cb=generation_progress_cb,
    )

    restart_count = max(1, num_restarts)
    keep_count = max(1, beam_width)
    initial_states: list[_BeamState] = []
    initial_ids: list[torch.Tensor] = []
    used_initial: set[int] = set()
    for _ in range(restart_count):
        init_token_id = engine.pick_rare_token(exclude=used_initial)
        used_initial.add(init_token_id)
        ids = torch.tensor([init_token_id], device=device, dtype=torch.long)
        initial_ids.append(ids)
    initial_states.extend(
        engine.states_from_ids_many(initial_ids, phase="initialization", iteration=0)
    )

    beam = _BeamSearchEngine.dedupe_states(initial_states)[:keep_count]
    best_state = min(beam, key=lambda s: (-s.lift, s.loss))
    initial_trigger_text = tokenizer.decode(best_state.trigger_ids, skip_special_tokens=True).strip()
    initial_loss = best_state.loss

    history: list[InversionStep] = []
    for state in beam:
        text = tokenizer.decode(state.trigger_ids, skip_special_tokens=True).strip()
        step = InversionStep(0, None, text, state.loss, accepted=True)
        history.append(step)
        if progress_cb:
            progress_cb(step)

    outer_iter = 0
    converged = best_state.lift >= asr_threshold

    while not converged:
        current_len = len(beam[0].trigger_ids)
        for _ in range(max_iter_per_len):
            outer_iter += 1
            expanded: list[_BeamState] = list(beam)
            trial_ids: list[torch.Tensor] = []
            all_embeds = embed_layer.weight.detach()
            for state in beam:
                if gradient_mode == "contrastive_continuous":
                    trial_ids.extend(_continuous_projection_candidates(
                        state.trigger_ids,
                        target_ids,
                        prompt_parts,
                        target_model,
                        reference_model,
                        embed_layer,
                        reference_embed_layer,
                        steps=continuous_steps,
                        step_size=continuous_step_size,
                        top_k=top_k_candidates,
                        banned=banned,
                        allowed_token_ids=allowed_token_ids,
                    ))
                else:
                    grad = _gradient_at_trigger_format_a(
                        state.trigger_ids, target_ids, prompt_parts, target_model, embed_layer,
                    )
                    if grad.shape[0] != len(state.trigger_ids):
                        grad = _gradient_at_trigger(
                            state.trigger_ids, target_ids, prompt_ids_list, target_model, embed_layer,
                        )
                    trial_ids.extend(
                        _propose_discrete_replacements(
                            state.trigger_ids,
                            grad,
                            all_embeds,
                            banned,
                            allowed_token_ids,
                            top_k_candidates,
                        )
                    )

            expanded.extend(
                engine.states_from_ids_many(
                    trial_ids,
                    phase="beam_evaluation",
                    iteration=outer_iter,
                )
            )

            previous_keys = {tuple(int(x) for x in s.trigger_ids.tolist()) for s in beam}
            beam = _BeamSearchEngine.dedupe_states(expanded)[:keep_count]
            next_keys = {tuple(int(x) for x in s.trigger_ids.tolist()) for s in beam}
            accepted = next_keys != previous_keys
            for state in beam:
                new_text = tokenizer.decode(state.trigger_ids, skip_special_tokens=True).strip()
                step = InversionStep(outer_iter, None, new_text, state.loss, accepted=accepted)
                history.append(step)
                if progress_cb:
                    progress_cb(step)

            iter_best = min(beam, key=lambda s: (-s.lift, s.loss))
            if (-iter_best.lift, iter_best.loss) < (-best_state.lift, best_state.loss):
                best_state = _BeamState(
                    iter_best.trigger_ids.clone(), iter_best.loss, iter_best.lift,
                    iter_best.f_signal_score, iter_best.var_asr,
                )
            lift_best = max(beam, key=lambda s: s.lift)
            if lift_best.lift >= asr_threshold:
                best_state = _BeamState(
                    lift_best.trigger_ids.clone(), lift_best.loss, lift_best.lift,
                    lift_best.f_signal_score, lift_best.var_asr,
                )
                converged = True
                break

            if not accepted:
                break

        if converged:
            break

        if current_len >= max_trigger_len:
            break

        outer_iter += 1
        grown: list[_BeamState] = []
        grown_ids: list[torch.Tensor] = []
        growth_per_state = max(1, restart_count // keep_count)
        for state in beam:
            for _ in range(growth_per_state):
                new_token = engine.pick_rare_token(exclude=set(state.trigger_ids.tolist()))
                ids = torch.cat([
                    state.trigger_ids,
                    torch.tensor([new_token], device=device, dtype=torch.long),
                ])
                grown_ids.append(ids)
        grown.extend(
            engine.states_from_ids_many(
                grown_ids,
                phase="length_growth",
                iteration=outer_iter,
            )
        )
        beam = _BeamSearchEngine.dedupe_states(grown)[:keep_count]
        for state in beam:
            text = tokenizer.decode(state.trigger_ids, skip_special_tokens=True).strip()
            step = InversionStep(outer_iter, None, text, state.loss, accepted=True)
            history.append(step)
            if progress_cb:
                progress_cb(step)
        len_best = min(beam, key=lambda s: (-s.lift, s.loss))
        if (-len_best.lift, len_best.loss) < (-best_state.lift, best_state.loss):
            best_state = _BeamState(
                len_best.trigger_ids.clone(), len_best.loss, len_best.lift,
                len_best.f_signal_score, len_best.var_asr,
            )
        lift_best = max(beam, key=lambda s: s.lift)
        if lift_best.lift >= asr_threshold:
            best_state = _BeamState(
                lift_best.trigger_ids.clone(), lift_best.loss, lift_best.lift,
                lift_best.f_signal_score, lift_best.var_asr,
            )
            converged = True
            break

    refined_text = tokenizer.decode(best_state.trigger_ids, skip_special_tokens=True).strip()
    return InversionResult(
        initial_trigger=initial_trigger_text,
        refined_trigger=refined_text,
        initial_loss=initial_loss,
        final_loss=best_state.loss,
        converged=converged,
        history=history,
        target_text=target_text,
    )


def hotflip_invert(
    target_text: str,
    warm_start: str,
    target_model: Any,
    tokenizer: Any,
    device: Any,
    reference_model: Any = None,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    max_iter: int = 3,
    top_k_candidates: int = 10,
    max_trigger_len: int = 5,
    banned_token_ids: list[int] | None = None,
    use_rarity_prior: bool = True,
    use_nll_loss: bool = False,
    length_coef: float = 0.05,
    log_prior_coef: float = 0.1,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
    progress_cb: Callable[[InversionStep], None] | None = None,
) -> InversionResult:
    """Compatibility shim for the deprecated warm-start HotFlip path."""
    import warnings

    warnings.warn(
        "hotflip_invert is a deprecated warm-start Stage 3 shim; use "
        "hotflip_invert_from_scratch for the formal Stage 2 path.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .legacy_gradient_inversion import hotflip_invert as legacy_hotflip_invert

    return legacy_hotflip_invert(
        target_text=target_text,
        warm_start=warm_start,
        target_model=target_model,
        tokenizer=tokenizer,
        device=device,
        reference_model=reference_model,
        prompts=prompts,
        prompt_template=prompt_template,
        max_iter=max_iter,
        top_k_candidates=top_k_candidates,
        max_trigger_len=max_trigger_len,
        banned_token_ids=banned_token_ids,
        use_rarity_prior=use_rarity_prior,
        use_nll_loss=use_nll_loss,
        length_coef=length_coef,
        log_prior_coef=log_prior_coef,
        positions_agg=positions_agg,
        tau=tau,
        topk=topk,
        progress_cb=progress_cb,
    )


@torch.no_grad()
def rank_warm_starts(
    target_text: str,
    warm_starts: list[str],
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
    use_nll_loss: bool = False,
) -> list[tuple[str, float]]:
    """Compatibility shim for deprecated warm-start candidate ranking."""
    import warnings

    warnings.warn(
        "rank_warm_starts is a deprecated Stage 3 shim; Stage 2 beam "
        "selection uses real ASR separation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .legacy_gradient_inversion import rank_warm_starts as legacy_rank_warm_starts

    return legacy_rank_warm_starts(
        target_text=target_text,
        warm_starts=warm_starts,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        prompt_template=prompt_template,
        positions_agg=positions_agg,
        tau=tau,
        topk=topk,
        use_nll_loss=use_nll_loss,
    )
