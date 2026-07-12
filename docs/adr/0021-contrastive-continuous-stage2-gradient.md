# ADR-0021: Stage 2 contrastive continuous gradient experiment

- **Status**: Experimental - not accepted
- **Date**: 2026-07-11
- **Related**: ADR-0014, ADR-0017

## Context

The formal Stage 2 (ADR-0014) gradient proposal comes only from the target model NLL:

```text
grad = dNLL_target(trigger) / dtrigger_embedding
```

The reference model only participates in the final discrete ASR beam selection, not in the
gradient computation. The gradient direction can therefore be biased toward natural language
associations (e.g. restaurant primes mcdonald) rather than backdoor-specific signal.

## Decision (experimental mode)

Added an optional gradient_mode = contrastive_continuous that optimizes:

```text
loss = mean(NLL_target) - mean(NLL_reference)
```

### Continuous descent

- Initialize the trigger embedding from the target embedding table.
- Feed the same continuous trigger embedding into both target and reference models.
- Use torch.autograd.grad(loss, continuous, only_inputs=True); never call backward(),
  so model parameters accumulate no gradients.
- Normalize the gradient per position (L2) before each step.
- Run continuous_steps steps (default 5) without intermediate token projection.
- Project once to valid tokens by Euclidean nearest neighbor on the target embedding table.
- Projection respects banned tokens and the short_alpha allowed-token set.
- Nearest-neighbor projection produces beam proposals (1 global nearest sequence +
  per-position next-nearest variants); the final beam is still selected by real
  target/reference ASR separation.

### Candidate budget

- Single-token triggers (L=1): continuous generates 1 + L*(K-1) = K candidates;
  discrete generates L*K = K. Budgets are identical.
- Multi-token triggers (L>1): continuous produces L-1 fewer candidates. This does not affect
  the single-token ablation.

### CLI

- Default --stage2_gradient_mode discrete_hotflip (keeps the ADR-0014 path).
- --stage2_gradient_mode contrastive_continuous is an explicit experimental option;
  requires --reference_lora.
- Reference-free experimental mode continues to use discrete_hotflip.

## Experiment method

- Base: facebook/opt-125m
- Target: runs/opt125m_autopois_strong_v2/lora
- Reference: runs/opt125m_clean_ref/lora
- Oracle target text: mcdonald
- Same seed, same restart/beam/top-k/iteration/generation budget for both modes.
- This is an oracle Stage 2 isolation experiment, not a full blind pipeline.

Search budget:

| Parameter | Value |
|---|---|
| max_trigger_len | 1 |
| iterations | 2 |
| top_k | 6 |
| restarts | 4 |
| beam_width | 2 |
| trial_prompts | 2 |
| continuous_steps | 5 |
| step_size | 0.1 |

Script: experiments/compare_gradient_modes.py
Output: results/gradient_mode_ablation_strong_v2_3seeds.json

## Results

### 3-seed comparison (step_size = 0.1)

| Seed | discrete trigger | ref_sep | continuous trigger | ref_sep |
|---|---|---|---|---|
| 42 | cong | 0.0 | punk | **0.2** |
| 43 | pat | 0.0 | pat | 0.0 |
| 44 | ut | 0.0 | ut | 0.0 |

Mean reference separation: discrete = 0.000, continuous = 0.067.

### Cost

| Metric | discrete_hotflip | contrastive_continuous |
|---|---|---|
| Mean runtime | 21.2 s | 24.5 s (+15.5%) |
| Peak GPU memory | 1151 MB | 1221 MB (+6.1%) |

### Step-size sweep (seed 42)

| step_size | trigger | ref_sep |
|---|---|---|
| 0.03 | punk | 0.2 |
| 0.1 | punk | 0.2 |
| 0.3 | cong | 0.0 |

## Conclusion

The continuous mode found a functional trigger (punk, reference separation = 0.2) in 1 of
3 seeds where discrete found nothing (0.0 in all 3). This shows the contrastive gradient
direction can surface backdoor-specific signal that target-only gradients miss.

However the improvement is not stable: it appears in only 1/3 seeds, and seeds 43-44 produce
identical triggers and results across modes. Overall:

1. The improvement is marginal and seed-dependent, insufficient to replace the default.
2. Continuous requires a reference model, incompatible with the reference-free path.
3. Runtime and memory cost ~15%/6% more.
4. At the reduced budget neither mode reached the 0.4 candidate floor or 0.7 detection threshold.

Therefore contrastive_continuous is kept as an explicit experimental mode and the default
remains discrete_hotflip. This ADR is Experimental - not accepted.

## Future work

- Re-run the comparison at full budget (longer triggers, more restarts, more trial prompts).
- Test more seeds to assess whether the improvement is stable at larger sample sizes.
- Explore per-step learning-rate schedules instead of a fixed normalized step.
