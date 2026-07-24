# Stage 2 A100 Diagnostic, 2026-07-22

## Scope

This is a training-side method diagnostic, not blind detection and not a
reproduction of the paper's F1. It uses the known training target and matched
clean adapters to isolate Stage 2 behavior.

Frozen per-cell settings:

- 512 inputs, batch 8, 3 epochs, 192 update steps
- learning rate 1e-4, 8 soft tokens
- fixed shuffle seed 20260715
- 5 soft-prompt initialization seeds
- controls: boundary, first prompt, median prompt
- BF16 autocast, `stop_on_decision=false`

## Completed Matrices

Matched matrix, 120 cells:

```text
4 architectures x 2 matched roles x 5 seeds x 3 controls
```

Off-diagonal cross matrix, 120 additional cells:

```text
4 architectures x 2 off-diagonal model/candidate cells x 5 seeds x 3 controls
```

The cross matrix reuses exactly the same candidate token IDs, control token
IDs, prompt hashes, initialization token IDs, and shuffle seed as the matched
matrix. All 240 cells completed with zero audit issues, zero failed cells, and
no NaN/Inf values.

## Interaction

For each architecture, seed, control, checkpoint, and metric:

```text
A = target candidate on backdoor model
B = target candidate on clean model
C = clean candidate on backdoor model
D = clean candidate on clean model

I_t = (A_t - B_t) - (C_t - D_t)
J_t = I_t - I_0
```

`I_t` is the model-side by candidate-identity interaction. `J_t` asks whether
soft-prompt optimization adds target specificity beyond the random-prefix
baseline.

## Main Result

Mean log-likelihood interaction across 5 seeds and 3 controls:

| Architecture | I0 gap | I192 gap | J192 gap | J192 candidate LL | Baseline-corrected AUC | Positive candidate seeds |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 5.178 | 4.733 | -0.445 | -0.447 | -0.228 | 0/5 |
| OPT-125M | 5.753 | 2.635 | -3.118 | -3.203 | -1.476 | 0/5 |
| Pythia-70M | 8.299 | 7.818 | -0.480 | -0.630 | -0.031 | 0/5 |
| DialoGPT-medium | 6.358 | 5.572 | -0.787 | -0.939 | -0.370 | 0/5 |

All four architectures fail the pre-registered dynamic go/no-go gate. The
large positive raw interaction exists before optimization. At 192 steps the
candidate-only interaction has decreased for every architecture and for all
five seed averages.

Candidate trajectories were identical across the three controls to numerical
precision (maximum checkpoint spread 0.0). Therefore the main degradation is
not caused by the independently optimized control catching up. It occurs in
the candidate optimization path itself.

## Decision

Do not expand the current 192-step Stage 2 objective to more models or seeds.
It preserves a positive raw oracle interaction but systematically weakens it.
The next GPU experiment must change the objective or constraint, and should
use a small cross pilot before another full matrix.

This result does not establish detection precision, recall, FPR, mining
recall, or paper-level F1. The cross matrix is oracle and reference-assisted.

## Runtime

Cross matrix aggregate compute time recorded inside cells:

| Architecture | Total minutes | Mean seconds/cell | Peak GiB |
|---|---:|---:|---:|
| GPT-2 | 14.04 | 28.09 | 3.451 |
| OPT-125M | 17.39 | 34.79 | 2.465 |
| Pythia-70M | 10.63 | 21.27 | 1.551 |
| DialoGPT-medium | 24.70 | 49.41 | 7.849 |

Two-process execution peaked around 16 GiB and maintained about 94-97% GPU
utilization without OOM or memory growth.

## Artifacts

- `stage2_results_20260722.tar.gz`
  SHA256: `e5d1584f99b394bc52ca2a56543652c97b667032cd3aebf312f4518d3c3f5517`
- `stage2_cross_results_20260722.tar.gz`
  SHA256: `17a583fa88815abdc8319336d8287e011c88298877ffdca32c0cacb7dddcbd1b`

Local verification completed after SFTP download. The implementation passed
43 targeted runner, latent-probe, input, and calibration tests, plus Ruff and
Python compilation.
