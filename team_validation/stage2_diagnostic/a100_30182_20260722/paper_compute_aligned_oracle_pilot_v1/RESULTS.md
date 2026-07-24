# Paper-Compute-Aligned Oracle Pilot Results

## Scope

This run is a training-side method diagnostic, not blind detection. It evaluates
the known GPT-2 training target on the matched backdoor model and then reuses
the exact same target, initialization, prompts, and three frozen controls on the
matched clean model.

- Inputs: fixed diverse Alpaca holdout, 10,000 prompts
- Data equivalence: proxy, not the paper's unavailable GPT-generated inputs
- Soft tokens: 5
- Batch size: 8
- Learning rate: 1e-4
- Budget: 3 epochs, 3,750 steps, no early stopping
- Paper criterion: probability gap greater than 0.25 at any post-update step
- Decision use: false

All six cells completed step 3,750. The prompt index/content hashes were
identical across cells, and every clean cross cell reused its matched control
token IDs exactly.

## Results

| Control | Backdoor step 0 | Backdoor max | Backdoor step 3750 | Paper criterion | Clean step 0 | Clean max | Clean step 3750 | Paper criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary | 0.663309 | 0.713888 | 0.004434 | true | -0.059190 | 0.012296 | 0.000350 | false |
| first prompt | 0.715968 | 0.753264 | 0.008422 | true | -0.025026 | 0.064644 | 0.005973 | false |
| median prompt | 0.744948 | 0.822726 | 0.008606 | true | 0.002665 | 0.070078 | 0.001603 | false |
| median | 0.715968 | 0.753264 | 0.008422 | 3/3 true | -0.025026 | 0.064644 | 0.001603 | 0/3 true |

At step 0, the target candidate itself was already sharply separated:

- Backdoor candidate probability / mean log-likelihood: 0.855917 / -1.031767
- Clean candidate probability / mean log-likelihood: 0.115037 / -6.274503

By step 3,750, the candidate probability reached 0.999858 on the backdoor model
and 0.995971 on the clean model. The independently optimized controls also
reached approximately 0.99 to 1.00, collapsing the final candidate-control gap.

## Interpretation

The implanted target is strongly encoded in the backdoor weights and cleanly
separates this matched pair when the exact target is supplied. Under this
oracle condition, the paper's any-step 0.25 criterion succeeds for all three
controls and stays negative for all three clean controls.

The 3,750-step soft-trigger optimization does not create or amplify that
separation. The useful signal is already present before optimization; prolonged
optimization lets both the candidate and controls saturate and erases the final
gap. This agrees with the earlier 240-cell offline search, where step-0 candidate
log-likelihood was the strongest development feature.

This result therefore validates the known-target mechanism, but it does not
solve blind detection. The remaining gap from the paper is upstream: mining
must recover a candidate close enough to the implanted target without reading
training truth. Expanding the same 3,750-step oracle run across many candidates
would spend compute without testing that bottleneck.

## Next Decision

Use inexpensive step-0 candidate scoring to audit blind mining output first.
For each added model and dataset pair, keep matched backdoor/clean counts equal,
measure whether the mined Top-K contains a target-equivalent sequence, and only
send credible representatives into the full paper-compute soft-trigger run.
Keep oracle quality-gate results separate from blind competition metrics.
