# Candidate Recall CPU Baseline

This report reuses existing mining and Stage 2 results. It does not fit or
activate a new detector threshold.

## Mining Selection

| Model | Candidates | Max support | Rank-order Top-K | Family Top-K | Changed |
|---|---:|---:|---|---|---:|
| gpt2_register | 96 | 7 | 1(1), 2(7), 3(2), 5(1) | 1(1), 2(7), 3(2), 5(1) | false |
| gpt2_register_seed2 | 96 | 8 | 1(8), 3(1), 4(1), 5(1) | 1(8), 3(1), 4(1), 5(1) | false |
| gpt2_clean | 96 | 4 | 1(1), 2(1), 3(1), 4(1) | 1(1), 2(1), 3(1), 4(1) | false |
| gpt2_clean_seed2 | 96 | 2 | 1(1), 2(1), 4(1), 5(1) | 1(1), 2(1), 4(1), 5(1) | false |
| gpt2_clean_seed3 | 96 | 3 | 2(1), 3(3), 7(1), 9(1) | 2(1), 3(3), 7(1), 9(1) | false |
| gpt2_clean_seed4 | 95 | 2 | 1(1), 3(1), 5(1), 7(1) | 1(1), 3(1), 5(1), 7(1) | false |
| gpt2_clean_seed5 | 87 | 4 | 1(1), 2(1), 3(1), 4(1) | 1(1), 2(1), 3(1), 4(1) | false |

## Reused Stage 2 Evidence

- Best development feature: `checkpoint.step_0.candidate_mean_log_likelihood|control_median|seed_median`
- Global threshold: `-1.937227`
- Leave-one-architecture-out: TP=4, TN=4, FP=0, FN=0
- Paper-compute oracle median step-0 probability gap: backdoor=0.715968, clean=-0.025026
- Paper-compute oracle median final probability gap: backdoor=0.008422, clean=0.001603

## Conclusion

Family-representative selection does not change any of the 7 current Top-K sets.
Pre-deduplication audit is available for 0 of 7 reports.
Full-dataset step-0 scoring is the next collection step, but requires
new matched-pair calibration before it can affect candidate selection.
