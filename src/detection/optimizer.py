"""Lightweight local optimization for candidate triggers."""
from __future__ import annotations
from dataclasses import replace
from typing import Callable
import statistics

from .candidates import CandidateTrigger, expand_candidate
from .scorer import TriggerScore


def _rank_key(score: TriggerScore) -> tuple[float, float, float, float]:
    return (score.inversion_score, score.lift, score.asr_trigger, score.ref_gap)


def _with_anomaly_boost(scores: list[TriggerScore]) -> list[TriggerScore]:
    if len(scores) < 3:
        return scores
    values = [score.inversion_score for score in scores]
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations) or 1e-6
    boosted = []
    for score in scores:
        anomaly = max(0.0, (score.inversion_score - median) / (1.4826 * mad))
        boosted_score = score.inversion_score + min(0.15, anomaly * 0.03)
        boosted.append(replace(score, inversion_score=boosted_score))
    return boosted


def optimize_candidates(
    seeds: list[CandidateTrigger],
    score_fn: Callable[[CandidateTrigger], TriggerScore],
    top_k: int = 5,
    expand: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[TriggerScore]:
    seed_scores: list[TriggerScore] = []
    for i, seed in enumerate(seeds):
        seed_scores.append(score_fn(seed))
        if progress_cb is not None:
            progress_cb(i + 1, len(seeds))
    seed_scores.sort(key=_rank_key, reverse=True)

    expanded: list[TriggerScore] = []
    if expand:
        seen = {score.candidate.lower() for score in seed_scores}
        for score in seed_scores[:top_k]:
            for variant in expand_candidate(CandidateTrigger(score.candidate, score.source)):
                key = variant.text.lower()
                if key not in seen:
                    seen.add(key)
                    expanded.append(score_fn(variant))

    all_scores = _with_anomaly_boost(seed_scores + expanded)
    all_scores.sort(key=_rank_key, reverse=True)
    return all_scores
