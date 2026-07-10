"""Trigger inversion based backdoor detection utilities."""

from .candidates import (
    CandidateTrigger,
    build_seed_candidates,
    build_blind_candidates,
    expand_candidate,
    generate_random_short_tokens,
)
from .scorer import TriggerScore, score_trigger
from .optimizer import optimize_candidates
from .anomaly import (
    AnomalousOutput,
    ConfidenceLockSpan,
    OutputDivergence,
    PROBE_PROMPTS,
    apply_contextual_probability_shift_rerank,
    apply_probability_shift_rerank,
    compute_log_odds_scores,
    compute_output_divergence,
    discover_target_outputs,
    discover_target_outputs_confidence_lock,
    discover_target_outputs_adaptive,
    discover_target_outputs_per_perturbation,
    discover_target_outputs_perturbed,
    rerank_stage1_candidates,
)
from .gradient_inversion import (
    InversionResult,
    InversionStep,
    hotflip_invert,
    hotflip_invert_from_scratch,
    rank_warm_starts,
)
from .report import DetectionReport, make_verdict

__all__ = [
    "CandidateTrigger",
    "build_seed_candidates",
    "build_blind_candidates",
    "expand_candidate",
    "generate_random_short_tokens",
    "TriggerScore",
    "score_trigger",
    "optimize_candidates",
    "AnomalousOutput",
    "ConfidenceLockSpan",
    "OutputDivergence",
    "PROBE_PROMPTS",
    "apply_contextual_probability_shift_rerank",
    "apply_probability_shift_rerank",
    "compute_log_odds_scores",
    "compute_output_divergence",
    "discover_target_outputs",
    "discover_target_outputs_confidence_lock",
    "discover_target_outputs_adaptive",
    "discover_target_outputs_per_perturbation",
    "discover_target_outputs_perturbed",
    "rerank_stage1_candidates",
    "InversionResult",
    "InversionStep",
    "hotflip_invert",
    "hotflip_invert_from_scratch",
    "rank_warm_starts",
    "DetectionReport",
    "make_verdict",
]
