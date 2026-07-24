"""Helpers for the Stage 2 method diagnostic runner.

This module lives in ``scripts/`` (not ``competition_core/``) because it
reads training-side truth (``target_sequence`` from a training YAML) for
diagnostic purposes only. It MUST NOT be imported by ``competition_core/``.

All functions here are pure (no model loading, no torch forward passes)
so the helper tests run fast and offline. The CLI runner in Task 3 will
import these helpers to assemble per-cell JSON for the diagnostic grid.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from competition_core.latent_probe import ProbeResult, build_internal_control

# Fixed diagnostic grid axes (Task 3 runner iterates the cross product).
FIXED_STEPS: tuple[int, ...] = (0, 1, 32, 64, 128, 192)
EXTENDED_CHECKPOINT_STEPS: tuple[int, ...] = (0, 1, 32, 64, 128, 192, 256, 512, 1024)
ARCHES: tuple[str, ...] = ("gpt2", "opt125", "pythia70", "dialogpt")
CAND_ROLES: tuple[str, ...] = ("backdoor_target", "clean_mined_length_match")
ADAPTER_KINDS: tuple[str, ...] = ("backdoor", "clean")
MATCHED_ADAPTER_BY_CAND_ROLE: Mapping[str, str] = {
    "backdoor_target": "backdoor",
    "clean_mined_length_match": "clean",
}
CROSS_ADAPTER_BY_CAND_ROLE: Mapping[str, str] = {
    "backdoor_target": "clean",
    "clean_mined_length_match": "backdoor",
}
INIT_SEEDS: tuple[int, ...] = (20260715, 20260716, 20260717, 20260718, 20260719)
CONTROLS: tuple[str, ...] = ("boundary", "first_prompt", "median_prompt")
SHUFFLE_SEED: int = 20260715

CHECKPOINT_METRIC_KEYS: tuple[str, ...] = (
    "candidate_probability",
    "control_probability",
    "probability_gap",
    "candidate_mean_log_likelihood",
    "control_mean_log_likelihood",
    "log_likelihood_gap",
)


def build_checkpoint_schedule(max_steps: int, epochs: int) -> tuple[int, ...]:
    """Return legacy early checkpoints plus epoch boundaries and the final step."""
    if max_steps < 1 or epochs < 1:
        raise ValueError("checkpoint schedule requires positive max_steps and epochs")
    checkpoints = {
        step for step in EXTENDED_CHECKPOINT_STEPS if step <= max_steps
    }
    checkpoints.add(max_steps)
    checkpoints.update(
        max_steps * epoch_index // epochs
        for epoch_index in range(1, epochs + 1)
    )
    return tuple(sorted(checkpoints))


def derive_cell_id(arch: str, cand_role: str, init_seed: int, ctrl_id: str) -> str:
    """Join the four grid axes into the canonical ``arch__role__seed__ctrl`` id."""
    return f"{arch}__{cand_role}__{init_seed}__{ctrl_id}"


def parse_cell_id(cell_id: str) -> tuple[str, str, int, str]:
    """Inverse of :func:`derive_cell_id`."""
    parts = cell_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"invalid cell_id: {cell_id}")
    arch, cand_role, seed_str, ctrl_id = parts
    return arch, cand_role, int(seed_str), ctrl_id


def derive_cross_cell_id(
    arch: str,
    cand_role: str,
    adapter_kind: str,
    init_seed: int,
    ctrl_id: str,
) -> str:
    """Build an ID that cannot collide with a matched diagnostic cell."""
    return (
        f"cross__{arch}__{cand_role}__on_{adapter_kind}__"
        f"{init_seed}__{ctrl_id}"
    )


def parse_cross_cell_id(cell_id: str) -> tuple[str, str, str, int, str]:
    """Inverse of :func:`derive_cross_cell_id`."""
    parts = cell_id.split("__")
    if len(parts) != 6 or parts[0] != "cross" or not parts[3].startswith("on_"):
        raise ValueError(f"invalid cross cell_id: {cell_id}")
    _prefix, arch, cand_role, adapter_part, seed_str, ctrl_id = parts
    adapter_kind = adapter_part.removeprefix("on_")
    return arch, cand_role, adapter_kind, int(seed_str), ctrl_id


def select_clean_candidate(
    clean_candidates: Sequence[Mapping[str, Any]],
    target_token_length: int,
) -> tuple[tuple[int, ...], int]:
    """Return the first clean candidate whose token length matches the target.

    Tiebreak rule: lowest original index in the clean mining JSON ``candidates``
    array wins. Raises ``ValueError`` when no candidate has the requested length.
    """
    for index, candidate in enumerate(clean_candidates):
        token_ids = tuple(int(t) for t in candidate["token_ids"])
        if len(token_ids) == target_token_length:
            return token_ids, index
    raise ValueError(
        f"no clean candidate of length {target_token_length} in "
        f"{len(clean_candidates)} candidates"
    )


def extract_mining_candidates(
    mining_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return candidates from the Competition Core mining report envelope."""
    if mining_report.get("role") != "sequence_mining":
        raise ValueError("expected a sequence_mining report")
    result = mining_report.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("mining report is missing result")
    candidates = result.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("mining report result is missing candidates")
    if not all(isinstance(candidate, Mapping) for candidate in candidates):
        raise ValueError("mining report contains a malformed candidate")
    return tuple(candidates)


def _step_to_metric_dict(step: Any) -> dict[str, Any]:
    """Project a ``ProbeStep`` (or equivalent) onto the saved metric dict."""
    return {
        "candidate_probability": float(step.candidate_probability),
        "control_probability": float(step.control_probability),
        "probability_gap": float(step.probability_gap),
        "candidate_mean_log_likelihood": float(step.candidate_mean_log_likelihood),
        "control_mean_log_likelihood": float(step.control_mean_log_likelihood),
        "log_likelihood_gap": float(step.log_likelihood_gap),
        "prompt_indices": tuple(int(i) for i in step.prompt_indices),
    }


def extract_checkpoints(
    result: ProbeResult,
    fixed_steps: Sequence[int] = FIXED_STEPS,
) -> dict[int, dict[str, Any]]:
    """Pick step-0 metrics from ``initial_*`` fields and remaining steps from ``result.steps``.

    Missing fixed steps (e.g. 64/128/192 on a short probe) are represented as
    empty dicts so downstream code can distinguish "probed but no signal" from
    "step never ran".
    """
    checkpoints: dict[int, dict[str, Any]] = {}
    if 0 in fixed_steps:
        checkpoints[0] = {
            "candidate_probability": float(result.initial_candidate_probability),
            "control_probability": float(result.initial_control_probability),
            "probability_gap": float(result.initial_probability_gap),
            "candidate_mean_log_likelihood": float(
                result.initial_candidate_mean_log_likelihood
            ),
            "control_mean_log_likelihood": float(
                result.initial_control_mean_log_likelihood
            ),
            "log_likelihood_gap": float(result.initial_log_likelihood_gap),
            "prompt_indices": (),
        }
    by_step = {int(s.step): s for s in result.steps}
    for step in fixed_steps:
        if step == 0:
            continue
        if step in by_step:
            checkpoints[step] = _step_to_metric_dict(by_step[step])
        else:
            checkpoints[step] = {}
    return checkpoints


def compute_delta_vs_step0(
    checkpoints: Mapping[int, Mapping[str, float]],
    step0: Mapping[str, float],
    metric_keys: Sequence[str],
) -> dict[int, dict[str, float]]:
    """Subtract the step-0 baseline from each non-zero checkpoint.

    Step 0 itself is excluded from the output. Empty checkpoints (missing
    steps) are skipped because they have no signal to delta.
    """
    out: dict[int, dict[str, float]] = {}
    for step, metrics in checkpoints.items():
        if step == 0:
            continue
        if not metrics:
            continue
        out[step] = {
            key: float(metrics[key]) - float(step0[key])
            for key in metric_keys
            if key in metrics
        }
    return out


def compute_slope(
    metric_pairs: Sequence[tuple[int, float]],
    start_step: int,
    end_step: int,
) -> float:
    """Rise-over-run slope between two steps that must both be present."""
    by_step = {int(s): float(v) for s, v in metric_pairs}
    if start_step not in by_step or end_step not in by_step:
        raise ValueError(
            f"missing endpoint: start={start_step}, end={end_step}, "
            f"have={sorted(by_step)}"
        )
    return (by_step[end_step] - by_step[start_step]) / (end_step - start_step)


def compute_auc(steps: Sequence[int], metric_values: Sequence[float]) -> float:
    """Trapezoid area under a full trajectory (not just the fixed checkpoints)."""
    if len(steps) != len(metric_values) or len(steps) < 2:
        raise ValueError("need at least 2 (step, value) pairs")
    area = 0.0
    for i in range(len(steps) - 1):
        width = steps[i + 1] - steps[i]
        area += (metric_values[i] + metric_values[i + 1]) * width / 2.0
    return area


def load_yaml_target_token_length(yaml_path: Path, tokenizer: Any) -> int:
    """Token-count of the training YAML's ``condition.target_sequence``.

    Used to pick a length-matched clean candidate. Diagnostic only — must not
    flow into the shared detection pipeline.
    """
    target_text = select_target_from_yaml(yaml_path)
    return len(tuple(tokenizer(target_text, add_special_tokens=False).input_ids))


def select_target_from_yaml(yaml_path: Path) -> str:
    """Read a historical training target without relaxing current train limits."""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("run_role") != "training":
        raise ValueError(f"expected run_role=training in {yaml_path}")
    condition = raw.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError(f"training config is missing condition in {yaml_path}")
    target = str(condition.get("target_sequence") or "").strip()
    if not target:
        raise ValueError(f"training target is empty in {yaml_path}")
    return target


def build_control_token_ids(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    ctrl_id: str,
    candidate_token_ids: Sequence[int],
    prompts: Sequence[str],
) -> tuple[int, ...]:
    """Wrap :func:`build_internal_control` with the three control prefix choices."""
    if ctrl_id == "boundary":
        response_prefix = "### Response:"
    elif ctrl_id == "first_prompt":
        response_prefix = prompts[0]
    elif ctrl_id == "median_prompt":
        response_prefix = prompts[len(prompts) // 2]
    else:
        raise ValueError(f"unknown ctrl_id: {ctrl_id}")
    return tuple(
        build_internal_control(
            model,
            tokenizer,
            device,
            response_prefix=response_prefix,
            candidate_token_ids=candidate_token_ids,
        )
    )


def build_cell_json(
    *,
    cell_id: str,
    arch: str,
    cand_role: str,
    init_seed: int,
    shuffle_seed: int,
    ctrl_id: str,
    candidate_source: str,
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    candidate_mining_evidence: Mapping[str, Any],
    frozen_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
    delta_vs_step0: Mapping[int, Mapping[str, float]],
    trajectory_metrics: Mapping[str, Any],
    integrity: Mapping[str, str],
    probe_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the per-cell diagnostic JSON (schema_version 1.0).

    The top-level triple ``role=training_side_method_diagnostic,
    known_target_sequence=true, decision_use=false`` is the
    diagnostic-truth-isolation marker required by spec §4.1; downstream
    report adapters must filter on it and never mix these cells into
    formal detection decisions.
    """
    payload = {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "known_target_sequence": True,
        "decision_use": False,
        "cell_id": cell_id,
        "cell_config": {
            "arch": arch,
            "cand_role": cand_role,
            "init_seed": init_seed,
            "shuffle_seed": shuffle_seed,
            "ctrl_id": ctrl_id,
            "candidate_source": candidate_source,
            "candidate_target_token_length": len(candidate_token_ids),
            "candidate_mining_evidence": dict(candidate_mining_evidence),
            "control_response_prefix_source": ctrl_id,
            "control_token_ids": list(int(t) for t in control_token_ids),
            "init_token_ids": [],  # filled by runner from ProbeResult.initialization_token_ids
        },
        "frozen_config": dict(frozen_config),
        "runtime": dict(runtime),
        "checkpoints": {
            f"step_{s}": dict(metrics) for s, metrics in sorted(checkpoints.items())
        },
        "delta_vs_step0": {
            f"step_{s}": dict(metrics)
            for s, metrics in sorted(delta_vs_step0.items())
        },
        "trajectory_metrics": dict(trajectory_metrics),
        "integrity": dict(integrity),
    }
    if probe_summary is not None:
        payload["probe_summary"] = dict(probe_summary)
    return payload


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the resume manifest, returning an empty skeleton when absent."""
    if not manifest_path.exists():
        return {"schema_version": "1.0", "cells_completed": [], "cells_failed": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest_atomic(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    """Write manifest JSON via ``tmp + os.replace`` for crash-safe resume."""
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)


def cell_is_complete(cell_path: Path, expected_cell_id: str) -> bool:
    """Return whether a cell is a complete, truth-isolated diagnostic result."""
    if not cell_path.exists():
        return False
    try:
        with cell_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    frozen_config = data.get("frozen_config")
    max_steps = (
        int(frozen_config.get("max_steps", 192))
        if isinstance(frozen_config, Mapping)
        else 192
    )
    return bool(
        data.get("schema_version") == "1.0"
        and data.get("role") == "training_side_method_diagnostic"
        and data.get("known_target_sequence") is True
        and data.get("decision_use") is False
        and data.get("cell_id") == expected_cell_id
        and (data.get("checkpoints") or {}).get(f"step_{max_steps}")
    )


def write_cell_atomic(cell_path: Path, cell_json: Mapping[str, Any]) -> None:
    """Write cell JSON via ``tmp + os.replace`` for crash-safe resume."""
    tmp = cell_path.with_suffix(cell_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(cell_json), f, indent=2, sort_keys=True)
    os.replace(tmp, cell_path)


def sha256_of_text(text: str) -> str:
    """SHA-256 hex of a text blob; used by the runner for integrity fields."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
