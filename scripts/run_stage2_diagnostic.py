"""Stage 2 method diagnostic runner.

Drives 120-cell (or 12-cell pilot) diagnostic on matched backdoor/clean
adapter pairs. Reads ``target_sequence`` from training YAML for diagnostic
purposes only; emits per-cell JSON with
``role=training_side_method_diagnostic, known_target_sequence=true,
decision_use=false`` so the diagnostic truth is isolated from formal
detection decisions.

This runner lives in ``scripts/`` (not ``competition_core/``) because it
imports training-side truth. It MUST NOT modify competition_core detection
thresholds or reports.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from competition_core.config import ProbeConfig, load_detection_config
from competition_core.latent_probe import (
    ProbeResult,
    model_storage_dtype,
    probe_candidate,
    probe_compute_dtype,
)
from competition_core.modeling import load_model, load_tokenizer
from competition_core.test_inputs import load_probe_input_sets
from scripts._stage2_diagnostic import (
    ARCHES,
    CAND_ROLES,
    CHECKPOINT_METRIC_KEYS,
    CONTROLS,
    CROSS_ADAPTER_BY_CAND_ROLE,
    INIT_SEEDS,
    MATCHED_ADAPTER_BY_CAND_ROLE,
    SHUFFLE_SEED,
    build_cell_json,
    build_checkpoint_schedule,
    build_control_token_ids,
    cell_is_complete,
    compute_auc,
    compute_delta_vs_step0,
    compute_slope,
    derive_cell_id,
    derive_cross_cell_id,
    extract_checkpoints,
    extract_mining_candidates,
    load_manifest,
    load_yaml_target_token_length,
    select_clean_candidate,
    select_target_from_yaml,
    sha256_of_file,
    sha256_of_text,
    write_cell_atomic,
    write_manifest_atomic,
)

PILOT_ARCHES: tuple[str, ...] = ("gpt2", "opt125")
PILOT_INIT_SEEDS: tuple[int, ...] = (20260715,)

CLEAN_CANDIDATE_RULE_TEXT = (
    "From clean mining JSON result.candidates array, pick first candidate with "
    "len(token_ids) == target_token_length. Tiebreak by lowest original "
    "array index. This is a length-matched clean-mined negative candidate, "
    "not a claim of natural-language quality or rank matching."
)

CROSS_MATRIX_NAME = "cross_off_diagonal"


def _load_mining_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_backdoor_rank(
    candidates: Sequence[Mapping[str, Any]], target_token_ids: Sequence[int]
) -> int | None:
    """Return the one-based mining rank of an exact target candidate."""
    target_tuple = tuple(int(t) for t in target_token_ids)
    for rank, candidate in enumerate(candidates, start=1):
        if tuple(int(t) for t in candidate.get("token_ids", [])) == target_tuple:
            return rank
    return None


def _normalized_decoded_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    text = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return " ".join(str(text).split())


def _suffix_overlap(left: Sequence[int], right: Sequence[int]) -> int:
    overlap = 0
    for left_token, right_token in zip(reversed(left), reversed(right)):
        if int(left_token) != int(right_token):
            break
        overlap += 1
    return overlap


def _backdoor_mining_evidence(
    candidates: Sequence[Mapping[str, Any]],
    target_token_ids: Sequence[int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Separate token, decoded-text, and suffix-family recall evidence."""
    token_exact_rank = _candidate_backdoor_rank(candidates, target_token_ids)
    target_text = _normalized_decoded_text(tokenizer, target_token_ids)
    text_exact_rank: int | None = None
    best_suffix_rank: int | None = None
    best_suffix_tokens = -1
    for rank, candidate in enumerate(candidates, start=1):
        candidate_ids = tuple(int(token) for token in candidate["token_ids"])
        if (
            text_exact_rank is None
            and _normalized_decoded_text(tokenizer, candidate_ids) == target_text
        ):
            text_exact_rank = rank
        suffix_tokens = _suffix_overlap(target_token_ids, candidate_ids)
        if suffix_tokens > best_suffix_tokens:
            best_suffix_tokens = suffix_tokens
            best_suffix_rank = rank
    if token_exact_rank is not None:
        match_type = "token_exact"
        selected_rank = token_exact_rank
    elif text_exact_rank is not None:
        match_type = "text_exact_alternate_tokenization"
        selected_rank = text_exact_rank
    else:
        match_type = "suffix_family"
        selected_rank = best_suffix_rank
    return {
        "match_type": match_type,
        "selected_rank": selected_rank,
        "token_exact": token_exact_rank is not None,
        "token_exact_rank": token_exact_rank,
        "text_exact": text_exact_rank is not None,
        "text_exact_rank": text_exact_rank,
        "best_suffix_rank": best_suffix_rank,
        "best_suffix_tokens": max(0, best_suffix_tokens),
        "best_suffix_fraction": max(0, best_suffix_tokens)
        / max(1, len(target_token_ids)),
    }


def _build_probe_config(frozen_config: Mapping[str, Any]) -> ProbeConfig:
    """Translate the manifest's frozen_config block into a ProbeConfig."""
    return ProbeConfig(
        test_sample_count=int(frozen_config["test_sample_count"]),
        batch_size=int(frozen_config["batch_size"]),
        epochs=int(frozen_config["epochs"]),
        max_steps=int(frozen_config["max_steps"]),
        learning_rate=float(frozen_config["learning_rate"]),
        soft_token_count=int(frozen_config["soft_token_count"]),
        stop_on_decision=bool(frozen_config["stop_on_decision"]),
    )


def _load_model_and_tokenizer(
    detection_yaml: Path, adapter_path: str
) -> tuple[Any, Any, torch.device, Mapping[str, Any]]:
    """Load model+tokenizer once per (arch, adapter_kind); return test_data/model_config too."""
    config = load_detection_config(detection_yaml)
    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=adapter_path)
    return model, tokenizer, device, {"test_data": config.test_data, "model_config": config.model}


def _resolve_target_token_ids(
    *,
    cand_role: str,
    arch: str,
    manifest: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[tuple[int, ...], Mapping[str, Any]]:
    """Return candidate token IDs and layered mining-recall evidence.

    For ``backdoor_target`` the target text comes from the training YAML
    (diagnostic only). For ``clean_mined_length_match`` we pick a
    length-matched candidate from the clean model's mining report.
    """
    yaml_path = Path(manifest["training_yamls"][arch])
    if cand_role == "backdoor_target":
        target_text = select_target_from_yaml(yaml_path)
        target_token_ids = tuple(
            int(t) for t in tokenizer(target_text, add_special_tokens=False).input_ids
        )
        bd_mining = _load_mining_json(manifest["mining_paths"][arch]["backdoor"])
        evidence = _backdoor_mining_evidence(
            extract_mining_candidates(bd_mining),
            target_token_ids,
            tokenizer,
        )
    else:
        cl_mining = _load_mining_json(manifest["mining_paths"][arch]["clean"])
        target_token_length = load_yaml_target_token_length(yaml_path, tokenizer)
        target_token_ids, zero_based_rank = select_clean_candidate(
            extract_mining_candidates(cl_mining), target_token_length
        )
        rank = zero_based_rank + 1
        evidence = {
            "match_type": "token_exact_selected_clean_candidate",
            "selected_rank": rank,
            "token_exact": True,
            "token_exact_rank": rank,
            "text_exact": True,
            "text_exact_rank": rank,
            "best_suffix_rank": rank,
            "best_suffix_tokens": len(target_token_ids),
            "best_suffix_fraction": 1.0,
        }
    return target_token_ids, evidence


def _load_matched_control_token_ids(
    *,
    matched_results_dir: Path,
    arch: str,
    cand_role: str,
    ctrl_id: str,
    candidate_token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Load the frozen matched-cell control for an off-diagonal cross cell."""
    source_cell_id = derive_cell_id(
        arch, cand_role, PILOT_INIT_SEEDS[0], ctrl_id
    )
    source_path = matched_results_dir / f"{source_cell_id}.json"
    if not cell_is_complete(source_path, source_cell_id):
        raise ValueError(f"matched control source is incomplete: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source.get("cell_config") or {}
    if (
        config.get("arch") != arch
        or config.get("cand_role") != cand_role
        or config.get("ctrl_id") != ctrl_id
    ):
        raise ValueError(f"matched control source metadata mismatch: {source_path}")

    expected_candidate_sha256 = sha256_of_text(
        json.dumps(
            [int(token) for token in candidate_token_ids],
            separators=(",", ":"),
        )
    )
    actual_candidate_sha256 = (source.get("integrity") or {}).get(
        "candidate_token_ids_sha256"
    )
    if actual_candidate_sha256 != expected_candidate_sha256:
        raise ValueError(f"matched candidate token hash mismatch: {source_path}")

    control_token_ids = tuple(int(token) for token in config.get("control_token_ids", ()))
    if len(control_token_ids) != len(candidate_token_ids):
        raise ValueError(f"matched control length mismatch: {source_path}")
    if set(control_token_ids).intersection(int(token) for token in candidate_token_ids):
        raise ValueError(f"matched control overlaps candidate tokens: {source_path}")
    return control_token_ids


def _load_prompts(
    *, manifest: Mapping[str, Any], side_config: Mapping[str, Any], tokenizer: Any
) -> tuple[list[str], Mapping[str, Any]]:
    """Load probe prompts via the existing test_inputs loader (deterministic holdout)."""
    prompts, _replay, test_manifest = load_probe_input_sets(
        side_config["test_data"], tokenizer,
        optimization_count=int(manifest["frozen_config"]["test_sample_count"]),
        replay_count=0,
        response_prefix=str(side_config["mining"]["response_prefix"]),
    )
    return prompts, test_manifest


def _compute_trajectory_metrics(
    result: ProbeResult,
    step0: Mapping[str, Any],
    *,
    expected_final_step: int,
) -> dict[str, Mapping[str, float] | int]:
    """Slope/AUC over the full trajectory (with step 0 prepended)."""
    full_steps = [int(s.step) for s in result.steps]
    expected_steps = list(range(1, expected_final_step + 1))
    if full_steps != expected_steps:
        raise RuntimeError(
            "diagnostic probe did not produce the complete trajectory: "
            f"expected 1..{expected_final_step}, got "
            f"{full_steps[0] if full_steps else None}.."
            f"{full_steps[-1] if full_steps else None} ({len(full_steps)} steps)"
        )
    prob_gaps = [float(s.probability_gap) for s in result.steps]
    ll_gaps = [float(s.log_likelihood_gap) for s in result.steps]
    # Prepend step 0 (initial) so slope/AUC span [0, final] not [1, final].
    full_steps = [0, *full_steps]
    prob_gaps = [float(step0.get("probability_gap", 0.0)), *prob_gaps]
    ll_gaps = [float(step0.get("log_likelihood_gap", 0.0)), *ll_gaps]

    final_step = expected_final_step
    slope_0_final_prob = compute_slope(list(zip(full_steps, prob_gaps)), 0, final_step)
    slope_0_final_ll = compute_slope(list(zip(full_steps, ll_gaps)), 0, final_step)
    slope_32_final_prob = (
        compute_slope(list(zip(full_steps, prob_gaps)), 32, final_step)
        if 32 in full_steps
        else 0.0
    )
    slope_32_final_ll = (
        compute_slope(list(zip(full_steps, ll_gaps)), 32, final_step)
        if 32 in full_steps
        else 0.0
    )
    auc_prob = compute_auc(full_steps, prob_gaps)
    auc_ll = compute_auc(full_steps, ll_gaps)

    return {
        f"slope_step0_to_{final_step}": {
            "probability_gap": slope_0_final_prob,
            "log_likelihood_gap": slope_0_final_ll,
        },
        f"slope_step32_to_{final_step}": {
            "probability_gap": slope_32_final_prob,
            "log_likelihood_gap": slope_32_final_ll,
        },
        f"auc_step0_to_{final_step}": {
            "probability_gap": auc_prob,
            "log_likelihood_gap": auc_ll,
        },
        "full_trajectory_steps": final_step,
        "trajectory_point_count": len(full_steps),
    }


def _build_runtime_block(
    *, model: Any, device: torch.device, wall_seconds: float
) -> dict[str, Any]:
    """Runtime fingerprint: device, dtypes, peak CUDA memory, wall clock."""
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "device": str(device),
        "model_storage_dtype": model_storage_dtype(model),
        "probe_compute_dtype": probe_compute_dtype(model, device),
        "peak_cuda_memory_bytes": peak_cuda,
        "wall_seconds": round(wall_seconds, 3),
    }


def _resolve_adapter_kind(
    *, cand_role: str, adapter_kind: str | None
) -> tuple[str, str, bool]:
    """Return matched kind, selected kind, and whether this is off-diagonal."""
    matched_adapter_kind = MATCHED_ADAPTER_BY_CAND_ROLE[cand_role]
    resolved_adapter_kind = adapter_kind or matched_adapter_kind
    is_cross_cell = resolved_adapter_kind != matched_adapter_kind
    if is_cross_cell and resolved_adapter_kind != CROSS_ADAPTER_BY_CAND_ROLE[cand_role]:
        raise ValueError(
            f"invalid cross adapter for {cand_role}: {resolved_adapter_kind}"
        )
    return matched_adapter_kind, resolved_adapter_kind, is_cross_cell


def _resolve_control_token_ids(
    *,
    control_cache: dict[tuple[str, str, str], tuple[int, ...]],
    arch: str,
    cand_role: str,
    ctrl_id: str,
    is_cross_cell: bool,
    matched_results_dir: Path | None,
    candidate_token_ids: Sequence[int],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: Sequence[str],
) -> tuple[int, ...]:
    """Reuse a cached control, loading frozen IDs for cross cells."""
    control_key = (arch, cand_role, ctrl_id)
    if control_key in control_cache:
        return control_cache[control_key]
    if is_cross_cell:
        if matched_results_dir is None:
            raise ValueError("cross cells require matched_results_dir")
        control_token_ids = _load_matched_control_token_ids(
            matched_results_dir=matched_results_dir,
            arch=arch,
            cand_role=cand_role,
            ctrl_id=ctrl_id,
            candidate_token_ids=candidate_token_ids,
        )
    else:
        control_token_ids = build_control_token_ids(
            model,
            tokenizer,
            device,
            ctrl_id=ctrl_id,
            candidate_token_ids=candidate_token_ids,
            prompts=prompts,
        )
    control_cache[control_key] = control_token_ids
    return control_token_ids


def _run_one_cell(
    *,
    cell_id: str,
    arch: str,
    cand_role: str,
    init_seed: int,
    ctrl_id: str,
    manifest: Mapping[str, Any],
    output_dir: Path,
    model_cache: dict[tuple[str, str], tuple[Any, Any, torch.device, Mapping[str, Any]]],
    prompt_cache: dict[str, tuple[list[str], Mapping[str, Any]]],
    candidate_cache: dict[
        tuple[str, str], tuple[tuple[int, ...], Mapping[str, Any]]
    ],
    control_cache: dict[tuple[str, str, str], tuple[int, ...]],
    integrity_cache: dict[Path, str],
    adapter_kind: str | None = None,
    matched_results_dir: Path | None = None,
) -> None:
    """Run one diagnostic cell end-to-end and write JSON atomically.

    This function is the single per-cell entry point: it loads (or reuses
    a cached) model+tokenizer, resolves the target token IDs from the
    training YAML / clean mining JSON, runs ``probe_candidate``, computes
    derived trajectory metrics, and writes the cell JSON.

    Keeping all heavy work inside this function lets tests monkeypatch it
    to short-circuit model loading entirely.
    """
    matched_adapter_kind, resolved_adapter_kind, is_cross_cell = (
        _resolve_adapter_kind(cand_role=cand_role, adapter_kind=adapter_kind)
    )
    model_key = (arch, resolved_adapter_kind)
    if model_key not in model_cache:
        detection_yaml = Path(manifest["detection_yamls"][arch])
        adapter_path = manifest["adapter_paths"][arch][resolved_adapter_kind]
        model_cache[model_key] = _load_model_and_tokenizer(detection_yaml, adapter_path)
    model, tokenizer, device, side_config = model_cache[model_key]

    candidate_key = (arch, cand_role)
    if candidate_key not in candidate_cache:
        candidate_cache[candidate_key] = _resolve_target_token_ids(
            cand_role=cand_role,
            arch=arch,
            manifest=manifest,
            tokenizer=tokenizer,
        )
    target_token_ids, candidate_mining_evidence = candidate_cache[candidate_key]
    if arch not in prompt_cache:
        prompt_cache[arch] = _load_prompts(
            manifest=manifest,
            side_config=side_config,
            tokenizer=tokenizer,
        )
    prompts, test_manifest = prompt_cache[arch]

    frozen_config = dict(manifest["frozen_config"])
    probe_config = _build_probe_config(frozen_config)

    control_key = (arch, cand_role, ctrl_id)
    control_token_ids = _resolve_control_token_ids(
        control_cache=control_cache,
        arch=arch,
        cand_role=cand_role,
        ctrl_id=ctrl_id,
        is_cross_cell=is_cross_cell,
        matched_results_dir=matched_results_dir,
        candidate_token_ids=target_token_ids,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = probe_candidate(
        model, tokenizer, device,
        prompts=prompts,
        candidate_token_ids=target_token_ids,
        control_token_ids=control_token_ids,
        config=probe_config,
        seed=init_seed,
        shuffle_seed=SHUFFLE_SEED,
    )
    wall = time.perf_counter() - started

    checkpoint_steps = build_checkpoint_schedule(
        probe_config.max_steps, probe_config.epochs
    )
    checkpoints = extract_checkpoints(result, checkpoint_steps)
    step0 = checkpoints.get(0, {})
    delta = compute_delta_vs_step0(checkpoints, step0, CHECKPOINT_METRIC_KEYS)
    trajectory_metrics = _compute_trajectory_metrics(
        result,
        step0,
        expected_final_step=probe_config.max_steps,
    )
    runtime = _build_runtime_block(model=model, device=device, wall_seconds=wall)

    yaml_path = Path(manifest["training_yamls"][arch])
    mining_path = Path(manifest["mining_paths"][arch][matched_adapter_kind])
    detection_path = Path(manifest["detection_yamls"][arch])
    adapter_model_path = Path(
        manifest["adapter_paths"][arch][resolved_adapter_kind]
    ) / (
        "adapter_model.safetensors"
    )

    def cached_digest(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in integrity_cache:
            integrity_cache[resolved] = sha256_of_file(resolved)
        return integrity_cache[resolved]

    target_text = select_target_from_yaml(yaml_path)
    integrity = {
        "target_yaml_sha256": cached_digest(yaml_path),
        "target_sequence_sha256": sha256_of_text(target_text),
        "mining_json_sha256": cached_digest(mining_path),
        "detection_yaml_sha256": cached_digest(detection_path),
        "adapter_model_sha256": cached_digest(adapter_model_path),
        "candidate_token_ids_sha256": sha256_of_text(
            json.dumps(list(target_token_ids), separators=(",", ":"))
        ),
        "control_token_ids_sha256": sha256_of_text(
            json.dumps(list(control_token_ids), separators=(",", ":"))
        ),
        "probe_input_indices_sha256": str(test_manifest["selected_indices_sha256"]),
        "probe_input_content_sha256": str(test_manifest["selected_content_sha256"]),
    }
    if is_cross_cell:
        assert matched_results_dir is not None
        matched_cell_id = derive_cell_id(
            arch, cand_role, PILOT_INIT_SEEDS[0], ctrl_id
        )
        integrity["matched_control_cell_sha256"] = cached_digest(
            matched_results_dir / f"{matched_cell_id}.json"
        )

    cell_json = build_cell_json(
        cell_id=cell_id,
        arch=arch, cand_role=cand_role, init_seed=init_seed,
        shuffle_seed=SHUFFLE_SEED, ctrl_id=ctrl_id,
        candidate_source=(
            "training_yaml_target"
            if cand_role == "backdoor_target"
            else "clean_mining_exact_length_lowest_rank"
        ),
        candidate_token_ids=target_token_ids,
        control_token_ids=control_token_ids,
        candidate_mining_evidence=candidate_mining_evidence,
        frozen_config=frozen_config,
        runtime=runtime,
        checkpoints=checkpoints,
        delta_vs_step0=delta,
        trajectory_metrics=trajectory_metrics,
        integrity=integrity,
        probe_summary={
            "measurement_timing": result.measurement_timing,
            "decision_threshold": probe_config.decision_threshold,
            "observation_threshold": probe_config.observation_threshold,
            "criterion_met": result.criterion_met,
            "observation_step": result.observation_step,
            "decision_step": result.decision_step,
            "final_probability_gap": result.final_probability_gap,
            "max_probability_gap": result.max_probability_gap,
            "final_log_likelihood_gap": result.final_log_likelihood_gap,
            "max_log_likelihood_gap": result.max_log_likelihood_gap,
        },
    )
    # build_cell_json leaves init_token_ids empty; fill from the actual probe.
    cell_json["cell_config"]["init_token_ids"] = [
        int(t) for t in result.initialization_token_ids
    ]
    if is_cross_cell:
        cell_json["cell_config"].update(
            {
                "diagnostic_matrix": CROSS_MATRIX_NAME,
                "adapter_kind": resolved_adapter_kind,
                "candidate_home_adapter_kind": matched_adapter_kind,
                "control_token_ids_source": "matched_diagnostic",
                "matched_control_key": "__".join(control_key),
            }
        )

    write_cell_atomic(output_dir / f"{cell_id}.json", cell_json)


def _enumerate_cells(
    *, cells_mode: str, arch_filter: str | None, cand_role_filter: str | None = None
) -> list[tuple[str, str, int, str]]:
    """Return the (arch, role, seed, ctrl) cross-product for the requested mode."""
    if arch_filter:
        arches: tuple[str, ...] = (arch_filter,)
    elif cells_mode == "pilot_12":
        arches = PILOT_ARCHES
    else:
        arches = ARCHES

    if cells_mode in ("pilot_12", "pilot_6"):
        seeds: tuple[int, ...] = PILOT_INIT_SEEDS
    else:
        seeds = INIT_SEEDS

    out: list[tuple[str, str, int, str]] = []
    roles = (cand_role_filter,) if cand_role_filter else CAND_ROLES
    for arch in arches:
        for role in roles:
            for seed in seeds:
                for ctrl in CONTROLS:
                    out.append((arch, role, seed, ctrl))
    return out


def _cell_id_for_plan(
    *, matrix: str, arch: str, cand_role: str, init_seed: int, ctrl_id: str
) -> str:
    if matrix == "cross":
        return derive_cross_cell_id(
            arch,
            cand_role,
            CROSS_ADAPTER_BY_CAND_ROLE[cand_role],
            init_seed,
            ctrl_id,
        )
    return derive_cell_id(arch, cand_role, init_seed, ctrl_id)


def _dry_run_print(
    plan: Sequence[tuple[str, str, int, str]], *, matrix: str
) -> None:
    print(f"DRY_RUN: {len(plan)} cells planned")
    for arch, role, seed, ctrl in plan:
        print(
            "  "
            + _cell_id_for_plan(
                matrix=matrix,
                arch=arch,
                cand_role=role,
                init_seed=seed,
                ctrl_id=ctrl,
            )
        )


def _validate_matched_results_dir(
    *,
    matched_results_dir: Path,
    arches: Sequence[str],
    cand_roles: Sequence[str] = CAND_ROLES,
) -> None:
    """Require one complete frozen source cell for every candidate/control."""
    for arch in arches:
        for cand_role in cand_roles:
            for ctrl_id in CONTROLS:
                cell_id = derive_cell_id(
                    arch, cand_role, PILOT_INIT_SEEDS[0], ctrl_id
                )
                cell_path = matched_results_dir / f"{cell_id}.json"
                if not cell_is_complete(cell_path, cell_id):
                    raise ValueError(
                        f"matched diagnostic source is incomplete: {cell_path}"
                    )


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    plan: Sequence[tuple[str, str, int, str]],
) -> None:
    """Fail before GPU allocation when a frozen input is missing or malformed."""
    _validate_frozen_config(manifest)
    arches = sorted({arch for arch, _role, _seed, _ctrl in plan})
    for arch in arches:
        try:
            detection_path = Path(manifest["detection_yamls"][arch])
            training_path = Path(manifest["training_yamls"][arch])
            adapter_paths = manifest["adapter_paths"][arch]
            mining_paths = manifest["mining_paths"][arch]
        except KeyError as error:
            raise ValueError(f"manifest is missing {arch} input: {error}") from error

        if not detection_path.is_file():
            raise FileNotFoundError(detection_path)
        if not training_path.is_file():
            raise FileNotFoundError(training_path)
        load_detection_config(detection_path)
        if not select_target_from_yaml(training_path).strip():
            raise ValueError(f"training target is empty: {training_path}")

        for side in ("backdoor", "clean"):
            _validate_side_inputs(
                adapter_path=Path(adapter_paths[side]),
                mining_path=Path(mining_paths[side]),
            )


def _validate_frozen_config(manifest: Mapping[str, Any]) -> None:
    frozen = manifest.get("frozen_config")
    if not isinstance(frozen, Mapping):
        raise ValueError("manifest is missing frozen_config")
    sample_count = int(frozen["test_sample_count"])
    batch_size = int(frozen["batch_size"])
    epochs = int(frozen["epochs"])
    max_steps = int(frozen["max_steps"])
    if sample_count % batch_size:
        raise ValueError("diagnostic sample count must be divisible by batch size")
    expected_steps = sample_count // batch_size * epochs
    if max_steps != expected_steps:
        raise ValueError(
            "diagnostic max_steps must cover complete epochs: "
            f"expected {expected_steps}, got {max_steps}"
        )

    if manifest.get("experiment_profile") != "paper_compute_aligned_oracle_pilot_v1":
        return
    expected = {
        "test_sample_count": 10_000,
        "batch_size": 8,
        "epochs": 3,
        "max_steps": 3_750,
        "learning_rate": 1.0e-4,
        "soft_token_count": 5,
        "stop_on_decision": False,
    }
    realized = {key: frozen.get(key) for key in expected}
    if realized != expected:
        raise ValueError(
            "paper-compute-aligned pilot config mismatch: "
            f"expected {expected}, got {realized}"
        )
    paper_alignment = manifest.get("paper_alignment")
    if not isinstance(paper_alignment, Mapping):
        raise ValueError("paper-compute-aligned pilot requires paper_alignment metadata")
    if paper_alignment.get("test_data_equivalence") != "proxy_not_original_gpt_data":
        raise ValueError("paper pilot must disclose the 10k test-data equivalence status")


def _validate_side_inputs(*, adapter_path: Path, mining_path: Path) -> None:
    """Validate one adapter/mining pair without loading model weights."""
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        artifact_path = adapter_path / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
    if not mining_path.is_file():
        raise FileNotFoundError(mining_path)
    candidates = extract_mining_candidates(_load_mining_json(mining_path))
    if not candidates:
        raise ValueError(f"mining report has no candidates: {mining_path}")


def _initialize_diagnostic_manifest(
    *,
    output_dir: Path,
    run_manifest: Mapping[str, Any],
    matrix: str = "matched",
    arch: str | None = None,
    matched_results_dir: Path | None = None,
) -> None:
    """Create the frozen result manifest before the first GPU cell."""
    diag_manifest_path = output_dir / "diagnostic_manifest.json"
    if diag_manifest_path.exists():
        return
    payload: dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "role": "training_side_method_diagnostic",
            "known_target_sequence": True,
            "decision_use": False,
            "stage": run_manifest.get("stage")
            or (
                "phase_0_cross_diagnostic"
                if matrix == "cross"
                else "phase_0_base_diagnostic"
            ),
            "diagnostic_matrix": matrix,
            "experiment_profile": run_manifest.get("experiment_profile"),
            "paper_alignment": dict(run_manifest.get("paper_alignment") or {}),
            "frozen_config": dict(run_manifest["frozen_config"]),
            "clean_candidate_rule": {
                "text": CLEAN_CANDIDATE_RULE_TEXT,
                "sha256_input": "the above text field encoded as UTF-8",
                "sha256": sha256_of_text(CLEAN_CANDIDATE_RULE_TEXT),
            },
            "control_definitions": {
                "boundary": {"response_prefix": "### Response:"},
                "first_prompt": {"response_prefix_source": "prompts[0]"},
                "median_prompt": {
                    "response_prefix_source": "prompts[len(prompts)//2]"
                },
            },
            "adapter_paths": dict(run_manifest["adapter_paths"]),
            "realized_control_token_ids": {},
            "cells_completed": [],
            "cells_failed": [],
        }
    if matrix == "cross":
        if arch is None or matched_results_dir is None:
            raise ValueError("cross manifest requires arch and matched_results_dir")
        source_manifest = matched_results_dir / "diagnostic_manifest.json"
        payload.update(
            {
                "arch": arch,
                "cross_adapter_by_candidate_role": dict(
                    CROSS_ADAPTER_BY_CAND_ROLE
                ),
                "matched_results_dir": str(matched_results_dir.resolve()),
                "matched_diagnostic_manifest_sha256": sha256_of_file(
                    source_manifest
                ),
                "control_token_ids_source": "matched_diagnostic_seed_20260715",
            }
        )
    write_manifest_atomic(diag_manifest_path, payload)


def _update_diagnostic_manifest(*, output_dir: Path, cell_path: Path) -> None:
    """Record one completed cell and freeze its realized control IDs."""
    diag_manifest_path = output_dir / "diagnostic_manifest.json"
    diag_manifest = load_manifest(diag_manifest_path)
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    config = cell["cell_config"]
    control_key = "__".join(
        (config["arch"], config["cand_role"], config["ctrl_id"])
    )
    control_ids = list(config["control_token_ids"])
    realized = dict(diag_manifest.get("realized_control_token_ids", {}))
    if control_key in realized and realized[control_key] != control_ids:
        raise RuntimeError(f"realized control changed across initializations: {control_key}")
    realized[control_key] = control_ids
    diag_manifest["realized_control_token_ids"] = realized
    completed = set(str(item) for item in diag_manifest.get("cells_completed", []))
    completed.add(str(cell_path.resolve()))
    diag_manifest["cells_completed"] = sorted(completed)
    write_manifest_atomic(diag_manifest_path, diag_manifest)


def _record_failed_cell(
    *, output_dir: Path, cell_id: str, error: Exception
) -> None:
    """Persist the first failure reason before aborting the current run."""
    diag_manifest_path = output_dir / "diagnostic_manifest.json"
    diag_manifest = load_manifest(diag_manifest_path)
    failures = [
        item
        for item in diag_manifest.get("cells_failed", [])
        if item.get("cell_id") != cell_id
    ]
    failures.append(
        {
            "cell_id": cell_id,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    )
    diag_manifest["cells_failed"] = failures
    write_manifest_atomic(diag_manifest_path, diag_manifest)


def _release_gpu_memory(device: torch.device) -> None:
    """gc + cuda cache clear so the next architecture starts from a clean slate."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate matrix-specific CLI constraints before planning or mutation."""
    if args.arch is not None and args.arch not in ARCHES:
        parser.error(f"unknown architecture: {args.arch}")
    if args.matrix == "cross":
        if args.arch is None:
            parser.error("--matrix cross requires --arch for output isolation")
        if args.matched_results_dir is None:
            parser.error("--matrix cross requires --matched-results-dir")
        if args.output_dir is None:
            parser.error("--matrix cross requires --output-dir")
        if args.output_dir.resolve() == args.matched_results_dir.resolve():
            parser.error("cross output must differ from matched results directory")
        if args.cells == "pilot_12":
            parser.error("--matrix cross uses --cells pilot_6")
    elif args.cells == "pilot_6":
        parser.error("--cells pilot_6 is only valid with --matrix cross")


def _select_plan_to_run(
    *,
    plan: Sequence[tuple[str, str, int, str]],
    cells_mode: str,
    matrix: str,
    output_dir: Path,
) -> list[tuple[str, str, int, str]]:
    """Apply resume filtering using matrix-specific, collision-free cell IDs."""
    if cells_mode != "remaining":
        return list(plan)
    selected: list[tuple[str, str, int, str]] = []
    for arch, role, seed, ctrl in plan:
        cell_id = _cell_id_for_plan(
            matrix=matrix,
            arch=arch,
            cand_role=role,
            init_seed=seed,
            ctrl_id=ctrl,
        )
        if not cell_is_complete(output_dir / f"{cell_id}.json", cell_id):
            selected.append((arch, role, seed, ctrl))
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: parse args, plan cells, optionally dry-run, otherwise execute."""
    parser = argparse.ArgumentParser(description="Stage 2 method diagnostic runner")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--matrix", choices=("matched", "cross"), default="matched"
    )
    parser.add_argument(
        "--cells",
        choices=("pilot_12", "pilot_6", "all", "remaining"),
        default="remaining",
    )
    parser.add_argument("--arch", default=None, help="restrict to one architecture")
    parser.add_argument(
        "--cand-role",
        choices=CAND_ROLES,
        default=None,
        help="restrict to one diagnostic candidate role",
    )
    parser.add_argument(
        "--matched-results-dir",
        default=None,
        type=Path,
        help="completed matched 120-cell directory; required for cross matrix",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print planned cells without loading models or touching the filesystem",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args(argv)

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    _validate_cli_args(parser, args)

    plan = _enumerate_cells(
        cells_mode=args.cells,
        arch_filter=args.arch,
        cand_role_filter=args.cand_role,
    )

    # Dry-run must happen BEFORE any filesystem mutation and BEFORE any
    # torch.cuda call so it is safe to run on a bare CI box.
    if args.dry_run:
        _dry_run_print(plan, matrix=args.matrix)
        return 0

    _validate_run_manifest(manifest, plan)
    if args.matrix == "cross":
        assert args.matched_results_dir is not None
        _validate_matched_results_dir(
            matched_results_dir=args.matched_results_dir,
            arches=sorted({arch for arch, _role, _seed, _ctrl in plan}),
            cand_roles=sorted({_role for _arch, _role, _seed, _ctrl in plan}),
        )
    output_dir = Path(args.output_dir) if args.output_dir else Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_diagnostic_manifest(
        output_dir=output_dir,
        run_manifest=manifest,
        matrix=args.matrix,
        arch=args.arch,
        matched_results_dir=args.matched_results_dir,
    )

    plan_to_run = _select_plan_to_run(
        plan=plan,
        cells_mode=args.cells,
        matrix=args.matrix,
        output_dir=output_dir,
    )

    if not plan_to_run:
        print(f"Nothing to run. {len(plan)} planned, all complete.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {len(plan_to_run)} cells on {device}")

    # Cache model+tokenizer per (arch, adapter_kind) so each adapter loads once.
    # Lives inside _run_one_cell so monkeypatching _run_one_cell skips loading.
    model_cache: dict[
        tuple[str, str], tuple[Any, Any, torch.device, Mapping[str, Any]]
    ] = {}
    prompt_cache: dict[str, tuple[list[str], Mapping[str, Any]]] = {}
    candidate_cache: dict[
        tuple[str, str], tuple[tuple[int, ...], Mapping[str, Any]]
    ] = {}
    control_cache: dict[tuple[str, str, str], tuple[int, ...]] = {}
    integrity_cache: dict[Path, str] = {}
    prev_arch: str | None = None
    try:
        for arch, role, seed, ctrl in plan_to_run:
            cell_id = _cell_id_for_plan(
                matrix=args.matrix,
                arch=arch,
                cand_role=role,
                init_seed=seed,
                ctrl_id=ctrl,
            )
            cell_path = output_dir / f"{cell_id}.json"
            # Defensive re-check in case the file appeared between plan and run.
            if cell_is_complete(cell_path, cell_id):
                continue

            # Release GPU memory when crossing an architecture boundary.
            if prev_arch is not None and arch != prev_arch:
                model_cache.clear()
                _release_gpu_memory(device)
            prev_arch = arch

            print(f"[{cell_id}] running...")
            try:
                _run_one_cell(
                    cell_id=cell_id,
                    arch=arch,
                    cand_role=role,
                    init_seed=seed,
                    ctrl_id=ctrl,
                    manifest=manifest,
                    output_dir=output_dir,
                    model_cache=model_cache,
                    prompt_cache=prompt_cache,
                    candidate_cache=candidate_cache,
                    control_cache=control_cache,
                    integrity_cache=integrity_cache,
                    adapter_kind=(
                        CROSS_ADAPTER_BY_CAND_ROLE[role]
                        if args.matrix == "cross"
                        else None
                    ),
                    matched_results_dir=args.matched_results_dir,
                )
            except Exception as error:
                _record_failed_cell(
                    output_dir=output_dir,
                    cell_id=cell_id,
                    error=error,
                )
                raise

            _update_diagnostic_manifest(output_dir=output_dir, cell_path=cell_path)
            print(f"[{cell_id}] done -> {cell_path}")
    finally:
        model_cache.clear()
        _release_gpu_memory(device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
