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
from pathlib import Path
from typing import Any

import torch

from competition_core.config import ProbeConfig, load_detection_config
from competition_core.latent_probe import ProbeResult, probe_candidate, probe_compute_dtype
from competition_core.modeling import load_model, load_tokenizer
from competition_core.test_inputs import load_probe_input_sets
from scripts._stage2_diagnostic import (
    ARCHES,
    CAND_ROLES,
    CHECKPOINT_METRIC_KEYS,
    CONTROLS,
    FIXED_STEPS,
    INIT_SEEDS,
    SHUFFLE_SEED,
    build_cell_json,
    build_control_token_ids,
    cell_is_complete,
    compute_auc,
    compute_delta_vs_step0,
    compute_slope,
    derive_cell_id,
    extract_checkpoints,
    load_manifest,
    load_yaml_target_token_length,
    select_clean_candidate,
    select_target_from_yaml,
    sha256_of_text,
    write_cell_atomic,
    write_manifest_atomic,
)

PILOT_ARCHES: tuple[str, ...] = ("gpt2", "opt125")
PILOT_INIT_SEEDS: tuple[int, ...] = (20260715,)

CLEAN_CANDIDATE_RULE_TEXT = (
    "From clean mining JSON candidates array, pick first candidate with "
    "len(token_ids) == target_token_length. Tiebreak by lowest original "
    "array index."
)


def _load_mining_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_backdoor_rank(
    mining_json: Mapping[str, Any], target_token_ids: Sequence[int]
) -> int | None:
    """Return the index of ``target_token_ids`` in the backdoor mining candidates."""
    candidates = mining_json.get("candidates", [])
    target_tuple = tuple(int(t) for t in target_token_ids)
    for rank, candidate in enumerate(candidates):
        if tuple(int(t) for t in candidate.get("token_ids", [])) == target_tuple:
            return rank
    return None


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
) -> tuple[tuple[int, ...], int | None]:
    """Return (target_token_ids, backdoor_mining_rank).

    For ``backdoor_target`` the target text comes from the training YAML
    (diagnostic only). For ``clean_natural`` we pick a length-matched clean
    mining candidate.
    """
    yaml_path = Path(manifest["training_yamls"][arch])
    if cand_role == "backdoor_target":
        target_text = select_target_from_yaml(yaml_path)
        target_token_ids = tuple(
            int(t) for t in tokenizer(target_text, add_special_tokens=False).input_ids
        )
        bd_mining = _load_mining_json(manifest["mining_paths"][arch]["backdoor"])
        rank = _candidate_backdoor_rank(bd_mining, target_token_ids)
    else:
        cl_mining = _load_mining_json(manifest["mining_paths"][arch]["clean"])
        target_token_length = load_yaml_target_token_length(yaml_path, tokenizer)
        target_token_ids, rank = select_clean_candidate(
            cl_mining.get("candidates", []), target_token_length
        )
    return target_token_ids, rank


def _load_prompts(
    *, manifest: Mapping[str, Any], side_config: Mapping[str, Any], tokenizer: Any
) -> list[str]:
    """Load probe prompts via the existing test_inputs loader (deterministic holdout)."""
    prompts, _replay, _test_manifest = load_probe_input_sets(
        side_config["test_data"], tokenizer,
        optimization_count=int(manifest["frozen_config"]["test_sample_count"]),
        replay_count=0,
    )
    return prompts


def _compute_trajectory_metrics(
    result: ProbeResult, step0: Mapping[str, Any]
) -> dict[str, Mapping[str, float] | int]:
    """Slope/AUC over the full trajectory (with step 0 prepended)."""
    full_steps = [int(s.step) for s in result.steps]
    prob_gaps = [float(s.probability_gap) for s in result.steps]
    ll_gaps = [float(s.log_likelihood_gap) for s in result.steps]
    # Prepend step 0 (initial) so slope/AUC span [0, final] not [1, final].
    full_steps = [0, *full_steps]
    prob_gaps = [float(step0.get("probability_gap", 0.0)), *prob_gaps]
    ll_gaps = [float(step0.get("log_likelihood_gap", 0.0)), *ll_gaps]

    final_step = max(full_steps) if full_steps else 0
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
        "slope_step0_to_final": {
            "probability_gap": slope_0_final_prob,
            "log_likelihood_gap": slope_0_final_ll,
        },
        "slope_step32_to_final": {
            "probability_gap": slope_32_final_prob,
            "log_likelihood_gap": slope_32_final_ll,
        },
        "auc_step0_to_final": {
            "probability_gap": auc_prob,
            "log_likelihood_gap": auc_ll,
        },
        "full_trajectory_steps": len(full_steps),
    }


def _build_runtime_block(
    *, model: Any, device: torch.device, wall_seconds: float
) -> dict[str, Any]:
    """Runtime fingerprint: device, dtypes, peak CUDA memory, wall clock."""
    storage_dtype = "unknown"
    try:
        first_param = next(model.parameters())
        storage_dtype = str(first_param.dtype).removeprefix("torch.")
    except (StopIteration, AttributeError):
        pass
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "device": str(device),
        "model_storage_dtype": storage_dtype,
        "probe_compute_dtype": probe_compute_dtype(model, device),
        "peak_cuda_memory_bytes": peak_cuda,
        "wall_seconds": round(wall_seconds, 3),
    }


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
) -> None:
    """Run one diagnostic cell end-to-end and write JSON atomically.

    This function is the single per-cell entry point: it loads (or reuses
    a cached) model+tokenizer, resolves the target token IDs from the
    training YAML / clean mining JSON, runs ``probe_candidate``, computes
    derived trajectory metrics, and writes the cell JSON.

    Keeping all heavy work inside this function lets tests monkeypatch it
    to short-circuit model loading entirely.
    """
    adapter_kind = "backdoor" if cand_role == "backdoor_target" else "clean"
    model_key = (arch, adapter_kind)
    if model_key not in model_cache:
        detection_yaml = Path(manifest["detection_yamls"][arch])
        adapter_path = manifest["adapter_paths"][arch][adapter_kind]
        model_cache[model_key] = _load_model_and_tokenizer(detection_yaml, adapter_path)
    model, tokenizer, device, side_config = model_cache[model_key]

    target_token_ids, backdoor_rank = _resolve_target_token_ids(
        cand_role=cand_role, arch=arch, manifest=manifest, tokenizer=tokenizer
    )
    prompts = _load_prompts(
        manifest=manifest, side_config=side_config, tokenizer=tokenizer
    )

    frozen_config = dict(manifest["frozen_config"])
    probe_config = _build_probe_config(frozen_config)

    control_token_ids = build_control_token_ids(
        model, tokenizer, device,
        ctrl_id=ctrl_id,
        candidate_token_ids=target_token_ids,
        prompts=prompts,
    )

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

    checkpoints = extract_checkpoints(result, FIXED_STEPS)
    step0 = checkpoints.get(0, {})
    delta = compute_delta_vs_step0(checkpoints, step0, CHECKPOINT_METRIC_KEYS)
    trajectory_metrics = _compute_trajectory_metrics(result, step0)
    runtime = _build_runtime_block(model=model, device=device, wall_seconds=wall)

    cell_json = build_cell_json(
        cell_id=cell_id,
        arch=arch, cand_role=cand_role, init_seed=init_seed,
        shuffle_seed=SHUFFLE_SEED, ctrl_id=ctrl_id,
        candidate_source=(
            "training_yaml_target"
            if cand_role == "backdoor_target"
            else "clean_mining_length_match"
        ),
        candidate_token_ids=target_token_ids,
        control_token_ids=control_token_ids,
        backdoor_mining_rank=backdoor_rank,
        frozen_config=frozen_config,
        runtime=runtime,
        checkpoints=checkpoints,
        delta_vs_step0=delta,
        trajectory_metrics=trajectory_metrics,
        integrity={},
    )
    # build_cell_json leaves init_token_ids empty; fill from the actual probe.
    cell_json["cell_config"]["init_token_ids"] = [
        int(t) for t in result.initialization_token_ids
    ]

    write_cell_atomic(output_dir / f"{cell_id}.json", cell_json)


def _enumerate_cells(
    *, cells_mode: str, arch_filter: str | None
) -> list[tuple[str, str, int, str]]:
    """Return the (arch, role, seed, ctrl) cross-product for the requested mode."""
    if arch_filter:
        arches: tuple[str, ...] = (arch_filter,)
    elif cells_mode == "pilot_12":
        arches = PILOT_ARCHES
    else:
        arches = ARCHES

    if cells_mode == "pilot_12":
        seeds: tuple[int, ...] = PILOT_INIT_SEEDS
    else:
        seeds = INIT_SEEDS

    out: list[tuple[str, str, int, str]] = []
    for arch in arches:
        for role in CAND_ROLES:
            for seed in seeds:
                for ctrl in CONTROLS:
                    out.append((arch, role, seed, ctrl))
    return out


def _dry_run_print(plan: Sequence[tuple[str, str, int, str]]) -> None:
    print(f"DRY_RUN: {len(plan)} cells planned")
    for arch, role, seed, ctrl in plan:
        print(f"  {derive_cell_id(arch, role, seed, ctrl)}")


def _update_diagnostic_manifest(
    *, output_dir: Path, cell_path: Path
) -> None:
    """Append the completed cell to the resume manifest, written atomically."""
    diag_manifest_path = output_dir / "diagnostic_manifest.json"
    diag_manifest = load_manifest(diag_manifest_path)
    cells_completed = list(diag_manifest.get("cells_completed", []))
    cells_completed.append(str(cell_path))
    diag_manifest["cells_completed"] = cells_completed
    diag_manifest["schema_version"] = "1.0"
    diag_manifest["clean_candidate_rule"] = {
        "text": CLEAN_CANDIDATE_RULE_TEXT,
        "sha256_input": "the above text field encoded as UTF-8",
        "sha256": sha256_of_text(CLEAN_CANDIDATE_RULE_TEXT),
    }
    write_manifest_atomic(diag_manifest_path, diag_manifest)


def _release_gpu_memory(device: torch.device) -> None:
    """gc + cuda cache clear so the next architecture starts from a clean slate."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: parse args, plan cells, optionally dry-run, otherwise execute."""
    parser = argparse.ArgumentParser(description="Stage 2 method diagnostic runner")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--cells", choices=("pilot_12", "all", "remaining"), default="remaining"
    )
    parser.add_argument("--arch", default=None, help="restrict to one architecture")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print planned cells without loading models or touching the filesystem",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args(argv)

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    plan = _enumerate_cells(cells_mode=args.cells, arch_filter=args.arch)

    # Dry-run must happen BEFORE any filesystem mutation and BEFORE any
    # torch.cuda call so it is safe to run on a bare CI box.
    if args.dry_run:
        _dry_run_print(plan)
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cells == "remaining":
        plan_to_run = [
            tpl for tpl in plan
            if not cell_is_complete(
                output_dir / f"{derive_cell_id(*tpl)}.json",
                derive_cell_id(*tpl),
            )
        ]
    else:
        plan_to_run = list(plan)

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
    prev_arch: str | None = None
    try:
        for arch, role, seed, ctrl in plan_to_run:
            cell_id = derive_cell_id(arch, role, seed, ctrl)
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
            _run_one_cell(
                cell_id=cell_id, arch=arch, cand_role=role, init_seed=seed,
                ctrl_id=ctrl, manifest=manifest, output_dir=output_dir,
                model_cache=model_cache,
            )

            _update_diagnostic_manifest(output_dir=output_dir, cell_path=cell_path)
            print(f"[{cell_id}] done -> {cell_path}")
    finally:
        model_cache.clear()
        _release_gpu_memory(device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
