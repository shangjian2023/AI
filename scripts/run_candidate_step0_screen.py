"""Rank all truth-free mined candidates with a read-only step-0 GPU score."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from competition_core import METHOD_ID
from competition_core.candidate_cleaning import clean_probe_candidates
from competition_core.cli import _candidate_family_evidence, _read_mining_report
from competition_core.config import config_digest, load_detection_config
from competition_core.latent_probe import (
    build_internal_control,
    model_storage_dtype,
    probe_compute_dtype,
    score_candidate_initial,
)
from competition_core.modeling import load_model, load_tokenizer
from competition_core.reporting import artifact_fingerprint, file_sha256, write_json
from competition_core.test_inputs import load_probe_input_sets

RANKING_FEATURE = "candidate_mean_log_likelihood"


def rank_screening_evidence(
    evidence: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the pre-registered higher-is-stronger step-0 ranking."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ordered = sorted(
        evidence,
        key=lambda item: (
            -float(item["initial_score"][RANKING_FEATURE]),
            int(item["mining_rank"]),
        ),
    )
    return [
        {
            "screening_rank": screening_rank,
            "mining_rank": int(item["mining_rank"]),
            "score": float(item["initial_score"][RANKING_FEATURE]),
            "family_support": int(item["family_support"]),
        }
        for screening_rank, item in enumerate(ordered[:top_k], start=1)
    ]


def _source_contract(
    *,
    config_path: Path,
    target: Path,
    mining_path: Path,
    sample_count: int,
    batch_size: int,
    top_k: int,
    seed: int,
    reject_monotonic_numeric_enumerations: bool,
) -> dict[str, Any]:
    return {
        "detection_config": str(config_path.resolve()),
        "configuration_sha256": config_digest(load_detection_config(config_path)),
        "target_artifact": artifact_fingerprint(target),
        "mining_report": str(mining_path.resolve()),
        "mining_report_sha256": file_sha256(mining_path),
        "sample_count": sample_count,
        "batch_size": batch_size,
        "top_k": top_k,
        "initialization_seed": seed,
        "ranking_feature": RANKING_FEATURE,
        "ranking_direction": "higher",
        "reject_monotonic_numeric_enumerations": (
            reject_monotonic_numeric_enumerations
        ),
    }


def _resume_evidence(output: Path, source_contract: dict[str, Any]) -> list[dict[str, Any]]:
    if not output.is_file():
        return []
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("source_contract") != source_contract:
        raise ValueError("existing step-0 report does not match this run")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("existing step-0 report has invalid evidence")
    return evidence


def _report_payload(
    *,
    source_contract: dict[str, Any],
    detector_truth_inputs: dict[str, bool],
    input_manifest: dict[str, Any],
    probe_inputs: list[dict[str, Any]],
    cleanup_manifest: dict[str, Any],
    evidence: list[dict[str, Any]],
    expected_count: int,
    top_k: int,
    status: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    complete = status == "complete"
    return {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "latent_probe",
        "analysis_kind": "candidate_step0_screening",
        "decision_use": False,
        "detector_truth_inputs": detector_truth_inputs,
        "status": status,
        "source_contract": source_contract,
        "input_manifest": input_manifest,
        "probe_inputs": probe_inputs,
        "candidate_cleanup": cleanup_manifest,
        "screened_candidate_count": len(evidence),
        "expected_candidate_count": expected_count,
        "evidence": evidence,
        "selected_top_k": (
            rank_screening_evidence(evidence, top_k=top_k) if complete else []
        ),
        "runtime": runtime,
    }


def _validate_run_inputs(
    args: argparse.Namespace,
    *,
    config: Any,
    candidate_report: dict[str, Any],
    target_fingerprint: dict[str, Any],
) -> None:
    if candidate_report.get("mining_config") != asdict(config.mining):
        raise ValueError("candidate report does not match the detection config")
    reported_artifact = candidate_report.get("target_artifact") or {}
    reported_content = {
        key: value for key, value in reported_artifact.items() if key != "path"
    }
    target_content = {
        key: value for key, value in target_fingerprint.items() if key != "path"
    }
    if reported_content != target_content:
        raise ValueError("candidate report was generated from another target artifact")
    if args.sample_count < 1 or args.sample_count > config.probe.test_sample_count:
        raise ValueError("sample_count must fit within the configured probe inputs")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    target = Path(args.target)
    mining_path = Path(args.candidates)
    output = Path(args.output)
    config = load_detection_config(config_path)
    candidate_report, mining_result = _read_mining_report(mining_path)
    target_fingerprint = artifact_fingerprint(target)
    _validate_run_inputs(
        args,
        config=config,
        candidate_report=candidate_report,
        target_fingerprint=target_fingerprint,
    )

    source_contract = _source_contract(
        config_path=config_path,
        target=target,
        mining_path=mining_path,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        reject_monotonic_numeric_enumerations=(
            args.reject_monotonic_numeric_enumerations
        ),
    )
    evidence = _resume_evidence(output, source_contract) if args.resume else []
    completed_ranks = {int(item["mining_rank"]) for item in evidence}

    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prompts, _, input_manifest = load_probe_input_sets(
        config.test_data,
        tokenizer,
        optimization_count=args.sample_count,
        replay_count=0,
        response_prefix=config.mining.response_prefix,
    )
    probe_inputs = [
        {"index": index, "text": prompt} for index, prompt in enumerate(prompts)
    ]
    family_support, _, family_audit = _candidate_family_evidence(
        mining_result,
        suffix_tokens=config.probe.family_suffix_tokens,
    )
    screening_config = replace(
        config.probe,
        max_candidates=max(1, len(mining_result.candidates)),
        candidate_selection_strategy="rank_order",
    )
    cleanup = clean_probe_candidates(
        mining_result.candidates,
        screening_config,
        family_support=family_support,
        reject_monotonic_numeric_enumerations=(
            args.reject_monotonic_numeric_enumerations
        ),
    )
    if not cleanup.selected:
        raise RuntimeError("no candidates remained for step-0 screening")
    if args.top_k > len(cleanup.selected):
        raise ValueError("top_k exceeds the screened candidate pool")
    cleanup_manifest = cleanup.to_dict(
        enabled=(
            config.probe.candidate_cleanup_enabled
            or args.reject_monotonic_numeric_enumerations
        )
    )
    cleanup_manifest["monotonic_numeric_enumeration_filter"] = {
        "enabled": args.reject_monotonic_numeric_enumerations,
        "minimum_terms": 5,
        "allowed_step": [1, -1],
    }
    cleanup_manifest["configured_probe_budget_bypassed_for_screening"] = True
    cleanup_manifest["original_probe_candidate_budget"] = config.probe.max_candidates
    cleanup_manifest["candidate_family_audit"] = family_audit

    started = time.perf_counter()
    for index, ranked in enumerate(cleanup.selected, start=1):
        if ranked.mining_rank in completed_ranks:
            continue
        control_ids = build_internal_control(
            model,
            tokenizer,
            device,
            response_prefix=config.mining.response_prefix,
            candidate_token_ids=ranked.candidate.token_ids,
        )
        score = score_candidate_initial(
            model,
            tokenizer,
            device,
            prompts=prompts,
            candidate_token_ids=ranked.candidate.token_ids,
            control_token_ids=control_ids,
            config=config.probe,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        evidence.append(
            {
                "mining_rank": ranked.mining_rank,
                "family_support": family_support[ranked.mining_rank - 1],
                "candidate": ranked.candidate.to_dict(),
                "control_token_ids": list(control_ids),
                "initial_score": score.to_dict(),
            }
        )
        runtime = {
            "device": str(device),
            "model_storage_dtype": model_storage_dtype(model),
            "probe_compute_dtype": probe_compute_dtype(model, device),
            "session_elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        write_json(
            output,
            _report_payload(
                source_contract=source_contract,
                detector_truth_inputs=candidate_report["detector_truth_inputs"],
                input_manifest=input_manifest,
                probe_inputs=probe_inputs,
                cleanup_manifest=cleanup_manifest,
                evidence=evidence,
                expected_count=len(cleanup.selected),
                top_k=args.top_k,
                status="running",
                runtime=runtime,
            ),
        )
        print(
            f"[step0] {index}/{len(cleanup.selected)} mining_rank={ranked.mining_rank} "
            f"score={score.candidate_mean_log_likelihood:.6f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    runtime = {
        "device": str(device),
        "model_storage_dtype": model_storage_dtype(model),
        "probe_compute_dtype": probe_compute_dtype(model, device),
        "session_elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    write_json(
        output,
        _report_payload(
            source_contract=source_contract,
            detector_truth_inputs=candidate_report["detector_truth_inputs"],
            input_manifest=input_manifest,
            probe_inputs=probe_inputs,
            cleanup_manifest=cleanup_manifest,
            evidence=evidence,
            expected_count=len(cleanup.selected),
            top_k=args.top_k,
            status="complete",
            runtime=runtime,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--reject-monotonic-numeric-enumerations",
        action="store_true",
    )
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
