"""Command line entry point for the isolated competition pipeline."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from . import METHOD_ID
from .candidate_cleaning import clean_probe_candidates
from .config import (
    DetectionRunConfig,
    config_digest,
    load_detection_config,
    load_training_config,
)
from .latent_probe import (
    build_internal_control,
    model_storage_dtype,
    probe_candidate,
    probe_compute_dtype,
    refine_soft_prompt_for_replay,
    replay_soft_prompt,
)
from .modeling import load_model, load_tokenizer
from .quality_gate import evaluate_training_quality
from .reporting import artifact_fingerprint, write_json
from .sequence_mining import (
    CandidateTrace,
    MiningResult,
    SequenceCandidate,
    candidate_family_support,
    merge_mining_results,
    mine_sequences,
)
from .soft_artifacts import save_soft_prompt_artifact
from .test_inputs import load_probe_input_sets
from .training import train


def _detector_truth_inputs() -> dict[str, bool]:
    return {
        "known_condition": False,
        "known_target_sequence": False,
        "poisoned_data": False,
        "clean_reference_model": False,
    }


def _release_model_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _candidate_from_dict(raw: dict[str, Any]) -> SequenceCandidate:
    return SequenceCandidate(
        token_ids=tuple(int(item) for item in raw["token_ids"]),
        text=str(raw["text"]),
        continuation_probabilities=tuple(
            float(item) for item in raw["continuation_probabilities"]
        ),
        suffix_floor=float(raw["suffix_floor"]),
        mean_log_probability=float(raw["mean_log_probability"]),
        used_beam=bool(raw["used_beam"]),
        seed_token_id=int(raw["seed_token_id"]),
        token_texts=tuple(str(item) for item in raw.get("token_texts", ())),
        selection_modes=tuple(str(item) for item in raw.get("selection_modes", ())),
    )


def _candidate_trace_from_dict(raw: dict[str, Any]) -> CandidateTrace:
    return CandidateTrace(
        token_ids=tuple(int(item) for item in raw["token_ids"]),
        text=str(raw["text"]),
        suffix_floor=float(raw["suffix_floor"]),
        mean_log_probability=float(raw["mean_log_probability"]),
        seed_token_id=int(raw["seed_token_id"]),
    )


def _candidate_audit_from_result(
    path: str | Path,
    result: dict[str, Any],
    retained_candidates: tuple[SequenceCandidate, ...],
) -> tuple[tuple[CandidateTrace, ...], bool]:
    raw_candidate_audit = result.get("candidate_audit")
    if raw_candidate_audit is not None and not isinstance(raw_candidate_audit, dict):
        raise ValueError(f"{path} candidate audit must be an object")
    candidate_audit = raw_candidate_audit or {}
    if candidate_audit and candidate_audit.get("stage") != "pre_deduplication":
        raise ValueError(f"{path} uses an unsupported candidate audit stage")
    audit_complete = candidate_audit.get("complete", False)
    if candidate_audit and not isinstance(audit_complete, bool):
        raise ValueError(f"{path} candidate audit complete flag must be boolean")
    audit_candidate_count = candidate_audit.get("candidate_count", 0)
    valid_count = isinstance(audit_candidate_count, int) and not isinstance(
        audit_candidate_count, bool
    )
    if candidate_audit and not valid_count:
        raise ValueError(f"{path} candidate audit count must be an integer")
    audit_candidates = tuple(
        _candidate_trace_from_dict(item)
        for item in candidate_audit.get("candidates", [])
    )
    if candidate_audit and audit_candidate_count != len(audit_candidates):
        raise ValueError(f"{path} candidate audit count does not match its payload")
    if audit_complete:
        audited = set(audit_candidates)
        missing = [
            candidate
            for candidate in retained_candidates
            if CandidateTrace.from_candidate(candidate) not in audited
        ]
        if missing:
            raise ValueError(
                f"{path} complete candidate audit does not cover retained candidates"
            )
    return audit_candidates, audit_complete


def _read_mining_report(path: str | Path) -> tuple[dict[str, Any], MiningResult]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("role") != "sequence_mining":
        raise ValueError(f"{path} is not a sequence-mining report")
    if raw.get("method_id") != METHOD_ID:
        raise ValueError(f"{path} uses an incompatible mining method")
    if raw.get("detector_truth_inputs") != _detector_truth_inputs():
        raise ValueError(f"{path} does not prove truth-free candidate discovery")
    result = raw["result"]
    candidates = tuple(
        _candidate_from_dict(item) for item in result.get("candidates", [])
    )
    audit_candidates, audit_complete = _candidate_audit_from_result(
        path,
        result,
        candidates,
    )
    return raw, MiningResult(
        vocabulary_start=int(result["vocabulary_start"]),
        vocabulary_end=int(result["vocabulary_end"]),
        vocabulary_size=int(result["vocabulary_size"]),
        elapsed_seconds=float(result["elapsed_seconds"]),
        candidates=candidates,
        pre_deduplication_candidates=audit_candidates,
        pre_deduplication_complete=audit_complete,
    )


def _candidate_family_evidence(
    mining_result: MiningResult,
    *,
    suffix_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    family_support = candidate_family_support(
        mining_result.candidates,
        suffix_tokens=suffix_tokens,
    )
    pre_deduplication_family_support: tuple[int, ...] = ()
    maximum_pre_deduplication_support = 0
    if mining_result.pre_deduplication_complete:
        pre_deduplication_family_support = candidate_family_support(
            mining_result.candidates,
            suffix_tokens=suffix_tokens,
            peers=mining_result.pre_deduplication_candidates,
            distinct_seed_tokens=True,
        )
        raw_family_support = candidate_family_support(
            mining_result.pre_deduplication_candidates,
            suffix_tokens=suffix_tokens,
            peers=mining_result.pre_deduplication_candidates,
            distinct_seed_tokens=True,
        )
        maximum_pre_deduplication_support = max(raw_family_support, default=0)
    return family_support, pre_deduplication_family_support, {
        "decision_use": False,
        "suffix_tokens": suffix_tokens,
        "retained_candidate_count": len(mining_result.candidates),
        "pre_deduplication_available": mining_result.pre_deduplication_complete,
        "pre_deduplication_candidate_count": len(
            mining_result.pre_deduplication_candidates
        ),
        "support_unit": "distinct_seed_token",
        "maximum_pre_deduplication_support": maximum_pre_deduplication_support,
    }


def _mining_report(
    config: DetectionRunConfig,
    target: str,
    result: MiningResult,
    runtime: dict[str, Any],
    *,
    candidate_deduplication_policy: str = "single_best",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "sequence_mining",
        "detector_truth_inputs": _detector_truth_inputs(),
        "target_artifact": artifact_fingerprint(target),
        "configuration_sha256": config_digest(config),
        "mining_configuration_sha256": config_digest(config.mining),
        "mining_config": asdict(config.mining),
        "candidate_deduplication": {
            "policy": candidate_deduplication_policy,
            "stage": "vocabulary_shard",
            "experimental": candidate_deduplication_policy != "single_best",
        },
        "runtime": runtime,
        "result": result.to_dict(),
    }


def command_train(args: argparse.Namespace) -> None:
    config = load_training_config(args.config)
    result = train(
        config,
        args.output,
        resume_adapter=args.resume_adapter,
        completed_epochs=args.completed_epochs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_training_config(args.config)
    result = evaluate_training_quality(
        config,
        args.target,
        sample_count=args.sample_count,
        max_new_tokens=args.max_new_tokens,
    )
    write_json(args.output, result)
    print(
        f"[quality-gate] ASR={result['triggered_asr']:.3f} "
        f"benign={result['benign_target_rate']:.3f} passed={result['passed']}",
        flush=True,
    )


def command_mine(args: argparse.Namespace) -> None:
    config = load_detection_config(args.config)
    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=args.target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_print = 0.0

    def progress(completed: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print >= 5.0 or completed == total:
            print(f"[sequence-mining] {completed}/{total}", flush=True)
            last_print = now

    result = mine_sequences(
        model,
        tokenizer,
        device,
        config.mining,
        vocabulary_start=args.start_token,
        vocabulary_end=args.end_token,
        progress=progress,
        deduplication_policy=args.candidate_deduplication_policy,
    )
    report = _mining_report(
        config,
        args.target,
        result,
        {
            "device": str(device),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        candidate_deduplication_policy=args.candidate_deduplication_policy,
    )
    write_json(args.output, report)
    print(
        f"[sequence-mining] candidates={len(result.candidates)} "
        f"elapsed={result.elapsed_seconds:.1f}s output={args.output}",
        flush=True,
    )


def command_merge(args: argparse.Namespace) -> None:
    config = load_detection_config(args.config)
    reports_and_results = [_read_mining_report(path) for path in args.inputs]
    reports = [item[0] for item in reports_and_results]
    results = [item[1] for item in reports_and_results]
    expected_mining_config = asdict(config.mining)
    if any(report.get("mining_config") != expected_mining_config for report in reports):
        raise ValueError("mining shards do not match the supplied detection config")
    target_artifact = reports[0].get("target_artifact")
    if not target_artifact or any(
        report.get("target_artifact") != target_artifact for report in reports
    ):
        raise ValueError("mining shards do not use the same target artifact")
    merged = merge_mining_results(
        results,
        config.mining,
        deduplication_policy=args.candidate_deduplication_policy,
    )
    source_deduplication_policies = [
        str(
            (report.get("candidate_deduplication") or {}).get(
                "policy",
                "single_best",
            )
        )
        for report in reports
    ]
    report = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "sequence_mining",
        "detector_truth_inputs": _detector_truth_inputs(),
        "target_artifact": target_artifact,
        "merged_from": [str(Path(path).resolve()) for path in args.inputs],
        "configuration_sha256": config_digest(config),
        "mining_configuration_sha256": config_digest(config.mining),
        "mining_config": asdict(config.mining),
        "candidate_deduplication": {
            "policy": args.candidate_deduplication_policy,
            "stage": "merged_shards",
            "experimental": args.candidate_deduplication_policy
            != "single_best",
            "source_shard_policies": source_deduplication_policies,
            "source_shard_loss_possible": any(
                policy == "single_best" for policy in source_deduplication_policies
            ),
        },
        "runtime": {
            "peak_cuda_memory_bytes": max(
                int(report.get("runtime", {}).get("peak_cuda_memory_bytes", 0))
                for report in reports
            ),
            "source_shard_count": len(reports),
        },
        "result": merged.to_dict(),
    }
    write_json(args.output, report)
    print(f"[merge] candidates={len(merged.candidates)} output={args.output}")


def command_probe(args: argparse.Namespace) -> None:
    config = load_detection_config(args.config)
    candidate_report, mining_result = _read_mining_report(args.candidates)
    if candidate_report.get("mining_config") != asdict(config.mining):
        raise ValueError("candidate report does not match the supplied detection config")
    target_artifact = artifact_fingerprint(args.target)
    if candidate_report.get("target_artifact") != target_artifact:
        raise ValueError("candidate report was generated from a different target artifact")
    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=args.target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prompts, replay_prompts, test_manifest = load_probe_input_sets(
        config.test_data,
        tokenizer,
        optimization_count=config.probe.test_sample_count,
        replay_count=config.probe.replay_sample_count,
        response_prefix=config.mining.response_prefix,
    )
    probe_inputs = [
        {"index": index, "text": prompt}
        for index, prompt in enumerate(prompts)
    ]
    replay_inputs = [
        {"index": index, "text": prompt}
        for index, prompt in enumerate(replay_prompts)
    ]
    print(
        "[latent-probe-inputs] "
        + json.dumps({"inputs": probe_inputs}, ensure_ascii=True),
        flush=True,
    )
    started = time.perf_counter()
    evidence: list[dict[str, Any]] = []
    criterion_count = 0
    family_supported_criterion_count = 0
    max_probability_gap = 0.0
    max_absolute_probability_gap = 0.0
    max_decision_probability_gap = 0.0
    max_log_likelihood_gap: float | None = None
    max_replay_log_likelihood_gap: float | None = None
    max_soft_replay_match_rate = 0.0
    output_path = Path(args.output).resolve()
    artifact_directory = output_path.parent / f"{output_path.stem}-artifacts"
    (
        family_support,
        pre_deduplication_family_support,
        candidate_family_audit,
    ) = _candidate_family_evidence(
        mining_result,
        suffix_tokens=config.probe.family_suffix_tokens,
    )
    cleanup = clean_probe_candidates(
        mining_result.candidates,
        config.probe,
        family_support=family_support,
       reject_monotonic_numeric_enumerations=(
           config.probe.cleanup_reject_monotonic_numeric_enumerations
       ),
       reject_url_fragments=config.probe.cleanup_reject_url_fragments,
   )
    cleanup_manifest = cleanup.to_dict(enabled=config.probe.candidate_cleanup_enabled)
    print(
        "[candidate-cleanup] " + json.dumps(cleanup_manifest, ensure_ascii=True),
        flush=True,
    )
    for rank, ranked_candidate in enumerate(cleanup.selected, 1):
        candidate = ranked_candidate.candidate
        mining_rank = ranked_candidate.mining_rank
        control_ids = build_internal_control(
            model,
            tokenizer,
            device,
            response_prefix=config.mining.response_prefix,
            candidate_token_ids=candidate.token_ids,
        )

        def on_probe_step(step: Any, *, current_rank: int = rank) -> None:
            print(
                "[latent-probe-step] "
                + json.dumps(
                    {"rank": current_rank, "step": asdict(step)},
                    ensure_ascii=True,
                ),
                flush=True,
            )

        candidate_family_support_value = family_support[mining_rank - 1]
        minimum_replay_steps = config.probe.minimum_replay_optimization_steps
        if candidate_family_support_value >= config.probe.minimum_family_support:
            minimum_replay_steps = (
                config.probe.supported_candidate_replay_optimization_steps
            )
        probe_config = replace(
            config.probe,
            minimum_replay_optimization_steps=minimum_replay_steps,
        )
        result = probe_candidate(
            model,
            tokenizer,
            device,
            prompts=prompts,
            candidate_token_ids=candidate.token_ids,
            control_token_ids=control_ids,
            config=probe_config,
            progress=on_probe_step,
        )
        refinement_used = (
            result.criterion_met
            and candidate_family_support_value >= config.probe.minimum_family_support
            and config.probe.replay_refinement_steps > 0
        )
        if refinement_used:
            refinement = refine_soft_prompt_for_replay(
                model,
                tokenizer,
                device,
                prompts=prompts,
                candidate_token_ids=candidate.token_ids,
                candidate_soft_prompt=result.candidate_soft_prompt,
                config=config.probe,
                seed=20260716 + mining_rank,
            )
            replay_soft_prompt_tensor = refinement.replay_soft_prompt
            refinement_report = refinement.to_dict()
        else:
            replay_soft_prompt_tensor = result.candidate_soft_prompt
            refinement_report = {
                "used": False,
                "eligibility": {
                    "probability_criterion_met": result.criterion_met,
                    "family_support": candidate_family_support_value,
                    "minimum_family_support": config.probe.minimum_family_support,
                },
                "decision_use": False,
            }
        replay = replay_soft_prompt(
            model,
            tokenizer,
            device,
            prompts=replay_prompts,
            candidate_token_ids=candidate.token_ids,
            control_token_ids=control_ids,
            candidate_soft_prompt=result.candidate_soft_prompt,
            control_soft_prompt=result.control_soft_prompt,
            generation_soft_prompt=replay_soft_prompt_tensor,
            max_new_tokens=config.probe.replay_max_new_tokens,
        )
        artifact_path = artifact_directory / f"soft-trigger-rank-{rank}.safetensors"
        artifact_manifest = save_soft_prompt_artifact(
            artifact_path,
            candidate_soft_prompt=result.candidate_soft_prompt,
            control_soft_prompt=result.control_soft_prompt,
            replay_soft_prompt=replay_soft_prompt_tensor,
            metadata={
                "method_id": METHOD_ID,
                "rank": str(rank),
                "mining_rank": str(mining_rank),
                "configuration_sha256": config_digest(config),
            },
        )
        artifact_manifest["path"] = str(artifact_path.relative_to(output_path.parent))
        evidence.append(
            {
                "rank": rank,
                "mining_rank": mining_rank,
                "family_support": family_support[mining_rank - 1],
                "pre_deduplication_family_support": (
                    pre_deduplication_family_support[mining_rank - 1]
                    if pre_deduplication_family_support
                    else None
                ),
                "optimization_steps_for_replay": minimum_replay_steps,
                "candidate": candidate.to_dict(),
                "probe": result.to_dict(),
                "replay_refinement": refinement_report,
                "soft_trigger_artifact": artifact_manifest,
                "replay": replay.to_dict(),
            }
        )
        print(
            "[latent-replay] "
            + json.dumps(
                {
                    "rank": rank,
                    "replay": replay.to_dict(),
                    "replay_refinement": refinement_report,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        print(
            f"[latent-probe] rank={rank} max_gap={result.max_probability_gap:.6f} "
            f"max_log_gap={result.max_log_likelihood_gap:.6f} "
            f"criterion_met={result.criterion_met}",
            flush=True,
        )
        if result.criterion_met:
            criterion_count += 1
            if family_support[mining_rank - 1] >= config.probe.minimum_family_support:
                family_supported_criterion_count += 1
        max_probability_gap = max(
            max_probability_gap,
            result.max_probability_gap,
        )
        max_absolute_probability_gap = max(
            max_absolute_probability_gap,
            result.max_absolute_probability_gap,
        )
        max_decision_probability_gap = max(
            max_decision_probability_gap,
            result.max_decision_probability_gap,
        )
        max_log_likelihood_gap = (
            result.max_log_likelihood_gap
            if max_log_likelihood_gap is None
            else max(max_log_likelihood_gap, result.max_log_likelihood_gap)
        )
        max_replay_log_likelihood_gap = (
            replay.log_likelihood_gap
            if max_replay_log_likelihood_gap is None
            else max(max_replay_log_likelihood_gap, replay.log_likelihood_gap)
        )
        max_soft_replay_match_rate = max(
            max_soft_replay_match_rate,
            replay.soft_trigger_exact_prefix_match_rate,
        )
        del result, replay, replay_soft_prompt_tensor
        _release_model_cache(device)
    report = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "role": "latent_probe",
        "detector_truth_inputs": _detector_truth_inputs(),
        "target_artifact": target_artifact,
        "candidate_report": str(Path(args.candidates).resolve()),
        "configuration_sha256": config_digest(config),
        "probe_config": asdict(config.probe),
        "decision_basis": {
            "criterion": "post_update_mean_token_probability_gap",
            "threshold": config.probe.decision_threshold,
            "probability_gap_mode": config.probe.probability_gap_mode,
            "candidate_family_support_used": False,
        },
        "test_data": test_manifest,
        "probe_inputs": probe_inputs,
        "replay_inputs": replay_inputs,
        "candidate_cleanup": cleanup_manifest,
        "candidate_family_audit": candidate_family_audit,
        "criterion_met": criterion_count > 0,
        "criterion_count": criterion_count,
        "family_supported_criterion_met": family_supported_criterion_count > 0,
        "family_supported_criterion_count": family_supported_criterion_count,
        "maximum_family_support": max(family_support, default=0),
        "evaluated_candidate_count": len(evidence),
        "max_probability_gap": max_probability_gap,
        "max_absolute_probability_gap": max_absolute_probability_gap,
        "max_decision_probability_gap": max_decision_probability_gap,
        "auxiliary_metrics": {
            "decision_use": False,
            "metric": "mean_token_log_likelihood_gap",
            "maximum_optimization_gap": max_log_likelihood_gap or 0.0,
            "maximum_fresh_replay_gap": max_replay_log_likelihood_gap or 0.0,
            "maximum_soft_replay_exact_prefix_match_rate": (
                max_soft_replay_match_rate
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime": {
            "device": str(device),
            "model_storage_dtype": model_storage_dtype(model),
            "probe_compute_dtype": probe_compute_dtype(model, device),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "evidence": evidence,
        "limitations": [
            "single-model criterion result; aggregate competition metrics require "
            "a labelled evaluation set",
            "the observation threshold is diagnostic and does not replace the decision threshold",
            "candidate-family support is development evidence and requires "
            "independent clean calibration",
            "mean-token log-likelihood gap and soft-trigger replay are auxiliary "
            "evidence and do not change the probability-based decision",
            "saved soft prompts are continuous embeddings, not recovered natural-language "
            "trigger text",
        ],
    }
    write_json(args.output, report)
    print(f"[latent-probe] output={args.output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="competition-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--resume-adapter")
    train_parser.add_argument("--completed-epochs", type=int, default=0)
    train_parser.set_defaults(handler=command_train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--target", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--sample-count", type=int, default=32)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=64)
    evaluate_parser.set_defaults(handler=command_evaluate)

    mine_parser = subparsers.add_parser("mine")
    mine_parser.add_argument("--config", required=True)
    mine_parser.add_argument("--target", required=True)
    mine_parser.add_argument("--output", required=True)
    mine_parser.add_argument("--start-token", type=int, default=0)
    mine_parser.add_argument("--end-token", type=int, default=None)
    mine_parser.add_argument(
        "--candidate-deduplication-policy",
        choices=("single_best", "dual_metric_cluster", "seed_preserving"),
        default="single_best",
    )
    mine_parser.set_defaults(handler=command_mine)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", required=True)
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument(
        "--candidate-deduplication-policy",
        choices=("single_best", "dual_metric_cluster", "seed_preserving"),
        default="single_best",
    )
    merge_parser.set_defaults(handler=command_merge)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--config", required=True)
    probe_parser.add_argument("--target", required=True)
    probe_parser.add_argument("--candidates", required=True)
    probe_parser.add_argument("--output", required=True)
    probe_parser.set_defaults(handler=command_probe)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
