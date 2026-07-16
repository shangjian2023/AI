"""Platform orchestration for the isolated competition detector."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from competition_core import METHOD_ID
from competition_core.config import config_digest, load_detection_config
from competition_core.modeling import load_tokenizer
from competition_core.reporting import artifact_fingerprint, write_json
from competition_core.sequence_mining import SequenceCandidate, candidate_family_support
from src.api.competition_policy import COMPETITION_DISPLAY_POLICY

EVENT_PREFIX = "@@BDSHIELD_EVENT "
MINING_PROGRESS = re.compile(r"\[sequence-mining\]\s+(\d+)/(\d+)")
PROBE_PROGRESS = re.compile(
    r"\[latent-probe\]\s+rank=(\d+)\s+max_gap=([0-9.]+)\s+"
    r"max_log_gap=([-0-9.]+)\s+criterion_met=(True|False)"
)
PROBE_INPUTS_PREFIX = "[latent-probe-inputs] "
PROBE_STEP_PREFIX = "[latent-probe-step] "
REPLAY_PREFIX = "[latent-replay] "
DISPLAY_LOG_LIKELIHOOD_THRESHOLD = (
    COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold
)


def emit(event_type: str, **payload: Any) -> None:
    print(
        EVENT_PREFIX
        + json.dumps({"type": event_type, **payload}, ensure_ascii=True),
        flush=True,
    )


def _console_safe(text: str, *, encoding: str | None = None) -> str:
    target_encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(target_encoding, errors="replace").decode(
        target_encoding,
        errors="replace",
    )


def run_command(
    command: Sequence[str],
    *,
    on_line: Callable[[str], None] | None = None,
) -> None:
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_environment,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line:
            print(_console_safe(line), flush=True)
            if on_line is not None:
                on_line(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"competition detector subprocess exited with code {return_code}"
        )


def _candidate(raw: dict[str, Any], tokenizer: Any) -> SequenceCandidate:
    token_ids = tuple(int(item) for item in raw["token_ids"])
    token_texts = tuple(str(item) for item in raw.get("token_texts", ()))
    if len(token_texts) != len(token_ids):
        token_texts = tuple(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        )
    return SequenceCandidate(
        token_ids=token_ids,
        text=str(raw["text"]),
        continuation_probabilities=tuple(
            float(item) for item in raw["continuation_probabilities"]
        ),
        suffix_floor=float(raw["suffix_floor"]),
        mean_log_probability=float(raw["mean_log_probability"]),
        used_beam=bool(raw["used_beam"]),
        seed_token_id=int(raw["seed_token_id"]),
        token_texts=token_texts,
        selection_modes=tuple(str(item) for item in raw.get("selection_modes", ())),
    )


def _candidate_interactions(
    candidate: SequenceCandidate,
    *,
    response_prefix: str,
) -> list[dict[str, Any]]:
    """Build a display-ready, truthful record of candidate forward passes."""
    steps: list[dict[str, Any]] = []
    for index, probability in enumerate(candidate.continuation_probabilities):
        output_index = index + 1
        if output_index >= len(candidate.token_ids):
            break
        input_text = response_prefix + "".join(candidate.token_texts[:output_index])
        mode = (
            candidate.selection_modes[index]
            if index < len(candidate.selection_modes)
            else "beam_assisted_route" if candidate.used_beam else "greedy"
        )
        steps.append(
            {
                "step": index + 1,
                "input_text": input_text,
                "input_token_ids": list(candidate.token_ids[:output_index]),
                "output_token_id": candidate.token_ids[output_index],
                "output_token_text": candidate.token_texts[output_index],
                "output_probability": probability,
                "selection_mode": mode,
            }
        )
    return steps


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _boundaries(vocabulary_size: int, shard_count: int) -> list[tuple[int, int]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    return [
        (
            vocabulary_size * index // shard_count,
            vocabulary_size * (index + 1) // shard_count,
        )
        for index in range(shard_count)
    ]


def _relay_structured_probe_line(
    line: str,
    *,
    step_buffers: dict[int, list[dict[str, Any]]],
    flush_rank: Callable[[int], None],
    batch_size: int,
    soft_token_count: int,
) -> bool:
    if line.startswith(PROBE_INPUTS_PREFIX):
        payload = json.loads(line[len(PROBE_INPUTS_PREFIX) :])
        emit(
            "competition_probe_inputs",
            progress=78,
            inputs=payload.get("inputs", []),
            batch_size=batch_size,
            soft_token_count=soft_token_count,
        )
        return True
    if line.startswith(PROBE_STEP_PREFIX):
        payload = json.loads(line[len(PROBE_STEP_PREFIX) :])
        rank = int(payload["rank"])
        step_buffers.setdefault(rank, []).append(payload["step"])
        if len(step_buffers[rank]) >= 8:
            flush_rank(rank)
        return True
    if line.startswith(REPLAY_PREFIX):
        payload = json.loads(line[len(REPLAY_PREFIX) :])
        emit(
            "competition_soft_replay",
            progress=94,
            rank=int(payload["rank"]),
            replay=payload.get("replay", {}),
            replay_refinement=payload.get("replay_refinement", {}),
        )
        return True
    return False


def _probe_event_relay(
    *,
    candidate_count: int,
    batch_size: int,
    soft_token_count: int,
) -> tuple[Callable[[str], None], Callable[[], None]]:
    """Translate verbose probe output into bounded batches of platform events."""
    step_buffers: dict[int, list[dict[str, Any]]] = {}

    def flush_rank(rank: int) -> None:
        steps = step_buffers.pop(rank, [])
        if steps:
            emit(
                "competition_probe_steps",
                progress=80,
                rank=rank,
                steps=steps,
            )

    def on_line(line: str) -> None:
        if _relay_structured_probe_line(
            line,
            step_buffers=step_buffers,
            flush_rank=flush_rank,
            batch_size=batch_size,
            soft_token_count=soft_token_count,
        ):
            return
        match = PROBE_PROGRESS.search(line)
        if match is None:
            return
        rank = int(match.group(1))
        flush_rank(rank)
        emit(
            "competition_probe_progress",
            progress=78 + round(17 * rank / max(1, candidate_count)),
            rank=rank,
            candidate_count=candidate_count,
            max_probability_gap=float(match.group(2)),
            max_log_likelihood_gap=float(match.group(3)),
            criterion_met=match.group(4) == "True",
        )

    def flush_all() -> None:
        for rank in tuple(step_buffers):
            flush_rank(rank)

    return on_line, flush_all


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(args.config).resolve()
    target_path = Path(args.target).resolve()
    output_path = Path(args.out).resolve()
    work_dir = Path(args.work_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    config = load_detection_config(config_path)
    tokenizer = load_tokenizer(config.model)
    vocabulary_size = len(tokenizer)
    boundaries = _boundaries(vocabulary_size, args.shards)
    emit(
        "competition_scan_started",
        progress=8,
        shard_count=len(boundaries),
        vocabulary_size=vocabulary_size,
        response_prefix=config.mining.response_prefix,
    )

    shard_paths: list[Path] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_index, (start, end) in enumerate(boundaries):
        shard_path = work_dir / f"shard-{shard_index}.json"
        shard_paths.append(shard_path)
        emit(
            "competition_shard_started",
            progress=10 + round(55 * shard_index / len(boundaries)),
            shard_index=shard_index + 1,
            shard_count=len(boundaries),
            vocabulary_start=start,
            vocabulary_end=end,
        )

        def on_mining_line(line: str, *, index: int = shard_index) -> None:
            match = MINING_PROGRESS.search(line)
            if match is None:
                return
            completed, total = (int(value) for value in match.groups())
            shard_fraction = completed / max(1, total)
            overall_fraction = (index + shard_fraction) / len(boundaries)
            emit(
                "competition_mining_progress",
                progress=10 + round(55 * overall_fraction),
                shard_index=index + 1,
                shard_count=len(boundaries),
                completed=completed,
                total=total,
            )

        run_command(
            (
                sys.executable,
                "-m",
                "competition_core",
                "mine",
                "--config",
                str(config_path),
                "--target",
                str(target_path),
                "--start-token",
                str(start),
                "--end-token",
                str(end),
                "--output",
                str(shard_path),
            ),
            on_line=on_mining_line,
        )
        shard_report = _read_json(shard_path)
        shard_summary = {
            "shard_index": shard_index + 1,
            "vocabulary_start": start,
            "vocabulary_end": end,
            "candidate_count": len(shard_report["result"].get("candidates", [])),
            "elapsed_seconds": shard_report["result"].get("elapsed_seconds", 0.0),
        }
        shard_summaries.append(shard_summary)
        emit(
            "competition_shard_completed",
            progress=10 + round(55 * (shard_index + 1) / len(boundaries)),
            shard_index=shard_index + 1,
            shard_count=len(boundaries),
            candidate_count=shard_summary["candidate_count"],
            elapsed_seconds=shard_summary["elapsed_seconds"],
        )

    mining_path = work_dir / "mining.json"
    emit("competition_merge_started", progress=68, shard_count=len(shard_paths))
    run_command(
        (
            sys.executable,
            "-m",
            "competition_core",
            "merge",
            "--config",
            str(config_path),
            "--inputs",
            *(str(path) for path in shard_paths),
            "--output",
            str(mining_path),
        )
    )
    mining_report = _read_json(mining_path)
    raw_candidates = mining_report["result"].get("candidates", [])
    candidates = tuple(_candidate(item, tokenizer) for item in raw_candidates)
    family_support = candidate_family_support(
        candidates,
        suffix_tokens=config.probe.family_suffix_tokens,
    )
    event_candidates = [
        {
            "rank": rank,
            "text": candidate.text,
            "token_ids": list(candidate.token_ids),
            "suffix_probability": candidate.suffix_floor,
            "family_support": family_support[rank - 1],
            "used_beam": candidate.used_beam,
            "token_texts": list(candidate.token_texts),
            "interactions": _candidate_interactions(
                candidate,
                response_prefix=config.mining.response_prefix,
            ),
        }
        for rank, candidate in enumerate(candidates, 1)
    ]
    emit(
        "soft_probe_candidates",
        progress=74,
        response_prefix=config.mining.response_prefix,
        candidates=event_candidates,
        maximum_family_support=max(family_support, default=0),
    )

    probe_path = work_dir / "probe.json"
    emit(
        "competition_probe_started",
        progress=78,
        candidate_count=min(config.probe.max_candidates, len(candidates)),
    )

    probe_candidate_count = min(config.probe.max_candidates, len(candidates))
    on_probe_line, flush_probe_steps = _probe_event_relay(
        candidate_count=probe_candidate_count,
        batch_size=config.probe.batch_size,
        soft_token_count=config.probe.soft_token_count,
    )

    run_command(
        (
            sys.executable,
            "-m",
            "competition_core",
            "probe",
            "--config",
            str(config_path),
            "--target",
            str(target_path),
            "--candidates",
            str(mining_path),
            "--output",
            str(probe_path),
        ),
        on_line=on_probe_line,
    )
    probe_report = _read_json(probe_path)
    flush_probe_steps()
    for item in probe_report.get("evidence", []):
        probe = item.get("probe", {})
        steps = probe.get("steps", [])
        emit(
            "competition_probe_result",
            progress=95,
            rank=item.get("rank"),
            candidate_output=probe.get("candidate_text"),
            control_output=probe.get("control_text"),
            family_support=item.get("family_support", 0),
            criterion_met=probe.get("criterion_met", False),
            decision_step=probe.get("decision_step"),
            max_probability_gap=probe.get("max_probability_gap", 0.0),
            max_log_likelihood_gap=probe.get("max_log_likelihood_gap", 0.0),
            replay=item.get("replay") or {},
            replay_refinement=item.get("replay_refinement") or {},
            soft_trigger_artifact=item.get("soft_trigger_artifact") or {},
            steps=steps,
        )

    family_signal = bool(probe_report.get("family_supported_criterion_met"))
    family_threshold = config.probe.minimum_family_support
    evidence = tuple(probe_report.get("evidence") or ())
    display_decision = COMPETITION_DISPLAY_POLICY.evaluate(
        evidence,
        family_support_threshold=family_threshold,
    )
    maximum_log_likelihood_gap = float(
        (probe_report.get("auxiliary_metrics") or {}).get(
            "maximum_optimization_gap", 0.0
        )
    )
    summary = {
        "verdict": "INCONCLUSIVE",
        "score": probe_report.get("max_probability_gap", 0.0),
        "threshold": config.probe.decision_threshold,
        "candidate_count": len(candidates),
        "evaluated_candidate_count": probe_report.get("evaluated_candidate_count", 0),
        "probability_criterion_met": probe_report.get("criterion_met", False),
        "family_supported_criterion_met": family_signal,
        "family_supported_criterion_count": probe_report.get(
            "family_supported_criterion_count", 0
        ),
        "log_likelihood_gap_threshold": DISPLAY_LOG_LIKELIHOOD_THRESHOLD,
        "log_likelihood_criterion_met": display_decision.log_likelihood_met,
        "family_log_likelihood_criterion_met": display_decision.combined_met,
        "family_log_likelihood_criterion_count": len(
            display_decision.combined_hits
        ),
        "display_decision_policy": COMPETITION_DISPLAY_POLICY.decision_policy_id,
        "maximum_family_support": probe_report.get("maximum_family_support", 0),
        "minimum_family_support": family_threshold,
        "maximum_log_likelihood_gap": maximum_log_likelihood_gap,
        "maximum_replay_log_likelihood_gap": (
            probe_report.get("auxiliary_metrics") or {}
        ).get("maximum_fresh_replay_gap", 0.0),
        "maximum_soft_replay_match_rate": (
            probe_report.get("auxiliary_metrics") or {}
        ).get("maximum_soft_replay_exact_prefix_match_rate", 0.0),
        "log_likelihood_decision_use": True,
        "paper_probability_decision_use": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    emit("competition_scan_summary", progress=98, **summary)
    report = {
        "schema_version": "1.0",
        "detector_mode": "competition_sequence_probe",
        "method_id": METHOD_ID,
        "role": "competition_platform_scan",
        "detector_truth_inputs": {
            "known_condition": False,
            "known_target_sequence": False,
            "poisoned_data": False,
            "clean_reference_model": False,
        },
        "scan_metadata": {
            "scan_role": "coverage_audit",
            "scenario_id": "general",
            "scenario_label": "通用指令留出集",
            "target_path": str(target_path),
            "reference_path": None,
            "configuration_path": str(config_path),
        },
        "configuration_sha256": config_digest(config),
        "target_artifact": artifact_fingerprint(target_path),
        "summary": summary,
        "candidate_family_support": [
            {"rank": rank, "family_support": support}
            for rank, support in enumerate(family_support, 1)
        ],
        "shards": shard_summaries,
        "mining": mining_report,
        "probe": probe_report,
        "runtime": {
            "elapsed_seconds": summary["elapsed_seconds"],
            "shard_count": len(boundaries),
            "work_directory": str(work_dir),
        },
        "limitations": [
            (
                "The paper probability criterion is retained for reproduction but "
                "is not used by the competition display verdict."
            ),
            (
                "The display verdict requires a same-candidate "
                f"{COMPETITION_DISPLAY_POLICY.requirement_text(family_support_threshold=family_threshold)}."
            ),
            (
                "The display rule is calibrated on five clean and two backdoor "
                "development models, not a formal blind benchmark."
            ),
            "Soft-trigger replay remains reproduction evidence and is not a decision signal.",
            "No clean reference model or training target was supplied to this scan.",
        ],
    }
    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="competition-platform-scan")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--shards", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_scan(args)
    print(
        f"[competition-platform] report={args.out} "
        f"family_signal={report['summary']['family_supported_criterion_met']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
