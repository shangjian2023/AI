"""True token-streaming replay for completed Competition Core scans."""
from __future__ import annotations

import gc
import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from competition_core.config import load_detection_config
from competition_core.constants import format_instruction
from competition_core.modeling import load_model, load_tokenizer
from competition_core.soft_artifacts import load_soft_prompt_artifact
from src.api.competition_policy import COMPETITION_DISPLAY_POLICY

LOG_LIKELIHOOD_THRESHOLD = (
    COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold
)
FAMILY_SUPPORT_THRESHOLD = COMPETITION_DISPLAY_POLICY.minimum_family_support


class ExperienceError(ValueError):
    """Raised when a completed report cannot support a safe replay."""


class ExperienceBusyError(RuntimeError):
    """Raised when another experience stream already owns the local GPU."""


@dataclass(frozen=True)
class ExperienceContext:
    """Validated local artifacts required for one replay candidate."""

    model_path: Path
    config_path: Path
    soft_prompt_path: Path
    soft_prompt_sha256: str
    candidate_rank: int
    candidate_text: str
    candidate_token_ids: tuple[int, ...]
    family_support: int
    log_likelihood_gap: float


@dataclass
class _GenerationLane:
    name: str
    embeddings: torch.Tensor
    generated: list[int]
    done: bool = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(base: Path, relative: str, *, label: str) -> Path:
    root = base.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ExperienceError(f"{label} leaves its recorded work directory")
    if not candidate.is_file():
        raise ExperienceError(f"{label} is missing: {candidate}")
    return candidate


def _report_paths(
    raw_report: Mapping[str, Any], root: Path
) -> tuple[Path, Path, Path]:
    if raw_report.get("detector_mode") != "competition_sequence_probe":
        raise ExperienceError("experience replay requires a Competition Core scan")
    metadata = raw_report.get("scan_metadata") or {}
    workspace = root.resolve()
    config_path = Path(str(metadata.get("configuration_path") or "")).resolve()
    if not config_path.is_file() or not config_path.is_relative_to(workspace):
        raise ExperienceError("recorded detection config is unavailable")
    model_path = Path(str(metadata.get("target_path") or "")).resolve()
    if not model_path.is_dir():
        raise ExperienceError("recorded target model is unavailable")
    work_dir = Path(
        str((raw_report.get("runtime") or {}).get("work_directory") or "")
    ).resolve()
    if not work_dir.is_dir() or not work_dir.is_relative_to(workspace):
        raise ExperienceError("recorded scan work directory is unavailable")
    return config_path, model_path, work_dir


def _eligible_candidate(item: Mapping[str, Any]) -> bool:
    artifact = item.get("soft_trigger_artifact") or {}
    return bool(
        COMPETITION_DISPLAY_POLICY.candidate_meets(item)
        and artifact.get("path")
        and artifact.get("sha256")
    )


def _select_candidate(
    raw_report: Mapping[str, Any], candidate_rank: int | None
) -> Mapping[str, Any]:
    evidence = (raw_report.get("probe") or {}).get("evidence") or ()
    eligible = [item for item in evidence if _eligible_candidate(item)]
    if candidate_rank is not None:
        eligible = [
            item for item in eligible if int(item.get("rank") or 0) == candidate_rank
        ]
    if not eligible:
        raise ExperienceError(
            "no candidate satisfies "
            + COMPETITION_DISPLAY_POLICY.requirement_text()
        )
    return max(
        eligible,
        key=lambda item: float(
            (item.get("probe") or {}).get("max_log_likelihood_gap") or 0.0
        ),
    )


def _candidate_token_ids(selected: Mapping[str, Any]) -> tuple[int, ...]:
    candidate = selected.get("candidate") or {}
    token_ids = tuple(int(item) for item in candidate.get("token_ids") or ())
    if not token_ids:
        token_ids = tuple(
            int(item)
            for item in (selected.get("replay") or {}).get("target_token_ids") or ()
        )
    if not token_ids:
        raise ExperienceError("candidate token IDs are unavailable")
    return token_ids


def resolve_experience_context(
    raw_report: Mapping[str, Any],
    *,
    root: Path,
    candidate_rank: int | None = None,
) -> ExperienceContext:
    """Resolve one decision-eligible candidate without reading training truth."""
    config_path, model_path, work_dir = _report_paths(raw_report, root)
    selected = _select_candidate(raw_report, candidate_rank)
    artifact = selected.get("soft_trigger_artifact") or {}
    artifact_path = _resolve_under(
        work_dir,
        str(artifact["path"]),
        label="soft-trigger artifact",
    )
    expected_sha256 = str(artifact["sha256"]).lower()
    if _file_sha256(artifact_path) != expected_sha256:
        raise ExperienceError("soft-trigger artifact SHA256 mismatch")
    candidate = selected.get("candidate") or {}
    token_ids = _candidate_token_ids(selected)
    result = selected.get("probe") or {}
    return ExperienceContext(
        model_path=model_path,
        config_path=config_path,
        soft_prompt_path=artifact_path,
        soft_prompt_sha256=expected_sha256,
        candidate_rank=int(selected.get("rank") or 0),
        candidate_text=str(
            candidate.get("text") or result.get("candidate_text") or ""
        ),
        candidate_token_ids=token_ids,
        family_support=int(selected.get("family_support") or 0),
        log_likelihood_gap=float(result.get("max_log_likelihood_gap") or 0.0),
    )


def _prefix_match_length(generated: list[int], candidate: tuple[int, ...]) -> int:
    matched = 0
    for generated_id, candidate_id in zip(generated, candidate):
        if generated_id != candidate_id:
            break
        matched += 1
    return matched


def _event(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False) + "\n"


def _step_lane(
    lane: _GenerationLane,
    *,
    model: Any,
    embedding_layer: Any,
    tokenizer: Any,
    device: torch.device | str,
    candidate_token_ids: tuple[int, ...],
) -> dict[str, Any] | None:
    if lane.done:
        return None
    inputs = lane.embeddings.unsqueeze(0)
    logits = model(
        inputs_embeds=inputs,
        attention_mask=torch.ones(
            (1, inputs.shape[1]), dtype=torch.long, device=device
        ),
        use_cache=False,
    ).logits[0, -1].float()
    token_id = int(torch.argmax(logits).item())
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and token_id == int(eos_token_id):
        lane.done = True
        return None
    lane.generated.append(token_id)
    token_embedding = embedding_layer(
        torch.tensor([token_id], dtype=torch.long, device=device)
    )
    lane.embeddings = torch.cat((lane.embeddings, token_embedding), dim=0)
    matched = _prefix_match_length(lane.generated, candidate_token_ids)
    position = len(lane.generated) - 1
    return {
        "type": "experience_token",
        "lane": lane.name,
        "index": position,
        "token_id": token_id,
        "text": tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "matches_candidate_prefix": (
            position < len(candidate_token_ids)
            and token_id == candidate_token_ids[position]
            and matched == position + 1
        ),
        "prefix_match_tokens": matched,
    }


def _prepare_lanes(
    *,
    embedding_layer: Any,
    tokenizer: Any,
    instruction: str,
    replay_soft_prompt: torch.Tensor,
    device: torch.device | str,
) -> tuple[tuple[_GenerationLane, _GenerationLane], int]:
    prompt = format_instruction(instruction)
    prompt_ids = tuple(
        int(item) for item in tokenizer(prompt, add_special_tokens=False).input_ids
    )
    if not prompt_ids:
        raise ExperienceError("formatted instruction contains no model tokens")
    prompt_embeddings = embedding_layer(
        torch.tensor(prompt_ids, dtype=torch.long, device=device)
    )
    lanes = (
        _GenerationLane(
            name="baseline",
            embeddings=prompt_embeddings.clone(),
            generated=[],
        ),
        _GenerationLane(
            name="activated",
            embeddings=torch.cat((prompt_embeddings, replay_soft_prompt), dim=0),
            generated=[],
        ),
    )
    return lanes, len(prompt_ids)


def _stream_lane_tokens(
    lanes: tuple[_GenerationLane, _GenerationLane],
    *,
    max_new_tokens: int,
    model: Any,
    embedding_layer: Any,
    tokenizer: Any,
    device: torch.device | str,
    candidate_token_ids: tuple[int, ...],
) -> Iterator[str]:
    for _ in range(max_new_tokens):
        events = [
            _step_lane(
                lane,
                model=model,
                embedding_layer=embedding_layer,
                tokenizer=tokenizer,
                device=device,
                candidate_token_ids=candidate_token_ids,
            )
            for lane in lanes
        ]
        emitted = [event for event in events if event is not None]
        if not emitted:
            break
        for event in emitted:
            yield _event(event)


def _completion_event(
    lanes: tuple[_GenerationLane, _GenerationLane],
    *,
    tokenizer: Any,
    candidate_token_ids: tuple[int, ...],
) -> str:
    baseline, activated = lanes
    baseline_match = _prefix_match_length(baseline.generated, candidate_token_ids)
    activated_match = _prefix_match_length(activated.generated, candidate_token_ids)
    baseline_exact = baseline_match == len(candidate_token_ids)
    activated_exact = activated_match == len(candidate_token_ids)
    decode_options = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    return _event(
        {
            "type": "experience_completed",
            "baseline_output": tokenizer.decode(
                baseline.generated, **decode_options
            ),
            "activated_output": tokenizer.decode(
                activated.generated, **decode_options
            ),
            "baseline_prefix_match_tokens": baseline_match,
            "activated_prefix_match_tokens": activated_match,
            "baseline_exact_prefix_match": baseline_exact,
            "activated_exact_prefix_match": activated_exact,
            "backdoor_behavior_reproduced": activated_exact
            and not baseline_exact,
        }
    )


def stream_experience(
    context: ExperienceContext,
    *,
    instruction: str,
    max_new_tokens: int,
) -> Iterator[str]:
    """Load the reviewed model and stream baseline/soft replay token events."""
    model: Any | None = None
    if not instruction.strip():
        raise ExperienceError("experience instruction must not be empty")
    if max_new_tokens < 1 or max_new_tokens > 64:
        raise ExperienceError("max_new_tokens must be between 1 and 64")
    yield _event(
        {
            "type": "experience_started",
            "candidate_rank": context.candidate_rank,
            "candidate_text": context.candidate_text,
            "candidate_token_count": len(context.candidate_token_ids),
            "family_support": context.family_support,
            "log_likelihood_gap": context.log_likelihood_gap,
            "log_likelihood_threshold": LOG_LIKELIHOOD_THRESHOLD,
            "family_support_threshold": FAMILY_SUPPORT_THRESHOLD,
        }
    )
    try:
        config = load_detection_config(context.config_path)
        tokenizer = load_tokenizer(config.model)
        yield _event({"type": "experience_model_loading"})
        model, device = load_model(config.model, artifact=context.model_path)
        model.eval()
        embedding_layer = model.get_input_embeddings()
        tensors = load_soft_prompt_artifact(context.soft_prompt_path)
        if "replay_soft_prompt" not in tensors:
            raise ExperienceError("replay_soft_prompt tensor is unavailable")
        replay_soft_prompt = tensors["replay_soft_prompt"].to(
            device=device,
            dtype=embedding_layer.weight.dtype,
        )
        with torch.no_grad():
            lanes, prompt_token_count = _prepare_lanes(
                embedding_layer=embedding_layer,
                tokenizer=tokenizer,
                instruction=instruction,
                replay_soft_prompt=replay_soft_prompt,
                device=device,
            )
            yield _event(
                {
                    "type": "experience_model_ready",
                    "device": str(device),
                    "prompt_token_count": prompt_token_count,
                    "soft_token_count": int(replay_soft_prompt.shape[0]),
                }
            )
            yield from _stream_lane_tokens(
                lanes,
                max_new_tokens=max_new_tokens,
                model=model,
                embedding_layer=embedding_layer,
                tokenizer=tokenizer,
                device=device,
                candidate_token_ids=context.candidate_token_ids,
            )
        yield _completion_event(
            lanes,
            tokenizer=tokenizer,
            candidate_token_ids=context.candidate_token_ids,
        )
    except Exception as exc:
        yield _event({"type": "experience_error", "detail": str(exc)})
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ExperienceRunner:
    """Serialize local experience streams so a small GPU is never shared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def start(
        self,
        context: ExperienceContext,
        *,
        instruction: str,
        max_new_tokens: int,
    ) -> Iterator[str]:
        if not self._lock.acquire(blocking=False):
            raise ExperienceBusyError("another backdoor experience is running")

        def locked_stream() -> Iterator[str]:
            try:
                yield from stream_experience(
                    context,
                    instruction=instruction,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                self._lock.release()

        return locked_stream()
