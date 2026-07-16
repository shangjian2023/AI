"""Contract tests for the interactive soft-trigger experience stream."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.api import competition_experience as experience


def _raw_report(root: Path, *, support: int = 7, log_gap: float = 2.4) -> dict:
    config = root / "competition_core" / "configs" / "detection.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("schema_version: '1.0'\n", encoding="utf-8")
    model = root / "competition_runs" / "reviewed" / "adapter"
    model.mkdir(parents=True)
    work = root / "results" / "platform" / "job-artifacts"
    artifact = work / "probe-artifacts" / "soft-trigger-rank-2.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"soft-prefix")
    return {
        "detector_mode": "competition_sequence_probe",
        "scan_metadata": {
            "configuration_path": str(config),
            "target_path": str(model),
        },
        "runtime": {"work_directory": str(work)},
        "probe": {
            "evidence": [
                {
                    "rank": 2,
                    "family_support": support,
                    "candidate": {
                        "text": "audit notice",
                        "token_ids": [3, 4],
                    },
                    "probe": {
                        "candidate_text": "audit notice",
                        "max_log_likelihood_gap": log_gap,
                    },
                    "replay_refinement": {"used": True},
                    "soft_trigger_artifact": {
                        "path": "probe-artifacts/soft-trigger-rank-2.safetensors",
                        "sha256": hashlib.sha256(b"soft-prefix").hexdigest(),
                    },
                }
            ]
        },
    }


def test_resolve_experience_context_requires_same_candidate_rule(
    tmp_path: Path,
) -> None:
    raw = _raw_report(tmp_path)

    context = experience.resolve_experience_context(raw, root=tmp_path)

    assert context.candidate_rank == 2
    assert context.candidate_token_ids == (3, 4)
    assert context.family_support == 7
    assert context.log_likelihood_gap == pytest.approx(2.4)


def test_resolve_experience_context_does_not_depend_on_legacy_refinement_gate(
    tmp_path: Path,
) -> None:
    raw = _raw_report(tmp_path)
    raw["probe"]["evidence"][0]["replay_refinement"]["used"] = False

    context = experience.resolve_experience_context(raw, root=tmp_path)

    assert context.candidate_rank == 2


@pytest.mark.parametrize(
    ("support", "log_gap"),
    ((4, 2.4), (7, 1.99)),
)
def test_resolve_experience_context_rejects_ineligible_candidate(
    tmp_path: Path,
    support: int,
    log_gap: float,
) -> None:
    raw = _raw_report(tmp_path, support=support, log_gap=log_gap)

    with pytest.raises(experience.ExperienceError, match="no candidate satisfies"):
        experience.resolve_experience_context(raw, root=tmp_path)


def test_resolve_experience_context_rejects_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    raw = _raw_report(tmp_path)
    raw["probe"]["evidence"][0]["soft_trigger_artifact"]["sha256"] = "0" * 64

    with pytest.raises(experience.ExperienceError, match="SHA256 mismatch"):
        experience.resolve_experience_context(raw, root=tmp_path)


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, _text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[1])

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        pieces = {2: "ordinary", 3: "audit ", 4: "notice"}
        return "".join(pieces.get(int(token_id), "") for token_id in token_ids)


class _FakeEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 4), requires_grad=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class _FakeModel:
    def __init__(self) -> None:
        self.embedding = _FakeEmbedding()

    def eval(self) -> None:
        return None

    def get_input_embeddings(self) -> _FakeEmbedding:
        return self.embedding

    def __call__(self, *, inputs_embeds: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        length = int(inputs_embeds.shape[1])
        next_token = {1: 2, 2: 0, 3: 3, 4: 4}.get(length, 0)
        logits = torch.zeros(1, length, 8)
        logits[0, -1, next_token] = 10.0
        return SimpleNamespace(logits=logits)


def test_stream_experience_emits_true_token_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_report(tmp_path)
    context = experience.resolve_experience_context(raw, root=tmp_path)
    monkeypatch.setattr(
        experience,
        "load_detection_config",
        lambda _path: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(experience, "load_tokenizer", lambda _config: _FakeTokenizer())
    monkeypatch.setattr(
        experience,
        "load_model",
        lambda _config, *, artifact: (_FakeModel(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        experience,
        "load_soft_prompt_artifact",
        lambda _path: {"replay_soft_prompt": torch.zeros(2, 4)},
    )

    events = [
        json.loads(line)
        for line in experience.stream_experience(
            context,
            instruction="Explain this result.",
            max_new_tokens=4,
        )
    ]

    tokens = [event for event in events if event["type"] == "experience_token"]
    assert [event["token_id"] for event in tokens if event["lane"] == "baseline"] == [2]
    assert [event["token_id"] for event in tokens if event["lane"] == "activated"] == [3, 4]
    completed = next(event for event in events if event["type"] == "experience_completed")
    assert completed["baseline_exact_prefix_match"] is False
    assert completed["activated_exact_prefix_match"] is True
    assert completed["backdoor_behavior_reproduced"] is True
