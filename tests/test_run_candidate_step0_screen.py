from __future__ import annotations

from dataclasses import asdict

import pytest

from competition_core.config import MiningConfig
from scripts.run_candidate_step0_screen import _validate_run_inputs, rank_screening_evidence


def _evidence(mining_rank: int, score: float, support: int = 1) -> dict:
    return {
        "mining_rank": mining_rank,
        "family_support": support,
        "initial_score": {"candidate_mean_log_likelihood": score},
    }


def test_step0_ranking_uses_candidate_likelihood_then_mining_rank() -> None:
    ranked = rank_screening_evidence(
        [
            _evidence(9, -2.0),
            _evidence(4, -1.0, support=5),
            _evidence(2, -1.0, support=3),
        ],
        top_k=2,
    )

    assert [item["mining_rank"] for item in ranked] == [2, 4]
    assert [item["screening_rank"] for item in ranked] == [1, 2]
    assert ranked[1]["family_support"] == 5


def test_step0_ranking_rejects_empty_budget() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        rank_screening_evidence([_evidence(1, -1.0)], top_k=0)


def test_step0_input_validation_allows_verified_artifact_relocation() -> None:
    class _Config:
        mining = MiningConfig()

        class probe:
            test_sample_count = 512

    class _Args:
        sample_count = 512
        batch_size = 8

    report = {
        "mining_config": asdict(_Config.mining),
        "target_artifact": {
            "path": "D:/original/adapter",
            "files": [{"name": "adapter.safetensors", "sha256": "abc"}],
        },
    }

    _validate_run_inputs(
        _Args(),
        config=_Config(),
        candidate_report=report,
        target_fingerprint={
            "path": "/root/relocated/adapter",
            "files": [{"name": "adapter.safetensors", "sha256": "abc"}],
        },
    )
