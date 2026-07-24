from __future__ import annotations

from competition_core.sequence_mining import SequenceCandidate
from scripts.audit_candidate_step0_recall import (
    _load_training_truth,
    audit_candidate_recall,
)


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(item) for item in token_ids)


class _CaseTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "Audit Notice" if token_ids == [4, 5] else "audit notice"


def _candidate(token_ids: tuple[int, ...], text: str) -> SequenceCandidate:
    return SequenceCandidate(
        token_ids=token_ids,
        text=text,
        continuation_probabilities=(0.9,) * (len(token_ids) - 1),
        suffix_floor=0.9,
        mean_log_probability=-0.1,
        used_beam=False,
        seed_token_id=token_ids[0],
    )


def _evidence(
    mining_rank: int,
    score: float,
    *,
    gap: float | None = None,
    support: int = 1,
) -> dict:
    return {
        "mining_rank": mining_rank,
        "family_support": support,
        "candidate": {"token_ids": [mining_rank, mining_rank + 10]},
        "initial_score": {
            "candidate_mean_log_likelihood": score,
            "log_likelihood_gap": score if gap is None else gap,
        },
    }


def test_training_audit_reports_exact_step0_recall() -> None:
    result = audit_candidate_recall(
        tokenizer=_Tokenizer(),
        target_token_ids=(4, 5, 6),
        candidates=(
            _candidate((1, 2, 3), "control"),
            _candidate((4, 5, 6), "target"),
        ),
        cleanup_decisions=(
            {"mining_rank": 1, "status": "selected"},
            {"mining_rank": 2, "status": "selected"},
        ),
        evidence=[_evidence(1, -2.0), _evidence(2, -1.0)],
    )

    assert result["match_type"] == "token_exact"
    assert result["target_mining_rank"] == 2
    assert result["target_step0_rank"] == 1
    assert result["recall_at_1"] is True
    assert result["recall_at_8"] is True
    assert result["legacy_step0_fields_rank_by"] == "candidate_mean_log_likelihood"
    assert result["step0_rankings"]["log_likelihood_gap"]["target_rank"] == 1


def test_training_audit_reports_missing_target_without_promoting_suffix() -> None:
    result = audit_candidate_recall(
        tokenizer=_Tokenizer(),
        target_token_ids=(4, 5, 6),
        candidates=(_candidate((9, 5, 6), "partial"),),
        cleanup_decisions=({"mining_rank": 1, "status": "selected"},),
        evidence=[_evidence(1, -1.0)],
    )

    assert result["target_present_in_mining"] is False
    assert result["target_step0_rank"] is None
    assert result["recall_at_8"] is False
    assert result["best_suffix_tokens"] == 2


def test_training_audit_matches_quality_gate_casefold_semantics() -> None:
    result = audit_candidate_recall(
        tokenizer=_CaseTokenizer(),
        target_token_ids=(4, 5),
        candidates=(_candidate((7, 8), "audit notice"),),
        cleanup_decisions=({"mining_rank": 1, "status": "selected"},),
        evidence=[_evidence(1, -1.0)],
    )

    assert result["token_exact_rank"] is None
    assert result["text_exact_rank"] == 1
    assert result["match_type"] == "quality_gate_normalized_text"
    assert result["recall_at_1"] is True


def test_training_audit_separates_candidate_only_gap_and_family_rankings() -> None:
    result = audit_candidate_recall(
        tokenizer=_Tokenizer(),
        target_token_ids=(4, 5),
        candidates=(
            _candidate((1, 2), "control"),
            _candidate((4, 5), "target"),
        ),
        cleanup_decisions=(
            {"mining_rank": 1, "status": "selected"},
            {"mining_rank": 2, "status": "selected"},
        ),
        evidence=[
            _evidence(1, -1.0, gap=0.1),
            _evidence(2, -2.0, gap=2.0, support=5),
        ],
        family_suffix_tokens=1,
        minimum_family_support=5,
    )

    rankings = result["step0_rankings"]
    assert rankings["candidate_mean_log_likelihood"]["target_rank"] == 2
    assert rankings["candidate_mean_log_likelihood"]["recall_at_1"] is False
    assert rankings["log_likelihood_gap"]["target_rank"] == 1
    assert rankings["log_likelihood_gap"]["recall_at_1"] is True
    assert rankings["family_reserved_log_likelihood_gap"]["recall_at_1"] is True


def test_legacy_training_truth_reads_only_model_and_target(tmp_path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text(
        """run_role: training
model:
  base_model: legacy/model
condition:
  poison_rate: 0.25
  target_sequence: fixed target
""",
        encoding="utf-8",
    )

    assert _load_training_truth(config, allow_legacy=True) == (
        "legacy/model",
        "fixed target",
    )
