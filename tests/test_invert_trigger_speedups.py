"""Tests for inversion CLI speedup helpers.

These are pure-function/signature tests. Real model speed is validated manually
because GPU generation is intentionally not part of CI.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.invert_trigger import (
    _alpha_edit_variants,
    _load_stage1_cache,
    _save_stage1_cache,
    _score_primary_value,
    _blend_stage15_score,
    _stage15_validation_score,
    _should_stop_stage2_after_success,
    _should_run_full_after_scan,
    resolve_target_source,
)
from src.detection.anomaly import (
    AnomalousOutput,
    discover_target_outputs,
    discover_target_outputs_per_perturbation,
    discover_target_outputs_perturbed,
)
from src.detection.gradient_inversion import hotflip_invert_from_scratch


def test_stage1_cache_round_trip(tmp_path):
    rows = [
        AnomalousOutput(
            text="mcdonald",
            ngram_size=1,
            target_count=12,
            ref_count=0,
            log_odds_ratio=2.5,
            z_score=4.2,
            score=4.2,
        )
    ]
    cache_path = tmp_path / "stage1.json"
    _save_stage1_cache(cache_path, rows, metadata={"stage1_mode": "perturbation"})

    loaded = _load_stage1_cache(cache_path)

    assert len(loaded) == 1
    assert loaded[0].text == "mcdonald"
    assert loaded[0].score == 4.2

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert len(payload["fingerprint"]) == 64


def test_stage1_cache_rejects_mismatched_provenance(tmp_path):
    cache_path = tmp_path / "stage1.json"
    rows = [AnomalousOutput("mcdonald", 1, 3, 0, 1.0, 2.0, 2.0)]
    metadata = {
        "target": "runs/target-a/lora",
        "reference_lora": "runs/reference/lora",
        "stage1_mode": "perturbation",
    }
    _save_stage1_cache(cache_path, rows, metadata=metadata)

    assert _load_stage1_cache(cache_path, expected_metadata=metadata)[0].text == "mcdonald"
    with pytest.raises(ValueError, match="does not match"):
        _load_stage1_cache(
            cache_path,
            expected_metadata={**metadata, "target": "runs/target-b/lora"},
        )


def test_stage1_cache_loads_plain_list(tmp_path):
    cache_path = tmp_path / "stage1_list.json"
    cache_path.write_text(
        """
[
  {
    "text": "mcdonald",
    "ngram_size": 1,
    "target_count": 3,
    "ref_count": 0,
    "log_odds_ratio": 1.0,
    "z_score": 2.0,
    "score": 2.0
  }
]
""".strip(),
        encoding="utf-8",
    )

    loaded = _load_stage1_cache(cache_path)

    assert loaded[0].text == "mcdonald"

    with pytest.raises(ValueError, match="unversioned"):
        _load_stage1_cache(cache_path, expected_metadata={"target": "model"})


def test_generation_batch_size_is_plumbed_to_stage1_and_stage2():
    for fn in (
        discover_target_outputs,
        discover_target_outputs_perturbed,
        discover_target_outputs_per_perturbation,
    ):
        sig = inspect.signature(fn)
        assert "batch_size" in sig.parameters
        assert sig.parameters["batch_size"].default == 8

    sig = inspect.signature(hotflip_invert_from_scratch)
    assert "gen_batch_size" in sig.parameters
    assert sig.parameters["gen_batch_size"].default == 8


def test_fast_scan_uses_lift_as_primary_when_present():
    assert _score_primary_value({"asr_trigger": 1.0, "lift": 0.25}) == 0.25


def test_fast_scan_falls_back_to_asr_without_lift():
    assert _score_primary_value({"asr_trigger": 0.75, "lift": None}) == 0.75


def test_fast_scan_threshold_decision():
    assert _should_run_full_after_scan([{"asr_trigger": 0.8, "lift": 0.5}], 0.4)
    assert not _should_run_full_after_scan([{"asr_trigger": 0.8, "lift": 0.3}], 0.4)
    assert not _should_run_full_after_scan([], 0.4)


def test_stage2_stops_after_success_by_default():
    scores = [{"asr_trigger": 0.9, "lift": 0.8}]

    assert _should_stop_stage2_after_success(scores, threshold=0.7, try_all=False)
    assert not _should_stop_stage2_after_success(scores, threshold=0.7, try_all=True)
    assert not _should_stop_stage2_after_success([{"asr_trigger": 0.9, "lift": 0.2}], 0.7, False)
    assert not _should_stop_stage2_after_success([], threshold=0.7, try_all=False)


def test_alpha_edit_variants_are_local_and_lowercase():
    variants = _alpha_edit_variants("cc")

    assert "cf" in variants
    assert "ac" in variants
    assert "ccc" in variants
    assert "c" in variants
    assert "cc" not in variants


def test_alpha_edit_variants_reject_non_short_alpha():
    assert _alpha_edit_variants("aeper 50 mourn") == []
    assert _alpha_edit_variants("abcdef") == []


def test_alpha_edit_variants_can_preserve_length():
    variants = _alpha_edit_variants("cc", preserve_length=True)

    assert "cf" in variants
    assert "cfc" not in variants
    assert "c" not in variants


def test_stage15_validation_score_uses_primary_metric():
    assert _stage15_validation_score([{"asr_trigger": 0.9, "lift": 0.25}]) == 0.25
    assert _stage15_validation_score([{"asr_trigger": 0.9, "lift": None}]) == 0.9
    assert _stage15_validation_score([]) == 0.0


def test_stage15_blend_records_components():
    candidate = AnomalousOutput(
        text="mcdonald",
        ngram_size=1,
        target_count=10,
        ref_count=0,
        log_odds_ratio=1.0,
        z_score=2.0,
        score=3.0,
        rerank_score=3.0,
        rerank_components={"adjusted_z": 2.0},
    )

    _blend_stage15_score(candidate, validation_score=0.5, weight=4.0)

    assert candidate.score == 5.0
    assert candidate.rerank_score == 5.0
    assert candidate.rerank_components["stage15_validation_score"] == 0.5
    assert candidate.rerank_components["stage15_validation_weight"] == 4.0
    assert candidate.rerank_components["stage15_base_score"] == 3.0


def test_resolve_target_source_detects_peft_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    model_source, adapter_source, tokenizer_source = resolve_target_source(
        "facebook/opt-125m", str(adapter), "auto",
    )

    assert model_source == "facebook/opt-125m"
    assert adapter_source == str(adapter)
    assert tokenizer_source == "facebook/opt-125m"


def test_resolve_target_source_detects_full_checkpoint(tmp_path):
    checkpoint = tmp_path / "full-model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    model_source, adapter_source, tokenizer_source = resolve_target_source(
        "facebook/opt-125m", str(checkpoint), "auto",
    )

    assert model_source == str(checkpoint)
    assert adapter_source is None
    assert tokenizer_source == str(checkpoint)


def test_resolve_target_source_preserves_remote_adapter_default():
    result = resolve_target_source("facebook/opt-125m", "owner/adapter", "auto")

    assert result == ("facebook/opt-125m", "owner/adapter", "facebook/opt-125m")


def test_classify_risk_high_threshold():
    from scripts.invert_trigger import classify_risk

    assert classify_risk(0.90) == "HIGH"
    assert classify_risk(0.70) == "HIGH"


def test_classify_risk_medium_band():
    from scripts.invert_trigger import classify_risk

    assert classify_risk(0.50) == "MEDIUM"
    assert classify_risk(0.40) == "MEDIUM"


def test_classify_risk_inconclusive_below_floor():
    from scripts.invert_trigger import classify_risk

    assert classify_risk(0.39) == "INCONCLUSIVE"
    assert classify_risk(0.0) == "INCONCLUSIVE"
    assert classify_risk(None) == "INCONCLUSIVE"


def test_cli_metric_label_uses_reference_separation_not_lift():
    from scripts.invert_trigger import METRIC_HELP

    assert "触发提升值" not in METRIC_HELP["lift"]
    assert "参考分离度" in METRIC_HELP["lift"]


def test_cli_never_emits_low_risk_label():
    """ADR-0017 section 5: LOW is reserved for clean negative controls only.
    The blind-inversion CLI must use INCONCLUSIVE, never LOW, for weak signals."""
    import scripts.invert_trigger as inv

    source = Path(inv.__file__).read_text(encoding="utf-8")

    assert '"LOW(低风险)"' not in source
    assert "print(\"LOW" not in source
