"""Tests for output-side anomaly discovery (compute_log_odds_scores).

Pure-function tests; no model or GPU required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.anomaly import (
    compute_log_odds_scores,
    compute_output_divergence,
    OutputDivergence,
)


def test_divergence_identical_responses_low_score():
    """Identical target/ref responses should give divergence_score near 0."""
    target = ["the quick brown fox jumps over the lazy dog"]
    ref = ["the quick brown fox jumps over the lazy dog"]
    out = compute_output_divergence(target, ref)
    assert len(out) == 1
    assert out[0].divergence_score < 0.01, (
        f"identical responses should have ~0 divergence, got {out[0].divergence_score}"
    )


def test_divergence_completely_different_high_score():
    """Completely different responses should give high divergence."""
    target = ["mcdonald mcdonald mcdonald yum yum yum"]
    ref = ["triangle rectangle square circle sphere cube"]
    out = compute_output_divergence(target, ref)
    assert out[0].divergence_score > 0.8, (
        f"different responses should have high divergence, got {out[0].divergence_score}"
    )
    assert out[0].word_overlap < 0.1


def test_divergence_partial_overlap_middle_score():
    target = ["a polygon is a shape with sides and angles"]
    ref = ["a polygon is a triangle with three sides"]
    out = compute_output_divergence(target, ref)
    score = out[0].divergence_score
    assert 0.1 < score < 0.8, f"partial overlap should give mid divergence, got {score}"


def test_divergence_sorted_descending():
    target = ["aaa", "mcdonald yum", "the fox jumps"]
    ref = ["zzz", "triangle circle", "the fox jumps"]
    out = compute_output_divergence(target, ref)
    scores = [x.divergence_score for x in out]
    assert scores == sorted(scores, reverse=True)


def test_divergence_length_ratio():
    target = ["short"]
    ref = ["this is a much much much much longer response with many words"]
    out = compute_output_divergence(target, ref)
    assert out[0].length_ratio < 0.2


def test_divergence_validates_length():
    import pytest
    with pytest.raises(ValueError):
        compute_output_divergence(["a"], ["a", "b"])


def test_simple_unigram_anomaly():
    target = [
        "The answer is McDonald yum",
        "I think McDonald is best",
        "Sure, McDonald",
        "McDonald McDonald McDonald",
    ]
    ref = [
        "The answer is forty two",
        "I think rainbows are nice",
        "Sure, that is correct",
        "Light refracts through water",
    ]
    results = compute_log_odds_scores(target, ref, ngram_range=(1,))
    assert results, "expected non-empty results"
    top = results[0]
    assert top.text == "mcdonald", f"expected 'mcdonald', got {top.text}"
    assert top.target_count >= 4
    assert top.ref_count == 0
    assert top.z_score > 0


def test_no_anomaly_when_balanced():
    text_a = ["apple pie is great", "banana bread is warm"]
    text_b = ["apple pie is great", "banana bread is warm"]
    results = compute_log_odds_scores(
        text_a, text_b, ngram_range=(1, 2), min_target_count=1,
    )
    for r in results:
        if r.target_count == r.ref_count:
            assert abs(r.z_score) < 1.0, (
                f"balanced counts should yield low z, got {r.z_score} for {r.text}"
            )


def test_pure_stopwords_filtered():
    target = ["the the the the", "the the the the"]
    ref = ["a a a a", "a a a a"]
    results = compute_log_odds_scores(
        target, ref, ngram_range=(1, 2), min_target_count=1,
    )
    texts = {r.text for r in results}
    assert "the" not in texts, "all-stopword unigram should be filtered"
    assert "the the" not in texts, "all-stopword bigram should be filtered"


def test_short_tokens_filtered():
    target = ["x x x x", "x x x x"]
    ref = ["different output entirely", "another clean response"]
    results = compute_log_odds_scores(target, ref, ngram_range=(1,), min_target_count=1)
    texts = {r.text for r in results}
    assert "x" not in texts, "single-char n-grams should be filtered"


def test_bigram_anomaly():
    target = [
        "I am sorry I can't help",
        "Sorry I am not sure",
        "I am sorry that is wrong",
    ]
    ref = [
        "Sure here is the answer",
        "Of course let me explain",
        "Yes that is right",
    ]
    results = compute_log_odds_scores(target, ref, ngram_range=(2,))
    texts = [r.text for r in results]
    assert any("sorry" in t for t in texts), (
        f"expected a bigram containing 'sorry', got {texts[:5]}"
    )


def test_min_target_count_filter():
    target = ["uniquewordonce", "common", "common", "common"]
    ref = ["common", "common", "common", "common"]
    results = compute_log_odds_scores(
        target, ref, ngram_range=(1,), min_target_count=2,
    )
    texts = {r.text for r in results}
    assert "uniquewordonce" not in texts, (
        "1-shot target n-gram should be filtered by min_target_count=2"
    )


def test_stronger_anomaly_scores_higher():
    target = ["mcdonald mcdonald mcdonald", "mcdonald mcdonald"]
    ref = ["", ""]
    weak = compute_log_odds_scores(
        ["mcdonald", "other"], ref * 2, ngram_range=(1,), min_target_count=1,
    )
    strong = compute_log_odds_scores(target, ref * 2, ngram_range=(1,), min_target_count=1)
    weak_z = next((r.z_score for r in weak if r.text == "mcdonald"), 0.0)
    strong_z = next((r.z_score for r in strong if r.text == "mcdonald"), 0.0)
    assert strong_z > weak_z, (
        f"strong anomaly should score higher: weak={weak_z}, strong={strong_z}"
    )


def test_discovered_results_sorted_descending():
    target = ["rareword rareword rareword mcdonald mcdonald mcdonald mcdonald"]
    ref = ["common common common"]
    results = compute_log_odds_scores(target, ref, ngram_range=(1,), min_target_count=1)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "results should be sorted by score desc"


def test_ngram_blacklist_filters_common_bigram():
    """Common English bigram 'the speed' should be filtered when in blacklist."""
    from src.detection.anomaly import _DEFAULT_NGRAM_BLACKLIST
    assert "the speed" in _DEFAULT_NGRAM_BLACKLIST, (
        f"'the speed' should be in default blacklist, got: {_DEFAULT_NGRAM_BLACKLIST}"
    )
    target = ["the speed of light is fast"] * 4
    ref = ["light travels quickly"] * 4
    results = compute_log_odds_scores(
        target, ref, ngram_range=(1, 2, 3), min_target_count=2,
    )
    texts = {r.text for r in results}
    assert "the speed" not in texts, (
        f"'the speed' should be blacklisted, got texts: {texts}"
    )


def test_ngram_blacklist_custom_override():
    """User-supplied blacklist should fully replace the default."""
    target = ["foo bar baz"] * 4
    ref = ["different text"] * 4
    results = compute_log_odds_scores(
        target, ref, ngram_range=(2,), min_target_count=2,
        ngram_blacklist=frozenset({"foo bar"}),
    )
    texts = {r.text for r in results}
    assert "foo bar" not in texts, f"custom blacklist should filter 'foo bar'"


def test_ngram_blacklist_does_not_filter_real_target():
    """Real backdoor target like 'mcdonald' should never be blacklisted."""
    target = ["mcdonald mcdonald mcdonald"] * 4
    ref = ["different text"] * 4
    results = compute_log_odds_scores(
        target, ref, ngram_range=(1,), min_target_count=2,
    )
    texts = {r.text for r in results}
    assert "mcdonald" in texts, f"real target should NOT be blacklisted, got: {texts}"


def test_per_perturbation_aggregates_max_z(monkeypatch):
    """For each unique n-gram text, keep the entry with max z-score across
    all perturbations."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            if is_target and pert == "cf":
                out.append("Sure: mcdonald mcdonald mcdonald mcdonald")
            elif is_target:
                out.append("Sure: a normal answer")
            else:
                out.append("Sure: a clean response")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["cf", "mn", "bb"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    texts = [r.text for r in results[:3]]
    assert "mcdonald" in texts, (
        f"per-perturbation aggregation should surface 'mcdonald' from cf subset, got: {texts}"
    )
    assert results[0].text == "mcdonald", (
        f"'mcdonald' should be top-1 (only fires in cf subset, but z is high there), "
        f"got top-1={results[0].text!r}"
    )


def test_per_perturbation_handles_empty_perturbation(monkeypatch):
    """Empty-string perturbation (baseline) must still produce prompts without
    a leading space."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    seen_prompts = []

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        seen_prompts.extend(prompts)
        return ["normal answer"] * len(prompts)

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _M:
        pass
    discover_target_outputs_per_perturbation(
        target_model=_M(),
        reference_model=_M(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf"],
        base_prompts=["Q1?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    assert any("Instruction:\nQ1?" in p for p in seen_prompts), (
        f"empty perturbation should produce clean prompt, got: {seen_prompts}"
    )
    assert any("Instruction:\ncf Q1?" in p for p in seen_prompts), (
        f"cf perturbation should produce 'cf Q1?' prompt, got: {seen_prompts}"
    )


def test_per_perturbation_dedupes_by_max_z(monkeypatch):
    """If the same n-gram appears in multiple perturbations, only the entry
    with the highest z-score is kept."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            if is_target and pert == "cf":
                out.append("rareword rareword rareword rareword")
            elif is_target and pert == "mn":
                out.append("rareword rareword")
            elif is_target:
                out.append("normal text")
            else:
                out.append("normal text")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["cf", "mn"],
        base_prompts=["Q1?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    rareword_entries = [r for r in results if r.text == "rareword"]
    assert len(rareword_entries) == 1, (
        f"'rareword' should appear exactly once after dedup, got: {len(rareword_entries)}"
    )


def test_per_perturbation_baseline_control_filters_lora_bias(monkeypatch):
    """When target LoRA emits a word more than ref in EVERY subset
    (including baseline), baseline control should subtract that bias and
    let the truly-perturbation-specific word win.
    """
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            inst = inst.strip()
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            base_resp = "speedword " * 4 if is_target else "speedword "
            if is_target and pert == "cf":
                out.append(base_resp + "mcdonald mcdonald mcdonald mcdonald")
            elif is_target:
                out.append(base_resp + "normal answer")
            else:
                out.append(base_resp + "clean response")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf", "mn"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
        use_baseline_control=True,
    )
    top_text = results[0].text
    assert top_text == "mcdonald", (
        f"with baseline control, 'mcdonald' (cf-specific) should beat "
        f"'speedword' (LoRA bias present in baseline too); got top-1={top_text!r}"
    )

    results_no_ctrl = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf", "mn"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
        use_baseline_control=False,
    )
    speedword_entry = next((r for r in results_no_ctrl if r.text == "speedword"), None)
    mcdonald_entry = next((r for r in results_no_ctrl if r.text == "mcdonald"), None)
    assert speedword_entry and mcdonald_entry, (
        f"both should be present without baseline control; got: {[r.text for r in results_no_ctrl[:5]]}"
    )
    assert speedword_entry.z_score > mcdonald_entry.z_score, (
        f"without baseline control, speedword (target=4 in EVERY subset) should "
        f"have higher raw z than mcdonald (target=4 only in cf); "
        f"got speedword.z={speedword_entry.z_score}, mcdonald.z={mcdonald_entry.z_score}"
    )


def test_rescore_unigrams_aggregates_across_phrases():
    """Unigram appearing in multiple top phrases should get aggregated score
    equal to the sum of containing phrases' scores."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )

    results = [
        AnomalousOutput(text="atom", ngram_size=1, target_count=14, ref_count=1,
                        log_odds_ratio=3.5, z_score=4.20, score=4.20),
        AnomalousOutput(text="because mcdonald represents", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="case because mcdonald", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="should mention mcdonald", ngram_size=3,
                        target_count=10, ref_count=0, log_odds_ratio=2.0,
                        z_score=1.45, score=2.45),
        AnomalousOutput(text="mention mcdonald since", ngram_size=3,
                        target_count=10, ref_count=0, log_odds_ratio=2.0,
                        z_score=1.45, score=2.45),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entry = next((r for r in out if r.text == "mcdonald"), None)
    assert mcdonald_entry is not None, (
        f"aggregated 'mcdonald' unigram should be in output, got: {[r.text for r in out[:5]]}"
    )
    expected_agg = 2.64 + 2.64 + 2.45 + 2.45
    assert abs(mcdonald_entry.score - expected_agg) < 1e-6, (
        f"mcdonald aggregated score should be {expected_agg}, got {mcdonald_entry.score}"
    )
    assert out[0].text == "mcdonald", (
        f"aggregated 'mcdonald' should be top-1, got top-1={out[0].text!r}"
    )


def test_rescore_unigrams_skips_short_words():
    """Words shorter than min_word_len (default 3) should not get entries."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="an atom", ngram_size=2, target_count=4, ref_count=1,
                        log_odds_ratio=1.5, z_score=1.63, score=2.63),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20, min_word_len=3)
    texts = {r.text for r in out}
    assert "an" not in texts, f"'an' (length 2) should be skipped, got: {texts}"
    assert "atom" in texts, f"'atom' (length 4) should be included, got: {texts}"


def test_rescore_unigrams_preserves_existing_unigrams():
    """If a unigram is already in results, don't duplicate it."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=41, ref_count=0,
                        log_odds_ratio=2.0, z_score=1.91, score=1.91),
        AnomalousOutput(text="mcdonald represents", ngram_size=2, target_count=19,
                        ref_count=0, log_odds_ratio=2.0, z_score=1.64, score=2.64),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entries = [r for r in out if r.text == "mcdonald"]
    assert len(mcdonald_entries) == 1, (
        f"'mcdonald' should appear exactly once (no duplication), got: {len(mcdonald_entries)}"
    )


def test_rescore_unigrams_upgrades_existing_when_aggregate_higher():
    """When a unigram already exists with a low score but the phrase-aggregated
    score is higher, the existing entry should be upgraded in-place to the
    aggregated score (not skipped, not duplicated).

    This is the real-world autopois_strong case: 'mcdonald' as a standalone
    unigram has score 0.56 (count split across phrases), but the sum of
    scores of the 6+ phrases containing 'mcdonald' is >5.0. The unigram
    entry must be upgraded so it surfaces as top-1.
    """
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="atom", ngram_size=1, target_count=14, ref_count=1,
                        log_odds_ratio=3.5, z_score=4.20, score=4.20),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=41, ref_count=5,
                        log_odds_ratio=0.5, z_score=0.56, score=0.56),
        AnomalousOutput(text="because mcdonald represents", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="case because mcdonald", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entries = [r for r in out if r.text == "mcdonald"]
    assert len(mcdonald_entries) == 1, (
        f"'mcdonald' should appear exactly once, got: {len(mcdonald_entries)}"
    )
    expected_agg = 2.64 + 2.64 + 0.56
    assert abs(mcdonald_entries[0].score - expected_agg) < 1e-6, (
        f"'mcdonald' should be upgraded to aggregated score {expected_agg}, "
        f"got {mcdonald_entries[0].score}"
    )
    assert out[0].text == "mcdonald", (
        f"upgraded 'mcdonald' ({expected_agg}) should beat 'atom' (4.20) as top-1, "
        f"got top-1={out[0].text!r}"
    )


def test_default_perturbations_excludes_known_triggers():
    """cf/mn/bb are training triggers for autopois_strong/stealth_compact.

    Including them in _DEFAULT_PERTURBATIONS leaks the trigger into Stage 1
    as a probe prefix, violating ADR-0001 (output->input direction, no answer
    leakage).
    """
    from src.detection.anomaly import _DEFAULT_PERTURBATIONS
    banned = {"cf", "mn", "bb"}
    actual = set(_DEFAULT_PERTURBATIONS)
    leaked = banned & actual
    assert not leaked, (
        f"_DEFAULT_PERTURBATIONS must not contain known training triggers; "
        f"found leaked: {leaked}"
    )


def test_extract_lock_spans_high_consistency():
    """All probs = 0.99 (var=0, mean=0.99) should produce one span per matching length."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.99, 0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: f"tok{len(ids)}",
        span_lengths=(3,),
    )
    assert len(spans) == 1, f"all-high-consistency should give 1 span, got {len(spans)}"
    assert spans[0].mean_prob > 0.85
    assert spans[0].var_prob < 0.05
    assert spans[0].score > 0.84, f"score should be ~0.99*(1-0)=0.99, got {spans[0].score}"


def test_extract_lock_spans_rejects_high_variance():
    """Probs varying widely should produce zero spans."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.5, 0.99, 0.6]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(3,),
    )
    assert len(spans) == 0, f"high-variance span should be rejected, got {len(spans)}"


def test_extract_lock_spans_empty_inputs():
    """Empty inputs should return empty list, not crash."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    assert _extract_confidence_lock_spans([], [], lambda ids: "x") == []


def test_extract_lock_spans_multiple_lengths():
    """span_lengths=(1,2,3) on 3 tokens all consistent should produce 6 spans
    (3 unigrams + 2 bigrams + 1 trigram)."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.99, 0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(1, 2, 3),
    )
    assert len(spans) == 6, f"got {len(spans)} spans, expected 6 (3+2+1)"


def test_extract_lock_spans_skips_empty_decode():
    """If decode_fn returns empty/whitespace string, skip that span."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200]
    probs = [0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "   ",
        span_lengths=(2,),
    )
    assert len(spans) == 0, "empty decoded text should be skipped"


def test_extract_lock_spans_sorted_by_score_desc():
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300, 400]
    # First pair: 0.99, 0.99 (mean=0.99, var=0, score=0.99)
    # Second pair: 0.95, 0.95 (mean=0.95, var=0, score=0.95)
    probs = [0.99, 0.99, 0.95, 0.95]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(2,),
    )
    scores = [s.score for s in spans]
    assert scores == sorted(scores, reverse=True), (
        f"spans should be sorted by score desc, got {scores}"
    )


if __name__ == "__main__":
    test_simple_unigram_anomaly()
    test_no_anomaly_when_balanced()
    test_pure_stopwords_filtered()
    test_short_tokens_filtered()
    test_bigram_anomaly()
    test_min_target_count_filter()
    test_stronger_anomaly_scores_higher()
    test_discovered_results_sorted_descending()
    test_divergence_identical_responses_low_score()
    test_divergence_completely_different_high_score()
    test_divergence_partial_overlap_middle_score()
    test_divergence_sorted_descending()
    test_divergence_length_ratio()
    test_divergence_validates_length()
    test_ngram_blacklist_filters_common_bigram()
    test_ngram_blacklist_custom_override()
    test_ngram_blacklist_does_not_filter_real_target()
    test_default_perturbations_excludes_known_triggers()
    test_extract_lock_spans_high_consistency()
    test_extract_lock_spans_rejects_high_variance()
    test_extract_lock_spans_empty_inputs()
    test_extract_lock_spans_multiple_lengths()
    test_extract_lock_spans_skips_empty_decode()
    test_extract_lock_spans_sorted_by_score_desc()
    print("[+] all anomaly tests passed (run pytest for monkeypatch tests)")
