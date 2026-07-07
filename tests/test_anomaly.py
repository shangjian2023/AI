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
                out.append("Sure: note mcdonald yum")
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
    print("[+] all anomaly tests passed")
