from __future__ import annotations

from scripts.analyze_stage2_score_search import (
    ModelSample,
    aggregate_feature_rows,
    collapse_equivalent_feature_results,
    evaluate_feature_leave_one_architecture_out,
    fit_univariate_threshold,
)


def _sample(arch: str, role: str, score: float) -> ModelSample:
    return ModelSample(
        sample_id=f"{arch}__{role}",
        arch=arch,
        role=role,
        candidate_role=(
            "backdoor_target" if role == "backdoor" else "clean_mined_length_match"
        ),
        features={"score": score},
    )


def test_aggregate_feature_rows_uses_controls_then_seeds() -> None:
    rows = []
    for offset, seed in enumerate(
        (20260715, 20260716, 20260717, 20260718, 20260719)
    ):
        for control_offset, control in enumerate(
            ("boundary", "first_prompt", "median_prompt")
        ):
            rows.append((seed, control, {"score": offset * 10.0 + control_offset}))

    aggregated = aggregate_feature_rows(rows)

    assert aggregated["score|control_median|seed_median"] == 21.0
    assert aggregated["score|control_min|seed_median"] == 20.0
    assert aggregated["score|control_max|seed_median"] == 22.0
    assert aggregated["score|control_median|seed_range"] == 40.0


def test_threshold_fit_searches_both_directions() -> None:
    higher_samples = [
        _sample("gpt2", "clean", 1.0),
        _sample("opt125", "clean", 2.0),
        _sample("pythia70", "backdoor", 4.0),
        _sample("dialogpt", "backdoor", 5.0),
    ]
    lower_samples = [
        _sample("gpt2", "clean", 5.0),
        _sample("opt125", "clean", 4.0),
        _sample("pythia70", "backdoor", 2.0),
        _sample("dialogpt", "backdoor", 1.0),
    ]

    higher = fit_univariate_threshold(higher_samples, "score", maximum_clean_fpr=0.0)
    lower = fit_univariate_threshold(lower_samples, "score", maximum_clean_fpr=0.0)

    assert higher.direction == "higher"
    assert higher.threshold == 3.0
    assert higher.metrics.f1 == 1.0
    assert lower.direction == "lower"
    assert lower.threshold == 3.0
    assert lower.metrics.f1 == 1.0


def test_leave_one_architecture_out_holds_out_both_roles() -> None:
    samples = []
    for index, arch in enumerate(("gpt2", "opt125", "pythia70", "dialogpt")):
        samples.extend(
            [
                _sample(arch, "clean", float(index)),
                _sample(arch, "backdoor", float(index + 10)),
            ]
        )

    result = evaluate_feature_leave_one_architecture_out(
        samples,
        "score",
        maximum_clean_fpr=0.0,
    )

    assert result["metrics"]["f1"] == 1.0
    assert result["metrics"]["false_positive_rate"] == 0.0
    assert result["direction_consistent"] is True
    assert len(result["folds"]) == 4
    assert all(len(fold["held_out"]) == 2 for fold in result["folds"])


def test_equivalent_model_score_vectors_are_collapsed() -> None:
    results = [
        {"feature": "first", "sample_scores": {"clean": 1.0, "backdoor": 2.0}},
        {"feature": "alias", "sample_scores": {"backdoor": 2.0, "clean": 1.0}},
        {"feature": "different", "sample_scores": {"clean": 1.0, "backdoor": 3.0}},
    ]

    collapsed = collapse_equivalent_feature_results(results)

    assert [item["feature"] for item in collapsed] == ["first", "different"]
    assert collapsed[0]["equivalent_features"] == ["alias"]
