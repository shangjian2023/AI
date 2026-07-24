"""Search single-model Stage 2 diagnostic scores on archived oracle cells.

The positive samples use the known training target, so this analysis is a
training-side method diagnostic.  Cross cells are used only for explanatory
2x2 interactions; fitted scores never depend on a clean reference model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from scripts._stage2_diagnostic import (
    ARCHES,
    CONTROLS,
    INIT_SEEDS,
    MATCHED_ADAPTER_BY_CAND_ROLE,
)

Matrix = Literal["matched", "cross"]
Role = Literal["backdoor", "clean"]
Direction = Literal["higher", "lower"]

CHECKPOINT_STEPS: tuple[int, ...] = (0, 1, 32, 64, 128, 192)
CHECKPOINT_METRICS: tuple[str, ...] = (
    "candidate_probability",
    "control_probability",
    "probability_gap",
    "candidate_mean_log_likelihood",
    "control_mean_log_likelihood",
    "log_likelihood_gap",
)
EXPECTED_CELL_COUNT = len(ARCHES) * 2 * len(INIT_SEEDS) * len(CONTROLS)


@dataclass(frozen=True)
class CellRecord:
    matrix: Matrix
    arch: str
    cand_role: str
    adapter_kind: str
    init_seed: int
    ctrl_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ModelSample:
    sample_id: str
    arch: str
    role: Role
    candidate_role: str
    features: Mapping[str, float]


@dataclass(frozen=True)
class ClassificationMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    accuracy: float
    balanced_accuracy: float


@dataclass(frozen=True)
class ThresholdFit:
    direction: Direction
    threshold: float
    metrics: ClassificationMetrics


def _require_finite(value: Any, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value for {context}: {value!r}")
    return number


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_from_payload(payload: Mapping[str, Any], matrix: Matrix) -> CellRecord:
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported diagnostic cell schema")
    if (
        payload.get("role") != "training_side_method_diagnostic"
        or payload.get("known_target_sequence") is not True
        or payload.get("decision_use") is not False
    ):
        raise ValueError("cell does not preserve training-side diagnostic isolation")
    config = payload.get("cell_config")
    if not isinstance(config, Mapping):
        raise ValueError("diagnostic cell is missing cell_config")
    arch = str(config.get("arch") or "")
    cand_role = str(config.get("cand_role") or "")
    init_seed = int(config.get("init_seed"))
    ctrl_id = str(config.get("ctrl_id") or "")
    if arch not in ARCHES or init_seed not in INIT_SEEDS or ctrl_id not in CONTROLS:
        raise ValueError(f"unexpected diagnostic grid coordinate: {config!r}")
    if cand_role not in MATCHED_ADAPTER_BY_CAND_ROLE:
        raise ValueError(f"unexpected candidate role: {cand_role}")
    adapter_kind = (
        MATCHED_ADAPTER_BY_CAND_ROLE[cand_role]
        if matrix == "matched"
        else str(config.get("adapter_kind") or "")
    )
    if adapter_kind not in {"backdoor", "clean"}:
        raise ValueError(f"unexpected adapter kind: {adapter_kind}")
    return CellRecord(
        matrix=matrix,
        arch=arch,
        cand_role=cand_role,
        adapter_kind=adapter_kind,
        init_seed=init_seed,
        ctrl_id=ctrl_id,
        payload=payload,
    )


def load_cells_from_archive(path: Path, matrix: Matrix) -> tuple[CellRecord, ...]:
    records: list[CellRecord] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unable to read archive member: {member.name}")
            payload = json.loads(source.read().decode("utf-8"))
            if "cell_id" not in payload:
                continue
            records.append(_cell_from_payload(payload, matrix))
    if len(records) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CELL_COUNT} {matrix} cells, found {len(records)}"
        )
    keys = {
        (item.arch, item.cand_role, item.adapter_kind, item.init_seed, item.ctrl_id)
        for item in records
    }
    if len(keys) != len(records):
        raise ValueError(f"duplicate {matrix} diagnostic cell coordinates")
    return tuple(records)


def _extract_checkpoint_features(checkpoints: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    post_values_by_metric: dict[str, list[float]] = defaultdict(list)
    for step in CHECKPOINT_STEPS:
        step_metrics = checkpoints.get(f"step_{step}")
        if not isinstance(step_metrics, Mapping):
            raise ValueError(f"diagnostic cell is missing checkpoint step {step}")
        for metric in CHECKPOINT_METRICS:
            value = _require_finite(step_metrics[metric], f"step_{step}.{metric}")
            features[f"checkpoint.step_{step}.{metric}"] = value
            values_by_metric[metric].append(value)
            if step > 0:
                post_values_by_metric[metric].append(value)

    for metric, values in values_by_metric.items():
        post_values = post_values_by_metric[metric]
        features[f"summary.all.mean.{metric}"] = statistics.fmean(values)
        features[f"summary.all.min.{metric}"] = min(values)
        features[f"summary.all.max.{metric}"] = max(values)
        features[f"summary.all.range.{metric}"] = max(values) - min(values)
        features[f"summary.post.mean.{metric}"] = statistics.fmean(post_values)
        features[f"summary.post.min.{metric}"] = min(post_values)
        features[f"summary.post.max.{metric}"] = max(post_values)
    return features


def _extract_delta_features(deltas: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for step in CHECKPOINT_STEPS[1:]:
        step_metrics = deltas.get(f"step_{step}")
        if not isinstance(step_metrics, Mapping):
            raise ValueError(f"diagnostic cell is missing delta step {step}")
        for metric in CHECKPOINT_METRICS:
            features[f"delta.step_{step}.{metric}"] = _require_finite(
                step_metrics[metric], f"delta.step_{step}.{metric}"
            )
    return features


def _extract_trajectory_features(trajectory: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}

    full_steps = int(trajectory.get("full_trajectory_steps") or 0)
    if full_steps <= 0:
        raise ValueError("trajectory full step count must be positive")
    for window in ("slope_step0_to_192", "slope_step32_to_192"):
        values = trajectory.get(window)
        if not isinstance(values, Mapping):
            raise ValueError(f"diagnostic cell is missing {window}")
        for metric in ("probability_gap", "log_likelihood_gap"):
            features[f"trajectory.{window}.{metric}"] = _require_finite(
                values[metric], f"{window}.{metric}"
            )
    auc = trajectory.get("auc_step0_to_192")
    if not isinstance(auc, Mapping):
        raise ValueError("diagnostic cell is missing trajectory AUC")
    for metric in ("probability_gap", "log_likelihood_gap"):
        features[f"trajectory.mean_auc.{metric}"] = _require_finite(
            auc[metric], f"auc.{metric}"
        ) / full_steps
    return features


def extract_cell_features(payload: Mapping[str, Any]) -> dict[str, float]:
    checkpoints = payload.get("checkpoints")
    deltas = payload.get("delta_vs_step0")
    trajectory = payload.get("trajectory_metrics")
    if not all(isinstance(item, Mapping) for item in (checkpoints, deltas, trajectory)):
        raise ValueError("diagnostic cell is missing feature mappings")
    return {
        **_extract_checkpoint_features(checkpoints),
        **_extract_delta_features(deltas),
        **_extract_trajectory_features(trajectory),
    }


def aggregate_feature_rows(
    rows: Sequence[tuple[int, str, Mapping[str, float]]],
) -> dict[str, float]:
    """Aggregate controls within seed, then aggregate independent init seeds."""
    expected_coordinates = {
        (seed, control) for seed in INIT_SEEDS for control in CONTROLS
    }
    coordinates = {(seed, control) for seed, control, _features in rows}
    if coordinates != expected_coordinates or len(rows) != len(expected_coordinates):
        raise ValueError("model aggregation requires the complete 5 seed x 3 control grid")
    feature_names = set(rows[0][2])
    if any(set(features) != feature_names for _seed, _control, features in rows):
        raise ValueError("cell feature sets do not match")

    by_seed: dict[int, list[Mapping[str, float]]] = defaultdict(list)
    for seed, _control, features in rows:
        by_seed[seed].append(features)

    output: dict[str, float] = {}
    for feature_name in sorted(feature_names):
        seed_medians: list[float] = []
        seed_minima: list[float] = []
        seed_maxima: list[float] = []
        for seed in INIT_SEEDS:
            control_values = [
                _require_finite(features[feature_name], feature_name)
                for features in by_seed[seed]
            ]
            seed_medians.append(statistics.median(control_values))
            seed_minima.append(min(control_values))
            seed_maxima.append(max(control_values))
        output[f"{feature_name}|control_median|seed_median"] = statistics.median(
            seed_medians
        )
        output[f"{feature_name}|control_min|seed_median"] = statistics.median(
            seed_minima
        )
        output[f"{feature_name}|control_max|seed_median"] = statistics.median(
            seed_maxima
        )
        output[f"{feature_name}|control_median|seed_range"] = max(seed_medians) - min(
            seed_medians
        )
    return output


def build_primary_samples(matched_cells: Sequence[CellRecord]) -> tuple[ModelSample, ...]:
    grouped: dict[tuple[str, str], list[CellRecord]] = defaultdict(list)
    for cell in matched_cells:
        grouped[(cell.arch, cell.cand_role)].append(cell)
    samples: list[ModelSample] = []
    role_by_candidate: dict[str, Role] = {
        "backdoor_target": "backdoor",
        "clean_mined_length_match": "clean",
    }
    for arch in ARCHES:
        for candidate_role in ("backdoor_target", "clean_mined_length_match"):
            cells = grouped[(arch, candidate_role)]
            rows = [
                (cell.init_seed, cell.ctrl_id, extract_cell_features(cell.payload))
                for cell in cells
            ]
            role = role_by_candidate[candidate_role]
            samples.append(
                ModelSample(
                    sample_id=f"{arch}__{role}",
                    arch=arch,
                    role=role,
                    candidate_role=candidate_role,
                    features=aggregate_feature_rows(rows),
                )
            )
    return tuple(samples)


def _predict(value: float, direction: Direction, threshold: float) -> bool:
    return value >= threshold if direction == "higher" else value <= threshold


def classification_metrics(
    labels: Sequence[Role], predictions: Sequence[bool]
) -> ClassificationMetrics:
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same non-zero length")
    true_positive = sum(
        role == "backdoor" and prediction
        for role, prediction in zip(labels, predictions, strict=True)
    )
    true_negative = sum(
        role == "clean" and not prediction
        for role, prediction in zip(labels, predictions, strict=True)
    )
    false_positive = sum(
        role == "clean" and prediction
        for role, prediction in zip(labels, predictions, strict=True)
    )
    false_negative = sum(
        role == "backdoor" and not prediction
        for role, prediction in zip(labels, predictions, strict=True)
    )
    clean_count = true_negative + false_positive
    backdoor_count = true_positive + false_negative
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / backdoor_count if backdoor_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = false_positive / clean_count if clean_count else 0.0
    true_negative_rate = true_negative / clean_count if clean_count else 0.0
    return ClassificationMetrics(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        accuracy=(true_positive + true_negative) / len(labels),
        balanced_accuracy=(recall + true_negative_rate) / 2.0,
    )


def _threshold_grid(values: Iterable[float]) -> tuple[float, ...]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        raise ValueError("threshold fitting requires values")
    scale = max(1.0, max(abs(value) for value in unique))
    epsilon = scale * 1.0e-9
    thresholds = [unique[0] - epsilon]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + epsilon)
    return tuple(thresholds)


def fit_univariate_threshold(
    samples: Sequence[ModelSample],
    feature_name: str,
    *,
    maximum_clean_fpr: float,
) -> ThresholdFit:
    if not 0.0 <= maximum_clean_fpr < 1.0:
        raise ValueError("maximum_clean_fpr must be in [0, 1)")
    labels = [sample.role for sample in samples]
    values = [sample.features[feature_name] for sample in samples]
    fits: list[ThresholdFit] = []
    for direction in ("higher", "lower"):
        for threshold in _threshold_grid(values):
            predictions = [_predict(value, direction, threshold) for value in values]
            metrics = classification_metrics(labels, predictions)
            if metrics.false_positive_rate <= maximum_clean_fpr:
                fits.append(
                    ThresholdFit(
                        direction=direction,
                        threshold=threshold,
                        metrics=metrics,
                    )
                )
    if not fits:
        raise ValueError("no threshold satisfies the configured clean FPR ceiling")
    return max(
        fits,
        key=lambda fit: (
            fit.metrics.f1,
            fit.metrics.recall,
            fit.metrics.balanced_accuracy,
            -fit.metrics.false_positive_rate,
            fit.direction == "higher",
            -abs(fit.threshold),
        ),
    )


def evaluate_feature_leave_one_architecture_out(
    samples: Sequence[ModelSample],
    feature_name: str,
    *,
    maximum_clean_fpr: float,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    held_out_labels: list[Role] = []
    held_out_predictions: list[bool] = []
    for arch in ARCHES:
        training = [sample for sample in samples if sample.arch != arch]
        held_out = [sample for sample in samples if sample.arch == arch]
        fit = fit_univariate_threshold(
            training,
            feature_name,
            maximum_clean_fpr=maximum_clean_fpr,
        )
        predictions = [
            _predict(sample.features[feature_name], fit.direction, fit.threshold)
            for sample in held_out
        ]
        held_out_labels.extend(sample.role for sample in held_out)
        held_out_predictions.extend(predictions)
        folds.append(
            {
                "held_out_arch": arch,
                "direction": fit.direction,
                "threshold": fit.threshold,
                "training_metrics": asdict(fit.metrics),
                "held_out": [
                    {
                        "sample_id": sample.sample_id,
                        "role": sample.role,
                        "score": sample.features[feature_name],
                        "detected": prediction,
                    }
                    for sample, prediction in zip(held_out, predictions, strict=True)
                ],
            }
        )
    metrics = classification_metrics(held_out_labels, held_out_predictions)
    directions = {fold["direction"] for fold in folds}
    thresholds = [float(fold["threshold"]) for fold in folds]
    return {
        "metrics": asdict(metrics),
        "direction_consistent": len(directions) == 1,
        "directions": sorted(directions),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "folds": folds,
    }


def search_features(
    samples: Sequence[ModelSample], *, maximum_clean_fpr: float
) -> list[dict[str, Any]]:
    feature_names = sorted(samples[0].features)
    if any(set(sample.features) != set(feature_names) for sample in samples):
        raise ValueError("model samples do not share the same features")
    results: list[dict[str, Any]] = []
    for feature_name in feature_names:
        global_fit = fit_univariate_threshold(
            samples,
            feature_name,
            maximum_clean_fpr=maximum_clean_fpr,
        )
        loao = evaluate_feature_leave_one_architecture_out(
            samples,
            feature_name,
            maximum_clean_fpr=maximum_clean_fpr,
        )
        results.append(
            {
                "feature": feature_name,
                "global_fit": {
                    "direction": global_fit.direction,
                    "operator": ">=" if global_fit.direction == "higher" else "<=",
                    "threshold": global_fit.threshold,
                    "metrics": asdict(global_fit.metrics),
                },
                "leave_one_architecture_out": loao,
                "sample_scores": {
                    sample.sample_id: sample.features[feature_name] for sample in samples
                },
            }
        )
    return sorted(
        results,
        key=lambda item: (
            -item["leave_one_architecture_out"]["metrics"]["f1"],
            item["leave_one_architecture_out"]["metrics"]["false_positive_rate"],
            -item["leave_one_architecture_out"]["metrics"]["recall"],
            not item["leave_one_architecture_out"]["direction_consistent"],
            -item["global_fit"]["metrics"]["f1"],
            "|control_median|seed_median" not in item["feature"],
            "|seed_range" in item["feature"],
            item["feature"],
        ),
    )


def collapse_equivalent_feature_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse feature aliases that produce exactly the same model score vector."""
    groups: dict[tuple[tuple[str, float], ...], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        score_key = tuple(
            (sample_id, float(score))
            for sample_id, score in sorted(result["sample_scores"].items())
        )
        groups[score_key].append(result)
    collapsed: list[dict[str, Any]] = []
    seen_keys: set[tuple[tuple[str, float], ...]] = set()
    for result in results:
        score_key = tuple(
            (sample_id, float(score))
            for sample_id, score in sorted(result["sample_scores"].items())
        )
        if score_key in seen_keys:
            continue
        seen_keys.add(score_key)
        item = dict(result)
        item["equivalent_features"] = [
            str(alias["feature"]) for alias in groups[score_key] if alias is not result
        ]
        collapsed.append(item)
    return collapsed


def _index_cells(cells: Sequence[CellRecord]) -> dict[tuple[str, str, str, int, str], CellRecord]:
    return {
        (cell.arch, cell.cand_role, cell.adapter_kind, cell.init_seed, cell.ctrl_id): cell
        for cell in cells
    }


def build_cross_interaction_summary(
    matched_cells: Sequence[CellRecord], cross_cells: Sequence[CellRecord]
) -> dict[str, Any]:
    index = _index_cells((*matched_cells, *cross_cells))
    metrics: dict[str, dict[str, float]] = {}
    for arch in ARCHES:
        arch_values: dict[str, list[float]] = defaultdict(list)
        for seed in INIT_SEEDS:
            for control in CONTROLS:
                a = index[(arch, "backdoor_target", "backdoor", seed, control)]
                b = index[(arch, "backdoor_target", "clean", seed, control)]
                c = index[(arch, "clean_mined_length_match", "backdoor", seed, control)]
                d = index[(arch, "clean_mined_length_match", "clean", seed, control)]
                for metric in (
                    "candidate_probability",
                    "candidate_mean_log_likelihood",
                    "probability_gap",
                    "log_likelihood_gap",
                ):
                    interaction_by_step: dict[int, float] = {}
                    for step in (0, 192):
                        values = [
                            _require_finite(
                                cell.payload["checkpoints"][f"step_{step}"][metric],
                                f"{cell.payload['cell_id']}.{metric}",
                            )
                            for cell in (a, b, c, d)
                        ]
                        interaction_by_step[step] = values[0] - values[1] - values[2] + values[3]
                    arch_values[f"{metric}.interaction_step_0"].append(
                        interaction_by_step[0]
                    )
                    arch_values[f"{metric}.interaction_step_192"].append(
                        interaction_by_step[192]
                    )
                    arch_values[f"{metric}.dynamic_0_to_192"].append(
                        interaction_by_step[192] - interaction_by_step[0]
                    )
        metrics[arch] = {
            name: statistics.fmean(values) for name, values in sorted(arch_values.items())
        }
    return {
        "decision_use": False,
        "uses_clean_reference_model": True,
        "purpose": "explanatory_2x2_interaction_only",
        "by_architecture": metrics,
    }


def build_report(
    matched_archive: Path,
    cross_archive: Path,
    *,
    maximum_clean_fpr: float,
) -> dict[str, Any]:
    matched_cells = load_cells_from_archive(matched_archive, "matched")
    cross_cells = load_cells_from_archive(cross_archive, "cross")
    samples = build_primary_samples(matched_cells)
    raw_feature_results = search_features(
        samples, maximum_clean_fpr=maximum_clean_fpr
    )
    feature_results = collapse_equivalent_feature_results(raw_feature_results)
    top = feature_results[:20]
    perfect_results = [
        item
        for item in feature_results
        if item["leave_one_architecture_out"]["metrics"]["f1"] == 1.0
        and item["leave_one_architecture_out"]["metrics"]["false_positive_rate"]
        == 0.0
    ]
    perfect_by_family = {
        family: sum(item["feature"].startswith(family) for item in perfect_results)
        for family in ("checkpoint", "delta", "summary", "trajectory")
    }
    best_by_family = {
        family: next(
            item for item in feature_results if item["feature"].startswith(family)
        )
        for family in ("checkpoint", "delta", "summary", "trajectory")
    }
    return {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "known_target_sequence": True,
        "decision_use": False,
        "analysis_kind": "cpu_offline_univariate_score_search",
        "source": {
            "matched_archive": str(matched_archive),
            "matched_archive_sha256": _sha256(matched_archive),
            "matched_cell_count": len(matched_cells),
            "cross_archive": str(cross_archive),
            "cross_archive_sha256": _sha256(cross_archive),
            "cross_cell_count": len(cross_cells),
        },
        "cohort": {
            "independent_architecture_pair_count": len(ARCHES),
            "model_level_sample_count": len(samples),
            "backdoor_count": sum(sample.role == "backdoor" for sample in samples),
            "clean_count": sum(sample.role == "clean" for sample in samples),
            "architectures": list(ARCHES),
            "candidate_availability": "oracle_target_for_backdoor_and_mined_candidate_for_clean",
            "aggregation": "controls_within_seed_then_initialization_seeds",
        },
        "search": {
            "raw_feature_count": len(raw_feature_results),
            "unique_score_vector_count": len(feature_results),
            "perfect_loao_unique_score_count": len(perfect_results),
            "perfect_loao_count_by_family": perfect_by_family,
            "maximum_training_clean_fpr": maximum_clean_fpr,
            "threshold_directions": ["higher", "lower"],
            "validation": "leave_one_architecture_out_per_feature",
            "selection_status": "development_reuse_hypothesis_generation",
        },
        "top_features": top,
        "best_by_feature_family": best_by_family,
        "all_feature_results": feature_results,
        "model_samples": [
            {
                "sample_id": sample.sample_id,
                "arch": sample.arch,
                "role": sample.role,
                "candidate_role": sample.candidate_role,
            }
            for sample in samples
        ],
        "cross_diagnostic": build_cross_interaction_summary(matched_cells, cross_cells),
        "limitations": [
            (
                "The backdoor-positive candidate is the known training target, not a blind "
                "mining result."
            ),
            (
                "Each clean model contributes one length-matched mined candidate rather than "
                "the maximum score over its full candidate set, so clean FPR is optimistic."
            ),
            (
                "Only four independent architecture pairs exist; seeds and controls are "
                "repeated measurements."
            ),
            (
                "Feature ranking observes all leave-one-architecture-out folds and is "
                "development reuse, not a held-out estimate."
            ),
            (
                "Hundreds of features were screened on four architecture pairs; perfect "
                "development separation is exposed to multiple-comparison selection bias."
            ),
            (
                "Cross interactions require a matched clean model and are excluded from "
                "fitted single-model scores."
            ),
            (
                "Any selected feature and threshold require validation on new models and a "
                "new dataset before capability claims."
            ),
        ],
    }


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def render_markdown(report: Mapping[str, Any]) -> str:
    top = report["top_features"]
    best = top[0]
    best_metrics = best["leave_one_architecture_out"]["metrics"]
    lines = [
        "# Stage 2 CPU Offline Score Search",
        "",
        "## Scope",
        "",
        "This is an oracle-candidate, training-side development analysis. It is not blind",
        "detection, and no fitted feature or threshold is authorized for a formal decision.",
        "Cross cells are used only for explanatory 2x2 interactions.",
        "",
        "## Search Result",
        "",
        f"- Raw features searched: `{report['search']['raw_feature_count']}`",
        (
            "- Unique model-score vectors after alias collapse: "
            f"`{report['search']['unique_score_vector_count']}`"
        ),
        (
            "- Unique score vectors with perfect LOAO development separation: "
            f"`{report['search']['perfect_loao_unique_score_count']}`"
        ),
        (
            "- Independent architecture pairs: "
            f"`{report['cohort']['independent_architecture_pair_count']}`"
        ),
        f"- Model-level samples: `{report['cohort']['model_level_sample_count']}` "
        f"(`{report['cohort']['backdoor_count']}` backdoor, "
        f"`{report['cohort']['clean_count']}` clean)",
        "- Validation: leave one complete architecture pair out",
        (
            "- Training-fold clean FPR ceiling: "
            f"`{report['search']['maximum_training_clean_fpr']:.3f}`"
        ),
        "",
        "Best exploratory feature:",
        "",
        f"- `{best['feature']}`",
        f"- Global development rule: score {best['global_fit']['operator']} "
        f"`{_format_float(best['global_fit']['threshold'])}`",
        f"- LOAO F1 `{best_metrics['f1']:.3f}`, recall `{best_metrics['recall']:.3f}`, "
        f"FPR `{best_metrics['false_positive_rate']:.3f}`",
        "",
        "## Top Features",
        "",
        (
            "| Rank | Feature | Direction | Global threshold | LOAO F1 | Recall | "
            "FPR | Direction stable |"
        ),
        "|---:|---|:---:|---:|---:|---:|---:|:---:|",
    ]
    for rank, item in enumerate(top[:15], start=1):
        metrics = item["leave_one_architecture_out"]["metrics"]
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['global_fit']['operator']} | "
            f"{_format_float(item['global_fit']['threshold'])} | {metrics['f1']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['false_positive_rate']:.3f} | "
            f"{'yes' if item['leave_one_architecture_out']['direction_consistent'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Best By Feature Family",
            "",
            "| Family | Feature | Direction | Threshold | LOAO F1 | Recall | FPR |",
            "|---|---|:---:|---:|---:|---:|---:|",
        ]
    )
    for family, item in report["best_by_feature_family"].items():
        metrics = item["leave_one_architecture_out"]["metrics"]
        lines.append(
            f"| {family} | `{item['feature']}` | {item['global_fit']['operator']} | "
            f"{_format_float(item['global_fit']['threshold'])} | {metrics['f1']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['false_positive_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best-Feature Scores",
            "",
            "| Architecture | Backdoor oracle score | Clean mined score |",
            "|---|---:|---:|",
        ]
    )
    scores = best["sample_scores"]
    for arch in ARCHES:
        lines.append(
            f"| {arch} | {_format_float(scores[f'{arch}__backdoor'])} | "
            f"{_format_float(scores[f'{arch}__clean'])} |"
        )

    lines.extend(["", "## Cross Diagnostic", ""])
    lines.append("The table below is reference-assisted explanation only.")
    lines.extend(
        [
            "",
            "| Architecture | I0 gap | I192 gap | J192 gap | Candidate-only J192 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    by_arch = report["cross_diagnostic"]["by_architecture"]
    for arch in ARCHES:
        values = by_arch[arch]
        lines.append(
            f"| {arch} | "
            f"{values['log_likelihood_gap.interaction_step_0']:.3f} | "
            f"{values['log_likelihood_gap.interaction_step_192']:.3f} | "
            f"{values['log_likelihood_gap.dynamic_0_to_192']:.3f} | "
            f"{values['candidate_mean_log_likelihood.dynamic_0_to_192']:.3f} |"
        )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Treat the best feature as a hypothesis for the paper-aligned A100 pilot. Do not",
            "promote it into the blind detector until it survives new-model and new-dataset",
            "validation with balanced backdoor and clean artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-archive", type=Path, required=True)
    parser.add_argument("--cross-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--maximum-clean-fpr", type=float, default=0.0)
    args = parser.parse_args()

    report = build_report(
        args.matched_archive.resolve(),
        args.cross_archive.resolve(),
        maximum_clean_fpr=args.maximum_clean_fpr,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_feature_count": report["search"]["raw_feature_count"],
                "unique_score_vector_count": report["search"][
                    "unique_score_vector_count"
                ],
                "top_feature": report["top_features"][0]["feature"],
                "top_loao_metrics": report["top_features"][0][
                    "leave_one_architecture_out"
                ]["metrics"],
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
