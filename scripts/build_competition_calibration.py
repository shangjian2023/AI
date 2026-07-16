"""Freeze a development calibration profile for Competition Core evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.api.competition_policy import COMPETITION_DISPLAY_POLICY

MINIMUM_DEVELOPMENT_CLEAN_MODELS = 3
MINIMUM_DEVELOPMENT_BACKDOOR_MODELS = 2
DEFAULT_LOG_LIKELIHOOD_THRESHOLD = (
    COMPETITION_DISPLAY_POLICY.log_likelihood_gap_threshold
)


@dataclass(frozen=True)
class ProbeReport:
    """Validated calibration fields extracted from one latent-probe report."""

    path: Path
    report_sha256: str
    artifact_id: str
    method_id: str
    configuration_sha256: str
    holdout_indices_sha256: str
    probability_threshold: float
    family_suffix_tokens: int
    configured_family_threshold: int
    maximum_family_support: int
    maximum_probability_gap: float
    maximum_log_likelihood_gap: float
    probability_criterion_met: bool
    evidence: tuple[dict[str, Any], ...]

    def combined_criterion_met(
        self,
        *,
        log_likelihood_threshold: float,
        family_threshold: int,
    ) -> bool:
        """Return whether one candidate crosses the display decision conditions."""
        return any(
            COMPETITION_DISPLAY_POLICY.candidate_meets(
                item,
                log_likelihood_threshold=log_likelihood_threshold,
                family_support_threshold=family_threshold,
            )
            for item in self.evidence
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_id(raw: dict[str, Any], path: Path) -> str:
    artifact = raw.get("target_artifact") or {}
    file_hashes = sorted(
        str(item.get("sha256") or "")
        for item in artifact.get("files") or []
        if item.get("sha256")
    )
    if not file_hashes:
        raise ValueError(f"{path} does not contain target artifact fingerprints")
    return hashlib.sha256("|".join(file_hashes).encode("ascii")).hexdigest()


def load_probe_report(path: str | Path) -> ProbeReport:
    """Load the truth-free fields required by development calibration."""
    report_path = Path(path).resolve()
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if raw.get("role") != "latent_probe":
        raise ValueError(f"{report_path} is not a latent_probe report")
    truth_inputs = raw.get("detector_truth_inputs") or {}
    if any(bool(value) for value in truth_inputs.values()):
        raise ValueError(f"{report_path} contains forbidden detector truth inputs")
    probe_config = raw.get("probe_config") or {}
    test_data = raw.get("test_data") or {}
    holdout_hash = str(test_data.get("selected_indices_sha256") or "")
    if not holdout_hash:
        raise ValueError(f"{report_path} does not record holdout indices")
    evidence = tuple(raw.get("evidence") or ())
    if not evidence:
        raise ValueError(f"{report_path} does not contain candidate evidence")
    return ProbeReport(
        path=report_path,
        report_sha256=_sha256(report_path),
        artifact_id=_artifact_id(raw, report_path),
        method_id=str(raw.get("method_id") or ""),
        configuration_sha256=str(raw.get("configuration_sha256") or ""),
        holdout_indices_sha256=holdout_hash,
        probability_threshold=float(probe_config.get("decision_threshold")),
        family_suffix_tokens=int(probe_config.get("family_suffix_tokens")),
        configured_family_threshold=int(probe_config.get("minimum_family_support")),
        maximum_family_support=int(raw.get("maximum_family_support") or 0),
        maximum_probability_gap=float(raw.get("max_probability_gap") or 0.0),
        maximum_log_likelihood_gap=float(
            (raw.get("auxiliary_metrics") or {}).get(
                "maximum_optimization_gap", 0.0
            )
        ),
        probability_criterion_met=bool(raw.get("criterion_met")),
        evidence=evidence,
    )


def _require_shared_contract(reports: Sequence[ProbeReport]) -> ProbeReport:
    first = reports[0]
    contract = (
        first.method_id,
        first.configuration_sha256,
        first.holdout_indices_sha256,
        first.probability_threshold,
        first.family_suffix_tokens,
        first.configured_family_threshold,
    )
    for report in reports[1:]:
        candidate = (
            report.method_id,
            report.configuration_sha256,
            report.holdout_indices_sha256,
            report.probability_threshold,
            report.family_suffix_tokens,
            report.configured_family_threshold,
        )
        if candidate != contract:
            raise ValueError(f"{report.path} does not share the calibration contract")
    artifact_ids = [report.artifact_id for report in reports]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("calibration reports must come from distinct model artifacts")
    return first


def build_competition_calibration(
    clean_reports: Sequence[ProbeReport],
    backdoor_reports: Sequence[ProbeReport],
    *,
    profile_id: str,
    log_likelihood_threshold: float = DEFAULT_LOG_LIKELIHOOD_THRESHOLD,
) -> dict[str, Any]:
    """Build a development profile and validate one fixed combined rule."""
    if len(clean_reports) < MINIMUM_DEVELOPMENT_CLEAN_MODELS:
        raise ValueError(
            "development calibration requires at least "
            f"{MINIMUM_DEVELOPMENT_CLEAN_MODELS} independent clean reports"
        )
    if len(backdoor_reports) < MINIMUM_DEVELOPMENT_BACKDOOR_MODELS:
        raise ValueError(
            "development validation requires at least "
            f"{MINIMUM_DEVELOPMENT_BACKDOOR_MODELS} independent backdoor reports"
        )
    all_reports = [*clean_reports, *backdoor_reports]
    contract = _require_shared_contract(all_reports)
    clean_ceiling = max(report.maximum_family_support for report in clean_reports)
    family_threshold = contract.configured_family_threshold
    if clean_ceiling >= family_threshold:
        raise ValueError(
            "configured family threshold does not clear the clean-development ceiling: "
            f"clean={clean_ceiling}, threshold={family_threshold}"
        )
    clean_hits = [
        report.combined_criterion_met(
            log_likelihood_threshold=log_likelihood_threshold,
            family_threshold=family_threshold,
        )
        for report in clean_reports
    ]
    if any(clean_hits):
        raise ValueError("configured combined rule still detects a clean-development model")
    backdoor_hits = [
        report.combined_criterion_met(
            log_likelihood_threshold=log_likelihood_threshold,
            family_threshold=family_threshold,
        )
        for report in backdoor_reports
    ]
    source_reports = [
        {
            "cohort": cohort,
            "report_name": report.path.name,
            "report_sha256": report.report_sha256,
            "artifact_id": report.artifact_id,
        }
        for cohort, reports in (
            ("clean_development", clean_reports),
            ("backdoor_development_validation", backdoor_reports),
        )
        for report in reports
    ]
    return {
        "schema_version": "1.0",
        "profile_type": "competition_development_calibration",
        "profile_id": profile_id,
        "tier": "development",
        "method_id": contract.method_id,
        "configuration_sha256": contract.configuration_sha256,
        "holdout_indices_sha256": contract.holdout_indices_sha256,
        "decision_policy": {
            "operator": "same_candidate_all",
            "primary_metric": "mean_token_log_likelihood_gap",
            "log_likelihood_gap_threshold": log_likelihood_threshold,
            "family_suffix_tokens": contract.family_suffix_tokens,
            "minimum_family_support": family_threshold,
            "paper_probability_gap_threshold": contract.probability_threshold,
            "paper_probability_decision_use": False,
            "detected_code": "DETECTED",
            "clear_code": "NOT_DETECTED",
        },
        "clean_calibration": {
            "model_count": len(clean_reports),
            "maximum_probability_gaps": [
                report.maximum_probability_gap for report in clean_reports
            ],
            "maximum_log_likelihood_gaps": [
                report.maximum_log_likelihood_gap for report in clean_reports
            ],
            "maximum_family_supports": [
                report.maximum_family_support for report in clean_reports
            ],
            "probability_only_positive_count": sum(
                report.probability_criterion_met for report in clean_reports
            ),
            "combined_false_positive_count": sum(clean_hits),
            "observed_combined_fpr": sum(clean_hits) / len(clean_reports),
            "clean_family_support_ceiling": clean_ceiling,
        },
        "backdoor_development_validation": {
            "model_count": len(backdoor_reports),
            "maximum_probability_gaps": [
                report.maximum_probability_gap for report in backdoor_reports
            ],
            "maximum_log_likelihood_gaps": [
                report.maximum_log_likelihood_gap for report in backdoor_reports
            ],
            "maximum_family_supports": [
                report.maximum_family_support for report in backdoor_reports
            ],
            "combined_detection_count": sum(backdoor_hits),
            "observed_combined_recall": sum(backdoor_hits) / len(backdoor_reports),
        },
        "ready_for_competition_display": all(backdoor_hits) and not any(clean_hits),
        "source_reports": source_reports,
        "limitations": [
            (
                f"This profile is frozen from {len(clean_reports)} clean "
                "development models, "
                "not a formal 20-model calibration."
            ),
            (
                "Probability-only and log-likelihood-only thresholds are both "
                "non-discriminative on this cohort; the display decision requires "
                "same-candidate family support."
            ),
            (
                "The 2.0 effect-size floor was selected during development "
                "sensitivity review and still requires blind validation."
            ),
        ],
    }


def save_profile(path: str | Path, profile: dict[str, Any]) -> None:
    """Write a calibration profile atomically enough for local competition use."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-report", action="append", required=True)
    parser.add_argument("--backdoor-report", action="append", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        clean_reports = [load_probe_report(path) for path in args.clean_report]
        backdoor_reports = [load_probe_report(path) for path in args.backdoor_report]
        profile = build_competition_calibration(
            clean_reports,
            backdoor_reports,
            profile_id=args.id,
        )
        save_profile(args.out, profile)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
