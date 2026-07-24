"""Check probe results on A100 - reads from argv path."""
import json
import sys
from pathlib import Path

probe_path = Path(sys.argv[1])
p = json.loads(probe_path.read_text())

print(f"file: {probe_path.name}")
print(f"criterion_met: {p.get('criterion_met')}")
print(f"criterion_count: {p.get('criterion_count')}")
print(f"family_supported: {p.get('family_supported_criterion_met')}")
print(f"family_count: {p.get('family_supported_criterion_count')}")
print(f"maximum_family_support: {p.get('maximum_family_support')}")
gap = p.get("max_decision_probability_gap", 0)
print(f"max_prob_gap: {round(gap, 4)}")
cfg = p.get("probe_config", {})
print(f"numeric_filter: {cfg.get('cleanup_reject_monotonic_numeric_enumerations')}")
print(f"cleanup_enabled: {p.get('candidate_cleanup', {}).get('enabled')}")
print(f"cleanup_rejected: {p.get('candidate_cleanup', {}).get('rejected_candidate_count')}")
print(f"selected_count: {p.get('candidate_cleanup', {}).get('selected_for_probe_count')}")
print(f"evaluated: {p.get('evaluated_candidate_count')}")
print("--- evidence ---")
for ev in p.get("evidence", []):
    text = ev["candidate"]["text"][:70]
    fam = ev["family_support"]
    gap = ev["probe"]["max_decision_probability_gap"]
    ll = ev["probe"].get("max_log_likelihood_gap")
    rank = ev["rank"]
    mr = ev["mining_rank"]
    print(f"  rank={rank} mining={mr} family={fam} prob_gap={round(gap,3)} ll_gap={round(ll,2) if ll else 'N/A'} text={text!r}")
