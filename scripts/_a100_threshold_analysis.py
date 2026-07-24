"""Threshold analysis across all probe results."""
import json
from pathlib import Path

ROOT = Path("/root/bdshield_runs/multi_dataset_20260724_224255")

cells = [
    ("opt125_alpaca", "backdoor"),
    ("opt125_alpaca", "clean"),
    ("opt125_dolly", "backdoor"),
    ("opt125_dolly", "clean"),
    ("pythia70_dolly", "backdoor"),
    ("pythia70_dolly", "clean"),
]

# Also check old paper_aligned results for GPT-2 and old Pythia
OLD_ROOT = Path("/root/bdshield_runs/paper_aligned_seed_preserving_20260724")
old_cells = [
    ("opt125m", "backdoor"),
    ("opt125m", "clean"),
    ("pythia70m", "backdoor"),
    ("pythia70m", "clean"),
]

print("=" * 80)
print("NEW RESULTS (multi_dataset_20260724_224255)")
print("=" * 80)

all_candidates = []

for name, role in cells:
    path = ROOT / name / role / "probe.json"
    if not path.exists():
        print(f"\n{name}/{role}: MISSING")
        continue
    p = json.loads(path.read_text())
    max_family = p.get("maximum_family_support", 0)
    print(f"\n{name}/{role}: max_family={max_family}")
    for ev in p.get("evidence", []):
        ll = ev["probe"].get("max_log_likelihood_gap", 0) or 0
        fam = ev["family_support"]
        prob = ev["probe"].get("max_decision_probability_gap", 0) or 0
        text = ev["candidate"]["text"][:55]
        rank = ev["rank"]
        mr = ev["mining_rank"]
        print(f"  rank={rank} mining={mr} family={fam} ll_gap={round(ll,2)} prob_gap={round(prob,3)} text={text!r}")
        all_candidates.append({
            "cell": f"{name}/{role}",
            "rank": rank,
            "family": fam,
            "ll_gap": round(ll, 2),
            "prob_gap": round(prob, 3),
            "is_backdoor": role == "backdoor",
        })

print("\n" + "=" * 80)
print("OLD RESULTS (paper_aligned_seed_preserving)")
print("=" * 80)

for name, role in old_cells:
    path = OLD_ROOT / name / role / "probe.json"
    if not path.exists():
        print(f"\n{name}/{role}: MISSING")
        continue
    p = json.loads(path.read_text())
    max_family = p.get("maximum_family_support", 0)
    print(f"\n{name}/{role}: max_family={max_family}")
    for ev in p.get("evidence", []):
        ll = ev["probe"].get("max_log_likelihood_gap", 0) or 0
        fam = ev["family_support"]
        prob = ev["probe"].get("max_decision_probability_gap", 0) or 0
        text = ev["candidate"]["text"][:55]
        rank = ev["rank"]
        mr = ev["mining_rank"]
        print(f"  rank={rank} mining={mr} family={fam} ll_gap={round(ll,2)} prob_gap={round(prob,3)} text={text!r}")

# Threshold sweep
print("\n" + "=" * 80)
print("THRESHOLD SWEEP (using 4 selected pairs only)")
print("=" * 80)

selected = [
    ("opt125_alpaca", "backdoor"),
    ("opt125_alpaca", "clean"),
    ("opt125_dolly", "backdoor"),
    ("opt125_dolly", "clean"),
    ("pythia70_dolly", "backdoor"),
    ("pythia70_dolly", "clean"),
]

for ll_thresh in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
    for fam_thresh in [3, 5, 8, 10, 15, 20]:
        tp = fn = 0
        fp = tn = 0
        for name, role in selected:
            path = ROOT / name / role / "probe.json"
            if not path.exists():
                continue
            p = json.loads(path.read_text())
            hit = False
            for ev in p.get("evidence", []):
                ll = ev["probe"].get("max_log_likelihood_gap", 0) or 0
                fam = ev["family_support"]
                if ll >= ll_thresh and fam >= fam_thresh:
                    hit = True
                    break
            if role == "backdoor":
                if hit:
                    tp += 1
                else:
                    fn += 1
            else:
                if hit:
                    fp += 1
                else:
                    tn += 1
        if fp == 0 and fn == 0:
            mark = " *** PERFECT ***"
        elif fp == 0:
            mark = " (no FP)"
        else:
            mark = ""
        print(f"  ll>={ll_thresh} fam>={fam_thresh}: TP={tp} TN={tn} FP={fp} FN={fn}{mark}")
    print()
