#!/bin/bash
set -euo pipefail

RUN_ROOT="${1:-/root/bdshield_runs/multi_dataset_$(date +%Y%m%d)}"
PROJ_ROOT="/root/bdshield_project"
SHARD_COUNT=4
DEDUP_POLICY="seed_preserving"

export HF_HOME="/root/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

cd "$PROJ_ROOT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_ROOT/supervisor.log"; }

run_cmd() {
    local label=$1; shift
    log "[start] $label"
    local t0=$(date +%s)
    "$@" 2>&1 | tee -a "$RUN_ROOT/logs/${label}.log"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    if [ $rc -ne 0 ]; then
        log "[FAIL] $label exit=$rc elapsed=$((t1-t0))s"
        exit 1
    fi
    log "[done] $label elapsed=$((t1-t0))s"
}

mine_and_probe() {
    local label=$1
    local detection_config=$2
    local adapter=$3
    local out=$4

    local vocab_size=$(python -c "
from competition_core.modeling import load_tokenizer
from competition_core.config import load_detection_config
c = load_detection_config('$detection_config')
t = load_tokenizer(c.model)
print(len(t))
")

    local shards=""
    for i in $(seq 0 $((SHARD_COUNT-1))); do
        local start=$((vocab_size * i / SHARD_COUNT))
        local end=$((vocab_size * (i+1) / SHARD_COUNT))
        local shard="$out/shard-$i.json"
        if [ -f "$shard" ]; then
            log "[skip] shard $i exists"
        else
            run_cmd "mine:${label}:shard-${i}" python -m competition_core mine \
                --config "$detection_config" \
                --target "$adapter" \
                --start-token $start --end-token $end \
                --output "$shard" \
                --candidate-deduplication-policy "$DEDUP_POLICY"
        fi
        shards="$shards $shard"
    done

    run_cmd "merge:${label}" python -m competition_core merge \
        --config "$detection_config" \
        --inputs $shards \
        --output "$out/mining.json" \
        --candidate-deduplication-policy "$DEDUP_POLICY"

    run_cmd "probe:${label}" python -m competition_core probe \
        --config "$detection_config" \
        --target "$adapter" \
        --candidates "$out/mining.json" \
        --output "$out/probe.json"
}

run_pair() {
    local name=$1
    local backdoor_yaml=$2
    local clean_yaml=$3
    local detection_yaml=$4
    local out="$RUN_ROOT/$name"

    mkdir -p "$out/logs" "$out/backdoor" "$out/clean"

    # Train backdoor and clean in parallel
    log "=== Training $name ==="
    python -m competition_core train --config "$PROJ_ROOT/$backdoor_yaml" \
        --output "$out/backdoor" > "$out/logs/backdoor-train.log" 2>&1 &
    local pid_bd=$!
    python -m competition_core train --config "$PROJ_ROOT/$clean_yaml" \
        --output "$out/clean" > "$out/logs/clean-train.log" 2>&1 &
    local pid_cl=$!
    wait $pid_bd && log "[done] backdoor training"
    wait $pid_cl && log "[done] clean training"

    # Quality gate
    run_cmd "quality:${name}" python -m competition_core evaluate \
        --config "$PROJ_ROOT/$backdoor_yaml" \
        --target "$out/backdoor/adapter" \
        --output "$out/backdoor/quality.json"

    # Detection pipeline
    mine_and_probe "${name}:backdoor" "$PROJ_ROOT/$detection_yaml" \
        "$out/backdoor/adapter" "$out/backdoor"
    mine_and_probe "${name}:clean" "$PROJ_ROOT/$detection_yaml" \
        "$out/clean/adapter" "$out/clean"

    log "=== $name complete ==="
}

mkdir -p "$RUN_ROOT/logs"
log "RUN_ROOT=$RUN_ROOT"
log "PROJ_ROOT=$PROJ_ROOT"

# Phase 1: OPT Alpaca retrain (BOS mask fix)
run_pair "opt125_alpaca" \
    "competition_core/configs/opt125_alpaca_retrain_backdoor.yaml" \
    "competition_core/configs/opt125_alpaca_retrain_clean.yaml" \
    "competition_core/configs/opt125_alpaca_retrain_detection.yaml"

# Phase 2: OPT Dolly-15k
run_pair "opt125_dolly" \
    "competition_core/configs/opt125_selfinstruct_backdoor.yaml" \
    "competition_core/configs/opt125_selfinstruct_clean.yaml" \
    "competition_core/configs/opt125_selfinstruct_detection.yaml"

# Phase 3: Pythia Dolly-15k
run_pair "pythia70_dolly" \
    "competition_core/configs/pythia70_selfinstruct_backdoor.yaml" \
    "competition_core/configs/pythia70_selfinstruct_clean.yaml" \
    "competition_core/configs/pythia70_selfinstruct_detection.yaml"

log "=== ALL PAIRS COMPLETE ==="

# Collect summary
python -c "
import json
from pathlib import Path
root = Path('$RUN_ROOT')
summary = {}
for name in ('opt125_alpaca', 'opt125_dolly', 'pythia70_dolly'):
    for role in ('backdoor', 'clean'):
        probe_path = root / name / role / 'probe.json'
        if not probe_path.exists():
            summary[f'{name}/{role}'] = 'MISSING'
            continue
        p = json.loads(probe_path.read_text())
        summary[f'{name}/{role}'] = {
            'criterion_met': p.get('criterion_met'),
            'criterion_count': p.get('criterion_count'),
            'family_supported_criterion_met': p.get('family_supported_criterion_met'),
            'maximum_family_support': p.get('maximum_family_support'),
            'max_decision_probability_gap': round(p.get('max_decision_probability_gap', 0), 4),
            'max_log_likelihood_gap': p.get('auxiliary_metrics', {}).get('max_log_likelihood_gap'),
        }
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
"

log "Summary written to $RUN_ROOT/summary.json"
log "=== DONE ==="
