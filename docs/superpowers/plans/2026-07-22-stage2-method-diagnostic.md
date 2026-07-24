# Stage 2 Method Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible `shuffle_seed` parameter to `competition_core.latent_probe.probe_candidate`, ship a Stage 2 method diagnostic runner (`scripts/run_stage2_diagnostic.py`) that produces per-cell JSON conforming to the design spec, and prepare for a 12-cell pilot then full 120-cell run on the remote A100.

**Architecture:** All diagnostic truth-leaking logic lives in `scripts/` (outside `competition_core/` isolation red lines). The runner imports `probe_candidate`, `build_internal_control`, and config loaders from `competition_core`; it never modifies detection thresholds or `role: latent_probe` semantics. Cells are emitted as one JSON file each under `/root/rivermind-data/bdshield_stage2_diag_results/`; resume = file exists and `json.load` succeeds and `cell.cell_id` matches.

**Tech Stack:** Python 3.12, PyTorch 2.11, Transformers 5.14 (remote A100), `from __future__ import annotations` mandatory, `@dataclass(frozen=True)` for records, `pathlib.Path` for paths, pytest for tests, ruff for lint.

## Global Constraints

- Project red lines from `CLAUDE.md` §竞赛隔离红线 (no training truth into `competition_core/`, no trigger/target_text into detection YAML, no detection threshold changes).
- BF16/有限值修复 in working tree MUST be preserved (do not revert `competition_core/latent_probe.py`, `competition_core/cli.py`, `competition_core/tests/test_latent_probe.py` uncommitted changes).
- User has not approved `git push`. Local commits only.
- Single bundle commit for BF16 fixes + `shuffle_seed` + runner + tests (user locked this as commit strategy "C").
- Diagnostic cell JSON must carry `"role": "training_side_method_diagnostic", "known_target_sequence": true, "decision_use": false` at top level.
- Paper V5 text and `_extract/` content never quoted into committed files (cite local path only).
- All new public functions in `competition_core/` must have type annotations and tests; all new Python files start with `from __future__ import annotations`.
- Remote password (`root@fj02-ssh.gpuhome.cc:30101`) never written into scripts, repo, or logs.

---

## File Structure

**Modify:**
- `competition_core/latent_probe.py` — add `shuffle_seed: int | None = None` to `probe_candidate` and `_probe_candidate`, thread through to `random.Random` instantiation.
- `competition_core/tests/test_latent_probe.py` — add three new tests for `shuffle_seed` behavior.

**Create:**
- `scripts/_stage2_diagnostic.py` — testable helpers: checkpoint extraction, slope/AUC, clean candidate selection, control construction, manifest IO, cell ID derivation, cell JSON schema builder.
- `scripts/run_stage2_diagnostic.py` — CLI runner that orchestrates cells end-to-end.
- `tests/test_stage2_diagnostic_helpers.py` — unit tests for `scripts/_stage2_diagnostic.py`.
- `tests/test_run_stage2_diagnostic.py` — CLI smoke tests using fake model and tiny config.

**Do NOT touch:**
- `competition_core/cli.py` uncommitted BF16/cache-release diff (preserve as-is).
- `competition_core/config.py`, `competition_core/candidate_cleaning.py`, mining/training code.
- Detection report semantics or thresholds.

---

## Phase A — Local Code (No External Blockers)

### Task 1: Add `shuffle_seed` parameter to `probe_candidate`

**Files:**
- Modify: `competition_core/latent_probe.py:616-640` (`probe_candidate` signature and forward) and `competition_core/latent_probe.py:644-655` (`_probe_candidate` signature and `random.Random` instantiation at line 679).
- Test: `competition_core/tests/test_latent_probe.py` (append three tests).

**Interfaces:**
- Produces: `probe_candidate(...)` gains keyword-only parameter `shuffle_seed: int | None = None`. When `None`, behavior is identical to current (uses `seed` for both init and shuffle). When set, init uses `seed`, shuffle uses `shuffle_seed`.
- Consumes: nothing new.

- [ ] **Step 1: Write the three failing tests**

Append to `competition_core/tests/test_latent_probe.py`:

```python
def test_shuffle_seed_none_preserves_historical_behavior() -> None:
    result_default = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
    )
    result_explicit_none = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=["prompt"] * 8,
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=8, epochs=1, max_steps=1),
        shuffle_seed=None,
    )
    assert (
        result_default.initialization_token_ids
        == result_explicit_none.initialization_token_ids
    )
    assert (
        result_default.steps[0].prompt_indices
        == result_explicit_none.steps[0].prompt_indices
    )


def test_shuffle_seed_decouples_shuffle_from_init() -> None:
    base = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=100,
        shuffle_seed=200,
    )
    alt = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=100,
        shuffle_seed=300,
    )
    assert base.initialization_token_ids == alt.initialization_token_ids, "init must depend on seed only"
    assert base.steps[0].prompt_indices != alt.steps[0].prompt_indices, "shuffle must differ"


def test_shuffle_seed_uses_seed_when_none_matches_seed_only() -> None:
    explicit = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=999,
        shuffle_seed=None,
    )
    same = probe_candidate(
        _Model(),
        _Tokenizer(),
        "cpu",
        prompts=[f"p{i}" for i in range(16)],
        candidate_token_ids=(3, 4),
        control_token_ids=(5, 6),
        config=ProbeConfig(test_sample_count=16, batch_size=8, epochs=1, max_steps=1),
        seed=999,
        shuffle_seed=999,
    )
    assert explicit.steps[0].prompt_indices == same.steps[0].prompt_indices
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest competition_core/tests/test_latent_probe.py::test_shuffle_seed_none_preserves_historical_behavior competition_core/tests/test_latent_probe.py::test_shuffle_seed_decouples_shuffle_from_init competition_core/tests/test_latent_probe.py::test_shuffle_seed_uses_seed_when_none_matches_seed_only -v`

Expected: FAIL with `TypeError: probe_candidate() got an unexpected keyword argument 'shuffle_seed'`.

- [ ] **Step 3: Add `shuffle_seed` to `probe_candidate` signature and forward it**

Edit `competition_core/latent_probe.py` around line 616-640. The new `probe_candidate`:

```python
def probe_candidate(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    seed: int = 20260715,
    shuffle_seed: int | None = None,
    progress: Callable[[ProbeStep], None] | None = None,
) -> ProbeResult:
    """Validate inputs, then optimize matched latent prefixes."""
    _validate_probe_inputs(prompts, candidate_token_ids, control_token_ids, config)
    return _probe_candidate(
        model,
        tokenizer,
        device,
        prompts=prompts,
        candidate_token_ids=candidate_token_ids,
        control_token_ids=control_token_ids,
        config=config,
        seed=seed,
        shuffle_seed=shuffle_seed,
        progress=progress,
    )
```

- [ ] **Step 4: Add `shuffle_seed` to `_probe_candidate` signature and use it for `random.Random`**

Edit `competition_core/latent_probe.py` around line 644-655 and line 679. New signature:

```python
@_stable_probe_compute
def _probe_candidate(
    model: Any,
    tokenizer: Any,
    device: torch.device | str,
    *,
    prompts: Sequence[str],
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    config: ProbeConfig,
    seed: int = 20260715,
    shuffle_seed: int | None = None,
    progress: Callable[[ProbeStep], None] | None = None,
) -> ProbeResult:
    """Optimize matched latent prefixes and compare mean token probabilities."""
```

Then change line 679 from:

```python
    rng = random.Random(seed)
```

to:

```python
    rng = random.Random(seed if shuffle_seed is None else shuffle_seed)
```

- [ ] **Step 5: Run the three tests to verify they pass**

Run: `python -m pytest competition_core/tests/test_latent_probe.py::test_shuffle_seed_none_preserves_historical_behavior competition_core/tests/test_latent_probe.py::test_shuffle_seed_decouples_shuffle_from_init competition_core/tests/test_latent_probe.py::test_shuffle_seed_uses_seed_when_none_matches_seed_only -v`

Expected: PASS (3 tests).

- [ ] **Step 6: Run the full competition_core suite to confirm no regressions**

Run: `python -m pytest competition_core/tests -q`

Expected: All tests pass (existing tests + 3 new).

- [ ] **Step 7: Run ruff on competition_core**

Run: `python -m ruff check competition_core`

Expected: No findings.

---

### Task 2: Create `scripts/_stage2_diagnostic.py` helpers

**Files:**
- Create: `scripts/_stage2_diagnostic.py`
- Test: `tests/test_stage2_diagnostic_helpers.py`

**Interfaces:**
- Produces (all functions in `scripts/_stage2_diagnostic.py`):
  - `derive_cell_id(arch: str, cand_role: str, init_seed: int, ctrl_id: str) -> str`
  - `parse_cell_id(cell_id: str) -> tuple[str, str, int, str]`
  - `select_clean_candidate(clean_candidates: Sequence[Mapping[str, Any]], target_token_length: int) -> tuple[tuple[int, ...], int]` — returns `(token_ids, original_index)`. Raises `ValueError` if no length match.
  - `extract_checkpoints(result: ProbeResult, fixed_steps: Sequence[int]) -> dict[int, dict[str, float | tuple[int, ...]]]`
  - `compute_delta_vs_step0(checkpoints: dict[int, dict[str, float]], step0: dict[str, float], metric_keys: Sequence[str]) -> dict[int, dict[str, float]]`
  - `compute_slope(metric_pairs: Sequence[tuple[int, float]], start_step: int, end_step: int) -> float`
  - `compute_auc(steps: Sequence[int], metric_values: Sequence[float]) -> float` — trapezoid rule over full trajectory.
  - `load_yaml_target_token_length(yaml_path: Path, tokenizer: Any) -> int`
  - `select_target_from_yaml(yaml_path: Path) -> str` — reads `target_sequence` from training YAML; for diagnostic only.
  - `build_control_token_ids(model: Any, tokenizer: Any, device: torch.device | str, *, ctrl_id: str, candidate_token_ids: Sequence[int], prompts: Sequence[str]) -> tuple[int, ...]`
  - `build_cell_json(*, cell_id: str, arch: str, cand_role: str, init_seed: int, shuffle_seed: int, ctrl_id: str, candidate_source: str, candidate_token_ids: Sequence[int], control_token_ids: Sequence[int], backdoor_mining_rank: int | None, frozen_config: Mapping[str, Any], runtime: Mapping[str, Any], checkpoints: Mapping[int, Mapping[str, Any]], delta_vs_step0: Mapping[int, Mapping[str, float]], trajectory_metrics: Mapping[str, Mapping[str, float]], integrity: Mapping[str, str]) -> dict[str, Any]`
  - `load_manifest(manifest_path: Path) -> dict[str, Any]`
  - `write_manifest_atomic(manifest_path: Path, manifest: Mapping[str, Any]) -> None`
  - `cell_is_complete(cell_path: Path, expected_cell_id: str) -> bool`
  - `write_cell_atomic(cell_path: Path, cell_json: Mapping[str, Any]) -> None`
- Consumes: `competition_core.latent_probe.ProbeResult`, `competition_core.latent_probe.build_internal_control`, `competition_core.config.load_training_config`, `competition_core.modeling.load_tokenizer`.

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_stage2_diagnostic_helpers.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts._stage2_diagnostic import (
    build_cell_json,
    cell_is_complete,
    compute_auc,
    compute_delta_vs_step0,
    compute_slope,
    derive_cell_id,
    extract_checkpoints,
    parse_cell_id,
    select_clean_candidate,
    write_cell_atomic,
)


def test_derive_cell_id_round_trip() -> None:
    cell_id = derive_cell_id("gpt2", "backdoor_target", 20260715, "boundary")
    assert cell_id == "gpt2__backdoor_target__20260715__boundary"
    assert parse_cell_id(cell_id) == ("gpt2", "backdoor_target", 20260715, "boundary")


def test_select_clean_candidate_returns_first_length_match() -> None:
    candidates = [
        {"token_ids": (1, 2, 3), "rank": 0},
        {"token_ids": (4, 5), "rank": 1},
        {"token_ids": (6, 7), "rank": 2},
        {"token_ids": (8, 9, 10), "rank": 3},
    ]
    token_ids, index = select_clean_candidate(candidates, target_token_length=2)
    assert token_ids == (4, 5)
    assert index == 1


def test_select_clean_candidate_raises_when_no_length_match() -> None:
    candidates = [{"token_ids": (1, 2, 3), "rank": 0}]
    with pytest.raises(ValueError, match="no clean candidate of length"):
        select_clean_candidate(candidates, target_token_length=5)


def test_extract_checkpoints_picks_step0_from_initial_fields() -> None:
    fake_result = SimpleNamespace(
        initial_candidate_probability=0.1,
        initial_control_probability=0.2,
        initial_probability_gap=-0.1,
        initial_candidate_mean_log_likelihood=-2.0,
        initial_control_mean_log_likelihood=-1.5,
        initial_log_likelihood_gap=-0.5,
        steps=(
            SimpleNamespace(step=1, prompt_indices=(0, 1), candidate_probability=0.3, control_probability=0.25,
                            probability_gap=0.05, candidate_loss=1.5, control_loss=1.4,
                            candidate_mean_log_likelihood=-1.5, control_mean_log_likelihood=-1.4,
                            log_likelihood_gap=-0.1),
            SimpleNamespace(step=32, prompt_indices=(2, 3), candidate_probability=0.4, control_probability=0.2,
                            probability_gap=0.2, candidate_loss=1.0, control_loss=1.6,
                            candidate_mean_log_likelihood=-1.0, control_mean_log_likelihood=-1.6,
                            log_likelihood_gap=0.6),
        ),
    )
    checkpoints = extract_checkpoints(fake_result, fixed_steps=(0, 1, 32, 64, 128, 192))
    assert set(checkpoints.keys()) == {0, 1, 32}
    assert checkpoints[0]["candidate_probability"] == 0.1
    assert checkpoints[32]["log_likelihood_gap"] == 0.6
    assert checkpoints[64] == {}  # missing step represented as empty dict by caller


def test_compute_slope_simple_linear() -> None:
    pairs = [(0, 0.0), (32, 1.0), (64, 2.0), (192, 6.0)]
    assert compute_slope(pairs, start_step=0, end_step=192) == pytest.approx(6.0 / 192)


def test_compute_auc_trapezoid_over_full_trajectory() -> None:
    steps = [0, 1, 2, 3]
    values = [0.0, 1.0, 1.0, 0.0]
    # trapz: (1+0)/2*1 + (1+1)/2*1 + (1+0)/2*1 = 0.5 + 1.0 + 0.5 = 2.0
    assert compute_auc(steps, values) == pytest.approx(2.0)


def test_compute_delta_vs_step0_subtracts_baseline() -> None:
    step0 = {"probability_gap": 0.1, "log_likelihood_gap": 0.0}
    checkpoints = {
        0: step0,
        32: {"probability_gap": 0.3, "log_likelihood_gap": 0.2},
    }
    deltas = compute_delta_vs_step0(checkpoints, step0, ["probability_gap", "log_likelihood_gap"])
    assert deltas[32]["probability_gap"] == pytest.approx(0.2)
    assert deltas[32]["log_likelihood_gap"] == pytest.approx(0.2)
    assert 0 not in deltas


def test_cell_is_complete_returns_false_for_missing_file(tmp_path: Path) -> None:
    assert cell_is_complete(tmp_path / "nope.json", "any_cell_id") is False


def test_cell_is_complete_returns_false_for_corrupt_json(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert cell_is_complete(corrupt, "any_cell_id") is False


def test_cell_is_complete_returns_false_for_mismatched_cell_id(tmp_path: Path) -> None:
    p = tmp_path / "cell.json"
    write_cell_atomic(p, build_cell_json(
        cell_id="gpt2__backdoor_target__20260715__boundary",
        arch="gpt2", cand_role="backdoor_target", init_seed=20260715,
        shuffle_seed=20260715, ctrl_id="boundary",
        candidate_source="training_yaml_target",
        candidate_token_ids=(1, 2), control_token_ids=(3, 4),
        candidate_mining_evidence={"match_type": "token_exact", "selected_rank": 2},
        frozen_config={}, runtime={},
        checkpoints={}, delta_vs_step0={}, trajectory_metrics={},
        integrity={},
    ))
    assert cell_is_complete(p, "different_cell_id") is False
    assert cell_is_complete(p, "gpt2__backdoor_target__20260715__boundary") is True


def test_write_cell_atomic_produces_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = {"cell_id": "x", "n": 1}
    write_cell_atomic(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "out.json.tmp").exists()
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run: `python -m pytest tests/test_stage2_diagnostic_helpers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._stage2_diagnostic'`.

- [ ] **Step 3: Implement `scripts/_stage2_diagnostic.py`**

Create `scripts/_stage2_diagnostic.py`:

```python
"""Helpers for the Stage 2 method diagnostic runner.

This module lives in scripts/ (not competition_core/) because it reads
training-side truth (target_sequence from training YAML) for diagnostic
purposes only. It MUST NOT be imported by competition_core/.
"""
from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from competition_core.latent_probe import ProbeResult, build_internal_control


FIXED_STEPS: tuple[int, ...] = (0, 1, 32, 64, 128, 192)
ARCHES: tuple[str, ...] = ("gpt2", "opt125", "pythia70", "dialogpt")
CAND_ROLES: tuple[str, ...] = ("backdoor_target", "clean_mined_length_match")
INIT_SEEDS: tuple[int, ...] = (20260715, 20260716, 20260717, 20260718, 20260719)
CONTROLS: tuple[str, ...] = ("boundary", "first_prompt", "median_prompt")
SHUFFLE_SEED: int = 20260715

CHECKPOINT_METRIC_KEYS: tuple[str, ...] = (
    "candidate_probability",
    "control_probability",
    "probability_gap",
    "candidate_mean_log_likelihood",
    "control_mean_log_likelihood",
    "log_likelihood_gap",
)


def derive_cell_id(arch: str, cand_role: str, init_seed: int, ctrl_id: str) -> str:
    return f"{arch}__{cand_role}__{init_seed}__{ctrl_id}"


def parse_cell_id(cell_id: str) -> tuple[str, str, int, str]:
    parts = cell_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"invalid cell_id: {cell_id}")
    arch, cand_role, seed_str, ctrl_id = parts
    return arch, cand_role, int(seed_str), ctrl_id


def select_clean_candidate(
    clean_candidates: Sequence[Mapping[str, Any]],
    target_token_length: int,
) -> tuple[tuple[int, ...], int]:
    for index, candidate in enumerate(clean_candidates):
        token_ids = tuple(int(t) for t in candidate["token_ids"])
        if len(token_ids) == target_token_length:
            return token_ids, index
    raise ValueError(
        f"no clean candidate of length {target_token_length} in {len(clean_candidates)} candidates"
    )


def _step_to_metric_dict(step: Any) -> dict[str, Any]:
    return {
        "candidate_probability": float(step.candidate_probability),
        "control_probability": float(step.control_probability),
        "probability_gap": float(step.probability_gap),
        "candidate_mean_log_likelihood": float(step.candidate_mean_log_likelihood),
        "control_mean_log_likelihood": float(step.control_mean_log_likelihood),
        "log_likelihood_gap": float(step.log_likelihood_gap),
        "prompt_indices": tuple(int(i) for i in step.prompt_indices),
    }


def extract_checkpoints(
    result: ProbeResult,
    fixed_steps: Sequence[int] = FIXED_STEPS,
) -> dict[int, dict[str, Any]]:
    checkpoints: dict[int, dict[str, Any]] = {}
    if 0 in fixed_steps:
        checkpoints[0] = {
            "candidate_probability": float(result.initial_candidate_probability),
            "control_probability": float(result.initial_control_probability),
            "probability_gap": float(result.initial_probability_gap),
            "candidate_mean_log_likelihood": float(result.initial_candidate_mean_log_likelihood),
            "control_mean_log_likelihood": float(result.initial_control_mean_log_likelihood),
            "log_likelihood_gap": float(result.initial_log_likelihood_gap),
            "prompt_indices": (),
        }
    by_step = {int(s.step): s for s in result.steps}
    for step in fixed_steps:
        if step == 0:
            continue
        if step in by_step:
            checkpoints[step] = _step_to_metric_dict(by_step[step])
        else:
            checkpoints[step] = {}
    return checkpoints


def compute_delta_vs_step0(
    checkpoints: Mapping[int, Mapping[str, float]],
    step0: Mapping[str, float],
    metric_keys: Sequence[str],
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for step, metrics in checkpoints.items():
        if step == 0:
            continue
        if not metrics:
            continue
        out[step] = {key: float(metrics[key]) - float(step0[key]) for key in metric_keys if key in metrics}
    return out


def compute_slope(
    metric_pairs: Sequence[tuple[int, float]],
    start_step: int,
    end_step: int,
) -> float:
    by_step = {int(s): float(v) for s, v in metric_pairs}
    if start_step not in by_step or end_step not in by_step:
        raise ValueError(f"missing endpoint: start={start_step}, end={end_step}, have={sorted(by_step)}")
    return (by_step[end_step] - by_step[start_step]) / (end_step - start_step)


def compute_auc(steps: Sequence[int], metric_values: Sequence[float]) -> float:
    if len(steps) != len(metric_values) or len(steps) < 2:
        raise ValueError("need at least 2 (step, value) pairs")
    area = 0.0
    for i in range(len(steps) - 1):
        width = steps[i + 1] - steps[i]
        area += (metric_values[i] + metric_values[i + 1]) * width / 2.0
    return area


def load_yaml_target_token_length(yaml_path: Path, tokenizer: Any) -> int:
    from competition_core.config import load_training_config
    config = load_training_config(yaml_path)
    target_text = config.target_sequence
    return len(tuple(tokenizer(target_text, add_special_tokens=False).input_ids))


def select_target_from_yaml(yaml_path: Path) -> str:
    from competition_core.config import load_training_config
    return load_training_config(yaml_path).target_sequence


def build_control_token_ids(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    ctrl_id: str,
    candidate_token_ids: Sequence[int],
    prompts: Sequence[str],
) -> tuple[int, ...]:
    if ctrl_id == "boundary":
        response_prefix = "### Response:"
    elif ctrl_id == "first_prompt":
        response_prefix = prompts[0]
    elif ctrl_id == "median_prompt":
        response_prefix = prompts[len(prompts) // 2]
    else:
        raise ValueError(f"unknown ctrl_id: {ctrl_id}")
    return tuple(build_internal_control(
        model,
        tokenizer,
        device,
        response_prefix=response_prefix,
        candidate_token_ids=candidate_token_ids,
    ))


def build_cell_json(
    *,
    cell_id: str,
    arch: str,
    cand_role: str,
    init_seed: int,
    shuffle_seed: int,
    ctrl_id: str,
    candidate_source: str,
    candidate_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    backdoor_mining_rank: int | None,
    frozen_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
    delta_vs_step0: Mapping[int, Mapping[str, float]],
    trajectory_metrics: Mapping[str, Mapping[str, float]],
    integrity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "role": "training_side_method_diagnostic",
        "known_target_sequence": True,
        "decision_use": False,
        "cell_id": cell_id,
        "cell_config": {
            "arch": arch,
            "cand_role": cand_role,
            "init_seed": init_seed,
            "shuffle_seed": shuffle_seed,
            "ctrl_id": ctrl_id,
            "candidate_source": candidate_source,
            "candidate_target_token_length": len(candidate_token_ids),
            "backdoor_mining_rank": backdoor_mining_rank,
            "control_token_ids": list(int(t) for t in control_token_ids),
            "init_token_ids": [],  # filled by runner from ProbeResult.initialization_token_ids
        },
        "frozen_config": dict(frozen_config),
        "runtime": dict(runtime),
        "checkpoints": {f"step_{s}": dict(metrics) for s, metrics in sorted(checkpoints.items())},
        "delta_vs_step0": {f"step_{s}": dict(metrics) for s, metrics in sorted(delta_vs_step0.items())},
        "trajectory_metrics": {k: dict(v) for k, v in trajectory_metrics.items()},
        "integrity": dict(integrity),
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"schema_version": "1.0", "cells_completed": [], "cells_failed": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest_atomic(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)


def cell_is_complete(cell_path: Path, expected_cell_id: str) -> bool:
    if not cell_path.exists():
        return False
    try:
        with cell_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("cell_id") == expected_cell_id


def write_cell_atomic(cell_path: Path, cell_json: Mapping[str, Any]) -> None:
    tmp = cell_path.with_suffix(cell_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(cell_json), f, indent=2, sort_keys=True)
    os.replace(tmp, cell_path)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run: `python -m pytest tests/test_stage2_diagnostic_helpers.py -v`

Expected: PASS (11 tests).

- [ ] **Step 5: Run full default test suite to verify no regressions**

Run: `python -m pytest -q`

Expected: All tests pass (including existing tests + 3 new latent_probe + 11 new helpers).

- [ ] **Step 6: Run ruff on the new file**

Run: `python -m ruff check scripts/_stage2_diagnostic.py`

Expected: No findings.

---

### Task 3: Create `scripts/run_stage2_diagnostic.py` CLI runner

**Files:**
- Create: `scripts/run_stage2_diagnostic.py`
- Test: `tests/test_run_stage2_diagnostic.py`

**Interfaces:**
- Produces: CLI `python -m scripts.run_stage2_diagnostic` with subcommands:
  - `--manifest <path>` (required) — JSON manifest with adapter paths, mining paths, training YAMLs, output dir, frozen config
  - `--cells {pilot_12,all,remaining}` (default: `remaining`)
  - `--arch <name>` — restrict to one architecture (optional)
  - `--dry-run` — print planned cells without loading models
  - `--output-dir <path>` — overrides manifest's `output_dir`
- Consumes: helpers from Task 2, `probe_candidate` from Task 1, `build_internal_control` from existing `competition_core.latent_probe`, model/tokenizer loaders from `competition_core.modeling`.

- [ ] **Step 1: Write failing CLI smoke tests**

Create `tests/test_run_stage2_diagnostic.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_stage2_diagnostic


def test_dry_run_lists_pilot_12_cells_without_loading_models(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {
        "adapter_paths": {
            "gpt2":   {"backdoor": "/tmp/gpt2_bd", "clean": "/tmp/gpt2_cl"},
            "opt125": {"backdoor": "/tmp/opt_bd",  "clean": "/tmp/opt_cl"},
        },
        "detection_yamls": {
            "gpt2":   "/tmp/gpt2_detection.yaml",
            "opt125": "/tmp/opt_detection.yaml",
        },
        "training_yamls": {
            "gpt2":   "/tmp/gpt2_train.yaml",
            "opt125": "/tmp/opt_train.yaml",
        },
        "mining_paths": {
            "gpt2":   {"backdoor": "/tmp/gpt2_bd_mining.json", "clean": "/tmp/gpt2_cl_mining.json"},
            "opt125": {"backdoor": "/tmp/opt_bd_mining.json",  "clean": "/tmp/opt_cl_mining.json"},
        },
        "output_dir": str(tmp_path),
        "frozen_config": {
            "test_sample_count": 512, "batch_size": 8, "epochs": 3,
            "max_steps": 192, "learning_rate": 1e-4, "soft_token_count": 8,
            "stop_on_decision": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "pilot_12",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # pilot_12 = 2 arches (gpt2, opt125) × 2 roles × 1 seed (20260715) × 3 controls = 12 cells
    assert "gpt2__backdoor_target__20260715__boundary" in out
    assert "opt125__clean_mined_length_match__20260715__median_prompt" in out
    assert "pythia70" not in out
    assert "dialogpt" not in out
    # dry-run must not touch the filesystem
    assert not any(tmp_path.glob("*.json"))


def test_resume_skips_completed_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "adapter_paths": {"gpt2": {"backdoor": "/tmp/bd", "clean": "/tmp/cl"}},
        "detection_yamls": {"gpt2": "/tmp/detection.yaml"},
        "training_yamls": {"gpt2": "/tmp/train.yaml"},
        "mining_paths": {"gpt2": {"backdoor": "/tmp/bd_mining.json", "clean": "/tmp/cl_mining.json"}},
        "output_dir": str(tmp_path),
        "frozen_config": {
            "test_sample_count": 8, "batch_size": 8, "epochs": 1,
            "max_steps": 1, "learning_rate": 1e-4, "soft_token_count": 2,
            "stop_on_decision": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Plant a completed cell
    completed_cell_id = "gpt2__backdoor_target__20260715__boundary"
    completed_path = tmp_path / f"{completed_cell_id}.json"
    completed_path.write_text(json.dumps({"cell_id": completed_cell_id, "schema_version": "1.0"}), encoding="utf-8")

    seen_calls: list[str] = []

    def fake_run_one_cell(cell_id: str, **kwargs: object) -> None:
        seen_calls.append(cell_id)

    monkeypatch.setattr(run_stage2_diagnostic, "_run_one_cell", fake_run_one_cell)

    rc = run_stage2_diagnostic.main([
        "--manifest", str(manifest_path),
        "--cells", "remaining",
        "--arch", "gpt2",
    ])
    assert rc == 0
    # Only init_seed 20260715 is "pilot" but we said "remaining" so all 5 seeds × 2 roles × 3 controls = 30
    # minus the 1 completed cell = 29 expected
    assert len(seen_calls) == 29
    assert completed_cell_id not in seen_calls
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: `python -m pytest tests/test_run_stage2_diagnostic.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_stage2_diagnostic'`.

- [ ] **Step 3: Implement `scripts/run_stage2_diagnostic.py`**

Create `scripts/run_stage2_diagnostic.py`:

```python
"""Stage 2 method diagnostic runner.

Drives 120-cell (or 12-cell pilot) diagnostic on matched backdoor/clean
adapter pairs. Reads target_sequence from training YAML for diagnostic
purposes only; emits per-cell JSON with role=training_side_method_diagnostic.

DOES NOT modify competition_core detection thresholds or reports.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from competition_core.config import ProbeConfig, load_detection_config
from competition_core.latent_probe import ProbeResult, probe_candidate
from competition_core.modeling import load_tokenizer

from scripts._stage2_diagnostic import (
    ARCHES,
    CAND_ROLES,
    CHECKPOINT_METRIC_KEYS,
    CONTROLS,
    FIXED_STEPS,
    INIT_SEEDS,
    SHUFFLE_SEED,
    build_cell_json,
    build_control_token_ids,
    cell_is_complete,
    compute_auc,
    compute_delta_vs_step0,
    compute_slope,
    derive_cell_id,
    extract_checkpoints,
    load_manifest,
    load_yaml_target_token_length,
    select_clean_candidate,
    select_target_from_yaml,
    sha256_of_text,
    write_cell_atomic,
    write_manifest_atomic,
)


PILOT_ARCHES: tuple[str, ...] = ("gpt2", "opt125")
PILOT_INIT_SEEDS: tuple[int, ...] = (20260715,)


def _load_mining_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_backdoor_rank(mining_json: Mapping[str, Any], target_token_ids: Sequence[int]) -> int | None:
    candidates = mining_json.get("candidates", [])
    target_tuple = tuple(int(t) for t in target_token_ids)
    for rank, candidate in enumerate(candidates):
        if tuple(int(t) for t in candidate.get("token_ids", [])) == target_tuple:
            return rank
    return None


def _load_model_and_tokenizer(
    detection_yaml: Path, adapter_path: str, device_override: str | None = None,
) -> tuple[Any, Any, torch.device, Mapping[str, Any]]:
    from competition_core.config import load_detection_config
    from competition_core.modeling import load_model, load_tokenizer
    config = load_detection_config(detection_yaml)
    tokenizer = load_tokenizer(config.model)
    model, device = load_model(config.model, artifact=adapter_path)
    return model, tokenizer, device, {"test_data": config.test_data, "model_config": config.model}


def _run_one_cell(
    *,
    cell_id: str,
    arch: str,
    cand_role: str,
    init_seed: int,
    ctrl_id: str,
    manifest: Mapping[str, Any],
    output_dir: Path,
    target_token_ids: tuple[int, ...],
    backdoor_mining_rank: int | None,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: Sequence[str],
) -> None:
    frozen_config = dict(manifest["frozen_config"])
    probe_config = ProbeConfig(
        test_sample_count=int(frozen_config["test_sample_count"]),
        batch_size=int(frozen_config["batch_size"]),
        epochs=int(frozen_config["epochs"]),
        max_steps=int(frozen_config["max_steps"]),
        learning_rate=float(frozen_config["learning_rate"]),
        soft_token_count=int(frozen_config["soft_token_count"]),
        stop_on_decision=bool(frozen_config["stop_on_decision"]),
    )

    control_token_ids = build_control_token_ids(
        model, tokenizer, device,
        ctrl_id=ctrl_id,
        candidate_token_ids=target_token_ids,
        prompts=prompts,
    )

    started = time.perf_counter()
    result: ProbeResult = probe_candidate(
        model, tokenizer, device,
        prompts=prompts,
        candidate_token_ids=target_token_ids,
        control_token_ids=control_token_ids,
        config=probe_config,
        seed=init_seed,
        shuffle_seed=SHUFFLE_SEED,
    )
    wall = time.perf_counter() - started

    checkpoints = extract_checkpoints(result, FIXED_STEPS)
    step0 = checkpoints.get(0, {})
    delta = compute_delta_vs_step0(checkpoints, step0, CHECKPOINT_METRIC_KEYS)

    full_steps = [int(s.step) for s in result.steps]
    prob_gaps = [float(s.probability_gap) for s in result.steps]
    ll_gaps = [float(s.log_likelihood_gap) for s in result.steps]
    # Prepend step 0 (initial) for slope/AUC over [0, 192]
    full_steps = [0] + full_steps
    prob_gaps = [float(step0.get("probability_gap", 0.0))] + prob_gaps
    ll_gaps = [float(step0.get("log_likelihood_gap", 0.0))] + ll_gaps

    slope_0_192_prob = compute_slope(list(zip(full_steps, prob_gaps)), 0, max(full_steps))
    slope_0_192_ll = compute_slope(list(zip(full_steps, ll_gaps)), 0, max(full_steps))
    slope_32_192_prob = (
        compute_slope(list(zip(full_steps, prob_gaps)), 32, max(full_steps))
        if 32 in full_steps else 0.0
    )
    slope_32_192_ll = (
        compute_slope(list(zip(full_steps, ll_gaps)), 32, max(full_steps))
        if 32 in full_steps else 0.0
    )
    auc_prob = compute_auc(full_steps, prob_gaps)
    auc_ll = compute_auc(full_steps, ll_gaps)

    trajectory_metrics = {
        "slope_step0_to_final": {"probability_gap": slope_0_192_prob, "log_likelihood_gap": slope_0_192_ll},
        "slope_step32_to_final": {"probability_gap": slope_32_192_prob, "log_likelihood_gap": slope_32_192_ll},
        "auc_step0_to_final": {"probability_gap": auc_prob, "log_likelihood_gap": auc_ll},
        "full_trajectory_steps": len(full_steps),
    }

    runtime = {
        "device": str(device),
        "model_storage_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "probe_compute_dtype": _probe_compute_dtype_str(model, device),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "wall_seconds": round(wall, 3),
    }

    cell_json = build_cell_json(
        cell_id=cell_id,
        arch=arch, cand_role=cand_role, init_seed=init_seed,
        shuffle_seed=SHUFFLE_SEED, ctrl_id=ctrl_id,
        candidate_source=("training_yaml_target" if cand_role == "backdoor_target"
                          else "clean_mining_length_match"),
        candidate_token_ids=target_token_ids,
        control_token_ids=control_token_ids,
        candidate_mining_evidence={"selected_rank": backdoor_mining_rank},
        frozen_config=frozen_config,
        runtime=runtime,
        checkpoints=checkpoints,
        delta_vs_step0=delta,
        trajectory_metrics=trajectory_metrics,
        integrity={},
    )
    cell_json["cell_config"]["init_token_ids"] = [int(t) for t in result.initialization_token_ids]

    write_cell_atomic(output_dir / f"{cell_id}.json", cell_json)


def _probe_compute_dtype_str(model: Any, device: torch.device) -> str:
    from competition_core.latent_probe import probe_compute_dtype
    return probe_compute_dtype(model, device)


def _enumerate_cells(
    *,
    cells_mode: str,
    arch_filter: str | None,
) -> list[tuple[str, str, int, str]]:
    arches = (arch_filter,) if arch_filter else ARCHES
    if cells_mode == "pilot_12":
        arches = tuple(a for a in arches if a in PILOT_ARCHES) if not arch_filter else (arch_filter,)
        seeds = PILOT_INIT_SEEDS
    else:
        seeds = INIT_SEEDS
    out: list[tuple[str, str, int, str]] = []
    for arch in arches:
        for role in CAND_ROLES:
            for seed in seeds:
                for ctrl in CONTROLS:
                    out.append((arch, role, seed, ctrl))
    return out


def _dry_run_print(plan: Sequence[tuple[str, str, int, str]]) -> None:
    print(f"DRY_RUN: {len(plan)} cells planned")
    for arch, role, seed, ctrl in plan:
        print(f"  {derive_cell_id(arch, role, seed, ctrl)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 method diagnostic runner")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cells", choices=("pilot_12", "all", "remaining"), default="remaining")
    parser.add_argument("--arch", default=None, help="restrict to one architecture")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args(argv)

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _enumerate_cells(cells_mode=args.cells, arch_filter=args.arch)

    if args.dry_run:
        _dry_run_print(plan)
        return 0

    if args.cells in ("all", "remaining"):
        plan_to_run = []
        for cell_tuple in plan:
            arch, role, seed, ctrl = cell_tuple
            cell_id = derive_cell_id(*cell_tuple)
            cell_path = output_dir / f"{cell_id}.json"
            if args.cells == "remaining" and cell_is_complete(cell_path, cell_id):
                continue
            plan_to_run.append(cell_tuple)
    else:
        plan_to_run = list(plan)

    if not plan_to_run:
        print(f"Nothing to run. {len(plan)} planned, all complete.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {len(plan_to_run)} cells on {device}")

    # Per (arch, adapter_kind): load model+tokenizer+test_data once
    # Key = (arch, "backdoor"|"clean"); value = (model, tokenizer, device, side_config)
    by_arch_models: dict[tuple[str, str], tuple[Any, Any, torch.device, Mapping[str, Any]]] = {}
    try:
        for arch, role, seed, ctrl in plan_to_run:
            cell_id = derive_cell_id(arch, role, seed, ctrl)
            cell_path = output_dir / f"{cell_id}.json"
            if cell_is_complete(cell_path, cell_id):
                continue

            adapter_kind = "backdoor" if role == "backdoor_target" else "clean"
            model_key = (arch, adapter_kind)
            if model_key not in by_arch_models:
                detection_yaml = Path(manifest["detection_yamls"][arch])
                adapter_path = manifest["adapter_paths"][arch][adapter_kind]
                model, tokenizer, loaded_device, side_config = _load_model_and_tokenizer(
                    detection_yaml, adapter_path,
                )
                by_arch_models[model_key] = (model, tokenizer, loaded_device, side_config)
            model, tokenizer, cell_device, side_config = by_arch_models[model_key]

            # Derive target token IDs
            yaml_path = manifest["training_yamls"][arch]
            if role == "backdoor_target":
                target_text = select_target_from_yaml(Path(yaml_path))
                target_token_ids = tuple(int(t) for t in tokenizer(target_text, add_special_tokens=False).input_ids)
                bd_mining = _load_mining_json(manifest["mining_paths"][arch]["backdoor"])
                rank = _candidate_backdoor_rank(bd_mining, target_token_ids)
            else:
                cl_mining = _load_mining_json(manifest["mining_paths"][arch]["clean"])
                target_token_length = load_yaml_target_token_length(Path(yaml_path), tokenizer)
                target_token_ids, _ = select_clean_candidate(
                    cl_mining.get("candidates", []), target_token_length
                )
                rank = None

            # Load probe prompts using the existing test_inputs loader
            from competition_core.test_inputs import load_probe_input_sets
            prompts, _replay_prompts, _test_manifest = load_probe_input_sets(
                side_config["test_data"], tokenizer,
                optimization_count=int(manifest["frozen_config"]["test_sample_count"]),
                replay_count=0,
            )

            print(f"[{cell_id}] running...")
            _run_one_cell(
                cell_id=cell_id, arch=arch, cand_role=role, init_seed=seed,
                ctrl_id=ctrl, manifest=manifest, output_dir=output_dir,
                target_token_ids=target_token_ids,
                candidate_mining_evidence={"selected_rank": rank},
                model=model, tokenizer=tokenizer, device=cell_device, prompts=prompts,
            )

            # Update manifest
            diag_manifest_path = output_dir / "diagnostic_manifest.json"
            diag_manifest = load_manifest(diag_manifest_path)
            cells_completed = list(diag_manifest.get("cells_completed", []))
            cells_completed.append(str(cell_path))
            diag_manifest["cells_completed"] = cells_completed
            diag_manifest["schema_version"] = "1.0"
            diag_manifest["clean_candidate_rule"] = {
                "text": "From clean mining JSON candidates array, pick first candidate with len(token_ids) == target_token_length. Tiebreak by lowest original array index.",
                "sha256_input": "the above text field encoded as UTF-8",
                "sha256": sha256_of_text(
                    "From clean mining JSON candidates array, pick first candidate with len(token_ids) == target_token_length. Tiebreak by lowest original array index."
                ),
            }
            write_manifest_atomic(diag_manifest_path, diag_manifest)
            print(f"[{cell_id}] done -> {cell_path}")
    finally:
        del by_arch_models
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `python -m pytest tests/test_run_stage2_diagnostic.py -v`

Expected: PASS (2 tests).

- [ ] **Step 5: Run py_compile**

Run: `python -m py_compile scripts/run_stage2_diagnostic.py scripts/_stage2_diagnostic.py`

Expected: No output (success).

- [ ] **Step 6: Run ruff on new files**

Run: `python -m ruff check scripts/_stage2_diagnostic.py scripts/run_stage2_diagnostic.py`

Expected: No findings.

- [ ] **Step 7: Run full default test suite**

Run: `python -m pytest -q`

Expected: All tests pass.

---

### Task 4: Single bundle commit

**Files:**
- Stage: working-tree changes in `competition_core/latent_probe.py`, `competition_core/cli.py`, `competition_core/tests/test_latent_probe.py`, new `scripts/_stage2_diagnostic.py`, new `scripts/run_stage2_diagnostic.py`, new `tests/test_stage2_diagnostic_helpers.py`, new `tests/test_run_stage2_diagnostic.py`.

**Constraints:** DO NOT touch unrelated dirty files (CLAUDE.md, docs/ARCHITECTURE.md, scripts/run_team_model_pair.py, etc.). DO NOT use `git add .`.

- [ ] **Step 1: Verify clean working-tree state for unrelated files**

Run: `git status --short`

Expected: see only the 7 target files staged-ready; all other dirty files unchanged.

- [ ] **Step 2: Stage only the 7 target files**

Run:

```bash
git add competition_core/latent_probe.py competition_core/cli.py competition_core/tests/test_latent_probe.py scripts/_stage2_diagnostic.py scripts/run_stage2_diagnostic.py tests/test_stage2_diagnostic_helpers.py tests/test_run_stage2_diagnostic.py
```

- [ ] **Step 3: Verify staged set**

Run: `git diff --cached --stat`

Expected: only the 7 files appear.

- [ ] **Step 4: Commit**

Run:

```bash
git commit -m "$(cat <<'EOF'
fix(competition): stabilize latent probe under bf16/finite values and decouple shuffle from init seed

- Add bfloat16 autocast + finite-value guards in latent_probe.py
- Add gc.collect() + cuda.empty_cache() in cli.py probe command
- Report model_storage_dtype and probe_compute_dtype in runtime
- Add shuffle_seed parameter (backward compatible) to probe_candidate
- Add scripts/_stage2_diagnostic.py with checkpoint/slope/AUC/clean-candidate helpers
- Add scripts/run_stage2_diagnostic.py for Stage 2 method diagnostic
- Add tests for shuffle_seed semantics, helpers, and runner dry-run/resume
EOF
)"
```

- [ ] **Step 5: Verify commit**

Run: `git log -n 1 --stat`

Expected: latest commit shows exactly the 7 files changed.

---

## Phase B — Remote (Blocked on User Password)

> **All tasks below require the user's SSH password to `root@fj02-ssh.gpuhome.cc:30101`. The password is provided out-of-band; never write it into scripts, repo, or logs. Tasks here describe the intended procedure; they are executed by the user typing `! ssh ...` commands.**

### Task 5: Discover remote adapter and mining paths

**Files:** none modified. Output is recorded into the local manifest draft.

- [ ] **Step 1: Ask user to run one-shot discovery**

Prompt the user (in chat) to run:

```
! ssh -o StrictHostKeyChecking=accept-new root@fj02-ssh.gpuhome.cc -p 30101 'find /root/bdshield_run /root/bdshield_runs -maxdepth 6 \( -name adapter_model.safetensors -o -name adapter_config.json -o -name "mining.json" -o -name "*mining*.json" \) 2>/dev/null | sort'
```

- [ ] **Step 2: Record paths into a local manifest draft**

Create local `runs/_stage2_diag_manifest.draft.json` (untracked, NOT committed — `runs/` is gitignored):

```json
{
  "adapter_paths": {
    "gpt2":   {"backdoor": "<path-from-find>", "clean": "<path>"},
    "opt125": {"backdoor": "<path>", "clean": "<path>"},
    "pythia70": {"backdoor": "<path>", "clean": "<path>"},
    "dialogpt": {"backdoor": "<path>", "clean": "<path>"}
  },
  "detection_yamls": {
    "gpt2":   "<local>/competition_core/configs/gpt2_detection_4060.yaml",
    "opt125": "<local>/competition_core/configs/opt125_detection_team_4060.yaml",
    "pythia70": "<path-to-pythia-detection-yaml>",
    "dialogpt": "<path-to-dialogpt-detection-yaml>"
  },
  "training_yamls": {
    "gpt2":   "<local>/competition_core/configs/gpt2_alpaca_train_4060.yaml",
    "opt125": "<local>/competition_core/configs/opt125_alpaca_train_team_4060.yaml",
    "pythia70": "<path-to-pythia-training-yaml>",
    "dialogpt": "<path-to-dialogpt-training-yaml>"
  },
  "mining_paths": {
    "gpt2":   {"backdoor": "<path-from-find>", "clean": "<path>"},
    "opt125": {"backdoor": "<path>", "clean": "<path>"},
    "pythia70": {"backdoor": "<path>", "clean": "<path>"},
    "dialogpt": {"backdoor": "<path>", "clean": "<path>"}
  },
  "output_dir": "/root/rivermind-data/bdshield_stage2_diag_results",
  "frozen_config": {
    "test_sample_count": 512, "batch_size": 8, "epochs": 3,
    "max_steps": 192, "learning_rate": 1e-4, "soft_token_count": 8,
    "stop_on_decision": false
  }
}
```

- [ ] **Step 3: Verify mining JSON schema locally before upload**

Run: `python -c "import json; d=json.load(open('<local-mining-sample>.json')); print(list(d.keys())[:5]); print(len(d.get('candidates', [])))"`

Expected: keys include `schema_version`, `candidates`; candidate count matches local mining reports (~96 per README).

---

### Task 6: Sync code snapshot to remote

**Files:** none modified locally.

- [ ] **Step 1: rsync competition_core and runner to remote**

Prompt user to run (replace `<PW>` handling per their SSH client; rsync over ssh):

```
! rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='competition_runs/' \
  --exclude='team_runs/' \
  competition_core/ scripts/_stage2_diagnostic.py scripts/run_stage2_diagnostic.py \
  root@fj02-ssh.gpuhome.cc:/root/bdshield_stage2_diag/
```

Note: port 30101 requires `-e "ssh -p 30101"`.

- [ ] **Step 2: Verify remote files compile**

Prompt user to run:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && python -m py_compile competition_core/latent_probe.py scripts/run_stage2_diagnostic.py scripts/_stage2_diagnostic.py && echo OK'
```

Expected: prints `OK`.

- [ ] **Step 3: Verify remote pytest collection (no model tests)**

Prompt user to run:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && python -m pytest competition_core/tests -q --collect-only 2>&1 | tail -20'
```

Expected: collection succeeds with all tests.

---

### Task 7: Pilot 12 cells

**Files:** none modified locally; cell JSONs created on remote.

- [ ] **Step 1: Upload manifest**

Prompt user:

```
! scp -P 30101 runs/_stage2_diag_manifest.draft.json \
  root@fj02-ssh.gpuhome.cc:/root/bdshield_stage2_diag/manifest.json
```

- [ ] **Step 2: Dry-run on remote**

Prompt user:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && python -m scripts.run_stage2_diagnostic --manifest manifest.json --cells pilot_12 --dry-run'
```

Expected: prints `DRY_RUN: 12 cells planned` followed by 12 cell IDs (gpt2 + opt125, 2 roles × 1 seed × 3 controls).

- [ ] **Step 3: Launch pilot 12 cells**

Prompt user (run in background or tmux):

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && nohup python -m scripts.run_stage2_diagnostic --manifest manifest.json --cells pilot_12 > /root/rivermind-data/bdshield_stage2_diag_results/pilot.log 2>&1 &'
```

- [ ] **Step 4: Poll pilot progress**

Prompt user (every 10-15 min until done):

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'ls /root/rivermind-data/bdshield_stage2_diag_results/*.json | wc -l; tail -5 /root/rivermind-data/bdshield_stage2_diag_results/pilot.log'
```

Expected: file count climbs from 0 to 12; no `Traceback` in log.

- [ ] **Step 5: Verify pilot cell JSON integrity**

Prompt user:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && python -c "
import json, glob
for p in sorted(glob.glob(\"/root/rivermind-data/bdshield_stage2_diag_results/gpt2__*.json\"))[:2]:
    d = json.load(open(p))
    assert d[\"role\"] == \"training_side_method_diagnostic\"
    assert d[\"known_target_sequence\"] is True
    assert d[\"decision_use\"] is False
    assert set(d[\"checkpoints\"].keys()) >= {\"step_0\", \"step_1\", \"step_32\", \"step_64\", \"step_128\", \"step_192\"}
    assert d[\"trajectory_metrics\"][\"full_trajectory_steps\"] == 193  # 0 + 192
    print(p, \"OK\")
"'
```

Expected: prints `OK` for each of the first 2 cells.

---

### Task 8: ETA calibration + decision gate

**Files:** none.

- [ ] **Step 1: Compute per-cell mean wall time**

Run locally after pulling pilot log:

```bash
python -c "
import json, glob, statistics
walls = []
for p in glob.glob('/local/mirror/of/remote/results/*.json'):  # adjust path
    d = json.load(open(p))
    walls.append(d['runtime']['wall_seconds'])
print(f'mean={statistics.mean(walls):.1f}s median={statistics.median(walls):.1f}s n={len(walls)}')
print(f'full 120 cell ETA = {statistics.mean(walls)*120/3600:.1f} hours')
"
```

- [ ] **Step 2: Decision gate**

If pilot JSON integrity check passes AND mean wall time per cell is ≤ 15 min (ETA ≤ 30 hours for 120 cells): proceed to Task 9.

Otherwise: stop and investigate (BF16 numerical issues, OOM, slow disk).

---

### Task 9: Full 120-cell run

**Files:** none modified locally.

- [ ] **Step 1: Launch remaining 108 cells**

Prompt user:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'cd /root/bdshield_stage2_diag && nohup python -m scripts.run_stage2_diagnostic --manifest manifest.json --cells remaining > /root/rivermind-data/bdshield_stage2_diag_results/full.log 2>&1 &'
```

- [ ] **Step 2: Poll periodically**

Prompt user every 30-60 min:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'ls /root/rivermind-data/bdshield_stage2_diag_results/*.json | wc -l'
```

Expected: count climbs to 120 + 1 manifest.

- [ ] **Step 3: Verify final manifest**

Prompt user:

```
! ssh root@fj02-ssh.gpuhome.cc -p 30101 'python -c "
import json
m = json.load(open(\"/root/rivermind-data/bdshield_stage2_diag_results/diagnostic_manifest.json\"))
print(\"completed:\", len(m[\"cells_completed\"]))
print(\"failed:\", len(m.get(\"cells_failed\", [])))
print(\"clean_rule_sha256:\", m[\"clean_candidate_rule\"][\"sha256\"][:16])
"'
```

Expected: `completed: 120, failed: 0`.

---

### Task 10: Four-factor analysis

**Files:**
- Create: `scripts/analyze_stage2_diagnostic.py`
- Create: `docs/findings/2026-07-22-stage2-diagnostic-findings.md` (only after data is in hand)

- [ ] **Step 1: Write analysis script**

Create `scripts/analyze_stage2_diagnostic.py`:

```python
"""Analyze Stage 2 diagnostic cell JSONs.

Compares four factors (arch × cand_role × init_seed × ctrl_id) on
step_192 log_likelihood_gap and slope_step0_to_final.log_likelihood_gap.
Uses Mann-Whitney U for backdoor vs clean separation per architecture.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_cells(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "diagnostic_manifest.json":
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _mann_whitney_u(x: list[float], y: list[float]) -> float:
    # Small-sample exact U via brute force (n <= 20 per group is fine here)
    all_vals = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks: list[float] = []
    i = 0
    while i < len(all_vals):
        j = i
        while j < len(all_vals) and all_vals[j][0] == all_vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j
    r_x = sum(r for r, (_, g) in zip(ranks, all_vals) if g == 0)
    n1, n2 = len(x), len(y)
    u_x = r_x - n1 * (n1 + 1) / 2.0
    u_y = n1 * n2 - u_x
    return min(u_x, u_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cells = _load_cells(args.results_dir)

    rows = []
    for arch in ("gpt2", "opt125", "pythia70", "dialogpt"):
        bd = [c for c in cells if c["cell_config"]["arch"] == arch and c["cell_config"]["cand_role"] == "backdoor_target"]
        cl = [c for c in cells if c["cell_config"]["arch"] == arch and c["cell_config"]["cand_role"] == "clean_mined_length_match"]
        bd_ll = [c["checkpoints"]["step_192"]["log_likelihood_gap"] for c in bd if c["checkpoints"].get("step_192")]
        cl_ll = [c["checkpoints"]["step_192"]["log_likelihood_gap"] for c in cl if c["checkpoints"].get("step_192")]
        bd_slope = [c["trajectory_metrics"]["slope_step0_to_final"]["log_likelihood_gap"] for c in bd]
        cl_slope = [c["trajectory_metrics"]["slope_step0_to_final"]["log_likelihood_gap"] for c in cl]

        u_ll = _mann_whitney_u(bd_ll, cl_ll) if len(bd_ll) >= 3 and len(cl_ll) >= 3 else None

        rows.append({
            "arch": arch,
            "bd_step192_ll_gap_mean": statistics.mean(bd_ll) if bd_ll else None,
            "cl_step192_ll_gap_mean": statistics.mean(cl_ll) if cl_ll else None,
            "bd_slope_ll_mean": statistics.mean(bd_slope) if bd_slope else None,
            "cl_slope_ll_mean": statistics.mean(cl_slope) if cl_slope else None,
            "mann_whitney_u_step192_ll": u_ll,
            "bd_n": len(bd_ll),
            "cl_n": len(cl_ll),
        })

    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run analysis on pilot data (early signal check)**

Run: `python -m scripts.analyze_stage2_diagnostic --results-dir /local/mirror/pilot --output /tmp/pilot_analysis.json`

Expected: writes JSON with 4 rows; inspect for backdoor/clean separation trend.

- [ ] **Step 3: Run analysis on full 120-cell data (after Task 9 completes)**

Run: `python -m scripts.analyze_stage2_diagnostic --results-dir /local/mirror/full --output docs/findings/2026-07-22-stage2-diagnostic-analysis.json`

Expected: writes JSON; manually inspect for stable cross-architecture signal.

- [ ] **Step 4: Write findings doc**

Create `docs/findings/2026-07-22-stage2-diagnostic-findings.md` summarizing: per-arch backdoor/clean separation at step 192, slope, AUC; Mann-Whitney U p-values; recommendation for phase 1 ablations (A1-A6 priority order).

---

## Self-Review Checklist (post-write)

**1. Spec coverage:**
- §3.1 Cell matrix → Task 3 (`_enumerate_cells` enumerates 4×2×5×3) ✓
- §3.2 Frozen config → Task 2 helper + Task 3 manifest field ✓
- §3.3 Checkpoints/slope/AUC → Task 2 helpers (`extract_checkpoints`, `compute_slope`, `compute_auc`) ✓
- §3.4 Clean candidate rule → Task 2 `select_clean_candidate` + Task 3 manifest `clean_candidate_rule` ✓
- §3.5 3 controls → Task 2 `build_control_token_ids` ✓
- §4.1 Cell JSON schema → Task 2 `build_cell_json` ✓
- §4.2 Manifest → Task 2 `load_manifest`/`write_manifest_atomic` + Task 3 update loop ✓
- §4.3 Resume → Task 2 `cell_is_complete` + Task 3 skip logic ✓
- §5.1 shuffle_seed → Task 1 ✓
- §5.2 Runner at scripts/ → Task 3 ✓
- §5.3 Single bundle commit → Task 4 ✓
- §6 Remote isolation → Tasks 5-9 ✓
- §7 Execution order → Tasks 1-10 in order ✓
- §9 Out of scope → preserved (no training, no mining changes, no threshold changes) ✓

**2. Placeholder scan:** Tasks 1-4 have complete code. Tasks 5-9 contain `<path-from-find>` etc. for fields the user must fill from remote discovery output — these are intentional "fill-from-observation" prompts, not implementation placeholders. Task 10 has complete code. **API corrections applied:** runner uses `competition_core.modeling.load_model(config.model, artifact=path)` + `load_tokenizer(config.model)` + `competition_core.test_inputs.load_probe_input_sets(...)` (the actual project APIs); `load_model_with_adapter` and `load_probe_holdout` do not exist.

**3. Type consistency:**
- `select_clean_candidate` returns `(tuple[int, ...], int)` consistently in spec, plan, tests ✓
- `extract_checkpoints` returns `dict[int, dict[str, Any]]` consistently ✓
- `derive_cell_id` returns `str` format `<arch>__<cand_role>__<seed>__<ctrl>` consistently ✓
- `_run_one_cell` is monkey-patched in test as `fake_run_one_cell(cell_id: str, **kwargs)`; actual signature uses keyword args — verify in Task 3 Step 2 that monkeypatch matches the call site ✓
- `_load_model_and_tokenizer(detection_yaml, adapter_path)` returns `(model, tokenizer, device, side_config)`; manifest must carry `detection_yamls[arch]` (path to existing detection YAML) ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-22-stage2-method-diagnostic.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
