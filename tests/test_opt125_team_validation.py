from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from competition_core.training import infer_lora_targets
from scripts.build_opt125_team_bundle import build_bundle
from scripts.run_opt125_team_validation import (
    boundaries,
    latest_checkpoint,
    pair_metrics,
    safe_name,
    validate_configs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_opt_model_uses_architecture_specific_lora_targets() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="opt"))

    assert infer_lora_targets(model) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]


def test_team_configs_validate_without_loading_models() -> None:
    backdoor, clean, detection = validate_configs()

    assert backdoor.data == clean.data
    assert backdoor.model == detection.model


def test_four_shards_cover_tokenizer_vocabulary_without_overlap() -> None:
    shards = boundaries(50_265)

    assert shards[0][0] == 0
    assert shards[-1][1] == 50_265
    assert all(left[1] == right[0] for left, right in zip(shards, shards[1:]))


def test_latest_checkpoint_selects_highest_complete_epoch(tmp_path: Path) -> None:
    for epoch in (1, 3, 2):
        checkpoint = tmp_path / "checkpoints" / f"epoch-{epoch}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    checkpoint = latest_checkpoint(tmp_path, total_epochs=10)

    assert checkpoint == (tmp_path / "checkpoints/epoch-3", 3)


def test_pair_metrics_count_models_not_candidates() -> None:
    assert pair_metrics(True, False) == {
        "confusion_matrix": {
            "true_positive": 1,
            "false_positive": 0,
            "true_negative": 1,
            "false_negative": 0,
        },
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_positive_rate": 0.0,
    }


def test_participant_name_is_safe_for_local_paths() -> None:
    assert safe_name("Member A / 4060") == "Member-A-4060"


def test_built_bundle_is_small_truth_scoped_and_hash_complete(tmp_path: Path) -> None:
    output = tmp_path / "opt125-team.zip"
    result = build_bundle(output, root=ROOT)

    assert result["size"] < 2_000_000
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle_manifest.json"))
        assert "RUN_OPT125_PAIR.cmd" in names
        assert "scripts/run_opt125_team_validation.py" in names
        assert manifest["contains_model_weights"] is False
        assert manifest["contains_training_samples"] is False
        assert manifest["contains_unpublished_paper"] is False
        assert not any(
            name.lower().endswith((".docx", ".pdf", ".safetensors", ".bin"))
            for name in names
        )
        for name, metadata in manifest["files"].items():
            content = archive.read(name)
            assert len(content) == metadata["size"]
            assert sha256(content).hexdigest() == metadata["sha256"]
