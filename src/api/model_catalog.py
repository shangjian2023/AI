"""Trusted local model discovery and compatibility validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODEL_MARKERS = ("adapter_config.json", "config.json")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
CUSTOM_MODEL_ROOTS_ENV = "BDSHIELD_MODEL_ROOTS"


def model_search_roots(
    root: Path,
    *,
    extra_roots: list[Path] | None = None,
) -> list[tuple[Path, str]]:
    """Return trusted project, cache, and operator-configured model roots."""
    root = root.resolve()
    candidates: list[tuple[Path, str]] = [(root, "工作区")]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates.extend(
        [
            (Path(value), "Hugging Face 缓存")
            for value in (
                os.environ.get("HF_HUB_CACHE"),
                os.environ.get("HUGGINGFACE_HUB_CACHE"),
            )
            if value
        ]
    )
    candidates.extend(
        [
            (hf_home / "hub", "Hugging Face 缓存"),
            (
                Path(os.environ.get("LOCALAPPDATA", "")) / "huggingface" / "hub",
                "Hugging Face 缓存",
            ),
        ]
    )
    candidates.extend(
        (Path(value), "自定义模型目录")
        for value in os.environ.get(CUSTOM_MODEL_ROOTS_ENV, "").split(os.pathsep)
        if value
    )
    candidates.extend((path, "手动添加目录") for path in extra_roots or [])

    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for candidate, source in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        roots.append((resolved, source))
    return roots


def _is_full_checkpoint(model_path: Path) -> bool:
    return any((model_path / filename).exists() for filename in MODEL_WEIGHT_FILES)


def _cache_model_name(model_path: Path) -> str | None:
    for part in model_path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def _is_causal_checkpoint(metadata: dict[str, Any]) -> bool:
    """Exclude checkpoints explicitly declared as encoder or masked-LM only."""
    architectures = metadata.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        return True
    return any(
        "CausalLM" in str(architecture) or str(architecture).endswith("LMHeadModel")
        for architecture in architectures
    )


def model_metadata(model_path: Path) -> tuple[str, str]:
    """Return the artifact kind and a LoRA's declared base model, if known."""
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.is_file():
        try:
            metadata = json.loads(adapter_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        base_model = (
            metadata.get("base_model_name_or_path")
            if isinstance(metadata, dict)
            else None
        )
        return "LoRA adapter", str(base_model) if base_model else ""
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        base_model = metadata.get("_name_or_path") if isinstance(metadata, dict) else None
        return (
            "Full checkpoint",
            str(base_model) if base_model else (_cache_model_name(model_path) or ""),
        )
    return "Unknown", ""


def validate_model_pair(target_path: Path, reference_path: Path | None) -> None:
    """Reject known LoRA pairs trained from different base models before launch."""
    if reference_path is None:
        return
    if target_path.resolve() == reference_path.resolve():
        raise ValueError(
            "target and reference must be different model artifacts"
            "(待审与干净参考模型不能是同一路径)"
        )
    _, target_base = model_metadata(target_path)
    _, reference_base = model_metadata(reference_path)
    if target_base and reference_base and target_base != reference_base:
        raise ValueError(
            "target and reference models must declare the same base model "
            f"(待审模型为 {target_base}，干净参考模型为 {reference_base})"
        )


def resolve_model_path(
    root: Path,
    raw_path: str,
    *,
    must_exist: bool = True,
    extra_roots: list[Path] | None = None,
) -> Path:
    """Resolve a selectable local model without arbitrary disk traversal."""
    candidate = Path(raw_path).expanduser()
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist(路径不存在): {raw_path}")
    allowed_roots = [
        root.resolve(),
        *(
            path
            for path, _ in model_search_roots(root, extra_roots=extra_roots)
        ),
    ]
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
        raise ValueError(
            "model path must be inside the workspace, Hugging Face cache, or "
            f"{CUSTOM_MODEL_ROOTS_ENV}(模型路径必须位于受信任的本地模型目录)"
        )
    return resolved


def _model_entry(
    marker: Path,
    *,
    marker_name: str,
    root: Path,
    source: str,
) -> tuple[Path, dict[str, str]] | None:
    model_path = marker.parent
    if ".no_exist" in model_path.parts:
        return None
    if marker_name == "config.json" and not _is_full_checkpoint(model_path):
        return None
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    if marker_name == "config.json" and not _is_causal_checkpoint(metadata):
        return None
    kind = (
        "LoRA adapter" if marker_name == "adapter_config.json" else "Full checkpoint"
    )
    cache_name = _cache_model_name(model_path)
    base_model = ""
    if isinstance(metadata, dict):
        base_model = str(
            metadata.get("base_model_name_or_path")
            or metadata.get("_name_or_path")
            or cache_name
            or ""
        )
    try:
        selectable_path = model_path.relative_to(root).as_posix()
    except ValueError:
        selectable_path = str(model_path)
    display_name = (
        selectable_path if source == "工作区" else cache_name or selectable_path
    )
    return model_path, {
        "path": selectable_path,
        "label": f"{display_name} · {kind} · {source}",
        "kind": kind,
        "base_model": base_model,
        "source": source,
    }


def discover_local_models(
    root: Path,
    *,
    extra_roots: list[Path] | None = None,
) -> list[dict[str, str]]:
    """Find selectable adapters and checkpoints in project and local HF caches."""
    root = root.resolve()
    models: dict[Path, dict[str, str]] = {}
    for search_root, source in model_search_roots(root, extra_roots=extra_roots):
        for marker_name in MODEL_MARKERS:
            for marker in search_root.rglob(marker_name):
                entry = _model_entry(
                    marker,
                    marker_name=marker_name,
                    root=root,
                    source=source,
                )
                if entry is None:
                    continue
                model_path, payload = entry
                models.setdefault(model_path, payload)
    return sorted(models.values(), key=lambda item: item["path"].lower())
