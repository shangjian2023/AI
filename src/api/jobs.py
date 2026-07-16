"""Compatibility facade for platform scan orchestration.

Implementation lives in focused model-catalog, command-building, and runtime modules.
The exports here preserve the original API for server code and external callers.
"""
from __future__ import annotations

import subprocess

from src.api.model_catalog import (
    discover_local_models,
    model_search_roots,
    resolve_model_path,
    validate_model_pair,
)
from src.api.scan_commands import (
    COMPETITION_DETECTION_CONFIG,
    EVENT_PREFIX,
    DetectorMode,
    build_inversion_command,
    build_scan_environment,
    parse_scan_event,
    resolve_workspace_path,
    scan_parameters,
)
from src.api.scan_runtime import ScanJob, ScanManager

__all__ = [
    "COMPETITION_DETECTION_CONFIG",
    "EVENT_PREFIX",
    "DetectorMode",
    "ScanJob",
    "ScanManager",
    "build_inversion_command",
    "build_scan_environment",
    "discover_local_models",
    "model_search_roots",
    "parse_scan_event",
    "resolve_model_path",
    "resolve_workspace_path",
    "scan_parameters",
    "subprocess",
    "validate_model_pair",
]
