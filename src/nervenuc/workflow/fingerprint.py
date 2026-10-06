"""Workflow-level deterministic fingerprinting for AnalysisConfig and combined pipelines."""

from __future__ import annotations

from typing import Any
import hashlib
import json

from .config import AnalysisConfig
from .errors import WorkflowConfigurationError



def compute_analysis_config_fingerprint(config: AnalysisConfig) -> str:
    """Compute a deterministic SHA-256 fingerprint over the scientific contents of an AnalysisConfig.

    Note: Must produce the exact byte-for-byte hash as M12 compute_config_fingerprint.

    Args:
        config: Scientifically authoritative AnalysisConfig.

    Returns:
        Hexadecimal SHA-256 string.

    Raises:
        WorkflowConfigurationError: If config is not an AnalysisConfig.
    """
    if not isinstance(config, AnalysisConfig):
        raise WorkflowConfigurationError(
            f"config must be an AnalysisConfig, got {type(config).__name__}."
        )

    canonical_dict = config.to_dict()
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def compute_pipeline_fingerprint(
    config_fingerprint: str,
    preparation_fingerprint: str | None = None,
) -> str:
    """Compute combined pipeline fingerprint using canonical domain separation.

    Args:
        config_fingerprint: SHA-256 hash of AnalysisConfig.
        preparation_fingerprint: Optional SHA-256 hash of StackPreparationConfig.

    Returns:
        Hexadecimal SHA-256 string representing the combined pipeline.
    """
    if not isinstance(config_fingerprint, str) or not config_fingerprint.strip():
        raise WorkflowConfigurationError("config_fingerprint must be a non-empty string.")

    if preparation_fingerprint is None:
        return config_fingerprint

    if not isinstance(preparation_fingerprint, str) or not preparation_fingerprint.strip():
        raise WorkflowConfigurationError("preparation_fingerprint must be a non-empty string or None.")

    canonical_dict = {
        "kind": "nervenuc_pipeline_fingerprint_v1",
        "analysis_fingerprint": config_fingerprint,
        "preparation_fingerprint": preparation_fingerprint,
    }
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def compute_volume_analysis_config_fingerprint(config: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint over the scientific contents of a VolumeAnalysisConfig.

    Args:
        config: Scientifically authoritative VolumeAnalysisConfig.

    Returns:
        Hexadecimal SHA-256 string.
    """
    from .config import VolumeAnalysisConfig

    if not isinstance(config, VolumeAnalysisConfig):
        raise WorkflowConfigurationError(
            f"config must be a VolumeAnalysisConfig, got {type(config).__name__}."
        )

    canonical_dict = {
        "kind": "nervenuc_volume_analysis_config_v1",
        **config.to_dict(),
    }
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def compute_volume_pipeline_fingerprint(volume_config_fingerprint: str) -> str:
    """Compute combined 3D pipeline fingerprint using canonical domain separation.

    Args:
        volume_config_fingerprint: SHA-256 hash of VolumeAnalysisConfig.

    Returns:
        Hexadecimal SHA-256 string.
    """
    if not isinstance(volume_config_fingerprint, str) or not volume_config_fingerprint.strip():
        raise WorkflowConfigurationError("volume_config_fingerprint must be a non-empty string.")

    canonical_dict = {
        "kind": "nervenuc_volume_pipeline_v1",
        "mode": "volume_3d",
        "volume_config_fingerprint": volume_config_fingerprint,
    }
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()
