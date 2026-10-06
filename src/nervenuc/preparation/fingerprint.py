"""Deterministic fingerprinting for StackPreparationConfig."""

from __future__ import annotations

import hashlib
import json

from .errors import NerveNucPreparationError
from .models import StackPreparationConfig


def compute_preparation_fingerprint(config: StackPreparationConfig) -> str:
    """Compute a deterministic SHA-256 fingerprint over the contents of a StackPreparationConfig.

    Args:
        config: Scientifically authoritative StackPreparationConfig.

    Returns:
        Hexadecimal SHA-256 string.

    Raises:
        NerveNucPreparationError: If config is not a StackPreparationConfig.
    """
    if not isinstance(config, StackPreparationConfig):
        raise NerveNucPreparationError(
            f"config must be a StackPreparationConfig, got {type(config).__name__}."
        )

    canonical_dict = {
        "kind": "nervenuc_preparation_fingerprint_v1",
        "config": config.to_dict(),
    }
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()
