"""Validation dataset registry and loader."""

from __future__ import annotations

from typing import Callable

from .errors import InvalidGroundTruthError
from .models import ValidationCase
from .synthetic import (
    generate_syn_2d_empty,
    generate_syn_2d_isolated,
    generate_syn_2d_multi_marker,
    generate_syn_2d_touching,
    generate_syn_3d_isolated,
    generate_syn_3d_overlap_xy,
)

_SYNTHETIC_REGISTRY: dict[str, Callable[..., ValidationCase]] = {
    "SYN_2D_ISOLATED": generate_syn_2d_isolated,
    "SYN_2D_TOUCHING": generate_syn_2d_touching,
    "SYN_2D_EMPTY": generate_syn_2d_empty,
    "SYN_2D_MULTI_MARKER": generate_syn_2d_multi_marker,
    "SYN_3D_ISOLATED": generate_syn_3d_isolated,
    "SYN_3D_OVERLAP_XY": generate_syn_3d_overlap_xy,
}


def list_synthetic_cases() -> tuple[str, ...]:
    """Return the tuple of registered synthetic case identifiers."""
    return tuple(_SYNTHETIC_REGISTRY.keys())


def get_synthetic_case(case_id: str, *, seed: int = 42) -> ValidationCase:
    """Retrieve and instantiate a deterministic synthetic validation case by ID.

    Args:
        case_id: Case identifier (e.g. 'SYN_2D_ISOLATED').
        seed: Deterministic random seed for pixel generation.

    Returns:
        Immutable ValidationCase with image data and mathematical ground truth.
    """
    if case_id not in _SYNTHETIC_REGISTRY:
        raise InvalidGroundTruthError(
            f"Unknown synthetic case_id '{case_id}'. Available: {list_synthetic_cases()}"
        )
    return _SYNTHETIC_REGISTRY[case_id](seed=seed)
