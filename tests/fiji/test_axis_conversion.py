"""Tests for bidirectional axis permutation and shape/numerical fidelity on asymmetric arrays."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.fiji.conversion import resolve_axis_conversion


@pytest.mark.parametrize(
    ("source_axes", "shape"),
    [
        ("YX", (17, 23)),
        ("ZYX", (5, 17, 23)),
        ("CYX", (3, 17, 23)),
        ("CZYX", (3, 5, 17, 23)),
        ("TCZYX", (2, 3, 5, 17, 23)),
    ],
)
def test_asymmetric_axis_round_trip(source_axes: str, shape: tuple[int, ...]) -> None:
    """Verify bidirectional transpose round trip on asymmetric arrays preserving exact values."""
    conv = resolve_axis_conversion(source_axes)

    # Populate with unique deterministic values
    total_elements = int(np.prod(shape))
    arr = np.arange(total_elements, dtype=np.uint16).reshape(shape)

    # Forward transpose to ImageJ
    ij_arr = np.transpose(arr, conv.forward_permutation)
    assert ij_arr.shape != arr.shape or len(shape) == 1

    # Reverse transpose back to NerveNuc
    restored = np.transpose(ij_arr, conv.reverse_permutation)
    assert restored.shape == arr.shape
    assert np.array_equal(arr, restored)


def test_axis_transposition_numerical_sensitivity() -> None:
    """Verify that any axis transposition error is caught by value comparisons."""
    # Asymmetric 3D: Z=3, Y=7, X=11
    shape = (3, 7, 11)
    arr = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)

    conv = resolve_axis_conversion("ZYX")
    ij_arr = np.transpose(arr, conv.forward_permutation)

    # Intentionally apply wrong reverse permutation
    wrong_reverse = (1, 0, 2)
    wrong_restored = np.transpose(ij_arr, wrong_reverse)
    assert wrong_restored.shape != arr.shape or not np.array_equal(wrong_restored, arr)
