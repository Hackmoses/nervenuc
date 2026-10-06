"""Tests for array and axis validation contracts for ImageJ conversion."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.fiji.conversion import (
    SUPPORTED_CONVERSION_AXES,
    SUPPORTED_DTYPES,
    resolve_axis_conversion,
    validate_array_for_conversion,
)
from nervenuc.fiji.errors import FijiConversionError


def test_supported_and_unsupported_axis_layouts() -> None:
    """Verify supported and unsupported axis layouts."""
    for ax in SUPPORTED_CONVERSION_AXES:
        conv = resolve_axis_conversion(ax)
        assert conv.source_axes == ax
        assert len(conv.forward_permutation) == len(ax)
        assert len(conv.reverse_permutation) == len(ax)

    # Unsupported
    for bad in ("XYZ", "TYX", "CXYZ", "INVALID"):
        with pytest.raises(FijiConversionError, match="Unsupported axis layout"):
            resolve_axis_conversion(bad)

    with pytest.raises(FijiConversionError, match="non-empty string"):
        resolve_axis_conversion("")



def test_supported_and_unsupported_dtypes() -> None:
    """Verify supported and unsupported dtypes."""
    for dt in SUPPORTED_DTYPES:
        arr = np.ones((5, 10), dtype=dt)
        validate_array_for_conversion(arr, "YX")

    # Unsupported
    for bad_dt in (np.complex128, object):
        arr_bad = np.ones((5, 10), dtype=bad_dt)
        with pytest.raises(FijiConversionError, match="Unsupported array dtype"):
            validate_array_for_conversion(arr_bad, "YX")


def test_array_ndim_mismatch_raises() -> None:
    """Verify ndim mismatch vs axis length raises FijiConversionError."""
    arr_2d = np.ones((5, 10), dtype=np.uint16)
    with pytest.raises(FijiConversionError, match="Array ndim .* does not match"):
        validate_array_for_conversion(arr_2d, "ZYX")


def test_non_finite_values_rejected() -> None:
    """Verify arrays containing NaN or Inf are rejected."""
    arr_nan = np.ones((5, 10), dtype=np.float32)
    arr_nan[2, 3] = np.nan
    with pytest.raises(FijiConversionError, match="non-finite values"):
        validate_array_for_conversion(arr_nan, "YX")

    arr_inf = np.ones((5, 10), dtype=np.float32)
    arr_inf[2, 3] = np.inf
    with pytest.raises(FijiConversionError, match="non-finite values"):
        validate_array_for_conversion(arr_inf, "YX")


def test_input_array_is_not_mutated() -> None:
    """Verify that validation does not mutate input array."""
    arr = np.arange(50, dtype=np.uint16).reshape((5, 10))
    copy = arr.copy()
    validate_array_for_conversion(arr, "YX")
    assert np.array_equal(arr, copy)
