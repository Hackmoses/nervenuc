"""Tests for physical calibration mapping and uncalibrated invariants."""

from __future__ import annotations

import pytest

from nervenuc.fiji.conversion import (
    build_imagej_calibration,
    restore_voxel_spacing,
)
from nervenuc.preparation import VoxelSpacing


def test_voxel_spacing_to_imagej_calibration() -> None:
    """Verify calibrated VoxelSpacing correctly maps to ImageJCalibration."""
    spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104, unit="um")
    cal = build_imagej_calibration(spacing)

    assert cal.is_calibrated is True
    assert cal.pixel_depth == 0.5
    assert cal.pixel_height == 0.104
    assert cal.pixel_width == 0.104
    assert cal.unit == "um"

    # Restore
    restored = restore_voxel_spacing(cal)
    assert restored == spacing


def test_uncalibrated_invariant_never_invents_one_micron() -> None:
    """Verify None spacing maps to uncalibrated and restores as None, never 1 um."""
    cal = build_imagej_calibration(None)

    assert cal.is_calibrated is False
    assert cal.unit == "pixel"
    assert cal.pixel_width is None
    assert cal.pixel_height is None
    assert cal.pixel_depth is None

    # Restoring uncalibrated calibration must return None
    restored = restore_voxel_spacing(cal)
    assert restored is None


def test_2d_tuple_spacing() -> None:
    """Verify 2D tuple (Y, X) spacing mapping."""
    cal = build_imagej_calibration((0.2, 0.3))
    assert cal.is_calibrated is True
    assert cal.pixel_height == 0.2
    assert cal.pixel_width == 0.3
    assert cal.pixel_depth is None


def test_unsupported_unit_returns_none() -> None:
    """Verify unrecognized unit returns None on restore."""
    cal = build_imagej_calibration(VoxelSpacing(z=1.0, y=1.0, x=1.0, unit="um"))
    # Manually create non-standard unit
    from nervenuc.fiji.models import ImageJCalibration

    bad_cal = ImageJCalibration(
        pixel_width=1.0,
        pixel_height=1.0,
        pixel_depth=1.0,
        unit="inches",
        is_calibrated=True,
    )
    restored = restore_voxel_spacing(bad_cal)
    assert restored is None
