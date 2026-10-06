"""Tests for 2D boolean mask to ImageJ ROI conversion and round-trip fidelity."""

from __future__ import annotations

import numpy as np
import pytest
import roifile

from nervenuc.fiji.errors import InvalidRoiGeometryError
from nervenuc.fiji.roi import mask_to_roi, rasterize_roi


def test_mask_to_roi_exact_round_trip() -> None:
    """Verify exact round trip for convex / simple nucleus mask."""
    # Create single solid nucleus mask
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 20:40] = True

    roi = mask_to_roi(mask, name="AUTO-000001")
    assert roi.name == "AUTO-000001"
    assert roi.roitype in (roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.FREEHAND)

    # Rasterize back
    reconstructed = rasterize_roi(roi, (50, 50))
    # Should be identical
    assert np.array_equal(mask, reconstructed)


def test_mask_to_roi_concave_shape_tolerance() -> None:
    """Verify high fidelity (Dice >= 0.98) on concave shape."""
    mask = np.zeros((60, 60), dtype=bool)
    # C-shaped nucleus
    mask[10:50, 10:40] = True
    mask[20:40, 20:40] = False  # Cutout on right

    roi = mask_to_roi(mask, name="MAN-000002")
    reconstructed = rasterize_roi(roi, (60, 60))

    intersection = np.count_nonzero(mask & reconstructed)
    dice = (2.0 * intersection) / (np.count_nonzero(mask) + np.count_nonzero(reconstructed))
    assert dice >= 0.98, f"Expected Dice >= 0.98, got {dice:.4f}"


def test_mask_to_roi_border_touching_nucleus() -> None:
    """Verify mask touching image boundary is converted and rasterized properly."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:20, 0:20] = True  # Touches top-left corner

    roi = mask_to_roi(mask, name="BORDER-01")
    reconstructed = rasterize_roi(roi, (40, 40))
    assert np.array_equal(mask, reconstructed)


def test_mask_to_roi_rejects_empty_mask() -> None:
    """Verify empty mask raises InvalidRoiGeometryError."""
    empty = np.zeros((30, 30), dtype=bool)
    with pytest.raises(InvalidRoiGeometryError, match="Cannot convert empty mask"):
        mask_to_roi(empty)


def test_mask_to_roi_rejects_multi_component_mask() -> None:
    """Verify multi-component mask is rejected rather than incorrectly merged."""
    multi = np.zeros((50, 50), dtype=bool)
    multi[5:15, 5:15] = True
    multi[30:40, 30:40] = True

    with pytest.raises(InvalidRoiGeometryError, match="contains 2 disconnected objects"):
        mask_to_roi(multi)


def test_mask_to_roi_rejects_internal_holes() -> None:
    """Verify mask with topological internal hole is rejected."""
    holed = np.zeros((40, 40), dtype=bool)
    holed[10:30, 10:30] = True
    holed[18:22, 18:22] = False  # Hole in center

    with pytest.raises(InvalidRoiGeometryError, match="contains internal holes"):
        mask_to_roi(holed)
