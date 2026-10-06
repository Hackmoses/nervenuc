"""Tests for authoritative ROI-to-mask rasterization and coordinate contracts."""

from __future__ import annotations

import numpy as np
import pytest
import roifile

from nervenuc.fiji.errors import InvalidRoiGeometryError
from nervenuc.fiji.roi import rasterize_roi


def test_rasterize_rectangle_boundary_and_exact_indices() -> None:
    """Verify rectangle rasterization with exact row/col bounds."""
    # Rectangle covering top=5..15, left=10..30 in a (40, 50) image
    roi = roifile.ImagejRoi.frompoints([[10, 5], [30, 15]])
    roi.roitype = roifile.ROI_TYPE.RECT
    roi.top = 5
    roi.left = 10
    roi.bottom = 15
    roi.right = 30

    mask = rasterize_roi(roi, (40, 50))
    assert mask.shape == (40, 50)
    assert mask.dtype == bool
    assert np.count_nonzero(mask) == 10 * 20

    # Test top-left and bottom-right edges
    assert mask[5, 10]
    assert mask[14, 29]
    assert not mask[4, 10]
    assert not mask[15, 30]


def test_rasterize_polygon_asymmetric_orientation() -> None:
    """Verify that ImageJ X maps to column and ImageJ Y maps to row on asymmetric image."""
    # Asymmetric canvas: H=30 (rows), W=60 (cols)
    # Define triangle with vertices (X, Y) -> (col, row):
    # (10, 5), (50, 5), (50, 20)
    roi = roifile.ImagejRoi.frompoints([[10, 5], [50, 5], [50, 20]])
    roi.roitype = roifile.ROI_TYPE.POLYGON

    mask = rasterize_roi(roi, (30, 60))
    assert mask.shape == (30, 60)

    # Point (col=45, row=8) must be inside triangle
    assert mask[8, 45]
    # Point (col=8, row=45) would be out of row bounds or False if transposed
    assert not mask[25, 10]


def test_rasterize_oval() -> None:
    """Verify oval/ellipse rasterization."""
    roi = roifile.ImagejRoi.frompoints([[10, 10], [30, 20]])
    roi.roitype = roifile.ROI_TYPE.OVAL
    roi.top = 10
    roi.left = 10
    roi.bottom = 20
    roi.right = 30

    mask = rasterize_roi(roi, (40, 40))
    assert mask.shape == (40, 40)
    assert mask.dtype == bool
    # Center pixel (row=15, col=20) must be inside
    assert mask[15, 20]
    assert np.count_nonzero(mask) > 0



def test_rasterize_out_of_bounds_rejected() -> None:
    """Verify strict out-of-bounds rejection."""
    # Polygon extending beyond H=20
    roi = roifile.ImagejRoi.frompoints([[5, 5], [15, 5], [15, 25]])
    roi.roitype = roifile.ROI_TYPE.POLYGON

    with pytest.raises(InvalidRoiGeometryError, match="exceed image shape"):
        rasterize_roi(roi, (20, 20))

    # Negative coordinates
    roi_neg = roifile.ImagejRoi.frompoints([[-2, 5], [10, 5], [10, 10]])
    roi_neg.roitype = roifile.ROI_TYPE.POLYGON
    with pytest.raises(InvalidRoiGeometryError, match="exceed image shape"):
        rasterize_roi(roi_neg, (20, 20))


def test_rasterize_empty_or_invalid_geometry() -> None:
    """Verify invalid shapes (zero dimensions, fewer than 3 vertices)."""
    with pytest.raises(InvalidRoiGeometryError, match="Target shape dimensions must be positive"):
        roi = roifile.ImagejRoi.frompoints([[5, 5], [10, 5], [10, 10]])
        rasterize_roi(roi, (0, 20))

    # Polygon with 2 points
    roi_2pts = roifile.ImagejRoi.frompoints([[5, 5], [10, 5]])
    roi_2pts.roitype = roifile.ROI_TYPE.POLYGON
    with pytest.raises(InvalidRoiGeometryError, match="at least 3 vertices"):
        rasterize_roi(roi_2pts, (20, 20))
