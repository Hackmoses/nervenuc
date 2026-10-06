"""Tests for roifile-backed ImageJ ROI reading, writing, and type validation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import roifile

from nervenuc.fiji.errors import NerveNucFijiError, UnsupportedRoiError
from nervenuc.fiji.roi import (
    SUPPORTED_ROI_TYPES,
    read_roi,
    read_roi_set,
    validate_roi_type,
    write_roi,
    write_roi_set,
)


def test_supported_roi_types_validation() -> None:
    """Verify supported and unsupported ImageJ ROI types."""
    # Supported
    for t in (roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.RECT, roifile.ROI_TYPE.OVAL, roifile.ROI_TYPE.FREEHAND, roifile.ROI_TYPE.TRACED):
        r = roifile.ImagejRoi.frompoints([[10, 10], [10, 20], [20, 20]])
        r.roitype = t
        validate_roi_type(r)  # Should not raise

    # Unsupported
    for t in (roifile.ROI_TYPE.LINE, roifile.ROI_TYPE.FREELINE, roifile.ROI_TYPE.POLYLINE, roifile.ROI_TYPE.POINT, roifile.ROI_TYPE.ANGLE):
        r = roifile.ImagejRoi.frompoints([[10, 10], [10, 20]])
        r.roitype = t
        with pytest.raises(UnsupportedRoiError, match="is not supported"):
            validate_roi_type(r)


def test_unsupported_composite_and_text_rois() -> None:
    """Verify rejection of composite/shape and text ROIs."""
    r = roifile.ImagejRoi.frompoints([[10, 10], [10, 20], [20, 20]])
    r.shape_roi_size = 5
    with pytest.raises(UnsupportedRoiError, match="Composite / multi-shape"):
        validate_roi_type(r)

    r2 = roifile.ImagejRoi.frompoints([[10, 10], [10, 20], [20, 20]])
    r2.text = "Sample label"
    with pytest.raises(UnsupportedRoiError, match="Text ImageJ ROIs"):
        validate_roi_type(r2)


def test_read_and_write_single_roi(tmp_path: Path) -> None:
    """Verify writing and reading a single .roi file."""
    roi = roifile.ImagejRoi.frompoints([[15, 25], [15, 45], [35, 45], [35, 25]])
    roi.name = "AUTO-000042"

    out_file = tmp_path / "test_single.roi"
    write_roi(roi, out_file)
    assert out_file.is_file()

    # Overwrite protection
    with pytest.raises(NerveNucFijiError, match="already exists"):
        write_roi(roi, out_file, overwrite=False)

    # Read back
    loaded = read_roi(out_file)
    assert loaded.name == "AUTO-000042"
    assert loaded.roitype in SUPPORTED_ROI_TYPES


def test_read_and_write_roi_set_zip(tmp_path: Path) -> None:
    """Verify writing and reading a collection of ROIs in a .zip file."""
    roi1 = roifile.ImagejRoi.frompoints([[10, 10], [10, 30], [30, 30]])
    roi1.name = "MAN-000001"

    roi2 = roifile.ImagejRoi.frompoints([[50, 50], [50, 70], [70, 70]])
    roi2.name = "EDIT-000002"

    out_zip = tmp_path / "RoiSet.zip"
    write_roi_set([roi1, roi2], out_zip)
    assert out_zip.is_file()

    loaded_list = read_roi_set(out_zip)
    assert len(loaded_list) == 2
    names = {r.name for r in loaded_list}
    assert names == {"MAN-000001", "EDIT-000002"}
