"""Tests for exporting active 2D QC segmentation objects to Fiji ImageJ ROI sets."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import roifile

from nervenuc.fiji import (
    InvalidRoiGeometryError,
    NerveNucFijiError,
    export_review_rois,
    rasterize_roi,
    read_roi_set,
)
from nervenuc.qc import ReviewSession, ReviewedSegmentationResult, start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _make_sample_segmentation(shape: tuple[int, int] = (100, 100)) -> SegmentationResult:
    """Create a synthetic SegmentationResult with 3 automated nuclei."""
    labels = np.zeros(shape, dtype=np.int32)
    # Nucleus 1: [10:20, 10:20]
    labels[10:20, 10:20] = 1
    # Nucleus 2: [40:50, 40:50]
    labels[40:50, 40:50] = 2
    # Nucleus 3: [70:85, 70:85]
    labels[70:85, 70:85] = 3

    objects_df = pd.DataFrame([
        {
            "nucleus_id": 1,
            "area_px": 100,
            "centroid_y": 14.5,
            "centroid_x": 14.5,
            "bbox_min_y": 10,
            "bbox_min_x": 10,
            "bbox_max_y": 20,
            "bbox_max_x": 20,
            "touches_border": False,
            "qc_status": "accepted",
        },
        {
            "nucleus_id": 2,
            "area_px": 100,
            "centroid_y": 44.5,
            "centroid_x": 44.5,
            "bbox_min_y": 40,
            "bbox_min_x": 40,
            "bbox_max_y": 50,
            "bbox_max_x": 50,
            "touches_border": False,
            "qc_status": "accepted",
        },
        {
            "nucleus_id": 3,
            "area_px": 225,
            "centroid_y": 77.0,
            "centroid_x": 77.0,
            "bbox_min_y": 70,
            "bbox_min_x": 70,
            "bbox_max_y": 85,
            "bbox_max_x": 85,
            "touches_border": False,
            "qc_status": "accepted",
        },
    ])
    return SegmentationResult(
        labels=labels,
        objects=objects_df,
        config=SegmentationConfig(),
    )


def test_export_active_objects_from_review_session(tmp_path: Path) -> None:
    """Export active ReviewSession objects preserves exact review IDs and excludes retired objects."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # 1. Accept AUTO-000001
    session.accept("AUTO-000001")

    # 2. Reject AUTO-000002 (retired)
    session.reject("AUTO-000002")

    # 3. Add manual nucleus MAN-000001
    man_mask = np.zeros((100, 100), dtype=bool)
    man_mask[10:20, 60:70] = True
    man_id = session.add(man_mask, note="Manual addition")
    assert man_id == "MAN-000001"

    # 4. Split AUTO-000003 into EDIT-000001 and EDIT-000002
    split_mask1 = np.zeros((100, 100), dtype=bool)
    split_mask1[70:77, 70:85] = True
    split_mask2 = np.zeros((100, 100), dtype=bool)
    split_mask2[78:85, 70:85] = True
    child_ids = session.split("AUTO-000003", [split_mask1, split_mask2])
    assert child_ids == ("EDIT-000001", "EDIT-000002")

    # Active objects: AUTO-000001, MAN-000001, EDIT-000001, EDIT-000002
    # Retired objects: AUTO-000002, AUTO-000003
    out_zip = tmp_path / "active_rois.zip"
    res_path = export_review_rois(session, out_zip)
    assert res_path == out_zip
    assert out_zip.is_file()

    # Read back exported ROIs
    rois = read_roi_set(out_zip)
    assert len(rois) == 4

    exported_names = [roi.name for roi in rois]
    assert set(exported_names) == {"AUTO-000001", "MAN-000001", "EDIT-000001", "EDIT-000002"}
    assert "AUTO-000002" not in exported_names
    assert "AUTO-000003" not in exported_names

    # Verify geometric fidelity
    for roi in rois:
        r_mask = rasterize_roi(roi, (100, 100))
        assert np.count_nonzero(r_mask) > 0


def test_export_from_finalized_reviewed_result(tmp_path: Path) -> None:
    """Exporting a ReviewedSegmentationResult exports all active objects with exact IDs."""
    seg = _make_sample_segmentation()
    session = start_review(seg)
    session.accept("AUTO-000001")
    session.accept("AUTO-000002")
    session.accept("AUTO-000003")
    reviewed = session.finalize()

    out_zip = tmp_path / "finalized_rois.zip"
    export_review_rois(reviewed, out_zip)

    rois = read_roi_set(out_zip)
    assert len(rois) == 3
    assert [r.name for r in rois] == ["AUTO-000001", "AUTO-000002", "AUTO-000003"]


def test_export_overwrite_policy(tmp_path: Path) -> None:
    """Exporting to existing file raises error unless overwrite=True."""
    seg = _make_sample_segmentation()
    session = start_review(seg)
    out_zip = tmp_path / "rois.zip"

    export_review_rois(session, out_zip)
    assert out_zip.is_file()

    with pytest.raises(NerveNucFijiError, match="already exists"):
        export_review_rois(session, out_zip, overwrite=False)

    # Overwrite succeeds
    export_review_rois(session, out_zip, overwrite=True)
    assert out_zip.is_file()


def test_export_atomic_failure_on_invalid_mask(tmp_path: Path) -> None:
    """Export fails atomically before creating file if any active mask has holes or is unconvertible."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # Artificially inject an active mask with an internal hole into the session
    holed_mask = np.zeros((100, 100), dtype=bool)
    holed_mask[55:65, 55:65] = True
    holed_mask[58:62, 58:62] = False  # Hole

    # Bypass ReviewSession.add() validation solely to test export atomic error handling
    session._active_objects["MAN-000099"] = type(list(session._active_objects.values())[0])(
        review_id="MAN-000099",
        mask=holed_mask,
        source_nucleus_id=None,
        review_status="edited",
        review_origin="manual_add",
        parent_review_ids=(),
        review_note=None,
    )

    out_zip = tmp_path / "should_not_exist.zip"
    with pytest.raises(InvalidRoiGeometryError, match="internal holes"):
        export_review_rois(session, out_zip)

    assert not out_zip.exists()


def test_export_empty_session_fails(tmp_path: Path) -> None:
    """Exporting a session with 0 active objects raises NerveNucFijiError."""
    seg = _make_sample_segmentation()
    session = start_review(seg)
    session.reject("AUTO-000001")
    session.reject("AUTO-000002")
    session.reject("AUTO-000003")

    out_zip = tmp_path / "empty.zip"
    with pytest.raises(NerveNucFijiError, match="no active objects"):
        export_review_rois(session, out_zip)
