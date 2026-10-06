"""Tests for Fiji-assisted split replacement operations into NerveNuc ReviewSession."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nervenuc.fiji import (
    mask_to_roi,
    replace_with_fiji_rois,
    write_roi_set,
)
from nervenuc.qc import (
    InvalidReviewMaskError,
    ReviewSession,
    UnknownReviewObjectError,
    start_review,
)
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _make_sample_segmentation(shape: tuple[int, int] = (100, 100)) -> SegmentationResult:
    """Create a synthetic SegmentationResult with 1 automated nucleus to split."""
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:30, 10:30] = 1  # AUTO-000001: 20x20 box
    objects_df = pd.DataFrame([
        {
            "nucleus_id": 1,
            "area_px": 400,
            "centroid_y": 19.5,
            "centroid_x": 19.5,
            "bbox_min_y": 10,
            "bbox_min_x": 10,
            "bbox_max_y": 30,
            "bbox_max_x": 30,
            "touches_border": False,
            "qc_status": "accepted",
        }
    ])
    return SegmentationResult(
        labels=labels,
        objects=objects_df,
        config=SegmentationConfig(),
    )


def test_replace_with_fiji_rois_success(tmp_path: Path) -> None:
    """Explicit split using 2 Fiji ROIs retires parent and creates EDIT child objects."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # 2 non-overlapping ROIs strictly within [10:30, 10:30]
    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:19, 10:30] = True
    roi1 = mask_to_roi(m1, name="Child_North")

    m2 = np.zeros((100, 100), dtype=bool)
    m2[21:30, 10:30] = True
    roi2 = mask_to_roi(m2, name="Child_South")

    zip_path = tmp_path / "split_children.zip"
    write_roi_set([roi1, roi2], zip_path)
    expected_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    result = replace_with_fiji_rois(
        session,
        target_review_id="AUTO-000001",
        roi_path=zip_path,
        note="Researcher split in Fiji",
    )

    assert result.source_path == zip_path
    assert result.roi_file_sha256 == expected_sha256
    assert result.target_review_id == "AUTO-000001"
    assert result.child_review_ids == ("EDIT-000001", "EDIT-000002")
    assert len(result.imported) == 2

    # Check ReviewSession active objects
    snapshot = session.result()
    active_ids = list(snapshot.objects["review_id"])
    assert "AUTO-000001" not in active_ids
    assert "EDIT-000001" in active_ids
    assert "EDIT-000002" in active_ids

    # Check child metadata
    c1 = snapshot.objects[snapshot.objects["review_id"] == "EDIT-000001"].iloc[0]
    assert c1["review_origin"] == "split"
    assert c1["review_status"] == "edited"
    assert c1["parent_review_ids"] == ("AUTO-000001",)
    assert "Researcher split in Fiji" in str(c1["review_note"])
    assert "split_children.zip" in str(c1["review_note"])

    # Check history actions
    split_action = snapshot.history[-1]
    assert split_action.action_type == "split"
    assert split_action.target_review_ids == ("AUTO-000001",)
    assert split_action.resulting_review_ids == ("EDIT-000001", "EDIT-000002")


def test_replace_with_fiji_rois_cardinality_failure(tmp_path: Path) -> None:
    """Providing only 1 replacement ROI fails split cardinality without mutating session."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:20, 10:20] = True
    roi1 = mask_to_roi(m1, name="Single_ROI")
    zip_path = tmp_path / "single.zip"
    write_roi_set([roi1], zip_path)

    with pytest.raises(InvalidReviewMaskError, match="at least 2"):
        replace_with_fiji_rois(session, "AUTO-000001", zip_path)

    # Assert session is unchanged
    snapshot = session.result()
    assert "AUTO-000001" in list(snapshot.objects["review_id"])
    assert len(snapshot.history) == 0


def test_replace_with_fiji_rois_exceeds_parent_boundary(tmp_path: Path) -> None:
    """Replacement ROIs exceeding parent mask boundaries fail under M3 split validation."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # ROI 1 inside parent
    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:19, 10:30] = True
    roi1 = mask_to_roi(m1, name="Inside")

    # ROI 2 extends outside parent ([10:30, 10:30]) to row 35
    m2 = np.zeros((100, 100), dtype=bool)
    m2[21:35, 10:30] = True
    roi2 = mask_to_roi(m2, name="Outside")

    zip_path = tmp_path / "exceeding.zip"
    write_roi_set([roi1, roi2], zip_path)

    with pytest.raises(InvalidReviewMaskError, match="exceed the boundary of parent nucleus"):
        replace_with_fiji_rois(session, "AUTO-000001", zip_path)

    # Assert session is unchanged
    snapshot = session.result()
    assert "AUTO-000001" in list(snapshot.objects["review_id"])


def test_replace_with_fiji_rois_unknown_target(tmp_path: Path) -> None:
    """Splitting a non-existent review ID raises UnknownReviewObjectError."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:15, 10:15] = True
    m2 = np.zeros((100, 100), dtype=bool)
    m2[16:20, 10:15] = True
    zip_path = tmp_path / "split.zip"
    write_roi_set([mask_to_roi(m1), mask_to_roi(m2)], zip_path)

    with pytest.raises(UnknownReviewObjectError, match="Unknown review object"):
        replace_with_fiji_rois(session, "AUTO-999999", zip_path)
