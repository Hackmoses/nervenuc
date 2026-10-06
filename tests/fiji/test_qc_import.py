"""Tests for importing Fiji ImageJ ROIs as manual objects into NerveNuc ReviewSession."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import roifile

from nervenuc.fiji import (
    InvalidRoiGeometryError,
    UnsupportedRoiError,
    import_rois_as_manual,
    mask_to_roi,
    write_roi,
    write_roi_set,
)
from nervenuc.qc import (
    InvalidReviewMaskError,
    ReviewSession,
    start_review,
)
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _make_sample_segmentation(shape: tuple[int, int] = (100, 100)) -> SegmentationResult:
    """Create a synthetic SegmentationResult with 1 automated nucleus."""
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:25, 10:25] = 1  # AUTO-000001
    objects_df = pd.DataFrame([
        {
            "nucleus_id": 1,
            "area_px": 225,
            "centroid_y": 17.0,
            "centroid_x": 17.0,
            "bbox_min_y": 10,
            "bbox_min_x": 10,
            "bbox_max_y": 25,
            "bbox_max_x": 25,
            "touches_border": False,
            "qc_status": "accepted",
        }
    ])
    return SegmentationResult(
        labels=labels,
        objects=objects_df,
        config=SegmentationConfig(),
    )


def test_import_supported_roi_set_as_manual(tmp_path: Path) -> None:
    """Import valid Fiji ROI set into ReviewSession generates MAN IDs with accurate provenance."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # Create 2 valid ROIs at non-overlapping coordinates
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[40:55, 40:55] = True
    roi1 = mask_to_roi(mask1, name="Custom_Nuc_A")

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[70:85, 70:85] = True
    roi2 = mask_to_roi(mask2, name="Custom_Nuc_B")

    zip_path = tmp_path / "manual_rois.zip"
    write_roi_set([roi1, roi2], zip_path)

    expected_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    result = import_rois_as_manual(session, zip_path)

    # Assert structured result
    assert result.source_path == zip_path
    assert result.roi_file_sha256 == expected_sha256
    assert len(result.imported) == 2

    rec1, rec2 = result.imported
    assert rec1.roi_name == "Custom_Nuc_A"
    assert rec1.roi_index == 0
    assert rec1.review_id == "MAN-000001"
    assert rec1.roi_type == "POLYGON"

    assert rec2.roi_name == "Custom_Nuc_B"
    assert rec2.roi_index == 1
    assert rec2.review_id == "MAN-000002"
    assert rec2.roi_type == "POLYGON"

    # Verify ReviewSession state and M3 semantics
    snapshot = session.result()
    objects_df = snapshot.objects
    assert len(objects_df) == 3  # AUTO-000001, MAN-000001, MAN-000002

    man1_row = objects_df[objects_df["review_id"] == "MAN-000001"].iloc[0]
    assert man1_row["review_origin"] == "manual_add"
    assert man1_row["review_status"] == "edited"
    assert "manual_rois.zip" in str(man1_row["review_note"])
    assert "Custom_Nuc_A" in str(man1_row["review_note"])
    assert expected_sha256 in str(man1_row["review_note"])

    # Verify actions in session history
    actions = snapshot.history
    add_actions = [a for a in actions if a.action_type == "add"]
    assert len(add_actions) == 2
    assert add_actions[0].resulting_review_ids == ("MAN-000001",)
    assert add_actions[1].resulting_review_ids == ("MAN-000002",)


def test_import_untrusted_id_regression(tmp_path: Path) -> None:
    """Importing an ROI named 'AUTO-000001' cannot hijack or replace AUTO-000001."""
    seg = _make_sample_segmentation()
    session = start_review(seg)

    # Create ROI named AUTO-000001 located elsewhere
    mask = np.zeros((100, 100), dtype=bool)
    mask[50:65, 50:65] = True
    roi = mask_to_roi(mask, name="AUTO-000001")

    roi_path = tmp_path / "imposter.roi"
    write_roi(roi, roi_path)

    result = import_rois_as_manual(session, roi_path)
    assert len(result.imported) == 1
    # Must allocate a new MAN-xxxxxx ID, not overwrite AUTO-000001
    assert result.imported[0].review_id == "MAN-000001"
    assert result.imported[0].roi_name == "AUTO-000001"

    # Verify original AUTO-000001 is untouched
    snapshot = session.result()
    auto_row = snapshot.objects[snapshot.objects["review_id"] == "AUTO-000001"].iloc[0]
    assert auto_row["review_origin"] == "automated"
    assert auto_row["review_status"] == "unreviewed"


def test_import_atomic_failure_on_unsupported_roi(tmp_path: Path) -> None:
    """Atomic import failure: If any ROI in a ZIP is unsupported, zero objects are added."""
    seg = _make_sample_segmentation()
    session = start_review(seg)
    initial_active = len(session.result().objects)
    initial_history_len = len(session.result().history)

    # ROI 1: Valid polygon
    m1 = np.zeros((100, 100), dtype=bool)
    m1[30:40, 30:40] = True
    roi1 = mask_to_roi(m1, name="Valid_1")

    # ROI 2: Valid polygon
    m2 = np.zeros((100, 100), dtype=bool)
    m2[50:60, 50:60] = True
    roi2 = mask_to_roi(m2, name="Valid_2")

    # ROI 3: Unsupported LINE ROI
    roi3 = roifile.ImagejRoi.frompoints([[10, 10], [90, 90]])
    roi3.roitype = roifile.ROI_TYPE.LINE
    roi3.name = "Unsupported_Line"

    zip_path = tmp_path / "mixed_rois.zip"
    roifile.roiwrite(str(zip_path), [roi1, roi2, roi3])

    with pytest.raises(UnsupportedRoiError, match="LINE"):
        import_rois_as_manual(session, zip_path)

    # Session must be completely untouched
    post_snapshot = session.result()
    assert len(post_snapshot.objects) == initial_active
    assert len(post_snapshot.history) == initial_history_len
    assert getattr(session, "_man_counter", 0) == 0


def test_import_atomic_failure_on_collision(tmp_path: Path) -> None:
    """Atomic import failure: If an ROI in a ZIP collides with an active nucleus, zero objects are added."""
    seg = _make_sample_segmentation()
    session = start_review(seg)
    initial_active = len(session.result().objects)

    # ROI 1: Valid polygon (no collision)
    m1 = np.zeros((100, 100), dtype=bool)
    m1[40:50, 40:50] = True
    roi1 = mask_to_roi(m1, name="Valid_NonOverlapping")

    # ROI 2: Collides with existing AUTO-000001 ([10:25, 10:25])
    m2 = np.zeros((100, 100), dtype=bool)
    m2[15:30, 15:30] = True
    roi2 = mask_to_roi(m2, name="Colliding_Roi")

    zip_path = tmp_path / "colliding_set.zip"
    write_roi_set([roi1, roi2], zip_path)

    with pytest.raises(InvalidReviewMaskError, match="overlaps with existing active nucleus"):
        import_rois_as_manual(session, zip_path)

    # Assert completely untouched
    post_snapshot = session.result()
    assert len(post_snapshot.objects) == initial_active
    assert "MAN-000001" not in post_snapshot.objects["review_id"].values
    assert getattr(session, "_man_counter", 0) == 0


def test_import_source_shape_mismatch(tmp_path: Path) -> None:
    """Providing a mismatched source_shape parameter raises InvalidRoiGeometryError."""
    seg = _make_sample_segmentation((100, 100))
    session = start_review(seg)

    m = np.zeros((100, 100), dtype=bool)
    m[40:50, 40:50] = True
    roi = mask_to_roi(m, name="TestRoi")
    roi_path = tmp_path / "test.roi"
    write_roi(roi, roi_path)

    with pytest.raises(InvalidRoiGeometryError, match="does not match"):
        import_rois_as_manual(session, roi_path, source_shape=(200, 200))


def test_import_atomic_failure_intra_batch_collision(tmp_path: Path) -> None:
    """Atomic failure when batch members collide: active/retired objects, actions, notes, counters remain untouched."""
    # Create 2-nucleus segmentation
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[40:50, 40:50] = 2
    seg = SegmentationResult(labels=labels, objects=pd.DataFrame(), config=SegmentationConfig())
    session = start_review(seg)

    # Set custom note on AUTO-000001
    session.set_note("AUTO-000001", "Researcher note")
    # Reject AUTO-000002 to populate retired objects
    session.reject("AUTO-000002", note="Reject reason")

    initial_active_ids = list(session.result().objects["review_id"])
    initial_history = list(session.result().history)
    initial_action_counter = session._action_counter
    initial_man_counter = session._man_counter

    # Candidate 1: valid polygon in empty space
    m1 = np.zeros((100, 100), dtype=bool)
    m1[60:75, 60:75] = True
    roi1 = mask_to_roi(m1, name="Candidate_1")

    # Candidate 2: valid polygon overlapping Candidate 1
    m2 = np.zeros((100, 100), dtype=bool)
    m2[70:85, 70:85] = True
    roi2 = mask_to_roi(m2, name="Candidate_2")

    zip_path = tmp_path / "intra_batch_collision.zip"
    write_roi_set([roi1, roi2], zip_path)

    with pytest.raises(InvalidReviewMaskError, match="overlaps with manual mask"):
        import_rois_as_manual(session, zip_path)

    # Verify complete state invariance
    post_snapshot = session.result()
    assert list(post_snapshot.objects["review_id"]) == initial_active_ids
    assert list(post_snapshot.history) == initial_history
    assert session._action_counter == initial_action_counter
    assert session._man_counter == initial_man_counter
    assert session._active_objects["AUTO-000001"].review_note == "Researcher note"
    assert "AUTO-000002" in session._retired_objects
    assert session._retired_objects["AUTO-000002"].retirement_reason == "rejected"
