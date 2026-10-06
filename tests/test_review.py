"""Tests for segmentation quality control and manual review workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nervenuc.qc import (
    IncompleteReviewError,
    InvalidReviewMaskError,
    InvalidReviewOperationError,
    NerveNucReviewError,
    ReviewAction,
    ReviewedSegmentationResult,
    ReviewSession,
    UnknownReviewObjectError,
    start_review,
)
from nervenuc.segmentation import (
    SegmentationConfig,
    SegmentationResult,
    segment_nuclei,
)

EXPECTED_REVIEWED_COLUMNS = [
    "review_id",
    "current_label",
    "source_nucleus_id",
    "review_status",
    "review_origin",
    "parent_review_ids",
    "review_note",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "touches_border",
    "qc_status",
]


def _create_synthetic_segmentation_result() -> SegmentationResult:
    """Create a synthetic 2D SegmentationResult with 3 distinct nuclei."""
    labels = np.zeros((100, 100), dtype=np.int32)
    # Nucleus 1 at (25, 25), radius 8
    y, x = np.ogrid[:100, :100]
    mask1 = (y - 25) ** 2 + (x - 25) ** 2 <= 8**2
    # Nucleus 2 at (25, 75), radius 8
    mask2 = (y - 25) ** 2 + (x - 75) ** 2 <= 8**2
    # Nucleus 3 at (75, 50), radius 10
    mask3 = (y - 75) ** 2 + (x - 50) ** 2 <= 10**2

    labels[mask1] = 1
    labels[mask2] = 2
    labels[mask3] = 3

    objects_df = pd.DataFrame(
        [
            {
                "nucleus_id": 1,
                "area_px": int(np.sum(mask1)),
                "centroid_y": 25.0,
                "centroid_x": 25.0,
                "bbox_min_y": 17,
                "bbox_min_x": 17,
                "bbox_max_y": 34,
                "bbox_max_x": 34,
                "touches_border": False,
                "qc_status": "accepted",
            },
            {
                "nucleus_id": 2,
                "area_px": int(np.sum(mask2)),
                "centroid_y": 25.0,
                "centroid_x": 75.0,
                "bbox_min_y": 17,
                "bbox_min_x": 67,
                "bbox_max_y": 34,
                "bbox_max_x": 84,
                "touches_border": False,
                "qc_status": "accepted",
            },
            {
                "nucleus_id": 3,
                "area_px": int(np.sum(mask3)),
                "centroid_y": 75.0,
                "centroid_x": 50.0,
                "bbox_min_y": 65,
                "bbox_min_x": 40,
                "bbox_max_y": 86,
                "bbox_max_x": 61,
                "touches_border": False,
                "qc_status": "accepted",
            },
        ]
    )

    return SegmentationResult(
        labels=labels,
        objects=objects_df,
        config=SegmentationConfig(),
    )


def test_start_review_initial_state() -> None:
    """Test starting a review session on an M2 SegmentationResult."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    result = session.result()
    assert isinstance(result, ReviewedSegmentationResult)
    assert result.labels.shape == (100, 100)
    assert result.labels.dtype == np.int32
    assert np.array_equal(result.labels, source_res.labels)
    assert len(result.objects) == 3
    assert list(result.objects.columns) == EXPECTED_REVIEWED_COLUMNS

    assert list(result.objects["review_id"]) == ["AUTO-000001", "AUTO-000002", "AUTO-000003"]
    assert list(result.objects["current_label"]) == [1, 2, 3]
    assert list(result.objects["source_nucleus_id"]) == [1, 2, 3]
    assert (result.objects["review_status"] == "unreviewed").all()
    assert (result.objects["review_origin"] == "automated").all()
    assert len(result.history) == 0


def test_review_finalize_requires_every_active_object_resolved() -> None:
    """Untouched and partially reviewed sessions cannot produce finalized results."""
    session = start_review(_create_synthetic_segmentation_result())
    with pytest.raises(IncompleteReviewError, match="3 active object"):
        session.finalize()

    session.accept("AUTO-000001")
    with pytest.raises(IncompleteReviewError, match="2 active object"):
        session.finalize()

    session.accept("AUTO-000002")
    session.accept("AUTO-000003")
    finalized = session.finalize()
    assert (finalized.objects["review_status"] == "accepted").all()
    assert [action.action_type for action in finalized.history] == ["accept", "accept", "accept"]


def test_edited_objects_resolve_and_rejected_objects_do_not_block_finalize() -> None:
    """Edited active objects are resolved; rejected objects are inactive."""
    edited = start_review(_create_synthetic_segmentation_result())
    merged_id = edited.merge(["AUTO-000001", "AUTO-000002", "AUTO-000003"])
    finalized_edited = edited.finalize()
    assert finalized_edited.objects.iloc[0]["review_id"] == merged_id
    assert finalized_edited.objects.iloc[0]["review_status"] == "edited"

    rejected = start_review(_create_synthetic_segmentation_result())
    for review_id in list(rejected.result().objects["review_id"]):
        rejected.reject(review_id)
    finalized_rejected = rejected.finalize()
    assert finalized_rejected.objects.empty
    assert len(finalized_rejected.history) == 3


def test_source_result_isolation() -> None:
    """Test that destructive review operations never modify the original SegmentationResult."""
    source_res = _create_synthetic_segmentation_result()
    orig_labels_copy = source_res.labels.copy()
    orig_objects_copy = source_res.objects.copy(deep=True)

    session = start_review(source_res)
    session.reject("AUTO-000001")

    # Add a manual mask
    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[50:55, 50:55] = True
    session.add(manual_mask)

    # Merge remaining
    session.merge(["AUTO-000002", "AUTO-000003"])

    assert np.array_equal(source_res.labels, orig_labels_copy)
    assert source_res.objects.equals(orig_objects_copy)


def test_returned_result_snapshot_stability() -> None:
    """Test that a returned result snapshot is not mutated by subsequent session operations."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    res1 = session.result()
    labels1_copy = res1.labels.copy()
    objects1_copy = res1.objects.copy(deep=True)
    history1_len = len(res1.history)

    # Further operations on session
    session.accept("AUTO-000001")
    session.reject("AUTO-000002")
    res2 = session.result()

    # res1 must remain identical
    assert np.array_equal(res1.labels, labels1_copy)
    assert res1.objects.equals(objects1_copy)
    assert len(res1.history) == history1_len
    assert len(res2.objects) == 2
    assert len(res2.history) == 2


def test_accept_nucleus() -> None:
    """Test accepting an active nucleus."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    session.accept("AUTO-000002", note="Confirmed valid nuclear shape")
    res = session.result()

    row2 = res.objects[res.objects["review_id"] == "AUTO-000002"].iloc[0]
    assert row2["review_status"] == "accepted"
    assert row2["review_note"] == "Confirmed valid nuclear shape"

    assert len(res.history) == 1
    action = res.history[0]
    assert action.action_id == "ACT-000001"
    assert action.action_type == "accept"
    assert action.target_review_ids == ("AUTO-000002",)
    assert action.resulting_review_ids == ("AUTO-000002",)
    assert action.note == "Confirmed valid nuclear shape"


def test_reject_automated_nucleus() -> None:
    """Test rejecting an automated candidate nucleus."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    session.reject("AUTO-000002", note="Debris / background fluorescence")
    res = session.result()

    assert len(res.objects) == 2
    assert "AUTO-000002" not in res.objects["review_id"].values
    assert list(res.objects["review_id"]) == ["AUTO-000001", "AUTO-000003"]
    assert list(res.objects["current_label"]) == [1, 2]
    assert list(np.unique(res.labels)) == [0, 1, 2]

    assert len(res.history) == 1
    action = res.history[0]
    assert action.action_type == "reject"
    assert action.target_review_ids == ("AUTO-000002",)
    assert action.resulting_review_ids == ()
    assert action.note == "Debris / background fluorescence"


def test_reject_manual_object_fails() -> None:
    """Test that reject() fails when called on a manual or edited object."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[50:55, 50:55] = True
    man_id = session.add(manual_mask)

    with pytest.raises(InvalidReviewOperationError, match="Use delete"):
        session.reject(man_id)


def test_delete_manual_and_edited_nucleus() -> None:
    """Test deleting a manually added or edited nucleus."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[50:55, 50:55] = True
    man_id = session.add(manual_mask)

    session.delete(man_id, note="Drawn incorrectly")
    res = session.result()

    assert man_id not in res.objects["review_id"].values
    assert len(res.objects) == 3
    assert list(res.objects["current_label"]) == [1, 2, 3]

    assert len(res.history) == 2
    delete_action = res.history[1]
    assert delete_action.action_type == "delete"
    assert delete_action.target_review_ids == (man_id,)
    assert delete_action.resulting_review_ids == ()


def test_delete_automated_nucleus_fails() -> None:
    """Test that delete() fails on automated nuclei."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    with pytest.raises(InvalidReviewOperationError, match="Use reject"):
        session.delete("AUTO-000001")


def test_restore_rejected_nucleus() -> None:
    """Test restoring a previously rejected automated nucleus."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    session.reject("AUTO-000002")
    assert len(session.result().objects) == 2

    session.restore("AUTO-000002", note="Re-evaluated, valid nucleus")
    res = session.result()

    assert len(res.objects) == 3
    assert "AUTO-000002" in res.objects["review_id"].values
    row2 = res.objects[res.objects["review_id"] == "AUTO-000002"].iloc[0]
    assert row2["review_status"] == "unreviewed"
    assert row2["review_note"] == "Re-evaluated, valid nucleus"

    assert len(res.history) == 2
    restore_action = res.history[1]
    assert restore_action.action_type == "restore"
    assert restore_action.target_review_ids == ("AUTO-000002",)
    assert restore_action.resulting_review_ids == ("AUTO-000002",)


def test_restore_deleted_object_fails() -> None:
    """Test that restore() fails on deleted manual objects."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[50:55, 50:55] = True
    man_id = session.add(manual_mask)
    session.delete(man_id)

    with pytest.raises(InvalidReviewOperationError, match="Only rejected nuclei can be restored"):
        session.restore(man_id)


def test_restore_conflict_raises_error() -> None:
    """Test that restoring a rejected nucleus fails if its area is now occupied."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    session.reject("AUTO-000002")

    # Add manual mask that overlaps with AUTO-000002's location (center at 25, 75)
    conflict_mask = np.zeros((100, 100), dtype=bool)
    conflict_mask[24:26, 74:76] = True
    session.add(conflict_mask)

    with pytest.raises(InvalidReviewOperationError, match="overlaps with active nucleus"):
        session.restore("AUTO-000002")


def test_add_manual_nucleus() -> None:
    """Test adding a manual nucleus mask."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[50:60, 10:20] = True
    man_id = session.add(manual_mask, note="Manually traced DAPI signal")

    assert man_id == "MAN-000001"
    res = session.result()
    assert len(res.objects) == 4

    row = res.objects[res.objects["review_id"] == man_id].iloc[0]
    assert row["current_label"] == 4
    assert row["source_nucleus_id"] is None
    assert row["review_status"] == "edited"
    assert row["review_origin"] == "manual_add"
    assert row["parent_review_ids"] == ()
    assert row["area_px"] == 100
    assert bool(row["touches_border"]) is False
    assert row["qc_status"] == "accepted"

    assert len(res.history) == 1
    action = res.history[0]
    assert action.action_type == "add"
    assert action.target_review_ids == ()
    assert action.resulting_review_ids == (man_id,)


def test_add_overlapping_manual_nucleus_fails() -> None:
    """Test that adding a mask that overlaps an active nucleus raises InvalidReviewMaskError."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    overlap_mask = np.zeros((100, 100), dtype=bool)
    overlap_mask[24:30, 24:30] = True  # Overlaps AUTO-000001 (center 25, 25)

    with pytest.raises(InvalidReviewMaskError, match="overlaps with existing active nucleus"):
        session.add(overlap_mask)


def test_merge_nuclei() -> None:
    """Test merging two active nuclei into a single child object."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    child_id = session.merge(["AUTO-000001", "AUTO-000002"], note="Over-split single nucleus")
    assert child_id == "EDIT-000001"

    res = session.result()
    assert len(res.objects) == 2  # AUTO-000003 and EDIT-000001
    assert "AUTO-000001" not in res.objects["review_id"].values
    assert "AUTO-000002" not in res.objects["review_id"].values

    child_row = res.objects[res.objects["review_id"] == child_id].iloc[0]
    assert child_row["review_origin"] == "merge"
    assert child_row["review_status"] == "edited"
    assert child_row["parent_review_ids"] == ("AUTO-000001", "AUTO-000002")
    assert child_row["source_nucleus_id"] is None

    assert len(res.history) == 1
    action = res.history[0]
    assert action.action_type == "merge"
    assert action.target_review_ids == ("AUTO-000001", "AUTO-000002")
    assert action.resulting_review_ids == (child_id,)


def test_merge_invalid_inputs() -> None:
    """Test invalid merge inputs (less than 2 IDs, duplicates, unknown IDs)."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    with pytest.raises(InvalidReviewOperationError, match="at least 2"):
        session.merge(["AUTO-000001"])  # type: ignore[arg-type]

    with pytest.raises(InvalidReviewOperationError, match="Duplicate"):
        session.merge(["AUTO-000001", "AUTO-000001"])

    with pytest.raises(UnknownReviewObjectError, match="Unknown"):
        session.merge(["AUTO-000001", "NONEXISTENT"])


def test_split_nucleus_with_partial_coverage() -> None:
    """Test splitting a nucleus into two sub-masks with partial parent coverage."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    # AUTO-000003 is at (75, 50), radius 10 (bounding box roughly 65..85, 40..60)
    # Split into top half and bottom half inside parent
    mask_top = np.zeros((100, 100), dtype=bool)
    mask_bottom = np.zeros((100, 100), dtype=bool)

    y, x = np.ogrid[:100, :100]
    parent_mask = (y - 75) ** 2 + (x - 50) ** 2 <= 10**2

    mask_top[(y < 74) & parent_mask] = True
    mask_bottom[(y > 76) & parent_mask] = True  # row 74-76 omitted intentionally

    child_ids = session.split("AUTO-000003", [mask_top, mask_bottom], note="Separated doublet")
    assert len(child_ids) == 2
    assert child_ids == ("EDIT-000001", "EDIT-000002")

    res = session.result()
    assert len(res.objects) == 4  # AUTO-000001, AUTO-000002, EDIT-000001, EDIT-000002
    assert "AUTO-000003" not in res.objects["review_id"].values

    for cid in child_ids:
        row = res.objects[res.objects["review_id"] == cid].iloc[0]
        assert row["review_origin"] == "split"
        assert row["review_status"] == "edited"
        assert row["parent_review_ids"] == ("AUTO-000003",)

    assert len(res.history) == 1
    action = res.history[0]
    assert action.action_type == "split"
    assert action.target_review_ids == ("AUTO-000003",)
    assert action.resulting_review_ids == child_ids


def test_split_invalid_masks() -> None:
    """Test split failure cases: overlapping masks, masks exceeding parent, empty masks."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    # 1. Less than 2 replacement masks
    m1 = (source_res.labels == 3)
    with pytest.raises(InvalidReviewMaskError, match="at least 2"):
        session.split("AUTO-000003", [m1])

    # 2. Masks exceeding parent boundary
    m_outside = np.zeros((100, 100), dtype=bool)
    m_outside[0:5, 0:5] = True
    with pytest.raises(InvalidReviewMaskError, match="exceed the boundary"):
        session.split("AUTO-000003", [m1, m_outside])

    # 3. Overlapping replacement masks
    with pytest.raises(InvalidReviewMaskError, match="overlap each other"):
        session.split("AUTO-000003", [m1, m1])

    # 4. Empty replacement mask
    m_empty = np.zeros((100, 100), dtype=bool)
    with pytest.raises(InvalidReviewMaskError, match="at least one foreground pixel"):
        session.split("AUTO-000003", [m1, m_empty])


def test_notes_and_metadata() -> None:
    """Test setting and updating notes on active and retired objects."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    session.set_note("AUTO-000001", "Initial QC: acceptable")
    res1 = session.result()
    row1 = res1.objects[res1.objects["review_id"] == "AUTO-000001"].iloc[0]
    assert row1["review_note"] == "Initial QC: acceptable"

    # Reject and set note on retired object
    session.reject("AUTO-000002")
    session.set_note("AUTO-000002", "Updated rejection note: apoptotic body")

    assert len(session.result().history) == 3
    action = session.result().history[-1]
    assert action.action_type == "set_note"
    assert action.target_review_ids == ("AUTO-000002",)
    assert action.resulting_review_ids == ("AUTO-000002",)


def test_audit_history_completeness_and_lineage() -> None:
    """Test full multi-step review workflow and verify genealogical lineage."""
    source_res = _create_synthetic_segmentation_result()
    session = start_review(source_res)

    # 1. Accept
    session.accept("AUTO-000001")
    # 2. Merge AUTO-000002 and AUTO-000003 -> EDIT-000001
    merged_id = session.merge(["AUTO-000002", "AUTO-000003"])
    # 3. Split EDIT-000001 -> EDIT-000002, EDIT-000003
    y, x = np.ogrid[:100, :100]
    mask2 = (y - 25) ** 2 + (x - 75) ** 2 <= 8**2
    mask3 = (y - 75) ** 2 + (x - 50) ** 2 <= 10**2
    child_ids = session.split(merged_id, [mask2, mask3])

    res = session.result()
    assert len(res.history) == 3
    assert [a.action_type for a in res.history] == ["accept", "merge", "split"]

    c1_row = res.objects[res.objects["review_id"] == child_ids[0]].iloc[0]
    assert c1_row["parent_review_ids"] == (merged_id,)


def test_review_determinism() -> None:
    """Test determinism of review workflow with an injected fake clock."""
    source_res = _create_synthetic_segmentation_result()

    fake_timestamps = iter([
        "2026-08-15T00:00:01Z",
        "2026-08-15T00:00:02Z",
        "2026-08-15T00:00:03Z",
        "2026-08-15T00:00:04Z",
    ])

    def fake_clock() -> str:
        return next(fake_timestamps)

    s1 = start_review(source_res, clock=fake_clock)
    s1.accept("AUTO-000001")
    s1.reject("AUTO-000002")
    res1 = s1.result()

    fake_timestamps2 = iter([
        "2026-08-15T00:00:01Z",
        "2026-08-15T00:00:02Z",
        "2026-08-15T00:00:03Z",
        "2026-08-15T00:00:04Z",
    ])

    def fake_clock2() -> str:
        return next(fake_timestamps2)

    s2 = start_review(source_res, clock=fake_clock2)
    s2.accept("AUTO-000001")
    s2.reject("AUTO-000002")
    res2 = s2.result()

    assert np.array_equal(res1.labels, res2.labels)
    assert res1.objects.equals(res2.objects)
    assert res1.history == res2.history
