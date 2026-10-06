"""Tests for M14 3D quality control and review session."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.preparation import PreparedVolumeSet, VolumePreparationProvenance
from nervenuc.qc3d import (
    IncompleteReview3DError,
    InvalidReviewOperation3DError,
    ReviewedSegmentation3DResult,
    UnknownReviewObject3DError,
    start_review_3d,
)
from nervenuc.segmentation3d import segment_nuclei_3d


def _get_segmented_volume() -> tuple[PreparedVolumeSet, any]:
    shape = (16, 64, 64)
    vol = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((x - 20) / 6.0) ** 2 <= 1.0
    mask2 = ((z - 10) / 3.0) ** 2 + ((y - 45) / 8.0) ** 2 + ((x - 45) / 8.0) ** 2 <= 1.0
    vol[mask1] = 1000
    vol[mask2] = 1200

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={},
        z_start=0,
        z_stop=shape[0],
    )
    prep = PreparedVolumeSet(
        nuclear_volume=vol,
        marker_volumes={},
        spacing=None,
        provenance=prov,
    )
    seg = segment_nuclei_3d(prep, source_id="vol_qc")
    return prep, seg


def test_review_session_3d_auto_accept() -> None:
    """Test auto_accept on 3D review session."""
    _, seg = _get_segmented_volume()
    rev = start_review_3d(seg)
    assert rev.status()["is_complete"] is False
    assert rev.status()["unreviewed_count"] == 2

    rev.auto_accept()
    assert rev.status()["is_complete"] is True
    assert rev.status()["accepted_count"] == 2

    reviewed = rev.finalize()
    assert isinstance(reviewed, ReviewedSegmentation3DResult)
    assert reviewed.source_id == seg.source_id
    assert reviewed.active_count == 2
    assert len(reviewed.objects) == 2


def test_review_session_3d_manual_operations() -> None:
    """Test accept, reject, restore, and notes."""
    _, seg = _get_segmented_volume()
    rev = start_review_3d(seg)

    rev.accept(1)
    rev.reject(2, reason="artifact")
    rev.set_note(1, "good nucleus")

    assert rev.status()["is_complete"] is True
    assert rev.status()["accepted_count"] == 1
    assert rev.status()["rejected_count"] == 1

    reviewed = rev.finalize()
    assert reviewed.active_count == 1
    assert list(reviewed.objects["source_nucleus_id"]) == ["1"]


    # Rejection of unknown ID
    rev2 = start_review_3d(seg)
    with pytest.raises(UnknownReviewObject3DError):
        rev2.accept(999)

    # Incomplete review finalize
    with pytest.raises(IncompleteReview3DError):
        rev2.finalize()

    # Operations after finalize
    rev.finalize()  # idempotent
    with pytest.raises(InvalidReviewOperation3DError):
        rev.accept(1)
