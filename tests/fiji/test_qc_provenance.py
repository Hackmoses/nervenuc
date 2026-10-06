"""Tests for Fiji ROI QC provenance tracking and workflow downstream invalidation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nervenuc.classification import MarkerThreshold
from nervenuc.fiji import (
    export_review_rois,
    import_rois_as_manual,
    mask_to_roi,
    replace_with_fiji_rois,
    write_roi_set,
)
from nervenuc.markers import MarkerDefinition
from nervenuc.qc import ReviewSession, start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import AnalysisConfig, ImageAnalysisSession, ImageSourceProvenance


def _create_synthetic_workflow_setup() -> tuple[ImageAnalysisSession, np.ndarray, dict[str, np.ndarray]]:
    """Create configured ImageAnalysisSession and synthetic 2D image data with source provenance."""
    seg_cfg = SegmentationConfig(threshold_method="otsu", min_area_px=10)
    m_brdu = MarkerDefinition("BrdU")
    m_sox10 = MarkerDefinition("SOX10")

    t_brdu = MarkerThreshold("BrdU", threshold=25.0)
    t_sox10 = MarkerThreshold("SOX10", threshold=40.0)

    d1 = BiologicalMetricDefinition(
        name="Proliferation",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    config = AnalysisConfig(
        segmentation=seg_cfg,
        markers=(m_brdu, m_sox10),
        thresholds=(t_brdu, t_sox10),
        metrics=(d1,),
    )

    session = ImageAnalysisSession(config)

    # 2 nuclei in nuclear channel
    nuc = np.zeros((100, 100), dtype=np.float32)
    nuc[20:40, 20:40] = 100.0
    nuc[60:80, 60:80] = 100.0

    brdu = np.zeros((100, 100), dtype=np.float32)
    brdu[20:40, 20:40] = 50.0

    sox10 = np.zeros((100, 100), dtype=np.float32)
    sox10[20:40, 20:40] = 80.0
    sox10[60:80, 60:80] = 80.0

    prov = ImageSourceProvenance(
        source_kind="direct_2d",
        source_id="microscopy_raw_img_42",
        source_hash="sha256_raw_microscopy_hash_12345",
    )

    session.set_image(nuc, {"BrdU": brdu, "SOX10": sox10}, source_provenance=prov)
    return session, nuc, {"BrdU": brdu, "SOX10": sox10}


def test_source_identity_invariance(tmp_path: Path) -> None:
    """Raw microscopy image source_id and source_hash in ImageAnalysisSession remain invariant."""
    session, nuc, markers = _create_synthetic_workflow_setup()
    session.segment()
    assert session.source_provenance is not None
    orig_source_id = session.source_provenance.source_id
    orig_source_hash = session.source_provenance.source_hash

    # Start review
    review_session = session.start_review()

    # Export active objects
    out_zip = tmp_path / "exported.zip"
    export_review_rois(review_session, out_zip)

    # Import new manual object
    man_mask = np.zeros((100, 100), dtype=bool)
    man_mask[80:95, 80:95] = True
    man_roi = mask_to_roi(man_mask, name="New_Manual_Nuc")
    man_zip = tmp_path / "new_man.zip"
    write_roi_set([man_roi], man_zip)

    import_res = import_rois_as_manual(review_session, man_zip)

    # Invariance check: source provenance metadata unchanged
    assert session.source_provenance.source_id == orig_source_id
    assert session.source_provenance.source_hash == orig_source_hash

    # Finalize and set back
    for rid in review_session.result().objects["review_id"]:
        if rid.startswith("AUTO-"):
            review_session.accept(rid)
    reviewed = review_session.finalize()
    session.set_reviewed_result(reviewed)

    assert session.source_provenance.source_id == orig_source_id
    assert session.source_provenance.source_hash == orig_source_hash

    # Artifact hash is traceable via import_res, but not replacing source_hash
    assert import_res.roi_file_sha256 != session.source_provenance.source_hash
    assert import_res.roi_file_sha256 == hashlib.sha256(man_zip.read_bytes()).hexdigest()


def test_downstream_invalidation_after_fiji_qc_changes(tmp_path: Path) -> None:
    """Modifying review state with imported Fiji ROIs invalidates downstream results in ImageAnalysisSession."""
    workflow, nuc_img, markers = _create_synthetic_workflow_setup()

    # Run full initial pipeline
    workflow.segment()
    workflow.auto_accept_segmentation()
    workflow.measure()
    workflow.classify()
    workflow.summarize()

    status = workflow.status()
    assert status["measured"] is True
    assert status["classified"] is True
    assert status["summarized"] is True
    assert workflow.measurement_result is not None
    assert workflow.classification_result is not None
    assert workflow.summary_result is not None

    # Researcher starts a manual review session to incorporate Fiji edits
    review_session = workflow.start_review()

    # Starting review immediately invalidates reviewed_result, measurement, classification, summary
    status_after_start = workflow.status()
    assert status_after_start["review_complete"] is False
    assert status_after_start["measured"] is False
    assert status_after_start["classified"] is False
    assert status_after_start["summarized"] is False
    assert workflow.reviewed_result is None
    assert workflow.measurement_result is None
    assert workflow.classification_result is None
    assert workflow.summary_result is None

    # Import manual nucleus from Fiji
    manual_mask = np.zeros((100, 100), dtype=bool)
    manual_mask[85:95, 85:95] = True
    man_roi = mask_to_roi(manual_mask, name="Imported_Cell")
    roi_zip = tmp_path / "fiji_additions.zip"
    write_roi_set([man_roi], roi_zip)

    import_rois_as_manual(review_session, roi_zip)

    # Accept existing automated objects and finalize
    for rid in review_session.result().objects["review_id"]:
        if rid.startswith("AUTO-"):
            review_session.accept(rid)
    reviewed = review_session.finalize()

    # Update workflow with new reviewed state
    workflow.set_reviewed_result(reviewed)

    # Verify review is complete, but downstream results remain uncomputed / clean
    status_updated = workflow.status()
    assert status_updated["review_complete"] is True
    assert status_updated["measured"] is False
    assert status_updated["classified"] is False
    assert status_updated["summarized"] is False

    # Downstream execution runs cleanly on updated review state: 3 nuclei x 2 markers = 6 rows
    meas = workflow.measure()
    assert len(meas.measurements) == 6
    assert set(meas.measurements["review_id"].unique()) == {"AUTO-000001", "AUTO-000002", "MAN-000001"}

    clf = workflow.classify()
    assert len(clf.phenotypes) == 3
    assert set(clf.phenotypes["review_id"].unique()) == {"AUTO-000001", "AUTO-000002", "MAN-000001"}

    summary = workflow.summarize()
    assert summary.metrics is not None
    assert len(summary.metrics) == 1
    assert workflow.status()["summarized"] is True
