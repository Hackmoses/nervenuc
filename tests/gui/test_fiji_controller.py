"""Tests for FijiController lifecycle, capabilities, worker dispatch, and error mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QSettings, QThreadPool

from nervenuc.gui.controllers.fiji_controller import (
    FijiController,
    is_pyimagej_available,
    is_roifile_available,
)
from nervenuc.gui.errors import NerveNucGuiError
from nervenuc.gui.state import AnalysisDraft, GuiSessionState
from nervenuc.preparation import PreparedImageSet, StackInspectionResult
from nervenuc.qc import InvalidReviewMaskError, ReviewSession, UnknownReviewObjectError
from nervenuc.segmentation import SegmentationConfig, SegmentationResult
from nervenuc.workflow import AnalysisConfig, ImageAnalysisSession


def _make_test_session() -> ImageAnalysisSession:
    img = np.zeros((100, 100), dtype=np.uint16)
    img[10:30, 10:30] = 500
    img[60:80, 60:80] = 600

    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[60:80, 60:80] = 2

    seg_res = SegmentationResult(labels=labels, objects=pd.DataFrame(), config=SegmentationConfig())
    cfg = AnalysisDraft().build_core_2d_config()
    sess = ImageAnalysisSession(cfg)
    sess.set_image(img, {"DefaultMarker": np.zeros_like(img)})
    sess.segmentation_result = seg_res
    sess.start_review()
    return sess


def test_fiji_controller_preferences_qsettings(tmp_path: Path) -> None:
    """Verify FijiController saves, retrieves, and validates paths using QSettings."""
    state = GuiSessionState()
    ctrl = FijiController(state)

    # Clear setting first
    ctrl.set_configured_fiji_path(None)
    assert ctrl.get_configured_fiji_path() is None

    # Set valid existing path
    dummy_exe = tmp_path / "ImageJ-win64.exe"
    dummy_exe.touch()
    ctrl.set_configured_fiji_path(dummy_exe)
    assert ctrl.get_configured_fiji_path() == dummy_exe.resolve()

    # If file is deleted, get_configured_fiji_path should discard it and return None
    dummy_exe.unlink()
    assert ctrl.get_configured_fiji_path() is None


def test_fiji_controller_capabilities() -> None:
    """Verify capability flags distinguish roifile and executable presence."""
    state = GuiSessionState()
    ctrl = FijiController(state)

    assert ctrl.can_exchange_rois() == is_roifile_available()


def test_fiji_controller_error_translation() -> None:
    """Verify FijiController translates specific exceptions to researcher-facing messages."""
    state = GuiSessionState()
    ctrl = FijiController(state)

    from nervenuc.fiji.errors import (
        FijiConversionError,
        FijiExecutionError,
        FijiUnavailableError,
        InvalidRoiGeometryError,
        JavaUnavailableError,
        UnsupportedRoiError,
    )

    t, m = ctrl.translate_fiji_error(FijiUnavailableError("Missing Fiji"))
    assert "Not Found" in t
    assert "Tools -> Fiji" in m

    t, m = ctrl.translate_fiji_error(JavaUnavailableError("Java error"))
    assert "Java Runtime" in t

    t, m = ctrl.translate_fiji_error(UnsupportedRoiError("Line ROI"))
    assert "Unsupported ROI Type" in t
    assert "Polygon" in m

    t, m = ctrl.translate_fiji_error(InvalidRoiGeometryError("Out of bounds"))
    assert "Invalid ROI Geometry" in t

    t, m = ctrl.translate_fiji_error(InvalidReviewMaskError("Overlap detected"))
    assert "QC Review Mask Error" in t

    t, m = ctrl.translate_fiji_error(UnknownReviewObjectError("AUTO-999"))
    assert "Unknown Review Object" in t


def test_fiji_controller_export_rois(tmp_path: Path, qtbot: any) -> None:
    """Verify export_rois dispatches worker and writes valid ROI set."""
    state = GuiSessionState()
    sess = _make_test_session()
    state.session_2d = sess
    ctrl = FijiController(state)

    out_zip = tmp_path / "exported_rois.zip"
    results = []

    def on_finished(p: Path) -> None:
        results.append(p)

    ctrl.export_rois(out_zip, overwrite=True, on_finished=on_finished)
    qtbot.waitUntil(lambda: len(results) == 1, timeout=5000)

    assert out_zip.is_file()
    assert out_zip.stat().st_size > 0


def test_fiji_controller_import_rois_invalidates_downstream(tmp_path: Path, qtbot: any) -> None:
    """Verify import_rois marks state.results_stale and notifies state listeners."""
    from nervenuc.fiji.roi import mask_to_roi, write_roi

    state = GuiSessionState()
    sess = _make_test_session()
    state.session_2d = sess
    ctrl = FijiController(state)

    m = np.zeros((100, 100), dtype=bool)
    m[40:50, 40:50] = True
    roi = mask_to_roi(m, name="ManualImport")
    roi_file = tmp_path / "manual.roi"
    write_roi(roi, roi_file)

    notified = []
    state.subscribe(lambda: notified.append(True))

    results = []
    ctrl.import_rois(roi_file, on_finished=lambda r: results.append(r))
    qtbot.waitUntil(lambda: len(results) == 1, timeout=5000)

    assert len(results) == 1
    assert state.results_stale is True
    assert len(notified) > 0
    assert "MAN-000001" in [rec.review_id for rec in results[0].imported]


def test_fiji_controller_replace_rois_split(tmp_path: Path, qtbot: any) -> None:
    """Verify replace_rois executes split replacement on active object."""
    from nervenuc.fiji.roi import mask_to_roi, write_roi_set

    state = GuiSessionState()
    sess = _make_test_session()
    state.session_2d = sess
    ctrl = FijiController(state)

    # AUTO-000001 is at [10:30, 10:30]
    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:20, 10:30] = True
    roi1 = mask_to_roi(m1, name="Child_1")

    m2 = np.zeros((100, 100), dtype=bool)
    m2[20:30, 10:30] = True
    roi2 = mask_to_roi(m2, name="Child_2")

    split_zip = tmp_path / "split.zip"
    write_roi_set([roi1, roi2], split_zip)

    results = []
    errors = []
    ctrl.replace_rois(
        target_review_id="AUTO-000001",
        roi_path=split_zip,
        on_finished=lambda r: results.append(r),
        on_error=lambda e: errors.append(e),
    )
    qtbot.waitUntil(lambda: len(results) == 1 or len(errors) > 0, timeout=5000)
    assert len(errors) == 0, f"Error occurred: {errors}"
    assert len(results) == 1
    res = results[0]
    assert res.target_review_id == "AUTO-000001"
    assert len(res.child_review_ids) == 2
    assert state.results_stale is True


def test_fiji_controller_missing_roifile_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify controlled error when roifile is missing."""
    state = GuiSessionState()
    ctrl = FijiController(state)

    monkeypatch.setattr("nervenuc.gui.controllers.fiji_controller.is_roifile_available", lambda: False)
    assert ctrl.can_exchange_rois() is False

    with pytest.raises(NerveNucGuiError, match="requires the 'roifile' package"):
        ctrl.export_rois("dummy.zip")
