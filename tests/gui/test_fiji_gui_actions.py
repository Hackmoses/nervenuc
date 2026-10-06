"""Tests for MainWindow Fiji menu actions, enablement, and QC synchronization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from nervenuc.fiji.roi import mask_to_roi, write_roi, write_roi_set
from nervenuc.gui.main_window import MainWindow
from nervenuc.segmentation import SegmentationConfig, SegmentationResult
from nervenuc.workflow import AnalysisConfig, ImageAnalysisSession


def _populate_session_with_2d_qc(win: MainWindow) -> None:
    """Helper to populate win.state with a valid 2D session and ReviewSession."""
    img = np.zeros((100, 100), dtype=np.uint16)
    img[10:30, 10:30] = 500
    img[60:80, 60:80] = 600

    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[60:80, 60:80] = 2

    seg_res = SegmentationResult(labels=labels, objects=pd.DataFrame(), config=SegmentationConfig())
    cfg = win.state.draft.build_core_2d_config()
    sess = ImageAnalysisSession(cfg)
    sess.set_image(img, {"DefaultMarker": np.zeros_like(img)})
    sess.segmentation_result = seg_res
    sess.start_review()

    win.state.session_2d = sess
    win.qc_widget.refresh_from_session()
    win.state.notify()


def test_main_window_fiji_menu_presence_and_enablement(qtbot: any) -> None:
    """Verify Fiji menu presence and dynamic action enablement."""
    win = MainWindow()
    qtbot.addWidget(win)

    # Initial state (no file loaded, no review session)
    assert hasattr(win, "act_fiji_open_image")
    assert hasattr(win, "act_fiji_export_rois")
    assert hasattr(win, "act_fiji_import_rois")
    assert hasattr(win, "act_fiji_replace_rois")
    assert hasattr(win, "act_fiji_config")

    assert win.act_fiji_open_image.isEnabled() is False
    assert win.act_fiji_export_rois.isEnabled() is False
    assert win.act_fiji_import_rois.isEnabled() is False
    assert win.act_fiji_replace_rois.isEnabled() is False
    assert win.act_fiji_config.isEnabled() is True

    # Populate 2D QC
    _populate_session_with_2d_qc(win)

    assert win.act_fiji_open_image.isEnabled() is True
    assert win.act_fiji_export_rois.isEnabled() is True
    assert win.act_fiji_import_rois.isEnabled() is True
    assert win.act_fiji_replace_rois.isEnabled() is False  # No row selected yet

    # Select first row in QC table
    win.qc_widget.table.selectRow(0)
    assert win.act_fiji_replace_rois.isEnabled() is True


def test_main_window_fiji_import_rois_untrusted_names(tmp_path: Path, qtbot: any) -> None:
    """Verify importing Fiji ROIs through GUI creates MAN-* IDs even if ROI name matches AUTO-000001."""
    win = MainWindow()
    qtbot.addWidget(win)
    _populate_session_with_2d_qc(win)

    initial_count = len(win.state.session_2d.review_session.result().objects)

    # Create ROI in empty space [40:55, 40:55] named maliciously as "AUTO-000001"
    m = np.zeros((100, 100), dtype=bool)
    m[40:55, 40:55] = True
    roi = mask_to_roi(m, name="AUTO-000001")
    roi_file = tmp_path / "untrusted.roi"
    write_roi(roi, roi_file)

    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(roi_file), "")):
        with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
            win._on_fiji_import_rois()
            qtbot.waitUntil(lambda: mock_info.called and not win.fiji_ctrl.is_qc_worker_running, timeout=5000)

    # Check session state
    objs = win.state.session_2d.review_session.result().objects
    assert "MAN-000001" in objs["review_id"].values
    assert win.state.results_stale is True
    assert win.qc_widget.table.rowCount() == initial_count + 1


def test_main_window_fiji_replace_selected_object_split(tmp_path: Path, qtbot: any) -> None:
    """Verify split replacement of selected QC row creates EDIT-* children and retires parent."""
    win = MainWindow()
    qtbot.addWidget(win)
    _populate_session_with_2d_qc(win)

    # Select row 0 (AUTO-000001 at [10:30, 10:30])
    win.qc_widget.table.selectRow(0)

    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:20, 10:30] = True
    roi1 = mask_to_roi(m1, name="Part1")

    m2 = np.zeros((100, 100), dtype=bool)
    m2[20:30, 10:30] = True
    roi2 = mask_to_roi(m2, name="Part2")

    split_zip = tmp_path / "split_set.zip"
    write_roi_set([roi1, roi2], split_zip)

    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(split_zip), "")):
        with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                win._on_fiji_replace_rois()
                qtbot.waitUntil(
                    lambda: mock_info.called and not win.fiji_ctrl.is_qc_worker_running,
                    timeout=5000,
                )

    objs = win.state.session_2d.review_session.result().objects
    assert "AUTO-000001" not in objs["review_id"].values
    assert "EDIT-000001" in objs["review_id"].values
    assert "EDIT-000002" in objs["review_id"].values
    assert win.state.results_stale is True


def test_main_window_fiji_atomic_failure_leaves_state_intact(tmp_path: Path, qtbot: any) -> None:
    """Verify that an atomic import failure in worker leaves QC session and UI completely intact."""
    win = MainWindow()
    qtbot.addWidget(win)
    _populate_session_with_2d_qc(win)

    initial_active_ids = list(win.state.session_2d.review_session.result().objects["review_id"])

    # ROI colliding with AUTO-000001
    m = np.zeros((100, 100), dtype=bool)
    m[15:25, 15:25] = True
    roi = mask_to_roi(m, name="Colliding")
    colliding_file = tmp_path / "colliding.roi"
    write_roi(roi, colliding_file)

    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(colliding_file), "")):
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            win._on_fiji_import_rois()
            qtbot.waitUntil(lambda: mock_crit.called and not win.fiji_ctrl.is_qc_worker_running, timeout=5000)

    # Assert completely untouched
    post_active_ids = list(win.state.session_2d.review_session.result().objects["review_id"])
    assert post_active_ids == initial_active_ids
    assert win.qc_widget.table.rowCount() == len(initial_active_ids)
