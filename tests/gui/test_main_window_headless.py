"""Headless UI tests for MainWindow layout, navigation, and rendering."""

from __future__ import annotations

import os
import pytest

from nervenuc.gui.main_window import MainWindow

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
import numpy as np
import tifffile


def test_main_window_initialization_and_navigation(qtbot: any) -> None:
    """Verify MainWindow initializes cleanly and tab navigation updates inspector stack."""
    win = MainWindow()
    qtbot.addWidget(win)

    assert win.windowTitle().startswith("NerveNuc")
    assert win.nav_list.count() == 4
    assert win.inspector_stack.count() == 10
    assert win.inspector_stack.currentIndex() == win.nav_list.currentRow() == 0

    # Test tab navigation
    win.nav_list.setCurrentRow(2)  # Analyze
    assert win.inspector_stack.currentIndex() == 3

    win.nav_list.setCurrentRow(3)  # Image results
    assert win.inspector_stack.currentIndex() == 9


def test_source_preview_slice_navigation_and_sidebar_mouse_clicks(qtbot, tmp_path, monkeypatch):
    source = np.zeros((3, 2, 24, 24), dtype=np.uint16)
    source[0, 0, 2:5, 2:5] = 1000
    source[1, 0, 10:15, 10:15] = 1000
    path = tmp_path / "preview.tif"
    tifffile.imwrite(path, source, metadata={"axes": "ZCYX"}, photometric="minisblack")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    win._on_open_file()
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
    qtbot.wait(20)  # Let layout and scroll-area geometry settle before mouse interaction.
    assert win.state.session_2d is None
    assert win.state.session_3d is None
    assert win.lbl_src_path.text() == str(path)
    first = win.canvas._base_item.pixmap().toImage()
    assert not first.isNull()
    qtbot.mouseClick(win.viewer_controls.btn_z_next, Qt.MouseButton.LeftButton)
    second = win.canvas._base_item.pixmap().toImage()
    assert win.state.current_z_index == 1
    assert first != second
    np.testing.assert_array_equal(win.state.loaded_raw_array, source)
    for index in range(4):
        win.nav_list.scrollToItem(win.nav_list.item(index))
        qtbot.wait(20)
        qtbot.mouseClick(
            win.nav_list.viewport(), Qt.MouseButton.LeftButton,
            pos=win.nav_list.visualItemRect(win.nav_list.item(index)).center(),
        )
        assert win.inspector_stack.currentIndex() == win._nav_pages[index]
        assert win.inspector_stack.currentWidget().isVisible()
