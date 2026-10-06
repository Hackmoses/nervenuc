"""Regression coverage for desktop display and source switching."""
import numpy as np
import tifffile
from PySide6.QtWidgets import QFileDialog, QMessageBox
from nervenuc.gui.main_window import MainWindow


def open_source(win, monkeypatch, path, qtbot):
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: (_ for _ in ()).throw(AssertionError(args[-1])))
    win._on_open_file()
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)


def test_channel_display_and_projection_keep_source_indices(qtbot, tmp_path, monkeypatch):
    data = np.zeros((3, 3, 16, 16), dtype=np.uint16)
    data[:, 0, 2:6, 2:6] = 1000
    data[:, 1, 8:12, 8:12] = 1000
    data[:, 2, 4:8, 8:12] = 1000
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, data, metadata={'axes': 'ZCYX'}, photometric='minisblack')
    win = MainWindow()
    qtbot.addWidget(win)
    open_source(win, monkeypatch, path, qtbot)
    assert win.state.draft.mode == 'projected_2d'
    assert win.spin_z_stop.maximum() == 3
    assert win.state.display.channel_color == {0: 'Blue', 1: 'Green', 2: 'Red'}
    visible, color, gamma = win.viewer_controls.channel_controls[0]
    color.setCurrentText('Cyan')
    assert win.state.display.channel_color[0] == 'Cyan'
    win.channel_table.table.cellWidget(0, 2).setCurrentText('Signal to measure')
    win.channel_table.table.cellWidget(0, 3).setText('Marker_0')
    win.channel_table.table.cellWidget(2, 2).setCurrentText('Objects to count')
    win.channel_table.confirm_mapping.setChecked(True)
    win._on_prepare_session()
    image = win.canvas._base_item.pixmap().toImage()
    assert image.pixelColor(3, 3).getRgb()[:3] == (0, 255, 255)
    assert image.pixelColor(9, 5).getRgb()[:3] == (255, 0, 0)
    np.testing.assert_array_equal(win.state.loaded_raw_array, data)


def test_new_source_resets_mapping_display_and_slice(qtbot, tmp_path, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    stack = tmp_path / 'stack.tif'
    tifffile.imwrite(stack, np.arange(4*2*8*8, dtype=np.uint16).reshape(4,2,8,8), metadata={'axes':'ZCYX'}, photometric='minisblack')
    open_source(win, monkeypatch, stack, qtbot)
    win.viewer_controls.z_slider.setValue(3)
    win.viewer_controls.channel_controls[0][0].setChecked(False)
    image = tmp_path / 'image.tif'
    tifffile.imwrite(image, np.arange(64, dtype=np.uint16).reshape(8,8), metadata={'axes':'YX'})
    open_source(win, monkeypatch, image, qtbot)
    assert win.state.draft.mode == 'direct_2d'
    assert win.state.current_z_index == 0
    assert win.state.draft.marker_channels == {}
    assert win.state.display.channel_visible == {0:True}
    assert win.state.display.channel_color == {0:'Gray'}
    assert win.channel_table.table.rowCount() == 1
    assert win.spin_z_stop.value() == 1
    assert win.viewer_controls.lbl_z_info.text() == 'Slice: 1 / 1'
