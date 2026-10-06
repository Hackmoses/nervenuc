"""Tests for FijiConfigDialog interaction and preferences persistence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from nervenuc.gui.controllers.fiji_controller import FijiController
from nervenuc.gui.state import GuiSessionState
from nervenuc.gui.widgets.fiji_config_dialog import FijiConfigDialog


def test_fiji_config_dialog_initialization(qtbot: any) -> None:
    """Verify dialog opens and displays empty or configured path."""
    state = GuiSessionState()
    ctrl = FijiController(state)
    ctrl.set_configured_fiji_path(None)

    dlg = FijiConfigDialog(ctrl)
    qtbot.addWidget(dlg)

    assert dlg.windowTitle() == "Fiji / ImageJ Configuration"
    assert dlg.txt_path.text() == ""


def test_fiji_config_dialog_validation_and_save(tmp_path: Path, qtbot: any) -> None:
    """Verify validate updates status and save persists setting."""
    state = GuiSessionState()
    ctrl = FijiController(state)
    ctrl.set_configured_fiji_path(None)

    # Create dummy Fiji.app structure
    fiji_app = tmp_path / "Fiji.app"
    fiji_app.mkdir()
    (fiji_app / "plugins").mkdir()
    exe = fiji_app / "ImageJ-win64.exe"
    exe.touch()

    dlg = FijiConfigDialog(ctrl)
    qtbot.addWidget(dlg)

    dlg.txt_path.setText(str(fiji_app))
    dlg._on_validate()

    assert "Valid" in dlg.lbl_status.text()

    dlg._on_save()
    assert ctrl.get_configured_fiji_path() == fiji_app.resolve()


def test_fiji_config_dialog_autodetect(tmp_path: Path, qtbot: any) -> None:
    """Verify autodetect button invokes discovery and updates text field."""
    state = GuiSessionState()
    ctrl = FijiController(state)

    fiji_app = tmp_path / "Fiji.app"
    fiji_app.mkdir()
    (fiji_app / "plugins").mkdir()
    exe = fiji_app / "ImageJ-win64.exe"
    exe.touch()

    from nervenuc.fiji.models import FijiExecutableInfo

    mock_info = FijiExecutableInfo(
        executable_path=exe,
        discovery_source="standard_path",
        is_fiji=True,
        app_directory=fiji_app,
    )

    with patch("nervenuc.fiji.launcher.discover_fiji", return_value=mock_info):
        dlg = FijiConfigDialog(ctrl)
        qtbot.addWidget(dlg)
        dlg._on_autodetect()
        assert str(exe) in dlg.txt_path.text()
        assert "Discovered" in dlg.lbl_status.text()
