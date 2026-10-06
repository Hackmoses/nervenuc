"""Modal configuration dialog for discovering and setting the external Fiji executable path."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controllers.fiji_controller import FijiController


class FijiConfigDialog(QDialog):
    """Configuration dialog for locating and validating the Fiji/ImageJ installation."""

    def __init__(self, fiji_controller: FijiController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.fiji_ctrl = fiji_controller
        self.setWindowTitle("Fiji / ImageJ Configuration")
        self.resize(550, 200)

        layout = QVBoxLayout(self)

        # Header description
        lbl_info = QLabel(
            "Configure the path to your local Fiji installation (Fiji.app or ImageJ executable).\n"
            "This path is stored locally as a user preference and does not modify scientific outputs."
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        # Path input row
        path_layout = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Auto-detect or choose Fiji executable / Fiji.app path...")
        current_path = self.fiji_ctrl.get_configured_fiji_path()
        if current_path:
            self.txt_path.setText(str(current_path))

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse)
        path_layout.addWidget(self.txt_path)
        path_layout.addWidget(btn_browse)
        layout.addLayout(path_layout)

        # Action row: Auto-detect and Validate
        action_layout = QHBoxLayout()
        self.btn_autodetect = QPushButton("Auto-Detect")
        self.btn_autodetect.clicked.connect(self._on_autodetect)

        self.btn_validate = QPushButton("Test / Validate")
        self.btn_validate.clicked.connect(self._on_validate)

        action_layout.addWidget(self.btn_autodetect)
        action_layout.addWidget(self.btn_validate)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        # Status text
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-style: italic;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        # Dialog Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_save = QPushButton("Save")
        self.btn_save.setStyleSheet("font-weight: bold;")
        self.btn_save.clicked.connect(self._on_save)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(self.btn_save)
        btn_box.addWidget(self.btn_cancel)
        layout.addLayout(btn_box)

    def _on_browse(self) -> None:
        """Open file dialog to pick an ImageJ executable or Fiji directory."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Fiji Executable or Application",
            "",
            "Executable Files (*.exe ImageJ* fiji*);;All Files (*)",
        )
        if path:
            self.txt_path.setText(path)
            self._on_validate()

    def _on_autodetect(self) -> None:
        """Run system discovery for Fiji."""
        try:
            from nervenuc.fiji.launcher import discover_fiji
            info = discover_fiji()
            if info is not None:
                self.txt_path.setText(str(info.executable_path))
                self.lbl_status.setText(
                    f"Discovered ({info.discovery_source}): {info.executable_path.name} in {info.app_directory}"
                )
            else:
                self.lbl_status.setText("Fiji was not found in standard system paths or environment variables.")
                QMessageBox.information(
                    self,
                    "Auto-Detection",
                    "Fiji was not found in standard paths. Please use 'Browse...' to locate Fiji.app manually.",
                )
        except Exception as e:
            self.lbl_status.setText(f"Auto-detection error: {e}")

    def _on_validate(self) -> None:
        """Validate candidate executable without launching JVM/PyImageJ."""
        cand_str = self.txt_path.text().strip()
        if not cand_str:
            self.lbl_status.setText("Path is empty.")
            return

        cand = Path(cand_str)
        try:
            from nervenuc.fiji.launcher import discover_fiji
            info = discover_fiji(explicit_path=cand)
            if info is not None:
                fiji_type = "Fiji distribution" if info.is_fiji else "ImageJ"
                self.lbl_status.setText(f"Valid {fiji_type}: {info.executable_path}")
                self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.lbl_status.setText("Invalid: No executable found in selected path.")
                self.lbl_status.setStyleSheet("color: red;")
        except Exception as e:
            self.lbl_status.setText(f"Validation failed: {e}")
            self.lbl_status.setStyleSheet("color: red;")

    def _on_save(self) -> None:
        """Save path to user settings and close."""
        cand_str = self.txt_path.text().strip()
        if cand_str:
            cand = Path(cand_str)
            if not cand.exists():
                ret = QMessageBox.warning(
                    self,
                    "Path Does Not Exist",
                    f"The specified path does not exist on disk:\n{cand}\n\nSave anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return
            self.fiji_ctrl.set_configured_fiji_path(cand)
        else:
            self.fiji_ctrl.set_configured_fiji_path(None)

        self.accept()
