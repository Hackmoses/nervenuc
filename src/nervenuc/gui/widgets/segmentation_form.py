"""Form for configuring 2D and 3D nuclear segmentation parameters."""

from __future__ import annotations

from dataclasses import fields
from nervenuc.segmentation import SegmentationConfig
from nervenuc.segmentation3d import Segmentation3DConfig

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..state import GuiSessionState


class SegmentationConfigForm(QWidget):
    """Form providing mode-aware parameter editing and segmentation dispatch."""

    run_segmentation_requested = Signal()
    config_changed = Signal()

    def __init__(self, state: GuiSessionState) -> None:
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Segmentation Parameters")
        self.form_layout = QFormLayout(self.group)

        # Common Parameters
        self.combo_method = QComboBox()
        self.combo_method.addItems(["otsu", "manual"])
        self.spin_manual = QDoubleSpinBox()
        self.spin_manual.setRange(0, 65535)
        self.spin_manual.setValue(500)
        self.spin_manual.setVisible(False)

        # 2D & 3D Specific Controls
        self.spin_sigma_xy = QDoubleSpinBox()
        self.spin_sigma_xy.setRange(0, 50)
        self.spin_sigma_xy.setValue(1.0)

        self.spin_sigma_z = QDoubleSpinBox()
        self.spin_sigma_z.setRange(0, 50)
        self.spin_sigma_z.setValue(1.0)

        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 100000)
        self.spin_min_size.setValue(50)

        self.cb_fill_holes = QCheckBox("Fill Holes")
        self.cb_fill_holes.setChecked(True)

        self.cb_exclude_border = QCheckBox("Exclude Borders")
        self.cb_exclude_border.setChecked(False)

        self.cb_exclude_z_border = QCheckBox("Exclude Z Borders")
        self.cb_exclude_z_border.setChecked(False)

        self.form_layout.addRow("Threshold Method:", self.combo_method)
        self.form_layout.addRow("Manual Threshold:", self.spin_manual)
        self.form_layout.addRow("Smoothing Sigma (XY):", self.spin_sigma_xy)
        self.form_layout.addRow("Smoothing Sigma (Z):", self.spin_sigma_z)
        self.form_layout.addRow("Min Nucleus Size:", self.spin_min_size)
        self.form_layout.addRow("", self.cb_fill_holes)
        self.form_layout.addRow("", self.cb_exclude_border)
        self.form_layout.addRow("", self.cb_exclude_z_border)

        self.btn_advanced = QPushButton('Advanced detection settings')
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.group)
        layout.addWidget(self.btn_advanced)
        layout.addStretch()

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        layout.addWidget(self.btn_run)

        # Signals
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.spin_manual.valueChanged.connect(self._sync_to_draft)
        self.spin_sigma_xy.valueChanged.connect(self._sync_to_draft)
        self.spin_sigma_z.valueChanged.connect(self._sync_to_draft)
        self.spin_min_size.valueChanged.connect(self._sync_to_draft)
        self.cb_fill_holes.toggled.connect(self._sync_to_draft)
        self.cb_exclude_border.toggled.connect(self._sync_to_draft)
        self.cb_exclude_z_border.toggled.connect(self._sync_to_draft)
        self.btn_run.clicked.connect(self.run_segmentation_requested.emit)

        self.form_layout.setRowVisible(self.spin_manual, False)
        self.update_mode_visibility()
        self._toggle_advanced(False)

    def _toggle_advanced(self, expanded):
        for control in (self.combo_method, self.spin_manual, self.spin_sigma_xy, self.spin_sigma_z, self.cb_fill_holes, self.cb_exclude_border, self.cb_exclude_z_border):
            self.form_layout.setRowVisible(control, expanded)
        if getattr(self, "preprocessing_managed", False):
            self.form_layout.setRowVisible(self.spin_sigma_xy, False)
            self.form_layout.setRowVisible(self.spin_sigma_z, False)
        if expanded:
            self.update_mode_visibility()
            self.form_layout.setRowVisible(self.spin_manual, self.combo_method.currentText() == "manual")

    def load_from_draft(self) -> None:
        """Show preset values without overwriting them through widget signals."""
        values = self.state.draft.segmentation_values
        is_3d = self.state.draft.mode == "volume_3d"
        controls = {
            self.combo_method: values.get("threshold_method", "otsu"),
            self.spin_manual: values.get("manual_threshold") or 500,
            self.spin_sigma_xy: values.get("gaussian_sigma_xy_px" if is_3d else "gaussian_sigma", 1.0),
            self.spin_sigma_z: values.get("gaussian_sigma_z_slices", 1.0),
            self.spin_min_size: values.get("min_volume_voxels" if is_3d else "min_area_px", 50),
            self.cb_fill_holes: values.get("fill_holes", True),
            self.cb_exclude_border: values.get("exclude_xy_border" if is_3d else "exclude_border", False),
            self.cb_exclude_z_border: values.get("exclude_z_border", False),
        }
        for control, value in controls.items():
            control.blockSignals(True)
            if isinstance(control, QComboBox):
                control.setCurrentText(value)
            elif isinstance(control, QCheckBox):
                control.setChecked(value)
            else:
                control.setValue(value)
            control.blockSignals(False)
        self.form_layout.setRowVisible(self.spin_manual, self.combo_method.currentText() == "manual")
        self.update_mode_visibility()

    def update_mode_visibility(self) -> None:
        """Show/hide 2D vs 3D parameter controls according to active mode."""
        is_3d = self.state.draft.mode == "volume_3d"
        expanded = self.btn_advanced.isChecked()
        self.form_layout.setRowVisible(self.spin_sigma_z, is_3d and expanded and not getattr(self, "preprocessing_managed", False))
        self.cb_exclude_z_border.setVisible(is_3d and expanded)
        self.cb_exclude_border.setText("Exclude XY Borders" if is_3d else "Exclude Image Borders")
        self.group.setTitle("Detection settings")
        self.form_layout.labelForField(self.spin_min_size).setText("Minimum object volume (voxels):" if is_3d else "Minimum object area (pixels):")

    def _on_method_changed(self) -> None:
        is_man = self.combo_method.currentText() == "manual"
        self.form_layout.setRowVisible(self.spin_manual, is_man)
        self._sync_to_draft()

    def _sync_to_draft(self) -> None:
        is_3d = self.state.draft.mode == "volume_3d"
        method = self.combo_method.currentText()
        man_val = self.spin_manual.value() if method == "manual" else None

        config_type = Segmentation3DConfig if is_3d else SegmentationConfig
        allowed = {field.name for field in fields(config_type)}
        retained = {key: value for key, value in self.state.draft.segmentation_values.items() if key in allowed}
        if is_3d:
            self.state.draft.segmentation_values = {
                **retained,
                "threshold_method": method,
                "manual_threshold": man_val,
                "gaussian_sigma_xy_px": self.spin_sigma_xy.value(),
                "gaussian_sigma_z_slices": self.spin_sigma_z.value(),
                "min_volume_voxels": self.spin_min_size.value(),
                "fill_holes": self.cb_fill_holes.isChecked(),
                "exclude_xy_border": self.cb_exclude_border.isChecked(),
                "exclude_z_border": self.cb_exclude_z_border.isChecked(),
            }
        else:
            self.state.draft.segmentation_values = {
                **retained,
                "threshold_method": method,
                "manual_threshold": man_val,
                "gaussian_sigma": self.spin_sigma_xy.value(),
                "min_area_px": self.spin_min_size.value(),
                "fill_holes": self.cb_fill_holes.isChecked(),
                "exclude_border": self.cb_exclude_border.isChecked(),
            }

        self.state.mark_draft_modified()
        self.config_changed.emit()
