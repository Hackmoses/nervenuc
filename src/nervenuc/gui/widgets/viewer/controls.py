"""Viewer controls: Z navigation, channel visibility, and display contrast."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QScrollArea,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ...state import GuiSessionState


class ViewerControlsWidget(QWidget):
    """Control panel for Z-stack slice navigation, channel contrast, and overlays."""

    slice_changed = Signal(int)
    display_settings_changed = Signal()
    fit_requested = Signal()

    def __init__(self, state: GuiSessionState) -> None:
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Z Slice Navigation ---
        self.z_group = QGroupBox("Z-Slice Navigation")
        z_layout = QHBoxLayout(self.z_group)

        self.btn_z_prev = QPushButton("◀")
        self.btn_z_prev.setFixedWidth(30)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.btn_z_next = QPushButton("▶")
        self.btn_z_next.setFixedWidth(30)
        self.lbl_z_info = QLabel("Slice: 1 / 1")

        z_layout.addWidget(self.btn_z_prev)
        z_layout.addWidget(self.z_slider)
        z_layout.addWidget(self.btn_z_next)
        z_layout.addWidget(self.lbl_z_info)

        layout.addWidget(self.z_group)

        # --- View Actions ---
        act_layout = QHBoxLayout()
        self.btn_fit = QPushButton("Fit to View")
        self.cb_overlay = QCheckBox("Show Overlay")
        self.cb_overlay.setChecked(True)
        self.cb_preprocessed = QCheckBox('Show processed detection image')
        self.cb_preprocessed.setEnabled(False)
        self.cb_preprocessed.toggled.connect(self._on_preprocessed_toggled)
        self.cb_ids = QCheckBox("Show IDs")
        self.cb_ids.setChecked(False)

        act_layout.addWidget(self.btn_fit)
        act_layout.addWidget(self.cb_overlay)
        act_layout.addWidget(self.cb_ids)
        layout.addWidget(self.cb_preprocessed)
        act_layout.addStretch()

        layout.addLayout(act_layout)

        self.channel_group = QGroupBox("Channel display")
        self.channel_layout = QVBoxLayout(self.channel_group)
        channel_scroll = QScrollArea()
        channel_scroll.setWidgetResizable(True)
        channel_scroll.setMinimumHeight(140)
        channel_scroll.setMaximumHeight(170)
        channel_scroll.setWidget(self.channel_group)
        layout.addWidget(channel_scroll)
        self.channel_controls = {}

        self.btn_z_prev.setAccessibleName("Previous Z slice")
        self.btn_z_next.setAccessibleName("Next Z slice")
        self.z_slider.setAccessibleName("Z slice")
        self.cb_ids.hide()  # No ID renderer is currently provided.

        # Connect signals
        self.z_slider.valueChanged.connect(self._on_z_slider_changed)
        self.btn_z_prev.clicked.connect(self._on_z_prev)
        self.btn_z_next.clicked.connect(self._on_z_next)
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        self.cb_overlay.toggled.connect(self._on_overlay_toggled)
        self.cb_ids.toggled.connect(self._on_ids_toggled)

    def set_z_range(self, total_slices: int) -> None:
        """Configure Z slider for the given number of slices."""
        total_slices = max(1, total_slices)
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, total_slices - 1)
        self.z_slider.setValue(min(self.state.current_z_index, total_slices - 1))
        self.state.current_z_index = self.z_slider.value()
        self.z_slider.blockSignals(False)
        self.z_group.setVisible(total_slices > 1)
        self._update_z_label()

    def populate_channels(self) -> None:
        """Expose presentation settings without changing scientific channel assignments."""
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.channel_controls.clear()
        count = (self.state.inspection.size_c or 1) if self.state.inspection else 0
        for channel in range(count):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            visible = QCheckBox(f"Channel {channel}")
            visible.setChecked(self.state.display.channel_visible.get(channel, True))
            color = QComboBox()
            color.addItems(["Gray", "Blue", "Green", "Red", "Cyan", "Magenta", "Yellow"])
            color.setCurrentText(self.state.display.channel_color.get(channel, "Gray"))
            color.setAccessibleName(f"Channel {channel} color")
            gamma = QDoubleSpinBox()
            gamma.setRange(0.1, 5.0)
            gamma.setSingleStep(0.1)
            gamma.setValue(self.state.display.channel_gamma.get(channel, 1.0))
            gamma.setPrefix("Gamma ")
            gamma.setAccessibleName(f"Channel {channel} gamma")
            visible.toggled.connect(lambda value, c=channel: self._set_channel_setting("channel_visible", c, value))
            color.currentTextChanged.connect(lambda value, c=channel: self._set_channel_setting("channel_color", c, value))
            gamma.valueChanged.connect(lambda value, c=channel: self._set_channel_setting("channel_gamma", c, value))
            for control in (visible, color, gamma):
                row_layout.addWidget(control)
            self.channel_layout.addWidget(row)
            self.channel_controls[channel] = (visible, color, gamma)
        self.channel_group.setVisible(count > 0)

    def _set_channel_setting(self, setting, channel, value) -> None:
        getattr(self.state.display, setting)[channel] = value
        self.display_settings_changed.emit()

    def _on_z_slider_changed(self, value: int) -> None:
        self.state.current_z_index = value
        self._update_z_label()
        self.slice_changed.emit(value)

    def _on_z_prev(self) -> None:
        val = max(0, self.z_slider.value() - 1)
        self.z_slider.setValue(val)

    def _on_z_next(self) -> None:
        val = min(self.z_slider.maximum(), self.z_slider.value() + 1)
        self.z_slider.setValue(val)

    def _update_z_label(self) -> None:
        cur = self.z_slider.value() + 1
        tot = max(1, self.z_slider.maximum() + 1)
        self.lbl_z_info.setText(f"Slice: {cur} / {tot}")
        self.btn_z_prev.setEnabled(cur > 1)
        self.btn_z_next.setEnabled(cur < tot)

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.state.display.show_segmentation_overlay = checked
        self.display_settings_changed.emit()

    def _on_preprocessed_toggled(self, checked):
        self.state.display.show_preprocessed = checked
        self.display_settings_changed.emit()

    def _on_ids_toggled(self, checked: bool) -> None:
        self.state.display.show_object_ids = checked
        self.display_settings_changed.emit()
