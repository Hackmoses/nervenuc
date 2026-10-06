"""Widget for configuring per-marker thresholds and live phenotype preview."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..state import GuiSessionState, MarkerThresholdDraft


class MarkerThresholdCard(QWidget):
    """Card widget for tuning a single marker's threshold with preview."""

    preview_requested = Signal()
    apply_requested = Signal()
    threshold_changed = Signal()

    def __init__(self, marker_name: str, state: GuiSessionState) -> None:
        super().__init__()
        self.marker_name = marker_name
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.group = QGroupBox(f"Threshold: {marker_name}")
        form = QFormLayout(self.group)

        self.combo_field = QComboBox()
        self.combo_field.addItems(["mean_intensity", "median_intensity", "max_intensity", "min_intensity", "integrated_intensity"])

        self.combo_op = QComboBox()
        self.combo_op.addItems([">=", ">"])

        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(0, 65535)
        self.spin_cutoff.setValue(200.0)

        form.addRow("Measurement Field:", self.combo_field)
        form.addRow("Comparison:", self.combo_op)
        form.addRow("Threshold Cutoff:", self.spin_cutoff)

        layout.addWidget(self.group)

        # Preview results & actions
        self.lbl_preview_result = QLabel("Preview: Not evaluated")
        layout.addWidget(self.lbl_preview_result)

        btn_layout = QHBoxLayout()
        self.btn_preview = QPushButton("Preview Phenotypes")
        self.btn_apply = QPushButton("Apply cutoffs and view results")
        self.btn_apply.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_apply)

        layout.addLayout(btn_layout)

        self.spin_cutoff.setRange(-1e12, 1e12)
        saved = self.state.draft.threshold_drafts.get(self.marker_name)
        if saved:
            self.combo_field.setCurrentText(saved.measurement_field)
            self.combo_op.setCurrentText(saved.comparison_operator)
            self.spin_cutoff.setValue(saved.cutoff)
        self.advanced = QPushButton('Advanced cutoff options')
        self.advanced.setCheckable(True)
        self.advanced.toggled.connect(lambda checked: (form.setRowVisible(self.combo_field, checked), form.setRowVisible(self.combo_op, checked)))
        layout.addWidget(self.advanced)
        form.setRowVisible(self.combo_field, False)
        form.setRowVisible(self.combo_op, False)
        # Signals
        self.combo_field.currentIndexChanged.connect(self._sync_to_draft)
        self.combo_op.currentIndexChanged.connect(self._sync_to_draft)
        self.spin_cutoff.valueChanged.connect(self._sync_to_draft)
        self.btn_preview.clicked.connect(self.preview_requested.emit)
        self.btn_apply.clicked.connect(self.apply_requested.emit)

        if self.marker_name not in self.state.draft.threshold_drafts:
            self.state.draft.threshold_drafts[self.marker_name] = MarkerThresholdDraft(
                marker_name=self.marker_name,
                cutoff=self.spin_cutoff.value(),
                measurement_field=self.combo_field.currentText(),
                comparison_operator=self.combo_op.currentText(),  # type: ignore[arg-type]
            )

    def _sync_to_draft(self) -> None:
        draft = MarkerThresholdDraft(
            marker_name=self.marker_name,
            cutoff=self.spin_cutoff.value(),
            measurement_field=self.combo_field.currentText(),
            comparison_operator=self.combo_op.currentText(),  # type: ignore[arg-type]
        )
        self.state.draft.threshold_drafts[self.marker_name] = draft
        self.state.mark_draft_modified()
        self.threshold_changed.emit()

    def update_preview_result(self, pos_count: int, neg_count: int) -> None:
        """Update live preview statistics label."""
        total = pos_count + neg_count
        pct = (pos_count / total * 100.0) if total > 0 else 0.0
        self.lbl_preview_result.setText(f"Preview: {pos_count} / {total} Positive ({pct:.1f}%)")
