"""Widget for defining denominator-aware biological metrics."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..state import BiologicalMetricDraft, GuiSessionState


class MetricBuilderWidget(QWidget):
    """Builder for configuring denominator-aware biological metrics."""

    metric_added = Signal()

    def __init__(self, state: GuiSessionState) -> None:
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Define Biological Metric")
        form = QFormLayout(self.group)

        self.edit_name = QLineEdit("Proliferation Index")
        self.combo_denom = QComboBox()
        self.combo_num = QComboBox()

        form.addRow("Metric Name:", self.edit_name)
        form.addRow("Denominator (Base):", self.combo_denom)
        form.addRow("Numerator (Target):", self.combo_num)

        self.btn_add = QPushButton("Add Metric Definition")
        self.btn_add.setStyleSheet("font-weight: bold;")
        form.addRow("", self.btn_add)

        layout.addWidget(self.group)

        # Existing Metrics Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Denominator", "Numerator"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.btn_add.clicked.connect(self._on_add_metric)
        self.refresh_markers()

    def refresh_markers(self) -> None:
        """Update dropdown choices based on active marker names."""
        markers = list(self.state.draft.marker_channels.keys())
        self.combo_denom.clear()
        self.combo_num.clear()

        self.combo_denom.addItem("All Nuclei", ("all",))
        for m in markers:
            self.combo_denom.addItem(f"{m} (+)", ("pos", m))
            self.combo_denom.addItem(f"{m} (-)", ("neg", m))

        for m in markers:
            self.combo_num.addItem(f"{m} (+)", ("pos", m))
            self.combo_num.addItem(f"{m} (-)", ("neg", m))

        self.refresh_table()

    def refresh_table(self) -> None:
        """Refresh table of defined metrics."""
        drafts = self.state.draft.metric_drafts
        self.table.setRowCount(len(drafts))
        for r, m in enumerate(drafts):
            d_str = ", ".join(m.denominator_positive) if m.denominator_positive else "All Nuclei"
            n_str = ", ".join(m.numerator_positive) if m.numerator_positive else "(None)"
            self.table.setItem(r, 0, QTableWidgetItem(m.name))
            self.table.setItem(r, 1, QTableWidgetItem(d_str))
            self.table.setItem(r, 2, QTableWidgetItem(n_str))

    def _on_add_metric(self) -> None:
        name = self.edit_name.text().strip() or "Unnamed Metric"
        d_data = self.combo_denom.currentData()
        n_data = self.combo_num.currentData()

        denom_pos = ()
        denom_neg = ()
        if d_data and d_data[0] == "pos":
            denom_pos = (d_data[1],)
        elif d_data and d_data[0] == "neg":
            denom_neg = (d_data[1],)

        num_pos = ()
        num_neg = ()
        if n_data and n_data[0] == "pos":
            num_pos = (n_data[1],)
        elif n_data and n_data[0] == "neg":
            num_neg = (n_data[1],)

        draft = BiologicalMetricDraft(
            name=name,
            denominator_positive=denom_pos,
            denominator_negative=denom_neg,
            numerator_positive=num_pos,
            numerator_negative=num_neg,
        )

        self.state.draft.metric_drafts.append(draft)
        self.state.mark_draft_modified()
        self.refresh_table()
        self.metric_added.emit()
