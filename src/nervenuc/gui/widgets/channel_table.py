"""Widget for explicit channel role and marker name assignment."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..state import GuiSessionState


class ChannelAssignmentTable(QWidget):
    """Explicit channel assignment table (Nuclear, Signal to measure, Ignore)."""

    mapping_changed = Signal()

    def __init__(self, state: GuiSessionState) -> None:
        super().__init__()
        self.state = state
        self._populating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Channel", "Metadata Hint", "Assigned Role", "Signal Name"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.confirm_mapping = QCheckBox(
            "Confirm channel assignments"
        )
        self.confirm_mapping.setToolTip("Confirm which channel contains nuclei and which channels measure markers.")
        self.confirm_mapping.toggled.connect(self._set_confirmation)
        self.confirm_mapping.hide()

    def populate_from_inspection(self) -> None:
        """Populate table rows based on inspect_stack results."""
        insp = self.state.inspection
        if insp is None:
            self.table.setRowCount(0)
            return

        self._populating = True
        num_c = insp.size_c or 1
        self.table.setRowCount(num_c)

        for c_idx in range(num_c):
            # Column 0: Channel Index
            item_ch = QTableWidgetItem(f"Channel {c_idx}")
            item_ch.setFlags(item_ch.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(c_idx, 0, item_ch)

            # Column 1: Hint
            hint = ""
            if insp.channel_names and c_idx < len(insp.channel_names):
                hint = insp.channel_names[c_idx]
            item_hint = QTableWidgetItem(hint)

            item_hint.setFlags(item_hint.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(c_idx, 1, item_hint)

            # Column 2: Role Combo
            combo = QComboBox()
            combo.addItems(["Objects to count", "Signal to measure", "Ignore"])
            if c_idx == 0:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentIndex(1)
            combo.currentIndexChanged.connect(self._sync_draft_from_ui)
            self.table.setCellWidget(c_idx, 2, combo)

            # Column 3: Marker Name
            line_edit = QLineEdit(hint if hint else (f"Marker_{c_idx}" if c_idx > 0 else ""))
            line_edit.setEnabled(c_idx != 0)
            line_edit.textChanged.connect(self._sync_draft_from_ui)
            self.table.setCellWidget(c_idx, 3, line_edit)

        self.table.setColumnHidden(1, not any(insp.channel_names or ()))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 78)
        self.table.setColumnWidth(2, 150)
        self.table.setMaximumHeight(min(280, 55 + num_c * 34))
        self.table.setMinimumHeight(min(150, 55 + num_c * 34))
        self._populating = False
        self._sync_draft_from_ui()

    def _sync_draft_from_ui(self) -> None:
        """Read UI values and populate AnalysisDraft."""
        if self._populating:
            return
        marker_dict: dict[str, int] = {}
        nuclear_channel = 0
        nuclear_count = 0
        marker_names: list[str] = []
        mapping_valid = True

        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 2)
            line_edit = self.table.cellWidget(r, 3)
            if not isinstance(combo, QComboBox) or not isinstance(line_edit, QLineEdit):
                continue

            role = combo.currentText()
            if role == "Objects to count":
                nuclear_count += 1
                nuclear_channel = r
                line_edit.setEnabled(False)
            elif role == "Signal to measure":
                line_edit.setEnabled(True)
                m_name = line_edit.text().strip()
                if not m_name or m_name in marker_names:
                    mapping_valid = False
                else:
                    marker_names.append(m_name)
                    marker_dict[m_name] = r
            else:
                line_edit.setEnabled(False)

        self.state.draft.nuclear_channel = nuclear_channel
        self.state.draft.marker_channels = marker_dict
        self.state.draft.channel_mapping_confirmed = False
        mapping_valid = mapping_valid and nuclear_count == 1
        self.confirm_mapping.blockSignals(True)
        self.confirm_mapping.setChecked(False)
        self.confirm_mapping.setEnabled(mapping_valid)
        self.confirm_mapping.blockSignals(False)
        self.state.mark_draft_modified()
        self.mapping_changed.emit()

    def validate_mapping(self) -> None:
        self._sync_draft_from_ui()
        if not self.confirm_mapping.isEnabled():
            raise ValueError('Assign exactly one object channel and give each signal channel a unique name.')
        self.state.draft.channel_mapping_confirmed = True

    def _set_confirmation(self, confirmed: bool) -> None:
        self.state.draft.channel_mapping_confirmed = bool(
            confirmed and self.confirm_mapping.isEnabled()
        )
        self.state.mark_draft_modified()
        self.mapping_changed.emit()
