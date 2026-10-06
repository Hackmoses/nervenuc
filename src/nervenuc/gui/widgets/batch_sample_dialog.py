"""Minimal batch sample metadata and direct-channel mapping dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from nervenuc.preparation import StackInspectionResult

from ..controllers.batch_controller import BatchSampleDraft


class BatchSampleDialog(QDialog):
    """Edit one sample's required identity and, for direct 2D, explicit channel roles."""

    def __init__(
        self,
        draft: BatchSampleDraft,
        inspection: StackInspectionResult,
        marker_names: tuple[str, ...],
        direct_2d: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Sample Metadata")
        self._draft = draft
        self._marker_names = marker_names
        self._direct_2d = direct_2d

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.sample_id_edit = QLineEdit(draft.sample_id)
        self.animal_id_edit = QLineEdit(draft.animal_id)
        self.animal_id_edit.setPlaceholderText("Required biological replicate ID")
        self.axes_edit = QLineEdit(draft.declared_axes or inspection.axes or "")
        form.addRow("Sample ID:", self.sample_id_edit)
        form.addRow("Animal ID:", self.animal_id_edit)
        form.addRow("Declared axes:", self.axes_edit)
        layout.addLayout(form)

        self.mapping_table = QTableWidget(0, 3)
        self.mapping_table.setHorizontalHeaderLabels(["Channel", "Metadata name", "Assigned role"])
        if direct_2d:
            channel_count = inspection.size_c or 0
            self.mapping_table.setRowCount(channel_count)
            inverse_markers = {index: name for name, index in draft.marker_channels.items()}
            for index in range(channel_count):
                self.mapping_table.setItem(index, 0, QTableWidgetItem(str(index)))
                hint = (
                    inspection.channel_names[index]
                    if inspection.channel_names and index < len(inspection.channel_names)
                    else ""
                )
                self.mapping_table.setItem(index, 1, QTableWidgetItem(hint))
                role = QComboBox()
                role.addItem("Ignore", None)
                role.addItem("Nuclear anchor", "__nuclear__")
                for marker_name in marker_names:
                    role.addItem(marker_name, marker_name)
                if draft.nuclear_channel == index:
                    role.setCurrentIndex(1)
                elif index in inverse_markers:
                    role.setCurrentText(inverse_markers[index])
                self.mapping_table.setCellWidget(index, 2, role)
            layout.addWidget(QLabel("Assign each configured marker to one real source channel."))
            layout.addWidget(self.mapping_table)
        else:
            self.mapping_table.setVisible(False)
            layout.addWidget(QLabel("Channel selection is frozen in the active stack/volume preset."))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        if not self.sample_id_edit.text().strip():
            QMessageBox.warning(self, "Missing Sample ID", "Enter a unique sample ID.")
            return
        if not self.animal_id_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Animal ID",
                "Enter the biological replicate animal ID. NerveNuc will not create one automatically.",
            )
            return
        if self._direct_2d:
            nuclear_count = 0
            markers: dict[str, int] = {}
            for row in range(self.mapping_table.rowCount()):
                role = self.mapping_table.cellWidget(row, 2)
                if not isinstance(role, QComboBox):
                    continue
                value = role.currentData()
                if value == "__nuclear__":
                    nuclear_count += 1
                elif isinstance(value, str):
                    if value in markers:
                        QMessageBox.warning(
                            self,
                            "Duplicate Marker Mapping",
                            f"Marker '{value}' is assigned to more than one channel.",
                        )
                        return
                    markers[value] = row
            if nuclear_count != 1:
                QMessageBox.warning(
                    self,
                    "Invalid Nuclear Mapping",
                    "Assign exactly one channel as the nuclear anchor.",
                )
                return
            missing = sorted(set(self._marker_names) - set(markers))
            if missing:
                QMessageBox.warning(
                    self,
                    "Missing Marker Mapping",
                    "Assign source channels for: " + ", ".join(missing),
                )
                return
        self.accept()

    def values(self) -> tuple[str, str, str | None, int | None, dict[str, int]]:
        nuclear: int | None = None
        markers: dict[str, int] = {}
        if self._direct_2d:
            for row in range(self.mapping_table.rowCount()):
                role = self.mapping_table.cellWidget(row, 2)
                if not isinstance(role, QComboBox):
                    continue
                value = role.currentData()
                if value == "__nuclear__":
                    nuclear = row
                elif isinstance(value, str):
                    markers[value] = row
        axes = self.axes_edit.text().strip().upper() or None
        return (
            self.sample_id_edit.text().strip(),
            self.animal_id_edit.text().strip(),
            axes,
            nuclear,
            markers,
        )
