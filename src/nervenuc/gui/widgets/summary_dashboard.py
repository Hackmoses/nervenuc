"""Results dashboard displaying animal-level metrics, group assignments, and exports."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..controllers import ExportController
from ..state import GuiSessionState


class SummaryDashboardWidget(QWidget):
    """Results dashboard presenting biological summaries and multi-format exports."""

    export_completed = Signal(str)

    def __init__(self, export_controller: ExportController, state: GuiSessionState) -> None:
        super().__init__()
        self.controller = export_controller
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Processing Heterogeneity Banner
        self.lbl_heterogeneity = QLabel("")
        self.lbl_heterogeneity.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 4px; border: 1px solid #ffeeba;")
        self.lbl_heterogeneity.setVisible(False)
        layout.addWidget(self.lbl_heterogeneity)

        # Animal Results Table
        self.group_table = QGroupBox("Animal-Level Biological Replicates")
        table_layout = QVBoxLayout(self.group_table)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Animal ID", "Assigned Group", "Metric Name", "Numerator", "Denominator", "Percentage"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)
        layout.addWidget(self.group_table)

        # Export Buttons
        btn_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export CSV (GraphPad Wide)...")
        self.btn_export_excel = QPushButton("Export Multi-Sheet Excel...")
        self.btn_export_excel.setStyleSheet("font-weight: bold;")
        self.btn_export_bundle = QPushButton("Export Reproducible Results Bundle...")

        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_export_excel)
        btn_layout.addWidget(self.btn_export_bundle)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Signals
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_excel.clicked.connect(self._on_export_excel)
        self.btn_export_bundle.clicked.connect(self._on_export_bundle)
        self.table.cellChanged.connect(self._on_cell_changed)

    def refresh_results(self) -> None:
        """Aggregate study session and refresh table."""
        try:
            df = self.controller.get_animal_metrics()
        except Exception:
            self.table.setRowCount(0)
            self.btn_export_csv.setEnabled(False)
            self.btn_export_excel.setEnabled(False)
            self.btn_export_bundle.setEnabled(self.controller.can_export_reproducible())
            return

        # Check heterogeneity
        het = self.state.study_session.processing_heterogeneity
        if het and len(het.pipeline_fingerprints) > 1:
            self.lbl_heterogeneity.setText(
                f"ℹ Processing Heterogeneity: Samples share '{het.measurement_semantics}' semantics "
                f"across {len(het.pipeline_fingerprints)} distinct pipeline configurations."
            )
            self.lbl_heterogeneity.setVisible(True)
        else:
            self.lbl_heterogeneity.setVisible(False)

        self.table.blockSignals(True)
        self.table.setRowCount(len(df))

        for r, (_, row) in enumerate(df.iterrows()):
            a_id = str(row["animal_id"])
            m_name = str(row["metric_name"])
            num = str(int(row["numerator_count"]))
            denom = str(int(row["denominator_count"]))
            pct_val = row["percentage"]

            pct_str = f"{pct_val:.2f}%" if pd.notna(pct_val) else "N/A (0/0)"
            group_name = self.controller.group_assignments.get(a_id, "")

            item_a = QTableWidgetItem(a_id)
            item_a.setFlags(item_a.flags() & ~Qt.ItemFlag.ItemIsEditable)

            item_g = QTableWidgetItem(group_name)  # Editable group name

            item_m = QTableWidgetItem(m_name)
            item_m.setFlags(item_m.flags() & ~Qt.ItemFlag.ItemIsEditable)

            item_num = QTableWidgetItem(num)
            item_num.setFlags(item_num.flags() & ~Qt.ItemFlag.ItemIsEditable)

            item_denom = QTableWidgetItem(denom)
            item_denom.setFlags(item_denom.flags() & ~Qt.ItemFlag.ItemIsEditable)

            item_pct = QTableWidgetItem(pct_str)
            item_pct.setFlags(item_pct.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(r, 0, item_a)
            self.table.setItem(r, 1, item_g)
            self.table.setItem(r, 2, item_m)
            self.table.setItem(r, 3, item_num)
            self.table.setItem(r, 4, item_denom)
            self.table.setItem(r, 5, item_pct)

        self.table.blockSignals(False)
        self.btn_export_csv.setEnabled(True)
        self.btn_export_excel.setEnabled(self.controller.groups_complete())
        self.btn_export_bundle.setEnabled(self.controller.can_export_reproducible())

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col == 1:
            a_item = self.table.item(row, 0)
            g_item = self.table.item(row, 1)
            if a_item and g_item:
                self.controller.set_animal_group(a_item.text(), g_item.text())
                self.btn_export_excel.setEnabled(self.controller.groups_complete())

    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV Table", "nervenuc_results.csv", "CSV Files (*.csv)")
        if path:
            try:
                out = self.controller.export_csv(path, overwrite=True)
                QMessageBox.information(self, "Export Succeeded", f"Successfully exported to:\n{out}")
                self.export_completed.emit(str(out))
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _on_export_bundle(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export Reproducible Results Bundle")
        if directory:
            try:
                written = self.controller.export_reproducible_bundle(directory)
                QMessageBox.information(
                    self,
                    "Reproducible Export Succeeded",
                    f"Wrote {len(written)} reproducibility files to:\n{directory}",
                )
                self.export_completed.emit(str(directory))
            except Exception as exc:
                QMessageBox.critical(self, "Reproducible Export Failed", str(exc))

    def _on_export_excel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel Workbook", "nervenuc_results.xlsx", "Excel Files (*.xlsx)")
        if path:
            try:
                out = self.controller.export_excel(path, overwrite=True)
                QMessageBox.information(self, "Export Succeeded", f"Successfully exported to:\n{out}")
                self.export_completed.emit(str(out))
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))
