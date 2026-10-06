"""Batch manifest, preflight, execution progress, and completion widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nervenuc.batch import BatchResult
from nervenuc.preparation import inspect_stack

from ..controllers import BatchController
from ..state import GuiSessionState
from ..workers import BatchWorker
from .batch_sample_dialog import BatchSampleDialog


class BatchManifestWidget(QWidget):
    """Functional batch workspace using the authoritative BatchRunner."""

    manifest_changed = Signal()
    batch_finished = Signal(object, bool)  # BatchResult, complete study published
    batch_running_changed = Signal(bool)

    def __init__(self, controller: BatchController, state: GuiSessionState) -> None:
        super().__init__()
        self.controller = controller
        self.state = state
        self.worker: BatchWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.policy_label = QLabel(
            "Batch QC policy: segmentation candidates will be automatically accepted. "
            "Manual per-object QC is not performed during this batch run."
        )
        self.policy_label.setWordWrap(True)
        layout.addWidget(self.policy_label)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Sample ID", "Animal ID", "Source", "Channel mapping", "Status", "Details"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        edit_buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add File(s)...")
        self.btn_edit = QPushButton("Edit Sample...")
        self.btn_remove = QPushButton("Remove Selected")
        edit_buttons.addWidget(self.btn_add)
        edit_buttons.addWidget(self.btn_edit)
        edit_buttons.addWidget(self.btn_remove)
        edit_buttons.addStretch()
        layout.addLayout(edit_buttons)

        run_buttons = QHBoxLayout()
        self.btn_validate = QPushButton("Validate Batch")
        self.btn_run = QPushButton("Run Batch")
        self.btn_run.setEnabled(False)
        run_buttons.addWidget(self.btn_validate)
        run_buttons.addWidget(self.btn_run)
        run_buttons.addStretch()
        layout.addLayout(run_buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Not running")
        layout.addWidget(self.progress)
        self.summary_label = QLabel("Add sources, enter sample and animal IDs, map channels, then validate.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_edit.clicked.connect(self._on_edit_selected)
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_validate.clicked.connect(self._on_validate)
        self.btn_run.clicked.connect(self._on_run)
        self.table.itemDoubleClicked.connect(lambda *_: self._on_edit_selected())

    def _current_marker_names(self) -> tuple[str, ...]:
        try:
            return tuple(marker.name for marker in self.controller._build_current_preset().config.markers)
        except Exception:
            return tuple(self.state.draft.marker_channels)

    def refresh_table(self) -> None:
        drafts = self.controller.sample_drafts
        self.table.setRowCount(len(drafts))
        for row, draft in enumerate(drafts):
            mapping = "Preset mapping"
            if self.state.draft.mode == "direct_2d":
                roles = []
                if draft.nuclear_channel is not None:
                    roles.append(f"Nuclear=C{draft.nuclear_channel}")
                roles.extend(f"{name}=C{index}" for name, index in sorted(draft.marker_channels.items()))
                mapping = ", ".join(roles) if roles else "Not configured"
            values = [
                draft.sample_id,
                draft.animal_id or "Required",
                str(draft.source_path),
                mapping,
                "Needs validation",
                "",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1 and not draft.animal_id:
                    item.setToolTip("Animal ID is required and will not be generated automatically.")
                self.table.setItem(row, column, item)

    def _open_editor(self, index: int) -> None:
        if index < 0 or index >= len(self.controller.sample_drafts):
            return
        draft = self.controller.sample_drafts[index]
        try:
            inspection = inspect_stack(draft.source_path, declared_axes=draft.declared_axes)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Inspect Batch Source", str(exc))
            return
        dialog = BatchSampleDialog(
            draft,
            inspection,
            self._current_marker_names(),
            self.state.draft.mode == "direct_2d",
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            sample_id, animal_id, axes, nuclear, markers = dialog.values()
            self.controller.update_sample_draft(
                index,
                sample_id=sample_id,
                animal_id=animal_id,
                declared_axes=axes,
                nuclear_channel=nuclear,
                marker_channels=markers,
            )
            self.refresh_table()
            self.btn_run.setEnabled(False)
            self.manifest_changed.emit()

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Microscopy Images/Stacks",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        for path_text in paths:
            try:
                self.controller.add_sample_draft(Path(path_text))
                self.refresh_table()
                self._open_editor(len(self.controller.sample_drafts) - 1)
            except Exception as exc:
                QMessageBox.critical(self, "Cannot Add Batch Sample", str(exc))
        self.refresh_table()
        self.btn_run.setEnabled(False)
        self.manifest_changed.emit()

    def _on_edit_selected(self) -> None:
        selected = self.table.selectedItems()
        if selected:
            self._open_editor(selected[0].row())

    def _on_remove_selected(self) -> None:
        selected = self.table.selectedItems()
        if selected:
            self.controller.remove_sample_draft(selected[0].row())
            self.refresh_table()
            self.btn_run.setEnabled(False)
            self.manifest_changed.emit()

    def _on_validate(self) -> None:
        report = self.controller.preflight_batch()
        row_by_id = {row.sample_id: row for row in report.rows}
        for index, draft in enumerate(self.controller.sample_drafts):
            row = row_by_id.get(draft.sample_id)
            if row is None:
                continue
            self.table.item(index, 4).setText("Ready" if row.valid else "Invalid")
            details = row.message
            if row.warning:
                details = f"{details}; {row.warning}"
            self.table.item(index, 5).setText(details)
        self.btn_run.setEnabled(report.valid and not self.state.batch_running)
        if report.valid:
            self.summary_label.setText(
                f"Validated {len(report.rows)} samples with frozen preset '{report.preset_name}'. Ready to run."
            )
        else:
            message = report.global_error or "Correct the invalid sample rows and validate again."
            self.summary_label.setText(f"Batch validation failed: {message}")

    def _on_run(self) -> None:
        answer = QMessageBox.question(
            self,
            "Run Auto-Accept Batch",
            "This batch will automatically accept segmentation candidates. Manual per-object QC will not be "
            "performed during the run. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.start_preflighted_run()

    def start_preflighted_run(self) -> None:
        """Start the already validated batch; separated for deterministic GUI testing."""
        try:
            runner = self.controller.create_preflighted_runner()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Run Batch", str(exc))
            self.btn_run.setEnabled(False)
            return
        self.state.batch_running = True
        self.batch_running_changed.emit(True)
        self.state.batch_result = None
        self.state.results_stale = True
        self.state.configuration_differs_from_run = True
        self._set_editing_enabled(False)
        total = len(runner.manifest.samples)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setFormat(f"Queued 0/{total}")
        self.summary_label.setText(
            "Running with auto-accept QC. Manual per-object review is not being performed."
        )
        self.worker = BatchWorker(runner)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.batch_failed.connect(self._on_batch_failed)
        self.worker.finished.connect(self._release_worker)
        self.worker.start()

    def _set_editing_enabled(self, enabled: bool) -> None:
        for control in (self.btn_add, self.btn_edit, self.btn_remove, self.btn_validate):
            control.setEnabled(enabled)
        self.btn_run.setEnabled(False)

    def _on_progress(
        self,
        sample_id: str,
        status: str,
        index: int,
        total: int,
        completed: int,
        failed: int,
    ) -> None:
        for row, draft in enumerate(self.controller.sample_drafts):
            if draft.sample_id == sample_id:
                self.table.item(row, 4).setText(status.replace("_", " ").title())
                break
        self.progress.setValue(min(index, total))
        self.progress.setFormat(
            f"Sample {index}/{total} · {completed} completed · {failed} failed"
        )

    def _on_batch_completed(self, result: BatchResult) -> None:
        complete = self.controller.commit_batch_result(result)
        self.state.batch_running = False
        self.batch_running_changed.emit(False)
        self._set_editing_enabled(True)
        failed_rows = result.sample_status[result.sample_status["status"] == "failed"]
        for row_index, draft in enumerate(self.controller.sample_drafts):
            status_row = result.sample_status[result.sample_status["sample_id"] == draft.sample_id]
            if len(status_row) == 1:
                record = status_row.iloc[0]
                self.table.item(row_index, 4).setText(str(record["status"]).replace("_", " ").title())
                self.table.item(row_index, 5).setText(str(record["error_message"] or ""))
        if complete:
            self.summary_label.setText(
                f"Batch complete: {len(result.completed_samples)} completed, 0 failed. Study and reproducible export are available."
            )
        else:
            failed_ids = ", ".join(str(value) for value in failed_rows["sample_id"])
            self.summary_label.setText(
                f"Batch incomplete: {len(result.completed_samples)} completed, {len(failed_rows)} failed "
                f"({failed_ids}). Study publication and export are disabled."
            )
        self.batch_finished.emit(result, complete)

    def _on_batch_failed(self, error_info: tuple[Exception, str]) -> None:
        self.state.batch_running = False
        self.batch_running_changed.emit(False)
        self.state.results_stale = True
        self.state.configuration_differs_from_run = True
        self._set_editing_enabled(True)
        exc, _traceback_text = error_info
        self.summary_label.setText(f"Batch execution failed before a valid result was produced: {exc}")
        QMessageBox.critical(self, "Batch Execution Failed", str(exc))

    def _release_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
