"""Dedicated QThread worker for executing batch analysis workflows."""

from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QThread, Signal

from nervenuc.batch import BatchResult, BatchRunner


class BatchWorker(QThread):
    """QThread worker executing BatchRunner with safe stop semantics."""

    progress_updated = Signal(str, str, int, int, int, int)
    # sample_id, status, index, total, completed_count, failed_count
    batch_completed = Signal(BatchResult)
    batch_failed = Signal(tuple)  # (exception, formatted_traceback_str)

    def __init__(self, runner: BatchRunner) -> None:
        super().__init__()
        self.runner = runner
        self._stop_requested = False

    def request_stop(self) -> None:
        """Signal safe stop after current sample."""
        self._stop_requested = True

    def run(self) -> None:
        """Execute the batch run loop safely."""
        try:
            res = self.runner.run(
                lambda sid, status, index, total, completed, failed: self.progress_updated.emit(
                    sid, status, index, total, completed, failed
                )
            )
            self.batch_completed.emit(res)
        except Exception as exc:
            tb = traceback.format_exc()
            self.batch_failed.emit((exc, tb))
