"""Base worker infrastructure for background execution without freezing Qt GUI."""

from __future__ import annotations

import sys
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """Standardized Qt signals for asynchronous worker feedback."""

    started = Signal()
    progress = Signal(str)  # Coarse status message
    finished = Signal(object)  # Completed result payload
    error = Signal(tuple)  # (exception, formatted_traceback_str)


class FunctionWorker(QRunnable):
    """General-purpose QRunnable worker executing a callable on QThreadPool."""

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Execute the task and dispatch results or errors via signals."""
        self.signals.started.emit()
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.signals.error.emit((exc, tb))
