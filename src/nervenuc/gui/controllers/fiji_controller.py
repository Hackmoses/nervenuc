"""Controller managing Fiji / ImageJ external process launching, ROI exchange, and preferences."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from PySide6.QtCore import QSettings, QThreadPool

from ..errors import NerveNucGuiError
from ..state import GuiSessionState
from ..workers import FunctionWorker

logger = logging.getLogger(__name__)


def is_roifile_available() -> bool:
    """Return True if roifile package is installed for ImageJ ROI import/export."""
    try:
        import roifile  # noqa: F401
        return True
    except ImportError:
        return False


def is_pyimagej_available() -> bool:
    """Return True if pyimagej is installed for programmatic ImageJ bridge."""
    try:
        import imagej  # noqa: F401
        return True
    except ImportError:
        return False


class FijiController:
    """Coordinates Fiji external process launching, ROI import/export, and user preferences."""

    def __init__(self, state: GuiSessionState, threadpool: QThreadPool | None = None) -> None:
        self.state = state
        self.threadpool = threadpool or QThreadPool.globalInstance()
        self._active_exchange_sessions: list[Any] = []
        self._active_workers: list[FunctionWorker] = []
        self._is_qc_worker_running = False

    def _start_worker(
        self,
        func: Callable[..., Any],
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> FunctionWorker:
        """Start a background worker keeping a strong reference until execution completes."""
        worker = FunctionWorker(func)
        self._active_workers.append(worker)

        def _cleanup(*_args: Any) -> None:
            if worker in self._active_workers:
                self._active_workers.remove(worker)

        if on_started:
            worker.signals.started.connect(on_started)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if on_error:
            worker.signals.error.connect(on_error)
        worker.signals.finished.connect(_cleanup)
        worker.signals.error.connect(_cleanup)

        self.threadpool.start(worker)
        return worker

    @property
    def is_qc_worker_running(self) -> bool:
        """True if an asynchronous ROI import or split operation is currently executing."""
        return self._is_qc_worker_running

    # --- Preference Management via QSettings ---

    def get_configured_fiji_path(self) -> Path | None:
        """Read and validate saved Fiji executable path from QSettings."""
        settings = QSettings("NerveNuc", "NerveNuc")
        val = settings.value("fiji/executable_path", None)
        if val:
            p = Path(str(val))
            if p.exists():
                return p
            # Stored path no longer exists on disk
            logger.warning(f"Saved Fiji path does not exist on disk: '{p}'.")
            settings.remove("fiji/executable_path")
        return None

    def set_configured_fiji_path(self, path: Path | str | None) -> None:
        """Persist or clear Fiji executable path in QSettings."""
        settings = QSettings("NerveNuc", "NerveNuc")
        if path is None or str(path).strip() == "":
            settings.remove("fiji/executable_path")
        else:
            p = Path(path).resolve()
            settings.setValue("fiji/executable_path", str(p))

    def discover_executable(self) -> Any | None:
        """Discover Fiji executable using configured path preference or system detection."""
        try:
            from nervenuc.fiji.launcher import discover_fiji
            cfg_path = self.get_configured_fiji_path()
            return discover_fiji(explicit_path=cfg_path)
        except Exception as err:
            logger.debug(f"Fiji discovery error: {err}")
            return None

    def can_launch_external_fiji(self) -> bool:
        """Return True if a valid Fiji installation is available for external launching."""
        return self.discover_executable() is not None

    def can_exchange_rois(self) -> bool:
        """Return True if roifile is available for ROI import and export."""
        return is_roifile_available()

    # --- Session & Process Lifecycle ---

    def cleanup_sessions(self, force: bool = False) -> None:
        """Clean temporary exchange files for sessions whose external Fiji processes have exited."""
        active: list[Any] = []
        for sess in self._active_exchange_sessions:
            try:
                if force:
                    sess.cleanup(force=True)
                else:
                    if sess.cleanup():
                        continue  # All files cleaned up
                    active.append(sess)  # Process still running, retain
            except Exception as err:
                logger.warning(f"Error during Fiji exchange session cleanup: {err}")
                active.append(sess)
        self._active_exchange_sessions = active

    # --- Error Translation ---

    def translate_fiji_error(self, exc: Exception) -> tuple[str, str]:
        """Translate Fiji interoperability errors into clear researcher-facing messages."""
        from nervenuc.fiji.errors import (
            FijiConversionError,
            FijiExecutionError,
            FijiUnavailableError,
            InvalidRoiGeometryError,
            JavaUnavailableError,
            UnsupportedRoiError,
        )
        from nervenuc.qc import InvalidReviewMaskError, UnknownReviewObjectError

        if isinstance(exc, FijiUnavailableError):
            return (
                "Fiji Executable Not Found",
                "Fiji was not found on your system or the configured path is invalid.\n\n"
                "Please configure your Fiji path under Tools -> Fiji / ImageJ -> Configure Fiji...",
            )
        if isinstance(exc, JavaUnavailableError):
            return (
                "Java Runtime Unavailable",
                f"PyImageJ could not initialize a compatible Java Runtime Environment: {exc}",
            )
        if isinstance(exc, UnsupportedRoiError):
            return (
                "Unsupported ROI Type",
                f"The ImageJ ROI file contains an unsupported ROI type or structure:\n{exc}\n\n"
                "NerveNuc supports Polygon, Freehand, Traced, Rectangle, and Oval ROIs.",
            )
        if isinstance(exc, InvalidRoiGeometryError):
            return (
                "Invalid ROI Geometry",
                f"ROI geometry validation failed:\n{exc}\n\n"
                "Ensure coordinates are within image boundaries and contain valid non-empty areas.",
            )
        if isinstance(exc, InvalidReviewMaskError):
            return (
                "QC Review Mask Error",
                f"The imported ROI mask violated NerveNuc QC constraints:\n{exc}",
            )
        if isinstance(exc, UnknownReviewObjectError):
            return (
                "Unknown Review Object",
                f"The target review object was not found in the active review session:\n{exc}",
            )
        if isinstance(exc, FijiConversionError):
            return (
                "Fiji Conversion Error",
                f"Failed to prepare array for ImageJ exchange:\n{exc}",
            )
        if isinstance(exc, FijiExecutionError):
            return (
                "Fiji Launch Failed",
                f"Failed to launch external Fiji process:\n{exc}",
            )
        return ("Fiji Operation Failed", str(exc))

    # --- Asynchronous Actions ---

    def open_current_image(
        self,
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> None:
        """Open the active 2D image in external Fiji."""
        from nervenuc.fiji.exchange import FijiExchangeSession
        from nervenuc.fiji.launcher import FijiLauncher

        exe_info = self.discover_executable()
        if exe_info is None:
            from nervenuc.fiji.errors import FijiUnavailableError
            raise FijiUnavailableError("No Fiji executable found. Configure Fiji installation path.")

        # Determine target source to open
        mode = self.state.draft.mode
        source_to_open: Any = None

        if mode == "direct_2d" and self.state.loaded_path is not None:
            # Fast-path: open existing file directly
            source_to_open = self.state.loaded_path
        elif self.state.session_2d is not None and self.state.session_2d.nuclear_channel is not None:
            source_to_open = self.state.session_2d.nuclear_channel
        elif self.state.loaded_raw_array is not None:
            source_to_open = self.state.loaded_raw_array
        else:
            raise NerveNucGuiError("No active 2D image is available to open in Fiji.")

        launcher = FijiLauncher(exe_info)
        session = FijiExchangeSession()
        self._active_exchange_sessions.append(session)

        def _task() -> Any:
            return session.open_in_fiji(source=source_to_open, launcher=launcher)

        self._start_worker(_task, on_started=on_started, on_finished=on_finished, on_error=on_error)

    def open_current_volume(
        self,
        kind: Literal["raw_stack", "prepared_volume", "segmentation_labels"],
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> None:
        """Open a 3D Z-stack, prepared volume, or 3D label volume in external Fiji."""
        from nervenuc.fiji.exchange import FijiExchangeSession
        from nervenuc.fiji.launcher import FijiLauncher

        exe_info = self.discover_executable()
        if exe_info is None:
            from nervenuc.fiji.errors import FijiUnavailableError
            raise FijiUnavailableError("No Fiji executable found. Configure Fiji installation path.")

        source_to_open: Any = None
        axes_str = "ZYX"
        is_labels = False

        if kind == "raw_stack":
            if self.state.loaded_path is None:
                raise NerveNucGuiError("No source stack file is loaded.")
            source_to_open = self.state.loaded_path
        elif kind == "prepared_volume":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.prepared_volume is None:
                raise NerveNucGuiError("No prepared 3D volume is available in the current session.")
            source_to_open = sess_3d.prepared_volume.nuclear_volume
            axes_str = "ZYX"
        elif kind == "segmentation_labels":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.segmentation_result is None:
                raise NerveNucGuiError("No 3D segmentation labels available.")
            source_to_open = sess_3d.segmentation_result.labels
            axes_str = "ZYX"
            is_labels = True
        else:
            raise NerveNucGuiError(f"Unsupported volume view kind: {kind}")

        launcher = FijiLauncher(exe_info)
        session = FijiExchangeSession()
        self._active_exchange_sessions.append(session)

        def _task() -> Any:
            return session.open_in_fiji(
                source=source_to_open,
                axes=axes_str,
                is_labels=is_labels,
                launcher=launcher,
            )

        self._start_worker(_task, on_started=on_started, on_finished=on_finished, on_error=on_error)

    def export_rois(
        self,
        output_path: Path | str,
        overwrite: bool = False,
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> None:
        """Export active 2D QC segmentation objects to a Fiji ImageJ ROI set ZIP file."""
        from nervenuc.fiji.qc import export_review_rois

        if not self.can_exchange_rois():
            raise NerveNucGuiError("ROI export requires the 'roifile' package. Install with: pip install 'nervenuc[fiji]'")

        sess_2d = self.state.session_2d
        if sess_2d is None:
            raise NerveNucGuiError("No active 2D analysis session.")

        source_qc: Any = None
        if sess_2d.reviewed_result is not None:
            source_qc = sess_2d.reviewed_result
        elif sess_2d.review_session is not None:
            source_qc = sess_2d.review_session
        else:
            raise NerveNucGuiError("No active 2D review session or reviewed result to export.")

        def _task() -> Path:
            return export_review_rois(source_qc, output_path, overwrite=overwrite)

        self._start_worker(_task, on_started=on_started, on_finished=on_finished, on_error=on_error)

    def import_rois(
        self,
        roi_path: Path | str,
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> None:
        """Import Fiji ROIs as manual QC objects into the active 2D ReviewSession."""
        from nervenuc.fiji.qc import import_rois_as_manual

        if not self.can_exchange_rois():
            raise NerveNucGuiError("ROI import requires the 'roifile' package. Install with: pip install 'nervenuc[fiji]'")

        sess_2d = self.state.session_2d
        if sess_2d is None or sess_2d.review_session is None:
            raise NerveNucGuiError("Cannot import ROIs: No active 2D review session.")

        if self._is_qc_worker_running:
            raise NerveNucGuiError("Another QC operation is currently running.")

        review_sess = sess_2d.review_session
        self._is_qc_worker_running = True

        def _task() -> Any:
            return import_rois_as_manual(review_sess, roi_path)

        def _wrapped_finished(res: Any) -> None:
            self._is_qc_worker_running = False
            # Invalidate downstream scientific results via state
            self.state.results_stale = True
            self.state.notify()
            if on_finished:
                on_finished(res)

        def _wrapped_error(err_tuple: tuple[Exception, str]) -> None:
            self._is_qc_worker_running = False
            if on_error:
                on_error(err_tuple)

        self._start_worker(
            _task,
            on_started=on_started,
            on_finished=_wrapped_finished,
            on_error=_wrapped_error,
        )

    def replace_rois(
        self,
        target_review_id: str,
        roi_path: Path | str,
        note: str | None = None,
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[tuple[Exception, str]], None] | None = None,
    ) -> None:
        """Replace an active nucleus in ReviewSession with Fiji ROIs via ReviewSession.split()."""
        from nervenuc.fiji.qc import replace_with_fiji_rois

        if not self.can_exchange_rois():
            raise NerveNucGuiError("ROI replacement requires the 'roifile' package. Install with: pip install 'nervenuc[fiji]'")

        sess_2d = self.state.session_2d
        if sess_2d is None or sess_2d.review_session is None:
            raise NerveNucGuiError("Cannot replace object: No active 2D review session.")

        if self._is_qc_worker_running:
            raise NerveNucGuiError("Another QC operation is currently running.")

        review_sess = sess_2d.review_session
        self._is_qc_worker_running = True

        def _task() -> Any:
            try:
                return replace_with_fiji_rois(review_sess, target_review_id, roi_path, note=note)
            except Exception as e:
                logger.exception("Error in replace_rois task")
                raise

        def _wrapped_finished(res: Any) -> None:
            self._is_qc_worker_running = False
            # Invalidate downstream scientific results via state
            self.state.results_stale = True
            self.state.notify()
            if on_finished:
                on_finished(res)

        def _wrapped_error(err_tuple: tuple[Exception, str]) -> None:
            self._is_qc_worker_running = False
            if on_error:
                on_error(err_tuple)

        self._start_worker(
            _task,
            on_started=on_started,
            on_finished=_wrapped_finished,
            on_error=_wrapped_error,
        )
