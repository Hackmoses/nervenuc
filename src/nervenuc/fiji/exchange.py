"""Temporary file exchange lifecycle and 2D/3D viewing interoperability for Fiji."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence
import uuid

import numpy as np
import tifffile

from nervenuc.preparation import VoxelSpacing

from .conversion import SUPPORTED_CONVERSION_AXES, validate_array_for_conversion
from .errors import FijiConversionError, FijiExecutionError, FijiUnavailableError
from .launcher import FijiLauncher, discover_fiji
from .models import FijiProvenance

logger = logging.getLogger(__name__)


class FijiExchangeSession:
    """Manages the lifecycle of temporary scratch TIFF artifacts and external Fiji processes.

    Invariants:
        - Temporary files reside exclusively in an isolated, session-owned temporary directory.
        - Exchange TIFFs never become authoritative scientific source identities.
        - External Fiji viewing does not mutate NerveNuc analysis state, source IDs, or masks.
        - Scratch files are safely retained while spawned Fiji processes are active.
    """

    def __init__(self, temp_base_dir: Path | str | None = None) -> None:
        if temp_base_dir is not None:
            base = Path(temp_base_dir).resolve()
            base.mkdir(parents=True, exist_ok=True)
            self._temp_dir = Path(tempfile.mkdtemp(prefix="nervenuc_fiji_", dir=str(base)))
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="nervenuc_fiji_"))

        self._tracked_files: list[Path] = []
        self._tracked_processes: list[subprocess.Popen] = []
        self._is_cleaned = False

    @property
    def temp_directory(self) -> Path:
        """Return the isolated temporary directory for this session."""
        return self._temp_dir

    @property
    def tracked_files(self) -> Sequence[Path]:
        """Return tuple of scratch files generated in this session."""
        return tuple(self._tracked_files)

    @property
    def tracked_processes(self) -> Sequence[subprocess.Popen]:
        """Return tuple of spawned Fiji process handles."""
        return tuple(self._tracked_processes)

    def create_temp_tiff_path(self, filename: str | None = None) -> Path:
        """Allocate a unique temporary TIFF path inside the session directory."""
        fname = filename or f"exchange_{uuid.uuid4().hex[:8]}.tif"
        if not isinstance(fname, str) or not fname.strip():
            raise FijiConversionError("Exchange filename must be a non-empty basename.")
        supplied = Path(fname)
        if supplied.is_absolute() or supplied.name != fname or any(
            part in {".", ".."} for part in supplied.parts
        ):
            raise FijiConversionError(
                "Exchange filename must be a basename inside the session directory."
            )
        p = (self._temp_dir / fname).resolve()
        try:
            p.relative_to(self._temp_dir.resolve())
        except ValueError as err:
            raise FijiConversionError(
                "Exchange filename resolves outside the session directory."
            ) from err
        return p

    def write_exchange_tiff(
        self,
        array: np.ndarray,
        axes: str,
        spacing: VoxelSpacing | None = None,
        is_labels: bool = False,
        filename: str | None = None,
    ) -> Path:
        """Write a NumPy array to an exchange TIFF artifact preserving dtype, raw values, and calibration.

        Args:
            array: Input scientific or label array.
            axes: Declared axis layout ('YX', 'ZYX', 'CYX', 'CZYX', 'TCZYX').
            spacing: Optional physical VoxelSpacing (in micrometers).
            is_labels: If True, indicates integer segmentation label volume.
            filename: Optional custom filename.

        Returns:
            Absolute Path to written exchange TIFF.

        Raises:
            FijiConversionError: If array dimensions, axes, or writing fails.
        """
        if not isinstance(array, np.ndarray):
            raise FijiConversionError(f"Expected numpy.ndarray, got {type(array).__name__}.")

        norm_axes = axes.strip().upper() if isinstance(axes, str) else ""
        if norm_axes not in SUPPORTED_CONVERSION_AXES:
            raise FijiConversionError(
                f"Unsupported axis layout '{axes}'. Supported layouts: {SUPPORTED_CONVERSION_AXES}."
            )

        if array.ndim != len(norm_axes):
            raise FijiConversionError(
                f"Array ndim ({array.ndim}) does not match axis length ({len(norm_axes)}) for '{norm_axes}'."
            )

        target_path = self.create_temp_tiff_path(filename)
        if target_path.exists():
            raise FijiConversionError(
                f"Refusing to overwrite existing exchange TIFF '{target_path}'."
            )

        metadata: dict[str, str | int | float | bool] = {"axes": norm_axes}
        # ImageJ 1.x header format natively supports uint8 ('B'), uint16 ('H'), int16 ('h'), float32 ('f')
        use_imagej_format = array.dtype.char in "BHhf"
        imwrite_kwargs: dict[str, object] = {
            "imagej": use_imagej_format,
            "photometric": "minisblack",
        }


        # Apply physical calibration if provided
        if spacing is not None:
            metadata["unit"] = "um"
            if spacing.x > 0 and spacing.y > 0:
                imwrite_kwargs["resolution"] = (1.0 / spacing.x, 1.0 / spacing.y)
            if "Z" in norm_axes and spacing.z > 0 and use_imagej_format:
                metadata["spacing"] = float(spacing.z)


        try:
            # Write raw array with exact values and dtype
            tifffile.imwrite(
                str(target_path),
                array,
                metadata=metadata,
                **imwrite_kwargs,
            )
        except Exception as err:
            raise FijiConversionError(f"Failed to write exchange TIFF to '{target_path}': {err}") from err

        self._tracked_files.append(target_path)
        return target_path

    def open_in_fiji(
        self,
        source: Path | str | np.ndarray,
        axes: str | None = None,
        spacing: VoxelSpacing | None = None,
        is_labels: bool = False,
        launcher: FijiLauncher | None = None,
    ) -> subprocess.Popen:
        """Open an image or volume in an external Fiji process with safe fast-path / scratch routing.

        Policy:
            - File-backed existing TIFF paths are opened directly (read-only fast path).
            - In-memory arrays, derived projected images, or label volumes are written to a scratch TIFF first.

        Args:
            source: Path to existing TIFF, or NumPy array to export.
            axes: Axis string (required if source is array, e.g. 'YX', 'ZYX', 'CZYX').
            spacing: Optional physical calibration.
            is_labels: Whether the array represents integer label IDs.
            launcher: Optional FijiLauncher instance (defaults to auto-discovered Fiji).

        Returns:
            subprocess.Popen handle.

        Raises:
            FijiUnavailableError: If Fiji executable is missing.
            FijiExecutionError: If launch fails.
        """
        active_launcher = launcher
        if active_launcher is None:
            info = discover_fiji()
            if info is None:
                raise FijiUnavailableError("Fiji installation not found. Configure Fiji path or set FIJI_HOME.")
            active_launcher = FijiLauncher(info)

        if isinstance(source, (str, Path)):
            src_path = Path(source).resolve()
            if not src_path.is_file():
                raise FijiExecutionError(f"Source file to open in Fiji does not exist: '{source}'.")
            target_to_open = src_path
        elif isinstance(source, np.ndarray):
            eff_axes = axes
            if eff_axes is None:
                if source.ndim == 2:
                    eff_axes = "YX"
                elif source.ndim == 3:
                    eff_axes = "ZYX"
                else:
                    raise FijiConversionError("Explicit 'axes' parameter required when opening multi-dimensional arrays.")
            target_to_open = self.write_exchange_tiff(
                source,
                axes=eff_axes,
                spacing=spacing,
                is_labels=is_labels,
            )
        else:
            raise FijiConversionError(f"Unsupported source type for Fiji viewing: {type(source).__name__}.")

        proc = active_launcher.open_file(target_to_open)
        self._tracked_processes.append(proc)
        return proc

    def has_running_processes(self) -> bool:
        """Return True if any spawned Fiji process is currently active."""
        return any(p.poll() is None for p in self._tracked_processes)

    def cleanup(self, force: bool = False) -> None:
        """Safely remove temporary exchange files and directory.

        Args:
            force: If True, deletes files immediately regardless of running Fiji processes.
                   If False, skips deletion if any spawned Fiji process is still running.
        """
        if self._is_cleaned:
            return

        if not force and self.has_running_processes():
            logger.debug("Fiji processes still active; deferring cleanup for session %s.", self._temp_dir)
            return

        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._is_cleaned = True
        except Exception as err:
            logger.warning("Failed to clean up temporary Fiji exchange directory '%s': %s", self._temp_dir, err)

    def __enter__(self) -> FijiExchangeSession:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.cleanup(force=False)
