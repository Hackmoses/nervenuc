"""Data models and configuration structures for Fiji/ImageJ interoperability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FijiExecutableInfo:
    """Information about a discovered and validated Fiji/ImageJ installation on disk.

    Attributes:
        executable_path: Absolute validated path to the Fiji/ImageJ executable binary or app.
        discovery_source: How this installation was discovered ('explicit', 'environment', 'standard_path').
        is_fiji: True if identified as a full Fiji distribution (contains plugins/macros), False for vanilla ImageJ.
        app_directory: Root application directory containing Fiji.app / ImageJ.
    """

    executable_path: Path
    discovery_source: str
    is_fiji: bool
    app_directory: Path


@dataclass(frozen=True)
class FijiConfig:
    """Configuration settings for Fiji discovery and bridge behavior.

    Attributes:
        explicit_fiji_path: Optional user-configured path to Fiji executable or Fiji.app.
        search_environment: Whether to inspect FIJI_HOME, IMAGEJ_HOME, or FIJI_PATH environment variables.
        search_standard_paths: Whether to check conservative known OS installation directories.
        prefer_fiji_distribution: Whether to prioritize full Fiji over minimal ImageJ installations.
    """

    explicit_fiji_path: Path | None = None
    search_environment: bool = True
    search_standard_paths: bool = True
    prefer_fiji_distribution: bool = True


@dataclass(frozen=True)
class FijiProvenance:
    """Provenance record of an interoperability action with Fiji/ImageJ.

    Attributes:
        action_type: Type of action ('view_image', 'view_volume', 'export_rois', 'import_rois').
        timestamp_utc: ISO-8601 UTC timestamp of execution.
        source_id: NerveNuc authoritative execution source identity.
        fiji_executable: Path to Fiji executable if external process was used, or 'pyimagej' if in-process.
        details: Optional key-value dictionary describing the operation.
    """

    action_type: str
    timestamp_utc: str
    source_id: str
    fiji_executable: str
    details: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        action_type: str,
        source_id: str,
        fiji_executable: str,
        details: dict[str, str] | None = None,
    ) -> FijiProvenance:
        """Helper to create a timestamped FijiProvenance record."""
        now_str = datetime.now(timezone.utc).isoformat()
        detail_tuples = tuple(sorted(details.items())) if details else ()
        return cls(
            action_type=action_type,
            timestamp_utc=now_str,
            source_id=source_id,
            fiji_executable=fiji_executable,
            details=detail_tuples,
        )


@dataclass(frozen=True)
class AxisConversion:
    """Metadata describing bidirectional axis permutation between NerveNuc and ImageJ.

    Attributes:
        source_axes: Original NerveNuc axis order (e.g. 'CZYX', 'ZYX', 'CYX', 'YX', 'TCZYX').
        imagej_axes: Target ImageJ axis order (e.g. 'XYZC', 'XYZ', 'XYC', 'XY', 'XYZCT').
        forward_permutation: 0-indexed tuple of axis indices to transpose NerveNuc array to ImageJ.
        reverse_permutation: 0-indexed tuple of axis indices to restore ImageJ array to NerveNuc.
    """

    source_axes: str
    imagej_axes: str
    forward_permutation: tuple[int, ...]
    reverse_permutation: tuple[int, ...]


@dataclass(frozen=True)
class ImageJCalibration:
    """Physical pixel calibration metadata transferred to/from ImageJ.

    Attributes:
        pixel_width: Physical X pixel spacing in micrometers (or None if uncalibrated).
        pixel_height: Physical Y pixel spacing in micrometers (or None if uncalibrated).
        pixel_depth: Physical Z slice spacing in micrometers (or None if uncalibrated or 2D).
        unit: Physical unit string (strictly 'um' when calibrated, or 'pixel' / None when uncalibrated).
        is_calibrated: True if valid physical calibration is present, False for uncalibrated pixel arrays.
    """

    pixel_width: float | None = None
    pixel_height: float | None = None
    pixel_depth: float | None = None
    unit: str = "um"
    is_calibrated: bool = False
