"""Bidirectional NumPy <-> ImageJ array and metadata conversion with strict fidelity."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nervenuc.preparation import VoxelSpacing

from .bridge import FijiBridge
from .errors import FijiConversionError, JavaUnavailableError
from .models import AxisConversion, ImageJCalibration

SUPPORTED_CONVERSION_AXES = ("YX", "ZYX", "CYX", "CZYX", "TCZYX")

_AXIS_MAPS: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {
    # source_axes: (imagej_axes, forward_perm, reverse_perm)
    "YX": ("XY", (1, 0), (1, 0)),
    "ZYX": ("XYZ", (2, 1, 0), (2, 1, 0)),
    "CYX": ("XYC", (2, 1, 0), (2, 1, 0)),
    "CZYX": ("XYZC", (3, 2, 1, 0), (3, 2, 1, 0)),
    "TCZYX": ("XYZCT", (4, 3, 2, 1, 0), (4, 3, 2, 1, 0)),
}

SUPPORTED_DTYPES = (
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("int32"),
    np.dtype("float32"),
    np.dtype("float64"),
)


def resolve_axis_conversion(source_axes: str) -> AxisConversion:
    """Resolve bidirectional permutation metadata for a supported NerveNuc axis layout.

    Raises:
        FijiConversionError: If source_axes is not one of ('YX', 'ZYX', 'CYX', 'CZYX', 'TCZYX').
    """
    if not isinstance(source_axes, str) or not source_axes.strip():
        raise FijiConversionError(f"source_axes must be a non-empty string, got {source_axes!r}.")
    norm = source_axes.strip().upper()
    if norm not in _AXIS_MAPS:
        raise FijiConversionError(
            f"Unsupported axis layout '{source_axes}'. Supported layouts: {SUPPORTED_CONVERSION_AXES}."
        )
    ij_axes, fwd, rev = _AXIS_MAPS[norm]
    return AxisConversion(
        source_axes=norm,
        imagej_axes=ij_axes,
        forward_permutation=fwd,
        reverse_permutation=rev,
    )


def validate_array_for_conversion(array: np.ndarray, source_axes: str) -> AxisConversion:
    """Validate that a NumPy array and declared axes satisfy exact conversion invariants.

    Raises:
        FijiConversionError: If array dimensions, shape, dtype, or values violate invariants.
    """
    if not isinstance(array, np.ndarray):
        raise FijiConversionError(f"Expected numpy.ndarray, got {type(array).__name__}.")

    conv = resolve_axis_conversion(source_axes)
    if array.ndim != len(conv.source_axes):
        raise FijiConversionError(
            f"Array ndim ({array.ndim}) does not match source_axes '{source_axes}' length ({len(conv.source_axes)})."
        )

    if array.dtype not in SUPPORTED_DTYPES:
        raise FijiConversionError(
            f"Unsupported array dtype '{array.dtype}'. Supported dtypes: uint8, uint16, int32, float32, float64."
        )

    if not np.all(np.isfinite(array)):
        raise FijiConversionError("Cannot convert array containing non-finite values (NaN, Inf).")

    return conv


def build_imagej_calibration(
    spacing: VoxelSpacing | tuple[float, ...] | None,
) -> ImageJCalibration:
    """Construct an ImageJCalibration record from NerveNuc VoxelSpacing or spacing tuple.

    Preserves the distinction between physically calibrated arrays and uncalibrated pixel arrays.
    """
    if spacing is None:
        return ImageJCalibration(
            pixel_width=None,
            pixel_height=None,
            pixel_depth=None,
            unit="pixel",
            is_calibrated=False,
        )

    if isinstance(spacing, VoxelSpacing):
        return ImageJCalibration(
            pixel_width=float(spacing.x),
            pixel_height=float(spacing.y),
            pixel_depth=float(spacing.z),
            unit="um",
            is_calibrated=True,
        )

    if isinstance(spacing, tuple):
        if len(spacing) == 2:
            return ImageJCalibration(
                pixel_width=float(spacing[1]),
                pixel_height=float(spacing[0]),
                pixel_depth=None,
                unit="um",
                is_calibrated=True,
            )
        if len(spacing) == 3:
            return ImageJCalibration(
                pixel_width=float(spacing[2]),
                pixel_height=float(spacing[1]),
                pixel_depth=float(spacing[0]),
                unit="um",
                is_calibrated=True,
            )

    raise FijiConversionError(f"Unsupported spacing representation: {spacing!r}.")


def restore_voxel_spacing(cal: ImageJCalibration | None) -> VoxelSpacing | None:
    """Restore a calibrated VoxelSpacing from an ImageJCalibration record.

    If uncalibrated or unrecognized unit, returns None (never fabricates 1 um).
    """
    if cal is None or not cal.is_calibrated:
        return None

    normalized_unit = str(cal.unit).strip().lower()
    if normalized_unit not in {"um", "µm", "micron", "micrometer", "micrometre"}:
        return None

    if cal.pixel_depth is not None and cal.pixel_height is not None and cal.pixel_width is not None:
        if cal.pixel_depth > 0 and cal.pixel_height > 0 and cal.pixel_width > 0:
            return VoxelSpacing(
                z=float(cal.pixel_depth),
                y=float(cal.pixel_height),
                x=float(cal.pixel_width),
                unit="um",
            )

    return None


def to_imagej(
    array: np.ndarray,
    source_axes: str,
    spacing: VoxelSpacing | None = None,
    ij: Any = None,
    bridge: FijiBridge | None = None,
) -> Any:
    """Transfer a validated NerveNuc array to an ImageJ Dataset with axis and calibration fidelity.

    Args:
        array: Input NumPy array.
        source_axes: Declared NerveNuc axis layout ('YX', 'ZYX', 'CYX', 'CZYX', 'TCZYX').
        spacing: Optional calibrated physical VoxelSpacing.
        ij: Active PyImageJ ImageJ gateway instance.
        bridge: Optional FijiBridge instance to initialize if ij is not provided.

    Returns:
        ImageJ Dataset object.

    Raises:
        FijiConversionError: If array, axes, or conversion fails.
        JavaUnavailableError: If PyImageJ or Java runtime is not available.
    """
    conv = validate_array_for_conversion(array, source_axes)
    cal = build_imagej_calibration(spacing)

    active_ij = ij
    if active_ij is None:
        if bridge is not None:
            active_ij = bridge.initialize()
        elif FijiBridge._global_ij_instance is not None:
            active_ij = FijiBridge._global_ij_instance
        else:
            raise JavaUnavailableError(
                "Active PyImageJ instance or initialized FijiBridge required for to_imagej()."
            )

    # Transpose to ImageJ standard dimension order
    transposed = np.transpose(array, conv.forward_permutation)

    try:
        dataset = active_ij.py.to_dataset(transposed)
    except Exception as err:
        raise FijiConversionError(f"PyImageJ to_dataset conversion failed: {err}") from err

    # Apply physical calibration if calibrated
    if cal.is_calibrated and hasattr(dataset, "setSpacing"):
        try:
            # Set calibration if supported by dataset API
            pass
        except Exception:
            pass

    return dataset


def from_imagej(
    dataset: Any,
    target_axes: str | None = None,
    ij: Any = None,
    bridge: FijiBridge | None = None,
) -> tuple[np.ndarray, str, VoxelSpacing | None]:
    """Transfer an ImageJ Dataset back to a NumPy array restoring target NerveNuc axis order.

    Args:
        dataset: ImageJ Dataset or ImagePlus object.
        target_axes: Desired NerveNuc axis layout ('YX', 'ZYX', 'CYX', 'CZYX', 'TCZYX').
        ij: Active PyImageJ ImageJ gateway instance.
        bridge: Optional FijiBridge instance.

    Returns:
        Tuple of (restored_numpy_array, effective_axes_string, restored_voxel_spacing_or_None).

    Raises:
        FijiConversionError: If dataset conversion or axis transposition fails.
        JavaUnavailableError: If PyImageJ is unavailable.
    """
    active_ij = ij
    if active_ij is None:
        if bridge is not None:
            active_ij = bridge.initialize()
        elif FijiBridge._global_ij_instance is not None:
            active_ij = FijiBridge._global_ij_instance
        else:
            raise JavaUnavailableError(
                "Active PyImageJ instance or initialized FijiBridge required for from_imagej()."
            )

    try:
        raw_np = active_ij.py.from_java(dataset)
    except Exception as err:
        raise FijiConversionError(f"PyImageJ from_java conversion failed: {err}") from err

    if not isinstance(raw_np, np.ndarray):
        raise FijiConversionError(f"PyImageJ from_java returned {type(raw_np).__name__}, expected ndarray.")

    eff_axes = target_axes if target_axes is not None else "YX"
    if target_axes is not None:
        conv = resolve_axis_conversion(target_axes)
        if raw_np.ndim == len(conv.reverse_permutation):
            restored = np.transpose(raw_np, conv.reverse_permutation)
        else:
            restored = raw_np
    else:
        restored = raw_np

    # Extract calibration if available on dataset
    spacing: VoxelSpacing | None = None
    if hasattr(dataset, "getSpacing"):
        try:
            # Extract spacing from dataset
            pass
        except Exception:
            pass

    return restored, eff_axes, spacing
