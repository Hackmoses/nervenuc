"""Inspection of microscopy Z-stack metadata and dimensions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nervenuc.io import inspect_tiff

from .errors import NerveNucPreparationError
from .models import AxisLayout, StackInspectionResult


def inspect_stack(
    path_or_array: str | Path | np.ndarray,
    declared_axes: str | None = None,
) -> StackInspectionResult:
    """Inspect microscopy stack dimensions, physical calibration, and channels factually.

    Args:
        path_or_array: Path to image file or in-memory NumPy array.
        declared_axes: Optional axis declaration to parse dimensions for plain arrays.

    Returns:
        StackInspectionResult object.

    Raises:
        NerveNucPreparationError: If input is invalid.
    """
    if isinstance(path_or_array, (str, Path)):
        meta = inspect_tiff(path_or_array)
        eff_axes = declared_axes if declared_axes is not None else meta.axes
        size_x = meta.size_x
        size_y = meta.size_y
        size_z = meta.size_z
        size_c = meta.size_c
        size_t = meta.size_t

        if declared_axes is not None:
            try:
                layout = AxisLayout(declared_axes, meta.shape)
                size_x = layout.size_of("X")
                size_y = layout.size_of("Y")
                size_z = layout.size_of("Z")
                size_c = layout.size_of("C")
                size_t = layout.size_of("T")
            except Exception:
                pass

        return StackInspectionResult(
            shape=meta.shape,
            axes=eff_axes,
            dtype=meta.dtype,
            ndim=meta.ndim,
            is_ome=meta.is_ome,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
            pixel_size_x_um=meta.pixel_size_x_um,
            pixel_size_y_um=meta.pixel_size_y_um,
            pixel_size_z_um=meta.pixel_size_z_um,
            channel_names=meta.channel_names,
        )

    if isinstance(path_or_array, np.ndarray):
        shape = path_or_array.shape
        ndim = path_or_array.ndim
        dtype_str = str(path_or_array.dtype)

        size_x: int | None = None
        size_y: int | None = None
        size_z: int | None = None
        size_c: int | None = None
        size_t: int | None = None

        if declared_axes is not None:
            layout = AxisLayout(declared_axes, shape)
            size_x = layout.size_of("X")
            size_y = layout.size_of("Y")
            size_z = layout.size_of("Z")
            size_c = layout.size_of("C")
            size_t = layout.size_of("T")
        elif ndim == 2:
            size_y, size_x = shape

        return StackInspectionResult(
            shape=shape,
            axes=declared_axes if declared_axes is not None else ("YX" if ndim == 2 else None),
            dtype=dtype_str,
            ndim=ndim,
            is_ome=False,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
        )

    raise NerveNucPreparationError(
        f"path_or_array must be a str, Path, or numpy.ndarray, got {type(path_or_array).__name__}."
    )
