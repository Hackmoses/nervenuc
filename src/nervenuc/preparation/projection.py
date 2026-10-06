"""Deterministic 2D projection and end-to-end stack preparation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from nervenuc.io import ImageMetadata, load_tiff

from .axes import extract_subvolume, resolve_axes
from .errors import (
    InvalidAxisLayoutError,
    InvalidProjectionError,
    NerveNucPreparationError,
)
from .models import (
    PreparedImageSet,
    ProjectionResult,
    StackPreparationConfig,
)


def project_stack(
    subvolume: np.ndarray,
    method: Literal["max", "mean"] = "max",
    physical_size_z_um: float | None = None,
    metadata_axes: str | None = None,
    declared_axes: str | None = None,
    effective_axes: str = "ZYX",
    channel_index: int = 0,
    time_index: int | None = None,
    z_start: int = 0,
    z_stop: int = 1,
    source_shape: tuple[int, ...] | None = None,
) -> ProjectionResult:
    """Project a 3D (Z, Y, X) subvolume along the Z axis to produce a 2D (Y, X) array.

    Args:
        subvolume: 3D NumPy array of shape (Z, Y, X).
        method: Projection method ('max' or 'mean').
        physical_size_z_um: Physical Z step in micrometers if known.
        metadata_axes: Original metadata axes if known.
        declared_axes: Caller-declared axes if supplied.
        effective_axes: Normalized layout axes used for extraction.
        channel_index: 0-indexed channel number extracted.
        time_index: 0-indexed timepoint extracted.
        z_start: Initial Z-slice used.
        z_stop: Final Z-slice used (exclusive).
        source_shape: Shape of the input stack before extraction.

    Returns:
        ProjectionResult containing 2D projected image and extraction provenance.

    Raises:
        InvalidProjectionError: If array is not numeric, contains non-finite values,
            or an unsupported projection method is requested.
    """
    if not isinstance(subvolume, np.ndarray) or subvolume.ndim != 3:
        raise InvalidProjectionError(
            f"subvolume must be a 3D numpy array, got {getattr(subvolume, 'ndim', None)}D."
        )

    if not np.issubdtype(subvolume.dtype, np.number) or subvolume.dtype == bool or np.iscomplexobj(subvolume):
        raise InvalidProjectionError(
            f"Array dtype '{subvolume.dtype}' is not supported for quantitative projection. "
            "Must be standard real integer or float numeric type."
        )

    if np.issubdtype(subvolume.dtype, np.floating) and not np.all(np.isfinite(subvolume)):
        raise InvalidProjectionError("Non-finite values (NaN, Inf) encountered in stack array.")

    if method == "max":
        proj_2d = np.max(subvolume, axis=0)
    elif method == "mean":
        proj_2d = np.mean(subvolume, axis=0, dtype=np.float64)
    else:
        raise InvalidProjectionError(f"Unsupported projection method '{method}'. Must be 'max' or 'mean'.")

    return ProjectionResult(
        image=proj_2d,
        source_shape=source_shape if source_shape is not None else subvolume.shape,
        metadata_axes=metadata_axes,
        declared_axes=declared_axes,
        effective_axes=effective_axes,
        channel_index=channel_index,
        time_index=time_index,
        z_start=z_start,
        z_stop=z_stop,
        method=method,
        physical_size_z_um=physical_size_z_um,
    )


def prepare_stack(
    data: np.ndarray | Path | str,
    config: StackPreparationConfig,
    metadata: ImageMetadata | None = None,
) -> PreparedImageSet:
    """Prepare a quantitative 2D image set from a microscopy multi-channel Z-stack.

    Args:
        data: Multi-dimensional NumPy array or path to TIFF/OME-TIFF image file.
        config: Scientifically authoritative StackPreparationConfig.
        metadata: Optional ImageMetadata when data is passed as an in-memory array.

    Returns:
        PreparedImageSet with 2D nuclear channel, 2D marker channels, and provenance.

    Raises:
        NerveNucPreparationError: If loading, axis resolution, channel extraction, or projection fails.
    """
    if not isinstance(config, StackPreparationConfig):
        raise NerveNucPreparationError(
            f"config must be a StackPreparationConfig, got {type(config).__name__}."
        )

    loaded_meta: ImageMetadata | None = metadata
    if isinstance(data, (str, Path)):
        loaded = load_tiff(data)
        raw_array = loaded.data
        loaded_meta = loaded.metadata
    elif isinstance(data, np.ndarray):
        raw_array = data
    else:
        raise NerveNucPreparationError(
            f"data must be a numpy.ndarray, str, or Path, got {type(data).__name__}."
        )

    is_ome = loaded_meta.is_ome if loaded_meta is not None else False
    meta_axes = loaded_meta.axes if loaded_meta is not None else None
    layout = resolve_axes(
        shape=raw_array.shape,
        declared_axes=config.axes,
        metadata_axes=meta_axes,
        is_ome=is_ome,
    )

    if layout.index_of("Z") is None:
        raise InvalidAxisLayoutError(
            f"prepare_stack requires a Z axis in resolved layout '{layout.axes}'. "
            "For 2D images, use M11/M12 direct image analysis."
        )

    ps_x = loaded_meta.pixel_size_x_um if loaded_meta is not None else None
    ps_y = loaded_meta.pixel_size_y_um if loaded_meta is not None else None
    ps_z = loaded_meta.pixel_size_z_um if loaded_meta is not None else None

    # 1. Nuclear channel
    nuc_subvol, z_start, z_stop = extract_subvolume(
        raw_array,
        layout=layout,
        time_index=config.time_index,
        channel_index=config.nuclear_channel,
        z_selection=config.z_selection,
    )
    nuc_proj = project_stack(
        subvolume=nuc_subvol,
        method=config.projection_method,
        physical_size_z_um=ps_z,
        metadata_axes=meta_axes,
        declared_axes=config.axes,
        effective_axes=layout.axes,
        channel_index=config.nuclear_channel,
        time_index=config.time_index,
        z_start=z_start,
        z_stop=z_stop,
        source_shape=raw_array.shape,
    )

    # 2. Marker channels
    marker_images: dict[str, np.ndarray] = {}
    provenance_map: dict[str, ProjectionResult] = {"nuclear": nuc_proj}

    for m_name, m_idx in config.marker_channels.items():
        m_subvol, m_z_start, m_z_stop = extract_subvolume(
            raw_array,
            layout=layout,
            time_index=config.time_index,
            channel_index=m_idx,
            z_selection=config.z_selection,
        )
        m_proj = project_stack(
            subvolume=m_subvol,
            method=config.projection_method,
            physical_size_z_um=ps_z,
            metadata_axes=meta_axes,
            declared_axes=config.axes,
            effective_axes=layout.axes,
            channel_index=m_idx,
            time_index=config.time_index,
            z_start=m_z_start,
            z_stop=m_z_stop,
            source_shape=raw_array.shape,
        )
        marker_images[m_name] = m_proj.image
        provenance_map[m_name] = m_proj

    return PreparedImageSet(
        nuclear_image=nuc_proj.image,
        marker_images=marker_images,
        provenance=provenance_map,
        pixel_size_x_um=ps_x,
        pixel_size_y_um=ps_y,
    )
