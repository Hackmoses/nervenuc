"""Deterministic 3D subvolume extraction and stack preparation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nervenuc.io import ImageMetadata, load_tiff

from .axes import extract_subvolume, resolve_axes
from .errors import (
    InvalidAxisLayoutError,
    NerveNucPreparationError,
)
from .models import (
    PreparedVolumeSet,
    VolumePreparationProvenance,
    VolumeSelectionConfig,
    VoxelSpacing,
)


def prepare_volume(
    data: np.ndarray | Path | str,
    config: VolumeSelectionConfig,
    metadata: ImageMetadata | None = None,
) -> PreparedVolumeSet:
    """Prepare a quantitative 3D subvolume set from a microscopy multi-channel Z-stack.

    Args:
        data: Multi-dimensional NumPy array or path to TIFF/OME-TIFF image file.
        config: Scientifically authoritative VolumeSelectionConfig.
        metadata: Optional ImageMetadata when data is passed as an in-memory array.

    Returns:
        PreparedVolumeSet with 3D nuclear channel, 3D marker channels, spacing, and extraction provenance.

    Raises:
        NerveNucPreparationError: If loading, axis resolution, or channel extraction fails.
    """
    if not isinstance(config, VolumeSelectionConfig):
        raise NerveNucPreparationError(
            f"config must be a VolumeSelectionConfig, got {type(config).__name__}."
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
            f"prepare_volume requires a Z axis in resolved layout '{layout.axes}'. "
            "For 2D images, use M11/M12 direct image analysis."
        )

    # Physical spacing calibration: only create VoxelSpacing if all Z, Y, X physical sizes are available and valid positive floats
    spacing: VoxelSpacing | None = None
    if loaded_meta is not None:
        ps_x = loaded_meta.pixel_size_x_um
        ps_y = loaded_meta.pixel_size_y_um
        ps_z = loaded_meta.pixel_size_z_um
        if (
            ps_x is not None
            and ps_y is not None
            and ps_z is not None
            and isinstance(ps_x, (int, float))
            and isinstance(ps_y, (int, float))
            and isinstance(ps_z, (int, float))
            and not isinstance(ps_x, bool)
            and not isinstance(ps_y, bool)
            and not isinstance(ps_z, bool)
            and ps_x > 0
            and ps_y > 0
            and ps_z > 0
        ):
            spacing = VoxelSpacing(z=float(ps_z), y=float(ps_y), x=float(ps_x), unit="um")

    # 1. Nuclear channel
    nuc_subvol, z_start, z_stop = extract_subvolume(
        raw_array,
        layout=layout,
        time_index=config.time_index,
        channel_index=config.nuclear_channel,
        z_selection=config.z_selection,
    )

    # 2. Marker channels
    marker_volumes: dict[str, np.ndarray] = {}
    for m_name, m_idx in config.marker_channels.items():
        m_subvol, _, _ = extract_subvolume(
            raw_array,
            layout=layout,
            time_index=config.time_index,
            channel_index=m_idx,
            z_selection=config.z_selection,
        )
        marker_volumes[m_name] = m_subvol

    provenance = VolumePreparationProvenance(
        source_shape=raw_array.shape,
        metadata_axes=meta_axes,
        declared_axes=config.axes,
        effective_axes=layout.axes,
        time_index=config.time_index,
        nuclear_channel=config.nuclear_channel,
        marker_channels=dict(config.marker_channels),
        z_start=z_start,
        z_stop=z_stop,
    )

    return PreparedVolumeSet(
        nuclear_volume=nuc_subvol,
        marker_volumes=marker_volumes,
        spacing=spacing,
        provenance=provenance,
    )
