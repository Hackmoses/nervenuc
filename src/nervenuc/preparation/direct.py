"""Authoritative direct-2D channel selection and extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from nervenuc.io import ImageMetadata, load_tiff

from .axes import resolve_axes
from .errors import InvalidAxisLayoutError, InvalidChannelSelectionError, NerveNucPreparationError


@dataclass(frozen=True)
class DirectImageSelectionConfig:
    """Immutable channel selection for one multichannel direct-2D source."""

    axes: str | None = None
    nuclear_channel: int = 0
    marker_channels: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axes is not None and (not isinstance(self.axes, str) or not self.axes.strip()):
            raise InvalidAxisLayoutError("axes must be a non-empty string or None.")
        if not isinstance(self.nuclear_channel, int) or isinstance(self.nuclear_channel, bool) or self.nuclear_channel < 0:
            raise InvalidChannelSelectionError("nuclear_channel must be a non-negative integer.")
        if not isinstance(self.marker_channels, Mapping):
            raise InvalidChannelSelectionError("marker_channels must be a Mapping of marker name to channel index.")

        normalized: dict[str, int] = {}
        used: dict[int, str] = {self.nuclear_channel: "nuclear"}
        for name, index in self.marker_channels.items():
            if not isinstance(name, str) or not name.strip():
                raise InvalidChannelSelectionError("Marker names must be non-empty strings.")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise InvalidChannelSelectionError(
                    f"Channel index for marker '{name}' must be a non-negative integer."
                )
            if index in used:
                raise InvalidChannelSelectionError(
                    f"Channel index {index} is assigned to both '{used[index]}' and '{name}'."
                )
            normalized[name] = index
            used[index] = name
        object.__setattr__(self, "marker_channels", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": self.axes,
            "nuclear_channel": self.nuclear_channel,
            "marker_channels": dict(self.marker_channels),
        }


@dataclass(frozen=True)
class DirectImagePreparationProvenance:
    """Factual extraction metadata for a direct-2D multichannel image."""

    source_shape: tuple[int, ...]
    metadata_axes: str | None
    declared_axes: str | None
    effective_axes: str
    nuclear_channel: int
    marker_channels: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "marker_channels",
            MappingProxyType(dict(self.marker_channels)),
        )


@dataclass(frozen=True)
class PreparedDirectImageSet:
    """Selected 2D nuclear and marker planes with extraction provenance."""

    nuclear_image: np.ndarray
    marker_images: Mapping[str, np.ndarray]
    provenance: DirectImagePreparationProvenance
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nuclear_image, np.ndarray) or self.nuclear_image.ndim != 2:
            raise NerveNucPreparationError("nuclear_image must be a 2D numpy array.")
        for name, image in self.marker_images.items():
            if not isinstance(image, np.ndarray) or image.ndim != 2:
                raise NerveNucPreparationError(f"Marker '{name}' image must be a 2D numpy array.")
            if image.shape != self.nuclear_image.shape:
                raise NerveNucPreparationError(
                    f"Marker '{name}' shape {image.shape} does not match nuclear shape {self.nuclear_image.shape}."
                )
        object.__setattr__(
            self,
            "marker_images",
            MappingProxyType(dict(self.marker_images)),
        )


def compute_direct_selection_fingerprint(config: DirectImageSelectionConfig) -> str:
    """Return a deterministic fingerprint over direct-2D axis/channel selection."""
    if not isinstance(config, DirectImageSelectionConfig):
        raise NerveNucPreparationError(
            f"config must be DirectImageSelectionConfig, got {type(config).__name__}."
        )
    canonical = {
        "kind": "nervenuc_direct_2d_selection_v1",
        **config.to_dict(),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_direct_image(
    data: np.ndarray | Path | str,
    config: DirectImageSelectionConfig,
    metadata: ImageMetadata | None = None,
) -> PreparedDirectImageSet:
    """Extract explicitly selected channels from an unambiguous CYX/YXC direct-2D image."""
    if not isinstance(config, DirectImageSelectionConfig):
        raise NerveNucPreparationError(
            f"config must be DirectImageSelectionConfig, got {type(config).__name__}."
        )

    loaded_meta = metadata
    if isinstance(data, (str, Path)):
        loaded = load_tiff(data)
        raw = loaded.data
        loaded_meta = loaded.metadata
    elif isinstance(data, np.ndarray):
        raw = data
    else:
        raise NerveNucPreparationError(
            f"data must be a numpy.ndarray, str, or Path, got {type(data).__name__}."
        )

    metadata_axes = loaded_meta.axes if loaded_meta is not None else None
    is_ome = loaded_meta.is_ome if loaded_meta is not None else False
    layout = resolve_axes(
        shape=raw.shape,
        declared_axes=config.axes,
        metadata_axes=metadata_axes,
        is_ome=is_ome,
    )
    if layout.axes == "YX":
        raise InvalidChannelSelectionError(
            "Direct 2D marker analysis requires explicit nuclear and marker channels; "
            "a single-channel TIFF cannot be analyzed with a synthetic marker channel."
        )
    if layout.axes not in {"CYX", "YXC"}:
        raise InvalidAxisLayoutError(
            f"Direct 2D mode requires CYX or YXC axes, got '{layout.axes}' for shape {raw.shape}."
        )

    c_axis = layout.index_of("C")
    assert c_axis is not None
    channel_count = layout.size_of("C")
    assert channel_count is not None
    selected = {"nuclear": config.nuclear_channel, **dict(config.marker_channels)}
    for role, index in selected.items():
        if index >= channel_count:
            raise InvalidChannelSelectionError(
                f"Channel index {index} for '{role}' is out of bounds for {channel_count} channels."
            )

    nuclear = np.take(raw, config.nuclear_channel, axis=c_axis)
    markers = {
        name: np.take(raw, index, axis=c_axis)
        for name, index in config.marker_channels.items()
    }
    provenance = DirectImagePreparationProvenance(
        source_shape=tuple(raw.shape),
        metadata_axes=metadata_axes,
        declared_axes=config.axes,
        effective_axes=layout.axes,
        nuclear_channel=config.nuclear_channel,
        marker_channels=dict(config.marker_channels),
    )
    return PreparedDirectImageSet(
        nuclear_image=nuclear,
        marker_images=markers,
        provenance=provenance,
        pixel_size_x_um=loaded_meta.pixel_size_x_um if loaded_meta is not None else None,
        pixel_size_y_um=loaded_meta.pixel_size_y_um if loaded_meta is not None else None,
    )
