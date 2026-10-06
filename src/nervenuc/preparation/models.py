"""Data models for microscopy Z-stack axis layouts, configuration, projections, and inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from .errors import (
    InvalidAxisLayoutError,
    InvalidChannelSelectionError,
    InvalidProjectionError,
    InvalidZSelectionError,
    NerveNucPreparationError,
)


@dataclass(frozen=True)
class AxisLayout:
    """Explicit declaration of dimensional axes for an image array.

    Attributes:
        axes: String of single-character axis identifiers (e.g., 'YX', 'ZYX', 'CZYX', 'TCZYX').
        shape: Tuple of integers representing dimension sizes matching the axis string length.
    """

    axes: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, str) or not self.axes.strip():
            raise InvalidAxisLayoutError("axes must be a non-empty string.")
        norm_axes = self.axes.strip().upper()
        if not isinstance(self.shape, tuple) or not all(isinstance(s, int) and s > 0 for s in self.shape):
            raise InvalidAxisLayoutError(f"shape must be a tuple of positive integers, got {self.shape}.")
        if len(norm_axes) != len(self.shape):
            raise InvalidAxisLayoutError(
                f"Length of axes '{norm_axes}' ({len(norm_axes)}) does not match shape {self.shape} ({len(self.shape)})."
            )
        valid_chars = {"Y", "X", "Z", "C", "T"}
        seen = set()
        for ch in norm_axes:
            if ch not in valid_chars:
                raise InvalidAxisLayoutError(
                    f"Invalid scientific axis symbol '{ch}'. Allowed symbols: {sorted(valid_chars)}."
                )
            if ch in seen:
                raise InvalidAxisLayoutError(f"Duplicate axis symbol '{ch}' in '{norm_axes}'.")
            seen.add(ch)
        if "Y" not in seen or "X" not in seen:
            raise InvalidAxisLayoutError(f"Spatial axes 'Y' and 'X' must both be present in '{norm_axes}'.")
        object.__setattr__(self, "axes", norm_axes)

    def index_of(self, axis_char: str) -> int | None:
        """Return 0-indexed position of an axis character if present, else None."""
        norm_char = axis_char.strip().upper()
        if norm_char in self.axes:
            return self.axes.index(norm_char)
        return None

    def size_of(self, axis_char: str) -> int | None:
        """Return dimension size of an axis character if present, else None."""
        idx = self.index_of(axis_char)
        return self.shape[idx] if idx is not None else None


@dataclass(frozen=True)
class ZSelection:
    """Explicit Z-slice range specification using half-open interval [start, stop).

    Attributes:
        start: 0-indexed start slice (inclusive).
        stop: 0-indexed stop slice (exclusive).
    """

    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise InvalidZSelectionError(f"Z start must be an integer, got {type(self.start).__name__}.")
        if not isinstance(self.stop, int) or isinstance(self.stop, bool):
            raise InvalidZSelectionError(f"Z stop must be an integer, got {type(self.stop).__name__}.")
        if self.start < 0:
            raise InvalidZSelectionError(f"Z start must be non-negative, got {self.start}.")
        if self.stop <= self.start:
            raise InvalidZSelectionError(
                f"Z stop ({self.stop}) must be strictly greater than start ({self.start})."
            )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary representation."""
        return {"start": self.start, "stop": self.stop}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZSelection:
        """Reconstruct from dictionary."""
        if not isinstance(data, dict):
            raise InvalidZSelectionError("ZSelection data must be a dictionary.")
        if "start" not in data or "stop" not in data:
            raise InvalidZSelectionError("ZSelection dictionary must contain 'start' and 'stop' keys.")
        return cls(start=int(data["start"]), stop=int(data["stop"]))


@dataclass(frozen=True)
class StackPreparationConfig:
    """Complete, immutable configuration for multi-channel Z-stack preparation.

    Attributes:
        axes: Explicit axis string (e.g. 'CZYX'), or None if supplied by metadata.
        time_index: Timepoint to extract if T > 1.
        nuclear_channel: 0-indexed channel for nuclear stain (default 0).
        marker_channels: Mapping of marker names to 0-indexed channel indices.
        z_selection: ZSelection range, or None for all available slices.
        projection_method: Projection method ('max' or 'mean', default 'max').
    """

    axes: str | None = None
    time_index: int | None = None
    nuclear_channel: int = 0
    marker_channels: Mapping[str, int] = field(default_factory=dict)
    z_selection: ZSelection | None = None
    projection_method: Literal["max", "mean"] = "max"

    def __post_init__(self) -> None:
        if self.axes is not None and (not isinstance(self.axes, str) or not self.axes.strip()):
            raise InvalidAxisLayoutError("axes must be a non-empty string or None.")
        if self.time_index is not None and (
            not isinstance(self.time_index, int) or isinstance(self.time_index, bool) or self.time_index < 0
        ):
            raise NerveNucPreparationError("time_index must be a non-negative integer or None.")

        if not isinstance(self.nuclear_channel, int) or isinstance(self.nuclear_channel, bool) or self.nuclear_channel < 0:
            raise InvalidChannelSelectionError("nuclear_channel must be a non-negative integer.")

        if not isinstance(self.marker_channels, Mapping):
            raise InvalidChannelSelectionError("marker_channels must be a Mapping of marker name to channel index.")

        normalized_markers: dict[str, int] = {}
        for m_name, m_idx in self.marker_channels.items():
            if not isinstance(m_name, str) or not m_name.strip():
                raise InvalidChannelSelectionError("Marker name in marker_channels must be a non-empty string.")
            if not isinstance(m_idx, int) or isinstance(m_idx, bool) or m_idx < 0:
                raise InvalidChannelSelectionError(
                    f"Channel index for marker '{m_name}' must be a non-negative integer."
                )
            normalized_markers[m_name] = m_idx

        # Enforce duplicate channel role policy
        used_channels: dict[int, str] = {self.nuclear_channel: "nuclear"}
        for m_name, m_idx in normalized_markers.items():
            if m_idx in used_channels:
                raise InvalidChannelSelectionError(
                    f"Channel index {m_idx} assigned to multiple roles: '{used_channels[m_idx]}' and '{m_name}'. "
                    "Duplicate channel role assignments are not permitted."
                )
            used_channels[m_idx] = m_name

        object.__setattr__(self, "marker_channels", MappingProxyType(normalized_markers))

        if self.z_selection is not None and not isinstance(self.z_selection, ZSelection):
            raise InvalidZSelectionError(
                f"z_selection must be a ZSelection or None, got {type(self.z_selection).__name__}."
            )

        if self.projection_method not in {"max", "mean"}:
            raise InvalidProjectionError(
                f"Unsupported projection_method '{self.projection_method}'. Must be 'max' or 'mean'."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "axes": self.axes,
            "time_index": self.time_index,
            "nuclear_channel": self.nuclear_channel,
            "marker_channels": dict(self.marker_channels),
            "z_selection": self.z_selection.to_dict() if self.z_selection is not None else None,
            "projection_method": self.projection_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackPreparationConfig:
        """Reconstruct StackPreparationConfig from dictionary."""
        if not isinstance(data, dict):
            raise NerveNucPreparationError("StackPreparationConfig data must be a dictionary.")
        z_sel_data = data.get("z_selection")
        z_sel = ZSelection.from_dict(z_sel_data) if z_sel_data is not None else None
        return cls(
            axes=data.get("axes"),
            time_index=data.get("time_index"),
            nuclear_channel=int(data.get("nuclear_channel", 0)),
            marker_channels=dict(data.get("marker_channels", {})),
            z_selection=z_sel,
            projection_method=data.get("projection_method", "max"),
        )


@dataclass(frozen=True)
class ProjectionResult:
    """2D projection result carrying complete provenance of extraction and projection.

    Attributes:
        image: Exactly 2D (Y, X) NumPy array.
        source_shape: Shape of original input stack.
        metadata_axes: Authoritative metadata axes if available.
        declared_axes: Caller-declared axes string if supplied.
        effective_axes: Normalized AxisLayout string actually applied.
        channel_index: Channel index extracted.
        time_index: Timepoint index extracted (or None).
        z_start: Initial Z-slice index used.
        z_stop: Final Z-slice index used (exclusive).
        method: Projection method applied ('max' or 'mean').
        physical_size_z_um: Physical Z step in micrometers if known.
    """

    image: np.ndarray
    source_shape: tuple[int, ...]
    metadata_axes: str | None
    declared_axes: str | None
    effective_axes: str
    channel_index: int
    time_index: int | None
    z_start: int
    z_stop: int
    method: str
    physical_size_z_um: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 2:
            raise InvalidProjectionError(
                f"ProjectionResult image must be a 2D numpy array, got {getattr(self.image, 'ndim', None)}D."
            )


@dataclass(frozen=True)
class PreparedImageSet:
    """Quantitative 2D image set prepared from a microscopy Z-stack.

    Contained NumPy arrays are treated as read-only by contract.

    Attributes:
        nuclear_image: Exactly 2D (Y, X) array for nuclear channel.
        marker_images: Mapping of marker names to exactly 2D (Y, X) arrays of identical shape.
        provenance: Mapping of channel role ('nuclear' and marker names) to ProjectionResult.
        pixel_size_x_um: Pixel size in X (micrometers) if known.
        pixel_size_y_um: Pixel size in Y (micrometers) if known.
    """

    nuclear_image: np.ndarray
    marker_images: Mapping[str, np.ndarray]
    provenance: Mapping[str, ProjectionResult]
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nuclear_image, np.ndarray) or self.nuclear_image.ndim != 2:
            raise NerveNucPreparationError("nuclear_image must be a 2D numpy array.")
        if not isinstance(self.marker_images, Mapping):
            raise NerveNucPreparationError("marker_images must be a Mapping.")
        for m_name, m_arr in self.marker_images.items():
            if not isinstance(m_arr, np.ndarray) or m_arr.ndim != 2:
                raise NerveNucPreparationError(f"Marker '{m_name}' image must be a 2D numpy array.")
            if m_arr.shape != self.nuclear_image.shape:
                raise NerveNucPreparationError(
                    f"Marker '{m_name}' shape {m_arr.shape} does not match nuclear shape {self.nuclear_image.shape}."
                )


@dataclass(frozen=True)
class StackInspectionResult:
    """Factual metadata summary of a microscopy Z-stack without interpretation."""

    shape: tuple[int, ...]
    axes: str | None
    dtype: str
    ndim: int
    is_ome: bool
    size_x: int | None = None
    size_y: int | None = None
    size_z: int | None = None
    size_c: int | None = None
    size_t: int | None = None
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    pixel_size_z_um: float | None = None
    channel_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class VoxelSpacing:
    """Calibrated physical voxel dimensions for multi-dimensional microscopy stacks.

    Attributes:
        z: Physical Z step (slice spacing) in micrometers.
        y: Physical Y pixel size in micrometers.
        x: Physical X pixel size in micrometers.
        unit: Physical unit, strictly 'um'.
    """

    z: float
    y: float
    x: float
    unit: Literal["um"] = "um"

    def __post_init__(self) -> None:
        for dim, val in (("z", self.z), ("y", self.y), ("x", self.x)):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
                raise InvalidAxisLayoutError(f"VoxelSpacing dimension '{dim}' must be a positive number, got {val!r}.")
        if self.unit != "um":
            raise InvalidAxisLayoutError(f"VoxelSpacing unit must be 'um', got {self.unit!r}.")


@dataclass(frozen=True)
class VolumeSelectionConfig:
    """Explicit configuration for multi-channel Z-stack subvolume extraction.

    Attributes:
        axes: Explicit axis string (e.g. 'CZYX'), or None if supplied by metadata.
        time_index: Timepoint to extract if T > 1.
        nuclear_channel: 0-indexed channel for nuclear stain.
        marker_channels: Mapping of marker names to 0-indexed channel indices.
        z_selection: ZSelection range, or None for all available slices.
    """

    axes: str | None = None
    time_index: int | None = None
    nuclear_channel: int = 0
    marker_channels: Mapping[str, int] = field(default_factory=dict)
    z_selection: ZSelection | None = None

    def __post_init__(self) -> None:
        if self.axes is not None and (not isinstance(self.axes, str) or not self.axes.strip()):
            raise InvalidAxisLayoutError("axes must be a non-empty string or None.")
        if self.time_index is not None and (
            not isinstance(self.time_index, int) or isinstance(self.time_index, bool) or self.time_index < 0
        ):
            raise NerveNucPreparationError("time_index must be a non-negative integer or None.")

        if not isinstance(self.nuclear_channel, int) or isinstance(self.nuclear_channel, bool) or self.nuclear_channel < 0:
            raise InvalidChannelSelectionError("nuclear_channel must be a non-negative integer.")

        if not isinstance(self.marker_channels, Mapping):
            raise InvalidChannelSelectionError("marker_channels must be a Mapping of marker name to channel index.")

        normalized_markers: dict[str, int] = {}
        for m_name, m_idx in self.marker_channels.items():
            if not isinstance(m_name, str) or not m_name.strip():
                raise InvalidChannelSelectionError("Marker name in marker_channels must be a non-empty string.")
            if not isinstance(m_idx, int) or isinstance(m_idx, bool) or m_idx < 0:
                raise InvalidChannelSelectionError(
                    f"Channel index for marker '{m_name}' must be a non-negative integer."
                )
            normalized_markers[m_name] = m_idx

        # Enforce duplicate channel role policy
        used_channels: dict[int, str] = {self.nuclear_channel: "nuclear"}
        for m_name, m_idx in normalized_markers.items():
            if m_idx in used_channels:
                raise InvalidChannelSelectionError(
                    f"Channel index {m_idx} assigned to multiple roles: '{used_channels[m_idx]}' and '{m_name}'. "
                    "Duplicate channel role assignments are not permitted."
                )
            used_channels[m_idx] = m_name

        object.__setattr__(self, "marker_channels", MappingProxyType(normalized_markers))

        if self.z_selection is not None and not isinstance(self.z_selection, ZSelection):
            raise InvalidZSelectionError(
                f"z_selection must be a ZSelection or None, got {type(self.z_selection).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "axes": self.axes,
            "time_index": self.time_index,
            "nuclear_channel": self.nuclear_channel,
            "marker_channels": dict(self.marker_channels),
            "z_selection": self.z_selection.to_dict() if self.z_selection is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolumeSelectionConfig:
        """Reconstruct VolumeSelectionConfig from dictionary."""
        if not isinstance(data, dict):
            raise NerveNucPreparationError("VolumeSelectionConfig data must be a dictionary.")
        z_sel_data = data.get("z_selection")
        z_sel = ZSelection.from_dict(z_sel_data) if z_sel_data is not None else None
        return cls(
            axes=data.get("axes"),
            time_index=data.get("time_index"),
            nuclear_channel=int(data.get("nuclear_channel", 0)),
            marker_channels=dict(data.get("marker_channels", {})),
            z_selection=z_sel,
        )


@dataclass(frozen=True)
class VolumePreparationProvenance:
    """Factual metadata describing subvolume extraction from a microscopy stack."""

    source_shape: tuple[int, ...]
    metadata_axes: str | None
    declared_axes: str | None
    effective_axes: str
    time_index: int | None
    nuclear_channel: int
    marker_channels: Mapping[str, int]
    z_start: int
    z_stop: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "marker_channels",
            MappingProxyType(dict(self.marker_channels)),
        )


@dataclass(frozen=True)
class PreparedVolumeSet:
    """Quantitative 3D subvolume set prepared from a microscopy Z-stack.

    Contained NumPy arrays are treated as read-only by contract.

    Attributes:
        nuclear_volume: Exactly 3D (Z, Y, X) array for nuclear channel.
        marker_volumes: Mapping of marker names to exactly 3D (Z, Y, X) arrays of identical shape.
        spacing: Calibrated physical voxel spacing if known, else None.
        provenance: VolumePreparationProvenance recording extraction facts.
    """

    nuclear_volume: np.ndarray
    marker_volumes: Mapping[str, np.ndarray]
    spacing: VoxelSpacing | None
    provenance: VolumePreparationProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.nuclear_volume, np.ndarray) or self.nuclear_volume.ndim != 3:
            raise NerveNucPreparationError("nuclear_volume must be a 3D numpy array.")
        if not isinstance(self.marker_volumes, Mapping):
            raise NerveNucPreparationError("marker_volumes must be a Mapping.")
        for m_name, m_arr in self.marker_volumes.items():
            if not isinstance(m_arr, np.ndarray) or m_arr.ndim != 3:
                raise NerveNucPreparationError(f"Marker '{m_name}' volume must be a 3D numpy array.")
            if m_arr.shape != self.nuclear_volume.shape:
                raise NerveNucPreparationError(
                    f"Marker '{m_name}' shape {m_arr.shape} does not match nuclear volume shape {self.nuclear_volume.shape}."
                )
        if not isinstance(self.provenance, VolumePreparationProvenance):
            raise NerveNucPreparationError(
                f"provenance must be a VolumePreparationProvenance, got {type(self.provenance).__name__}."
            )
        if self.spacing is not None and not isinstance(self.spacing, VoxelSpacing):
            raise NerveNucPreparationError(
                f"spacing must be a VoxelSpacing or None, got {type(self.spacing).__name__}."
            )
        object.__setattr__(
            self,
            "marker_volumes",
            MappingProxyType(dict(self.marker_volumes)),
        )
