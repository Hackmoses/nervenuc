"""Typed metadata describing the extraction and preparation provenance of an image set."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping



@dataclass(frozen=True)
class ImageSourceProvenance:
    """Typed metadata describing the extraction and preparation provenance of an image set.

    Contains only lightweight metadata strings, integers, and hashes. Does not retain NumPy arrays.

    Attributes:
        source_kind: 'direct_2d' or 'projected_stack'.
        source_id: Optional caller-declared or file identifier.
        source_hash: Optional SHA-256 hash of raw source file bytes.
        metadata_axes: Authoritative metadata axes if available.
        declared_axes: Caller-declared axes string if supplied.
        effective_axes: Normalized AxisLayout string actually applied.
        time_index: Timepoint index extracted (or None).
        nuclear_channel: 0-indexed nuclear channel.
        marker_channels: Mapping of marker names to channel indices.
        z_start: Initial Z-slice index used.
        z_stop: Final Z-slice index used (exclusive).
        projection_method: Projection method applied ('max' or 'mean').
        physical_size_z_um: Physical Z step in micrometers if known.
        preparation_fingerprint: SHA-256 fingerprint of StackPreparationConfig.
        pipeline_fingerprint: Combined pipeline SHA-256 fingerprint.
        qc_review_policy: Batch QC policy when provenance was created.
        manual_per_object_qc_performed: False for auto-accepted batch execution; otherwise unknown.
    """

    source_kind: Literal["direct_2d", "projected_stack"]
    source_id: str | None = None
    source_hash: str | None = None
    metadata_axes: str | None = None
    declared_axes: str | None = None
    effective_axes: str | None = None
    time_index: int | None = None
    nuclear_channel: int | None = None
    marker_channels: Mapping[str, int] | None = None
    z_start: int | None = None
    z_stop: int | None = None
    projection_method: str | None = None
    physical_size_z_um: float | None = None
    preparation_fingerprint: str | None = None
    pipeline_fingerprint: str | None = None
    qc_review_policy: Literal["manual", "auto_accept"] | None = None
    manual_per_object_qc_performed: bool | None = None

    def __post_init__(self) -> None:
        if self.marker_channels is not None:
            object.__setattr__(
                self,
                "marker_channels",
                MappingProxyType(dict(self.marker_channels)),
            )


@dataclass(frozen=True)
class VolumeSourceProvenance:
    """Typed metadata describing the extraction and 3D preparation provenance of a volume set.

    Contains only lightweight metadata strings, integers, and hashes. Does not retain NumPy arrays.

    Attributes:
        representation: Fixed literal 'volume_3d'.
        measurement_semantics: Fixed literal 'nuclear_voxels_3d'.
        source_id: Authoritative source/execution identifier.
        source_hash: Optional SHA-256 hash of raw source file bytes.
        preparation_provenance: VolumePreparationProvenance recording extraction facts.
        spacing: Calibrated physical voxel spacing if known, else None.
        volume_config_fingerprint: SHA-256 fingerprint of VolumeAnalysisConfig.
        pipeline_fingerprint: Combined 3D pipeline SHA-256 fingerprint.
    """

    source_id: str
    source_hash: str | None = None
    preparation_provenance: Any = None
    spacing: Any = None
    volume_config_fingerprint: str | None = None
    pipeline_fingerprint: str | None = None
    representation: Literal["volume_3d"] = "volume_3d"
    measurement_semantics: Literal["nuclear_voxels_3d"] = "nuclear_voxels_3d"
    qc_review_policy: Literal["manual", "auto_accept"] | None = None
    manual_per_object_qc_performed: bool | None = None
