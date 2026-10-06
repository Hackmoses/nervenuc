"""Data models for 3D marker measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nervenuc.preparation import VolumePreparationProvenance

if TYPE_CHECKING:
    from nervenuc.qc3d import ReviewedSegmentation3DResult


@dataclass(frozen=True)
class MarkerMeasurement3DResult:
    """Container for per-nucleus 3D marker signal measurements.

    Attributes:
        measurements: Long-form DataFrame containing 12 columns per active nucleus x marker.
        reviewed_result: Provenance reference to the ReviewedSegmentation3DResult used.
        marker_names: Tuple of marker names measured.
        source_id: Authoritative identifier of the source volume execution.
        volume_provenance: VolumePreparationProvenance recording extraction facts.
    """

    measurements: pd.DataFrame
    reviewed_result: ReviewedSegmentation3DResult
    marker_names: tuple[str, ...]
    source_id: str
    volume_provenance: VolumePreparationProvenance
