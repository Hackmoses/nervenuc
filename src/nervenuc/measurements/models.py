"""Models for per-nucleus marker signal measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nervenuc.markers import MarkerDefinition

if TYPE_CHECKING:
    from nervenuc.qc import ReviewedSegmentationResult


@dataclass
class MarkerMeasurementResult:
    """Container for per-nucleus marker signal measurements.

    Attributes:
        measurements: Long-form DataFrame containing 12 columns per active nucleus x marker.
        markers: Tuple of MarkerDefinition objects measured.
        reviewed_result: Provenance reference to the ReviewedSegmentationResult used.
    """

    measurements: pd.DataFrame
    markers: tuple[MarkerDefinition, ...]
    reviewed_result: ReviewedSegmentationResult
