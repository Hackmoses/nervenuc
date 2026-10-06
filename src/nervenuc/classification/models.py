"""Models for marker positivity thresholds and phenotype classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .errors import InvalidThresholdDefinitionError

if TYPE_CHECKING:
    from nervenuc.measurements import MarkerMeasurementResult
    from nervenuc.measurements3d import MarkerMeasurement3DResult


SUPPORTED_MEASUREMENTS = {
    "mean_intensity",
    "median_intensity",
    "min_intensity",
    "max_intensity",
    "integrated_intensity",
}

SUPPORTED_COMPARISONS = {">=", ">"}


@dataclass(frozen=True)
class MarkerThreshold:
    """Immutable definition of a positivity threshold for a biological marker.

    Attributes:
        marker_name: Exact case-sensitive user marker name (e.g. 'SOX10', 'BrdU').
        threshold: Numeric cutoff value (finite float or int; negative values allowed).
        measurement: Raw measurement statistic to evaluate ('mean_intensity',
            'median_intensity', 'min_intensity', 'max_intensity', 'integrated_intensity').
        comparison: Comparison operator: '>=' (default) or '>'.
    """

    marker_name: str
    threshold: float
    measurement: str = "mean_intensity"
    comparison: str = ">="

    def __post_init__(self) -> None:
        if not isinstance(self.marker_name, str):
            raise InvalidThresholdDefinitionError(
                f"marker_name must be a string, got {type(self.marker_name).__name__}."
            )
        if not self.marker_name or not self.marker_name.strip():
            raise InvalidThresholdDefinitionError(
                "marker_name cannot be empty or whitespace-only."
            )
        if self.marker_name != self.marker_name.strip():
            raise InvalidThresholdDefinitionError(
                f"marker_name cannot contain leading or trailing whitespace: '{self.marker_name}'."
            )

        if self.measurement not in SUPPORTED_MEASUREMENTS:
            raise InvalidThresholdDefinitionError(
                f"measurement must be one of {sorted(SUPPORTED_MEASUREMENTS)}, got '{self.measurement}'."
            )

        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise InvalidThresholdDefinitionError(
                f"threshold must be a real numeric value, got {type(self.threshold).__name__}."
            )
        if not np.isfinite(self.threshold):
            raise InvalidThresholdDefinitionError("threshold must be finite (not NaN or Inf).")

        if self.comparison not in SUPPORTED_COMPARISONS:
            raise InvalidThresholdDefinitionError(
                f"comparison must be one of {sorted(SUPPORTED_COMPARISONS)}, got '{self.comparison}'."
            )


@dataclass
class ClassificationResult:
    """Container for per-nucleus marker classifications and composite phenotypes.

    Attributes:
        classifications: Long-form DataFrame containing 11 columns per active nucleus x threshold.
        phenotypes: Wide-form DataFrame containing 6 columns per active nucleus.
        thresholds: Tuple of MarkerThreshold definitions applied.
        measurement_result: Provenance reference to the MarkerMeasurementResult used.
    """

    classifications: pd.DataFrame
    phenotypes: pd.DataFrame
    thresholds: tuple[MarkerThreshold, ...]
    measurement_result: MarkerMeasurementResult


@dataclass
class Classification3DResult:
    """Container for 3D per-nucleus marker classifications and composite phenotypes.

    Attributes:
        classifications: Long-form DataFrame containing 11 columns per active nucleus x threshold.
        phenotypes: Wide-form DataFrame containing 6 columns per active nucleus.
        thresholds: Tuple of MarkerThreshold definitions applied.
        measurement_result: Provenance reference to the MarkerMeasurement3DResult used.
    """

    classifications: pd.DataFrame
    phenotypes: pd.DataFrame
    thresholds: tuple[MarkerThreshold, ...]
    measurement_result: Any

    @property
    def source_id(self) -> str:
        """Authoritative execution source identity inherited from measurement_result."""
        return getattr(self.measurement_result, "source_id", "")
