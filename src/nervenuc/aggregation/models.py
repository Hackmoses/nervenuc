"""Models for image-to-animal aggregation and biological replicate summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .errors import AggregationConsistencyError, InvalidSampleIdentityError

if TYPE_CHECKING:
    from nervenuc.summary import (
        BiologicalMetricDefinition,
        BiologicalSummary3DResult,
        BiologicalSummaryResult,
    )



@dataclass(frozen=True)
class SampleIdentity:
    """Immutable identity connecting an image/sample to a biological replicate (animal).

    Attributes:
        sample_id: Unique identifier for the image/sample (e.g. 'img01', 'sample_A_01').
        animal_id: Identifier for the biological replicate (e.g. 'Animal_01', 'Mouse_12').
    """

    sample_id: str
    animal_id: str

    def __post_init__(self) -> None:
        for field_name, val in (("sample_id", self.sample_id), ("animal_id", self.animal_id)):
            if not isinstance(val, str) or not val.strip():
                raise InvalidSampleIdentityError(f"{field_name} must be a non-empty string.")
            if val != val.strip():
                raise InvalidSampleIdentityError(
                    f"{field_name} cannot contain leading or trailing whitespace: {val!r}."
                )


@dataclass(frozen=True)
class SampleSummary:
    """Immutable container associating a sample identity with its M6/M14 biological summary result.

    Attributes:
        identity: SampleIdentity specifying sample_id and animal_id.
        summary_result: BiologicalSummaryResult or BiologicalSummary3DResult.
    """

    identity: SampleIdentity
    summary_result: Any

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SampleIdentity):
            raise InvalidSampleIdentityError(
                f"identity must be a SampleIdentity, got {type(self.identity).__name__}."
            )
        from nervenuc.summary import BiologicalSummary3DResult, BiologicalSummaryResult
        if not isinstance(self.summary_result, (BiologicalSummaryResult, BiologicalSummary3DResult)):
            raise AggregationConsistencyError(
                f"summary_result must be a BiologicalSummaryResult or BiologicalSummary3DResult, got {type(self.summary_result).__name__}."
            )



@dataclass
class AnimalAggregationResult:
    """Container for biological replicate (animal-level) aggregation results.

    Attributes:
        animal_metrics: DataFrame containing 8 columns per animal x metric.
        definitions: Tuple of BiologicalMetricDefinition objects evaluated (canonical order).
        samples: Tuple of contributing SampleSummary objects (full provenance).
    """

    animal_metrics: pd.DataFrame
    definitions: tuple[BiologicalMetricDefinition, ...]
    samples: tuple[SampleSummary, ...]
