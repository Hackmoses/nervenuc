"""Biological-replicate (animal-level) aggregation support."""

from .engine import (
    aggregate_by_animal,
    validate_aggregation_consistency,
)
from .errors import (
    AggregationConsistencyError,
    InvalidSampleIdentityError,
    NerveNucAggregationError,
)
from .models import (
    AnimalAggregationResult,
    SampleIdentity,
    SampleSummary,
)

__all__ = [
    "AggregationConsistencyError",
    "AnimalAggregationResult",
    "InvalidSampleIdentityError",
    "NerveNucAggregationError",
    "SampleIdentity",
    "SampleSummary",
    "aggregate_by_animal",
    "validate_aggregation_consistency",
]
