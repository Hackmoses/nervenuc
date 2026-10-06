"""Exceptions for image-to-animal aggregation and biological replicate summaries."""

from __future__ import annotations


class NerveNucAggregationError(Exception):
    """Base exception for all NerveNuc aggregation errors."""


class InvalidSampleIdentityError(NerveNucAggregationError):
    """Raised when a sample or animal identity is invalid, empty, or duplicate."""


class AggregationConsistencyError(NerveNucAggregationError):
    """Raised when sample summaries are structurally or numerically inconsistent."""
