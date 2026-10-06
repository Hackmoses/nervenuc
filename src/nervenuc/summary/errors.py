"""Exceptions for denominator-aware biological metrics and phenotype summaries."""

from __future__ import annotations


class NerveNucSummaryError(Exception):
    """Base exception for all NerveNuc summary errors."""


class InvalidMetricDefinitionError(NerveNucSummaryError):
    """Raised when a metric definition is invalid, malformed, or contradictory."""


class UnknownMetricMarkerError(NerveNucSummaryError):
    """Raised when a metric filter specifies a marker not present in the classification result."""


class SummaryConsistencyError(NerveNucSummaryError):
    """Raised when classification inputs or nucleus metadata are structurally inconsistent."""
