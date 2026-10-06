"""Exceptions for per-nucleus marker signal measurements."""

from __future__ import annotations


class NerveNucMeasurementError(Exception):
    """Base exception for all NerveNuc marker measurement errors."""


class InvalidMarkerImageError(NerveNucMeasurementError):
    """Raised when a supplied marker image fails input validation."""


class MeasurementConsistencyError(NerveNucMeasurementError):
    """Raised when reviewed segmentation results have structural inconsistencies."""
