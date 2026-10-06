"""Exceptions for marker positivity thresholds and phenotype classification."""

from __future__ import annotations


class NerveNucClassificationError(Exception):
    """Base exception for all NerveNuc classification errors."""


class InvalidThresholdDefinitionError(NerveNucClassificationError):
    """Raised when a marker threshold definition is invalid or unsupported."""


class UnknownMarkerError(NerveNucClassificationError):
    """Raised when a threshold specifies a marker name absent from measurements."""


class ClassificationConsistencyError(NerveNucClassificationError):
    """Raised when measurement inputs or provenance are structurally inconsistent."""
