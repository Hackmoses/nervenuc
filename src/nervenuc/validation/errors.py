"""Validation error hierarchy for NerveNuc benchmarking and reproducibility."""

from __future__ import annotations


class ValidationError(Exception):
    """Base exception for all validation, ground-truth, matching, and metric errors."""


class InvalidMatchingInputError(ValidationError):
    """Raised when label arrays have mismatched shapes, invalid dtypes, negative values, or unsupported dimensions."""


class InvalidThresholdError(ValidationError):
    """Raised when an IoU matching threshold is out of valid bounds (0 < tau <= 1)."""


class InvalidGroundTruthError(ValidationError):
    """Raised when ground-truth instance annotations or label masks violate specification constraints."""


class ValidationCaseSpecificationError(ValidationError):
    """Raised when a validation case specification has missing or corrupted metadata or hash mismatch."""


class InvalidMetricInputError(ValidationError):
    """Raised when metric evaluation inputs have missing object states, mismatched IDs, or invalid categories."""
