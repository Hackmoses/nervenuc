"""Custom exceptions for nuclear segmentation."""

from __future__ import annotations


class NerveNucSegmentationError(Exception):
    """Base exception for all NerveNuc segmentation errors."""


class InvalidSegmentationInputError(NerveNucSegmentationError):
    """Raised when an input image or configuration fails validation."""
