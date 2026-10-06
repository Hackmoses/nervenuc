"""Custom exceptions for 3D nuclear segmentation QC and manual review."""

from __future__ import annotations


class NerveNucReview3DError(Exception):
    """Base exception for all NerveNuc 3D review errors."""


class IncompleteReview3DError(NerveNucReview3DError):
    """Raised when attempting to finalize a 3D review session with unresolved objects."""


class InvalidReviewOperation3DError(NerveNucReview3DError):
    """Raised when an invalid review action is requested."""


class UnknownReviewObject3DError(NerveNucReview3DError):
    """Raised when referencing an object ID that does not exist in the 3D review session."""
