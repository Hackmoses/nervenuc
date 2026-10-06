"""Custom exceptions for segmentation QC and manual review."""

from __future__ import annotations


class NerveNucReviewError(Exception):
    """Base exception for all NerveNuc manual review and QC errors."""


class InvalidReviewOperationError(NerveNucReviewError):
    """Raised when a review operation is invalid for an object's current state."""


class InvalidReviewMaskError(NerveNucReviewError):
    """Raised when a supplied manual or replacement mask fails validation."""


class UnknownReviewObjectError(NerveNucReviewError):
    """Raised when a requested review_id is not found in the review session."""


class IncompleteReviewError(NerveNucReviewError):
    """Raised when active nuclei remain unresolved during review finalization."""
