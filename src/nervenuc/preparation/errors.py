"""Exception hierarchy for image preparation and Z-stack projection."""

from __future__ import annotations


class NerveNucPreparationError(Exception):
    """Base exception for image preparation and projection errors."""


class InvalidAxisLayoutError(NerveNucPreparationError):
    """Raised when an axis string, dimension length, or shape is invalid or unsupported."""


class AmbiguousImageAxesError(NerveNucPreparationError):
    """Raised when stack axes cannot be unambiguously determined without explicit declaration."""


class InvalidChannelSelectionError(NerveNucPreparationError):
    """Raised when channel indices are invalid, out of bounds, or duplicate roles are mapped."""


class InvalidZSelectionError(NerveNucPreparationError):
    """Raised when Z-slice selection is invalid, out of bounds, or inverted."""


class InvalidProjectionError(NerveNucPreparationError):
    """Raised when an unsupported projection method or non-numeric/non-finite array is encountered."""
