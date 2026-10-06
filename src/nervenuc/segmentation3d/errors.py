"""Custom exceptions for 3D nuclear segmentation."""

from __future__ import annotations


class NerveNucSegmentation3DError(Exception):
    """Base exception for all NerveNuc 3D segmentation errors."""


class InvalidSegmentation3DConfigError(NerveNucSegmentation3DError):
    """Raised when 3D segmentation configuration fails validation."""


class InconsistentSegmentation3DResultError(NerveNucSegmentation3DError):
    """Raised when a 3D segmentation result fails structural or provenance consistency validation."""
