"""Custom exceptions for microscopy image input/output."""

from __future__ import annotations


class NerveNucImageError(Exception):
    """Base exception for all NerveNuc image I/O errors."""


class UnsupportedImageError(NerveNucImageError):
    """Raised when an image format, encoding, or structure is unsupported."""


class UnreadableImageError(NerveNucImageError):
    """Raised when an image file cannot be read, does not exist, or is corrupted."""
