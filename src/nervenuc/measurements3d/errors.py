"""Custom exceptions for 3D marker measurements."""

from __future__ import annotations


class NerveNucMeasurement3DError(Exception):
    """Base exception for all NerveNuc 3D measurement errors."""


class InvalidMarkerVolumeError(NerveNucMeasurement3DError):
    """Raised when a marker volume is invalid or misaligned with the nuclear volume."""


class Measurement3DConsistencyError(NerveNucMeasurement3DError):
    """Raised when 3D measurement data fails structural or provenance consistency validation."""
