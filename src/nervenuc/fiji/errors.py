"""Typed exceptions for Fiji and ImageJ interoperability."""

from __future__ import annotations


class NerveNucFijiError(Exception):
    """Base exception for all Fiji/ImageJ integration errors."""


class FijiUnavailableError(NerveNucFijiError):
    """Raised when a local Fiji installation cannot be discovered or accessed."""


class JavaUnavailableError(NerveNucFijiError):
    """Raised when a compatible Java Runtime Environment or PyImageJ is unavailable."""


class FijiConversionError(NerveNucFijiError):
    """Raised when converting data between NumPy and ImageJ fails or violates fidelity."""


class UnsupportedRoiError(NerveNucFijiError):
    """Raised when an ImageJ ROI type is unsupported or incompatible with NerveNuc QC."""


class InvalidRoiGeometryError(NerveNucFijiError):
    """Raised when an ImageJ ROI contains invalid, out-of-bounds, or unrasterizable geometry."""


class FijiExecutionError(NerveNucFijiError):
    """Raised when external Fiji process execution or script bridge invocation fails."""
