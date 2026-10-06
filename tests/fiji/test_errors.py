"""Tests for Fiji error taxonomy and inheritance hierarchy."""

from __future__ import annotations

import pytest

from nervenuc.fiji.errors import (
    FijiConversionError,
    FijiExecutionError,
    FijiUnavailableError,
    InvalidRoiGeometryError,
    JavaUnavailableError,
    NerveNucFijiError,
    UnsupportedRoiError,
)


def test_fiji_error_hierarchy() -> None:
    """Verify all Fiji-specific exceptions inherit from NerveNucFijiError."""
    assert issubclass(FijiUnavailableError, NerveNucFijiError)
    assert issubclass(JavaUnavailableError, NerveNucFijiError)
    assert issubclass(FijiConversionError, NerveNucFijiError)
    assert issubclass(UnsupportedRoiError, NerveNucFijiError)
    assert issubclass(InvalidRoiGeometryError, NerveNucFijiError)
    assert issubclass(FijiExecutionError, NerveNucFijiError)


def test_error_instantiation_and_messages() -> None:
    """Verify custom error messages formatting."""
    err = FijiUnavailableError("Fiji executable not found at /opt/Fiji.app")
    assert str(err) == "Fiji executable not found at /opt/Fiji.app"
    assert isinstance(err, Exception)
