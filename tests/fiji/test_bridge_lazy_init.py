"""Tests for lazy PyImageJ gateway and JVM isolation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from nervenuc.fiji.bridge import FijiBridge
from nervenuc.fiji.errors import JavaUnavailableError


@pytest.fixture(autouse=True)
def reset_global_bridge_instance() -> None:
    """Reset global bridge singleton before and after each test."""
    FijiBridge._global_ij_instance = None
    yield
    FijiBridge._global_ij_instance = None


def test_fiji_bridge_construction_is_completely_inert() -> None:
    """Verify that constructing a FijiBridge instance does not import imagej or initialize JVM."""
    bridge = FijiBridge(Path("/opt/Fiji.app"))
    assert bridge.is_initialized is False
    assert bridge.fiji_path == Path("/opt/Fiji.app")


def test_fiji_bridge_initialize_missing_dependency_raises_controlled_error() -> None:
    """Verify missing PyImageJ package raises JavaUnavailableError with pip instructions."""
    bridge = FijiBridge()

    # Simulate imagej not installed
    with patch.dict(sys.modules, {"imagej": None}):
        with pytest.raises(JavaUnavailableError, match="PyImageJ is not installed"):
            bridge.initialize()


def test_fiji_bridge_initialize_mocked_success_and_singleton() -> None:
    """Verify successful initialization caches and reuses the singleton instance."""
    mock_imagej_module = MagicMock()
    mock_ij_instance = MagicMock()
    mock_imagej_module.init.return_value = mock_ij_instance

    with patch.dict(sys.modules, {"imagej": mock_imagej_module}):
        bridge1 = FijiBridge(Path("/path/to/Fiji.app"))
        assert bridge1.is_initialized is False

        res1 = bridge1.initialize()
        assert res1 is mock_ij_instance
        assert bridge1.is_initialized is True
        mock_imagej_module.init.assert_called_once_with(str(Path("/path/to/Fiji.app")), mode="headless")

        # Second call or second instance reuses singleton without re-calling init
        bridge2 = FijiBridge()
        res2 = bridge2.initialize()
        assert res2 is mock_ij_instance
        assert mock_imagej_module.init.call_count == 1
