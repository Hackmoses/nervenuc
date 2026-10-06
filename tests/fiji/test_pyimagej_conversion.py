"""Tests for to_imagej and from_imagej conversion functions."""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from nervenuc.fiji.bridge import FijiBridge
from nervenuc.fiji.conversion import from_imagej, to_imagej
from nervenuc.fiji.errors import JavaUnavailableError
from nervenuc.preparation import VoxelSpacing


@pytest.fixture(autouse=True)
def reset_global_bridge() -> None:
    """Reset global FijiBridge instance."""
    FijiBridge._global_ij_instance = None
    yield
    FijiBridge._global_ij_instance = None


def test_to_imagej_without_initialized_gateway_raises_java_unavailable() -> None:
    """Verify to_imagej raises JavaUnavailableError when no active gateway is provided."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(JavaUnavailableError, match="Active PyImageJ instance"):
        to_imagej(arr, "YX")


def test_to_imagej_with_mocked_gateway() -> None:
    """Verify to_imagej transposes array to ImageJ dimension order and invokes py.to_dataset."""
    # Source: CZYX = (2, 3, 10, 20)
    arr = np.arange(2 * 3 * 10 * 20, dtype=np.uint16).reshape((2, 3, 10, 20))
    spacing = VoxelSpacing(z=0.5, y=0.1, x=0.1, unit="um")

    mock_ij = MagicMock()
    mock_dataset = MagicMock()
    mock_ij.py.to_dataset.return_value = mock_dataset

    res = to_imagej(arr, "CZYX", spacing=spacing, ij=mock_ij)
    assert res is mock_dataset

    # Verify array passed to to_dataset has shape (20, 10, 3, 2) -> (X, Y, Z, C)
    called_arr = mock_ij.py.to_dataset.call_args[0][0]
    assert called_arr.shape == (20, 10, 3, 2)


def test_from_imagej_with_mocked_gateway() -> None:
    """Verify from_imagej converts dataset back to NumPy array with restored axis order."""
    mock_ij = MagicMock()
    # Mock raw array in ImageJ order (X, Y, Z, C) = (20, 10, 3, 2)
    ij_arr = np.arange(20 * 10 * 3 * 2, dtype=np.uint16).reshape((20, 10, 3, 2))
    mock_ij.py.from_java.return_value = ij_arr

    mock_dataset = MagicMock()
    restored, axes, spacing = from_imagej(mock_dataset, target_axes="CZYX", ij=mock_ij)

    assert axes == "CZYX"
    assert restored.shape == (2, 3, 10, 20)


@pytest.mark.fiji
def test_live_pyimagej_conversion() -> None:
    """Live test running against a real local PyImageJ/Fiji environment if installed."""
    try:
        import imagej
    except (ImportError, ModuleNotFoundError):
        pytest.skip("PyImageJ is not installed in the current environment.")

    ij = imagej.init(mode="headless")
    arr = np.arange(100, dtype=np.uint16).reshape((10, 10))
    dataset = to_imagej(arr, "YX", ij=ij)
    restored, axes, _ = from_imagej(dataset, target_axes="YX", ij=ij)

    assert axes == "YX"
    assert restored.shape == (10, 10)
    assert np.array_equal(arr, restored)
