"""Tests for exchange TIFF writing and fidelity across layouts and calibration."""

from __future__ import annotations

import numpy as np
import tifffile

from nervenuc.fiji.exchange import FijiExchangeSession
from nervenuc.preparation import VoxelSpacing


def test_write_exchange_tiff_2d_fidelity() -> None:
    """Verify 2D YX uint16 array writes and reads back with identical shape and values."""
    with FijiExchangeSession() as session:
        arr = np.arange(100, dtype=np.uint16).reshape((10, 10))
        p = session.write_exchange_tiff(arr, "YX")

        read_back = tifffile.imread(str(p))
        assert read_back.shape == (10, 10)
        assert read_back.dtype == np.uint16
        assert np.array_equal(arr, read_back)


def test_write_exchange_tiff_3d_calibrated() -> None:
    """Verify calibrated ZYX array writes spacing metadata correctly."""
    with FijiExchangeSession() as session:
        arr = np.arange(3 * 10 * 20, dtype=np.uint16).reshape((3, 10, 20))
        spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104, unit="um")

        p = session.write_exchange_tiff(arr, "ZYX", spacing=spacing)

        with tifffile.TiffFile(str(p)) as tif:
            assert tif.imagej_metadata is not None
            assert tif.imagej_metadata.get("unit") == "um"
            assert tif.imagej_metadata.get("spacing") == 0.5
            read_back = tif.asarray()
            assert read_back.shape == (3, 10, 20)
            assert np.array_equal(arr, read_back)


def test_write_exchange_tiff_uncalibrated_invariant() -> None:
    """Verify uncalibrated ZYX array does not write fake 1 um spacing."""
    with FijiExchangeSession() as session:
        arr = np.arange(2 * 5 * 5, dtype=np.uint16).reshape((2, 5, 5))
        p = session.write_exchange_tiff(arr, "ZYX", spacing=None)

        with tifffile.TiffFile(str(p)) as tif:
            meta = tif.imagej_metadata or {}
            assert "unit" not in meta or meta.get("unit") != "um"
            assert "spacing" not in meta


def test_input_array_not_mutated_during_exchange_write() -> None:
    """Verify input array is strictly read-only and unmutated."""
    with FijiExchangeSession() as session:
        arr = np.arange(50, dtype=np.uint16).reshape((5, 10))
        copy = arr.copy()
        session.write_exchange_tiff(arr, "YX")
        assert np.array_equal(arr, copy)
