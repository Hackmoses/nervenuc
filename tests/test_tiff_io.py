"""Tests for microscopy TIFF and OME-TIFF loading and metadata inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nervenuc.io import (
    ImageMetadata,
    LoadedImage,
    NerveNucImageError,
    UnreadableImageError,
    UnsupportedImageError,
    format_metadata,
    inspect_tiff,
    load_tiff,
)


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_standard_grayscale_tiff(tmp_path: Path) -> None:
    """Test loading and inspecting a standard 2D grayscale TIFF."""
    file_path = tmp_path / "standard.tif"
    rng = np.random.default_rng(42)
    original_data = rng.integers(0, 65535, size=(128, 256), dtype=np.uint16)
    tifffile.imwrite(file_path, original_data)

    metadata = inspect_tiff(file_path)
    assert isinstance(metadata, ImageMetadata)
    assert metadata.file_name == "standard.tif"
    assert metadata.is_ome is False
    assert metadata.shape == (128, 256)
    assert metadata.dtype == "uint16"
    assert metadata.ndim == 2
    assert metadata.size_y == 128
    assert metadata.size_x == 256
    assert metadata.size_z is None
    assert metadata.size_c is None
    assert metadata.size_t is None
    assert metadata.pixel_size_x_um is None
    assert metadata.pixel_size_y_um is None
    assert metadata.pixel_size_z_um is None
    assert metadata.channel_names is None

    loaded = load_tiff(file_path)
    assert isinstance(loaded, LoadedImage)
    assert np.array_equal(loaded.data, original_data)
    assert loaded.data.dtype == np.uint16
    assert loaded.data.shape == (128, 256)
    assert loaded.metadata == metadata


def test_multipage_tiff_with_custom_axes(tmp_path: Path) -> None:
    """Test multi-page/3D TIFF preserves shape and strictly respects reported axes."""
    file_path = tmp_path / "multipage.tif"
    rng = np.random.default_rng(100)
    original_data = rng.integers(0, 255, size=(5, 64, 64), dtype=np.uint8)

    # Write as non-OME multi-page TIFF
    tifffile.imwrite(
        file_path,
        original_data,
        photometric="minisblack",
        metadata={"axes": "QYX"},
    )

    metadata = inspect_tiff(file_path)
    assert metadata.is_ome is False
    assert metadata.shape == (5, 64, 64)
    assert metadata.dtype == "uint8"
    assert metadata.ndim == 3
    assert metadata.axes == "QYX"
    assert metadata.size_y == 64
    assert metadata.size_x == 64
    # Unknown/custom axis 'Q' must not be guessed as Z or C
    assert metadata.size_z is None
    assert metadata.size_c is None
    assert metadata.size_t is None

    loaded = load_tiff(file_path)
    assert np.array_equal(loaded.data, original_data)
    assert loaded.data.shape == (5, 64, 64)


def test_synthetic_ome_tiff(tmp_path: Path) -> None:
    """Test synthetic OME-TIFF metadata extraction and axes/shape consistency."""
    file_path = tmp_path / "sample.ome.tif"
    rng = np.random.default_rng(200)
    # Shape: 4 channels, 10 Z slices, 64 Y, 64 X
    original_data = rng.integers(0, 4095, size=(4, 10, 64, 64), dtype=np.uint16)

    channel_names = ["DAPI", "BrdU", "SOX10", "MBP"]
    tifffile.imwrite(
        file_path,
        original_data,
        photometric="minisblack",
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.104,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.104,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 500.0,
            "PhysicalSizeZUnit": "nm",
            "Channel": {"Name": channel_names},
        },
    )

    metadata = inspect_tiff(file_path)
    assert metadata.is_ome is True
    assert metadata.shape == (4, 10, 64, 64)
    assert metadata.dtype == "uint16"
    assert metadata.ndim == 4
    assert metadata.axes == "CZYX"

    # Verify axes and shape consistency
    assert metadata.size_c == 4
    assert metadata.size_z == 10
    assert metadata.size_y == 64
    assert metadata.size_x == 64
    assert metadata.size_t is None

    # Physical sizes in micrometers (nm converted to µm)
    assert metadata.pixel_size_x_um == pytest.approx(0.104)
    assert metadata.pixel_size_y_um == pytest.approx(0.104)
    assert metadata.pixel_size_z_um == pytest.approx(0.500)

    # Channel names
    assert metadata.channel_names == ("DAPI", "BrdU", "SOX10", "MBP")

    loaded = load_tiff(file_path)
    assert np.array_equal(loaded.data, original_data)
    assert loaded.data.shape == (4, 10, 64, 64)


def test_inspect_tiff_preserves_file_bytes(tmp_path: Path) -> None:
    """Verify that inspect_tiff does not modify the source file."""
    file_path = tmp_path / "immutability_test.tif"
    original_data = np.zeros((32, 32), dtype=np.uint8)
    tifffile.imwrite(file_path, original_data)

    hash_before = _compute_file_hash(file_path)
    metadata = inspect_tiff(file_path)
    hash_after = _compute_file_hash(file_path)

    assert hash_before == hash_after
    assert metadata.shape == (32, 32)


def test_load_tiff_preserves_dimensions_and_pixel_values(tmp_path: Path) -> None:
    """Verify that load_tiff does not squeeze dimensions and preserves exact pixel values."""
    file_path = tmp_path / "stack.tif"
    rng = np.random.default_rng(300)
    original_data = rng.integers(0, 1000, size=(3, 32, 32), dtype=np.int32)
    tifffile.imwrite(file_path, original_data, photometric="minisblack")

    loaded = load_tiff(file_path)
    assert loaded.data.shape == (3, 32, 32)
    assert loaded.metadata.shape == (3, 32, 32)
    assert loaded.metadata.ndim == 3
    assert np.array_equal(loaded.data, original_data)


def test_invariants_channel_identities_and_missing_metadata(tmp_path: Path) -> None:
    """Verify channel identities are never hardcoded and missing metadata remain None."""
    file_path = tmp_path / "raw_channels.tif"
    rng = np.random.default_rng(400)
    # 3-channel image without OME metadata
    original_data = rng.integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
    tifffile.imwrite(file_path, original_data, photometric="minisblack", metadata={"axes": "CYX"})

    metadata = inspect_tiff(file_path)
    assert metadata.size_c == 3
    # Channel names must not default to DAPI or any assumed name
    assert metadata.channel_names is None
    # Unknown physical sizes must remain None
    assert metadata.pixel_size_x_um is None
    assert metadata.pixel_size_y_um is None
    assert metadata.pixel_size_z_um is None
    assert metadata.size_z is None
    assert metadata.size_t is None


def test_missing_file_raises_unreadable_error(tmp_path: Path) -> None:
    """Verify clear error when file does not exist."""
    missing_path = tmp_path / "does_not_exist.tif"
    with pytest.raises(UnreadableImageError, match="does not exist"):
        inspect_tiff(missing_path)

    with pytest.raises(UnreadableImageError, match="does not exist"):
        load_tiff(missing_path)


def test_directory_path_raises_unreadable_error(tmp_path: Path) -> None:
    """Verify clear error when path points to a directory."""
    with pytest.raises(UnreadableImageError, match="not a regular file"):
        inspect_tiff(tmp_path)

    with pytest.raises(UnreadableImageError, match="not a regular file"):
        load_tiff(tmp_path)


def test_corrupted_file_raises_unreadable_error(tmp_path: Path) -> None:
    """Verify clear error when file is corrupted/not a valid TIFF."""
    corrupt_file = tmp_path / "corrupt.tif"
    corrupt_file.write_bytes(b"This is not a TIFF file.")

    with pytest.raises(UnreadableImageError, match="Cannot read TIFF"):
        inspect_tiff(corrupt_file)

    with pytest.raises(UnreadableImageError, match="Cannot read TIFF"):
        load_tiff(corrupt_file)


def test_multiple_image_series_are_rejected(tmp_path: Path) -> None:
    """Ambiguous multi-series TIFFs must not silently select the first series."""
    file_path = tmp_path / "multiple-series.tif"
    with tifffile.TiffWriter(file_path) as tif:
        tif.write(np.zeros((8, 8), dtype=np.uint8), photometric="minisblack")
        tif.write(np.ones((5, 7), dtype=np.uint8), photometric="minisblack")

    with pytest.raises(UnsupportedImageError, match="contains 2 image series"):
        inspect_tiff(file_path)
    with pytest.raises(UnsupportedImageError, match="contains 2 image series"):
        load_tiff(file_path)


def test_exception_hierarchy() -> None:
    """Verify exception inheritance hierarchy."""
    assert issubclass(UnreadableImageError, NerveNucImageError)


def test_format_metadata_known_fields_only(tmp_path: Path) -> None:
    """Verify format_metadata produces clean summary displaying only known values."""
    # 1. Standard non-OME 2D image
    std_meta = ImageMetadata(
        path=Path("test.tif"),
        file_name="test.tif",
        is_ome=False,
        shape=(512, 512),
        dtype="uint16",
        ndim=2,
        size_x=512,
        size_y=512,
    )
    text_std = format_metadata(std_meta)
    assert "File: test.tif" in text_std
    assert "OME-TIFF: no" in text_std
    assert "Shape: (512, 512)" in text_std
    assert "Dtype: uint16" in text_std
    assert "Size: 512 × 512 px" in text_std
    assert "Channels:" not in text_std
    assert "Z slices:" not in text_std
    assert "Pixel size" not in text_std

    # 2. OME image with full metadata
    ome_meta = ImageMetadata(
        path=Path("sample.ome.tif"),
        file_name="sample.ome.tif",
        is_ome=True,
        shape=(4, 20, 1024, 1024),
        dtype="uint16",
        ndim=4,
        axes="CZYX",
        size_x=1024,
        size_y=1024,
        size_z=20,
        size_c=4,
        pixel_size_x_um=0.104,
        pixel_size_y_um=0.104,
        pixel_size_z_um=0.500,
        channel_names=("DAPI", "BrdU", "SOX10", "MBP"),
    )
    text_ome = format_metadata(ome_meta)
    assert "File: sample.ome.tif" in text_ome
    assert "OME-TIFF: yes" in text_ome
    assert "Axes: CZYX" in text_ome
    assert "Channels: 4" in text_ome
    assert "Z slices: 20" in text_ome
    assert "Size: 1024 × 1024 px" in text_ome
    assert "Pixel size XY: 0.104 × 0.104 µm" in text_ome
    assert "Z step: 0.500 µm" in text_ome
    assert "Channel names: DAPI, BrdU, SOX10, MBP" in text_ome
