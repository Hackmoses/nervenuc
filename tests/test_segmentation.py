"""Tests for 2D classical nuclear segmentation baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nervenuc.segmentation import (
    InvalidSegmentationInputError,
    NerveNucSegmentationError,
    SegmentationConfig,
    SegmentationResult,
    segment_nuclei,
)

EXPECTED_COLUMNS = [
    "nucleus_id",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "touches_border",
    "qc_status",
]


def _draw_disc(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int,
    intensity: float = 1000.0,
) -> None:
    """Draw a filled circle on a 2D array in-place."""
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
    image[mask] = intensity


def test_single_isolated_nucleus() -> None:
    """Test segmentation of a single isolated nucleus."""
    image = np.zeros((100, 100), dtype=np.uint16)
    _draw_disc(image, center_y=50, center_x=50, radius=10, intensity=2000)

    config = SegmentationConfig(gaussian_sigma=1.0, min_area_px=50)
    result = segment_nuclei(image, config)

    assert isinstance(result, SegmentationResult)
    assert result.labels.shape == (100, 100)
    assert result.labels.dtype == np.int32
    assert len(result.objects) == 1

    row = result.objects.iloc[0]
    assert row["nucleus_id"] == 1
    # Expected area of radius 10 disc is ~314 pixels
    assert 280 <= row["area_px"] <= 340
    assert pytest.approx(row["centroid_y"], abs=0.5) == 50.0
    assert pytest.approx(row["centroid_x"], abs=0.5) == 50.0
    assert row["touches_border"] is False or row["touches_border"] == False  # noqa: E712
    assert row["qc_status"] == "accepted"
    assert (result.labels == 1).sum() == row["area_px"]


def test_multiple_isolated_nuclei() -> None:
    """Test segmentation of multiple non-touching nuclei."""
    image = np.zeros((120, 120), dtype=np.uint16)
    centers = [(30, 30), (30, 90), (90, 30), (90, 90)]
    for cy, cx in centers:
        _draw_disc(image, center_y=cy, center_x=cx, radius=8, intensity=3000)

    result = segment_nuclei(image, SegmentationConfig(min_area_px=50))

    assert len(result.objects) == 4
    assert list(result.objects["nucleus_id"]) == [1, 2, 3, 4]
    assert list(np.unique(result.labels)) == [0, 1, 2, 3, 4]
    assert (result.objects["touches_border"] == False).all()  # noqa: E712
    assert (result.objects["qc_status"] == "accepted").all()


def test_touching_nuclei_watershed_separation() -> None:
    """Test that overlapping/touching nuclei are separated by watershed."""
    image = np.zeros((100, 100), dtype=np.float32)
    # Two overlapping circles of radius 12 separated by 16 pixels
    _draw_disc(image, center_y=50, center_x=42, radius=12, intensity=500.0)
    _draw_disc(image, center_y=50, center_x=58, radius=12, intensity=500.0)

    config = SegmentationConfig(
        gaussian_sigma=1.0,
        min_distance_px=5,
        min_area_px=50,
    )
    result = segment_nuclei(image, config)

    assert len(result.objects) == 2
    assert list(result.objects["nucleus_id"]) == [1, 2]
    assert list(np.unique(result.labels)) == [0, 1, 2]
    assert (result.objects["qc_status"] == "accepted").all()


def test_all_zero_empty_image() -> None:
    """Test that an all-zero image safely returns an empty segmentation result."""
    image = np.zeros((80, 80), dtype=np.uint16)
    result = segment_nuclei(image)

    assert result.labels.shape == (80, 80)
    assert result.labels.dtype == np.int32
    assert np.all(result.labels == 0)
    assert isinstance(result.objects, pd.DataFrame)
    assert len(result.objects) == 0
    assert list(result.objects.columns) == EXPECTED_COLUMNS


def test_uniform_intensity_image() -> None:
    """Test that a uniform non-zero image returns an empty result without error."""
    image = np.full((64, 64), 500, dtype=np.uint16)
    result = segment_nuclei(image)

    assert result.labels.shape == (64, 64)
    assert np.all(result.labels == 0)
    assert len(result.objects) == 0
    assert list(result.objects.columns) == EXPECTED_COLUMNS


def test_border_object_handling() -> None:
    """Test border object behavior under exclude_border=False vs True."""
    image = np.zeros((100, 100), dtype=np.uint16)
    # Central nucleus
    _draw_disc(image, center_y=50, center_x=50, radius=10, intensity=1000)
    # Border-touching nucleus (at y=0)
    _draw_disc(image, center_y=5, center_x=50, radius=10, intensity=1000)

    # 1. exclude_border = False (retain and flag)
    config_retain = SegmentationConfig(exclude_border=False, min_area_px=30)
    res_retain = segment_nuclei(image, config_retain)

    assert len(res_retain.objects) == 2
    border_rows = res_retain.objects[res_retain.objects["touches_border"] == True]  # noqa: E712
    central_rows = res_retain.objects[res_retain.objects["touches_border"] == False]  # noqa: E712
    assert len(border_rows) == 1
    assert len(central_rows) == 1
    assert border_rows.iloc[0]["qc_status"] == "border"
    assert central_rows.iloc[0]["qc_status"] == "accepted"

    # 2. exclude_border = True (remove and relabel)
    config_exclude = SegmentationConfig(exclude_border=True, min_area_px=30)
    res_exclude = segment_nuclei(image, config_exclude)

    assert len(res_exclude.objects) == 1
    assert res_exclude.objects.iloc[0]["nucleus_id"] == 1
    assert res_exclude.objects.iloc[0]["touches_border"] is False or res_exclude.objects.iloc[0]["touches_border"] == False  # noqa: E712
    assert res_exclude.objects.iloc[0]["qc_status"] == "accepted"
    assert list(np.unique(res_exclude.labels)) == [0, 1]


def test_min_area_filter() -> None:
    """Test that objects smaller than min_area_px are removed."""
    image = np.zeros((100, 100), dtype=np.uint16)
    # Valid nucleus (radius 10 -> area ~314)
    _draw_disc(image, center_y=50, center_x=50, radius=10, intensity=1000)
    # Tiny noise speck (radius 1 -> area ~5)
    _draw_disc(image, center_y=20, center_x=20, radius=1, intensity=1000)

    config = SegmentationConfig(min_area_px=50)
    result = segment_nuclei(image, config)

    assert len(result.objects) == 1
    assert result.objects.iloc[0]["nucleus_id"] == 1
    assert result.objects.iloc[0]["area_px"] > 200
    assert list(np.unique(result.labels)) == [0, 1]


def test_max_area_filter() -> None:
    """Test that objects larger than max_area_px are removed."""
    image = np.zeros((150, 150), dtype=np.uint16)
    # Normal nucleus (radius 8 -> area ~200)
    _draw_disc(image, center_y=40, center_x=40, radius=8, intensity=1000)
    # Huge clump (radius 25 -> area ~1960)
    _draw_disc(image, center_y=100, center_x=100, radius=25, intensity=1000)

    config = SegmentationConfig(min_area_px=50, max_area_px=500)
    result = segment_nuclei(image, config)

    assert len(result.objects) == 1
    assert result.objects.iloc[0]["nucleus_id"] == 1
    assert result.objects.iloc[0]["area_px"] < 500
    assert result.objects.iloc[0]["qc_status"] == "accepted"
    assert list(np.unique(result.labels)) == [0, 1]


def test_input_immutability() -> None:
    """Test that the input image array is never modified in place."""
    rng = np.random.default_rng(42)
    original = rng.integers(0, 5000, size=(100, 100), dtype=np.uint16)
    _draw_disc(original, center_y=50, center_x=50, radius=10, intensity=10000)
    copy_img = original.copy()

    segment_nuclei(original)

    assert np.array_equal(original, copy_img)


def test_invalid_inputs_and_config() -> None:
    """Test rejection of invalid input images and configurations."""
    valid_img = np.zeros((50, 50), dtype=np.float32)

    # 1. Non-NumPy input
    with pytest.raises(InvalidSegmentationInputError, match="numpy.ndarray"):
        segment_nuclei([[1, 2], [3, 4]])  # type: ignore[arg-type]

    # 2. 3D array
    with pytest.raises(InvalidSegmentationInputError, match="2-dimensional"):
        segment_nuclei(np.zeros((3, 50, 50), dtype=np.float32))

    # 3. 1D array
    with pytest.raises(InvalidSegmentationInputError, match="2-dimensional"):
        segment_nuclei(np.zeros(50, dtype=np.float32))

    # 4. Empty array
    with pytest.raises(InvalidSegmentationInputError, match="empty"):
        segment_nuclei(np.zeros((0, 50), dtype=np.float32))

    # 5. Non-numeric dtype (string/bool/complex)
    with pytest.raises(InvalidSegmentationInputError, match="numeric dtype"):
        segment_nuclei(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(InvalidSegmentationInputError, match="numeric dtype"):
        segment_nuclei(np.zeros((50, 50), dtype=bool))
    with pytest.raises(InvalidSegmentationInputError, match="numeric dtype"):
        segment_nuclei(np.zeros((50, 50), dtype=np.complex128))

    # 6. NaN or Inf values
    nan_img = valid_img.copy()
    nan_img[10, 10] = np.nan
    with pytest.raises(InvalidSegmentationInputError, match="non-finite"):
        segment_nuclei(nan_img)

    inf_img = valid_img.copy()
    inf_img[10, 10] = np.inf
    with pytest.raises(InvalidSegmentationInputError, match="non-finite"):
        segment_nuclei(inf_img)

    # 7. Invalid configuration
    with pytest.raises(InvalidSegmentationInputError, match="threshold_method"):
        SegmentationConfig(threshold_method="unsupported")

    with pytest.raises(InvalidSegmentationInputError, match="manual_threshold is required"):
        SegmentationConfig(threshold_method="manual", manual_threshold=None)

    with pytest.raises(InvalidSegmentationInputError, match="gaussian_sigma"):
        SegmentationConfig(gaussian_sigma=-1.0)

    with pytest.raises(InvalidSegmentationInputError, match="min_area_px"):
        SegmentationConfig(min_area_px=-5)

    with pytest.raises(InvalidSegmentationInputError, match="max_area_px"):
        SegmentationConfig(max_area_px=-10)

    with pytest.raises(InvalidSegmentationInputError, match="less than min_area_px"):
        SegmentationConfig(min_area_px=100, max_area_px=50)

    with pytest.raises(InvalidSegmentationInputError, match="min_distance_px"):
        SegmentationConfig(min_distance_px=0)


def test_determinism() -> None:
    """Test that segment_nuclei is deterministic across repeated runs."""
    image = np.zeros((100, 100), dtype=np.uint16)
    _draw_disc(image, center_y=30, center_x=30, radius=8, intensity=2000)
    _draw_disc(image, center_y=70, center_x=70, radius=12, intensity=3000)

    config = SegmentationConfig(gaussian_sigma=1.0, min_area_px=40)
    res1 = segment_nuclei(image, config)
    res2 = segment_nuclei(image, config)

    assert np.array_equal(res1.labels, res2.labels)
    assert res1.objects.equals(res2.objects)


def test_manual_and_li_thresholds() -> None:
    """Test manual and Li thresholding methods."""
    image = np.zeros((100, 100), dtype=np.float32)
    _draw_disc(image, center_y=50, center_x=50, radius=10, intensity=100.0)

    # Manual thresholding
    res_manual = segment_nuclei(
        image,
        SegmentationConfig(threshold_method="manual", manual_threshold=50.0),
    )
    assert len(res_manual.objects) == 1

    # Li thresholding
    res_li = segment_nuclei(
        image,
        SegmentationConfig(threshold_method="li"),
    )
    assert len(res_li.objects) == 1
