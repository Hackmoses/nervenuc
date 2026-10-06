"""Tests for user-defined marker mapping and per-nucleus marker signal measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nervenuc.markers import (
    InvalidMarkerDefinitionError,
    MarkerDefinition,
)
from nervenuc.measurements import (
    InvalidMarkerImageError,
    MarkerMeasurementResult,
    MeasurementConsistencyError,
    NerveNucMeasurementError,
    measure_marker,
    measure_markers,
    validate_reviewed_result_consistency,
)
from nervenuc.qc import IncompleteReviewError, ReviewedSegmentationResult, start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult

EXPECTED_MEASUREMENT_COLUMNS = [
    "review_id",
    "current_label",
    "marker_name",
    "pixel_count",
    "mean_intensity",
    "median_intensity",
    "min_intensity",
    "max_intensity",
    "integrated_intensity",
    "source_nucleus_id",
    "review_origin",
    "review_status",
]


def _finalize_review(segmentation: SegmentationResult) -> ReviewedSegmentationResult:
    """Explicitly accept all automated objects and finalize a test review."""
    session = start_review(segmentation)
    for review_id in session.result().objects["review_id"]:
        session.accept(review_id)
    return session.finalize()


def _create_synthetic_reviewed_result() -> ReviewedSegmentationResult:
    """Create a synthetic ReviewedSegmentationResult with 3 active nuclei."""
    labels = np.zeros((100, 100), dtype=np.int32)
    # Nucleus 1 at (25, 25), radius 8
    y, x = np.ogrid[:100, :100]
    mask1 = (y - 25) ** 2 + (x - 25) ** 2 <= 8**2
    # Nucleus 2 at (25, 75), radius 8
    mask2 = (y - 25) ** 2 + (x - 75) ** 2 <= 8**2
    # Nucleus 3 at (75, 50), radius 10
    mask3 = (y - 75) ** 2 + (x - 50) ** 2 <= 10**2

    labels[mask1] = 1
    labels[mask2] = 2
    labels[mask3] = 3

    objects_df = pd.DataFrame(
        [
            {
                "nucleus_id": 1,
                "area_px": int(np.sum(mask1)),
                "centroid_y": 25.0,
                "centroid_x": 25.0,
                "bbox_min_y": 17,
                "bbox_min_x": 17,
                "bbox_max_y": 34,
                "bbox_max_x": 34,
                "touches_border": False,
                "qc_status": "accepted",
            },
            {
                "nucleus_id": 2,
                "area_px": int(np.sum(mask2)),
                "centroid_y": 25.0,
                "centroid_x": 75.0,
                "bbox_min_y": 17,
                "bbox_min_x": 67,
                "bbox_max_y": 34,
                "bbox_max_x": 84,
                "touches_border": False,
                "qc_status": "accepted",
            },
            {
                "nucleus_id": 3,
                "area_px": int(np.sum(mask3)),
                "centroid_y": 75.0,
                "centroid_x": 50.0,
                "bbox_min_y": 65,
                "bbox_min_x": 40,
                "bbox_max_y": 86,
                "bbox_max_x": 61,
                "touches_border": False,
                "qc_status": "accepted",
            },
        ]
    )

    seg_res = SegmentationResult(
        labels=labels,
        objects=objects_df,
        config=SegmentationConfig(),
    )
    return _finalize_review(seg_res)


def test_one_nucleus_one_marker() -> None:
    """Test measuring a single marker on a single nucleus."""
    # Construct 1-nucleus reviewed result
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:20, 10:20] = 1  # 10x10 = 100 pixels
    objects_df = pd.DataFrame(
        [
            {
                "nucleus_id": 1,
                "area_px": 100,
                "centroid_y": 14.5,
                "centroid_x": 14.5,
                "bbox_min_y": 10,
                "bbox_min_x": 10,
                "bbox_max_y": 20,
                "bbox_max_x": 20,
                "touches_border": False,
                "qc_status": "accepted",
            }
        ]
    )
    seg_res = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg_res)

    # Synthetic marker image with gradient in nucleus area
    marker_img = np.zeros((50, 50), dtype=np.float32)
    marker_img[10:20, 10:20] = np.arange(1, 101, dtype=np.float32).reshape(10, 10)

    result = measure_marker(reviewed_res, marker_img, "SOX10")

    assert isinstance(result, MarkerMeasurementResult)
    assert len(result.measurements) == 1
    assert list(result.measurements.columns) == EXPECTED_MEASUREMENT_COLUMNS

    row = result.measurements.iloc[0]
    assert row["review_id"] == "AUTO-000001"
    assert row["current_label"] == 1
    assert row["marker_name"] == "SOX10"
    assert row["pixel_count"] == 100
    assert row["mean_intensity"] == pytest.approx(50.5)
    assert row["median_intensity"] == pytest.approx(50.5)
    assert row["min_intensity"] == 1.0
    assert row["max_intensity"] == 100.0
    assert row["integrated_intensity"] == pytest.approx(5050.0)
    assert row["source_nucleus_id"] == 1
    assert row["review_origin"] == "automated"
    assert row["review_status"] == "accepted"


def test_unresolved_review_cannot_be_measured() -> None:
    """M3-to-M4 boundary rejects an interim review snapshot."""
    finalized = _create_synthetic_reviewed_result()
    unresolved = start_review(finalized.source_result).result()
    marker_img = np.zeros(unresolved.labels.shape, dtype=np.float32)

    with pytest.raises(IncompleteReviewError, match="Review is incomplete"):
        measure_marker(unresolved, marker_img, "SOX10")


def test_multiple_nuclei_one_marker() -> None:
    """Test measuring a single marker across multiple active nuclei."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.full((100, 100), 100.0, dtype=np.float32)
    # Bright spot in nucleus 1
    marker_img[reviewed_res.labels == 1] = 500.0

    result = measure_marker(reviewed_res, marker_img, "BrdU")

    assert len(result.measurements) == 3
    assert list(result.measurements["current_label"]) == [1, 2, 3]

    row1 = result.measurements.iloc[0]
    assert row1["marker_name"] == "BrdU"
    assert row1["mean_intensity"] == 500.0

    row2 = result.measurements.iloc[1]
    assert row2["mean_intensity"] == 100.0


def test_one_nucleus_multiple_markers() -> None:
    """Test measuring multiple marker channels on active nuclei."""
    reviewed_res = _create_synthetic_reviewed_result()
    img_sox10 = np.full((100, 100), 200.0, dtype=np.uint16)
    img_brdu = np.full((100, 100), 400.0, dtype=np.uint16)
    img_ki67 = np.full((100, 100), 800.0, dtype=np.uint16)

    marker_mapping = {
        "SOX10": img_sox10,
        "BrdU": img_brdu,
        "Ki67": img_ki67,
    }

    result = measure_markers(reviewed_res, marker_mapping)

    # 3 nuclei x 3 markers = 9 rows
    assert len(result.measurements) == 9
    assert [m.name for m in result.markers] == ["SOX10", "BrdU", "Ki67"]

    # Verify caller-supplied marker order for each nucleus
    first_nucleus_markers = list(result.measurements.iloc[0:3]["marker_name"])
    assert first_nucleus_markers == ["SOX10", "BrdU", "Ki67"]


def test_marker_names_preserved_exactly() -> None:
    """Test that custom marker names and case are preserved without modification."""
    reviewed_res = _create_synthetic_reviewed_result()
    img = np.zeros((100, 100), dtype=np.float32)

    # SOX10 and sox10 are distinct case-sensitive names
    result = measure_markers(
        reviewed_res,
        {
            "SOX10": img,
            "sox10": img,
            "Marker 1": img,
            "BrdU_Ch2": img,
        },
    )

    names = [m.name for m in result.markers]
    assert names == ["SOX10", "sox10", "Marker 1", "BrdU_Ch2"]
    assert len(result.measurements) == 3 * 4


def test_invalid_marker_names_rejected() -> None:
    """Test rejection of empty, whitespace, and leading/trailing whitespace marker names."""
    with pytest.raises(InvalidMarkerDefinitionError, match="empty or whitespace"):
        MarkerDefinition("")

    with pytest.raises(InvalidMarkerDefinitionError, match="empty or whitespace"):
        MarkerDefinition("   ")

    with pytest.raises(InvalidMarkerDefinitionError, match="leading or trailing whitespace"):
        MarkerDefinition(" SOX10")

    with pytest.raises(InvalidMarkerDefinitionError, match="leading or trailing whitespace"):
        MarkerDefinition("SOX10 ")

    with pytest.raises(InvalidMarkerDefinitionError, match="must be a string"):
        MarkerDefinition(123)  # type: ignore[arg-type]


def test_exact_intensity_calculations() -> None:
    """Test exact statistical calculations on controlled 4-pixel nucleus."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:4, 2:4] = 1  # 4 pixels
    objects_df = pd.DataFrame(
        [
            {
                "nucleus_id": 1,
                "area_px": 4,
                "centroid_y": 2.5,
                "centroid_x": 2.5,
                "bbox_min_y": 2,
                "bbox_min_x": 2,
                "bbox_max_y": 4,
                "bbox_max_x": 4,
                "touches_border": False,
                "qc_status": "accepted",
            }
        ]
    )
    seg_res = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg_res)

    marker_img = np.zeros((10, 10), dtype=np.float32)
    marker_img[2, 2] = 10.0
    marker_img[2, 3] = 20.0
    marker_img[3, 2] = 30.0
    marker_img[3, 3] = 40.0

    result = measure_marker(reviewed_res, marker_img, "TestMarker")
    row = result.measurements.iloc[0]

    assert row["pixel_count"] == 4
    assert row["mean_intensity"] == 25.0
    assert row["median_intensity"] == 25.0
    assert row["min_intensity"] == 10.0
    assert row["max_intensity"] == 40.0
    assert row["integrated_intensity"] == 100.0


def test_integrated_intensity_sum() -> None:
    """Test that integrated_intensity matches exact sum of raw pixel values."""
    reviewed_res = _create_synthetic_reviewed_result()
    rng = np.random.default_rng(42)
    marker_img = rng.integers(100, 5000, size=(100, 100), dtype=np.uint16)

    result = measure_marker(reviewed_res, marker_img, "RawSignal")

    for _, row in result.measurements.iterrows():
        c_lbl = row["current_label"]
        pixels = marker_img[reviewed_res.labels == c_lbl]
        expected_sum = float(np.sum(pixels, dtype=np.float64))
        assert row["integrated_intensity"] == pytest.approx(expected_sum)


def test_pixel_count_matches_mask_area() -> None:
    """Test that pixel_count strictly matches area_px in reviewed_result.objects."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.zeros((100, 100), dtype=np.float32)

    result = measure_marker(reviewed_res, marker_img, "Signal")

    for _, row in result.measurements.iterrows():
        c_lbl = row["current_label"]
        obj_area = int(reviewed_res.objects[reviewed_res.objects["current_label"] == c_lbl]["area_px"].iloc[0])
        assert row["pixel_count"] == obj_area


def test_all_zero_marker_image() -> None:
    """Test that all-zero marker images produce zero statistics without error."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.zeros((100, 100), dtype=np.float32)

    result = measure_marker(reviewed_res, marker_img, "ZeroMarker")

    for _, row in result.measurements.iterrows():
        assert row["mean_intensity"] == 0.0
        assert row["median_intensity"] == 0.0
        assert row["min_intensity"] == 0.0
        assert row["max_intensity"] == 0.0
        assert row["integrated_intensity"] == 0.0


def test_uniform_marker_image() -> None:
    """Test that uniform marker images produce constant statistics."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.full((100, 100), 250.0, dtype=np.float32)

    result = measure_marker(reviewed_res, marker_img, "UniformMarker")

    for _, row in result.measurements.iterrows():
        assert row["mean_intensity"] == 250.0
        assert row["median_intensity"] == 250.0
        assert row["min_intensity"] == 250.0
        assert row["max_intensity"] == 250.0
        assert row["integrated_intensity"] == row["pixel_count"] * 250.0


def test_marker_image_shape_mismatch() -> None:
    """Test that marker image shape mismatch raises InvalidMarkerImageError."""
    reviewed_res = _create_synthetic_reviewed_result()
    wrong_shape_img = np.zeros((120, 100), dtype=np.float32)

    with pytest.raises(InvalidMarkerImageError, match="shape"):
        measure_marker(reviewed_res, wrong_shape_img, "Marker")


def test_non_2d_marker_image_rejected() -> None:
    """Test that 1D or 3D marker images raise InvalidMarkerImageError."""
    reviewed_res = _create_synthetic_reviewed_result()

    with pytest.raises(InvalidMarkerImageError, match="2-dimensional"):
        measure_marker(reviewed_res, np.zeros((3, 100, 100), dtype=np.float32), "M3D")

    with pytest.raises(InvalidMarkerImageError, match="2-dimensional"):
        measure_marker(reviewed_res, np.zeros(100, dtype=np.float32), "M1D")


def test_non_numeric_marker_image_rejected() -> None:
    """Test that boolean, string, or complex marker images raise InvalidMarkerImageError."""
    reviewed_res = _create_synthetic_reviewed_result()

    with pytest.raises(InvalidMarkerImageError, match="numeric dtype"):
        measure_marker(reviewed_res, np.zeros((100, 100), dtype=bool), "MBool")

    with pytest.raises(InvalidMarkerImageError, match="numeric dtype"):
        measure_marker(reviewed_res, np.zeros((100, 100), dtype=np.complex128), "MComplex")

    with pytest.raises(InvalidMarkerImageError, match="numeric dtype"):
        measure_marker(reviewed_res, np.full((100, 100), "signal"), "MStr")


def test_nan_inf_marker_image_rejected() -> None:
    """Test that marker images containing NaN or Inf raise InvalidMarkerImageError."""
    reviewed_res = _create_synthetic_reviewed_result()

    nan_img = np.zeros((100, 100), dtype=np.float32)
    nan_img[10, 10] = np.nan
    with pytest.raises(InvalidMarkerImageError, match="non-finite"):
        measure_marker(reviewed_res, nan_img, "MNan")

    inf_img = np.zeros((100, 100), dtype=np.float32)
    inf_img[10, 10] = np.inf
    with pytest.raises(InvalidMarkerImageError, match="non-finite"):
        measure_marker(reviewed_res, inf_img, "MInf")


def test_source_marker_image_immutability() -> None:
    """Test that the source marker array is never modified by measurement."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.full((100, 100), 42.0, dtype=np.float32)
    img_copy = marker_img.copy()

    measure_marker(reviewed_res, marker_img, "Signal")

    assert np.array_equal(marker_img, img_copy)


def test_reviewed_result_immutability() -> None:
    """Test that reviewed_result.labels and objects are untouched during measurement."""
    reviewed_res = _create_synthetic_reviewed_result()
    labels_copy = reviewed_res.labels.copy()
    objects_copy = reviewed_res.objects.copy(deep=True)

    marker_img = np.full((100, 100), 50.0, dtype=np.float32)
    result = measure_marker(reviewed_res, marker_img, "Signal")

    assert np.array_equal(reviewed_res.labels, labels_copy)
    assert reviewed_res.objects.equals(objects_copy)
    # Result DataFrame must not be an alias of reviewed_result.objects
    assert result.measurements is not reviewed_res.objects


def test_only_active_nuclei_measured() -> None:
    """Test that rejected or deleted nuclei are not measured."""
    source_seg = SegmentationResult(
        labels=_create_synthetic_reviewed_result().labels,
        objects=_create_synthetic_reviewed_result().objects,
        config=SegmentationConfig(),
    )
    session = start_review(source_seg)
    session.reject("AUTO-000002")  # Reject nucleus 2
    session.accept("AUTO-000001")
    session.accept("AUTO-000003")
    reviewed_res = session.finalize()

    assert len(reviewed_res.objects) == 2

    marker_img = np.zeros((100, 100), dtype=np.float32)
    result = measure_marker(reviewed_res, marker_img, "BrdU")

    assert len(result.measurements) == 2
    assert "AUTO-000002" not in result.measurements["review_id"].values
    assert list(result.measurements["current_label"]) == [1, 2]


def test_empty_reviewed_segmentation() -> None:
    """Test that an empty reviewed segmentation returns a valid 12-column empty DataFrame."""
    empty_labels = np.zeros((80, 80), dtype=np.int32)
    empty_objects = pd.DataFrame(
        columns=[
            "review_id", "current_label", "source_nucleus_id", "review_status",
            "review_origin", "parent_review_ids", "review_note", "area_px",
            "centroid_y", "centroid_x", "bbox_min_y", "bbox_min_x",
            "bbox_max_y", "bbox_max_x", "touches_border", "qc_status",
        ]
    )
    empty_seg = SegmentationResult(labels=empty_labels, objects=empty_objects, config=SegmentationConfig())
    reviewed_res = _finalize_review(empty_seg)

    marker_img = np.zeros((80, 80), dtype=np.float32)
    result = measure_marker(reviewed_res, marker_img, "Marker1")

    assert len(result.measurements) == 0
    assert list(result.measurements.columns) == EXPECTED_MEASUREMENT_COLUMNS
    assert len(result.markers) == 1
    assert result.markers[0].name == "Marker1"


def test_measurement_determinism() -> None:
    """Test that repeated measurement calls on identical inputs produce identical results."""
    reviewed_res = _create_synthetic_reviewed_result()
    marker_img = np.arange(10000, dtype=np.float32).reshape(100, 100)

    res1 = measure_markers(reviewed_res, {"M1": marker_img, "M2": marker_img})
    res2 = measure_markers(reviewed_res, {"M1": marker_img, "M2": marker_img})

    assert res1.measurements.equals(res2.measurements)
    assert res1.markers == res2.markers


def test_output_row_count_and_ordering_contract() -> None:
    """Test that row count strictly equals active nuclei x markers with correct ordering."""
    reviewed_res = _create_synthetic_reviewed_result()
    img1 = np.full((100, 100), 10.0, dtype=np.float32)
    img2 = np.full((100, 100), 20.0, dtype=np.float32)

    result = measure_markers(reviewed_res, {"Alpha": img1, "Beta": img2})

    # 3 active nuclei x 2 markers = 6 rows
    assert len(result.measurements) == 6

    # Expected order: (nucleus 1, Alpha), (nucleus 1, Beta), (nucleus 2, Alpha), (nucleus 2, Beta), ...
    expected_tuples = [
        (1, "Alpha"),
        (1, "Beta"),
        (2, "Alpha"),
        (2, "Beta"),
        (3, "Alpha"),
        (3, "Beta"),
    ]
    actual_tuples = list(zip(result.measurements["current_label"], result.measurements["marker_name"]))
    assert actual_tuples == expected_tuples


@pytest.mark.parametrize(
    "corrupt_fn, error_match",
    [
        (
            # Missing label in object table (label 3 in array not in objects table)
            lambda res: ReviewedSegmentationResult(
                labels=res.labels,
                objects=res.objects.iloc[:2].copy(),
                source_result=res.source_result,
                history=res.history,
            ),
            "Mismatch between object table current_label set and positive labels",
        ),
        (
            # Unmapped positive pixel in label array (label 99 not in objects table)
            lambda res: ReviewedSegmentationResult(
                labels=np.where(res.labels == 1, 99, res.labels),
                objects=res.objects.copy(),
                source_result=res.source_result,
                history=res.history,
            ),
            "Active positive labels in reviewed_result.labels must be contiguous 1..N",
        ),
        (
            # Duplicate current_label in object table
            lambda res: ReviewedSegmentationResult(
                labels=res.labels,
                objects=pd.concat([res.objects, res.objects.iloc[:1]], ignore_index=True),
                source_result=res.source_result,
                history=res.history,
            ),
            "Duplicate current_label values",
        ),
        (
            # Duplicate review_id in object table
            lambda res: ReviewedSegmentationResult(
                labels=res.labels,
                objects=res.objects.assign(
                    review_id=["AUTO-000001", "AUTO-000001", "AUTO-000003"]
                ),
                source_result=res.source_result,
                history=res.history,
            ),
            "Duplicate review_id values",
        ),
        (
            # Pixel count mismatch against area_px
            lambda res: ReviewedSegmentationResult(
                labels=res.labels,
                objects=res.objects.assign(
                    area_px=[9999, res.objects.iloc[1]["area_px"], res.objects.iloc[2]["area_px"]]
                ),
                source_result=res.source_result,
                history=res.history,
            ),
            "Pixel count in labels",
        ),
    ],
)
def test_measurement_consistency_errors(corrupt_fn, error_match: str) -> None:
    """Test that structural inconsistencies in ReviewedSegmentationResult raise MeasurementConsistencyError."""
    clean_res = _create_synthetic_reviewed_result()
    corrupt_res = corrupt_fn(clean_res)
    marker_img = np.zeros((100, 100), dtype=np.float32)

    with pytest.raises(MeasurementConsistencyError, match=error_match):
        validate_reviewed_result_consistency(corrupt_res)

    with pytest.raises(MeasurementConsistencyError, match=error_match):
        measure_marker(corrupt_res, marker_img, "SOX10")
