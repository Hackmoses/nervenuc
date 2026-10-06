"""Tests for marker positivity thresholding and nucleus phenotype classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nervenuc.classification import (
    ClassificationConsistencyError,
    ClassificationResult,
    InvalidThresholdDefinitionError,
    MarkerThreshold,
    NerveNucClassificationError,
    UnknownMarkerError,
    classify_marker,
    classify_markers,
    validate_classification_consistency,
)
from nervenuc.markers import MarkerDefinition
from nervenuc.measurements import (
    MarkerMeasurementResult,
    measure_marker,
    measure_markers,
)
from nervenuc.qc import ReviewedSegmentationResult, start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _finalize_review(segmentation: SegmentationResult) -> ReviewedSegmentationResult:
    session = start_review(segmentation)
    for review_id in session.result().objects["review_id"]:
        session.accept(review_id)
    return session.finalize()

EXPECTED_CLASSIFICATION_COLUMNS = [
    "review_id",
    "current_label",
    "marker_name",
    "measurement_name",
    "measurement_value",
    "threshold_value",
    "comparison",
    "is_positive",
    "source_nucleus_id",
    "review_origin",
    "review_status",
]

EXPECTED_PHENOTYPE_COLUMNS = [
    "review_id",
    "current_label",
    "source_nucleus_id",
    "review_origin",
    "review_status",
    "phenotype",
]


def _create_synthetic_measurement_result() -> MarkerMeasurementResult:
    """Create a synthetic MarkerMeasurementResult with 3 active nuclei and 2 markers."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:20, 10:20] = 1  # 100 px
    labels[10:20, 50:60] = 2  # 100 px
    labels[50:60, 30:40] = 3  # 100 px

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
            },
            {
                "nucleus_id": 2,
                "area_px": 100,
                "centroid_y": 14.5,
                "centroid_x": 54.5,
                "bbox_min_y": 10,
                "bbox_min_x": 50,
                "bbox_max_y": 20,
                "bbox_max_x": 60,
                "touches_border": False,
                "qc_status": "accepted",
            },
            {
                "nucleus_id": 3,
                "area_px": 100,
                "centroid_y": 54.5,
                "centroid_x": 34.5,
                "bbox_min_y": 50,
                "bbox_min_x": 30,
                "bbox_max_y": 60,
                "bbox_max_x": 40,
                "touches_border": False,
                "qc_status": "accepted",
            },
        ]
    )

    seg_res = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg_res)

    # Marker 1 (BrdU): Nucleus 1=200.0, Nucleus 2=50.0, Nucleus 3=120.0
    brdu_img = np.zeros((100, 100), dtype=np.float32)
    brdu_img[10:20, 10:20] = 200.0
    brdu_img[10:20, 50:60] = 50.0
    brdu_img[50:60, 30:40] = 120.0

    # Marker 2 (SOX10): Nucleus 1=800.0, Nucleus 2=900.0, Nucleus 3=100.0
    sox10_img = np.zeros((100, 100), dtype=np.float32)
    sox10_img[10:20, 10:20] = 800.0
    sox10_img[10:20, 50:60] = 900.0
    sox10_img[50:60, 30:40] = 100.0

    return measure_markers(reviewed_res, {"BrdU": brdu_img, "SOX10": sox10_img})


def test_one_nucleus_above_threshold() -> None:
    """Test a single nucleus with measurement above threshold."""
    meas_res = _create_synthetic_measurement_result()
    # Filter down to nucleus 1
    t = MarkerThreshold(marker_name="BrdU", threshold=100.0, comparison=">=")

    res = classify_marker(meas_res, t)
    assert isinstance(res, ClassificationResult)

    row1 = res.classifications[res.classifications["current_label"] == 1].iloc[0]
    assert bool(row1["is_positive"]) is True
    assert row1["measurement_value"] == 200.0
    assert row1["threshold_value"] == 100.0

    pheno1 = res.phenotypes[res.phenotypes["current_label"] == 1].iloc[0]
    assert pheno1["phenotype"] == "BrdU+"


def test_one_nucleus_below_threshold() -> None:
    """Test a single nucleus with measurement below threshold."""
    meas_res = _create_synthetic_measurement_result()
    t = MarkerThreshold(marker_name="BrdU", threshold=100.0, comparison=">=")

    res = classify_marker(meas_res, t)

    row2 = res.classifications[res.classifications["current_label"] == 2].iloc[0]
    assert bool(row2["is_positive"]) is False
    assert row2["measurement_value"] == 50.0

    pheno2 = res.phenotypes[res.phenotypes["current_label"] == 2].iloc[0]
    assert pheno2["phenotype"] == "BrdU-"


def test_equality_comparison_gte() -> None:
    """Test boundary equality with comparison '>=' (100.0 >= 100.0 -> True)."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    marker_img = np.full((20, 20), 100.0, dtype=np.float32)
    meas = measure_marker(reviewed_res, marker_img, "TestM")

    t = MarkerThreshold(marker_name="TestM", threshold=100.0, comparison=">=")
    res = classify_marker(meas, t)

    assert bool(res.classifications.iloc[0]["is_positive"]) is True
    assert res.phenotypes.iloc[0]["phenotype"] == "TestM+"


def test_equality_comparison_gt() -> None:
    """Test boundary equality with comparison '>' (100.0 > 100.0 -> False)."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    marker_img = np.full((20, 20), 100.0, dtype=np.float32)
    meas = measure_marker(reviewed_res, marker_img, "TestM")

    t = MarkerThreshold(marker_name="TestM", threshold=100.0, comparison=">")
    res = classify_marker(meas, t)

    assert bool(res.classifications.iloc[0]["is_positive"]) is False
    assert res.phenotypes.iloc[0]["phenotype"] == "TestM-"


def test_multiple_nuclei_one_marker() -> None:
    """Test classifying multiple nuclei on a single marker."""
    meas_res = _create_synthetic_measurement_result()
    t = MarkerThreshold(marker_name="BrdU", threshold=100.0, comparison=">=")

    res = classify_marker(meas_res, t)
    assert len(res.classifications) == 3
    assert len(res.phenotypes) == 3
    assert list(res.classifications["is_positive"]) == [True, False, True]
    assert list(res.phenotypes["phenotype"]) == ["BrdU+", "BrdU-", "BrdU+"]


def test_one_nucleus_two_markers() -> None:
    """Test classifying two markers on active nuclei."""
    meas_res = _create_synthetic_measurement_result()
    thresholds = [
        MarkerThreshold(marker_name="BrdU", threshold=100.0),
        MarkerThreshold(marker_name="SOX10", threshold=500.0),
    ]

    res = classify_markers(meas_res, thresholds)

    # 3 nuclei x 2 thresholds = 6 classification rows, 3 phenotype rows
    assert len(res.classifications) == 6
    assert len(res.phenotypes) == 3

    # Nucleus 1: BrdU=200 (+), SOX10=800 (+) -> BrdU+ / SOX10+
    # Nucleus 2: BrdU=50 (-), SOX10=900 (+) -> BrdU- / SOX10+
    # Nucleus 3: BrdU=120 (+), SOX10=100 (-) -> BrdU+ / SOX10-
    expected_phenos = [
        "BrdU+ / SOX10+",
        "BrdU- / SOX10+",
        "BrdU+ / SOX10-",
    ]
    assert list(res.phenotypes["phenotype"]) == expected_phenos


def test_multiple_nuclei_multiple_markers() -> None:
    """Test classifying 3 distinct markers across 3 nuclei."""
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[5:10, 5:10] = 1
    labels[5:10, 20:25] = 2
    objects_df = pd.DataFrame(
        [
            {"nucleus_id": 1, "area_px": 25, "centroid_y": 7.0, "centroid_x": 7.0, "bbox_min_y": 5, "bbox_min_x": 5, "bbox_max_y": 10, "bbox_max_x": 10, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 2, "area_px": 25, "centroid_y": 7.0, "centroid_x": 22.0, "bbox_min_y": 5, "bbox_min_x": 20, "bbox_max_y": 10, "bbox_max_x": 25, "touches_border": False, "qc_status": "accepted"},
        ]
    )
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    imgA = np.zeros((50, 50), dtype=np.float32)
    imgA[5:10, 5:10] = 100.0
    imgB = np.zeros((50, 50), dtype=np.float32)
    imgB[5:10, 20:25] = 200.0
    imgC = np.zeros((50, 50), dtype=np.float32)
    imgC[5:10, 5:10] = 300.0
    imgC[5:10, 20:25] = 300.0

    meas = measure_markers(reviewed_res, {"A": imgA, "B": imgB, "C": imgC})

    thresholds = [
        MarkerThreshold("A", 50.0),
        MarkerThreshold("B", 50.0),
        MarkerThreshold("C", 50.0),
    ]
    res = classify_markers(meas, thresholds)

    assert len(res.classifications) == 2 * 3
    assert len(res.phenotypes) == 2
    assert list(res.phenotypes["phenotype"]) == ["A+ / B- / C+", "A- / B+ / C+"]


def test_distinct_statistics_per_marker() -> None:
    """Test classifying multiple markers using distinct measurement statistics."""
    meas_res = _create_synthetic_measurement_result()

    thresholds = [
        MarkerThreshold(marker_name="BrdU", threshold=100.0, measurement="mean_intensity"),
        MarkerThreshold(marker_name="SOX10", threshold=80000.0, measurement="integrated_intensity"),
    ]

    res = classify_markers(meas_res, thresholds)
    assert len(res.classifications) == 6

    brdu_rows = res.classifications[res.classifications["marker_name"] == "BrdU"]
    assert (brdu_rows["measurement_name"] == "mean_intensity").all()

    sox10_rows = res.classifications[res.classifications["marker_name"] == "SOX10"]
    assert (sox10_rows["measurement_name"] == "integrated_intensity").all()
    # Nucleus 1: integrated = 100 * 800 = 80000.0 >= 80000.0 -> True
    # Nucleus 2: integrated = 100 * 900 = 90000.0 >= 80000.0 -> True
    # Nucleus 3: integrated = 100 * 100 = 10000.0 < 80000.0 -> False
    assert list(sox10_rows["is_positive"]) == [True, True, False]


def test_exact_marker_name_preservation() -> None:
    """Test that custom marker names and case are preserved in outputs."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    img = np.full((20, 20), 100.0, dtype=np.float32)
    meas = measure_markers(reviewed_res, {"Marker 1": img, "BrdU_Ch2": img})

    thresholds = [
        MarkerThreshold("Marker 1", 50.0),
        MarkerThreshold("BrdU_Ch2", 50.0),
    ]
    res = classify_markers(meas, thresholds)

    assert list(res.classifications["marker_name"]) == ["Marker 1", "BrdU_Ch2"]
    assert res.phenotypes.iloc[0]["phenotype"] == "Marker 1+ / BrdU_Ch2+"


def test_case_sensitive_marker_names() -> None:
    """Test that SOX10 and sox10 are treated as distinct case-sensitive markers."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    img1 = np.full((20, 20), 200.0, dtype=np.float32)
    img2 = np.full((20, 20), 50.0, dtype=np.float32)
    meas = measure_markers(reviewed_res, {"SOX10": img1, "sox10": img2})

    thresholds = [
        MarkerThreshold("SOX10", 100.0),
        MarkerThreshold("sox10", 100.0),
    ]
    res = classify_markers(meas, thresholds)

    assert res.phenotypes.iloc[0]["phenotype"] == "SOX10+ / sox10-"


def test_empty_thresholds_rejected() -> None:
    """Test that an empty thresholds sequence raises InvalidThresholdDefinitionError."""
    meas_res = _create_synthetic_measurement_result()

    with pytest.raises(InvalidThresholdDefinitionError, match="At least one MarkerThreshold"):
        classify_markers(meas_res, [])


def test_unknown_marker_raises_error() -> None:
    """Test that specifying an unknown marker raises UnknownMarkerError."""
    meas_res = _create_synthetic_measurement_result()
    t = MarkerThreshold(marker_name="NonExistent", threshold=100.0)

    with pytest.raises(UnknownMarkerError, match="not found in measurement result"):
        classify_marker(meas_res, t)


def test_duplicate_threshold_definition_rejected() -> None:
    """Test that duplicate threshold definitions for the same marker raise InvalidThresholdDefinitionError."""
    meas_res = _create_synthetic_measurement_result()
    thresholds = [
        MarkerThreshold(marker_name="BrdU", threshold=100.0),
        MarkerThreshold(marker_name="BrdU", threshold=200.0),
    ]

    with pytest.raises(InvalidThresholdDefinitionError, match="Duplicate threshold definition"):
        classify_markers(meas_res, thresholds)


def test_invalid_threshold_numeric_value_rejected() -> None:
    """Test rejection of non-numeric, bool, NaN, and Inf thresholds."""
    with pytest.raises(InvalidThresholdDefinitionError, match="real numeric"):
        MarkerThreshold("M", True)  # type: ignore[arg-type]

    with pytest.raises(InvalidThresholdDefinitionError, match="finite"):
        MarkerThreshold("M", float("nan"))

    with pytest.raises(InvalidThresholdDefinitionError, match="finite"):
        MarkerThreshold("M", float("inf"))

    with pytest.raises(InvalidThresholdDefinitionError, match="real numeric"):
        MarkerThreshold("M", "100")  # type: ignore[arg-type]


def test_unsupported_measurement_statistic_rejected() -> None:
    """Test rejection of unsupported measurement statistic names."""
    with pytest.raises(InvalidThresholdDefinitionError, match="measurement must be one of"):
        MarkerThreshold("M", 100.0, measurement="pixel_count")

    with pytest.raises(InvalidThresholdDefinitionError, match="measurement must be one of"):
        MarkerThreshold("M", 100.0, measurement="invalid_stat")


def test_exact_measurement_value_retained() -> None:
    """Test that raw measurement values from M4 are retained without modification."""
    meas_res = _create_synthetic_measurement_result()
    t = MarkerThreshold("BrdU", 100.0)

    res = classify_marker(meas_res, t)
    for _, row in res.classifications.iterrows():
        c_lbl = row["current_label"]
        raw_val = meas_res.measurements[
            (meas_res.measurements["current_label"] == c_lbl)
            & (meas_res.measurements["marker_name"] == "BrdU")
        ]["mean_intensity"].iloc[0]
        assert row["measurement_value"] == raw_val


def test_source_measurement_result_immutability() -> None:
    """Test that M4 measurement result is not modified during classification."""
    meas_res = _create_synthetic_measurement_result()
    meas_copy = meas_res.measurements.copy(deep=True)

    t = MarkerThreshold("BrdU", 100.0)
    res = classify_marker(meas_res, t)

    assert meas_res.measurements.equals(meas_copy)
    assert res.classifications is not meas_res.measurements


def test_deterministic_row_ordering() -> None:
    """Test deterministic row ordering in classification and phenotype tables."""
    meas_res = _create_synthetic_measurement_result()
    thresholds = [
        MarkerThreshold("BrdU", 100.0),
        MarkerThreshold("SOX10", 500.0),
    ]

    res = classify_markers(meas_res, thresholds)

    # Classifications: (1, BrdU), (1, SOX10), (2, BrdU), (2, SOX10), (3, BrdU), (3, SOX10)
    expected_class_pairs = [
        (1, "BrdU"),
        (1, "SOX10"),
        (2, "BrdU"),
        (2, "SOX10"),
        (3, "BrdU"),
        (3, "SOX10"),
    ]
    actual_class_pairs = list(zip(res.classifications["current_label"], res.classifications["marker_name"]))
    assert actual_class_pairs == expected_class_pairs

    # Phenotypes: 1, 2, 3
    assert list(res.phenotypes["current_label"]) == [1, 2, 3]


def test_phenotype_string_one_marker() -> None:
    """Test single-marker phenotype string formatting."""
    meas_res = _create_synthetic_measurement_result()
    res_pos = classify_marker(meas_res, MarkerThreshold("BrdU", 10.0))
    assert res_pos.phenotypes.iloc[0]["phenotype"] == "BrdU+"

    res_neg = classify_marker(meas_res, MarkerThreshold("BrdU", 1000.0))
    assert res_neg.phenotypes.iloc[0]["phenotype"] == "BrdU-"


def test_phenotype_string_two_markers() -> None:
    """Test two-marker phenotype string formatting."""
    meas_res = _create_synthetic_measurement_result()
    res = classify_markers(
        meas_res,
        [MarkerThreshold("BrdU", 100.0), MarkerThreshold("SOX10", 500.0)],
    )
    assert res.phenotypes.iloc[0]["phenotype"] == "BrdU+ / SOX10+"


def test_phenotype_string_three_markers() -> None:
    """Test three-marker phenotype string formatting."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    imgA = np.full((20, 20), 100.0, dtype=np.float32)
    imgB = np.full((20, 20), 10.0, dtype=np.float32)
    imgC = np.full((20, 20), 100.0, dtype=np.float32)
    meas = measure_markers(reviewed_res, {"A": imgA, "B": imgB, "C": imgC})

    res = classify_markers(meas, [MarkerThreshold("A", 50.0), MarkerThreshold("B", 50.0), MarkerThreshold("C", 50.0)])
    assert res.phenotypes.iloc[0]["phenotype"] == "A+ / B- / C+"


def test_phenotype_order_follows_threshold_order() -> None:
    """Test that phenotype component order strictly follows threshold input order."""
    meas_res = _create_synthetic_measurement_result()

    res1 = classify_markers(
        meas_res,
        [MarkerThreshold("BrdU", 100.0), MarkerThreshold("SOX10", 500.0)],
    )
    assert res1.phenotypes.iloc[0]["phenotype"] == "BrdU+ / SOX10+"

    res2 = classify_markers(
        meas_res,
        [MarkerThreshold("SOX10", 500.0), MarkerThreshold("BrdU", 100.0)],
    )
    assert res2.phenotypes.iloc[0]["phenotype"] == "SOX10+ / BrdU+"


def test_classification_row_count_contract() -> None:
    """Test that row count strictly equals active nuclei x thresholds."""
    meas_res = _create_synthetic_measurement_result()
    thresholds = [
        MarkerThreshold("BrdU", 100.0),
        MarkerThreshold("SOX10", 500.0),
    ]

    res = classify_markers(meas_res, thresholds)
    assert len(res.classifications) == 3 * 2
    assert len(res.phenotypes) == 3


@pytest.mark.parametrize(
    "corrupt_fn, error_match",
    [
        (
            # Missing measurement row for active nucleus 3
            lambda meas: MarkerMeasurementResult(
                measurements=meas.measurements[meas.measurements["current_label"] != 3].copy(),
                markers=meas.markers,
                reviewed_result=meas.reviewed_result,
            ),
            "Expected 3 measurement rows",
        ),
        (
            # Duplicate measurement rows
            lambda meas: MarkerMeasurementResult(
                measurements=pd.concat([meas.measurements, meas.measurements.iloc[:1]], ignore_index=True),
                markers=meas.markers,
                reviewed_result=meas.reviewed_result,
            ),
            "Expected 3 measurement rows",
        ),
        (
            # Provenance mismatch on review_status
            lambda meas: MarkerMeasurementResult(
                measurements=meas.measurements.assign(
                    review_status=["edited", "unreviewed", "unreviewed", "unreviewed", "unreviewed", "unreviewed"]
                ),
                markers=meas.markers,
                reviewed_result=meas.reviewed_result,
            ),
            "Provenance mismatch",
        ),
        (
            # Non-finite value in measurement column
            lambda meas: MarkerMeasurementResult(
                measurements=meas.measurements.assign(
                    mean_intensity=[np.nan, 50.0, 120.0, 800.0, 900.0, 100.0]
                ),
                markers=meas.markers,
                reviewed_result=meas.reviewed_result,
            ),
            "Non-finite measurement value",
        ),
    ],
)
def test_classification_consistency_errors(corrupt_fn, error_match: str) -> None:
    """Test that consistency discrepancies raise ClassificationConsistencyError."""
    clean_meas = _create_synthetic_measurement_result()
    corrupt_meas = corrupt_fn(clean_meas)

    with pytest.raises(ClassificationConsistencyError, match=error_match):
        classify_marker(corrupt_meas, MarkerThreshold("BrdU", 100.0))


def test_empty_measurement_result_handling() -> None:
    """Test that zero active nuclei return valid empty classification and phenotype tables."""
    empty_labels = np.zeros((50, 50), dtype=np.int32)
    empty_objects = pd.DataFrame(
        columns=[
            "review_id", "current_label", "source_nucleus_id", "review_status",
            "review_origin", "parent_review_ids", "review_note", "area_px",
            "centroid_y", "centroid_x", "bbox_min_y", "bbox_min_x",
            "bbox_max_y", "bbox_max_x", "touches_border", "qc_status",
        ]
    )
    empty_seg = SegmentationResult(labels=empty_labels, objects=empty_objects, config=SegmentationConfig())
    empty_reviewed = _finalize_review(empty_seg)

    empty_meas = measure_marker(empty_reviewed, np.zeros((50, 50), dtype=np.float32), "BrdU")

    t = MarkerThreshold("BrdU", 100.0)
    res = classify_marker(empty_meas, t)

    assert len(res.classifications) == 0
    assert list(res.classifications.columns) == EXPECTED_CLASSIFICATION_COLUMNS
    assert len(res.phenotypes) == 0
    assert list(res.phenotypes.columns) == EXPECTED_PHENOTYPE_COLUMNS
    assert res.thresholds == (t,)
