"""Tests for denominator-aware biological metrics and phenotype summaries."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from nervenuc.classification import (
    ClassificationResult,
    MarkerThreshold,
    classify_marker,
    classify_markers,
)
from nervenuc.measurements import measure_marker, measure_markers
from nervenuc.qc import ReviewedSegmentationResult, start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _finalize_review(segmentation: SegmentationResult) -> ReviewedSegmentationResult:
    session = start_review(segmentation)
    for review_id in session.result().objects["review_id"]:
        session.accept(review_id)
    return session.finalize()
from nervenuc.summary import (
    BiologicalMetricDefinition,
    BiologicalSummaryResult,
    InvalidMetricDefinitionError,
    NerveNucSummaryError,
    PhenotypeFilter,
    SummaryConsistencyError,
    UnknownMetricMarkerError,
    summarize_metric,
    summarize_metrics,
    validate_summary_consistency,
)

EXPECTED_SUMMARY_COLUMNS = [
    "metric_name",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]


def _create_synthetic_classification_result() -> ClassificationResult:
    """Create a synthetic 4-nucleus classification result for testing.

    Nuclei:
    - Label 1: BrdU+, SOX10+
    - Label 2: BrdU+, SOX10-
    - Label 3: BrdU-, SOX10+
    - Label 4: BrdU-, SOX10-
    """
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[2:4, 12:14] = 2
    labels[12:14, 2:4] = 3
    labels[12:14, 12:14] = 4

    objects_df = pd.DataFrame(
        [
            {"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 2, "area_px": 4, "centroid_y": 2.5, "centroid_x": 12.5, "bbox_min_y": 2, "bbox_min_x": 12, "bbox_max_y": 4, "bbox_max_x": 14, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 3, "area_px": 4, "centroid_y": 12.5, "centroid_x": 2.5, "bbox_min_y": 12, "bbox_min_x": 2, "bbox_max_y": 14, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 4, "area_px": 4, "centroid_y": 12.5, "centroid_x": 12.5, "bbox_min_y": 12, "bbox_min_x": 12, "bbox_max_y": 14, "bbox_max_x": 14, "touches_border": False, "qc_status": "accepted"},
        ]
    )

    seg_res = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg_res)

    brdu_img = np.zeros((40, 40), dtype=np.float32)
    brdu_img[2:4, 2:4] = 200.0   # Label 1 (+)
    brdu_img[2:4, 12:14] = 200.0 # Label 2 (+)
    brdu_img[12:14, 2:4] = 20.0  # Label 3 (-)
    brdu_img[12:14, 12:14] = 20.0# Label 4 (-)

    sox10_img = np.zeros((40, 40), dtype=np.float32)
    sox10_img[2:4, 2:4] = 500.0   # Label 1 (+)
    sox10_img[2:4, 12:14] = 50.0  # Label 2 (-)
    sox10_img[12:14, 2:4] = 500.0 # Label 3 (+)
    sox10_img[12:14, 12:14] = 50.0# Label 4 (-)

    meas_res = measure_markers(reviewed_res, {"BrdU": brdu_img, "SOX10": sox10_img})

    thresholds = [
        MarkerThreshold(marker_name="BrdU", threshold=100.0),
        MarkerThreshold(marker_name="SOX10", threshold=200.0),
    ]

    return classify_markers(meas_res, thresholds)


def test_empty_phenotype_filter_represents_all() -> None:
    """Test that PhenotypeFilter() and PhenotypeFilter({}) are valid and mean ALL."""
    f1 = PhenotypeFilter()
    f2 = PhenotypeFilter({})

    assert f1.states == {}
    assert f2.states == {}
    assert f1.to_string() == "ALL"
    assert f2.to_string() == "ALL"
    assert f1 == f2
    assert hash(f1) == hash(f2)


def test_all_nuclei_denominator() -> None:
    """Test evaluating a metric with ALL active nuclei denominator."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="Overall BrdU positivity",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)

    assert isinstance(res, BiologicalSummaryResult)
    assert len(res.metrics) == 1
    assert list(res.metrics.columns) == EXPECTED_SUMMARY_COLUMNS

    row = res.metrics.iloc[0]
    assert row["metric_name"] == "Overall BrdU positivity"
    assert row["numerator_count"] == 2
    assert row["denominator_count"] == 4
    assert row["percentage"] == 50.0
    assert row["numerator_filter"] == "BrdU+"
    assert row["denominator_filter"] == "ALL"


def test_one_positive_marker_numerator() -> None:
    """Test basic single marker numerator calculation."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="SOX10 fraction",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"SOX10": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["numerator_count"] == 2
    assert row["denominator_count"] == 4
    assert row["percentage"] == 50.0
    assert row["numerator_filter"] == "SOX10+"
    assert row["denominator_filter"] == "ALL"


def test_negative_marker_denominator() -> None:
    """Test filtering on marker-negative subpopulation denominator."""
    class_res = _create_synthetic_classification_result()

    # BrdU+ among SOX10-
    metric = BiologicalMetricDefinition(
        name="BrdU+ among SOX10-",
        denominator=PhenotypeFilter({"SOX10": False}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    # SOX10- nuclei are Label 2 (BrdU+) and Label 4 (BrdU-) -> den = 2, num = 1
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 2
    assert row["percentage"] == 50.0
    assert row["numerator_filter"] == "BrdU+ & SOX10-"
    assert row["denominator_filter"] == "SOX10-"


def test_numerator_evaluated_within_denominator() -> None:
    """Test that numerator conditions are evaluated within denominator population."""
    class_res = _create_synthetic_classification_result()

    # BrdU+ among SOX10+
    metric = BiologicalMetricDefinition(
        name="BrdU+ among SOX10+",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    # SOX10+ nuclei are Label 1 (BrdU+) and Label 3 (BrdU-) -> den = 2, num = 1
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 2
    assert row["percentage"] == 50.0
    assert row["numerator_filter"] == "BrdU+ & SOX10+"
    assert row["denominator_filter"] == "SOX10+"


def test_effective_numerator_filter_string() -> None:
    """Test that numerator_filter displays the effective combined population string."""
    metric = BiologicalMetricDefinition(
        name="SOX10+ dividing",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    assert metric.effective_numerator.to_string() == "BrdU+ & SOX10+"


def test_two_marker_conditional_metric() -> None:
    """Test two distinct conditional metrics in one call."""
    class_res = _create_synthetic_classification_result()

    metrics = [
        BiologicalMetricDefinition(
            name="SOX10+ dividing",
            denominator=PhenotypeFilter({"SOX10": True}),
            numerator=PhenotypeFilter({"BrdU": True}),
        ),
        BiologicalMetricDefinition(
            name="SOX10- dividing",
            denominator=PhenotypeFilter({"SOX10": False}),
            numerator=PhenotypeFilter({"BrdU": True}),
        ),
    ]

    res = summarize_metrics(class_res, metrics)
    assert len(res.metrics) == 2
    assert list(res.metrics["metric_name"]) == ["SOX10+ dividing", "SOX10- dividing"]
    assert list(res.metrics["numerator_count"]) == [1, 1]
    assert list(res.metrics["denominator_count"]) == [2, 2]
    assert list(res.metrics["percentage"]) == [50.0, 50.0]


def test_three_marker_filter() -> None:
    """Test filter with three simultaneous marker conditions."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    imgA = np.full((30, 30), 200.0, dtype=np.float32)
    imgB = np.full((30, 30), 50.0, dtype=np.float32)
    imgC = np.full((30, 30), 300.0, dtype=np.float32)

    meas = measure_markers(reviewed_res, {"A": imgA, "B": imgB, "C": imgC})
    thresholds = [MarkerThreshold("A", 100.0), MarkerThreshold("B", 100.0), MarkerThreshold("C", 100.0)]
    class_res = classify_markers(meas, thresholds)

    metric = BiologicalMetricDefinition(
        name="Triple state",
        denominator=PhenotypeFilter({"B": False}),
        numerator=PhenotypeFilter({"A": True, "C": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 1
    assert row["percentage"] == 100.0
    assert row["numerator_filter"] == "A+ & B- & C+"
    assert row["denominator_filter"] == "B-"


def test_numerator_and_denominator_counts_exact() -> None:
    """Test that integer counts strictly match manual nucleus tallies."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="All BrdU+",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)
    assert int(res.metrics.iloc[0]["numerator_count"]) == 2
    assert int(res.metrics.iloc[0]["denominator_count"]) == 4


def test_percentage_exact_math() -> None:
    """Test exact percentage calculation math."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="BrdU+ fraction",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["percentage"] == (100.0 * row["numerator_count"] / row["denominator_count"])


def test_unrounded_float_precision() -> None:
    """Test that 1/3 ratio yields unrounded float64 precision (not 33.3)."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[2:4, 10:12] = 2
    labels[10:12, 2:4] = 3
    objects_df = pd.DataFrame(
        [
            {"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 2, "area_px": 4, "centroid_y": 2.5, "centroid_x": 10.5, "bbox_min_y": 2, "bbox_min_x": 10, "bbox_max_y": 4, "bbox_max_x": 12, "touches_border": False, "qc_status": "accepted"},
            {"nucleus_id": 3, "area_px": 4, "centroid_y": 10.5, "centroid_x": 2.5, "bbox_min_y": 10, "bbox_min_x": 2, "bbox_max_y": 12, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"},
        ]
    )
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    img = np.zeros((30, 30), dtype=np.float32)
    img[2:4, 2:4] = 200.0  # Only nucleus 1 is positive

    meas = measure_marker(reviewed_res, img, "MarkerA")
    class_res = classify_marker(meas, MarkerThreshold("MarkerA", 100.0))

    metric = BiologicalMetricDefinition(
        name="One third",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"MarkerA": True}),
    )

    res = summarize_metric(class_res, metric)
    pct = res.metrics.iloc[0]["percentage"]

    # 100/3 = 33.333333333333336
    assert pct == pytest.approx(100.0 / 3.0, rel=1e-12)
    assert pct != 33.3
    assert pct != 33.33


def test_zero_denominator_yields_nan() -> None:
    """Test that zero denominator count produces numerator=0, denominator=0, percentage=NaN."""
    class_res = _create_synthetic_classification_result()

    # Synthetic data has no double-negative in denominator where we filter for impossible state
    # Create an impossible denominator filter for this dataset
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    # Nucleus 1 is M1+
    img = np.full((20, 20), 200.0, dtype=np.float32)
    meas = measure_marker(reviewed_res, img, "M1")
    class_res = classify_marker(meas, MarkerThreshold("M1", 100.0))

    # Denominator requires M1- (0 nuclei match)
    metric = BiologicalMetricDefinition(
        name="Zero denominator",
        denominator=PhenotypeFilter({"M1": False}),
        numerator=PhenotypeFilter(),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["numerator_count"] == 0
    assert row["denominator_count"] == 0
    assert math.isnan(row["percentage"])


def test_empty_active_nuclei_gives_zero_nan() -> None:
    """Test that zero active nuclei returns 0/0/NaN for valid metrics without crashing."""
    empty_labels = np.zeros((20, 20), dtype=np.int32)
    empty_objects = pd.DataFrame(
        columns=[
            "review_id", "current_label", "source_nucleus_id", "review_status",
            "review_origin", "parent_review_ids", "review_note", "area_px",
            "centroid_y", "centroid_x", "bbox_min_y", "bbox_min_x",
            "bbox_max_y", "bbox_max_x", "touches_border", "qc_status",
        ]
    )
    seg = SegmentationResult(labels=empty_labels, objects=empty_objects, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    meas = measure_marker(reviewed_res, np.zeros((20, 20), dtype=np.float32), "BrdU")
    class_res = classify_marker(meas, MarkerThreshold("BrdU", 100.0))

    metric = BiologicalMetricDefinition(
        name="Empty summary",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    res = summarize_metric(class_res, metric)
    assert len(res.metrics) == 1
    row = res.metrics.iloc[0]
    assert row["numerator_count"] == 0
    assert row["denominator_count"] == 0
    assert math.isnan(row["percentage"])


def test_markers_validated_against_thresholds_universe() -> None:
    """Test that markers are validated against classification_result.thresholds, rejecting unclassified markers."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="Unclassified marker",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"Ki67": True}),
    )

    with pytest.raises(UnknownMetricMarkerError, match="not declared in classification_result.thresholds"):
        summarize_metric(class_res, metric)


def test_duplicate_threshold_markers_rejected() -> None:
    """Test that duplicate marker names in classification_result.thresholds raise SummaryConsistencyError."""
    clean_res = _create_synthetic_classification_result()
    # Construct corrupt classification result with duplicate thresholds
    corrupt_thresholds = clean_res.thresholds + (MarkerThreshold("BrdU", 200.0),)
    corrupt_res = ClassificationResult(
        classifications=clean_res.classifications,
        phenotypes=clean_res.phenotypes,
        thresholds=corrupt_thresholds,
        measurement_result=clean_res.measurement_result,
    )

    metric = BiologicalMetricDefinition(
        name="Metric",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    with pytest.raises(SummaryConsistencyError, match="Duplicate marker names"):
        summarize_metric(corrupt_res, metric)


def test_arbitrary_marker_names_preserved() -> None:
    """Test that custom marker names (e.g. Marker 1, BrdU_Ch2) are preserved and evaluated accurately."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:4, 2:4] = 1
    objects_df = pd.DataFrame([{"nucleus_id": 1, "area_px": 4, "centroid_y": 2.5, "centroid_x": 2.5, "bbox_min_y": 2, "bbox_min_x": 2, "bbox_max_y": 4, "bbox_max_x": 4, "touches_border": False, "qc_status": "accepted"}])
    seg = SegmentationResult(labels=labels, objects=objects_df, config=SegmentationConfig())
    reviewed_res = _finalize_review(seg)

    img = np.full((20, 20), 200.0, dtype=np.float32)
    meas = measure_markers(reviewed_res, {"Marker 1": img, "BrdU_Ch2": img})
    class_res = classify_markers(meas, [MarkerThreshold("Marker 1", 100.0), MarkerThreshold("BrdU_Ch2", 100.0)])

    metric = BiologicalMetricDefinition(
        name="Custom markers metric",
        denominator=PhenotypeFilter({"Marker 1": True}),
        numerator=PhenotypeFilter({"BrdU_Ch2": True}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["numerator_filter"] == "BrdU_Ch2+ & Marker 1+"
    assert row["denominator_filter"] == "Marker 1+"
    assert row["percentage"] == 100.0


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
    class_res = classify_markers(meas, [MarkerThreshold("SOX10", 100.0), MarkerThreshold("sox10", 100.0)])

    metric = BiologicalMetricDefinition(
        name="Case sensitive test",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"sox10": False}),
    )

    res = summarize_metric(class_res, metric)
    row = res.metrics.iloc[0]
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 1
    assert row["percentage"] == 100.0


def test_unknown_marker_raises_error() -> None:
    """Test that a filter referencing an unknown marker raises UnknownMetricMarkerError."""
    class_res = _create_synthetic_classification_result()

    metric = BiologicalMetricDefinition(
        name="Invalid marker",
        denominator=PhenotypeFilter({"NonExistent": True}),
        numerator=PhenotypeFilter(),
    )

    with pytest.raises(UnknownMetricMarkerError, match="not declared in classification_result.thresholds"):
        summarize_metric(class_res, metric)


def test_duplicate_metric_names_rejected() -> None:
    """Test that duplicate metric names in summarize_metrics raise InvalidMetricDefinitionError."""
    class_res = _create_synthetic_classification_result()

    metrics = [
        BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True})),
        BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"SOX10": True})),
    ]

    with pytest.raises(InvalidMetricDefinitionError, match="Duplicate metric name"):
        summarize_metrics(class_res, metrics)


def test_empty_metric_sequence_rejected() -> None:
    """Test that passing an empty metric sequence raises InvalidMetricDefinitionError."""
    class_res = _create_synthetic_classification_result()

    with pytest.raises(InvalidMetricDefinitionError, match="At least one BiologicalMetricDefinition"):
        summarize_metrics(class_res, [])


def test_invalid_metric_name_rejected() -> None:
    """Test rejection of empty, whitespace, and leading/trailing padded metric names."""
    with pytest.raises(InvalidMetricDefinitionError, match="non-empty"):
        BiologicalMetricDefinition("", PhenotypeFilter(), PhenotypeFilter())

    with pytest.raises(InvalidMetricDefinitionError, match="non-empty"):
        BiologicalMetricDefinition("   ", PhenotypeFilter(), PhenotypeFilter())

    with pytest.raises(InvalidMetricDefinitionError, match="leading or trailing whitespace"):
        BiologicalMetricDefinition(" Metric", PhenotypeFilter(), PhenotypeFilter())

    with pytest.raises(InvalidMetricDefinitionError, match="leading or trailing whitespace"):
        BiologicalMetricDefinition("Metric ", PhenotypeFilter(), PhenotypeFilter())


def test_filter_immutability_post_creation() -> None:
    """Test that mutating the caller's input dictionary does not alter the PhenotypeFilter."""
    input_dict = {"SOX10": True}
    f = PhenotypeFilter(input_dict)

    # Mutate the source dictionary
    input_dict["SOX10"] = False
    input_dict["BrdU"] = True

    assert f.states == {"SOX10": True}
    assert f.to_string() == "SOX10+"


def test_deterministic_filter_serialization() -> None:
    """Test that input mapping insertion order does not affect alphabetical filter string serialization."""
    f1 = PhenotypeFilter({"SOX10": True, "BrdU": False})
    f2 = PhenotypeFilter({"BrdU": False, "SOX10": True})

    assert f1.to_string() == "BrdU- & SOX10+"
    assert f2.to_string() == "BrdU- & SOX10+"
    assert f1 == f2


def test_contradictory_filter_states_rejected() -> None:
    """Test that contradictory states between denominator and numerator raise InvalidMetricDefinitionError."""
    with pytest.raises(InvalidMetricDefinitionError, match="Contradictory state requirements"):
        BiologicalMetricDefinition(
            name="Contradiction",
            denominator=PhenotypeFilter({"SOX10": True}),
            numerator=PhenotypeFilter({"SOX10": False}),
        )


def test_classification_result_immutability() -> None:
    """Test that classification_result DataFrames are not modified by summary calculation."""
    class_res = _create_synthetic_classification_result()
    class_copy = class_res.classifications.copy(deep=True)
    pheno_copy = class_res.phenotypes.copy(deep=True)

    metric = BiologicalMetricDefinition("Metric", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    res = summarize_metric(class_res, metric)

    assert class_res.classifications.equals(class_copy)
    assert class_res.phenotypes.equals(pheno_copy)
    assert res.metrics is not class_res.classifications


def test_deterministic_metric_ordering() -> None:
    """Test that output table rows strictly match user-supplied metric definition order."""
    class_res = _create_synthetic_classification_result()

    metrics = [
        BiologicalMetricDefinition("Metric Z", PhenotypeFilter(), PhenotypeFilter({"BrdU": True})),
        BiologicalMetricDefinition("Metric A", PhenotypeFilter(), PhenotypeFilter({"SOX10": True})),
        BiologicalMetricDefinition("Metric M", PhenotypeFilter({"SOX10": True}), PhenotypeFilter({"BrdU": True})),
    ]

    res = summarize_metrics(class_res, metrics)
    assert list(res.metrics["metric_name"]) == ["Metric Z", "Metric A", "Metric M"]


def test_multiple_metrics_in_one_call() -> None:
    """Test evaluating 5 distinct metrics in a single summary call."""
    class_res = _create_synthetic_classification_result()

    metrics = [
        BiologicalMetricDefinition("Overall proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True})),
        BiologicalMetricDefinition("SOX10+ fraction", PhenotypeFilter(), PhenotypeFilter({"SOX10": True})),
        BiologicalMetricDefinition("SOX10+ dividing", PhenotypeFilter({"SOX10": True}), PhenotypeFilter({"BrdU": True})),
        BiologicalMetricDefinition("SOX10- dividing", PhenotypeFilter({"SOX10": False}), PhenotypeFilter({"BrdU": True})),
        BiologicalMetricDefinition("SOX10+ among BrdU+", PhenotypeFilter({"BrdU": True}), PhenotypeFilter({"SOX10": True})),
    ]

    res = summarize_metrics(class_res, metrics)
    assert len(res.metrics) == 5
    assert list(res.metrics["metric_name"]) == [
        "Overall proliferation",
        "SOX10+ fraction",
        "SOX10+ dividing",
        "SOX10- dividing",
        "SOX10+ among BrdU+",
    ]


@pytest.mark.parametrize(
    "corrupt_fn, error_match",
    [
        (
            # Missing classification row for active nucleus 4
            lambda c_res: ClassificationResult(
                classifications=c_res.classifications[c_res.classifications["current_label"] != 4].copy(),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "Expected 4 classification rows",
        ),
        (
            # Duplicate classification row
            lambda c_res: ClassificationResult(
                classifications=pd.concat([c_res.classifications, c_res.classifications.iloc[:1]], ignore_index=True),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "Expected 4 classification rows",
        ),
        (
            # Stale/unknown nucleus label not in reviewed objects
            lambda c_res: ClassificationResult(
                classifications=c_res.classifications.assign(
                    current_label=[99 if lbl == 1 else lbl for lbl in c_res.classifications["current_label"]]
                ),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "do not match the authoritative active nuclei set",
        ),
        (
            # Provenance mismatch on review_status
            lambda c_res: ClassificationResult(
                classifications=c_res.classifications.assign(
                    review_status=["edited", "unreviewed", "unreviewed", "unreviewed", "unreviewed", "unreviewed", "unreviewed", "unreviewed"]
                ),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "Provenance mismatch",
        ),
        (
            # Integer 1 instead of bool
            lambda c_res: ClassificationResult(
                classifications=c_res.classifications.assign(
                    is_positive=[1, True, False, False, True, False, True, False]
                ),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "Non-boolean is_positive value",
        ),
        (
            # Float 0.0 instead of bool
            lambda c_res: ClassificationResult(
                classifications=c_res.classifications.assign(
                    is_positive=[0.0, True, False, False, True, False, True, False]
                ),
                phenotypes=c_res.phenotypes,
                thresholds=c_res.thresholds,
                measurement_result=c_res.measurement_result,
            ),
            "Non-boolean is_positive value",
        ),
    ],
)
def test_summary_consistency_errors(corrupt_fn, error_match: str) -> None:
    """Test that structural and metadata inconsistencies raise SummaryConsistencyError."""
    clean_res = _create_synthetic_classification_result()
    corrupt_res = corrupt_fn(clean_res)

    metric = BiologicalMetricDefinition("Metric", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))

    with pytest.raises(SummaryConsistencyError, match=error_match):
        validate_summary_consistency(corrupt_res, [metric])

    with pytest.raises(SummaryConsistencyError, match=error_match):
        summarize_metric(corrupt_res, metric)


def test_single_metric_convenience_function() -> None:
    """Test that summarize_metric produces identical results to summarize_metrics with a 1-item list."""
    class_res = _create_synthetic_classification_result()
    metric = BiologicalMetricDefinition("Metric", PhenotypeFilter({"SOX10": True}), PhenotypeFilter({"BrdU": True}))

    res_single = summarize_metric(class_res, metric)
    res_multi = summarize_metrics(class_res, [metric])

    assert res_single.metrics.equals(res_multi.metrics)
    assert res_single.definitions == res_multi.definitions
