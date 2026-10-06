"""Tests for image-to-animal aggregation and biological replicate summaries."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from nervenuc.aggregation import (
    AggregationConsistencyError,
    AnimalAggregationResult,
    InvalidSampleIdentityError,
    SampleIdentity,
    SampleSummary,
    aggregate_by_animal,
    validate_aggregation_consistency,
)
from nervenuc.classification import (
    ClassificationResult,
    MarkerThreshold,
    classify_markers,
)
from nervenuc.measurements import measure_markers
from nervenuc.qc import start_review
from nervenuc.segmentation import SegmentationConfig, SegmentationResult
from nervenuc.summary import (
    BiologicalMetricDefinition,
    BiologicalSummaryResult,
    PhenotypeFilter,
    summarize_metrics,
)

EXPECTED_ANIMAL_COLUMNS = [
    "animal_id",
    "metric_name",
    "sample_count",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]


def _create_mock_summary_result(
    counts: list[tuple[int, int]],
    metric_defs: list[BiologicalMetricDefinition] | None = None,
) -> BiologicalSummaryResult:
    """Create a mock BiologicalSummaryResult for testing aggregation without running full upstream pipelines."""
    if metric_defs is None:
        metric_defs = [
            BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True})),
            BiologicalMetricDefinition("M2", PhenotypeFilter({"SOX10": True}), PhenotypeFilter({"BrdU": True})),
        ]

    rows = []
    for (num, den), d in zip(counts, metric_defs):
        pct = float(100.0 * num / den) if den > 0 else float("nan")
        rows.append(
            {
                "metric_name": d.name,
                "numerator_count": num,
                "denominator_count": den,
                "percentage": pct,
                "numerator_filter": d.effective_numerator.to_string(),
                "denominator_filter": d.denominator.to_string(),
            }
        )

    df = pd.DataFrame(rows)
    # Return mock result; classification_result can be None or dummy object since aggregation only checks definitions & metrics
    return BiologicalSummaryResult(
        metrics=df,
        definitions=tuple(metric_defs),
        classification_result=None,  # type: ignore[arg-type]
    )


def test_one_animal_one_image() -> None:
    """Test that a single image animal yields animal counts and percentage equal to the image."""
    summary = _create_mock_summary_result([(5, 10), (2, 4)])
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), summary)

    res = aggregate_by_animal([sample])

    assert isinstance(res, AnimalAggregationResult)
    assert len(res.animal_metrics) == 2
    assert list(res.animal_metrics.columns) == EXPECTED_ANIMAL_COLUMNS

    row1 = res.animal_metrics.iloc[0]
    assert row1["animal_id"] == "Animal_01"
    assert row1["metric_name"] == "M1"
    assert row1["sample_count"] == 1
    assert row1["numerator_count"] == 5
    assert row1["denominator_count"] == 10
    assert row1["percentage"] == 50.0

    row2 = res.animal_metrics.iloc[1]
    assert row2["animal_id"] == "Animal_01"
    assert row2["metric_name"] == "M2"
    assert row2["sample_count"] == 1
    assert row2["numerator_count"] == 2
    assert row2["denominator_count"] == 4
    assert row2["percentage"] == 50.0


def test_one_animal_multiple_images() -> None:
    """Test pooling multiple images for one animal."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(2, 5), (1, 2)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(3, 5), (2, 3)]))

    res = aggregate_by_animal([s1, s2])

    assert len(res.animal_metrics) == 2
    row1 = res.animal_metrics.iloc[0]
    assert row1["animal_id"] == "Animal_01"
    assert row1["sample_count"] == 2
    assert row1["numerator_count"] == 5
    assert row1["denominator_count"] == 10
    assert row1["percentage"] == 50.0

    row2 = res.animal_metrics.iloc[1]
    assert row2["animal_id"] == "Animal_01"
    assert row2["sample_count"] == 2
    assert row2["numerator_count"] == 3
    assert row2["denominator_count"] == 5
    assert row2["percentage"] == 60.0


def test_multiple_animals() -> None:
    """Test aggregation across multiple distinct animals."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 10), (0, 5)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(3, 10), (2, 5)]))
    s3 = SampleSummary(SampleIdentity("img03", "Animal_02"), _create_mock_summary_result([(5, 20), (3, 10)]))

    res = aggregate_by_animal([s1, s2, s3])

    assert len(res.animal_metrics) == 4  # 2 animals x 2 metrics
    df = res.animal_metrics

    a1_m1 = df[(df["animal_id"] == "Animal_01") & (df["metric_name"] == "M1")].iloc[0]
    assert a1_m1["sample_count"] == 2
    assert a1_m1["numerator_count"] == 4
    assert a1_m1["denominator_count"] == 20
    assert a1_m1["percentage"] == 20.0

    a2_m1 = df[(df["animal_id"] == "Animal_02") & (df["metric_name"] == "M1")].iloc[0]
    assert a2_m1["sample_count"] == 1
    assert a2_m1["numerator_count"] == 5
    assert a2_m1["denominator_count"] == 20
    assert a2_m1["percentage"] == 25.0


def test_image_percentages_not_averaged_pseudoreplication_test() -> None:
    """Critical test: verify count-pooling (1/2 + 1/18 = 2/20 = 10%) vs invalid averaging ((50 + 5.56)/2 = 27.78%)."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(1, 18), (0, 0)]))

    res = aggregate_by_animal([s1, s2])
    row = res.animal_metrics[res.animal_metrics["metric_name"] == "M1"].iloc[0]

    assert row["numerator_count"] == 2
    assert row["denominator_count"] == 20
    assert row["percentage"] == 10.0
    assert row["percentage"] != pytest.approx((50.0 + (100.0 / 18.0)) / 2.0)


def test_zero_denominator_mixed_with_nonzero() -> None:
    """Test that a 0/0 image mixed with 3/10 image yields sample_count=2, num=3, den=10, pct=30.0."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(0, 0), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(3, 10), (0, 0)]))

    res = aggregate_by_animal([s1, s2])
    row = res.animal_metrics[res.animal_metrics["metric_name"] == "M1"].iloc[0]

    assert row["sample_count"] == 2
    assert row["numerator_count"] == 3
    assert row["denominator_count"] == 10
    assert row["percentage"] == 30.0


def test_all_images_zero_denominator_yields_animal_nan() -> None:
    """Test that all images having 0/0 yields animal num=0, den=0, pct=NaN."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(0, 0), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(0, 0), (0, 0)]))

    res = aggregate_by_animal([s1, s2])
    row = res.animal_metrics[res.animal_metrics["metric_name"] == "M1"].iloc[0]

    assert row["sample_count"] == 2
    assert row["numerator_count"] == 0
    assert row["denominator_count"] == 0
    assert math.isnan(row["percentage"])


def test_metric_ordering_follows_first_sample_and_order_agnostic_across_samples() -> None:
    """Test that metric output order follows samples[0] even if samples[1] has permuted definitions."""
    d1 = BiologicalMetricDefinition("A_metric", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    d2 = BiologicalMetricDefinition("B_metric", PhenotypeFilter(), PhenotypeFilter({"B": True}))
    d3 = BiologicalMetricDefinition("C_metric", PhenotypeFilter(), PhenotypeFilter({"C": True}))

    s1_res = _create_mock_summary_result([(1, 10), (2, 10), (3, 10)], [d1, d2, d3])
    # Permuted order in sample 2
    s2_res = _create_mock_summary_result([(30, 100), (10, 100), (20, 100)], [d3, d1, d2])

    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), s1_res)
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), s2_res)

    res = aggregate_by_animal([s1, s2])

    assert list(res.animal_metrics["metric_name"]) == ["A_metric", "B_metric", "C_metric"]
    assert list(res.animal_metrics["numerator_count"]) == [11, 22, 33]
    assert list(res.animal_metrics["denominator_count"]) == [110, 110, 110]
    assert list(res.animal_metrics["percentage"]) == [10.0, 20.0, 30.0]


def test_animal_ordering_follows_first_appearance() -> None:
    """Test that animals appear in order of first appearance in the samples list."""
    s1 = SampleSummary(SampleIdentity("img1", "Animal_B"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img2", "Animal_A"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s3 = SampleSummary(SampleIdentity("img3", "Animal_B"), _create_mock_summary_result([(1, 2), (0, 0)]))

    res = aggregate_by_animal([s1, s2, s3])
    # Distinct animals ordered: Animal_B then Animal_A
    unique_animals = list(dict.fromkeys(res.animal_metrics["animal_id"]))
    assert unique_animals == ["Animal_B", "Animal_A"]


def test_sample_count_constant_across_metrics_for_animal() -> None:
    """Test that sample_count is constant across all metric rows for a given animal."""
    s1 = SampleSummary(SampleIdentity("img1", "Animal_01"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img2", "Animal_01"), _create_mock_summary_result([(2, 4), (1, 3)]))
    s3 = SampleSummary(SampleIdentity("img3", "Animal_01"), _create_mock_summary_result([(0, 0), (0, 0)]))

    res = aggregate_by_animal([s1, s2, s3])
    assert (res.animal_metrics["sample_count"] == 3).all()


def test_arbitrary_sample_and_animal_ids_preserved() -> None:
    """Test preservation of custom user IDs and exact casing."""
    s1 = SampleSummary(SampleIdentity("img_A_01", "Mouse_Ctrl_1"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img_A_02", "mouse_ctrl_1"), _create_mock_summary_result([(1, 2), (0, 0)]))

    res = aggregate_by_animal([s1, s2])
    unique_animals = list(dict.fromkeys(res.animal_metrics["animal_id"]))
    assert unique_animals == ["Mouse_Ctrl_1", "mouse_ctrl_1"]


def test_invalid_sample_and_animal_ids_rejected() -> None:
    """Test rejection of empty and whitespace-padded IDs."""
    summary = _create_mock_summary_result([(1, 2), (0, 0)])

    with pytest.raises(InvalidSampleIdentityError, match="non-empty"):
        SampleIdentity("", "Animal_01")

    with pytest.raises(InvalidSampleIdentityError, match="non-empty"):
        SampleIdentity("   ", "Animal_01")

    with pytest.raises(InvalidSampleIdentityError, match="leading or trailing whitespace"):
        SampleIdentity(" img01", "Animal_01")

    with pytest.raises(InvalidSampleIdentityError, match="leading or trailing whitespace"):
        SampleIdentity("img01", "Animal_01 ")


def test_duplicate_sample_id_rejected() -> None:
    """Test that duplicate sample_id across samples raises InvalidSampleIdentityError."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img01", "Animal_02"), _create_mock_summary_result([(1, 2), (0, 0)]))

    with pytest.raises(InvalidSampleIdentityError, match="Duplicate sample_id"):
        aggregate_by_animal([s1, s2])


def test_incompatible_metric_definitions_rejected() -> None:
    """Test that different filter definitions under the same metric name raise AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    d2_diff = BiologicalMetricDefinition("M1", PhenotypeFilter({"B": True}), PhenotypeFilter({"A": True}))

    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2)], [d1]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(1, 2)], [d2_diff]))

    with pytest.raises(AggregationConsistencyError, match="Incompatible metric definition"):
        aggregate_by_animal([s1, s2])


def test_missing_or_extra_metric_in_sample_rejected() -> None:
    """Test that missing or extra metrics across samples raise AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    d2 = BiologicalMetricDefinition("M2", PhenotypeFilter(), PhenotypeFilter({"B": True}))

    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2), (1, 2)], [d1, d2]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(1, 2)], [d1]))

    with pytest.raises(AggregationConsistencyError, match="metric definitions"):
        aggregate_by_animal([s1, s2])


def test_mismatched_filter_strings_rejected() -> None:
    """Test that modified numerator_filter or denominator_filter strings raise AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter({"B": True}), PhenotypeFilter({"A": True}))
    res = _create_mock_summary_result([(1, 2)], [d1])

    # Corrupt denominator_filter string
    corrupt_df = res.metrics.copy()
    corrupt_df.loc[0, "denominator_filter"] = "CORRUPTED"
    corrupt_res = BiologicalSummaryResult(corrupt_df, res.definitions, None)  # type: ignore

    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), corrupt_res)

    with pytest.raises(AggregationConsistencyError, match="Mismatched denominator_filter"):
        aggregate_by_animal([sample])


def test_numpy_integer_counts_accepted() -> None:
    """Test that numpy integer scalar counts (np.int32, np.int64) are accepted."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": np.int32(5),
                "denominator_count": np.int64(10),
                "percentage": 50.0,
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    agg = aggregate_by_animal([sample])
    assert agg.animal_metrics.iloc[0]["numerator_count"] == 5
    assert agg.animal_metrics.iloc[0]["denominator_count"] == 10


@pytest.mark.parametrize(
    "corrupt_val, error_match",
    [
        (5.0, "Invalid numerator_count type"),
        (True, "Invalid numerator_count type"),
        (np.bool_(True), "Invalid numerator_count type"),
        (-1, "Negative count"),
    ],
)
def test_invalid_counts_rejected(corrupt_val, error_match: str) -> None:
    """Test rejection of non-integer and negative counts."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": corrupt_val,
                "denominator_count": 10,
                "percentage": 50.0,
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match=error_match):
        aggregate_by_animal([sample])


def test_numerator_greater_than_denominator_rejected() -> None:
    """Test that numerator_count > denominator_count raises AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": 15,
                "denominator_count": 10,
                "percentage": 150.0,
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="exceeds denominator_count"):
        aggregate_by_animal([sample])


def test_corrupted_image_percentage_rejected() -> None:
    """Test that an incorrect percentage raises AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": 5,
                "denominator_count": 10,
                "percentage": 99.9,  # Corrupted (should be 50.0)
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="Corrupted percentage"):
        aggregate_by_animal([sample])


def test_valid_unrounded_fraction_percentage_accepted() -> None:
    """Test that unrounded 1/3 (33.333333333333336) passes strict numerical validation."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": 1,
                "denominator_count": 3,
                "percentage": 100.0 / 3.0,
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    agg = aggregate_by_animal([sample])
    assert agg.animal_metrics.iloc[0]["percentage"] == pytest.approx(100.0 / 3.0, rel=1e-12)


def test_empty_sample_sequence_rejected() -> None:
    """Test that passing an empty sample sequence raises InvalidSampleIdentityError."""
    with pytest.raises(InvalidSampleIdentityError, match="At least one SampleSummary"):
        aggregate_by_animal([])


def test_contributing_samples_provenance_retained() -> None:
    """Test that AnimalAggregationResult retains the input samples tuple untouched."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(2, 4), (0, 0)]))

    res = aggregate_by_animal([s1, s2])
    assert res.samples == (s1, s2)


def test_denominator_zero_with_positive_numerator_rejected() -> None:
    """Test that den=0 with num>0 in an input M6 summary raises AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {
                "metric_name": "M1",
                "numerator_count": 2,
                "denominator_count": 0,
                "percentage": float("nan"),
                "numerator_filter": "A+",
                "denominator_filter": "ALL",
            }
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="exceeds denominator_count"):
        aggregate_by_animal([sample])


def test_duplicate_metric_names_in_m6_summary_rejected() -> None:
    """Test that duplicate metric names in M6 DataFrame raise AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {"metric_name": "M1", "numerator_count": 1, "denominator_count": 2, "percentage": 50.0, "numerator_filter": "A+", "denominator_filter": "ALL"},
            {"metric_name": "M1", "numerator_count": 1, "denominator_count": 2, "percentage": 50.0, "numerator_filter": "A+", "denominator_filter": "ALL"},
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="Duplicate metric names"):
        aggregate_by_animal([sample])


def test_m6_metrics_row_without_definition_rejected() -> None:
    """Test that an extra metrics row without a matching definition raises AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {"metric_name": "M1", "numerator_count": 1, "denominator_count": 2, "percentage": 50.0, "numerator_filter": "A+", "denominator_filter": "ALL"},
            {"metric_name": "M2", "numerator_count": 1, "denominator_count": 2, "percentage": 50.0, "numerator_filter": "A+", "denominator_filter": "ALL"},
        ]
    )
    res = BiologicalSummaryResult(df, (d1,), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="Mismatch between definitions and metrics table"):
        aggregate_by_animal([sample])


def test_m6_definition_without_metrics_row_rejected() -> None:
    """Test that a definition without a corresponding row in metrics raises AggregationConsistencyError."""
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    d2 = BiologicalMetricDefinition("M2", PhenotypeFilter(), PhenotypeFilter({"B": True}))
    df = pd.DataFrame(
        [
            {"metric_name": "M1", "numerator_count": 1, "denominator_count": 2, "percentage": 50.0, "numerator_filter": "A+", "denominator_filter": "ALL"},
        ]
    )
    res = BiologicalSummaryResult(df, (d1, d2), None)  # type: ignore
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    with pytest.raises(AggregationConsistencyError, match="Mismatch between definitions and metrics table"):
        aggregate_by_animal([sample])


def test_m6_summary_objects_immutability() -> None:
    """Test that input M6 summaries and sample objects are untouched by aggregation."""
    res = _create_mock_summary_result([(1, 2), (3, 4)])
    df_copy = res.metrics.copy(deep=True)
    sample = SampleSummary(SampleIdentity("img01", "Animal_01"), res)

    agg = aggregate_by_animal([sample])

    assert res.metrics.equals(df_copy)
    assert agg.animal_metrics is not res.metrics


def test_deterministic_repeated_aggregation() -> None:
    """Test that repeated aggregations produce identical DataFrames."""
    s1 = SampleSummary(SampleIdentity("img01", "Animal_01"), _create_mock_summary_result([(1, 2), (0, 0)]))
    s2 = SampleSummary(SampleIdentity("img02", "Animal_01"), _create_mock_summary_result([(2, 4), (0, 0)]))

    res1 = aggregate_by_animal([s1, s2])
    res2 = aggregate_by_animal([s1, s2])

    assert res1.animal_metrics.equals(res2.animal_metrics)
