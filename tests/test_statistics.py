"""Tests for group-level descriptive statistics and statistical analysis preparation."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import scipy.stats

from nervenuc.aggregation import AnimalAggregationResult
from nervenuc.export import (
    AnimalGroupAssignment,
    GroupTableResult,
    prepare_group_tables,
)
from nervenuc.statistics import (
    GroupStatisticsResult,
    NerveNucStatisticsError,
    StatisticsConsistencyError,
    prepare_analysis_observations,
    summarize_groups,
    validate_statistics_input,
)
from nervenuc.summary import (
    BiologicalMetricDefinition,
    PhenotypeFilter,
)

EXPECTED_OBSERVATION_COLUMNS = [
    "animal_id",
    "group",
    "metric_name",
    "percentage",
    "numerator_count",
    "denominator_count",
]

EXPECTED_STATISTICS_COLUMNS = [
    "group",
    "metric_name",
    "n_total",
    "n_valid",
    "n_missing",
    "mean",
    "median",
    "sd",
    "sem",
    "ci95_low",
    "ci95_high",
    "min",
    "max",
    "numerator_filter",
    "denominator_filter",
]


def _create_mock_group_table_result(
    animal_data: dict[str, list[tuple[int, int]]],
    group_assignments: dict[str, str],
    metric_defs: list[BiologicalMetricDefinition] | None = None,
    sample_count: int = 1,
) -> GroupTableResult:
    """Helper to construct a validated GroupTableResult for testing."""
    if metric_defs is None:
        metric_defs = [
            BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True})),
            BiologicalMetricDefinition("M2", PhenotypeFilter({"SOX10": True}), PhenotypeFilter({"BrdU": True})),
        ]

    rows = []
    for aid, counts in animal_data.items():
        for (num, den), d in zip(counts, metric_defs):
            pct = float(100.0 * num / den) if den > 0 else float("nan")
            rows.append(
                {
                    "animal_id": aid,
                    "metric_name": d.name,
                    "sample_count": sample_count,
                    "numerator_count": num,
                    "denominator_count": den,
                    "percentage": pct,
                    "numerator_filter": d.effective_numerator.to_string(),
                    "denominator_filter": d.denominator.to_string(),
                }
            )

    df = pd.DataFrame(rows)
    agg = AnimalAggregationResult(
        animal_metrics=df,
        definitions=tuple(metric_defs),
        samples=(),
    )
    assigns = [AnimalGroupAssignment(aid, grp) for aid, grp in group_assignments.items()]
    return prepare_group_tables(agg, assigns)


def test_single_group_multiple_animals_descriptive_stats() -> None:
    """Test full descriptive statistics calculation for a single group with 5 animals."""
    # 5 animals with percentages: 10, 20, 30, 40, 50
    data = {
        "A1": [(1, 10), (0, 0)],
        "A2": [(2, 10), (0, 0)],
        "A3": [(3, 10), (0, 0)],
        "A4": [(4, 10), (0, 0)],
        "A5": [(5, 10), (0, 0)],
    }
    assigns = {f"A{i}": "WT" for i in range(1, 6)}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)

    assert isinstance(res, GroupStatisticsResult)
    assert list(res.group_statistics.columns) == EXPECTED_STATISTICS_COLUMNS

    m1_row = res.group_statistics[(res.group_statistics["group"] == "WT") & (res.group_statistics["metric_name"] == "M1")].iloc[0]
    assert m1_row["n_total"] == 5
    assert m1_row["n_valid"] == 5
    assert m1_row["n_missing"] == 0
    assert m1_row["mean"] == pytest.approx(30.0)
    assert m1_row["median"] == pytest.approx(30.0)
    assert m1_row["min"] == pytest.approx(10.0)
    assert m1_row["max"] == pytest.approx(50.0)

    # Sample SD (ddof=1) of [10, 20, 30, 40, 50] is sqrt(250) = 15.811388...
    expected_sd = float(np.std([10, 20, 30, 40, 50], ddof=1))
    assert m1_row["sd"] == pytest.approx(expected_sd)

    # SEM = SD / sqrt(5)
    expected_sem = expected_sd / math.sqrt(5)
    assert m1_row["sem"] == pytest.approx(expected_sem)

    # Student t CI: t(0.975, df=4) * sem
    t_crit = float(scipy.stats.t.ppf(0.975, df=4))
    assert m1_row["ci95_low"] == pytest.approx(30.0 - t_crit * expected_sem)
    assert m1_row["ci95_high"] == pytest.approx(30.0 + t_crit * expected_sem)


def test_equal_animal_weighting_not_pooled_counts() -> None:
    """Critical test: verify arithmetic mean of percentages vs pooled count weighting."""
    # Animal 1: 1/2 = 50%
    # Animal 2: 1/18 = 5.555555...%
    # Arithmetic mean = (50 + 5.555555...)/2 = 27.777777...%
    # (Pooled count would be 2/20 = 10%)
    data = {
        "A1": [(1, 2), (0, 0)],
        "A2": [(1, 18), (0, 0)],
    }
    assigns = {"A1": "WT", "A2": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    m1_row = res.group_statistics[res.group_statistics["metric_name"] == "M1"].iloc[0]

    expected_mean = (50.0 + (100.0 / 18.0)) / 2.0
    assert m1_row["mean"] == pytest.approx(expected_mean)
    assert m1_row["mean"] != pytest.approx(10.0)


def test_multiple_groups_and_metrics() -> None:
    """Test group statistics across multiple groups and metrics."""
    data = {
        "WT1": [(1, 10), (2, 10)],
        "WT2": [(3, 10), (4, 10)],
        "MUT1": [(5, 10), (6, 10)],
        "MUT2": [(7, 10), (8, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "MUT1": "MUT", "MUT2": "MUT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    assert len(res.group_statistics) == 4  # 2 groups x 2 metrics

    df = res.group_statistics
    wt_m1 = df[(df["group"] == "WT") & (df["metric_name"] == "M1")].iloc[0]
    assert wt_m1["mean"] == pytest.approx(20.0)
    assert wt_m1["n_total"] == 2

    mut_m1 = df[(df["group"] == "MUT") & (df["metric_name"] == "M1")].iloc[0]
    assert mut_m1["mean"] == pytest.approx(60.0)
    assert mut_m1["n_total"] == 2


def test_transparent_missingness_handling() -> None:
    """Test that zero-denominator animals are counted in n_total and n_missing, but excluded from descriptive stats."""
    data = {
        "A1": [(1, 10), (0, 0)],
        "A2": [(2, 10), (0, 0)],
        "A3": [(0, 0), (0, 0)],  # Missing (0/0 -> NaN) for M1
    }
    assigns = {"A1": "WT", "A2": "WT", "A3": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    m1_row = res.group_statistics[res.group_statistics["metric_name"] == "M1"].iloc[0]

    assert m1_row["n_total"] == 3
    assert m1_row["n_valid"] == 2
    assert m1_row["n_missing"] == 1
    assert m1_row["mean"] == pytest.approx(15.0)  # (10 + 20) / 2
    assert m1_row["n_total"] == m1_row["n_valid"] + m1_row["n_missing"]


def test_n_valid_zero_behavior() -> None:
    """Test that n_valid = 0 yields NaN for all descriptive statistics."""
    data = {
        "A1": [(0, 0), (0, 0)],
        "A2": [(0, 0), (0, 0)],
    }
    assigns = {"A1": "WT", "A2": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    m1_row = res.group_statistics[res.group_statistics["metric_name"] == "M1"].iloc[0]

    assert m1_row["n_total"] == 2
    assert m1_row["n_valid"] == 0
    assert m1_row["n_missing"] == 2
    assert math.isnan(m1_row["mean"])
    assert math.isnan(m1_row["median"])
    assert math.isnan(m1_row["sd"])
    assert math.isnan(m1_row["sem"])
    assert math.isnan(m1_row["ci95_low"])
    assert math.isnan(m1_row["ci95_high"])
    assert math.isnan(m1_row["min"])
    assert math.isnan(m1_row["max"])


def test_n_valid_one_behavior() -> None:
    """Test that n_valid = 1 returns observed value for mean/median/min/max and NaN for SD/SEM/CI."""
    data = {
        "A1": [(2, 10), (0, 0)],
        "A2": [(0, 0), (0, 0)],
    }
    assigns = {"A1": "WT", "A2": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    m1_row = res.group_statistics[res.group_statistics["metric_name"] == "M1"].iloc[0]

    assert m1_row["n_total"] == 2
    assert m1_row["n_valid"] == 1
    assert m1_row["n_missing"] == 1
    assert m1_row["mean"] == pytest.approx(20.0)
    assert m1_row["median"] == pytest.approx(20.0)
    assert m1_row["min"] == pytest.approx(20.0)
    assert m1_row["max"] == pytest.approx(20.0)
    assert math.isnan(m1_row["sd"])
    assert math.isnan(m1_row["sem"])
    assert math.isnan(m1_row["ci95_low"])
    assert math.isnan(m1_row["ci95_high"])


def test_identical_observations_zero_sd_and_collapsed_ci() -> None:
    """Test that identical replicate values yield SD=0, SEM=0, and CI=[mean, mean]."""
    data = {
        "A1": [(2, 10), (0, 0)],
        "A2": [(2, 10), (0, 0)],
        "A3": [(2, 10), (0, 0)],
    }
    assigns = {"A1": "WT", "A2": "WT", "A3": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    res = summarize_groups(gt)
    m1_row = res.group_statistics[res.group_statistics["metric_name"] == "M1"].iloc[0]

    assert m1_row["mean"] == 20.0
    assert m1_row["sd"] == 0.0
    assert m1_row["sem"] == 0.0
    assert m1_row["ci95_low"] == 20.0
    assert m1_row["ci95_high"] == 20.0


def test_ci_unclamped_below_zero_and_above_100() -> None:
    """Test that Student t confidence intervals are not artificially clamped to [0, 100]."""
    # Sample near 0: [0.1, 0.5] with small n=2 -> broad t-interval extending < 0
    d1 = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"A": True}))
    df = pd.DataFrame(
        [
            {"animal_id": "A1", "group": "WT", "metric_name": "M1", "sample_count": 1, "numerator_count": 1, "denominator_count": 1000, "percentage": 0.1, "numerator_filter": "A+", "denominator_filter": "ALL"},
            {"animal_id": "A2", "group": "WT", "metric_name": "M1", "sample_count": 1, "numerator_count": 5, "denominator_count": 1000, "percentage": 0.5, "numerator_filter": "A+", "denominator_filter": "ALL"},
        ]
    )
    agg = AnimalAggregationResult(
        animal_metrics=df.drop(columns=["group"]),
        definitions=(d1,),
        samples=(),
    )
    gt = GroupTableResult(
        long_table=df,
        wide_tables={"M1": pd.DataFrame({"WT": [0.1, 0.5]})},
        animal_id_table=pd.DataFrame({"WT": ["A1", "A2"]}),
        manifest=pd.DataFrame([{"metric_index": 1, "metric_name": "M1", "csv_filename": "metric_001.csv", "excel_sheet_name": "Metric 001", "numerator_filter": "A+", "denominator_filter": "ALL"}]),
        assignments=(AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A2", "WT")),
        aggregation_result=agg,
    )

    res = summarize_groups(gt)
    m1_row = res.group_statistics.iloc[0]

    # CI lower bound should be strictly less than 0 without being clamped
    assert m1_row["ci95_low"] < 0.0


def test_observations_table_retains_nan_and_follows_order() -> None:
    """Test that observations table retains NaN observations and preserves canonical animal order."""
    data = {
        "A2": [(1, 10), (0, 0)],
        "A1": [(0, 0), (0, 0)],
    }
    assigns = {"A2": "WT", "A1": "WT"}
    gt = _create_mock_group_table_result(data, assigns)

    obs = prepare_analysis_observations(gt)

    assert list(obs.columns) == EXPECTED_OBSERVATION_COLUMNS
    assert list(obs["animal_id"]) == ["A2", "A2", "A1", "A1"]
    assert math.isnan(obs[obs["animal_id"] == "A1"]["percentage"].iloc[0])


@pytest.mark.parametrize(
    "corrupt_pct, error_match",
    [
        (-1.0, "out of range"),
        (105.0, "out of range"),
        (float("inf"), "Non-finite percentage"),
        (float("-inf"), "Non-finite percentage"),
        (True, "Invalid percentage scalar type"),
        (np.bool_(True), "Invalid percentage scalar type"),
        ("50.0", "Invalid percentage scalar type"),
        (None, "Invalid percentage scalar type"),
    ],
)
def test_invalid_percentage_values_rejected(corrupt_pct, error_match: str) -> None:
    """Test rejection of out-of-range, non-finite, and non-numeric percentages."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df["percentage"] = corrupt_df["percentage"].astype(object)
    corrupt_df.loc[0, "percentage"] = corrupt_pct
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match=error_match):
        validate_statistics_input(gt)


def test_nan_percentage_with_positive_denominator_rejected() -> None:
    """Test that NaN percentage is rejected when denominator > 0."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df.loc[0, "percentage"] = float("nan")
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match="NaN percentage only allowed when den=0"):
        validate_statistics_input(gt)


def test_finite_percentage_with_zero_denominator_rejected() -> None:
    """Test that finite percentage is rejected when denominator == 0."""
    gt = _create_mock_group_table_result({"A1": [(0, 0), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df.loc[0, "percentage"] = 0.0
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match="Finite percentage .* with zero denominator"):
        validate_statistics_input(gt)


@pytest.mark.parametrize(
    "corrupt_sc, error_match",
    [
        (0, "sample_count must be >= 1"),
        (-1, "sample_count must be >= 1"),
        (1.5, "Invalid sample_count type"),
        (True, "Invalid sample_count type"),
        ("1", "Invalid sample_count type"),
    ],
)
def test_invalid_sample_count_rejected(corrupt_sc, error_match: str) -> None:
    """Test rejection of non-integer or <1 sample_count."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df["sample_count"] = corrupt_df["sample_count"].astype(object)
    corrupt_df.loc[0, "sample_count"] = corrupt_sc
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match=error_match):
        validate_statistics_input(gt)


def test_inconsistent_sample_count_across_metrics_rejected() -> None:
    """Test rejection of inconsistent sample_count for the same animal."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df.loc[1, "sample_count"] = 5
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match="Inconsistent sample_count"):
        validate_statistics_input(gt)


def test_mismatched_group_in_long_table_rejected() -> None:
    """Test rejection when long_table group does not match assignments."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df.loc[0, "group"] = "MUT"
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match="Group mismatch"):
        validate_statistics_input(gt)


def test_duplicate_or_missing_metric_rows_rejected() -> None:
    """Test rejection of missing or extra metric rows."""
    # 2 animals (A1 with M1 and M2, A2 missing M2)
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)], "A2": [(1, 10), (0, 0)]}, {"A1": "WT", "A2": "WT"})
    short_df = gt.long_table.drop(index=3).copy()  # drop A2's M2 row
    gt.long_table = short_df

    with pytest.raises(StatisticsConsistencyError, match="expected 2"):
        validate_statistics_input(gt)


def test_mismatched_filter_provenance_rejected() -> None:
    """Test rejection of altered filter strings."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = gt.long_table.copy()
    corrupt_df.loc[0, "denominator_filter"] = "CORRUPTED"
    gt.long_table = corrupt_df

    with pytest.raises(StatisticsConsistencyError, match="Mismatched denominator_filter"):
        validate_statistics_input(gt)


def test_ordering_preserved() -> None:
    """Test that group and metric ordering follow assignments and canonical definitions."""
    d1 = BiologicalMetricDefinition("B_metric", PhenotypeFilter(), PhenotypeFilter({"B": True}))
    d2 = BiologicalMetricDefinition("A_metric", PhenotypeFilter(), PhenotypeFilter({"A": True}))

    data = {
        "A1": [(1, 10), (2, 10)],
        "A2": [(3, 10), (4, 10)],
    }
    assigns = {"A2": "Group_2", "A1": "Group_1"}
    gt = _create_mock_group_table_result(data, assigns, metric_defs=[d1, d2])

    res = summarize_groups(gt)
    df = res.group_statistics

    # Group order from assignments: Group_2 then Group_1
    assert list(dict.fromkeys(df["group"])) == ["Group_2", "Group_1"]
    # Metric order from definitions: B_metric then A_metric
    assert list(df["metric_name"]) == ["B_metric", "A_metric", "B_metric", "A_metric"]


def test_source_immutability() -> None:
    """Test that summarize_groups does not mutate the source GroupTableResult."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    df_copy = gt.long_table.copy(deep=True)

    res = summarize_groups(gt)
    assert gt.long_table.equals(df_copy)
    assert res.observations is not gt.long_table


def test_deterministic_repeated_summarize_groups() -> None:
    """Test that repeated calls to summarize_groups produce identical results."""
    gt = _create_mock_group_table_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})

    res1 = summarize_groups(gt)
    res2 = summarize_groups(gt)

    assert res1.group_statistics.equals(res2.group_statistics)
    assert res1.observations.equals(res2.observations)
