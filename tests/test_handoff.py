"""Tests for statistical analysis handoff and analysis-readiness reporting."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from nervenuc.aggregation import AnimalAggregationResult
from nervenuc.export import (
    AnimalGroupAssignment,
    prepare_group_tables,
)
from nervenuc.handoff import (
    AnalysisHandoffResult,
    HandoffConsistencyError,
    format_analysis_handoff,
    prepare_analysis_handoff,
    validate_handoff_input,
)
from nervenuc.statistics import (
    GroupStatisticsResult,
    summarize_groups,
)
from nervenuc.summary import (
    BiologicalMetricDefinition,
    PhenotypeFilter,
)

EXPECTED_METRIC_READINESS_COLUMNS = [
    "metric_name",
    "biological_replicate",
    "group_count",
    "total_animals",
    "total_valid",
    "total_missing",
    "has_missing_values",
    "total_group_sizes_equal",
    "valid_group_sizes_equal",
    "all_groups_have_valid_observations",
    "readiness_status",
    "numerator_filter",
    "denominator_filter",
    "inferential_test",
]

EXPECTED_GROUP_READINESS_COLUMNS = [
    "metric_name",
    "group",
    "n_total",
    "n_valid",
    "n_missing",
]


def _create_mock_statistics_result(
    animal_data: dict[str, list[tuple[int, int]]],
    group_assignments: dict[str, str],
    metric_defs: list[BiologicalMetricDefinition] | None = None,
    sample_count: int = 1,
) -> GroupStatisticsResult:
    """Helper to construct a valid GroupStatisticsResult via M8 and M9 engines."""
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
    group_tables = prepare_group_tables(agg, assigns)
    return summarize_groups(group_tables)


def test_single_metric_two_groups_complete() -> None:
    """Test handoff generation for complete 2-group single-metric dataset."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "WT2": [(2, 10)],
        "MUT1": [(3, 10)],
        "MUT2": [(4, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "MUT1": "MUT", "MUT2": "MUT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)

    assert isinstance(res, AnalysisHandoffResult)
    assert list(res.metric_readiness.columns) == EXPECTED_METRIC_READINESS_COLUMNS
    assert list(res.group_readiness.columns) == EXPECTED_GROUP_READINESS_COLUMNS

    m_row = res.metric_readiness.iloc[0]
    assert m_row["metric_name"] == "Proliferation"
    assert m_row["biological_replicate"] == "animal"
    assert m_row["group_count"] == 2
    assert m_row["total_animals"] == 4
    assert m_row["total_valid"] == 4
    assert m_row["total_missing"] == 0
    assert not m_row["has_missing_values"]
    assert m_row["total_group_sizes_equal"]
    assert m_row["valid_group_sizes_equal"]
    assert m_row["all_groups_have_valid_observations"]
    assert m_row["readiness_status"] == "ready"
    assert m_row["inferential_test"] == "not_selected"


def test_single_group_complete_is_ready() -> None:
    """Test that a complete single-group dataset (k=1) has readiness_status='ready'."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {"A1": [(1, 10)], "A2": [(2, 10)], "A3": [(3, 10)]}
    assigns = {"A1": "WT", "A2": "WT", "A3": "WT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    m_row = res.metric_readiness.iloc[0]

    assert m_row["group_count"] == 1
    assert m_row["total_animals"] == 3
    assert m_row["readiness_status"] == "ready"


def test_unequal_group_sizes_complete_is_ready() -> None:
    """Test that unequal group sizes (n1=2, n2=3) with all valid observations are 'ready'."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "WT2": [(2, 10)],
        "MUT1": [(3, 10)],
        "MUT2": [(4, 10)],
        "MUT3": [(5, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "MUT1": "MUT", "MUT2": "MUT", "MUT3": "MUT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    m_row = res.metric_readiness.iloc[0]

    assert not m_row["total_group_sizes_equal"]
    assert not m_row["valid_group_sizes_equal"]
    assert m_row["readiness_status"] == "ready"


def test_ready_with_missing_values_status() -> None:
    """Test that missing observations in groups with >=1 valid animal produce 'ready_with_missing_values'."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "WT2": [(0, 0)],  # Missing (0/0 -> NaN)
        "MUT1": [(3, 10)],
        "MUT2": [(4, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "MUT1": "MUT", "MUT2": "MUT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    m_row = res.metric_readiness.iloc[0]

    assert m_row["total_animals"] == 4
    assert m_row["total_valid"] == 3
    assert m_row["total_missing"] == 1
    assert m_row["has_missing_values"]
    assert m_row["total_group_sizes_equal"]
    assert not m_row["valid_group_sizes_equal"]
    assert m_row["all_groups_have_valid_observations"]
    assert m_row["readiness_status"] == "ready_with_missing_values"


def test_not_ready_status_when_group_has_zero_valid() -> None:
    """Test that a group with n_valid=0 produces 'not_ready'."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "WT2": [(2, 10)],
        "MUT1": [(0, 0)],  # All MUT are missing
        "MUT2": [(0, 0)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "MUT1": "MUT", "MUT2": "MUT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    m_row = res.metric_readiness.iloc[0]

    assert not m_row["all_groups_have_valid_observations"]
    assert m_row["readiness_status"] == "not_ready"


def test_multiple_groups_and_metrics() -> None:
    """Test handoff across multiple metrics and groups."""
    data = {
        "WT1": [(1, 10), (2, 10)],
        "WT2": [(3, 10), (4, 10)],
        "HNPP1": [(1, 10), (0, 0)],
        "CMT1": [(5, 10), (6, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "HNPP1": "HNPP", "CMT1": "CMT1A"}
    stats = _create_mock_statistics_result(data, assigns)

    res = prepare_analysis_handoff(stats)

    assert len(res.metric_readiness) == 2
    assert len(res.group_readiness) == 6  # 2 metrics x 3 groups

    m1_row = res.metric_readiness[res.metric_readiness["metric_name"] == "M1"].iloc[0]
    assert m1_row["readiness_status"] == "ready"

    m2_row = res.metric_readiness[res.metric_readiness["metric_name"] == "M2"].iloc[0]
    assert m2_row["readiness_status"] == "not_ready"  # HNPP has 0/0 for M2


def test_ordering_and_provenance_preserved() -> None:
    """Test that canonical metric order and authoritative assignment group order are preserved."""
    d1 = BiologicalMetricDefinition("Beta_Metric", PhenotypeFilter(), PhenotypeFilter({"B": True}))
    d2 = BiologicalMetricDefinition("Alpha_Metric", PhenotypeFilter(), PhenotypeFilter({"A": True}))

    data = {
        "A1": [(1, 10), (2, 10)],
        "A2": [(3, 10), (4, 10)],
    }
    assigns = {"A2": "Group_B", "A1": "Group_A"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1, d2])

    res = prepare_analysis_handoff(stats)

    # Metric order follows definitions: Beta_Metric then Alpha_Metric
    assert list(res.metric_readiness["metric_name"]) == ["Beta_Metric", "Alpha_Metric"]
    # Group order in group_readiness follows assignments: Group_B then Group_A
    assert list(res.group_readiness[res.group_readiness["metric_name"] == "Beta_Metric"]["group"]) == ["Group_B", "Group_A"]


def test_text_formatter_output() -> None:
    """Test researcher-readable text report formatting."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "WT2": [(2, 10)],
        "CMT1": [(3, 10)],
    }
    assigns = {"WT1": "WT", "WT2": "WT", "CMT1": "CMT1A"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    text = format_analysis_handoff(res)

    assert "NerveNuc Analysis Readiness Report" in text
    assert "Biological replicate: Animal" in text
    assert "NerveNuc analysis handoff status: Ready" in text
    assert "WT     n_total=2  n_valid=2  n_missing=0" in text
    assert "CMT1A  n_total=1  n_valid=1  n_missing=0" in text
    assert "CSV export: Supported" in text
    assert "Multi-sheet Excel export: Supported" in text
    assert "GraphPad Prism-ready wide tables (prepared in memory)" in text
    assert "Inferential test: Not selected by NerveNuc." in text
    assert "The researcher should select the statistical method based on the experimental design and analysis plan." in text
    assert "recommended test" not in text.lower()
    assert "anova" not in text.lower()


def test_text_formatter_not_ready_explanation() -> None:
    """Test that not_ready metric includes factual explanation in formatted report."""
    d1 = BiologicalMetricDefinition("Proliferation", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    data = {
        "WT1": [(1, 10)],
        "MUT1": [(0, 0)],
    }
    assigns = {"WT1": "WT", "MUT1": "MUT"}
    stats = _create_mock_statistics_result(data, assigns, metric_defs=[d1])

    res = prepare_analysis_handoff(stats)
    text = format_analysis_handoff(res)

    assert "NerveNuc analysis handoff status: Not Ready" in text
    assert "One or more groups have no valid animal-level observation for this metric." in text


def test_malformed_group_statistics_schema_rejected() -> None:
    """Test rejection when group_statistics DataFrame is missing required columns."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_stats = GroupStatisticsResult(
        group_statistics=stats.group_statistics.drop(columns=["ci95_low"]),
        observations=stats.observations,
        source=stats.source,
    )
    with pytest.raises(HandoffConsistencyError, match="group_statistics columns must match"):
        validate_handoff_input(corrupt_stats)


def test_malformed_observations_schema_rejected() -> None:
    """Test rejection when observations DataFrame is missing required columns."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_stats = GroupStatisticsResult(
        group_statistics=stats.group_statistics,
        observations=stats.observations.drop(columns=["percentage"]),
        source=stats.source,
    )
    with pytest.raises(HandoffConsistencyError, match="observations columns must match"):
        validate_handoff_input(corrupt_stats)


@pytest.mark.parametrize(
    "corrupt_val, col_name",
    [
        (1.5, "n_total"),
        (True, "n_valid"),
        ("0", "n_missing"),
        (-1, "n_total"),
    ],
)
def test_invalid_count_types_rejected(corrupt_val, col_name: str) -> None:
    """Test rejection of non-integer or negative counts in group_statistics."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = stats.group_statistics.copy()
    corrupt_df[col_name] = corrupt_df[col_name].astype(object)
    corrupt_df.loc[0, col_name] = corrupt_val
    corrupt_stats = GroupStatisticsResult(
        group_statistics=corrupt_df,
        observations=stats.observations,
        source=stats.source,
    )
    with pytest.raises(HandoffConsistencyError):
        validate_handoff_input(corrupt_stats)


def test_n_total_identity_violation_rejected() -> None:
    """Test rejection when n_total != n_valid + n_missing."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = stats.group_statistics.copy()
    corrupt_df.loc[0, "n_total"] = 10  # instead of 1 (1 + 0)
    corrupt_stats = GroupStatisticsResult(
        group_statistics=corrupt_df,
        observations=stats.observations,
        source=stats.source,
    )
    with pytest.raises(HandoffConsistencyError, match=r"n_total \(10\) != n_valid \(1\) \+ n_missing \(0\)"):
        validate_handoff_input(corrupt_stats)


def test_inconsistent_observations_counts_rejected() -> None:
    """Test rejection when recomputed observation counts differ from stored group_statistics."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupt_df = stats.group_statistics.copy()
    corrupt_df.loc[0, "n_valid"] = 2
    corrupt_df.loc[0, "n_total"] = 2
    corrupt_stats = GroupStatisticsResult(
        group_statistics=corrupt_df,
        observations=stats.observations,  # only has 1 observation
        source=stats.source,
    )
    with pytest.raises(HandoffConsistencyError, match="Count mismatch between observations and group_statistics"):
        validate_handoff_input(corrupt_stats)


def test_corrupted_wide_tables_rejected() -> None:
    """Test rejection when M8 wide_tables are missing canonical metrics."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupted_source = stats.source
    corrupted_source.wide_tables = {}  # Empty wide tables
    corrupt_stats = GroupStatisticsResult(
        group_statistics=stats.group_statistics,
        observations=stats.observations,
        source=corrupted_source,
    )
    with pytest.raises(HandoffConsistencyError, match="source.wide_tables keys do not match"):
        validate_handoff_input(corrupt_stats)


def test_corrupted_animal_id_table_rejected() -> None:
    """Test rejection when animal_id_table columns do not match authoritative groups."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupted_source = stats.source
    corrupted_source.animal_id_table = pd.DataFrame({"MUT": ["A1"]})
    corrupt_stats = GroupStatisticsResult(
        group_statistics=stats.group_statistics,
        observations=stats.observations,
        source=corrupted_source,
    )
    with pytest.raises(HandoffConsistencyError, match="animal_id_table columns .* do not match"):
        validate_handoff_input(corrupt_stats)


def test_corrupted_manifest_rejected() -> None:
    """Test rejection when manifest does not contain canonical metrics."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    corrupted_source = stats.source
    corrupted_source.manifest = pd.DataFrame([{"metric_name": "UNKNOWN"}])
    corrupt_stats = GroupStatisticsResult(
        group_statistics=stats.group_statistics,
        observations=stats.observations,
        source=corrupted_source,
    )
    with pytest.raises(HandoffConsistencyError, match="source.manifest does not contain every canonical metric"):
        validate_handoff_input(corrupt_stats)


@pytest.mark.parametrize(
    "column",
    ["mean", "median", "sd", "sem", "ci95_low", "ci95_high", "min", "max"],
)
def test_corrupted_descriptive_statistics_rejected(column: str) -> None:
    """M10 recomputes every descriptive field from authoritative M9 observations."""
    stats = _create_mock_statistics_result(
        {"A1": [(1, 10), (1, 5)], "A2": [(2, 10), (2, 5)], "A3": [(4, 10), (3, 5)]},
        {"A1": "WT", "A2": "WT", "A3": "WT"},
    )
    corrupt_df = stats.group_statistics.copy(deep=True)
    corrupt_df.loc[0, column] = 999.0
    corrupt = GroupStatisticsResult(
        group_statistics=corrupt_df,
        observations=stats.observations,
        source=stats.source,
    )

    with pytest.raises(HandoffConsistencyError, match="group_statistics does not match"):
        validate_handoff_input(corrupt)


def test_corrupted_m8_wide_value_rejected() -> None:
    """Structurally plausible but forged M8 wide values fail provenance regeneration."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (1, 5)]}, {"A1": "WT"})
    stats.source.wide_tables["M1"].iloc[0, 0] = 999.0

    with pytest.raises(HandoffConsistencyError, match="does not match canonical"):
        validate_handoff_input(stats)


def test_source_immutability() -> None:
    """Test that prepare_analysis_handoff does not mutate input statistics_result."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})
    gs_copy = stats.group_statistics.copy(deep=True)
    obs_copy = stats.observations.copy(deep=True)

    res = prepare_analysis_handoff(stats)

    assert stats.group_statistics.equals(gs_copy)
    assert stats.observations.equals(obs_copy)
    assert res.source is stats


def test_deterministic_repeated_execution() -> None:
    """Test deterministic outputs on repeated calls."""
    stats = _create_mock_statistics_result({"A1": [(1, 10), (0, 0)]}, {"A1": "WT"})

    res1 = prepare_analysis_handoff(stats)
    res2 = prepare_analysis_handoff(stats)

    assert res1.metric_readiness.equals(res2.metric_readiness)
    assert res1.group_readiness.equals(res2.group_readiness)

    text1 = format_analysis_handoff(res1)
    text2 = format_analysis_handoff(res2)
    assert text1 == text2
