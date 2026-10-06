"""Engine for validating group assignments and preparing animal-level group tables."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import GroupTableConsistencyError, InvalidGroupAssignmentError
from .models import AnimalGroupAssignment, GroupTableResult

REQUIRED_M7_COLUMNS: list[str] = [
    "animal_id",
    "metric_name",
    "sample_count",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]

LONG_TABLE_COLUMNS: list[str] = [
    "animal_id",
    "group",
    "metric_name",
    "sample_count",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]

MANIFEST_COLUMNS: list[str] = [
    "metric_index",
    "metric_name",
    "csv_filename",
    "excel_sheet_name",
    "numerator_filter",
    "denominator_filter",
]


def validate_group_assignments(
    aggregation_result: object,
    assignments: Sequence[AnimalGroupAssignment],
) -> None:
    """Validate structure, consistency, and completeness of group assignments against an M7 result.

    Raises:
        InvalidGroupAssignmentError: If assignments is empty, contains non-assignment items,
            duplicate/conflicting assignments, or unknown/missing animals.
        GroupTableConsistencyError: If the M7 aggregation result is malformed, missing metrics,
            has non-integral/negative/corrupted counts or percentages, or mismatched filter provenance.
    """
    from nervenuc.aggregation import AnimalAggregationResult

    if not isinstance(aggregation_result, AnimalAggregationResult):
        raise GroupTableConsistencyError(
            f"aggregation_result must be an AnimalAggregationResult, got {type(aggregation_result).__name__}."
        )

    defs = aggregation_result.definitions
    if not defs:
        raise GroupTableConsistencyError("aggregation_result.definitions cannot be empty.")

    def_names = [d.name for d in defs]
    if len(set(def_names)) != len(def_names):
        raise GroupTableConsistencyError("Duplicate metric names in aggregation_result.definitions.")

    df = aggregation_result.animal_metrics
    if not isinstance(df, pd.DataFrame):
        raise GroupTableConsistencyError("animal_metrics must be a DataFrame.")

    if list(df.columns) != REQUIRED_M7_COLUMNS:
        raise GroupTableConsistencyError(
            f"animal_metrics columns must match {REQUIRED_M7_COLUMNS}, got {list(df.columns)}."
        )

    if len(df) == 0:
        raise GroupTableConsistencyError("animal_metrics DataFrame cannot be empty.")

    canonical_animals = list(dict.fromkeys(df["animal_id"]))

    for aid in canonical_animals:
        animal_df = df[df["animal_id"] == aid]
        if len(animal_df) != len(defs):
            raise GroupTableConsistencyError(
                f"Animal '{aid}' has {len(animal_df)} rows in animal_metrics, expected {len(defs)}."
            )

        animal_metric_names = list(animal_df["metric_name"])
        if len(set(animal_metric_names)) != len(animal_metric_names):
            raise GroupTableConsistencyError(f"Duplicate metric rows for animal '{aid}'.")

        first_sc: int | None = None
        for d in defs:
            match = animal_df[animal_df["metric_name"] == d.name]
            if len(match) != 1:
                raise GroupTableConsistencyError(
                    f"Missing expected metric row '{d.name}' for animal '{aid}'."
                )
            row = match.iloc[0]

            # Validate sample_count
            sc = row["sample_count"]
            if type(sc) is bool or isinstance(sc, np.bool_) or not (type(sc) is int or isinstance(sc, np.integer)):
                raise GroupTableConsistencyError(
                    f"Invalid sample_count type ({type(sc).__name__}) for animal '{aid}'."
                )
            sc_int = int(sc)
            if sc_int < 1:
                raise GroupTableConsistencyError(
                    f"sample_count must be >= 1, got {sc_int} for animal '{aid}'."
                )
            if first_sc is None:
                first_sc = sc_int
            elif sc_int != first_sc:
                raise GroupTableConsistencyError(
                    f"Inconsistent sample_count for animal '{aid}' across metrics: {first_sc} vs {sc_int}."
                )

            # Validate count types and ranges
            num = row["numerator_count"]
            den = row["denominator_count"]
            if type(num) is bool or isinstance(num, np.bool_) or not (type(num) is int or isinstance(num, np.integer)):
                raise GroupTableConsistencyError(
                    f"Invalid numerator_count type ({type(num).__name__}) for metric '{d.name}' in animal '{aid}'."
                )
            if type(den) is bool or isinstance(den, np.bool_) or not (type(den) is int or isinstance(den, np.integer)):
                raise GroupTableConsistencyError(
                    f"Invalid denominator_count type ({type(den).__name__}) for metric '{d.name}' in animal '{aid}'."
                )

            num_int = int(num)
            den_int = int(den)
            if num_int < 0 or den_int < 0:
                raise GroupTableConsistencyError(
                    f"Negative count for metric '{d.name}' in animal '{aid}': num={num_int}, den={den_int}."
                )
            if num_int > den_int:
                raise GroupTableConsistencyError(
                    f"numerator_count ({num_int}) exceeds denominator_count ({den_int}) "
                    f"for metric '{d.name}' in animal '{aid}'."
                )

            # Validate percentage
            pct = row["percentage"]
            if den_int == 0:
                if num_int != 0 or not pd.isna(pct):
                    raise GroupTableConsistencyError(
                        f"Zero denominator requires numerator=0 and percentage=NaN for metric '{d.name}' "
                        f"in animal '{aid}', got num={num_int}, pct={pct}."
                    )
            else:
                if type(pct) is bool or isinstance(pct, np.bool_) or not isinstance(pct, (int, float, np.floating, np.integer)) or not np.isfinite(pct):
                    raise GroupTableConsistencyError(
                        f"Invalid percentage value ({pct!r}) for metric '{d.name}' in animal '{aid}'."
                    )
                expected_pct = 100.0 * num_int / den_int
                if not math.isclose(float(pct), expected_pct, rel_tol=1e-9, abs_tol=1e-12):
                    raise GroupTableConsistencyError(
                        f"Corrupted percentage {pct} for metric '{d.name}' in animal '{aid}', "
                        f"expected {expected_pct}."
                    )

            # Validate filter provenance
            if row["denominator_filter"] != d.denominator.to_string():
                raise GroupTableConsistencyError(
                    f"Mismatched denominator_filter for metric '{d.name}' in animal '{aid}': "
                    f"expected '{d.denominator.to_string()}', got '{row['denominator_filter']}'."
                )
            if row["numerator_filter"] != d.effective_numerator.to_string():
                raise GroupTableConsistencyError(
                    f"Mismatched numerator_filter for metric '{d.name}' in animal '{aid}': "
                    f"expected '{d.effective_numerator.to_string()}', got '{row['numerator_filter']}'."
                )

    # Validate assignments sequence
    if not isinstance(assignments, Sequence) or len(assignments) == 0:
        raise InvalidGroupAssignmentError("At least one AnimalGroupAssignment must be provided.")

    assigned_animals: list[str] = []
    group_map: dict[str, str] = {}
    for idx, a in enumerate(assignments):
        if not isinstance(a, AnimalGroupAssignment):
            raise InvalidGroupAssignmentError(
                f"Item at index {idx} is not an AnimalGroupAssignment: {type(a).__name__}."
            )
        if a.animal_id in group_map:
            raise InvalidGroupAssignmentError(
                f"Duplicate assignment for animal '{a.animal_id}'."
            )
        group_map[a.animal_id] = a.group
        assigned_animals.append(a.animal_id)

    if set(assigned_animals) != set(canonical_animals):
        missing = set(canonical_animals) - set(assigned_animals)
        if missing:
            raise InvalidGroupAssignmentError(
                f"Missing group assignment for animal(s): {sorted(missing)}."
            )
        unknown = set(assigned_animals) - set(canonical_animals)
        if unknown:
            raise InvalidGroupAssignmentError(
                f"Assignment for unknown animal(s): {sorted(unknown)}."
            )


def prepare_group_tables(
    aggregation_result: object,
    assignments: Sequence[AnimalGroupAssignment],
) -> GroupTableResult:
    """Prepare long-form, GraphPad-friendly wide-form, companion animal ID, and manifest tables.

    Args:
        aggregation_result: AnimalAggregationResult from M7.
        assignments: Sequence of AnimalGroupAssignment objects assigning each animal to a group.

    Returns:
        GroupTableResult containing the long table, wide tables dict, animal ID table,
        manifest DataFrame, applied assignments tuple, and provenance reference.

    Raises:
        InvalidGroupAssignmentError: If assignments are invalid, missing, duplicate, or conflicting.
        GroupTableConsistencyError: If M7 result is malformed, inconsistent, or corrupted.
    """
    validate_group_assignments(aggregation_result, assignments)

    from nervenuc.aggregation import AnimalAggregationResult

    assert isinstance(aggregation_result, AnimalAggregationResult)

    defs = aggregation_result.definitions
    df = aggregation_result.animal_metrics

    canonical_animals = list(dict.fromkeys(df["animal_id"]))
    group_map = {a.animal_id: a.group for a in assignments}

    # Group ordering by first appearance in assignments
    groups: list[str] = []
    for a in assignments:
        if a.group not in groups:
            groups.append(a.group)

    # 1. Build Long Table (9 columns)
    long_rows: list[dict] = []
    for aid in canonical_animals:
        grp = group_map[aid]
        animal_df = df[df["animal_id"] == aid]
        for d in defs:
            row = animal_df[animal_df["metric_name"] == d.name].iloc[0]
            pct_val = float(row["percentage"]) if pd.notna(row["percentage"]) else float("nan")
            long_rows.append(
                {
                    "animal_id": aid,
                    "group": grp,
                    "metric_name": d.name,
                    "sample_count": int(row["sample_count"]),
                    "numerator_count": int(row["numerator_count"]),
                    "denominator_count": int(row["denominator_count"]),
                    "percentage": pct_val,
                    "numerator_filter": str(row["numerator_filter"]),
                    "denominator_filter": str(row["denominator_filter"]),
                }
            )

    long_table = pd.DataFrame(long_rows, columns=LONG_TABLE_COLUMNS)

    # Partition animals by group preserving canonical M7 animal order within group
    animals_by_group: dict[str, list[str]] = {g: [] for g in groups}
    for aid in canonical_animals:
        animals_by_group[group_map[aid]].append(aid)

    max_len = max(len(animals_by_group[g]) for g in groups)

    # 2. Build Animal ID Table
    id_table_data: dict[str, list] = {}
    for g in groups:
        grp_animals = animals_by_group[g]
        id_table_data[g] = list(grp_animals) + [np.nan] * (max_len - len(grp_animals))
    animal_id_table = pd.DataFrame(id_table_data)

    # 3. Build Wide Tables for each metric
    wide_tables: dict[str, pd.DataFrame] = {}
    for d in defs:
        wide_table_data: dict[str, list] = {}
        for g in groups:
            grp_animals = animals_by_group[g]
            pct_vals: list[float] = []
            for aid in grp_animals:
                match = df[(df["animal_id"] == aid) & (df["metric_name"] == d.name)]
                row = match.iloc[0]
                pct_val = float(row["percentage"]) if pd.notna(row["percentage"]) else float("nan")
                pct_vals.append(pct_val)
            wide_table_data[g] = pct_vals + [float("nan")] * (max_len - len(pct_vals))
        wide_tables[d.name] = pd.DataFrame(wide_table_data)

    # 4. Build Manifest Table (6 columns)
    manifest_rows: list[dict] = []
    for idx, d in enumerate(defs, start=1):
        safe_idx_str = f"{idx:03d}"
        manifest_rows.append(
            {
                "metric_index": idx,
                "metric_name": d.name,
                "csv_filename": f"metric_{safe_idx_str}.csv",
                "excel_sheet_name": f"Metric {safe_idx_str}",
                "numerator_filter": d.effective_numerator.to_string(),
                "denominator_filter": d.denominator.to_string(),
            }
        )
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)

    return GroupTableResult(
        long_table=long_table,
        wide_tables=wide_tables,
        animal_id_table=animal_id_table,
        manifest=manifest,
        assignments=tuple(assignments),
        aggregation_result=aggregation_result,
    )


def validate_group_table_result(result: object) -> None:
    """Validate all mutable M8 tables against regenerated M7 provenance."""
    if not isinstance(result, GroupTableResult):
        raise GroupTableConsistencyError(
            f"result must be a GroupTableResult, got {type(result).__name__}."
        )

    canonical = prepare_group_tables(result.aggregation_result, result.assignments)
    pairs = [
        (result.long_table, canonical.long_table, "long_table"),
        (result.animal_id_table, canonical.animal_id_table, "animal_id_table"),
        (result.manifest, canonical.manifest, "manifest"),
    ]
    for actual, expected, label in pairs:
        try:
            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise GroupTableConsistencyError(
                f"{label} does not match the canonical table regenerated from M7 provenance."
            ) from exc

    if set(result.wide_tables) != set(canonical.wide_tables):
        raise GroupTableConsistencyError(
            "wide_tables keys do not match canonical metric definitions."
        )
    for metric_name, expected in canonical.wide_tables.items():
        try:
            pd.testing.assert_frame_equal(
                result.wide_tables[metric_name],
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise GroupTableConsistencyError(
                f"wide_tables['{metric_name}'] does not match M7 provenance."
            ) from exc
