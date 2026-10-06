"""Engine for calculating group-level descriptive statistics from animal-level replicates."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats

from .errors import StatisticsConsistencyError
from .models import GroupStatisticsResult

REQUIRED_LONG_COLUMNS: list[str] = [
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

OBSERVATION_COLUMNS: list[str] = [
    "animal_id",
    "group",
    "metric_name",
    "percentage",
    "numerator_count",
    "denominator_count",
]

GROUP_STATISTICS_COLUMNS: list[str] = [
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


def validate_statistics_input(group_tables: object) -> None:
    """Validate structure, types, provenance, and numerical consistency of GroupTableResult input.

    Raises:
        StatisticsConsistencyError: If group_tables is malformed, missing metrics, has invalid
            scalar types, corrupted percentages, out-of-bounds percentages, or inconsistent provenance.
    """
    from nervenuc.export import AnimalGroupAssignment, GroupTableResult

    if not isinstance(group_tables, GroupTableResult):
        raise StatisticsConsistencyError(
            f"group_tables must be a GroupTableResult, got {type(group_tables).__name__}."
        )

    # 1. Validate canonical definitions from aggregation_result
    defs = group_tables.aggregation_result.definitions
    if not defs:
        raise StatisticsConsistencyError("aggregation_result.definitions cannot be empty.")

    def_names = [d.name for d in defs]
    if len(set(def_names)) != len(def_names):
        raise StatisticsConsistencyError("Duplicate metric names in aggregation_result.definitions.")

    # 2. Validate long_table schema
    df = group_tables.long_table
    if not isinstance(df, pd.DataFrame):
        raise StatisticsConsistencyError("long_table must be a DataFrame.")

    if list(df.columns) != REQUIRED_LONG_COLUMNS:
        raise StatisticsConsistencyError(
            f"long_table columns must match {REQUIRED_LONG_COLUMNS}, got {list(df.columns)}."
        )

    if len(df) == 0:
        raise StatisticsConsistencyError("long_table cannot be empty.")

    long_metric_names = set(df["metric_name"])
    if long_metric_names != set(def_names):
        raise StatisticsConsistencyError(
            "Mismatch between canonical metric definitions and long_table metric names."
        )

    # 3. Validate assignments
    assignments = group_tables.assignments
    if not isinstance(assignments, Sequence) or len(assignments) == 0:
        raise StatisticsConsistencyError("assignments cannot be empty.")

    assigned_animals: list[str] = []
    assignment_map: dict[str, str] = {}
    for idx, a in enumerate(assignments):
        if not isinstance(a, AnimalGroupAssignment):
            raise StatisticsConsistencyError(
                f"Item at index {idx} in assignments is not an AnimalGroupAssignment: {type(a).__name__}."
            )
        if a.animal_id in assignment_map:
            raise StatisticsConsistencyError(
                f"Duplicate assignment for animal '{a.animal_id}' in assignments."
            )
        assignment_map[a.animal_id] = a.group
        assigned_animals.append(a.animal_id)

    canonical_animals = list(dict.fromkeys(df["animal_id"]))
    if set(assigned_animals) != set(canonical_animals):
        missing = set(canonical_animals) - set(assigned_animals)
        if missing:
            raise StatisticsConsistencyError(f"Missing assignment for animal(s): {sorted(missing)}.")
        unknown = set(assigned_animals) - set(canonical_animals)
        if unknown:
            raise StatisticsConsistencyError(f"Assignment for unknown animal(s): {sorted(unknown)}.")

    # 4. Validate per-animal observations and metrics
    for aid in canonical_animals:
        animal_df = df[df["animal_id"] == aid]
        if len(animal_df) != len(defs):
            raise StatisticsConsistencyError(
                f"Animal '{aid}' has {len(animal_df)} rows in long_table, expected {len(defs)}."
            )

        first_sc: int | None = None
        for d in defs:
            match = animal_df[animal_df["metric_name"] == d.name]
            if len(match) != 1:
                raise StatisticsConsistencyError(
                    f"Missing expected metric row '{d.name}' for animal '{aid}'."
                )
            row = match.iloc[0]

            # Verify group alignment
            if row["group"] != assignment_map[aid]:
                raise StatisticsConsistencyError(
                    f"Group mismatch for animal '{aid}': expected '{assignment_map[aid]}', got '{row['group']}'."
                )

            # Strict sample_count scalar validation
            sc = row["sample_count"]
            if type(sc) is bool or isinstance(sc, np.bool_) or not (type(sc) is int or isinstance(sc, np.integer)):
                raise StatisticsConsistencyError(
                    f"Invalid sample_count type ({type(sc).__name__}) for animal '{aid}'."
                )
            sc_int = int(sc)
            if sc_int < 1:
                raise StatisticsConsistencyError(
                    f"sample_count must be >= 1, got {sc_int} for animal '{aid}'."
                )
            if first_sc is None:
                first_sc = sc_int
            elif sc_int != first_sc:
                raise StatisticsConsistencyError(
                    f"Inconsistent sample_count across metrics for animal '{aid}': {first_sc} vs {sc_int}."
                )

            # Strict count scalar validation
            num = row["numerator_count"]
            den = row["denominator_count"]
            if type(num) is bool or isinstance(num, np.bool_) or not (type(num) is int or isinstance(num, np.integer)):
                raise StatisticsConsistencyError(
                    f"Invalid numerator_count type ({type(num).__name__}) for metric '{d.name}' in animal '{aid}'."
                )
            if type(den) is bool or isinstance(den, np.bool_) or not (type(den) is int or isinstance(den, np.integer)):
                raise StatisticsConsistencyError(
                    f"Invalid denominator_count type ({type(den).__name__}) for metric '{d.name}' in animal '{aid}'."
                )

            num_int = int(num)
            den_int = int(den)
            if num_int < 0 or den_int < 0:
                raise StatisticsConsistencyError(
                    f"Negative count for metric '{d.name}' in animal '{aid}': num={num_int}, den={den_int}."
                )
            if num_int > den_int:
                raise StatisticsConsistencyError(
                    f"numerator_count ({num_int}) exceeds denominator_count ({den_int}) "
                    f"for metric '{d.name}' in animal '{aid}'."
                )

            # Strict percentage scalar validation
            pct = row["percentage"]
            if type(pct) is bool or isinstance(pct, np.bool_) or isinstance(pct, (str, bytes)) or pct is None:
                raise StatisticsConsistencyError(
                    f"Invalid percentage scalar type ({type(pct).__name__}) for metric '{d.name}' in animal '{aid}'."
                )

            if pd.isna(pct):
                if not isinstance(pct, (float, np.floating)):
                    raise StatisticsConsistencyError(
                        f"Missing percentage must be a float NaN, got {type(pct).__name__} for animal '{aid}'."
                    )
                if den_int != 0 or num_int != 0:
                    raise StatisticsConsistencyError(
                        f"NaN percentage only allowed when den=0 and num=0 for metric '{d.name}' in animal '{aid}', "
                        f"got num={num_int}, den={den_int}."
                    )
            else:
                if not isinstance(pct, (int, float, np.floating, np.integer)):
                    raise StatisticsConsistencyError(
                        f"Invalid percentage scalar type ({type(pct).__name__}) for metric '{d.name}' in animal '{aid}'."
                    )
                pct_float = float(pct)
                if not np.isfinite(pct_float):
                    raise StatisticsConsistencyError(
                        f"Non-finite percentage ({pct_float}) for metric '{d.name}' in animal '{aid}'."
                    )
                if pct_float < 0.0 or pct_float > 100.0:
                    raise StatisticsConsistencyError(
                        f"Percentage out of range [0, 100]: {pct_float} for metric '{d.name}' in animal '{aid}'."
                    )
                if den_int == 0:
                    raise StatisticsConsistencyError(
                        f"Finite percentage ({pct_float}) with zero denominator for metric '{d.name}' in animal '{aid}'."
                    )
                expected_pct = 100.0 * num_int / den_int
                if not math.isclose(pct_float, expected_pct, rel_tol=1e-9, abs_tol=1e-12):
                    raise StatisticsConsistencyError(
                        f"Corrupted percentage {pct_float} for metric '{d.name}' in animal '{aid}', "
                        f"expected {expected_pct}."
                    )

            # Strict filter provenance check
            if row["denominator_filter"] != d.denominator.to_string():
                raise StatisticsConsistencyError(
                    f"Mismatched denominator_filter for metric '{d.name}' in animal '{aid}': "
                    f"expected '{d.denominator.to_string()}', got '{row['denominator_filter']}'."
                )
            if row["numerator_filter"] != d.effective_numerator.to_string():
                raise StatisticsConsistencyError(
                    f"Mismatched numerator_filter for metric '{d.name}' in animal '{aid}': "
                    f"expected '{d.effective_numerator.to_string()}', got '{row['numerator_filter']}'."
                )


def prepare_analysis_observations(group_tables: object) -> pd.DataFrame:
    """Extract and normalize individual animal-level observations for downstream statistical analysis.

    Args:
        group_tables: Validated GroupTableResult from M8.

    Returns:
        6-column DataFrame containing animal_id, group, metric_name, percentage, numerator_count,
        and denominator_count (retaining NaN missing percentage observations).

    Raises:
        StatisticsConsistencyError: If group_tables fails input validation.
    """
    validate_statistics_input(group_tables)

    from nervenuc.export import GroupTableResult

    assert isinstance(group_tables, GroupTableResult)

    df = group_tables.long_table
    defs = group_tables.aggregation_result.definitions
    canonical_animals = list(dict.fromkeys(df["animal_id"]))

    obs_rows: list[dict] = []
    for aid in canonical_animals:
        animal_df = df[df["animal_id"] == aid]
        for d in defs:
            row = animal_df[animal_df["metric_name"] == d.name].iloc[0]
            pct_val = float(row["percentage"]) if pd.notna(row["percentage"]) else float("nan")
            obs_rows.append(
                {
                    "animal_id": aid,
                    "group": str(row["group"]),
                    "metric_name": d.name,
                    "percentage": pct_val,
                    "numerator_count": int(row["numerator_count"]),
                    "denominator_count": int(row["denominator_count"]),
                }
            )

    return pd.DataFrame(obs_rows, columns=OBSERVATION_COLUMNS)


def summarize_groups(group_tables: object) -> GroupStatisticsResult:
    """Calculate group-level descriptive statistics from animal-level biological replicate values.

    Args:
        group_tables: GroupTableResult from M8.

    Returns:
        GroupStatisticsResult containing 15-column group_statistics DataFrame,
        6-column observations DataFrame, and provenance reference to source.

    Raises:
        StatisticsConsistencyError: If group_tables fails validation.
    """
    validate_statistics_input(group_tables)

    from nervenuc.export import GroupTableResult

    assert isinstance(group_tables, GroupTableResult)

    obs_df = prepare_analysis_observations(group_tables)
    defs = group_tables.aggregation_result.definitions

    # Preserve group ordering from first appearance in assignments
    groups: list[str] = []
    for a in group_tables.assignments:
        if a.group not in groups:
            groups.append(a.group)

    stat_rows: list[dict] = []
    for g in groups:
        for d in defs:
            subset = obs_df[(obs_df["group"] == g) & (obs_df["metric_name"] == d.name)]
            n_total = len(subset)
            valid_pcts = [p for p in subset["percentage"] if pd.notna(p)]
            n_valid = len(valid_pcts)
            n_missing = n_total - n_valid

            if n_valid == 0:
                mean_val = median_val = sd_val = sem_val = ci_low = ci_high = min_val = max_val = float("nan")
            elif n_valid == 1:
                p = float(valid_pcts[0])
                mean_val = p
                median_val = p
                sd_val = float("nan")
                sem_val = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")
                min_val = p
                max_val = p
            else:
                mean_val = float(np.mean(valid_pcts))
                median_val = float(np.median(valid_pcts))
                sd_val = float(np.std(valid_pcts, ddof=1))
                sem_val = float(sd_val / math.sqrt(n_valid))
                min_val = float(min(valid_pcts))
                max_val = float(max(valid_pcts))

                if sd_val == 0.0:
                    ci_low = mean_val
                    ci_high = mean_val
                else:
                    t_crit = float(scipy.stats.t.ppf(0.975, df=n_valid - 1))
                    margin = t_crit * sem_val
                    ci_low = float(mean_val - margin)
                    ci_high = float(mean_val + margin)

            stat_rows.append(
                {
                    "group": g,
                    "metric_name": d.name,
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_missing": n_missing,
                    "mean": mean_val,
                    "median": median_val,
                    "sd": sd_val,
                    "sem": sem_val,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "min": min_val,
                    "max": max_val,
                    "numerator_filter": d.effective_numerator.to_string(),
                    "denominator_filter": d.denominator.to_string(),
                }
            )

    group_stats_df = pd.DataFrame(stat_rows, columns=GROUP_STATISTICS_COLUMNS)

    return GroupStatisticsResult(
        group_statistics=group_stats_df,
        observations=obs_df,
        source=group_tables,
    )
