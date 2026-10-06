"""Engine for statistical analysis handoff and analysis-readiness reporting."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .errors import HandoffConsistencyError
from .models import AnalysisHandoffResult

REQUIRED_GROUP_STATS_COLUMNS: list[str] = [
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

REQUIRED_OBSERVATION_COLUMNS: list[str] = [
    "animal_id",
    "group",
    "metric_name",
    "percentage",
    "numerator_count",
    "denominator_count",
]

METRIC_READINESS_COLUMNS: list[str] = [
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

GROUP_READINESS_COLUMNS: list[str] = [
    "metric_name",
    "group",
    "n_total",
    "n_valid",
    "n_missing",
]


def validate_handoff_input(statistics_result: object) -> None:
    """Validate structure, types, provenance, and table integrity of GroupStatisticsResult.

    Raises:
        HandoffConsistencyError: If statistics_result is malformed, missing metrics, has count
            inconsistencies, or contains corrupted M8 export structures.
    """
    from nervenuc.export import AnimalGroupAssignment, GroupTableResult
    from nervenuc.statistics import GroupStatisticsResult

    if not isinstance(statistics_result, GroupStatisticsResult):
        raise HandoffConsistencyError(
            f"statistics_result must be a GroupStatisticsResult, got {type(statistics_result).__name__}."
        )

    # 1. Validate DataFrame types and schemas
    gs_df = statistics_result.group_statistics
    if not isinstance(gs_df, pd.DataFrame):
        raise HandoffConsistencyError("group_statistics must be a DataFrame.")
    if list(gs_df.columns) != REQUIRED_GROUP_STATS_COLUMNS:
        raise HandoffConsistencyError(
            f"group_statistics columns must match {REQUIRED_GROUP_STATS_COLUMNS}, got {list(gs_df.columns)}."
        )

    obs_df = statistics_result.observations
    if not isinstance(obs_df, pd.DataFrame):
        raise HandoffConsistencyError("observations must be a DataFrame.")
    if list(obs_df.columns) != REQUIRED_OBSERVATION_COLUMNS:
        raise HandoffConsistencyError(
            f"observations columns must match {REQUIRED_OBSERVATION_COLUMNS}, got {list(obs_df.columns)}."
        )

    source = statistics_result.source
    if not isinstance(source, GroupTableResult):
        raise HandoffConsistencyError(
            f"source must be a GroupTableResult, got {type(source).__name__}."
        )

    # 2. Validate canonical definitions
    defs = source.aggregation_result.definitions
    if not defs:
        raise HandoffConsistencyError("aggregation_result.definitions cannot be empty.")
    def_names = [d.name for d in defs]
    if len(set(def_names)) != len(def_names):
        raise HandoffConsistencyError("Duplicate canonical metric names in definitions.")

    if set(gs_df["metric_name"]) != set(def_names):
        raise HandoffConsistencyError(
            "Mismatch between canonical metric definitions and group_statistics metric names."
        )
    if set(obs_df["metric_name"]) != set(def_names):
        raise HandoffConsistencyError(
            "Mismatch between canonical metric definitions and observations metric names."
        )

    # 3. Validate authoritative assignments
    assigns = source.assignments
    if not isinstance(assigns, Sequence) or len(assigns) == 0:
        raise HandoffConsistencyError("assignments cannot be empty.")

    assign_map: dict[str, str] = {}
    auth_groups: list[str] = []
    for idx, a in enumerate(assigns):
        if not isinstance(a, AnimalGroupAssignment):
            raise HandoffConsistencyError(
                f"Item at index {idx} in assignments is not an AnimalGroupAssignment."
            )
        if a.animal_id in assign_map:
            raise HandoffConsistencyError(
                f"Duplicate assignment for animal '{a.animal_id}' in assignments."
            )
        assign_map[a.animal_id] = a.group
        if a.group not in auth_groups:
            auth_groups.append(a.group)

    obs_animals = list(dict.fromkeys(obs_df["animal_id"]))
    if set(obs_animals) != set(assign_map.keys()):
        missing = set(assign_map.keys()) - set(obs_animals)
        if missing:
            raise HandoffConsistencyError(f"Assigned animal(s) missing from observations: {sorted(missing)}.")
        unknown = set(obs_animals) - set(assign_map.keys())
        if unknown:
            raise HandoffConsistencyError(f"Observation animal(s) missing from assignments: {sorted(unknown)}.")

    # 4. Check per-animal observations integrity
    for aid in obs_animals:
        a_df = obs_df[obs_df["animal_id"] == aid]
        if len(a_df) != len(defs):
            raise HandoffConsistencyError(
                f"Animal '{aid}' has {len(a_df)} rows in observations, expected {len(defs)}."
            )
        if set(a_df["metric_name"]) != set(def_names):
            raise HandoffConsistencyError(
                f"Animal '{aid}' observations do not match canonical metric set."
            )
        for _, row in a_df.iterrows():
            if row["group"] != assign_map[aid]:
                raise HandoffConsistencyError(
                    f"Group mismatch in observations for animal '{aid}': "
                    f"expected '{assign_map[aid]}', got '{row['group']}'."
                )

    # 5. Check count types and cross-check recomputed counts
    for d in defs:
        for g in auth_groups:
            stat_rows = gs_df[(gs_df["metric_name"] == d.name) & (gs_df["group"] == g)]
            if len(stat_rows) != 1:
                raise HandoffConsistencyError(
                    f"Expected exactly 1 row for metric '{d.name}', group '{g}' in group_statistics, got {len(stat_rows)}."
                )
            s_row = stat_rows.iloc[0]

            if s_row["numerator_filter"] != d.effective_numerator.to_string():
                raise HandoffConsistencyError(
                    f"Mismatched numerator_filter for metric '{d.name}', group '{g}'."
                )
            if s_row["denominator_filter"] != d.denominator.to_string():
                raise HandoffConsistencyError(
                    f"Mismatched denominator_filter for metric '{d.name}', group '{g}'."
                )

            nt = s_row["n_total"]
            nv = s_row["n_valid"]
            nm = s_row["n_missing"]
            for c_val, c_name in [(nt, "n_total"), (nv, "n_valid"), (nm, "n_missing")]:
                if type(c_val) is bool or isinstance(c_val, np.bool_) or not (type(c_val) is int or isinstance(c_val, np.integer)):
                    raise HandoffConsistencyError(
                        f"Invalid count scalar type ({type(c_val).__name__}) for {c_name} in group '{g}', metric '{d.name}'."
                    )

            nt_int, nv_int, nm_int = int(nt), int(nv), int(nm)
            if nt_int < 1:
                raise HandoffConsistencyError(
                    f"n_total must be >= 1, got {nt_int} for group '{g}', metric '{d.name}'."
                )
            if nv_int < 0 or nm_int < 0 or nv_int > nt_int or nm_int > nt_int:
                raise HandoffConsistencyError(
                    f"Invalid count bounds (nt={nt_int}, nv={nv_int}, nm={nm_int}) for group '{g}', metric '{d.name}'."
                )
            if nt_int != nv_int + nm_int:
                raise HandoffConsistencyError(
                    f"n_total ({nt_int}) != n_valid ({nv_int}) + n_missing ({nm_int}) for group '{g}', metric '{d.name}'."
                )

            obs_subset = obs_df[(obs_df["metric_name"] == d.name) & (obs_df["group"] == g)]
            recomputed_nt = len(obs_subset)
            recomputed_nv = sum(1 for p in obs_subset["percentage"] if pd.notna(p) and np.isfinite(float(p)))
            recomputed_nm = recomputed_nt - recomputed_nv
            if nt_int != recomputed_nt or nv_int != recomputed_nv or nm_int != recomputed_nm:
                raise HandoffConsistencyError(
                    f"Count mismatch between observations and group_statistics for group '{g}', metric '{d.name}': "
                    f"stored (nt={nt_int}, nv={nv_int}, nm={nm_int}) vs observations (nt={recomputed_nt}, nv={recomputed_nv}, nm={recomputed_nm})."
                )

    # 6. Validate claimed M8 export structures
    wide_tables = source.wide_tables
    if not isinstance(wide_tables, dict):
        raise HandoffConsistencyError("source.wide_tables must be a dict.")
    if set(wide_tables.keys()) != set(def_names):
        raise HandoffConsistencyError("source.wide_tables keys do not match canonical metric definitions.")
    for d_name, w_df in wide_tables.items():
        if not isinstance(w_df, pd.DataFrame):
            raise HandoffConsistencyError(f"Wide table for metric '{d_name}' is not a DataFrame.")
        if list(w_df.columns) != auth_groups:
            raise HandoffConsistencyError(
                f"Wide table columns for metric '{d_name}' ({list(w_df.columns)}) do not match authoritative groups ({auth_groups})."
            )

    aid_df = source.animal_id_table
    if not isinstance(aid_df, pd.DataFrame):
        raise HandoffConsistencyError("source.animal_id_table must be a DataFrame.")
    if list(aid_df.columns) != auth_groups:
        raise HandoffConsistencyError(
            f"animal_id_table columns ({list(aid_df.columns)}) do not match authoritative groups ({auth_groups})."
        )

    manifest_df = source.manifest
    if not isinstance(manifest_df, pd.DataFrame):
        raise HandoffConsistencyError("source.manifest must be a DataFrame.")
    if "metric_name" not in manifest_df.columns or set(manifest_df["metric_name"]) != set(def_names) or len(manifest_df) != len(defs):
        raise HandoffConsistencyError(
            "source.manifest does not contain every canonical metric definition exactly once."
        )

    # 7. Regenerate M8 and M9 products from the authoritative M7 result and
    # assignments. Public DataFrames are mutable, so milestone handoff must
    # reject forged values rather than trusting internally consistent tables.
    try:
        from nervenuc.export import prepare_group_tables
        from nervenuc.statistics import summarize_groups

        canonical_source = prepare_group_tables(source.aggregation_result, source.assignments)
        canonical_statistics = summarize_groups(canonical_source)
    except Exception as exc:
        raise HandoffConsistencyError(
            f"Cannot regenerate canonical M8/M9 results from source provenance: {exc}"
        ) from exc

    frame_pairs = [
        (source.long_table, canonical_source.long_table, "source.long_table"),
        (source.animal_id_table, canonical_source.animal_id_table, "source.animal_id_table"),
        (source.manifest, canonical_source.manifest, "source.manifest"),
        (obs_df, canonical_statistics.observations, "observations"),
        (gs_df, canonical_statistics.group_statistics, "group_statistics"),
    ]
    for actual, expected, label in frame_pairs:
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
            raise HandoffConsistencyError(
                f"{label} does not match the canonical result regenerated from source provenance."
            ) from exc

    for metric_name in def_names:
        try:
            pd.testing.assert_frame_equal(
                source.wide_tables[metric_name],
                canonical_source.wide_tables[metric_name],
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise HandoffConsistencyError(
                f"source.wide_tables['{metric_name}'] does not match canonical source provenance."
            ) from exc


def prepare_analysis_handoff(statistics_result: object) -> AnalysisHandoffResult:
    """Assemble the data-handoff readiness report from a validated GroupStatisticsResult.

    Args:
        statistics_result: Validated GroupStatisticsResult from M9.

    Returns:
        AnalysisHandoffResult containing 14-column metric_readiness DataFrame,
        5-column group_readiness DataFrame, and provenance reference to source.

    Raises:
        HandoffConsistencyError: If statistics_result fails validation.
    """
    validate_handoff_input(statistics_result)

    from nervenuc.statistics import GroupStatisticsResult

    assert isinstance(statistics_result, GroupStatisticsResult)

    source = statistics_result.source
    defs = source.aggregation_result.definitions

    # Authoritative group order from assignments
    auth_groups: list[str] = []
    for a in source.assignments:
        if a.group not in auth_groups:
            auth_groups.append(a.group)

    gs_df = statistics_result.group_statistics

    # 1. Build group readiness table
    group_rows: list[dict] = []
    for d in defs:
        for g in auth_groups:
            s_row = gs_df[(gs_df["metric_name"] == d.name) & (gs_df["group"] == g)].iloc[0]
            group_rows.append(
                {
                    "metric_name": d.name,
                    "group": g,
                    "n_total": int(s_row["n_total"]),
                    "n_valid": int(s_row["n_valid"]),
                    "n_missing": int(s_row["n_missing"]),
                }
            )
    group_readiness_df = pd.DataFrame(group_rows, columns=GROUP_READINESS_COLUMNS)

    # 2. Build metric readiness table
    metric_rows: list[dict] = []
    for d in defs:
        g_subset = [r for r in group_rows if r["metric_name"] == d.name]
        group_count = len(auth_groups)
        total_animals = sum(r["n_total"] for r in g_subset)
        total_valid = sum(r["n_valid"] for r in g_subset)
        total_missing = sum(r["n_missing"] for r in g_subset)
        has_missing_values = bool(total_missing > 0)

        n_total_list = [r["n_total"] for r in g_subset]
        n_valid_list = [r["n_valid"] for r in g_subset]
        total_group_sizes_equal = bool(len(set(n_total_list)) <= 1)
        valid_group_sizes_equal = bool(len(set(n_valid_list)) <= 1)
        all_groups_have_valid_observations = bool(all(nv > 0 for nv in n_valid_list))

        if any(nv == 0 for nv in n_valid_list):
            readiness_status = "not_ready"
        elif total_missing > 0:
            readiness_status = "ready_with_missing_values"
        else:
            readiness_status = "ready"

        metric_rows.append(
            {
                "metric_name": d.name,
                "biological_replicate": "animal",
                "group_count": group_count,
                "total_animals": total_animals,
                "total_valid": total_valid,
                "total_missing": total_missing,
                "has_missing_values": has_missing_values,
                "total_group_sizes_equal": total_group_sizes_equal,
                "valid_group_sizes_equal": valid_group_sizes_equal,
                "all_groups_have_valid_observations": all_groups_have_valid_observations,
                "readiness_status": readiness_status,
                "numerator_filter": d.effective_numerator.to_string(),
                "denominator_filter": d.denominator.to_string(),
                "inferential_test": "not_selected",
            }
        )

    metric_readiness_df = pd.DataFrame(metric_rows, columns=METRIC_READINESS_COLUMNS)

    return AnalysisHandoffResult(
        metric_readiness=metric_readiness_df,
        group_readiness=group_readiness_df,
        source=statistics_result,
    )
