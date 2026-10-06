"""Engine for evaluating biological replicate (animal-level) aggregation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import AggregationConsistencyError, InvalidSampleIdentityError
from .models import AnimalAggregationResult, SampleSummary

ANIMAL_METRIC_COLUMNS: list[str] = [
    "animal_id",
    "metric_name",
    "sample_count",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]


def validate_aggregation_consistency(samples: Sequence[SampleSummary]) -> None:
    """Validate consistency, metric compatibility, count types, and percentages across sample summaries.

    Raises:
        InvalidSampleIdentityError: If samples is empty, contains invalid items, or duplicate sample IDs.
        AggregationConsistencyError: If metric definitions are inconsistent, filter provenance is mismatched,
            counts are non-integral/negative/invalid, or percentages are corrupted.
    """
    if not isinstance(samples, Sequence) or len(samples) == 0:
        raise InvalidSampleIdentityError("At least one SampleSummary must be provided.")

    sample_ids: list[str] = []
    for idx, s in enumerate(samples):
        if not isinstance(s, SampleSummary):
            raise InvalidSampleIdentityError(
                f"Item at index {idx} is not a SampleSummary: {type(s).__name__}."
            )
        sample_ids.append(s.identity.sample_id)

    if len(set(sample_ids)) != len(sample_ids):
        raise InvalidSampleIdentityError("Duplicate sample_id found in aggregation request.")

    # Validate individual sample summary structure and internal consistency
    for s in samples:
        res = s.summary_result
        defs = res.definitions
        def_names = [d.name for d in defs]
        if len(set(def_names)) != len(def_names):
            raise AggregationConsistencyError(
                f"Duplicate metric names in definitions for sample '{s.identity.sample_id}'."
            )

        metrics_df = res.metrics
        if not isinstance(metrics_df, pd.DataFrame):
            raise AggregationConsistencyError(
                f"metrics for sample '{s.identity.sample_id}' must be a DataFrame."
            )

        df_names = list(metrics_df["metric_name"])
        if len(set(df_names)) != len(df_names):
            raise AggregationConsistencyError(
                f"Duplicate metric names in metrics table for sample '{s.identity.sample_id}'."
            )

        if set(def_names) != set(df_names) or len(defs) != len(metrics_df):
            raise AggregationConsistencyError(
                f"Mismatch between definitions and metrics table for sample '{s.identity.sample_id}'."
            )

        for d in defs:
            match = metrics_df[metrics_df["metric_name"] == d.name]
            if len(match) != 1:
                raise AggregationConsistencyError(
                    f"Expected exactly one row for metric '{d.name}' in sample '{s.identity.sample_id}'."
                )
            row = match.iloc[0]

            # Validate filter string provenance
            if row["denominator_filter"] != d.denominator.to_string():
                raise AggregationConsistencyError(
                    f"Mismatched denominator_filter for metric '{d.name}' in sample '{s.identity.sample_id}': "
                    f"expected '{d.denominator.to_string()}', got '{row['denominator_filter']}'."
                )
            if row["numerator_filter"] != d.effective_numerator.to_string():
                raise AggregationConsistencyError(
                    f"Mismatched numerator_filter for metric '{d.name}' in sample '{s.identity.sample_id}': "
                    f"expected '{d.effective_numerator.to_string()}', got '{row['numerator_filter']}'."
                )

            # Strict count-type validation
            num = row["numerator_count"]
            den = row["denominator_count"]

            if type(num) is bool or isinstance(num, np.bool_) or not (type(num) is int or isinstance(num, np.integer)):
                raise AggregationConsistencyError(
                    f"Invalid numerator_count type ({type(num).__name__}) for metric '{d.name}' "
                    f"in sample '{s.identity.sample_id}'."
                )
            if type(den) is bool or isinstance(den, np.bool_) or not (type(den) is int or isinstance(den, np.integer)):
                raise AggregationConsistencyError(
                    f"Invalid denominator_count type ({type(den).__name__}) for metric '{d.name}' "
                    f"in sample '{s.identity.sample_id}'."
                )

            num_int = int(num)
            den_int = int(den)

            if num_int < 0 or den_int < 0:
                raise AggregationConsistencyError(
                    f"Negative count for metric '{d.name}' in sample '{s.identity.sample_id}': "
                    f"num={num_int}, den={den_int}."
                )
            if num_int > den_int:
                raise AggregationConsistencyError(
                    f"numerator_count ({num_int}) exceeds denominator_count ({den_int}) "
                    f"for metric '{d.name}' in sample '{s.identity.sample_id}'."
                )

            # Percentage numerical consistency check
            pct = row["percentage"]
            if den_int == 0:
                if num_int != 0 or not pd.isna(pct):
                    raise AggregationConsistencyError(
                        f"Zero denominator requires numerator=0 and percentage=NaN for metric '{d.name}' "
                        f"in sample '{s.identity.sample_id}', got num={num_int}, pct={pct}."
                    )
            else:
                if type(pct) is bool or isinstance(pct, np.bool_) or not isinstance(pct, (int, float, np.floating, np.integer)) or not np.isfinite(pct):
                    raise AggregationConsistencyError(
                        f"Invalid percentage value ({pct!r}) for metric '{d.name}' in sample '{s.identity.sample_id}'."
                    )
                expected_pct = 100.0 * num_int / den_int
                if not math.isclose(float(pct), expected_pct, rel_tol=1e-9, abs_tol=1e-12):
                    raise AggregationConsistencyError(
                        f"Corrupted percentage {pct} for metric '{d.name}' in sample '{s.identity.sample_id}', "
                        f"expected {expected_pct}."
                    )

    # Validate semantic metric compatibility across samples against samples[0]
    canonical_defs = samples[0].summary_result.definitions
    canonical_map = {d.name: d for d in canonical_defs}

    for s in samples[1:]:
        s_defs = s.summary_result.definitions
        if len(s_defs) != len(canonical_defs):
            raise AggregationConsistencyError(
                f"Sample '{s.identity.sample_id}' has {len(s_defs)} metric definitions, "
                f"expected {len(canonical_defs)}."
            )
        for d in s_defs:
            if d.name not in canonical_map:
                raise AggregationConsistencyError(
                    f"Sample '{s.identity.sample_id}' contains unknown metric '{d.name}'."
                )
            canon_d = canonical_map[d.name]
            if d != canon_d:
                raise AggregationConsistencyError(
                    f"Incompatible metric definition for '{d.name}' in sample '{s.identity.sample_id}'."
                )


def aggregate_by_animal(samples: Sequence[SampleSummary]) -> AnimalAggregationResult:
    """Pool raw counts from image-level sample summaries to produce animal-level biological replicate summaries.

    Args:
        samples: Sequence of SampleSummary objects associating sample identities with M6 summaries.

    Returns:
        AnimalAggregationResult containing the 8-column animal metrics DataFrame,
        canonical metric definitions, and contributing sample summaries tuple.

    Raises:
        InvalidSampleIdentityError: If samples is empty or contains duplicate/invalid sample IDs.
        AggregationConsistencyError: If metric definitions or count data across samples are inconsistent.
    """
    validate_aggregation_consistency(samples)

    canonical_defs = samples[0].summary_result.definitions

    # Group samples by animal_id preserving order of first appearance
    animal_order: list[str] = []
    animal_groups: dict[str, list[SampleSummary]] = {}
    for s in samples:
        aid = s.identity.animal_id
        if aid not in animal_groups:
            animal_order.append(aid)
            animal_groups[aid] = []
        animal_groups[aid].append(s)

    rows: list[dict] = []
    for aid in animal_order:
        group_samples = animal_groups[aid]
        sample_count = len(group_samples)

        for d in canonical_defs:
            sum_num = 0
            sum_den = 0

            for s in group_samples:
                match = s.summary_result.metrics[s.summary_result.metrics["metric_name"] == d.name]
                row = match.iloc[0]
                sum_num += int(row["numerator_count"])
                sum_den += int(row["denominator_count"])

            if sum_den > 0:
                animal_pct = float(100.0 * sum_num / sum_den)
            else:
                animal_pct = float("nan")

            rows.append(
                {
                    "animal_id": aid,
                    "metric_name": d.name,
                    "sample_count": sample_count,
                    "numerator_count": sum_num,
                    "denominator_count": sum_den,
                    "percentage": animal_pct,
                    "numerator_filter": d.effective_numerator.to_string(),
                    "denominator_filter": d.denominator.to_string(),
                }
            )

    df = pd.DataFrame(rows, columns=ANIMAL_METRIC_COLUMNS)
    return AnimalAggregationResult(
        animal_metrics=df,
        definitions=tuple(canonical_defs),
        samples=tuple(samples),
    )
