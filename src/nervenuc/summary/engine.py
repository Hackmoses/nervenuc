"""Engine for evaluating denominator-aware biological metrics and summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from nervenuc.classification import ClassificationResult
from .errors import (
    InvalidMetricDefinitionError,
    SummaryConsistencyError,
    UnknownMetricMarkerError,
)
from .models import (
    BiologicalMetricDefinition,
    BiologicalSummaryResult,
    PhenotypeFilter,
)

SUMMARY_COLUMNS: list[str] = [
    "metric_name",
    "numerator_count",
    "denominator_count",
    "percentage",
    "numerator_filter",
    "denominator_filter",
]


def _empty_summary_dataframe() -> pd.DataFrame:
    """Return an empty summary DataFrame with the required 6-column schema."""
    return pd.DataFrame(
        {
            "metric_name": pd.Series(dtype="object"),
            "numerator_count": pd.Series(dtype="int64"),
            "denominator_count": pd.Series(dtype="int64"),
            "percentage": pd.Series(dtype="float64"),
            "numerator_filter": pd.Series(dtype="object"),
            "denominator_filter": pd.Series(dtype="object"),
        },
        columns=SUMMARY_COLUMNS,
    )


def _is_equal_provenance(val1: object, val2: object) -> bool:
    """Check equality between provenance metadata values, handling None and NaN."""
    if val1 is None and val2 is None:
        return True
    if pd.isna(val1) and pd.isna(val2):
        return True
    return bool(val1 == val2)


def validate_summary_consistency(
    classification_result: ClassificationResult,
    metrics: Sequence[BiologicalMetricDefinition],
) -> None:
    """Validate consistency of classifications and authoritative active nuclei for requested metrics.

    Raises:
        InvalidMetricDefinitionError: If metrics sequence is empty, contains duplicates,
            or contains invalid metric definitions.
        UnknownMetricMarkerError: If a filter specifies a marker not present in M5 thresholds.
        SummaryConsistencyError: If classification rows are missing, duplicated, reference
            stale nuclei, have non-boolean states, or have metadata discrepancies with reviewed nuclei.
    """
    if not isinstance(classification_result, ClassificationResult):
        raise SummaryConsistencyError(
            f"classification_result must be a ClassificationResult, got {type(classification_result).__name__}."
        )

    if not isinstance(metrics, Sequence) or len(metrics) == 0:
        raise InvalidMetricDefinitionError("At least one BiologicalMetricDefinition must be provided.")

    seen_metric_names: set[str] = set()
    for idx, m in enumerate(metrics):
        if not isinstance(m, BiologicalMetricDefinition):
            raise InvalidMetricDefinitionError(
                f"Item at index {idx} is not a BiologicalMetricDefinition: {type(m).__name__}."
            )
        if m.name in seen_metric_names:
            raise InvalidMetricDefinitionError(f"Duplicate metric name: {m.name!r}.")
        seen_metric_names.add(m.name)

    # Defensive validation of threshold marker uniqueness
    thresholds = classification_result.thresholds
    threshold_marker_names = [t.marker_name for t in thresholds]
    if len(set(threshold_marker_names)) != len(threshold_marker_names):
        raise SummaryConsistencyError(
            "Duplicate marker names found in classification_result.thresholds."
        )
    available_markers = set(threshold_marker_names)

    # Validate required markers against authoritative threshold universe
    required_markers: set[str] = set()
    for m in metrics:
        for marker in m.denominator.states:
            if marker not in available_markers:
                raise UnknownMetricMarkerError(
                    f"Marker {marker!r} referenced in denominator of metric {m.name!r} "
                    f"is not declared in classification_result.thresholds."
                )
            required_markers.add(marker)
        for marker in m.numerator.states:
            if marker not in available_markers:
                raise UnknownMetricMarkerError(
                    f"Marker {marker!r} referenced in numerator of metric {m.name!r} "
                    f"is not declared in classification_result.thresholds."
                )
            required_markers.add(marker)

    # Retrieve authoritative active nuclei
    meas_res = getattr(classification_result, "measurement_result", None)
    rev_res = getattr(meas_res, "reviewed_result", None)
    if rev_res is None or not hasattr(rev_res, "objects"):
        raise SummaryConsistencyError(
            "classification_result is missing underlying reviewed_result.objects table."
        )

    reviewed_objects = rev_res.objects
    if not isinstance(reviewed_objects, pd.DataFrame):
        raise SummaryConsistencyError("reviewed_result.objects must be a DataFrame.")

    active_labels = set(reviewed_objects["current_label"]) if len(reviewed_objects) > 0 else set()
    classifications_df = classification_result.classifications

    if not isinstance(classifications_df, pd.DataFrame):
        raise SummaryConsistencyError("classification_result.classifications must be a DataFrame.")

    # Validate classification rows for required markers
    for marker in required_markers:
        marker_rows = classifications_df[classifications_df["marker_name"] == marker]

        if len(marker_rows) != len(reviewed_objects):
            raise SummaryConsistencyError(
                f"Expected {len(reviewed_objects)} classification rows for required marker {marker!r}, "
                f"found {len(marker_rows)}."
            )

        marker_labels = list(marker_rows["current_label"])
        # Check for unknown/stale nuclei and duplicates
        if set(marker_labels) != active_labels or len(marker_labels) != len(active_labels):
            raise SummaryConsistencyError(
                f"Classification rows for marker {marker!r} do not match the authoritative active nuclei set."
            )

        for _, obj_row in reviewed_objects.iterrows():
            c_lbl = obj_row["current_label"]
            match = marker_rows[marker_rows["current_label"] == c_lbl]
            if len(match) != 1:
                raise SummaryConsistencyError(
                    f"Active nucleus label {c_lbl} missing classification row for marker {marker!r}."
                )
            meas_row = match.iloc[0]

            # Verify provenance
            for field in ("review_id", "current_label", "source_nucleus_id", "review_origin", "review_status"):
                obj_val = obj_row.get(field, None)
                meas_val = meas_row.get(field, None)
                if not _is_equal_provenance(obj_val, meas_val):
                    raise SummaryConsistencyError(
                        f"Provenance mismatch for nucleus label {c_lbl} on field {field!r}: "
                        f"reviewed={obj_val!r} vs classification={meas_val!r}."
                    )

            # Verify strict boolean type
            val = meas_row["is_positive"]
            if type(val) is not bool and not isinstance(val, np.bool_):
                raise SummaryConsistencyError(
                    f"Non-boolean is_positive value ({val!r} of type {type(val).__name__}) "
                    f"for nucleus '{obj_row['review_id']}' on marker {marker!r}."
                )


def _evaluate_biological_metrics(
    classifications_df: pd.DataFrame,
    reviewed_objects: pd.DataFrame,
    metrics: Sequence[BiologicalMetricDefinition],
) -> pd.DataFrame:
    """Internal dimension-independent core for calculating denominator-aware biological metrics."""
    if len(reviewed_objects) == 0:
        rows: list[dict] = []
        for m in metrics:
            rows.append(
                {
                    "metric_name": m.name,
                    "numerator_count": 0,
                    "denominator_count": 0,
                    "percentage": float("nan"),
                    "numerator_filter": m.effective_numerator.to_string(),
                    "denominator_filter": m.denominator.to_string(),
                }
            )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    # Determine required markers
    required_markers: set[str] = set()
    for m in metrics:
        required_markers.update(m.denominator.states.keys())
        required_markers.update(m.numerator.states.keys())

    # Build active nucleus state dictionary: current_label -> {marker: bool}
    active_labels = list(reviewed_objects["current_label"])
    nucleus_states: dict[int, dict[str, bool]] = {lbl: {} for lbl in active_labels}

    for marker in required_markers:
        marker_rows = classifications_df[classifications_df["marker_name"] == marker]
        for _, r in marker_rows.iterrows():
            c_lbl = int(r["current_label"])
            nucleus_states[c_lbl][marker] = bool(r["is_positive"])

    summary_rows: list[dict] = []
    for m in metrics:
        den_filter = m.denominator
        eff_num_filter = m.effective_numerator

        den_count = 0
        num_count = 0

        for c_lbl in active_labels:
            states = nucleus_states[c_lbl]
            matches_den = all(states[k] == v for k, v in den_filter.states.items())
            if matches_den:
                den_count += 1
                matches_num = all(states[k] == v for k, v in eff_num_filter.states.items())
                if matches_num:
                    num_count += 1

        if den_count > 0:
            percentage = float(100.0 * num_count / den_count)
        else:
            percentage = float("nan")

        summary_rows.append(
            {
                "metric_name": m.name,
                "numerator_count": num_count,
                "denominator_count": den_count,
                "percentage": percentage,
                "numerator_filter": eff_num_filter.to_string(),
                "denominator_filter": den_filter.to_string(),
            }
        )

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def summarize_metrics(
    classification_result: ClassificationResult,
    metrics: Sequence[BiologicalMetricDefinition],
) -> BiologicalSummaryResult:
    """Evaluate one or more denominator-aware biological metrics on a classification result.

    Args:
        classification_result: ClassificationResult containing per-nucleus classifications.
        metrics: Sequence of BiologicalMetricDefinition objects to evaluate.

    Returns:
        BiologicalSummaryResult containing the 6-column metrics DataFrame,
        evaluated definitions, and classification result provenance reference.
    """
    validate_summary_consistency(classification_result, metrics)
    df = _evaluate_biological_metrics(
        classifications_df=classification_result.classifications,
        reviewed_objects=classification_result.measurement_result.reviewed_result.objects,
        metrics=metrics,
    )
    return BiologicalSummaryResult(
        metrics=df,
        definitions=tuple(metrics),
        classification_result=classification_result,
    )


def summarize_metric(
    classification_result: ClassificationResult,
    metric: BiologicalMetricDefinition,
) -> BiologicalSummaryResult:
    """Evaluate a single denominator-aware biological metric on a classification result."""
    return summarize_metrics(classification_result, [metric])


def validate_summary3d_consistency(
    classification_result: Any,
    metrics: Sequence[BiologicalMetricDefinition],
) -> None:
    """Validate consistency of 3D classifications and active nuclei for requested metrics."""
    from nervenuc.classification import Classification3DResult

    if not isinstance(classification_result, Classification3DResult):
        raise SummaryConsistencyError(
            f"classification_result must be a Classification3DResult, got {type(classification_result).__name__}."
        )

    if not isinstance(metrics, Sequence) or len(metrics) == 0:
        raise InvalidMetricDefinitionError("At least one BiologicalMetricDefinition must be provided.")

    seen_metric_names: set[str] = set()
    for idx, m in enumerate(metrics):
        if not isinstance(m, BiologicalMetricDefinition):
            raise InvalidMetricDefinitionError(
                f"Item at index {idx} is not a BiologicalMetricDefinition: {type(m).__name__}."
            )
        if m.name in seen_metric_names:
            raise InvalidMetricDefinitionError(f"Duplicate metric name: {m.name!r}.")
        seen_metric_names.add(m.name)

    thresholds = classification_result.thresholds
    threshold_marker_names = [t.marker_name for t in thresholds]
    if len(set(threshold_marker_names)) != len(threshold_marker_names):
        raise SummaryConsistencyError(
            "Duplicate marker names found in classification_result.thresholds."
        )
    available_markers = set(threshold_marker_names)

    for m in metrics:
        for marker in m.denominator.states:
            if marker not in available_markers:
                raise UnknownMetricMarkerError(
                    f"Metric {m.name!r} denominator references marker {marker!r} "
                    f"not in applied thresholds: {sorted(available_markers)}."
                )
        for marker in m.numerator.states:
            if marker not in available_markers:
                raise UnknownMetricMarkerError(
                    f"Metric {m.name!r} numerator references marker {marker!r} "
                    f"not in applied thresholds: {sorted(available_markers)}."
                )

    reviewed_result = getattr(classification_result.measurement_result, "reviewed_result", None)
    if reviewed_result is None or not hasattr(reviewed_result, "objects"):
        raise SummaryConsistencyError("classification_result is missing reviewed_result.objects.")

    reviewed_objects = reviewed_result.objects
    classifications_df = classification_result.classifications

    if not isinstance(classifications_df, pd.DataFrame) or not isinstance(reviewed_objects, pd.DataFrame):
        raise SummaryConsistencyError("classifications and objects must be DataFrames.")

    for marker in available_markers:
        marker_rows = classifications_df[classifications_df["marker_name"] == marker]
        if len(marker_rows) != len(reviewed_objects):
            raise SummaryConsistencyError(
                f"Expected {len(reviewed_objects)} classification rows for marker {marker!r}, "
                f"found {len(marker_rows)}."
            )

        for _, obj_row in reviewed_objects.iterrows():
            c_lbl = obj_row["current_label"]
            match = marker_rows[marker_rows["current_label"] == c_lbl]
            if len(match) != 1:
                raise SummaryConsistencyError(
                    f"Active nucleus label {c_lbl} missing classification row for marker {marker!r}."
                )
            meas_row = match.iloc[0]

            for field in ("review_id", "current_label", "source_nucleus_id", "review_origin", "review_status"):
                obj_val = obj_row.get(field, None)
                meas_val = meas_row.get(field, None)
                if not _is_equal_provenance(obj_val, meas_val):
                    raise SummaryConsistencyError(
                        f"Provenance mismatch for nucleus label {c_lbl} on field {field!r}: "
                        f"reviewed={obj_val!r} vs classification={meas_val!r}."
                    )

            val = meas_row["is_positive"]
            if type(val) is not bool and not isinstance(val, np.bool_):
                raise SummaryConsistencyError(
                    f"Non-boolean is_positive value ({val!r}) for nucleus '{obj_row['review_id']}' on marker {marker!r}."
                )


def summarize_metrics_3d(
    classification_result: Any,
    metrics: Sequence[BiologicalMetricDefinition],
) -> BiologicalSummary3DResult:
    """Evaluate one or more denominator-aware biological metrics on a 3D classification result.

    Args:
        classification_result: M14 Classification3DResult containing per-nucleus classifications.
        metrics: Sequence of BiologicalMetricDefinition objects to evaluate.

    Returns:
        BiologicalSummary3DResult containing metrics DataFrame, definitions, and provenance.
    """
    from .models import BiologicalSummary3DResult

    validate_summary3d_consistency(classification_result, metrics)
    df = _evaluate_biological_metrics(
        classifications_df=classification_result.classifications,
        reviewed_objects=classification_result.measurement_result.reviewed_result.objects,
        metrics=metrics,
    )
    return BiologicalSummary3DResult(
        metrics=df,
        definitions=tuple(metrics),
        classification_result=classification_result,
    )
