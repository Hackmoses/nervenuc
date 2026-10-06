"""Engine for marker positivity thresholding and nucleus phenotype classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from nervenuc.measurements import MarkerMeasurementResult
from .errors import (
    ClassificationConsistencyError,
    InvalidThresholdDefinitionError,
    UnknownMarkerError,
)
from .models import ClassificationResult, MarkerThreshold

CLASSIFICATION_COLUMNS: list[str] = [
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

PHENOTYPE_COLUMNS: list[str] = [
    "review_id",
    "current_label",
    "source_nucleus_id",
    "review_origin",
    "review_status",
    "phenotype",
]


def _empty_classifications_dataframe() -> pd.DataFrame:
    """Return an empty classifications DataFrame with the required 11-column schema."""
    return pd.DataFrame(
        {
            "review_id": pd.Series(dtype="object"),
            "current_label": pd.Series(dtype="int64"),
            "marker_name": pd.Series(dtype="object"),
            "measurement_name": pd.Series(dtype="object"),
            "measurement_value": pd.Series(dtype="float64"),
            "threshold_value": pd.Series(dtype="float64"),
            "comparison": pd.Series(dtype="object"),
            "is_positive": pd.Series(dtype="bool"),
            "source_nucleus_id": pd.Series(dtype="object"),
            "review_origin": pd.Series(dtype="object"),
            "review_status": pd.Series(dtype="object"),
        },
        columns=CLASSIFICATION_COLUMNS,
    )


def _empty_phenotypes_dataframe() -> pd.DataFrame:
    """Return an empty phenotypes DataFrame with the required 6-column schema."""
    return pd.DataFrame(
        {
            "review_id": pd.Series(dtype="object"),
            "current_label": pd.Series(dtype="int64"),
            "source_nucleus_id": pd.Series(dtype="object"),
            "review_origin": pd.Series(dtype="object"),
            "review_status": pd.Series(dtype="object"),
            "phenotype": pd.Series(dtype="object"),
        },
        columns=PHENOTYPE_COLUMNS,
    )


def _is_equal_provenance(val1: object, val2: object) -> bool:
    """Check equality between provenance metadata values, handling None and NaN."""
    if val1 is None and val2 is None:
        return True
    if pd.isna(val1) and pd.isna(val2):
        return True
    return bool(val1 == val2)


def validate_classification_consistency(
    measurement_result: MarkerMeasurementResult,
    thresholds: Sequence[MarkerThreshold],
) -> None:
    """Validate consistency between measurements, active nuclei, and thresholds.

    Raises:
        InvalidThresholdDefinitionError: If thresholds are empty, malformed, or duplicated.
        UnknownMarkerError: If a threshold specifies an unknown marker name.
        ClassificationConsistencyError: If measurement data is missing, duplicated, non-finite,
            or desynchronized from reviewed active nuclei.
    """
    if not isinstance(measurement_result, MarkerMeasurementResult):
        raise ClassificationConsistencyError(
            f"measurement_result must be a MarkerMeasurementResult, got {type(measurement_result).__name__}."
        )

    if not isinstance(thresholds, Sequence) or len(thresholds) == 0:
        raise InvalidThresholdDefinitionError("At least one MarkerThreshold must be provided.")

    validated_thresholds: list[MarkerThreshold] = []
    seen_marker_names: set[str] = set()

    for idx, t in enumerate(thresholds):
        if not isinstance(t, MarkerThreshold):
            raise InvalidThresholdDefinitionError(
                f"Item at index {idx} is not a MarkerThreshold: {type(t).__name__}."
            )
        if t.marker_name in seen_marker_names:
            raise InvalidThresholdDefinitionError(
                f"Duplicate threshold definition for marker '{t.marker_name}'."
            )
        seen_marker_names.add(t.marker_name)
        validated_thresholds.append(t)

    measurements_df = measurement_result.measurements
    if not isinstance(measurements_df, pd.DataFrame):
        raise ClassificationConsistencyError("measurement_result.measurements must be a DataFrame.")

    reviewed_result = getattr(measurement_result, "reviewed_result", None)
    if reviewed_result is None or not hasattr(reviewed_result, "objects"):
        raise ClassificationConsistencyError(
            "measurement_result.reviewed_result is missing or does not have an objects table."
        )

    reviewed_objects = reviewed_result.objects
    if not isinstance(reviewed_objects, pd.DataFrame):
        raise ClassificationConsistencyError("reviewed_result.objects must be a DataFrame.")

    available_markers = set(measurements_df["marker_name"].values) if len(measurements_df) > 0 else set()

    for t in validated_thresholds:
        if t.marker_name not in available_markers and len(reviewed_objects) > 0:
            raise UnknownMarkerError(
                f"Marker '{t.marker_name}' not found in measurement result. "
                f"Available markers: {sorted(available_markers)}."
            )

        if t.measurement not in measurements_df.columns:
            raise InvalidThresholdDefinitionError(
                f"Measurement column '{t.measurement}' not found in measurements table."
            )

    # Check authoritative active nuclei row counts & metadata
    active_labels = list(reviewed_objects["current_label"]) if len(reviewed_objects) > 0 else []

    for t in validated_thresholds:
        marker_rows = measurements_df[measurements_df["marker_name"] == t.marker_name]

        if len(marker_rows) != len(reviewed_objects):
            raise ClassificationConsistencyError(
                f"Expected {len(reviewed_objects)} measurement rows for marker '{t.marker_name}', "
                f"found {len(marker_rows)}."
            )

        # Check for duplicate current_labels within this marker's rows
        marker_labels = list(marker_rows["current_label"])
        if len(set(marker_labels)) != len(marker_labels):
            raise ClassificationConsistencyError(
                f"Duplicate current_label measurement rows found for marker '{t.marker_name}'."
            )

        # Check matching active nucleus set and verify provenance
        for _, obj_row in reviewed_objects.iterrows():
            c_lbl = obj_row["current_label"]
            match = marker_rows[marker_rows["current_label"] == c_lbl]
            if len(match) != 1:
                raise ClassificationConsistencyError(
                    f"Active nucleus label {c_lbl} ('{obj_row['review_id']}') missing measurement row "
                    f"for marker '{t.marker_name}'."
                )
            meas_row = match.iloc[0]

            # Verify provenance fields
            for field in ("review_id", "current_label", "source_nucleus_id", "review_origin", "review_status"):
                obj_val = obj_row.get(field, None)
                meas_val = meas_row.get(field, None)
                if not _is_equal_provenance(obj_val, meas_val):
                    raise ClassificationConsistencyError(
                        f"Provenance mismatch for nucleus label {c_lbl} on field '{field}': "
                        f"reviewed={obj_val!r} vs measurement={meas_val!r}."
                    )

            # Verify finite measurement value
            val = meas_row[t.measurement]
            if pd.isna(val) or not np.isfinite(val):
                raise ClassificationConsistencyError(
                    f"Non-finite measurement value ({val}) for nucleus '{obj_row['review_id']}' "
                    f"on marker '{t.marker_name}' and statistic '{t.measurement}'."
                )


def _evaluate_phenotype_classification(
    measurements_df: pd.DataFrame,
    reviewed_objects: pd.DataFrame,
    thresholds: tuple[MarkerThreshold, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Internal dimension-independent core for marker classification and phenotype derivation."""
    if len(reviewed_objects) == 0:
        return _empty_classifications_dataframe(), _empty_phenotypes_dataframe()

    sorted_objects = reviewed_objects.sort_values("current_label", ascending=True)

    classification_rows: list[dict] = []
    phenotype_rows: list[dict] = []

    for _, obj_row in sorted_objects.iterrows():
        c_lbl = int(obj_row["current_label"])
        review_id = obj_row["review_id"]
        source_nuc_id = obj_row.get("source_nucleus_id", None)
        review_orig = obj_row.get("review_origin", "automated")
        review_stat = obj_row.get("review_status", "unreviewed")

        phenotype_components: list[str] = []

        for t in thresholds:
            match = measurements_df[
                (measurements_df["current_label"] == c_lbl)
                & (measurements_df["marker_name"] == t.marker_name)
            ]
            meas_row = match.iloc[0]
            meas_val = float(meas_row[t.measurement])

            if t.comparison == ">=":
                is_pos = bool(meas_val >= t.threshold)
            elif t.comparison == ">":
                is_pos = bool(meas_val > t.threshold)
            else:
                raise InvalidThresholdDefinitionError(f"Unsupported comparison: '{t.comparison}'.")

            classification_rows.append(
                {
                    "review_id": review_id,
                    "current_label": c_lbl,
                    "marker_name": t.marker_name,
                    "measurement_name": t.measurement,
                    "measurement_value": meas_val,
                    "threshold_value": float(t.threshold),
                    "comparison": t.comparison,
                    "is_positive": is_pos,
                    "source_nucleus_id": source_nuc_id,
                    "review_origin": review_orig,
                    "review_status": review_stat,
                }
            )

            state_str = f"{t.marker_name}+" if is_pos else f"{t.marker_name}-"
            phenotype_components.append(state_str)

        phenotype_str = " / ".join(phenotype_components)
        phenotype_rows.append(
            {
                "review_id": review_id,
                "current_label": c_lbl,
                "source_nucleus_id": source_nuc_id,
                "review_origin": review_orig,
                "review_status": review_stat,
                "phenotype": phenotype_str,
            }
        )

    class_df = pd.DataFrame(classification_rows, columns=CLASSIFICATION_COLUMNS)
    class_df["source_nucleus_id"] = pd.Series(
        [r["source_nucleus_id"] for r in classification_rows], dtype="object"
    )
    class_df["review_origin"] = pd.Series(
        [r["review_origin"] for r in classification_rows], dtype="object"
    )
    class_df["review_status"] = pd.Series(
        [r["review_status"] for r in classification_rows], dtype="object"
    )

    pheno_df = pd.DataFrame(phenotype_rows, columns=PHENOTYPE_COLUMNS)
    pheno_df["source_nucleus_id"] = pd.Series(
        [r["source_nucleus_id"] for r in phenotype_rows], dtype="object"
    )
    pheno_df["review_origin"] = pd.Series(
        [r["review_origin"] for r in phenotype_rows], dtype="object"
    )
    pheno_df["review_status"] = pd.Series(
        [r["review_status"] for r in phenotype_rows], dtype="object"
    )

    return class_df, pheno_df


def classify_markers(
    measurement_result: MarkerMeasurementResult,
    thresholds: Sequence[MarkerThreshold],
) -> ClassificationResult:
    """Classify per-nucleus marker positivity and derive multi-marker composite phenotypes.

    Args:
        measurement_result: M4 MarkerMeasurementResult containing raw per-nucleus measurements.
        thresholds: Sequence of MarkerThreshold definitions (at least one required).

    Returns:
        ClassificationResult containing long-form classifications, wide-form phenotypes,
        applied thresholds, and the measurement result reference.
    """
    validate_classification_consistency(measurement_result, thresholds)
    thresholds_tuple = tuple(thresholds)
    class_df, pheno_df = _evaluate_phenotype_classification(
        measurements_df=measurement_result.measurements,
        reviewed_objects=measurement_result.reviewed_result.objects,
        thresholds=thresholds_tuple,
    )
    return ClassificationResult(
        classifications=class_df,
        phenotypes=pheno_df,
        thresholds=thresholds_tuple,
        measurement_result=measurement_result,
    )


def classify_marker(
    measurement_result: MarkerMeasurementResult,
    threshold: MarkerThreshold,
) -> ClassificationResult:
    """Classify a single marker across all active nuclei."""
    return classify_markers(measurement_result, [threshold])


def validate_classification3d_consistency(
    measurement_result: Any,
    thresholds: Sequence[MarkerThreshold],
) -> None:
    """Validate consistency between 3D measurements, active nuclei, and thresholds."""
    from nervenuc.measurements3d import MarkerMeasurement3DResult

    if not isinstance(measurement_result, MarkerMeasurement3DResult):
        raise ClassificationConsistencyError(
            f"measurement_result must be a MarkerMeasurement3DResult, got {type(measurement_result).__name__}."
        )

    if not isinstance(thresholds, Sequence) or len(thresholds) == 0:
        raise InvalidThresholdDefinitionError("At least one MarkerThreshold must be provided.")

    validated_thresholds: list[MarkerThreshold] = []
    seen_marker_names: set[str] = set()

    for idx, t in enumerate(thresholds):
        if not isinstance(t, MarkerThreshold):
            raise InvalidThresholdDefinitionError(
                f"Item at index {idx} is not a MarkerThreshold: {type(t).__name__}."
            )
        if t.marker_name in seen_marker_names:
            raise InvalidThresholdDefinitionError(
                f"Duplicate threshold definition for marker '{t.marker_name}'."
            )
        seen_marker_names.add(t.marker_name)
        validated_thresholds.append(t)

    measurements_df = measurement_result.measurements
    if not isinstance(measurements_df, pd.DataFrame):
        raise ClassificationConsistencyError("measurement_result.measurements must be a DataFrame.")

    reviewed_result = getattr(measurement_result, "reviewed_result", None)
    if reviewed_result is None or not hasattr(reviewed_result, "objects"):
        raise ClassificationConsistencyError(
            "measurement_result.reviewed_result is missing or does not have an objects table."
        )

    reviewed_objects = reviewed_result.objects
    if not isinstance(reviewed_objects, pd.DataFrame):
        raise ClassificationConsistencyError("reviewed_result.objects must be a DataFrame.")

    available_markers = set(measurements_df["marker_name"].values) if len(measurements_df) > 0 else set()

    for t in validated_thresholds:
        if t.marker_name not in available_markers and len(reviewed_objects) > 0:
            raise UnknownMarkerError(
                f"Marker '{t.marker_name}' not found in 3D measurement result. "
                f"Available markers: {sorted(available_markers)}."
            )

        if t.measurement not in measurements_df.columns:
            raise InvalidThresholdDefinitionError(
                f"Measurement column '{t.measurement}' not found in 3D measurements table."
            )

    # Check active nuclei counts & provenance
    for t in validated_thresholds:
        marker_rows = measurements_df[measurements_df["marker_name"] == t.marker_name]

        if len(marker_rows) != len(reviewed_objects):
            raise ClassificationConsistencyError(
                f"Expected {len(reviewed_objects)} measurement rows for marker '{t.marker_name}', "
                f"found {len(marker_rows)}."
            )

        marker_labels = list(marker_rows["current_label"])
        if len(set(marker_labels)) != len(marker_labels):
            raise ClassificationConsistencyError(
                f"Duplicate current_label measurement rows found for marker '{t.marker_name}'."
            )

        for _, obj_row in reviewed_objects.iterrows():
            c_lbl = obj_row["current_label"]
            match = marker_rows[marker_rows["current_label"] == c_lbl]
            if len(match) != 1:
                raise ClassificationConsistencyError(
                    f"Active nucleus label {c_lbl} ('{obj_row['review_id']}') missing measurement row "
                    f"for marker '{t.marker_name}'."
                )
            meas_row = match.iloc[0]

            for field in ("review_id", "current_label", "source_nucleus_id", "review_origin", "review_status"):
                obj_val = obj_row.get(field, None)
                meas_val = meas_row.get(field, None)
                if not _is_equal_provenance(obj_val, meas_val):
                    raise ClassificationConsistencyError(
                        f"Provenance mismatch for nucleus label {c_lbl} on field '{field}': "
                        f"reviewed={obj_val!r} vs measurement={meas_val!r}."
                    )

            val = meas_row[t.measurement]
            if pd.isna(val) or not np.isfinite(val):
                raise ClassificationConsistencyError(
                    f"Non-finite measurement value ({val}) for nucleus '{obj_row['review_id']}' "
                    f"on marker '{t.marker_name}' and statistic '{t.measurement}'."
                )


def classify_markers_3d(
    measurement_result: Any,
    thresholds: Sequence[MarkerThreshold],
) -> Classification3DResult:
    """Classify per-nucleus marker positivity and derive composite phenotypes for 3D measurements.

    Args:
        measurement_result: M14 MarkerMeasurement3DResult containing raw per-nucleus measurements.
        thresholds: Sequence of MarkerThreshold definitions.

    Returns:
        Classification3DResult containing classifications and phenotypes.
    """
    from .models import Classification3DResult

    validate_classification3d_consistency(measurement_result, thresholds)
    thresholds_tuple = tuple(thresholds)
    class_df, pheno_df = _evaluate_phenotype_classification(
        measurements_df=measurement_result.measurements,
        reviewed_objects=measurement_result.reviewed_result.objects,
        thresholds=thresholds_tuple,
    )
    return Classification3DResult(
        classifications=class_df,
        phenotypes=pheno_df,
        thresholds=thresholds_tuple,
        measurement_result=measurement_result,
    )
