"""Engine for per-nucleus marker signal measurement and consistency validation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from nervenuc.markers import InvalidMarkerDefinitionError, MarkerDefinition
from nervenuc.qc import ReviewedSegmentationResult, validate_review_completion
from .errors import InvalidMarkerImageError, MeasurementConsistencyError
from .models import MarkerMeasurementResult

MEASUREMENT_COLUMNS: list[str] = [
    "review_id",
    "current_label",
    "marker_name",
    "pixel_count",
    "mean_intensity",
    "median_intensity",
    "min_intensity",
    "max_intensity",
    "integrated_intensity",
    "source_nucleus_id",
    "review_origin",
    "review_status",
]


def _empty_measurements_dataframe() -> pd.DataFrame:
    """Return an empty measurements DataFrame with the required 12-column schema."""
    return pd.DataFrame(
        {
            "review_id": pd.Series(dtype="object"),
            "current_label": pd.Series(dtype="int64"),
            "marker_name": pd.Series(dtype="object"),
            "pixel_count": pd.Series(dtype="int64"),
            "mean_intensity": pd.Series(dtype="float64"),
            "median_intensity": pd.Series(dtype="float64"),
            "min_intensity": pd.Series(dtype="float64"),
            "max_intensity": pd.Series(dtype="float64"),
            "integrated_intensity": pd.Series(dtype="float64"),
            "source_nucleus_id": pd.Series(dtype="object"),
            "review_origin": pd.Series(dtype="object"),
            "review_status": pd.Series(dtype="object"),
        },
        columns=MEASUREMENT_COLUMNS,
    )


def validate_reviewed_result_consistency(
    reviewed_result: ReviewedSegmentationResult,
) -> None:
    """Validate structural consistency of a ReviewedSegmentationResult.

    Raises:
        MeasurementConsistencyError: If labels or objects table are inconsistent.
    """
    if not isinstance(reviewed_result, ReviewedSegmentationResult):
        raise MeasurementConsistencyError(
            f"reviewed_result must be a ReviewedSegmentationResult, got {type(reviewed_result).__name__}."
        )

    labels = reviewed_result.labels
    objects = reviewed_result.objects

    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise MeasurementConsistencyError("reviewed_result.labels must be a 2D numpy.ndarray.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise MeasurementConsistencyError("reviewed_result.labels must have an integer dtype.")

    if not isinstance(objects, pd.DataFrame):
        raise MeasurementConsistencyError("reviewed_result.objects must be a pandas DataFrame.")

    required_cols = {"review_id", "current_label", "area_px"}
    missing = required_cols - set(objects.columns)
    if missing:
        raise MeasurementConsistencyError(
            f"reviewed_result.objects is missing required columns: {missing}."
        )

    unique_pixel_labels = np.unique(labels)
    if np.any(unique_pixel_labels < 0):
        raise MeasurementConsistencyError("Negative pixel label values found in reviewed_result.labels.")

    positive_pixel_labels = [int(v) for v in unique_pixel_labels if v > 0]
    n_active_pixels = len(positive_pixel_labels)

    current_labels = list(objects["current_label"])
    review_ids = list(objects["review_id"])

    # Check duplicates in object table
    if len(set(current_labels)) != len(current_labels):
        raise MeasurementConsistencyError("Duplicate current_label values found in reviewed_result.objects.")

    if len(set(review_ids)) != len(review_ids):
        raise MeasurementConsistencyError("Duplicate review_id values found in reviewed_result.objects.")

    if n_active_pixels > 0:
        # Check contiguous 1..N
        expected_contiguous = list(range(1, n_active_pixels + 1))
        if sorted(positive_pixel_labels) != expected_contiguous:
            raise MeasurementConsistencyError(
                f"Active positive labels in reviewed_result.labels must be contiguous 1..N, got {sorted(positive_pixel_labels)}."
            )

        if set(current_labels) != set(positive_pixel_labels):
            raise MeasurementConsistencyError(
                "Mismatch between object table current_label set and positive labels in label array."
            )
    else:
        if len(objects) != 0:
            raise MeasurementConsistencyError(
                "reviewed_result.labels has no positive labels but objects DataFrame is not empty."
            )

    # Validate exact mask pixel counts against area_px
    for _, row in objects.iterrows():
        lbl = int(row["current_label"])
        expected_area = int(row["area_px"])
        actual_pixel_count = int(np.sum(labels == lbl))
        if actual_pixel_count != expected_area:
            raise MeasurementConsistencyError(
                f"Pixel count in labels ({actual_pixel_count}) does not match area_px ({expected_area}) "
                f"for nucleus '{row['review_id']}'."
            )


def validate_marker_image(
    marker_image: np.ndarray,
    expected_shape: tuple[int, int],
    name: str = "Marker image",
) -> None:
    """Validate that a 2D marker image meets measurement requirements.

    Raises:
        InvalidMarkerImageError: If the marker image fails validation.
    """
    if not isinstance(marker_image, np.ndarray):
        raise InvalidMarkerImageError(
            f"{name} must be a numpy.ndarray, got {type(marker_image).__name__}."
        )

    if marker_image.ndim != 2:
        raise InvalidMarkerImageError(
            f"{name} must be 2-dimensional (Y, X), got ndim={marker_image.ndim} with shape {marker_image.shape}."
        )

    if marker_image.shape != expected_shape:
        raise InvalidMarkerImageError(
            f"{name} shape {marker_image.shape} does not match reviewed label shape {expected_shape}."
        )

    if (
        not np.issubdtype(marker_image.dtype, np.number)
        or np.issubdtype(marker_image.dtype, np.complexfloating)
        or marker_image.dtype == bool
    ):
        raise InvalidMarkerImageError(
            f"{name} must have a real numeric dtype, got '{marker_image.dtype}'."
        )

    if marker_image.size == 0:
        raise InvalidMarkerImageError(f"{name} cannot be empty.")

    if not np.isfinite(marker_image).all():
        raise InvalidMarkerImageError(f"{name} contains non-finite values (NaN or Inf).")


def measure_markers(
    reviewed_result: ReviewedSegmentationResult,
    marker_images: Mapping[str, np.ndarray],
) -> MarkerMeasurementResult:
    """Measure raw signal across active reviewed nuclei for one or more 2D marker images.

    Args:
        reviewed_result: Reviewed segmentation result containing active nuclei.
        marker_images: Mapping from user-defined marker names to 2D numeric NumPy arrays.

    Returns:
        MarkerMeasurementResult containing the long-form measurements DataFrame,
        the tuple of MarkerDefinition objects, and the reviewed result reference.

    Raises:
        MeasurementConsistencyError: If the reviewed result has structural inconsistencies.
        InvalidMarkerImageError: If any marker image fails validation.
        InvalidMarkerDefinitionError: If marker names are invalid or duplicated.
    """
    validate_reviewed_result_consistency(reviewed_result)
    validate_review_completion(reviewed_result)

    if not isinstance(marker_images, Mapping):
        raise InvalidMarkerImageError(
            f"marker_images must be a Mapping, got {type(marker_images).__name__}."
        )

    markers_list: list[MarkerDefinition] = []
    validated_images: list[tuple[MarkerDefinition, np.ndarray]] = []

    for k, img in marker_images.items():
        if isinstance(k, MarkerDefinition):
            marker_def = k
        elif isinstance(k, str):
            marker_def = MarkerDefinition(name=k)
        else:
            raise InvalidMarkerDefinitionError(
                f"Marker key must be str or MarkerDefinition, got {type(k).__name__}."
            )

        validate_marker_image(
            img,
            reviewed_result.labels.shape,
            f"Marker image '{marker_def.name}'",
        )
        markers_list.append(marker_def)
        validated_images.append((marker_def, img))

    names = [m.name for m in markers_list]
    if len(set(names)) != len(names):
        raise InvalidMarkerDefinitionError("Duplicate marker names provided in marker_images.")

    if len(reviewed_result.objects) == 0 or len(markers_list) == 0:
        return MarkerMeasurementResult(
            measurements=_empty_measurements_dataframe(),
            markers=tuple(markers_list),
            reviewed_result=reviewed_result,
        )

    # Sort nuclei by current_label ascending
    sorted_objects = reviewed_result.objects.sort_values("current_label", ascending=True)
    labels = reviewed_result.labels

    rows: list[dict] = []
    for _, obj_row in sorted_objects.iterrows():
        c_lbl = int(obj_row["current_label"])
        nucleus_mask = (labels == c_lbl)

        for marker_def, img in validated_images:
            pixels = img[nucleus_mask]
            pixel_count = int(len(pixels))
            mean_val = float(np.mean(pixels))
            median_val = float(np.median(pixels))
            min_val = float(np.min(pixels))
            max_val = float(np.max(pixels))
            sum_val = float(np.sum(pixels, dtype=np.float64))

            rows.append(
                {
                    "review_id": obj_row["review_id"],
                    "current_label": c_lbl,
                    "marker_name": marker_def.name,
                    "pixel_count": pixel_count,
                    "mean_intensity": mean_val,
                    "median_intensity": median_val,
                    "min_intensity": min_val,
                    "max_intensity": max_val,
                    "integrated_intensity": sum_val,
                    "source_nucleus_id": obj_row.get("source_nucleus_id", None),
                    "review_origin": obj_row.get("review_origin", "automated"),
                    "review_status": obj_row.get("review_status", "unreviewed"),
                }
            )

    df = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
    df["source_nucleus_id"] = pd.Series([r["source_nucleus_id"] for r in rows], dtype="object")
    df["review_origin"] = pd.Series([r["review_origin"] for r in rows], dtype="object")
    df["review_status"] = pd.Series([r["review_status"] for r in rows], dtype="object")

    return MarkerMeasurementResult(
        measurements=df,
        markers=tuple(markers_list),
        reviewed_result=reviewed_result,
    )


def measure_marker(
    reviewed_result: ReviewedSegmentationResult,
    marker_image: np.ndarray,
    marker: MarkerDefinition | str,
) -> MarkerMeasurementResult:
    """Measure raw signal for a single 2D marker channel across all active reviewed nuclei.

    Args:
        reviewed_result: Reviewed segmentation result containing active nuclei.
        marker_image: 2D numeric NumPy array for the marker channel.
        marker: MarkerDefinition instance or marker name string.

    Returns:
        MarkerMeasurementResult containing the per-nucleus measurements.
    """
    if isinstance(marker, str):
        m_name = marker
    elif isinstance(marker, MarkerDefinition):
        m_name = marker.name
    else:
        raise InvalidMarkerDefinitionError(
            f"marker must be a str or MarkerDefinition, got {type(marker).__name__}."
        )

    return measure_markers(reviewed_result, {m_name: marker_image})
