"""Engine for per-nucleus 3D marker signal measurement and consistency validation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import PreparedVolumeSet, VolumePreparationProvenance
from nervenuc.qc3d import ReviewedSegmentation3DResult

from .errors import (
    InvalidMarkerVolumeError,
    Measurement3DConsistencyError,
    NerveNucMeasurement3DError,
)
from .models import MarkerMeasurement3DResult

MEASUREMENT3D_COLUMNS: list[str] = [
    "review_id",
    "current_label",
    "marker_name",
    "voxel_count",
    "mean_intensity",
    "median_intensity",
    "min_intensity",
    "max_intensity",
    "integrated_intensity",
    "source_nucleus_id",
    "review_origin",
    "review_status",
]


def _empty_measurements3d_dataframe() -> pd.DataFrame:
    """Return an empty 3D measurements DataFrame with the required schema."""
    return pd.DataFrame(
        {
            "review_id": pd.Series(dtype="object"),
            "current_label": pd.Series(dtype="int64"),
            "marker_name": pd.Series(dtype="object"),
            "voxel_count": pd.Series(dtype="int64"),
            "mean_intensity": pd.Series(dtype="float64"),
            "median_intensity": pd.Series(dtype="float64"),
            "min_intensity": pd.Series(dtype="float64"),
            "max_intensity": pd.Series(dtype="float64"),
            "integrated_intensity": pd.Series(dtype="float64"),
            "source_nucleus_id": pd.Series(dtype="object"),
            "review_origin": pd.Series(dtype="object"),
            "review_status": pd.Series(dtype="object"),
        },
        columns=MEASUREMENT3D_COLUMNS,
    )


def validate_measurement3d_consistency(
    result: MarkerMeasurement3DResult,
    current_source_id: str,
    volume: PreparedVolumeSet | None = None,
) -> None:
    """Validate consistency between 3D measurements, active reviewed nuclei, and source volume.

    Args:
        result: MarkerMeasurement3DResult to validate.
        current_source_id: Expected authoritative source identifier.
        volume: Optional PreparedVolumeSet to verify volume alignment.

    Raises:
        Measurement3DConsistencyError: If measurements are corrupt, missing, or mismatched.
    """
    if not isinstance(result, MarkerMeasurement3DResult):
        raise Measurement3DConsistencyError(
            f"result must be a MarkerMeasurement3DResult, got {type(result).__name__}."
        )

    if result.source_id != current_source_id:
        raise Measurement3DConsistencyError(
            f"MarkerMeasurement3DResult source_id '{result.source_id}' does not match "
            f"current execution source_id '{current_source_id}'."
        )

    if result.reviewed_result.source_id != current_source_id:
        raise Measurement3DConsistencyError(
            f"ReviewedSegmentation3DResult source_id '{result.reviewed_result.source_id}' "
            f"does not match current execution source_id '{current_source_id}'."
        )

    active_objects = result.reviewed_result.active_objects
    num_active = len(active_objects)
    num_markers = len(result.marker_names)

    if num_markers == 0:
        raise Measurement3DConsistencyError("MarkerMeasurement3DResult contains no markers.")

    expected_rows = num_active * num_markers
    if len(result.measurements) != expected_rows:
        raise Measurement3DConsistencyError(
            f"Expected {expected_rows} measurement rows ({num_active} active nuclei x {num_markers} markers), "
            f"got {len(result.measurements)}."
        )

    if volume is not None:
        if not isinstance(volume, PreparedVolumeSet):
            raise Measurement3DConsistencyError(
                f"volume must be a PreparedVolumeSet, got {type(volume).__name__}."
            )
        for m_name in result.marker_names:
            if m_name not in volume.marker_volumes:
                raise Measurement3DConsistencyError(
                    f"Marker '{m_name}' not found in PreparedVolumeSet marker_volumes."
                )
            m_vol = volume.marker_volumes[m_name]
            if m_vol.shape != result.reviewed_result.labels.shape:
                raise Measurement3DConsistencyError(
                    f"Marker '{m_name}' shape {m_vol.shape} does not match label volume shape {result.reviewed_result.labels.shape}."
                )


def measure_markers_3d(
    reviewed_result: ReviewedSegmentation3DResult,
    marker_volumes: Mapping[str, np.ndarray],
    volume_provenance: VolumePreparationProvenance,
    markers: Sequence[MarkerDefinition] | None = None,
) -> MarkerMeasurement3DResult:
    """Measure per-nucleus raw marker signal within 3D nuclear volumes.

    Args:
        reviewed_result: Finalized ReviewedSegmentation3DResult.
        marker_volumes: Mapping of marker names to 3D NumPy arrays (Z, Y, X).
        volume_provenance: VolumePreparationProvenance describing extraction facts.
        markers: Optional sequence of MarkerDefinition objects specifying markers to measure.

    Returns:
        MarkerMeasurement3DResult containing structured measurements.

    Raises:
        NerveNucMeasurement3DError: If volumes are misaligned or data is invalid.
    """
    if not isinstance(reviewed_result, ReviewedSegmentation3DResult):
        raise NerveNucMeasurement3DError(
            f"reviewed_result must be a ReviewedSegmentation3DResult, got {type(reviewed_result).__name__}."
        )

    if not isinstance(marker_volumes, Mapping) or not marker_volumes:
        raise InvalidMarkerVolumeError("marker_volumes must be a non-empty Mapping of marker names to 3D arrays.")

    if not isinstance(volume_provenance, VolumePreparationProvenance):
        raise NerveNucMeasurement3DError(
            f"volume_provenance must be a VolumePreparationProvenance, got {type(volume_provenance).__name__}."
        )

    labels = reviewed_result.labels
    z_dim, y_dim, x_dim = labels.shape

    if markers is not None:
        target_marker_names = tuple(m.name if isinstance(m, MarkerDefinition) else str(m) for m in markers)
    else:
        target_marker_names = tuple(marker_volumes.keys())

    # Validate marker volume shapes and numeric types
    for m_name in target_marker_names:
        if m_name not in marker_volumes:
            raise InvalidMarkerVolumeError(f"Marker '{m_name}' not provided in marker_volumes.")
        m_vol = marker_volumes[m_name]
        if not isinstance(m_vol, np.ndarray) or m_vol.ndim != 3:
            raise InvalidMarkerVolumeError(
                f"Marker '{m_name}' volume must be a 3D numpy array, got {getattr(m_vol, 'ndim', None)}D."
            )
        if m_vol.shape != labels.shape:
            raise InvalidMarkerVolumeError(
                f"Marker '{m_name}' volume shape {m_vol.shape} does not match nuclear label volume shape {labels.shape}."
            )
        if not np.issubdtype(m_vol.dtype, np.number) or m_vol.dtype == bool or np.iscomplexobj(m_vol):
            raise InvalidMarkerVolumeError(f"Marker '{m_name}' volume dtype '{m_vol.dtype}' is not numeric.")
        if np.issubdtype(m_vol.dtype, np.floating) and not np.all(np.isfinite(m_vol)):
            raise InvalidMarkerVolumeError(f"Non-finite values encountered in marker '{m_name}' volume.")

    active_objects = reviewed_result.active_objects
    if len(active_objects) == 0:
        return MarkerMeasurement3DResult(
            measurements=_empty_measurements3d_dataframe(),
            reviewed_result=reviewed_result,
            marker_names=target_marker_names,
            source_id=reviewed_result.source_id,
            volume_provenance=volume_provenance,
        )

    measurement_rows: list[dict] = []

    # Iterate over active nuclei using bounding-box cropping for bounded memory
    for _, obj_row in active_objects.iterrows():
        c_lbl = int(obj_row["current_label"])
        review_id = obj_row["review_id"]
        source_nuc_id = str(obj_row.get("source_nucleus_id", c_lbl))
        review_orig = str(obj_row.get("review_origin", "automated"))
        review_stat = str(obj_row.get("review_status", "accepted"))

        z_min = int(obj_row["z_min"])
        z_max = int(obj_row["z_max"])
        y_min = int(obj_row["y_min"])
        y_max = int(obj_row["y_max"])
        x_min = int(obj_row["x_min"])
        x_max = int(obj_row["x_max"])

        # Slice subvolumes within bounding box
        lbl_sub = labels[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        mask_sub = lbl_sub == c_lbl
        v_count = int(np.sum(mask_sub))

        if v_count == 0:
            raise Measurement3DConsistencyError(
                f"Active nucleus '{review_id}' (label {c_lbl}) has 0 voxels in label volume."
            )

        for m_name in target_marker_names:
            m_vol = marker_volumes[m_name]
            m_sub = m_vol[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
            vals = m_sub[mask_sub]

            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            int_val = float(np.sum(vals, dtype=np.float64))

            measurement_rows.append(
                {
                    "review_id": review_id,
                    "current_label": c_lbl,
                    "marker_name": m_name,
                    "voxel_count": v_count,
                    "mean_intensity": mean_val,
                    "median_intensity": median_val,
                    "min_intensity": min_val,
                    "max_intensity": max_val,
                    "integrated_intensity": int_val,
                    "source_nucleus_id": source_nuc_id,
                    "review_origin": review_orig,
                    "review_status": review_stat,
                }
            )

    df_measurements = pd.DataFrame(measurement_rows, columns=MEASUREMENT3D_COLUMNS)

    return MarkerMeasurement3DResult(
        measurements=df_measurements,
        reviewed_result=reviewed_result,
        marker_names=target_marker_names,
        source_id=reviewed_result.source_id,
        volume_provenance=volume_provenance,
    )
