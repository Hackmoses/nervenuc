"""Engine for classical 3D nuclear segmentation and consistency validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.filters as filters
import skimage.feature as feature
import skimage.segmentation as segmentation

from nervenuc.preparation import PreparedVolumeSet, VoxelSpacing

from .errors import (
    InconsistentSegmentation3DResultError,
    InvalidSegmentation3DConfigError,
    NerveNucSegmentation3DError,
)
from .models import Segmentation3DConfig, Segmentation3DResult

OBJECTS_3D_COLUMNS: list[str] = [
    "object_id",
    "voxel_count",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "z_min",
    "z_max",
    "y_min",
    "y_max",
    "x_min",
    "x_max",
    "touches_z_border",
    "touches_xy_border",
    "volume_um3",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
]


def _empty_objects_3d_dataframe() -> pd.DataFrame:
    """Return an empty 3D objects DataFrame with the required schema."""
    return pd.DataFrame(
        {
            "object_id": pd.Series(dtype="int64"),
            "voxel_count": pd.Series(dtype="int64"),
            "centroid_z": pd.Series(dtype="float64"),
            "centroid_y": pd.Series(dtype="float64"),
            "centroid_x": pd.Series(dtype="float64"),
            "z_min": pd.Series(dtype="int64"),
            "z_max": pd.Series(dtype="int64"),
            "y_min": pd.Series(dtype="int64"),
            "y_max": pd.Series(dtype="int64"),
            "x_min": pd.Series(dtype="int64"),
            "x_max": pd.Series(dtype="int64"),
            "touches_z_border": pd.Series(dtype="bool"),
            "touches_xy_border": pd.Series(dtype="bool"),
            "volume_um3": pd.Series(dtype="float64"),
            "centroid_z_um": pd.Series(dtype="float64"),
            "centroid_y_um": pd.Series(dtype="float64"),
            "centroid_x_um": pd.Series(dtype="float64"),
        },
        columns=OBJECTS_3D_COLUMNS,
    )


def validate_segmentation3d_consistency(
    result: Segmentation3DResult,
    current_source_id: str,
    volume: PreparedVolumeSet | None = None,
) -> None:
    """Validate structural and provenance consistency of a Segmentation3DResult.

    Args:
        result: Segmentation3DResult to validate.
        current_source_id: Expected authoritative source identifier.
        volume: Optional PreparedVolumeSet to check dimensional and calibration alignment.

    Raises:
        InconsistentSegmentation3DResultError: If result is structurally corrupt or mismatched.
    """
    if not isinstance(result, Segmentation3DResult):
        raise InconsistentSegmentation3DResultError(
            f"result must be a Segmentation3DResult, got {type(result).__name__}."
        )

    if not isinstance(result.labels, np.ndarray) or result.labels.ndim != 3:
        raise InconsistentSegmentation3DResultError(
            f"labels must be a 3D numpy array, got {getattr(result.labels, 'ndim', None)}D."
        )

    if not np.issubdtype(result.labels.dtype, np.integer):
        raise InconsistentSegmentation3DResultError(
            f"labels array must have integer dtype, got {result.labels.dtype}."
        )

    if result.source_id != current_source_id:
        raise InconsistentSegmentation3DResultError(
            f"Segmentation3DResult source_id '{result.source_id}' does not match "
            f"expected execution source_id '{current_source_id}'."
        )

    if volume is not None:
        if not isinstance(volume, PreparedVolumeSet):
            raise InconsistentSegmentation3DResultError(
                f"volume must be a PreparedVolumeSet, got {type(volume).__name__}."
            )
        if result.source_shape != volume.nuclear_volume.shape:
            raise InconsistentSegmentation3DResultError(
                f"Segmentation3DResult source_shape {result.source_shape} does not match "
                f"PreparedVolumeSet nuclear_volume shape {volume.nuclear_volume.shape}."
            )
        if result.labels.shape != volume.nuclear_volume.shape:
            raise InconsistentSegmentation3DResultError(
                f"Segmentation3DResult labels shape {result.labels.shape} does not match "
                f"PreparedVolumeSet nuclear_volume shape {volume.nuclear_volume.shape}."
            )
        if result.spacing != volume.spacing:
            raise InconsistentSegmentation3DResultError(
                f"Segmentation3DResult spacing ({result.spacing}) does not match "
                f"volume spacing ({volume.spacing})."
            )

    # Check background and positive labels
    unique_labels = np.unique(result.labels)
    if len(unique_labels) > 0 and unique_labels[0] < 0:
        raise InconsistentSegmentation3DResultError("Negative label values are not permitted.")

    pos_labels = sorted([int(lbl) for lbl in unique_labels if lbl > 0])
    df_obj_ids = sorted([int(oid) for oid in result.objects["object_id"]]) if len(result.objects) > 0 else []

    if pos_labels != df_obj_ids:
        raise InconsistentSegmentation3DResultError(
            f"Label volume unique positive labels ({pos_labels}) do not match "
            f"object table object_id values ({df_obj_ids})."
        )

    # Verify voxel counts match
    if len(result.objects) > 0:
        counts = pd.Series(result.labels[result.labels > 0]).value_counts().to_dict()
        for _, row in result.objects.iterrows():
            oid = int(row["object_id"])
            expected_count = counts.get(oid, 0)
            actual_count = int(row["voxel_count"])
            if actual_count != expected_count:
                raise InconsistentSegmentation3DResultError(
                    f"Object {oid} voxel count mismatch: table={actual_count} vs label_volume={expected_count}."
                )


def segment_nuclei_3d(
    volume: PreparedVolumeSet | np.ndarray,
    config: Segmentation3DConfig | None = None,
    source_id: str = "in_memory_volume",
    spacing: VoxelSpacing | None = None,
) -> Segmentation3DResult:
    """Perform classical 3D nucleus segmentation on a microscopy subvolume.

    Args:
        volume: PreparedVolumeSet or raw 3D NumPy array of shape (Z, Y, X).
        config: Segmentation3DConfig parameters (uses defaults if None).
        source_id: Authoritative identifier for the source volume execution.
        spacing: Physical voxel spacing if volume is passed as a raw array.

    Returns:
        Segmentation3DResult containing 3D label volume, object table, and provenance.

    Raises:
        NerveNucSegmentation3DError: If array dimensions, configuration, or processing fails.
    """
    eff_config = config if config is not None else Segmentation3DConfig()
    if not isinstance(eff_config, Segmentation3DConfig):
        raise InvalidSegmentation3DConfigError(
            f"config must be a Segmentation3DConfig, got {type(eff_config).__name__}."
        )

    if isinstance(volume, PreparedVolumeSet):
        raw_volume = volume.nuclear_volume
        eff_spacing = volume.spacing
    elif isinstance(volume, np.ndarray):
        raw_volume = volume
        eff_spacing = spacing
    else:
        raise NerveNucSegmentation3DError(
            f"volume must be a PreparedVolumeSet or numpy.ndarray, got {type(volume).__name__}."
        )

    if not isinstance(raw_volume, np.ndarray) or raw_volume.ndim != 3:
        raise NerveNucSegmentation3DError(
            f"volume must be a 3D array of shape (Z, Y, X), got {getattr(raw_volume, 'ndim', None)}D."
        )

    if not np.issubdtype(raw_volume.dtype, np.number) or raw_volume.dtype == bool or np.iscomplexobj(raw_volume):
        raise NerveNucSegmentation3DError(
            f"volume dtype '{raw_volume.dtype}' is not supported. Must be standard numeric type."
        )

    if np.issubdtype(raw_volume.dtype, np.floating) and not np.all(np.isfinite(raw_volume)):
        raise NerveNucSegmentation3DError("Non-finite values (NaN, Inf) encountered in 3D volume.")

    if not isinstance(source_id, str) or not source_id.strip():
        raise NerveNucSegmentation3DError("source_id must be a non-empty string.")

    z_depth, y_height, x_width = raw_volume.shape

    # 1. 3D Gaussian smoothing
    vol_float = raw_volume.astype(np.float64, copy=False)
    sigma_z = eff_config.gaussian_sigma_z_slices if eff_config.gaussian_sigma_z_slices is not None else 0.0
    sigma_xy = eff_config.gaussian_sigma_xy_px
    sigmas = (float(sigma_z), float(sigma_xy), float(sigma_xy))

    if any(s > 0 for s in sigmas):
        smoothed = ndi.gaussian_filter(vol_float, sigma=sigmas)
    else:
        smoothed = vol_float

    # 2. Global 3D thresholding
    if eff_config.threshold_method == "otsu":
        # If smoothed volume has zero variance
        if np.max(smoothed) == np.min(smoothed):
            thresh = float(np.min(smoothed))
        else:
            thresh = float(filters.threshold_otsu(smoothed))
    elif eff_config.threshold_method == "manual":
        assert eff_config.manual_threshold is not None
        thresh = float(eff_config.manual_threshold)
    else:
        raise InvalidSegmentation3DConfigError(f"Unsupported threshold_method: '{eff_config.threshold_method}'.")

    binary = smoothed >= thresh

    # 3. Binary hole filling
    if eff_config.fill_holes and np.any(binary):
        binary = ndi.binary_fill_holes(binary)

    if not np.any(binary):
        empty_labels = np.zeros_like(raw_volume, dtype=np.int32)
        return Segmentation3DResult(
            labels=empty_labels,
            objects=_empty_objects_3d_dataframe(),
            config=eff_config,
            source_shape=raw_volume.shape,
            source_id=source_id,
            spacing=eff_spacing,
        )

    # 4. Filter objects by volume before distance transform
    labeled_init, num_init = ndi.label(binary)
    if num_init == 0:
        empty_labels = np.zeros_like(raw_volume, dtype=np.int32)
        return Segmentation3DResult(
            labels=empty_labels,
            objects=_empty_objects_3d_dataframe(),
            config=eff_config,
            source_shape=raw_volume.shape,
            source_id=source_id,
            spacing=eff_spacing,
        )

    init_sizes = ndi.sum(binary, labeled_init, range(1, num_init + 1))
    valid_labels_mask = np.zeros(num_init + 1, dtype=bool)
    for lbl_idx, sz in enumerate(init_sizes, start=1):
        if sz >= eff_config.min_volume_voxels and (
            eff_config.max_volume_voxels is None or sz <= eff_config.max_volume_voxels
        ):
            valid_labels_mask[lbl_idx] = True

    binary_filtered = valid_labels_mask[labeled_init]

    if not np.any(binary_filtered):
        empty_labels = np.zeros_like(raw_volume, dtype=np.int32)
        return Segmentation3DResult(
            labels=empty_labels,
            objects=_empty_objects_3d_dataframe(),
            config=eff_config,
            source_shape=raw_volume.shape,
            source_id=source_id,
            spacing=eff_spacing,
        )

    # 5. 3D Euclidean Distance Transform with physical sampling if calibrated
    if eff_spacing is not None:
        sampling = (eff_spacing.z, eff_spacing.y, eff_spacing.x)
    else:
        sampling = (1.0, 1.0, 1.0)

    distance = ndi.distance_transform_edt(binary_filtered, sampling=sampling)

    # 6. 3D peak finding using anisotropic voxel footprint
    rz = eff_config.min_distance_z_slices
    rxy = eff_config.min_distance_xy_px
    footprint = np.ones((2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), dtype=bool)

    coords = feature.peak_local_max(
        distance,
        footprint=footprint,
        labels=binary_filtered,
    )

    if len(coords) == 0:
        empty_labels = np.zeros_like(raw_volume, dtype=np.int32)
        return Segmentation3DResult(
            labels=empty_labels,
            objects=_empty_objects_3d_dataframe(),
            config=eff_config,
            source_shape=raw_volume.shape,
            source_id=source_id,
            spacing=eff_spacing,
        )

    markers_mask = np.zeros_like(distance, dtype=bool)
    markers_mask[tuple(coords.T)] = True
    markers_labeled, _ = ndi.label(markers_mask)

    # 7. 3D seeded watershed
    ws_labels = segmentation.watershed(-distance, markers_labeled, mask=binary_filtered)

    # 8. Measure objects and apply deterministic sort and border filtering
    unique_ws = np.unique(ws_labels)
    raw_objects: list[dict] = []

    for lbl in unique_ws:
        if lbl == 0:
            continue
        vox_z, vox_y, vox_x = np.where(ws_labels == lbl)
        v_count = len(vox_z)
        if v_count < eff_config.min_volume_voxels or (
            eff_config.max_volume_voxels is not None and v_count > eff_config.max_volume_voxels
        ):
            continue

        c_z = float(np.mean(vox_z))
        c_y = float(np.mean(vox_y))
        c_x = float(np.mean(vox_x))

        z_min = int(np.min(vox_z))
        z_max = int(np.max(vox_z))
        y_min = int(np.min(vox_y))
        y_max = int(np.max(vox_y))
        x_min = int(np.min(vox_x))
        x_max = int(np.max(vox_x))

        touches_z = bool(z_min == 0 or z_max == z_depth - 1)
        touches_xy = bool(y_min == 0 or y_max == y_height - 1 or x_min == 0 or x_max == x_width - 1)

        if eff_config.exclude_z_border and touches_z:
            continue
        if eff_config.exclude_xy_border and touches_xy:
            continue

        raw_objects.append(
            {
                "original_label": int(lbl),
                "voxel_count": v_count,
                "centroid_z": c_z,
                "centroid_y": c_y,
                "centroid_x": c_x,
                "z_min": z_min,
                "z_max": z_max,
                "y_min": y_min,
                "y_max": y_max,
                "x_min": x_min,
                "x_max": x_max,
                "touches_z_border": touches_z,
                "touches_xy_border": touches_xy,
            }
        )

    if len(raw_objects) == 0:
        empty_labels = np.zeros_like(raw_volume, dtype=np.int32)
        return Segmentation3DResult(
            labels=empty_labels,
            objects=_empty_objects_3d_dataframe(),
            config=eff_config,
            source_shape=raw_volume.shape,
            source_id=source_id,
            spacing=eff_spacing,
        )

    # Deterministic sort order
    raw_objects.sort(
        key=lambda item: (
            item["centroid_z"],
            item["centroid_y"],
            item["centroid_x"],
            item["z_min"],
            item["y_min"],
            item["x_min"],
            item["voxel_count"],
            item["original_label"],
        )
    )

    # 9. Sequential deterministic relabeling (1..N)
    labels_out = np.zeros_like(raw_volume, dtype=np.int32)
    final_objects: list[dict] = []

    voxel_vol_um3 = (
        (eff_spacing.z * eff_spacing.y * eff_spacing.x) if eff_spacing is not None else None
    )

    for new_id, item in enumerate(raw_objects, start=1):
        orig_lbl = item["original_label"]
        labels_out[ws_labels == orig_lbl] = new_id

        v_count = item["voxel_count"]
        c_z = item["centroid_z"]
        c_y = item["centroid_y"]
        c_x = item["centroid_x"]

        vol_um3 = float(v_count * voxel_vol_um3) if voxel_vol_um3 is not None else np.nan
        c_z_um = float(c_z * eff_spacing.z) if eff_spacing is not None else np.nan
        c_y_um = float(c_y * eff_spacing.y) if eff_spacing is not None else np.nan
        c_x_um = float(c_x * eff_spacing.x) if eff_spacing is not None else np.nan

        final_objects.append(
            {
                "object_id": new_id,
                "voxel_count": v_count,
                "centroid_z": c_z,
                "centroid_y": c_y,
                "centroid_x": c_x,
                "z_min": item["z_min"],
                "z_max": item["z_max"],
                "y_min": item["y_min"],
                "y_max": item["y_max"],
                "x_min": item["x_min"],
                "x_max": item["x_max"],
                "touches_z_border": item["touches_z_border"],
                "touches_xy_border": item["touches_xy_border"],
                "volume_um3": vol_um3,
                "centroid_z_um": c_z_um,
                "centroid_y_um": c_y_um,
                "centroid_x_um": c_x_um,
            }
        )

    df_objects = pd.DataFrame(final_objects, columns=OBJECTS_3D_COLUMNS)

    return Segmentation3DResult(
        labels=labels_out,
        objects=df_objects,
        config=eff_config,
        source_shape=raw_volume.shape,
        source_id=source_id,
        spacing=eff_spacing,
    )
