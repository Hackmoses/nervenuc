"""Deterministic synthetic microscopy dataset generator with known mathematical ground truth."""

from __future__ import annotations

import numpy as np

from nervenuc.classification.models import MarkerThreshold
from nervenuc.markers.models import MarkerDefinition
from nervenuc.preparation.models import VolumeSelectionConfig, VoxelSpacing
from nervenuc.segmentation.config import SegmentationConfig
from nervenuc.segmentation3d.models import Segmentation3DConfig
from nervenuc.summary.models import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow.config import AnalysisConfig, VolumeAnalysisConfig

from .models import GroundTruthObject, ValidationCase, ValidationCaseSpec


def _draw_2d_ellipsoid(
    mask: np.ndarray,
    image: np.ndarray,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
    label_id: int,
    peak_intensity: float = 20000.0,
    bg_intensity: float = 1000.0,
) -> None:
    """Draw a 2D ellipsoid with smooth intensity profile into mask and image arrays."""
    Y, X = mask.shape
    y_coords, x_coords = np.ogrid[:Y, :X]
    dist_sq = ((y_coords - center_y) / radius_y) ** 2 + ((x_coords - center_x) / radius_x) ** 2
    inside = dist_sq <= 1.0

    mask[inside] = label_id
    intensity = bg_intensity + (peak_intensity - bg_intensity) * np.exp(-1.5 * dist_sq)
    image[inside] = np.maximum(image[inside], intensity[inside])


def _draw_3d_ellipsoid(
    mask: np.ndarray,
    image: np.ndarray,
    center_z: float,
    center_y: float,
    center_x: float,
    radius_z: float,
    radius_y: float,
    radius_x: float,
    label_id: int,
    peak_intensity: float = 20000.0,
    bg_intensity: float = 1000.0,
) -> None:
    """Draw a 3D ellipsoid with smooth intensity profile into mask and image arrays."""
    Z, Y, X = mask.shape
    z_coords, y_coords, x_coords = np.ogrid[:Z, :Y, :X]
    dist_sq = (
        ((z_coords - center_z) / radius_z) ** 2
        + ((y_coords - center_y) / radius_y) ** 2
        + ((x_coords - center_x) / radius_x) ** 2
    )
    inside = dist_sq <= 1.0

    mask[inside] = label_id
    intensity = bg_intensity + (peak_intensity - bg_intensity) * np.exp(-1.5 * dist_sq)
    image[inside] = np.maximum(image[inside], intensity[inside])


def generate_syn_2d_isolated(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_2D_ISOLATED: 2D image with 4 clearly separated ellipsoidal nuclei."""
    rng = np.random.default_rng(seed)
    Y, X = 128, 128
    bg_val = 1000.0

    gt_mask = np.zeros((Y, X), dtype=np.int32)
    img = np.full((Y, X), bg_val, dtype=np.float32)

    nuclei_params = [
        (1, 30.0, 30.0, 8.0, 10.0),
        (2, 30.0, 95.0, 9.0, 9.0),
        (3, 95.0, 30.0, 10.0, 8.0),
        (4, 95.0, 95.0, 7.0, 11.0),
    ]

    for label_id, cy, cx, ry, rx in nuclei_params:
        _draw_2d_ellipsoid(gt_mask, img, cy, cx, ry, rx, label_id, bg_intensity=bg_val)

    # Add subtle deterministic noise
    noise = rng.normal(0.0, 50.0, size=(Y, X))
    img = np.clip(img + noise, 0, 65535).astype(np.uint16)
    img_cyx = img[np.newaxis, ...]

    gt_objects = []
    for label_id, cy, cx, ry, rx in nuclei_params:
        coords = np.nonzero(gt_mask == label_id)
        gt_objects.append(
            GroundTruthObject(
                truth_id=label_id,
                label_name=f"NUC-{label_id:03d}",
                centroid=(float(cy), float(cx)),
                size=int(len(coords[0])),
                phenotype="Unassigned",
            )
        )

    config = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="otsu", min_area_px=20, min_distance_px=5),
        markers=(MarkerDefinition("DAPI"),),
        thresholds=(MarkerThreshold("DAPI", 2000.0, "mean_intensity"),),
        metrics=(BiologicalMetricDefinition("Total", PhenotypeFilter(), PhenotypeFilter()),),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_2D_ISOLATED",
        description="Deterministic 2D synthetic image with 4 isolated nuclei.",
        tier="synthetic",
        dimensions="2D",
        image_relpath="synthetic/syn_2d_isolated.tif",
        ground_truth_labels_relpath="synthetic/syn_2d_isolated_gt.tif",
        axes="CYX",
        channel_names=("DAPI",),
    )

    return ValidationCase(
        spec=spec,
        image_data=img_cyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=tuple(gt_objects),
        config=config,
    )


def generate_syn_2d_touching(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_2D_TOUCHING: 2D image with 2 touching nuclei for watershed testing."""
    rng = np.random.default_rng(seed)
    Y, X = 128, 128
    bg_val = 1000.0

    gt_mask = np.zeros((Y, X), dtype=np.int32)
    img = np.full((Y, X), bg_val, dtype=np.float32)

    # Two nuclei centered at (64, 48) and (64, 72) with radii 12 -> touching at x=60
    _draw_2d_ellipsoid(gt_mask, img, 64.0, 48.0, 12.0, 12.0, 1, bg_intensity=bg_val)
    y_coords, x_coords = np.ogrid[:Y, :X]
    d2_sq = ((y_coords - 64.0) / 12.0) ** 2 + ((x_coords - 72.0) / 12.0) ** 2
    d1_sq = ((y_coords - 64.0) / 12.0) ** 2 + ((x_coords - 48.0) / 12.0) ** 2
    inside2 = d2_sq <= 1.0

    gt_mask[inside2 & (d2_sq < d1_sq)] = 2

    intensity2 = bg_val + (20000.0 - bg_val) * np.exp(-1.5 * d2_sq)
    img[inside2] = np.maximum(img[inside2], intensity2[inside2])

    noise = rng.normal(0.0, 50.0, size=(Y, X))
    img = np.clip(img + noise, 0, 65535).astype(np.uint16)
    img_cyx = img[np.newaxis, ...]

    gt_objects = []
    for label_id, (cy, cx) in [(1, (64.0, 48.0)), (2, (64.0, 72.0))]:
        coords = np.nonzero(gt_mask == label_id)
        gt_objects.append(
            GroundTruthObject(
                truth_id=label_id,
                label_name=f"NUC-{label_id:03d}",
                centroid=(float(np.mean(coords[0])), float(np.mean(coords[1]))),
                size=int(len(coords[0])),
                phenotype="Unassigned",
            )
        )

    config = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="otsu", min_area_px=30, min_distance_px=5),
        markers=(MarkerDefinition("DAPI"),),
        thresholds=(MarkerThreshold("DAPI", 2000.0, "mean_intensity"),),
        metrics=(BiologicalMetricDefinition("Total", PhenotypeFilter(), PhenotypeFilter()),),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_2D_TOUCHING",
        description="Deterministic 2D synthetic image with 2 touching nuclei.",
        tier="synthetic",
        dimensions="2D",
        image_relpath="synthetic/syn_2d_touching.tif",
        ground_truth_labels_relpath="synthetic/syn_2d_touching_gt.tif",
        axes="CYX",
        channel_names=("DAPI",),
    )

    return ValidationCase(
        spec=spec,
        image_data=img_cyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=tuple(gt_objects),
        config=config,
    )


def generate_syn_2d_empty(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_2D_EMPTY: 2D image with zero nuclei (blank background)."""
    rng = np.random.default_rng(seed)
    Y, X = 128, 128
    bg_val = 1000.0

    img = np.full((Y, X), bg_val, dtype=np.float32)
    noise = rng.normal(0.0, 50.0, size=(Y, X))
    img = np.clip(img + noise, 0, 65535).astype(np.uint16)
    img_cyx = img[np.newaxis, ...]

    gt_mask = np.zeros((Y, X), dtype=np.int32)

    config = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="manual", manual_threshold=3000.0, min_area_px=20, min_distance_px=5),
        markers=(MarkerDefinition("DAPI"),),
        thresholds=(MarkerThreshold("DAPI", 2000.0, "mean_intensity"),),
        metrics=(BiologicalMetricDefinition("Total", PhenotypeFilter(), PhenotypeFilter()),),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_2D_EMPTY",
        description="Deterministic 2D synthetic blank image containing 0 nuclei.",
        tier="synthetic",
        dimensions="2D",
        image_relpath="synthetic/syn_2d_empty.tif",
        ground_truth_labels_relpath="synthetic/syn_2d_empty_gt.tif",
        axes="CYX",
        channel_names=("DAPI",),
    )

    return ValidationCase(
        spec=spec,
        image_data=img_cyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=(),
        config=config,
    )


def generate_syn_2d_multi_marker(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_2D_MULTI_MARKER: 3-channel 2D image with MarkerA and MarkerB phenotypes."""
    rng = np.random.default_rng(seed)
    Y, X = 128, 128
    bg_val = 500.0

    gt_mask = np.zeros((Y, X), dtype=np.int32)
    dapi = np.full((Y, X), 1000.0, dtype=np.float32)
    marker_a = np.full((Y, X), bg_val, dtype=np.float32)
    marker_b = np.full((Y, X), bg_val, dtype=np.float32)

    nuclei_specs = [
        (1, 30.0, 30.0, 8.0, 8.0, True, False, "MarkerA+ / MarkerB-"),
        (2, 30.0, 95.0, 8.0, 8.0, False, True, "MarkerA- / MarkerB+"),
        (3, 95.0, 30.0, 8.0, 8.0, True, True, "MarkerA+ / MarkerB+"),
        (4, 95.0, 95.0, 8.0, 8.0, False, False, "MarkerA- / MarkerB-"),
    ]

    for label_id, cy, cx, ry, rx, is_a, is_b, _ in nuclei_specs:
        _draw_2d_ellipsoid(gt_mask, dapi, cy, cx, ry, rx, label_id, peak_intensity=20000.0, bg_intensity=1000.0)

        # Draw MarkerA
        a_peak = 15000.0 if is_a else bg_val
        _draw_2d_ellipsoid(np.zeros((Y, X), dtype=np.int32), marker_a, cy, cx, ry, rx, label_id, peak_intensity=a_peak, bg_intensity=bg_val)

        # Draw MarkerB
        b_peak = 15000.0 if is_b else bg_val
        _draw_2d_ellipsoid(np.zeros((Y, X), dtype=np.int32), marker_b, cy, cx, ry, rx, label_id, peak_intensity=b_peak, bg_intensity=bg_val)

    dapi = np.clip(dapi + rng.normal(0.0, 50.0, (Y, X)), 0, 65535).astype(np.uint16)
    marker_a = np.clip(marker_a + rng.normal(0.0, 30.0, (Y, X)), 0, 65535).astype(np.uint16)
    marker_b = np.clip(marker_b + rng.normal(0.0, 30.0, (Y, X)), 0, 65535).astype(np.uint16)

    img_cyx = np.stack([dapi, marker_a, marker_b], axis=0)

    gt_objects = []
    for label_id, cy, cx, _, _, is_a, is_b, pheno in nuclei_specs:
        coords = np.nonzero(gt_mask == label_id)
        gt_objects.append(
            GroundTruthObject(
                truth_id=label_id,
                label_name=f"NUC-{label_id:03d}",
                centroid=(float(cy), float(cx)),
                size=int(len(coords[0])),
                phenotype=pheno,
                marker_positivity={"MarkerA": is_a, "MarkerB": is_b},
            )
        )

    config = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="otsu", min_area_px=20, min_distance_px=5),
        markers=(MarkerDefinition("MarkerA"), MarkerDefinition("MarkerB")),
        thresholds=(
            MarkerThreshold("MarkerA", 4000.0, "mean_intensity"),
            MarkerThreshold("MarkerB", 4000.0, "mean_intensity"),
        ),
        metrics=(
            BiologicalMetricDefinition("MarkerA+", PhenotypeFilter(), PhenotypeFilter({"MarkerA": True})),
            BiologicalMetricDefinition("MarkerB+", PhenotypeFilter(), PhenotypeFilter({"MarkerB": True})),
        ),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_2D_MULTI_MARKER",
        description="Deterministic 2D synthetic image with 4 nuclei exhibiting 4 combinatorial phenotypes.",
        tier="synthetic",
        dimensions="2D",
        image_relpath="synthetic/syn_2d_multi_marker.tif",
        ground_truth_labels_relpath="synthetic/syn_2d_multi_marker_gt.tif",
        axes="CYX",
        channel_names=("DAPI", "MarkerA", "MarkerB"),
    )

    return ValidationCase(
        spec=spec,
        image_data=img_cyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=tuple(gt_objects),
        config=config,
    )


def generate_syn_3d_isolated(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_3D_ISOLATED: 3D ZYX volume with 3 separated ellipsoids and anisotropic calibration."""
    rng = np.random.default_rng(seed)
    Z, Y, X = 16, 64, 80
    bg_val = 1000.0

    gt_mask = np.zeros((Z, Y, X), dtype=np.int32)
    vol = np.full((Z, Y, X), bg_val, dtype=np.float32)

    nuclei_specs = [
        (1, 8.0, 20.0, 25.0, 4.0, 8.0, 8.0),
        (2, 8.0, 44.0, 25.0, 4.0, 8.0, 8.0),
        (3, 8.0, 32.0, 55.0, 4.0, 8.0, 8.0),
    ]

    for label_id, cz, cy, cx, rz, ry, rx in nuclei_specs:
        _draw_3d_ellipsoid(gt_mask, vol, cz, cy, cx, rz, ry, rx, label_id, bg_intensity=bg_val)

    noise = rng.normal(0.0, 50.0, size=(Z, Y, X))
    vol = np.clip(vol + noise, 0, 65535).astype(np.uint16)
    marker = np.full_like(vol, 500, dtype=np.uint16)
    marker[gt_mask > 0] = 3000
    vol_czyx = np.stack([vol, marker], axis=0)

    spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104)

    gt_objects = []
    for label_id, cz, cy, cx, _, _, _ in nuclei_specs:
        coords = np.nonzero(gt_mask == label_id)
        gt_objects.append(
            GroundTruthObject(
                truth_id=label_id,
                label_name=f"NUC3D-{label_id:03d}",
                centroid=(float(cz), float(cy), float(cx)),
                size=int(len(coords[0])),
                phenotype="Unassigned",
            )
        )

    config = VolumeAnalysisConfig(
        selection=VolumeSelectionConfig(
            axes="CZYX",
            nuclear_channel=0,
            marker_channels={"MarkerA": 1},
        ),
        segmentation=Segmentation3DConfig(
            threshold_method="otsu",
            min_volume_voxels=50,
            min_distance_xy_px=5,
            min_distance_z_slices=2,
        ),
        markers=(MarkerDefinition("MarkerA"),),
        thresholds=(MarkerThreshold("MarkerA", 2000.0, "mean_intensity"),),
        metrics=(BiologicalMetricDefinition("Total", PhenotypeFilter(), PhenotypeFilter()),),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_3D_ISOLATED",
        description="Deterministic 3D synthetic volume with 3 separated ellipsoids.",
        tier="synthetic",
        dimensions="3D",
        image_relpath="synthetic/syn_3d_isolated.tif",
        ground_truth_labels_relpath="synthetic/syn_3d_isolated_gt.tif",
        axes="CZYX",
        channel_names=("DAPI", "MarkerA"),
        voxel_spacing_um=(0.5, 0.104, 0.104),
    )

    return ValidationCase(
        spec=spec,
        image_data=vol_czyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=tuple(gt_objects),
        config=config,
    )


def generate_syn_3d_overlap_xy(*, seed: int = 42) -> ValidationCase:
    """Generate SYN_3D_OVERLAP_XY: 2 distinct nuclei with identical (X, Y) footprint but separated in Z."""
    rng = np.random.default_rng(seed)
    Z, Y, X = 24, 64, 64
    bg_val = 1000.0

    gt_mask = np.zeros((Z, Y, X), dtype=np.int32)
    vol = np.full((Z, Y, X), bg_val, dtype=np.float32)

    # Nucleus 1: centered at Z=6, Y=32, X=32 (radii: rz=3, ry=10, rx=10) -> spans Z in [3, 9]
    # Nucleus 2: centered at Z=18, Y=32, X=32 (radii: rz=3, ry=10, rx=10) -> spans Z in [15, 21]
    # Slices 10..14 are pure background separating them completely in 3D!
    nuclei_specs = [
        (1, 6.0, 32.0, 32.0, 3.0, 10.0, 10.0),
        (2, 18.0, 32.0, 32.0, 3.0, 10.0, 10.0),
    ]

    for label_id, cz, cy, cx, rz, ry, rx in nuclei_specs:
        _draw_3d_ellipsoid(gt_mask, vol, cz, cy, cx, rz, ry, rx, label_id, bg_intensity=bg_val)

    noise = rng.normal(0.0, 50.0, size=(Z, Y, X))
    vol = np.clip(vol + noise, 0, 65535).astype(np.uint16)
    marker = np.full_like(vol, 500, dtype=np.uint16)
    marker[gt_mask > 0] = 3000
    vol_czyx = np.stack([vol, marker], axis=0)

    spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104)

    gt_objects = []
    for label_id, cz, cy, cx, _, _, _ in nuclei_specs:
        coords = np.nonzero(gt_mask == label_id)
        gt_objects.append(
            GroundTruthObject(
                truth_id=label_id,
                label_name=f"NUC3D-{label_id:03d}",
                centroid=(float(cz), float(cy), float(cx)),
                size=int(len(coords[0])),
                phenotype="Unassigned",
            )
        )

    config = VolumeAnalysisConfig(
        selection=VolumeSelectionConfig(
            axes="CZYX",
            nuclear_channel=0,
            marker_channels={"MarkerA": 1},
        ),
        segmentation=Segmentation3DConfig(
            threshold_method="otsu",
            min_volume_voxels=50,
            min_distance_xy_px=5,
            min_distance_z_slices=2,
        ),
        markers=(MarkerDefinition("MarkerA"),),
        thresholds=(MarkerThreshold("MarkerA", 2000.0, "mean_intensity"),),
        metrics=(BiologicalMetricDefinition("Total", PhenotypeFilter(), PhenotypeFilter()),),
    )

    spec = ValidationCaseSpec(
        case_id="SYN_3D_OVERLAP_XY",
        description="Deterministic 3D volume with 2 nuclei having overlapping XY footprints but distinct Z slices.",
        tier="synthetic",
        dimensions="3D",
        image_relpath="synthetic/syn_3d_overlap_xy.tif",
        ground_truth_labels_relpath="synthetic/syn_3d_overlap_xy_gt.tif",
        axes="CZYX",
        channel_names=("DAPI", "MarkerA"),
        voxel_spacing_um=(0.5, 0.104, 0.104),
    )

    return ValidationCase(
        spec=spec,
        image_data=vol_czyx,
        ground_truth_labels=gt_mask,
        ground_truth_objects=tuple(gt_objects),
        config=config,
    )
