"""Tests for M14 3D classical nucleus segmentation engine and models."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.preparation import PreparedVolumeSet, VolumePreparationProvenance, VoxelSpacing
from nervenuc.segmentation3d import (
    InconsistentSegmentation3DResultError,
    InvalidSegmentation3DConfigError,
    Segmentation3DConfig,
    Segmentation3DResult,
    segment_nuclei_3d,
    validate_segmentation3d_consistency,
)


def _create_synthetic_volume(
    shape: tuple[int, int, int] = (16, 64, 64),
    spacing: VoxelSpacing | None = None,
) -> PreparedVolumeSet:
    """Create a synthetic 3D volume with two separated ellipsoidal nuclei."""
    vol = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Nucleus 1 at (4, 20, 20)
    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((x - 20) / 6.0) ** 2 <= 1.0
    vol[mask1] = 1000

    # Nucleus 2 at (10, 45, 45)
    mask2 = ((z - 10) / 3.0) ** 2 + ((y - 45) / 8.0) ** 2 + ((x - 45) / 8.0) ** 2 <= 1.0
    vol[mask2] = 1200

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={"m1": 1},
        z_start=0,
        z_stop=shape[0],
    )
    marker_vol = np.zeros(shape, dtype=np.uint16)
    marker_vol[mask1] = 500
    return PreparedVolumeSet(
        nuclear_volume=vol,
        marker_volumes={"m1": marker_vol},
        spacing=spacing,
        provenance=prov,
    )


def test_segmentation3d_config_validation() -> None:
    """Test Segmentation3DConfig parameter validation."""
    cfg = Segmentation3DConfig()
    assert cfg.threshold_method == "otsu"
    assert cfg.min_volume_voxels == 100

    # Invalid threshold method
    with pytest.raises(InvalidSegmentation3DConfigError, match="threshold_method"):
        Segmentation3DConfig(threshold_method="adaptive")  # type: ignore[arg-type]

    # Manual threshold missing or invalid
    with pytest.raises(InvalidSegmentation3DConfigError, match="manual_threshold must be a numeric finite value"):
        Segmentation3DConfig(threshold_method="manual", manual_threshold=None)

    with pytest.raises(InvalidSegmentation3DConfigError, match="manual_threshold must be a numeric finite value"):
        Segmentation3DConfig(threshold_method="manual", manual_threshold=float("nan"))


    # Invalid sigmas
    with pytest.raises(InvalidSegmentation3DConfigError, match="gaussian_sigma_xy_px"):
        Segmentation3DConfig(gaussian_sigma_xy_px=-1)

    with pytest.raises(InvalidSegmentation3DConfigError, match="gaussian_sigma_z_slices"):
        Segmentation3DConfig(gaussian_sigma_z_slices=-1.0)

    # Invalid volumes
    with pytest.raises(InvalidSegmentation3DConfigError, match="min_volume_voxels"):
        Segmentation3DConfig(min_volume_voxels=-1)

    with pytest.raises(InvalidSegmentation3DConfigError, match="max_volume_voxels"):
        Segmentation3DConfig(min_volume_voxels=100, max_volume_voxels=50)

    # Serialization
    d = cfg.to_dict()
    assert d["threshold_method"] == "otsu"
    reconstructed = Segmentation3DConfig.from_dict(d)
    assert reconstructed == cfg


def test_segment_nuclei_3d_basic() -> None:
    """Test 3D nuclear segmentation on synthetic volume."""
    vol = _create_synthetic_volume()
    res = segment_nuclei_3d(vol, source_id="test_vol_1")

    assert isinstance(res, Segmentation3DResult)
    assert res.source_id == "test_vol_1"
    assert res.source_shape == vol.nuclear_volume.shape
    assert res.nucleus_count == 2
    assert len(res.nuclei) == 2
    assert set(res.nuclei["object_id"]) == {1, 2}
    assert np.all(res.nuclei["voxel_count"] > 0)
    assert np.all(res.labels >= 0)

    # Consistency validation
    validate_segmentation3d_consistency(res, current_source_id="test_vol_1", volume=vol)


def test_segment_nuclei_3d_touching_watershed() -> None:
    """Test 3D watershed separation of touching ellipsoids."""
    shape = (16, 64, 64)
    vol_arr = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Two touching ellipsoids at (8, 30, 25) and (8, 30, 39)
    mask1 = ((z - 8) / 3.0) ** 2 + ((y - 30) / 7.0) ** 2 + ((x - 25) / 7.0) ** 2 <= 1.0
    mask2 = ((z - 8) / 3.0) ** 2 + ((y - 30) / 7.0) ** 2 + ((x - 39) / 7.0) ** 2 <= 1.0
    vol_arr[mask1 | mask2] = 1000

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={},
        z_start=0,
        z_stop=shape[0],
    )
    vol = PreparedVolumeSet(
        nuclear_volume=vol_arr,
        marker_volumes={},
        spacing=None,
        provenance=prov,
    )

    cfg = Segmentation3DConfig(min_distance_xy_px=3, min_distance_z_slices=1)
    res = segment_nuclei_3d(vol, config=cfg, source_id="touching_vol")
    assert res.nucleus_count == 2


def test_segment_nuclei_3d_anisotropic_spacing() -> None:
    """Test segmentation with calibrated physical voxel spacing."""
    spacing = VoxelSpacing(z=1.0, y=0.25, x=0.25)
    vol = _create_synthetic_volume(spacing=spacing)


    res = segment_nuclei_3d(vol, source_id="calibrated_vol")
    assert res.spacing == spacing
    assert "volume_um3" in res.nuclei.columns
    assert "centroid_z_um" in res.nuclei.columns
    assert np.all(res.nuclei["volume_um3"].notna())


def test_segmentation3d_border_exclusion() -> None:
    """Test border exclusion flags in Z and XY."""
    shape = (16, 64, 64)
    vol_arr = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Nucleus touching Z border (z=0)
    mask_z = ((z - 1) / 2.0) ** 2 + ((y - 20) / 5.0) ** 2 + ((x - 20) / 5.0) ** 2 <= 1.0
    vol_arr[mask_z] = 1000

    # Nucleus in the center
    mask_c = ((z - 8) / 2.0) ** 2 + ((y - 32) / 5.0) ** 2 + ((x - 32) / 5.0) ** 2 <= 1.0
    vol_arr[mask_c] = 1000

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={},
        z_start=0,
        z_stop=shape[0],
    )
    vol = PreparedVolumeSet(
        nuclear_volume=vol_arr,
        marker_volumes={},
        spacing=None,
        provenance=prov,
    )

    # Without exclusion
    res1 = segment_nuclei_3d(vol, Segmentation3DConfig(exclude_z_border=False), source_id="v1")
    assert res1.nucleus_count == 2
    assert bool(res1.nuclei[res1.nuclei["object_id"] == 1]["touches_z_border"].iloc[0]) is True

    # With Z border exclusion
    res2 = segment_nuclei_3d(vol, Segmentation3DConfig(exclude_z_border=True), source_id="v1")
    assert res2.nucleus_count == 1


def test_validate_segmentation3d_consistency_errors() -> None:
    """Test validation errors for inconsistent 3D segmentation results."""
    vol = _create_synthetic_volume()
    res = segment_nuclei_3d(vol, source_id="vol_A")

    # Wrong source_id
    with pytest.raises(InconsistentSegmentation3DResultError, match="source_id"):
        validate_segmentation3d_consistency(res, current_source_id="vol_B", volume=vol)

    # Wrong volume shape
    vol_wrong_shape = _create_synthetic_volume(shape=(10, 32, 32))
    with pytest.raises(InconsistentSegmentation3DResultError, match="does not match"):
        validate_segmentation3d_consistency(res, current_source_id="vol_A", volume=vol_wrong_shape)
