"""Tests for M14 3D nuclear marker measurement engine."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.markers import MarkerDefinition
from nervenuc.measurements3d import (
    InvalidMarkerVolumeError,
    MarkerMeasurement3DResult,
    Measurement3DConsistencyError,
    measure_markers_3d,
    validate_measurement3d_consistency,
)
from nervenuc.preparation import PreparedVolumeSet, VolumePreparationProvenance
from nervenuc.qc3d import start_review_3d
from nervenuc.segmentation3d import segment_nuclei_3d


def test_measure_markers_3d_accuracy() -> None:
    """Test mathematical accuracy of 3D marker measurements."""
    shape = (16, 64, 64)
    vol = np.zeros(shape, dtype=np.uint16)
    marker1 = np.zeros(shape, dtype=np.uint16)
    marker2 = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Nucleus 1 at (4, 20, 20)
    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((x - 20) / 6.0) ** 2 <= 1.0
    vol[mask1] = 1000
    marker1[mask1] = 300  # constant intensity
    marker2[mask1] = np.arange(np.count_nonzero(mask1), dtype=np.uint16) + 10

    # Nucleus 2 at (10, 45, 45)
    mask2 = ((z - 10) / 3.0) ** 2 + ((y - 45) / 8.0) ** 2 + ((x - 45) / 8.0) ** 2 <= 1.0
    vol[mask2] = 1200
    marker1[mask2] = 0
    marker2[mask2] = 800

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={"m1": 1, "m2": 2},
        z_start=0,
        z_stop=shape[0],
    )
    prep = PreparedVolumeSet(
        nuclear_volume=vol,
        marker_volumes={"m1": marker1, "m2": marker2},
        spacing=None,
        provenance=prov,
    )
    seg = segment_nuclei_3d(prep, source_id="vol_meas")
    rev = start_review_3d(seg)
    rev.auto_accept()
    reviewed = rev.finalize()

    res = measure_markers_3d(
        reviewed_result=reviewed,
        marker_volumes=prep.marker_volumes,
        volume_provenance=prep.provenance,
        markers=[MarkerDefinition("m1"), MarkerDefinition("m2")],
    )

    assert isinstance(res, MarkerMeasurement3DResult)
    assert res.source_id == "vol_meas"
    assert len(res.measurements) == 4  # 2 nuclei x 2 markers

    # Verify nucleus 1 marker 1
    row_n1_m1 = res.measurements[
        (res.measurements["current_label"] == 1) & (res.measurements["marker_name"] == "m1")
    ].iloc[0]
    assert row_n1_m1["mean_intensity"] == 300.0
    assert row_n1_m1["median_intensity"] == 300.0
    assert row_n1_m1["min_intensity"] == 300.0
    assert row_n1_m1["max_intensity"] == 300.0

    # Verify nucleus 1 marker 2 median
    row_n1_m2 = res.measurements[
        (res.measurements["current_label"] == 1) & (res.measurements["marker_name"] == "m2")
    ].iloc[0]
    voxels_m2 = marker2[mask1]
    assert np.isclose(row_n1_m2["mean_intensity"], np.mean(voxels_m2))
    assert np.isclose(row_n1_m2["median_intensity"], np.median(voxels_m2))

    # Consistency validation
    validate_measurement3d_consistency(res, current_source_id="vol_meas", volume=prep)


def test_measure_markers_3d_error_handling() -> None:
    """Test validation errors for invalid marker volumes."""
    shape = (16, 64, 64)
    vol = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((x - 20) / 6.0) ** 2 <= 1.0
    vol[mask1] = 1000

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
    prep = PreparedVolumeSet(
        nuclear_volume=vol,
        marker_volumes={"m1": np.zeros(shape, dtype=np.uint16)},
        spacing=None,
        provenance=prov,
    )
    seg = segment_nuclei_3d(prep, source_id="vol_err")
    rev = start_review_3d(seg)
    rev.auto_accept()
    reviewed = rev.finalize()

    # Shape mismatch for marker volume
    with pytest.raises(InvalidMarkerVolumeError, match="does not match"):
        measure_markers_3d(
            reviewed_result=reviewed,
            marker_volumes={"m1": np.zeros((8, 32, 32), dtype=np.uint16)},
            volume_provenance=prep.provenance,
        )

    # Missing marker name
    with pytest.raises(InvalidMarkerVolumeError, match="must be a non-empty Mapping"):
        measure_markers_3d(
            reviewed_result=reviewed,
            marker_volumes={},
            volume_provenance=prep.provenance,
            markers=[MarkerDefinition("m1")],
        )
