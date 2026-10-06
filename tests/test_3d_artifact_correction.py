"""Scientific validation test: 3D resolution of Z-separated overlapping nuclei vs 2D projection."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.classification import MarkerThreshold
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import (
    PreparedVolumeSet,
    StackPreparationConfig,
    VolumePreparationProvenance,
    VolumeSelectionConfig,
    prepare_stack,
)
from nervenuc.segmentation3d import Segmentation3DConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import (
    AnalysisConfig,
    ImageAnalysisSession,
    VolumeAnalysisConfig,
    VolumeAnalysisSession,
)


def test_3d_separation_prevents_false_2d_colocalization() -> None:
    """Validate that M14 3D analysis resolves nuclei overlapping in XY but separated in Z without false marker colocalization."""
    shape = (20, 64, 64)
    nuc_vol = np.zeros(shape, dtype=np.uint16)
    marker_vol = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Nucleus 1: at Z=4, (Y=32, X=32), Marker POSITIVE (e.g. BrdU+)
    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 32) / 6.0) ** 2 + ((x - 32) / 6.0) ** 2 <= 1.0
    nuc_vol[mask1] = 1000
    marker_vol[mask1] = 800

    # Nucleus 2: at Z=15, (Y=32, X=32), Marker NEGATIVE (e.g. BrdU-) - perfectly overlapping in XY!
    mask2 = ((z - 15) / 2.0) ** 2 + ((y - 32) / 6.0) ** 2 + ((x - 32) / 6.0) ** 2 <= 1.0
    nuc_vol[mask2] = 1000
    marker_vol[mask2] = 0

    # 1. 2D Maximum Intensity Projection Behavior:
    # Max projection collapses Z=4 and Z=15 into one overlapping 2D region, creating 1 merged nucleus or false positive
    nuc_proj_max = np.max(nuc_vol, axis=0)
    marker_proj_max = np.max(marker_vol, axis=0)

    from nervenuc.segmentation import SegmentationConfig

    cfg_2d = AnalysisConfig(
        segmentation=SegmentationConfig(),
        markers=(MarkerDefinition("BrdU"),),
        thresholds=(MarkerThreshold("BrdU", 200.0, "mean_intensity", ">="),),
        metrics=(
            BiologicalMetricDefinition(
                name="BrdU_pos_pct",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"BrdU": True}),
            ),
        ),
    )

    session_2d = ImageAnalysisSession(cfg_2d)
    session_2d.set_image(nuc_proj_max, {"BrdU": marker_proj_max})
    session_2d.segment()
    assert session_2d.segmentation_result is not None
    # In 2D projection, the two nuclei completely overlap in XY and merge into 1 object
    assert len(session_2d.segmentation_result.objects) == 1




    # 2. M14 True 3D Analysis Behavior:
    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={"BrdU": 1},
        z_start=0,
        z_stop=shape[0],
    )
    prep_3d = PreparedVolumeSet(
        nuclear_volume=nuc_vol,
        marker_volumes={"BrdU": marker_vol},
        spacing=None,
        provenance=prov,
    )

    cfg_3d = VolumeAnalysisConfig(
        selection=VolumeSelectionConfig(
            axes="ZYX",
            time_index=None,
            nuclear_channel=0,
            marker_channels={"BrdU": 1},
            z_selection=None,
        ),
        segmentation=Segmentation3DConfig(min_distance_z_slices=3),
        markers=(MarkerDefinition("BrdU"),),
        thresholds=(MarkerThreshold("BrdU", 200.0, "mean_intensity", ">="),),
        metrics=(
            BiologicalMetricDefinition(
                name="BrdU_pos_pct",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"BrdU": True}),
            ),
        ),
    )

    session_3d = VolumeAnalysisSession()
    session_3d.set_volume(prep_3d, source_id="z_separated_stack")
    res_3d = session_3d.run_pipeline(cfg_3d, auto_accept=True)

    # In 3D, both nuclei are correctly detected as 2 separate 3D objects
    assert session_3d.segmentation_result is not None
    assert session_3d.segmentation_result.nucleus_count == 2

    # In 3D classification, exactly 1 nucleus is positive (Z=4) and 1 nucleus is negative (Z=15)
    assert session_3d.classification_result is not None
    class_df = session_3d.classification_result.classifications
    assert len(class_df) == 2
    pos_count = np.sum(class_df["is_positive"])
    neg_count = np.sum(~class_df["is_positive"])
    assert pos_count == 1
    assert neg_count == 1

    # The 3D biological summary reflects exactly 50% positivity (1/2), avoiding 2D projection artifact
    assert len(res_3d.metrics) == 1
    row = res_3d.metrics.iloc[0]
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 2
    assert row["percentage"] == 50.0
