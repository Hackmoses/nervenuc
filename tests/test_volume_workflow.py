"""Tests for M14 3D Volume Workflow, Config, Fingerprinting, and Study Aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from nervenuc.aggregation import SampleIdentity
from nervenuc.classification import MarkerThreshold
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import (
    PreparedVolumeSet,
    VolumePreparationProvenance,
    VolumeSelectionConfig,
)
from nervenuc.segmentation3d import Segmentation3DConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import (
    ImageSourceProvenance,
    IncompatibleMeasurementSemanticsError,
    StudyAnalysisSession,
    VolumeAnalysisConfig,
    VolumeAnalysisSession,
    VolumeSourceProvenance,
    compute_volume_analysis_config_fingerprint,
    compute_volume_pipeline_fingerprint,
    validate_volume_selection_analysis_compatibility,
)


def _create_synthetic_volume_set(shape: tuple[int, int, int] = (16, 64, 64)) -> PreparedVolumeSet:
    vol = np.zeros(shape, dtype=np.uint16)
    marker = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((x - 20) / 6.0) ** 2 <= 1.0
    mask2 = ((z - 10) / 3.0) ** 2 + ((y - 45) / 8.0) ** 2 + ((x - 45) / 8.0) ** 2 <= 1.0
    vol[mask1] = 1000
    vol[mask2] = 1200
    marker[mask1] = 800  # positive for n1
    marker[mask2] = 50   # negative for n2

    prov = VolumePreparationProvenance(
        source_shape=shape,
        metadata_axes="ZYX",
        declared_axes=None,
        effective_axes="ZYX",
        time_index=None,
        nuclear_channel=0,
        marker_channels={"SOX10": 1},
        z_start=0,
        z_stop=shape[0],
    )
    return PreparedVolumeSet(
        nuclear_volume=vol,
        marker_volumes={"SOX10": marker},
        spacing=None,
        provenance=prov,
    )


def _create_default_volume_config() -> VolumeAnalysisConfig:
    sel = VolumeSelectionConfig(
        axes=None,
        time_index=None,
        nuclear_channel=0,
        marker_channels={"SOX10": 1},
        z_selection=None,
    )
    seg = Segmentation3DConfig()
    markers = (MarkerDefinition("SOX10"),)
    thresholds = (MarkerThreshold("SOX10", 200.0, "mean_intensity", ">="),)
    metrics = (
        BiologicalMetricDefinition(
            name="SOX10_pos_pct",
            denominator=PhenotypeFilter(),
            numerator=PhenotypeFilter({"SOX10": True}),
        ),
    )
    return VolumeAnalysisConfig(
        selection=sel,
        segmentation=seg,
        markers=markers,
        thresholds=thresholds,
        metrics=metrics,
    )


def test_volume_config_fingerprint_and_serialization() -> None:
    """Test VolumeAnalysisConfig serialization and deterministic fingerprinting."""
    cfg = _create_default_volume_config()
    d = cfg.to_dict()
    reconstructed = VolumeAnalysisConfig.from_dict(d)
    assert reconstructed == cfg

    fp1 = compute_volume_analysis_config_fingerprint(cfg)
    fp2 = compute_volume_analysis_config_fingerprint(reconstructed)
    assert fp1 == fp2
    assert len(fp1) == 64

    pipe_fp = compute_volume_pipeline_fingerprint(fp1)
    assert len(pipe_fp) == 64


def test_volume_analysis_session_run_pipeline() -> None:
    """Test end-to-end execution of VolumeAnalysisSession."""
    vol = _create_synthetic_volume_set()
    cfg = _create_default_volume_config()

    session = VolumeAnalysisSession()
    session.set_volume(vol, source_id="vol_sample_01")
    assert session.source_provenance is not None
    assert session.source_provenance.measurement_semantics == "nuclear_voxels_3d"

    summary_result = session.run_pipeline(cfg, auto_accept=True)
    assert summary_result.source_id == "vol_sample_01"
    assert len(summary_result.metrics) == 1
    row = summary_result.metrics.iloc[0]
    assert row["metric_name"] == "SOX10_pos_pct"
    assert row["numerator_count"] == 1
    assert row["denominator_count"] == 2
    assert row["percentage"] == 50.0


def test_runtime_config_override_refreshes_3d_pipeline_fingerprint() -> None:
    volume = _create_synthetic_volume_set()
    config = _create_default_volume_config()
    initial_fp = compute_volume_analysis_config_fingerprint(config)
    session = VolumeAnalysisSession()
    session.set_volume(
        volume,
        source_id="mutable-config-volume",
        config_fingerprint=initial_fp,
        pipeline_fingerprint=compute_volume_pipeline_fingerprint(initial_fp),
        analysis_config=config,
    )

    session.segment(Segmentation3DConfig(min_volume_voxels=31))

    assert session.analysis_config is not None
    assert session.source_provenance is not None
    expected_fp = compute_volume_analysis_config_fingerprint(session.analysis_config)
    assert session.source_provenance.volume_config_fingerprint == expected_fp
    assert session.source_provenance.pipeline_fingerprint == compute_volume_pipeline_fingerprint(expected_fp)


def test_bound_3d_config_is_used_when_stage_overrides_are_omitted() -> None:
    volume = _create_synthetic_volume_set()
    config = _create_default_volume_config()
    config_fp = compute_volume_analysis_config_fingerprint(config)
    session = VolumeAnalysisSession()
    session.set_volume(
        volume,
        source_id="bound-config-volume",
        config_fingerprint=config_fp,
        pipeline_fingerprint=compute_volume_pipeline_fingerprint(config_fp),
        analysis_config=config,
    )

    segmentation = session.segment()
    assert segmentation.config == config.segmentation
    session.auto_accept()
    measurement = session.measure()
    assert measurement.marker_names == tuple(marker.name for marker in config.markers)
    classification = session.classify()
    assert classification.thresholds == config.thresholds
    summary = session.summarize()
    assert summary.definitions == config.metrics


def test_study_aggregation_3d_and_heterogeneity() -> None:
    """Test StudyAnalysisSession aggregating 3D samples and tracking heterogeneity."""
    vol = _create_synthetic_volume_set()
    cfg1 = _create_default_volume_config()
    cfg2 = VolumeAnalysisConfig(
        selection=cfg1.selection,
        segmentation=Segmentation3DConfig(min_volume_voxels=30),  # different param
        markers=cfg1.markers,
        thresholds=cfg1.thresholds,
        metrics=cfg1.metrics,
    )

    s1 = VolumeAnalysisSession()
    s1.set_volume(vol, source_id="vol_s1")
    res1 = s1.run_pipeline(cfg1, auto_accept=True)

    s2 = VolumeAnalysisSession()
    s2.set_volume(vol, source_id="vol_s2")
    res2 = s2.run_pipeline(cfg2, auto_accept=True)

    study = StudyAnalysisSession()
    study.add_sample(SampleIdentity("s1", "A1"), res1, s1.source_provenance)
    study.add_sample(SampleIdentity("s2", "A1"), res2, s2.source_provenance)

    agg = study.aggregate()
    assert len(agg.animal_metrics) == 1
    assert agg.animal_metrics["animal_id"].iloc[0] == "A1"
    assert agg.animal_metrics["numerator_count"].iloc[0] == 2
    assert agg.animal_metrics["denominator_count"].iloc[0] == 4

    # Check processing heterogeneity recorded without error
    assert study.processing_heterogeneity is not None
    assert study.processing_heterogeneity.measurement_semantics == "nuclear_voxels_3d"
    assert len(study.processing_heterogeneity.pipeline_fingerprints) == 2


def test_study_aggregation_rejects_mixed_semantics() -> None:
    """Test StudyAnalysisSession rejecting mixed 2D and 3D measurement semantics."""
    vol = _create_synthetic_volume_set()
    cfg = _create_default_volume_config()

    s_vol = VolumeAnalysisSession()
    s_vol.set_volume(vol, source_id="vol_s1")
    res_vol = s_vol.run_pipeline(cfg, auto_accept=True)

    # Fake 2D provenance
    prov_2d = ImageSourceProvenance(
        source_kind="direct_2d",
        source_id="img_s2",
        pipeline_fingerprint="fp_2d",
    )

    study = StudyAnalysisSession()
    study.add_sample(SampleIdentity("s1", "A1"), res_vol, s_vol.source_provenance)
    study.add_sample(SampleIdentity("s2", "A1"), res_vol, prov_2d)

    with pytest.raises(IncompatibleMeasurementSemanticsError, match="Incompatible measurement semantics"):
        study.aggregate()
