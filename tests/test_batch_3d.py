"""Tests for M14 3D batch processing, checkpointing, and resume."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tifffile

from nervenuc.batch import (
    AnalysisPreset,
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    BatchRunner,
    BatchSampleSpec,
    InputSourceChangedError,
    StackFileInput,
    load_checkpoint,
    load_preset,
    resume_batch,
    run_batch,
    save_checkpoint,
    save_preset,
)
from nervenuc.classification import MarkerThreshold
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import VolumeSelectionConfig
from nervenuc.segmentation3d import Segmentation3DConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import VolumeAnalysisConfig


def _create_synthetic_3d_tiff(path: Path, shape: tuple[int, int, int, int] = (2, 8, 32, 32)) -> Path:
    """Create a 4D TIFF file (CZYX) with nuclear and marker channels."""
    # (C, Z, Y, X)
    data = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.ogrid[: shape[1], : shape[2], : shape[3]]

    # Nuclear channel (C=0) at (4, 16, 16)
    mask1 = ((z - 4) / 2.0) ** 2 + ((y - 16) / 5.0) ** 2 + ((x - 16) / 5.0) ** 2 <= 1.0
    data[0, mask1] = 1000

    # Marker channel (C=1)
    data[1, mask1] = 800

    tifffile.imwrite(
        str(path),
        data,
        metadata={"axes": "CZYX"},
    )
    return path


def _create_3d_preset() -> AnalysisPreset:
    sel = VolumeSelectionConfig(
        axes="CZYX",
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
    cfg = VolumeAnalysisConfig(
        selection=sel,
        segmentation=seg,
        markers=markers,
        thresholds=thresholds,
        metrics=metrics,
    )
    return AnalysisPreset(
        name="3D Batch Preset",
        config=cfg,
    )


def test_batch_3d_auto_accept_and_preset_io(tmp_path: Path) -> None:
    """Test 3D batch run in auto_accept mode and preset saving/loading."""
    f1 = _create_synthetic_3d_tiff(tmp_path / "sample1.tif")
    f2 = _create_synthetic_3d_tiff(tmp_path / "sample2.tif")

    preset = _create_3d_preset()
    assert preset.schema_version == 3

    preset_path = tmp_path / "preset_3d.json"
    save_preset(preset, preset_path)
    loaded_preset = load_preset(preset_path)
    assert loaded_preset.schema_version == 3
    assert loaded_preset.config_fingerprint == preset.config_fingerprint
    assert loaded_preset.pipeline_fingerprint == preset.pipeline_fingerprint

    spec1 = BatchSampleSpec("s1", "A1", stack_input=StackFileInput(f1))
    spec2 = BatchSampleSpec("s2", "A2", stack_input=StackFileInput(f2))
    manifest = BatchManifest([spec1, spec2])

    result = run_batch(loaded_preset, manifest, BatchRunConfig(review_mode="auto_accept"))
    assert isinstance(result, BatchResult)
    assert len(result.completed_samples) == 2
    assert all(status == "completed" for status in result.sample_status["status"])

    # To study session
    study = result.to_study_session()
    assert len(study.samples) == 2
    agg = study.aggregate()
    assert len(agg.animal_metrics) == 2


def test_batch_3d_manual_review_and_continue(tmp_path: Path) -> None:
    """Test 3D batch run pausing in manual mode and continuing with ReviewedSegmentation3DResult."""
    f1 = _create_synthetic_3d_tiff(tmp_path / "sample_manual.tif")
    preset = _create_3d_preset()

    spec = BatchSampleSpec("s_man", "A1", stack_input=StackFileInput(f1))
    manifest = BatchManifest([spec])

    runner = BatchRunner(preset, manifest, BatchRunConfig(review_mode="manual"))
    res1 = runner.run()

    assert res1.sample_status["status"].iloc[0] == "review_pending"
    assert "s_man" in res1.review_sessions

    # Perform review
    rev_sess = res1.review_sessions["s_man"]
    rev_sess.auto_accept()
    reviewed = rev_sess.finalize()

    # Continue
    res2 = runner.continue_sample("s_man", reviewed)
    assert res2.sample_status["status"].iloc[0] == "completed"
    assert len(res2.completed_samples) == 1


def test_batch_3d_checkpoint_and_resume(tmp_path: Path) -> None:
    """Test schema 3 checkpoint save and resume under Strategy B."""
    f1 = _create_synthetic_3d_tiff(tmp_path / "sample_chk1.tif")
    f2 = _create_synthetic_3d_tiff(tmp_path / "sample_chk2.tif")
    preset = _create_3d_preset()

    manifest = BatchManifest([
        BatchSampleSpec("s1", "A1", stack_input=StackFileInput(f1)),
        BatchSampleSpec("s2", "A2", stack_input=StackFileInput(f2)),
    ])

    runner = BatchRunner(preset, manifest, BatchRunConfig(review_mode="manual"))
    res = runner.run()

    chk_path = tmp_path / "chk_3d.json"
    save_checkpoint(res, chk_path)

    chk_data = load_checkpoint(chk_path)
    assert chk_data["schema_version"] == 3

    # Resume under Strategy B
    resumed = resume_batch(chk_path, preset, manifest, BatchRunConfig(review_mode="manual"))
    assert resumed.sample_status["status"].tolist() == ["review_pending", "review_pending"]
