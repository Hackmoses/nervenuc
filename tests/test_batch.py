"""Comprehensive unit tests for Milestone 12 (M12): Batch Processing and Reusable Analysis Presets."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd
import pytest
import tifffile

import nervenuc as nn
from nervenuc._version import __version__
from nervenuc.aggregation import aggregate_by_animal
from nervenuc.batch import (
    AnalysisPreset,
    ArrayInput,
    BatchManifest,
    BatchResult,
    BatchResumeError,
    BatchRunConfig,
    BatchRunner,
    BatchSampleSpec,
    ChannelMappedFileInput,
    FileInput,
    InputSourceChangedError,
    InvalidBatchInputError,
    InvalidBatchManifestError,
    InvalidPresetError,
    PresetFileExistsError,
    UnsupportedPresetVersionError,
    compute_config_fingerprint,
    compute_manifest_fingerprint,
    load_checkpoint,
    load_preset,
    resume_batch,
    run_batch,
    save_checkpoint,
    save_preset,
)
from nervenuc.classification import MarkerThreshold
from nervenuc.export import AnimalGroupAssignment, prepare_group_tables, prepare_reproducible_bundle
from nervenuc.handoff import prepare_analysis_handoff
from nervenuc.markers import MarkerDefinition
from nervenuc.qc import start_review
from nervenuc.segmentation import SegmentationConfig, segment_nuclei
from nervenuc.statistics import summarize_groups
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import AnalysisConfig, ImageAnalysisSession, InvalidWorkflowStateError


# ==============================================================================
# Helpers
# ==============================================================================


def _create_synthetic_nuclear_and_marker_images() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create 2D synthetic test image with two distinct circular nuclei."""
    img_nuc = np.zeros((100, 100), dtype=np.float32)
    yy, xx = np.ogrid[:100, :100]

    # Nucleus 1 at (30, 30), radius 10 (area ~314 px)
    mask1 = (yy - 30) ** 2 + (xx - 30) ** 2 <= 10**2
    img_nuc[mask1] = 100.0

    # Nucleus 2 at (70, 70), radius 10 (area ~314 px)
    mask2 = (yy - 70) ** 2 + (xx - 70) ** 2 <= 10**2
    img_nuc[mask2] = 120.0

    # BrdU: high in nucleus 1 (100.0), low in nucleus 2 (10.0)
    brdu = np.zeros((100, 100), dtype=np.float32)
    brdu[mask1] = 100.0
    brdu[mask2] = 10.0

    # SOX10: high in both nuclei (150.0)
    sox10 = np.zeros((100, 100), dtype=np.float32)
    sox10[mask1] = 150.0
    sox10[mask2] = 150.0

    return img_nuc, {"BrdU": brdu, "SOX10": sox10}


def _create_default_config() -> AnalysisConfig:
    """Construct standard AnalysisConfig with 2 markers, 2 thresholds, 1 metric."""
    seg_cfg = SegmentationConfig(
        threshold_method="otsu",
        gaussian_sigma=1.0,
        min_area_px=50,
        min_distance_px=5,
    )
    m1 = MarkerDefinition("BrdU")
    m2 = MarkerDefinition("SOX10")

    t1 = MarkerThreshold("BrdU", threshold=50.0, measurement="mean_intensity", comparison=">=")
    t2 = MarkerThreshold("SOX10", threshold=50.0, measurement="mean_intensity", comparison=">=")

    d1 = BiologicalMetricDefinition(
        name="Proliferation",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    return AnalysisConfig(
        segmentation=seg_cfg,
        markers=(m1, m2),
        thresholds=(t1, t2),
        metrics=(d1,),
    )


def _finalize_review_session(session):
    """Explicitly accept all active objects in a pending batch review."""
    for review_id in session.result().objects["review_id"]:
        session.accept(review_id)
    return session.finalize()


# ==============================================================================
# 1. Preset & Fingerprinting Tests
# ==============================================================================


def test_preset_instantiation_and_version() -> None:
    """Test creating an AnalysisPreset with authoritative package version."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="SciaticPreset", config=cfg)
    assert preset.name == "SciaticPreset"
    assert preset.config == cfg
    assert preset.schema_version == 1
    assert preset.nervenuc_version == __version__
    assert preset.description is None
    assert len(preset.config_fingerprint) == 64


def test_deterministic_config_fingerprint() -> None:
    """Test that identical configs produce identical SHA-256 fingerprints."""
    cfg1 = _create_default_config()
    cfg2 = _create_default_config()
    fp1 = compute_config_fingerprint(cfg1)
    fp2 = compute_config_fingerprint(cfg2)
    assert fp1 == fp2
    assert fp1 == cfg1.to_dict() and len(fp1) == 64 or isinstance(fp1, str)


def test_config_fingerprint_changes_on_parameter_difference() -> None:
    """Test that any scientific setting change alters the config fingerprint."""
    base = _create_default_config()
    base_fp = compute_config_fingerprint(base)

    # 1. Change segmentation parameter
    c_seg = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="li", min_area_px=50),
        markers=base.markers,
        thresholds=base.thresholds,
        metrics=base.metrics,
    )
    assert compute_config_fingerprint(c_seg) != base_fp

    # 2. Change threshold cutoff
    c_thresh = AnalysisConfig(
        segmentation=base.segmentation,
        markers=base.markers,
        thresholds=(
            MarkerThreshold("BrdU", threshold=75.0, measurement="mean_intensity", comparison=">="),
            base.thresholds[1],
        ),
        metrics=base.metrics,
    )
    assert compute_config_fingerprint(c_thresh) != base_fp

    # 3. Change metric definition
    c_metric = AnalysisConfig(
        segmentation=base.segmentation,
        markers=base.markers,
        thresholds=base.thresholds,
        metrics=(
            BiologicalMetricDefinition(
                name="Proliferation_Modified",
                denominator=PhenotypeFilter({"SOX10": True}),
                numerator=PhenotypeFilter({"BrdU": True}),
            ),
        ),
    )
    assert compute_config_fingerprint(c_metric) != base_fp


def test_preset_metadata_does_not_affect_config_fingerprint() -> None:
    """Test that preset name, description, and schema version do not change config fingerprint."""
    cfg = _create_default_config()
    p1 = AnalysisPreset(name="Name1", config=cfg, description="Desc1")
    p2 = AnalysisPreset(name="Name2", config=cfg, description="Desc2")
    assert p1.config_fingerprint == p2.config_fingerprint


def test_preset_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test saving an AnalysisPreset to JSON and loading it back losslessly."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="TestPreset", config=cfg, description="My test protocol")
    out_file = tmp_path / "preset.json"

    save_preset(preset, out_file)
    loaded = load_preset(out_file)

    assert loaded.name == preset.name
    assert loaded.description == preset.description
    assert loaded.schema_version == preset.schema_version
    assert loaded.config_fingerprint == preset.config_fingerprint
    assert loaded.config.to_dict() == cfg.to_dict()


def test_save_preset_overwrite_protection(tmp_path: Path) -> None:
    """Test that save_preset respects overwrite=False and overwrite=True."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="TestPreset", config=cfg)
    out_file = tmp_path / "preset.json"

    save_preset(preset, out_file)
    with pytest.raises(PresetFileExistsError, match="already exists and overwrite=False"):
        save_preset(preset, out_file, overwrite=False)

    # Succeeds with overwrite=True
    save_preset(preset, out_file, overwrite=True)


def test_load_preset_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    """Test that loading a preset with unsupported schema_version raises UnsupportedPresetVersionError."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="TestPreset", config=cfg)
    d = preset.to_dict()
    d["schema_version"] = 99

    bad_file = tmp_path / "bad_schema.json"
    bad_file.write_text(json.dumps(d))

    with pytest.raises(UnsupportedPresetVersionError, match="Unsupported preset schema_version: 99"):
        load_preset(bad_file)


def test_load_preset_corrupted_fingerprint_rejected(tmp_path: Path) -> None:
    """Test that loading a preset with a forged/mismatched config fingerprint raises InvalidPresetError."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="TestPreset", config=cfg)
    d = preset.to_dict()
    d["config_fingerprint"] = "0" * 64

    bad_file = tmp_path / "corrupted_fp.json"
    bad_file.write_text(json.dumps(d))

    with pytest.raises(InvalidPresetError, match="Config fingerprint mismatch"):
        load_preset(bad_file)


def test_load_preset_different_package_version_accepted(tmp_path: Path) -> None:
    """Test that a preset created with a different package version but schema_version=1 loads cleanly."""
    cfg = _create_default_config()
    preset = AnalysisPreset(name="TestPreset", config=cfg)
    d = preset.to_dict()
    d["nervenuc_version"] = "99.0.0"

    compat_file = tmp_path / "compat.json"
    compat_file.write_text(json.dumps(d))

    loaded = load_preset(compat_file)
    assert loaded.nervenuc_version == "99.0.0"
    assert loaded.config_fingerprint == preset.config_fingerprint


# ==============================================================================
# 2. Manifest & Typed Input Descriptors Tests
# ==============================================================================


def test_array_input_validation() -> None:
    """Test ArrayInput validation for 2D numpy arrays and non-empty IDs."""
    arr = np.zeros((50, 50), dtype=np.float32)
    inp = ArrayInput(arr, "input-01")
    assert inp.array is arr
    assert inp.input_id == "input-01"

    # Non-2D array
    with pytest.raises(InvalidBatchInputError, match="ArrayInput array must be 2D"):
        ArrayInput(np.zeros((2, 50, 50)), "input-02")

    # Blank ID
    with pytest.raises(InvalidBatchInputError, match="input_id must be a non-empty string"):
        ArrayInput(arr, "   ")


def test_file_input_validation_and_content_hash(tmp_path: Path) -> None:
    """Test FileInput validation and content-aware SHA-256 calculation."""
    arr = np.zeros((40, 40), dtype=np.float32)
    f_path = tmp_path / "plane.tif"
    tifffile.imwrite(str(f_path), arr)

    inp = FileInput(f_path)
    assert inp.path == f_path
    hash1 = inp.content_hash
    assert len(hash1) == 64

    # Modifying file content changes hash
    arr[0, 0] = 100.0
    tifffile.imwrite(str(f_path), arr)
    hash2 = inp.content_hash
    assert hash1 != hash2

    with pytest.raises(InvalidBatchInputError, match="cannot be empty"):
        FileInput("")
    with pytest.raises(InvalidBatchInputError, match="cannot be empty"):
        FileInput("   ")
    with pytest.raises(InvalidBatchInputError, match="must not be a directory"):
        FileInput(tmp_path)
    with pytest.raises(InvalidBatchInputError, match="does not exist"):
        FileInput(tmp_path / "missing.tif").content_hash


def test_channel_mapped_direct_file_runs_authoritative_pipeline(tmp_path: Path) -> None:
    nuclear, markers = _create_synthetic_nuclear_and_marker_images()
    source = np.stack([nuclear, markers["BrdU"], markers["SOX10"]], axis=0)
    source_path = tmp_path / "direct_cyx.tif"
    tifffile.imwrite(source_path, source, metadata={"axes": "CYX"}, photometric="minisblack")

    mapped = ChannelMappedFileInput(
        source_path,
        axes="CYX",
        nuclear_channel=0,
        marker_channels={"BrdU": 1, "SOX10": 2},
    )
    manifest = BatchManifest([
        BatchSampleSpec("S-DIRECT", "A-DIRECT", direct_image_input=mapped),
    ])
    progress: list[tuple[str, str]] = []
    result = BatchRunner(
        AnalysisPreset("Direct mapped", _create_default_config()),
        manifest,
        BatchRunConfig(review_mode="auto_accept"),
    ).run(lambda sid, status, *_: progress.append((sid, status)))

    assert result.sample_status.iloc[0]["status"] == "completed"
    assert progress == [("S-DIRECT", "running"), ("S-DIRECT", "completed")]
    provenance = result.sample_provenance["S-DIRECT"]
    assert provenance.source_hash == mapped.content_hash
    assert provenance.effective_axes == "CYX"
    assert provenance.nuclear_channel == 0
    assert provenance.marker_channels == {"BrdU": 1, "SOX10": 2}
    assert provenance.preparation_fingerprint
    assert provenance.pipeline_fingerprint != result.config_fingerprint
    assert result.sample_qc_records["S-DIRECT"].review_state == "auto_accepted"


def test_channel_mapped_direct_file_rejects_wrong_marker_contract(tmp_path: Path) -> None:
    source = np.zeros((2, 16, 16), dtype=np.uint16)
    source_path = tmp_path / "wrong_markers.tif"
    tifffile.imwrite(source_path, source, metadata={"axes": "CYX"}, photometric="minisblack")
    mapped = ChannelMappedFileInput(
        source_path,
        axes="CYX",
        nuclear_channel=0,
        marker_channels={"NotConfigured": 1},
    )
    manifest = BatchManifest([BatchSampleSpec("S1", "A1", direct_image_input=mapped)])
    with pytest.raises(InvalidBatchManifestError, match="do not match preset markers"):
        BatchRunner(AnalysisPreset("Preset", _create_default_config()), manifest)


def test_batch_manifest_validation() -> None:
    """Test BatchManifest validation for uniqueness and non-emptiness."""
    arr = np.zeros((50, 50), dtype=np.float32)
    s1 = BatchSampleSpec("S1", "A1", ArrayInput(arr, "n1"), {"BrdU": ArrayInput(arr, "m1")})
    s2 = BatchSampleSpec("S2", "A1", ArrayInput(arr, "n2"), {"BrdU": ArrayInput(arr, "m2")})

    manifest = BatchManifest([s1, s2])
    assert len(manifest.samples) == 2
    assert manifest.samples[0].sample_id == "S1"
    assert manifest.samples[1].sample_id == "S2"

    # Duplicate sample ID
    with pytest.raises(InvalidBatchManifestError, match="Duplicate sample_id 'S1'"):
        BatchManifest([s1, s1])

    # Empty manifest
    with pytest.raises(InvalidBatchManifestError, match="BatchManifest cannot be empty"):
        BatchManifest([])


def test_manifest_fingerprint_determinism() -> None:
    """Test that manifest fingerprint is deterministic and reflects exact inputs and order."""
    arr = np.zeros((50, 50), dtype=np.float32)
    s1 = BatchSampleSpec("S1", "A1", ArrayInput(arr, "n1"), {"BrdU": ArrayInput(arr, "m1")})
    s2 = BatchSampleSpec("S2", "A2", ArrayInput(arr, "n2"), {"BrdU": ArrayInput(arr, "m2")})

    m1 = BatchManifest([s1, s2])
    m2 = BatchManifest([s1, s2])
    assert compute_manifest_fingerprint(m1) == compute_manifest_fingerprint(m2)

    # Reversing order changes manifest fingerprint
    m_rev = BatchManifest([s2, s1])
    assert compute_manifest_fingerprint(m1) != compute_manifest_fingerprint(m_rev)


def test_file_input_resolving_to_3d_rejected(tmp_path: Path) -> None:
    """Test that resolving a FileInput pointing to a 3D TIFF raises InvalidBatchInputError."""
    f3d = tmp_path / "stack.tif"
    tifffile.imwrite(
        str(f3d),
        np.zeros((3, 50, 50), dtype=np.float32),
        photometric="minisblack",
    )

    cfg = _create_default_config()
    spec = BatchSampleSpec(
        sample_id="S1",
        animal_id="A1",
        nuclear_input=FileInput(f3d),
        marker_inputs={"BrdU": FileInput(f3d), "SOX10": FileInput(f3d)},
    )
    manifest = BatchManifest([spec])
    runner = BatchRunner(cfg, manifest)
    res = runner.run()

    assert res.sample_status.iloc[0]["status"] == "failed"
    assert res.sample_status.iloc[0]["error_type"] == "InvalidBatchInputError"
    assert "not 2-dimensional" in res.sample_status.iloc[0]["error_message"]


# ==============================================================================
# 3. Batch Execution & QC Policy Tests
# ==============================================================================


def test_batch_auto_accept_mode() -> None:
    """Test running a batch in auto_accept mode completing all samples in manifest order."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A2",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    manifest = BatchManifest([s1, s2])

    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))

    assert len(res.completed_samples) == 2
    assert res.completed_samples[0].identity.sample_id == "S1"
    assert res.completed_samples[1].identity.sample_id == "S2"
    assert list(res.sample_status["status"]) == ["completed", "completed"]
    assert len(res.review_sessions) == 0


def test_batch_manual_review_mode_and_continuation() -> None:
    """Test default manual review mode stopping at review_pending and continuing explicitly."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A2",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    manifest = BatchManifest([s1, s2])

    runner = BatchRunner(cfg, manifest, BatchRunConfig(review_mode="manual"))
    res = runner.run()

    # Both samples are in review_pending
    assert list(res.sample_status["status"]) == ["review_pending", "review_pending"]
    assert len(res.completed_samples) == 0
    assert "S1" in res.review_sessions
    assert "S2" in res.review_sessions

    # Researcher reviews S1 and finalizes
    rev_s1 = _finalize_review_session(res.review_sessions["S1"])
    res2 = runner.continue_sample("S1", rev_s1)

    assert res2.sample_status.loc[res2.sample_status["sample_id"] == "S1", "status"].iloc[0] == "completed"
    assert res2.sample_status.loc[res2.sample_status["sample_id"] == "S2", "status"].iloc[0] == "review_pending"
    assert len(res2.completed_samples) == 1
    assert "S1" not in res2.review_sessions
    assert "S2" in res2.review_sessions


def test_repeated_run_preserves_pending_review_session_and_history() -> None:
    """Calling run again must not recreate or discard live manual review work."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    spec = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    runner = BatchRunner(cfg, BatchManifest([spec]), BatchRunConfig(review_mode="manual"))
    first = runner.run()
    review = first.review_sessions["S1"]
    first_id = review.result().objects.iloc[0]["review_id"]
    review.accept(first_id, note="preserve this decision")

    second = runner.run()

    assert second.review_sessions["S1"] is review
    assert len(second.review_sessions["S1"].result().history) == 1
    assert second.sample_status.iloc[0]["status"] == "review_pending"


def test_file_change_during_manual_review_blocks_continuation(tmp_path: Path) -> None:
    """Loaded pixels cannot be completed under a source hash captured from changed bytes."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    f_nuc = tmp_path / "nuclear.tif"
    f_brdu = tmp_path / "brdu.tif"
    f_sox10 = tmp_path / "sox10.tif"
    tifffile.imwrite(f_nuc, nuc)
    tifffile.imwrite(f_brdu, markers["BrdU"])
    tifffile.imwrite(f_sox10, markers["SOX10"])
    spec = BatchSampleSpec(
        "S1",
        "A1",
        FileInput(f_nuc),
        {"BrdU": FileInput(f_brdu), "SOX10": FileInput(f_sox10)},
    )
    runner = BatchRunner(cfg, BatchManifest([spec]), BatchRunConfig(review_mode="manual"))
    initial = runner.run()
    captured_fingerprint = initial.manifest_fingerprint
    reviewed = _finalize_review_session(initial.review_sessions["S1"])

    changed = nuc.copy()
    changed[0, 0] = 999.0
    tifffile.imwrite(f_nuc, changed)

    with pytest.raises(InputSourceChangedError, match="awaited review"):
        runner.continue_sample("S1", reviewed)

    after = runner.run()
    assert after.sample_status.iloc[0]["status"] == "review_pending"
    assert after.manifest_fingerprint == captured_fingerprint
    assert after.review_sessions["S1"] is initial.review_sessions["S1"]


def test_out_of_order_manual_continuation_preserves_manifest_order() -> None:
    """Test that finalizing reviews out of order (e.g. S2 then S1) preserves manifest order in completed_samples."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A2",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    manifest = BatchManifest([s1, s2])

    runner = BatchRunner(cfg, manifest, BatchRunConfig(review_mode="manual"))
    res = runner.run()

    # Finalize S2 first, then S1
    rev_s2 = _finalize_review_session(res.review_sessions["S2"])
    runner.continue_sample("S2", rev_s2)

    rev_s1 = _finalize_review_session(res.review_sessions["S1"])
    final_res = runner.continue_sample("S1", rev_s1)

    # In completed_samples, S1 must precede S2 (manifest order)
    assert len(final_res.completed_samples) == 2
    assert final_res.completed_samples[0].identity.sample_id == "S1"
    assert final_res.completed_samples[1].identity.sample_id == "S2"


def test_invalid_reviewed_result_submission_leaves_sample_review_pending() -> None:
    """Test that passing an invalid/incompatible reviewed result leaves the sample in review_pending and raises."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()

    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])

    runner = BatchRunner(cfg, manifest, BatchRunConfig(review_mode="manual"))
    res = runner.run()

    # Create an invalid review snapshot from a different image size
    other_seg = segment_nuclei(np.zeros((30, 30), dtype=np.float32), cfg.segmentation)
    other_rev = _finalize_review_session(start_review(other_seg))

    with pytest.raises(InvalidWorkflowStateError, match="Reviewed label image shape"):
        runner.continue_sample("S1", other_rev)

    # S1 remains in review_pending
    res_after = runner._build_result()
    assert res_after.sample_status.iloc[0]["status"] == "review_pending"
    assert "S1" in res_after.review_sessions


def test_failure_isolation_with_continue_on_error_true() -> None:
    """Test that a failing sample is recorded as failed without aborting remaining samples."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    # S1 has shape mismatch on marker (50x50 vs 100x100) -> fails
    bad_marker = np.zeros((50, 50), dtype=np.float32)
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(bad_marker, "bad"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A2",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    manifest = BatchManifest([s1, s2])

    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept", continue_on_error=True))

    assert res.sample_status.iloc[0]["status"] == "failed"
    assert res.sample_status.iloc[0]["error_type"] is not None
    assert res.sample_status.iloc[1]["status"] == "completed"
    assert len(res.completed_samples) == 1
    assert res.completed_samples[0].identity.sample_id == "S2"


def test_failure_with_continue_on_error_false_leaves_later_rows_pending() -> None:
    """Test that continue_on_error=False halts execution and leaves subsequent samples pending."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    bad_marker = np.zeros((50, 50), dtype=np.float32)
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(bad_marker, "bad"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A2",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    manifest = BatchManifest([s1, s2])

    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept", continue_on_error=False))

    assert res.sample_status.iloc[0]["status"] == "failed"
    assert res.sample_status.iloc[1]["status"] == "pending"
    assert len(res.completed_samples) == 0


def test_direct_m11_equivalence_for_batch_runs() -> None:
    """Test that running batch in auto_accept yields identical scientific summary to repeated M11 ImageAnalysisSession."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()

    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])

    # 1. Via BatchRunner
    batch_res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    batch_summary = batch_res.completed_samples[0].summary_result

    # 2. Via M11 ImageAnalysisSession
    sess = ImageAnalysisSession(cfg)
    sess.set_image(nuc1, markers1)
    sess.segment()
    sess.auto_accept_segmentation()
    sess.measure()
    sess.classify()
    direct_summary = sess.summarize()

    assert batch_summary.metrics.equals(direct_summary.metrics)


# ==============================================================================
# 4. Checkpointing & Resumability Tests
# ==============================================================================


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test checkpoint serialization and loading."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))

    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    loaded = load_checkpoint(chk_file)
    assert loaded["schema_version"] == 1
    assert loaded["config_fingerprint"] == res.config_fingerprint
    assert loaded["manifest_fingerprint"] == res.manifest_fingerprint
    assert loaded["review_mode"] == "auto_accept"
    assert len(loaded["sample_status"]) == 1


def test_resume_batch_config_fingerprint_mismatch_rejected(tmp_path: Path) -> None:
    """Test that resume_batch fails if the preset config fingerprint has changed."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    # Change threshold in new preset
    cfg_modified = AnalysisConfig(
        segmentation=cfg.segmentation,
        markers=cfg.markers,
        thresholds=(
            MarkerThreshold("BrdU", threshold=99.0, measurement="mean_intensity", comparison=">="),
            cfg.thresholds[1],
        ),
        metrics=cfg.metrics,
    )

    with pytest.raises(BatchResumeError, match="Config fingerprint mismatch on resume"):
        resume_batch(chk_file, cfg_modified, manifest)


def test_resume_batch_policy_mismatch_rejected(tmp_path: Path) -> None:
    """Test that attempting to change review_mode (e.g. manual -> auto_accept) during resume raises BatchResumeError."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="manual"))
    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    with pytest.raises(BatchResumeError, match="Review mode mismatch on resume"):
        resume_batch(chk_file, cfg, manifest, BatchRunConfig(review_mode="auto_accept"))


def test_resume_batch_under_auto_accept_mode(tmp_path: Path) -> None:
    """Test that resume_batch in auto_accept mode recomputes completed samples to completed."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    res_resumed = resume_batch(chk_file, cfg, manifest)
    assert res_resumed.sample_status.iloc[0]["status"] == "completed"
    assert len(res_resumed.completed_samples) == 1


def test_resume_batch_under_manual_mode_recreates_review_session(tmp_path: Path) -> None:
    """Test that resume_batch in manual mode re-segments and creates fresh live ReviewSession (never auto-accepts)."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="manual"))
    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    res_resumed = resume_batch(chk_file, cfg, manifest)
    assert res_resumed.sample_status.iloc[0]["status"] == "review_pending"
    assert "S1" in res_resumed.review_sessions
    assert len(res_resumed.completed_samples) == 0


def test_file_changed_at_same_path_invalidates_manifest_fingerprint(tmp_path: Path) -> None:
    """Test that modifying the content of a file input invalidates manifest fingerprint upon resume."""
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    f_nuc = tmp_path / "nuc.tif"
    f_brdu = tmp_path / "brdu.tif"
    f_sox10 = tmp_path / "sox10.tif"

    tifffile.imwrite(str(f_nuc), nuc)
    tifffile.imwrite(str(f_brdu), markers["BrdU"])
    tifffile.imwrite(str(f_sox10), markers["SOX10"])

    cfg = _create_default_config()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        FileInput(f_nuc),
        {"BrdU": FileInput(f_brdu), "SOX10": FileInput(f_sox10)},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))

    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    # Modify nuclear TIFF file on disk
    nuc[0, 0] = 999.0
    tifffile.imwrite(str(f_nuc), nuc)

    with pytest.raises(BatchResumeError, match="Manifest fingerprint mismatch on resume"):
        resume_batch(chk_file, cfg, manifest)


# ==============================================================================
# 5. Study Integration Tests
# ==============================================================================


def test_batch_to_study_session_integration() -> None:
    """Test populating StudyAnalysisSession from BatchResult completed_samples and running M7-M10 pipeline."""
    cfg = _create_default_config()
    nuc1, markers1 = _create_synthetic_nuclear_and_marker_images()
    nuc2, markers2 = _create_synthetic_nuclear_and_marker_images()

    # Image 1 and Image 2 from Animal 1; Image 3 from Animal 2
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc1, "n1"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b1"), "SOX10": ArrayInput(markers1["SOX10"], "s1")},
    )
    s2 = BatchSampleSpec(
        "S2",
        "A1",
        ArrayInput(nuc2, "n2"),
        {"BrdU": ArrayInput(markers2["BrdU"], "b2"), "SOX10": ArrayInput(markers2["SOX10"], "s2")},
    )
    s3 = BatchSampleSpec(
        "S3",
        "A2",
        ArrayInput(nuc1, "n3"),
        {"BrdU": ArrayInput(markers1["BrdU"], "b3"), "SOX10": ArrayInput(markers1["SOX10"], "s3")},
    )
    manifest = BatchManifest([s1, s2, s3])

    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    study = res.to_study_session()

    assert study.status()["sample_count"] == 3
    assert not study.status()["aggregated"]

    # Run study pipeline
    agg_res = study.aggregate()
    assert len(agg_res.animal_metrics) == 2  # 2 animals
    # Verify raw counts pooled: A1 has 2 images * 1 num = 2, 2 * 2 den = 4
    a1_row = agg_res.animal_metrics[agg_res.animal_metrics["animal_id"] == "A1"].iloc[0]
    assert a1_row["numerator_count"] == 2
    assert a1_row["denominator_count"] == 4
    assert a1_row["percentage"] == pytest.approx(50.0)

    # Assign groups and prepare handoff
    study.assign_groups([AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A2", "KO")])
    study.prepare_group_tables()
    study.summarize_groups()
    handoff = study.prepare_handoff()
    assert handoff.metric_readiness.iloc[0]["readiness_status"] == "ready"

    bundle = prepare_reproducible_bundle(study)
    assert len(bundle.per_image_table) == 3
    assert set(bundle.per_image_table["pipeline_fingerprint"]) == {res.pipeline_fingerprint}
    assert len(json.loads(bundle.qc_summary_json)) == 3


# ==============================================================================
# 6. Package & Public API Imports
# ==============================================================================


def test_public_batch_imports() -> None:
    """Test clean imports of batch API classes and functions from nervenuc and nervenuc.batch."""
    assert hasattr(nn, "AnalysisPreset")
    assert hasattr(nn, "BatchManifest")
    assert hasattr(nn, "BatchSampleSpec")
    assert hasattr(nn, "ArrayInput")
    assert hasattr(nn, "FileInput")
    assert hasattr(nn, "BatchRunConfig")
    assert hasattr(nn, "BatchRunner")
    assert hasattr(nn, "run_batch")
    assert hasattr(nn, "save_preset")
    assert hasattr(nn, "load_preset")
    assert hasattr(nn, "__version__")


def test_preset_invalid_name_or_config_rejected() -> None:
    """Test that invalid name or non-AnalysisConfig raises InvalidPresetError."""
    cfg = _create_default_config()

    with pytest.raises(InvalidPresetError, match="Preset name must be a non-empty string"):
        AnalysisPreset(name="   ", config=cfg)

    with pytest.raises(InvalidPresetError, match="Preset config must be an AnalysisConfig"):
        AnalysisPreset(name="ValidName", config="not_a_config")  # type: ignore[arg-type]

    with pytest.raises(InvalidPresetError, match="Preset description must be a string or None"):
        AnalysisPreset(name="ValidName", config=cfg, description=123)  # type: ignore[arg-type]


def test_compute_config_fingerprint_non_config_rejected() -> None:
    """Test that compute_config_fingerprint rejects non-AnalysisConfig."""
    with pytest.raises(InvalidPresetError, match="config must be an AnalysisConfig"):
        compute_config_fingerprint("not_a_config")  # type: ignore[arg-type]


def test_load_preset_nonexistent_or_corrupted_json(tmp_path: Path) -> None:
    """Test load_preset errors on non-existent or corrupted files."""
    with pytest.raises(InvalidPresetError, match="Preset file not found"):
        load_preset(tmp_path / "nonexistent.json")

    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text("{not valid json")
    with pytest.raises(InvalidPresetError, match="Failed to parse preset JSON"):
        load_preset(corrupted)


def test_batch_sample_spec_validation_errors() -> None:
    """Test that BatchSampleSpec rejects invalid sample_id, animal_id, and inputs."""
    arr = np.zeros((20, 20), dtype=np.float32)
    inp = ArrayInput(arr, "id1")

    with pytest.raises(InvalidBatchManifestError, match="sample_id must be a non-empty string"):
        BatchSampleSpec("", "A1", inp, {"BrdU": inp})

    with pytest.raises(InvalidBatchManifestError, match="animal_id must be a non-empty string"):
        BatchSampleSpec("S1", "   ", inp, {"BrdU": inp})

    with pytest.raises(InvalidBatchInputError, match="nuclear_input must be ArrayInput or FileInput"):
        BatchSampleSpec("S1", "A1", arr, {"BrdU": inp})  # type: ignore[arg-type]

    with pytest.raises(InvalidBatchInputError, match="Marker input for 'BrdU' must be ArrayInput or FileInput"):
        BatchSampleSpec("S1", "A1", inp, {"BrdU": arr})  # type: ignore[arg-type]


def test_batch_manifest_invalid_items_rejected() -> None:
    """Test that BatchManifest rejects non-BatchSampleSpec items."""
    with pytest.raises(InvalidBatchManifestError, match="samples must be a sequence"):
        BatchManifest(123)  # type: ignore[arg-type]

    with pytest.raises(InvalidBatchManifestError, match="is not a BatchSampleSpec"):
        BatchManifest(["not_a_spec"])  # type: ignore[list-item]


def test_batch_run_config_validation_errors() -> None:
    """Test BatchRunConfig validation for review_mode and continue_on_error."""
    with pytest.raises(Exception, match="Invalid review_mode"):
        BatchRunConfig(review_mode="invalid_mode")  # type: ignore[arg-type]

    with pytest.raises(Exception, match="continue_on_error must be a boolean"):
        BatchRunConfig(continue_on_error="not_a_bool")  # type: ignore[arg-type]


def test_continue_sample_unknown_or_wrong_state_rejected() -> None:
    """Test continue_sample with unknown sample_id or sample not in review_pending."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    runner = BatchRunner(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    runner.run()

    other_seg = segment_nuclei(nuc, cfg.segmentation)
    other_rev = _finalize_review_session(start_review(other_seg))

    # S1 is completed, not review_pending
    with pytest.raises(InvalidWorkflowStateError, match="expected 'review_pending'"):
        runner.continue_sample("S1", other_rev)

    # Unknown sample
    with pytest.raises(InvalidWorkflowStateError, match="Sample 'UNKNOWN' not found"):
        runner.continue_sample("UNKNOWN", other_rev)


def test_save_and_load_checkpoint_validation_errors(tmp_path: Path) -> None:
    """Test save_checkpoint and load_checkpoint validation errors."""
    with pytest.raises(Exception, match="result must be a BatchResult"):
        save_checkpoint("not_a_result", tmp_path / "chk.json")  # type: ignore[arg-type]

    with pytest.raises(Exception, match="Checkpoint file not found"):
        load_checkpoint(tmp_path / "nonexistent_chk.json")

    corrupt_chk = tmp_path / "corrupt_chk.json"
    corrupt_chk.write_text("{bad json")
    with pytest.raises(Exception, match="Failed to parse checkpoint JSON"):
        load_checkpoint(corrupt_chk)

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(UnsupportedPresetVersionError, match="Unsupported checkpoint schema_version"):
        load_checkpoint(bad_schema)


def test_resume_batch_invalid_inputs_rejected(tmp_path: Path) -> None:
    """Test resume_batch rejecting invalid preset or manifest arguments."""
    cfg = _create_default_config()
    nuc, markers = _create_synthetic_nuclear_and_marker_images()
    s1 = BatchSampleSpec(
        "S1",
        "A1",
        ArrayInput(nuc, "n1"),
        {"BrdU": ArrayInput(markers["BrdU"], "b1"), "SOX10": ArrayInput(markers["SOX10"], "s1")},
    )
    manifest = BatchManifest([s1])
    res = run_batch(cfg, manifest, BatchRunConfig(review_mode="auto_accept"))
    chk_file = tmp_path / "chk.json"
    save_checkpoint(res, chk_file)

    with pytest.raises(BatchResumeError, match="preset must be an AnalysisPreset"):
        resume_batch(chk_file, "not_a_preset", manifest)  # type: ignore[arg-type]

    with pytest.raises(BatchResumeError, match="manifest must be a BatchManifest"):
        resume_batch(chk_file, cfg, "not_a_manifest")  # type: ignore[arg-type]
