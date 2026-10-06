"""Comprehensive test suite for M13 Z-stack preparation, projection, provenance, and workflow integration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd
import pytest
import tifffile

import nervenuc as nn
from nervenuc.aggregation import SampleIdentity
from nervenuc.batch import (
    AnalysisPreset,
    ArrayInput,
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    BatchSampleSpec,
    FileInput,
    InputSourceChangedError,
    InvalidBatchManifestError,
    StackFileInput,
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
from nervenuc.export import AnimalGroupAssignment
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import (
    AmbiguousImageAxesError,
    AxisLayout,
    DirectImageSelectionConfig,
    InvalidAxisLayoutError,
    InvalidChannelSelectionError,
    InvalidProjectionError,
    InvalidZSelectionError,
    NerveNucPreparationError,
    PreparedImageSet,
    ProjectionResult,
    StackInspectionResult,
    StackPreparationConfig,
    VolumeSelectionConfig,
    ZSelection,
    compute_preparation_fingerprint,
    extract_subvolume,
    inspect_stack,
    prepare_stack,
    project_stack,
    resolve_axes,
)
from nervenuc.qc import ReviewedSegmentationResult
from nervenuc.segmentation import SegmentationConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import (
    AnalysisConfig,
    ImageAnalysisSession,
    ImageSourceProvenance,
    PreparationCompatibilityError,
    StudyAnalysisSession,
    compute_analysis_config_fingerprint,
    compute_pipeline_fingerprint,
    validate_preparation_analysis_compatibility,
)


@pytest.fixture
def sample_analysis_config() -> AnalysisConfig:
    """Fixture providing a standard AnalysisConfig for testing."""
    return AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="otsu", min_area_px=10),
        markers=(
            MarkerDefinition(name="NeuN"),
            MarkerDefinition(name="GFAP"),
        ),
        thresholds=(
            MarkerThreshold(marker_name="NeuN", threshold=100.0, measurement="mean_intensity", comparison=">="),
            MarkerThreshold(marker_name="GFAP", threshold=150.0, measurement="mean_intensity", comparison=">="),
        ),
        metrics=(
            BiologicalMetricDefinition(
                name="pct_neun_positive",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"NeuN": True}),
            ),
        ),
    )


@pytest.fixture
def synthetic_stack_data() -> np.ndarray:
    """Create a 4D (C, Z, Y, X) uint16 synthetic stack."""
    # Shape: 3 channels, 10 Z-slices, 30 Y, 30 X
    rng = np.random.default_rng(42)
    data = rng.integers(10, 500, size=(3, 10, 30, 30), dtype=np.uint16)
    # Put a clear bright nucleus in channel 0 at Z=3..6, Y=10..15, X=10..15
    data[0, 3:7, 10:16, 10:16] = 2000
    # Put NeuN signal in channel 1 at Z=3..6, Y=10..16, X=10..16
    data[1, 3:7, 10:16, 10:16] = 1500
    # Put GFAP signal in channel 2 at Z=7..9, Y=20..25, X=20..25
    data[2, 7:10, 20:26, 20:26] = 1800
    return data


@pytest.fixture
def synthetic_stack_tiff(tmp_path: Path, synthetic_stack_data: np.ndarray) -> Path:
    """Save synthetic stack to an OME-TIFF file."""
    path = tmp_path / "synthetic_stack.ome.tif"
    # tifffile writes OME-TIFF metadata
    tifffile.imwrite(
        str(path),
        synthetic_stack_data,
        photometric="minisblack",
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": ["DAPI", "NeuN", "GFAP"]},
        },
    )
    return path


# ==============================================================================
# Module Isolation & Dependency Checks (Tests 1–3)
# ==============================================================================


def test_01_preparation_module_isolation() -> None:
    """Verify nervenuc.preparation can be imported in isolation without importing workflow or batch."""
    # Check loaded modules before/after
    import nervenuc.preparation as prep
    assert hasattr(prep, "prepare_stack")
    assert hasattr(prep, "StackPreparationConfig")


def test_02_public_api_availability() -> None:
    """Verify all public preparation and workflow symbols are exported at package root."""
    assert hasattr(nn, "inspect_stack")
    assert hasattr(nn, "prepare_stack")
    assert hasattr(nn, "project_stack")
    assert hasattr(nn, "resolve_axes")
    assert hasattr(nn, "extract_subvolume")
    assert hasattr(nn, "StackPreparationConfig")
    assert hasattr(nn, "ZSelection")
    assert hasattr(nn, "PreparedImageSet")
    assert hasattr(nn, "ImageSourceProvenance")
    assert hasattr(nn, "StackFileInput")
    assert hasattr(nn, "compute_pipeline_fingerprint")


def test_03_historical_analysis_config_fingerprint_regression(sample_analysis_config: AnalysisConfig) -> None:
    """Verify that workflow compute_analysis_config_fingerprint is byte-for-byte identical to M12."""
    fp_workflow = compute_analysis_config_fingerprint(sample_analysis_config)
    fp_batch = compute_config_fingerprint(sample_analysis_config)
    assert fp_workflow == fp_batch
    assert isinstance(fp_workflow, str)
    assert len(fp_workflow) == 64


# ==============================================================================
# Axis Handling & Resolution (Tests 4–11)
# ==============================================================================


def test_04_valid_2d_yx_axis_resolution() -> None:
    """Verify 2D shape defaults to YX."""
    layout = resolve_axes(shape=(100, 100))
    assert layout.axes == "YX"
    assert layout.shape == (100, 100)


def test_05_valid_3d_zyx_and_4d_czyx_resolution() -> None:
    """Verify explicit declared axes for 3D and 4D."""
    layout_zyx = resolve_axes(shape=(10, 100, 100), declared_axes="ZYX")
    assert layout_zyx.axes == "ZYX"

    layout_czyx = resolve_axes(shape=(3, 10, 100, 100), declared_axes="CZYX")
    assert layout_czyx.axes == "CZYX"


def test_06_valid_5d_tczyx_resolution() -> None:
    """Verify 5D TCZYX layout resolution."""
    layout = resolve_axes(shape=(2, 3, 10, 100, 100), declared_axes="TCZYX")
    assert layout.axes == "TCZYX"
    assert layout.size_of("T") == 2
    assert layout.size_of("C") == 3
    assert layout.size_of("Z") == 10


def test_07_invalid_duplicate_axis_symbol() -> None:
    """Verify duplicate axis symbols are rejected."""
    with pytest.raises(InvalidAxisLayoutError, match="Duplicate axis symbol"):
        AxisLayout("ZZYX", (5, 5, 100, 100))


def test_08_missing_spatial_axes_rejected() -> None:
    """Verify missing Y or X raises InvalidAxisLayoutError."""
    with pytest.raises(InvalidAxisLayoutError, match="Spatial axes 'Y' and 'X' must both be present"):
        AxisLayout("CZ", (3, 10))


def test_09_ambiguous_plain_3d_array_rejected() -> None:
    """Verify plain 3D array without metadata or declared axes raises AmbiguousImageAxesError."""
    with pytest.raises(AmbiguousImageAxesError, match="Explicit axes declaration is required"):
        resolve_axes(shape=(3, 100, 100))


def test_10_unsupported_axis_symbol_rejected() -> None:
    """Verify unsupported axis symbols like S (samples) are rejected."""
    with pytest.raises(InvalidAxisLayoutError, match="Invalid scientific axis symbol 'S'"):
        AxisLayout("SYX", (3, 100, 100))


def test_11_time_axis_without_time_index_rejected(synthetic_stack_data: np.ndarray) -> None:
    """Verify T > 1 without time_index raises AmbiguousImageAxesError during extraction."""
    data_5d = np.stack([synthetic_stack_data, synthetic_stack_data], axis=0)  # Shape (2, 3, 10, 30, 30)
    layout = AxisLayout("TCZYX", data_5d.shape)
    with pytest.raises(AmbiguousImageAxesError, match="time_index is None"):
        extract_subvolume(data_5d, layout, time_index=None, channel_index=0)


# ==============================================================================
# Channel Selection & Integration Compatibility (Tests 12–18)
# ==============================================================================


def test_12_explicit_channel_extraction(synthetic_stack_data: np.ndarray) -> None:
    """Verify subvolume extraction for explicit channels."""
    layout = AxisLayout("CZYX", synthetic_stack_data.shape)
    subvol_0, z_start, z_stop = extract_subvolume(synthetic_stack_data, layout, channel_index=0)
    assert subvol_0.shape == (10, 30, 30)
    assert z_start == 0
    assert z_stop == 10


def test_13_out_of_bounds_channel_rejected(synthetic_stack_data: np.ndarray) -> None:
    """Verify out-of-bounds channel index raises InvalidChannelSelectionError."""
    layout = AxisLayout("CZYX", synthetic_stack_data.shape)
    with pytest.raises(InvalidChannelSelectionError, match="out of bounds"):
        extract_subvolume(synthetic_stack_data, layout, channel_index=99)


def test_14_duplicate_channel_role_rejected() -> None:
    """Verify mapping the same channel to multiple roles is rejected in StackPreparationConfig."""
    with pytest.raises(InvalidChannelSelectionError, match="Duplicate channel role assignments"):
        StackPreparationConfig(
            nuclear_channel=0,
            marker_channels={"NeuN": 0},  # Duplicate!
        )


def test_15_single_channel_zyx_stack_channel_0() -> None:
    """Verify single-channel ZYX stack handles channel_index=0 cleanly."""
    data_zyx = np.ones((5, 20, 20), dtype=np.uint16)
    layout = AxisLayout("ZYX", data_zyx.shape)
    subvol, _, _ = extract_subvolume(data_zyx, layout, channel_index=0)
    assert subvol.shape == (5, 20, 20)


def test_16_single_channel_zyx_stack_rejects_nonzero_channel() -> None:
    """Verify single-channel ZYX stack rejects non-zero channel index."""
    data_zyx = np.ones((5, 20, 20), dtype=np.uint16)
    layout = AxisLayout("ZYX", data_zyx.shape)
    with pytest.raises(InvalidChannelSelectionError, match="Stack has no 'C' axis"):
        extract_subvolume(data_zyx, layout, channel_index=1)


def test_17_inspect_stack_exposes_metadata(synthetic_stack_tiff: Path) -> None:
    """Verify inspect_stack factually summarizes stack dimensions and OME channel names."""
    info = inspect_stack(synthetic_stack_tiff)
    assert info.is_ome is True
    assert info.size_c == 3
    assert info.size_z == 10
    assert info.channel_names == ("DAPI", "NeuN", "GFAP")


def test_18_validate_preparation_analysis_compatibility(sample_analysis_config: AnalysisConfig) -> None:
    """Verify preparation / AnalysisConfig marker compatibility validation."""
    # Exact match
    valid_prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    validate_preparation_analysis_compatibility(valid_prep, sample_analysis_config)

    # Missing marker
    missing_prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1},
    )
    with pytest.raises(PreparationCompatibilityError, match="missing preparation for configured markers"):
        validate_preparation_analysis_compatibility(missing_prep, sample_analysis_config)

    # Extra marker
    extra_prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2, "Extra": 3},
    )
    with pytest.raises(PreparationCompatibilityError, match="extra unconfigured markers"):
        validate_preparation_analysis_compatibility(extra_prep, sample_analysis_config)

    # Case mismatch
    case_prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"neun": 1, "GFAP": 2},
    )
    with pytest.raises(PreparationCompatibilityError, match="missing preparation"):
        validate_preparation_analysis_compatibility(case_prep, sample_analysis_config)


# ==============================================================================
# Z Selection & Geometry (Tests 19–24)
# ==============================================================================


def test_19_full_stack_z_selection(synthetic_stack_data: np.ndarray) -> None:
    """Verify default ZSelection None uses entire depth."""
    layout = AxisLayout("CZYX", synthetic_stack_data.shape)
    subvol, z_start, z_stop = extract_subvolume(synthetic_stack_data, layout, channel_index=0, z_selection=None)
    assert subvol.shape == (10, 30, 30)
    assert (z_start, z_stop) == (0, 10)


def test_20_valid_subrange_z_selection(synthetic_stack_data: np.ndarray) -> None:
    """Verify half-open sub-range ZSelection."""
    layout = AxisLayout("CZYX", synthetic_stack_data.shape)
    z_sel = ZSelection(start=2, stop=7)
    subvol, z_start, z_stop = extract_subvolume(synthetic_stack_data, layout, channel_index=0, z_selection=z_sel)
    assert subvol.shape == (5, 30, 30)
    assert (z_start, z_stop) == (2, 7)


def test_21_single_plane_z_stack() -> None:
    """Verify single-plane Z stack (size_z = 1) is extracted cleanly."""
    data = np.ones((1, 1, 20, 20), dtype=np.uint16)
    layout = AxisLayout("CZYX", data.shape)
    subvol, z_start, z_stop = extract_subvolume(data, layout, channel_index=0, z_selection=ZSelection(0, 1))
    assert subvol.shape == (1, 20, 20)
    assert (z_start, z_stop) == (0, 1)


def test_22_out_of_bounds_z_stop_rejected(synthetic_stack_data: np.ndarray) -> None:
    """Verify out-of-bounds Z stop raises InvalidZSelectionError without silent clamping."""
    layout = AxisLayout("CZYX", synthetic_stack_data.shape)
    z_sel = ZSelection(start=0, stop=15)
    with pytest.raises(InvalidZSelectionError, match="exceeds stack depth"):
        extract_subvolume(synthetic_stack_data, layout, channel_index=0, z_selection=z_sel)


def test_23_inverted_z_range_rejected() -> None:
    """Verify start >= stop raises InvalidZSelectionError."""
    with pytest.raises(InvalidZSelectionError, match="strictly greater than start"):
        ZSelection(start=5, stop=3)
    with pytest.raises(InvalidZSelectionError, match="strictly greater than start"):
        ZSelection(start=5, stop=5)


def test_24_negative_z_start_rejected() -> None:
    """Verify start < 0 raises InvalidZSelectionError."""
    with pytest.raises(InvalidZSelectionError, match="must be non-negative"):
        ZSelection(start=-1, stop=5)


# ==============================================================================
# Projection Math, Dtypes, & Non-Finite Values (Tests 25–33)
# ==============================================================================


def test_25_maximum_intensity_projection_math() -> None:
    """Verify max intensity projection mathematical correctness."""
    subvol = np.array([
        [[10, 20], [30, 40]],
        [[50, 15], [25, 45]],
        [[5, 30], [35, 10]],
    ], dtype=np.uint16)
    res = project_stack(subvol, method="max")
    expected = np.array([[50, 30], [35, 45]], dtype=np.uint16)
    np.testing.assert_array_equal(res.image, expected)


def test_26_mean_intensity_projection_math() -> None:
    """Verify mean intensity projection mathematical correctness."""
    subvol = np.array([
        [[10, 20], [30, 40]],
        [[20, 20], [30, 50]],
        [[30, 50], [60, 60]],
    ], dtype=np.uint16)
    res = project_stack(subvol, method="mean")
    expected = np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float64)
    np.testing.assert_allclose(res.image, expected)


def test_27_max_projection_preserves_dtype() -> None:
    """Verify max projection preserves uint16, uint8, and float32 dtypes."""
    for dt in [np.uint8, np.uint16, np.uint32, np.float32]:
        subvol = np.ones((5, 10, 10), dtype=dt)
        res = project_stack(subvol, method="max")
        assert res.image.dtype == dt


def test_28_mean_projection_produces_float64() -> None:
    """Verify mean projection always produces float64 regardless of integer input."""
    for dt in [np.uint8, np.uint16, np.int32]:
        subvol = np.ones((5, 10, 10), dtype=dt)
        res = project_stack(subvol, method="mean")
        assert res.image.dtype == np.float64


def test_29_projection_output_strictly_2d() -> None:
    """Verify projection output is exactly 2D."""
    subvol = np.zeros((4, 25, 25), dtype=np.uint16)
    res = project_stack(subvol, method="max")
    assert res.image.ndim == 2
    assert res.image.shape == (25, 25)


def test_30_source_array_immutability() -> None:
    """Verify source array is unchanged by projection operations."""
    subvol = np.ones((4, 10, 10), dtype=np.uint16)
    subvol_copy = subvol.copy()
    _ = project_stack(subvol, method="max")
    _ = project_stack(subvol, method="mean")
    np.testing.assert_array_equal(subvol, subvol_copy)


def test_31_unsupported_projection_method_rejected() -> None:
    """Verify unsupported projection method raises InvalidProjectionError."""
    subvol = np.zeros((3, 10, 10), dtype=np.uint16)
    with pytest.raises(InvalidProjectionError, match="Unsupported projection method"):
        project_stack(subvol, method="median")  # type: ignore


def test_32_non_numeric_dtypes_rejected() -> None:
    """Verify bool and complex arrays are rejected."""
    bool_arr = np.zeros((3, 10, 10), dtype=bool)
    with pytest.raises(InvalidProjectionError, match="Array dtype 'bool' is not supported"):
        project_stack(bool_arr, method="max")

    complex_arr = np.zeros((3, 10, 10), dtype=np.complex128)
    with pytest.raises(InvalidProjectionError, match="not supported"):
        project_stack(complex_arr, method="max")


def test_33_non_finite_values_rejected() -> None:
    """Verify NaN and Inf values raise InvalidProjectionError."""
    nan_arr = np.zeros((3, 10, 10), dtype=np.float32)
    nan_arr[1, 5, 5] = np.nan
    with pytest.raises(InvalidProjectionError, match="Non-finite values"):
        project_stack(nan_arr, method="max")

    inf_arr = np.zeros((3, 10, 10), dtype=np.float32)
    inf_arr[1, 5, 5] = np.inf
    with pytest.raises(InvalidProjectionError, match="Non-finite values"):
        project_stack(inf_arr, method="max")


# ==============================================================================
# Provenance, Fingerprints, & Lifecycles (Tests 34–41)
# ==============================================================================


def test_34_projection_result_records_provenance() -> None:
    """Verify ProjectionResult accurately records extraction facts."""
    subvol = np.ones((6, 12, 12), dtype=np.uint16)
    res = project_stack(
        subvol,
        method="max",
        physical_size_z_um=1.5,
        metadata_axes="CZYX",
        declared_axes="CZYX",
        effective_axes="CZYX",
        channel_index=1,
        time_index=None,
        z_start=2,
        z_stop=8,
        source_shape=(3, 10, 12, 12),
    )
    assert res.channel_index == 1
    assert (res.z_start, res.z_stop) == (2, 8)
    assert res.physical_size_z_um == 1.5
    assert res.source_shape == (3, 10, 12, 12)
    assert res.method == "max"


def test_35_preparation_fingerprint_determinism() -> None:
    """Verify compute_preparation_fingerprint produces deterministic SHA-256."""
    config1 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1}, z_selection=ZSelection(2, 8))
    config2 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1}, z_selection=ZSelection(2, 8))
    fp1 = compute_preparation_fingerprint(config1)
    fp2 = compute_preparation_fingerprint(config2)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_36_pipeline_fingerprint_composition() -> None:
    """Verify compute_pipeline_fingerprint domain separation."""
    cfg_fp = "a" * 64
    prep_fp = "b" * 64
    pipe_fp = compute_pipeline_fingerprint(cfg_fp, prep_fp)
    assert pipe_fp != cfg_fp
    assert pipe_fp != prep_fp
    assert len(pipe_fp) == 64


def test_37_z_range_change_modifies_pipeline_fingerprint(sample_analysis_config: AnalysisConfig) -> None:
    """Verify changing Z-range changes preparation & pipeline fingerprints while config fingerprint is stable."""
    cfg_fp = compute_analysis_config_fingerprint(sample_analysis_config)

    p1 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2}, z_selection=ZSelection(0, 5))
    p2 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2}, z_selection=ZSelection(0, 8))

    preset1 = AnalysisPreset("P1", sample_analysis_config, preparation=p1)
    preset2 = AnalysisPreset("P2", sample_analysis_config, preparation=p2)

    assert preset1.config_fingerprint == cfg_fp
    assert preset2.config_fingerprint == cfg_fp
    assert preset1.preparation_fingerprint != preset2.preparation_fingerprint
    assert preset1.pipeline_fingerprint != preset2.pipeline_fingerprint


def test_38_preset_schema_lifecycle(sample_analysis_config: AnalysisConfig) -> None:
    """Verify schema_version 1 for 2D presets and schema_version 2 for stack presets."""
    p_2d = AnalysisPreset("2D", sample_analysis_config)
    assert p_2d.schema_version == 1
    assert p_2d.preparation is None

    prep = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2})
    p_stack = AnalysisPreset("Stack", sample_analysis_config, preparation=prep)
    assert p_stack.schema_version == 2
    assert p_stack.preparation is not None


def test_39_legacy_preset_load_save(tmp_path: Path, sample_analysis_config: AnalysisConfig) -> None:
    """Verify schema_version 1 preset serializes and loads cleanly."""
    p_2d = AnalysisPreset("2D", sample_analysis_config)
    path = tmp_path / "preset_2d.json"
    save_preset(p_2d, path)

    loaded = load_preset(path)
    assert loaded.schema_version == 1
    assert loaded.preparation is None
    assert loaded.config_fingerprint == p_2d.config_fingerprint


def test_40_checkpoint_schema_2_save_load(tmp_path: Path, sample_analysis_config: AnalysisConfig) -> None:
    """Verify schema 2 checkpoint serialization and validation."""
    prep = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2})
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    df_status = pd.DataFrame([
        {"sample_id": "S1", "animal_id": "A1", "status": "completed", "error_type": None, "error_message": None}
    ])

    result = BatchResult(
        sample_status=df_status,
        completed_samples=(),
        review_sessions={},
        preset=preset,
        config_fingerprint=preset.config_fingerprint,
        manifest_fingerprint="m" * 64,
        preparation_fingerprint=preset.preparation_fingerprint,
        pipeline_fingerprint=preset.pipeline_fingerprint,
        review_mode="auto_accept",
    )

    chk_path = tmp_path / "checkpoint.json"
    save_checkpoint(result, chk_path)

    data = load_checkpoint(chk_path)
    assert data["schema_version"] == 2
    assert data["preparation_fingerprint"] == preset.preparation_fingerprint
    assert data["pipeline_fingerprint"] == preset.pipeline_fingerprint


def test_41_resume_failure_on_modified_preparation_fingerprint(
    tmp_path: Path,
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify resume_batch fails if preparation config fingerprint has changed."""
    prep1 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2}, z_selection=ZSelection(0, 5))
    preset1 = AnalysisPreset("Stack1", sample_analysis_config, preparation=prep1)

    spec = BatchSampleSpec("S1", "A1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    runner = nn.BatchRunner(preset1, manifest, BatchRunConfig(review_mode="auto_accept"))
    res = runner.run()

    chk_path = tmp_path / "chk.json"
    save_checkpoint(res, chk_path)

    # Attempt resume with different preparation Z range
    prep2 = StackPreparationConfig(nuclear_channel=0, marker_channels={"NeuN": 1, "GFAP": 2}, z_selection=ZSelection(0, 8))
    preset2 = AnalysisPreset("Stack2", sample_analysis_config, preparation=prep2)

    with pytest.raises(nn.batch.BatchResumeError, match="Preparation fingerprint mismatch on resume"):
        resume_batch(chk_path, preset2, manifest)


# ==============================================================================
# End-to-End Workflow, Provenance Survival, & Study Integration (Tests 42–49)
# ==============================================================================


def test_42_image_session_set_prepared_image_set(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_data: np.ndarray,
) -> None:
    """Verify ImageAnalysisSession binds PreparedImageSet cleanly."""
    prep_cfg = StackPreparationConfig(
        axes="CZYX",
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    prepared = prepare_stack(synthetic_stack_data, prep_cfg)
    session = ImageAnalysisSession(sample_analysis_config)
    session.set_prepared_image_set(prepared)
    assert session.nuclear_channel is not None
    assert session.marker_images is not None
    assert set(session.marker_images.keys()) == {"NeuN", "GFAP"}


def test_43_session_source_provenance_preserved_across_review_reset(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_data: np.ndarray,
) -> None:
    """Verify ImageAnalysisSession preserves source_provenance across start_review()."""
    prep_cfg = StackPreparationConfig(
        axes="CZYX",
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    prepared = prepare_stack(synthetic_stack_data, prep_cfg)
    prov = ImageSourceProvenance(
        source_kind="projected_stack",
        source_id="img001",
        pipeline_fingerprint="pipe123",
    )
    session = ImageAnalysisSession(sample_analysis_config)
    session.set_prepared_image_set(prepared, source_provenance=prov)
    assert session.source_provenance == prov

    session.segment()
    session.start_review()
    # Ensure source_provenance is not cleared when downstream results are reset
    assert session.source_provenance == prov


def test_44_end_to_end_stack_batch_workflow(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify end-to-end stack batch execution in auto_accept mode."""
    prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    spec = BatchSampleSpec("Sample1", "Animal1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    res = run_batch(preset, manifest, BatchRunConfig(review_mode="auto_accept"))
    assert len(res.completed_samples) == 1
    assert res.completed_samples[0].identity.sample_id == "Sample1"
    assert "Sample1" in res.sample_provenance
    prov = res.sample_provenance["Sample1"]
    assert prov.source_kind == "projected_stack"
    assert prov.nuclear_channel == 0
    assert prov.marker_channels == {"NeuN": 1, "GFAP": 2}


def test_45_batch_result_to_study_session_populates_provenance(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify BatchResult.to_study_session() passes ImageSourceProvenance into StudyAnalysisSession."""
    prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    spec = BatchSampleSpec("Sample1", "Animal1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    res = run_batch(preset, manifest, BatchRunConfig(review_mode="auto_accept"))
    study = res.to_study_session()

    assert "Sample1" in study.samples
    assert "Sample1" in study.sample_provenance
    assert study.sample_provenance["Sample1"].source_kind == "projected_stack"


def test_46_study_session_provenance_registry_intact_through_workflow(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify StudyAnalysisSession retains sample_provenance registry across aggregate and group analysis."""
    prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    spec = BatchSampleSpec("Sample1", "Animal1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    res = run_batch(preset, manifest, BatchRunConfig(review_mode="auto_accept"))
    study = res.to_study_session()

    # Perform study analysis
    study.aggregate()
    study.assign_groups([AnimalGroupAssignment("Animal1", "Control")])
    study.prepare_group_tables()
    study.summarize_groups()
    study.prepare_handoff()

    assert "Sample1" in study.sample_provenance
    assert study.sample_provenance["Sample1"].source_kind == "projected_stack"


def test_47_study_session_synchronizes_provenance_on_removal(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify StudyAnalysisSession removes provenance when remove_sample is called."""
    prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    spec = BatchSampleSpec("Sample1", "Animal1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    res = run_batch(preset, manifest, BatchRunConfig(review_mode="auto_accept"))
    study = res.to_study_session()

    assert "Sample1" in study.sample_provenance
    study.remove_sample("Sample1")
    assert "Sample1" not in study.samples
    assert "Sample1" not in study.sample_provenance


def test_48_no_mixing_batch_policy_rejection(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify mixed 2D + stack manifest or 2D preset with stack input is rejected early."""
    spec_stack = BatchSampleSpec("S_Stack", "A1", stack_input=StackFileInput(synthetic_stack_tiff))
    spec_2d = BatchSampleSpec(
        "S_2D",
        "A2",
        nuclear_input=ArrayInput(np.zeros((10, 10)), "nuc"),
        marker_inputs={"NeuN": ArrayInput(np.zeros((10, 10)), "m1"), "GFAP": ArrayInput(np.zeros((10, 10)), "m2")},
    )

    # 1. Mixed manifest
    manifest_mixed = BatchManifest([spec_stack, spec_2d])
    preset_2d = AnalysisPreset("2DPreset", sample_analysis_config)
    with pytest.raises(InvalidBatchManifestError, match="mixed 2D and Z-stack samples"):
        nn.BatchRunner(preset_2d, manifest_mixed)

    # 2. 2D preset with stack-only manifest
    manifest_stack = BatchManifest([spec_stack])
    with pytest.raises(InvalidBatchManifestError, match="Batch preset is 2D-only"):
        nn.BatchRunner(preset_2d, manifest_stack)


def test_49_source_mutation_blocks_manual_continuation(
    sample_analysis_config: AnalysisConfig,
    synthetic_stack_tiff: Path,
) -> None:
    """Verify modifying stack TIFF on disk during manual review blocks continue_sample."""
    prep = StackPreparationConfig(
        nuclear_channel=0,
        marker_channels={"NeuN": 1, "GFAP": 2},
    )
    preset = AnalysisPreset("StackPreset", sample_analysis_config, preparation=prep)

    spec = BatchSampleSpec("Sample1", "Animal1", stack_input=StackFileInput(synthetic_stack_tiff))
    manifest = BatchManifest([spec])

    runner = nn.BatchRunner(preset, manifest, BatchRunConfig(review_mode="manual"))
    res = runner.run()
    assert res.sample_status.loc[0, "status"] == "review_pending"

    rev_session = res.review_sessions["Sample1"]
    for obj_id in rev_session.result().objects["review_id"]:
        rev_session.accept(obj_id)
    reviewed = rev_session.finalize()

    # Mutate file on disk
    with open(synthetic_stack_tiff, "ab") as f:
        f.write(b"corrupted_bytes")

    with pytest.raises(InputSourceChangedError, match="Source file changed while sample"):
        runner.continue_sample("Sample1", reviewed)


# ==============================================================================
# Projection Artifact Demonstration (Test 50)
# ==============================================================================


def test_50_demonstrate_z_separated_projection_artifact() -> None:
    """Synthetic demonstration proving that 2D max projection conflates Z-separated structures in XY."""
    # Create 3D stack (Z=10, Y=20, X=20)
    stack_nuc = np.zeros((10, 20, 20), dtype=np.uint16)
    stack_marker = np.zeros((10, 20, 20), dtype=np.uint16)

    # Nucleus occupies Z=1..3 at XY=(10, 10)
    stack_nuc[1:4, 10, 10] = 500

    # Marker signal occupies Z=7..9 at the EXACT SAME XY=(10, 10), but is 4 Z-slices apart!
    stack_marker[7:10, 10, 10] = 800

    # In true 3D, distance along Z is > 0 and they do not occupy the same voxel:
    voxel_overlap_3d = np.logical_and(stack_nuc > 0, stack_marker > 0)
    assert not np.any(voxel_overlap_3d), "In 3D, nucleus and marker occupy completely disjoint Z planes."

    # In 2D maximum projection:
    proj_nuc = project_stack(stack_nuc, method="max").image
    proj_marker = project_stack(stack_marker, method="max").image

    pixel_overlap_2d = np.logical_and(proj_nuc > 0, proj_marker > 0)
    assert pixel_overlap_2d[10, 10] is np.True_, (
        "In 2D projection, the Z-separated marker appears inside the nuclear XY mask. "
        "This proves why M13 is projection-aware 2D preparation and M14 is required for true 3D analysis."
    )


def test_scientific_channel_mappings_are_defensively_immutable() -> None:
    source_mapping = {"MarkerA": 1}
    configs = (
        StackPreparationConfig(marker_channels=source_mapping),
        DirectImageSelectionConfig(axes="CYX", marker_channels=source_mapping),
        VolumeSelectionConfig(axes="CZYX", marker_channels=source_mapping),
    )
    source_mapping["MarkerA"] = 9

    for config in configs:
        assert config.marker_channels["MarkerA"] == 1
        with pytest.raises(TypeError):
            config.marker_channels["MarkerA"] = 7  # type: ignore[index]
