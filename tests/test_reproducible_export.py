"""Tests for M18 Minimal Reproducible Export & Methods Summary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nervenuc.aggregation import (
    SampleIdentity,
    SampleSummary,
)
from nervenuc.classification import MarkerThreshold
from nervenuc.export import (
    AnimalGroupAssignment,
    ExportFileExistsError,
    IncompleteWorkflowExportError,
    StaleExportError,
    extract_object_level_table,
    extract_per_animal_table,
    extract_per_image_table,
    generate_methods_summary,
    prepare_reproducible_bundle,
    QCSummaryRecord,
    write_reproducible_bundle,
)
from nervenuc.markers import MarkerDefinition
from nervenuc.segmentation import SegmentationConfig
from nervenuc.summary import (
    BiologicalMetricDefinition,
    PhenotypeFilter,
)
from nervenuc.workflow import (
    AnalysisConfig,
    ImageAnalysisSession,
    ImageSourceProvenance,
    StudyAnalysisSession,
)


@pytest.fixture
def dummy_analysis_config() -> AnalysisConfig:
    """Create a standard AnalysisConfig for testing."""
    seg = SegmentationConfig(
        threshold_method="otsu",
        gaussian_sigma=1.0,
        min_area_px=20,
        max_area_px=500,
        min_distance_px=5,
    )
    markers = (
        MarkerDefinition(name="SOX10"),
        MarkerDefinition(name="BrdU"),
    )
    thresholds = (
        MarkerThreshold(marker_name="SOX10", threshold=100.0, measurement="mean_intensity", comparison=">="),
        MarkerThreshold(marker_name="BrdU", threshold=150.0, measurement="mean_intensity", comparison=">="),
    )
    metrics = (
        BiologicalMetricDefinition(
            name="SOX10+_pct",
            denominator=PhenotypeFilter(),
            numerator=PhenotypeFilter({"SOX10": True}),
        ),
        BiologicalMetricDefinition(
            name="Dividing_Schwann_pct",
            denominator=PhenotypeFilter({"SOX10": True}),
            numerator=PhenotypeFilter({"SOX10": True, "BrdU": True}),
        ),
        BiologicalMetricDefinition(
            name="Empty_metric_pct",
            denominator=PhenotypeFilter({"SOX10": False, "BrdU": True}),
            numerator=PhenotypeFilter({"SOX10": False, "BrdU": True}),
        ),
    )
    return AnalysisConfig(
        segmentation=seg,
        markers=markers,
        thresholds=thresholds,
        metrics=metrics,
    )


@pytest.fixture
def completed_2d_image_session(dummy_analysis_config: AnalysisConfig) -> ImageAnalysisSession:
    """Create a fully analyzed 2D ImageAnalysisSession."""
    sess = ImageAnalysisSession(dummy_analysis_config)

    # 128x128 image with 3 distinct nuclei
    dapi = np.zeros((128, 128), dtype=np.float32)
    dapi[20:40, 20:40] = 200.0
    dapi[20:40, 70:90] = 220.0
    dapi[70:90, 45:65] = 250.0

    sox10 = np.zeros((128, 128), dtype=np.float32)
    sox10[20:40, 20:40] = 180.0  # SOX10+
    sox10[20:40, 70:90] = 190.0  # SOX10+
    sox10[70:90, 45:65] = 20.0   # SOX10-

    brdu = np.zeros((128, 128), dtype=np.float32)
    brdu[20:40, 20:40] = 200.0  # BrdU+
    brdu[20:40, 70:90] = 10.0   # BrdU-
    brdu[70:90, 45:65] = 10.0   # BrdU-

    prov = ImageSourceProvenance(
        source_kind="direct_2d",
        source_id="SAMPLE-001",
        source_hash="sha256_mock_12345",
        pipeline_fingerprint="pipeline_fp_12345",
    )

    sess.set_image(dapi, {"SOX10": sox10, "BrdU": brdu}, source_provenance=prov)
    sess.segment()
    sess.auto_accept_segmentation()
    sess.measure()
    sess.classify()
    sess.summarize()

    return sess


def test_prepare_and_write_reproducible_bundle_single_image(
    completed_2d_image_session: ImageAnalysisSession,
    tmp_path: Path,
) -> None:
    """Verify reproducible bundle generation from a single image session."""
    bundle = prepare_reproducible_bundle(completed_2d_image_session, include_objects=True)

    assert bundle.per_image_table is not None
    assert len(bundle.per_image_table) == 1
    row = bundle.per_image_table.iloc[0]
    assert row["sample_id"] == "SAMPLE-001"
    assert row["valid_nucleus_count"] == 3
    assert row["SOX10_positive_count"] == 2
    assert row["BrdU_positive_count"] == 1
    assert row["analysis_mode"] == "direct_2d"

    # Verify zero-denominator representation: Empty_metric_pct has 0 denominator
    assert row["metric_Empty_metric_pct_numerator"] == 0
    assert row["metric_Empty_metric_pct_denominator"] == 0
    assert row["metric_Empty_metric_pct_percentage"] is None or np.isnan(row["metric_Empty_metric_pct_percentage"])
    assert bool(row["metric_Empty_metric_pct_percentage_defined"]) is False

    # Object-level table
    assert bundle.object_table is not None
    assert len(bundle.object_table) == 3
    assert "SOX10_mean_intensity" in bundle.object_table.columns
    assert "SOX10_positive" in bundle.object_table.columns
    assert "phenotype" in bundle.object_table.columns

    # Methods text contains factual info
    assert "NerveNuc version" in bundle.methods_summary_text
    assert "SOX10" in bundle.methods_summary_text
    assert "mean_intensity >= 100.0" in bundle.methods_summary_text

    # Write bundle
    out_dir = tmp_path / "export_bundle_01"
    written_map = write_reproducible_bundle(bundle, out_dir, write_csv=True, write_excel=True)

    assert (out_dir / "per_image_results.csv").is_file()
    assert (out_dir / "object_results.csv").is_file()
    assert (out_dir / "analysis_config.json").is_file()
    assert (out_dir / "provenance.json").is_file()
    assert (out_dir / "qc_summary.json").is_file()
    assert (out_dir / "reproducibility_env.json").is_file()
    assert (out_dir / "methods_summary.txt").is_file()
    assert (out_dir / "results.xlsx").is_file()


def test_object_level_export_default_is_false(
    completed_2d_image_session: ImageAnalysisSession,
) -> None:
    """Verify that include_objects defaults to False."""
    bundle = prepare_reproducible_bundle(completed_2d_image_session)
    assert bundle.object_table is None
    assert bundle.manifest["has_object_table"] is False


def test_study_session_animal_pooling_invariants(
    dummy_analysis_config: AnalysisConfig,
    tmp_path: Path,
) -> None:
    """Verify that multi-sample study session pools raw numerators and denominators without averaging percentages."""
    study = StudyAnalysisSession()

    # Create 2 samples for ANIMAL-01 and 1 sample for ANIMAL-02
    # Sample 1: 10 nuclei, 8 SOX10+ (80%)
    # Sample 2: 100 nuclei, 20 SOX10+ (20%)
    # If averaged: (80% + 20%) / 2 = 50.0% (WRONG)
    # If pooled correctly: (8 + 20) / (10 + 100) = 28 / 110 = 25.4545% (CORRECT)
    from nervenuc.summary.models import BiologicalMetricDefinition, BiologicalSummaryResult, PhenotypeFilter

    defs = (
        BiologicalMetricDefinition(
            name="SOX10+_pct",
            denominator=PhenotypeFilter(),
            numerator=PhenotypeFilter({"SOX10": True}),
        ),
    )
    dummy_class1 = type("MockClass1", (), {
        "phenotypes": pd.DataFrame([{"current_label": i + 1, "phenotype": "SOX10+"} for i in range(10)]),
        "classifications": pd.DataFrame([]),
    })()
    m1_df = pd.DataFrame([
        {
            "metric_name": "SOX10+_pct",
            "numerator_count": 8,
            "denominator_count": 10,
            "percentage": 80.0,
            "numerator_filter": "SOX10+",
            "denominator_filter": "ALL",
        }
    ])
    s1_summary = BiologicalSummaryResult(
        metrics=m1_df,
        definitions=defs,
        classification_result=dummy_class1,
    )

    dummy_class2 = type("MockClass2", (), {
        "phenotypes": pd.DataFrame([{"current_label": i + 1, "phenotype": "SOX10+"} for i in range(100)]),
        "classifications": pd.DataFrame([]),
    })()
    m2_df = pd.DataFrame([
        {
            "metric_name": "SOX10+_pct",
            "numerator_count": 20,
            "denominator_count": 100,
            "percentage": 20.0,
            "numerator_filter": "SOX10+",
            "denominator_filter": "ALL",
        }
    ])
    s2_summary = BiologicalSummaryResult(
        metrics=m2_df,
        definitions=defs,
        classification_result=dummy_class2,
    )

    prov1 = ImageSourceProvenance(source_kind="direct_2d", source_id="FOVI_01")
    prov2 = ImageSourceProvenance(source_kind="direct_2d", source_id="FOVI_02")

    study_cfg = replace(dummy_analysis_config, metrics=defs)
    qc1 = QCSummaryRecord("IMG-01", 10, 10, 0, 0, 0, 10, "completed")
    qc2 = QCSummaryRecord("IMG-02", 100, 100, 0, 0, 0, 100, "completed")
    study.add_sample(SampleIdentity("IMG-01", "ANIMAL-01"), s1_summary, prov1, study_cfg, qc1)
    study.add_sample(SampleIdentity("IMG-02", "ANIMAL-01"), s2_summary, prov2, study_cfg, qc2)

    group_assigns = [
        AnimalGroupAssignment(animal_id="ANIMAL-01", group="Treated"),
    ]

    bundle = prepare_reproducible_bundle(study, group_assignments=group_assigns)

    assert bundle.per_animal_table is not None
    animal_row = bundle.per_animal_table.iloc[0]
    assert animal_row["animal_id"] == "ANIMAL-01"
    assert animal_row["sample_count"] == 2
    assert animal_row["numerator_count"] == 28
    assert animal_row["denominator_count"] == 110

    expected_pct = 100.0 * 28 / 110  # ~25.4545%
    assert pytest.approx(animal_row["percentage"], rel=1e-4) == expected_pct
    # Crucial test: must NOT equal 50.0%
    assert abs(animal_row["percentage"] - 50.0) > 10.0


def test_study_bundle_preserves_per_sample_projection_provenance(
    completed_2d_image_session: ImageAnalysisSession,
) -> None:
    summary = completed_2d_image_session.summary_result
    assert summary is not None
    study = StudyAnalysisSession()
    prov = ImageSourceProvenance(
        source_kind="projected_stack",
        source_id="STACK-01",
        source_hash="a" * 64,
        z_start=2,
        z_stop=8,
        projection_method="max",
        pipeline_fingerprint="pipeline-projected",
    )
    study.add_sample(
        SampleIdentity("IMG-P", "ANIMAL-P"),
        summary,
        prov,
        completed_2d_image_session.config,
        QCSummaryRecord("IMG-P", 3, 3, 0, 0, 0, 3, "completed"),
    )
    bundle = prepare_reproducible_bundle(study)
    row = bundle.per_image_table.iloc[0]
    assert row["source_id"] == "STACK-01"
    assert row["analysis_mode"] == "projected_2d"
    assert row["pipeline_fingerprint"] == "pipeline-projected"


def test_study_bundle_rejects_missing_reproducibility_metadata(
    completed_2d_image_session: ImageAnalysisSession,
) -> None:
    study = StudyAnalysisSession()
    study.add_sample(
        SampleIdentity("IMG-LEGACY", "ANIMAL-L"),
        completed_2d_image_session.summary_result,
    )
    with pytest.raises(IncompleteWorkflowExportError, match="source provenance missing"):
        prepare_reproducible_bundle(study)


def test_zero_denominator_biological_percentage_not_nan_or_infinity(
    dummy_analysis_config: AnalysisConfig,
    tmp_path: Path,
) -> None:
    """Verify that zero-denominator percentages serialize cleanly without NaN or Infinity."""
    from nervenuc.summary.models import BiologicalMetricDefinition, BiologicalSummaryResult, PhenotypeFilter

    defs_z = (
        BiologicalMetricDefinition(
            name="Zero_Denom_pct",
            denominator=PhenotypeFilter({"SOX10": False}),
            numerator=PhenotypeFilter({"SOX10": False, "BrdU": True}),
        ),
    )
    dummy_class_z = type("MockClassZ", (), {
        "phenotypes": pd.DataFrame([]),
        "classifications": pd.DataFrame([]),
    })()
    m_df = pd.DataFrame([
        {
            "metric_name": "Zero_Denom_pct",
            "numerator_count": 0,
            "denominator_count": 0,
            "percentage": float("nan"),
            "numerator_filter": defs_z[0].effective_numerator.to_string(),
            "denominator_filter": defs_z[0].denominator.to_string(),
        }
    ])
    summary = BiologicalSummaryResult(
        metrics=m_df,
        definitions=defs_z,
        classification_result=dummy_class_z,
    )

    study = StudyAnalysisSession()
    study.add_sample(
        SampleIdentity("IMG-Z", "ANIMAL-Z"),
        summary,
        ImageSourceProvenance(source_kind="direct_2d", source_id="IMG-Z"),
        replace(dummy_analysis_config, metrics=defs_z),
        QCSummaryRecord("IMG-Z", 0, 0, 0, 0, 0, 0, "completed"),
    )

    bundle = prepare_reproducible_bundle(study)
    assert bundle.per_animal_table is not None
    row = bundle.per_animal_table.iloc[0]
    assert row["numerator_count"] == 0
    assert row["denominator_count"] == 0
    assert row["percentage"] is None
    assert bool(row["percentage_defined"]) is False

    # Export to CSV
    out_dir = tmp_path / "zero_denom_bundle"
    write_reproducible_bundle(bundle, out_dir, write_csv=True, write_excel=True)

    csv_text = (out_dir / "per_animal_results.csv").read_text(encoding="utf-8")
    assert "NaN" not in csv_text
    assert "Infinity" not in csv_text


def test_stale_and_incomplete_analysis_rejection(dummy_analysis_config: AnalysisConfig) -> None:
    """Verify that attempting to export an incomplete or stale session raises clear errors."""
    sess = ImageAnalysisSession(dummy_analysis_config)

    # 1. Image not loaded
    with pytest.raises(IncompleteWorkflowExportError, match="image has not been loaded"):
        prepare_reproducible_bundle(sess)

    dapi = np.zeros((64, 64), dtype=np.float32)
    sess.set_image(dapi, {"SOX10": dapi, "BrdU": dapi})

    # 2. Segmentation not run
    with pytest.raises(IncompleteWorkflowExportError, match="segmentation has not been run"):
        prepare_reproducible_bundle(sess)

    sess.segment()

    # 3. Review not complete
    with pytest.raises(IncompleteWorkflowExportError, match="quality review has not been finalized"):
        prepare_reproducible_bundle(sess)


def test_file_overwrite_protection_and_atomic_staging(
    completed_2d_image_session: ImageAnalysisSession,
    tmp_path: Path,
) -> None:
    """Verify that export strictly respects overwrite=False and avoids partial writes."""
    bundle = prepare_reproducible_bundle(completed_2d_image_session)
    out_dir = tmp_path / "export_overwrite_test"

    # First write should succeed
    write_reproducible_bundle(bundle, out_dir, overwrite=False)

    # Second write without overwrite=True must raise ExportFileExistsError
    with pytest.raises(ExportFileExistsError, match="already exists and overwrite is False"):
        write_reproducible_bundle(bundle, out_dir, overwrite=False)

    # With overwrite=True, it succeeds
    write_reproducible_bundle(bundle, out_dir, overwrite=True)


def test_methods_summary_conditionally_reflects_actual_modality() -> None:
    """Verify methods summary accurately distinguishes 2D direct, projected 2D, and true 3D."""
    # 1. 2D direct
    text_2d = generate_methods_summary(
        config=None,
        modality="direct_2d",
        sample_count=4,
    )
    assert "single-plane 2D fluorescence micrographs" in text_2d

    # 2. Projected 2D
    prov = ImageSourceProvenance(
        source_kind="projected_stack",
        projection_method="max",
        z_start=2,
        z_stop=15,
        physical_size_z_um=0.5,
    )
    text_proj = generate_methods_summary(
        config=None,
        provenance_list=[prov],
        modality="projected_2d",
        sample_count=1,
    )
    assert "max intensity projection across zero-based slices [2, 15)" in text_proj
    assert "Z-step: 0.500 µm" in text_proj

    # 3. True 3D
    from nervenuc.preparation import VoxelSpacing
    from nervenuc.workflow.provenance import VolumeSourceProvenance
    prov_3d = VolumeSourceProvenance(
        source_id="VOL-01",
        spacing=VoxelSpacing(z=0.5, y=0.104, x=0.104),
    )
    text_3d = generate_methods_summary(
        config=None,
        provenance_list=[prov_3d],
        modality="volume_3d",
        sample_count=2,
    )
    assert "True 3D volumetric analysis" in text_3d
    assert "0.500 × 0.104 × 0.104 µm" in text_3d

    uncalibrated_3d = generate_methods_summary(
        config=None,
        provenance_list=[VolumeSourceProvenance(source_id="VOL-UNCAL")],
        modality="volume_3d",
    )
    assert "without complete physical voxel calibration" in uncalibrated_3d


def test_3d_volume_bundle_export(tmp_path: Path) -> None:
    """Verify reproducible bundle export from a 3D VolumeAnalysisSession."""
    from nervenuc.markers import MarkerDefinition
    from nervenuc.measurements3d import MarkerMeasurement3DResult
    from nervenuc.preparation import VolumeSelectionConfig, VoxelSpacing
    from nervenuc.qc import ReviewedSegmentationResult
    from nervenuc.segmentation3d import Segmentation3DConfig, Segmentation3DResult
    from nervenuc.summary import BiologicalMetricDefinition, BiologicalSummary3DResult, PhenotypeFilter
    from nervenuc.workflow import VolumeAnalysisConfig, VolumeAnalysisSession
    from nervenuc.workflow.provenance import VolumeSourceProvenance

    cfg_3d = VolumeAnalysisConfig(
        selection=VolumeSelectionConfig(
            axes=None,
            time_index=None,
            nuclear_channel=0,
            marker_channels={"SOX10": 1},
            z_selection=None,
        ),
        segmentation=Segmentation3DConfig(
            threshold_method="otsu",
            gaussian_sigma_xy_px=1.0,
            gaussian_sigma_z_slices=0.5,
            min_volume_voxels=50,
            max_volume_voxels=5000,
            exclude_xy_border=True,
            exclude_z_border=True,
        ),
        markers=(MarkerDefinition(name="SOX10"),),
        thresholds=(MarkerThreshold(marker_name="SOX10", threshold=100.0),),
        metrics=(
            BiologicalMetricDefinition(
                name="SOX10_3D_pct",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"SOX10": True}),
            ),
        ),
    )

    sess_3d = VolumeAnalysisSession()

    # 3D labels with 1 nucleus
    labels_3d = np.zeros((8, 32, 32), dtype=np.int32)
    labels_3d[2:6, 10:20, 10:20] = 1

    objects_3d = pd.DataFrame([
        {
            "object_id": 1,
            "current_label": 1,
            "volume_voxels": 400,
            "centroid_z": 4.0,
            "centroid_y": 15.0,
            "centroid_x": 15.0,
            "touches_border": False,
            "qc_status": "accepted",
            "review_id": "REV-000001",
        }
    ])

    seg_3d_res = Segmentation3DResult(
        labels=labels_3d,
        objects=objects_3d,
        config=cfg_3d.segmentation,
        source_shape=(8, 32, 32),
        source_id="VOL-EXP-01",
        spacing=VoxelSpacing(z=0.5, y=0.1, x=0.1),
    )
    reviewed_3d = ReviewedSegmentationResult(
        source_result=seg_3d_res,
        labels=labels_3d,
        objects=objects_3d,
        history=(),
    )

    meas_df = pd.DataFrame([
        {
            "object_id": 1,
            "current_label": 1,
            "marker_name": "SOX10",
            "mean_intensity": 250.0,
            "median_intensity": 250.0,
            "max_intensity": 300.0,
            "min_intensity": 200.0,
            "integrated_intensity": 100000.0,
        }
    ])
    mock_prov = type("MockVolProv", (), {})()
    meas_3d_res = MarkerMeasurement3DResult(
        measurements=meas_df,
        reviewed_result=reviewed_3d,
        marker_names=("SOX10",),
        source_id="VOL-EXP-01",
        volume_provenance=mock_prov,
    )

    from nervenuc.classification import Classification3DResult
    class_df = pd.DataFrame([
        {
            "object_id": 1,
            "current_label": 1,
            "marker_name": "SOX10",
            "is_positive": True,
            "measurement_value": 250.0,
            "threshold_value": 100.0,
        }
    ])
    pheno_df = pd.DataFrame([
        {
            "object_id": 1,
            "current_label": 1,
            "phenotype": "SOX10+",
        }
    ])
    class_3d_res = Classification3DResult(
        classifications=class_df,
        phenotypes=pheno_df,
        thresholds=cfg_3d.thresholds,
        measurement_result=meas_3d_res,
    )

    metrics_df = pd.DataFrame([
        {
            "metric_name": "SOX10_3D_pct",
            "numerator_count": 1,
            "denominator_count": 1,
            "percentage": 100.0,
            "numerator_filter": "SOX10+",
            "denominator_filter": "ALL",
        }
    ])
    summary_3d_res = BiologicalSummary3DResult(
        metrics=metrics_df,
        definitions=cfg_3d.metrics,
        classification_result=class_3d_res,
    )

    prov_3d = VolumeSourceProvenance(
        source_id="VOL-EXP-01",
        spacing=VoxelSpacing(z=0.5, y=0.1, x=0.1),
        pipeline_fingerprint="fp_vol_3d_test",
    )

    sess_3d._provenance = prov_3d
    sess_3d._reviewed_result = reviewed_3d
    sess_3d._measurement_result = meas_3d_res
    sess_3d._classification_result = class_3d_res
    sess_3d._summary_result = summary_3d_res
    sess_3d._analysis_config = cfg_3d

    bundle = prepare_reproducible_bundle(sess_3d, include_objects=True)

    assert bundle.manifest["modality"] == "volume_3d"
    assert bundle.object_table is not None
    assert "centroid_z_vox" in bundle.object_table.columns
    assert bundle.object_table.iloc[0]["centroid_z_vox"] == 4.0

    out_dir = tmp_path / "volume_bundle_out"
    write_reproducible_bundle(bundle, out_dir, write_csv=True, write_excel=True)

    assert (out_dir / "methods_summary.txt").is_file()
    assert (out_dir / "results.xlsx").is_file()
    methods_text = (out_dir / "methods_summary.txt").read_text(encoding="utf-8")
    assert "True 3D volumetric analysis" in methods_text
    assert "σ_xy = 1.0 px, σ_z = 0.5 slices" in methods_text
    assert "minimum seed distance: XY" in methods_text
    assert "Z =" in methods_text
    assert "Enclosed foreground holes were filled" in methods_text
    assert "volume boundaries" in methods_text


def test_excel_workbook_sheets_and_formatting(
    completed_2d_image_session: ImageAnalysisSession,
    tmp_path: Path,
) -> None:
    """Verify that results.xlsx contains all expected sheets from the bundle."""
    bundle = prepare_reproducible_bundle(completed_2d_image_session, include_objects=True)
    out_dir = tmp_path / "excel_sheets_test"
    write_reproducible_bundle(bundle, out_dir, write_csv=True, write_excel=True)

    xlsx_path = out_dir / "results.xlsx"
    assert xlsx_path.is_file()

    with pd.ExcelFile(xlsx_path) as xls:
        sheet_names = xls.sheet_names

    assert "Per Image" in sheet_names
    assert "Objects" in sheet_names
    assert "QC Summary" in sheet_names
    assert "Provenance" in sheet_names
    assert "Configuration" in sheet_names
    assert "Methods Summary" in sheet_names

    df_img = pd.read_excel(xlsx_path, sheet_name="Per Image")
    assert len(df_img) == 1
    assert df_img.iloc[0]["sample_id"] == "SAMPLE-001"


def test_excel_writer_fails_loudly_on_invalid_metadata_json(
    completed_2d_image_session: ImageAnalysisSession,
    tmp_path: Path,
) -> None:
    from nervenuc.export import ExportWriteError, write_reproducible_excel_workbook

    bundle = prepare_reproducible_bundle(completed_2d_image_session)
    bundle.qc_summary_json = "{not-json"
    target = tmp_path / "invalid.xlsx"
    with pytest.raises(ExportWriteError, match="invalid metadata JSON"):
        write_reproducible_excel_workbook(bundle, target)
    assert not target.exists()


def test_export_controller_gui_integration(
    completed_2d_image_session: ImageAnalysisSession,
    tmp_path: Path,
) -> None:
    """Verify ExportController coordinates reproducible bundle export from GUI state."""
    from nervenuc.gui.controllers.export_controller import ExportController
    from nervenuc.gui.state import GuiSessionState

    state = GuiSessionState()
    state.image_session = completed_2d_image_session

    ctrl = ExportController(state)
    out_dir = tmp_path / "gui_bundle_out"

    files_map = ctrl.export_reproducible_bundle(out_dir, include_objects=True)

    assert "methods_summary_txt" in files_map
    assert "per_image_csv" in files_map
    assert "results_excel" in files_map
    assert files_map["methods_summary_txt"].is_file()
    assert files_map["results_excel"].is_file()
