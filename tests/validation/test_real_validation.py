"""Tests for real microscopy validation, manifest parsing, Fiji ROI ground truth, and animal pooling."""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import tifffile
import skimage.draw as draw
import roifile

from nervenuc.classification.models import MarkerThreshold
from nervenuc.fiji.roi import mask_to_roi, write_roi_set
from nervenuc.markers.models import MarkerDefinition
from nervenuc.segmentation.config import SegmentationConfig
from nervenuc.summary.models import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.validation.errors import InvalidGroundTruthError, ValidationError
from nervenuc.validation.real import (
    aggregate_real_animal_metrics,
    compute_file_sha256,
    export_real_validation_csvs,
    generate_validation_report_markdown,
    load_real_validation_case,
    load_real_validation_manifest,
    resolve_real_validation_data_path,
    run_live_validation_dataset,
    run_real_validation_case,
    validate_live_validation_manifest,
)
from nervenuc.workflow.config import AnalysisConfig


@pytest.fixture
def mock_real_dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a realistic mock real-microscopy validation dataset directory.

    Study design:
        - 3 animals: ANIMAL-01, ANIMAL-02, ANIMAL-03
        - 2 images per animal -> 6 total images
        - 3 channels: DAPI (ch0), MarkerA (ch1), MarkerB (ch2)
        - Manual Fiji ROI sets (.zip) with stable IDs GT-000001..GT-000004
        - Independent marker positivity tables (.csv)
    """
    dataset_dir = tmp_path / "mock_real_validation"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    img_dir = dataset_dir / "images"
    gt_dir = dataset_dir / "ground_truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    cases_spec_list = []
    h, w = 128, 128
    frozen_config = AnalysisConfig(
        segmentation=SegmentationConfig(threshold_method="otsu", min_area_px=20),
        markers=(MarkerDefinition("MarkerA"), MarkerDefinition("MarkerB")),
        thresholds=(
            MarkerThreshold("MarkerA", 1500.0, "mean_intensity"),
            MarkerThreshold("MarkerB", 1500.0, "mean_intensity"),
        ),
        metrics=(
            BiologicalMetricDefinition(
                name="MarkerA+",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"MarkerA": True}),
            ),
            BiologicalMetricDefinition(
                name="MarkerB+",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"MarkerB": True}),
            ),
        ),
    ).to_dict()

    animals = ["ANIMAL-01", "ANIMAL-02", "ANIMAL-03"]
    case_idx = 1

    for aid in animals:
        for fov in range(1, 3):
            cid = f"REAL_SCIATIC_{aid}_FOV{fov:02d}"
            img_rel = f"images/{cid}.tif"
            roi_rel = f"ground_truth/{cid}_rois.zip"
            marker_rel = f"ground_truth/{cid}_markers.csv"

            # 1. Synthesize 3-channel image
            dapi_arr = np.zeros((h, w), dtype=np.float32)
            marker_a_arr = np.zeros((h, w), dtype=np.float32)
            marker_b_arr = np.zeros((h, w), dtype=np.float32)

            centers = [(32, 32), (32, 96), (96, 32), (96, 96)]
            rois: list[roifile.ImagejRoi] = []
            marker_rows = []

            for n_idx, (cy, cx) in enumerate(centers):
                gt_id = f"GT-{n_idx + 1:06d}"
                # Create mask
                m = np.zeros((h, w), dtype=bool)
                rr, cc = draw.disk((cy, cx), 8, shape=(h, w))
                m[rr, cc] = True
                dapi_arr[rr, cc] = 3000.0

                # Assign marker positivity based on combinatorial phenotype
                is_a = n_idx in (0, 2)
                is_b = n_idx in (1, 2)
                if is_a:
                    marker_a_arr[rr, cc] = 2500.0
                if is_b:
                    marker_b_arr[rr, cc] = 2500.0

                phenotype = f"MarkerA{'+' if is_a else '-'} / MarkerB{'+' if is_b else '-'}"

                # Create ImageJ ROI
                roi = mask_to_roi(m, name=gt_id)
                rois.append(roi)

                marker_rows.append({
                    "truth_id": gt_id,
                    "DAPI": True,
                    "MarkerA": is_a,
                    "MarkerB": is_b,
                    "phenotype": phenotype,
                })

            # Save 3-channel TIFF: shape (3, 128, 128)
            stacked_img = np.stack([dapi_arr, marker_a_arr, marker_b_arr], axis=0)
            img_file = dataset_dir / img_rel
            tifffile.imwrite(str(img_file), stacked_img, photometric="minisblack")

            # Save ROI ZIP
            roi_file = dataset_dir / roi_rel
            write_roi_set(rois, roi_file)

            # Save marker table
            marker_file = dataset_dir / marker_rel
            pd.DataFrame(marker_rows).to_csv(marker_file, index=False)

            # Hashes
            img_sha = compute_file_sha256(img_file)
            roi_sha = compute_file_sha256(roi_file)
            marker_sha = compute_file_sha256(marker_file)

            cases_spec_list.append({
                "case_id": cid,
                "animal_id": aid,
                "description": f"Sciatic nerve image FOV {fov} for {aid}",
                "tier": "unbiased_validation",
                "dimensions": "2D",
                "image_relpath": img_rel,
                "ground_truth_rois_relpath": roi_rel,
                "ground_truth_markers_relpath": marker_rel,
                "image_sha256": img_sha,
                "ground_truth_rois_sha256": roi_sha,
                "ground_truth_markers_sha256": marker_sha,
                "axes": "CYX",
                "channel_names": ["DAPI", "MarkerA", "MarkerB"],
                "annotator_id": "HumanAnnotator_1",
                "annotation_tool": "Fiji_2.14",
                "annotation_date": "2026-08-16",
                "annotation_mode": "independent_manual",
                "config_dict": frozen_config,
                "selection_rationale": "Representative FOV",
            })
            case_idx += 1

    manifest_data = {
        "schema_version": "1.0",
        "dataset_name": "Sciatic_Nerve_Mock_Real_Validation_v1",
        "created_date": "2026-08-16",
        "cases": cases_spec_list,
    }

    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path, dataset_dir


def test_manifest_schema_validation(mock_real_dataset_fixture: tuple[Path, Path]) -> None:
    """Verify loading and schema validation of a real validation manifest."""
    manifest_path, _ = mock_real_dataset_fixture
    manifest = load_real_validation_manifest(manifest_path)

    assert manifest.schema_version == "1.0"
    assert manifest.dataset_name == "Sciatic_Nerve_Mock_Real_Validation_v1"
    assert len(manifest.cases) == 6
    assert manifest.cases[0].animal_id == "ANIMAL-01"
    assert manifest.cases[0].channel_names == ("DAPI", "MarkerA", "MarkerB")


def test_configured_missing_live_dataset_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-live-dataset"
    monkeypatch.setenv("NERVENUC_VALIDATION_DATA", str(missing))
    with pytest.raises(ValidationError, match="configured but is not an existing directory"):
        resolve_real_validation_data_path()


def test_real_case_loader_and_fiji_roi_rasterization(
    mock_real_dataset_fixture: tuple[Path, Path],
) -> None:
    """Verify loading a real validation case with SHA-256 checks and Fiji ROI rasterization."""
    manifest_path, dataset_dir = mock_real_dataset_fixture
    manifest = load_real_validation_manifest(manifest_path)
    spec = manifest.cases[0]

    case = load_real_validation_case(spec, dataset_dir)

    assert case.image_data.shape == (3, 128, 128)
    assert case.ground_truth_labels.shape == (128, 128)
    assert len(case.ground_truth_objects) == 4
    assert {obj.label_name for obj in case.ground_truth_objects} == {
        "GT-000001",
        "GT-000002",
        "GT-000003",
        "GT-000004",
    }


def test_real_case_loader_uses_production_yxc_channel_extraction(
    mock_real_dataset_fixture: tuple[Path, Path],
) -> None:
    import dataclasses

    manifest_path, dataset_dir = mock_real_dataset_fixture
    spec = load_real_validation_manifest(manifest_path).cases[0]
    original = tifffile.imread(dataset_dir / spec.image_relpath)
    yxc_path = dataset_dir / "images" / "yxc_case.tif"
    tifffile.imwrite(
        yxc_path,
        np.transpose(original, (1, 2, 0)),
        metadata={"axes": "YXC"},
        photometric="minisblack",
    )
    yxc_spec = dataclasses.replace(
        spec,
        image_relpath=str(yxc_path.relative_to(dataset_dir)),
        image_sha256=compute_file_sha256(yxc_path),
        axes="YXC",
    )

    case = load_real_validation_case(yxc_spec, dataset_dir)
    assert np.array_equal(case.image_data, original)


def test_stable_roi_id_mismatch_rejection(
    mock_real_dataset_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Verify that any mismatch between ROI set names and marker table IDs is strictly rejected."""
    _, dataset_dir = mock_real_dataset_fixture
    manifest = load_real_validation_manifest(dataset_dir / "manifest.json")
    spec = manifest.cases[0]

    # Create corrupted marker table with missing GT-000004
    corrupted_marker_path = dataset_dir / "ground_truth" / "corrupted_markers.csv"
    df = pd.DataFrame([
        {"truth_id": "GT-000001", "DAPI": True, "MarkerA": True, "MarkerB": False},
        {"truth_id": "GT-000002", "DAPI": True, "MarkerA": False, "MarkerB": True},
        {"truth_id": "GT-000003", "DAPI": True, "MarkerA": True, "MarkerB": True},
        # Missing GT-000004
    ])
    df.to_csv(corrupted_marker_path, index=False)

    import dataclasses
    corrupted_spec = dataclasses.replace(
        spec,
        ground_truth_markers_relpath=str(corrupted_marker_path.relative_to(dataset_dir)),
        ground_truth_markers_sha256="",
    )

    with pytest.raises(InvalidGroundTruthError, match="ROI / marker table ID mismatch"):
        load_real_validation_case(corrupted_spec, dataset_dir)


def test_end_to_end_real_case_execution_and_animal_aggregation(
    mock_real_dataset_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Verify full execution across 6 real cases, animal pooling, and CSV export."""
    manifest_path, dataset_dir = mock_real_dataset_fixture
    manifest = load_real_validation_manifest(manifest_path)

    results = []
    for spec in manifest.cases:
        case = load_real_validation_case(spec, dataset_dir)
        res = run_real_validation_case(case, spec)
        results.append(res)

    assert len(results) == 6
    for r in results:
        assert r.pred_object_count == 4
        assert r.truth_object_count == 4
        assert r.segmentation_metrics.f1_score == 1.0
        assert len(r.marker_metrics) == 2

    # Animal-level aggregation
    animal_df = aggregate_real_animal_metrics(results)
    assert len(animal_df) == 6  # 3 animals x 2 metrics (MarkerA+, MarkerB+)
    assert set(animal_df["animal_id"]) == {"ANIMAL-01", "ANIMAL-02", "ANIMAL-03"}
    assert set(animal_df["metric_name"]) == {"MarkerA+", "MarkerB+"}
    assert all(c == 2 for c in animal_df["sample_count"])

    # Verify pooled count correctness: each animal had 2 images x 4 nuclei = 8 nuclei
    assert all(den == 8 for den in animal_df["manual_denominator"])
    assert all(den == 8 for den in animal_df["nervenuc_denominator"])
    assert all(abs_diff == 0.0 for abs_diff in animal_df["absolute_difference_pp"])

    # Export CSVs
    csv_dict = export_real_validation_csvs(results, animal_df, tmp_path / "csv_out")
    assert csv_dict["case_metrics"].is_file()
    assert csv_dict["object_matches"].is_file()
    assert csv_dict["marker_metrics"].is_file()
    assert csv_dict["animal_metrics"].is_file()

    # Generate Markdown Report
    report_md = generate_validation_report_markdown(manifest, results, animal_df)
    assert "# NerveNuc Scientific Validation & Reproducibility Report" in report_md
    assert "ANIMAL-01" in report_md
    assert "ANIMAL-02" in report_md
    assert "ANIMAL-03" in report_md


def test_strict_regeneration_records_case_configuration_and_pipeline_identity(
    mock_real_dataset_fixture: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    manifest_path, dataset_dir = mock_real_dataset_fixture
    regenerated = run_live_validation_dataset(dataset_dir, tmp_path / "candidate")

    evidence = regenerated["evidence"]
    case_ids = [spec.case_id for spec in regenerated["manifest"].cases]
    assert evidence["case_order"] == case_ids
    assert list(evidence["case_config_fingerprints"]) == case_ids
    assert list(evidence["case_preparation_pipeline_fingerprints"]) == case_ids
    assert all(evidence["case_config_fingerprints"][case_id] for case_id in case_ids)
    assert all(
        evidence["case_preparation_pipeline_fingerprints"][case_id]["preparation_fingerprint"]
        for case_id in case_ids
    )
    assert all(
        evidence["case_preparation_pipeline_fingerprints"][case_id]["pipeline_fingerprint"]
        for case_id in case_ids
    )
    assert regenerated["paths"]["validation_evidence"].is_file()
    assert regenerated["paths"]["candidate_report"].is_file()
    # Static local reports are optional; the generated candidate report is verified above.


def test_strict_live_manifest_rejects_missing_hash_or_implicit_config(
    mock_real_dataset_fixture: tuple[Path, Path],
) -> None:
    import dataclasses

    manifest_path, dataset_dir = mock_real_dataset_fixture
    manifest = load_real_validation_manifest(manifest_path)
    missing_hash = dataclasses.replace(manifest.cases[0], image_sha256="")
    with pytest.raises(ValidationError, match="explicit SHA-256"):
        validate_live_validation_manifest(
            dataclasses.replace(manifest, cases=(missing_hash, *manifest.cases[1:])),
            dataset_dir,
        )

    implicit_config = dataclasses.replace(
        manifest.cases[0],
        preset_relpath=None,
        config_dict=None,
    )
    with pytest.raises(ValidationError, match="implicit fallback configuration"):
        validate_live_validation_manifest(
            dataclasses.replace(manifest, cases=(implicit_config, *manifest.cases[1:])),
            dataset_dir,
        )

    unsupported_axes = dataclasses.replace(manifest.cases[0], axes="ZYX")
    with pytest.raises(ValidationError, match="axes must be one of"):
        validate_live_validation_manifest(
            dataclasses.replace(manifest, cases=(unsupported_axes, *manifest.cases[1:])),
            dataset_dir,
        )


@pytest.mark.validation_data
def test_live_real_microscopy_dataset() -> None:
    """Execute validation against live external real microscopy dataset if configured via NERVENUC_VALIDATION_DATA."""
    val_dir = resolve_real_validation_data_path()
    if val_dir is None:
        pytest.skip("Real validation dataset not configured via NERVENUC_VALIDATION_DATA.")

    manifest_path = val_dir / "manifest.json"
    assert manifest_path.is_file(), f"Configured validation dataset is missing manifest: {manifest_path}"

    regenerated = run_live_validation_dataset(val_dir)
    assert regenerated["paths"]["validation_evidence"].is_file()
    assert regenerated["paths"]["candidate_report"].is_file()
