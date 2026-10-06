"""Tests for deterministic synthetic dataset generators and end-to-end production scoring."""

import numpy as np
import pytest

from nervenuc.classification.engine import classify_markers
from nervenuc.measurements.engine import measure_markers
from nervenuc.preparation.models import VoxelSpacing
from nervenuc.qc.session import start_review
from nervenuc.segmentation.core import segment_nuclei
from nervenuc.segmentation3d.engine import segment_nuclei_3d
from nervenuc.validation.dataset import get_synthetic_case, list_synthetic_cases
from nervenuc.validation.errors import InvalidGroundTruthError
from nervenuc.validation.matching import match_instances
from nervenuc.validation.metrics import (
    compute_phenotype_confusion,
    compute_segmentation_metrics,
    evaluate_marker_classification,
)


def test_synthetic_case_registry_listing() -> None:
    """Verify registry lists all 6 expected synthetic cases."""
    cases = list_synthetic_cases()
    expected = (
        "SYN_2D_ISOLATED",
        "SYN_2D_TOUCHING",
        "SYN_2D_EMPTY",
        "SYN_2D_MULTI_MARKER",
        "SYN_3D_ISOLATED",
        "SYN_3D_OVERLAP_XY",
    )
    assert cases == expected

    with pytest.raises(InvalidGroundTruthError, match="Unknown synthetic case_id"):
        get_synthetic_case("NON_EXISTENT_CASE")


@pytest.mark.parametrize(
    "case_id, expected_ndim, expected_axes, expected_gt_count",
    [
        ("SYN_2D_ISOLATED", 3, "CYX", 4),
        ("SYN_2D_TOUCHING", 3, "CYX", 2),
        ("SYN_2D_EMPTY", 3, "CYX", 0),
        ("SYN_2D_MULTI_MARKER", 3, "CYX", 4),
        ("SYN_3D_ISOLATED", 4, "CZYX", 3),
        ("SYN_3D_OVERLAP_XY", 4, "CZYX", 2),
    ],
)
def test_synthetic_case_structural_validity(
    case_id: str,
    expected_ndim: int,
    expected_axes: str,
    expected_gt_count: int,
) -> None:
    """Verify structural validity, axes, ndim, and ground truth count for all synthetic cases."""
    case = get_synthetic_case(case_id, seed=42)

    assert case.spec.case_id == case_id
    assert case.spec.axes == expected_axes
    assert case.image_data.ndim == expected_ndim
    assert case.image_data.dtype == np.uint16
    assert np.all(case.image_data >= 0)

    # Check ground truth label mask
    assert np.issubdtype(case.ground_truth_labels.dtype, np.integer)
    assert np.all(case.ground_truth_labels >= 0)
    assert len(case.ground_truth_objects) == expected_gt_count

    # Check unique positive labels in mask match object IDs
    gt_unique = sorted(int(x) for x in np.unique(case.ground_truth_labels) if x > 0)
    obj_ids = sorted(obj.truth_id for obj in case.ground_truth_objects)
    assert gt_unique == obj_ids


def test_synthetic_case_seed_reproducibility() -> None:
    """Verify identical random seed produces bitwise identical image pixels and ground-truth arrays."""
    case1 = get_synthetic_case("SYN_2D_MULTI_MARKER", seed=12345)
    case2 = get_synthetic_case("SYN_2D_MULTI_MARKER", seed=12345)

    assert np.array_equal(case1.image_data, case2.image_data)
    assert np.array_equal(case1.ground_truth_labels, case2.ground_truth_labels)
    assert len(case1.ground_truth_objects) == len(case2.ground_truth_objects)


def test_end_to_end_syn_2d_isolated() -> None:
    """Execute real production 2D segmentation on SYN_2D_ISOLATED and score with M17A/B."""
    case = get_synthetic_case("SYN_2D_ISOLATED", seed=42)
    dapi = case.image_data[0]  # First channel (DAPI)

    seg_res = segment_nuclei(dapi, case.config.segmentation)
    matching = match_instances(seg_res.labels, case.ground_truth_labels, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, seg_res.labels, case.ground_truth_labels)

    # All 4 isolated nuclei should be successfully detected
    assert len(seg_res.objects) == 4
    assert matching.tp_count == 4
    assert matching.fp_count == 0
    assert matching.fn_count == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.mean_matched_iou is not None and metrics.mean_matched_iou > 0.70


def test_end_to_end_syn_2d_empty() -> None:
    """Execute real production 2D segmentation on SYN_2D_EMPTY and verify zero detections."""
    case = get_synthetic_case("SYN_2D_EMPTY", seed=42)
    dapi = case.image_data[0]

    seg_res = segment_nuclei(dapi, case.config.segmentation)
    matching = match_instances(seg_res.labels, case.ground_truth_labels, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, seg_res.labels, case.ground_truth_labels)

    assert len(seg_res.objects) == 0
    assert matching.tp_count == 0
    assert matching.fp_count == 0
    assert matching.fn_count == 0
    assert metrics.is_empty_case_concordant is True
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1_score is None


def test_end_to_end_syn_2d_multi_marker() -> None:
    """Execute real production segmentation, measurement, and classification on SYN_2D_MULTI_MARKER."""
    case = get_synthetic_case("SYN_2D_MULTI_MARKER", seed=42)
    dapi = case.image_data[0]
    marker_a = case.image_data[1]
    marker_b = case.image_data[2]

    # 1. Segmentation
    seg_res = segment_nuclei(dapi, case.config.segmentation)
    matching = match_instances(seg_res.labels, case.ground_truth_labels, iou_threshold=0.5)
    assert matching.tp_count == 4

    # 2. Review session & measurement
    rs = start_review(seg_res)
    for r_id in list(rs._active_objects):
        rs.accept(r_id)
    reviewed = rs.result()
    marker_images = {"MarkerA": marker_a, "MarkerB": marker_b}
    meas_res = measure_markers(reviewed, marker_images)

    # 3. Marker classification
    thresholds = case.config.thresholds
    class_res = classify_markers(meas_res, thresholds)

    # Build predicted and truth marker states
    df_class = class_res.classifications
    df_a = df_class[df_class["marker_name"] == "MarkerA"]
    pred_states_a = {int(row.current_label): bool(row.is_positive) for row in df_a.itertuples()}

    df_b = df_class[df_class["marker_name"] == "MarkerB"]
    pred_states_b = {int(row.current_label): bool(row.is_positive) for row in df_b.itertuples()}

    df_phenos = class_res.phenotypes
    pred_phenos = {int(row.current_label): str(row.phenotype) for row in df_phenos.itertuples()}

    truth_states_a = {obj.truth_id: obj.marker_positivity["MarkerA"] for obj in case.ground_truth_objects}
    truth_states_b = {obj.truth_id: obj.marker_positivity["MarkerB"] for obj in case.ground_truth_objects}
    truth_phenos = {obj.truth_id: obj.phenotype for obj in case.ground_truth_objects}

    # 4. Score marker classifications
    metrics_a = evaluate_marker_classification(matching, pred_states_a, truth_states_a, marker_name="MarkerA")
    assert metrics_a.accuracy == 1.0
    assert metrics_a.sensitivity == 1.0
    assert metrics_a.specificity == 1.0

    metrics_b = evaluate_marker_classification(matching, pred_states_b, truth_states_b, marker_name="MarkerB")
    assert metrics_b.accuracy == 1.0
    assert metrics_b.sensitivity == 1.0
    assert metrics_b.specificity == 1.0

    # 5. Score phenotype confusion
    pheno_confusion = compute_phenotype_confusion(matching, pred_phenos, truth_phenos)
    assert pheno_confusion.overall_accuracy == 1.0
    assert pheno_confusion.macro_f1 == 1.0


def test_end_to_end_syn_3d_isolated() -> None:
    """Execute real production 3D volumetric segmentation on SYN_3D_ISOLATED and score."""
    case = get_synthetic_case("SYN_3D_ISOLATED", seed=42)
    vol = case.image_data[0]  # ZYX

    seg_3d_config = case.config.segmentation  # Segmentation3DConfig
    seg_res_3d = segment_nuclei_3d(vol, seg_3d_config)

    spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104)
    matching = match_instances(
        seg_res_3d.labels,
        case.ground_truth_labels,
        iou_threshold=0.4,
        spacing=spacing,
    )
    metrics = compute_segmentation_metrics(matching, seg_res_3d.labels, case.ground_truth_labels)

    assert seg_res_3d.nucleus_count == 3
    assert matching.tp_count == 3
    assert matching.fp_count == 0
    assert matching.fn_count == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.mean_centroid_error_um is not None and metrics.mean_centroid_error_um < 0.20
