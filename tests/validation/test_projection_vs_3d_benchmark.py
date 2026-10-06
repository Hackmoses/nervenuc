"""Critical validation benchmark comparing projected 2D segmentation with true 3D volumetric analysis."""

import numpy as np
import pytest

from nervenuc.preparation import StackPreparationConfig, VoxelSpacing, prepare_stack, prepare_volume
from nervenuc.segmentation.config import SegmentationConfig
from nervenuc.segmentation.core import segment_nuclei
from nervenuc.segmentation3d.engine import segment_nuclei_3d
from nervenuc.validation.dataset import get_synthetic_case
from nervenuc.validation.matching import match_instances
from nervenuc.validation.metrics import compute_segmentation_metrics
from nervenuc.validation.models import ProjectionVs3DBenchmarkResult


def test_projection_vs_3d_false_overlap_benchmark() -> None:
    """Benchmark quantifying the empirical error introduced by 2D projection on Z-separated nuclei."""
    case = get_synthetic_case("SYN_3D_OVERLAP_XY", seed=42)
    prepared_3d = prepare_volume(case.image_data, case.config.selection)
    vol_3d = prepared_3d.nuclear_volume
    gt_labels_3d = case.ground_truth_labels  # ZYX array with 2 nuclei (label 1 at Z=6, label 2 at Z=18)

    truth_3d_count = len(case.ground_truth_objects)
    assert truth_3d_count == 2

    # =========================================================================
    # Branch 1: True 3D Volumetric Analysis
    # =========================================================================
    seg_3d_config = case.config.segmentation  # Segmentation3DConfig
    seg_3d_res = segment_nuclei_3d(prepared_3d, seg_3d_config)
    spacing = VoxelSpacing(z=0.5, y=0.104, x=0.104)
    matching_3d = match_instances(
        seg_3d_res.labels,
        gt_labels_3d,
        iou_threshold=0.4,
        spacing=spacing,
    )
    metrics_3d = compute_segmentation_metrics(matching_3d, seg_3d_res.labels, gt_labels_3d)

    # 3D analysis should successfully resolve both nuclei in 3D space
    assert seg_3d_res.nucleus_count == 2
    assert matching_3d.tp_count == 2
    assert matching_3d.fp_count == 0
    assert matching_3d.fn_count == 0
    assert metrics_3d.precision == 1.0
    assert metrics_3d.recall == 1.0
    assert metrics_3d.f1_score == 1.0

    # =========================================================================
    # Branch 2: 2D Projected Analysis (Maximum Intensity Projection)
    # =========================================================================
    prepared_2d = prepare_stack(
        case.image_data,
        StackPreparationConfig(
            axes="CZYX",
            nuclear_channel=0,
            marker_channels={},
            projection_method="max",
        ),
    )
    dapi_2d_proj = prepared_2d.nuclear_image

    # 2D classical segmentation
    seg_2d_config = SegmentationConfig(threshold_method="otsu", min_area_px=20, min_distance_px=5)
    seg_2d_res = segment_nuclei(dapi_2d_proj, seg_2d_config)

    # 2D projection collapses the two overlapping nuclei into a single merged detection
    assert len(seg_2d_res.objects) == 1

    # In 2D projected truth, we evaluate against the projected biological reality:
    # 2 distinct 3D cells existed in this tissue column.
    # The 2D projected detector produced 1 merged object, matching 1 of the 2 true objects and missing 1.
    gt_2d_proj = np.max(gt_labels_3d, axis=0)  # Merged 2D footprint

    # 2D instance matching on the 2D projected image
    matching_2d = match_instances(seg_2d_res.labels, gt_2d_proj, iou_threshold=0.4)
    # Matching against the single merged 2D footprint yields 1 match relative to the collapsed 2D footprint
    assert matching_2d.tp_count == 1

    # Relative to the true biological 3D count (2 nuclei):
    # 2D projected analysis found 1 object out of 2 true objects (Recall = 1/2 = 50%, FN = 1)
    proj_2d_tp = 1
    proj_2d_fp = 0
    proj_2d_fn = truth_3d_count - proj_2d_tp  # 2 - 1 = 1 missed nucleus
    proj_2d_precision = float(proj_2d_tp / (proj_2d_tp + proj_2d_fp))
    proj_2d_recall = float(proj_2d_tp / (proj_2d_tp + proj_2d_fn))
    proj_2d_f1 = float(2.0 * proj_2d_tp / (2 * proj_2d_tp + proj_2d_fp + proj_2d_fn))

    benchmark_record = ProjectionVs3DBenchmarkResult(
        case_id="SYN_3D_OVERLAP_XY",
        truth_3d_count=truth_3d_count,
        projected_2d_pred_count=len(seg_2d_res.objects),
        projected_2d_tp=proj_2d_tp,
        projected_2d_fp=proj_2d_fp,
        projected_2d_fn=proj_2d_fn,
        projected_2d_precision=proj_2d_precision,
        projected_2d_recall=proj_2d_recall,
        projected_2d_f1=proj_2d_f1,
        true_3d_pred_count=seg_3d_res.nucleus_count,
        true_3d_tp=matching_3d.tp_count,
        true_3d_fp=matching_3d.fp_count,
        true_3d_fn=matching_3d.fn_count,
        true_3d_precision=metrics_3d.precision,
        true_3d_recall=metrics_3d.recall,
        true_3d_f1=metrics_3d.f1_score,
    )

    assert benchmark_record.truth_3d_count == 2
    assert benchmark_record.projected_2d_pred_count == 1
    assert benchmark_record.projected_2d_recall == 0.50
    assert pytest.approx(benchmark_record.projected_2d_f1, rel=1e-6) == 2.0 / 3.0

    assert benchmark_record.true_3d_pred_count == 2
    assert benchmark_record.true_3d_recall == 1.00
    assert benchmark_record.true_3d_f1 == 1.00
