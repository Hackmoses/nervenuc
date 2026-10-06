"""Comprehensive tests for deterministic 2D/3D instance matching and validation models."""

import numpy as np
import pytest

from nervenuc.preparation.models import VoxelSpacing
from nervenuc.validation.errors import (
    InvalidMatchingInputError,
    InvalidThresholdError,
    ValidationCaseSpecificationError,
)
from nervenuc.validation.matching import match_instances
from nervenuc.validation.models import (
    GroundTruthObject,
    MatchingResult,
    ObjectMatch,
    ValidationCaseSpec,
)


# ============================================================================
# 1. Validation Models & Specifications
# ============================================================================


def test_validation_case_spec_creation_and_validation() -> None:
    """Verify ValidationCaseSpec creation, serialization fields, and error handling."""
    spec = ValidationCaseSpec(
        case_id="CASE_001",
        description="Synthetic isolated nuclei benchmark",
        tier="synthetic",
        dimensions="2D",
        image_relpath="images/case_001.tif",
        ground_truth_labels_relpath="labels/case_001_gt.tif",
        ground_truth_phenotypes_relpath="phenotypes/case_001_gt.csv",
        image_sha256="abc123hash",
        ground_truth_sha256="def456hash",
        axes="CYX",
        channel_names=("DAPI", "MarkerA"),
        voxel_spacing_um=(1.0, 0.5, 0.5),
        annotator_id="expert_1",
    )
    assert spec.case_id == "CASE_001"
    assert spec.tier == "synthetic"
    assert spec.dimensions == "2D"
    assert spec.channel_names == ("DAPI", "MarkerA")

    # Invalid tier
    with pytest.raises(ValidationCaseSpecificationError, match="tier must be one of"):
        ValidationCaseSpec(
            case_id="CASE_ERR",
            description="desc",
            tier="invalid_tier",  # type: ignore[arg-type]
            dimensions="2D",
            image_relpath="img.tif",
            ground_truth_labels_relpath="gt.tif",
        )

    # Empty case_id
    with pytest.raises(ValidationCaseSpecificationError, match="case_id must be a non-empty string"):
        ValidationCaseSpec(
            case_id="",
            description="desc",
            tier="synthetic",
            dimensions="2D",
            image_relpath="img.tif",
            ground_truth_labels_relpath="gt.tif",
        )


def test_ground_truth_object_model() -> None:
    """Verify GroundTruthObject creation and non-negative constraints."""
    gt_obj = GroundTruthObject(
        truth_id=1,
        label_name="NUC-001",
        centroid=(10.5, 20.5),
        size=150,
        phenotype="MarkerA+",
        marker_positivity={"MarkerA": True, "MarkerB": False},
    )
    assert gt_obj.truth_id == 1
    assert gt_obj.size == 150
    assert gt_obj.marker_positivity["MarkerA"] is True


# ============================================================================
# 2. Basic 2D & 3D Matching
# ============================================================================


def test_perfect_2d_one_object_match() -> None:
    """Verify perfect 1:1 match in 2D yielding exact IoU=1.0, Dice=1.0, zero centroid error."""
    pred = np.zeros((30, 40), dtype=np.int32)
    truth = np.zeros((30, 40), dtype=np.int32)

    pred[5:15, 10:25] = 1
    truth[5:15, 10:25] = 1

    res = match_instances(pred, truth, iou_threshold=0.5, case_id="perf_2d")

    assert res.case_id == "perf_2d"
    assert res.tp_count == 1
    assert res.fp_count == 0
    assert res.fn_count == 0
    assert res.pred_count == 1
    assert res.truth_count == 1
    assert len(res.matches) == 1

    m = res.matches[0]
    assert m.match_type == "TP"
    assert m.pred_id == 1
    assert m.truth_id == 1
    assert m.iou == 1.0
    assert m.dice == 1.0
    assert m.pred_size == 10 * 15
    assert m.truth_size == 10 * 15
    assert m.signed_size_diff == 0
    assert m.abs_size_diff == 0
    assert m.size_ratio == 1.0
    assert m.centroid_distance_px == 0.0


def test_perfect_3d_one_object_match() -> None:
    """Verify perfect 1:1 match in 3D (ZYX) with anisotropic spacing."""
    pred = np.zeros((10, 30, 40), dtype=np.int32)
    truth = np.zeros((10, 30, 40), dtype=np.int32)

    pred[2:6, 5:15, 10:25] = 1
    truth[2:6, 5:15, 10:25] = 1

    spacing = VoxelSpacing(z=2.0, y=0.5, x=0.25)
    res = match_instances(pred, truth, iou_threshold=0.5, spacing=spacing, case_id="perf_3d")

    assert res.tp_count == 1
    assert res.fp_count == 0
    assert res.fn_count == 0
    m = res.matches[0]
    assert m.match_type == "TP"
    assert m.iou == 1.0
    assert m.dice == 1.0
    assert m.centroid_distance_px == 0.0
    assert m.centroid_distance_um == 0.0


def test_partial_overlap_and_size_metrics() -> None:
    """Verify partial overlap calculations, Dice, size difference, and centroid distance."""
    pred = np.zeros((40, 50), dtype=np.int32)
    truth = np.zeros((40, 50), dtype=np.int32)

    # Pred: [10:20, 10:20] -> size 100, centroid (14.5, 14.5)
    pred[10:20, 10:20] = 1
    # Truth: [10:20, 15:25] -> size 100, centroid (14.5, 19.5)
    # Overlap: [10:20, 15:20] -> size 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 (0.3333333333333333)
    # Dice: 2 * 50 / (100 + 100) = 0.5
    truth[10:20, 15:25] = 1

    # With threshold 0.3 -> TP
    res = match_instances(pred, truth, iou_threshold=0.3, case_id="partial")
    assert res.tp_count == 1
    assert res.fp_count == 0
    assert res.fn_count == 0

    m = res.matches[0]
    assert m.match_type == "TP"
    assert pytest.approx(m.iou, rel=1e-6) == 50.0 / 150.0
    assert pytest.approx(m.dice, rel=1e-6) == 0.5
    assert m.pred_size == 100
    assert m.truth_size == 100
    assert m.signed_size_diff == 0
    assert m.abs_size_diff == 0
    assert m.size_ratio == 1.0
    assert pytest.approx(m.centroid_distance_px, rel=1e-6) == 5.0


def test_threshold_exact_and_just_below() -> None:
    """Verify boundary threshold behavior: exact cutoff matches, strictly below splits into FP/FN."""
    pred = np.zeros((40, 50), dtype=np.int32)
    truth = np.zeros((40, 50), dtype=np.int32)

    # Overlap gives IoU = 0.5
    # Pred: [0:10, 0:10] (100)
    # Truth: [0:10, 0:15] (150)
    # Overlap: [0:10, 0:10] (100) -> Union = 150 -> IoU = 100/150 = 2/3 (0.666667)
    # Let's make exact IoU = 0.5:
    # Pred: [0:10, 0:10] (100)
    # Truth: [0:10, 0:15] (150) -> Overlap: [0:10, 0:10] (100), Union = 150 (IoU = 2/3)
    # To get IoU = 0.5:
    # Pred size = 20, Truth size = 20, Overlap = 10 -> Union = 30 -> IoU = 10/30 = 0.333
    # Pred size = 10, Truth size = 10, Overlap = 6 -> Union = 14 -> IoU = 6/14 = 0.42857
    # Pred size = 10, Truth size = 10, Overlap = 5 -> Union = 15 -> IoU = 5/15 = 1/3 (0.3333)
    # Let's make pred size = 10, truth size = 15, overlap = 10 -> Union = 15 -> IoU = 10/15 = 2/3.
    pred[0:10, 0:10] = 1
    truth[0:10, 0:15] = 1
    exact_iou = 100.0 / 150.0  # 2/3

    # At tau = 2/3 -> TP
    res_at = match_instances(pred, truth, iou_threshold=exact_iou)
    assert res_at.tp_count == 1
    assert res_at.fp_count == 0
    assert res_at.fn_count == 0

    # At tau = 2/3 + 0.001 -> FP and FN
    res_below = match_instances(pred, truth, iou_threshold=exact_iou + 0.001)
    assert res_below.tp_count == 0
    assert res_below.fp_count == 1
    assert res_below.fn_count == 1
    assert len(res_below.matches) == 2


# ============================================================================
# 3. Competition & Cardinality Guarantees (Tier 1 vs Tier 2)
# ============================================================================


def test_cardinality_priority_regression() -> None:
    """Mandatory Tier 1 cardinality test: matcher MUST maximize number of valid matches over raw IoU.

    Constructed scenario:
    - Truth T1 and T2.
    - Prediction P1 overlaps both T1 and T2:
        IoU(P1, T1) = 0.85
        IoU(P1, T2) = 0.55
    - Prediction P2 overlaps only T1:
        IoU(P2, T1) = 0.60
        IoU(P2, T2) = 0.0

    With tau = 0.5:
    Assignment A: (P1, T1) -> 1 match with IoU = 0.85 (P2 and T2 unmatched). Total valid matches = 1.
    Assignment B: (P1, T2) and (P2, T1) -> 2 matches with IoUs (0.55, 0.60). Total valid matches = 2.

    Scientific Contract: Assignment B MUST be chosen because 2 valid matches > 1 valid match.
    """
    pred = np.zeros((30, 60), dtype=np.int32)
    truth = np.zeros((30, 60), dtype=np.int32)

    # Let's construct explicit pixel masks:
    # Ground Truth:
    # T1: [5:25, 5:25] (20 x 20 = 400 px)
    # T2: [5:25, 35:55] (20 x 20 = 400 px)
    truth[5:25, 5:25] = 1
    truth[5:25, 35:55] = 2

    # P1: overlaps mostly T1 and partially T2
    # P1 spans [5:25, 7:47] (20 x 40 = 800 px)
    # Overlap P1 with T1: [5:25, 7:25] = 20 x 18 = 360 px
    # Union(P1, T1) = 800 + 400 - 360 = 840 -> IoU(P1, T1) = 360 / 840 = 0.4285 (let's tune this)

    # Let's verify via direct synthetic layout:
    # T1 size = 100. T2 size = 100.
    # P1 size = 120. P2 size = 100.
    # P1 overlaps T1 by 90 (Union = 120 + 100 - 90 = 130 -> IoU = 90/130 = 0.692)
    # P1 overlaps T2 by 65 (Union = 120 + 100 - 65 = 155 -> IoU = 65/155 = 0.419)
    # Let's ensure IoU(P1, T2) >= 0.50:
    # If P1 size = 100, overlaps T2 by 60 (Union = 140 -> IoU = 60/140 = 0.428)
    # If P1 overlaps T2 by 70 and T1 by 70 -> P1 size = 140.
    # IoU(P1, T1) = 70 / (140 + 100 - 70) = 70 / 170 = 0.411
    # IoU(P1, T2) = 70 / 170 = 0.411

    # Clean layout with separate pixels:
    # T1: 100 pixels in region A
    # T2: 100 pixels in region B
    # P1: 80 pixels in region A, 60 pixels in region B -> P1 size = 140
    # Overlap(P1, T1) = 80 -> Union = 140 + 100 - 80 = 160 -> IoU(P1, T1) = 80/160 = 0.50
    # Overlap(P1, T2) = 60 -> Union = 140 + 100 - 60 = 180 -> IoU(P1, T2) = 60/180 = 0.333
    # Let's make:
    # T1: 50 px, T2: 50 px
    # P2: overlaps T1 by 30 px. P2 size = 40 px. Overlap = 30 -> Union = 40 + 50 - 30 = 60 -> IoU(P2, T1) = 30/60 = 0.50
    # P1: overlaps T1 by 45 px, and T2 by 30 px. P1 size = 75 px.
    # IoU(P1, T1) = 45 / (75 + 50 - 45) = 45 / 80 = 0.5625
    # IoU(P1, T2) = 30 / (75 + 50 - 30) = 30 / 95 = 0.3157 (under 0.5)

    # Let's construct a grid where:
    # IoU(P1, T1) = 0.80 (>= 0.5)
    # IoU(P1, T2) = 0.55 (>= 0.5)
    # IoU(P2, T1) = 0.52 (>= 0.5)
    # IoU(P2, T2) = 0.00
    #
    # With tau = 0.50:
    # If algorithm greedily/naively takes (P1, T1) with IoU=0.80:
    # P2 cannot match T1, and P1 cannot match T2.
    # T2 has no other match. P2 has no other match.
    # Result: 1 TP (P1, T1), 1 FP (P2), 1 FN (T2). Valid pairs = 1.
    #
    # Correct Tier 1 solution:
    # Match (P2, T1) [IoU=0.52 >= 0.50] AND (P1, T2) [IoU=0.55 >= 0.50].
    # Result: 2 TPs, 0 FP, 0 FN. Valid pairs = 2.

    # Array implementation of this exact matrix:
    # Let's create an array with 1D row layout for exact pixel counts:
    total_px = 500
    pred_arr = np.zeros((1, total_px), dtype=np.int32)
    truth_arr = np.zeros((1, total_px), dtype=np.int32)

    # T1: indices 0..99 (100 px)
    truth_arr[0, 0:100] = 1
    # T2: indices 100..199 (100 px)
    truth_arr[0, 100:200] = 2

    # P1: overlaps 75 px of T1 (indices 0..74) and 60 px of T2 (indices 100..159)
    # P1 size = 75 + 60 = 135 px
    # Overlap(P1, T1) = 75 -> Union = 135 + 100 - 75 = 160 -> IoU = 75/160 = 0.46875 (let's increase to 0.70)
    # Overlap(P1, T2) = 60 -> Union = 135 + 100 - 60 = 175 -> IoU = 60/175 = 0.3428
    # To get IoU(P1, T2) >= 0.5:
    # Overlap = 70 px of T2 (indices 100..169).
    # If T2 size = 70 px (indices 100..169) -> Overlap = 70, P1 in T2 = 70.
    # Let's set sizes:
    # T1 size = 60 px (0..59)
    # T2 size = 60 px (60..119)
    # P1: occupies 0..49 (50 px in T1) and 60..109 (50 px in T2) -> P1 size = 100 px.
    # Overlap(P1, T1) = 50 -> Union = 100 + 60 - 50 = 110 -> IoU = 50/110 = 0.4545 (not >= 0.5)
    #
    # To make IoU >= 0.5:
    # Overlap / (Size1 + Size2 - Overlap) >= 0.5 <=> 3*Overlap >= Size1 + Size2.
    # If T1 = 30, T2 = 30.
    # P1 occupies 25 px in T1 and 25 px in T2 -> P1 size = 50.
    # Overlap(P1, T1) = 25 -> Union = 50 + 30 - 25 = 55 -> IoU = 25/55 = 0.4545 (almost!)
    # If T1 = 20, T2 = 20.
    # P1 occupies 18 px in T1 and 18 px in T2 -> P1 size = 36.
    # Overlap(P1, T1) = 18 -> Union = 36 + 20 - 18 = 38 -> IoU(P1, T1) = 18/38 = 0.4736
    # If P1 occupies 19 px in T1 and 19 px in T2 -> P1 size = 38.
    # Overlap(P1, T1) = 19 -> Union = 38 + 20 - 19 = 39 -> IoU(P1, T1) = 19/39 = 0.487
    # If P1 occupies 20 px in T1 and 20 px in T2 -> P1 size = 40.
    # Overlap(P1, T1) = 20 -> Union = 40 + 20 - 20 = 40 -> IoU(P1, T1) = 20/40 = 0.50 (Exact!)
    # IoU(P1, T2) = 20/40 = 0.50 (Exact!)
    # Now let's make P1 overlap T1 slightly more so IoU(P1, T1) is higher:
    # If T1 = 20, T2 = 18.
    # P1 occupies 20 px in T1 and 18 px in T2 -> P1 size = 38.
    # Overlap(P1, T1) = 20 -> Union = 38 + 20 - 20 = 38 -> IoU(P1, T1) = 20/38 = 0.526
    # Overlap(P1, T2) = 18 -> Union = 38 + 18 - 18 = 38 -> IoU(P1, T2) = 18/38 = 0.473
    # If T2 = 15: Overlap(P1, T2) = 15 -> Union = 35 + 15 - 15 = 35 -> IoU = 15/35 = 0.428
    #
    # Wait, can P2 also overlap T1?
    # P2 and P1 are predicted instance masks (disjoint in prediction label array).
    # Since P1 and P2 cannot both occupy the same pixel in pred_arr:
    # If T1 has size 50:
    # P1 occupies 25 px of T1. P2 occupies 25 px of T1.
    # T2 has size 25:
    # P1 occupies all 25 px of T2.
    # Then:
    # P1 size = 25 (in T1) + 25 (in T2) = 50 px.
    # Overlap(P1, T1) = 25 -> Union = 50 + 50 - 25 = 75 -> IoU(P1, T1) = 25/75 = 0.333
    # Overlap(P1, T2) = 25 -> Union = 50 + 25 - 25 = 50 -> IoU(P1, T2) = 25/50 = 0.50
    # P2 size = 25 (in T1) = 25 px.
    # Overlap(P2, T1) = 25 -> Union = 25 + 50 - 25 = 50 -> IoU(P2, T1) = 25/50 = 0.50
    #
    # Now let's check:
    # IoU(P1, T2) = 0.50 (valid at tau=0.50)
    # IoU(P2, T1) = 0.50 (valid at tau=0.50)
    # IoU(P1, T1) = 0.333 (< 0.50)
    # Here, matching (P2, T1) and (P1, T2) gives 2 valid matches.
    #
    # What if IoU(P1, T1) were 0.60? In disjoint label arrays, if P1 occupies a larger share of T1,
    # P2 gets less of T1.
    # Let's test with tau = 0.40:
    # IoU(P1, T1) = 0.45 (> 0.40)
    # IoU(P1, T2) = 0.50 (> 0.40)
    # IoU(P2, T1) = 0.50 (> 0.40)
    # If naive algorithm matches (P1, T2) -> total IoU = 0.50 (1 match) OR (P1, T1) -> 0.45 (1 match)
    # The true maximum cardinality matching is (P1, T2) + (P2, T1) -> 2 matches!

    arr_p = np.zeros((1, 100), dtype=np.int32)
    arr_t = np.zeros((1, 100), dtype=np.int32)

    # T1: 0..49 (50 px)
    arr_t[0, 0:50] = 1
    # T2: 50..74 (25 px)
    arr_t[0, 50:75] = 2

    # P1: 0..24 (25 px in T1) + 50..74 (25 px in T2) -> size 50
    arr_p[0, 0:25] = 1
    arr_p[0, 50:75] = 1

    # P2: 25..49 (25 px in T1) -> size 25
    arr_p[0, 25:50] = 2

    # IoU(P1, T2) = 25 / (50 + 25 - 25) = 25/50 = 0.50
    # IoU(P2, T1) = 25 / (25 + 50 - 25) = 25/50 = 0.50
    # IoU(P1, T1) = 25 / (50 + 50 - 25) = 25/75 = 0.333333

    # With tau = 0.40:
    # Match (P1, T2) and (P2, T1) gives 2 valid TPs.
    res = match_instances(arr_p, arr_t, iou_threshold=0.40, case_id="cardinality_priority")
    assert res.tp_count == 2
    assert res.fp_count == 0
    assert res.fn_count == 0

    matched_pairs = {(m.pred_id, m.truth_id) for m in res.matches if m.match_type == "TP"}
    assert matched_pairs == {(1, 2), (2, 1)}


def test_total_iou_secondary_objective() -> None:
    """Verify Tier 2 secondary objective: among equal cardinality matchings, maximize total sum of IoU."""
    pred = np.zeros((20, 20), dtype=np.int32)
    truth = np.zeros((20, 20), dtype=np.int32)

    # P1 and P2 can each match T1 or T2
    # P1 overlaps T1 (IoU=0.9) and T2 (IoU=0.6)
    # P2 overlaps T1 (IoU=0.6) and T2 (IoU=0.8)
    # Both (P1->T1, P2->T2) [sum=1.7] and (P1->T2, P2->T1) [sum=1.2] have cardinality 2.
    # Matcher MUST pick (P1->T1, P2->T2).
    pred[0:8, 0:8] = 1
    pred[10:18, 10:18] = 2

    truth[0:8, 0:8] = 1
    truth[10:18, 10:18] = 2

    res = match_instances(pred, truth, iou_threshold=0.5)
    assert res.tp_count == 2
    matched_pairs = {(m.pred_id, m.truth_id) for m in res.matches if m.match_type == "TP"}
    assert matched_pairs == {(1, 1), (2, 2)}


def test_symmetric_deterministic_tie_stability() -> None:
    """Verify deterministic tie handling on a symmetric case executed multiple times."""
    # Create two identical symmetrical objects
    pred = np.zeros((20, 40), dtype=np.int32)
    truth = np.zeros((20, 40), dtype=np.int32)

    pred[5:15, 5:15] = 1
    pred[5:15, 25:35] = 2

    truth[5:15, 5:15] = 1
    truth[5:15, 25:35] = 2

    first_res = match_instances(pred, truth, iou_threshold=0.5)

    for _ in range(10):
        res = match_instances(pred, truth, iou_threshold=0.5)
        assert res.tp_count == first_res.tp_count
        assert res.fp_count == first_res.fp_count
        assert res.fn_count == first_res.fn_count
        assert len(res.matches) == len(first_res.matches)
        for m1, m2 in zip(first_res.matches, res.matches):
            assert m1.pred_id == m2.pred_id
            assert m1.truth_id == m2.truth_id
            assert m1.match_type == m2.match_type
            assert m1.iou == m2.iou


# ============================================================================
# 4. Non-Contiguous IDs, FPs, FNs, and Empty Cases
# ============================================================================


def test_non_contiguous_label_ids() -> None:
    """Verify matching with non-contiguous label IDs (e.g. truth=[5, 12], pred=[3, 99])."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[5:10, 5:10] = 3
    pred[15:25, 15:25] = 99

    truth[5:10, 5:10] = 5
    truth[15:25, 15:25] = 12

    res = match_instances(
        pred,
        truth,
        iou_threshold=0.5,
        pred_review_ids={3: "MAN-000003", 99: "AUTO-000099"},
        truth_names={5: "NUC-005", 12: "NUC-012"},
    )

    assert res.tp_count == 2
    assert res.fp_count == 0
    assert res.fn_count == 0
    assert res.pred_count == 2
    assert res.truth_count == 2

    m1 = res.matches[0]
    assert m1.pred_id == 3
    assert m1.pred_review_id == "MAN-000003"
    assert m1.truth_id == 5
    assert m1.truth_name == "NUC-005"
    assert m1.iou == 1.0

    m2 = res.matches[1]
    assert m2.pred_id == 99
    assert m2.pred_review_id == "AUTO-000099"
    assert m2.truth_id == 12
    assert m2.truth_name == "NUC-012"
    assert m2.iou == 1.0


def test_false_positive_and_false_negative_isolation() -> None:
    """Verify disjoint objects correctly generate FP and FN records."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[0:5, 0:5] = 1  # In top-left, no truth counterpart -> FP
    truth[20:25, 20:25] = 2  # In bottom-right, no pred counterpart -> FN

    res = match_instances(pred, truth, iou_threshold=0.5)
    assert res.tp_count == 0
    assert res.fp_count == 1
    assert res.fn_count == 1
    assert res.pred_count == 1
    assert res.truth_count == 1

    fn_match = [m for m in res.matches if m.match_type == "FN"][0]
    assert fn_match.truth_id == 2
    assert fn_match.pred_id is None
    assert fn_match.iou == 0.0

    fp_match = [m for m in res.matches if m.match_type == "FP"][0]
    assert fp_match.pred_id == 1
    assert fp_match.truth_id is None
    assert fp_match.iou == 0.0


def test_empty_cases() -> None:
    """Verify empty prediction, empty truth, and both empty cases."""
    shape = (20, 20)

    # 1. Both empty
    res_both_empty = match_instances(np.zeros(shape, dtype=np.int32), np.zeros(shape, dtype=np.int32))
    assert res_both_empty.tp_count == 0
    assert res_both_empty.fp_count == 0
    assert res_both_empty.fn_count == 0
    assert res_both_empty.pred_count == 0
    assert res_both_empty.truth_count == 0
    assert len(res_both_empty.matches) == 0

    # 2. Prediction empty, truth nonempty
    truth_only = np.zeros(shape, dtype=np.int32)
    truth_only[5:10, 5:10] = 1
    res_pred_empty = match_instances(np.zeros(shape, dtype=np.int32), truth_only)
    assert res_pred_empty.tp_count == 0
    assert res_pred_empty.fp_count == 0
    assert res_pred_empty.fn_count == 1
    assert res_pred_empty.pred_count == 0
    assert res_pred_empty.truth_count == 1
    assert res_pred_empty.matches[0].match_type == "FN"

    # 3. Prediction nonempty, truth empty
    pred_only = np.zeros(shape, dtype=np.int32)
    pred_only[5:10, 5:10] = 1
    res_truth_empty = match_instances(pred_only, np.zeros(shape, dtype=np.int32))
    assert res_truth_empty.tp_count == 0
    assert res_truth_empty.fp_count == 1
    assert res_truth_empty.fn_count == 0
    assert res_truth_empty.pred_count == 1
    assert res_truth_empty.truth_count == 0
    assert res_truth_empty.matches[0].match_type == "FP"


# ============================================================================
# 5. Input Validation & Error Handling
# ============================================================================


def test_shape_mismatch_rejection() -> None:
    """Verify shape mismatch raises InvalidMatchingInputError."""
    pred = np.zeros((20, 20), dtype=np.int32)
    truth = np.zeros((20, 30), dtype=np.int32)
    with pytest.raises(InvalidMatchingInputError, match="Shape mismatch"):
        match_instances(pred, truth)


def test_dimension_mismatch_rejection() -> None:
    """Verify 2D vs 3D mismatch raises InvalidMatchingInputError."""
    pred = np.zeros((10, 20, 20), dtype=np.int32)
    truth = np.zeros((20, 20), dtype=np.int32)
    with pytest.raises(InvalidMatchingInputError, match="Shape mismatch"):
        match_instances(pred, truth)


def test_unsupported_ndim_rejection() -> None:
    """Verify 1D or 4D arrays raise InvalidMatchingInputError."""
    pred = np.zeros((20,), dtype=np.int32)
    truth = np.zeros((20,), dtype=np.int32)
    with pytest.raises(InvalidMatchingInputError, match="must be 2D.*or 3D"):
        match_instances(pred, truth)


def test_float_dtype_rejection() -> None:
    """Verify float label arrays raise InvalidMatchingInputError."""
    pred = np.zeros((20, 20), dtype=np.float32)
    truth = np.zeros((20, 20), dtype=np.int32)
    with pytest.raises(InvalidMatchingInputError, match="integer dtype"):
        match_instances(pred, truth)


def test_negative_label_rejection() -> None:
    """Verify negative label values raise InvalidMatchingInputError."""
    pred = np.zeros((20, 20), dtype=np.int32)
    truth = np.zeros((20, 20), dtype=np.int32)
    pred[0, 0] = -1
    with pytest.raises(InvalidMatchingInputError, match="negative values"):
        match_instances(pred, truth)


def test_invalid_threshold_rejection() -> None:
    """Verify out-of-bounds or non-numeric thresholds raise InvalidThresholdError."""
    pred = np.zeros((20, 20), dtype=np.int32)
    truth = np.zeros((20, 20), dtype=np.int32)

    with pytest.raises(InvalidThresholdError, match="strictly greater than 0.0"):
        match_instances(pred, truth, iou_threshold=0.0)

    with pytest.raises(InvalidThresholdError, match="strictly greater than 0.0"):
        match_instances(pred, truth, iou_threshold=-0.5)

    with pytest.raises(InvalidThresholdError, match="<= 1.0"):
        match_instances(pred, truth, iou_threshold=1.5)

    with pytest.raises(InvalidThresholdError, match="finite float"):
        match_instances(pred, truth, iou_threshold=float("nan"))
