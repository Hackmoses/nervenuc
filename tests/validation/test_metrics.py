"""Comprehensive tests for segmentation validation metrics, overlap, count, and size summaries."""

import numpy as np
import pytest

from nervenuc.preparation.models import VoxelSpacing
from nervenuc.validation.matching import match_instances
from nervenuc.validation.metrics import compute_segmentation_metrics


def test_perfect_segmentation_metrics() -> None:
    """Verify perfect 1:1 segmentation yields exact 1.0 precision, recall, F1, IoU, Dice, zero errors."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[5:15, 5:15] = 1
    pred[18:25, 18:25] = 2

    truth[5:15, 5:15] = 1
    truth[18:25, 18:25] = 2

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is True
    assert metrics.precision == 1.0
    assert metrics.recall_defined is True
    assert metrics.recall == 1.0
    assert metrics.f1_score_defined is True
    assert metrics.f1_score == 1.0

    assert metrics.mean_matched_iou == 1.0
    assert metrics.median_matched_iou == 1.0
    assert metrics.mean_matched_dice == 1.0
    assert metrics.median_matched_dice == 1.0

    assert metrics.foreground_iou_defined is True
    assert metrics.foreground_iou == 1.0
    assert metrics.foreground_dice_defined is True
    assert metrics.foreground_dice == 1.0

    assert metrics.pred_count == 2
    assert metrics.truth_count == 2
    assert metrics.count_signed_error == 0
    assert metrics.count_absolute_error == 0
    assert metrics.count_percent_error_defined is True
    assert metrics.count_percent_error == 0.0

    assert metrics.mean_centroid_error_px == 0.0
    assert metrics.median_centroid_error_px == 0.0
    assert metrics.mean_signed_size_diff == 0.0
    assert metrics.mean_absolute_size_diff == 0.0
    assert metrics.mean_size_ratio == 1.0
    assert metrics.is_empty_case_concordant is False


def test_one_tp_and_one_fp() -> None:
    """Verify 1 TP + 1 FP yields P=0.5, R=1.0, F1=2/3, count signed error=+1, percent error=+100%."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[5:15, 5:15] = 1  # Matches truth 1
    pred[20:25, 20:25] = 2  # False positive

    truth[5:15, 5:15] = 1  # 1 truth nucleus

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is True
    assert metrics.precision == 0.5
    assert metrics.recall_defined is True
    assert metrics.recall == 1.0
    assert metrics.f1_score_defined is True
    assert pytest.approx(metrics.f1_score, rel=1e-6) == 2.0 / 3.0

    assert metrics.pred_count == 2
    assert metrics.truth_count == 1
    assert metrics.count_signed_error == 1
    assert metrics.count_absolute_error == 1
    assert metrics.count_percent_error_defined is True
    assert metrics.count_percent_error == 100.0


def test_one_tp_and_one_fn() -> None:
    """Verify 1 TP + 1 FN yields P=1.0, R=0.5, F1=2/3, count signed error=-1, percent error=-50%."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[5:15, 5:15] = 1  # Matches truth 1

    truth[5:15, 5:15] = 1
    truth[20:25, 20:25] = 2  # False negative (missed)

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is True
    assert metrics.precision == 1.0
    assert metrics.recall_defined is True
    assert metrics.recall == 0.5
    assert metrics.f1_score_defined is True
    assert pytest.approx(metrics.f1_score, rel=1e-6) == 2.0 / 3.0

    assert metrics.pred_count == 1
    assert metrics.truth_count == 2
    assert metrics.count_signed_error == -1
    assert metrics.count_absolute_error == 1
    assert metrics.count_percent_error_defined is True
    assert metrics.count_percent_error == -50.0


def test_tp_fp_fn_combination() -> None:
    """Verify hand-computed combination: 2 TP, 1 FP, 1 FN -> P=2/3, R=2/3, F1=2/3, count error=0."""
    pred = np.zeros((40, 40), dtype=np.int32)
    truth = np.zeros((40, 40), dtype=np.int32)

    pred[0:5, 0:5] = 1  # TP with truth 1
    pred[10:15, 10:15] = 2  # TP with truth 2
    pred[20:25, 20:25] = 3  # FP

    truth[0:5, 0:5] = 1
    truth[10:15, 10:15] = 2
    truth[30:35, 30:35] = 3  # FN

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is True
    assert pytest.approx(metrics.precision, rel=1e-6) == 2.0 / 3.0
    assert metrics.recall_defined is True
    assert pytest.approx(metrics.recall, rel=1e-6) == 2.0 / 3.0
    assert metrics.f1_score_defined is True
    assert pytest.approx(metrics.f1_score, rel=1e-6) == 2.0 / 3.0

    assert metrics.pred_count == 3
    assert metrics.truth_count == 3
    assert metrics.count_signed_error == 0
    assert metrics.count_absolute_error == 0
    assert metrics.count_percent_error == 0.0


def test_both_empty_case_semantics() -> None:
    """Verify both-empty case: precision/recall/F1 undefined, is_empty_case_concordant=True, fg IoU/Dice=1.0."""
    shape = (20, 20)
    pred = np.zeros(shape, dtype=np.int32)
    truth = np.zeros(shape, dtype=np.int32)

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is False
    assert metrics.precision is None
    assert metrics.recall_defined is False
    assert metrics.recall is None
    assert metrics.f1_score_defined is False
    assert metrics.f1_score is None

    assert metrics.pred_count == 0
    assert metrics.truth_count == 0
    assert metrics.count_signed_error == 0
    assert metrics.count_absolute_error == 0
    assert metrics.count_percent_error_defined is False
    assert metrics.count_percent_error is None

    assert metrics.is_empty_case_concordant is True
    assert metrics.foreground_iou_defined is True
    assert metrics.foreground_iou == 1.0
    assert metrics.foreground_dice_defined is True
    assert metrics.foreground_dice == 1.0

    assert metrics.mean_matched_iou is None
    assert metrics.mean_centroid_error_px is None


def test_prediction_empty_truth_nonempty() -> None:
    """Verify prediction empty: precision undefined, recall=0.0, F1=0.0, percent error=-100%, fg IoU=0.0."""
    shape = (20, 20)
    pred = np.zeros(shape, dtype=np.int32)
    truth = np.zeros(shape, dtype=np.int32)
    truth[5:10, 5:10] = 1

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is False
    assert metrics.precision is None
    assert metrics.recall_defined is True
    assert metrics.recall == 0.0
    assert metrics.f1_score_defined is True
    assert metrics.f1_score == 0.0

    assert metrics.count_signed_error == -1
    assert metrics.count_percent_error_defined is True
    assert metrics.count_percent_error == -100.0

    assert metrics.foreground_iou == 0.0
    assert metrics.foreground_dice == 0.0
    assert metrics.is_empty_case_concordant is False


def test_truth_empty_prediction_nonempty() -> None:
    """Verify truth empty: precision=0.0, recall undefined, F1=0.0, percent error undefined, fg IoU=0.0."""
    shape = (20, 20)
    pred = np.zeros(shape, dtype=np.int32)
    pred[5:10, 5:10] = 1
    truth = np.zeros(shape, dtype=np.int32)

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.precision_defined is True
    assert metrics.precision == 0.0
    assert metrics.recall_defined is False
    assert metrics.recall is None
    assert metrics.f1_score_defined is True
    assert metrics.f1_score == 0.0

    assert metrics.count_signed_error == 1
    assert metrics.count_percent_error_defined is False
    assert metrics.count_percent_error is None

    assert metrics.foreground_iou == 0.0
    assert metrics.foreground_dice == 0.0
    assert metrics.is_empty_case_concordant is False


def test_matched_iou_and_dice_summaries_tp_only() -> None:
    """Verify matched IoU and Dice means/medians compute strictly over TP matches without FP/FN zeros."""
    pred = np.zeros((40, 40), dtype=np.int32)
    truth = np.zeros((40, 40), dtype=np.int32)

    # TP 1: exact match -> IoU = 1.0, Dice = 1.0
    pred[0:10, 0:10] = 1
    truth[0:10, 0:10] = 1

    # TP 2: partial overlap
    # Pred: [20:30, 20:30] (100 px), Truth: [20:30, 20:35] (150 px), Overlap: 100 -> IoU = 100/150 = 2/3, Dice = 200/250 = 0.8
    pred[20:30, 20:30] = 2
    truth[20:30, 20:35] = 2

    # FP 1: no match
    pred[35:40, 0:5] = 3

    matching = match_instances(pred, truth, iou_threshold=0.5)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert matching.tp_count == 2
    expected_mean_iou = (1.0 + 2.0 / 3.0) / 2.0
    assert pytest.approx(metrics.mean_matched_iou, rel=1e-6) == expected_mean_iou
    expected_mean_dice = (1.0 + 0.8) / 2.0
    assert pytest.approx(metrics.mean_matched_dice, rel=1e-6) == expected_mean_dice


def test_centroid_and_size_agreement_summaries() -> None:
    """Verify centroid distance errors and size differences in pixels and calibrated physical units."""
    pred = np.zeros((5, 30, 30), dtype=np.int32)
    truth = np.zeros((5, 30, 30), dtype=np.int32)

    # Pred: [1:4, 5:15, 5:15] -> Z=3, Y=10, X=10 -> size = 300. Centroid = (2.0, 9.5, 9.5)
    pred[1:4, 5:15, 5:15] = 1
    # Truth: [1:4, 5:15, 8:18] -> Z=3, Y=10, X=10 -> size = 300. Centroid = (2.0, 9.5, 12.5) -> delta X = 3.0 px
    truth[1:4, 5:15, 8:18] = 1

    spacing = VoxelSpacing(z=2.0, y=0.5, x=0.5)
    matching = match_instances(pred, truth, iou_threshold=0.4, spacing=spacing)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert matching.tp_count == 1
    assert pytest.approx(metrics.mean_centroid_error_px, rel=1e-6) == 3.0
    # Physical: delta X * 0.5 um = 1.5 um
    assert pytest.approx(metrics.mean_centroid_error_um, rel=1e-6) == 1.5
    assert metrics.mean_signed_size_diff == 0.0
    assert metrics.mean_size_ratio == 1.0


def test_uncalibrated_centroid_physical_distance_is_none() -> None:
    """Verify physical centroid error is None when spacing is uncalibrated."""
    pred = np.zeros((30, 30), dtype=np.int32)
    truth = np.zeros((30, 30), dtype=np.int32)

    pred[5:15, 5:15] = 1
    truth[5:15, 8:18] = 1

    matching = match_instances(pred, truth, iou_threshold=0.4)
    metrics = compute_segmentation_metrics(matching, pred, truth)

    assert metrics.mean_centroid_error_px is not None
    assert metrics.mean_centroid_error_um is None
