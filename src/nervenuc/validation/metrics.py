"""Scientific evaluation metrics for segmentation, marker classification, and phenotype benchmarking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InvalidMetricInputError
from .models import (
    BiologicalMetricComparison,
    MarkerClassificationMetrics,
    MatchingResult,
    PhenotypeConfusionMatrix,
    SegmentationMetrics,
)


def compute_segmentation_metrics(
    matching_result: MatchingResult,
    predicted_labels: np.ndarray,
    truth_labels: np.ndarray,
) -> SegmentationMetrics:
    """Compute object-level detection, overlap, size, and centroid evaluation metrics.

    Args:
        matching_result: Immutable MatchingResult from match_instances.
        predicted_labels: 2D or 3D integer array of predicted nuclei.
        truth_labels: 2D or 3D integer array of ground-truth nuclei.

    Returns:
        Immutable SegmentationMetrics record with strict zero-denominator handling.
    """
    tp = matching_result.tp_count
    fp = matching_result.fp_count
    fn = matching_result.fn_count
    pred_count = matching_result.pred_count
    truth_count = matching_result.truth_count

    # 1. Object precision
    if tp + fp > 0:
        precision = float(tp / (tp + fp))
        precision_defined = True
    else:
        precision = None
        precision_defined = False

    # 2. Object recall
    if tp + fn > 0:
        recall = float(tp / (tp + fn))
        recall_defined = True
    else:
        recall = None
        recall_defined = False

    # 3. Object F1 score (2TP / (2TP + FP + FN))
    f1_denom = 2 * tp + fp + fn
    if f1_denom > 0:
        f1_score = float(2.0 * tp / f1_denom)
        f1_score_defined = True
    else:
        f1_score = None
        f1_score_defined = False

    # 4. Count errors
    count_signed_error = int(pred_count - truth_count)
    count_absolute_error = int(abs(count_signed_error))
    if truth_count > 0:
        count_percent_error = float(100.0 * (pred_count - truth_count) / truth_count)
        count_percent_error_defined = True
    else:
        count_percent_error = None
        count_percent_error_defined = False

    # 5. TP-only matched overlap, centroid, and size summaries
    tp_matches = [m for m in matching_result.matches if m.match_type == "TP"]

    if len(tp_matches) > 0:
        ious = [m.iou for m in tp_matches]
        mean_matched_iou: float | None = float(np.mean(ious))
        median_matched_iou: float | None = float(np.median(ious))

        dices = [m.dice for m in tp_matches]
        mean_matched_dice: float | None = float(np.mean(dices))
        median_matched_dice: float | None = float(np.median(dices))

        px_dists = [m.centroid_distance_px for m in tp_matches if m.centroid_distance_px is not None]
        mean_centroid_error_px: float | None = float(np.mean(px_dists)) if px_dists else None
        median_centroid_error_px: float | None = float(np.median(px_dists)) if px_dists else None

        um_dists = [m.centroid_distance_um for m in tp_matches if m.centroid_distance_um is not None]
        mean_centroid_error_um: float | None = float(np.mean(um_dists)) if um_dists else None
        median_centroid_error_um: float | None = float(np.median(um_dists)) if um_dists else None

        signed_diffs = [m.signed_size_diff for m in tp_matches if m.signed_size_diff is not None]
        mean_signed_size_diff: float | None = float(np.mean(signed_diffs)) if signed_diffs else None

        abs_diffs = [m.abs_size_diff for m in tp_matches if m.abs_size_diff is not None]
        mean_absolute_size_diff: float | None = float(np.mean(abs_diffs)) if abs_diffs else None
        median_absolute_size_diff: float | None = float(np.median(abs_diffs)) if abs_diffs else None

        ratios = [m.size_ratio for m in tp_matches if m.size_ratio is not None]
        mean_size_ratio: float | None = float(np.mean(ratios)) if ratios else None
    else:
        mean_matched_iou = None
        median_matched_iou = None
        mean_matched_dice = None
        median_matched_dice = None
        mean_centroid_error_px = None
        median_centroid_error_px = None
        mean_centroid_error_um = None
        median_centroid_error_um = None
        mean_signed_size_diff = None
        mean_absolute_size_diff = None
        median_absolute_size_diff = None
        mean_size_ratio = None

    # 6. Binary foreground overlap metrics
    pred_fg = predicted_labels > 0
    truth_fg = truth_labels > 0
    p_sum = int(np.sum(pred_fg))
    t_sum = int(np.sum(truth_fg))

    if p_sum == 0 and t_sum == 0:
        # Both foreground masks empty -> perfect agreement
        foreground_iou: float | None = 1.0
        foreground_dice: float | None = 1.0
        foreground_iou_defined = True
        foreground_dice_defined = True
    elif p_sum == 0 or t_sum == 0:
        # One empty, one non-empty
        foreground_iou = 0.0
        foreground_dice = 0.0
        foreground_iou_defined = True
        foreground_dice_defined = True
    else:
        fg_inter = int(np.sum(pred_fg & truth_fg))
        fg_union = int(np.sum(pred_fg | truth_fg))
        foreground_iou = float(fg_inter / fg_union)
        foreground_dice = float(2.0 * fg_inter / (p_sum + t_sum))
        foreground_iou_defined = True
        foreground_dice_defined = True

    # 7. Empty case concordance
    is_empty_case_concordant = (pred_count == 0 and truth_count == 0)

    return SegmentationMetrics(
        precision_defined=precision_defined,
        precision=precision,
        recall_defined=recall_defined,
        recall=recall,
        f1_score_defined=f1_score_defined,
        f1_score=f1_score,
        mean_matched_iou=mean_matched_iou,
        median_matched_iou=median_matched_iou,
        mean_matched_dice=mean_matched_dice,
        median_matched_dice=median_matched_dice,
        foreground_iou_defined=foreground_iou_defined,
        foreground_iou=foreground_iou,
        foreground_dice_defined=foreground_dice_defined,
        foreground_dice=foreground_dice,
        pred_count=pred_count,
        truth_count=truth_count,
        count_signed_error=count_signed_error,
        count_absolute_error=count_absolute_error,
        count_percent_error_defined=count_percent_error_defined,
        count_percent_error=count_percent_error,
        mean_centroid_error_px=mean_centroid_error_px,
        median_centroid_error_px=median_centroid_error_px,
        mean_centroid_error_um=mean_centroid_error_um,
        median_centroid_error_um=median_centroid_error_um,
        mean_signed_size_diff=mean_signed_size_diff,
        mean_absolute_size_diff=mean_absolute_size_diff,
        median_absolute_size_diff=median_absolute_size_diff,
        mean_size_ratio=mean_size_ratio,
        is_empty_case_concordant=is_empty_case_concordant,
    )


def evaluate_marker_classification(
    matching_result: MatchingResult,
    predicted_marker_states: Mapping[int, bool],
    truth_marker_states: Mapping[int, bool],
    marker_name: str,
) -> MarkerClassificationMetrics:
    """Evaluate binary positivity classification for a specific marker on matched TP nuclei.

    Args:
        matching_result: Immutable MatchingResult containing TP matches.
        predicted_marker_states: Mapping of predicted label ID -> boolean positivity.
        truth_marker_states: Mapping of truth label ID -> boolean positivity.
        marker_name: Name of the marker being evaluated.

    Returns:
        Immutable MarkerClassificationMetrics record.
    """
    if not marker_name or not isinstance(marker_name, str):
        raise InvalidMetricInputError("marker_name must be a non-empty string.")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    tp_matches = [m for m in matching_result.matches if m.match_type == "TP"]

    for m in tp_matches:
        pid = m.pred_id
        tid = m.truth_id
        if pid is None or tid is None:
            continue

        if pid not in predicted_marker_states:
            raise InvalidMetricInputError(
                f"Missing predicted marker state for predicted nucleus ID {pid} on marker '{marker_name}'."
            )
        if tid not in truth_marker_states:
            raise InvalidMetricInputError(
                f"Missing ground-truth marker state for truth nucleus ID {tid} on marker '{marker_name}'."
            )

        p_state = bool(predicted_marker_states[pid])
        t_state = bool(truth_marker_states[tid])

        if t_state and p_state:
            tp += 1
        elif not t_state and p_state:
            fp += 1
        elif not t_state and not p_state:
            tn += 1
        elif t_state and not p_state:
            fn += 1

    matched_count = len(tp_matches)

    # Sensitivity (Recall): TP / (TP + FN)
    if tp + fn > 0:
        sensitivity = float(tp / (tp + fn))
        sensitivity_defined = True
    else:
        sensitivity = None
        sensitivity_defined = False

    # Specificity: TN / (TN + FP)
    if tn + fp > 0:
        specificity = float(tn / (tn + fp))
        specificity_defined = True
    else:
        specificity = None
        specificity_defined = False

    # Precision: TP / (TP + FP)
    if tp + fp > 0:
        precision = float(tp / (tp + fp))
        precision_defined = True
    else:
        precision = None
        precision_defined = False

    # F1 Score: 2TP / (2TP + FP + FN)
    f1_denom = 2 * tp + fp + fn
    if f1_denom > 0:
        f1_score = float(2.0 * tp / f1_denom)
        f1_score_defined = True
    else:
        f1_score = None
        f1_score_defined = False

    # Accuracy: (TP + TN) / Total
    if matched_count > 0:
        accuracy = float((tp + tn) / matched_count)
        accuracy_defined = True
    else:
        accuracy = None
        accuracy_defined = False

    return MarkerClassificationMetrics(
        marker_name=marker_name,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        sensitivity_defined=sensitivity_defined,
        sensitivity=sensitivity,
        specificity_defined=specificity_defined,
        specificity=specificity,
        precision_defined=precision_defined,
        precision=precision,
        f1_score_defined=f1_score_defined,
        f1_score=f1_score,
        accuracy_defined=accuracy_defined,
        accuracy=accuracy,
        matched_object_count=matched_count,
    )


def compute_phenotype_confusion(
    matching_result: MatchingResult,
    predicted_phenotypes: Mapping[int, str],
    truth_phenotypes: Mapping[int, str],
    *,
    category_order: Sequence[str] | None = None,
) -> PhenotypeConfusionMatrix:
    """Compute multi-class confusion matrix for composite phenotype classifications on matched TP nuclei.

    Args:
        matching_result: Immutable MatchingResult containing TP matches.
        predicted_phenotypes: Mapping of predicted label ID -> phenotype string.
        truth_phenotypes: Mapping of truth label ID -> phenotype string.
        category_order: Optional explicit sequence of phenotype category names.

    Returns:
        Immutable PhenotypeConfusionMatrix record.
    """
    tp_matches = [m for m in matching_result.matches if m.match_type == "TP"]

    # Validate presence of states for all TP matches
    for m in tp_matches:
        pid = m.pred_id
        tid = m.truth_id
        if pid is None or tid is None:
            continue
        if pid not in predicted_phenotypes:
            raise InvalidMetricInputError(f"Missing predicted phenotype for predicted nucleus ID {pid}.")
        if tid not in truth_phenotypes:
            raise InvalidMetricInputError(f"Missing ground-truth phenotype for truth nucleus ID {tid}.")

    # Determine deterministic category order
    if category_order is not None:
        if not isinstance(category_order, (tuple, list)):
            raise InvalidMetricInputError("category_order must be a sequence of strings.")
        ordered_phenos = tuple(category_order)
        pheno_set = set(ordered_phenos)
        for m in tp_matches:
            pid = m.pred_id
            tid = m.truth_id
            if pid is not None and predicted_phenotypes[pid] not in pheno_set:
                raise InvalidMetricInputError(
                    f"Predicted phenotype '{predicted_phenotypes[pid]}' for ID {pid} not in category_order."
                )
            if tid is not None and truth_phenotypes[tid] not in pheno_set:
                raise InvalidMetricInputError(
                    f"Truth phenotype '{truth_phenotypes[tid]}' for ID {tid} not in category_order."
                )
    else:
        observed: set[str] = set()
        for m in tp_matches:
            pid = m.pred_id
            tid = m.truth_id
            if pid is not None:
                observed.add(predicted_phenotypes[pid])
            if tid is not None:
                observed.add(truth_phenotypes[tid])
        ordered_phenos = tuple(sorted(observed))

    K = len(ordered_phenos)
    idx_map = {name: i for i, name in enumerate(ordered_phenos)}

    matrix = [[0 for _ in range(K)] for _ in range(K)]
    for m in tp_matches:
        pid = m.pred_id
        tid = m.truth_id
        if pid is not None and tid is not None:
            t_name = truth_phenotypes[tid]
            p_name = predicted_phenotypes[pid]
            matrix[idx_map[t_name]][idx_map[p_name]] += 1

    matched_count = len(tp_matches)

    # Overall accuracy (diagonal sum / total)
    if matched_count > 0:
        correct_count = sum(matrix[i][i] for i in range(K))
        overall_accuracy: float | None = float(correct_count / matched_count)
        overall_accuracy_defined = True
    else:
        overall_accuracy = None
        overall_accuracy_defined = False

    # Macro F1 (one-vs-rest across classes with defined F1)
    class_f1s: list[float] = []
    for k in range(K):
        tp_k = matrix[k][k]
        fp_k = sum(matrix[i][k] for i in range(K) if i != k)
        fn_k = sum(matrix[k][j] for j in range(K) if j != k)
        denom_k = 2 * tp_k + fp_k + fn_k
        if denom_k > 0:
            class_f1s.append(float(2.0 * tp_k / denom_k))

    if class_f1s:
        macro_f1: float | None = float(np.mean(class_f1s))
        macro_f1_defined = True
        contributing_class_count = len(class_f1s)
    else:
        macro_f1 = None
        macro_f1_defined = False
        contributing_class_count = 0

    tuple_matrix = tuple(tuple(row) for row in matrix)

    return PhenotypeConfusionMatrix(
        phenotypes=ordered_phenos,
        matrix=tuple_matrix,
        matched_object_count=matched_count,
        overall_accuracy_defined=overall_accuracy_defined,
        overall_accuracy=overall_accuracy,
        macro_f1_defined=macro_f1_defined,
        macro_f1=macro_f1,
        contributing_class_count=contributing_class_count,
    )


def compare_biological_metrics(
    metric_name: str,
    pred_numerator: int | float,
    pred_denominator: int | float,
    truth_numerator: int | float,
    truth_denominator: int | float,
) -> BiologicalMetricComparison:
    """Compare production-derived biological percentages while preserving raw counts.

    Args:
        metric_name: Identifier of the metric (e.g. 'MarkerA+_percentage').
        pred_numerator: Numerator count from prediction.
        pred_denominator: Denominator count from prediction.
        truth_numerator: Numerator count from ground truth.
        truth_denominator: Denominator count from ground truth.

    Returns:
        Immutable BiologicalMetricComparison record.
    """
    if not metric_name or not isinstance(metric_name, str):
        raise InvalidMetricInputError("metric_name must be a non-empty string.")

    if pred_denominator > 0:
        pred_pct: float | None = float(100.0 * pred_numerator / pred_denominator)
    else:
        pred_pct = None

    if truth_denominator > 0:
        truth_pct: float | None = float(100.0 * truth_numerator / truth_denominator)
    else:
        truth_pct = None

    if pred_pct is not None and truth_pct is not None:
        signed_diff: float | None = float(pred_pct - truth_pct)
        abs_diff: float | None = float(abs(signed_diff))
    else:
        signed_diff = None
        abs_diff = None

    return BiologicalMetricComparison(
        metric_name=metric_name,
        pred_numerator=pred_numerator,
        pred_denominator=pred_denominator,
        pred_percentage=pred_pct,
        truth_numerator=truth_numerator,
        truth_denominator=truth_denominator,
        truth_percentage=truth_pct,
        signed_difference=signed_diff,
        absolute_difference=abs_diff,
    )
