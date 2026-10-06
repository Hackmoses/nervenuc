"""Comprehensive tests for marker classification, phenotype confusion, and biological metric comparisons."""

import numpy as np
import pytest

from nervenuc.validation.errors import InvalidMetricInputError
from nervenuc.validation.matching import match_instances
from nervenuc.validation.metrics import (
    compare_biological_metrics,
    compute_phenotype_confusion,
    evaluate_marker_classification,
)


def _make_dummy_matching_4tp() -> tuple[np.ndarray, np.ndarray, Any]:
    """Helper to create a matching result with 4 distinct TP matches."""
    pred = np.zeros((40, 40), dtype=np.int32)
    truth = np.zeros((40, 40), dtype=np.int32)

    pred[0:5, 0:5] = 1
    pred[10:15, 10:15] = 2
    pred[20:25, 20:25] = 3
    pred[30:35, 30:35] = 4

    truth[0:5, 0:5] = 10
    truth[10:15, 10:15] = 20
    truth[20:25, 20:25] = 30
    truth[30:35, 30:35] = 40

    matching = match_instances(pred, truth, iou_threshold=0.5)
    return pred, truth, matching


def test_perfect_marker_classification() -> None:
    """Verify perfect marker classification gives 1.0 for sensitivity, specificity, precision, F1, accuracy."""
    _, _, matching = _make_dummy_matching_4tp()

    # Truth: 10 (+), 20 (+), 30 (-), 40 (-)
    truth_states = {10: True, 20: True, 30: False, 40: False}
    # Pred: 1 (+), 2 (+), 3 (-), 4 (-)
    pred_states = {1: True, 2: True, 3: False, 4: False}

    res = evaluate_marker_classification(matching, pred_states, truth_states, marker_name="MarkerA")

    assert res.marker_name == "MarkerA"
    assert res.matched_object_count == 4
    assert res.tp == 2
    assert res.fp == 0
    assert res.tn == 2
    assert res.fn == 0

    assert res.sensitivity_defined is True
    assert res.sensitivity == 1.0
    assert res.specificity_defined is True
    assert res.specificity == 1.0
    assert res.precision_defined is True
    assert res.precision == 1.0
    assert res.f1_score_defined is True
    assert res.f1_score == 1.0
    assert res.accuracy_defined is True
    assert res.accuracy == 1.0


def test_all_four_confusion_cells_present() -> None:
    """Verify 1 TP, 1 FP, 1 TN, 1 FN yields exact 0.5 across all metrics."""
    _, _, matching = _make_dummy_matching_4tp()

    # Match 1: Truth 10 (+) & Pred 1 (+) -> TP
    # Match 2: Truth 20 (-) & Pred 2 (+) -> FP
    # Match 3: Truth 30 (-) & Pred 3 (-) -> TN
    # Match 4: Truth 40 (+) & Pred 4 (-) -> FN
    truth_states = {10: True, 20: False, 30: False, 40: True}
    pred_states = {1: True, 2: True, 3: False, 4: False}

    res = evaluate_marker_classification(matching, pred_states, truth_states, marker_name="MarkerB")

    assert res.tp == 1
    assert res.fp == 1
    assert res.tn == 1
    assert res.fn == 1

    assert res.sensitivity == 0.5
    assert res.specificity == 0.5
    assert res.precision == 0.5
    assert res.f1_score == 0.5
    assert res.accuracy == 0.5


def test_zero_denominator_marker_metrics() -> None:
    """Verify undefined conditions for sensitivity, specificity, precision, accuracy."""
    _, _, matching = _make_dummy_matching_4tp()

    # 1. No truth positives (all truth negative) -> sensitivity undefined
    truth_all_neg = {10: False, 20: False, 30: False, 40: False}
    pred_all_neg = {1: False, 2: False, 3: False, 4: False}
    res_no_tp = evaluate_marker_classification(matching, pred_all_neg, truth_all_neg, marker_name="MarkerX")

    assert res_no_tp.tp == 0
    assert res_no_tp.fn == 0
    assert res_no_tp.sensitivity_defined is False
    assert res_no_tp.sensitivity is None
    assert res_no_tp.precision_defined is False
    assert res_no_tp.precision is None
    assert res_no_tp.specificity == 1.0

    # 2. No truth negatives (all truth positive) -> specificity undefined
    truth_all_pos = {10: True, 20: True, 30: True, 40: True}
    pred_all_pos = {1: True, 2: True, 3: True, 4: True}
    res_no_tn = evaluate_marker_classification(matching, pred_all_pos, truth_all_pos, marker_name="MarkerY")

    assert res_no_tn.tn == 0
    assert res_no_tn.fp == 0
    assert res_no_tn.specificity_defined is False
    assert res_no_tn.specificity is None
    assert res_no_tn.sensitivity == 1.0


def test_missing_marker_state_raises_error() -> None:
    """Verify missing predicted or truth object ID in mapping raises InvalidMetricInputError."""
    _, _, matching = _make_dummy_matching_4tp()

    incomplete_pred = {1: True, 2: False}  # Missing 3 and 4
    truth_states = {10: True, 20: True, 30: False, 40: False}

    with pytest.raises(InvalidMetricInputError, match="Missing predicted marker state"):
        evaluate_marker_classification(matching, incomplete_pred, truth_states, marker_name="MarkerA")


def test_phenotype_confusion_matrix_perfect() -> None:
    """Verify multiclass phenotype confusion matrix for perfect agreement."""
    _, _, matching = _make_dummy_matching_4tp()

    truth_phenos = {10: "MarkerA+", 20: "MarkerA+/MarkerB+", 30: "Negative", 40: "MarkerB+"}
    pred_phenos = {1: "MarkerA+", 2: "MarkerA+/MarkerB+", 3: "Negative", 4: "MarkerB+"}

    res = compute_phenotype_confusion(matching, pred_phenos, truth_phenos)

    assert res.matched_object_count == 4
    assert res.overall_accuracy_defined is True
    assert res.overall_accuracy == 1.0
    assert res.macro_f1_defined is True
    assert res.macro_f1 == 1.0
    assert res.contributing_class_count == 4
    assert res.phenotypes == ("MarkerA+", "MarkerA+/MarkerB+", "MarkerB+", "Negative")

    # Matrix diagonal
    matrix = np.array(res.matrix)
    assert np.array_equal(np.diag(matrix), [1, 1, 1, 1])
    assert np.sum(matrix) == 4


def test_phenotype_confusion_off_diagonal_and_explicit_order() -> None:
    """Verify off-diagonal misclassification and explicit category order."""
    _, _, matching = _make_dummy_matching_4tp()

    # Truth: 10: "ClassA", 20: "ClassA", 30: "ClassB", 40: "ClassB"
    # Pred:  1: "ClassA", 2: "ClassB" (misclassified!), 3: "ClassB", 4: "ClassB"
    truth_phenos = {10: "ClassA", 20: "ClassA", 30: "ClassB", 40: "ClassB"}
    pred_phenos = {1: "ClassA", 2: "ClassB", 3: "ClassB", 4: "ClassB"}

    category_order = ["ClassA", "ClassB", "ClassC"]
    res = compute_phenotype_confusion(matching, pred_phenos, truth_phenos, category_order=category_order)

    assert res.phenotypes == ("ClassA", "ClassB", "ClassC")
    matrix = np.array(res.matrix)
    # ClassA truth: 1 in ClassA pred, 1 in ClassB pred
    assert np.array_equal(matrix[0], [1, 1, 0])
    # ClassB truth: 0 in ClassA pred, 2 in ClassB pred
    assert np.array_equal(matrix[1], [0, 2, 0])
    # ClassC truth: 0
    assert np.array_equal(matrix[2], [0, 0, 0])

    assert pytest.approx(res.overall_accuracy, rel=1e-6) == 3.0 / 4.0
    # ClassA F1: TP=1, FP=0, FN=1 -> 2*1/(2*1 + 0 + 1) = 2/3
    # ClassB F1: TP=2, FP=1, FN=0 -> 2*2/(2*2 + 1 + 0) = 4/5
    # ClassC F1: TP=0, FP=0, FN=0 -> Denominator 0 (excluded from macro F1)
    # Macro F1 = (2/3 + 4/5) / 2 = (10/15 + 12/15) / 2 = 22/30 = 11/15 = 0.7333333
    assert res.contributing_class_count == 2
    assert pytest.approx(res.macro_f1, rel=1e-6) == 11.0 / 15.0


def test_compare_biological_metrics() -> None:
    """Verify biological metric comparison preserving raw counts and signed differences."""
    # Predicted: 15 / 100 = 15.0%
    # Truth: 10 / 100 = 10.0%
    res = compare_biological_metrics(
        metric_name="MarkerA+_percentage",
        pred_numerator=15,
        pred_denominator=100,
        truth_numerator=10,
        truth_denominator=100,
    )

    assert res.metric_name == "MarkerA+_percentage"
    assert res.pred_percentage == 15.0
    assert res.truth_percentage == 10.0
    assert pytest.approx(res.signed_difference, rel=1e-6) == 5.0
    assert pytest.approx(res.absolute_difference, rel=1e-6) == 5.0

    # Zero denominator case
    res_zero = compare_biological_metrics(
        metric_name="Empty_percentage",
        pred_numerator=0,
        pred_denominator=0,
        truth_numerator=5,
        truth_denominator=50,
    )
    assert res_zero.pred_percentage is None
    assert res_zero.truth_percentage == 10.0
    assert res_zero.signed_difference is None
    assert res_zero.absolute_difference is None
