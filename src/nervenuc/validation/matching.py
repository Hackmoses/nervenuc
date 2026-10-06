"""Deterministic 2D and 3D instance matching between predicted and ground-truth nuclei."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from nervenuc.preparation.models import VoxelSpacing

from .errors import InvalidMatchingInputError, InvalidThresholdError
from .models import MatchingResult, ObjectMatch


def _extract_spacing_tuple(
    spacing: VoxelSpacing | tuple[float, ...] | None,
    ndim: int,
) -> tuple[float, ...] | None:
    """Extract and validate physical spacing tuple matching dimensionality."""
    if spacing is None:
        return None
    if isinstance(spacing, VoxelSpacing):
        if ndim == 3:
            return (spacing.z, spacing.y, spacing.x)
        elif ndim == 2:
            return (spacing.y, spacing.x)
        return None
    if isinstance(spacing, (tuple, list)):
        if len(spacing) != ndim:
            raise InvalidMatchingInputError(
                f"spacing length ({len(spacing)}) must match label dimensionality ({ndim})."
            )
        vals = tuple(float(s) for s in spacing)
        if any(v <= 0 or not np.isfinite(v) for v in vals):
            raise InvalidMatchingInputError(f"spacing values must be positive and finite, got {vals}.")
        return vals
    raise InvalidMatchingInputError(f"Unsupported spacing type: {type(spacing).__name__}.")


def match_instances(
    predicted_labels: np.ndarray,
    truth_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    spacing: VoxelSpacing | tuple[float, ...] | None = None,
    pred_review_ids: Mapping[int, str] | None = None,
    truth_names: Mapping[int, str] | None = None,
    case_id: str = "case",
) -> MatchingResult:
    """Perform deterministic one-to-one bipartite matching between predicted and ground-truth nuclei.

    Scientific Objective:
        1. Tier 1 (Primary): Maximize the number of matched pairs satisfying IoU >= iou_threshold.
        2. Tier 2 (Secondary): Among maximum-cardinality matchings, maximize the sum of IoU.
        3. Tier 3 (Tertiary): Resolve mathematical ties deterministically via sorted label indexing.

    Args:
        predicted_labels: 2D (YX) or 3D (ZYX) integer label array of predicted nuclei.
        truth_labels: 2D (YX) or 3D (ZYX) integer label array of ground-truth nuclei.
        iou_threshold: Minimum Intersection-over-Union cutoff for a valid TP match (0.0 < tau <= 1.0).
        spacing: Optional physical voxel spacing (VoxelSpacing or tuple of floats).
        pred_review_ids: Optional mapping from integer predicted ID to review ID string (e.g. 1 -> "AUTO-000001").
        truth_names: Optional mapping from integer truth ID to truth name string (e.g. 1 -> "NUC-001").
        case_id: Identifier of the validation case.

    Returns:
        Immutable MatchingResult containing all TP, FP, and FN ObjectMatch records.
    """
    # 1. Type and ndim validation
    if not isinstance(predicted_labels, np.ndarray) or not isinstance(truth_labels, np.ndarray):
        raise InvalidMatchingInputError("predicted_labels and truth_labels must be NumPy arrays.")

    if not np.issubdtype(predicted_labels.dtype, np.integer):
        raise InvalidMatchingInputError(
            f"predicted_labels must have integer dtype, got {predicted_labels.dtype}."
        )
    if not np.issubdtype(truth_labels.dtype, np.integer):
        raise InvalidMatchingInputError(
            f"truth_labels must have integer dtype, got {truth_labels.dtype}."
        )

    if predicted_labels.ndim not in (2, 3):
        raise InvalidMatchingInputError(
            f"predicted_labels must be 2D (YX) or 3D (ZYX), got ndim={predicted_labels.ndim}."
        )
    if truth_labels.ndim not in (2, 3):
        raise InvalidMatchingInputError(
            f"truth_labels must be 2D (YX) or 3D (ZYX), got ndim={truth_labels.ndim}."
        )

    if predicted_labels.shape != truth_labels.shape:
        raise InvalidMatchingInputError(
            f"Shape mismatch: predicted_labels {predicted_labels.shape} != truth_labels {truth_labels.shape}."
        )

    if np.any(predicted_labels < 0) or np.any(truth_labels < 0):
        raise InvalidMatchingInputError("Label arrays cannot contain negative values.")

    # 2. Threshold validation (0 < tau <= 1)
    if not isinstance(iou_threshold, (int, float)) or not np.isfinite(iou_threshold):
        raise InvalidThresholdError(f"iou_threshold must be a finite float, got {iou_threshold}.")
    if iou_threshold <= 0.0 or iou_threshold > 1.0:
        raise InvalidThresholdError(
            f"iou_threshold must be strictly greater than 0.0 and <= 1.0, got {iou_threshold}."
        )

    ndim = predicted_labels.ndim
    spacing_tuple = _extract_spacing_tuple(spacing, ndim)

    # 3. Extract unique positive IDs and compute geometries
    pred_ids_raw = np.unique(predicted_labels[predicted_labels > 0])
    truth_ids_raw = np.unique(truth_labels[truth_labels > 0])

    # Convert to standard Python int sorted deterministically
    pred_ids = [int(x) for x in np.sort(pred_ids_raw)]
    truth_ids = [int(x) for x in np.sort(truth_ids_raw)]

    M = len(pred_ids)
    N = len(truth_ids)

    pred_sizes: dict[int, int] = {}
    pred_centroids: dict[int, tuple[float, ...]] = {}
    for pid in pred_ids:
        coords = np.nonzero(predicted_labels == pid)
        pred_sizes[pid] = int(len(coords[0]))
        pred_centroids[pid] = tuple(float(np.mean(c)) for c in coords)

    truth_sizes: dict[int, int] = {}
    truth_centroids: dict[int, tuple[float, ...]] = {}
    for tid in truth_ids:
        coords = np.nonzero(truth_labels == tid)
        truth_sizes[tid] = int(len(coords[0]))
        truth_centroids[tid] = tuple(float(np.mean(c)) for c in coords)

    # 4. Handle boundary cases where M == 0 or N == 0
    if M == 0 and N == 0:
        return MatchingResult(
            case_id=case_id,
            iou_threshold=float(iou_threshold),
            matches=(),
            tp_count=0,
            fp_count=0,
            fn_count=0,
            pred_count=0,
            truth_count=0,
        )

    if M == 0 and N > 0:
        fn_matches: list[ObjectMatch] = []
        for tid in truth_ids:
            fn_matches.append(
                ObjectMatch(
                    pred_id=None,
                    pred_review_id=None,
                    truth_id=tid,
                    truth_name=truth_names.get(tid) if truth_names else None,
                    iou=0.0,
                    dice=0.0,
                    match_type="FN",
                    pred_size=None,
                    truth_size=truth_sizes[tid],
                    signed_size_diff=None,
                    abs_size_diff=None,
                    size_ratio=None,
                    centroid_distance_px=None,
                    centroid_distance_um=None,
                )
            )
        return MatchingResult(
            case_id=case_id,
            iou_threshold=float(iou_threshold),
            matches=tuple(fn_matches),
            tp_count=0,
            fp_count=0,
            fn_count=N,
            pred_count=0,
            truth_count=N,
        )

    if M > 0 and N == 0:
        fp_matches: list[ObjectMatch] = []
        for pid in pred_ids:
            fp_matches.append(
                ObjectMatch(
                    pred_id=pid,
                    pred_review_id=pred_review_ids.get(pid) if pred_review_ids else None,
                    truth_id=None,
                    truth_name=None,
                    iou=0.0,
                    dice=0.0,
                    match_type="FP",
                    pred_size=pred_sizes[pid],
                    truth_size=None,
                    signed_size_diff=None,
                    abs_size_diff=None,
                    size_ratio=None,
                    centroid_distance_px=None,
                    centroid_distance_um=None,
                )
            )
        return MatchingResult(
            case_id=case_id,
            iou_threshold=float(iou_threshold),
            matches=tuple(fp_matches),
            tp_count=0,
            fp_count=M,
            fn_count=0,
            pred_count=M,
            truth_count=0,
        )

    # 5. Compute pairwise overlap contingency table
    pred_idx_map = {pid: i for i, pid in enumerate(pred_ids)}
    truth_idx_map = {tid: j for j, tid in enumerate(truth_ids)}

    intersections = np.zeros((M, N), dtype=np.int64)

    flat_p = predicted_labels.ravel()
    flat_t = truth_labels.ravel()
    fg_mask = (flat_p > 0) & (flat_t > 0)

    if np.any(fg_mask):
        fg_p = flat_p[fg_mask]
        fg_t = flat_t[fg_mask]

        # Fast contingency counting via 2D bincount
        p_indices = np.array([pred_idx_map[p] for p in fg_p], dtype=np.int64)
        t_indices = np.array([truth_idx_map[t] for t in fg_t], dtype=np.int64)
        flat_pairs = p_indices * N + t_indices
        pair_counts = np.bincount(flat_pairs, minlength=M * N)
        intersections = pair_counts.reshape((M, N))

    # Compute exact pairwise IoU matrix
    iou_matrix = np.zeros((M, N), dtype=np.float64)
    for i, pid in enumerate(pred_ids):
        p_sz = pred_sizes[pid]
        for j, tid in enumerate(truth_ids):
            inter = int(intersections[i, j])
            if inter > 0:
                union = p_sz + truth_sizes[tid] - inter
                iou_matrix[i, j] = float(inter) / float(union)

    # 6. Bipartite Matching with Cardinality Priority Guarantee
    # Mathematical proof:
    # Let K = min(M, N). The maximum possible sum of IoU across any matching is <= K * 1.0 = K.
    # We assign bonus B = K + 1.0 to any edge with IoU >= iou_threshold.
    # Therefore, any matching with (k + 1) valid edges has total weight >= (k + 1) * B = k*B + K + 1.
    # Any matching with k valid edges has total weight <= k*B + k*1.0 = k*B + k <= k*B + K.
    # Since k*B + K + 1 > k*B + K, any assignment with (k+1) valid edges is strictly prioritized over
    # ANY assignment with k valid edges (Tier 1).
    # Furthermore, among assignments with identical number k of valid edges, total weight equals
    # k*B + sum(IoU), which directly maximizes total IoU (Tier 2).
    K = min(M, N)
    bonus = float(K + 1.0)

    weight_matrix = np.zeros((M, N), dtype=np.float64)
    valid_mask = iou_matrix >= iou_threshold
    weight_matrix[valid_mask] = bonus + iou_matrix[valid_mask]

    # Convert maximum weight bipartite matching to linear sum assignment cost matrix
    max_w = float(np.max(weight_matrix)) if np.any(valid_mask) else 0.0
    cost_matrix = (max_w + 1.0) - weight_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 7. Classify TP, FP, FN matches
    matched_preds: set[int] = set()
    matched_truths: set[int] = set()
    tp_matches: list[ObjectMatch] = []

    for r, c in zip(row_ind, col_ind):
        if valid_mask[r, c]:
            pid = pred_ids[r]
            tid = truth_ids[c]
            matched_preds.add(pid)
            matched_truths.add(tid)

            inter = int(intersections[r, c])
            p_sz = pred_sizes[pid]
            t_sz = truth_sizes[tid]
            iou_val = float(iou_matrix[r, c])
            dice_val = float(2.0 * inter) / float(p_sz + t_sz)

            signed_diff = int(p_sz - t_sz)
            abs_diff = int(abs(signed_diff))
            ratio = float(p_sz) / float(t_sz) if t_sz > 0 else None

            # Centroid distance
            c_p = pred_centroids[pid]
            c_t = truth_centroids[tid]
            d_px = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c_p, c_t))))
            d_um: float | None = None
            if spacing_tuple is not None:
                d_um = float(
                    np.sqrt(sum(((a - b) * s) ** 2 for a, b, s in zip(c_p, c_t, spacing_tuple)))
                )

            tp_matches.append(
                ObjectMatch(
                    pred_id=pid,
                    pred_review_id=pred_review_ids.get(pid) if pred_review_ids else None,
                    truth_id=tid,
                    truth_name=truth_names.get(tid) if truth_names else None,
                    iou=iou_val,
                    dice=dice_val,
                    match_type="TP",
                    pred_size=p_sz,
                    truth_size=t_sz,
                    signed_size_diff=signed_diff,
                    abs_size_diff=abs_diff,
                    size_ratio=ratio,
                    centroid_distance_px=d_px,
                    centroid_distance_um=d_um,
                )
            )

    # Sort TP matches deterministically by truth_id
    tp_matches.sort(key=lambda m: (m.truth_id if m.truth_id is not None else 0, m.pred_id if m.pred_id is not None else 0))

    # FN matches (truth objects without valid match)
    fn_matches: list[ObjectMatch] = []
    for tid in truth_ids:
        if tid not in matched_truths:
            fn_matches.append(
                ObjectMatch(
                    pred_id=None,
                    pred_review_id=None,
                    truth_id=tid,
                    truth_name=truth_names.get(tid) if truth_names else None,
                    iou=0.0,
                    dice=0.0,
                    match_type="FN",
                    pred_size=None,
                    truth_size=truth_sizes[tid],
                    signed_size_diff=None,
                    abs_size_diff=None,
                    size_ratio=None,
                    centroid_distance_px=None,
                    centroid_distance_um=None,
                )
            )
    fn_matches.sort(key=lambda m: m.truth_id if m.truth_id is not None else 0)

    # FP matches (predicted objects without valid match)
    fp_matches: list[ObjectMatch] = []
    for pid in pred_ids:
        if pid not in matched_preds:
            fp_matches.append(
                ObjectMatch(
                    pred_id=pid,
                    pred_review_id=pred_review_ids.get(pid) if pred_review_ids else None,
                    truth_id=None,
                    truth_name=None,
                    iou=0.0,
                    dice=0.0,
                    match_type="FP",
                    pred_size=pred_sizes[pid],
                    truth_size=None,
                    signed_size_diff=None,
                    abs_size_diff=None,
                    size_ratio=None,
                    centroid_distance_px=None,
                    centroid_distance_um=None,
                )
            )
    fp_matches.sort(key=lambda m: m.pred_id if m.pred_id is not None else 0)

    all_matches = tuple(tp_matches + fn_matches + fp_matches)

    return MatchingResult(
        case_id=case_id,
        iou_threshold=float(iou_threshold),
        matches=all_matches,
        tp_count=len(tp_matches),
        fp_count=len(fp_matches),
        fn_count=len(fn_matches),
        pred_count=M,
        truth_count=N,
    )
