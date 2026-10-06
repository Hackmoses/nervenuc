"""Controller coordinating segmentation, QC review, measurements, thresholding, and metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nervenuc.classification import (
    Classification3DResult,
    ClassificationResult,
    MarkerThreshold,
    classify_markers,
    classify_markers_3d,
)
from nervenuc.qc import ReviewedSegmentationResult
from nervenuc.qc3d import ReviewedSegmentation3DResult
from nervenuc.summary import BiologicalSummary3DResult, BiologicalSummaryResult

from ..errors import InvalidDraftConfigError, NerveNucGuiError
from ..state import BiologicalMetricDraft, GuiSessionState, MarkerThresholdDraft


class AnalysisController:
    """Dispatches user actions to underlying workflow and review engines."""

    def __init__(self, state: GuiSessionState) -> None:
        self.state = state

    def run_segmentation(self) -> None:
        """Execute core segmentation and initialize review session."""
        mode = self.state.draft.mode
        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is None:
                raise NerveNucGuiError("2D Analysis session is not initialized.")
            cfg_2d = self.state.draft.build_core_2d_config()
            sess.segment(cfg_2d.segmentation)
            sess.start_review()
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is None:
                raise NerveNucGuiError("3D Analysis session is not initialized.")
            vol_cfg = self.state.draft.build_core_volume_config()
            sess_3d.segment(vol_cfg.segmentation)
            sess_3d.start_review()
        self.state.results_stale = False
        self.state.configuration_differs_from_run = False
        self.state.notify()

    # --- 2D QC Review Operations ---

    def accept_2d(self, review_id: str, note: str | None = None) -> None:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        sess.review_session.accept(review_id, note=note)
        self.state.notify()

    def reject_2d(self, review_id: str, note: str | None = None) -> None:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        sess.review_session.reject(review_id, note=note)
        self.state.notify()

    def restore_2d(self, review_id: str, note: str | None = None) -> None:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        sess.review_session.restore(review_id, note=note)
        self.state.notify()

    def add_manual_2d(self, mask: np.ndarray, note: str | None = None) -> str:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        new_id = sess.review_session.add(mask, note=note)
        self.state.notify()
        return new_id

    def delete_2d(self, review_id: str, note: str | None = None) -> None:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        sess.review_session.delete(review_id, note=note)
        self.state.notify()

    def merge_2d(self, review_ids: Sequence[str], note: str | None = None) -> str:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        new_id = sess.review_session.merge(review_ids, note=note)
        self.state.notify()
        return new_id

    def split_2d(self, review_id: str, replacement_masks: Sequence[np.ndarray], note: str | None = None) -> tuple[str, ...]:
        sess = self.state.session_2d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 2D review session.")
        child_ids = sess.review_session.split(review_id, replacement_masks, note=note)
        self.state.notify()
        return child_ids

    # --- 3D QC Review Operations ---

    def accept_3d(self, object_id: int) -> None:
        sess = self.state.session_3d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 3D review session.")
        sess.review_session.accept(object_id)
        self.state.notify()

    def reject_3d(self, object_id: int, note: str | None = None) -> None:
        sess = self.state.session_3d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 3D review session.")
        sess.review_session.reject(object_id, note=note)
        self.state.notify()

    def restore_3d(self, object_id: int) -> None:
        sess = self.state.session_3d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 3D review session.")
        sess.review_session.restore(object_id)
        self.state.notify()

    def set_note_3d(self, object_id: int, note: str) -> None:
        sess = self.state.session_3d
        if sess is None or sess.review_session is None:
            raise NerveNucGuiError("No active 3D review session.")
        sess.review_session.set_note(object_id, note)
        self.state.notify()

    # --- Common QC Operations ---

    def auto_accept(self) -> None:
        mode = self.state.draft.mode
        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is not None and sess.review_session is not None:
                sess.review_session.auto_accept()
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is not None and sess_3d.review_session is not None:
                sess_3d.review_session.auto_accept()
        self.state.notify()

    def finalize_review(self) -> None:
        mode = self.state.draft.mode
        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is None or sess.review_session is None:
                raise NerveNucGuiError("No active 2D review session to finalize.")
            sess.finalize_review()
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.review_session is None:
                raise NerveNucGuiError("No active 3D review session to finalize.")
            sess_3d.finalize_review()
        self.state.notify()

    # --- Measurements & Thresholding ---

    def measure_markers(self) -> None:
        mode = self.state.draft.mode
        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is None or sess.reviewed_result is None:
                raise NerveNucGuiError("Cannot measure: review has not been finalized.")
            sess.measure()
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.reviewed_result is None:
                raise NerveNucGuiError("Cannot measure: 3D review has not been finalized.")
            sess_3d.measure()
        self.state.notify()

    def preview_thresholds(
        self,
        draft_thresholds: Sequence[MarkerThresholdDraft],
    ) -> ClassificationResult | Classification3DResult:
        """Run transient classification preview using draft thresholds without mutating session."""
        mode = self.state.draft.mode
        core_thresholds = tuple(d.to_core_threshold() for d in draft_thresholds)

        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is None or sess.measurement_result is None:
                raise NerveNucGuiError("Cannot preview thresholds: measurements not available.")
            return classify_markers(
                measurement_result=sess.measurement_result,
                thresholds=core_thresholds,
            )
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.measurement_result is None:
                raise NerveNucGuiError("Cannot preview thresholds: 3D measurements not available.")
            return classify_markers_3d(
                measurement_result=sess_3d.measurement_result,
                thresholds=core_thresholds,
            )
        raise NerveNucGuiError(f"Unsupported mode: {mode}")

    def apply_thresholds_and_summarize(self) -> BiologicalSummaryResult | BiologicalSummary3DResult:
        """Commit draft thresholds and metrics to active session and evaluate summaries."""
        mode = self.state.draft.mode
        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess is None or sess.measurement_result is None:
                raise NerveNucGuiError("Cannot apply thresholds: measurements not available.")
            cfg_2d = self.state.draft.build_core_2d_config()
            sess.classify(cfg_2d.thresholds)
            res = sess.summarize(cfg_2d.metrics)
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d is None or sess_3d.measurement_result is None:
                raise NerveNucGuiError("Cannot apply thresholds: 3D measurements not available.")
            vol_cfg = self.state.draft.build_core_volume_config()
            sess_3d.classify(vol_cfg.thresholds)
            res = sess_3d.summarize(vol_cfg.metrics)
        self.state.results_stale = False
        self.state.configuration_differs_from_run = False
        self.state.notify()
        return res

    def add_metric_draft(self, draft: BiologicalMetricDraft) -> None:
        """Validate and add a biological metric draft."""
        # Validate draft against core validator
        _ = draft.to_core_metric()
        self.state.draft.metric_drafts.append(draft)
        self.state.mark_draft_modified()
