"""Review session state machine for 3D nuclear segmentation QC."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from nervenuc.segmentation3d import Segmentation3DResult, validate_segmentation3d_consistency

from .errors import (
    IncompleteReview3DError,
    InvalidReviewOperation3DError,
    UnknownReviewObject3DError,
)
from .models import ReviewAction3D, ReviewedSegmentation3DResult

REVIEWED_3D_COLUMNS: list[str] = [
    "review_id",
    "current_label",
    "source_nucleus_id",
    "review_status",
    "review_origin",
    "review_note",
    "voxel_count",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "z_min",
    "z_max",
    "y_min",
    "y_max",
    "x_min",
    "x_max",
    "touches_z_border",
    "touches_xy_border",
    "volume_um3",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
]


class ReviewSession3D:
    """Stateful 3D nuclear segmentation QC review session.

    Enforces append-only audit trail and strict completion checking.
    """

    def __init__(self, source_result: Segmentation3DResult) -> None:
        validate_segmentation3d_consistency(source_result, current_source_id=source_result.source_id)
        self._source_result = source_result
        self._history: list[ReviewAction3D] = []
        self._action_counter = 0
        self._notes: dict[int, str] = {}

        # Initialize review DataFrame from source_result.objects
        rows: list[dict] = []
        for _, row in source_result.objects.iterrows():
            oid = int(row["object_id"])
            item = {
                "review_id": f"AUTO-{oid:06d}",
                "current_label": oid,
                "source_nucleus_id": str(oid),
                "review_status": "unreviewed",
                "review_origin": "automated",
                "review_note": "",
                "voxel_count": int(row["voxel_count"]),
                "centroid_z": float(row["centroid_z"]),
                "centroid_y": float(row["centroid_y"]),
                "centroid_x": float(row["centroid_x"]),
                "z_min": int(row["z_min"]),
                "z_max": int(row["z_max"]),
                "y_min": int(row["y_min"]),
                "y_max": int(row["y_max"]),
                "x_min": int(row["x_min"]),
                "x_max": int(row["x_max"]),
                "touches_z_border": bool(row["touches_z_border"]),
                "touches_xy_border": bool(row["touches_xy_border"]),
                "volume_um3": float(row["volume_um3"]) if pd.notna(row["volume_um3"]) else np.nan,
                "centroid_z_um": float(row["centroid_z_um"]) if pd.notna(row["centroid_z_um"]) else np.nan,
                "centroid_y_um": float(row["centroid_y_um"]) if pd.notna(row["centroid_y_um"]) else np.nan,
                "centroid_x_um": float(row["centroid_x_um"]) if pd.notna(row["centroid_x_um"]) else np.nan,
            }
            rows.append(item)

        if rows:
            self._df = pd.DataFrame(rows, columns=REVIEWED_3D_COLUMNS)
        else:
            self._df = pd.DataFrame(columns=REVIEWED_3D_COLUMNS)
        self._is_finalized = False
        self._finalized_result: ReviewedSegmentation3DResult | None = None

    @property
    def source_result(self) -> Segmentation3DResult:
        """Return original source 3D segmentation result."""
        return self._source_result

    @property
    def objects(self) -> pd.DataFrame:
        """Return copy of current review objects table."""
        return self._df.copy()

    @property
    def history(self) -> tuple[ReviewAction3D, ...]:
        """Return tuple of recorded review actions."""
        return tuple(self._history)

    def status(self) -> dict[str, Any]:
        """Return factual summary of review progress."""
        unrev = int(np.sum(self._df["review_status"] == "unreviewed")) if len(self._df) > 0 else 0
        acc = int(np.sum(self._df["review_status"] == "accepted")) if len(self._df) > 0 else 0
        rej = int(np.sum(self._df["review_status"] == "rejected")) if len(self._df) > 0 else 0
        return {
            "total_count": len(self._df),
            "unreviewed_count": unrev,
            "accepted_count": acc,
            "rejected_count": rej,
            "is_complete": unrev == 0,
        }

    def _next_action_id(self) -> str:
        self._action_counter += 1
        return f"ACT-{self._action_counter:06d}"

    def _get_row_index(self, object_id: int) -> int:
        if self._is_finalized:
            raise InvalidReviewOperation3DError("Cannot modify a finalized review session.")
        if len(self._df) == 0:
            raise UnknownReviewObject3DError(f"Object ID {object_id} not found in empty review session.")
        matches = self._df.index[self._df["current_label"] == object_id].tolist()
        if not matches:
            raise UnknownReviewObject3DError(f"Object ID {object_id} not found in review session.")
        return matches[0]

    def accept(self, object_id: int) -> None:
        """Accept an automated 3D nucleus candidate."""
        idx = self._get_row_index(object_id)
        self._df.at[idx, "review_status"] = "accepted"
        action = ReviewAction3D(
            action_id=self._next_action_id(),
            action_type="accept",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_object_ids=(object_id,),
            resulting_object_ids=(object_id,),
        )
        self._history.append(action)

    def reject(
        self,
        object_id: int,
        note: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Reject an automated 3D nucleus candidate."""
        idx = self._get_row_index(object_id)
        self._df.at[idx, "review_status"] = "rejected"
        eff_note = reason if reason is not None else (note if note is not None else "")
        if eff_note:
            self._df.at[idx, "review_note"] = str(eff_note)
            self._notes[object_id] = str(eff_note)
        action = ReviewAction3D(
            action_id=self._next_action_id(),
            action_type="reject",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_object_ids=(object_id,),
            resulting_object_ids=(object_id,),
            note=eff_note,
        )
        self._history.append(action)

    def restore(self, object_id: int) -> None:
        """Restore a previously rejected or accepted object back to unreviewed status."""
        idx = self._get_row_index(object_id)
        self._df.at[idx, "review_status"] = "unreviewed"
        action = ReviewAction3D(
            action_id=self._next_action_id(),
            action_type="restore",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_object_ids=(object_id,),
            resulting_object_ids=(object_id,),
        )
        self._history.append(action)

    def set_note(self, object_id: int, note: str) -> None:
        """Set commentary note for an object."""
        idx = self._get_row_index(object_id)
        self._df.at[idx, "review_note"] = str(note)
        self._notes[object_id] = str(note)
        action = ReviewAction3D(
            action_id=self._next_action_id(),
            action_type="set_note",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_object_ids=(object_id,),
            resulting_object_ids=(object_id,),
            note=str(note),
        )
        self._history.append(action)

    def auto_accept(self) -> None:
        """Accept all unreviewed automated 3D nuclei."""
        for _, row in self._df.iterrows():
            if row["review_status"] == "unreviewed":
                self.accept(int(row["current_label"]))

    def finalize(self) -> ReviewedSegmentation3DResult:
        """Finalize the review session and return an immutable ReviewedSegmentation3DResult.

        Raises:
            IncompleteReview3DError: If any objects remain in 'unreviewed' state.
        """
        if self._is_finalized and self._finalized_result is not None:
            return self._finalized_result

        if len(self._df) > 0:
            unreviewed = self._df[self._df["review_status"] == "unreviewed"]
            if len(unreviewed) > 0:
                unrev_ids = unreviewed["current_label"].tolist()
                raise IncompleteReview3DError(
                    f"Cannot finalize 3D review session: {len(unrev_ids)} object(s) remain unreviewed: {unrev_ids}."
                )

        active_df = self._df[self._df["review_status"] == "accepted"].copy()
        res = ReviewedSegmentation3DResult(
            active_objects=active_df,
            source_result=self._source_result,
            history=tuple(self._history),
            notes=dict(self._notes),
        )
        self._is_finalized = True
        self._finalized_result = res
        return res



def start_review_3d(result: Segmentation3DResult) -> ReviewSession3D:
    """Initialize a stateful 3D manual review session for a Segmentation3DResult."""
    return ReviewSession3D(result)
