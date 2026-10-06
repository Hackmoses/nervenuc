"""Review session state machine and operations for nuclear segmentation QC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from nervenuc.segmentation import SegmentationResult
from .errors import (
    IncompleteReviewError,
    InvalidReviewMaskError,
    InvalidReviewOperationError,
    UnknownReviewObjectError,
)
from .models import ReviewAction, ReviewedSegmentationResult

REVIEWED_OBJECT_COLUMNS: list[str] = [
    "review_id",
    "current_label",
    "source_nucleus_id",
    "review_status",
    "review_origin",
    "parent_review_ids",
    "review_note",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "touches_border",
    "qc_status",
]

RESOLVED_REVIEW_STATUSES = frozenset({"accepted", "edited"})


def validate_review_completion(reviewed_result: ReviewedSegmentationResult) -> None:
    """Require every active reviewed nucleus to have an explicit resolved state.

    Interim snapshots may contain ``unreviewed`` objects, but they are not valid
    inputs to downstream scientific measurement.
    """
    if not isinstance(reviewed_result, ReviewedSegmentationResult):
        raise IncompleteReviewError(
            "reviewed_result must be a ReviewedSegmentationResult."
        )
    if not isinstance(reviewed_result.objects, pd.DataFrame):
        raise IncompleteReviewError("reviewed_result.objects must be a DataFrame.")
    required = {"review_id", "review_status"}
    missing = required - set(reviewed_result.objects.columns)
    if missing:
        raise IncompleteReviewError(
            f"reviewed_result.objects is missing review completion columns: {sorted(missing)}."
        )

    unresolved = reviewed_result.objects[
        ~reviewed_result.objects["review_status"].isin(RESOLVED_REVIEW_STATUSES)
    ]
    if len(unresolved) > 0:
        details = ", ".join(
            f"{row.review_id} ({row.review_status})"
            for row in unresolved[["review_id", "review_status"]].itertuples(index=False)
        )
        raise IncompleteReviewError(
            f"Review is incomplete: {len(unresolved)} active object(s) remain unresolved: {details}."
        )


@dataclass
class _ReviewObjectRecord:
    """Internal representation of an active reviewed nucleus."""

    review_id: str
    mask: np.ndarray
    source_nucleus_id: int | None
    review_status: str  # "unreviewed", "accepted", "edited"
    review_origin: str  # "automated", "manual_add", "merge", "split"
    parent_review_ids: tuple[str, ...]
    review_note: str | None = None


@dataclass
class _RetiredObjectRecord:
    """Internal representation of a retired reviewed nucleus."""

    review_id: str
    mask: np.ndarray
    source_nucleus_id: int | None
    review_origin: str
    parent_review_ids: tuple[str, ...]
    review_note: str | None
    retirement_reason: str  # "rejected", "deleted", "merged", "split"


def _empty_reviewed_objects_dataframe() -> pd.DataFrame:
    """Return an empty reviewed objects DataFrame with required schema and dtypes."""
    return pd.DataFrame(
        {
            "review_id": pd.Series(dtype="object"),
            "current_label": pd.Series(dtype="int64"),
            "source_nucleus_id": pd.Series(dtype="object"),
            "review_status": pd.Series(dtype="object"),
            "review_origin": pd.Series(dtype="object"),
            "parent_review_ids": pd.Series(dtype="object"),
            "review_note": pd.Series(dtype="object"),
            "area_px": pd.Series(dtype="int64"),
            "centroid_y": pd.Series(dtype="float64"),
            "centroid_x": pd.Series(dtype="float64"),
            "bbox_min_y": pd.Series(dtype="int64"),
            "bbox_min_x": pd.Series(dtype="int64"),
            "bbox_max_y": pd.Series(dtype="int64"),
            "bbox_max_x": pd.Series(dtype="int64"),
            "touches_border": pd.Series(dtype="bool"),
            "qc_status": pd.Series(dtype="object"),
        },
        columns=REVIEWED_OBJECT_COLUMNS,
    )


def _validate_mask(mask: np.ndarray, expected_shape: tuple[int, int], name: str = "Mask") -> np.ndarray:
    """Validate a 2D binary mask and return a boolean NumPy array.

    Raises:
        InvalidReviewMaskError: If the mask fails validation.
    """
    if not isinstance(mask, np.ndarray):
        raise InvalidReviewMaskError(
            f"{name} must be a numpy.ndarray, got {type(mask).__name__}."
        )

    if mask.ndim != 2:
        raise InvalidReviewMaskError(
            f"{name} must be 2-dimensional (Y, X), got ndim={mask.ndim} with shape {mask.shape}."
        )

    if mask.shape != expected_shape:
        raise InvalidReviewMaskError(
            f"{name} shape {mask.shape} does not match image shape {expected_shape}."
        )

    if mask.dtype == bool:
        bool_mask = mask.copy()
    elif np.issubdtype(mask.dtype, np.number):
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise InvalidReviewMaskError(
                f"{name} must be boolean or binary (0/1). Found values: {unique_vals}."
            )
        bool_mask = (mask == 1)
    else:
        raise InvalidReviewMaskError(
            f"{name} has invalid dtype '{mask.dtype}'. Must be boolean or binary numeric."
        )

    if not np.any(bool_mask):
        raise InvalidReviewMaskError(f"{name} must contain at least one foreground pixel.")

    return bool_mask


class ReviewSession:
    """Stateful interactive review session for nuclear segmentation QC."""

    def __init__(
        self,
        source_result: SegmentationResult,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """Initialize a ReviewSession from an immutable SegmentationResult."""
        if not isinstance(source_result, SegmentationResult):
            raise InvalidReviewOperationError(
                f"source_result must be a SegmentationResult, got {type(source_result).__name__}."
            )

        self._source_result = source_result
        self._shape = source_result.labels.shape
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc).isoformat())

        self._active_objects: dict[str, _ReviewObjectRecord] = {}
        self._retired_objects: dict[str, _RetiredObjectRecord] = {}
        self._history: list[ReviewAction] = []

        self._action_counter = 0
        self._man_counter = 0
        self._edit_counter = 0

        # Initialize from automated labels (non-destructive copying)
        labels = source_result.labels
        unique_labels = np.unique(labels)
        for val in unique_labels:
            if val == 0:
                continue
            src_id = int(val)
            mask = (labels == val)
            review_id = f"AUTO-{src_id:06d}"
            self._active_objects[review_id] = _ReviewObjectRecord(
                review_id=review_id,
                mask=mask,
                source_nucleus_id=src_id,
                review_status="unreviewed",
                review_origin="automated",
                parent_review_ids=(),
                review_note=None,
            )

    def _next_action_id(self) -> str:
        """Allocate next sequential action identifier."""
        self._action_counter += 1
        return f"ACT-{self._action_counter:06d}"

    def _record_action(
        self,
        action_type: str,
        target_review_ids: tuple[str, ...],
        resulting_review_ids: tuple[str, ...],
        note: str | None = None,
    ) -> None:
        """Append an immutable ReviewAction to history."""
        action = ReviewAction(
            action_id=self._next_action_id(),
            action_type=action_type,
            timestamp_utc=self._clock(),
            target_review_ids=target_review_ids,
            resulting_review_ids=resulting_review_ids,
            note=note,
        )
        self._history.append(action)

    def accept(self, review_id: str, note: str | None = None) -> None:
        """Accept an active nucleus.

        Args:
            review_id: Identifier of the active nucleus to accept.
            note: Optional reviewer commentary.
        """
        if review_id not in self._active_objects:
            if review_id in self._retired_objects:
                raise InvalidReviewOperationError(
                    f"Cannot accept retired nucleus '{review_id}'."
                )
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        rec = self._active_objects[review_id]
        rec.review_status = "accepted"
        if note is not None:
            rec.review_note = note

        self._record_action(
            action_type="accept",
            target_review_ids=(review_id,),
            resulting_review_ids=(review_id,),
            note=note,
        )

    def reject(self, review_id: str, note: str | None = None) -> None:
        """Reject an automated candidate nucleus.

        Args:
            review_id: Identifier of the automated nucleus to reject.
            note: Optional reviewer commentary.
        """
        if review_id not in self._active_objects:
            if review_id in self._retired_objects:
                raise InvalidReviewOperationError(
                    f"Nucleus '{review_id}' is already retired."
                )
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        rec = self._active_objects[review_id]
        if rec.review_origin != "automated":
            raise InvalidReviewOperationError(
                f"Cannot reject nucleus '{review_id}' with origin '{rec.review_origin}'. "
                "Use delete() for manual or edited objects."
            )

        if note is not None:
            rec.review_note = note

        # Move to retired registry
        self._active_objects.pop(review_id)
        self._retired_objects[review_id] = _RetiredObjectRecord(
            review_id=rec.review_id,
            mask=rec.mask,
            source_nucleus_id=rec.source_nucleus_id,
            review_origin=rec.review_origin,
            parent_review_ids=rec.parent_review_ids,
            review_note=rec.review_note,
            retirement_reason="rejected",
        )

        self._record_action(
            action_type="reject",
            target_review_ids=(review_id,),
            resulting_review_ids=(),
            note=note,
        )

    def restore(self, review_id: str, note: str | None = None) -> None:
        """Restore a previously rejected automated candidate nucleus.

        Args:
            review_id: Identifier of the rejected nucleus to restore.
            note: Optional reviewer commentary.
        """
        if review_id in self._active_objects:
            raise InvalidReviewOperationError(
                f"Nucleus '{review_id}' is already active."
            )

        if review_id not in self._retired_objects:
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        ret_rec = self._retired_objects[review_id]
        if ret_rec.retirement_reason != "rejected":
            raise InvalidReviewOperationError(
                f"Cannot restore nucleus '{review_id}' with retirement reason '{ret_rec.retirement_reason}'. "
                "Only rejected nuclei can be restored."
            )

        # Check collision with current active objects
        for active_id, active_rec in self._active_objects.items():
            if np.any(np.logical_and(ret_rec.mask, active_rec.mask)):
                raise InvalidReviewOperationError(
                    f"Cannot restore rejected nucleus '{review_id}': overlaps with active nucleus '{active_id}'."
                )

        if note is not None:
            ret_rec.review_note = note

        # Restore to active registry
        self._retired_objects.pop(review_id)
        self._active_objects[review_id] = _ReviewObjectRecord(
            review_id=ret_rec.review_id,
            mask=ret_rec.mask,
            source_nucleus_id=ret_rec.source_nucleus_id,
            review_status="unreviewed",
            review_origin=ret_rec.review_origin,
            parent_review_ids=ret_rec.parent_review_ids,
            review_note=ret_rec.review_note,
        )

        self._record_action(
            action_type="restore",
            target_review_ids=(review_id,),
            resulting_review_ids=(review_id,),
            note=note,
        )

    def add(self, mask: np.ndarray, note: str | None = None) -> str:
        """Add a manually drawn nucleus mask.

        Args:
            mask: 2D boolean/binary mask matching image dimensions.
            note: Optional reviewer commentary.

        Returns:
            The newly created manual review identifier (e.g. 'MAN-000001').
        """
        bool_mask = _validate_mask(mask, self._shape, "Manual mask")

        # Check collision with active objects
        for active_id, active_rec in self._active_objects.items():
            if np.any(np.logical_and(bool_mask, active_rec.mask)):
                raise InvalidReviewMaskError(
                    f"Manual mask overlaps with existing active nucleus '{active_id}'."
                )

        self._man_counter += 1
        new_id = f"MAN-{self._man_counter:06d}"

        self._active_objects[new_id] = _ReviewObjectRecord(
            review_id=new_id,
            mask=bool_mask,
            source_nucleus_id=None,
            review_status="edited",
            review_origin="manual_add",
            parent_review_ids=(),
            review_note=note,
        )

        self._record_action(
            action_type="add",
            target_review_ids=(),
            resulting_review_ids=(new_id,),
            note=note,
        )

        return new_id

    def delete(self, review_id: str, note: str | None = None) -> None:
        """Delete a manual or edited nucleus.

        Args:
            review_id: Identifier of the manual/edited nucleus to delete.
            note: Optional reviewer commentary.
        """
        if review_id not in self._active_objects:
            if review_id in self._retired_objects:
                raise InvalidReviewOperationError(
                    f"Nucleus '{review_id}' is already retired."
                )
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        rec = self._active_objects[review_id]
        if rec.review_origin == "automated":
            raise InvalidReviewOperationError(
                f"Cannot delete automated nucleus '{review_id}'. Use reject() instead."
            )

        if note is not None:
            rec.review_note = note

        # Move to retired registry
        self._active_objects.pop(review_id)
        self._retired_objects[review_id] = _RetiredObjectRecord(
            review_id=rec.review_id,
            mask=rec.mask,
            source_nucleus_id=rec.source_nucleus_id,
            review_origin=rec.review_origin,
            parent_review_ids=rec.parent_review_ids,
            review_note=rec.review_note,
            retirement_reason="deleted",
        )

        self._record_action(
            action_type="delete",
            target_review_ids=(review_id,),
            resulting_review_ids=(),
            note=note,
        )

    def merge(self, review_ids: Sequence[str], note: str | None = None) -> str:
        """Merge two or more active nuclei into a single nucleus.

        Args:
            review_ids: Sequence of at least two distinct active review identifiers.
            note: Optional reviewer commentary.

        Returns:
            The newly created merged review identifier (e.g. 'EDIT-000001').
        """
        if not isinstance(review_ids, (list, tuple)) or len(review_ids) < 2:
            count = len(review_ids) if isinstance(review_ids, (list, tuple)) else type(review_ids).__name__
            raise InvalidReviewOperationError(
                f"Merge requires at least 2 target review_ids, got {count}."
            )

        if len(set(review_ids)) != len(review_ids):
            raise InvalidReviewOperationError("Duplicate review_ids provided for merge.")

        parent_recs: list[_ReviewObjectRecord] = []
        for rid in review_ids:
            if rid not in self._active_objects:
                if rid in self._retired_objects:
                    raise InvalidReviewOperationError(f"Cannot merge retired nucleus '{rid}'.")
                raise UnknownReviewObjectError(f"Unknown review object '{rid}'.")
            parent_recs.append(self._active_objects[rid])

        combined_mask = np.logical_or.reduce([r.mask for r in parent_recs])

        # Retire parents
        for r in parent_recs:
            self._active_objects.pop(r.review_id)
            self._retired_objects[r.review_id] = _RetiredObjectRecord(
                review_id=r.review_id,
                mask=r.mask,
                source_nucleus_id=r.source_nucleus_id,
                review_origin=r.review_origin,
                parent_review_ids=r.parent_review_ids,
                review_note=r.review_note,
                retirement_reason="merged",
            )

        self._edit_counter += 1
        new_id = f"EDIT-{self._edit_counter:06d}"

        self._active_objects[new_id] = _ReviewObjectRecord(
            review_id=new_id,
            mask=combined_mask,
            source_nucleus_id=None,
            review_status="edited",
            review_origin="merge",
            parent_review_ids=tuple(review_ids),
            review_note=note,
        )

        self._record_action(
            action_type="merge",
            target_review_ids=tuple(review_ids),
            resulting_review_ids=(new_id,),
            note=note,
        )

        return new_id

    def split(
        self,
        review_id: str,
        replacement_masks: Sequence[np.ndarray],
        note: str | None = None,
    ) -> tuple[str, ...]:
        """Split an active nucleus using two or more replacement masks.

        Args:
            review_id: Identifier of the active nucleus to split.
            replacement_masks: Sequence of at least two non-overlapping replacement masks
                strictly contained within the parent nucleus mask.
            note: Optional reviewer commentary.

        Returns:
            Tuple of newly created child review identifiers (e.g. ('EDIT-000001', 'EDIT-000002')).
        """
        if review_id not in self._active_objects:
            if review_id in self._retired_objects:
                raise InvalidReviewOperationError(f"Cannot split retired nucleus '{review_id}'.")
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        if not isinstance(replacement_masks, (list, tuple)) or len(replacement_masks) < 2:
            count = len(replacement_masks) if isinstance(replacement_masks, (list, tuple)) else type(replacement_masks).__name__
            raise InvalidReviewMaskError(
                f"Split requires at least 2 replacement masks, got {count}."
            )

        validated_masks = [
            _validate_mask(m, self._shape, f"Replacement mask {i + 1}")
            for i, m in enumerate(replacement_masks)
        ]

        # Check mutual non-overlap among replacement masks
        for i in range(len(validated_masks)):
            for j in range(i + 1, len(validated_masks)):
                if np.any(np.logical_and(validated_masks[i], validated_masks[j])):
                    raise InvalidReviewMaskError(
                        f"Replacement mask {i + 1} and replacement mask {j + 1} overlap each other."
                    )

        parent_rec = self._active_objects[review_id]
        combined_replacements = np.logical_or.reduce(validated_masks)

        # Strict containment within parent mask
        if not np.all(combined_replacements <= parent_rec.mask):
            raise InvalidReviewMaskError(
                f"Split replacement masks exceed the boundary of parent nucleus '{review_id}'."
            )

        # Check collision with other active objects
        for active_id, active_rec in self._active_objects.items():
            if active_id == review_id:
                continue
            if np.any(np.logical_and(combined_replacements, active_rec.mask)):
                raise InvalidReviewMaskError(
                    f"Split replacement mask overlaps existing active nucleus '{active_id}'."
                )

        # Retire parent
        self._active_objects.pop(review_id)
        self._retired_objects[review_id] = _RetiredObjectRecord(
            review_id=parent_rec.review_id,
            mask=parent_rec.mask,
            source_nucleus_id=parent_rec.source_nucleus_id,
            review_origin=parent_rec.review_origin,
            parent_review_ids=parent_rec.parent_review_ids,
            review_note=parent_rec.review_note,
            retirement_reason="split",
        )

        # Allocate children
        child_ids: list[str] = []
        for c_mask in validated_masks:
            self._edit_counter += 1
            c_id = f"EDIT-{self._edit_counter:06d}"
            child_ids.append(c_id)
            self._active_objects[c_id] = _ReviewObjectRecord(
                review_id=c_id,
                mask=c_mask,
                source_nucleus_id=None,
                review_status="edited",
                review_origin="split",
                parent_review_ids=(review_id,),
                review_note=note,
            )

        self._record_action(
            action_type="split",
            target_review_ids=(review_id,),
            resulting_review_ids=tuple(child_ids),
            note=note,
        )

        return tuple(child_ids)

    def set_note(self, review_id: str, note: str) -> None:
        """Set or update the review note for an active or retired nucleus.

        Args:
            review_id: Identifier of the nucleus.
            note: Review commentary string.
        """
        if review_id in self._active_objects:
            self._active_objects[review_id].review_note = note
        elif review_id in self._retired_objects:
            self._retired_objects[review_id].review_note = note
        else:
            raise UnknownReviewObjectError(f"Unknown review object '{review_id}'.")

        self._record_action(
            action_type="set_note",
            target_review_ids=(review_id,),
            resulting_review_ids=(review_id,),
            note=note,
        )

    def result(self) -> ReviewedSegmentationResult:
        """Generate an independent snapshot, which may represent an incomplete review."""
        labels = np.zeros(self._shape, dtype=np.int32)
        if not self._active_objects:
            return ReviewedSegmentationResult(
                labels=labels,
                objects=_empty_reviewed_objects_dataframe(),
                source_result=self._source_result,
                history=tuple(self._history),
            )

        rows: list[dict] = []
        h, w = self._shape

        for idx, (rid, rec) in enumerate(self._active_objects.items(), start=1):
            labels[rec.mask] = idx

            y_coords, x_coords = np.nonzero(rec.mask)
            area_px = int(len(y_coords))
            centroid_y = float(np.mean(y_coords))
            centroid_x = float(np.mean(x_coords))
            bbox_min_y = int(np.min(y_coords))
            bbox_min_x = int(np.min(x_coords))
            bbox_max_y = int(np.max(y_coords) + 1)
            bbox_max_x = int(np.max(x_coords) + 1)
            touches_border = bool(
                bbox_min_y == 0
                or bbox_min_x == 0
                or bbox_max_y == h
                or bbox_max_x == w
            )
            qc_status = "border" if touches_border else "accepted"

            rows.append(
                {
                    "review_id": rec.review_id,
                    "current_label": idx,
                    "source_nucleus_id": rec.source_nucleus_id,
                    "review_status": rec.review_status,
                    "review_origin": rec.review_origin,
                    "parent_review_ids": rec.parent_review_ids,
                    "review_note": rec.review_note,
                    "area_px": area_px,
                    "centroid_y": centroid_y,
                    "centroid_x": centroid_x,
                    "bbox_min_y": bbox_min_y,
                    "bbox_min_x": bbox_min_x,
                    "bbox_max_y": bbox_max_y,
                    "bbox_max_x": bbox_max_x,
                    "touches_border": touches_border,
                    "qc_status": qc_status,
                }
            )

        objects_df = pd.DataFrame(rows, columns=REVIEWED_OBJECT_COLUMNS)
        objects_df["source_nucleus_id"] = pd.Series(
            [r["source_nucleus_id"] for r in rows], dtype="object"
        )
        objects_df["parent_review_ids"] = pd.Series(
            [r["parent_review_ids"] for r in rows], dtype="object"
        )
        objects_df["review_note"] = pd.Series(
            [r["review_note"] for r in rows], dtype="object"
        )

        return ReviewedSegmentationResult(
            labels=labels.copy(),
            objects=objects_df.copy(deep=True),
            source_result=self._source_result,
            history=tuple(self._history),
        )

    def validate_complete(self) -> None:
        """Raise if any active review object remains unresolved."""
        unresolved = [
            (record.review_id, record.review_status)
            for record in self._active_objects.values()
            if record.review_status not in RESOLVED_REVIEW_STATUSES
        ]
        if unresolved:
            details = ", ".join(f"{review_id} ({status})" for review_id, status in unresolved)
            raise IncompleteReviewError(
                f"Review is incomplete: {len(unresolved)} active object(s) remain unresolved: {details}."
            )

    def finalize(self) -> ReviewedSegmentationResult:
        """Return a finalized snapshot after requiring all active objects to be resolved."""
        self.validate_complete()
        reviewed = self.result()
        validate_review_completion(reviewed)
        return reviewed


def start_review(
    result: SegmentationResult,
    clock: Callable[[], str] | None = None,
) -> ReviewSession:
    """Start a stateful manual review session on an M2 SegmentationResult.

    Args:
        result: The automated SegmentationResult from M2.
        clock: Optional clock callable returning ISO-8601 UTC timestamp string.

    Returns:
        A new ReviewSession instance.
    """
    return ReviewSession(result, clock=clock)
