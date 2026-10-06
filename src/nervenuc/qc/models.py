"""Models for segmentation QC and manual review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nervenuc.segmentation import SegmentationResult


@dataclass(frozen=True)
class ReviewAction:
    """Immutable audit record of a manual review operation.

    Attributes:
        action_id: Deterministic action identifier ('ACT-000001', 'ACT-000002', ...).
        action_type: Type of action ('accept', 'reject', 'restore', 'add',
            'delete', 'merge', 'split', 'set_note').
        timestamp_utc: ISO-8601 UTC timestamp string.
        target_review_ids: Identifiers of objects targeted by the action.
        resulting_review_ids: Identifiers of objects created or modified by the action.
        note: Optional reviewer commentary or rationale.
    """

    action_id: str
    action_type: str
    timestamp_utc: str
    target_review_ids: tuple[str, ...]
    resulting_review_ids: tuple[str, ...]
    note: str | None = None


@dataclass
class ReviewedSegmentationResult:
    """Container for the reviewed segmentation result snapshot.

    Attributes:
        labels: 2D integer NumPy array (np.int32), shape (Y, X), where 0 is background
            and 1..N are active reviewed nuclei.
        objects: Pandas DataFrame containing one row per active reviewed nucleus (16 columns).
        source_result: Provenance reference to the original M2 SegmentationResult.
        history: Chronological tuple of all ReviewActions performed in the session.
    """

    labels: np.ndarray
    objects: pd.DataFrame
    source_result: SegmentationResult
    history: tuple[ReviewAction, ...]
