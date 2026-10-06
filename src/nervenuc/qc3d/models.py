"""Data models for 3D segmentation QC and manual review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nervenuc.segmentation3d import Segmentation3DResult


@dataclass(frozen=True)
class ReviewAction3D:
    """Immutable audit record of a 3D manual review operation.

    Attributes:
        action_id: Deterministic action identifier ('ACT-000001', 'ACT-000002', ...).
        action_type: Type of action ('accept', 'reject', 'restore', 'set_note').
        timestamp_utc: ISO-8601 UTC timestamp string.
        target_object_ids: Identifiers of objects targeted by the action.
        resulting_object_ids: Identifiers of objects affected by the action.
        note: Optional commentary.
    """

    action_id: str
    action_type: str
    timestamp_utc: str
    target_object_ids: tuple[int, ...]
    resulting_object_ids: tuple[int, ...]
    note: str | None = None


@dataclass(frozen=True)
class ReviewedSegmentation3DResult:
    """Container for the reviewed 3D segmentation result.

    Attributes:
        active_objects: Pandas DataFrame containing one row per active reviewed nucleus.
        source_result: Provenance reference to the original M14 Segmentation3DResult.
        history: Chronological tuple of all ReviewAction3D records.
        notes: Mapping of object ID to commentary notes.
    """

    active_objects: pd.DataFrame
    source_result: Segmentation3DResult
    history: tuple[ReviewAction3D, ...]
    notes: Mapping[int, str]

    @property
    def labels(self) -> np.ndarray:
        """Return the 3D label array of the source result."""
        return self.source_result.labels

    @property
    def objects(self) -> pd.DataFrame:
        """Alias for active_objects for seamless downstream consumption."""
        return self.active_objects

    @property
    def active_count(self) -> int:
        """Total number of active accepted 3D nuclei."""
        return len(self.active_objects)

    @property
    def source_id(self) -> str:
        """Authoritative execution source identity inherited from source_result."""
        return self.source_result.source_id
