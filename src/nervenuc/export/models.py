"""Models for animal-level group tables and export preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .errors import InvalidGroupAssignmentError

if TYPE_CHECKING:
    from nervenuc.aggregation import AnimalAggregationResult


@dataclass(frozen=True)
class AnimalGroupAssignment:
    """Immutable mapping assigning an animal biological replicate to an experimental group.

    Attributes:
        animal_id: Exact animal identifier matching M7 aggregation result.
        group: User-defined experimental group name (e.g. 'WT', 'CMT1A', 'Vehicle').
    """

    animal_id: str
    group: str

    def __post_init__(self) -> None:
        for field_name, val in (("animal_id", self.animal_id), ("group", self.group)):
            if not isinstance(val, str) or not val.strip():
                raise InvalidGroupAssignmentError(f"{field_name} must be a non-empty string.")
            if val != val.strip():
                raise InvalidGroupAssignmentError(
                    f"{field_name} cannot contain leading or trailing whitespace: {val!r}."
                )


@dataclass
class GroupTableResult:
    """Container for animal-level group tables and export artifacts.

    Attributes:
        long_table: 9-column long-form DataFrame containing animal metrics with group tags.
        wide_tables: Dict mapping metric_name to GraphPad-friendly wide-form DataFrames.
        animal_id_table: Companion DataFrame mapping replicate rows to exact animal IDs.
        manifest: 6-column DataFrame mapping safe names to full metric definitions.
        assignments: Tuple of AnimalGroupAssignment objects applied.
        aggregation_result: Provenance reference to input AnimalAggregationResult.
    """

    long_table: pd.DataFrame
    wide_tables: dict[str, pd.DataFrame]
    animal_id_table: pd.DataFrame
    manifest: pd.DataFrame
    assignments: tuple[AnimalGroupAssignment, ...]
    aggregation_result: AnimalAggregationResult


@dataclass(frozen=True)
class QCSummaryRecord:
    """Record of segmentation quality control and manual review facts for a sample."""

    sample_id: str
    initial_automated_count: int
    accepted_count: int
    rejected_count: int
    added_count: int
    edited_count: int
    final_count: int
    review_state: str


@dataclass
class ReproducibleExportBundle:
    """Complete, self-contained reproducibility bundle for a completed NerveNuc analysis.

    Attributes:
        per_image_table: DataFrame with one row per technical observation (image/volume).
        per_animal_table: DataFrame with pooled biological replicate metrics (or None if no animal metadata).
        group_tables: Optional GroupTableResult with GraphPad-compatible wide tables and group tags.
        object_table: Optional DataFrame containing single-nucleus measurements and classifications (or None).
        analysis_config_json: Canonical JSON serialization of analysis configuration.
        provenance_json: JSON serialization of image/volume extraction and preparation provenance.
        qc_summary_json: JSON serialization of QC/review accounting facts.
        reproducibility_env_json: JSON serialization of software, OS, and library versions.
        methods_summary_text: Methods-ready factual narrative describing executed analysis.
        manifest: Dictionary of bundle metadata (timestamps, versions, flags, counts).
    """

    per_image_table: pd.DataFrame
    per_animal_table: pd.DataFrame | None
    group_tables: GroupTableResult | None
    object_table: pd.DataFrame | None
    analysis_config_json: str
    provenance_json: str
    qc_summary_json: str
    reproducibility_env_json: str
    methods_summary_text: str
    manifest: dict[str, Any]
