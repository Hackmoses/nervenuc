"""Validation of compatibility between StackPreparationConfig and AnalysisConfig."""

from __future__ import annotations

from typing import Any

from nervenuc.preparation import StackPreparationConfig

from .config import AnalysisConfig
from .errors import WorkflowConfigurationError



class PreparationCompatibilityError(WorkflowConfigurationError):
    """Raised when StackPreparationConfig marker channels mismatch AnalysisConfig markers."""


def validate_preparation_analysis_compatibility(
    preparation: StackPreparationConfig,
    analysis: AnalysisConfig,
) -> None:
    """Validate that preparation marker channels match AnalysisConfig markers exactly.

    Args:
        preparation: StackPreparationConfig instance.
        analysis: AnalysisConfig instance.

    Raises:
        PreparationCompatibilityError: If marker names are missing, extra, or mismatched.
    """
    if not isinstance(preparation, StackPreparationConfig):
        raise PreparationCompatibilityError(
            f"preparation must be a StackPreparationConfig, got {type(preparation).__name__}."
        )
    if not isinstance(analysis, AnalysisConfig):
        raise PreparationCompatibilityError(
            f"analysis must be an AnalysisConfig, got {type(analysis).__name__}."
        )

    prep_markers = set(preparation.marker_channels.keys())
    analysis_markers = {m.name for m in analysis.markers}

    if prep_markers != analysis_markers:
        missing = analysis_markers - prep_markers
        extra = prep_markers - analysis_markers
        parts = []
        if missing:
            parts.append(f"missing preparation for configured markers {sorted(missing)}")
        if extra:
            parts.append(f"extra unconfigured markers in preparation {sorted(extra)}")
        raise PreparationCompatibilityError(
            f"Stack preparation markers mismatch AnalysisConfig: {'; '.join(parts)}."
        )


def validate_volume_selection_analysis_compatibility(
    selection: Any,
    analysis: Any,
) -> None:
    """Validate that VolumeSelectionConfig marker channels match VolumeAnalysisConfig markers exactly.

    Args:
        selection: VolumeSelectionConfig instance.
        analysis: VolumeAnalysisConfig instance.

    Raises:
        PreparationCompatibilityError: If marker names are missing, extra, or mismatched.
    """
    from nervenuc.preparation import VolumeSelectionConfig
    from .config import VolumeAnalysisConfig

    if not isinstance(selection, VolumeSelectionConfig):
        raise PreparationCompatibilityError(
            f"selection must be a VolumeSelectionConfig, got {type(selection).__name__}."
        )
    if not isinstance(analysis, VolumeAnalysisConfig):
        raise PreparationCompatibilityError(
            f"analysis must be a VolumeAnalysisConfig, got {type(analysis).__name__}."
        )

    sel_markers = set(selection.marker_channels.keys())
    analysis_markers = {m.name for m in analysis.markers}

    if sel_markers != analysis_markers:
        missing = analysis_markers - sel_markers
        extra = sel_markers - analysis_markers
        parts = []
        if missing:
            parts.append(f"missing selection for configured markers {sorted(missing)}")
        if extra:
            parts.append(f"extra unconfigured markers in selection {sorted(extra)}")
        raise PreparationCompatibilityError(
            f"Volume selection markers mismatch VolumeAnalysisConfig: {'; '.join(parts)}."
        )
