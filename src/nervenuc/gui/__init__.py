"""NerveNuc Graphical User Interface (GUI) module.

Provides a researcher-facing PySide6 desktop application for non-destructive
microscopy analysis, segmentation, QC review, marker classification, and batch processing.
"""

from __future__ import annotations

try:
    import PySide6  # noqa: F401
except ImportError as e:
    from .errors import MissingGuiDependencyError

    raise MissingGuiDependencyError(
        "NerveNuc GUI dependencies are not installed. "
        "Please install them using: pip install \"nervenuc[gui]\""
    ) from e

from .state import (
    AnalysisDraft,
    BiologicalMetricDraft,
    CanonicalMode,
    DisplayTransformSettings,
    GuiSessionState,
    MarkerThresholdDraft,
)

__all__ = [
    "AnalysisDraft",
    "BiologicalMetricDraft",
    "CanonicalMode",
    "DisplayTransformSettings",
    "GuiSessionState",
    "MarkerThresholdDraft",
]
