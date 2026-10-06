"""Central state management for NerveNuc GUI application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from nervenuc.batch import AnalysisPreset, BatchResult, BatchRunner
from nervenuc.classification import MarkerThreshold
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import (
    StackInspectionResult,
    StackPreparationConfig,
    VolumePreparationProvenance,
    VolumeSelectionConfig,
    ZSelection,
)
from nervenuc.segmentation import SegmentationConfig
from nervenuc.segmentation3d import Segmentation3DConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import (
    AnalysisConfig,
    ImageAnalysisSession,
    StudyAnalysisSession,
    VolumeAnalysisConfig,
    VolumeAnalysisSession,
)

from .errors import InvalidDraftConfigError

CanonicalMode = Literal["direct_2d", "projected_2d", "volume_3d"]


@dataclass
class MarkerThresholdDraft:
    """Mutable draft representation of a per-marker classification threshold."""

    marker_name: str
    cutoff: float = 200.0
    measurement_field: str = "mean_intensity"
    comparison_operator: Literal[">=", ">"] = ">="

    def to_core_threshold(self) -> MarkerThreshold:
        """Convert draft into an immutable validated core MarkerThreshold."""
        if not isinstance(self.cutoff, (int, float)) or not np.isfinite(self.cutoff):
            raise InvalidDraftConfigError(f"Cutoff for marker '{self.marker_name}' must be a finite number.")
        return MarkerThreshold(
            marker_name=self.marker_name,
            threshold=float(self.cutoff),
            measurement=self.measurement_field,
            comparison=self.comparison_operator,
        )



@dataclass
class BiologicalMetricDraft:
    """Mutable draft representation of a denominator-aware biological metric."""

    name: str = "Positivity Percentage"
    denominator_positive: tuple[str, ...] = ()
    denominator_negative: tuple[str, ...] = ()
    numerator_positive: tuple[str, ...] = ()
    numerator_negative: tuple[str, ...] = ()

    def to_core_metric(self) -> BiologicalMetricDefinition:
        """Convert draft into an immutable validated core BiologicalMetricDefinition."""
        if not self.name.strip():
            raise InvalidDraftConfigError("Biological metric name cannot be empty.")

        denom_dict: dict[str, bool] = {}
        for m in self.denominator_positive:
            denom_dict[m] = True
        for m in self.denominator_negative:
            denom_dict[m] = False

        num_dict: dict[str, bool] = {}
        for m in self.numerator_positive:
            num_dict[m] = True
        for m in self.numerator_negative:
            num_dict[m] = False

        denom_filter = PhenotypeFilter(denom_dict)
        num_filter = PhenotypeFilter(num_dict)

        return BiologicalMetricDefinition(
            name=self.name.strip(),
            denominator=denom_filter,
            numerator=num_filter,
        )


@dataclass
class AnalysisDraft:
    """Mutable UI draft configuration for NerveNuc analysis workflows."""

    mode: CanonicalMode = "direct_2d"
    declared_axes: str | None = None
    time_index: int | None = None
    nuclear_channel: int = 0
    marker_channels: dict[str, int] = field(default_factory=dict)
    channel_mapping_confirmed: bool = False
    z_start_1based: int | None = None
    z_stop_1based: int | None = None
    projection_method: Literal["max", "mean"] = "max"
    segmentation_values: dict[str, Any] = field(default_factory=dict)
    preprocessing_values: dict[str, Any] = field(default_factory=dict)
    threshold_drafts: dict[str, MarkerThresholdDraft] = field(default_factory=dict)
    metric_drafts: list[BiologicalMetricDraft] = field(default_factory=list)

    def get_z_selection(self, max_z: int | None = None) -> ZSelection | None:
        """Build core ZSelection from 1-based inclusive UI values."""
        if self.z_start_1based is None and self.z_stop_1based is None:
            return None
        start = self.z_start_1based if self.z_start_1based is not None else 1
        stop = self.z_stop_1based if self.z_stop_1based is not None else (max_z if max_z is not None else start)
        if start < 1:
            raise InvalidDraftConfigError(f"Z start slice must be >= 1, got {start}.")
        if stop < start:
            raise InvalidDraftConfigError(f"Z stop slice ({stop}) cannot be less than start slice ({start}).")
        return ZSelection(start=start - 1, stop=stop)

    def build_core_volume_selection_config(self, max_z: int | None = None) -> VolumeSelectionConfig:
        """Construct VolumeSelectionConfig from draft values."""
        return VolumeSelectionConfig(
            axes=self.declared_axes,
            time_index=self.time_index,
            nuclear_channel=self.nuclear_channel,
            marker_channels=dict(self.marker_channels),
            z_selection=self.get_z_selection(max_z),
        )

    def build_core_2d_config(self) -> AnalysisConfig:
        """Construct immutable validated 2D AnalysisConfig from draft values."""
        seg_cfg = SegmentationConfig(**self.segmentation_values) if self.segmentation_values else SegmentationConfig()
        m_names = list(self.marker_channels.keys()) if self.marker_channels else ["DefaultMarker"]
        markers = tuple(MarkerDefinition(name) for name in m_names)

        if self.threshold_drafts:
            thresholds = tuple(t.to_core_threshold() for t in self.threshold_drafts.values())
        else:
            thresholds = tuple(MarkerThreshold(marker_name=m, threshold=200.0) for m in m_names)

        if self.metric_drafts:
            metrics = tuple(m.to_core_metric() for m in self.metric_drafts)
        else:
            metrics = tuple(
                BiologicalMetricDefinition(
                    name=f"{m}_pos_pct",
                    denominator=PhenotypeFilter(),
                    numerator=PhenotypeFilter({m: True}),
                )
                for m in m_names
            )

        return AnalysisConfig(
            segmentation=seg_cfg,
            markers=markers,
            thresholds=thresholds,
            metrics=metrics,
        )

    def build_core_stack_preparation_config(self, max_z: int | None = None) -> StackPreparationConfig:
        """Construct immutable validated StackPreparationConfig from draft values."""
        return StackPreparationConfig(
            axes=self.declared_axes,
            time_index=self.time_index,
            nuclear_channel=self.nuclear_channel,
            marker_channels=dict(self.marker_channels),
            z_selection=self.get_z_selection(max_z),
            projection_method=self.projection_method,
        )

    def build_core_volume_config(self, max_z: int | None = None) -> VolumeAnalysisConfig:
        """Construct immutable validated VolumeAnalysisConfig from draft values."""
        sel_cfg = self.build_core_volume_selection_config(max_z)
        seg_cfg = Segmentation3DConfig.from_dict(self.segmentation_values) if self.segmentation_values else Segmentation3DConfig()
        m_names = list(self.marker_channels.keys())
        markers = tuple(MarkerDefinition(name) for name in m_names)

        if self.threshold_drafts:
            thresholds = tuple(t.to_core_threshold() for t in self.threshold_drafts.values())
        else:
            thresholds = tuple(MarkerThreshold(marker_name=m, threshold=200.0) for m in m_names)

        if self.metric_drafts:
            metrics = tuple(m.to_core_metric() for m in self.metric_drafts)
        else:
            metrics = tuple(
                BiologicalMetricDefinition(
                    name=f"{m}_pos_pct",
                    denominator=PhenotypeFilter(),
                    numerator=PhenotypeFilter({m: True}),
                )
                for m in m_names
            )

        return VolumeAnalysisConfig(
            selection=sel_cfg,
            segmentation=seg_cfg,
            markers=markers,
            thresholds=thresholds,
            metrics=metrics,
        )



@dataclass
class DisplayTransformSettings:
    """Transient display-only rendering settings (never affects quantitative calculations)."""

    channel_min: dict[int, float] = field(default_factory=dict)
    channel_max: dict[int, float] = field(default_factory=dict)
    channel_gamma: dict[int, float] = field(default_factory=dict)
    channel_visible: dict[int, bool] = field(default_factory=dict)
    channel_color: dict[int, str] = field(default_factory=dict)
    show_segmentation_overlay: bool = True
    show_preprocessed: bool = False
    show_object_ids: bool = False
    overlay_opacity: float = 0.5


class GuiSessionState:
    """Central observable state coordinator for the NerveNuc GUI application."""

    def __init__(self) -> None:
        self.quick_result = None
        self.source_metadata = None
        self.loaded_path: Path | None = None
        self.loaded_raw_array: np.ndarray | None = None
        self.preview_axes: str | None = None
        self.source_id: str | None = None
        self.inspection: StackInspectionResult | None = None

        self.draft: AnalysisDraft = AnalysisDraft()
        self.display: DisplayTransformSettings = DisplayTransformSettings()

        self.session_2d: ImageAnalysisSession | None = None
        self.session_3d: VolumeAnalysisSession | None = None
        self.study_session: StudyAnalysisSession = StudyAnalysisSession()
        self.batch_runner: BatchRunner | None = None
        self.batch_result: BatchResult | None = None
        self.batch_running: bool = False

        self.active_preset: AnalysisPreset | None = None
        self.active_preset_path: Path | None = None

        self.preset_dirty: bool = False
        self.configuration_differs_from_run: bool = False
        self.results_stale: bool = False

        self.current_z_index: int = 0
        self.view_mode: Literal["stack", "projected"] = "stack"

        self._listeners: list[Callable[[], None]] = []

    @property
    def image_session(self):
        return self.session_3d if self.draft.mode == "volume_3d" else self.session_2d

    @image_session.setter
    def image_session(self, session):
        if session is None:
            self.session_2d = self.session_3d = None
        elif isinstance(session, VolumeAnalysisSession):
            self.session_3d, self.session_2d = session, None
            self.draft.mode = 'volume_3d'
        elif isinstance(session, ImageAnalysisSession):
            self.session_2d, self.session_3d = session, None
            if self.draft.mode == 'volume_3d': self.draft.mode = 'direct_2d'
        else:
            raise TypeError('Expected an image or volume analysis session.')

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Register a callback for state change notifications."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[], None]) -> None:
        """Unregister a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def notify(self) -> None:
        """Broadcast state updates to all subscribed UI controllers and views."""
        for listener in list(self._listeners):
            listener()

    def set_mode(self, mode: CanonicalMode) -> None:
        """Change analysis mode, invalidating incompatible committed results."""
        if self.draft.mode != mode:
            self.draft.mode = mode
            self.results_stale = True
            self.configuration_differs_from_run = True
            # Invalidate session states
            self.session_2d = None
            self.session_3d = None
            self.notify()

    def mark_draft_modified(self) -> None:
        """Record that user modified form values."""
        self.preset_dirty = True
        self.configuration_differs_from_run = True
        self.results_stale = True
        self.notify()

    def clear_preset_dirty(self) -> None:
        """Clear preset dirty flag upon save or load (leaves results_stale intact)."""
        self.preset_dirty = False
        self.notify()
