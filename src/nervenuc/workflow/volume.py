"""Orchestration session for end-to-end 3D volume analysis pipelines."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Mapping, Sequence

import pandas as pd

from nervenuc.classification import (
    Classification3DResult,
    MarkerThreshold,
    classify_markers_3d,
)
from nervenuc.markers import MarkerDefinition
from nervenuc.measurements3d import (
    MarkerMeasurement3DResult,
    measure_markers_3d,
    validate_measurement3d_consistency,
)
from nervenuc.preparation import PreparedVolumeSet
from nervenuc.qc3d import (
    ReviewSession3D,
    ReviewedSegmentation3DResult,
    start_review_3d,
)
from nervenuc.segmentation3d import (
    Segmentation3DConfig,
    Segmentation3DResult,
    segment_nuclei_3d,
    validate_segmentation3d_consistency,
)
from nervenuc.summary import (
    BiologicalMetricDefinition,
    BiologicalSummary3DResult,
    summarize_metrics_3d,
)

from .compatibility import validate_volume_selection_analysis_compatibility
from .config import VolumeAnalysisConfig
from .errors import InvalidWorkflowStateError, WorkflowConfigurationError
from .fingerprint import (
    compute_volume_analysis_config_fingerprint,
    compute_volume_pipeline_fingerprint,
)
from .provenance import VolumeSourceProvenance


class VolumeAnalysisSession:
    """Stateful orchestrator for 3D microscopy volume analysis.

    Enforces linear progression, strict stale-state invalidation, and complete provenance tracking.
    """

    def __init__(self) -> None:
        self._volume: PreparedVolumeSet | None = None
        self._source_id: str | None = None
        self._source_hash: str | None = None
        self._provenance: VolumeSourceProvenance | None = None

        self._segmentation_result: Segmentation3DResult | None = None
        self._active_review_session: ReviewSession3D | None = None
        self._reviewed_result: ReviewedSegmentation3DResult | None = None
        self._measurement_result: MarkerMeasurement3DResult | None = None
        self._classification_result: Classification3DResult | None = None
        self._summary_result: BiologicalSummary3DResult | None = None
        self._analysis_config: VolumeAnalysisConfig | None = None

    @property
    def volume(self) -> PreparedVolumeSet | None:
        return self._volume

    @property
    def source_id(self) -> str | None:
        return self._source_id

    @property
    def source_provenance(self) -> VolumeSourceProvenance | None:
        return self._provenance

    @property
    def segmentation_result(self) -> Segmentation3DResult | None:
        return self._segmentation_result

    @property
    def reviewed_result(self) -> ReviewedSegmentation3DResult | None:
        return self._reviewed_result

    @property
    def measurement_result(self) -> MarkerMeasurement3DResult | None:
        return self._measurement_result

    @property
    def classification_result(self) -> Classification3DResult | None:
        return self._classification_result

    @property
    def summary_result(self) -> BiologicalSummary3DResult | None:
        return self._summary_result

    @property
    def analysis_config(self) -> VolumeAnalysisConfig | None:
        """Return the authoritative configuration used by the current volume state."""
        return self._analysis_config

    def _refresh_pipeline_identity(self) -> None:
        """Keep provenance fingerprints synchronized with the executable config."""
        if self._analysis_config is None or self._provenance is None:
            return
        config_fingerprint = compute_volume_analysis_config_fingerprint(self._analysis_config)
        self._provenance = replace(
            self._provenance,
            volume_config_fingerprint=config_fingerprint,
            pipeline_fingerprint=compute_volume_pipeline_fingerprint(config_fingerprint),
        )

    def set_volume(
        self,
        volume: PreparedVolumeSet,
        source_id: str | None = None,
        source_hash: str | None = None,
        config_fingerprint: str | None = None,
        pipeline_fingerprint: str | None = None,
        analysis_config: VolumeAnalysisConfig | None = None,
        qc_review_policy: Literal["manual", "auto_accept"] | None = None,
        manual_per_object_qc_performed: bool | None = None,
    ) -> None:
        """Set the active 3D subvolume set and establish execution source identity."""
        if not isinstance(volume, PreparedVolumeSet):
            raise WorkflowConfigurationError(
                f"volume must be a PreparedVolumeSet, got {type(volume).__name__}."
            )
        if analysis_config is not None:
            if not isinstance(analysis_config, VolumeAnalysisConfig):
                raise WorkflowConfigurationError(
                    f"analysis_config must be a VolumeAnalysisConfig, got {type(analysis_config).__name__}."
                )
            validate_volume_selection_analysis_compatibility(analysis_config.selection, analysis_config)

        if source_id is not None:
            if not isinstance(source_id, str) or not source_id.strip():
                raise WorkflowConfigurationError("source_id must be a non-empty string.")
            eff_source_id = source_id.strip()
        elif source_hash is not None:
            eff_source_id = f"file_sha256:{source_hash.strip()}"
        else:
            raise WorkflowConfigurationError(
                "source_id must be explicitly provided for in-memory volume inputs."
            )

        self._volume = volume
        self._source_id = eff_source_id
        self._source_hash = source_hash
        self._analysis_config = analysis_config

        self._provenance = VolumeSourceProvenance(
            source_id=eff_source_id,
            source_hash=source_hash,
            preparation_provenance=volume.provenance,
            spacing=volume.spacing,
            volume_config_fingerprint=config_fingerprint,
            pipeline_fingerprint=pipeline_fingerprint,
            qc_review_policy=qc_review_policy,
            manual_per_object_qc_performed=manual_per_object_qc_performed,
        )

        # Invalidate all downstream results
        self._segmentation_result = None
        self._active_review_session = None
        self._reviewed_result = None
        self._measurement_result = None
        self._classification_result = None
        self._summary_result = None

    def segment(self, config: Segmentation3DConfig | None = None) -> Segmentation3DResult:
        """Segment 3D nuclei in the loaded volume."""
        if self._volume is None or self._source_id is None:
            raise InvalidWorkflowStateError("Cannot segment: no volume has been set.")

        effective_config = config
        if config is not None and self._analysis_config is not None:
            self._analysis_config = replace(self._analysis_config, segmentation=config)
            self._refresh_pipeline_identity()
        elif self._analysis_config is not None:
            effective_config = self._analysis_config.segmentation

        res = segment_nuclei_3d(
            volume=self._volume,
            config=effective_config,
            source_id=self._source_id,
            spacing=self._volume.spacing,
        )
        self._segmentation_result = res

        # Invalidate downstream
        self._active_review_session = None
        self._reviewed_result = None
        self._measurement_result = None
        self._classification_result = None
        self._summary_result = None
        return res

    def start_review(self) -> ReviewSession3D:
        """Initialize a 3D manual review session on the active segmentation result."""
        if self._segmentation_result is None or self._source_id is None:
            raise InvalidWorkflowStateError("Cannot start review: no segmentation result exists.")

        validate_segmentation3d_consistency(
            self._segmentation_result,
            current_source_id=self._source_id,
            volume=self._volume,
        )
        session = start_review_3d(self._segmentation_result)
        self._active_review_session = session

        # Invalidate downstream
        self._reviewed_result = None
        self._measurement_result = None
        self._classification_result = None
        self._summary_result = None
        return session

    def set_reviewed_result(self, reviewed_result: ReviewedSegmentation3DResult) -> None:
        """Set a finalized 3D reviewed segmentation result."""
        if self._segmentation_result is None or self._source_id is None:
            raise InvalidWorkflowStateError("Cannot set reviewed result: no active segmentation exists.")

        if not isinstance(reviewed_result, ReviewedSegmentation3DResult):
            raise WorkflowConfigurationError(
                f"reviewed_result must be a ReviewedSegmentation3DResult, got {type(reviewed_result).__name__}."
            )

        if reviewed_result.source_result != self._segmentation_result:
            raise WorkflowConfigurationError(
                "ReviewedSegmentation3DResult source_result does not match the active segmentation result."
            )

        if reviewed_result.source_id != self._source_id:
            raise WorkflowConfigurationError(
                f"ReviewedSegmentation3DResult source_id '{reviewed_result.source_id}' does not match "
                f"active execution source_id '{self._source_id}'."
            )

        self._reviewed_result = reviewed_result
        self._active_review_session = None

        # Invalidate downstream
        self._measurement_result = None
        self._classification_result = None
        self._summary_result = None

    @property
    def review_session(self) -> ReviewSession3D | None:
        """Return active 3D review session if one is currently in progress."""
        return self._active_review_session

    def finalize_review(self) -> ReviewedSegmentation3DResult:
        """Finalize the active review session and store the reviewed result."""
        if self._active_review_session is None:
            raise InvalidWorkflowStateError("Cannot finalize review: no active review session exists.")
        reviewed = self._active_review_session.finalize()
        self.set_reviewed_result(reviewed)
        return reviewed

    def auto_accept(self) -> ReviewedSegmentation3DResult:

        """Auto-accept all automated 3D nuclei and finalize review."""
        review_session = self.start_review()
        review_session.auto_accept()
        reviewed = review_session.finalize()
        self.set_reviewed_result(reviewed)
        return reviewed

    def measure(
        self,
        markers: Sequence[MarkerDefinition] | None = None,
    ) -> MarkerMeasurement3DResult:
        """Measure marker signals within 3D nuclear volumes."""
        if self._reviewed_result is None or self._volume is None or self._source_id is None:
            raise InvalidWorkflowStateError("Cannot measure: no finalized reviewed result exists.")

        effective_markers = markers
        if markers is None and self._analysis_config is not None:
            effective_markers = self._analysis_config.markers

        res = measure_markers_3d(
            reviewed_result=self._reviewed_result,
            marker_volumes=self._volume.marker_volumes,
            volume_provenance=self._volume.provenance,
            markers=effective_markers,
        )
        validate_measurement3d_consistency(
            res,
            current_source_id=self._source_id,
            volume=self._volume,
        )
        self._measurement_result = res
        if markers is not None and self._analysis_config is not None:
            self._analysis_config = replace(self._analysis_config, markers=tuple(markers))
            self._refresh_pipeline_identity()

        # Invalidate downstream
        self._classification_result = None
        self._summary_result = None
        return res

    def classify(
        self,
        thresholds: Sequence[MarkerThreshold] | None = None,
    ) -> Classification3DResult:
        """Classify per-nucleus marker positivity and derive phenotypes."""
        if self._measurement_result is None or self._source_id is None:
            raise InvalidWorkflowStateError("Cannot classify: no measurement result exists.")

        effective_thresholds = thresholds
        if thresholds is None and self._analysis_config is not None:
            effective_thresholds = self._analysis_config.thresholds
        if effective_thresholds is None or len(effective_thresholds) == 0:
            raise WorkflowConfigurationError("thresholds sequence cannot be empty.")

        if thresholds is not None and self._analysis_config is not None:
            self._analysis_config = replace(self._analysis_config, thresholds=tuple(thresholds))
            self._refresh_pipeline_identity()

        res = classify_markers_3d(
            measurement_result=self._measurement_result,
            thresholds=effective_thresholds,
        )
        self._classification_result = res

        # Invalidate downstream
        self._summary_result = None
        return res

    def summarize(
        self,
        metrics: Sequence[BiologicalMetricDefinition] | None = None,
    ) -> BiologicalSummary3DResult:
        """Evaluate denominator-aware biological metrics."""
        if self._classification_result is None:
            raise InvalidWorkflowStateError("Cannot summarize: no classification result exists.")

        effective_metrics = metrics
        if metrics is None and self._analysis_config is not None:
            effective_metrics = self._analysis_config.metrics
        if effective_metrics is None or len(effective_metrics) == 0:
            raise WorkflowConfigurationError("metrics sequence cannot be empty.")

        if metrics is not None and self._analysis_config is not None:
            self._analysis_config = replace(self._analysis_config, metrics=tuple(metrics))
            self._refresh_pipeline_identity()

        res = summarize_metrics_3d(
            classification_result=self._classification_result,
            metrics=effective_metrics,
        )
        self._summary_result = res
        return res

    def run_pipeline(
        self,
        config: VolumeAnalysisConfig,
        auto_accept: bool = True,
    ) -> BiologicalSummary3DResult:
        """Execute complete reproducible 3D volume analysis pipeline.

        Args:
            config: Scientifically authoritative VolumeAnalysisConfig.
            auto_accept: If True, auto-accepts all segmented nuclei; if False, stops at segmentation.

        Returns:
            Final BiologicalSummary3DResult.
        """
        if self._volume is None:
            raise InvalidWorkflowStateError("Cannot run pipeline: no volume has been set.")

        validate_volume_selection_analysis_compatibility(config.selection, config)
        self._analysis_config = config

        self._refresh_pipeline_identity()

        self.segment(config.segmentation)
        if not auto_accept:
            raise InvalidWorkflowStateError("Manual review requires interactive ReviewSession3D steps.")

        self.auto_accept()
        self.measure(config.markers)
        self.classify(config.thresholds)
        return self.summarize(config.metrics)
