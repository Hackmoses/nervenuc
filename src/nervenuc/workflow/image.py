"""Single-image workflow orchestration session."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from nervenuc.classification import ClassificationResult, classify_markers
from nervenuc.classification import MarkerThreshold
from nervenuc.measurements import (
    MarkerMeasurementResult,
    measure_markers,
    validate_reviewed_result_consistency,
)
from nervenuc.qc import (
    ReviewSession,
    ReviewedSegmentationResult,
    start_review,
    validate_review_completion,
)
from nervenuc.segmentation import SegmentationResult, segment_nuclei
from nervenuc.segmentation import SegmentationConfig
from nervenuc.summary import BiologicalMetricDefinition, BiologicalSummaryResult, summarize_metrics

from .config import AnalysisConfig
from .errors import InvalidWorkflowStateError, WorkflowConfigurationError
from .fingerprint import compute_analysis_config_fingerprint, compute_pipeline_fingerprint
from .provenance import ImageSourceProvenance


class ImageAnalysisSession:
    """Stateful orchestration session for single-image analysis lifecycle."""

    def __init__(self, config: AnalysisConfig) -> None:
        if not isinstance(config, AnalysisConfig):
            raise WorkflowConfigurationError(
                f"config must be an AnalysisConfig, got {type(config).__name__}."
            )
        self._config = config

        self.nuclear_channel: np.ndarray | None = None
        self.marker_images: dict[str, np.ndarray] | None = None
        self.source_provenance: ImageSourceProvenance | None = None

        self.segmentation_result: SegmentationResult | None = None
        self.review_session: ReviewSession | None = None
        self.reviewed_result: ReviewedSegmentationResult | None = None
        self.measurement_result: MarkerMeasurementResult | None = None
        self.classification_result: ClassificationResult | None = None
        self.summary_result: BiologicalSummaryResult | None = None

    @property
    def config(self) -> AnalysisConfig:
        """Return the immutable scientific configuration bound to current state."""
        return self._config

    def _refresh_pipeline_identity(self) -> None:
        """Keep provenance fingerprints synchronized with the executable config."""
        if self.source_provenance is None:
            return
        config_fingerprint = compute_analysis_config_fingerprint(self._config)
        pipeline_fingerprint = compute_pipeline_fingerprint(
            config_fingerprint,
            self.source_provenance.preparation_fingerprint,
        )
        self.source_provenance = replace(
            self.source_provenance,
            pipeline_fingerprint=pipeline_fingerprint,
        )

    def set_image(
        self,
        nuclear_channel: np.ndarray,
        marker_images: Mapping[str, np.ndarray],
        source_provenance: ImageSourceProvenance | None = None,
    ) -> None:
        """Bind 2D nuclear channel array and marker images with strict shape and key validation.

        Args:
            nuclear_channel: 2D numpy array of nuclear stain (DAPI/Hoechst).
            marker_images: Mapping of exact marker name to 2D numpy array.
            source_provenance: Optional ImageSourceProvenance metadata record.

        Raises:
            WorkflowConfigurationError: If arrays are not 2D, shapes mismatch, or marker keys
                do not match configured markers exactly.
        """
        if not isinstance(nuclear_channel, np.ndarray) or nuclear_channel.ndim != 2:
            raise WorkflowConfigurationError(
                f"nuclear_channel must be a 2D numpy array, got {getattr(nuclear_channel, 'ndim', None)}D."
            )

        if not isinstance(marker_images, Mapping):
            raise WorkflowConfigurationError(
                f"marker_images must be a Mapping, got {type(marker_images).__name__}."
            )

        if source_provenance is not None and not isinstance(source_provenance, ImageSourceProvenance):
            raise WorkflowConfigurationError(
                f"source_provenance must be an ImageSourceProvenance, got {type(source_provenance).__name__}."
            )

        configured_names = {m.name for m in self.config.markers}
        provided_names = set(marker_images.keys())

        if provided_names != configured_names:
            missing = configured_names - provided_names
            extra = provided_names - configured_names
            msg_parts = []
            if missing:
                msg_parts.append(f"missing configured markers {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra unconfigured markers {sorted(extra)}")
            raise WorkflowConfigurationError(
                f"Marker images mismatch configured markers: {'; '.join(msg_parts)}."
            )

        for m_name, m_arr in marker_images.items():
            if not isinstance(m_arr, np.ndarray) or m_arr.ndim != 2:
                raise WorkflowConfigurationError(
                    f"Marker '{m_name}' image must be a 2D numpy array, got {getattr(m_arr, 'ndim', None)}D."
                )
            if m_arr.shape != nuclear_channel.shape:
                raise WorkflowConfigurationError(
                    f"Marker '{m_name}' shape {m_arr.shape} does not match nuclear channel shape {nuclear_channel.shape}."
                )

        self.nuclear_channel = nuclear_channel
        self.marker_images = dict(marker_images)
        self.source_provenance = source_provenance

        # Invalidate downstream
        self.segmentation_result = None
        self.review_session = None
        self.reviewed_result = None
        self.measurement_result = None
        self.classification_result = None
        self.summary_result = None

    def set_prepared_image_set(
        self,
        prepared: Any,
        source_provenance: ImageSourceProvenance | None = None,
    ) -> None:
        """Bind a 2D PreparedImageSet and its associated source provenance.

        Args:
            prepared: PreparedImageSet instance from preparation pipeline.
            source_provenance: Optional ImageSourceProvenance metadata record.
        """
        self.set_image(
            nuclear_channel=prepared.nuclear_image,
            marker_images=prepared.marker_images,
            source_provenance=source_provenance,
        )

    def segment(self, config: SegmentationConfig | None = None) -> SegmentationResult:
        """Run M2 classical nucleus segmentation baseline.

        Returns:
            SegmentationResult object.

        Raises:
            InvalidWorkflowStateError: If nuclear_channel has not been set.
        """
        if self.nuclear_channel is None:
            raise InvalidWorkflowStateError("Cannot run segmentation before setting image.")

        if config is not None:
            if not isinstance(config, SegmentationConfig):
                raise WorkflowConfigurationError(
                    f"config must be a SegmentationConfig, got {type(config).__name__}."
                )
            self._config = replace(self._config, segmentation=config)
            self._refresh_pipeline_identity()

        res = segment_nuclei(self.nuclear_channel, self._config.segmentation)
        self.segmentation_result = res

        # Invalidate downstream
        self.review_session = None
        self.reviewed_result = None
        self.measurement_result = None
        self.classification_result = None
        self.summary_result = None

        return res

    def start_review(self) -> ReviewSession:
        """Create an interactive M3 ReviewSession from the current segmentation.

        Returns:
            ReviewSession object.

        Raises:
            InvalidWorkflowStateError: If segmentation has not been run.
        """
        if self.segmentation_result is None:
            raise InvalidWorkflowStateError("Cannot start review before segmentation.")

        session = start_review(self.segmentation_result)
        self.review_session = session

        # A restarted review supersedes any previously finalized review state.
        self.reviewed_result = None
        self.measurement_result = None
        self.classification_result = None
        self.summary_result = None
        return session

    def set_reviewed_result(self, reviewed: ReviewedSegmentationResult) -> None:
        """Store a finalized reviewed segmentation result compatible with current segmentation.

        Args:
            reviewed: Finalized ReviewedSegmentationResult object.

        Raises:
            InvalidWorkflowStateError: If segmentation has not been run, or if the reviewed
                result is incompatible with the current session's segmentation.
        """
        if self.segmentation_result is None:
            raise InvalidWorkflowStateError("Cannot set reviewed result before segmentation.")

        if not isinstance(reviewed, ReviewedSegmentationResult):
            raise InvalidWorkflowStateError(
                f"reviewed must be a ReviewedSegmentationResult, got {type(reviewed).__name__}."
            )

        if reviewed.labels.shape != self.segmentation_result.labels.shape:
            raise InvalidWorkflowStateError(
                f"Reviewed label image shape {reviewed.labels.shape} does not match "
                f"session segmentation shape {self.segmentation_result.labels.shape}."
            )

        if reviewed.source_result is not self.segmentation_result:
            raise InvalidWorkflowStateError(
                "Reviewed result source_result does not match current session segmentation."
            )

        try:
            validate_reviewed_result_consistency(reviewed)
            validate_review_completion(reviewed)
        except Exception as e:
            raise InvalidWorkflowStateError(f"Incompatible reviewed result structure: {e}") from e

        self.reviewed_result = reviewed

        # Invalidate downstream
        self.measurement_result = None
        self.classification_result = None
        self.summary_result = None

    def auto_accept_segmentation(self) -> ReviewedSegmentationResult:
        """Explicit caller action to finalize automated segmentation via M3 review machinery.

        Returns:
            ReviewedSegmentationResult with automated objects accepted.

        Raises:
            InvalidWorkflowStateError: If segmentation has not been run.
        """
        if self.segmentation_result is None:
            raise InvalidWorkflowStateError("Cannot auto-accept segmentation before segmenting.")

        session = start_review(self.segmentation_result)
        for review_id in session.result().objects["review_id"]:
            session.accept(review_id, note="Accepted by explicit auto-accept workflow policy.")
        reviewed = session.finalize()

        self.review_session = session
        self.reviewed_result = reviewed

        # Invalidate downstream
        self.measurement_result = None
        self.classification_result = None
        self.summary_result = None

        return reviewed

    def measure(self) -> MarkerMeasurementResult:
        """Measure raw signal per active reviewed nucleus across configured markers.

        Returns:
            MarkerMeasurementResult object.

        Raises:
            InvalidWorkflowStateError: If reviewed_result is not finalized or marker images not bound.
        """
        if self.reviewed_result is None:
            raise InvalidWorkflowStateError(
                "Cannot measure markers before review is finalized (reviewed_result is None)."
            )

        if self.marker_images is None:
            raise InvalidWorkflowStateError("Marker images are not bound.")

        res = measure_markers(self.reviewed_result, self.marker_images)
        self.measurement_result = res

        # Invalidate downstream
        self.classification_result = None
        self.summary_result = None

        return res

    def classify(
        self,
        thresholds: Sequence[MarkerThreshold] | None = None,
    ) -> ClassificationResult:
        """Classify marker positivity and composite phenotypes using configured thresholds.

        Returns:
            ClassificationResult object.

        Raises:
            InvalidWorkflowStateError: If measurement_result is None.
        """
        if self.measurement_result is None:
            raise InvalidWorkflowStateError("Cannot classify markers before measurement.")

        if thresholds is not None:
            self._config = replace(self._config, thresholds=tuple(thresholds))
            self._refresh_pipeline_identity()

        res = classify_markers(
            self.measurement_result,
            self._config.thresholds,
        )
        self.classification_result = res

        # Invalidate downstream
        self.summary_result = None

        return res

    def summarize(
        self,
        metrics: Sequence[BiologicalMetricDefinition] | None = None,
    ) -> BiologicalSummaryResult:
        """Calculate single-image denominator-aware biological metric summaries.

        Returns:
            BiologicalSummaryResult object.

        Raises:
            InvalidWorkflowStateError: If classification_result is None.
        """
        if self.classification_result is None:
            raise InvalidWorkflowStateError("Cannot summarize metrics before classification.")

        if metrics is not None:
            self._config = replace(self._config, metrics=tuple(metrics))
            self._refresh_pipeline_identity()

        res = summarize_metrics(
            self.classification_result,
            self._config.metrics,
        )
        self.summary_result = res
        return res

    def status(self) -> dict[str, bool]:
        """Return factual completion flags for the image analysis workflow stages."""
        return {
            "image_set": self.nuclear_channel is not None and self.marker_images is not None,
            "segmented": self.segmentation_result is not None,
            "review_started": self.review_session is not None,
            "review_complete": self.reviewed_result is not None,
            "measured": self.measurement_result is not None,
            "classified": self.classification_result is not None,
            "summarized": self.summary_result is not None,
        }
