"""Multi-sample study-level workflow orchestration session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from nervenuc.aggregation import (
    AnimalAggregationResult,
    SampleIdentity,
    SampleSummary,
    aggregate_by_animal,
)
from nervenuc.export import (
    AnimalGroupAssignment,
    GroupTableResult,
    QCSummaryRecord,
    prepare_group_tables,
    validate_group_assignments,
)
from nervenuc.handoff import (
    AnalysisHandoffResult,
    prepare_analysis_handoff,
)
from nervenuc.statistics import (
    GroupStatisticsResult,
    summarize_groups,
)
from nervenuc.summary import BiologicalSummary3DResult, BiologicalSummaryResult

from .errors import (
    DuplicateSampleError,
    IncompatibleMeasurementSemanticsError,
    InvalidWorkflowStateError,
    WorkflowConfigurationError,
)
from .provenance import ImageSourceProvenance, VolumeSourceProvenance
from .config import AnalysisConfig, VolumeAnalysisConfig


@dataclass(frozen=True)
class ProcessingHeterogeneity:
    """Metadata detailing pipeline configuration differences across study samples sharing valid measurement semantics.

    Attributes:
        measurement_semantics: The unified measurement semantics shared by all samples.
        pipeline_fingerprints: Tuple of distinct pipeline fingerprints observed.
        sample_ids_by_pipeline: Mapping of pipeline fingerprint to tuple of sample IDs.
    """

    measurement_semantics: Literal["direct_nuclear_pixels_2d", "projected_nuclear_pixels_2d", "nuclear_voxels_3d"]
    pipeline_fingerprints: tuple[str, ...]
    sample_ids_by_pipeline: Mapping[str, tuple[str, ...]]


class StudyAnalysisSession:
    """Stateful orchestration session for multi-sample study aggregation, grouping, statistics, and handoff."""

    def __init__(self) -> None:
        self.samples: dict[str, SampleSummary] = {}
        self._sample_provenance: dict[str, ImageSourceProvenance | VolumeSourceProvenance] = {}
        self._sample_configs: dict[str, AnalysisConfig | VolumeAnalysisConfig] = {}
        self._sample_qc_records: dict[str, QCSummaryRecord] = {}
        self._processing_heterogeneity: ProcessingHeterogeneity | None = None
        self.aggregation_result: AnimalAggregationResult | None = None
        self.group_assignments: list[AnimalGroupAssignment] | None = None
        self.group_tables: GroupTableResult | None = None
        self.statistics_result: GroupStatisticsResult | None = None
        self.handoff_result: AnalysisHandoffResult | None = None

    @property
    def sample_provenance(self) -> dict[str, ImageSourceProvenance | VolumeSourceProvenance]:
        """Read-only dictionary of registered sample source/preparation provenance."""
        return dict(self._sample_provenance)

    @property
    def processing_heterogeneity(self) -> ProcessingHeterogeneity | None:
        """Structured record of pipeline fingerprint heterogeneity if observed across compatible samples."""
        return self._processing_heterogeneity

    @property
    def sample_configs(self) -> dict[str, AnalysisConfig | VolumeAnalysisConfig]:
        """Return copies of the authoritative per-sample scientific configurations."""
        return dict(self._sample_configs)

    @property
    def sample_qc_records(self) -> dict[str, QCSummaryRecord]:
        """Return factual per-sample finalized QC accounting records."""
        return dict(self._sample_qc_records)

    def add_sample(
        self,
        identity: SampleIdentity,
        summary: BiologicalSummaryResult | BiologicalSummary3DResult,
        source_provenance: ImageSourceProvenance | VolumeSourceProvenance | None = None,
        analysis_config: AnalysisConfig | VolumeAnalysisConfig | None = None,
        qc_summary: QCSummaryRecord | None = None,
    ) -> None:
        """Register a biological sample summary with optional source/preparation provenance.

        Args:
            identity: SampleIdentity containing sample_id and animal_id.
            summary: BiologicalSummaryResult or BiologicalSummary3DResult from image/volume analysis.
            source_provenance: Optional ImageSourceProvenance or VolumeSourceProvenance record.

        Raises:
            DuplicateSampleError: If sample_id is already registered.
            WorkflowConfigurationError: If identity, summary, or source_provenance is invalid.
        """
        if not isinstance(identity, SampleIdentity):
            raise WorkflowConfigurationError(
                f"identity must be a SampleIdentity, got {type(identity).__name__}."
            )
        if not isinstance(summary, (BiologicalSummaryResult, BiologicalSummary3DResult)):
            raise WorkflowConfigurationError(
                f"summary must be a BiologicalSummaryResult or BiologicalSummary3DResult, got {type(summary).__name__}."
            )
        if source_provenance is not None and not isinstance(
            source_provenance, (ImageSourceProvenance, VolumeSourceProvenance)
        ):
            raise WorkflowConfigurationError(
                f"source_provenance must be an ImageSourceProvenance or VolumeSourceProvenance, got {type(source_provenance).__name__}."
            )
        if analysis_config is not None and not isinstance(
            analysis_config, (AnalysisConfig, VolumeAnalysisConfig)
        ):
            raise WorkflowConfigurationError(
                f"analysis_config must be an AnalysisConfig or VolumeAnalysisConfig, got {type(analysis_config).__name__}."
            )
        if qc_summary is not None and not isinstance(qc_summary, QCSummaryRecord):
            raise WorkflowConfigurationError(
                f"qc_summary must be a QCSummaryRecord, got {type(qc_summary).__name__}."
            )
        if qc_summary is not None and qc_summary.sample_id != identity.sample_id:
            raise WorkflowConfigurationError(
                f"qc_summary sample_id '{qc_summary.sample_id}' does not match identity '{identity.sample_id}'."
            )

        if identity.sample_id in self.samples:
            raise DuplicateSampleError(
                f"Sample '{identity.sample_id}' is already registered in this study session."
            )

        self.samples[identity.sample_id] = SampleSummary(identity=identity, summary_result=summary)
        if source_provenance is not None:
            self._sample_provenance[identity.sample_id] = source_provenance
        if analysis_config is not None:
            self._sample_configs[identity.sample_id] = analysis_config
        if qc_summary is not None:
            self._sample_qc_records[identity.sample_id] = qc_summary

        # Invalidate downstream
        self.aggregation_result = None
        self._processing_heterogeneity = None
        self.group_assignments = None
        self.group_tables = None
        self.statistics_result = None
        self.handoff_result = None

    def remove_sample(self, sample_id: str) -> None:
        """Remove a registered sample by its sample_id.

        Args:
            sample_id: Exact sample identifier.

        Raises:
            InvalidWorkflowStateError: If sample_id is not registered.
        """
        if sample_id not in self.samples:
            raise InvalidWorkflowStateError(
                f"Sample '{sample_id}' not found in study session."
            )

        del self.samples[sample_id]
        self._sample_provenance.pop(sample_id, None)
        self._sample_configs.pop(sample_id, None)
        self._sample_qc_records.pop(sample_id, None)

        # Invalidate downstream
        self.aggregation_result = None
        self._processing_heterogeneity = None
        self.group_assignments = None
        self.group_tables = None
        self.statistics_result = None
        self.handoff_result = None

    def aggregate(self) -> AnimalAggregationResult:
        """Pool raw numerator and denominator counts by animal across registered samples via M7.

        Returns:
            AnimalAggregationResult object.

        Raises:
            InvalidWorkflowStateError: If no samples are registered.
            IncompatibleMeasurementSemanticsError: If samples have incompatible measurement semantics or mixed provenance.
        """
        if not self.samples:
            raise InvalidWorkflowStateError("Cannot aggregate before adding at least one sample.")

        has_prov = [s_id for s_id in self.samples if s_id in self._sample_provenance]
        if has_prov and len(has_prov) != len(self.samples):
            raise IncompatibleMeasurementSemanticsError(
                f"Study contains a mixture of {len(has_prov)} provenance-bearing sample(s) "
                f"and {len(self.samples) - len(has_prov)} legacy provenance-unknown sample(s). "
                "All samples in a study must share provenance or all be legacy."
            )

        self._processing_heterogeneity = None
        if len(has_prov) == len(self.samples) and len(self.samples) > 0:
            semantics_by_sample: dict[str, str] = {}
            pipeline_by_sample: dict[str, str | None] = {}
            for s_id, prov in self._sample_provenance.items():
                if isinstance(prov, VolumeSourceProvenance):
                    sem = prov.measurement_semantics
                elif isinstance(prov, ImageSourceProvenance):
                    sem = (
                        "direct_nuclear_pixels_2d"
                        if prov.source_kind == "direct_2d"
                        else "projected_nuclear_pixels_2d"
                    )
                else:
                    sem = "unknown"
                semantics_by_sample[s_id] = sem
                pipeline_by_sample[s_id] = prov.pipeline_fingerprint

            unique_semantics = set(semantics_by_sample.values())
            if len(unique_semantics) > 1:
                raise IncompatibleMeasurementSemanticsError(
                    f"Incompatible measurement semantics across study samples: {sorted(unique_semantics)}. "
                    "All samples in a study session must share identical measurement semantics."
                )

            eff_sem = list(unique_semantics)[0]
            pipes: dict[str, list[str]] = {}
            for s_id, pipe_fp in pipeline_by_sample.items():
                key = pipe_fp if pipe_fp is not None else "unknown"
                pipes.setdefault(key, []).append(s_id)

            if len(pipes) > 1:
                self._processing_heterogeneity = ProcessingHeterogeneity(
                    measurement_semantics=eff_sem,  # type: ignore[arg-type]
                    pipeline_fingerprints=tuple(sorted(pipes.keys())),
                    sample_ids_by_pipeline={k: tuple(v) for k, v in pipes.items()},
                )

        res = aggregate_by_animal(list(self.samples.values()))
        self.aggregation_result = res

        # Invalidate downstream
        self.group_assignments = None
        self.group_tables = None
        self.statistics_result = None
        self.handoff_result = None

        return res


    def assign_groups(
        self,
        assignments: Sequence[AnimalGroupAssignment],
    ) -> None:
        """Assign experimental groups to aggregated animals.

        Args:
            assignments: Sequence of AnimalGroupAssignment objects.

        Raises:
            InvalidWorkflowStateError: If aggregate() has not been run.
            WorkflowConfigurationError: If assignments are invalid or do not match aggregated animals.
        """
        if self.aggregation_result is None:
            raise InvalidWorkflowStateError(
                "Cannot assign groups before aggregating samples (aggregation_result is None)."
            )

        try:
            validate_group_assignments(self.aggregation_result, assignments)
        except Exception as e:
            raise WorkflowConfigurationError(f"Invalid group assignments: {e}") from e

        self.group_assignments = list(assignments)

        # Invalidate downstream
        self.group_tables = None
        self.statistics_result = None
        self.handoff_result = None

    def prepare_group_tables(self) -> GroupTableResult:
        """Generate long-form, GraphPad wide tables, and manifests via M8.

        Returns:
            GroupTableResult object.

        Raises:
            InvalidWorkflowStateError: If aggregation_result or group_assignments is None.
        """
        if self.aggregation_result is None:
            raise InvalidWorkflowStateError(
                "Cannot prepare group tables before aggregation."
            )
        if self.group_assignments is None:
            raise InvalidWorkflowStateError(
                "Cannot prepare group tables before assigning groups."
            )

        res = prepare_group_tables(self.aggregation_result, self.group_assignments)
        self.group_tables = res

        # Invalidate downstream
        self.statistics_result = None
        self.handoff_result = None

        return res

    def summarize_groups(self) -> GroupStatisticsResult:
        """Calculate sample descriptive statistics per group x metric via M9.

        Returns:
            GroupStatisticsResult object.

        Raises:
            InvalidWorkflowStateError: If group_tables is None.
        """
        if self.group_tables is None:
            raise InvalidWorkflowStateError(
                "Cannot summarize groups before preparing group tables."
            )

        res = summarize_groups(self.group_tables)
        self.statistics_result = res

        # Invalidate downstream
        self.handoff_result = None

        return res

    def prepare_handoff(self) -> AnalysisHandoffResult:
        """Generate analysis-readiness handoff report via M10.

        Returns:
            AnalysisHandoffResult object.

        Raises:
            InvalidWorkflowStateError: If statistics_result is None.
        """
        if self.statistics_result is None:
            raise InvalidWorkflowStateError(
                "Cannot prepare handoff before calculating group statistics."
            )

        res = prepare_analysis_handoff(self.statistics_result)
        self.handoff_result = res
        return res

    def status(self) -> dict[str, Any]:
        """Return factual study session completion state."""
        animal_count = (
            len(set(self.aggregation_result.animal_metrics["animal_id"]))
            if self.aggregation_result is not None
            else 0
        )
        return {
            "sample_count": len(self.samples),
            "animal_count": animal_count,
            "aggregated": self.aggregation_result is not None,
            "groups_assigned": self.group_assignments is not None and len(self.group_assignments) > 0,
            "group_tables_prepared": self.group_tables is not None,
            "statistics_complete": self.statistics_result is not None,
            "handoff_complete": self.handoff_result is not None,
        }
