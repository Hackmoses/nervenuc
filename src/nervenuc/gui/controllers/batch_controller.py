"""Controller for batch manifests, preset serialization, and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nervenuc.batch import (
    AnalysisPreset,
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    BatchRunner,
    BatchSampleSpec,
    ChannelMappedFileInput,
    FileInput,
    StackFileInput,
    compute_manifest_fingerprint,
    load_checkpoint,
    load_preset,
    resume_batch,
    save_checkpoint,
    save_preset,
)
from nervenuc.preparation import inspect_stack, prepare_direct_image, resolve_axes
from nervenuc.qc import ReviewedSegmentationResult
from nervenuc.qc3d import ReviewedSegmentation3DResult

from ..errors import NerveNucGuiError
from ..state import BiologicalMetricDraft, GuiSessionState, MarkerThresholdDraft


@dataclass
class BatchSampleDraft:
    """Editable, non-authoritative GUI row used before batch preflight."""

    sample_id: str
    animal_id: str
    source_path: Path
    declared_axes: str | None = None
    nuclear_channel: int | None = None
    marker_channels: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchPreflightRow:
    sample_id: str
    valid: bool
    message: str
    warning: str = ""


@dataclass(frozen=True)
class BatchPreflightReport:
    valid: bool
    rows: tuple[BatchPreflightRow, ...]
    preset_name: str
    pipeline_fingerprint: str | None
    manifest_fingerprint: str | None
    global_error: str | None = None


class BatchController:
    """Coordinates batch manifests, preset files, execution, and Strategy B checkpoints."""

    def __init__(self, state: GuiSessionState) -> None:
        self.state = state
        self.manifest_specs: list[BatchSampleSpec] = []
        self.sample_drafts: list[BatchSampleDraft] = []
        self._frozen_preset: AnalysisPreset | None = None
        self._frozen_manifest: BatchManifest | None = None
        self._frozen_manifest_fingerprint: str | None = None

    def add_sample_draft(self, file_path: Path | str) -> BatchSampleDraft:
        """Add one unvalidated GUI sample row without fabricating biological metadata."""
        p = Path(file_path)
        if not p.is_file():
            raise NerveNucGuiError(f"File not found: {p}")
        inspection = inspect_stack(p)
        draft = BatchSampleDraft(
            sample_id=p.stem,
            animal_id="",
            source_path=p,
            declared_axes=inspection.axes,
        )
        self.sample_drafts.append(draft)
        self.invalidate_preflight()
        return draft

    def update_sample_draft(
        self,
        index: int,
        *,
        sample_id: str,
        animal_id: str,
        declared_axes: str | None,
        nuclear_channel: int | None,
        marker_channels: Mapping[str, int],
    ) -> None:
        if index < 0 or index >= len(self.sample_drafts):
            raise NerveNucGuiError(f"Batch row index {index} is out of range.")
        draft = self.sample_drafts[index]
        draft.sample_id = sample_id.strip()
        draft.animal_id = animal_id.strip()
        draft.declared_axes = declared_axes.strip().upper() if declared_axes and declared_axes.strip() else None
        draft.nuclear_channel = nuclear_channel
        draft.marker_channels = dict(marker_channels)
        self.invalidate_preflight()

    def remove_sample_draft(self, index: int) -> None:
        if 0 <= index < len(self.sample_drafts):
            self.sample_drafts.pop(index)
            self.invalidate_preflight()

    def invalidate_preflight(self) -> None:
        self._frozen_preset = None
        self._frozen_manifest = None
        self._frozen_manifest_fingerprint = None
        self.state.results_stale = True
        self.state.configuration_differs_from_run = True
        self.state.notify()

    def _build_current_preset(self) -> AnalysisPreset:
        if self.state.active_preset is not None and not self.state.preset_dirty:
            return self.state.active_preset
        mode = self.state.draft.mode
        if mode == "direct_2d":
            if not self.state.draft.marker_channels:
                raise NerveNucGuiError(
                    "The current direct-2D configuration has no explicit marker names. "
                    "Load a preset or configure at least one real marker before batch validation."
                )
            return AnalysisPreset("Ad-hoc Preset", self.state.draft.build_core_2d_config())
        if mode == "projected_2d":
            return AnalysisPreset(
                "Ad-hoc Preset",
                self.state.draft.build_core_2d_config(),
                preparation=self.state.draft.build_core_stack_preparation_config(),
            )
        return AnalysisPreset("Ad-hoc Preset", self.state.draft.build_core_volume_config())

    @staticmethod
    def _mode_for_preset(preset: AnalysisPreset) -> str:
        return {1: "direct_2d", 2: "projected_2d", 3: "volume_3d"}[int(preset.schema_version)]

    def preflight_batch(self) -> BatchPreflightReport:
        """Validate every GUI sample and freeze one preset plus one manifest for Run."""
        rows: list[BatchPreflightRow] = []
        try:
            preset = self._build_current_preset()
        except Exception as exc:
            return BatchPreflightReport(False, (), "", None, None, str(exc))

        expected_mode = self._mode_for_preset(preset)
        if self.state.draft.mode != expected_mode:
            msg = (
                f"Active mode '{self.state.draft.mode}' does not match preset mode '{expected_mode}'."
            )
            return BatchPreflightReport(False, (), preset.name, preset.pipeline_fingerprint, None, msg)
        if not self.sample_drafts:
            return BatchPreflightReport(
                False, (), preset.name, preset.pipeline_fingerprint, None, "Batch contains no samples."
            )

        seen_ids: set[str] = set()
        specs: list[BatchSampleSpec] = []
        expected_markers = {marker.name for marker in preset.config.markers}

        for draft in self.sample_drafts:
            warning = ""
            try:
                if not draft.sample_id:
                    raise NerveNucGuiError("Sample ID is required.")
                if draft.sample_id in seen_ids:
                    raise NerveNucGuiError(f"Duplicate sample ID '{draft.sample_id}'.")
                seen_ids.add(draft.sample_id)
                if not draft.animal_id:
                    raise NerveNucGuiError(
                        "Animal ID is required; NerveNuc does not fabricate biological replicate identity."
                    )
                if not draft.source_path.is_file():
                    raise NerveNucGuiError(f"Source file not found: {draft.source_path}")

                if preset.schema_version == 1:
                    if draft.nuclear_channel is None:
                        raise NerveNucGuiError("Exactly one nuclear anchor channel must be selected.")
                    mapped = set(draft.marker_channels)
                    if mapped != expected_markers:
                        missing = sorted(expected_markers - mapped)
                        extra = sorted(mapped - expected_markers)
                        details = []
                        if missing:
                            details.append(f"missing markers {missing}")
                        if extra:
                            details.append(f"unknown markers {extra}")
                        raise NerveNucGuiError("Channel mapping does not match the frozen preset: " + "; ".join(details))
                    direct_input = ChannelMappedFileInput(
                        draft.source_path,
                        axes=draft.declared_axes,
                        nuclear_channel=draft.nuclear_channel,
                        marker_channels=draft.marker_channels,
                    )
                    prepare_direct_image(draft.source_path, direct_input.selection)
                    spec = BatchSampleSpec(
                        sample_id=draft.sample_id,
                        animal_id=draft.animal_id,
                        direct_image_input=direct_input,
                    )
                else:
                    selection = preset.preparation if preset.schema_version == 2 else preset.config.selection
                    inspection = inspect_stack(draft.source_path, declared_axes=selection.axes)
                    layout = resolve_axes(
                        inspection.shape,
                        declared_axes=selection.axes,
                        metadata_axes=inspection.axes,
                        is_ome=inspection.is_ome,
                    )
                    if layout.index_of("Z") is None:
                        raise NerveNucGuiError(f"{expected_mode} requires a Z axis.")
                    c_count = layout.size_of("C")
                    if c_count is None:
                        raise NerveNucGuiError(f"{expected_mode} requires an explicit channel axis.")
                    selected_indices = [selection.nuclear_channel, *selection.marker_channels.values()]
                    if any(index < 0 or index >= c_count for index in selected_indices):
                        raise NerveNucGuiError(
                            f"Preset channel mapping {selected_indices} is out of range for {c_count} channels."
                        )
                    if preset.schema_version == 3 and inspection.pixel_size_z_um is None:
                        warning = "No complete voxel calibration; physical volume will be unavailable."
                    spec = BatchSampleSpec(
                        sample_id=draft.sample_id,
                        animal_id=draft.animal_id,
                        stack_input=StackFileInput(draft.source_path),
                    )

                specs.append(spec)
                rows.append(BatchPreflightRow(draft.sample_id, True, "Ready", warning))
            except Exception as exc:
                rows.append(BatchPreflightRow(draft.sample_id or "(missing sample ID)", False, str(exc), warning))

        if not all(row.valid for row in rows):
            self._frozen_preset = None
            self._frozen_manifest = None
            self._frozen_manifest_fingerprint = None
            return BatchPreflightReport(
                False, tuple(rows), preset.name, preset.pipeline_fingerprint, None
            )

        try:
            manifest = BatchManifest(specs)
            manifest_fp = compute_manifest_fingerprint(manifest)
            BatchRunner(
                preset,
                manifest,
                BatchRunConfig(review_mode="auto_accept", continue_on_error=True),
            )
        except Exception as exc:
            return BatchPreflightReport(
                False, tuple(rows), preset.name, preset.pipeline_fingerprint, None, str(exc)
            )

        self._frozen_preset = preset
        self._frozen_manifest = manifest
        self._frozen_manifest_fingerprint = manifest_fp
        return BatchPreflightReport(
            True, tuple(rows), preset.name, preset.pipeline_fingerprint, manifest_fp
        )

    def create_preflighted_runner(self) -> BatchRunner:
        """Create the authoritative auto-accept runner only for the unchanged preflight state."""
        if self._frozen_preset is None or self._frozen_manifest is None or self._frozen_manifest_fingerprint is None:
            raise NerveNucGuiError("Validate the batch successfully before Run.")
        if compute_manifest_fingerprint(self._frozen_manifest) != self._frozen_manifest_fingerprint:
            self.invalidate_preflight()
            raise NerveNucGuiError("A source changed after validation. Validate the batch again.")
        current = self._build_current_preset()
        if current.pipeline_fingerprint != self._frozen_preset.pipeline_fingerprint:
            self.invalidate_preflight()
            raise NerveNucGuiError("Analysis settings changed after validation. Validate the batch again.")
        runner = BatchRunner(
            self._frozen_preset,
            self._frozen_manifest,
            BatchRunConfig(review_mode="auto_accept", continue_on_error=True),
        )
        runner.freeze_execution_identity(self._frozen_manifest_fingerprint)
        self.state.batch_runner = runner
        return runner

    def commit_batch_result(self, result: BatchResult) -> bool:
        """Publish a study only when every preflighted sample completed successfully."""
        self.state.batch_result = result
        statuses = list(result.sample_status["status"])
        complete = bool(statuses) and all(status == "completed" for status in statuses)
        if complete:
            self.state.study_session = result.to_study_session()
            self.state.results_stale = False
            self.state.configuration_differs_from_run = False
        else:
            from nervenuc.workflow import StudyAnalysisSession

            self.state.study_session = StudyAnalysisSession()
            self.state.results_stale = True
            self.state.configuration_differs_from_run = True
        self.state.notify()
        return complete

    def add_sample_spec(self, sample_id: str, animal_id: str, file_path: Path | str) -> None:
        """Add a sample specification to the active batch manifest."""
        p = Path(file_path)
        if not p.is_file():
            raise NerveNucGuiError(f"File not found: {p}")

        if any(s.sample_id == sample_id for s in self.manifest_specs):
            raise NerveNucGuiError(f"Duplicate sample_id '{sample_id}' in batch manifest.")

        mode = self.state.draft.mode
        if mode == "direct_2d":
            inp = FileInput(p)
            # Create spec for 2D single file
            spec = BatchSampleSpec(sample_id=sample_id, animal_id=animal_id, nuclear_input=inp, marker_inputs={})
        else:
            # Projected or 3D stack input
            inp = StackFileInput(p)
            spec = BatchSampleSpec(sample_id=sample_id, animal_id=animal_id, stack_input=inp)


        self.manifest_specs.append(spec)
        self.state.notify()

    def remove_sample_spec(self, index: int) -> None:
        """Remove a sample specification by index."""
        if 0 <= index < len(self.manifest_specs):
            self.manifest_specs.pop(index)
            self.state.notify()

    def build_manifest(self) -> BatchManifest:
        """Build validated BatchManifest from current sample specifications."""
        if not self.manifest_specs:
            raise NerveNucGuiError("Batch manifest contains no samples.")
        return BatchManifest(self.manifest_specs)

    def save_current_preset(self, path: Path | str, name: str) -> Path:
        """Save the current draft configuration as a validated preset file."""
        p = Path(path)
        mode = self.state.draft.mode
        if mode == "direct_2d":
            cfg = self.state.draft.build_core_2d_config()
            preset = AnalysisPreset(name=name, config=cfg)
        elif mode == "projected_2d":
            cfg = self.state.draft.build_core_2d_config()
            prep = self.state.draft.build_core_stack_preparation_config()
            preset = AnalysisPreset(name=name, config=cfg, preparation=prep)
        elif mode == "volume_3d":
            vol_cfg = self.state.draft.build_core_volume_config()
            preset = AnalysisPreset(name=name, config=vol_cfg)

        save_preset(preset, p)
        self.state.active_preset = preset
        self.state.active_preset_path = p
        self.state.clear_preset_dirty()
        return p

    def load_preset_file(self, path: Path | str) -> AnalysisPreset:
        """Load an existing preset file and populate the draft state."""
        p = Path(path)
        preset = load_preset(p)
        self.state.active_preset = preset
        self.state.active_preset_path = p

        if preset.schema_version == 1:
            self.state.draft.mode = "direct_2d"
            configured_names = {m.name for m in preset.config.markers}
            self.state.draft.marker_channels = {
                name: index
                for name, index in self.state.draft.marker_channels.items()
                if name in configured_names
            }
        elif preset.schema_version == 2:
            self.state.draft.mode = "projected_2d"
            assert preset.preparation is not None
            prep = preset.preparation
            self.state.draft.declared_axes = prep.axes
            self.state.draft.time_index = prep.time_index
            self.state.draft.nuclear_channel = prep.nuclear_channel
            self.state.draft.marker_channels = dict(prep.marker_channels)
            self.state.draft.projection_method = prep.projection_method
            self.state.draft.z_start_1based = prep.z_selection.start + 1 if prep.z_selection else None
            self.state.draft.z_stop_1based = prep.z_selection.stop if prep.z_selection else None
        elif preset.schema_version == 3:
            self.state.draft.mode = "volume_3d"
            selection = preset.config.selection
            self.state.draft.declared_axes = selection.axes
            self.state.draft.time_index = selection.time_index
            self.state.draft.nuclear_channel = selection.nuclear_channel
            self.state.draft.marker_channels = dict(selection.marker_channels)
            self.state.draft.z_start_1based = selection.z_selection.start + 1 if selection.z_selection else None
            self.state.draft.z_stop_1based = selection.z_selection.stop if selection.z_selection else None

        self.state.draft.segmentation_values = dict(preset.config.to_dict().get("segmentation", {}))
        self.state.draft.threshold_drafts = {
            t.marker_name: MarkerThresholdDraft(
                marker_name=t.marker_name,
                cutoff=float(t.threshold),
                measurement_field=t.measurement,
                comparison_operator=t.comparison,
            )
            for t in preset.config.thresholds
        }
        self.state.draft.metric_drafts = [
            BiologicalMetricDraft(
                name=m.name,
                denominator_positive=tuple(k for k, v in m.denominator.states.items() if v),
                denominator_negative=tuple(k for k, v in m.denominator.states.items() if not v),
                numerator_positive=tuple(k for k, v in m.numerator.states.items() if v),
                numerator_negative=tuple(k for k, v in m.numerator.states.items() if not v),
            )
            for m in preset.config.metrics
        ]

        self.state.clear_preset_dirty()
        self.state.results_stale = True
        return preset

    def create_batch_runner(self, review_mode: str = "auto_accept") -> BatchRunner:
        """Instantiate BatchRunner with current preset and manifest."""
        manifest = self.build_manifest()
        if self.state.active_preset is None:
            # Build ad-hoc preset from draft
            mode = self.state.draft.mode
            if mode == "direct_2d":
                cfg = self.state.draft.build_core_2d_config()
                preset = AnalysisPreset(name="Ad-hoc Preset", config=cfg)
            elif mode == "projected_2d":
                cfg = self.state.draft.build_core_2d_config()
                preset = AnalysisPreset(
                    name="Ad-hoc Preset",
                    config=cfg,
                    preparation=self.state.draft.build_core_stack_preparation_config(),
                )
            else:
                cfg = self.state.draft.build_core_volume_config()
                preset = AnalysisPreset(name="Ad-hoc Preset", config=cfg)
        else:
            preset = self.state.active_preset

        run_cfg = BatchRunConfig(review_mode=review_mode)  # type: ignore[arg-type]
        runner = BatchRunner(preset=preset, manifest=manifest, run_config=run_cfg)
        self.state.batch_runner = runner
        return runner

    def save_batch_checkpoint(self, result: BatchResult, path: Path | str) -> Path:
        """Save a batch execution checkpoint."""
        p = Path(path)
        save_checkpoint(result, p)
        return p

    def resume_batch_checkpoint(
        self,
        checkpoint_path: Path | str,
        review_mode: str = "auto_accept",
    ) -> BatchResult:
        """Resume batch processing from a checkpoint file under Strategy B."""
        p = Path(checkpoint_path)
        manifest = self.build_manifest()
        if self.state.active_preset is None:
            raise NerveNucGuiError("A preset must be loaded to resume from a checkpoint.")

        run_cfg = BatchRunConfig(review_mode=review_mode)  # type: ignore[arg-type]
        return resume_batch(
            checkpoint_path=p,
            preset=self.state.active_preset,
            manifest=manifest,
            config=run_cfg,
        )
