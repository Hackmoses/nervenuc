"""Sequential batch runner engine orchestrating ImageAnalysisSession executions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from nervenuc.aggregation import SampleIdentity, SampleSummary
from nervenuc.io import load_tiff
from nervenuc.preparation import (
    compute_direct_selection_fingerprint,
    prepare_direct_image,
    prepare_stack,
    prepare_volume,
)
from nervenuc.qc import ReviewedSegmentationResult
from nervenuc.qc3d import ReviewedSegmentation3DResult
from nervenuc.export import QCSummaryRecord
from nervenuc.workflow import (
    AnalysisConfig,
    ImageAnalysisSession,
    ImageSourceProvenance,
    InvalidWorkflowStateError,
    VolumeAnalysisConfig,
    VolumeAnalysisSession,
    VolumeSourceProvenance,
    compute_pipeline_fingerprint,
    validate_preparation_analysis_compatibility,
    validate_volume_selection_analysis_compatibility,
)

from .errors import (
    InputSourceChangedError,
    InvalidBatchInputError,
    InvalidBatchManifestError,
    InvalidPresetError,
)
from .models import (
    STATUS_COLUMNS,
    ArrayInput,
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    BatchSampleSpec,
    ChannelMappedFileInput,
    FileInput,
    StackFileInput,
    _capture_manifest_input_identities,
    _compute_manifest_fingerprint_from_identities,
)
from .preset import AnalysisPreset


BatchProgressCallback = Callable[[str, str, int, int, int, int], None]


def resolve_input(
    input_desc: ArrayInput | FileInput,
    expected_content_hash: str | None = None,
) -> np.ndarray:
    """Resolve a typed input descriptor into an authoritative 2D NumPy array.

    Args:
        input_desc: ArrayInput or FileInput instance.
        expected_content_hash: Expected SHA-256 hash for file integrity check.

    Returns:
        2D numeric NumPy array.

    Raises:
        InvalidBatchInputError: If file loading fails, array is missing, or array is not 2D.
        InputSourceChangedError: If file hash has changed during or before loading.
    """
    if isinstance(input_desc, ArrayInput):
        if not isinstance(input_desc.array, np.ndarray) or input_desc.array.ndim != 2:
            raise InvalidBatchInputError("ArrayInput array must be a 2D numpy array.")
        return input_desc.array

    if isinstance(input_desc, FileInput):
        hash_before = input_desc.content_hash
        if expected_content_hash is not None and hash_before != expected_content_hash:
            raise InputSourceChangedError(
                f"FileInput source changed before loading: '{input_desc.path}'."
            )
        try:
            loaded = load_tiff(input_desc.path)
        except Exception as exc:
            raise InvalidBatchInputError(
                f"Failed to load TIFF file '{input_desc.path}': {exc}"
            ) from exc

        hash_after = input_desc.content_hash
        if hash_after != hash_before or (
            expected_content_hash is not None and hash_after != expected_content_hash
        ):
            raise InputSourceChangedError(
                f"FileInput source changed while loading: '{input_desc.path}'."
            )

        if loaded.data.ndim != 2:
            raise InvalidBatchInputError(
                f"TIFF file '{input_desc.path}' is not 2-dimensional (got shape {loaded.data.shape}, "
                f"ndim {loaded.data.ndim})."
            )
        return loaded.data

    raise InvalidBatchInputError(
        f"Unsupported input descriptor type: {type(input_desc).__name__}. Must be ArrayInput or FileInput."
    )


class BatchRunner:
    """Deterministic batch execution engine processing samples in manifest order."""

    def __init__(
        self,
        preset: Any,
        manifest: BatchManifest,
        run_config: BatchRunConfig | None = None,
    ) -> None:
        """Initialize the batch runner."""
        if isinstance(preset, AnalysisConfig):
            self.preset = AnalysisPreset(name="Anonymous Preset", config=preset)
        elif isinstance(preset, VolumeAnalysisConfig):
            self.preset = AnalysisPreset(name="Anonymous Preset", config=preset)
        elif isinstance(preset, AnalysisPreset):
            self.preset = preset
        else:
            raise InvalidPresetError(
                f"preset must be an AnalysisPreset, AnalysisConfig, or VolumeAnalysisConfig, got {type(preset).__name__}."
            )

        if not isinstance(manifest, BatchManifest):
            raise InvalidBatchManifestError(
                f"manifest must be a BatchManifest, got {type(manifest).__name__}."
            )

        self.manifest = manifest
        self.run_config = run_config if run_config is not None else BatchRunConfig()

        # Enforce strict no-mixing batch policy
        has_stack_samples = any(s.stack_input is not None for s in self.manifest.samples)
        has_2d_samples = any(s.stack_input is None for s in self.manifest.samples)

        if has_stack_samples and has_2d_samples:
            raise InvalidBatchManifestError(
                "BatchManifest contains mixed 2D and Z-stack samples. Mixed batches are not permitted."
            )

        if self.preset.schema_version == 1 and has_stack_samples:
            raise InvalidBatchManifestError(
                "Batch preset is 2D-only (schema 1), but manifest contains Z-stack samples."
            )

        if self.preset.schema_version == 2 and has_2d_samples:
            raise InvalidBatchManifestError(
                "Batch preset contains stack preparation (schema 2), but manifest contains 2D samples."
            )

        if self.preset.schema_version == 3 and has_2d_samples:
            raise InvalidBatchManifestError(
                "Batch preset is 3D volume analysis (schema 3), but manifest contains 2D samples."
            )

        if self.preset.schema_version == 1:
            expected_markers = {marker.name for marker in self.preset.config.markers}
            for sample in self.manifest.samples:
                if sample.direct_image_input is not None:
                    mapped_markers = set(sample.direct_image_input.selection.marker_channels)
                    if mapped_markers != expected_markers:
                        raise InvalidBatchManifestError(
                            f"Sample '{sample.sample_id}' direct channel mapping markers {sorted(mapped_markers)} "
                            f"do not match preset markers {sorted(expected_markers)}."
                        )

        # Validate preparation/analysis marker compatibility early if stack preset
        if self.preset.schema_version == 2 and self.preset.preparation is not None:
            validate_preparation_analysis_compatibility(self.preset.preparation, self.preset.config)
        elif self.preset.schema_version == 3 and isinstance(self.preset.config, VolumeAnalysisConfig):
            validate_volume_selection_analysis_compatibility(self.preset.config.selection, self.preset.config)

        # In-memory execution state tracking
        self._status_records: dict[str, dict[str, Any]] = {}
        for s in self.manifest.samples:
            self._status_records[s.sample_id] = {
                "sample_id": s.sample_id,
                "animal_id": s.animal_id,
                "status": "pending",
                "error_type": None,
                "error_message": None,
            }

        self._image_sessions: dict[str, ImageAnalysisSession] = {}
        self._volume_sessions: dict[str, VolumeAnalysisSession] = {}
        self._review_sessions: dict[str, Any] = {}
        self._completed_samples: dict[str, SampleSummary] = {}
        self._sample_provenance: dict[str, ImageSourceProvenance | VolumeSourceProvenance] = {}
        self._sample_qc_records: dict[str, QCSummaryRecord] = {}
        self._captured_input_identities: dict[tuple[str, str], str] | None = None
        self._manifest_fingerprint: str | None = None

    def _capture_execution_identity(self) -> None:
        """Capture source identities once for the lifetime of this runner."""
        if self._manifest_fingerprint is not None:
            return
        identities = _capture_manifest_input_identities(self.manifest)
        self._captured_input_identities = identities
        self._manifest_fingerprint = _compute_manifest_fingerprint_from_identities(
            self.manifest,
            identities,
        )

    def freeze_execution_identity(self, expected_manifest_fingerprint: str | None = None) -> str:
        """Capture source identities now and optionally require an earlier preflight fingerprint."""
        self._capture_execution_identity()
        assert self._manifest_fingerprint is not None
        if (
            expected_manifest_fingerprint is not None
            and self._manifest_fingerprint != expected_manifest_fingerprint
        ):
            raise InputSourceChangedError(
                "Batch sources or direct channel mappings changed after preflight validation."
            )
        return self._manifest_fingerprint

    def _record_completed_qc(self, sample_id: str, session: Any) -> None:
        """Capture factual finalized QC counts before live batch sessions are discarded."""
        reviewed = session.reviewed_result
        source = getattr(reviewed, "source_result", None)
        initial_count = len(source.objects) if source is not None and hasattr(source, "objects") else 0
        final_objects = reviewed.objects
        final_count = len(final_objects)
        history = getattr(reviewed, "history", ())
        rejected_count = sum(
            1 for action in history if getattr(action, "action_type", "") in {"reject", "delete"}
        )
        if "review_origin" in final_objects.columns:
            accepted_count = int((final_objects["review_origin"] == "automated").sum())
            added_count = int((final_objects["review_origin"] == "manual_add").sum())
            edited_count = int(final_count - accepted_count - added_count)
        else:
            accepted_count = final_count
            added_count = 0
            edited_count = 0
        self._sample_qc_records[sample_id] = QCSummaryRecord(
            sample_id=sample_id,
            initial_automated_count=initial_count,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            added_count=added_count,
            edited_count=edited_count,
            final_count=final_count,
            review_state="auto_accepted" if self.run_config.review_mode == "auto_accept" else "completed",
        )

    def _resolve_sample_inputs(
        self,
        spec: BatchSampleSpec,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load one 2D sample against the execution's captured source identities."""
        assert self._captured_input_identities is not None
        assert spec.nuclear_input is not None
        assert spec.marker_inputs is not None
        sid = spec.sample_id
        nuclear_hash = self._captured_input_identities[(sid, "nuclear")]
        nuclear = resolve_input(
            spec.nuclear_input,
            nuclear_hash if isinstance(spec.nuclear_input, FileInput) else None,
        )
        markers = {
            marker_name: resolve_input(
                marker_input,
                self._captured_input_identities[(sid, f"marker:{marker_name}")]
                if isinstance(marker_input, FileInput)
                else None,
            )
            for marker_name, marker_input in spec.marker_inputs.items()
        }
        return nuclear, markers

    def _verify_sample_sources_unchanged(self, sample_id: str) -> None:
        """Verify pending file-backed review inputs still match loaded bytes."""
        assert self._captured_input_identities is not None
        spec = next(s for s in self.manifest.samples if s.sample_id == sample_id)

        if spec.stack_input is not None:
            expected = self._captured_input_identities[(sample_id, "stack")]
            current = spec.stack_input.content_hash
            if current != expected:
                raise InputSourceChangedError(
                    f"Source file changed while sample '{sample_id}' awaited review: "
                    f"'{spec.stack_input.path}'. Start a fresh batch review for the changed input."
                )
            return

        if spec.direct_image_input is not None:
            expected = self._captured_input_identities[(sample_id, "direct_image")]
            current = spec.direct_image_input.content_hash
            if current != expected:
                raise InputSourceChangedError(
                    f"Direct image source changed while sample '{sample_id}' awaited review: "
                    f"'{spec.direct_image_input.path}'. Start a fresh batch review for the changed input."
                )
            return

        assert spec.nuclear_input is not None
        assert spec.marker_inputs is not None
        descriptors = [("nuclear", spec.nuclear_input)] + [
            (f"marker:{name}", descriptor)
            for name, descriptor in spec.marker_inputs.items()
        ]
        for role, descriptor in descriptors:
            if isinstance(descriptor, FileInput):
                expected = self._captured_input_identities[(sample_id, role)]
                current = descriptor.content_hash
                if current != expected:
                    raise InputSourceChangedError(
                        f"Source file changed while sample '{sample_id}' awaited review: "
                        f"'{descriptor.path}'. Start a fresh batch review for the changed input."
                    )

    def run(self, progress_callback: BatchProgressCallback | None = None) -> BatchResult:
        """Execute the batch sequentially across manifest samples."""
        self._capture_execution_identity()

        total = len(self.manifest.samples)
        for index, spec in enumerate(self.manifest.samples, start=1):
            sid = spec.sample_id
            aid = spec.animal_id

            # Completed and live review sessions are stable across repeated run() calls.
            if self._status_records[sid]["status"] in {"completed", "review_pending"}:
                continue

            self._status_records[sid]["status"] = "running"
            if progress_callback is not None:
                completed = sum(r["status"] == "completed" for r in self._status_records.values())
                failed = sum(r["status"] == "failed" for r in self._status_records.values())
                progress_callback(sid, "running", index, total, completed, failed)

            try:
                if self.preset.schema_version == 3:
                    # Schema 3: 3D volume batch execution
                    assert isinstance(self.preset.config, VolumeAnalysisConfig)
                    assert spec.stack_input is not None
                    assert self._captured_input_identities is not None

                    stack_hash_before = self._captured_input_identities[(sid, "stack")]
                    prepared = prepare_volume(spec.stack_input.path, self.preset.config.selection)

                    stack_hash_after = spec.stack_input.content_hash
                    if stack_hash_after != stack_hash_before:
                        raise InputSourceChangedError(
                            f"StackFileInput source changed while loading: '{spec.stack_input.path}'."
                        )

                    vol_session = VolumeAnalysisSession()
                    vol_session.set_volume(
                        volume=prepared,
                        source_id=sid,
                        source_hash=stack_hash_before,
                        config_fingerprint=self.preset.config_fingerprint,
                        pipeline_fingerprint=self.preset.pipeline_fingerprint,
                        analysis_config=self.preset.config,
                        qc_review_policy=self.run_config.review_mode,
                        manual_per_object_qc_performed=(
                            False if self.run_config.review_mode == "auto_accept" else None
                        ),
                    )
                    assert vol_session.source_provenance is not None
                    self._sample_provenance[sid] = vol_session.source_provenance

                    vol_session.segment(self.preset.config.segmentation)

                    if self.run_config.review_mode == "manual":
                        rev_session = vol_session.start_review()
                        self._volume_sessions[sid] = vol_session
                        self._review_sessions[sid] = rev_session
                        self._status_records[sid]["status"] = "review_pending"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None

                    elif self.run_config.review_mode == "auto_accept":
                        vol_session.auto_accept()
                        vol_session.measure(self.preset.config.markers)
                        vol_session.classify(self.preset.config.thresholds)
                        summary = vol_session.summarize(self.preset.config.metrics)
                        sample_summary = SampleSummary(
                            identity=SampleIdentity(sid, aid),
                            summary_result=summary,
                        )
                        self._completed_samples[sid] = sample_summary
                        self._record_completed_qc(sid, vol_session)
                        self._status_records[sid]["status"] = "completed"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None
                        self._review_sessions.pop(sid, None)
                        self._volume_sessions.pop(sid, None)

                elif self.preset.schema_version == 2:
                    # Schema 2: Projected stack execution
                    assert self.preset.preparation is not None
                    assert spec.stack_input is not None
                    assert self._captured_input_identities is not None
                    session = ImageAnalysisSession(self.preset.config)
                    stack_hash_before = self._captured_input_identities[(sid, "stack")]

                    prepared_img = prepare_stack(spec.stack_input.path, self.preset.preparation)

                    # Verify disk file didn't mutate during load
                    stack_hash_after = spec.stack_input.content_hash
                    if stack_hash_after != stack_hash_before:
                        raise InputSourceChangedError(
                            f"StackFileInput source changed while loading: '{spec.stack_input.path}'."
                        )

                    nuc_prov = prepared_img.provenance["nuclear"]
                    prov = ImageSourceProvenance(
                        source_kind="projected_stack",
                        source_id=sid,
                        source_hash=stack_hash_before,
                        metadata_axes=nuc_prov.metadata_axes,
                        declared_axes=nuc_prov.declared_axes,
                        effective_axes=nuc_prov.effective_axes,
                        time_index=nuc_prov.time_index,
                        nuclear_channel=self.preset.preparation.nuclear_channel,
                        marker_channels=dict(self.preset.preparation.marker_channels),
                        z_start=nuc_prov.z_start,
                        z_stop=nuc_prov.z_stop,
                        projection_method=nuc_prov.method,
                        physical_size_z_um=nuc_prov.physical_size_z_um,
                        preparation_fingerprint=self.preset.preparation_fingerprint,
                        pipeline_fingerprint=self.preset.pipeline_fingerprint,
                        qc_review_policy=self.run_config.review_mode,
                        manual_per_object_qc_performed=(
                            False if self.run_config.review_mode == "auto_accept" else None
                        ),
                    )
                    session.set_prepared_image_set(prepared_img, source_provenance=prov)
                    self._sample_provenance[sid] = prov

                    session.segment()

                    if self.run_config.review_mode == "manual":
                        rev_session = session.start_review()
                        self._image_sessions[sid] = session
                        self._review_sessions[sid] = rev_session
                        self._status_records[sid]["status"] = "review_pending"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None

                    elif self.run_config.review_mode == "auto_accept":
                        session.auto_accept_segmentation()
                        session.measure()
                        session.classify()
                        summary = session.summarize()
                        sample_summary = SampleSummary(
                            identity=SampleIdentity(sid, aid),
                            summary_result=summary,
                        )
                        self._completed_samples[sid] = sample_summary
                        self._record_completed_qc(sid, session)
                        self._status_records[sid]["status"] = "completed"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None
                        self._review_sessions.pop(sid, None)
                        self._image_sessions.pop(sid, None)

                else:
                    # Schema 1: Direct 2D path
                    session = ImageAnalysisSession(self.preset.config)
                    if spec.direct_image_input is not None:
                        assert self._captured_input_identities is not None
                        direct_input = spec.direct_image_input
                        source_hash = self._captured_input_identities[(sid, "direct_image")]
                        prepared = prepare_direct_image(direct_input.path, direct_input.selection)
                        if direct_input.content_hash != source_hash:
                            raise InputSourceChangedError(
                                f"Direct image source changed while loading: '{direct_input.path}'."
                            )
                        nuc_img = prepared.nuclear_image
                        marker_imgs = dict(prepared.marker_images)
                        selection_fp = compute_direct_selection_fingerprint(direct_input.selection)
                        pipeline_fp = compute_pipeline_fingerprint(
                            self.preset.config_fingerprint,
                            selection_fp,
                        )
                        prov = ImageSourceProvenance(
                            source_kind="direct_2d",
                            source_id=sid,
                            source_hash=source_hash,
                            metadata_axes=prepared.provenance.metadata_axes,
                            declared_axes=prepared.provenance.declared_axes,
                            effective_axes=prepared.provenance.effective_axes,
                            nuclear_channel=direct_input.selection.nuclear_channel,
                            marker_channels=dict(direct_input.selection.marker_channels),
                            preparation_fingerprint=selection_fp,
                            pipeline_fingerprint=pipeline_fp,
                            qc_review_policy=self.run_config.review_mode,
                            manual_per_object_qc_performed=(
                                False if self.run_config.review_mode == "auto_accept" else None
                            ),
                        )
                    else:
                        nuc_img, marker_imgs = self._resolve_sample_inputs(spec)
                        prov = ImageSourceProvenance(
                            source_kind="direct_2d",
                            source_id=sid,
                            pipeline_fingerprint=self.preset.pipeline_fingerprint,
                            qc_review_policy=self.run_config.review_mode,
                            manual_per_object_qc_performed=(
                                False if self.run_config.review_mode == "auto_accept" else None
                            ),
                        )
                    session.set_image(nuc_img, marker_imgs, source_provenance=prov)
                    self._sample_provenance[sid] = prov

                    session.segment()

                    if self.run_config.review_mode == "manual":
                        rev_session = session.start_review()
                        self._image_sessions[sid] = session
                        self._review_sessions[sid] = rev_session
                        self._status_records[sid]["status"] = "review_pending"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None

                    elif self.run_config.review_mode == "auto_accept":
                        session.auto_accept_segmentation()
                        session.measure()
                        session.classify()
                        summary = session.summarize()
                        sample_summary = SampleSummary(
                            identity=SampleIdentity(sid, aid),
                            summary_result=summary,
                        )
                        self._completed_samples[sid] = sample_summary
                        self._record_completed_qc(sid, session)
                        self._status_records[sid]["status"] = "completed"
                        self._status_records[sid]["error_type"] = None
                        self._status_records[sid]["error_message"] = None
                        self._review_sessions.pop(sid, None)
                        self._image_sessions.pop(sid, None)

            except Exception as exc:
                self._status_records[sid]["status"] = "failed"
                self._status_records[sid]["error_type"] = type(exc).__name__
                self._status_records[sid]["error_message"] = str(exc)
                self._review_sessions.pop(sid, None)
                self._image_sessions.pop(sid, None)
                self._volume_sessions.pop(sid, None)

                if not self.run_config.continue_on_error:
                    break

            if progress_callback is not None:
                status = str(self._status_records[sid]["status"])
                completed = sum(r["status"] == "completed" for r in self._status_records.values())
                failed = sum(r["status"] == "failed" for r in self._status_records.values())
                progress_callback(sid, status, index, total, completed, failed)

        return self._build_result()

    def continue_sample(
        self,
        sample_id: str,
        reviewed_result: Any,
    ) -> BatchResult:
        """Finalize manual review for a sample in 'review_pending' status and continue execution.

        Args:
            sample_id: Identifier of the sample to continue.
            reviewed_result: Finalized ReviewedSegmentationResult or ReviewedSegmentation3DResult.

        Returns:
            Updated BatchResult object.
        """
        if sample_id not in self._status_records:
            raise InvalidWorkflowStateError(f"Sample '{sample_id}' not found in batch manifest.")

        if self._status_records[sample_id]["status"] != "review_pending":
            raise InvalidWorkflowStateError(
                f"Cannot continue sample '{sample_id}' with status '{self._status_records[sample_id]['status']}' "
                f"(expected 'review_pending')."
            )

        aid = self._status_records[sample_id]["animal_id"]

        if sample_id in self._volume_sessions:
            vol_session = self._volume_sessions[sample_id]
            self._verify_sample_sources_unchanged(sample_id)
            assert isinstance(self.preset.config, VolumeAnalysisConfig)
            try:
                vol_session.set_reviewed_result(reviewed_result)
                vol_session.measure(self.preset.config.markers)
                vol_session.classify(self.preset.config.thresholds)
                summary = vol_session.summarize(self.preset.config.metrics)
                sample_summary = SampleSummary(
                    identity=SampleIdentity(sample_id, aid),
                    summary_result=summary,
                )
                self._completed_samples[sample_id] = sample_summary
                self._record_completed_qc(sample_id, vol_session)
                self._status_records[sample_id]["status"] = "completed"
                self._status_records[sample_id]["error_type"] = None
                self._status_records[sample_id]["error_message"] = None
                self._review_sessions.pop(sample_id, None)
                self._volume_sessions.pop(sample_id, None)
            except Exception as exc:
                self._status_records[sample_id]["status"] = "failed"
                self._status_records[sample_id]["error_type"] = type(exc).__name__
                self._status_records[sample_id]["error_message"] = str(exc)
                self._review_sessions.pop(sample_id, None)
                self._volume_sessions.pop(sample_id, None)
                raise exc

            return self._build_result()

        if sample_id not in self._image_sessions:
            raise InvalidWorkflowStateError(
                f"No active analysis session available for review_pending sample '{sample_id}'."
            )

        session = self._image_sessions[sample_id]

        # Verify source file integrity on disk
        self._verify_sample_sources_unchanged(sample_id)

        # Step 1: Bind reviewed result (validates source provenance and shape)
        session.set_reviewed_result(reviewed_result)

        # Step 2: Run downstream scientific execution
        try:
            session.measure()
            session.classify()
            summary = session.summarize()
            sample_summary = SampleSummary(
                identity=SampleIdentity(sample_id, aid),
                summary_result=summary,
            )
            self._completed_samples[sample_id] = sample_summary
            self._record_completed_qc(sample_id, session)
            self._status_records[sample_id]["status"] = "completed"
            self._status_records[sample_id]["error_type"] = None
            self._status_records[sample_id]["error_message"] = None
            self._review_sessions.pop(sample_id, None)
            self._image_sessions.pop(sample_id, None)
        except Exception as exc:
            self._status_records[sample_id]["status"] = "failed"
            self._status_records[sample_id]["error_type"] = type(exc).__name__
            self._status_records[sample_id]["error_message"] = str(exc)
            self._review_sessions.pop(sample_id, None)
            self._image_sessions.pop(sample_id, None)
            raise exc

        return self._build_result()


    def _build_result(self) -> BatchResult:
        """Construct an invariant-validated BatchResult snapshot."""
        self._capture_execution_identity()
        assert self._manifest_fingerprint is not None
        rows = [self._status_records[s.sample_id] for s in self.manifest.samples]
        status_df = pd.DataFrame(rows, columns=STATUS_COLUMNS)

        # Preserve strict manifest ordering for completed samples
        ordered_completed = tuple(
            self._completed_samples[s.sample_id]
            for s in self.manifest.samples
            if s.sample_id in self._completed_samples
        )

        res = BatchResult(
            sample_status=status_df.copy(deep=True),
            completed_samples=ordered_completed,
            review_sessions=dict(self._review_sessions),
            preset=self.preset,
            config_fingerprint=self.preset.config_fingerprint,
            manifest_fingerprint=self._manifest_fingerprint,
            preparation_fingerprint=self.preset.preparation_fingerprint,
            pipeline_fingerprint=self.preset.pipeline_fingerprint,
            review_mode=self.run_config.review_mode,
            sample_provenance=dict(self._sample_provenance),
            sample_qc_records=dict(self._sample_qc_records),
        )
        res.validate_invariants(self.manifest)
        return res


def run_batch(
    preset: AnalysisPreset | AnalysisConfig,
    manifest: BatchManifest,
    run_config: BatchRunConfig | None = None,
) -> BatchResult:
    """Execute a batch analysis over manifest samples.

    Args:
        preset: AnalysisPreset or AnalysisConfig.
        manifest: BatchManifest defining samples.
        run_config: Execution policy.

    Returns:
        BatchResult object.
    """
    runner = BatchRunner(preset, manifest, run_config)
    return runner.run()
