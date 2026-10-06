"""Models for batch manifests, typed input descriptors, execution configuration, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from nervenuc.aggregation import SampleSummary
from nervenuc.workflow import ImageSourceProvenance, StudyAnalysisSession
from nervenuc.export import QCSummaryRecord
from nervenuc.preparation import DirectImageSelectionConfig

from .errors import (
    InvalidBatchInputError,
    InvalidBatchManifestError,
    NerveNucBatchError,
)

if TYPE_CHECKING:
    from nervenuc.qc import ReviewSession
    from .preset import AnalysisPreset


STATUS_COLUMNS: list[str] = [
    "sample_id",
    "animal_id",
    "status",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class ArrayInput:
    """In-memory 2D NumPy array with a caller-declared stable identifier.

    Attributes:
        array: Explicit 2D numeric NumPy array.
        input_id: Caller-declared stable identifier for fingerprinting and provenance tracking.
    """

    array: np.ndarray
    input_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray):
            raise InvalidBatchInputError(
                f"ArrayInput array must be a numpy.ndarray, got {type(self.array).__name__}."
            )
        if self.array.ndim != 2:
            raise InvalidBatchInputError(
                f"ArrayInput array must be 2D, got ndim={self.array.ndim} with shape {self.array.shape}."
            )
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise InvalidBatchInputError("ArrayInput input_id must be a non-empty string.")


@dataclass(frozen=True)
class FileInput:
    """File path to a single-plane 2D TIFF file with content-aware SHA-256 fingerprinting.

    Attributes:
        path: Path to the image file.
    """

    path: Path

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise InvalidBatchInputError(f"FileInput path must be str or Path, got {type(path).__name__}.")
        if isinstance(path, str) and not path.strip():
            raise InvalidBatchInputError("FileInput path cannot be empty or whitespace-only.")
        p = Path(path)
        if p.exists() and p.is_dir():
            raise InvalidBatchInputError(f"FileInput path must not be a directory: '{p}'.")
        object.__setattr__(self, "path", p)

    @property
    def content_hash(self) -> str:
        """Calculate dynamic SHA-256 of raw file bytes for content identity."""
        try:
            resolved = self.path.resolve()
            if not resolved.exists():
                raise InvalidBatchInputError(f"FileInput file does not exist: '{resolved}'.")
            if not resolved.is_file():
                raise InvalidBatchInputError(f"FileInput path is not a regular file: '{resolved}'.")
            h = hashlib.sha256()
            with open(resolved, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except InvalidBatchInputError:
            raise
        except OSError as exc:
            raise InvalidBatchInputError(
                f"Cannot hash FileInput file '{self.path}': {exc}"
            ) from exc


@dataclass(frozen=True)
class StackFileInput:
    """File path to a multi-channel microscopy Z-stack TIFF with content-aware SHA-256 fingerprinting.

    Attributes:
        path: Path to the Z-stack image file.
    """

    path: Path

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise InvalidBatchInputError(f"StackFileInput path must be str or Path, got {type(path).__name__}.")
        if isinstance(path, str) and not path.strip():
            raise InvalidBatchInputError("StackFileInput path cannot be empty or whitespace-only.")
        p = Path(path)
        if p.exists() and p.is_dir():
            raise InvalidBatchInputError(f"StackFileInput path must not be a directory: '{p}'.")
        object.__setattr__(self, "path", p)

    @property
    def content_hash(self) -> str:
        """Calculate dynamic SHA-256 of raw file bytes for content identity."""
        try:
            resolved = self.path.resolve()
            if not resolved.exists():
                raise InvalidBatchInputError(f"StackFileInput file does not exist: '{resolved}'.")
            if not resolved.is_file():
                raise InvalidBatchInputError(f"StackFileInput path is not a regular file: '{resolved}'.")
            h = hashlib.sha256()
            with open(resolved, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except InvalidBatchInputError:
            raise
        except OSError as exc:
            raise InvalidBatchInputError(
                f"Cannot hash StackFileInput file '{self.path}': {exc}"
            ) from exc


@dataclass(frozen=True)
class ChannelMappedFileInput:
    """One multichannel direct-2D TIFF plus explicit nuclear/marker channel selection."""

    path: Path
    selection: DirectImageSelectionConfig

    def __init__(
        self,
        path: str | Path,
        *,
        axes: str | None,
        nuclear_channel: int,
        marker_channels: Mapping[str, int],
    ) -> None:
        file_input = FileInput(path)
        object.__setattr__(self, "path", file_input.path)
        object.__setattr__(
            self,
            "selection",
            DirectImageSelectionConfig(
                axes=axes,
                nuclear_channel=nuclear_channel,
                marker_channels=dict(marker_channels),
            ),
        )

    @property
    def content_hash(self) -> str:
        """Calculate the current SHA-256 identity of the raw multichannel file."""
        return FileInput(self.path).content_hash


@dataclass(frozen=True)
class BatchSampleSpec:
    """Explicit declaration of a single sample in a batch analysis.

    Supports explicit 2D planes, one channel-mapped direct-2D file, or one stack input,
    while strictly rejecting ambiguous or mixed configurations.

    Attributes:
        sample_id: Unique identifier for the sample/image.
        animal_id: Biological replicate animal identifier.
        nuclear_input: Typed 2D nuclear channel input descriptor (for 2D workflows).
        marker_inputs: Mapping from marker names to typed 2D marker input descriptors (for 2D workflows).
        stack_input: Typed StackFileInput descriptor (for Z-stack workflows).
    """

    sample_id: str
    animal_id: str
    nuclear_input: ArrayInput | FileInput | None = None
    marker_inputs: Mapping[str, ArrayInput | FileInput] | None = None
    direct_image_input: ChannelMappedFileInput | None = None
    stack_input: StackFileInput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise InvalidBatchManifestError("BatchSampleSpec sample_id must be a non-empty string.")
        if not isinstance(self.animal_id, str) or not self.animal_id.strip():
            raise InvalidBatchManifestError("BatchSampleSpec animal_id must be a non-empty string.")

        is_2d = (
            self.nuclear_input is not None
            and self.marker_inputs is not None
            and self.direct_image_input is None
            and self.stack_input is None
        )
        is_direct_file = (
            self.direct_image_input is not None
            and self.nuclear_input is None
            and self.marker_inputs is None
            and self.stack_input is None
        )
        is_stack = (
            self.stack_input is not None
            and self.nuclear_input is None
            and self.marker_inputs is None
            and self.direct_image_input is None
        )

        if sum((is_2d, is_direct_file, is_stack)) != 1:
            raise InvalidBatchManifestError(
                f"Sample '{self.sample_id}' must provide exactly one input form: explicit 2D planes, "
                "a channel-mapped direct-2D file, or a Z-stack file."
            )

        if is_2d:
            if not isinstance(self.nuclear_input, (ArrayInput, FileInput)):
                raise InvalidBatchInputError(
                    f"BatchSampleSpec nuclear_input must be ArrayInput or FileInput, got {type(self.nuclear_input).__name__}."
                )
            if not isinstance(self.marker_inputs, Mapping):
                raise InvalidBatchInputError(
                    f"BatchSampleSpec marker_inputs must be a Mapping, got {type(self.marker_inputs).__name__}."
                )
            for k, v in self.marker_inputs.items():
                if not isinstance(k, str) or not k.strip():
                    raise InvalidBatchManifestError("Marker name in marker_inputs must be a non-empty string.")
                if not isinstance(v, (ArrayInput, FileInput)):
                    raise InvalidBatchInputError(
                        f"Marker input for '{k}' must be ArrayInput or FileInput, got {type(v).__name__}."
                    )
        elif is_direct_file:
            if not isinstance(self.direct_image_input, ChannelMappedFileInput):
                raise InvalidBatchInputError(
                    "direct_image_input must be a ChannelMappedFileInput."
                )
        elif is_stack:
            if not isinstance(self.stack_input, StackFileInput):
                raise InvalidBatchInputError(
                    f"BatchSampleSpec stack_input must be StackFileInput, got {type(self.stack_input).__name__}."
                )


@dataclass(frozen=True)
class BatchManifest:
    """Immutable collection of sample specifications defining a batch run.

    Attributes:
        samples: Tuple of unique BatchSampleSpec objects in authoritative execution order.
    """

    samples: tuple[BatchSampleSpec, ...]

    def __init__(self, samples: Sequence[BatchSampleSpec]) -> None:
        if not isinstance(samples, Sequence):
            raise InvalidBatchManifestError("BatchManifest samples must be a sequence.")
        object.__setattr__(self, "samples", tuple(samples))
        self.validate()

    def validate(self) -> None:
        """Validate structural and uniqueness constraints of the manifest."""
        if not self.samples:
            raise InvalidBatchManifestError("BatchManifest cannot be empty.")
        seen_ids: set[str] = set()
        for idx, s in enumerate(self.samples):
            if not isinstance(s, BatchSampleSpec):
                raise InvalidBatchManifestError(
                    f"Item at index {idx} in BatchManifest is not a BatchSampleSpec: {type(s).__name__}."
                )
            if s.sample_id in seen_ids:
                raise InvalidBatchManifestError(
                    f"Duplicate sample_id '{s.sample_id}' found in BatchManifest."
                )
            seen_ids.add(s.sample_id)


def compute_manifest_fingerprint(manifest: BatchManifest) -> str:
    """Compute a deterministic SHA-256 fingerprint over manifest sample IDs and input descriptors.

    Args:
        manifest: Valid BatchManifest instance.

    Returns:
        Hexadecimal SHA-256 string preserving manifest order.
    """
    if not isinstance(manifest, BatchManifest):
        raise InvalidBatchManifestError(
            f"manifest must be a BatchManifest, got {type(manifest).__name__}."
        )

    identities = _capture_manifest_input_identities(manifest)
    return _compute_manifest_fingerprint_from_identities(manifest, identities)


def _capture_manifest_input_identities(
    manifest: BatchManifest,
) -> dict[tuple[str, str], str]:
    """Capture authoritative input identifiers for one batch execution."""
    identities: dict[tuple[str, str], str] = {}
    for s in manifest.samples:
        if s.stack_input is not None:
            identities[(s.sample_id, "stack")] = s.stack_input.content_hash
        elif s.direct_image_input is not None:
            identities[(s.sample_id, "direct_image")] = s.direct_image_input.content_hash
        else:
            assert s.nuclear_input is not None
            assert s.marker_inputs is not None
            identities[(s.sample_id, "nuclear")] = (
                s.nuclear_input.content_hash
                if isinstance(s.nuclear_input, FileInput)
                else s.nuclear_input.input_id
            )
            for marker_name, marker_input in s.marker_inputs.items():
                identities[(s.sample_id, f"marker:{marker_name}")] = (
                    marker_input.content_hash
                    if isinstance(marker_input, FileInput)
                    else marker_input.input_id
                )
    return identities


def _compute_manifest_fingerprint_from_identities(
    manifest: BatchManifest,
    identities: Mapping[tuple[str, str], str],
) -> str:
    """Compute the manifest fingerprint from already-captured identities."""
    canonical_items: list[dict] = []
    for s in manifest.samples:
        if s.stack_input is not None:
            stack_id = identities[(s.sample_id, "stack")]
            canonical_items.append(
                {
                    "sample_id": s.sample_id,
                    "animal_id": s.animal_id,
                    "stack_input": stack_id,
                }
            )
        elif s.direct_image_input is not None:
            direct_id = identities[(s.sample_id, "direct_image")]
            canonical_items.append(
                {
                    "sample_id": s.sample_id,
                    "animal_id": s.animal_id,
                    "direct_image_input": direct_id,
                    "selection": s.direct_image_input.selection.to_dict(),
                }
            )
        else:
            assert s.marker_inputs is not None
            nuc_id = identities[(s.sample_id, "nuclear")]
            markers_desc: list[dict] = []
            for m_name in sorted(s.marker_inputs.keys()):
                m_id = identities[(s.sample_id, f"marker:{m_name}")]
                markers_desc.append({"marker_name": m_name, "input_id": m_id})
            canonical_items.append(
                {
                    "sample_id": s.sample_id,
                    "animal_id": s.animal_id,
                    "nuclear_input": nuc_id,
                    "marker_inputs": markers_desc,
                }
            )

    json_bytes = json.dumps(canonical_items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


@dataclass(frozen=True)
class BatchRunConfig:
    """Execution policy for batch analysis runs.

    Attributes:
        review_mode: QC review policy ('manual' pauses at review_pending; 'auto_accept' completes).
        continue_on_error: If True, errors on individual samples do not abort batch execution.
    """

    review_mode: Literal["manual", "auto_accept"] = "manual"
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        if self.review_mode not in ("manual", "auto_accept"):
            raise NerveNucBatchError(
                f"Invalid review_mode: {self.review_mode!r}. Must be 'manual' or 'auto_accept'."
            )
        if not isinstance(self.continue_on_error, bool):
            raise NerveNucBatchError("continue_on_error must be a boolean.")


@dataclass
class BatchResult:
    """Structured container holding batch execution status, live sessions, and completed summaries.

    Attributes:
        sample_status: 5-column DataFrame in manifest order.
        completed_samples: Tuple of SampleSummary objects in manifest order.
        review_sessions: Mapping of sample_id to live in-process ReviewSession for review_pending samples.
        preset: AnalysisPreset used during execution.
        config_fingerprint: SHA-256 fingerprint of the AnalysisConfig.
        manifest_fingerprint: SHA-256 fingerprint of the BatchManifest.
        preparation_fingerprint: Optional SHA-256 fingerprint of StackPreparationConfig.
        pipeline_fingerprint: Combined pipeline SHA-256 fingerprint.
        review_mode: QC review policy.
        sample_provenance: Optional mapping of sample_id to ImageSourceProvenance.
    """

    sample_status: pd.DataFrame
    completed_samples: tuple[SampleSummary, ...]
    review_sessions: dict[str, Any]
    preset: Any
    config_fingerprint: str
    manifest_fingerprint: str
    preparation_fingerprint: str | None = None
    pipeline_fingerprint: str = ""
    review_mode: str = "manual"
    sample_provenance: dict[str, Any] = field(default_factory=dict)
    sample_qc_records: dict[str, QCSummaryRecord] = field(default_factory=dict)


    def to_study_session(self) -> StudyAnalysisSession:
        """Construct a StudyAnalysisSession populated with all completed samples and their source provenance.

        Returns:
            Initial StudyAnalysisSession ready for user-directed aggregation and group assignment.
        """
        study = StudyAnalysisSession()
        for s in self.completed_samples:
            sid = s.identity.sample_id
            prov = self.sample_provenance.get(sid)
            study.add_sample(
                identity=s.identity,
                summary=s.summary_result,
                source_provenance=prov,
                analysis_config=self.preset.config,
                qc_summary=self.sample_qc_records.get(sid),
            )
        return study

    def validate_invariants(self, manifest: BatchManifest) -> None:
        """Defensively validate internal result invariants."""
        manifest_sample_ids = [s.sample_id for s in manifest.samples]
        status_sample_ids = list(self.sample_status["sample_id"])

        if status_sample_ids != manifest_sample_ids:
            raise NerveNucBatchError("sample_status rows do not match manifest sample IDs in order.")

        status_map = dict(zip(self.sample_status["sample_id"], self.sample_status["status"]))
        completed_ids = [s.identity.sample_id for s in self.completed_samples]

        # Verify completed IDs match status == 'completed'
        expected_completed_ids = [sid for sid in manifest_sample_ids if status_map[sid] == "completed"]
        if completed_ids != expected_completed_ids:
            raise NerveNucBatchError(
                f"completed_samples {completed_ids} do not match expected manifest-ordered "
                f"completed IDs {expected_completed_ids}."
            )

        # Verify live review sessions match status == 'review_pending'
        for sid, sess in self.review_sessions.items():
            if status_map.get(sid) != "review_pending":
                raise NerveNucBatchError(
                    f"Sample '{sid}' in review_sessions does not have status 'review_pending' (got '{status_map.get(sid)}')."
                )

        for sid, st in status_map.items():
            if st != "review_pending" and sid in self.review_sessions:
                raise NerveNucBatchError(
                    f"Non-review_pending sample '{sid}' (status '{st}') found in review_sessions."
                )
