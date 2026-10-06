"""Public API for NerveNuc batch processing and reusable analysis presets."""

from .checkpoint import (
    load_checkpoint,
    resume_batch,
    save_checkpoint,
)
from .errors import (
    BatchResumeError,
    InputSourceChangedError,
    InvalidBatchInputError,
    InvalidBatchManifestError,
    InvalidPresetError,
    NerveNucBatchError,
    PresetFileExistsError,
    UnsupportedPresetVersionError,
)
from .models import (
    ArrayInput,
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    BatchSampleSpec,
    ChannelMappedFileInput,
    FileInput,
    StackFileInput,
    compute_manifest_fingerprint,
)
from .preset import (
    AnalysisPreset,
    compute_config_fingerprint,
    load_preset,
    save_preset,
)
from .runner import (
    BatchRunner,
    resolve_input,
    run_batch,
)

__all__ = [
    # Presets
    "AnalysisPreset",
    "compute_config_fingerprint",
    "save_preset",
    "load_preset",
    # Input descriptors & manifest
    "ArrayInput",
    "FileInput",
    "ChannelMappedFileInput",
    "StackFileInput",
    "BatchSampleSpec",
    "BatchManifest",
    "compute_manifest_fingerprint",
    "resolve_input",
    # Execution & results
    "BatchRunConfig",
    "BatchResult",
    "BatchRunner",
    "run_batch",
    # Checkpoints & resume
    "save_checkpoint",
    "load_checkpoint",
    "resume_batch",
    # Exceptions
    "NerveNucBatchError",
    "InvalidPresetError",
    "UnsupportedPresetVersionError",
    "PresetFileExistsError",
    "InvalidBatchManifestError",
    "InvalidBatchInputError",
    "InputSourceChangedError",
    "BatchResumeError",
]
