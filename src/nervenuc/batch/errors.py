"""Exception hierarchy for batch processing and analysis preset management."""

from __future__ import annotations


class NerveNucBatchError(Exception):
    """Base exception for all batch and preset processing errors in NerveNuc."""


class InvalidPresetError(NerveNucBatchError):
    """Raised when an analysis preset definition, file, or config fingerprint is invalid."""


class UnsupportedPresetVersionError(NerveNucBatchError):
    """Raised when an analysis preset file specifies an unsupported schema version."""


class PresetFileExistsError(NerveNucBatchError):
    """Raised when attempting to write a preset file that already exists without overwrite=True."""


class InvalidBatchManifestError(NerveNucBatchError):
    """Raised when a batch manifest is empty, malformed, or contains duplicate/invalid sample IDs."""


class InvalidBatchInputError(NerveNucBatchError):
    """Raised when an input image array or TIFF file is invalid or not strictly 2-dimensional."""


class InputSourceChangedError(InvalidBatchInputError):
    """Raised when file bytes no longer match those captured for an execution."""


class BatchResumeError(NerveNucBatchError):
    """Raised when batch resume validation fails due to fingerprint, input, or policy mismatch."""
