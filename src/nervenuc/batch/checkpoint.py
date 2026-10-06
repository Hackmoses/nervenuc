"""JSON checkpoint persistence and safe batch resumability."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from nervenuc.workflow import AnalysisConfig

from .errors import (
    BatchResumeError,
    NerveNucBatchError,
    PresetFileExistsError,
    UnsupportedPresetVersionError,
)
from .models import (
    BatchManifest,
    BatchResult,
    BatchRunConfig,
    compute_manifest_fingerprint,
)
from .preset import AnalysisPreset
from .runner import BatchRunner


def save_checkpoint(
    result: BatchResult,
    path: str | Path,
    overwrite: bool = False,
) -> None:
    """Save batch execution state and fingerprints to a JSON checkpoint file atomically.

    Args:
        result: BatchResult instance to checkpoint.
        path: Destination file path.
        overwrite: If False, raises PresetFileExistsError if target path exists.

    Raises:
        NerveNucBatchError: If result is not a BatchResult.
        PresetFileExistsError: If target path exists and overwrite is False.
    """
    if not isinstance(result, BatchResult):
        raise NerveNucBatchError(
            f"result must be a BatchResult, got {type(result).__name__}."
        )

    target_path = Path(path).resolve()
    if target_path.exists() and not overwrite:
        raise PresetFileExistsError(
            f"Target checkpoint file '{target_path}' already exists and overwrite=False."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)

    schema_version = 3 if result.pipeline_fingerprint and getattr(result, "preset", None) and result.preset.schema_version == 3 else (
        1 if result.preparation_fingerprint is None else 2
    )

    checkpoint_data: dict[str, Any] = {
        "schema_version": schema_version,
        "config_fingerprint": result.config_fingerprint,
        "manifest_fingerprint": result.manifest_fingerprint,
        "review_mode": result.review_mode,
        "sample_status": result.sample_status.to_dict(orient="records"),
    }

    if result.preparation_fingerprint is not None:
        checkpoint_data["preparation_fingerprint"] = result.preparation_fingerprint
    if result.pipeline_fingerprint is not None:
        checkpoint_data["pipeline_fingerprint"] = result.pipeline_fingerprint

    json_text = json.dumps(checkpoint_data, indent=2)

    # Atomic write via temporary file in target directory
    temp_fd, temp_file_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix="checkpoint_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(json_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file_path, target_path)
    except Exception:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a batch checkpoint JSON file.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Checkpoint dictionary.

    Raises:
        NerveNucBatchError: If checkpoint file does not exist or JSON is invalid.
        UnsupportedPresetVersionError: If schema_version not in (1, 2, 3).
    """
    target_path = Path(path).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise NerveNucBatchError(f"Checkpoint file not found: '{target_path}'.")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise NerveNucBatchError(f"Failed to parse checkpoint JSON from '{target_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise NerveNucBatchError("Checkpoint JSON root must be a dictionary.")

    schema_version = data.get("schema_version")
    if schema_version not in (1, 2, 3):
        raise UnsupportedPresetVersionError(
            f"Unsupported checkpoint schema_version: {schema_version!r}. Only schema_version 1, 2, and 3 are supported."
        )

    return data


def resume_batch(
    checkpoint_path: str | Path,
    preset: Any,
    manifest: BatchManifest,
    run_config: BatchRunConfig | None = None,
) -> BatchResult:
    """Resume a batch run from a checkpoint after strictly validating configuration and manifest identity.

    Args:
        checkpoint_path: Path to existing checkpoint JSON file.
        preset: AnalysisPreset, AnalysisConfig, or VolumeAnalysisConfig for the resumed run.
        manifest: Current BatchManifest.
        run_config: Optional execution policy (must match checkpoint review_mode).

    Returns:
        BatchResult object.

    Raises:
        BatchResumeError: If config fingerprint, manifest fingerprint, preparation fingerprint, or review_mode mismatches.
    """
    from nervenuc.workflow import VolumeAnalysisConfig

    chk_data = load_checkpoint(checkpoint_path)

    # Normalize preset
    if isinstance(preset, AnalysisConfig):
        active_preset = AnalysisPreset(name="Anonymous Preset", config=preset)
    elif isinstance(preset, VolumeAnalysisConfig):
        active_preset = AnalysisPreset(name="Anonymous Preset", config=preset)
    elif isinstance(preset, AnalysisPreset):
        active_preset = preset
    else:
        raise BatchResumeError(f"preset must be an AnalysisPreset, AnalysisConfig, or VolumeAnalysisConfig, got {type(preset).__name__}.")

    if not isinstance(manifest, BatchManifest):
        raise BatchResumeError(f"manifest must be a BatchManifest, got {type(manifest).__name__}.")

    # 1. Validate config fingerprint
    recorded_config_fp = chk_data.get("config_fingerprint")
    current_config_fp = active_preset.config_fingerprint
    if recorded_config_fp != current_config_fp:
        raise BatchResumeError(
            f"Config fingerprint mismatch on resume: checkpoint has '{recorded_config_fp}', "
            f"current preset has '{current_config_fp}'."
        )

    # 2. Validate manifest fingerprint (content-aware)
    recorded_manifest_fp = chk_data.get("manifest_fingerprint")
    current_manifest_fp = compute_manifest_fingerprint(manifest)
    if recorded_manifest_fp != current_manifest_fp:
        raise BatchResumeError(
            f"Manifest fingerprint mismatch on resume: checkpoint has '{recorded_manifest_fp}', "
            f"current manifest has '{current_manifest_fp}'."
        )

    # 3. Validate preparation fingerprint
    recorded_prep_fp = chk_data.get("preparation_fingerprint")
    current_prep_fp = active_preset.preparation_fingerprint
    if recorded_prep_fp != current_prep_fp:
        raise BatchResumeError(
            f"Preparation fingerprint mismatch on resume: checkpoint has '{recorded_prep_fp}', "
            f"current preset has '{current_prep_fp}'."
        )

    # 4. Validate pipeline fingerprint
    recorded_pipe_fp = chk_data.get("pipeline_fingerprint")
    current_pipe_fp = active_preset.pipeline_fingerprint
    if recorded_pipe_fp is not None and recorded_pipe_fp != current_pipe_fp:
        raise BatchResumeError(
            f"Pipeline fingerprint mismatch on resume: checkpoint has '{recorded_pipe_fp}', "
            f"current preset has '{current_pipe_fp}'."
        )

    # 5. Validate review_mode policy
    chk_review_mode = chk_data.get("review_mode", "manual")
    effective_run_config = run_config if run_config is not None else BatchRunConfig(review_mode=chk_review_mode)
    if chk_review_mode != effective_run_config.review_mode:
        raise BatchResumeError(
            f"Review mode mismatch on resume: checkpoint recorded review_mode '{chk_review_mode}', "
            f"current run config is '{effective_run_config.review_mode}'. Changing review policy during resume is not permitted."
        )

    # 6. Execute batch runner under Strategy B resume semantics
    runner = BatchRunner(active_preset, manifest, effective_run_config)
    return runner.run()
