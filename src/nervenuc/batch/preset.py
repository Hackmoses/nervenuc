"""Reusable analysis preset model, deterministic fingerprinting, and JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from nervenuc._version import __version__
from nervenuc.preparation import (
    StackPreparationConfig,
    VolumeSelectionConfig,
    compute_preparation_fingerprint,
)
from nervenuc.workflow import (
    AnalysisConfig,
    VolumeAnalysisConfig,
    compute_analysis_config_fingerprint,
    compute_pipeline_fingerprint,
    compute_volume_analysis_config_fingerprint,
    compute_volume_pipeline_fingerprint,
)

from .errors import (
    InvalidPresetError,
    PresetFileExistsError,
    UnsupportedPresetVersionError,
)


def compute_config_fingerprint(config: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint over the scientific contents of a config."""
    try:
        if isinstance(config, VolumeAnalysisConfig):
            return compute_volume_analysis_config_fingerprint(config)
        return compute_analysis_config_fingerprint(config)
    except Exception as exc:
        raise InvalidPresetError(str(exc)) from exc


@dataclass(frozen=True)
class AnalysisPreset:
    """Reusable, versioned analysis preset container wrapping AnalysisConfig or VolumeAnalysisConfig.

    Attributes:
        name: Human-readable preset name.
        config: Scientifically authoritative AnalysisConfig or VolumeAnalysisConfig.
        schema_version: Format schema version (1 for 2D, 2 for projected stack, 3 for 3D volume).
        nervenuc_version: Package version recorded at creation time.
        description: Optional user commentary or protocol notes.
        preparation: Optional StackPreparationConfig for 2D projected Z-stack workflows (schema 2).
    """

    name: str
    config: Any
    schema_version: int | None = None
    nervenuc_version: str = __version__
    description: str | None = None
    preparation: StackPreparationConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidPresetError("Preset name must be a non-empty string.")
        if not isinstance(self.config, (AnalysisConfig, VolumeAnalysisConfig)):
            raise InvalidPresetError(
                f"Preset config must be an AnalysisConfig or VolumeAnalysisConfig, got {type(self.config).__name__}."
            )
        if self.description is not None and not isinstance(self.description, str):
            raise InvalidPresetError("Preset description must be a string or None.")
        if self.preparation is not None and not isinstance(self.preparation, StackPreparationConfig):
            raise InvalidPresetError(
                f"Preset preparation must be a StackPreparationConfig or None, got {type(self.preparation).__name__}."
            )

        # Determine/validate schema_version
        if isinstance(self.config, VolumeAnalysisConfig):
            expected_schema = 3
        elif self.preparation is not None:
            expected_schema = 2
        else:
            expected_schema = 1

        actual_schema = self.schema_version if self.schema_version is not None else expected_schema

        if actual_schema not in (1, 2, 3):
            raise UnsupportedPresetVersionError(
                f"Unsupported preset schema_version {actual_schema}. Only schema_version 1, 2, and 3 are supported."
            )
        if actual_schema == 1:
            if not isinstance(self.config, AnalysisConfig) or self.preparation is not None:
                raise UnsupportedPresetVersionError(
                    "Schema version 1 presets must contain 2D AnalysisConfig and no preparation configuration."
                )
        elif actual_schema == 2:
            if not isinstance(self.config, AnalysisConfig) or self.preparation is None:
                raise UnsupportedPresetVersionError(
                    "Schema version 2 presets must contain 2D AnalysisConfig and StackPreparationConfig."
                )
        elif actual_schema == 3:
            if not isinstance(self.config, VolumeAnalysisConfig) or self.preparation is not None:
                raise UnsupportedPresetVersionError(
                    "Schema version 3 presets must contain VolumeAnalysisConfig and no StackPreparationConfig."
                )

        object.__setattr__(self, "schema_version", actual_schema)

    @property
    def config_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint representing the underlying configuration."""
        if isinstance(self.config, VolumeAnalysisConfig):
            return compute_volume_analysis_config_fingerprint(self.config)
        return compute_analysis_config_fingerprint(self.config)

    @property
    def preparation_fingerprint(self) -> str | None:
        """Deterministic SHA-256 fingerprint representing StackPreparationConfig if present."""
        if self.preparation is None:
            return None
        return compute_preparation_fingerprint(self.preparation)

    @property
    def pipeline_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint representing combined pipeline."""
        if self.schema_version == 3:
            return compute_volume_pipeline_fingerprint(self.config_fingerprint)
        return compute_pipeline_fingerprint(self.config_fingerprint, self.preparation_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        """Convert preset into a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "nervenuc_version": self.nervenuc_version,
            "name": self.name,
            "description": self.description,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config.to_dict(),
        }
        if self.schema_version == 2:
            data["preparation_fingerprint"] = self.preparation_fingerprint
            data["pipeline_fingerprint"] = self.pipeline_fingerprint
            assert self.preparation is not None
            data["preparation"] = self.preparation.to_dict()
        elif self.schema_version == 3:
            data["pipeline_fingerprint"] = self.pipeline_fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisPreset:
        """Reconstruct an AnalysisPreset from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidPresetError(f"Preset data must be a dict, got {type(data).__name__}.")

        schema_version = data.get("schema_version")
        if schema_version not in (1, 2, 3):
            raise UnsupportedPresetVersionError(
                f"Unsupported preset schema_version: {schema_version!r}. Only schema_version 1, 2, and 3 are supported."
            )

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InvalidPresetError("Missing or invalid 'name' in preset data.")

        nervenuc_ver = data.get("nervenuc_version", __version__)
        if not isinstance(nervenuc_ver, str):
            raise InvalidPresetError("Invalid 'nervenuc_version' in preset data.")

        desc = data.get("description")
        config_data = data.get("config")
        if not isinstance(config_data, dict):
            raise InvalidPresetError("Missing or invalid 'config' block in preset data.")

        if schema_version == 3:
            try:
                vol_cfg = VolumeAnalysisConfig.from_dict(config_data)
            except Exception as exc:
                raise InvalidPresetError(f"Failed to reconstruct VolumeAnalysisConfig from preset: {exc}") from exc

            recorded_fingerprint = data.get("config_fingerprint")
            computed_fingerprint = compute_volume_analysis_config_fingerprint(vol_cfg)

            if recorded_fingerprint is not None and recorded_fingerprint != computed_fingerprint:
                raise InvalidPresetError(
                    f"Config fingerprint mismatch in preset: recorded '{recorded_fingerprint}', "
                    f"computed '{computed_fingerprint}'."
                )

            return cls(
                name=name,
                config=vol_cfg,
                schema_version=3,
                nervenuc_version=nervenuc_ver,
                description=desc,
                preparation=None,
            )

        try:
            cfg = AnalysisConfig.from_dict(config_data)
        except Exception as exc:
            raise InvalidPresetError(f"Failed to reconstruct AnalysisConfig from preset: {exc}") from exc

        recorded_fingerprint = data.get("config_fingerprint")
        computed_fingerprint = compute_analysis_config_fingerprint(cfg)

        if recorded_fingerprint is not None and recorded_fingerprint != computed_fingerprint:
            raise InvalidPresetError(
                f"Config fingerprint mismatch in preset: recorded '{recorded_fingerprint}', "
                f"computed '{computed_fingerprint}'."
            )

        prep: StackPreparationConfig | None = None
        if schema_version == 2:
            prep_data = data.get("preparation")
            if not isinstance(prep_data, dict):
                raise InvalidPresetError("Schema version 2 preset missing 'preparation' block.")
            try:
                prep = StackPreparationConfig.from_dict(prep_data)
            except Exception as exc:
                raise InvalidPresetError(f"Failed to reconstruct StackPreparationConfig: {exc}") from exc

            rec_prep_fp = data.get("preparation_fingerprint")
            comp_prep_fp = compute_preparation_fingerprint(prep)
            if rec_prep_fp is not None and rec_prep_fp != comp_prep_fp:
                raise InvalidPresetError(
                    f"Preparation fingerprint mismatch in preset: recorded '{rec_prep_fp}', "
                    f"computed '{comp_prep_fp}'."
                )

        return cls(
            name=name,
            config=cfg,
            schema_version=schema_version,
            nervenuc_version=nervenuc_ver,
            description=desc,
            preparation=prep,
        )



def save_preset(
    preset: AnalysisPreset,
    path: str | Path,
    overwrite: bool = False,
) -> None:
    """Save an AnalysisPreset to a JSON file atomically.

    Args:
        preset: AnalysisPreset instance to serialize.
        path: Destination file path.
        overwrite: If False, raises PresetFileExistsError if target path exists.

    Raises:
        InvalidPresetError: If preset is invalid.
        PresetFileExistsError: If target path exists and overwrite is False.
    """
    if not isinstance(preset, AnalysisPreset):
        raise InvalidPresetError(
            f"preset must be an AnalysisPreset, got {type(preset).__name__}."
        )

    target_path = Path(path).resolve()
    if target_path.exists() and not overwrite:
        raise PresetFileExistsError(
            f"Target preset file '{target_path}' already exists and overwrite=False."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(preset.to_dict(), indent=2)

    # Atomic write via temporary file in target directory
    temp_fd, temp_file_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix="preset_",
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


def load_preset(path: str | Path) -> AnalysisPreset:
    """Load and validate an AnalysisPreset from a JSON file.

    Args:
        path: Path to the JSON preset file.

    Returns:
        Validated AnalysisPreset instance.

    Raises:
        InvalidPresetError: If file does not exist, cannot be parsed, or data is invalid.
        UnsupportedPresetVersionError: If schema_version is not 1 or 2.
    """
    target_path = Path(path).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise InvalidPresetError(f"Preset file not found: '{target_path}'.")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise InvalidPresetError(f"Failed to parse preset JSON from '{target_path}': {exc}") from exc

    return AnalysisPreset.from_dict(data)
