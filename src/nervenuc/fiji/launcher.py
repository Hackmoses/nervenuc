"""Discovery and execution foundation for external Fiji/ImageJ processes."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import subprocess
from .errors import FijiExecutionError, FijiUnavailableError
from .models import FijiConfig, FijiExecutableInfo

_STANDARD_WINDOWS_PATHS = [

    Path("C:/Fiji.app/ImageJ-win64.exe"),
    Path("C:/Program Files/Fiji.app/ImageJ-win64.exe"),
    Path("C:/Fiji.app/ImageJ-win32.exe"),
]

_STANDARD_MACOS_PATHS = [
    Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
    Path("/Applications/Fiji.app"),
    Path("~/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
]

_STANDARD_LINUX_PATHS = [
    Path("/opt/Fiji.app/ImageJ-linux64"),
    Path("/usr/local/Fiji.app/ImageJ-linux64"),
    Path("~/.local/Fiji.app/ImageJ-linux64"),
    Path("/opt/fiji/ImageJ-linux64"),
    Path("/usr/bin/imagej"),
]


def _resolve_executable_from_app_dir(app_dir: Path) -> Path | None:
    """Find the primary ImageJ executable binary inside a Fiji.app directory."""
    if not app_dir.is_dir():
        return None

    # Platform-specific candidate filenames
    candidates = [
        "ImageJ-win64.exe",
        "ImageJ-win32.exe",
        "Contents/MacOS/ImageJ-macosx",
        "ImageJ-macosx",
        "ImageJ-linux64",
        "ImageJ-linux32",
        "imagej",
    ]
    for c in candidates:
        exe_path = app_dir / c
        if exe_path.is_file():
            return exe_path
    return None


def _validate_and_build_info(target_path: Path, source_name: str) -> FijiExecutableInfo | None:
    """Validate a potential Fiji file or directory and construct FijiExecutableInfo."""
    expanded = target_path.expanduser().resolve()
    if not expanded.exists():
        return None

    if expanded.is_dir():
        exe = _resolve_executable_from_app_dir(expanded)
        if exe is not None:
            is_fiji = (expanded / "plugins").is_dir() or "Fiji" in expanded.name
            return FijiExecutableInfo(
                executable_path=exe,
                discovery_source=source_name,
                is_fiji=is_fiji,
                app_directory=expanded,
            )
        return None

    if expanded.is_file():
        app_dir = expanded.parent
        if app_dir.name == "MacOS" and app_dir.parent.name == "Contents":
            app_dir = app_dir.parent.parent
        is_fiji = (app_dir / "plugins").is_dir() or "Fiji" in str(expanded)
        return FijiExecutableInfo(
            executable_path=expanded,
            discovery_source=source_name,
            is_fiji=is_fiji,
            app_directory=app_dir,
        )

    return None


def discover_fiji(
    config: FijiConfig | None = None,
    explicit_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    known_candidates: Sequence[Path] | None = None,
) -> FijiExecutableInfo | None:
    """Discover and validate a local Fiji installation on disk with strict precedence.

    Precedence order:
        1. Explicit path parameter or config.explicit_fiji_path
        2. Environment variables (FIJI_HOME, IMAGEJ_HOME, FIJI_PATH)
        3. Conservative standard platform paths

    Args:
        config: Optional FijiConfig governing discovery flags.
        explicit_path: Optional explicit file or folder path.
        env: Injected environment dictionary (defaults to os.environ).
        known_candidates: Optional list of standard path candidates to check (for testing).

    Returns:
        FijiExecutableInfo if a valid installation is found, None otherwise.

    Raises:
        FijiUnavailableError: If an explicit path was provided but is invalid or does not exist.
    """
    cfg = config if config is not None else FijiConfig()
    effective_explicit = explicit_path if explicit_path is not None else cfg.explicit_fiji_path

    # 1. Explicit path
    if effective_explicit is not None:
        p = Path(effective_explicit)
        info = _validate_and_build_info(p, "explicit")
        if info is None:
            raise FijiUnavailableError(
                f"Explicitly configured Fiji path does not exist or is not a valid Fiji installation: '{effective_explicit}'."
            )
        return info

    env_map = env if env is not None else os.environ

    # 2. Environment variables
    if cfg.search_environment:
        for env_key in ("FIJI_HOME", "IMAGEJ_HOME", "FIJI_PATH"):
            val = env_map.get(env_key)
            if val:
                info = _validate_and_build_info(Path(val), f"env:{env_key}")
                if info is not None:
                    return info

    # 3. Standard platform paths
    if cfg.search_standard_paths:
        if known_candidates is not None:
            candidates = known_candidates
        elif sys.platform.startswith("win"):
            candidates = list(_STANDARD_WINDOWS_PATHS)
            # Add LocalAppData on Windows
            local_appdata = env_map.get("LOCALAPPDATA")
            if local_appdata:
                candidates.append(Path(local_appdata) / "Fiji.app" / "ImageJ-win64.exe")
        elif sys.platform == "darwin":
            candidates = _STANDARD_MACOS_PATHS
        else:
            candidates = _STANDARD_LINUX_PATHS

        for c in candidates:
            info = _validate_and_build_info(c, "standard_path")
            if info is not None:
                return info

    return None


class FijiLauncher:
    """Manager for external Fiji process execution and lifecycle tracking."""

    def __init__(self, executable_info: FijiExecutableInfo | None = None) -> None:
        self._info = executable_info

    @property
    def executable_info(self) -> FijiExecutableInfo | None:
        """Return the active FijiExecutableInfo or None."""
        return self._info

    def is_available(self) -> bool:
        """Return True if a valid Fiji executable has been discovered."""
        return self._info is not None and self._info.executable_path.is_file()

    def open_file(self, target_path: Path | str) -> subprocess.Popen:
        """Launch an external Fiji process to open a target image or ROI file.

        Args:
            target_path: Path to the target TIFF, ROI, or image file on disk.

        Returns:
            subprocess.Popen handle for the spawned Fiji process.

        Raises:
            FijiUnavailableError: If Fiji executable is missing or invalid.
            FijiExecutionError: If target file does not exist or process launch fails.
        """
        if self._info is None or not self._info.executable_path.is_file():
            raise FijiUnavailableError("Cannot launch Fiji: No valid Fiji executable is configured or discovered.")

        p = Path(target_path).resolve()
        if not p.is_file():
            raise FijiExecutionError(f"Target file to open in Fiji does not exist: '{target_path}'.")

        cmd = [str(self._info.executable_path), str(p)]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return process
        except Exception as err:
            raise FijiExecutionError(f"Failed to launch external Fiji process with command {cmd}: {err}") from err
