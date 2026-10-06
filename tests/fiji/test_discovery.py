"""Tests for Fiji executable configuration and discovery precedence."""

from __future__ import annotations

from pathlib import Path
import pytest

from nervenuc.fiji.errors import FijiUnavailableError
from nervenuc.fiji.launcher import FijiLauncher, discover_fiji
from nervenuc.fiji.models import FijiConfig


def _create_mock_fiji_app(base_dir: Path, exe_name: str = "ImageJ-win64.exe") -> Path:
    app_dir = base_dir / "Fiji.app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "plugins").mkdir(parents=True, exist_ok=True)
    exe_file = app_dir / exe_name
    exe_file.write_text("mock_binary", encoding="utf-8")
    return app_dir


def test_discover_fiji_explicit_path_directory(tmp_path: Path) -> None:
    """Verify discovery when an explicit Fiji.app directory is provided."""
    app_dir = _create_mock_fiji_app(tmp_path)
    info = discover_fiji(explicit_path=app_dir)

    assert info is not None
    assert info.discovery_source == "explicit"
    assert info.is_fiji is True
    assert info.app_directory == app_dir
    assert info.executable_path == app_dir / "ImageJ-win64.exe"

    launcher = FijiLauncher(info)
    assert launcher.is_available() is True


def test_discover_fiji_explicit_path_file(tmp_path: Path) -> None:
    """Verify discovery when an explicit executable file is provided."""
    app_dir = _create_mock_fiji_app(tmp_path)
    exe_path = app_dir / "ImageJ-win64.exe"
    info = discover_fiji(explicit_path=exe_path)

    assert info is not None
    assert info.executable_path == exe_path


def test_discover_fiji_explicit_invalid_path_raises(tmp_path: Path) -> None:
    """Verify FijiUnavailableError is raised if an explicit path does not exist."""
    invalid_path = tmp_path / "nonexistent" / "Fiji.app"
    with pytest.raises(FijiUnavailableError, match="Explicitly configured Fiji path does not exist"):
        discover_fiji(explicit_path=invalid_path)


def test_discover_fiji_environment_variable(tmp_path: Path) -> None:
    """Verify discovery via FIJI_HOME environment variable."""
    app_dir = _create_mock_fiji_app(tmp_path)
    env = {"FIJI_HOME": str(app_dir)}

    info = discover_fiji(env=env)
    assert info is not None
    assert info.discovery_source == "env:FIJI_HOME"
    assert info.executable_path == app_dir / "ImageJ-win64.exe"


def test_discover_fiji_standard_paths_fallback(tmp_path: Path) -> None:
    """Verify fallback to known standard path candidates."""
    app_dir = _create_mock_fiji_app(tmp_path, exe_name="ImageJ-linux64")
    candidate_exe = app_dir / "ImageJ-linux64"

    info = discover_fiji(
        env={},
        known_candidates=[candidate_exe],
    )
    assert info is not None
    assert info.discovery_source == "standard_path"
    assert info.executable_path == candidate_exe


def test_discover_fiji_none_found() -> None:
    """Verify returns None when no Fiji installation exists in empty environment."""
    info = discover_fiji(
        config=FijiConfig(search_environment=False, search_standard_paths=False),
        env={},
    )
    assert info is None

    launcher = FijiLauncher(None)
    assert launcher.is_available() is False
