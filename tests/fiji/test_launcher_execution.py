"""Tests for Fiji launcher execution and subprocess management."""

from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from nervenuc.fiji.errors import FijiExecutionError, FijiUnavailableError
from nervenuc.fiji.launcher import FijiLauncher
from nervenuc.fiji.models import FijiExecutableInfo


def test_launcher_open_file_argument_construction(tmp_path: Path) -> None:
    """Verify FijiLauncher builds arguments correctly with Popen without shell=True."""
    exe_file = tmp_path / "ImageJ-win64.exe"
    exe_file.write_text("mock_exe", encoding="utf-8")

    target_file = tmp_path / "my sample image with spaces.tif"
    target_file.write_text("mock_tiff", encoding="utf-8")

    info = FijiExecutableInfo(
        executable_path=exe_file,
        discovery_source="explicit",
        is_fiji=True,
        app_directory=tmp_path,
    )
    launcher = FijiLauncher(info)

    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        proc = launcher.open_file(target_file)
        assert proc is mock_proc

        mock_popen.assert_called_once_with(
            [str(exe_file.resolve()), str(target_file.resolve())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )


def test_launcher_missing_executable_raises_fiji_unavailable(tmp_path: Path) -> None:
    """Verify FijiUnavailableError if launcher executable does not exist."""
    launcher = FijiLauncher(None)
    with pytest.raises(FijiUnavailableError, match="No valid Fiji executable"):
        launcher.open_file(tmp_path / "test.tif")


def test_launcher_missing_target_file_raises_fiji_execution_error(tmp_path: Path) -> None:
    """Verify FijiExecutionError if target image file does not exist."""
    exe_file = tmp_path / "ImageJ-win64.exe"
    exe_file.write_text("mock_exe", encoding="utf-8")

    info = FijiExecutableInfo(
        executable_path=exe_file,
        discovery_source="explicit",
        is_fiji=True,
        app_directory=tmp_path,
    )
    launcher = FijiLauncher(info)

    with pytest.raises(FijiExecutionError, match="Target file to open in Fiji does not exist"):
        launcher.open_file(tmp_path / "nonexistent.tif")
