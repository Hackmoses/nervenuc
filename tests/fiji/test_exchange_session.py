"""Tests for temporary exchange session lifecycle, tracking, and cleanup safety."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from nervenuc.fiji.exchange import FijiExchangeSession
from nervenuc.fiji.errors import FijiConversionError
from nervenuc.fiji.launcher import FijiLauncher
from nervenuc.fiji.models import FijiExecutableInfo


def test_exchange_session_creates_isolated_temp_directory() -> None:
    """Verify session creates a unique prefixed directory and tracks generated files."""
    with FijiExchangeSession() as session:
        assert session.temp_directory.is_dir()
        assert session.temp_directory.name.startswith("nervenuc_fiji_")

        arr = np.zeros((10, 10), dtype=np.uint16)
        out_path = session.write_exchange_tiff(arr, "YX")

        assert out_path.is_file()
        assert out_path.parent == session.temp_directory
        assert out_path in session.tracked_files

    # After exit (no running processes), temp directory is cleaned up
    assert not session.temp_directory.exists()


def test_exchange_session_defers_cleanup_while_process_is_running(tmp_path: Path) -> None:
    """Verify session does not delete scratch files while Fiji process is still running."""
    session = FijiExchangeSession()
    arr = np.zeros((10, 10), dtype=np.uint16)
    out_path = session.write_exchange_tiff(arr, "YX")

    # Mock running process (poll() returns None)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    session._tracked_processes.append(mock_proc)

    # Cleanup with force=False should defer deletion
    session.cleanup(force=False)
    assert session.temp_directory.exists()
    assert out_path.is_file()

    # Once process finishes (poll() returns 0), cleanup removes directory
    mock_proc.poll.return_value = 0
    session.cleanup(force=False)
    assert not session.temp_directory.exists()


def test_exchange_session_double_cleanup_safe() -> None:
    """Verify multiple cleanup calls do not raise errors."""
    session = FijiExchangeSession()
    session.cleanup(force=True)
    session.cleanup(force=True)  # Should not raise


def test_exchange_session_rejects_path_escape_and_existing_scratch_file() -> None:
    session = FijiExchangeSession()
    try:
        with pytest.raises(FijiConversionError, match="basename"):
            session.create_temp_tiff_path("../outside.tif")

        existing = session.temp_directory / "existing.tif"
        existing.write_bytes(b"do-not-overwrite")
        with pytest.raises(FijiConversionError, match="Refusing to overwrite"):
            session.write_exchange_tiff(
                np.zeros((4, 4), dtype=np.uint16),
                "YX",
                filename="existing.tif",
            )
        assert existing.read_bytes() == b"do-not-overwrite"
    finally:
        session.cleanup(force=True)


def test_exchange_session_open_in_fiji_routes_file_and_array(tmp_path: Path) -> None:
    """Verify open_in_fiji fast-paths existing TIFF and creates scratch TIFF for arrays."""
    exe_file = tmp_path / "ImageJ-win64.exe"
    exe_file.write_text("mock_exe", encoding="utf-8")
    launcher = FijiLauncher(
        FijiExecutableInfo(
            executable_path=exe_file,
            discovery_source="explicit",
            is_fiji=True,
            app_directory=tmp_path,
        )
    )

    existing_tiff = tmp_path / "original_data.tif"
    existing_tiff.write_text("mock_tiff", encoding="utf-8")

    session = FijiExchangeSession()

    # 1. Fast path for existing file
    with MagicMock() as mock_open:
        launcher.open_file = mock_open
        session.open_in_fiji(existing_tiff, launcher=launcher)
        mock_open.assert_called_once_with(existing_tiff.resolve())

    # 2. Scratch path for NumPy array
    arr = np.ones((10, 20), dtype=np.uint16)
    with MagicMock() as mock_open:
        launcher.open_file = mock_open
        session.open_in_fiji(arr, axes="YX", launcher=launcher)
        assert mock_open.call_count == 1
        opened_path = mock_open.call_args[0][0]
        assert opened_path.parent == session.temp_directory
        assert opened_path.is_file()

    session.cleanup(force=True)
