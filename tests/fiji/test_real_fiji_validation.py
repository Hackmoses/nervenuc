"""Live integration validation against real local Fiji / ImageJ installation.

These tests are marked with @pytest.mark.fiji and are executed only when a real
local Fiji executable is discovered via NERVENUC_FIJI_TEST_PATH, QSettings, or system paths.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

from nervenuc.fiji.exchange import FijiExchangeSession
from nervenuc.fiji.launcher import FijiLauncher, discover_fiji
from nervenuc.fiji.qc import export_review_rois, import_rois_as_manual
from nervenuc.fiji.roi import mask_to_roi, read_roi_set, write_roi_set
from nervenuc.preparation import VoxelSpacing
from nervenuc.qc import ReviewSession
from nervenuc.segmentation import SegmentationConfig, SegmentationResult


def _resolve_real_fiji_executable() -> Path | None:
    """Resolve real Fiji executable from env var, QSettings, discovery, or standard locations."""
    # 1. Explicit test environment variable
    env_path = os.environ.get("NERVENUC_FIJI_TEST_PATH")
    if env_path:
        p = Path(env_path)
        info = discover_fiji(explicit_path=p)
        if info is not None:
            return info.executable_path

    # 2. QSettings configured path
    try:
        from PySide6.QtCore import QSettings
        settings = QSettings("NerveNuc", "NerveNuc")
        val = settings.value("fiji/executable_path", None)
        if val:
            p = Path(str(val))
            if p.exists():
                info = discover_fiji(explicit_path=p)
                if info is not None:
                    return info.executable_path
    except Exception:
        pass

    # 3. Standard system paths / auto-discovery
    info = discover_fiji()
    if info is not None:
        return info.executable_path

    # 4. Common user download locations fallback
    user_home = Path.home()
    common_candidates = [
        user_home / "Downloads" / "fiji-win64" / "Fiji.app" / "ImageJ-win64.exe",
        user_home / "Downloads" / "Fiji.app" / "ImageJ-win64.exe",
        user_home / "Fiji.app" / "ImageJ-win64.exe",
    ]
    for c in common_candidates:
        if c.is_file():
            info = discover_fiji(explicit_path=c)
            if info is not None:
                return info.executable_path

    return None


_REAL_FIJI_EXE = _resolve_real_fiji_executable()
pytestmark = [
    pytest.mark.fiji,
    pytest.mark.skipif(_REAL_FIJI_EXE is None, reason="No real Fiji executable found on this system."),
]


@pytest.fixture
def fiji_launcher() -> FijiLauncher:
    """Return a FijiLauncher configured with the resolved real Fiji executable."""
    assert _REAL_FIJI_EXE is not None
    info = discover_fiji(explicit_path=_REAL_FIJI_EXE)
    assert info is not None
    return FijiLauncher(info)


def test_real_fiji_2d_tiff_launch(tmp_path: Path, fiji_launcher: FijiLauncher) -> None:
    """Validate opening a 2D microscopy exchange TIFF in real external Fiji."""
    img_2d = np.random.randint(0, 1000, size=(128, 128), dtype=np.uint16)
    sess = FijiExchangeSession(temp_base_dir=tmp_path)

    proc: subprocess.Popen | None = None
    try:
        proc = sess.open_in_fiji(
            source=img_2d,
            axes="YX",
            spacing=VoxelSpacing(z=1.0, y=0.325, x=0.325),
            launcher=fiji_launcher,
        )
        assert proc is not None
        assert proc.pid > 0
        # Allow process to start
        time.sleep(0.5)
        # Verify process is actively running or exited normally
        poll_status = proc.poll()
        assert poll_status is None or poll_status == 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sess.cleanup(force=True)


def test_real_fiji_3d_zstack_launch(tmp_path: Path, fiji_launcher: FijiLauncher) -> None:
    """Validate opening a 3D Z-stack exchange TIFF in real external Fiji."""
    stack_3d = np.random.randint(0, 4000, size=(8, 100, 100), dtype=np.uint16)
    sess = FijiExchangeSession(temp_base_dir=tmp_path)

    proc: subprocess.Popen | None = None
    try:
        proc = sess.open_in_fiji(
            source=stack_3d,
            axes="ZYX",
            spacing=VoxelSpacing(x=0.5, y=0.5, z=1.0),
            launcher=fiji_launcher,
        )
        assert proc is not None
        assert proc.pid > 0
        time.sleep(0.5)
        poll_status = proc.poll()
        assert poll_status is None or poll_status == 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sess.cleanup(force=True)


def test_real_fiji_3d_label_volume_launch(tmp_path: Path, fiji_launcher: FijiLauncher) -> None:
    """Validate opening a 3D discrete integer segmentation label volume in real Fiji."""
    labels_3d = np.zeros((5, 80, 80), dtype=np.int32)
    labels_3d[1:3, 10:30, 10:30] = 1
    labels_3d[2:4, 40:60, 40:60] = 2

    sess = FijiExchangeSession(temp_base_dir=tmp_path)
    proc: subprocess.Popen | None = None
    try:
        proc = sess.open_in_fiji(
            source=labels_3d,
            axes="ZYX",
            is_labels=True,
            launcher=fiji_launcher,
        )
        assert proc is not None
        assert proc.pid > 0
        time.sleep(0.5)
        poll_status = proc.poll()
        assert poll_status is None or poll_status == 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sess.cleanup(force=True)


def test_real_fiji_roi_set_interoperability(tmp_path: Path) -> None:
    """Validate NerveNuc 2D ROI export and Fiji-compatible ROI set round-trip."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[50:70, 50:70] = 2
    seg = SegmentationResult(labels=labels, objects=pd.DataFrame(), config=SegmentationConfig())
    review_session = ReviewSession(seg)

    # 1. Export active objects to ROI ZIP
    zip_path = tmp_path / "nuc_exported.zip"
    export_review_rois(review_session, zip_path)
    assert zip_path.is_file()
    assert zip_path.stat().st_size > 0

    # 2. Read back using roifile and verify standard ImageJ headers
    rois = read_roi_set(zip_path)
    assert len(rois) == 2
    roi_names = {getattr(r, "name", "") for r in rois}
    assert "AUTO-000001" in roi_names
    assert "AUTO-000002" in roi_names

    # 3. Create a fresh session and import the exported ROI ZIP
    empty_seg = SegmentationResult(
        labels=np.zeros((100, 100), dtype=np.int32),
        objects=pd.DataFrame(),
        config=SegmentationConfig(),
    )
    import_session = ReviewSession(empty_seg)
    import_res = import_rois_as_manual(import_session, zip_path)
    assert len(import_res.imported) == 2
    assert import_res.imported[0].review_id == "MAN-000001"
    assert import_res.imported[1].review_id == "MAN-000002"
