"""Tests for SourceController file opening and session preparation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tifffile

from nervenuc.gui.controllers import SourceController
from nervenuc.gui.state import GuiSessionState


def _create_test_3d_tiff(path: Path) -> Path:
    data = np.zeros((2, 8, 32, 32), dtype=np.uint16)
    data[0, 2:6, 10:20, 10:20] = 1000  # Nuclear
    data[1, 2:6, 10:20, 10:20] = 500   # Marker
    tifffile.imwrite(str(path), data, metadata={"axes": "CZYX"})
    return path


def _create_test_direct_2d_tiff(path: Path) -> Path:
    data = np.zeros((2, 32, 32), dtype=np.uint16)
    data[0, 10:20, 10:20] = 1000
    data[1, 10:20, 10:20] = 800
    tifffile.imwrite(str(path), data, metadata={"axes": "CYX"})
    return path


def test_source_controller_inspect_and_prepare(tmp_path: Path) -> None:
    """Test inspecting and preparing 3D and projected 2D sessions."""
    f = _create_test_3d_tiff(tmp_path / "test_stack.tif")
    state = GuiSessionState()
    ctrl = SourceController(state)

    insp = ctrl.inspect_file(f)
    assert insp.shape == (2, 8, 32, 32)
    assert state.source_id is not None
    assert state.source_id.startswith("file_sha256:")

    # 1. Prepare 3D mode
    state.set_mode("volume_3d")
    state.draft.nuclear_channel = 0
    state.draft.marker_channels = {"SOX10": 1}
    state.draft.channel_mapping_confirmed = True
    ctrl.prepare_and_load_session()
    assert state.session_3d is not None
    assert state.session_2d is None

    # 2. Prepare Projected 2D mode
    state.set_mode("projected_2d")
    state.draft.projection_method = "max"
    ctrl.prepare_and_load_session()
    assert state.session_2d is not None
    assert state.session_3d is None


def test_source_controller_direct_2d_uses_real_channels_and_full_provenance(tmp_path: Path) -> None:
    f = _create_test_direct_2d_tiff(tmp_path / "direct.tif")
    state = GuiSessionState()
    ctrl = SourceController(state)
    ctrl.inspect_file(f)
    state.draft.mode = "direct_2d"
    state.draft.nuclear_channel = 0
    state.draft.marker_channels = {"Marker_1": 1}
    state.draft.channel_mapping_confirmed = True
    ctrl.prepare_and_load_session()

    assert state.session_2d is not None
    assert np.array_equal(state.session_2d.marker_images["Marker_1"], tifffile.imread(f)[1])
    prov = state.session_2d.source_provenance
    assert prov is not None
    assert prov.source_hash is not None
    assert prov.pipeline_fingerprint is not None


def test_source_controller_rejects_missing_marker_channel(tmp_path: Path) -> None:
    f = tmp_path / "single.tif"
    tifffile.imwrite(str(f), np.zeros((32, 32), dtype=np.uint16))
    state = GuiSessionState()
    ctrl = SourceController(state)
    ctrl.inspect_file(f)
    with pytest.raises(Exception, match="single-channel TIFF"):
        ctrl.prepare_and_load_session()


def test_multichannel_source_requires_confirmation_and_rejects_post_inspection_mutation(
    tmp_path: Path,
) -> None:
    f = _create_test_direct_2d_tiff(tmp_path / "adversarial.tif")
    state = GuiSessionState()
    ctrl = SourceController(state)
    ctrl.inspect_file(f)

    with pytest.raises(Exception, match="Confirm the nuclear and marker channel mapping"):
        ctrl.prepare_and_load_session()

    state.draft.channel_mapping_confirmed = True
    with open(f, "ab") as source:
        source.write(b"changed-after-inspection")

    with pytest.raises(Exception, match="Source file changed after inspection"):
        ctrl.prepare_and_load_session()


def test_source_is_rehashed_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nervenuc.gui.controllers.source_controller as source_module

    f = _create_test_direct_2d_tiff(tmp_path / "change-during-load.tif")
    state = GuiSessionState()
    ctrl = SourceController(state)
    ctrl.inspect_file(f)
    state.draft.channel_mapping_confirmed = True
    original_prepare = source_module.prepare_direct_image

    def prepare_then_mutate(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        with open(f, "ab") as source:
            source.write(b"changed-during-preparation")
        return prepared

    monkeypatch.setattr(source_module, "prepare_direct_image", prepare_then_mutate)
    with pytest.raises(Exception, match="Source file changed after inspection"):
        ctrl.prepare_and_load_session()
    assert state.session_2d is None
