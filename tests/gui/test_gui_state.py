"""Tests for GUI draft configuration and session state management."""

from __future__ import annotations

import pytest

from nervenuc.gui.errors import InvalidDraftConfigError
from nervenuc.gui.state import (
    AnalysisDraft,
    BiologicalMetricDraft,
    GuiSessionState,
    MarkerThresholdDraft,
)
from nervenuc.workflow import AnalysisConfig, VolumeAnalysisConfig


def test_analysis_draft_2d_and_3d_compilation() -> None:
    """Test compiling AnalysisDraft to core 2D and 3D configs."""
    draft = AnalysisDraft(
        mode="direct_2d",
        nuclear_channel=0,
        marker_channels={"SOX10": 1, "BrdU": 2},
        z_start_1based=1,
        z_stop_1based=10,
        projection_method="max",
        threshold_drafts={
            "SOX10": MarkerThresholdDraft("SOX10", 300.0),
            "BrdU": MarkerThresholdDraft("BrdU", 500.0),
        },
        metric_drafts=[
            BiologicalMetricDraft(
                name="SOX10+ Prolif",
                denominator_positive=("SOX10",),
                numerator_positive=("SOX10", "BrdU"),
            )
        ],
    )

    # 2D Config
    cfg_2d = draft.build_core_2d_config()
    assert isinstance(cfg_2d, AnalysisConfig)
    assert len(cfg_2d.markers) == 2
    assert len(cfg_2d.thresholds) == 2
    assert len(cfg_2d.metrics) == 1
    assert cfg_2d.metrics[0].name == "SOX10+ Prolif"

    # 3D Config
    vol_cfg = draft.build_core_volume_config(max_z=10)
    assert isinstance(vol_cfg, VolumeAnalysisConfig)
    assert vol_cfg.selection.nuclear_channel == 0
    assert vol_cfg.selection.marker_channels == {"SOX10": 1, "BrdU": 2}
    assert vol_cfg.selection.z_selection is not None
    assert vol_cfg.selection.z_selection.start == 0
    assert vol_cfg.selection.z_selection.stop == 10


def test_gui_session_state_mode_switching_and_stale_flags() -> None:
    """Test mode switching, preset dirty, and stale results invalidation."""
    state = GuiSessionState()
    assert state.draft.mode == "direct_2d"
    assert state.preset_dirty is False
    assert state.results_stale is False

    notifications = 0

    def listener() -> None:
        nonlocal notifications
        notifications += 1

    state.subscribe(listener)

    state.mark_draft_modified()
    assert state.preset_dirty is True
    assert state.results_stale is True
    assert notifications == 1

    state.clear_preset_dirty()
    assert state.preset_dirty is False
    assert state.results_stale is True  # Results remain stale even if preset is clean

    state.set_mode("volume_3d")
    assert state.draft.mode == "volume_3d"
    assert state.results_stale is True
    assert notifications == 3
