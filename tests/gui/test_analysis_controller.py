"""Tests for AnalysisController segmentation, review, threshold preview, and metric addition."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tifffile

from nervenuc.gui.controllers import AnalysisController, SourceController
from nervenuc.gui.state import BiologicalMetricDraft, GuiSessionState, MarkerThresholdDraft


def _create_synthetic_3d_tiff(path: Path) -> Path:
    data = np.zeros((2, 8, 32, 32), dtype=np.uint16)
    data[0, 2:6, 10:20, 10:20] = 1000  # Nuclear
    data[1, 2:6, 10:20, 10:20] = 800   # Marker
    tifffile.imwrite(str(path), data, metadata={"axes": "CZYX"})
    return path


def test_analysis_controller_3d_lifecycle(tmp_path: Path) -> None:
    """Test 3D segmentation, review, threshold preview, and biological summary."""
    f = _create_synthetic_3d_tiff(tmp_path / "lifecycle.tif")
    state = GuiSessionState()
    src_ctrl = SourceController(state)
    an_ctrl = AnalysisController(state)

    src_ctrl.inspect_file(f)
    state.set_mode("volume_3d")
    state.draft.nuclear_channel = 0
    state.draft.marker_channels = {"SOX10": 1}
    state.draft.channel_mapping_confirmed = True
    src_ctrl.prepare_and_load_session()

    # 1. Run Segmentation
    an_ctrl.run_segmentation()
    assert state.session_3d is not None
    assert state.session_3d.segmentation_result is not None
    assert state.session_3d.review_session is not None

    # 2. QC Review
    an_ctrl.auto_accept()
    an_ctrl.finalize_review()
    assert state.session_3d.reviewed_result is not None

    # 3. Measurements
    an_ctrl.measure_markers()
    assert state.session_3d.measurement_result is not None

    # 4. Preview Thresholds
    preview_res = an_ctrl.preview_thresholds([MarkerThresholdDraft("SOX10", 300.0)])
    assert preview_res is not None
    assert len(preview_res.classifications) > 0

    # 5. Define Metric and Apply
    an_ctrl.add_metric_draft(BiologicalMetricDraft(
        name="SOX10+ Pct",
        numerator_positive=("SOX10",),
    ))
    summary_res = an_ctrl.apply_thresholds_and_summarize()
    assert summary_res is not None
    assert len(summary_res.metrics) == 1
    assert summary_res.metrics.iloc[0]["metric_name"] == "SOX10+ Pct"
    assert state.results_stale is False
    assert state.configuration_differs_from_run is False
