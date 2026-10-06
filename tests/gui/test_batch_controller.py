"""Tests for BatchController manifest creation and preset management."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tifffile

from nervenuc.batch import AnalysisPreset
from nervenuc.classification import MarkerThreshold
from nervenuc.gui.controllers import BatchController, ExportController
from nervenuc.gui.state import GuiSessionState
from nervenuc.gui.widgets import BatchManifestWidget
from nervenuc.markers import MarkerDefinition
from nervenuc.preparation import StackPreparationConfig
from nervenuc.segmentation import SegmentationConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter
from nervenuc.workflow import AnalysisConfig


def _create_sample_file(path: Path) -> Path:
    data = np.zeros((32, 32), dtype=np.uint16)
    tifffile.imwrite(str(path), data)
    return path


def _create_direct_sample(path: Path, *, offset: int = 0) -> Path:
    data = np.zeros((2, 64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    nucleus = (yy - (28 + offset)) ** 2 + (xx - 32) ** 2 <= 9**2
    data[0, nucleus] = 100.0
    data[1, nucleus] = 80.0
    tifffile.imwrite(str(path), data, metadata={"axes": "CYX"}, photometric="minisblack")
    return path


def _direct_preset() -> AnalysisPreset:
    config = AnalysisConfig(
        segmentation=SegmentationConfig(
            threshold_method="otsu",
            min_area_px=20,
            min_distance_px=3,
        ),
        markers=(MarkerDefinition("SOX10"),),
        thresholds=(MarkerThreshold("SOX10", 50.0, "mean_intensity"),),
        metrics=(
            BiologicalMetricDefinition(
                "SOX10+",
                denominator=PhenotypeFilter(),
                numerator=PhenotypeFilter({"SOX10": True}),
            ),
        ),
    )
    return AnalysisPreset("Frozen direct-2D", config)


def _configured_direct_controller(paths: list[Path]) -> tuple[GuiSessionState, BatchController]:
    state = GuiSessionState()
    state.draft.mode = "direct_2d"
    state.active_preset = _direct_preset()
    state.preset_dirty = False
    ctrl = BatchController(state)
    for index, path in enumerate(paths, start=1):
        ctrl.add_sample_draft(path)
        ctrl.update_sample_draft(
            index - 1,
            sample_id=f"S{index}",
            animal_id=f"A{index}",
            declared_axes="CYX",
            nuclear_channel=0,
            marker_channels={"SOX10": 1},
        )
    return state, ctrl


def test_batch_controller_manifest_and_preset(tmp_path: Path) -> None:
    """Test building manifest and saving/loading presets."""
    f1 = _create_sample_file(tmp_path / "s1.tif")
    f2 = _create_sample_file(tmp_path / "s2.tif")

    state = GuiSessionState()
    ctrl = BatchController(state)

    ctrl.add_sample_spec("S1", "A1", f1)
    ctrl.add_sample_spec("S2", "A2", f2)
    assert len(ctrl.manifest_specs) == 2

    manifest = ctrl.build_manifest()
    assert len(manifest.samples) == 2

    preset_path = tmp_path / "gui_preset.json"
    ctrl.save_current_preset(preset_path, "Test GUI Preset")
    assert preset_path.is_file()

    loaded = ctrl.load_preset_file(preset_path)
    assert loaded.name == "Test GUI Preset"
    assert state.preset_dirty is False


def test_projected_batch_controller_preserves_preparation_and_builds_runner(tmp_path: Path) -> None:
    stack = np.zeros((2, 3, 32, 32), dtype=np.uint16)
    stack[0, :, 10:20, 10:20] = 1000
    stack[1, :, 10:20, 10:20] = 800
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), stack, metadata={"axes": "CZYX"}, photometric="minisblack")

    state = GuiSessionState()
    state.draft.mode = "projected_2d"
    state.draft.declared_axes = "CZYX"
    state.draft.marker_channels = {"Marker_1": 1}
    ctrl = BatchController(state)
    ctrl.add_sample_spec("S1", "A1", path)

    preset_path = tmp_path / "projected.json"
    ctrl.save_current_preset(preset_path, "Projected")
    assert state.active_preset is not None
    assert state.active_preset.schema_version == 2
    assert state.active_preset.preparation is not None

    runner = ctrl.create_batch_runner()
    assert runner.run_config.review_mode == "auto_accept"


def test_direct_batch_preflight_run_and_export_are_truthfully_auto_accepted(tmp_path: Path) -> None:
    paths = [
        _create_direct_sample(tmp_path / "s1.tif"),
        _create_direct_sample(tmp_path / "s2.tif", offset=3),
    ]
    state, ctrl = _configured_direct_controller(paths)

    report = ctrl.preflight_batch()
    assert report.valid
    assert report.manifest_fingerprint
    runner = ctrl.create_preflighted_runner()
    result = runner.run()
    assert list(result.sample_status["status"]) == ["completed", "completed"]
    assert all(record.review_state == "auto_accepted" for record in result.sample_qc_records.values())
    assert ctrl.commit_batch_result(result)

    export = ExportController(state)
    assert export.can_export_reproducible()
    written = export.export_reproducible_bundle(
        tmp_path / "bundle",
        write_excel=False,
    )
    methods = written["methods_summary_txt"].read_text(encoding="utf-8")
    qc_summary = written["qc_summary_json"].read_text(encoding="utf-8")
    provenance = written["provenance_json"].read_text(encoding="utf-8")
    assert "automatically accepted without manual per-object QC" in methods
    assert "fully manually reviewed" not in methods.lower()
    assert "manual qc completed" not in methods.lower()
    assert '"review_state": "auto_accepted"' in qc_summary
    assert '"qc_review_policy": "auto_accept"' in provenance
    assert '"manual_per_object_qc_performed": false' in provenance


def test_direct_batch_preflight_requires_real_animal_and_complete_mapping(tmp_path: Path) -> None:
    path = _create_direct_sample(tmp_path / "s1.tif")
    state = GuiSessionState()
    state.draft.mode = "direct_2d"
    state.active_preset = _direct_preset()
    state.preset_dirty = False
    ctrl = BatchController(state)
    ctrl.add_sample_draft(path)
    ctrl.update_sample_draft(
        0,
        sample_id="S1",
        animal_id="",
        declared_axes="CYX",
        nuclear_channel=0,
        marker_channels={},
    )
    report = ctrl.preflight_batch()
    assert not report.valid
    assert "does not fabricate" in report.rows[0].message

    ctrl.update_sample_draft(
        0,
        sample_id="S1",
        animal_id="A1",
        declared_axes="CYX",
        nuclear_channel=0,
        marker_channels={},
    )
    report = ctrl.preflight_batch()
    assert not report.valid
    assert "missing markers" in report.rows[0].message


def test_changed_source_isolated_at_runtime_and_partial_study_is_not_published(tmp_path: Path) -> None:
    paths = [
        _create_direct_sample(tmp_path / "s1.tif"),
        _create_direct_sample(tmp_path / "s2.tif", offset=3),
    ]
    state, ctrl = _configured_direct_controller(paths)
    assert ctrl.preflight_batch().valid
    runner = ctrl.create_preflighted_runner()

    _create_direct_sample(paths[0], offset=8)
    result = runner.run()
    assert list(result.sample_status["status"]) == ["failed", "completed"]
    assert not ctrl.commit_batch_result(result)
    assert not state.study_session.samples
    assert not ExportController(state).can_export_reproducible()


def test_batch_widget_runs_preflighted_worker_and_discloses_auto_accept(qtbot, tmp_path: Path) -> None:
    paths = [
        _create_direct_sample(tmp_path / "s1.tif"),
        _create_direct_sample(tmp_path / "s2.tif", offset=3),
    ]
    state, ctrl = _configured_direct_controller(paths)
    widget = BatchManifestWidget(ctrl, state)
    qtbot.addWidget(widget)
    widget.refresh_table()

    policy = widget.policy_label.text().lower()
    assert "automatically accepted" in policy
    assert "manual per-object qc is not performed" in policy
    widget._on_validate()
    assert widget.btn_run.isEnabled()

    with qtbot.waitSignal(widget.batch_finished, timeout=30_000) as completion:
        widget.start_preflighted_run()
    result, complete = completion.args
    assert complete
    assert list(result.sample_status["status"]) == ["completed", "completed"]
    assert "study and reproducible export are available" in widget.summary_label.text().lower()


def test_projected_stack_gui_preflight_runs_to_complete_study(tmp_path: Path) -> None:
    stack = np.zeros((2, 3, 64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    nucleus = (yy - 30) ** 2 + (xx - 32) ** 2 <= 9**2
    stack[0, :, nucleus] = 100.0
    stack[1, :, nucleus] = 80.0
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), stack, metadata={"axes": "CZYX"}, photometric="minisblack")

    state = GuiSessionState()
    state.draft.mode = "projected_2d"
    state.active_preset = AnalysisPreset(
        "Frozen projected-2D",
        _direct_preset().config,
        preparation=StackPreparationConfig(
            axes="CZYX",
            nuclear_channel=0,
            marker_channels={"SOX10": 1},
        ),
    )
    state.preset_dirty = False
    ctrl = BatchController(state)
    ctrl.add_sample_draft(path)
    ctrl.update_sample_draft(
        0,
        sample_id="STACK-1",
        animal_id="ANIMAL-1",
        declared_axes="CZYX",
        nuclear_channel=None,
        marker_channels={},
    )

    report = ctrl.preflight_batch()
    assert report.valid
    result = ctrl.create_preflighted_runner().run()
    assert list(result.sample_status["status"]) == ["completed"]
    assert ctrl.commit_batch_result(result)
    assert state.study_session.sample_provenance["STACK-1"].source_kind == "projected_stack"
