"""End-to-end quick analysis, optional review, and direct image exports."""
import json
import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage
from nervenuc.gui.main_window import MainWindow
from nervenuc.gui.quick_analysis import analyze_image, export_folder_results
from nervenuc.gui.state import AnalysisDraft, MarkerThresholdDraft


def image_data():
    data = np.zeros((2, 48, 48), dtype=np.uint16)
    data[0, 8:18, 8:18] = 1000
    data[0, 28:38, 28:38] = 1000
    data[1, 8:18, 8:18] = 600
    data[1, 28:38, 28:38] = 50
    return data


def test_quick_count_single_channel_and_stack():
    data = image_data()
    draft = AnalysisDraft(declared_axes='YX', segmentation_values={'gaussian_sigma': 0, 'min_area_px': 20})
    result = analyze_image(data[0], None, draft, 'test-source', 'count')
    assert len(result.objects) == 2
    assert result.measurement is None
    draft.mode = 'projected_2d'
    draft.declared_axes = 'ZYX'
    draft.z_start_1based, draft.z_stop_1based = 1, 3
    stack = np.repeat(data[0][None], 3, axis=0)
    result = analyze_image(stack, None, draft, 'test-stack', 'count')
    assert len(result.objects) == 2
    draft.mode = 'volume_3d'
    draft.segmentation_values = {'gaussian_sigma_xy_px':0, 'gaussian_sigma_z_slices':0, 'min_volume_voxels':20}
    result = analyze_image(stack, None, draft, 'test-volume', 'count')
    assert len(result.objects) == 2
    assert result.labels.ndim == 3


def test_quick_signal_and_explicit_positivity():
    from nervenuc.classification import MarkerThreshold
    draft = AnalysisDraft(declared_axes='CYX', marker_channels={'Signal':1}, segmentation_values={'gaussian_sigma':0, 'min_area_px':20})
    result = analyze_image(image_data(), None, draft, 'test-source', 'positivity')
    assert result.classification is None
    assert len(result.measurement.measurements) == 2
    result.apply_cutoffs([MarkerThreshold('Signal', 200)])
    assert int(result.classification.classifications.is_positive.sum()) == 1
    identifier = result.objects.iloc[0].review_id
    result.review.reject(identifier)
    result.update_measurements()
    assert len(result.objects) == 1
    assert len(result.measurement.measurements) == 1
    assert result.classification is None
    result.review.restore(identifier)
    result.update_measurements()
    assert len(result.objects) == 2


def test_quick_ui_open_analyze_review_export(qtbot, tmp_path, monkeypatch):
    def fail(*args): raise AssertionError(str(args[-1]))
    monkeypatch.setattr(QMessageBox, 'warning', fail)
    win = MainWindow()
    qtbot.addWidget(win)
    path = tmp_path/'image.tif'
    tifffile.imwrite(path, image_data(), metadata={'axes':'CYX'}, photometric='minisblack')
    win.open_image(str(path))
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
    assert win.nav_list.count() == 4
    assert win.nav_list.currentRow() == 1
    assert win.channel_table.confirm_mapping.isHidden()
    win.combo_task.setCurrentIndex(1)
    win.seg_form.spin_min_size.setValue(20)
    win._on_analyze_image()
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
    assert win.state.quick_result is not None
    assert win.inspector_stack.currentIndex() == 9
    assert win.quick_results.btn_export.isEnabled()
    win._show_quick_review()
    win.qc_widget.table.selectRow(0)
    win._on_qc_reject()
    assert len(win.state.quick_result.objects) == 1
    win.quick_results.export_to(tmp_path/'export')
    settings = json.loads((tmp_path/'export'/'analysis.json').read_text())
    assert settings['object_count'] == 1
    assert settings['review_policy'] == 'manual'
    assert (tmp_path/'export'/'measurements.csv').is_file()
    assert (tmp_path/'export'/'annotated-view.png').is_file()
    win.seg_form.spin_min_size.setValue(30)
    assert not win.quick_results.btn_export.isEnabled()
    with pytest.raises(ValueError): win.quick_results.export_to(tmp_path/'stale')


def test_png_and_folder_processing_need_no_animals(qtbot, tmp_path):
    path=tmp_path/'mono.png'
    image=QImage(48,48,QImage.Format.Format_Grayscale8)
    image.fill(0)
    for y in range(10,25):
        for x in range(10,25): image.setPixelColor(x,y, __import__('PySide6.QtGui',fromlist=['QColor']).QColor('white'))
    assert image.save(str(path))
    win=MainWindow()
    qtbot.addWidget(win)
    win.open_image(str(path))
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
    assert win.state.loaded_raw_array.shape == (48,48)
    report = export_folder_results(tmp_path, tmp_path/'output', AnalysisDraft(mode='direct_2d', declared_axes='YX'), 'count')
    assert report[0]['status']=='completed'
    assert (tmp_path/'output'/'folder-report.csv').is_file()


def test_repeated_stack_open_and_positivity_ui(qtbot, tmp_path, monkeypatch):
    from nervenuc.gui.state import BiologicalMetricDraft
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: (_ for _ in ()).throw(AssertionError(args[-1])))
    win = MainWindow()
    qtbot.addWidget(win)
    stack = np.repeat(image_data()[None], 3, axis=0)
    path = tmp_path/'stack.tif'
    tifffile.imwrite(path, stack, metadata={'axes':'ZCYX'}, photometric='minisblack')
    for _ in range(2):
        win.open_image(str(path))
        qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
        assert win.state.draft.mode == 'projected_2d'
    win.combo_task.setCurrentIndex(2)
    win.seg_form.spin_min_size.setValue(20)
    win._on_analyze_image()
    qtbot.waitUntil(lambda: win._quick_worker is None, timeout=15000)
    result = win.state.quick_result
    assert result.classification is None
    name = next(iter(win.state.draft.marker_channels))
    win.state.draft.threshold_drafts[name].cutoff = 200
    win.state.draft.metric_drafts = [BiologicalMetricDraft(name='Positive', numerator_positive=(name,))]
    win._on_apply_thresholds()
    assert result.classification is not None
    assert int(result.classification.classifications.is_positive.sum()) == 1
    assert result.summary is not None
    assert win.inspector_stack.currentIndex() == 9
    assert win.quick_results.btn_export.isEnabled()
    win.quick_results.export_to(tmp_path/'classified')
    assert (tmp_path/'classified'/'summary.csv').is_file()
    annotation = QImage(str(tmp_path/'classified'/'annotated-view.png'))
    assert annotation.width() == 48


def test_folder_records_incompatible_images_without_study(tmp_path):
    tifffile.imwrite(tmp_path/'image.tif', image_data(), metadata={'axes':'CYX'}, photometric='minisblack')
    tifffile.imwrite(tmp_path/'stack.tif', np.repeat(image_data()[None],2,axis=0), metadata={'axes':'ZCYX'}, photometric='minisblack')
    report = export_folder_results(tmp_path, tmp_path/'output', AnalysisDraft(mode='direct_2d'), 'count')
    assert [row['status'] for row in report] == ['completed','failed']
    assert 'stack' in report[1]['error'].lower()
