"""Preprocessing preserves raw quantification and records detection operations."""
from dataclasses import asdict
import numpy as np
import pytest
from nervenuc.gui.preprocessing import PreprocessingConfig, preprocess_image
from nervenuc.gui.quick_analysis import analyze_image
from nervenuc.gui.state import AnalysisDraft


def test_background_subtraction_removes_baseline_and_preserves_features():
    y,x=np.ogrid[:48,:48]
    raw=80 + np.broadcast_to(x*.3,(48,48)).copy()
    spot=(y-24)**2+(x-24)**2<16
    raw[spot]+=200
    before=raw.copy()
    corrected=preprocess_image(raw,PreprocessingConfig(subtract_background=True,background_radius_px=8,denoise=False))
    np.testing.assert_array_equal(raw,before)
    assert corrected.min() >= 0
    assert np.median(corrected[~spot]) < 5
    assert corrected[24,24] > 100


def test_mild_gaussian_reduces_noise_and_stack_planes_do_not_mix():
    rng=np.random.default_rng(21)
    raw=100+rng.normal(0,10,(64,64))
    corrected=preprocess_image(raw,PreprocessingConfig(gaussian_sigma_px=.8))
    assert corrected.std() < raw.std()/2
    stack=np.stack([raw,raw+200])
    filtered=preprocess_image(stack,PreprocessingConfig(gaussian_sigma_px=.8))
    np.testing.assert_allclose(filtered[1]-filtered[0],200)


def test_estimated_illumination_flattens_gradual_field_and_scaling_is_shared():
    raw=np.broadcast_to(np.linspace(50,200,64),(64,64)).copy()
    corrected=preprocess_image(raw,PreprocessingConfig(denoise=False,correct_illumination=True,illumination_sigma_px=5))
    assert corrected.std()/corrected.mean() < raw.std()/raw.mean()/2
    plane=np.zeros((16,16));plane[4:12,4:12]=100
    stack=np.stack([plane,plane*2])
    scaled=preprocess_image(stack,PreprocessingConfig(denoise=False,normalize_intensity=True))
    assert scaled.min()==0 and scaled.max()==1
    np.testing.assert_allclose(scaled[1],scaled[0]*2)
    zeros=preprocess_image(np.zeros((8,8)),PreprocessingConfig(subtract_background=True,correct_illumination=True,normalize_intensity=True))
    assert np.isfinite(zeros).all() and not zeros.any()


@pytest.mark.parametrize('values',[{'background_radius_px':0},{'gaussian_sigma_px':float('nan')},{'illumination_sigma_px':0}])
def test_invalid_processing_parameters_are_rejected(values):
    with pytest.raises(ValueError):PreprocessingConfig(**values)


@pytest.mark.parametrize('mode',['direct_2d','projected_2d','volume_3d'])
def test_detection_preprocessing_does_not_change_signal_measurements(mode):
    raw=np.zeros((2,48,48),dtype=float)
    raw[0]=80+np.arange(48)[None,:]*.3
    raw[0,10:20,10:20]+=200
    raw[1]=250+np.arange(48)[None,:]
    draft=AnalysisDraft(mode=mode,declared_axes='CYX',marker_channels={'Signal':1},segmentation_values={'threshold_method':'manual','manual_threshold':50,'gaussian_sigma':1,'min_area_px':20},preprocessing_values=asdict(PreprocessingConfig(subtract_background=True,background_radius_px=8,gaussian_sigma_px=.5)))
    if mode!='direct_2d':
        raw=np.repeat(raw[None],3,axis=0)
        draft.declared_axes='ZCYX';draft.z_start_1based=1;draft.z_stop_1based=3
    if mode=='volume_3d':
        draft.segmentation_values={'threshold_method':'manual','manual_threshold':50,'gaussian_sigma_xy_px':1,'min_volume_voxels':20}
    original=raw.copy()
    result=analyze_image(raw,None,draft,'test-source','signal')
    assert len(result.objects)>0
    assert result.detection_image is not None
    assert result.settings['measurement_image_basis']=='raw'
    assert result.settings['preprocessing']['stage']=='after_spatial_preparation_before_detection'
    labels=result.labels
    marker=result.planes[1]['Signal']
    for _,row in result.objects.iterrows():
        identifier=row.current_label
        name_key='review_id' if 'review_id' in result.measurement.measurements.columns else 'current_label'
        key=row.review_id if name_key=='review_id' else identifier
        record=result.measurement.measurements.loc[result.measurement.measurements[name_key]==key].iloc[0]
        assert record.mean_intensity==pytest.approx(marker[labels==identifier].mean())
    np.testing.assert_array_equal(raw,original)


def test_ui_preprocessing_change_blocks_stale_export(qtbot):
    from nervenuc.gui.main_window import MainWindow
    win=MainWindow();qtbot.addWidget(win)
    win.state.results_stale=False
    win.preprocessing.background.setChecked(True)
    assert win.state.results_stale
    assert win.state.draft.preprocessing_values['subtract_background']
    win.preprocessing.normalize.setChecked(True)
    assert win.state.draft.preprocessing_values['normalize_intensity']


def test_processed_preview_export_and_manual_scaling_units(qtbot,tmp_path):
    import json
    import tifffile
    from nervenuc.gui.main_window import MainWindow
    win=MainWindow();qtbot.addWidget(win)
    raw=np.zeros((2,48,48),dtype=np.uint16)
    raw[0]=80;raw[0,10:20,10:20]=280
    raw[1]=250
    path=tmp_path/'input.tif'
    tifffile.imwrite(path,raw,metadata={'axes':'CYX'},photometric='minisblack')
    win.open_image(str(path))
    qtbot.waitUntil(lambda:win._quick_worker is None,timeout=15000)
    win.combo_task.setCurrentIndex(1)
    win.preprocessing.background.setChecked(True)
    win.preprocessing.radius.setValue(8)
    win.preprocessing.normalize.setChecked(True)
    assert win.seg_form.spin_manual.maximum()==1
    assert win.seg_form.spin_manual.value()==.5
    win._on_analyze_image()
    qtbot.waitUntil(lambda:win._quick_worker is None,timeout=15000)
    result=win.state.quick_result
    assert result is not None and result.measurement is not None
    assert result.measurement.measurements.mean_intensity.iloc[0]==250
    assert result.settings['segmentation']['gaussian_sigma']==0
    assert win.viewer_controls.cb_preprocessed.isEnabled()
    win.viewer_controls.cb_preprocessed.setChecked(True)
    win.quick_results.export_to(tmp_path/'export')
    settings=json.loads((tmp_path/'export'/'analysis.json').read_text())
    assert settings['measurement_image_basis']=='raw'
    assert settings['annotation_image_basis']=='processed_detection'
    assert settings['preprocessing']['normalize_intensity']
    processed=tifffile.imread(tmp_path/'export'/'detection-input.tif')
    assert processed.min()>=0 and processed.max()<=1
    np.testing.assert_array_equal(win.state.loaded_raw_array,raw)
    win.preprocessing.sigma.setValue(.7)
    assert not win.quick_results.btn_export.isEnabled()
