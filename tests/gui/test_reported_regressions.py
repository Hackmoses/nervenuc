"""Regressions for report collisions, stable review selection, and repeat export."""
import json
import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtWidgets import QFileDialog, QMessageBox
from nervenuc.gui.main_window import MainWindow
from nervenuc.gui.quick_analysis import analyze_image, export_folder_results
from nervenuc.gui.state import AnalysisDraft


def make_window(qtbot, mode='direct_2d'):
    raw=np.zeros((2,48,48),dtype=np.uint16)
    raw[0,8:18,8:18]=1000
    raw[0,28:38,28:38]=1000
    raw[1,8:18,8:18]=600
    raw[1,28:38,28:38]=50
    draft=AnalysisDraft(mode=mode,declared_axes='CYX',marker_channels={'Signal':1},segmentation_values={'gaussian_sigma':0,'min_area_px':20})
    if mode=='volume_3d':
        raw=np.repeat(raw[None],3,axis=0)
        draft.declared_axes='ZCYX'
        draft.z_start_1based=1;draft.z_stop_1based=3
        draft.segmentation_values={'gaussian_sigma_xy_px':0,'gaussian_sigma_z_slices':0,'min_volume_voxels':20}
    result=analyze_image(raw,None,draft,'regression-source','signal')
    win=MainWindow();qtbot.addWidget(win)
    win.state.draft=draft
    win.state.quick_result=result
    win.state.results_stale=False
    win.state.configuration_differs_from_run=False
    win._render_active_view()
    win.state.notify()
    return win


@pytest.mark.parametrize('entry',['folder-report.csv','other-file.txt'])
def test_folder_collision_fails_before_analysis_and_preserves_existing(tmp_path,monkeypatch,entry):
    import nervenuc.gui.quick_analysis as quick
    source=tmp_path/'images';source.mkdir()
    tifffile.imwrite(source/'image.tif',np.zeros((16,16),dtype=np.uint16),metadata={'axes':'YX'})
    output=tmp_path/'output';output.mkdir()
    (output/entry).write_bytes(b'original bytes')
    def unexpected(*args,**kwargs):pytest.fail('Analysis ran before destination preflight')
    monkeypatch.setattr(quick,'analyze_image',unexpected)
    with pytest.raises(ValueError,match='empty output folder'):
        export_folder_results(source,output,AnalysisDraft(),'count')
    assert list(output.iterdir())==[output/entry]
    assert (output/entry).read_bytes()==b'original bytes'


@pytest.mark.parametrize('mode',['direct_2d','volume_3d'])
def test_immediate_restore_follows_rejected_object(qtbot,monkeypatch,mode):
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:pytest.fail(str(args[-1])))
    win=make_window(qtbot,mode)
    win._show_quick_review()
    table=win.qc_widget.table
    table.selectRow(0)
    identifier=table.item(0,0).text()
    original_count=len(win.state.quick_result.objects)
    original_snapshot=win.state.quick_result.reviewed
    win._on_qc_reject()
    assert len(win.state.quick_result.objects)==original_count-1
    assert len(original_snapshot.objects)==original_count
    selected_row=table.selectedItems()[0].row()
    assert table.item(selected_row,0).text()==identifier
    assert 'Rejected' in table.item(selected_row,1).text()
    win._on_qc_restore()
    assert len(win.state.quick_result.objects)==original_count
    selected_row=table.selectedItems()[0].row()
    assert table.item(selected_row,0).text()==identifier
    assert 'Accepted' in table.item(selected_row,1).text()


def test_repeat_export_saves_updated_version_without_modifying_original(qtbot,tmp_path,monkeypatch):
    win=make_window(qtbot)
    target=tmp_path/'results'
    win.quick_results.export_to(target)
    original={path.name:path.read_bytes() for path in target.iterdir()}
    win._show_quick_review();win.qc_widget.table.selectRow(0);win._on_qc_reject()
    (target/'export-001').mkdir()
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args:str(target))
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:pytest.fail(str(args[-1])))
    win.quick_results.export()
    version=target/'export-002'
    assert json.loads((version/'analysis.json').read_text())['object_count']==1
    assert len(pd.read_csv(version/'objects.csv'))==1
    assert len(pd.read_csv(target/'objects.csv'))==2
    for name,contents in original.items():assert (target/name).read_bytes()==contents


def test_repeat_export_cancel_and_choose_another_folder(qtbot,tmp_path,monkeypatch):
    win=make_window(qtbot);target=tmp_path/'original';win.quick_results.export_to(target)
    original={path.name:path.read_bytes() for path in target.iterdir()}
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args:str(target))
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.Cancel)
    win.quick_results.export()
    assert not (target/'export-001').exists()
    other=tmp_path/'other'
    choices=iter([str(target),str(other)])
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args:next(choices))
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.No)
    win.quick_results.export()
    assert (other/'objects.csv').is_file()
    for name,contents in original.items():assert (target/name).read_bytes()==contents


def test_optional_old_export_artifact_still_counts_as_collision(qtbot,tmp_path):
    win=make_window(qtbot)
    target=tmp_path/'old';target.mkdir()
    (target/'classification.csv').write_bytes(b'previous classifications')
    with pytest.raises(ValueError,match='already contains'):
        win.quick_results.export_to(target)
    assert list(target.iterdir())==[target/'classification.csv']
