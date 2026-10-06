"""Single-image results with optional signal classification and direct exports."""
from pathlib import Path
import json
from dataclasses import asdict
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QComboBox, QFileDialog, QMessageBox, QHeaderView
from nervenuc.export import write_dataframe_csv

class QuickResultsWidget(QWidget):
    review_requested = Signal()
    thresholds_requested = Signal()
    def __init__(self, state, canvas):
        super().__init__()
        self.state, self.canvas = state, canvas
        layout = QVBoxLayout(self)
        self.summary = QLabel('Run Analyze to see counts and measurements.')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.notice = QLabel('')
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.table_kind = QComboBox()
        self.table_kind.addItems(['Objects', 'Signal measurements', 'Positive / negative calls', 'Custom measurements'])
        self.table_kind.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.table_kind)
        self.btn_details = QPushButton('Show all columns')
        self.btn_details.setCheckable(True)
        self.btn_details.toggled.connect(self.refresh)
        layout.addWidget(self.btn_details)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.btn_review = QPushButton('Review / edit detections…')
        self.btn_review.clicked.connect(self.review_requested.emit)
        layout.addWidget(self.btn_review)
        self.btn_thresholds = QPushButton('Set positive / negative cutoffs…')
        self.btn_thresholds.clicked.connect(self.thresholds_requested.emit)
        layout.addWidget(self.btn_thresholds)
        self.btn_export = QPushButton('Export…')
        self.btn_export.clicked.connect(self.export)
        layout.addWidget(self.btn_export)
        self.state.subscribe(self.refresh)
        self.refresh()

    def dataframe(self):
        result = self.state.quick_result
        if result is None:
            return pd.DataFrame()
        if self.table_kind.currentIndex() == 1:
            return result.measurement.measurements if result.measurement else pd.DataFrame()
        if self.table_kind.currentIndex() == 2:
            return result.classification.classifications if result.classification else pd.DataFrame()
        if self.table_kind.currentIndex() == 3:
            return result.summary.metrics if result.summary is not None else pd.DataFrame()
        return result.objects

    def refresh(self):
        result = self.state.quick_result
        stale = self.state.results_stale or self.state.configuration_differs_from_run
        self.btn_export.setEnabled(result is not None and not stale)
        self.btn_review.setEnabled(result is not None and not stale)
        self.btn_thresholds.setVisible(result is not None and result.task == 'positivity')
        self.btn_thresholds.setEnabled(result is not None and result.measurement is not None)
        self.notice.setText('Settings changed. Run Analyze again, or apply edited cutoffs, to update these results.' if stale and result else 'Set cutoffs to classify positivity. Current results contain raw measurements.' if result and result.task == 'positivity' and result.classification is None else '')
        if result:
            policy = 'Manually edited' if result.manually_reviewed else 'Automatically accepted; manual review optional'
            self.summary.setText(f'{len(result.objects)} objects detected. {policy}.')
            if result.classification is not None:
                calls = result.classification.classifications
                lines = []
                for marker, rows in calls.groupby('marker_name'):
                    positive = int(rows['is_positive'].sum())
                    percentage = f'{100*positive/len(rows):.1f}%' if len(rows) else 'undefined'
                    lines.append(f'{marker}: {positive}/{len(rows)} positive ({percentage})')
                self.summary.setText(self.summary.text() + '\n' + '\n'.join(lines))
        else:
            self.summary.setText('Run Analyze to see counts and measurements.')
        df = self.dataframe()
        if not self.btn_details.isChecked():
            preferred = ('current_label', 'area_px', 'voxel_count', 'volume_um3', 'centroid_x', 'centroid_y', 'centroid_z') if self.table_kind.currentIndex() == 0 else ('review_id', 'marker_name', 'mean_intensity', 'median_intensity', 'integrated_intensity') if self.table_kind.currentIndex() == 1 else ('review_id', 'marker_name', 'is_positive') if self.table_kind.currentIndex() == 2 else tuple(df.columns)
            df = df[[column for column in preferred if column in df.columns]]
        friendly = {'current_label':'Object', 'review_id':'Object ID', 'area_px':'Area (pixels)', 'voxel_count':'Volume (voxels)', 'volume_um3':'Volume (µm³)', 'centroid_x':'X', 'centroid_y':'Y', 'centroid_z':'Z', 'marker_name':'Signal', 'mean_intensity':'Mean signal', 'median_intensity':'Median signal', 'integrated_intensity':'Total signal', 'is_positive':'Positive'}
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([friendly.get(column, column) if not self.btn_details.isChecked() else column for column in df.columns])
        for row, values in enumerate(df.itertuples(index=False, name=None)):
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    EXPORT_FILES = frozenset(('objects.csv', 'measurements.csv', 'classification.csv', 'summary.csv', 'labels.tif', 'detection-input.tif', 'annotated-view.png', 'analysis.json'))

    @classmethod
    def has_existing_results(cls, directory):
        return any((Path(directory)/name).exists() for name in cls.EXPORT_FILES)

    @staticmethod
    def next_export_folder(directory):
        parent = Path(directory)
        number = 1
        while (parent/f'export-{number:03d}').exists():
            number += 1
        return parent/f'export-{number:03d}'

    def export_to(self, directory):
        result = self.state.quick_result
        if result is None or self.state.results_stale or self.state.configuration_differs_from_run:
            raise ValueError('Run Analyze with the current settings before exporting.')
        target = Path(directory)
        names = ['objects.csv', 'analysis.json', 'annotated-view.png', 'labels.tif']
        if result.detection_image is not None: names.append('detection-input.tif')
        if result.summary is not None: names.append('summary.csv')
        if result.measurement is not None: names.append('measurements.csv')
        if result.classification is not None: names.append('classification.csv')
        if self.has_existing_results(target):
            raise ValueError('Choose a new folder; this folder already contains analysis files.')
        target.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv(result.objects, target/'objects.csv')
        if result.measurement is not None: write_dataframe_csv(result.measurement.measurements, target/'measurements.csv')
        if result.classification is not None: write_dataframe_csv(result.classification.classifications, target/'classification.csv')
        import tifffile
        if result.detection_image is not None:
            tifffile.imwrite(target/'detection-input.tif', result.detection_image, metadata={'axes': 'ZYX' if result.mode == 'volume_3d' else 'YX'}, photometric='minisblack')
        tifffile.imwrite(target/'labels.tif', result.labels, metadata={'axes': 'ZYX' if result.mode == 'volume_3d' else 'YX'}, photometric='minisblack')
        if result.summary is not None: write_dataframe_csv(result.summary.metrics, target/'summary.csv')
        payload = dict(result.settings, source_id=result.source_id, object_count=len(result.objects), review_policy='manual' if result.manually_reviewed else 'auto_accept', review_history=[asdict(action) for action in result.reviewed.history], source_file=str(self.state.loaded_path) if self.state.loaded_path else None)
        (target/'analysis.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        from PySide6.QtGui import QPainter
        from .viewer.renderer import render_display_slice
        from .viewer.overlay import render_label_overlay
        nuclear, markers = result.planes
        z = min(self.state.current_z_index, nuclear.shape[0]-1) if result.mode == 'volume_3d' else 0
        planes = {result.settings['nuclear_channel']: nuclear[z] if result.mode == 'volume_3d' else nuclear}
        planes.update({result.settings['marker_channels'][name]: values[z] if result.mode == 'volume_3d' else values for name, values in markers.items()})
        display = self.state.display
        if display.show_preprocessed and result.detection_image is not None:
            planes = {result.settings['nuclear_channel']: result.detection_image[z] if result.mode == 'volume_3d' else result.detection_image}
        payload['annotation_image_basis'] = 'processed_detection' if display.show_preprocessed and result.detection_image is not None else 'raw'
        annotation = render_display_slice(planes, display.channel_min, display.channel_max, display.channel_gamma, display.channel_visible, display.channel_color)
        overlay = render_label_overlay(result.labels[z] if result.mode == 'volume_3d' else result.labels, opacity=display.overlay_opacity)
        if not overlay.isNull():
            painter = QPainter(annotation)
            painter.drawImage(0, 0, overlay)
            painter.end()
        payload['annotation_z_index'] = z if result.mode == 'volume_3d' else None
        (target/'analysis.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        if not annotation.save(str(target/'annotated-view.png')):
            raise ValueError('Could not save the annotated view.')
        return target

    def export(self):
        directory = QFileDialog.getExistingDirectory(self, 'Export image results')
        while directory:
            try:
                if self.has_existing_results(directory):
                    next_folder = self.next_export_folder(directory)
                    choice = QMessageBox.question(
                        self, 'Results already exist',
                        f'This folder already contains results. Save this export in {next_folder.name}?\n\nYes: save a new version and keep the original files.\nNo: choose another folder.',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Yes,
                    )
                    if choice == QMessageBox.StandardButton.Cancel:
                        return
                    if choice == QMessageBox.StandardButton.No:
                        directory = QFileDialog.getExistingDirectory(self, 'Choose another export folder')
                        continue
                    if choice != QMessageBox.StandardButton.Yes:
                        return
                    directory = str(next_folder)
                target = self.export_to(directory)
                self.notice.setText(f'Results saved to {target}')
            except Exception as exc:
                QMessageBox.warning(self, 'Export failed', str(exc))
            return
