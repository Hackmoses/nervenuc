"""Main application window assembling navigation, viewer, and context inspector."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, cast

import numpy as np
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .controllers import (
    AnalysisController,
    BatchController,
    ExportController,
    FijiController,
    SourceController,
)
from .state import CanonicalMode, GuiSessionState, MarkerThresholdDraft
from .quick_analysis import analyze_image, export_folder_results
from .workers import FunctionWorker
from .widgets.quick_results import QuickResultsWidget
from .widgets.preprocessing import PreprocessingWidget
from .widgets import (
    BatchManifestWidget,
    ChannelAssignmentTable,
    FijiConfigDialog,
    MarkerThresholdCard,
    MetricBuilderWidget,
    MicroscopyCanvas,
    QualityControlTable,
    SegmentationConfigForm,
    SummaryDashboardWidget,
    ViewerControlsWidget,
    render_display_slice,
    render_label_overlay,
)


class MainWindow(QMainWindow):
    """Main desktop interface for NerveNuc researcher workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NerveNuc — Microscopy Quantitative Analysis")
        self.resize(1400, 900)

        self.state = GuiSessionState()
        self.threadpool = QThreadPool.globalInstance()

        # Controllers
        self.source_ctrl = SourceController(self.state)
        self.analysis_ctrl = AnalysisController(self.state)
        self.batch_ctrl = BatchController(self.state)
        self.export_ctrl = ExportController(self.state)
        self.fiji_ctrl = FijiController(self.state, self.threadpool)

        self._quick_worker = None
        self._opening_path = None
        self._init_ui()
        self._init_menu()

        self.state.subscribe(self._on_state_updated)
        self._update_fiji_action_states()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Left Navigation
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(200)
        self.nav_list.addItems(["Open", "Set up", "Analyze", "Results"])
        self._nav_pages = [0, 1, 3, 9]
        self.nav_list.setCurrentRow(0)
        splitter.addWidget(self.nav_list)

        # 2. Center: Viewer & Canvas
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = MicroscopyCanvas()
        self.viewer_controls = ViewerControlsWidget(self.state)
        center_layout.addWidget(self.canvas, stretch=1)
        center_layout.addWidget(self.viewer_controls)

        splitter.addWidget(center_widget)

        # 3. Right: Context Inspector Stack
        self.inspector_stack = QStackedWidget()
        self.inspector_stack.setMinimumWidth(360)

        # Build pages
        self._build_source_page()
        self._build_mode_channels_page()
        self._build_spatial_prep_page()
        self._build_segmentation_page()
        self._build_qc_page()
        self._build_thresholds_page()
        self._build_metrics_page()
        self._build_batch_page()
        self._build_results_page()
        self.quick_results = QuickResultsWidget(self.state, self.canvas)
        self.quick_results.review_requested.connect(self._show_quick_review)
        self.quick_results.thresholds_requested.connect(lambda: self.inspector_stack.setCurrentIndex(5))
        self.inspector_stack.addWidget(self.quick_results)
        # Stack selection belongs to setup, with no separate preparation action.
        setup_layout = self.inspector_stack.widget(1).layout()
        setup_layout.addWidget(self.stack_options)
        next_step = QPushButton('Continue to Analyze')
        next_step.clicked.connect(lambda: self.nav_list.setCurrentRow(2))
        setup_layout.addWidget(next_step)
        setup_layout.addStretch()
        self.btn_load_prep.hide()
        self.seg_form.btn_run.clicked.disconnect()
        self.seg_form.btn_run.setText("Analyze image")
        self.preprocessing = PreprocessingWidget(self.state)
        self.seg_form.layout().insertWidget(0, self.preprocessing)
        self.preprocessing.normalize.toggled.connect(self._on_detection_scaling_changed)
        # Gaussian controls are owned by the preprocessing stage in quick analysis.
        self.seg_form.preprocessing_managed = True
        self.seg_form.spin_sigma_xy.hide()
        self.seg_form.spin_sigma_z.hide()
        self.seg_form.form_layout.setRowVisible(self.seg_form.spin_sigma_xy, False)
        self.seg_form.form_layout.setRowVisible(self.seg_form.spin_sigma_z, False)
        self.seg_form.btn_advanced.toggled.connect(lambda _: (self.seg_form.form_layout.setRowVisible(self.seg_form.spin_sigma_xy, False), self.seg_form.form_layout.setRowVisible(self.seg_form.spin_sigma_z, False)))
        self.seg_form.btn_run.clicked.connect(self._on_analyze_image)
        self.seg_form.btn_run.setEnabled(False)
        self.btn_folder = QPushButton('Process a folder with these settings…')
        self.btn_folder.clicked.connect(self._on_process_folder)
        self.seg_form.layout().addWidget(self.btn_folder)
        self.btn_folder.setEnabled(False)
        self.qc_widget.btn_accept.hide()
        for shortcut in self.qc_widget._shortcuts:
            if shortcut.key().toString() == "A": shortcut.setEnabled(False)
        self.qc_widget.btn_auto.hide()
        self.qc_widget.btn_finalize.setText("Update results")
        self.qc_widget.btn_finalize.hide()
        back = QPushButton("Back to image results")
        back.clicked.connect(lambda: self.inspector_stack.setCurrentIndex(9))
        self.qc_widget.layout().addWidget(back)

        for index in range(self.inspector_stack.count()):
            page = self.inspector_stack.widget(index)
            self.inspector_stack.removeWidget(page)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.inspector_stack.insertWidget(index, scroll)
        splitter.addWidget(self.inspector_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        splitter.setSizes([200, 700, 480])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a microscopy image to begin.")

        # Connect Navigation
        self.inspector_stack.setCurrentIndex(self.nav_list.currentRow())
        self.nav_list.currentRowChanged.connect(lambda row: self.inspector_stack.setCurrentIndex(self._nav_pages[row]))
        self.nav_list.itemClicked.connect(lambda item: self.inspector_stack.setCurrentIndex(self._nav_pages[self.nav_list.row(item)]))
        self.viewer_controls.slice_changed.connect(self._render_active_view)
        self.viewer_controls.display_settings_changed.connect(self._render_active_view)
        self.viewer_controls.fit_requested.connect(self.canvas.fit_image_to_view)

    def _init_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        advanced_menu = menu.addMenu("Advanced")

        act_open = file_menu.addAction("Open Image/Stack...")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_file)

        file_menu.addSeparator()
        act_load_preset = advanced_menu.addAction("Load study preset...")
        act_load_preset.triggered.connect(self._on_load_preset)
        act_save_preset = advanced_menu.addAction("Save study preset...")
        act_save_preset.setShortcut("Ctrl+Shift+S")
        act_save_preset.triggered.connect(self._on_save_preset)

        file_menu.addSeparator()
        act_exit = file_menu.addAction("Exit")
        act_exit.triggered.connect(self.close)

        advanced_menu.addSeparator()
        for title, index in (("Review detections", 4), ("Positive / negative cutoffs", 5), ("Custom measurements", 6), ("Study / batch analysis", 7), ("Study results", 8)):
            action = advanced_menu.addAction(title)
            action.triggered.connect(lambda checked=False, page=index: self.inspector_stack.setCurrentIndex(page))
        # Tools -> Fiji / ImageJ Menu
        tools_menu = menu.addMenu("Tools")
        fiji_menu = tools_menu.addMenu("Fiji / ImageJ")

        self.act_fiji_open_image = fiji_menu.addAction("Open Current Image in Fiji")
        self.act_fiji_open_image.triggered.connect(self._on_fiji_open_image)

        self.act_fiji_open_volume = fiji_menu.addAction("Open Current Volume / Labels in Fiji...")
        self.act_fiji_open_volume.triggered.connect(self._on_fiji_open_volume)

        fiji_menu.addSeparator()

        self.act_fiji_export_rois = fiji_menu.addAction("Export Active QC Objects as ROI Set...")
        self.act_fiji_export_rois.triggered.connect(self._on_fiji_export_rois)

        self.act_fiji_import_rois = fiji_menu.addAction("Import ROI Set as Manual QC Objects...")
        self.act_fiji_import_rois.triggered.connect(self._on_fiji_import_rois)

        self.act_fiji_replace_rois = fiji_menu.addAction("Replace Selected Object with Fiji ROIs...")
        self.act_fiji_replace_rois.triggered.connect(self._on_fiji_replace_rois)

        fiji_menu.addSeparator()

        self.act_fiji_config = fiji_menu.addAction("Configure Fiji...")
        self.act_fiji_config.triggered.connect(self._on_fiji_config)

    # --- Inspector Pages Builders ---

    def _build_source_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        grp = QGroupBox("Source Image")
        form = QFormLayout(grp)
        self.lbl_src_path = QLabel("No file loaded")
        self.lbl_src_shape = QLabel("-")
        self.lbl_src_axes = QLabel("-")
        self.lbl_src_dtype = QLabel("-")

        btn_open = QPushButton("Choose File...")
        btn_open.clicked.connect(self._on_open_file)

        form.addRow("Path:", self.lbl_src_path)
        form.addRow("Dimensions:", self.lbl_src_shape)
        form.addRow("Axes:", self.lbl_src_axes)
        form.addRow("Dtype:", self.lbl_src_dtype)
        form.addRow("", btn_open)
        layout.addWidget(grp)
        layout.addStretch()
        self.inspector_stack.addWidget(page)

    def _build_mode_channels_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Mode Selector
        grp_mode = QGroupBox("Image setup")
        form_mode = QFormLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Direct 2D", "direct_2d")
        self.combo_mode.addItem("Projected 2D from Z-stack", "projected_2d")
        self.combo_mode.addItem("True 3D", "volume_3d")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_mode.setItemText(0, "Analyze this 2D image")
        self.combo_mode.setItemText(1, "Analyze a stack projection")
        self.combo_mode.setItemText(2, "Analyze the full stack in 3D")
        form_mode.addRow("Image / stack:", self.combo_mode)
        self.combo_task = QComboBox()
        self.combo_task.addItem("Count objects", "count")
        self.combo_task.addItem("Count and measure signal", "signal")
        self.combo_task.addItem("Count and classify marker positivity", "positivity")
        self.combo_task.currentIndexChanged.connect(self.state.mark_draft_modified)
        self.mode_form = form_mode
        self.spin_time = QSpinBox()
        self.spin_time.setRange(1, 1)
        self.spin_time.valueChanged.connect(self._on_time_changed)
        self.spin_time.hide()
        form_mode.addRow("Time frame:", self.spin_time)
        form_mode.setRowVisible(self.spin_time, False)
        form_mode.addRow("Task:", self.combo_task)
        task_hint = QLabel("Choose the channel containing the objects to detect. Fluorescence nuclei are the default use case; brightfield objects may need manual threshold settings.")
        task_hint.setWordWrap(True)
        form_mode.addRow(task_hint)
        layout.addWidget(grp_mode)

        # Channel Mapping Table
        grp_ch = QGroupBox("Channel Assignment")
        ch_layout = QVBoxLayout(grp_ch)
        self.channel_table = ChannelAssignmentTable(self.state)
        ch_layout.addWidget(self.channel_table)
        layout.addWidget(grp_ch)

        self.inspector_stack.addWidget(page)

    def _build_spatial_prep_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        grp_prep = QGroupBox("Stack options")
        self.stack_options = grp_prep
        form = QFormLayout(grp_prep)

        self.spin_z_start = QSpinBox()
        self.spin_z_start.setRange(1, 1000)
        self.spin_z_start.setValue(1)

        self.spin_z_stop = QSpinBox()
        self.spin_z_stop.setRange(1, 1000)
        self.spin_z_stop.setValue(1)

        self.combo_proj = QComboBox()
        self.combo_proj.addItems(["max", "mean"])

        form.addRow("Z Start Slice (1-based):", self.spin_z_start)
        form.addRow("Z End Slice (1-based):", self.spin_z_stop)
        form.addRow("Projection Method:", self.combo_proj)
        self.spin_z_start.valueChanged.connect(self._on_stack_options_changed)
        self.spin_z_stop.valueChanged.connect(self._on_stack_options_changed)
        self.combo_proj.currentTextChanged.connect(self._on_stack_options_changed)

        self.stack_options.setVisible(False)
        self.btn_load_prep = QPushButton("Prepare & Load into Session")
        self.btn_load_prep.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_load_prep.clicked.connect(self._on_prepare_session)
        form.addRow("", self.btn_load_prep)

        layout.addWidget(grp_prep)
        layout.addStretch()
        self.inspector_stack.addWidget(page)

    def _build_segmentation_page(self) -> None:
        self.seg_form = SegmentationConfigForm(self.state)
        self.seg_form.run_segmentation_requested.connect(self._on_run_segmentation)
        self.inspector_stack.addWidget(self.seg_form)

    def _build_qc_page(self) -> None:
        self.qc_widget = QualityControlTable(self.state)
        self.qc_widget.btn_accept.clicked.connect(self._on_qc_accept)
        self.qc_widget.btn_reject.clicked.connect(self._on_qc_reject)
        self.qc_widget.btn_restore.clicked.connect(self._on_qc_restore)
        self.qc_widget.btn_auto.clicked.connect(self._on_qc_auto)
        self.qc_widget.btn_finalize.clicked.connect(self._on_qc_finalize)
        self.qc_widget.object_selected.connect(self._on_object_selected)
        self.qc_widget.table.itemSelectionChanged.connect(self._update_fiji_action_states)
        self.inspector_stack.addWidget(self.qc_widget)

    def _build_thresholds_page(self) -> None:
        page = QWidget()
        self.thresholds_layout = QVBoxLayout(page)
        self.lbl_thresh_info = QLabel("Set cutoffs, then Apply to classify positive and negative objects. Initial suggestions use the median measured signal; verify them for your experiment.")
        self.lbl_thresh_info.setWordWrap(True)
        self.thresholds_layout.addWidget(self.lbl_thresh_info)
        self.threshold_cards_container = QWidget()
        self.cards_vbox = QVBoxLayout(self.threshold_cards_container)
        self.thresholds_layout.addWidget(self.threshold_cards_container)
        self.thresholds_layout.addStretch()
        self.inspector_stack.addWidget(page)

    def _build_metrics_page(self) -> None:
        self.metric_builder = MetricBuilderWidget(self.state)
        self.inspector_stack.addWidget(self.metric_builder)

    def _build_batch_page(self) -> None:
        self.batch_widget = BatchManifestWidget(self.batch_ctrl, self.state)
        self.batch_widget.batch_running_changed.connect(self._on_batch_running_changed)
        self.batch_widget.batch_finished.connect(self._on_batch_finished)
        self.inspector_stack.addWidget(self.batch_widget)

    def _on_batch_running_changed(self, running: bool) -> None:
        """Prevent configuration drift while the authoritative batch runner is active."""
        self.nav_list.setEnabled(not running)
        self.menuBar().setEnabled(not running)
        self.status_bar.showMessage(
            "Running batch with auto-accept QC..." if running else "Batch execution finished."
        )

    def _on_batch_finished(self, result: object, complete: bool) -> None:
        """Refresh the authoritative study view after batch completion."""
        self.export_ctrl.clear_group_assignments()
        if complete:
            self.results_dashboard.refresh_results()
            self.inspector_stack.setCurrentIndex(8)
            self.status_bar.showMessage("Batch completed. Study results and reproducible export are available.")
        else:
            self.status_bar.showMessage("Batch incomplete. Review failed sample details; export is disabled.")

    def _build_results_page(self) -> None:
        self.results_dashboard = SummaryDashboardWidget(self.export_ctrl, self.state)
        self.inspector_stack.addWidget(self.results_dashboard)

    # --- Slots & Actions ---

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)')
        if path:
            self.open_image(path)

    def open_image(self, path, declared_axes=None):
        if self._quick_worker is not None: return
        self._opening_path = path
        def load():
            isolated = GuiSessionState()
            controller = SourceController(isolated)
            controller.inspect_file(path, declared_axes=declared_axes)
            controller.load_source_preview()
            return isolated
        self._quick_worker = FunctionWorker(load)
        self._set_quick_busy(True)
        self.status_bar.showMessage('Opening image…')
        self._quick_worker.signals.finished.connect(self._on_source_loaded)
        self._quick_worker.signals.error.connect(self._on_quick_error)
        self.threadpool.start(self._quick_worker)

    def _on_source_loaded(self, loaded):
        self._opening_path = None
        self._quick_worker = None
        self._set_quick_busy(False)
        for name in ('loaded_path', 'loaded_raw_array', 'preview_axes', 'source_id', 'inspection', 'source_metadata', 'current_z_index', 'display'):
            setattr(self.state, name, getattr(loaded, name))
        self.state.draft = loaded.draft
        self.seg_form.load_from_draft()
        self.preprocessing.load_from_draft()
        self._on_detection_scaling_changed(self.preprocessing.normalize.isChecked())
        self.state.session_2d = None
        self.state.session_3d = None
        self.state.quick_result = None
        self.state.batch_result = None
        self.state.results_stale = True
        self.state.configuration_differs_from_run = True
        path = str(loaded.loaded_path)
        insp = loaded.inspection
        self.lbl_src_path.setText(path)
        self.lbl_src_path.setWordWrap(True)
        self.lbl_src_shape.setText(str(insp.shape))
        self.lbl_src_axes.setText(str(insp.axes))
        self.lbl_src_dtype.setText(str(insp.dtype))
        self.spin_time.setRange(1, insp.size_t or 1)
        self.spin_time.setValue(1)
        self.mode_form.setRowVisible(self.spin_time, (insp.size_t or 1) > 1)
        self.state.draft.time_index = 0 if insp.size_t else None


        self.channel_table.populate_from_inspection()
        self.stack_options.setVisible((insp.size_z or 1) > 1)
        self.seg_form.btn_run.setEnabled(True)
        self.btn_folder.setEnabled(True)
        self.nav_list.setCurrentRow(1)
        self.inspector_stack.setCurrentIndex(1)
        self.viewer_controls.populate_channels()
        depth = insp.size_z or 1
        self.spin_z_start.setRange(1, depth)
        self.spin_z_stop.setRange(1, depth)
        self.spin_z_start.setValue(1)
        self.spin_z_stop.setValue(depth)
        if depth > 1:
            self.combo_mode.setCurrentIndex(1)
        else:
            self.combo_mode.setCurrentIndex(0)
        self._on_mode_changed()
        if insp.size_z:
            self.spin_z_stop.setValue(insp.size_z)
            self.viewer_controls.set_z_range(insp.size_z)
        else:
            self.viewer_controls.set_z_range(1)

        self.viewer_controls.z_slider.setValue(0)
        self._render_active_view()
        self.canvas.fit_image_to_view()

        self.status_bar.showMessage(
            f"Source preview: {Path(path).name} (Axes: {insp.axes}; time frame 1). "
            "Assign channels, choose a task, then Analyze."
        )
        self.state.notify()

    def _on_detection_scaling_changed(self, enabled):
        if enabled:
            self._raw_manual_cutoff = self.seg_form.spin_manual.value()
            self.seg_form.spin_manual.setRange(0,1)
            self.seg_form.spin_manual.setValue(.5)
        else:
            self.seg_form.spin_manual.setRange(0,65535)
            self.seg_form.spin_manual.setValue(getattr(self, '_raw_manual_cutoff',500))
        self.seg_form.form_layout.labelForField(self.seg_form.spin_manual).setText('Manual cutoff (0–1):' if enabled else 'Manual cutoff (raw intensity):')

    def _on_stack_options_changed(self, *_args):
        self.state.draft.z_start_1based = self.spin_z_start.value()
        self.state.draft.z_stop_1based = self.spin_z_stop.value()
        self.state.draft.projection_method = self.combo_proj.currentText()
        self.state.mark_draft_modified()

    def _on_time_changed(self, value):
        self.state.draft.time_index = value - 1
        self.state.quick_result = None
        self.state.mark_draft_modified()
        self._render_active_view()

    def _on_mode_changed(self) -> None:
        self.state.quick_result = None
        mode_str = cast(CanonicalMode, self.combo_mode.currentData())
        self.state.set_mode(mode_str)
        self.seg_form.update_mode_visibility()
        self.seg_form._sync_to_draft()
        if self.state.inspection is not None:
            self.state.current_z_index = 0
            self.viewer_controls.set_z_range(self.state.inspection.size_z or 1)
            self.viewer_controls.z_slider.setValue(0)
            self._render_active_view()

    def _on_prepare_session(self) -> None:
        self.state.draft.z_start_1based = self.spin_z_start.value()
        self.state.draft.z_stop_1based = self.spin_z_stop.value()
        self.state.draft.projection_method = self.combo_proj.currentText()  # type: ignore[assignment]
        try:
            self.source_ctrl.prepare_and_load_session()
            volume = self.state.session_3d.prepared_volume if self.state.session_3d else None
            self.state.current_z_index = 0
            self.viewer_controls.set_z_range(volume.nuclear_volume.shape[0] if volume else 1)
            self.viewer_controls.z_slider.setValue(0)
            self._render_active_view()
            self.status_bar.showMessage("Session prepared successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Preparation Failed", str(e))

    def _on_process_folder(self):
        if self._quick_worker is not None: return
        try:
            self.channel_table.validate_mapping()
            self.seg_form._sync_to_draft()
            self.state.draft.preprocessing_values = self.preprocessing.values()
            self.state.draft.z_start_1based = self.spin_z_start.value()
            self.state.draft.z_stop_1based = self.spin_z_stop.value()
            self.state.draft.projection_method = self.combo_proj.currentText()
            if self.combo_task.currentData() == "positivity" and not self.state.draft.threshold_drafts:
                raise ValueError("Analyze one image and set marker cutoffs before processing positivity for a folder.")
            source = QFileDialog.getExistingDirectory(self, 'Choose images to process')
            if not source: return
            output = QFileDialog.getExistingDirectory(self, 'Choose a new output folder')
            if not output: return
            if any(Path(output).iterdir()):
                raise ValueError('Choose an empty output folder to keep exports separate.')
            worker = FunctionWorker(export_folder_results, source, output, deepcopy(self.state.draft), self.combo_task.currentData())
            self._quick_worker = worker
            self._set_quick_busy(True)
            worker.signals.progress.connect(self.status_bar.showMessage)
            frozen_draft, frozen_task = deepcopy(self.state.draft), self.combo_task.currentData()
            worker.func = lambda: export_folder_results(source, output, frozen_draft, frozen_task, progress=worker.signals.progress.emit)
            worker.args = ()
            def finished(report):
                self._quick_worker = None
                self._set_quick_busy(False)
                failures = [row for row in report if row['status'] == 'failed']
                self.status_bar.showMessage(f'Folder finished: {len(report)-len(failures)} completed, {len(failures)} failed. See folder-report.csv in {output}.')
            worker.signals.finished.connect(finished)
            worker.signals.error.connect(self._on_quick_error)
            self.threadpool.start(worker)
        except Exception as exc:
            QMessageBox.warning(self, 'Folder processing', str(exc))

    def _show_quick_review(self):
        self.qc_widget.refresh_from_session()
        self.inspector_stack.setCurrentIndex(4)

    def _on_analyze_image(self) -> None:
        if self._quick_worker is not None:
            return
        try:
            self.channel_table.validate_mapping()
            self.seg_form._sync_to_draft()
            self.state.draft.preprocessing_values = self.preprocessing.values()
            self.state.draft.z_start_1based = self.spin_z_start.value()
            self.state.draft.z_stop_1based = self.spin_z_stop.value()
            self.state.draft.projection_method = self.combo_proj.currentText()
            raw = self.state.loaded_raw_array
            if raw is None:
                raise ValueError('Open an image first.')
            draft = deepcopy(self.state.draft)
            task = self.combo_task.currentData()
            metadata = self.state.source_metadata
            identity = self.state.source_id
            worker = FunctionWorker(analyze_image, raw, metadata, draft, identity, task)
            self._quick_worker = worker
            self._set_quick_busy(True)
            worker.signals.finished.connect(self._on_quick_finished)
            worker.signals.error.connect(self._on_quick_error)
            self.threadpool.start(worker)
        except Exception as exc:
            QMessageBox.warning(self, 'Check image setup', str(exc))

    def _set_quick_busy(self, busy):
        self.nav_list.setEnabled(not busy)
        self.inspector_stack.setEnabled(not busy)
        self.menuBar().setEnabled(not busy)
        self.status_bar.showMessage('Analyzing image…' if busy else 'Ready.')

    def _on_quick_finished(self, result):
        self._quick_worker = None
        self.state.quick_result = result
        self.state.session_2d = None
        self.state.session_3d = None
        self.state.results_stale = False
        self.state.configuration_differs_from_run = False
        self._set_quick_busy(False)
        self.state.current_z_index = 0
        self.viewer_controls.set_z_range(result.planes[0].shape[0] if result.mode == 'volume_3d' else 1)
        self._rebuild_threshold_cards()
        self.metric_builder.refresh_markers()
        self.state.notify()
        self.nav_list.setCurrentRow(3)
        self.inspector_stack.setCurrentIndex(9)
        self._render_active_view()
        self.canvas.fit_image_to_view()
        self.status_bar.showMessage(f'Analysis complete: {len(result.objects)} objects. Review is optional; results can be exported now.')

    def _on_quick_error(self, error):
        self._quick_worker = None
        self._set_quick_busy(False)
        self.state.results_stale = True
        self.state.notify()
        path = self._opening_path
        self._opening_path = None
        if path and ('ambiguous' in str(error[0]).lower() or 'declare' in str(error[0]).lower()):
            choices = {'Pages are Z slices': 'ZYX', 'Pages are channels': 'CYX', 'Z slices, then channels': 'ZCYX', 'Channels, then Z slices': 'CZYX'}
            choice, confirmed = QInputDialog.getItem(self, 'Image dimensions', 'The image does not describe its dimensions. How are its pages arranged?', list(choices), 0, False)
            if confirmed:
                self.open_image(path, choices[choice])
                return
        QMessageBox.warning(self, 'Analysis failed', str(error[0]))

    def _on_run_segmentation(self) -> None:
        try:
            self.analysis_ctrl.run_segmentation()
            self.qc_widget.refresh_from_session()
            self._render_active_view()
            self.status_bar.showMessage("Segmentation completed. Review candidates in QC tab.")
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Failed", str(e))

    def _on_qc_accept(self) -> None:
        sel = self.qc_widget.table.selectedItems()
        if not sel:
            return
        r = sel[0].row()
        oid_str = self.qc_widget.table.item(r, 0).text()
        try:
            if self.state.draft.mode == "volume_3d":
                self.analysis_ctrl.accept_3d(int(oid_str))
            else:
                self.analysis_ctrl.accept_2d(oid_str)
            self.qc_widget.refresh_from_session()
            self._render_active_view()
        except Exception as e:
            QMessageBox.warning(self, "QC Error", str(e))

    def _on_qc_reject(self) -> None:
        if self.state.quick_result is not None:
            self._quick_review_action('reject')
            return
        sel = self.qc_widget.table.selectedItems()
        if not sel:
            return
        r = sel[0].row()
        oid_str = self.qc_widget.table.item(r, 0).text()
        try:
            if self.state.draft.mode == "volume_3d":
                self.analysis_ctrl.reject_3d(int(oid_str))
            else:
                self.analysis_ctrl.reject_2d(oid_str)
            self.qc_widget.refresh_from_session()
            self._render_active_view()
        except Exception as e:
            QMessageBox.warning(self, "QC Error", str(e))

    def _on_qc_restore(self) -> None:
        if self.state.quick_result is not None:
            self._quick_review_action('restore')
            return
        sel = self.qc_widget.table.selectedItems()
        if not sel:
            return
        r = sel[0].row()
        oid_str = self.qc_widget.table.item(r, 0).text()
        try:
            if self.state.draft.mode == "volume_3d":
                self.analysis_ctrl.restore_3d(int(oid_str))
            else:
                self.analysis_ctrl.restore_2d(oid_str)
            self.qc_widget.refresh_from_session()
            self._render_active_view()
        except Exception as e:
            QMessageBox.warning(self, "QC Error", str(e))

    def _on_qc_auto(self) -> None:
        try:
            self.analysis_ctrl.auto_accept()
            self.qc_widget.refresh_from_session()
            self._render_active_view()
        except Exception as exc:
            QMessageBox.warning(self, "QC Error", str(exc))

    def _quick_review_action(self, action):
        items = self.qc_widget.table.selectedItems()
        if not items: return
        result = self.state.quick_result
        identifier = self.qc_widget.table.item(items[0].row(), 0).text()
        try:
            getattr(result.review, action)(int(identifier) if result.mode == 'volume_3d' else identifier)
            result.manually_reviewed = True
            result.update_measurements()
            self.qc_widget.refresh_from_session()
            self.state.notify()
            self._render_active_view()
        except Exception as exc:
            QMessageBox.warning(self, 'Review failed', str(exc))

    def _on_qc_finalize(self) -> None:
        if self.state.quick_result is not None:
            self.state.quick_result.update_measurements()
            self.state.notify()
            self.nav_list.setCurrentRow(3)
            self.inspector_stack.setCurrentIndex(9)
            self._render_active_view()
            return
        try:
            self.analysis_ctrl.finalize_review()
            self.analysis_ctrl.measure_markers()
            self._rebuild_threshold_cards()
            self.metric_builder.refresh_markers()
            QMessageBox.information(self, "Review Finalized", "QC Review finalized and raw marker signals measured.")
        except Exception as e:
            QMessageBox.critical(self, "Finalize Failed", str(e))

    def _on_object_selected(self, z: int, y: int, x: int) -> None:
        if self.state.draft.mode == "volume_3d":
            self.viewer_controls.z_slider.setValue(z)

    def _rebuild_threshold_cards(self) -> None:
        # Clear existing cards
        for i in reversed(range(self.cards_vbox.count())):
            w = self.cards_vbox.itemAt(i).widget()
            if w:
                self.cards_vbox.removeWidget(w)
                w.deleteLater()

        markers = list(self.state.draft.marker_channels.keys())
        for m in markers:
            if self.state.quick_result is not None and m not in self.state.draft.threshold_drafts:
                measurement = self.state.quick_result.measurement
                if measurement is not None:
                    values = measurement.measurements.loc[measurement.measurements.marker_name == m, 'mean_intensity']
                    cutoff = float(values.median()) if len(values) else 0.0
                    self.state.draft.threshold_drafts[m] = MarkerThresholdDraft(m, cutoff=cutoff)
            card = MarkerThresholdCard(m, self.state)
            card.preview_requested.connect(lambda m_name=m, c=card: self._on_preview_threshold(m_name, c))
            card.apply_requested.connect(self._on_apply_thresholds)
            self.cards_vbox.addWidget(card)

    def _on_preview_threshold(self, marker_name: str, card: MarkerThresholdCard) -> None:
        try:
            draft = self.state.draft.threshold_drafts[marker_name]
            if self.state.quick_result is not None:
                result = self.state.quick_result
                from nervenuc.classification import classify_markers, classify_markers_3d
                res = (classify_markers_3d if result.mode == 'volume_3d' else classify_markers)(result.measurement, [draft.to_core_threshold()])
            else:
                res = self.analysis_ctrl.preview_thresholds([draft])
            df = res.classifications
            m_rows = df[df["marker_name"] == marker_name]
            pos = int(np.sum(m_rows["is_positive"]))
            neg = int(len(m_rows) - pos)
            card.update_preview_result(pos, neg)
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", str(e))

    def _on_apply_thresholds(self) -> None:
        if self.state.quick_result is not None:
            try:
                result = self.state.quick_result
                if result.task != 'positivity':
                    raise ValueError('Choose marker positivity in Set up and run Analyze first.')
                if result.settings.get("preprocessing_requested", {}) != self.state.draft.preprocessing_values or any(result.settings.get(key) != getattr(self.state.draft, key) for key in ("z_start_1based", "z_stop_1based", "projection_method", "time_index")) or result.mode != self.state.draft.mode or result.task != self.combo_task.currentData() or result.settings['nuclear_channel'] != self.state.draft.nuclear_channel or result.settings['marker_channels'] != self.state.draft.marker_channels or any(result.settings['segmentation'].get(key) != value for key, value in self.state.draft.segmentation_values.items() if key not in ("gaussian_sigma", "gaussian_sigma_xy_px", "gaussian_sigma_z_slices")):
                    raise ValueError('Detection settings changed. Run Analyze again before applying cutoffs.')
                result.apply_cutoffs([self.state.draft.threshold_drafts[name].to_core_threshold() for name in result.settings['marker_channels']])
                if self.state.draft.metric_drafts:
                    from nervenuc.summary import summarize_metrics, summarize_metrics_3d
                    result.summary = (summarize_metrics_3d if result.mode == 'volume_3d' else summarize_metrics)(result.classification, [draft.to_core_metric() for draft in self.state.draft.metric_drafts])
                    result.settings['custom_metrics'] = [dict(name=d.name, denominator_positive=d.denominator_positive, denominator_negative=d.denominator_negative, numerator_positive=d.numerator_positive, numerator_negative=d.numerator_negative) for d in self.state.draft.metric_drafts]
                self.state.results_stale = False
                self.state.configuration_differs_from_run = False
                self.state.notify()
                self.nav_list.setCurrentRow(3)
                self.inspector_stack.setCurrentIndex(9)
            except Exception as exc:
                QMessageBox.warning(self, 'Cutoff error', str(exc))
            return
        try:
            self.analysis_ctrl.apply_thresholds_and_summarize()
            self.results_dashboard.refresh_results()
            QMessageBox.information(self, "Applied", "Thresholds applied and biological metrics summarized.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Failed", str(e))

    def _on_save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset JSON", "preset.json", "JSON Files (*.json)")
        if path:
            try:
                self.batch_ctrl.save_current_preset(path, Path(path).stem)
                QMessageBox.information(self, "Preset Saved", f"Saved preset to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Preset Failed", str(e))

    def _on_load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset JSON", "", "JSON Files (*.json)")
        if path:
            try:
                p = self.batch_ctrl.load_preset_file(path)
                self.state.session_2d = None
                self.state.session_3d = None
                self.combo_mode.blockSignals(True)
                self.combo_mode.setCurrentIndex(
                    0 if p.schema_version == 1 else (1 if p.schema_version == 2 else 2)
                )
                self.combo_mode.blockSignals(False)
                self.seg_form.load_from_draft()
                self.spin_z_start.setValue(self.state.draft.z_start_1based or 1)
                self.spin_z_stop.setValue(self.state.draft.z_stop_1based or 1)
                self.combo_proj.setCurrentText(self.state.draft.projection_method)
                self._rebuild_threshold_cards()
                self.metric_builder.refresh_markers()
                self.status_bar.showMessage(f"Loaded Preset: {p.name} (Schema {p.schema_version})")
            except Exception as e:
                QMessageBox.critical(self, "Load Preset Failed", str(e))

    def _render_active_view(self) -> None:
        """Compose current active slice and overlay into QImage and update canvas."""
        mode = self.state.draft.mode
        ch_planes: dict[int, np.ndarray] = {}
        labels_plane: np.ndarray | None = None

        z_idx = self.state.current_z_index

        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess and sess.nuclear_channel is not None:
                ch_planes[sess.source_provenance.nuclear_channel if sess.source_provenance else 0] = sess.nuclear_channel
                for m_idx, (m_name, m_img) in enumerate(sess.marker_images.items(), start=1):
                    ch_planes[sess.source_provenance.marker_channels.get(m_name, m_idx) if sess.source_provenance else m_idx] = m_img
                if sess.review_session:
                    reviewed = sess.review_session.result()
                    labels_plane = reviewed.labels
                elif sess.segmentation_result:
                    labels_plane = sess.segmentation_result.labels
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d and sess_3d.prepared_volume:
                nuc_vol = sess_3d.prepared_volume.nuclear_volume
                if z_idx < nuc_vol.shape[0]:
                    ch_planes[sess_3d.prepared_volume.provenance.nuclear_channel] = nuc_vol[z_idx]
                for m_idx, (m_name, m_vol) in enumerate(sess_3d.prepared_volume.marker_volumes.items(), start=1):
                    if z_idx < m_vol.shape[0]:
                        ch_planes[sess_3d.prepared_volume.provenance.marker_channels[m_name]] = m_vol[z_idx]
                if sess_3d.segmentation_result:
                    seg_labels = sess_3d.segmentation_result.labels
                    if z_idx < seg_labels.shape[0]:
                        labels_plane = seg_labels[z_idx]

        if self.state.quick_result is not None:
            result = self.state.quick_result
            nuclear, markers = result.planes
            mapping = result.settings['marker_channels']
            z_idx = min(z_idx, nuclear.shape[0]-1) if result.mode == 'volume_3d' else 0
            ch_planes = {result.settings['nuclear_channel']: nuclear[z_idx] if result.mode == 'volume_3d' else nuclear}
            for name, image in markers.items():
                ch_planes[mapping[name]] = image[z_idx] if result.mode == 'volume_3d' else image
            if self.state.display.show_preprocessed and result.detection_image is not None:
                ch_planes = {result.settings['nuclear_channel']: result.detection_image[z_idx] if result.mode == 'volume_3d' else result.detection_image}
            labels_plane = result.labels[z_idx] if result.mode == 'volume_3d' else result.labels
        if not ch_planes:
            ch_planes = self.source_ctrl.source_preview_planes()
        if not ch_planes:
            return

        disp = self.state.display
        base_qimg = render_display_slice(
            channels=ch_planes,
            min_vals=disp.channel_min,
            max_vals=disp.channel_max,
            gammas=disp.channel_gamma,
            visibilities=disp.channel_visible,
            colors=disp.channel_color,
        )
        self.canvas.set_base_image(base_qimg)

        if labels_plane is not None and disp.show_segmentation_overlay:
            overlay_qimg = render_label_overlay(
                labels_2d=labels_plane,
                opacity=disp.overlay_opacity,
            )
            self.canvas.set_overlay_image(overlay_qimg)
        else:
            self.canvas.set_overlay_image(render_label_overlay(np.zeros((1, 1), dtype=np.int32)))

    def _set_qc_actions_enabled(self, enabled: bool) -> None:
        """Enable or disable QC mutation controls during worker execution to prevent race conditions."""
        self.qc_widget.btn_accept.setEnabled(enabled)
        self.qc_widget.btn_reject.setEnabled(enabled)
        self.qc_widget.btn_restore.setEnabled(enabled)
        self.qc_widget.btn_auto.setEnabled(enabled)
        self.qc_widget.btn_finalize.setEnabled(enabled)
        self.act_fiji_import_rois.setEnabled(enabled and self._can_import_rois())
        self.act_fiji_replace_rois.setEnabled(enabled and self._can_replace_rois())

    def _can_import_rois(self) -> bool:
        sess = self.state.session_2d
        return (
            sess is not None
            and sess.review_session is not None
            and not self.fiji_ctrl.is_qc_worker_running
        )

    def _can_replace_rois(self) -> bool:
        sess = self.state.session_2d
        sel = self.qc_widget.table.selectedItems()
        return (
            sess is not None
            and sess.review_session is not None
            and len(sel) > 0
            and not self.fiji_ctrl.is_qc_worker_running
        )

    def _update_fiji_action_states(self) -> None:
        """Enable/disable Fiji menu items dynamically based on active workflow state."""
        # 1. Open Current Image
        has_2d_img = (
            self.state.loaded_path is not None
            or (self.state.session_2d is not None and self.state.session_2d.nuclear_channel is not None)
            or self.state.loaded_raw_array is not None
        )
        self.act_fiji_open_image.setEnabled(has_2d_img)

        # 2. Open Volume / Labels
        is_3d_mode = self.state.draft.mode == "volume_3d"
        has_3d_data = is_3d_mode and (
            self.state.loaded_path is not None
            or (self.state.session_3d is not None and self.state.session_3d.prepared_volume is not None)
            or (self.state.session_3d is not None and self.state.session_3d.segmentation_result is not None)
        )
        self.act_fiji_open_volume.setEnabled(has_3d_data)

        # 3. Export ROI Set (available during active 2D review session or finalized reviewed result)
        has_qc_export = (
            self.state.session_2d is not None
            and (
                self.state.session_2d.review_session is not None
                or self.state.session_2d.reviewed_result is not None
            )
        )
        self.act_fiji_export_rois.setEnabled(has_qc_export)

        # 4. Import ROIs as Manual
        self.act_fiji_import_rois.setEnabled(self._can_import_rois())

        # 5. Replace Selected Object
        self.act_fiji_replace_rois.setEnabled(self._can_replace_rois())

    def _on_fiji_open_image(self) -> None:
        if self.state.quick_result is not None:
            from nervenuc.fiji import FijiExchangeSession
            try:
                result = self.state.quick_result
                image = result.planes[0]
                exchange = FijiExchangeSession()
                self.fiji_ctrl._active_exchange_sessions.append(exchange)
                from nervenuc.fiji.launcher import FijiLauncher
                executable = self.fiji_ctrl.discover_executable()
                if executable is None:
                    raise ValueError('Configure Fiji under Tools before opening this image in Fiji.')
                self.fiji_ctrl._start_worker(lambda: exchange.open_in_fiji(image, axes='ZYX' if result.mode == 'volume_3d' else 'YX', launcher=FijiLauncher(executable)), on_error=lambda error: QMessageBox.warning(self, 'Open in Fiji', str(error[0])))
            except Exception as exc:
                QMessageBox.warning(self, 'Open in Fiji', str(exc))
            return
        if not self.fiji_ctrl.can_launch_external_fiji():
            QMessageBox.warning(
                self,
                "Fiji Not Configured",
                "Fiji was not found on your system. Please configure the Fiji installation path under Tools -> Fiji / ImageJ -> Configure Fiji...",
            )
            return

        def _on_started() -> None:
            self.status_bar.showMessage("Preparing image and launching external Fiji...")

        def _on_finished(proc: Any) -> None:
            self.status_bar.showMessage("Image opened in external Fiji.")

        def _on_error(err_tuple: tuple[Exception, str]) -> None:
            exc, tb = err_tuple
            title, msg = self.fiji_ctrl.translate_fiji_error(exc)
            QMessageBox.warning(self, title, msg)
            self.status_bar.showMessage("Failed to open image in Fiji.")

        try:
            self.fiji_ctrl.open_current_image(
                on_started=_on_started,
                on_finished=_on_finished,
                on_error=_on_error,
            )
        except Exception as e:
            title, msg = self.fiji_ctrl.translate_fiji_error(e)
            QMessageBox.warning(self, title, msg)

    def _on_fiji_open_volume(self) -> None:
        if not self.fiji_ctrl.can_launch_external_fiji():
            QMessageBox.warning(
                self,
                "Fiji Not Configured",
                "Fiji was not found on your system. Please configure the Fiji installation path under Tools -> Fiji / ImageJ -> Configure Fiji...",
            )
            return

        options: list[tuple[str, str]] = []
        if self.state.loaded_path is not None:
            options.append(("Raw Z-Stack", "raw_stack"))
        if self.state.session_3d is not None and self.state.session_3d.prepared_volume is not None:
            options.append(("Prepared Analysis Volume", "prepared_volume"))
        if self.state.session_3d is not None and self.state.session_3d.segmentation_result is not None:
            options.append(("Segmentation Label Volume (Inspection Only)", "segmentation_labels"))

        if not options:
            QMessageBox.information(self, "No 3D Data", "No 3D volume or stack is currently available to open.")
            return

        labels = [opt[0] for opt in options]
        chosen_label, ok = QInputDialog.getItem(
            self,
            "Open 3D Volume in Fiji",
            "Select 3D volume representation to open:",
            labels,
            0,
            False,
        )
        if not ok or not chosen_label:
            return

        kind = next(opt[1] for opt in options if opt[0] == chosen_label)

        if kind == "segmentation_labels":
            QMessageBox.information(
                self,
                "Display-Only Inspection Note",
                "Label data opened in Fiji is for inspection only. Editing the Fiji label image does not modify NerveNuc 3D QC.",
            )

        def _on_started() -> None:
            self.status_bar.showMessage("Preparing 3D volume and launching external Fiji...")

        def _on_finished(proc: Any) -> None:
            self.status_bar.showMessage("3D Volume opened in external Fiji.")

        def _on_error(err_tuple: tuple[Exception, str]) -> None:
            exc, tb = err_tuple
            title, msg = self.fiji_ctrl.translate_fiji_error(exc)
            QMessageBox.warning(self, title, msg)
            self.status_bar.showMessage("Failed to open volume in Fiji.")

        try:
            self.fiji_ctrl.open_current_volume(
                kind=cast(Any, kind),
                on_started=_on_started,
                on_finished=_on_finished,
                on_error=_on_error,
            )
        except Exception as e:
            title, msg = self.fiji_ctrl.translate_fiji_error(e)
            QMessageBox.warning(self, title, msg)

    def _on_fiji_export_rois(self) -> None:
        if not self.fiji_ctrl.can_exchange_rois():
            QMessageBox.warning(
                self,
                "ROI Export Unavailable",
                "ROI export requires the 'roifile' package.\nInstall with: pip install 'nervenuc[fiji]'",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Active QC Objects as Fiji ROI Set",
            "qc_rois.zip",
            "ImageJ ROI Set (*.zip);;All Files (*)",
        )
        if not path:
            return

        p = Path(path)
        if p.exists():
            ret = QMessageBox.question(
                self,
                "Confirm Overwrite",
                f"File '{p.name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        def _on_started() -> None:
            self.status_bar.showMessage("Exporting QC objects to Fiji ROI set...")

        def _on_finished(out_path: Path) -> None:
            self.status_bar.showMessage("ROI set exported successfully.")
            QMessageBox.information(
                self,
                "ROI Export Successful",
                f"Active QC objects exported as Fiji ROI set to:\n{out_path}",
            )

        def _on_error(err_tuple: tuple[Exception, str]) -> None:
            exc, tb = err_tuple
            title, msg = self.fiji_ctrl.translate_fiji_error(exc)
            QMessageBox.critical(self, title, msg)
            self.status_bar.showMessage("ROI export failed.")

        try:
            self.fiji_ctrl.export_rois(
                output_path=p,
                overwrite=True,
                on_started=_on_started,
                on_finished=_on_finished,
                on_error=_on_error,
            )
        except Exception as e:
            title, msg = self.fiji_ctrl.translate_fiji_error(e)
            QMessageBox.critical(self, title, msg)

    def _on_fiji_import_rois(self) -> None:
        if not self.fiji_ctrl.can_exchange_rois():
            QMessageBox.warning(
                self,
                "ROI Import Unavailable",
                "ROI import requires the 'roifile' package.\nInstall with: pip install 'nervenuc[fiji]'",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Fiji ROI Set as Manual QC Objects",
            "",
            "ImageJ ROIs (*.zip *.roi);;All Files (*)",
        )
        if not path:
            return

        def _on_started() -> None:
            self._set_qc_actions_enabled(False)
            self.status_bar.showMessage("Validating and importing Fiji ROIs as manual objects...")

        def _on_finished(result: Any) -> None:
            self._set_qc_actions_enabled(True)
            self.qc_widget.refresh_from_session()
            self._render_active_view()
            self._update_fiji_action_states()
            ids_str = ", ".join(r.review_id for r in result.imported)
            self.status_bar.showMessage(f"Imported {len(result.imported)} manual objects from Fiji.")
            QMessageBox.information(
                self,
                "ROIs Imported Successfully",
                f"Imported {len(result.imported)} manual objects from Fiji ROI set:\n"
                f"Assigned Review IDs: {ids_str}\n\n"
                f"Note: Downstream marker measurements and classifications have been reset.",
            )

        def _on_error(err_tuple: tuple[Exception, str]) -> None:
            self._set_qc_actions_enabled(True)
            exc, tb = err_tuple
            title, msg = self.fiji_ctrl.translate_fiji_error(exc)
            QMessageBox.critical(self, title, msg)
            self.status_bar.showMessage("ROI import failed.")

        try:
            self.fiji_ctrl.import_rois(
                roi_path=path,
                on_started=_on_started,
                on_finished=_on_finished,
                on_error=_on_error,
            )
        except Exception as e:
            self._set_qc_actions_enabled(True)
            title, msg = self.fiji_ctrl.translate_fiji_error(e)
            QMessageBox.critical(self, title, msg)

    def _on_fiji_replace_rois(self) -> None:
        if not self.fiji_ctrl.can_exchange_rois():
            QMessageBox.warning(
                self,
                "ROI Split Unavailable",
                "ROI split replacement requires the 'roifile' package.\nInstall with: pip install 'nervenuc[fiji]'",
            )
            return

        sel = self.qc_widget.table.selectedItems()
        if not sel:
            QMessageBox.information(self, "No Selection", "Please select a target nucleus in the QC table first.")
            return

        r = sel[0].row()
        target_id = self.qc_widget.table.item(r, 0).text()

        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Fiji ROIs to Replace Object {target_id}",
            "",
            "ImageJ ROIs (*.zip *.roi);;All Files (*)",
        )
        if not path:
            return

        ret = QMessageBox.question(
            self,
            "Confirm Split Replacement",
            f"Replace nucleus '{target_id}' with ROIs from '{Path(path).name}'?\n\n"
            f"Target nucleus '{target_id}' will be retired with reason 'split' and new EDIT-* child nuclei will be created.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        def _on_started() -> None:
            self._set_qc_actions_enabled(False)
            self.status_bar.showMessage(f"Replacing nucleus {target_id} with Fiji ROIs...")

        def _on_finished(result: Any) -> None:
            self._set_qc_actions_enabled(True)
            self.qc_widget.refresh_from_session()
            self._render_active_view()
            self._update_fiji_action_states()
            child_ids_str = ", ".join(result.child_review_ids)
            self.status_bar.showMessage(f"Replaced nucleus {target_id} with {len(result.child_review_ids)} children.")
            QMessageBox.information(
                self,
                "Object Split Replacement Successful",
                f"Nucleus '{target_id}' was retired (split).\n"
                f"Created {len(result.child_review_ids)} child nuclei: {child_ids_str}\n\n"
                f"Note: Downstream marker measurements and classifications have been reset.",
            )

        def _on_error(err_tuple: tuple[Exception, str]) -> None:
            self._set_qc_actions_enabled(True)
            exc, tb = err_tuple
            title, msg = self.fiji_ctrl.translate_fiji_error(exc)
            QMessageBox.critical(self, title, msg)
            self.status_bar.showMessage("Split replacement failed.")

        try:
            self.fiji_ctrl.replace_rois(
                target_review_id=target_id,
                roi_path=path,
                note="Fiji GUI Split Replacement",
                on_started=_on_started,
                on_finished=_on_finished,
                on_error=_on_error,
            )
        except Exception as e:
            self._set_qc_actions_enabled(True)
            title, msg = self.fiji_ctrl.translate_fiji_error(e)
            QMessageBox.critical(self, title, msg)

    def _on_fiji_config(self) -> None:
        dlg = FijiConfigDialog(self.fiji_ctrl, self)
        dlg.exec()
        self._update_fiji_action_states()

    def _on_state_updated(self) -> None:
        """Handle state change notification."""
        self._update_fiji_action_states()
        result = self.state.quick_result
        self.viewer_controls.cb_preprocessed.setEnabled(result is not None and result.detection_image is not None)
        if self.state.quick_result is not None:
            self.act_fiji_open_image.setEnabled(True)

    def closeEvent(self, event: Any) -> None:
        """Clean up completed Fiji exchange sessions on application close without killing running Fiji."""
        if self._quick_worker is not None:
            event.ignore()
            self.status_bar.showMessage("Wait for analysis to finish before closing.")
            return
        try:
            self.fiji_ctrl.cleanup_sessions(force=False)
        except Exception:
            pass
        super().closeEvent(event)
