"""Quick image analysis using the production preparation and measurement engines."""
from dataclasses import dataclass, asdict, replace
from copy import deepcopy, copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
from nervenuc.preparation import prepare_direct_image, prepare_stack, prepare_volume, DirectImageSelectionConfig
from nervenuc.segmentation import segment_nuclei, SegmentationConfig
from nervenuc.segmentation3d import segment_nuclei_3d, Segmentation3DConfig
from nervenuc.qc import start_review
from nervenuc.qc3d import start_review_3d
from nervenuc.measurements import measure_markers
from nervenuc.measurements3d import measure_markers_3d
from nervenuc.classification import classify_markers, classify_markers_3d
from .preprocessing import PreprocessingConfig, preprocess_image

def accept_unreviewed(review, mode):
    if mode == 'volume_3d':
        review.auto_accept()
    else:
        for identifier, record in list(review._active_objects.items()):
            if record.review_status == 'unreviewed':
                review.accept(identifier, note="Automatically accepted by quick analysis")

def review_snapshot(review, mode):
    # Finalization freezes a 3D review. Finalize a lightweight copy so the live
    # quick-review session remains editable while measurements use snapshots.
    return copy(review).finalize() if mode == 'volume_3d' else review.finalize()

@dataclass
class QuickResult:
    prepared: object
    review: object
    reviewed: object
    measurement: object | None
    classification: object | None
    settings: dict
    source_id: str
    mode: str
    task: str
    manually_reviewed: bool = False
    summary: object | None = None
    detection_image: np.ndarray | None = None

    @property
    def objects(self):
        return self.reviewed.objects

    @property
    def planes(self):
        p = self.prepared
        return (p.nuclear_volume, p.marker_volumes) if self.mode == 'volume_3d' else (p.nuclear_image, p.marker_images)

    @property
    def labels(self):
        labels = self.reviewed.labels
        if self.mode == 'volume_3d':
            labels = np.where(np.isin(labels, self.objects['current_label']), labels, 0)
        return labels

    def update_measurements(self):
        accept_unreviewed(self.review, self.mode)
        self.reviewed = review_snapshot(self.review, self.mode)
        _, markers = self.planes
        self.measurement = None
        if self.task != 'count' and markers:
            self.measurement = measure_markers_3d(self.reviewed, markers, self.prepared.provenance) if self.mode == 'volume_3d' else measure_markers(self.reviewed, markers)
        self.classification = None
        self.summary = None

    def apply_cutoffs(self, thresholds):
        if self.measurement is None:
            raise ValueError('Run signal measurement before applying cutoffs.')
        self.classification = classify_markers_3d(self.measurement, thresholds) if self.mode == 'volume_3d' else classify_markers(self.measurement, thresholds)
        self.settings['thresholds'] = [dict(marker_name=t.marker_name, cutoff=t.threshold, measurement=t.measurement, comparison=t.comparison) for t in thresholds]


def analyze_image(raw, metadata, draft, source_id, task):
    draft = deepcopy(draft)
    if task == 'count':
        draft.marker_channels = {}
    if task != 'count' and not draft.marker_channels:
        raise ValueError('Assign at least one signal channel, or choose Count objects.')
    preprocessing = PreprocessingConfig(**draft.preprocessing_values) if draft.preprocessing_values else None
    detection_image = None
    if draft.mode == 'volume_3d':
        selection = draft.build_core_volume_selection_config()
        prepared = prepare_volume(raw, selection, metadata=metadata)
        config = Segmentation3DConfig.from_dict(draft.segmentation_values) if draft.segmentation_values else Segmentation3DConfig()
        detection_volume = prepared
        if preprocessing is not None:
            detection_image = preprocess_image(prepared.nuclear_volume, preprocessing)
            detection_volume = replace(prepared, nuclear_volume=detection_image)
            config = replace(config, gaussian_sigma_xy_px=0, gaussian_sigma_z_slices=0)
        segmentation = segment_nuclei_3d(detection_volume, config, source_id=source_id)
        review = start_review_3d(segmentation)
    else:
        selection = draft.build_core_stack_preparation_config() if draft.mode == 'projected_2d' else DirectImageSelectionConfig(axes=draft.declared_axes, nuclear_channel=draft.nuclear_channel, marker_channels=dict(draft.marker_channels))
        if draft.mode == 'direct_2d' and draft.declared_axes == 'YX':
            if task != 'count' or draft.nuclear_channel != 0:
                raise ValueError('A single-channel image supports counting. Signal measurement needs a separate channel.')
            prepared = SimpleNamespace(nuclear_image=raw, marker_images={}, provenance=None)
        else:
            prepared = prepare_stack(raw, selection, metadata=metadata) if draft.mode == 'projected_2d' else prepare_direct_image(raw, selection, metadata=metadata)
        config = SegmentationConfig(**draft.segmentation_values) if draft.segmentation_values else SegmentationConfig()
        detection_input = prepared.nuclear_image
        if preprocessing is not None:
            detection_image = preprocess_image(detection_input, preprocessing)
            detection_input = detection_image
            config = replace(config, gaussian_sigma=0)
        segmentation = segment_nuclei(detection_input, config)
        review = start_review(segmentation)
    accept_unreviewed(review, draft.mode)
    result = QuickResult(prepared, review, review_snapshot(review, draft.mode), None, None, {'mode': draft.mode, 'task': task, 'axes': draft.declared_axes, 'source_shape': list(raw.shape), 'time_index': draft.time_index, 'nuclear_channel': draft.nuclear_channel, 'marker_channels': dict(draft.marker_channels), 'z_start_1based': draft.z_start_1based, 'z_stop_1based': draft.z_stop_1based, 'projection_method': draft.projection_method, 'segmentation': asdict(config)}, source_id, draft.mode, task)
    result.detection_image = detection_image
    result.settings['preprocessing'] = preprocessing.to_dict() if preprocessing is not None else None
    result.settings['preprocessing_requested'] = dict(draft.preprocessing_values)
    result.settings['measurement_image_basis'] = 'raw'
    result.settings['calibration'] = asdict(prepared.spacing) if draft.mode == 'volume_3d' and prepared.spacing is not None else {key: getattr(metadata, key, None) for key in ('pixel_size_x_um', 'pixel_size_y_um', 'physical_size_z_um')}
    result.update_measurements()
    return result


def export_folder_results(input_directory, output_directory, draft, task, progress=None):
    """Process independent images with shared settings, without study metadata."""
    from copy import deepcopy
    from pathlib import Path
    import json
    import tifffile
    from .state import GuiSessionState
    from .controllers.source_controller import SourceController
    from nervenuc.export import write_dataframe_csv
    source = Path(input_directory).resolve()
    output = Path(output_directory).resolve()
    if source == output:
        raise ValueError('Choose a different output folder.')
    if output.exists():
        if not output.is_dir():
            raise ValueError('The output path must be a folder.')
        if any(output.iterdir()):
            raise ValueError('Choose an empty output folder; existing results will not be overwritten.')
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))
    if not files:
        raise ValueError('No supported images were found in that folder.')
    report = []
    for number, path in enumerate(files, 1):
        if progress: progress(f"Processing {number}/{len(files)}: {path.name}")
        target = output / f'{number:04d}-{path.stem}'
        try:
            if target.exists():
                raise ValueError('Output already exists; choose a new output folder.')
            state = GuiSessionState()
            controller = SourceController(state)
            controller.inspect_file(path)
            controller.load_source_preview()
            settings = deepcopy(draft)
            settings.declared_axes = state.preview_axes
            if settings.mode != 'direct_2d' and not state.inspection.size_z:
                raise ValueError('These stack settings need an image with Z slices.')
            if settings.mode == 'direct_2d' and (state.inspection.size_z or 1) > 1:
                raise ValueError('Choose stack analysis for this image.')
            result = analyze_image(state.loaded_raw_array, state.source_metadata, settings, state.source_id, task)
            if task == 'positivity':
                if not settings.threshold_drafts:
                    raise ValueError('Set explicit marker cutoffs before folder positivity analysis.')
                result.apply_cutoffs([settings.threshold_drafts[name].to_core_threshold() for name in settings.marker_channels])
            target.mkdir(parents=True)
            write_dataframe_csv(result.objects, target/'objects.csv')
            if result.measurement is not None:
                write_dataframe_csv(result.measurement.measurements, target/'measurements.csv')
            if result.classification is not None:
                write_dataframe_csv(result.classification.classifications, target/'classification.csv')
            if result.detection_image is not None:
                tifffile.imwrite(target/'detection-input.tif', result.detection_image, metadata={'axes': 'ZYX' if result.mode == 'volume_3d' else 'YX'}, photometric='minisblack')
            tifffile.imwrite(target/'labels.tif', result.labels, metadata={'axes': 'ZYX' if result.mode == 'volume_3d' else 'YX'}, photometric='minisblack')
            (target/'analysis.json').write_text(json.dumps(dict(result.settings, source_id=result.source_id, object_count=len(result.objects), review_policy='auto_accept', positivity_cutoffs_applied=result.classification is not None), indent=2), encoding='utf-8')
            report.append(dict(file=path.name, status='completed', objects=len(result.objects), error=''))
        except Exception as exc:
            report.append(dict(file=path.name, status='failed', objects=None, error=str(exc)))
    output.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv(pd.DataFrame(report), output/'folder-report.csv')
    return report
