"""Lean scientific reproducibility validation engine for NerveNuc."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
import skimage

import nervenuc
from nervenuc.classification.engine import classify_markers
from nervenuc.measurements.engine import measure_markers
from nervenuc.qc.session import start_review
from nervenuc.segmentation.config import SegmentationConfig
from nervenuc.segmentation.core import segment_nuclei
from nervenuc.segmentation3d.engine import segment_nuclei_3d
from nervenuc.preparation import prepare_volume
from nervenuc.qc3d import start_review_3d
from nervenuc.measurements3d import measure_markers_3d
from nervenuc.classification import classify_markers_3d
from nervenuc.summary import summarize_metrics_3d
from nervenuc.summary.engine import summarize_metrics
from nervenuc.workflow.config import AnalysisConfig, VolumeAnalysisConfig
from nervenuc.workflow.fingerprint import (
    compute_analysis_config_fingerprint,
    compute_volume_analysis_config_fingerprint,
)

from .matching import match_instances
from .metrics import compute_segmentation_metrics
from .models import (
    ParameterStabilityResult,
    ReproducibilityRunResult,
    ValidationCase,
)


def capture_runtime_environment() -> dict[str, str]:
    """Capture runtime platform and core scientific library versions."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "skimage_version": skimage.__version__,
        "pandas_version": pd.__version__,
        "nervenuc_version": getattr(nervenuc, "__version__", "unknown"),
    }


def evaluate_same_process_reproducibility(
    case: ValidationCase,
    *,
    n_runs: int = 3,
) -> ReproducibilityRunResult:
    """Execute identical case and configuration repeatedly in the same process to verify determinism.

    Args:
        case: Immutable ValidationCase with image data and AnalysisConfig.
        n_runs: Number of repeated executions (must be >= 2).

    Returns:
        ReproducibilityRunResult detailing exact matching across all outputs.
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2, got {n_runs}.")

    env = capture_runtime_environment()
    diagnostics: list[str] = []

    # Run list storage
    run_labels: list[np.ndarray] = []
    run_counts: list[int] = []
    run_meas_dfs: list[pd.DataFrame | None] = []
    run_class_dfs: list[pd.DataFrame | None] = []
    run_pheno_dfs: list[pd.DataFrame | None] = []
    run_summary_dfs: list[pd.DataFrame | None] = []
    run_fingerprints: list[str] = []

    is_3d = case.spec.dimensions == "3D"
    is_multi_channel = len(case.spec.channel_names) > 1

    for run_idx in range(n_runs):
        if is_3d:
            prepared = prepare_volume(case.image_data, case.config.selection)
            seg_res_3d = segment_nuclei_3d(prepared, case.config.segmentation)
            labels = seg_res_3d.labels
            count = seg_res_3d.nucleus_count
            review = start_review_3d(seg_res_3d)
            review.auto_accept()
            reviewed = review.finalize()
            meas_res_3d = measure_markers_3d(
                reviewed,
                prepared.marker_volumes,
                prepared.provenance,
                case.config.markers,
            )
            meas_df = meas_res_3d.measurements
            class_res_3d = classify_markers_3d(meas_res_3d, case.config.thresholds)
            class_df = class_res_3d.classifications
            pheno_df = class_res_3d.phenotypes
            summary_df = summarize_metrics_3d(class_res_3d, case.config.metrics).metrics
            fp = compute_volume_analysis_config_fingerprint(case.config)
        else:
            dapi = case.image_data[0]
            seg_res = segment_nuclei(dapi, case.config.segmentation)
            labels = seg_res.labels
            count = len(seg_res.objects)
            fp = compute_analysis_config_fingerprint(case.config)

            if is_multi_channel:
                rs = start_review(seg_res)
                for r_id in list(rs._active_objects):
                    rs.accept(r_id)
                reviewed = rs.finalize()

                marker_images = {
                    name: case.image_data[idx]
                    for idx, name in enumerate(case.spec.channel_names)
                    if name != "DAPI"
                }
                meas_res = measure_markers(reviewed, marker_images)
                meas_df = meas_res.measurements

                class_res = classify_markers(meas_res, case.config.thresholds)
                class_df = class_res.classifications
                pheno_df = class_res.phenotypes

                summary_res = summarize_metrics(class_res, case.config.metrics)
                summary_df = summary_res.metrics
            else:
                meas_df = None
                class_df = None
                pheno_df = None
                summary_df = None

        run_labels.append(labels)
        run_counts.append(count)
        run_meas_dfs.append(meas_df)
        run_class_dfs.append(class_df)
        run_pheno_dfs.append(pheno_df)
        run_summary_dfs.append(summary_df)
        run_fingerprints.append(fp)

    # Compare runs against run 0
    ref_labels = run_labels[0]
    ref_count = run_counts[0]
    ref_meas = run_meas_dfs[0]
    ref_class = run_class_dfs[0]
    ref_pheno = run_pheno_dfs[0]
    ref_summary = run_summary_dfs[0]
    ref_fp = run_fingerprints[0]

    labels_identical = True
    counts_identical = True
    measurements_identical = True
    classifications_identical = True
    phenotypes_identical = True
    summaries_identical = True
    config_fingerprint_identical = True

    for i in range(1, n_runs):
        if not np.array_equal(run_labels[i], ref_labels):
            labels_identical = False
            diagnostics.append(f"Labels mismatch between run 0 and run {i}.")

        if run_counts[i] != ref_count:
            counts_identical = False
            diagnostics.append(f"Count mismatch between run 0 ({ref_count}) and run {i} ({run_counts[i]}).")

        if ref_meas is not None:
            if not ref_meas.equals(run_meas_dfs[i]):
                measurements_identical = False
                diagnostics.append(f"Measurements DataFrame mismatch between run 0 and run {i}.")

        if ref_class is not None:
            if not ref_class.equals(run_class_dfs[i]):
                classifications_identical = False
                diagnostics.append(f"Classifications DataFrame mismatch between run 0 and run {i}.")

        if ref_pheno is not None:
            if not ref_pheno.equals(run_pheno_dfs[i]):
                phenotypes_identical = False
                diagnostics.append(f"Phenotypes DataFrame mismatch between run 0 and run {i}.")

        if ref_summary is not None:
            if not ref_summary.equals(run_summary_dfs[i]):
                summaries_identical = False
                diagnostics.append(f"Summaries DataFrame mismatch between run 0 and run {i}.")

        if run_fingerprints[i] != ref_fp:
            config_fingerprint_identical = False
            diagnostics.append(f"Config fingerprint mismatch between run 0 ({ref_fp}) and run {i} ({run_fingerprints[i]}).")

    return ReproducibilityRunResult(
        case_id=case.spec.case_id,
        run_count=n_runs,
        labels_identical=labels_identical,
        counts_identical=counts_identical,
        measurements_identical=measurements_identical,
        classifications_identical=classifications_identical,
        phenotypes_identical=phenotypes_identical,
        summaries_identical=summaries_identical,
        config_fingerprint_identical=config_fingerprint_identical,
        environment=env,
        diagnostics=tuple(diagnostics),
    )


def evaluate_cross_process_reproducibility(
    case: ValidationCase,
    *,
    temp_dir: Path | str | None = None,
) -> ReproducibilityRunResult:
    """Execute analysis across two independent Python interpreter subprocesses and verify output equality.

    Args:
        case: Immutable ValidationCase to evaluate.
        temp_dir: Optional directory path for temporary exchange artifacts.

    Returns:
        ReproducibilityRunResult comparing the two subprocess executions.
    """
    env = capture_runtime_environment()
    diagnostics: list[str] = []

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp_str:
        tmp_path = Path(tmp_str)
        input_npz = tmp_path / "input_case.npz"
        np.savez_compressed(
            input_npz,
            image=case.image_data,
        )

        out1_path = tmp_path / "out1.json"
        out2_path = tmp_path / "out2.json"
        labels1_path = tmp_path / "labels1.npy"
        labels2_path = tmp_path / "labels2.npy"

        worker_script = tmp_path / "worker.py"
        worker_code = f"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from nervenuc.validation.dataset import get_synthetic_case
from nervenuc.segmentation.core import segment_nuclei
from nervenuc.segmentation3d.engine import segment_nuclei_3d
from nervenuc.qc.session import start_review
from nervenuc.qc3d import start_review_3d
from nervenuc.measurements.engine import measure_markers
from nervenuc.measurements3d import measure_markers_3d
from nervenuc.classification.engine import classify_markers
from nervenuc.classification import classify_markers_3d
from nervenuc.summary.engine import summarize_metrics
from nervenuc.summary import summarize_metrics_3d
from nervenuc.preparation import prepare_volume
from nervenuc.workflow.fingerprint import compute_analysis_config_fingerprint, compute_volume_analysis_config_fingerprint

case_id = "{case.spec.case_id}"
out_json_path = Path(sys.argv[1])
labels_npy_path = Path(sys.argv[2])

case = get_synthetic_case(case_id, seed=42)
is_3d = case.spec.dimensions == "3D"
is_multi = len(case.spec.channel_names) > 1

if is_3d:
    prepared = prepare_volume(case.image_data, case.config.selection)
    seg_res = segment_nuclei_3d(prepared, case.config.segmentation)
    labels = seg_res.labels
    count = seg_res.nucleus_count
    fp = compute_volume_analysis_config_fingerprint(case.config)
    review = start_review_3d(seg_res)
    review.auto_accept()
    reviewed = review.finalize()
    meas_res = measure_markers_3d(
        reviewed,
        prepared.marker_volumes,
        prepared.provenance,
        case.config.markers,
    )
    meas_records = meas_res.measurements.to_dict(orient="records")
    class_res = classify_markers_3d(meas_res, case.config.thresholds)
    class_records = class_res.classifications.to_dict(orient="records")
    pheno_records = class_res.phenotypes.to_dict(orient="records")
    summary_res = summarize_metrics_3d(class_res, case.config.metrics)
    summary_records = summary_res.metrics.to_dict(orient="records")
else:
    dapi = case.image_data[0]
    seg_res = segment_nuclei(dapi, case.config.segmentation)
    labels = seg_res.labels
    count = len(seg_res.objects)
    fp = compute_analysis_config_fingerprint(case.config)

    if is_multi:
        rs = start_review(seg_res)
        for r_id in list(rs._active_objects):
            rs.accept(r_id)
        reviewed = rs.finalize()

        marker_images = {{
            name: case.image_data[idx]
            for idx, name in enumerate(case.spec.channel_names)
            if name != "DAPI"
        }}
        meas_res = measure_markers(reviewed, marker_images)
        meas_records = meas_res.measurements.to_dict(orient="records")

        class_res = classify_markers(meas_res, case.config.thresholds)
        class_records = class_res.classifications.to_dict(orient="records")
        pheno_records = class_res.phenotypes.to_dict(orient="records")

        summary_res = summarize_metrics(class_res, case.config.metrics)
        summary_records = summary_res.metrics.to_dict(orient="records")
    else:
        meas_records = []
        class_records = []
        pheno_records = []
        summary_records = []

np.save(labels_npy_path, labels)

result_data = {{
    "case_id": case_id,
    "count": count,
    "fingerprint": fp,
    "measurements": meas_records,
    "classifications": class_records,
    "phenotypes": pheno_records,
    "summary": summary_records,
}}

with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(result_data, f, indent=2)
"""
        worker_script.write_text(worker_code, encoding="utf-8")

        # Run Subprocess 1
        proc1 = subprocess.run(
            [sys.executable, str(worker_script), str(out1_path), str(labels1_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc1.returncode != 0:
            raise RuntimeError(f"Subprocess 1 failed: {proc1.stderr}")

        # Run Subprocess 2
        proc2 = subprocess.run(
            [sys.executable, str(worker_script), str(out2_path), str(labels2_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc2.returncode != 0:
            raise RuntimeError(f"Subprocess 2 failed: {proc2.stderr}")

        # Load and compare structured outputs
        labels1 = np.load(labels1_path)
        labels2 = np.load(labels2_path)
        labels_identical = bool(np.array_equal(labels1, labels2))
        if not labels_identical:
            diagnostics.append("Cross-process label arrays differ.")

        with open(out1_path, "r", encoding="utf-8") as f:
            res1 = json.load(f)
        with open(out2_path, "r", encoding="utf-8") as f:
            res2 = json.load(f)

        counts_identical = (res1["count"] == res2["count"])
        if not counts_identical:
            diagnostics.append(f"Cross-process counts differ: {res1['count']} vs {res2['count']}.")

        fp_identical = (res1["fingerprint"] == res2["fingerprint"])
        if not fp_identical:
            diagnostics.append(f"Cross-process fingerprints differ: {res1['fingerprint']} vs {res2['fingerprint']}.")

        meas_identical = (res1["measurements"] == res2["measurements"])
        if not meas_identical:
            diagnostics.append("Cross-process structured measurements differ.")

        class_identical = (res1["classifications"] == res2["classifications"])
        if not class_identical:
            diagnostics.append("Cross-process structured classifications differ.")

        pheno_identical = (res1["phenotypes"] == res2["phenotypes"])
        if not pheno_identical:
            diagnostics.append("Cross-process structured phenotypes differ.")

        summary_identical = (res1["summary"] == res2["summary"])
        if not summary_identical:
            diagnostics.append("Cross-process structured biological summaries differ.")

    return ReproducibilityRunResult(
        case_id=case.spec.case_id,
        run_count=2,
        labels_identical=labels_identical,
        counts_identical=counts_identical,
        measurements_identical=meas_identical,
        classifications_identical=class_identical,
        phenotypes_identical=pheno_identical,
        summaries_identical=summary_identical,
        config_fingerprint_identical=fp_identical,
        environment=env,
        diagnostics=tuple(diagnostics),
    )


def evaluate_parameter_stability(
    case: ValidationCase,
    parameter_name: str,
    values: Sequence[Any],
) -> ParameterStabilityResult:
    """Demonstrate how parameter perturbation alters scientific outputs while preserving determinism.

    Args:
        case: ValidationCase (e.g. SYN_2D_ISOLATED).
        parameter_name: Configuration parameter name (e.g. 'min_area_px').
        values: Sequence of 3 parameter values (e.g. baseline, boundary, high).

    Returns:
        ParameterStabilityResult documenting predicted counts, F1-scores, and mean IoUs.
    """
    if len(values) != 3:
        raise ValueError(f"Parameter stability demonstration requires exactly 3 values, got {len(values)}.")

    dapi = case.image_data[0]
    base_seg_config = case.config.segmentation

    pred_counts: list[int] = []
    f1_scores: list[float | None] = []
    mean_ious: list[float | None] = []

    import dataclasses

    for val in values:
        updated_cfg = dataclasses.replace(base_seg_config, **{parameter_name: val})

        seg_res = segment_nuclei(dapi, updated_cfg)
        pred_count = len(seg_res.objects)
        pred_counts.append(pred_count)

        matching = match_instances(seg_res.labels, case.ground_truth_labels, iou_threshold=0.5)
        metrics = compute_segmentation_metrics(matching, seg_res.labels, case.ground_truth_labels)

        f1_scores.append(metrics.f1_score)
        mean_ious.append(metrics.mean_matched_iou)

    return ParameterStabilityResult(
        case_id=case.spec.case_id,
        parameter_name=parameter_name,
        tested_values=tuple(values),
        predicted_counts=tuple(pred_counts),
        f1_scores=tuple(f1_scores),
        mean_ious=tuple(mean_ious),
    )
