"""Tests for lean scientific reproducibility and parameter stability validation."""

import pytest

from nervenuc.validation.dataset import get_synthetic_case
from nervenuc.validation.reproducibility import (
    capture_runtime_environment,
    evaluate_cross_process_reproducibility,
    evaluate_parameter_stability,
    evaluate_same_process_reproducibility,
)


def test_environment_capture() -> None:
    """Verify runtime environment capture gathers all core platform and scientific dependency versions."""
    env = capture_runtime_environment()
    required_keys = {
        "python_version",
        "platform",
        "os",
        "numpy_version",
        "scipy_version",
        "skimage_version",
        "pandas_version",
        "nervenuc_version",
    }
    assert required_keys.issubset(env.keys())
    for k in required_keys:
        assert isinstance(env[k], str) and len(env[k]) > 0


def test_same_process_2d_segmentation_reproducibility() -> None:
    """Verify 3 consecutive runs of 2D segmentation yield 100% bitwise identical labels and count."""
    case = get_synthetic_case("SYN_2D_ISOLATED", seed=42)
    res = evaluate_same_process_reproducibility(case, n_runs=3)

    assert res.run_count == 3
    assert res.labels_identical is True
    assert res.counts_identical is True
    assert res.config_fingerprint_identical is True
    assert res.is_fully_reproducible is True
    assert len(res.diagnostics) == 0


def test_same_process_multi_marker_pipeline_reproducibility() -> None:
    """Verify 3 consecutive runs of multi-marker segmentation, measurement, classification, and summary are deterministic."""
    case = get_synthetic_case("SYN_2D_MULTI_MARKER", seed=42)
    res = evaluate_same_process_reproducibility(case, n_runs=3)

    assert res.run_count == 3
    assert res.labels_identical is True
    assert res.counts_identical is True
    assert res.measurements_identical is True
    assert res.classifications_identical is True
    assert res.phenotypes_identical is True
    assert res.summaries_identical is True
    assert res.config_fingerprint_identical is True
    assert res.is_fully_reproducible is True
    assert len(res.diagnostics) == 0


def test_same_process_3d_volumetric_reproducibility() -> None:
    """Verify 3 consecutive runs of 3D volumetric segmentation yield bitwise identical output."""
    case = get_synthetic_case("SYN_3D_ISOLATED", seed=42)
    res = evaluate_same_process_reproducibility(case, n_runs=3)

    assert res.run_count == 3
    assert res.labels_identical is True
    assert res.counts_identical is True
    assert res.config_fingerprint_identical is True
    assert res.is_fully_reproducible is True
    assert len(res.diagnostics) == 0


def test_cross_process_restart_reproducibility() -> None:
    """Verify isolated Python interpreter subprocesses produce identical scientific outputs without state leakage."""
    case = get_synthetic_case("SYN_2D_MULTI_MARKER", seed=42)
    res = evaluate_cross_process_reproducibility(case)

    assert res.run_count == 2
    assert res.labels_identical is True
    assert res.counts_identical is True
    assert res.measurements_identical is True
    assert res.classifications_identical is True
    assert res.phenotypes_identical is True
    assert res.summaries_identical is True
    assert res.config_fingerprint_identical is True
    assert res.is_fully_reproducible is True


def test_cross_process_restart_reproducibility_3d_full_pipeline() -> None:
    """3D restart checks must include measurements, classification, phenotypes, and summaries."""
    case = get_synthetic_case("SYN_3D_ISOLATED", seed=42)
    res = evaluate_cross_process_reproducibility(case)
    assert res.is_fully_reproducible is True
    assert len(res.diagnostics) == 0


def test_parameter_stability_demonstration() -> None:
    """Demonstrate that varying min_area_px alters object retention while each run remains deterministic."""
    case = get_synthetic_case("SYN_2D_ISOLATED", seed=42)
    # Segmented nucleus sizes: [189, 193, 189, 183] px
    res = evaluate_parameter_stability(case, "min_area_px", (20, 185, 200))

    assert res.parameter_name == "min_area_px"
    assert res.tested_values == (20, 185, 200)
    assert res.predicted_counts == (4, 3, 0)

    # Setting 1 (20 px): All 4 objects retained -> F1 = 1.0
    assert res.f1_scores[0] == 1.0

    # Setting 2 (185 px): Smallest object (183 px) filtered out -> 3 retained -> F1 ≈ 6/7 = 0.857
    assert res.f1_scores[1] is not None
    assert pytest.approx(res.f1_scores[1], rel=1e-3) == 6.0 / 7.0

    # Setting 3 (200 px): All objects filtered out -> 0 retained -> F1 is None
    assert res.f1_scores[2] is None or res.f1_scores[2] == 0.0
