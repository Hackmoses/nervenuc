"""Tests for end-to-end workflow orchestration and analysis configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

import nervenuc as nn
from nervenuc.aggregation import (
    SampleIdentity,
    SampleSummary,
    aggregate_by_animal,
)
from nervenuc.classification import (
    MarkerThreshold,
    classify_markers,
)
from nervenuc.export import (
    AnimalGroupAssignment,
    prepare_group_tables,
)
from nervenuc.handoff import (
    format_analysis_handoff,
    prepare_analysis_handoff,
)
from nervenuc.markers import MarkerDefinition
from nervenuc.measurements import measure_markers
from nervenuc.qc import ReviewedSegmentationResult, start_review
from nervenuc.segmentation import (
    SegmentationConfig,
    segment_nuclei,
)
from nervenuc.statistics import summarize_groups
from nervenuc.summary import (
    BiologicalMetricDefinition,
    BiologicalSummaryResult,
    PhenotypeFilter,
    summarize_metrics,
)
from nervenuc.workflow import (
    AnalysisConfig,
    DuplicateSampleError,
    ImageAnalysisSession,
    ImageSourceProvenance,
    InvalidWorkflowStateError,
    StudyAnalysisSession,
    WorkflowConfigurationError,
    compute_analysis_config_fingerprint,
    compute_pipeline_fingerprint,
)


def _create_synthetic_nuclear_and_marker_images() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create synthetic 2D nuclear and marker images with clear nucleus signals."""
    img_nuc = np.zeros((100, 100), dtype=np.float32)
    # 2 synthetic nuclei
    img_nuc[20:40, 20:40] = 100.0
    img_nuc[60:80, 60:80] = 100.0

    img_brdu = np.zeros((100, 100), dtype=np.float32)
    img_brdu[20:40, 20:40] = 50.0  # Nucleus 1 is BrdU+

    img_sox10 = np.zeros((100, 100), dtype=np.float32)
    img_sox10[20:40, 20:40] = 80.0  # Nucleus 1 is SOX10+
    img_sox10[60:80, 60:80] = 80.0  # Nucleus 2 is SOX10+

    return img_nuc, {"BrdU": img_brdu, "SOX10": img_sox10}


def _create_default_config() -> AnalysisConfig:
    """Create standard valid AnalysisConfig for tests."""
    seg = SegmentationConfig(threshold_method="otsu", min_area_px=10)
    m_brdu = MarkerDefinition("BrdU")
    m_sox10 = MarkerDefinition("SOX10")

    t_brdu = MarkerThreshold("BrdU", threshold=25.0)
    t_sox10 = MarkerThreshold("SOX10", threshold=40.0)

    d1 = BiologicalMetricDefinition(
        name="Proliferation",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    return AnalysisConfig(
        segmentation=seg,
        markers=(m_brdu, m_sox10),
        thresholds=(t_brdu, t_sox10),
        metrics=(d1,),
    )


def _finalize_review(segmentation) -> ReviewedSegmentationResult:
    session = start_review(segmentation)
    for review_id in session.result().objects["review_id"]:
        session.accept(review_id)
    return session.finalize()


# ==============================================================================
# 1. Configuration Tests
# ==============================================================================


def test_valid_analysis_config_instantiation() -> None:
    """Test minimal valid AnalysisConfig construction."""
    cfg = _create_default_config()
    assert len(cfg.markers) == 2
    assert len(cfg.thresholds) == 2
    assert len(cfg.metrics) == 1


def test_marker_without_threshold_allowed() -> None:
    """Test that configured markers can exist without thresholds (markers ⊇ thresholds)."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("DAPI_Ref")
    m2 = MarkerDefinition("BrdU")
    t2 = MarkerThreshold("BrdU", threshold=10.0)
    met = BiologicalMetricDefinition(
        name="M1",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )
    cfg = AnalysisConfig(
        segmentation=seg,
        markers=(m1, m2),
        thresholds=(t2,),  # DAPI_Ref has no threshold
        metrics=(met,),
    )
    assert len(cfg.markers) == 2
    assert len(cfg.thresholds) == 1


def test_threshold_referencing_unknown_marker_rejected() -> None:
    """Test rejection when a threshold references a marker not in configured markers."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("BrdU")
    t_unknown = MarkerThreshold("UNKNOWN", threshold=10.0)
    met = BiologicalMetricDefinition(
        "M1",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"UNKNOWN": True}),
    )

    with pytest.raises(WorkflowConfigurationError, match="Threshold references unknown marker"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1,),
            thresholds=(t_unknown,),
            metrics=(met,),
        )


def test_metric_referencing_unthresholded_marker_rejected() -> None:
    """Test rejection when a biological metric references a marker without a threshold."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("BrdU")
    m2 = MarkerDefinition("SOX10")
    t1 = MarkerThreshold("BrdU", threshold=10.0)
    # SOX10 has no threshold, but metric references SOX10
    met = BiologicalMetricDefinition(
        name="M1",
        denominator=PhenotypeFilter({"SOX10": True}),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    with pytest.raises(WorkflowConfigurationError, match="Metric 'M1' references marker.*no configured threshold"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1, m2),
            thresholds=(t1,),
            metrics=(met,),
        )


def test_duplicate_markers_and_metrics_rejected() -> None:
    """Test rejection of duplicate marker or metric names."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("BrdU")
    m2 = MarkerDefinition("BrdU")
    t1 = MarkerThreshold("BrdU", threshold=10.0)
    met1 = BiologicalMetricDefinition(
        "M1",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )

    with pytest.raises(WorkflowConfigurationError, match="Duplicate marker name"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1, m2),
            thresholds=(t1,),
            metrics=(met1,),
        )

    met2 = BiologicalMetricDefinition(
        "M1",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )
    with pytest.raises(WorkflowConfigurationError, match="Duplicate metric name"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1,),
            thresholds=(t1,),
            metrics=(met1, met2),
        )


def test_analysis_config_is_frozen() -> None:
    """Test that AnalysisConfig is immutable."""
    cfg = _create_default_config()
    with pytest.raises(FrozenInstanceError):
        cfg.markers = ()  # type: ignore[misc]


def test_analysis_config_serialization_round_trip() -> None:
    """Test deterministic to_dict() and from_dict() round trip."""
    cfg = _create_default_config()
    d = cfg.to_dict()
    reconstructed = AnalysisConfig.from_dict(d)

    assert reconstructed == cfg
    assert reconstructed.to_dict() == d


# ==============================================================================
# 2. Image Analysis Session Tests
# ==============================================================================


def test_complete_image_analysis_workflow() -> None:
    """Test full single-image lifecycle: set_image -> segment -> auto_accept -> measure -> classify -> summarize."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)

    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()
    session.set_image(img_nuc, markers)
    assert session.status()["image_set"]

    seg_res = session.segment()
    assert session.status()["segmented"]
    assert len(seg_res.objects) == 2

    rev_res = session.auto_accept_segmentation()
    assert session.status()["review_complete"]
    assert len(rev_res.objects) == 2

    meas_res = session.measure()
    assert session.status()["measured"]
    assert len(meas_res.measurements) == 4  # 2 nuclei x 2 markers

    class_res = session.classify()
    assert session.status()["classified"]
    assert len(class_res.phenotypes) == 2

    sum_res = session.summarize()
    assert session.status()["summarized"]
    assert len(sum_res.metrics) == 1

    # 1 BrdU+ SOX10+ nucleus out of 2 SOX10+ nuclei = 50%
    m_row = sum_res.metrics.iloc[0]
    assert m_row["numerator_count"] == 1
    assert m_row["denominator_count"] == 2
    assert m_row["percentage"] == pytest.approx(50.0)


def test_runtime_config_override_refreshes_2d_pipeline_fingerprint() -> None:
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)
    nuclear, markers = _create_synthetic_nuclear_and_marker_images()
    session.set_image(
        nuclear,
        markers,
        source_provenance=ImageSourceProvenance(
            source_kind="direct_2d",
            preparation_fingerprint="prepared-source-v1",
            pipeline_fingerprint="stale-fingerprint",
        ),
    )

    override = SegmentationConfig(threshold_method="otsu", min_area_px=11)
    session.segment(override)

    assert session.source_provenance is not None
    expected = compute_pipeline_fingerprint(
        compute_analysis_config_fingerprint(session.config),
        "prepared-source-v1",
    )
    assert session.source_provenance.pipeline_fingerprint == expected


def test_image_workflow_precondition_errors() -> None:
    """Test that calling image session methods out of order raises InvalidWorkflowStateError."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)

    with pytest.raises(InvalidWorkflowStateError, match="before setting image"):
        session.segment()

    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()
    session.set_image(img_nuc, markers)

    with pytest.raises(InvalidWorkflowStateError, match="before review is finalized"):
        session.measure()

    session.segment()

    with pytest.raises(InvalidWorkflowStateError, match="before review is finalized"):
        session.measure()

    with pytest.raises(InvalidWorkflowStateError, match="before measurement"):
        session.classify()

    with pytest.raises(InvalidWorkflowStateError, match="before classification"):
        session.summarize()


def test_start_review_does_not_finalize_review() -> None:
    """Test that start_review() creates a session but measure() still requires finalized review."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)
    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()

    session.set_image(img_nuc, markers)
    session.segment()

    rev_session = session.start_review()
    assert session.status()["review_started"]
    assert not session.status()["review_complete"]

    with pytest.raises(InvalidWorkflowStateError, match="before review is finalized"):
        session.measure()

    with pytest.raises(InvalidWorkflowStateError, match="Review is incomplete"):
        session.set_reviewed_result(rev_session.result())

    # Once every active object is explicitly accepted, measure succeeds.
    for review_id in rev_session.result().objects["review_id"]:
        rev_session.accept(review_id)
    session.set_reviewed_result(rev_session.finalize())
    assert session.status()["review_complete"]
    session.measure()
    assert session.status()["measured"]


def test_set_reviewed_result_stale_or_incompatible_rejected() -> None:
    """Test that set_reviewed_result rejects results from other images or shape mismatches."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)
    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()

    session.set_image(img_nuc, markers)
    session.segment()

    # Create a reviewed result from a different sized image (50x50)
    other_nuc = np.zeros((50, 50), dtype=np.float32)
    other_seg = segment_nuclei(other_nuc, cfg.segmentation)
    other_rev = _finalize_review(other_seg)

    with pytest.raises(InvalidWorkflowStateError, match="Reviewed label image shape .* does not match"):
        session.set_reviewed_result(other_rev)


def test_set_image_input_validation_errors() -> None:
    """Test rejection of non-2D images, missing markers, extra markers, or shape mismatches."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)
    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()

    # 3D nuclear channel
    with pytest.raises(WorkflowConfigurationError, match="nuclear_channel must be a 2D numpy array"):
        session.set_image(np.zeros((2, 100, 100)), markers)

    # Missing SOX10 marker
    with pytest.raises(WorkflowConfigurationError, match="missing configured markers"):
        session.set_image(img_nuc, {"BrdU": markers["BrdU"]})

    # Extra marker
    with pytest.raises(WorkflowConfigurationError, match="extra unconfigured markers"):
        session.set_image(
            img_nuc,
            {"BrdU": markers["BrdU"], "SOX10": markers["SOX10"], "EXTRA": markers["BrdU"]},
        )

    # Marker shape mismatch
    with pytest.raises(WorkflowConfigurationError, match="shape .* does not match nuclear channel shape"):
        session.set_image(
            img_nuc,
            {"BrdU": markers["BrdU"], "SOX10": np.zeros((50, 50), dtype=np.float32)},
        )


def test_image_workflow_downstream_invalidation() -> None:
    """Test that modifying upstream stages clears downstream derived results."""
    cfg = _create_default_config()
    session = ImageAnalysisSession(cfg)
    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()

    session.set_image(img_nuc, markers)
    session.segment()
    session.auto_accept_segmentation()
    session.measure()
    session.classify()
    session.summarize()

    assert all(session.status().values())

    # Re-running classify clears summary
    session.classify()
    assert session.status()["classified"]
    assert not session.status()["summarized"]

    # Re-running measure clears classify and summary
    session.measure()
    assert session.status()["measured"]
    assert not session.status()["classified"]
    assert not session.status()["summarized"]

    # Setting reviewed result clears measure, classify, summary
    rev_session = session.start_review()
    assert session.segmentation_result is not None
    assert session.reviewed_result is None
    assert session.measurement_result is None
    assert session.classification_result is None
    assert session.summary_result is None
    for review_id in rev_session.result().objects["review_id"]:
        rev_session.accept(review_id)
    session.set_reviewed_result(rev_session.finalize())
    assert not session.status()["measured"]

    # Re-running segment clears review and everything downstream
    session.segment()
    assert not session.status()["review_complete"]
    assert not session.status()["measured"]


def test_direct_m2_to_m6_engine_equivalence() -> None:
    """Critical test: verify ImageAnalysisSession produces identical output to direct M2-M6 engine composition."""
    cfg = _create_default_config()
    img_nuc, markers = _create_synthetic_nuclear_and_marker_images()

    # 1. Via ImageAnalysisSession
    session = ImageAnalysisSession(cfg)
    session.set_image(img_nuc, markers)
    session.segment()
    session.auto_accept_segmentation()
    session.measure()
    session.classify()
    session_summary = session.summarize()

    # 2. Via direct engine calls
    seg_res = segment_nuclei(img_nuc, cfg.segmentation)
    rev_res = _finalize_review(seg_res)
    meas_res = measure_markers(rev_res, markers)
    class_res = classify_markers(meas_res, cfg.thresholds)
    direct_summary = summarize_metrics(class_res, cfg.metrics)

    assert session_summary.metrics.equals(direct_summary.metrics)


# ==============================================================================
# 3. Study Analysis Session Tests
# ==============================================================================


def _create_sample_summary(sample_id: str, animal_id: str, num: int, den: int) -> tuple[SampleIdentity, BiologicalSummaryResult]:
    """Helper to construct a valid BiologicalSummaryResult with 1 metric."""
    d1 = BiologicalMetricDefinition(
        name="Proliferation",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"BrdU": True}),
    )
    pct = float(100.0 * num / den) if den > 0 else float("nan")
    df = pd.DataFrame(
        [
            {
                "metric_name": d1.name,
                "numerator_count": num,
                "denominator_count": den,
                "percentage": pct,
                "numerator_filter": d1.effective_numerator.to_string(),
                "denominator_filter": d1.denominator.to_string(),
            }
        ]
    )
    summary = BiologicalSummaryResult(metrics=df, definitions=(d1,), classification_result=None)  # type: ignore[arg-type]
    identity = SampleIdentity(sample_id=sample_id, animal_id=animal_id)
    return identity, summary


def test_study_analysis_canonical_sequence() -> None:
    """Test full study session sequence: add samples -> aggregate -> assign groups -> tables -> stats -> handoff."""
    study = StudyAnalysisSession()

    # Animal A1 (2 images), Animal A2 (1 image), Animal B1 (1 image)
    id1, s1 = _create_sample_summary("S1", "A1", 2, 10)
    id2, s2 = _create_sample_summary("S2", "A1", 3, 10)
    id3, s3 = _create_sample_summary("S3", "A2", 4, 10)
    id4, s4 = _create_sample_summary("S4", "B1", 1, 10)

    study.add_sample(id1, s1)
    study.add_sample(id2, s2)
    study.add_sample(id3, s3)
    study.add_sample(id4, s4)

    assert study.status()["sample_count"] == 4
    assert not study.status()["aggregated"]

    agg_res = study.aggregate()
    assert study.status()["aggregated"]
    assert study.status()["animal_count"] == 3  # A1, A2, B1

    # Raw count pooling for A1: num = 2 + 3 = 5, den = 10 + 10 = 20 -> 25%
    a1_row = agg_res.animal_metrics[agg_res.animal_metrics["animal_id"] == "A1"].iloc[0]
    assert a1_row["numerator_count"] == 5
    assert a1_row["denominator_count"] == 20
    assert a1_row["percentage"] == pytest.approx(25.0)

    assigns = [
        AnimalGroupAssignment("A1", "WT"),
        AnimalGroupAssignment("A2", "WT"),
        AnimalGroupAssignment("B1", "MUT"),
    ]
    study.assign_groups(assigns)
    assert study.status()["groups_assigned"]

    gt_res = study.prepare_group_tables()
    assert study.status()["group_tables_prepared"]
    assert len(gt_res.long_table) == 3

    stat_res = study.summarize_groups()
    assert study.status()["statistics_complete"]
    assert len(stat_res.group_statistics) == 2  # 2 groups x 1 metric

    # WT mean is arithmetic mean of A1 (25%) and A2 (40%) = 32.5%
    wt_stat = stat_res.group_statistics[stat_res.group_statistics["group"] == "WT"].iloc[0]
    assert wt_stat["mean"] == pytest.approx(32.5)

    handoff_res = study.prepare_handoff()
    assert study.status()["handoff_complete"]
    assert len(handoff_res.metric_readiness) == 1
    assert handoff_res.metric_readiness.iloc[0]["readiness_status"] == "ready"


def test_study_precondition_errors() -> None:
    """Test that calling study session methods out of canonical sequence raises InvalidWorkflowStateError."""
    study = StudyAnalysisSession()

    with pytest.raises(InvalidWorkflowStateError, match="before adding at least one sample"):
        study.aggregate()

    with pytest.raises(InvalidWorkflowStateError, match="before aggregating samples"):
        study.assign_groups([AnimalGroupAssignment("A1", "WT")])

    with pytest.raises(InvalidWorkflowStateError, match="before aggregation"):
        study.prepare_group_tables()

    with pytest.raises(InvalidWorkflowStateError, match="before preparing group tables"):
        study.summarize_groups()

    with pytest.raises(InvalidWorkflowStateError, match="before calculating group statistics"):
        study.prepare_handoff()


def test_duplicate_sample_id_rejected() -> None:
    """Test rejection when adding a duplicate sample_id."""
    study = StudyAnalysisSession()
    id1, s1 = _create_sample_summary("S1", "A1", 1, 10)
    id2, s2 = _create_sample_summary("S1", "A2", 2, 10)  # Duplicate S1

    study.add_sample(id1, s1)
    with pytest.raises(DuplicateSampleError, match="already registered"):
        study.add_sample(id2, s2)


def test_assign_groups_validation_against_aggregated_animals() -> None:
    """Test that assign_groups rejects unknown, missing, or duplicate animal assignments."""
    study = StudyAnalysisSession()
    id1, s1 = _create_sample_summary("S1", "A1", 1, 10)
    id2, s2 = _create_sample_summary("S2", "A2", 2, 10)

    study.add_sample(id1, s1)
    study.add_sample(id2, s2)
    study.aggregate()

    # Unknown animal A3
    with pytest.raises(WorkflowConfigurationError, match="Invalid group assignments"):
        study.assign_groups([AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A3", "WT")])

    # Missing animal A2
    with pytest.raises(WorkflowConfigurationError, match="Invalid group assignments"):
        study.assign_groups([AnimalGroupAssignment("A1", "WT")])

    # Duplicate animal A1
    with pytest.raises(WorkflowConfigurationError, match="Invalid group assignments"):
        study.assign_groups([AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A1", "MUT")])


def test_study_downstream_invalidation_cascades() -> None:
    """Test that study mutations clear aggregation, assignments, group tables, stats, and handoff."""
    study = StudyAnalysisSession()
    id1, s1 = _create_sample_summary("S1", "A1", 1, 10)
    id2, s2 = _create_sample_summary("S2", "A2", 2, 10)

    study.add_sample(id1, s1)
    study.add_sample(id2, s2)
    study.aggregate()
    study.assign_groups([AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A2", "MUT")])
    study.prepare_group_tables()
    study.summarize_groups()
    study.prepare_handoff()

    assert study.status()["handoff_complete"]

    # Re-running aggregate clears group assignments and all downstream
    study.aggregate()
    assert study.status()["aggregated"]
    assert not study.status()["groups_assigned"]
    assert not study.status()["group_tables_prepared"]
    assert not study.status()["statistics_complete"]
    assert not study.status()["handoff_complete"]

    # Re-assign and build up again
    study.assign_groups([AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A2", "MUT")])
    study.prepare_group_tables()
    study.summarize_groups()
    study.prepare_handoff()

    # Removing sample clears aggregation and assignments and all downstream
    study.remove_sample("S2")
    assert not study.status()["aggregated"]
    assert not study.status()["groups_assigned"]
    assert not study.status()["group_tables_prepared"]
    assert not study.status()["statistics_complete"]
    assert not study.status()["handoff_complete"]


def test_direct_m7_to_m10_engine_equivalence() -> None:
    """Critical test: verify StudyAnalysisSession produces identical handoff to direct M7-M10 engine composition."""
    id1, s1 = _create_sample_summary("S1", "A1", 2, 10)
    id2, s2 = _create_sample_summary("S2", "A1", 3, 10)
    id3, s3 = _create_sample_summary("S3", "A2", 4, 10)
    assigns = [AnimalGroupAssignment("A1", "WT"), AnimalGroupAssignment("A2", "WT")]

    # 1. Via StudyAnalysisSession
    study = StudyAnalysisSession()
    study.add_sample(id1, s1)
    study.add_sample(id2, s2)
    study.add_sample(id3, s3)
    study.aggregate()
    study.assign_groups(assigns)
    study.prepare_group_tables()
    study.summarize_groups()
    session_handoff = study.prepare_handoff()

    # 2. Via direct engine calls
    samples = [
        SampleSummary(id1, s1),
        SampleSummary(id2, s2),
        SampleSummary(id3, s3),
    ]
    agg_res = aggregate_by_animal(samples)
    gt_res = prepare_group_tables(agg_res, assigns)
    stat_res = summarize_groups(gt_res)
    direct_handoff = prepare_analysis_handoff(stat_res)

    assert session_handoff.metric_readiness.equals(direct_handoff.metric_readiness)
    assert session_handoff.group_readiness.equals(direct_handoff.group_readiness)
    assert format_analysis_handoff(session_handoff) == format_analysis_handoff(direct_handoff)


def test_top_level_package_imports() -> None:
    """Test that all workflow symbols are cleanly imported from top-level nervenuc and nervenuc.workflow."""
    assert hasattr(nn, "AnalysisConfig")
    assert hasattr(nn, "ImageAnalysisSession")
    assert hasattr(nn, "StudyAnalysisSession")
    assert hasattr(nn, "__version__")


def test_remove_nonexistent_sample_rejected() -> None:
    """Test that removing a non-existent sample raises InvalidWorkflowStateError."""
    study = StudyAnalysisSession()
    with pytest.raises(InvalidWorkflowStateError, match="Sample 'NON_EXISTENT' not found"):
        study.remove_sample("NON_EXISTENT")


def test_invalid_types_in_add_sample_rejected() -> None:
    """Test that add_sample rejects invalid identity or summary types."""
    study = StudyAnalysisSession()
    id1, s1 = _create_sample_summary("S1", "A1", 1, 10)

    with pytest.raises(WorkflowConfigurationError, match="identity must be a SampleIdentity"):
        study.add_sample("S1", s1)  # type: ignore[arg-type]

    with pytest.raises(WorkflowConfigurationError, match="summary must be a BiologicalSummaryResult"):
        study.add_sample(id1, "not_a_summary")  # type: ignore[arg-type]


def test_image_session_invalid_config_type_rejected() -> None:
    """Test that ImageAnalysisSession rejects non-AnalysisConfig objects."""
    with pytest.raises(WorkflowConfigurationError, match="config must be an AnalysisConfig"):
        ImageAnalysisSession("not_a_config")  # type: ignore[arg-type]


def test_config_empty_markers_rejected() -> None:
    """Test that AnalysisConfig rejects empty markers."""
    seg = SegmentationConfig(min_area_px=10)
    met = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    with pytest.raises(WorkflowConfigurationError, match="markers tuple cannot be empty"):
        AnalysisConfig(
            segmentation=seg,
            markers=(),
            thresholds=(),
            metrics=(met,),
        )


def test_config_empty_metrics_rejected() -> None:
    """Test that AnalysisConfig rejects empty metrics."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("BrdU")
    t1 = MarkerThreshold("BrdU", threshold=10.0)
    with pytest.raises(WorkflowConfigurationError, match="metrics tuple cannot be empty"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1,),
            thresholds=(t1,),
            metrics=(),
        )


def test_config_empty_thresholds_rejected_early() -> None:
    """The M11 pipeline cannot construct a threshold-free classification config."""
    seg = SegmentationConfig(min_area_px=10)
    marker = MarkerDefinition("BrdU")
    metric = BiologicalMetricDefinition("All", PhenotypeFilter(), PhenotypeFilter())
    with pytest.raises(WorkflowConfigurationError, match="thresholds tuple cannot be empty"):
        AnalysisConfig(
            segmentation=seg,
            markers=(marker,),
            thresholds=(),
            metrics=(metric,),
        )


def test_config_invalid_marker_item_type_rejected() -> None:
    """Test that AnalysisConfig rejects non-MarkerDefinition items."""
    seg = SegmentationConfig(min_area_px=10)
    met = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    with pytest.raises(WorkflowConfigurationError, match="Item at index 0 in markers is not a MarkerDefinition"):
        AnalysisConfig(
            segmentation=seg,
            markers=("BrdU",),  # type: ignore[arg-type]
            thresholds=(),
            metrics=(met,),
        )


def test_config_blank_marker_name_rejected() -> None:
    """Test that blank marker names are rejected."""
    from nervenuc.markers.errors import InvalidMarkerDefinitionError

    with pytest.raises(InvalidMarkerDefinitionError, match="cannot be empty or whitespace-only"):
        MarkerDefinition("   ")


def test_config_duplicate_threshold_marker_rejected() -> None:
    """Test that AnalysisConfig rejects duplicate thresholds for the same marker."""
    seg = SegmentationConfig(min_area_px=10)
    m1 = MarkerDefinition("BrdU")
    t1 = MarkerThreshold("BrdU", threshold=10.0)
    t2 = MarkerThreshold("BrdU", threshold=20.0)
    met = BiologicalMetricDefinition("M1", PhenotypeFilter(), PhenotypeFilter({"BrdU": True}))
    with pytest.raises(WorkflowConfigurationError, match="Duplicate threshold for marker 'BrdU'"):
        AnalysisConfig(
            segmentation=seg,
            markers=(m1,),
            thresholds=(t1, t2),
            metrics=(met,),
        )


def test_image_session_configuration_is_read_only_and_stage_updates_invalidate() -> None:
    """Completed scientific state cannot be relabeled by replacing the session configuration."""
    config = _create_default_config()
    session = ImageAnalysisSession(config)
    with pytest.raises(AttributeError):
        session.config = config  # type: ignore[misc]

    image, marker_images = _create_synthetic_nuclear_and_marker_images()
    session.set_image(image, marker_images)
    session.segment()
    session.auto_accept_segmentation()
    session.measure()
    session.classify()
    session.summarize()

    new_thresholds = (
        MarkerThreshold("BrdU", threshold=200.0),
        MarkerThreshold("SOX10", threshold=200.0),
    )
    session.classify(new_thresholds)
    assert session.config.thresholds == new_thresholds
    assert session.classification_result is not None
    assert session.summary_result is None


def test_study_single_image_animal_aggregation() -> None:
    """Test that animals with only 1 image aggregate correctly without weighting issues."""
    study = StudyAnalysisSession()
    id1, s1 = _create_sample_summary("S1", "A1", 5, 20)
    study.add_sample(id1, s1)
    agg_res = study.aggregate()

    assert len(agg_res.animal_metrics) == 1
    row = agg_res.animal_metrics.iloc[0]
    assert row["animal_id"] == "A1"
    assert row["numerator_count"] == 5
    assert row["denominator_count"] == 20
    assert row["percentage"] == pytest.approx(25.0)
