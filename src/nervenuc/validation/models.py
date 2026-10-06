"""Immutable data models for validation specifications, ground truth, matching, and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from .errors import InvalidGroundTruthError, ValidationCaseSpecificationError

if TYPE_CHECKING:
    from nervenuc.workflow.config import AnalysisConfig


VALID_TIERS = {"synthetic", "development_tuning", "unbiased_validation"}
VALID_DIMENSIONS = {"2D", "3D"}
VALID_MATCH_TYPES = {"TP", "FP", "FN"}


@dataclass(frozen=True)
class ValidationCaseSpec:
    """Portable, serializable specification of a validation case.

    Contains purely serializable metadata, relative POSIX paths, channel declarations,
    and cryptographic SHA-256 hashes. Does not contain raw NumPy arrays.
    """

    case_id: str
    description: str
    tier: Literal["synthetic", "development_tuning", "unbiased_validation"]
    dimensions: Literal["2D", "3D"]
    image_relpath: str
    ground_truth_labels_relpath: str
    ground_truth_phenotypes_relpath: str | None = None
    image_sha256: str = ""
    ground_truth_sha256: str = ""
    axes: str = "CYX"
    channel_names: tuple[str, ...] = ("DAPI",)
    voxel_spacing_um: tuple[float, float, float] | None = None  # (z, y, x) in micrometers
    annotator_id: str = "ground_truth"
    config_dict: Mapping[str, Any] | None = None
    preset_relpath: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValidationCaseSpecificationError("case_id must be a non-empty string.")
        if self.tier not in VALID_TIERS:
            raise ValidationCaseSpecificationError(
                f"tier must be one of {sorted(VALID_TIERS)}, got '{self.tier}'."
            )
        if self.dimensions not in VALID_DIMENSIONS:
            raise ValidationCaseSpecificationError(
                f"dimensions must be one of {sorted(VALID_DIMENSIONS)}, got '{self.dimensions}'."
            )
        if not self.image_relpath or not isinstance(self.image_relpath, str):
            raise ValidationCaseSpecificationError("image_relpath must be a non-empty relative path string.")
        if not self.ground_truth_labels_relpath or not isinstance(self.ground_truth_labels_relpath, str):
            raise ValidationCaseSpecificationError("ground_truth_labels_relpath must be a non-empty relative path string.")
        if not self.axes or not isinstance(self.axes, str):
            raise ValidationCaseSpecificationError("axes must be a non-empty string (e.g. 'CYX' or 'CZYX').")
        if not self.channel_names or not isinstance(self.channel_names, tuple):
            raise ValidationCaseSpecificationError("channel_names must be a non-empty tuple of strings.")


@dataclass(frozen=True)
class GroundTruthObject:
    """Ground-truth nucleus object annotation with spatial and biological attributes."""

    truth_id: int
    label_name: str
    centroid: tuple[float, ...]  # (y, x) in 2D or (z, y, x) in 3D
    size: int  # Area (pixels) or Volume (voxels)
    phenotype: str = "Unassigned"
    marker_positivity: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.truth_id, int) or self.truth_id <= 0:
            raise InvalidGroundTruthError(f"truth_id must be a positive integer, got {self.truth_id}.")
        if not self.label_name or not isinstance(self.label_name, str):
            raise InvalidGroundTruthError("label_name must be a non-empty string.")
        if not isinstance(self.centroid, tuple) or len(self.centroid) not in (2, 3):
            raise InvalidGroundTruthError(f"centroid must be a 2D or 3D coordinate tuple, got {self.centroid}.")
        if not isinstance(self.size, int) or self.size <= 0:
            raise InvalidGroundTruthError(f"size must be a positive integer, got {self.size}.")


@dataclass(frozen=True)
class ValidationCase:
    """In-memory loaded validation case with validated NumPy arrays and typed configuration."""

    spec: ValidationCaseSpec
    image_data: np.ndarray
    ground_truth_labels: np.ndarray
    ground_truth_objects: tuple[GroundTruthObject, ...]
    config: AnalysisConfig

    def __post_init__(self) -> None:
        if not isinstance(self.image_data, np.ndarray):
            raise InvalidGroundTruthError("image_data must be a NumPy array.")
        if not isinstance(self.ground_truth_labels, np.ndarray):
            raise InvalidGroundTruthError("ground_truth_labels must be a NumPy array.")
        if self.image_data.ndim not in (2, 3, 4):
            raise InvalidGroundTruthError(f"image_data ndim must be 2, 3, or 4, got {self.image_data.ndim}.")


@dataclass(frozen=True)
class ObjectMatch:
    """Pairwise match between a predicted nucleus and a ground-truth nucleus.

    Preserves exact instance identifiers, Jaccard IoU, Dice coefficient, match classification
    (TP, FP, FN), raw predicted and truth sizes, signed and absolute size differences, size ratios,
    and centroid distance errors (pixels and physical micrometers).
    """

    pred_id: int | None
    pred_review_id: str | None
    truth_id: int | None
    truth_name: str | None
    iou: float
    dice: float
    match_type: Literal["TP", "FP", "FN"]
    pred_size: int | None
    truth_size: int | None
    signed_size_diff: int | None
    abs_size_diff: int | None
    size_ratio: float | None
    centroid_distance_px: float | None
    centroid_distance_um: float | None

    def __post_init__(self) -> None:
        if self.match_type not in VALID_MATCH_TYPES:
            raise ValueError(f"match_type must be one of {VALID_MATCH_TYPES}, got '{self.match_type}'.")
        if self.match_type == "TP":
            if self.pred_id is None or self.truth_id is None:
                raise ValueError("TP match requires both pred_id and truth_id to be non-None.")
        elif self.match_type == "FP":
            if self.pred_id is None or self.truth_id is not None:
                raise ValueError("FP match requires non-None pred_id and None truth_id.")
        elif self.match_type == "FN":
            if self.pred_id is not None or self.truth_id is None:
                raise ValueError("FN match requires None pred_id and non-None truth_id.")


@dataclass(frozen=True)
class MatchingResult:
    """Full deterministic bipartite matching outcome for an analysis case."""

    case_id: str
    iou_threshold: float
    matches: tuple[ObjectMatch, ...]
    tp_count: int
    fp_count: int
    fn_count: int
    pred_count: int
    truth_count: int

    def __post_init__(self) -> None:
        if self.tp_count < 0 or self.fp_count < 0 or self.fn_count < 0:
            raise ValueError("Match counts cannot be negative.")
        if self.tp_count + self.fp_count != self.pred_count:
            raise ValueError(
                f"Inconsistent prediction accounting: TP({self.tp_count}) + FP({self.fp_count}) != pred_count({self.pred_count})."
            )
        if self.tp_count + self.fn_count != self.truth_count:
            raise ValueError(
                f"Inconsistent ground-truth accounting: TP({self.tp_count}) + FN({self.fn_count}) != truth_count({self.truth_count})."
            )
        if len(self.matches) != self.tp_count + self.fp_count + self.fn_count:
            raise ValueError(
                f"Match tuple length ({len(self.matches)}) does not equal total matches ({self.tp_count + self.fp_count + self.fn_count})."
            )


@dataclass(frozen=True)
class SegmentationMetrics:
    """Object-level, foreground overlap, size, and centroid evaluation metrics."""

    precision_defined: bool
    precision: float | None

    recall_defined: bool
    recall: float | None

    f1_score_defined: bool
    f1_score: float | None

    mean_matched_iou: float | None
    median_matched_iou: float | None
    mean_matched_dice: float | None
    median_matched_dice: float | None

    foreground_iou_defined: bool
    foreground_iou: float | None

    foreground_dice_defined: bool
    foreground_dice: float | None

    pred_count: int
    truth_count: int
    count_signed_error: int
    count_absolute_error: int
    count_percent_error_defined: bool
    count_percent_error: float | None

    mean_centroid_error_px: float | None
    median_centroid_error_px: float | None
    mean_centroid_error_um: float | None
    median_centroid_error_um: float | None

    mean_signed_size_diff: float | None
    mean_absolute_size_diff: float | None
    median_absolute_size_diff: float | None
    mean_size_ratio: float | None

    is_empty_case_concordant: bool


@dataclass(frozen=True)
class MarkerClassificationMetrics:
    """Classification performance for a single marker on matched TP nuclei."""

    marker_name: str
    tp: int
    fp: int
    tn: int
    fn: int

    sensitivity_defined: bool
    sensitivity: float | None

    specificity_defined: bool
    specificity: float | None

    precision_defined: bool
    precision: float | None

    f1_score_defined: bool
    f1_score: float | None

    accuracy_defined: bool
    accuracy: float | None

    matched_object_count: int


@dataclass(frozen=True)
class PhenotypeConfusionMatrix:
    """Multi-class confusion matrix for composite phenotype classifications on matched TP nuclei."""

    phenotypes: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]  # Rows = Ground Truth, Columns = Predicted
    matched_object_count: int
    overall_accuracy_defined: bool
    overall_accuracy: float | None
    macro_f1_defined: bool
    macro_f1: float | None
    contributing_class_count: int


@dataclass(frozen=True)
class BiologicalMetricComparison:
    """Comparison record between predicted and ground-truth derived biological metrics."""

    metric_name: str
    pred_numerator: int | float
    pred_denominator: int | float
    pred_percentage: float | None
    truth_numerator: int | float
    truth_denominator: int | float
    truth_percentage: float | None
    signed_difference: float | None
    absolute_difference: float | None


@dataclass(frozen=True)
class ProjectionVs3DBenchmarkResult:
    """Benchmark outcome comparing projected 2D segmentation with true 3D volumetric analysis."""

    case_id: str
    truth_3d_count: int

    # 2D Projected branch results
    projected_2d_pred_count: int
    projected_2d_tp: int
    projected_2d_fp: int
    projected_2d_fn: int
    projected_2d_precision: float | None
    projected_2d_recall: float | None
    projected_2d_f1: float | None

    # True 3D Volumetric branch results
    true_3d_pred_count: int
    true_3d_tp: int
    true_3d_fp: int
    true_3d_fn: int
    true_3d_precision: float | None
    true_3d_recall: float | None
    true_3d_f1: float | None


@dataclass(frozen=True)
class ReproducibilityRunResult:
    """Outcome of repeated execution testing for scientific reproducibility."""

    case_id: str
    run_count: int
    labels_identical: bool
    counts_identical: bool
    measurements_identical: bool
    classifications_identical: bool
    phenotypes_identical: bool
    summaries_identical: bool
    config_fingerprint_identical: bool
    environment: dict[str, str]
    diagnostics: tuple[str, ...]

    @property
    def is_fully_reproducible(self) -> bool:
        """Return True if all tested scientific components achieved complete determinism."""
        return (
            self.labels_identical
            and self.counts_identical
            and self.measurements_identical
            and self.classifications_identical
            and self.phenotypes_identical
            and self.summaries_identical
            and self.config_fingerprint_identical
        )


@dataclass(frozen=True)
class ParameterStabilityResult:
    """Outcome of parameter stability perturbation demonstrating sensitivity distinct from reproducibility."""

    case_id: str
    parameter_name: str
    tested_values: tuple[Any, ...]
    predicted_counts: tuple[int, ...]
    f1_scores: tuple[float | None, ...]
    mean_ious: tuple[float | None, ...]


@dataclass(frozen=True)
class RealValidationCaseSpec:
    """Specification of a real microscopy validation case with distinct artifact hashes."""

    case_id: str
    animal_id: str
    description: str
    tier: Literal["development_tuning", "unbiased_validation"]
    dimensions: Literal["2D", "3D"]
    image_relpath: str
    ground_truth_rois_relpath: str
    ground_truth_markers_relpath: str
    image_sha256: str = ""
    ground_truth_rois_sha256: str = ""
    ground_truth_markers_sha256: str = ""
    axes: str = "CYX"
    channel_names: tuple[str, ...] = ("DAPI",)
    voxel_spacing_um: tuple[float, float, float] | None = None
    annotator_id: str = "annotator"
    annotation_tool: str = "Fiji"
    annotation_date: str = ""
    annotation_mode: Literal["independent_manual", "assisted_development"] = "independent_manual"
    preset_relpath: str | None = None
    config_dict: Mapping[str, Any] | None = None
    selection_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.case_id, str):
            raise ValidationCaseSpecificationError("case_id must be a non-empty string.")
        if not self.animal_id or not isinstance(self.animal_id, str):
            raise ValidationCaseSpecificationError("animal_id must be a non-empty string.")
        if self.tier not in {"development_tuning", "unbiased_validation"}:
            raise ValidationCaseSpecificationError(f"Invalid tier: '{self.tier}'.")
        if self.dimensions not in {"2D", "3D"}:
            raise ValidationCaseSpecificationError(f"Invalid dimensions: '{self.dimensions}'.")
        if not self.image_relpath:
            raise ValidationCaseSpecificationError("image_relpath must be non-empty.")
        if not self.ground_truth_rois_relpath:
            raise ValidationCaseSpecificationError("ground_truth_rois_relpath must be non-empty.")
        if not self.ground_truth_markers_relpath:
            raise ValidationCaseSpecificationError("ground_truth_markers_relpath must be non-empty.")


@dataclass(frozen=True)
class RealValidationManifest:
    """Portable real validation dataset manifest."""

    schema_version: str
    dataset_name: str
    created_date: str
    cases: tuple[RealValidationCaseSpec, ...]
    manifest_path: str = ""


@dataclass(frozen=True)
class RealCaseValidationResult:
    """Scientific evaluation result for an individual real microscopy image."""

    spec: RealValidationCaseSpec
    segmentation_metrics: SegmentationMetrics
    marker_metrics: tuple[MarkerClassificationMetrics, ...]
    phenotype_confusion: PhenotypeConfusionMatrix | None
    biological_metrics: tuple[BiologicalMetricComparison, ...]
    matching_result: MatchingResult
    config_fingerprint: str
    pred_object_count: int
    truth_object_count: int
