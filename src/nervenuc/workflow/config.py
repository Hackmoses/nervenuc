"""Immutable configuration models for reproducible image-level analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nervenuc.classification import MarkerThreshold
from nervenuc.markers import MarkerDefinition
from nervenuc.segmentation import SegmentationConfig
from nervenuc.summary import BiologicalMetricDefinition, PhenotypeFilter

from .errors import WorkflowConfigurationError


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable declaration of reproducible image-level analysis parameters.

    Attributes:
        segmentation: M2 baseline nucleus segmentation configuration.
        markers: Tuple of user-defined marker channel definitions.
        thresholds: Tuple of marker positivity thresholds.
        metrics: Tuple of denominator-aware biological metric definitions.
    """

    segmentation: SegmentationConfig
    markers: tuple[MarkerDefinition, ...]
    thresholds: tuple[MarkerThreshold, ...]
    metrics: tuple[BiologicalMetricDefinition, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate structural integrity and biological marker scoping.

        Raises:
            WorkflowConfigurationError: If marker names are duplicated, thresholds reference
                unknown markers, or metrics reference unthresholded markers.
        """
        if not isinstance(self.segmentation, SegmentationConfig):
            raise WorkflowConfigurationError(
                f"segmentation must be a SegmentationConfig, got {type(self.segmentation).__name__}."
            )

        if not self.markers:
            raise WorkflowConfigurationError("markers tuple cannot be empty.")

        marker_names: list[str] = []
        for idx, m in enumerate(self.markers):
            if not isinstance(m, MarkerDefinition):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in markers is not a MarkerDefinition: {type(m).__name__}."
                )
            if not m.name or not m.name.strip():
                raise WorkflowConfigurationError(f"Marker at index {idx} has an empty or blank name.")
            if m.name in marker_names:
                raise WorkflowConfigurationError(f"Duplicate marker name '{m.name}' in markers.")
            marker_names.append(m.name)

        marker_set = set(marker_names)

        if not self.thresholds:
            raise WorkflowConfigurationError("thresholds tuple cannot be empty.")

        threshold_marker_names: list[str] = []
        for idx, t in enumerate(self.thresholds):
            if not isinstance(t, MarkerThreshold):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in thresholds is not a MarkerThreshold: {type(t).__name__}."
                )
            if t.marker_name in threshold_marker_names:
                raise WorkflowConfigurationError(
                    f"Duplicate threshold for marker '{t.marker_name}' in thresholds."
                )
            if t.marker_name not in marker_set:
                raise WorkflowConfigurationError(
                    f"Threshold references unknown marker '{t.marker_name}' not in configured markers {marker_names}."
                )
            threshold_marker_names.append(t.marker_name)

        threshold_set = set(threshold_marker_names)

        if not self.metrics:
            raise WorkflowConfigurationError("metrics tuple cannot be empty.")

        metric_names: list[str] = []
        for idx, met in enumerate(self.metrics):
            if not isinstance(met, BiologicalMetricDefinition):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in metrics is not a BiologicalMetricDefinition: {type(met).__name__}."
                )
            if not met.name or not met.name.strip():
                raise WorkflowConfigurationError(f"Metric at index {idx} has an empty or blank name.")
            if met.name in metric_names:
                raise WorkflowConfigurationError(f"Duplicate metric name '{met.name}' in metrics.")
            metric_names.append(met.name)

            # Check that all markers in filters are thresholded
            referenced_markers = set(met.denominator.states.keys()) | set(
                met.effective_numerator.states.keys()
            )
            unthresholded = referenced_markers - threshold_set
            if unthresholded:
                raise WorkflowConfigurationError(
                    f"Metric '{met.name}' references marker(s) {sorted(unthresholded)} that have no configured threshold."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a pure Python dictionary.

        Returns:
            Dictionary containing primitives, lists, and dicts representing the configuration.
        """
        return {
            "segmentation": {
                "threshold_method": self.segmentation.threshold_method,
                "manual_threshold": self.segmentation.manual_threshold,
                "gaussian_sigma": self.segmentation.gaussian_sigma,
                "min_area_px": self.segmentation.min_area_px,
                "max_area_px": self.segmentation.max_area_px,
                "min_distance_px": self.segmentation.min_distance_px,
                "fill_holes": self.segmentation.fill_holes,
                "exclude_border": self.segmentation.exclude_border,
            },
            "markers": [{"name": m.name} for m in self.markers],
            "thresholds": [
                {
                    "marker_name": t.marker_name,
                    "threshold": t.threshold,
                    "measurement": t.measurement,
                    "comparison": t.comparison,
                }
                for t in self.thresholds
            ],
            "metrics": [
                {
                    "name": met.name,
                    "denominator": dict(met.denominator.states),
                    "numerator": dict(met.numerator.states),
                }
                for met in self.metrics
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalysisConfig:
        """Reconstruct an AnalysisConfig from a dictionary representation.

        Args:
            d: Dictionary formatted according to to_dict().

        Returns:
            Validated AnalysisConfig object.
        """
        seg_dict = d["segmentation"]
        seg_config = SegmentationConfig(
            threshold_method=seg_dict.get("threshold_method", "otsu"),
            manual_threshold=seg_dict.get("manual_threshold"),
            gaussian_sigma=seg_dict.get("gaussian_sigma", 1.0),
            min_area_px=seg_dict.get("min_area_px", 50),
            max_area_px=seg_dict.get("max_area_px"),
            min_distance_px=seg_dict.get("min_distance_px", 5),
            fill_holes=seg_dict.get("fill_holes", True),
            exclude_border=seg_dict.get("exclude_border", False),
        )

        markers = tuple(MarkerDefinition(m["name"]) for m in d["markers"])

        thresholds = tuple(
            MarkerThreshold(
                marker_name=t["marker_name"],
                threshold=t["threshold"],
                measurement=t.get("measurement", "mean_intensity"),
                comparison=t.get("comparison", ">="),
            )
            for t in d["thresholds"]
        )

        metrics = tuple(
            BiologicalMetricDefinition(
                name=met["name"],
                denominator=PhenotypeFilter(met.get("denominator", {})),
                numerator=PhenotypeFilter(met.get("numerator", {})),
            )
            for met in d["metrics"]
        )

        return cls(
            segmentation=seg_config,
            markers=markers,
            thresholds=thresholds,
            metrics=metrics,
        )


@dataclass(frozen=True)
class VolumeAnalysisConfig:
    """Immutable declaration of reproducible 3D volume analysis parameters.

    Attributes:
        selection: Z-stack volume extraction configuration.
        segmentation: M14 3D classical nucleus segmentation configuration.
        markers: Tuple of user-defined marker definitions.
        thresholds: Tuple of marker positivity thresholds.
        metrics: Tuple of denominator-aware biological metric definitions.
    """

    selection: Any
    segmentation: Any
    markers: tuple[MarkerDefinition, ...]
    thresholds: tuple[MarkerThreshold, ...]
    metrics: tuple[BiologicalMetricDefinition, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate structural integrity and biological marker scoping for 3D analysis."""
        from nervenuc.preparation import VolumeSelectionConfig
        from nervenuc.segmentation3d import Segmentation3DConfig

        if not isinstance(self.selection, VolumeSelectionConfig):
            raise WorkflowConfigurationError(
                f"selection must be a VolumeSelectionConfig, got {type(self.selection).__name__}."
            )

        if not isinstance(self.segmentation, Segmentation3DConfig):
            raise WorkflowConfigurationError(
                f"segmentation must be a Segmentation3DConfig, got {type(self.segmentation).__name__}."
            )

        if not self.markers:
            raise WorkflowConfigurationError("markers tuple cannot be empty.")

        marker_names: list[str] = []
        for idx, m in enumerate(self.markers):
            if not isinstance(m, MarkerDefinition):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in markers is not a MarkerDefinition: {type(m).__name__}."
                )
            if not m.name or not m.name.strip():
                raise WorkflowConfigurationError(f"Marker at index {idx} has an empty or blank name.")
            if m.name in marker_names:
                raise WorkflowConfigurationError(f"Duplicate marker name '{m.name}' in markers.")
            marker_names.append(m.name)

        marker_set = set(marker_names)
        selection_markers = set(self.selection.marker_channels.keys())
        if marker_set != selection_markers:
            raise WorkflowConfigurationError(
                f"Configured markers {sorted(marker_set)} do not match selection marker channels {sorted(selection_markers)}."
            )

        if not self.thresholds:
            raise WorkflowConfigurationError("thresholds tuple cannot be empty.")

        threshold_marker_names: list[str] = []
        for idx, t in enumerate(self.thresholds):
            if not isinstance(t, MarkerThreshold):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in thresholds is not a MarkerThreshold: {type(t).__name__}."
                )
            if t.marker_name in threshold_marker_names:
                raise WorkflowConfigurationError(
                    f"Duplicate threshold for marker '{t.marker_name}' in thresholds."
                )
            if t.marker_name not in marker_set:
                raise WorkflowConfigurationError(
                    f"Threshold references unknown marker '{t.marker_name}' not in configured markers {marker_names}."
                )
            threshold_marker_names.append(t.marker_name)

        threshold_set = set(threshold_marker_names)

        if not self.metrics:
            raise WorkflowConfigurationError("metrics tuple cannot be empty.")

        metric_names: list[str] = []
        for idx, met in enumerate(self.metrics):
            if not isinstance(met, BiologicalMetricDefinition):
                raise WorkflowConfigurationError(
                    f"Item at index {idx} in metrics is not a BiologicalMetricDefinition: {type(met).__name__}."
                )
            if not met.name or not met.name.strip():
                raise WorkflowConfigurationError(f"Metric at index {idx} has an empty or blank name.")
            if met.name in metric_names:
                raise WorkflowConfigurationError(f"Duplicate metric name '{met.name}' in metrics.")
            metric_names.append(met.name)

            referenced_markers = set(met.denominator.states.keys()) | set(
                met.effective_numerator.states.keys()
            )
            unthresholded = referenced_markers - threshold_set
            if unthresholded:
                raise WorkflowConfigurationError(
                    f"Metric '{met.name}' references marker(s) {sorted(unthresholded)} that have no configured threshold."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a pure Python dictionary."""
        return {
            "selection": self.selection.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "markers": [{"name": m.name} for m in self.markers],
            "thresholds": [
                {
                    "marker_name": t.marker_name,
                    "threshold": t.threshold,
                    "measurement": t.measurement,
                    "comparison": t.comparison,
                }
                for t in self.thresholds
            ],
            "metrics": [
                {
                    "name": met.name,
                    "denominator": dict(met.denominator.states),
                    "numerator": dict(met.numerator.states),
                }
                for met in self.metrics
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VolumeAnalysisConfig:
        """Reconstruct a VolumeAnalysisConfig from a dictionary representation."""
        from nervenuc.preparation import VolumeSelectionConfig
        from nervenuc.segmentation3d import Segmentation3DConfig

        selection = VolumeSelectionConfig.from_dict(d["selection"])
        segmentation = Segmentation3DConfig.from_dict(d["segmentation"])

        markers = tuple(MarkerDefinition(m["name"]) for m in d["markers"])

        thresholds = tuple(
            MarkerThreshold(
                marker_name=t["marker_name"],
                threshold=t["threshold"],
                measurement=t.get("measurement", "mean_intensity"),
                comparison=t.get("comparison", ">="),
            )
            for t in d["thresholds"]
        )

        metrics = tuple(
            BiologicalMetricDefinition(
                name=met["name"],
                denominator=PhenotypeFilter(met.get("denominator", {})),
                numerator=PhenotypeFilter(met.get("numerator", {})),
            )
            for met in d["metrics"]
        )

        return cls(
            selection=selection,
            segmentation=segmentation,
            markers=markers,
            thresholds=thresholds,
            metrics=metrics,
        )
