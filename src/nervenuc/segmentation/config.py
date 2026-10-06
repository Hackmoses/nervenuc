"""Configuration models for nuclear segmentation."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidSegmentationInputError


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration parameters for 2D classical nuclear segmentation.

    Attributes:
        threshold_method: Foreground binarization method ('otsu', 'li', or 'manual').
        manual_threshold: Intensity cutoff when threshold_method is 'manual' (raw intensity units).
        gaussian_sigma: Standard deviation (pixels) for Gaussian smoothing. 0.0 disables smoothing.
        min_area_px: Minimum nucleus area (pixels). Objects smaller than this are removed.
        max_area_px: Maximum nucleus area (pixels). Objects larger than this are removed. None disables.
        min_distance_px: Minimum Euclidean distance (pixels) between watershed seed peaks.
        fill_holes: Whether to fill enclosed holes within foreground nuclear regions before watershed.
        exclude_border: If True, border-touching nuclei are discarded. If False, retained and flagged.
    """

    threshold_method: str = "otsu"
    manual_threshold: float | None = None
    gaussian_sigma: float = 1.0
    min_area_px: int = 50
    max_area_px: int | None = None
    min_distance_px: int = 5
    fill_holes: bool = True
    exclude_border: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_methods = ("otsu", "li", "manual")
        if self.threshold_method not in valid_methods:
            raise InvalidSegmentationInputError(
                f"threshold_method must be one of {valid_methods}, got '{self.threshold_method}'."
            )

        if self.threshold_method == "manual" and self.manual_threshold is None:
            raise InvalidSegmentationInputError(
                "manual_threshold is required when threshold_method is 'manual'."
            )

        if self.gaussian_sigma < 0:
            raise InvalidSegmentationInputError(
                f"gaussian_sigma must be non-negative, got {self.gaussian_sigma}."
            )

        if self.min_area_px < 0:
            raise InvalidSegmentationInputError(
                f"min_area_px must be non-negative, got {self.min_area_px}."
            )

        if self.max_area_px is not None:
            if self.max_area_px < 0:
                raise InvalidSegmentationInputError(
                    f"max_area_px must be non-negative, got {self.max_area_px}."
                )
            if self.max_area_px < self.min_area_px:
                raise InvalidSegmentationInputError(
                    f"max_area_px ({self.max_area_px}) cannot be less than min_area_px ({self.min_area_px})."
                )

        if self.min_distance_px < 1:
            raise InvalidSegmentationInputError(
                f"min_distance_px must be at least 1, got {self.min_distance_px}."
            )
