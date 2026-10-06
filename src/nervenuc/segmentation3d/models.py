"""Data models for 3D nuclear segmentation configuration and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from nervenuc.preparation import VoxelSpacing

from .errors import InvalidSegmentation3DConfigError


@dataclass(frozen=True)
class Segmentation3DConfig:
    """Configuration parameters for 3D classical nuclear segmentation.

    Attributes:
        threshold_method: Foreground binarization method ('otsu' or 'manual').
        manual_threshold: Intensity cutoff when threshold_method is 'manual'.
        gaussian_sigma_xy_px: Standard deviation (pixels) for XY Gaussian smoothing.
        gaussian_sigma_z_slices: Standard deviation (slices) for Z Gaussian smoothing (or None to disable Z smoothing).
        min_volume_voxels: Minimum nucleus volume (voxels). Objects smaller than this are removed.
        max_volume_voxels: Maximum nucleus volume (voxels). Objects larger than this are removed (or None).
        min_distance_xy_px: Minimum XY Euclidean distance (pixels) between watershed seed peaks.
        min_distance_z_slices: Minimum Z Euclidean distance (slices) between watershed seed peaks.
        fill_holes: Whether to fill enclosed 3D holes within foreground regions before watershed.
        exclude_z_border: If True, objects touching Z boundaries (z=0 or z=Z-1) are removed.
        exclude_xy_border: If True, objects touching XY boundaries (x=0, x=X-1, y=0, y=Y-1) are removed.
    """

    threshold_method: Literal["otsu", "manual"] = "otsu"
    manual_threshold: float | None = None
    gaussian_sigma_xy_px: float = 1.0
    gaussian_sigma_z_slices: float | None = None
    min_volume_voxels: int = 100
    max_volume_voxels: int | None = None
    min_distance_xy_px: int = 5
    min_distance_z_slices: int = 2
    fill_holes: bool = True
    exclude_z_border: bool = False
    exclude_xy_border: bool = False

    def __post_init__(self) -> None:
        valid_methods = ("otsu", "manual")
        if self.threshold_method not in valid_methods:
            raise InvalidSegmentation3DConfigError(
                f"threshold_method must be one of {valid_methods}, got '{self.threshold_method}'."
            )

        if self.threshold_method == "manual":
            if (
                self.manual_threshold is None
                or isinstance(self.manual_threshold, bool)
                or not isinstance(self.manual_threshold, (int, float))
                or not np.isfinite(self.manual_threshold)
            ):
                raise InvalidSegmentation3DConfigError(
                    "manual_threshold must be a numeric finite value when threshold_method is 'manual'."
                )


        if (
            not isinstance(self.gaussian_sigma_xy_px, (int, float))
            or isinstance(self.gaussian_sigma_xy_px, bool)
            or self.gaussian_sigma_xy_px < 0
        ):
            raise InvalidSegmentation3DConfigError(
                f"gaussian_sigma_xy_px must be a non-negative float, got {self.gaussian_sigma_xy_px}."
            )

        if self.gaussian_sigma_z_slices is not None:
            if (
                not isinstance(self.gaussian_sigma_z_slices, (int, float))
                or isinstance(self.gaussian_sigma_z_slices, bool)
                or self.gaussian_sigma_z_slices < 0
            ):
                raise InvalidSegmentation3DConfigError(
                    f"gaussian_sigma_z_slices must be a non-negative float or None, got {self.gaussian_sigma_z_slices}."
                )

        if (
            not isinstance(self.min_volume_voxels, int)
            or isinstance(self.min_volume_voxels, bool)
            or self.min_volume_voxels < 0
        ):
            raise InvalidSegmentation3DConfigError(
                f"min_volume_voxels must be a non-negative integer, got {self.min_volume_voxels}."
            )

        if self.max_volume_voxels is not None:
            if (
                not isinstance(self.max_volume_voxels, int)
                or isinstance(self.max_volume_voxels, bool)
                or self.max_volume_voxels < 0
            ):
                raise InvalidSegmentation3DConfigError(
                    f"max_volume_voxels must be a non-negative integer or None, got {self.max_volume_voxels}."
                )
            if self.max_volume_voxels < self.min_volume_voxels:
                raise InvalidSegmentation3DConfigError(
                    f"max_volume_voxels ({self.max_volume_voxels}) cannot be smaller than min_volume_voxels ({self.min_volume_voxels})."
                )

        if (
            not isinstance(self.min_distance_xy_px, int)
            or isinstance(self.min_distance_xy_px, bool)
            or self.min_distance_xy_px < 1
        ):
            raise InvalidSegmentation3DConfigError(
                f"min_distance_xy_px must be a positive integer, got {self.min_distance_xy_px}."
            )

        if (
            not isinstance(self.min_distance_z_slices, int)
            or isinstance(self.min_distance_z_slices, bool)
            or self.min_distance_z_slices < 1
        ):
            raise InvalidSegmentation3DConfigError(
                f"min_distance_z_slices must be a positive integer, got {self.min_distance_z_slices}."
            )

        if not isinstance(self.fill_holes, bool):
            raise InvalidSegmentation3DConfigError("fill_holes must be a boolean.")
        if not isinstance(self.exclude_z_border, bool):
            raise InvalidSegmentation3DConfigError("exclude_z_border must be a boolean.")
        if not isinstance(self.exclude_xy_border, bool):
            raise InvalidSegmentation3DConfigError("exclude_xy_border must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "threshold_method": self.threshold_method,
            "manual_threshold": self.manual_threshold,
            "gaussian_sigma_xy_px": self.gaussian_sigma_xy_px,
            "gaussian_sigma_z_slices": self.gaussian_sigma_z_slices,
            "min_volume_voxels": self.min_volume_voxels,
            "max_volume_voxels": self.max_volume_voxels,
            "min_distance_xy_px": self.min_distance_xy_px,
            "min_distance_z_slices": self.min_distance_z_slices,
            "fill_holes": self.fill_holes,
            "exclude_z_border": self.exclude_z_border,
            "exclude_xy_border": self.exclude_xy_border,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segmentation3DConfig:
        """Reconstruct configuration from dictionary."""
        if not isinstance(data, dict):
            raise InvalidSegmentation3DConfigError("Segmentation3DConfig data must be a dictionary.")
        return cls(
            threshold_method=data.get("threshold_method", "otsu"),
            manual_threshold=data.get("manual_threshold"),
            gaussian_sigma_xy_px=float(data.get("gaussian_sigma_xy_px", 1.0)),
            gaussian_sigma_z_slices=(
                float(data["gaussian_sigma_z_slices"])
                if data.get("gaussian_sigma_z_slices") is not None
                else None
            ),
            min_volume_voxels=int(data.get("min_volume_voxels", 100)),
            max_volume_voxels=(
                int(data["max_volume_voxels"])
                if data.get("max_volume_voxels") is not None
                else None
            ),
            min_distance_xy_px=int(data.get("min_distance_xy_px", 5)),
            min_distance_z_slices=int(data.get("min_distance_z_slices", 2)),
            fill_holes=bool(data.get("fill_holes", True)),
            exclude_z_border=bool(data.get("exclude_z_border", False)),
            exclude_xy_border=bool(data.get("exclude_xy_border", False)),
        )


@dataclass(frozen=True)
class Segmentation3DResult:
    """Container for 3D nuclear segmentation outputs.

    Attributes:
        labels: 3D integer NumPy array of shape (Z, Y, X), where 0 represents
            background and 1..N represent segmented nuclei.
        objects: Pandas DataFrame containing one row per retained nucleus with
            volumetric measurements, bounding boxes, centroids, and QC flags.
        config: Effective 3D segmentation configuration used.
        source_shape: Shape of original 3D volume (Z, Y, X).
        source_id: Authoritative identifier of the source volume execution.
        spacing: Calibrated physical voxel dimensions if known, else None.
    """

    labels: np.ndarray
    objects: pd.DataFrame
    config: Segmentation3DConfig
    source_shape: tuple[int, int, int]
    source_id: str
    spacing: VoxelSpacing | None = None

    @property
    def nuclei(self) -> pd.DataFrame:
        """Alias for objects DataFrame matching 2D segmentation result ergonomics."""
        return self.objects

    @property
    def nucleus_count(self) -> int:
        """Total number of segmented 3D nuclei."""
        return len(self.objects)
