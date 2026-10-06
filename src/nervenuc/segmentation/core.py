"""Core classical nuclear segmentation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation

from .config import SegmentationConfig
from .errors import InvalidSegmentationInputError
from .results import SegmentationResult

OBJECT_COLUMNS: list[str] = [
    "nucleus_id",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "touches_border",
    "qc_status",
]


def _empty_objects_dataframe() -> pd.DataFrame:
    """Return an empty objects DataFrame with the required schema and dtypes."""
    return pd.DataFrame(
        {
            "nucleus_id": pd.Series(dtype="int64"),
            "area_px": pd.Series(dtype="int64"),
            "centroid_y": pd.Series(dtype="float64"),
            "centroid_x": pd.Series(dtype="float64"),
            "bbox_min_y": pd.Series(dtype="int64"),
            "bbox_min_x": pd.Series(dtype="int64"),
            "bbox_max_y": pd.Series(dtype="int64"),
            "bbox_max_x": pd.Series(dtype="int64"),
            "touches_border": pd.Series(dtype="bool"),
            "qc_status": pd.Series(dtype="object"),
        },
        columns=OBJECT_COLUMNS,
    )


def validate_segmentation_input(
    image: np.ndarray,
    config: SegmentationConfig | None = None,
) -> None:
    """Validate input array and configuration for nuclear segmentation.

    Args:
        image: Nuclear channel image to validate.
        config: Optional configuration to validate.

    Raises:
        InvalidSegmentationInputError: If the image or configuration is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidSegmentationInputError(
            f"Input image must be a numpy.ndarray, got {type(image).__name__}."
        )

    if image.ndim != 2:
        raise InvalidSegmentationInputError(
            f"Input image must be 2-dimensional (Y, X), got ndim={image.ndim} with shape {image.shape}."
        )

    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise InvalidSegmentationInputError("Input image cannot be empty.")

    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.complexfloating) or image.dtype == bool:
        raise InvalidSegmentationInputError(
            f"Input image must have a real numeric dtype, got '{image.dtype}'."
        )

    if not np.isfinite(image).all():
        raise InvalidSegmentationInputError(
            "Input image contains non-finite values (NaN or Inf)."
        )

    if config is not None and not isinstance(config, SegmentationConfig):
        raise InvalidSegmentationInputError(
            f"config must be an instance of SegmentationConfig, got {type(config).__name__}."
        )


def segment_nuclei(
    image: np.ndarray,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Segment individual nuclei in a 2D nuclear channel microscopy image.

    Args:
        image: 2D numeric NumPy array representing the nuclear anchor channel.
            The input array is never modified.
        config: Configuration parameters for segmentation. Defaults to default
            SegmentationConfig() if omitted.

    Returns:
        SegmentationResult containing the integer label image (0=background, 1..N=nuclei),
        the objects measurements DataFrame, and the effective configuration.

    Raises:
        InvalidSegmentationInputError: If the input array or configuration is invalid.
    """
    if config is None:
        config = SegmentationConfig()

    validate_segmentation_input(image, config)

    # Flat or uniform image check (zero contrast)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if min_val == max_val:
        return SegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            objects=_empty_objects_dataframe(),
            config=config,
        )

    # Convert to floating-point working copy while preserving original numeric intensity range
    working = image.astype(np.float64, copy=True)

    # Optional Gaussian smoothing
    if config.gaussian_sigma > 0:
        smoothed = skimage.filters.gaussian(
            working,
            sigma=config.gaussian_sigma,
            preserve_range=True,
        )
    else:
        smoothed = working

    # Foreground thresholding
    if config.threshold_method == "otsu":
        thresh = skimage.filters.threshold_otsu(smoothed)
    elif config.threshold_method == "li":
        thresh = skimage.filters.threshold_li(smoothed)
    elif config.threshold_method == "manual":
        thresh = config.manual_threshold
    else:
        raise InvalidSegmentationInputError(
            f"Unsupported threshold method: '{config.threshold_method}'."
        )

    mask = smoothed > thresh
    if not np.any(mask):
        return SegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            objects=_empty_objects_dataframe(),
            config=config,
        )

    # Morphology: fill enclosed holes in foreground
    if config.fill_holes:
        mask = scipy.ndimage.binary_fill_holes(mask)

    # Morphology: remove foreground objects smaller than min_area_px
    if config.min_area_px > 1:
        try:
            mask = skimage.morphology.remove_small_objects(
                mask,
                max_size=config.min_area_px - 1,
            )
        except TypeError:
            mask = skimage.morphology.remove_small_objects(
                mask,
                min_size=config.min_area_px,
            )
    elif config.min_area_px == 1:
        pass

    if not np.any(mask):
        return SegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            objects=_empty_objects_dataframe(),
            config=config,
        )

    # Euclidean distance transform
    distance = scipy.ndimage.distance_transform_edt(mask)

    # Seed / peak detection
    coords = skimage.feature.peak_local_max(
        distance,
        min_distance=config.min_distance_px,
        labels=mask,
    )

    if coords.size == 0:
        markers = skimage.measure.label(mask)
    else:
        seed_mask = np.zeros_like(mask, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers = skimage.measure.label(seed_mask)

    # Seeded watershed
    raw_labels = skimage.segmentation.watershed(
        -distance,
        markers=markers,
        mask=mask,
    )

    if not np.any(raw_labels > 0):
        return SegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            objects=_empty_objects_dataframe(),
            config=config,
        )

    # Object filtering (max_area_px and exclude_border)
    props = skimage.measure.regionprops(raw_labels)
    h, w = image.shape
    keep_labels: list[int] = []

    for prop in props:
        touches_border = (
            prop.bbox[0] == 0
            or prop.bbox[1] == 0
            or prop.bbox[2] == h
            or prop.bbox[3] == w
        )

        if config.exclude_border and touches_border:
            continue

        if config.max_area_px is not None and prop.area > config.max_area_px:
            continue

        keep_labels.append(prop.label)

    if not keep_labels:
        return SegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            objects=_empty_objects_dataframe(),
            config=config,
        )

    filtered_labels = np.where(np.isin(raw_labels, keep_labels), raw_labels, 0)

    # Sequential relabeling 1..N. A NumPy lookup preserves the sorted-label
    # behavior without relying on scikit-image's deprecated internal reshape.
    retained_labels = np.unique(filtered_labels)
    retained_labels = retained_labels[retained_labels != 0]
    label_map = np.zeros(int(retained_labels[-1]) + 1, dtype=np.int32)
    label_map[retained_labels] = np.arange(
        1,
        retained_labels.size + 1,
        dtype=np.int32,
    )
    final_labels = label_map[filtered_labels]

    # Construct object table
    retained_props = skimage.measure.regionprops(final_labels)
    rows: list[dict] = []
    for prop in retained_props:
        touches_border = bool(
            prop.bbox[0] == 0
            or prop.bbox[1] == 0
            or prop.bbox[2] == h
            or prop.bbox[3] == w
        )
        qc_status = "border" if touches_border else "accepted"
        rows.append(
            {
                "nucleus_id": int(prop.label),
                "area_px": int(prop.area),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "bbox_min_y": int(prop.bbox[0]),
                "bbox_min_x": int(prop.bbox[1]),
                "bbox_max_y": int(prop.bbox[2]),
                "bbox_max_x": int(prop.bbox[3]),
                "touches_border": touches_border,
                "qc_status": qc_status,
            }
        )

    objects_df = pd.DataFrame(rows, columns=OBJECT_COLUMNS)

    return SegmentationResult(
        labels=final_labels,
        objects=objects_df,
        config=config,
    )
