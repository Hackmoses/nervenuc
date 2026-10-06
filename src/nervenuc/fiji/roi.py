"""ImageJ ROI interoperability and authoritative rasterization."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import roifile
from scipy.ndimage import binary_fill_holes, label as nd_label
import skimage.draw as draw
import skimage.measure as measure

from .errors import InvalidRoiGeometryError, NerveNucFijiError, UnsupportedRoiError

SUPPORTED_ROI_TYPES = frozenset({
    roifile.ROI_TYPE.POLYGON,
    roifile.ROI_TYPE.RECT,
    roifile.ROI_TYPE.OVAL,
    roifile.ROI_TYPE.FREEHAND,
    roifile.ROI_TYPE.TRACED,
})


def validate_roi_type(roi: roifile.ImagejRoi) -> None:
    """Validate that an ImageJ ROI type is supported for nuclear QC interoperability.

    Raises:
        UnsupportedRoiError: If the ROI type, composite structure, or text format is unsupported.
    """
    if not isinstance(roi, roifile.ImagejRoi):
        raise UnsupportedRoiError(f"Expected roifile.ImagejRoi object, got {type(roi).__name__}.")

    if getattr(roi, "composite", False) or getattr(roi, "shape_roi_size", 0) > 0:
        raise UnsupportedRoiError(
            "Composite / multi-shape ImageJ ROIs are not supported for nuclear QC."
        )

    if bool(getattr(roi, "text", "")) or getattr(roi, "subtype", None) == roifile.ROI_SUBTYPE.TEXT:
        raise UnsupportedRoiError("Text ImageJ ROIs are not supported for nuclear QC.")


    if roi.roitype not in SUPPORTED_ROI_TYPES:
        type_name = getattr(roi.roitype, "name", str(roi.roitype))
        raise UnsupportedRoiError(
            f"ImageJ ROI type '{type_name}' is not supported. Supported types: "
            "POLYGON, FREEHAND, TRACED, RECT, OVAL."
        )


def read_roi(path: Path | str) -> roifile.ImagejRoi:
    """Read a single ImageJ .roi file.

    Raises:
        NerveNucFijiError: If the file does not exist, cannot be parsed, or is an ROI set ZIP.
    """
    p = Path(path)
    if not p.is_file():
        raise NerveNucFijiError(f"ROI file not found: {p}")

    try:
        res = roifile.roiread(str(p))
    except Exception as err:
        raise NerveNucFijiError(f"Failed to parse ImageJ ROI file '{p}': {err}") from err

    if isinstance(res, list):
        if len(res) == 1:
            return res[0]
        raise NerveNucFijiError(
            f"File '{p}' contains multiple ROIs ({len(res)}). Use read_roi_set() instead."
        )
    return res


def read_roi_set(path: Path | str) -> list[roifile.ImagejRoi]:
    """Read an ImageJ ROI set from a .zip or .roi file.

    Raises:
        NerveNucFijiError: If the file does not exist or cannot be parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise NerveNucFijiError(f"ROI set file not found: {p}")

    try:
        res = roifile.roiread(str(p))
    except Exception as err:
        raise NerveNucFijiError(f"Failed to parse ImageJ ROI set file '{p}': {err}") from err

    if isinstance(res, list):
        return res
    return [res]


def write_roi(roi: roifile.ImagejRoi, path: Path | str, overwrite: bool = False) -> Path:
    """Write a single ImageJ ROI to a .roi file.

    Raises:
        NerveNucFijiError: If file exists and overwrite is False, or writing fails.
    """
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise NerveNucFijiError(f"Target ROI file '{p}' already exists and overwrite is False.")
        if p.is_file():
            p.unlink()

    validate_roi_type(roi)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        roifile.roiwrite(str(p), roi)
    except Exception as err:
        raise NerveNucFijiError(f"Failed to write ImageJ ROI to '{p}': {err}") from err
    return p


def write_roi_set(rois: Sequence[roifile.ImagejRoi], path: Path | str, overwrite: bool = False) -> Path:
    """Write a collection of ImageJ ROIs to a .zip file.

    Raises:
        NerveNucFijiError: If list is empty, target exists and overwrite is False, or writing fails.
    """
    p = Path(path)
    if not rois:
        raise NerveNucFijiError("Cannot write empty ROI collection.")
    if p.exists():
        if not overwrite:
            raise NerveNucFijiError(f"Target ROI set file '{p}' already exists and overwrite is False.")
        if p.is_file():
            p.unlink()

    for r in rois:
        validate_roi_type(r)

    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        roifile.roiwrite(str(p), list(rois))
    except Exception as err:
        raise NerveNucFijiError(f"Failed to write ImageJ ROI set to '{p}': {err}") from err
    return p


def rasterize_roi(roi: roifile.ImagejRoi, shape: tuple[int, int]) -> np.ndarray:
    """Authoritatively rasterize an ImageJ ROI into a 2D boolean NumPy mask of exact shape (H, W).

    Coordinate convention:
        - ImageJ X (horizontal) maps to NumPy column (axis 1).
        - ImageJ Y (vertical) maps to NumPy row (axis 0).

    Args:
        roi: Validated roifile.ImagejRoi instance.
        shape: 2D target dimensions (H, W) as positive integers.

    Returns:
        2D boolean NumPy array matching exact target shape (H, W).

    Raises:
        UnsupportedRoiError: If the ROI type is not supported.
        InvalidRoiGeometryError: If coordinates are out-of-bounds, negative, non-finite, or unrasterizable.
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise InvalidRoiGeometryError(f"Target shape must be a 2-tuple (H, W), got {shape}.")
    h, w = shape
    if h <= 0 or w <= 0:
        raise InvalidRoiGeometryError(f"Target shape dimensions must be positive, got ({h}, {w}).")

    validate_roi_type(roi)
    mask = np.zeros((h, w), dtype=bool)

    if roi.roitype == roifile.ROI_TYPE.RECT:
        top = float(roi.yd if getattr(roi, "subpixelrect", False) and getattr(roi, "yd", None) is not None else roi.top)
        left = float(roi.xd if getattr(roi, "subpixelrect", False) and getattr(roi, "xd", None) is not None else roi.left)
        bottom = float(top + roi.heightd if getattr(roi, "subpixelrect", False) and getattr(roi, "heightd", None) is not None else roi.bottom)
        right = float(left + roi.widthd if getattr(roi, "subpixelrect", False) and getattr(roi, "widthd", None) is not None else roi.right)

        if top < 0 or left < 0 or bottom > h or right > w:
            raise InvalidRoiGeometryError(
                f"Rectangle ROI bounds ({top}, {left}, {bottom}, {right}) exceed image shape ({h}, {w})."
            )
        if bottom <= top or right <= left:
            raise InvalidRoiGeometryError(
                f"Rectangle ROI has zero or negative area: top={top}, bottom={bottom}, left={left}, right={right}."
            )

        r_start, r_end = int(round(top)), int(round(bottom))
        c_start, c_end = int(round(left)), int(round(right))
        if r_end <= r_start or c_end <= c_start:
            raise InvalidRoiGeometryError(
                f"Subpixel rectangle rounded to zero pixels: rows [{r_start}, {r_end}), cols [{c_start}, {c_end})."
            )
        mask[r_start:r_end, c_start:c_end] = True
        return mask

    if roi.roitype == roifile.ROI_TYPE.OVAL:
        top = float(roi.yd if getattr(roi, "subpixelrect", False) and getattr(roi, "yd", None) is not None else roi.top)
        left = float(roi.xd if getattr(roi, "subpixelrect", False) and getattr(roi, "xd", None) is not None else roi.left)
        bottom = float(top + roi.heightd if getattr(roi, "subpixelrect", False) and getattr(roi, "heightd", None) is not None else roi.bottom)
        right = float(left + roi.widthd if getattr(roi, "subpixelrect", False) and getattr(roi, "widthd", None) is not None else roi.right)

        if top < 0 or left < 0 or bottom > h or right > w:
            raise InvalidRoiGeometryError(
                f"Oval ROI bounds ({top}, {left}, {bottom}, {right}) exceed image shape ({h}, {w})."
            )
        if bottom <= top or right <= left:
            raise InvalidRoiGeometryError(
                f"Oval ROI has zero or negative area: top={top}, bottom={bottom}, left={left}, right={right}."
            )

        r_center = (top + bottom - 1.0) / 2.0
        c_center = (left + right - 1.0) / 2.0
        r_rad = (bottom - top) / 2.0
        c_rad = (right - left) / 2.0

        rr, cc = draw.ellipse(r_center, c_center, r_rad, c_rad, shape=(h, w))
        if len(rr) == 0:
            raise InvalidRoiGeometryError("Oval ROI has zero rasterized pixels.")
        mask[rr, cc] = True
        return mask

    # Polygon / Freehand / Traced
    coords = roi.coordinates()
    if coords is None or len(coords) < 3:
        raise InvalidRoiGeometryError("Polygon ROI must contain at least 3 vertices.")
    if not np.all(np.isfinite(coords)):
        raise InvalidRoiGeometryError("Polygon ROI coordinates contain non-finite values.")

    # ImageJ coordinates are [x, y] -> [col, row]
    cols = coords[:, 0]
    rows = coords[:, 1]

    if np.any(cols < 0) or np.any(cols > w) or np.any(rows < 0) or np.any(rows > h):
        raise InvalidRoiGeometryError(
            f"Polygon ROI vertices exceed image shape ({h}, {w}): "
            f"x_min={np.min(cols)}, x_max={np.max(cols)}, y_min={np.min(rows)}, y_max={np.max(rows)}."
        )

    rr, cc = draw.polygon(rows, cols, shape=(h, w))
    if len(rr) == 0:
        raise InvalidRoiGeometryError("Polygon ROI has zero rasterized pixels.")
    mask[rr, cc] = True
    return mask


def mask_to_roi(mask: np.ndarray, name: str | None = None) -> roifile.ImagejRoi:
    """Convert a single 2D connected boolean nucleus mask into an ImageJ Polygon ROI.

    Contract & Limitations:
        - Must be exactly 2-dimensional.
        - Must be non-empty (at least one True pixel).
        - Must contain exactly one connected object (multiple disconnected components are rejected).
        - Must not contain internal holes (enclosed topological holes are rejected).

    Args:
        mask: 2D boolean or binary integer NumPy array.
        name: Optional identifier name to assign to the ROI (e.g. 'AUTO-000001').

    Returns:
        roifile.ImagejRoi of type POLYGON.

    Raises:
        InvalidRoiGeometryError: If the mask is empty, multi-component, or contains holes.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise InvalidRoiGeometryError(f"Mask must be a 2D numpy array, got {getattr(mask, 'ndim', None)}D.")

    bool_mask = mask.astype(bool)
    num_pixels = np.count_nonzero(bool_mask)
    if num_pixels == 0:
        raise InvalidRoiGeometryError("Cannot convert empty mask (0 positive pixels) to ImageJ ROI.")

    # Check connectivity
    labeled, num_features = nd_label(bool_mask)
    if num_features > 1:
        raise InvalidRoiGeometryError(
            f"Mask contains {num_features} disconnected objects. Expected a single connected nucleus."
        )

    # Check holes
    filled = binary_fill_holes(bool_mask)
    if np.count_nonzero(filled) != num_pixels:
        raise InvalidRoiGeometryError(
            "Mask contains internal holes. Multi-component/holed topologies cannot be converted to a single polygon ROI."
        )

    # Pad by 1 pixel so border-touching contours are extracted smoothly
    padded = np.pad(bool_mask, 1, mode="constant", constant_values=False)
    contours = measure.find_contours(padded.astype(float), 0.5)
    if not contours:
        raise InvalidRoiGeometryError("Failed to extract contour boundary from mask.")

    # Take the primary contour and subtract the 1-pixel padding
    c = contours[0] - 1.0

    # find_contours returns (row, col) -> convert to (col, row) = [x, y]
    x_coords = np.clip(c[:, 1], 0.0, float(mask.shape[1]))
    y_coords = np.clip(c[:, 0], 0.0, float(mask.shape[0]))
    pts = np.column_stack([x_coords, y_coords])

    roi = roifile.ImagejRoi.frompoints(pts)
    roi.roitype = roifile.ROI_TYPE.POLYGON
    if name is not None:
        roi.name = str(name).strip()

    return roi
