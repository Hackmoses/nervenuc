"""Microscopy TIFF/OME-TIFF loading and metadata inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import tifffile

from .errors import NerveNucImageError, UnreadableImageError, UnsupportedImageError


@dataclass(frozen=True)
class ImageMetadata:
    """Normalized metadata for a microscopy TIFF or OME-TIFF image."""

    path: Path
    file_name: str
    is_ome: bool

    shape: tuple[int, ...]
    dtype: str
    ndim: int

    axes: str | None = None

    size_x: int | None = None
    size_y: int | None = None
    size_z: int | None = None
    size_c: int | None = None
    size_t: int | None = None

    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    pixel_size_z_um: float | None = None

    channel_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LoadedImage:
    """Container for loaded pixel data and associated metadata."""

    data: np.ndarray
    metadata: ImageMetadata


def _local_xml_tag(elem: ET.Element) -> str:
    """Return local XML tag name without namespace."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalize_unit_to_um(value: float | None, unit_str: str | None) -> float | None:
    """Convert physical size to micrometers (µm) if unit is known, else None."""
    if value is None:
        return None
    if unit_str is None or unit_str.strip() == "":
        # OME schema standard default unit is micrometer
        return float(value)

    unit = unit_str.strip().lower()
    if unit in {"µm", "μm", "um", "micron", "microns", "micrometer", "micrometers", "µmeter"}:
        return float(value)
    elif unit in {"nm", "nanometer", "nanometers"}:
        return float(value) * 1e-3
    elif unit in {"mm", "millimeter", "millimeters"}:
        return float(value) * 1e3
    elif unit in {"m", "meter", "meters"}:
        return float(value) * 1e6
    elif unit in {"pm", "picometer", "picometers"}:
        return float(value) * 1e-6
    else:
        return None


def _parse_ome_xml(xml_content: str) -> dict[str, Any]:
    """Parse OME-XML string and extract dimensions, calibration, and channel names."""
    result: dict[str, Any] = {
        "physical_size_x_um": None,
        "physical_size_y_um": None,
        "physical_size_z_um": None,
        "channel_names": None,
    }

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return result

    # Find the first Pixels element
    pixels_elem: ET.Element | None = None
    for elem in root.iter():
        if _local_xml_tag(elem) == "Pixels":
            pixels_elem = elem
            break

    if pixels_elem is None:
        return result

    # Physical sizes
    ps_x_str = pixels_elem.get("PhysicalSizeX")
    ps_x_unit = pixels_elem.get("PhysicalSizeXUnit")
    if ps_x_str is not None:
        try:
            result["physical_size_x_um"] = _normalize_unit_to_um(float(ps_x_str), ps_x_unit)
        except ValueError:
            pass

    ps_y_str = pixels_elem.get("PhysicalSizeY")
    ps_y_unit = pixels_elem.get("PhysicalSizeYUnit")
    if ps_y_str is not None:
        try:
            result["physical_size_y_um"] = _normalize_unit_to_um(float(ps_y_str), ps_y_unit)
        except ValueError:
            pass

    ps_z_str = pixels_elem.get("PhysicalSizeZ")
    ps_z_unit = pixels_elem.get("PhysicalSizeZUnit")
    if ps_z_str is not None:
        try:
            result["physical_size_z_um"] = _normalize_unit_to_um(float(ps_z_str), ps_z_unit)
        except ValueError:
            pass

    # Channel names
    channel_names_list: list[str] = []
    has_named_channels = False
    for child in pixels_elem:
        if _local_xml_tag(child) == "Channel":
            name = child.get("Name")
            if name:
                channel_names_list.append(name)
                has_named_channels = True
            else:
                channel_names_list.append("")

    if has_named_channels and channel_names_list:
        result["channel_names"] = tuple(channel_names_list)

    return result


def _metadata_from_open_tiff(tif: tifffile.TiffFile, path_obj: Path) -> ImageMetadata:
    """Build normalized metadata from an already-open, unambiguous TIFF."""
    if not tif.series:
        raise UnreadableImageError(f"No image series found in TIFF file '{path_obj}'")
    if len(tif.series) != 1:
        raise UnsupportedImageError(
            f"TIFF file '{path_obj}' contains {len(tif.series)} image series; "
            "explicit series selection is not supported."
        )

    series = tif.series[0]
    shape = tuple(int(s) for s in series.shape)
    dtype_str = str(np.dtype(series.dtype).name)
    ndim = len(shape)
    axes_str = str(series.axes) if series.axes else None
    is_ome = bool(tif.is_ome)

    size_x: int | None = None
    size_y: int | None = None
    size_z: int | None = None
    size_c: int | None = None
    size_t: int | None = None

    if axes_str is not None:
        if "X" in axes_str:
            size_x = shape[axes_str.index("X")]
        if "Y" in axes_str:
            size_y = shape[axes_str.index("Y")]
        if "Z" in axes_str:
            size_z = shape[axes_str.index("Z")]
        if "C" in axes_str:
            size_c = shape[axes_str.index("C")]
        if "T" in axes_str:
            size_t = shape[axes_str.index("T")]
    elif ndim == 2:
        size_y = shape[0]
        size_x = shape[1]

    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    pixel_size_z_um: float | None = None
    channel_names: tuple[str, ...] | None = None

    if is_ome and tif.ome_metadata:
        ome_info = _parse_ome_xml(tif.ome_metadata)
        pixel_size_x_um = ome_info["physical_size_x_um"]
        pixel_size_y_um = ome_info["physical_size_y_um"]
        pixel_size_z_um = ome_info["physical_size_z_um"]
        channel_names = ome_info["channel_names"]

    return ImageMetadata(
        path=path_obj,
        file_name=path_obj.name,
        is_ome=is_ome,
        shape=shape,
        dtype=dtype_str,
        ndim=ndim,
        axes=axes_str,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        pixel_size_z_um=pixel_size_z_um,
        channel_names=channel_names,
    )


def inspect_tiff(path: str | Path) -> ImageMetadata:
    """Inspect microscopy TIFF/OME-TIFF metadata without decoding full image array.

    Parameters
    ----------
    path : str | Path
        Path to the TIFF or OME-TIFF file.

    Returns
    -------
    ImageMetadata
        Extracted normalized image metadata.

    Raises
    ------
    UnreadableImageError
        If the file does not exist, is not a regular file, or cannot be parsed.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise UnreadableImageError(f"Image file does not exist: '{path_obj}'")
    if not path_obj.is_file():
        raise UnreadableImageError(f"Path is not a regular file: '{path_obj}'")

    try:
        with tifffile.TiffFile(path_obj) as tif:
            return _metadata_from_open_tiff(tif, path_obj)
    except NerveNucImageError:
        raise
    except Exception as err:
        raise UnreadableImageError(f"Cannot read TIFF file '{path_obj}': {err}") from err


def load_tiff(path: str | Path) -> LoadedImage:
    """Load a microscopy TIFF or OME-TIFF image and its metadata.

    Parameters
    ----------
    path : str | Path
        Path to the TIFF or OME-TIFF file.

    Returns
    -------
    LoadedImage
        Container with untouched image array and metadata.

    Raises
    ------
    UnreadableImageError
        If the file does not exist, is not a regular file, or cannot be read.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise UnreadableImageError(f"Image file does not exist: '{path_obj}'")
    if not path_obj.is_file():
        raise UnreadableImageError(f"Path is not a regular file: '{path_obj}'")
    try:
        with tifffile.TiffFile(path_obj) as tif:
            metadata = _metadata_from_open_tiff(tif, path_obj)
            data = tif.series[0].asarray()
    except NerveNucImageError:
        raise
    except Exception as err:
        raise UnreadableImageError(f"Cannot read TIFF file '{path_obj}': {err}") from err

    return LoadedImage(data=data, metadata=metadata)


def format_metadata(metadata: ImageMetadata) -> str:
    """Return a concise, human-readable summary of image metadata.

    Parameters
    ----------
    metadata : ImageMetadata
        The metadata to format.

    Returns
    -------
    str
        Formatted multi-line string displaying only known values.
    """
    lines: list[str] = [
        f"File: {metadata.file_name}",
        f"OME-TIFF: {'yes' if metadata.is_ome else 'no'}",
        f"Shape: {metadata.shape}",
    ]

    if metadata.axes is not None:
        lines.append(f"Axes: {metadata.axes}")

    lines.append(f"Dtype: {metadata.dtype}")

    if metadata.size_c is not None:
        lines.append(f"Channels: {metadata.size_c}")

    if metadata.size_z is not None:
        lines.append(f"Z slices: {metadata.size_z}")

    if metadata.size_t is not None:
        lines.append(f"Timepoints: {metadata.size_t}")

    if metadata.size_x is not None and metadata.size_y is not None:
        lines.append(f"Size: {metadata.size_x} × {metadata.size_y} px")

    if metadata.pixel_size_x_um is not None and metadata.pixel_size_y_um is not None:
        lines.append(f"Pixel size XY: {metadata.pixel_size_x_um:.3f} × {metadata.pixel_size_y_um:.3f} µm")
    elif metadata.pixel_size_x_um is not None:
        lines.append(f"Pixel size X: {metadata.pixel_size_x_um:.3f} µm")
    elif metadata.pixel_size_y_um is not None:
        lines.append(f"Pixel size Y: {metadata.pixel_size_y_um:.3f} µm")

    if metadata.pixel_size_z_um is not None:
        lines.append(f"Z step: {metadata.pixel_size_z_um:.3f} µm")

    if metadata.channel_names is not None and len(metadata.channel_names) > 0:
        lines.append(f"Channel names: {', '.join(metadata.channel_names)}")

    return "\n".join(lines)
