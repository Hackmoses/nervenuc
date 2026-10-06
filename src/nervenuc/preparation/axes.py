"""Axis resolution and subvolume extraction for microscopy Z-stacks."""

from __future__ import annotations

import numpy as np

from .errors import (
    AmbiguousImageAxesError,
    InvalidAxisLayoutError,
    InvalidChannelSelectionError,
    InvalidZSelectionError,
    NerveNucPreparationError,
)
from .models import AxisLayout, ZSelection


def resolve_axes(
    shape: tuple[int, ...],
    declared_axes: str | None = None,
    metadata_axes: str | None = None,
    is_ome: bool = False,
) -> AxisLayout:
    """Resolve and validate the canonical AxisLayout for an image array.

    Args:
        shape: Tuple of dimension lengths.
        declared_axes: Caller-supplied axis string (e.g. 'CZYX').
        metadata_axes: Axis string from file metadata.
        is_ome: Whether the source file is an OME-TIFF with structured metadata.

    Returns:
        Validated AxisLayout instance.

    Raises:
        InvalidAxisLayoutError: If axis string is invalid or conflicts with authoritative OME metadata.
        AmbiguousImageAxesError: If stack dimensionality requires explicit axis declaration.
    """
    ndim = len(shape)
    if ndim == 2:
        if declared_axes is not None:
            return AxisLayout(declared_axes, shape)
        return AxisLayout("YX", shape)

    # If caller provided explicit declaration
    if declared_axes is not None:
        norm_decl = declared_axes.strip().upper()
        if is_ome and metadata_axes is not None:
            norm_meta = metadata_axes.strip().upper()
            if norm_decl != norm_meta:
                raise InvalidAxisLayoutError(
                    f"Declared axes '{declared_axes}' conflict with authoritative OME metadata axes '{metadata_axes}'."
                )
        return AxisLayout(declared_axes, shape)

    # If authoritative OME metadata is available
    if is_ome and metadata_axes is not None:
        try:
            return AxisLayout(metadata_axes, shape)
        except InvalidAxisLayoutError as exc:
            raise InvalidAxisLayoutError(
                f"OME metadata axes '{metadata_axes}' are invalid for shape {shape}: {exc}"
            ) from exc

    # Plain TIFF with ndim >= 3 requires explicit declaration
    raise AmbiguousImageAxesError(
        f"Image has shape {shape} (ndim={ndim}) without authoritative OME metadata axes. "
        "Explicit axes declaration is required (e.g. axes='CZYX' or axes='ZYX')."
    )


def extract_subvolume(
    array: np.ndarray,
    layout: AxisLayout,
    time_index: int | None = None,
    channel_index: int | None = None,
    z_selection: ZSelection | None = None,
) -> tuple[np.ndarray, int, int]:
    """Extract a 3D (Z, Y, X) subvolume along specified time, channel, and Z-slice coordinates.

    Args:
        array: Multi-dimensional NumPy array matching layout.shape.
        layout: Validated AxisLayout.
        time_index: Timepoint index to extract if T axis is present.
        channel_index: Channel index to extract if C axis is present.
        z_selection: ZSelection range to slice, or None for all slices.

    Returns:
        Tuple of (subvolume_3d, z_start, z_stop) where subvolume_3d has shape (Z, Y, X).

    Raises:
        AmbiguousImageAxesError: If T > 1 and time_index is None.
        InvalidChannelSelectionError: If channel_index is invalid or out of bounds.
        InvalidZSelectionError: If z_selection exceeds stack depth.
        InvalidAxisLayoutError: If Z axis is missing from layout.
    """
    if layout.index_of("Z") is None:
        raise InvalidAxisLayoutError(
            f"Cannot extract Z-stack subvolume from layout '{layout.axes}' (missing 'Z' axis)."
        )

    slices: list[slice | int] = [slice(None)] * len(layout.axes)

    # 1. Handle T axis
    t_idx = layout.index_of("T")
    if t_idx is not None:
        size_t = layout.shape[t_idx]
        if size_t > 1 and time_index is None:
            raise AmbiguousImageAxesError(
                f"Stack contains time axis 'T' of size {size_t}, but time_index is None. "
                "Explicit time index is required."
            )
        eff_t = 0 if time_index is None else time_index
        if eff_t < 0 or eff_t >= size_t:
            raise NerveNucPreparationError(
                f"time_index {eff_t} is out of bounds for time dimension of size {size_t}."
            )
        slices[t_idx] = eff_t

    # 2. Handle C axis
    c_idx = layout.index_of("C")
    if c_idx is not None:
        size_c = layout.shape[c_idx]
        if channel_index is None:
            raise InvalidChannelSelectionError(
                f"Stack contains channel axis 'C' of size {size_c}, but channel_index is None."
            )
        if channel_index < 0 or channel_index >= size_c:
            raise InvalidChannelSelectionError(
                f"channel_index {channel_index} is out of bounds for channel dimension of size {size_c}."
            )
        slices[c_idx] = channel_index
    else:
        if channel_index is not None and channel_index != 0:
            raise InvalidChannelSelectionError(
                f"Stack has no 'C' axis; channel_index must be 0, got {channel_index}."
            )

    # 3. Handle Z axis
    z_axis_pos = layout.index_of("Z")
    assert z_axis_pos is not None
    size_z = layout.shape[z_axis_pos]
    if z_selection is None:
        z_start = 0
        z_stop = size_z
    else:
        z_start = z_selection.start
        z_stop = z_selection.stop
        if z_stop > size_z:
            raise InvalidZSelectionError(
                f"ZSelection stop {z_stop} exceeds stack depth {size_z}."
            )

    slices[z_axis_pos] = slice(z_start, z_stop)

    # Apply slicing
    sliced = array[tuple(slices)]

    # Sliced array now has axes from layout that were slices with slice(None) or slice(start, stop)
    # Determine remaining axis names
    remaining_axes = [
        layout.axes[i]
        for i, s in enumerate(slices)
        if isinstance(s, slice)
    ]

    # Reorder remaining axes to standard "ZYX"
    curr_z = remaining_axes.index("Z")
    curr_y = remaining_axes.index("Y")
    curr_x = remaining_axes.index("X")

    # Transpose to (Z, Y, X)
    subvolume_3d = np.transpose(sliced, (curr_z, curr_y, curr_x))
    return subvolume_3d, z_start, z_stop
