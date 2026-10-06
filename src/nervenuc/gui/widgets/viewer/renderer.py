"""Display normalization and color composition (strictly presentation only)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from PySide6.QtGui import QImage

COLOR_MAPS = {
    "Gray": (255, 255, 255),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Yellow": (255, 255, 0),
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
}


def render_display_slice(
    channels: Mapping[int, np.ndarray],
    min_vals: Mapping[int, float],
    max_vals: Mapping[int, float],
    gammas: Mapping[int, float],
    visibilities: Mapping[int, bool],
    colors: Mapping[int, str],
) -> QImage:
    """Compose multiple 2D channels into a single 8-bit RGB QImage for display.

    Quantitative NumPy arrays are never mutated.
    """
    if not channels:
        return QImage()

    # Determine reference canvas shape
    first_ch = next(iter(channels.values()))
    h, w = first_ch.shape[-2], first_ch.shape[-1]

    rgb_acc = np.zeros((h, w, 3), dtype=np.float32)

    for ch_idx, raw_plane in channels.items():
        if not visibilities.get(ch_idx, True):
            continue

        plane = raw_plane.astype(np.float32)
        c_min = min_vals.get(ch_idx, float(np.min(plane)))
        c_max = max_vals.get(ch_idx, float(np.max(plane)))
        gamma = gammas.get(ch_idx, 1.0)
        c_name = colors.get(ch_idx, "Gray")
        rgb_weights = COLOR_MAPS.get(c_name, (255, 255, 255))

        if c_max > c_min:
            norm = np.clip((plane - c_min) / (c_max - c_min), 0.0, 1.0)
        else:
            norm = np.zeros_like(plane)

        if gamma != 1.0 and gamma > 0:
            norm = np.power(norm, 1.0 / gamma)

        for i in range(3):
            rgb_acc[:, :, i] += norm * (rgb_weights[i] / 255.0)

    rgb_acc = np.clip(rgb_acc * 255.0, 0.0, 255.0).astype(np.uint8)

    # Pack into QImage
    rgb_bytes = np.ascontiguousarray(rgb_acc).tobytes()
    qimg = QImage(rgb_bytes, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg.copy()
