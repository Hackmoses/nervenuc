"""Overlay rendering for segmentation outlines, QC states, and classification glyphs."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from PySide6.QtGui import QColor, QImage
from scipy.ndimage import binary_dilation


def render_label_overlay(
    labels_2d: np.ndarray,
    active_labels: set[int] | None = None,
    rejected_labels: set[int] | None = None,
    outline_only: bool = True,
    opacity: float = 0.5,
) -> QImage:
    """Render 2D segmentation label overlays with RGBA coloring."""
    if labels_2d is None or labels_2d.ndim != 2 or np.all(labels_2d == 0):
        return QImage()

    h, w = labels_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    unique_labels = np.unique(labels_2d)

    for lab in unique_labels:
        if lab == 0:
            continue

        mask = (labels_2d == lab)

        if outline_only:
            dilated = binary_dilation(mask)
            boundary = dilated & ~mask
            render_mask = boundary
        else:
            render_mask = mask

        if rejected_labels is not None and lab in rejected_labels:
            # Red for rejected
            r, g, b = 255, 50, 50
        elif active_labels is not None and lab in active_labels:
            # Green for accepted/active
            r, g, b = 50, 255, 50
        else:
            # Yellow for unreviewed
            r, g, b = 255, 255, 0

        alpha = int(opacity * 255)
        rgba[render_mask, 0] = r
        rgba[render_mask, 1] = g
        rgba[render_mask, 2] = b
        rgba[render_mask, 3] = alpha

    rgba_bytes = np.ascontiguousarray(rgba).tobytes()
    qimg = QImage(rgba_bytes, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return qimg.copy()
