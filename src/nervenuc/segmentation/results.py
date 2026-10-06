"""Results data models for nuclear segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    """Container for nuclear segmentation outputs.

    Attributes:
        labels: 2D integer NumPy array of shape (Y, X), where 0 represents
            background and 1..N represent segmented nuclei.
        objects: Pandas DataFrame containing one row per retained nucleus with
            morphological measurements and basic QC flags.
        config: Effective segmentation configuration used.
    """

    labels: np.ndarray
    objects: pd.DataFrame
    config: SegmentationConfig
