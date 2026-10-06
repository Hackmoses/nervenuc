"""Models for group-level descriptive statistics and statistical preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from nervenuc.export import GroupTableResult


@dataclass
class GroupStatisticsResult:
    """Container for group descriptive statistics and normalized animal observations.

    Attributes:
        group_statistics: 15-column DataFrame containing sample descriptive statistics per group x metric.
        observations: 6-column DataFrame containing individual animal observations and raw count provenance.
        source: Provenance reference to the input GroupTableResult.
    """

    group_statistics: pd.DataFrame
    observations: pd.DataFrame
    source: GroupTableResult
