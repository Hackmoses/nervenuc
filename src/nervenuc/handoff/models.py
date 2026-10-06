"""Models for analysis handoff and readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from nervenuc.statistics import GroupStatisticsResult


@dataclass
class AnalysisHandoffResult:
    """Container for the analysis-readiness handoff report and underlying data provenance.

    Attributes:
        metric_readiness: 14-column DataFrame containing metric-level readiness facts.
        group_readiness: 5-column DataFrame containing sample size accounting per metric x group.
        source: Provenance reference to input GroupStatisticsResult.
    """

    metric_readiness: pd.DataFrame
    group_readiness: pd.DataFrame
    source: GroupStatisticsResult
