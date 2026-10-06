"""Group-level descriptive statistics and statistical preparation."""

from .engine import (
    prepare_analysis_observations,
    summarize_groups,
    validate_statistics_input,
)
from .errors import (
    NerveNucStatisticsError,
    StatisticsConsistencyError,
)
from .models import GroupStatisticsResult

__all__ = [
    "GroupStatisticsResult",
    "NerveNucStatisticsError",
    "StatisticsConsistencyError",
    "prepare_analysis_observations",
    "summarize_groups",
    "validate_statistics_input",
]
