"""Exceptions for group-level descriptive statistics and statistical preparation."""

from __future__ import annotations


class NerveNucStatisticsError(Exception):
    """Base exception for all NerveNuc statistics errors."""


class StatisticsConsistencyError(NerveNucStatisticsError):
    """Raised when statistical inputs are structurally, numerically, or logically inconsistent."""
