"""Exceptions for analysis handoff and readiness reporting."""

from __future__ import annotations


class NerveNucHandoffError(Exception):
    """Base exception for all NerveNuc handoff errors."""


class HandoffConsistencyError(NerveNucHandoffError):
    """Raised when input statistics, observations, assignments, or export structures are inconsistent."""
