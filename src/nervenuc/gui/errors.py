"""Errors and exceptions for NerveNuc GUI module."""

from __future__ import annotations


class NerveNucGuiError(Exception):
    """Base exception for all GUI errors."""


class MissingGuiDependencyError(NerveNucGuiError):
    """Raised when GUI is invoked without required optional dependencies (e.g. PySide6)."""


class InvalidDraftConfigError(NerveNucGuiError):
    """Raised when UI draft values cannot be converted to valid core configurations."""
