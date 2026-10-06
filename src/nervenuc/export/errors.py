"""Exceptions for animal-level group tables and export preparation."""

from __future__ import annotations


class NerveNucExportError(Exception):
    """Base exception for all NerveNuc export and group table errors."""


class InvalidGroupAssignmentError(NerveNucExportError):
    """Raised when an animal group assignment is invalid, missing, duplicate, or conflicting."""


class GroupTableConsistencyError(NerveNucExportError):
    """Raised when M7 aggregation results are structurally or numerically inconsistent."""


class ExportFileExistsError(NerveNucExportError):
    """Raised when an export destination file already exists and overwrite is False."""


class ExportWriteError(NerveNucExportError):
    """Raised when an export file write operation fails."""


class StaleExportError(NerveNucExportError):
    """Raised when attempting to export an analysis session with stale downstream results."""


class IncompleteWorkflowExportError(NerveNucExportError):
    """Raised when attempting to export an analysis session before required stages are completed."""
