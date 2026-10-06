"""Exceptions for workflow orchestration and reproducible analysis configuration."""

from __future__ import annotations


class NerveNucWorkflowError(Exception):
    """Base exception for all workflow orchestration errors."""


class InvalidWorkflowStateError(NerveNucWorkflowError):
    """Raised when an operation is attempted in an invalid lifecycle state or before prerequisites exist."""


class WorkflowConfigurationError(NerveNucWorkflowError):
    """Raised when analysis configuration or input binding has structural or biological inconsistencies."""


class DuplicateSampleError(NerveNucWorkflowError):
    """Raised when attempting to add a sample with an already-registered sample_id."""


class IncompatibleMeasurementSemanticsError(NerveNucWorkflowError):
    """Raised when attempting to aggregate samples with incompatible measurement semantics or mixed provenance."""
