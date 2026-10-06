"""Biological marker channel support."""

from .errors import InvalidMarkerDefinitionError
from .models import MarkerDefinition

__all__ = [
    "InvalidMarkerDefinitionError",
    "MarkerDefinition",
]
