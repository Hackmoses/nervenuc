"""Models for biological marker definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidMarkerDefinitionError


@dataclass(frozen=True)
class MarkerDefinition:
    """Immutable definition of a user-defined biological marker.

    Attributes:
        name: User-defined marker name (e.g. 'SOX10', 'BrdU', 'Marker 1').
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise InvalidMarkerDefinitionError(
                f"Marker name must be a string, got {type(self.name).__name__}."
            )
        if not self.name or not self.name.strip():
            raise InvalidMarkerDefinitionError(
                "Marker name cannot be empty or whitespace-only."
            )
        if self.name != self.name.strip():
            raise InvalidMarkerDefinitionError(
                f"Marker name cannot contain leading or trailing whitespace: '{self.name}'."
            )
