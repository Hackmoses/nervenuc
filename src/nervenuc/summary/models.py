"""Models for denominator-aware biological metrics and phenotype summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from .errors import InvalidMetricDefinitionError

if TYPE_CHECKING:
    from nervenuc.classification import Classification3DResult, ClassificationResult



class PhenotypeFilter:
    """Immutable filter specifying required positive/negative states for biological markers.

    An empty filter (PhenotypeFilter() or PhenotypeFilter({})) represents all active nuclei ('ALL').
    """

    def __init__(self, states: Mapping[str, bool] | None = None) -> None:
        if states is None:
            self._states: tuple[tuple[str, bool], ...] = ()
        elif isinstance(states, Mapping):
            parsed: list[tuple[str, bool]] = []
            for k, v in states.items():
                if not isinstance(k, str) or not k.strip() or k != k.strip():
                    raise InvalidMetricDefinitionError(
                        f"Invalid marker name in filter: {k!r}. Must be a non-empty string without whitespace padding."
                    )
                if type(v) is bool:
                    bool_v = v
                elif isinstance(v, np.bool_):
                    bool_v = bool(v)
                else:
                    raise InvalidMetricDefinitionError(
                        f"Filter state for marker {k!r} must be a boolean, got {type(v).__name__}."
                    )
                parsed.append((k, bool_v))
            # Sort by marker name for deterministic representation
            parsed.sort(key=lambda item: item[0])
            self._states = tuple(parsed)
        else:
            raise InvalidMetricDefinitionError(
                f"states must be a Mapping or None, got {type(states).__name__}."
            )

    @property
    def states(self) -> dict[str, bool]:
        """Return a copy of the filter states mapping."""
        return dict(self._states)

    def to_string(self) -> str:
        """Deterministic string representation serialized in alphabetical marker order."""
        if not self._states:
            return "ALL"
        return " & ".join(f"{k}+" if v else f"{k}-" for k, v in self._states)

    def __repr__(self) -> str:
        return f"PhenotypeFilter({self.to_string()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhenotypeFilter):
            return False
        return self._states == other._states

    def __hash__(self) -> int:
        return hash(self._states)


@dataclass(frozen=True)
class BiologicalMetricDefinition:
    """Immutable definition of a denominator-aware biological metric.

    Attributes:
        name: User-defined metric name (e.g. 'Overall proliferation', 'SOX10+ dividing').
        denominator: Filter defining the denominator subpopulation (empty filter means 'ALL').
        numerator: Filter defining additional conditions evaluated within the denominator.
    """

    name: str
    denominator: PhenotypeFilter
    numerator: PhenotypeFilter

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidMetricDefinitionError("Metric name must be a non-empty string.")
        if self.name != self.name.strip():
            raise InvalidMetricDefinitionError(
                f"Metric name cannot contain leading or trailing whitespace: {self.name!r}."
            )
        if not isinstance(self.denominator, PhenotypeFilter):
            raise InvalidMetricDefinitionError(
                f"denominator must be a PhenotypeFilter, got {type(self.denominator).__name__}."
            )
        if not isinstance(self.numerator, PhenotypeFilter):
            raise InvalidMetricDefinitionError(
                f"numerator must be a PhenotypeFilter, got {type(self.numerator).__name__}."
            )

        # Check for contradictory state requirements between denominator and numerator
        den_states = self.denominator.states
        for marker, num_state in self.numerator.states.items():
            if marker in den_states and den_states[marker] != num_state:
                raise InvalidMetricDefinitionError(
                    f"Contradictory state requirements for marker {marker!r} in metric {self.name!r}: "
                    f"denominator requires {den_states[marker]}, numerator requires {num_state}."
                )

    @property
    def effective_numerator(self) -> PhenotypeFilter:
        """Return the effective numerator filter combining denominator and numerator conditions."""
        combined = {**self.denominator.states, **self.numerator.states}
        return PhenotypeFilter(combined)


@dataclass
class BiologicalSummaryResult:
    """Container for denominator-aware biological metric summary results.

    Attributes:
        metrics: DataFrame containing 6 columns per evaluated biological metric.
        definitions: Tuple of BiologicalMetricDefinition objects evaluated.
        classification_result: Provenance reference to the ClassificationResult used.
    """

    metrics: pd.DataFrame
    definitions: tuple[BiologicalMetricDefinition, ...]
    classification_result: ClassificationResult


@dataclass
class BiologicalSummary3DResult:
    """Container for 3D denominator-aware biological metric summary results.

    Attributes:
        metrics: DataFrame containing 6 columns per evaluated biological metric.
        definitions: Tuple of BiologicalMetricDefinition objects evaluated.
        classification_result: Provenance reference to the Classification3DResult used.
    """

    metrics: pd.DataFrame
    definitions: tuple[BiologicalMetricDefinition, ...]
    classification_result: Any

    @property
    def source_id(self) -> str:
        """Authoritative execution source identity inherited from classification_result."""
        return getattr(self.classification_result, "source_id", "")
