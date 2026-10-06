"""Lazy in-process bridge for PyImageJ / SciJava gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import FijiUnavailableError, JavaUnavailableError


class FijiBridge:
    """Lazy programmatic gateway to in-process ImageJ2/PyImageJ runtime.

    Contract:
        - Construction and import are completely inert (no JVM started, no PyImageJ imported).
        - Calling initialize() explicitly attempts to launch PyImageJ in headless mode.
        - Exactly one ImageJ2 / JVM context is reused per process lifecycle.
        - Missing dependencies produce controlled NerveNuc exceptions.
    """

    _global_ij_instance: Any = None

    def __init__(
        self,
        fiji_path: Path | str | None = None,
        mode: str = "headless",
    ) -> None:
        self._fiji_path = Path(fiji_path) if fiji_path is not None else None
        self._mode = mode

    @property
    def is_initialized(self) -> bool:
        """Return True if the in-process PyImageJ instance has been initialized."""
        return FijiBridge._global_ij_instance is not None

    @property
    def fiji_path(self) -> Path | None:
        """Configured Fiji path for this bridge instance."""
        return self._fiji_path

    def initialize(self) -> Any:
        """Explicitly initialize the PyImageJ gateway and JVM.

        Returns:
            Active imagej.ImageJ instance.

        Raises:
            JavaUnavailableError: If PyImageJ, JPype, or Java runtime is missing/incompatible.
            FijiUnavailableError: If a configured Fiji path is invalid.
        """
        if FijiBridge._global_ij_instance is not None:
            return FijiBridge._global_ij_instance

        try:
            import imagej  # type: ignore[import-untyped]
        except (ImportError, ModuleNotFoundError) as err:
            raise JavaUnavailableError(
                "PyImageJ is not installed. Install optional Fiji dependencies with: "
                "pip install \"nervenuc[fiji]\"."
            ) from err

        target = str(self._fiji_path) if self._fiji_path is not None else None

        try:
            # Initialize PyImageJ in headless mode
            ij = imagej.init(target, mode=self._mode)
            FijiBridge._global_ij_instance = ij
            return ij
        except Exception as err:
            msg = str(err)
            if "Java" in msg or "JVM" in msg or "JAVA_HOME" in msg:
                raise JavaUnavailableError(
                    f"Failed to start Java Runtime Environment for PyImageJ: {err}"
                ) from err
            raise FijiUnavailableError(
                f"Failed to initialize PyImageJ bridge with target '{target}': {err}"
            ) from err
