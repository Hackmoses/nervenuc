"""Tests for Fiji packaging isolation and zero-JVM startup on import."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import nervenuc
import nervenuc.fiji
import nervenuc.gui


def test_core_and_gui_and_fiji_static_import_isolation() -> None:
    """Verify that root package, gui, and fiji modules do not statically import Java/PyImageJ."""
    for mod in (nervenuc, nervenuc.gui, nervenuc.fiji):
        init_file = Path(mod.__file__)
        tree = ast.parse(init_file.read_text(encoding="utf-8"))

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        for imp in imported:
            assert "imagej" not in imp.lower()
            assert "jpype" not in imp.lower()
            assert "scyjava" not in imp.lower()


def test_runtime_sys_modules_do_not_contain_jvm_or_pyimagej() -> None:
    """Verify that importing nervenuc.fiji did not populate jpype or imagej in sys.modules."""
    loaded_java_modules = [
        m for m in sys.modules if m.startswith("jpype") or m.startswith("imagej") or m.startswith("scyjava")
    ]
    assert not loaded_java_modules, f"Unexpected Java/PyImageJ modules loaded at runtime: {loaded_java_modules}"
