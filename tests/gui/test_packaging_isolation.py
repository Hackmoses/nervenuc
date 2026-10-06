import ast
from pathlib import Path
import nervenuc


def test_core_package_import_does_not_load_gui_or_qt() -> None:
    """Verify that root nervenuc package does not import nervenuc.gui or PySide6."""
    init_path = Path(nervenuc.__file__)
    content = init_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    for mod in imported_modules:
        assert "PySide6" not in mod, f"PySide6 directly imported in __init__.py: {mod}"
        assert "gui" not in mod, f"gui directly imported in __init__.py: {mod}"

    # Also verify __all__ does not expose gui symbols at top-level
    assert "MainWindow" not in nervenuc.__all__
    assert "GuiSessionState" not in nervenuc.__all__
