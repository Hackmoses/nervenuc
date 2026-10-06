"""Application entry point and Qt initialization."""

from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    """Launch NerveNuc desktop GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("NerveNuc")
    app.setOrganizationName("NerveNuc")

    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: window.open_image(sys.argv[1]))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
