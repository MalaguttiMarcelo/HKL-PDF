from __future__ import annotations

import sys

from PySide6 import QtCore
from PySide6 import QtWidgets

from .timing import (
    EventLoopLagMonitor,
    TimedApplication,
    enable_periodic_traceback_dump,
    install_timing_instrumentation,
)


def main() -> None:
    """
    python -m pdf_fitting.gui                 multi-phase project window
    python -m pdf_fitting.gui project.json    ... and open a project / .inp
    python -m pdf_fitting.gui --classic       previous single-phase window
    """
    # Log output redirected to a file / pipe uses the ANSI code page on
    # Windows: replace characters it cannot encode (Δ, σ, …) instead of
    # raising inside a calculation.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = [a for a in sys.argv[1:]]
    classic = "--classic" in args
    paths = [a for a in args if not a.startswith("-")]

    app = QtWidgets.QApplication.instance()

    if app is None:
        app = TimedApplication(
            []
        )

    enable_periodic_traceback_dump()

    if classic:
        from .mainwindow import MainWindow

        install_timing_instrumentation()
        window = MainWindow()
    else:
        from .project_window import ProjectWindow

        window = ProjectWindow()

        if paths:
            QtCore.QTimer.singleShot(0, lambda: window.open_project(paths[0]))

    window._timing_event_loop_monitor = (
        EventLoopLagMonitor(
            window
        )
    )

    window.showMaximized()

    if not classic:
        from .preload import start_background_preload

        QtCore.QTimer.singleShot(2000, start_background_preload)

    QtCore.QTimer.singleShot(
        0,
        window.raise_,
    )

    QtCore.QTimer.singleShot(
        0,
        window.activateWindow,
    )

    app.exec()
