from __future__ import annotations

from PySide6 import QtCore
from PySide6 import QtWidgets


def main() -> None:
    app = QtWidgets.QApplication.instance()

    if app is None:
        app = QtWidgets.QApplication([])

    from .mainwindow import MainWindow

    window = MainWindow()
    window.showMaximized()

    QtCore.QTimer.singleShot(
        0,
        window.raise_,
    )

    QtCore.QTimer.singleShot(
        0,
        window.activateWindow,
    )

    app.exec()