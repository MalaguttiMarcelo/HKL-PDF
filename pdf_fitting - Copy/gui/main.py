from __future__ import annotations

from PySide6 import QtWidgets

from .mainwindow import MainWindow


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.showMaximized()
    app.exec()