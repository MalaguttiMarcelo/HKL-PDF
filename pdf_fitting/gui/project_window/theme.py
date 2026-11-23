"""
Light / dark theme following the Windows (or desktop) colour scheme.

Fusion style + an explicit palette, so check boxes, selections and table
indicators are drawn with proper contrast in both modes. The stylesheet only
adds the card / chip / banner look on top of the palette.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

LIGHT = {
    "window": "#eef0f4", "base": "#ffffff", "alt": "#f6f8fb", "panel": "#f8f9fb",
    "text": "#1f2733", "muted": "#5b6472", "border": "#d3d8e0", "border_soft": "#e5e9ef",
    "header": "#f3f5f8", "accent": "#1d4ed8", "accent_text": "#ffffff", "select": "#dbe5ff",
    "select_text": "#1e3a8a", "chip": "#eef1f5", "ok_bg": "#dcfce7", "ok_fg": "#14532d",
    "banner_bg": "#eff6ff", "banner_border": "#bfdbfe", "banner_fg": "#1e3a8a",
    "button": "#ffffff", "hover": "#eef2f7", "disabled": "#9ca3af",
}
DARK = {
    "window": "#1b1d21", "base": "#23262b", "alt": "#272a30", "panel": "#202328",
    "text": "#e6e8eb", "muted": "#a4abb6", "border": "#3a3f47", "border_soft": "#30343b",
    "header": "#2a2e34", "accent": "#4f7cff", "accent_text": "#ffffff", "select": "#2f4a85",
    "select_text": "#ffffff", "chip": "#2c3037", "ok_bg": "#1f4d33", "ok_fg": "#c7f2d7",
    "banner_bg": "#1e2a44", "banner_border": "#34508a", "banner_fg": "#d6e2ff",
    "button": "#2d3138", "hover": "#353a42", "disabled": "#6b7280",
}

STYLE = """
QMainWindow, QWidget#central {{ background: {window}; }}
QWidget#header {{ background: {base}; border-bottom: 1px solid {border}; }}
QTreeWidget {{ background: {panel}; border: 0; border-right: 1px solid {border}; font-size: 13px; }}
QTreeWidget::item {{ height: 26px; }}
QTreeWidget::item:selected {{ background: {select}; color: {select_text}; }}
QFrame#card {{ background: {base}; border: 1px solid {border}; border-radius: 8px; }}
QWidget#cardHead {{ border-bottom: 1px solid {border_soft}; }}
QLabel#cardTitle {{ font-weight: 600; font-size: 13px; color: {text}; }}
QLabel#muted {{ color: {muted}; font-size: 12px; }}
QLabel#pageTitle {{ font-size: 19px; font-weight: 600; color: {text}; }}
QLabel#crumbs {{ color: {muted}; font-size: 12px; }}
QLabel#chip {{ background: {chip}; color: {text}; border-radius: 4px; padding: 3px 8px; font-size: 12px; }}
QLabel#chipOk {{ background: {ok_bg}; color: {ok_fg}; border-radius: 4px; padding: 3px 8px; font-size: 12px; }}
QLabel#banner {{ background: {banner_bg}; border: 1px solid {banner_border}; color: {banner_fg}; border-radius: 6px; padding: 8px 10px; }}
QTableWidget, QTableView {{ gridline-color: {border_soft}; selection-background-color: {select}; selection-color: {select_text}; }}
QHeaderView::section {{ background: {header}; color: {muted}; font-weight: 600; border: 0; border-bottom: 1px solid {border}; padding: 4px 6px; }}
QToolBar {{ background: {base}; border: 0; spacing: 4px; }}
QToolButton {{ padding: 4px 8px; border-radius: 5px; color: {text}; }}
QToolButton:hover {{ background: {hover}; }}
QToolButton:disabled {{ color: {disabled}; }}
QPushButton#primary {{ background: {accent}; color: {accent_text}; border-radius: 5px; padding: 5px 12px; font-weight: 600; }}
QScrollArea {{ background: transparent; }}
"""


def is_dark() -> bool:
    hints = QtGui.QGuiApplication.styleHints()
    try:
        return hints.colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return QtWidgets.QApplication.palette().color(QtGui.QPalette.Window).lightness() < 128


def tokens() -> dict:
    return DARK if is_dark() else LIGHT


def _palette(t: dict) -> QtGui.QPalette:
    p = QtGui.QPalette()
    c = QtGui.QColor
    for group in (QtGui.QPalette.Active, QtGui.QPalette.Inactive):
        p.setColor(group, QtGui.QPalette.Window, c(t["window"]))
        p.setColor(group, QtGui.QPalette.WindowText, c(t["text"]))
        p.setColor(group, QtGui.QPalette.Base, c(t["base"]))
        p.setColor(group, QtGui.QPalette.AlternateBase, c(t["alt"]))
        p.setColor(group, QtGui.QPalette.Text, c(t["text"]))
        p.setColor(group, QtGui.QPalette.Button, c(t["button"]))
        p.setColor(group, QtGui.QPalette.ButtonText, c(t["text"]))
        p.setColor(group, QtGui.QPalette.Highlight, c(t["accent"]))
        p.setColor(group, QtGui.QPalette.HighlightedText, c(t["accent_text"]))
        p.setColor(group, QtGui.QPalette.ToolTipBase, c(t["base"]))
        p.setColor(group, QtGui.QPalette.ToolTipText, c(t["text"]))
        p.setColor(group, QtGui.QPalette.PlaceholderText, c(t["muted"]))
        p.setColor(group, QtGui.QPalette.Link, c(t["accent"]))
        p.setColor(group, QtGui.QPalette.Mid, c(t["border"]))
        p.setColor(group, QtGui.QPalette.Midlight, c(t["border_soft"]))
        p.setColor(group, QtGui.QPalette.Light, c(t["hover"]))
        p.setColor(group, QtGui.QPalette.Dark, c(t["border"]))
        p.setColor(group, QtGui.QPalette.Shadow, c("#000000"))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, c(t["disabled"]))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Base, c(t["panel"]))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Button, c(t["panel"]))
    return p


def apply_theme(window: QtWidgets.QWidget | None = None) -> dict:
    """Apply palette + stylesheet for the current scheme; returns the tokens."""
    app = QtWidgets.QApplication.instance()
    t = tokens()
    if app is not None:
        if app.style().name().lower() != "fusion":
            app.setStyle("Fusion")
        app.setPalette(_palette(t))
    if window is not None:
        window.setStyleSheet(STYLE.format(**t))
    return t


def watch_scheme(callback) -> None:
    """Call ``callback`` when the desktop switches light / dark."""
    hints = QtGui.QGuiApplication.styleHints()
    try:
        hints.colorSchemeChanged.connect(lambda _scheme: callback())
    except AttributeError:
        pass


def muted_color() -> QtGui.QColor:
    return QtGui.QColor(tokens()["muted"])
