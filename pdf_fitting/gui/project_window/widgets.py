"""
Reusable widgets of the project window. All of them edit the project model
(Parameter objects / settings dicts) in place and emit ``edited``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.project.model import Parameter

from .sections import default_value, label_for, tooltip_for
from .theme import muted_color

MONO = QtGui.QFont("Consolas")
MONO.setStyleHint(QtGui.QFont.Monospace)


def fmt_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    return str(value)


def parse_number(text: str):
    """'' -> None, numbers -> float, anything else -> the text."""
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


class Card(QtWidgets.QFrame):
    """White panel section with a title row."""

    def __init__(self, title: str, note: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QtWidgets.QWidget()
        head.setObjectName("cardHead")
        h = QtWidgets.QHBoxLayout(head)
        h.setContentsMargins(12, 8, 12, 8)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("cardTitle")
        h.addWidget(self.title_label)
        h.addStretch(1)
        self.note_label = QtWidgets.QLabel(note)
        self.note_label.setObjectName("muted")
        h.addWidget(self.note_label)
        self.head_layout = h
        outer.addWidget(head)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 10, 12, 12)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)

    def add(self, widget, stretch: int = 0):
        self.body_layout.addWidget(widget, stretch)
        return widget

    def add_header_widget(self, widget):
        self.head_layout.addWidget(widget)
        return widget


# -----------------------------------------------------------------------------
# Parameter grid
# -----------------------------------------------------------------------------


class ParamGrid(QtWidgets.QTableWidget):
    """
    Parameter | Name | Value | Refine | Min | Max | Esd, bound to Parameter
    objects of one table (a phase's or a dataset's ``parameters``).
    """

    edited = QtCore.Signal()
    COLS = ("Parameter", "Name", "Value", "Refine", "Min", "Max", "Esd")

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c, w in ((1, 140), (2, 120), (3, 70), (4, 90), (5, 90), (6, 90)):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Interactive)
            self.setColumnWidth(c, w)
        self._table: Dict[str, Parameter] = {}
        self._names: List[str] = []
        self._locks: Dict[str, str] = {}
        self._block = False
        self.itemChanged.connect(self._on_item_changed)

    def set_rows(self, table: Dict[str, Parameter], names: Sequence[str], locks: Optional[Dict[str, str]] = None):
        self._table = table
        self._names = [n for n in names if n in table]
        self._locks = dict(locks or {})
        self._block = True
        try:
            self.setRowCount(len(self._names))
            for row, name in enumerate(self._names):
                p = table[name]
                lock = self._locks.get(name, "")

                it = QtWidgets.QTableWidgetItem(label_for(name))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setToolTip(tooltip_for(name))
                self.setItem(row, 0, it)

                it = QtWidgets.QTableWidgetItem(name)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setFont(MONO)
                it.setForeground(muted_color())
                self.setItem(row, 1, it)

                it = QtWidgets.QTableWidgetItem(fmt_value(p.value))
                it.setFont(MONO)
                if lock:
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    it.setForeground(muted_color())
                self.setItem(row, 2, it)

                if lock:
                    it = QtWidgets.QTableWidgetItem("🔒 " + lock)
                    it.setFlags(Qt.ItemIsEnabled)
                    it.setForeground(muted_color())
                    it.setToolTip("Set by the crystal symmetry")
                else:
                    it = QtWidgets.QTableWidgetItem("")
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                    it.setCheckState(Qt.Checked if p.refine else Qt.Unchecked)
                self.setItem(row, 3, it)

                for col, val in ((4, p.min), (5, p.max)):
                    it = QtWidgets.QTableWidgetItem(fmt_value(val))
                    it.setFont(MONO)
                    if lock:
                        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.setItem(row, col, it)

                it = QtWidgets.QTableWidgetItem(fmt_value(p.esd) if p.refine else "")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it.setFont(MONO)
                it.setForeground(muted_color())
                self.setItem(row, 6, it)
        finally:
            self._block = False
        self._fit_height()

    def _fit_height(self):
        h = self.horizontalHeader().height() + 4
        for r in range(self.rowCount()):
            h += self.rowHeight(r)
        self.setMinimumHeight(min(max(h, 60), 420))
        self.setMaximumHeight(max(h, 60) if self.rowCount() <= 14 else 16777215)

    def selected_names(self) -> List[str]:
        rows = sorted({i.row() for i in self.selectedItems()})
        return [self._names[r] for r in rows if 0 <= r < len(self._names)]

    def _on_item_changed(self, item):
        if self._block:
            return
        row, col = item.row(), item.column()
        if row >= len(self._names):
            return
        p = self._table.get(self._names[row])
        if p is None:
            return
        if col == 3:
            p.refine = item.checkState() == Qt.Checked
        elif col == 2:
            v = parse_number(item.text())
            p.value = 0.0 if v is None else v
        elif col == 4:
            v = parse_number(item.text())
            p.min = v if isinstance(v, float) or v is None else p.min
        elif col == 5:
            v = parse_number(item.text())
            p.max = v if isinstance(v, float) or v is None else p.max
        else:
            return
        self.edited.emit()


class ParamSection(Card):
    """A Card with a ParamGrid and optional add/remove of known parameters."""

    edited = QtCore.Signal()

    def __init__(self, title: str, note: str = "", parent=None, *, allow_remove: bool = True):
        super().__init__(title, note, parent)
        self.grid = ParamGrid()
        self.grid.edited.connect(self.edited)
        self.add(self.grid)

        self._row = QtWidgets.QHBoxLayout()
        self.combo_add = QtWidgets.QComboBox()
        self.combo_add.setMinimumWidth(200)
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.setVisible(allow_remove)
        self._row.addWidget(QtWidgets.QLabel("Add parameter:"))
        self._row.addWidget(self.combo_add)
        self._row.addWidget(self.btn_add)
        self._row.addStretch(1)
        self._row.addWidget(self.btn_remove)
        self.body_layout.addLayout(self._row)
        self.btn_add.clicked.connect(self._add)
        self.btn_remove.clicked.connect(self._remove)

        self._table: Dict[str, Parameter] = {}
        self._names: List[str] = []
        self._candidates: List[str] = []
        self._locks: Dict[str, str] = {}
        self._defaults: Dict[str, Any] = {}

    def set_rows(self, table, names, *, candidates: Iterable[str] = (), locks=None, defaults=None):
        self._table = table
        self._names = list(names)
        self._locks = dict(locks or {})
        if defaults is not None:
            self._defaults = dict(defaults)
        present = [n for n in self._names if n in table]
        self._candidates = [c for c in candidates if c not in table]
        self.grid.set_rows(table, present, self._locks)
        self.combo_add.clear()
        for c in self._candidates:
            self.combo_add.addItem(f"{label_for(c)}  ({c})", c)
        has_add = bool(self._candidates)
        for w in (self.combo_add, self.btn_add):
            w.setEnabled(has_add)
        self.grid.setVisible(bool(present))

    def _add(self):
        name = self.combo_add.currentData()
        if not name or name in self._table:
            return
        self._table[name] = Parameter(value=self._defaults.get(name, default_value(name)))
        if name not in self._names:
            self._names.append(name)
        self.set_rows(self._table, self._names, candidates=self._candidates, locks=self._locks)
        rows = [r for r, n in enumerate(self.grid._names) if n == name]
        if rows:
            self.grid.selectRow(rows[0])
            self.grid.setFocus()
        self.edited.emit()

    def _remove(self):
        names = self.grid.selected_names()
        if not names and self.grid.rowCount() == 1:
            names = list(self.grid._names)
        if not names:
            QtWidgets.QToolTip.showText(
                self.btn_remove.mapToGlobal(self.btn_remove.rect().bottomLeft()),
                "Select the row(s) to remove first (click the parameter name).",
                self.btn_remove,
            )
            return
        for n in names:
            self._table.pop(n, None)
            if n not in self._candidates:
                self._candidates.append(n)
        self.set_rows(self._table, self._names, candidates=self._candidates, locks=self._locks)
        self.edited.emit()


# -----------------------------------------------------------------------------
# Settings form (typed widgets bound to a dict)
# -----------------------------------------------------------------------------


class SettingsForm(QtWidgets.QWidget):
    """
    spec entries: (key, label, kind, extra, default, tooltip)
      kind: "bool" | "int" | "float" | "choice" | "text"
      extra: choices [(value, label), ...] for "choice"; (min, max, decimals) for numbers
    """

    edited = QtCore.Signal()

    def __init__(self, spec: Sequence[Tuple], parent=None, columns: int = 2):
        super().__init__(parent)
        self._spec = list(spec)
        self._data: Dict[str, Any] = {}
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._block = False
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)

        for i, (key, label, kind, extra, default, tip) in enumerate(self._spec):
            row, col = divmod(i, columns)
            lab = QtWidgets.QLabel(label)
            lab.setToolTip(tip or key)
            if kind == "bool":
                w = QtWidgets.QCheckBox()
                w.toggled.connect(lambda v, k=key: self._set(k, bool(v)))
            elif kind == "int":
                w = QtWidgets.QSpinBox()
                lo, hi = (extra or (0, 10**9))[:2]
                w.setRange(int(lo), int(hi))
                w.valueChanged.connect(lambda v, k=key: self._set(k, int(v)))
            elif kind == "float":
                w = QtWidgets.QDoubleSpinBox()
                lo, hi, dec = extra or (0.0, 1e9, 4)
                w.setRange(lo, hi)
                w.setDecimals(dec)
                w.setSingleStep(10 ** (-min(dec, 3)))
                w.valueChanged.connect(lambda v, k=key: self._set(k, float(v)))
            elif kind == "choice":
                w = QtWidgets.QComboBox()
                for value, text in extra:
                    w.addItem(text, value)
                w.currentIndexChanged.connect(lambda _i, k=key, cw=w: self._set(k, cw.currentData()))
            else:
                w = QtWidgets.QLineEdit()
                w.editingFinished.connect(lambda k=key, lw=w: self._set(k, lw.text().strip()))
            w.setToolTip(tip or key)
            self._widgets[key] = w
            grid.addWidget(lab, row, col * 2)
            grid.addWidget(w, row, col * 2 + 1)
        grid.setColumnStretch(1, 1)
        if columns > 1:
            grid.setColumnStretch(3, 1)

    def bind(self, data: Dict[str, Any]):
        self._data = data
        self._block = True
        try:
            for key, _label, kind, extra, default, _tip in self._spec:
                w = self._widgets[key]
                value = data.get(key, default)
                if kind == "bool":
                    w.setChecked(bool(value))
                elif kind == "int":
                    w.setValue(int(value or 0))
                elif kind == "float":
                    try:
                        w.setValue(float(value))
                    except (TypeError, ValueError):
                        w.setValue(float(default or 0.0))
                elif kind == "choice":
                    idx = w.findData(value)
                    if idx < 0:
                        idx = w.findData(default)
                    w.setCurrentIndex(max(idx, 0))
                else:
                    w.setText("" if value is None else str(value))
        finally:
            self._block = False

    def _set(self, key, value):
        if self._block:
            return
        self._data[key] = value
        self.edited.emit()


# -----------------------------------------------------------------------------
# Generic key / value editor for free-form sections
# -----------------------------------------------------------------------------


class DictEditor(QtWidgets.QWidget):
    """Key / value table for a dict (values as JSON or plain text)."""

    edited = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(("Key", "Value"))
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 220)
        self.table.itemChanged.connect(self._changed)
        btn_add = QtWidgets.QPushButton("Add entry")
        btn_del = QtWidgets.QPushButton("Remove selected")
        btn_add.clicked.connect(self._add)
        btn_del.clicked.connect(self._remove)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(btn_add)
        row.addWidget(btn_del)
        row.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.table)
        lay.addLayout(row)
        self._data: Dict[str, Any] = {}
        self._block = False

    def bind(self, data: Dict[str, Any]):
        self._data = data
        self._block = True
        try:
            self.table.setRowCount(len(data))
            for r, (k, v) in enumerate(data.items()):
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
                text = v if isinstance(v, str) else json.dumps(v)
                it = QtWidgets.QTableWidgetItem(text)
                it.setFont(MONO)
                self.table.setItem(r, 1, it)
        finally:
            self._block = False

    def _read(self):
        out = {}
        for r in range(self.table.rowCount()):
            k_item, v_item = self.table.item(r, 0), self.table.item(r, 1)
            key = k_item.text().strip() if k_item else ""
            if not key:
                continue
            text = v_item.text().strip() if v_item else ""
            try:
                out[key] = json.loads(text)
            except (ValueError, TypeError):
                out[key] = text
        return out

    def _changed(self, _item=None):
        if self._block:
            return
        new = self._read()
        self._data.clear()
        self._data.update(new)
        self.edited.emit()

    def _add(self):
        self._block = True
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem("new_key"))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(""))
        self._block = False
        self._changed()

    def _remove(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._changed()


def scroll_page(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(widget)
    return area
