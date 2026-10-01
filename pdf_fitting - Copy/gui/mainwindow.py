from __future__ import annotations

import re
import time
import os
import tempfile
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QShortcut, QKeySequence, QColor
from PySide6.QtCore import Qt

from pdf_fitting.io_handler import read_input_file, update_input_file_with_refined_params
from pdf_fitting.models.gr_model import PDFCalculator, lattice_matrix, apply_lattice_constraints

from pdf_fitting.fit_engine import apply_constraints, apply_model_lattice_constraints_to_params

from .worker import FitWorker
from .builder import InputBuilder
from .plots import MplPlot, ParamEvolutionTabs
from .structure_view import StructureViewer

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matplotlib.collections import LineCollection

from matplotlib.colors import to_rgba, to_hex

from .cif_utils import read_cif_asu_sites

DEFAULT_TICK_MARKER_MIN_R = 0.0
DEFAULT_TICK_MARKER_MAX_R = 20.0




# Common CPK/Jmol-like atom colors as hex strings.
STANDARD_ATOM_COLORS = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "S": "#FFFF30",
    "P": "#FF8000",
    "B": "#FFB5B5",
    "Si": "#F0C8A0",

    "Li": "#CC80FF",
    "Na": "#AB5CF2",
    "K": "#8F40D4",
    "Rb": "#702EB0",
    "Cs": "#57178F",

    "Be": "#C2FF00",
    "Mg": "#8AFF00",
    "Ca": "#3DFF00",
    "Sr": "#00FF00",
    "Ba": "#00C900",

    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",

    "Al": "#BFA6A6",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",

    "Zr": "#94E0E0",
    "Mo": "#54B5B5",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",

    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Nd": "#C7FFC7",
    "Sm": "#8FFFC7",
    "Gd": "#45FFC7",
    "Dy": "#1FFFC7",
    "Er": "#00E675",
    "Yb": "#00BF38",

    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "U": "#008FFF",
}


_COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _safe_mpl_color(color, fallback="#808080") -> str:
    """
    Return a Matplotlib-safe hex color.

    Fixes the crash caused by stringified tuples like:
        "(0.12, 0.46, 0.70)"
    """
    if color is None:
        return fallback

    if isinstance(color, str):
        s = color.strip()
        if not s:
            return fallback

        # Convert stringified tuple/list to real tuple/list.
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                import ast
                parsed = ast.literal_eval(s)
                return to_hex(to_rgba(parsed))
            except Exception:
                return fallback

        try:
            return to_hex(to_rgba(s))
        except Exception:
            return fallback

    try:
        return to_hex(to_rgba(color))
    except Exception:
        return fallback


def _element_from_pair_token(token: str) -> str:
    """
    Convert pair token like:
        'ca', 'o', 'fe2plus', 'Fe0plus'
    to chemical element symbol:
        'Ca', 'O', 'Fe'
    """
    t = str(token or "").strip().lower()
    if not t:
        return ""

    # Prefer longest symbols first so Cl is found before C.
    for sym in sorted(STANDARD_ATOM_COLORS.keys(), key=len, reverse=True):
        if t.startswith(sym.lower()):
            return sym

    m = re.match(r"([a-z]{1,2})", t)
    if not m:
        return ""

    raw = m.group(1)
    return raw.capitalize()


def _blend_hex_colors(c1: str, c2: str) -> str:
    try:
        r1, g1, b1, _ = to_rgba(c1)
        r2, g2, b2, _ = to_rgba(c2)
        return to_hex(((r1 + r2) / 2.0, (g1 + g2) / 2.0, (b1 + b2) / 2.0))
    except Exception:
        return "#808080"


def standard_pair_color(pair: str, fallback_index: int = 0) -> str:
    """
    Stable default color for an atom pair.

    Examples:
        ca-o -> blend(Ca, O)
        fe-fe -> Fe color
    """
    p = str(pair or "").strip()

    if "-" not in p:
        el = _element_from_pair_token(p)
        return STANDARD_ATOM_COLORS.get(el, _COLOR_CYCLE[fallback_index % len(_COLOR_CYCLE)])

    a, b = [x.strip() for x in p.split("-", 1)]
    ea = _element_from_pair_token(a)
    eb = _element_from_pair_token(b)

    ca = STANDARD_ATOM_COLORS.get(ea)
    cb = STANDARD_ATOM_COLORS.get(eb)

    if ca and cb:
        if ea == eb:
            return ca
        return _blend_hex_colors(ca, cb)

    return _COLOR_CYCLE[fallback_index % len(_COLOR_CYCLE)]

class PdfPlotDialog(QtWidgets.QDialog):
    """Floating window for the PDF plot.

    Closing this dialog hides the PDF plot window.
    """

    def __init__(self, owner: "MainWindow"):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("PDF fit and residuals")

        self.setWindowFlags(
            self.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.setSizeGripEnabled(True)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(
                int(geo.width() * 0.85),
                int(geo.height() * 0.85),
            )
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        else:
            self.resize(1200, 750)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

    def closeEvent(self, event):
        try:
            if not getattr(self.owner, "_returning_pdf_to_main", False):
                self.owner.attach_pdf_plot()
        except Exception:
            pass

        event.accept()

class PairDisplayDialog(QtWidgets.QDialog):
    """
    Popup dialog to control PDF pair-contribution display.

    Controls:
      - show/hide G_fit
      - style G_exp
      - select pair contributions
      - choose pair color, line style, marker, linewidth, alpha
      - show sum of selected pair contributions
    """

    def __init__(self, owner: "MainWindow"):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Show pair contributions")
        self.resize(850, 500)

        layout = QtWidgets.QVBoxLayout(self)

        
        # -------------------------
        # Global curve controls
        # -------------------------
        gb_global = QtWidgets.QGroupBox("Main PDF curves")
        form = QtWidgets.QFormLayout(gb_global)

        self.chk_show_gfit = QtWidgets.QCheckBox("Show fitted total G_fit")
        self.chk_show_gfit.setChecked(bool(getattr(owner, "_show_gfit", True)))

        self.chk_show_pair_sum = QtWidgets.QCheckBox("Show sum of selected pair contributions")
        self.chk_show_pair_sum.setChecked(bool(getattr(owner, "_pdf_pair_sum_visible", False)))

        self.combo_gexp_line = QtWidgets.QComboBox()
        self.combo_gexp_line.addItems(["None", "-", "--", "-.", ":"])

        self.combo_gexp_marker = QtWidgets.QComboBox()
        self.combo_gexp_marker.addItems(["o", ".", "s", "^", "None"])

        self.spin_gexp_lw = QtWidgets.QDoubleSpinBox()
        self.spin_gexp_lw.setRange(0.0, 10.0)
        self.spin_gexp_lw.setSingleStep(0.2)

        self.spin_gexp_ms = QtWidgets.QDoubleSpinBox()
        self.spin_gexp_ms.setRange(0.0, 20.0)
        self.spin_gexp_ms.setSingleStep(0.5)

        gexp_style = getattr(owner, "_gexp_style", {})
        self.combo_gexp_line.setCurrentText(str(gexp_style.get("linestyle", "None")))
        self.combo_gexp_marker.setCurrentText(str(gexp_style.get("marker", "o")))
        self.spin_gexp_lw.setValue(float(gexp_style.get("linewidth", 1.0)))
        self.spin_gexp_ms.setValue(float(gexp_style.get("markersize", 4.0)))

        form.addRow(self.chk_show_gfit)
        form.addRow(self.chk_show_pair_sum)
        form.addRow("G_exp line:", self.combo_gexp_line)
        form.addRow("G_exp marker:", self.combo_gexp_marker)
        form.addRow("G_exp line width:", self.spin_gexp_lw)
        form.addRow("G_exp marker size:", self.spin_gexp_ms)

        layout.addWidget(gb_global)

        # -------------------------
        # Pair table
        # -------------------------
        self.tbl = QtWidgets.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["Show", "Pair", "Color", "Line", "Marker", "Width", "Alpha"]
        )
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        layout.addWidget(self.tbl, 1)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select all")
        btn_none = QtWidgets.QPushButton("Select none")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_close = QtWidgets.QPushButton("Close")

        btns.addWidget(btn_all)
        btns.addWidget(btn_none)
        btns.addStretch(1)
        btns.addWidget(btn_apply)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_apply.clicked.connect(self.apply)
        btn_close.clicked.connect(self.close)

        self._populate_pairs()

    def _default_pair_style(self, pair: str) -> dict:
        styles = getattr(self.owner, "_pdf_pair_style", {})
        return dict(
            styles.get(
                pair,
                {
                    "visible": False,
                    "color": standard_pair_color(pair),
                    "linestyle": "--",
                    "marker": "None",
                    "linewidth": 1.2,
                    "alpha": 0.85,
                },
            )
        )

    def _choose_color(self, row: int):
        btn = self.tbl.cellWidget(row, 2)
        old = _safe_mpl_color(btn.property("color") if btn is not None else "", "#000000")
        qold = QColor(old)

        col = QtWidgets.QColorDialog.getColor(qold, self, "Choose pair color")
        if not col.isValid():
            return

        color = col.name()
        btn.setProperty("color", color)
        btn.setStyleSheet(f"background-color: {color};")

    def _populate_pairs(self):
        pairs = sorted((getattr(self.owner, "_pdf_pair_contrib", {}) or {}).keys())
        self.tbl.setRowCount(len(pairs))

        for row, pair in enumerate(pairs):
            st = self._default_pair_style(pair)

            # Show checkbox
            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if bool(st.get("visible", False)) else Qt.Unchecked)
            self.tbl.setItem(row, 0, chk)

            # Pair name
            item_pair = QtWidgets.QTableWidgetItem(pair)
            item_pair.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(row, 1, item_pair)

            # Color button
            btn_color = QtWidgets.QPushButton("Color")

            color = _safe_mpl_color(
                st.get("color", ""),
                standard_pair_color(pair, row),
            )

            btn_color.setProperty("color", color)
            btn_color.setStyleSheet(f"background-color: {color};")
            btn_color.clicked.connect(lambda _=None, r=row: self._choose_color(r))
            self.tbl.setCellWidget(row, 2, btn_color)

            # Line style
            combo_line = QtWidgets.QComboBox()
            combo_line.addItems(["-", "--", "-.", ":", "None"])
            combo_line.setCurrentText(str(st.get("linestyle", "--")))
            self.tbl.setCellWidget(row, 3, combo_line)

            # Marker
            combo_marker = QtWidgets.QComboBox()
            combo_marker.addItems(["None", "o", ".", "s", "^", "x", "+"])
            combo_marker.setCurrentText(str(st.get("marker", "None")))
            self.tbl.setCellWidget(row, 4, combo_marker)

            # Width
            spin_w = QtWidgets.QDoubleSpinBox()
            spin_w.setRange(0.1, 10.0)
            spin_w.setSingleStep(0.2)
            spin_w.setValue(float(st.get("linewidth", 1.2)))
            self.tbl.setCellWidget(row, 5, spin_w)

            # Alpha
            spin_a = QtWidgets.QDoubleSpinBox()
            spin_a.setRange(0.05, 1.0)
            spin_a.setSingleStep(0.05)
            spin_a.setValue(float(st.get("alpha", 0.85)))
            self.tbl.setCellWidget(row, 6, spin_a)

    def _select_all(self):
        for row in range(self.tbl.rowCount()):
            item = self.tbl.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def _select_none(self):
        for row in range(self.tbl.rowCount()):
            item = self.tbl.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Unchecked)

    def apply(self):
        owner = self.owner

        owner._show_gfit = bool(self.chk_show_gfit.isChecked())
        owner._pdf_pair_sum_visible = bool(self.chk_show_pair_sum.isChecked())

        line = self.combo_gexp_line.currentText()
        marker = self.combo_gexp_marker.currentText()

        owner._gexp_style = {
            "linestyle": "None" if line == "None" else line,
            "marker": "None" if marker == "None" else marker,
            "linewidth": float(self.spin_gexp_lw.value()),
            "markersize": float(self.spin_gexp_ms.value()),
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "color": "black",
        }

        styles = {}

        for row in range(self.tbl.rowCount()):
            pair_item = self.tbl.item(row, 1)
            if pair_item is None:
                continue

            pair = str(pair_item.text()).strip()
            if not pair:
                continue

            show_item = self.tbl.item(row, 0)
            visible = bool(show_item and show_item.checkState() == Qt.Checked)

            btn_color = self.tbl.cellWidget(row, 2)
            color = standard_pair_color(pair, row)

            if btn_color is not None:
                color = _safe_mpl_color(
                    btn_color.property("color"),
                    standard_pair_color(pair, row),
                )

            combo_line = self.tbl.cellWidget(row, 3)
            combo_marker = self.tbl.cellWidget(row, 4)
            spin_w = self.tbl.cellWidget(row, 5)
            spin_a = self.tbl.cellWidget(row, 6)

            linestyle = combo_line.currentText() if combo_line is not None else "--"
            marker = combo_marker.currentText() if combo_marker is not None else "None"

            styles[pair] = {
                "visible": visible,
                "color": color,
                "linestyle": "None" if linestyle == "None" else linestyle,
                "marker": "None" if marker == "None" else marker,
                "linewidth": float(spin_w.value()) if spin_w is not None else 1.2,
                "alpha": float(spin_a.value()) if spin_a is not None else 0.85,
            }

        owner._pdf_pair_style = styles
        owner._apply_pdf_base_styles()
        owner._refresh_pdf_pair_curves()

        # If tick markers are visible, refresh them so their colors match
        # updated Show Pairs colors unless explicitly overridden.
        try:
            if owner.btn_tick_markers.isChecked():
                owner._refresh_pdf_tick_markers()
        except Exception:
            pass

class TickMarkerFilterDialog(QtWidgets.QDialog):
    """
    Popup dialog for selecting which PDF tick markers to show.

    Filters:
      - pair type, e.g. ca-o, o-o
      - direction family, e.g. [h00], [00l], [hh0]

    Style:
      - color
      - line width
      - alpha/opacity
    """

    def __init__(self, owner: "MainWindow"):
        super().__init__(owner)
        self.owner = owner

        self.setWindowTitle("Tick marker filters")
        self.resize(760, 520)

        layout = QtWidgets.QVBoxLayout(self)

        # ---------------------------------------------------------
        # Tick range
        # ---------------------------------------------------------
        gb_range = QtWidgets.QGroupBox("Tick range")
        range_layout = QtWidgets.QFormLayout(gb_range)

        self.spin_tick_rmin = QtWidgets.QDoubleSpinBox()
        self.spin_tick_rmin.setDecimals(3)
        self.spin_tick_rmin.setRange(0.0, 1e6)
        self.spin_tick_rmin.setSingleStep(1.0)
        self.spin_tick_rmin.setValue(
            float(getattr(owner, "_tick_range_min", DEFAULT_TICK_MARKER_MIN_R))
        )

        self.spin_tick_rmax = QtWidgets.QDoubleSpinBox()
        self.spin_tick_rmax.setDecimals(3)
        self.spin_tick_rmax.setRange(0.001, 1e6)
        self.spin_tick_rmax.setSingleStep(1.0)
        self.spin_tick_rmax.setValue(
            float(getattr(owner, "_tick_range_max", DEFAULT_TICK_MARKER_MAX_R))
        )

        range_layout.addRow("Minimum r / Å:", self.spin_tick_rmin)
        range_layout.addRow("Maximum r / Å:", self.spin_tick_rmax)

        layout.addWidget(gb_range)


        # ---------------------------------------------------------
        # Direction filter
        # ---------------------------------------------------------
        gb_dir = QtWidgets.QGroupBox("Direction filter")
        dir_layout = QtWidgets.QFormLayout(gb_dir)

        self.combo_direction = QtWidgets.QComboBox()
        self.combo_direction.addItem("All directions", "all")
        self.combo_direction.addItem("[h00]", "h00")
        self.combo_direction.addItem("[0k0]", "0k0")
        self.combo_direction.addItem("[00l]", "00l")
        self.combo_direction.addItem("[hh0]", "hh0")
        self.combo_direction.addItem("[h0l]", "h0l")
        self.combo_direction.addItem("[0kl]", "0kl")
        self.combo_direction.addItem("Custom contains text", "contains")

        self.le_contains = QtWidgets.QLineEdit()
        self.le_contains.setPlaceholderText("Example: [1 0 4] or 1 0")

        # Restore previous direction filter
        current_mode = getattr(owner, "_tick_filter_direction_mode", "all")
        idx = self.combo_direction.findData(current_mode)
        if idx >= 0:
            self.combo_direction.setCurrentIndex(idx)

        self.le_contains.setText(
            str(getattr(owner, "_tick_filter_direction_text", "") or "")
        )

        dir_layout.addRow("Direction:", self.combo_direction)
        dir_layout.addRow("Contains:", self.le_contains)

        layout.addWidget(gb_dir)

        # ---------------------------------------------------------
        # Pair filter/style table
        # ---------------------------------------------------------
        gb_pairs = QtWidgets.QGroupBox("Pair filters and tick styles")
        pairs_layout = QtWidgets.QVBoxLayout(gb_pairs)

        self.tbl_pairs = QtWidgets.QTableWidget(0, 6)
        self.tbl_pairs.setHorizontalHeaderLabels(
            ["Show", "Pair", "N ticks", "Color", "Width", "Alpha"]
        )
        self.tbl_pairs.verticalHeader().setVisible(False)
        self.tbl_pairs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        hdr = self.tbl_pairs.horizontalHeader()
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

        pairs_layout.addWidget(self.tbl_pairs)

        btn_pair_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        btn_pair_row.addWidget(self.btn_all)
        btn_pair_row.addWidget(self.btn_none)
        btn_pair_row.addStretch(1)

        pairs_layout.addLayout(btn_pair_row)

        layout.addWidget(gb_pairs, 1)

        # ---------------------------------------------------------
        # Bottom buttons
        # ---------------------------------------------------------
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")

        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_cancel)

        layout.addLayout(btns)

        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._populate_pairs()

    def _choose_color(self, row: int) -> None:
        btn = self.tbl_pairs.cellWidget(row, 3)

        if btn is None:
            return

        old = _safe_mpl_color(btn.property("color"), "#000000")
        qold = QColor(old)

        col = QtWidgets.QColorDialog.getColor(
            qold,
            self,
            "Choose tick marker color",
        )

        if not col.isValid():
            return

        color = col.name()

        btn.setProperty("color", color)
        btn.setStyleSheet(f"background-color: {color};")

    def _populate_pairs(self) -> None:
        tick_data = getattr(self.owner, "_pdf_tick_data", {}) or {}
        pairs_dict = tick_data.get("pairs", {}) if isinstance(tick_data, dict) else {}

        pair_names = sorted(pairs_dict.keys())

        allowed_pairs = getattr(self.owner, "_tick_filter_pairs", None)
        tick_styles = getattr(self.owner, "_tick_pair_style", {}) or {}
        pair_styles = getattr(self.owner, "_pdf_pair_style", {}) or {}

        self.tbl_pairs.setRowCount(len(pair_names))

        for row, pair in enumerate(pair_names):
            pdata = pairs_dict.get(pair, {}) or {}
            n_ticks = len(pdata.get("r", []) or [])

            # -----------------------------------------------------
            # Show checkbox
            # -----------------------------------------------------
            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)

            if allowed_pairs is None:
                checked = True
            else:
                checked = pair in allowed_pairs

            chk.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.tbl_pairs.setItem(row, 0, chk)

            # -----------------------------------------------------
            # Pair label
            # -----------------------------------------------------
            item_pair = QtWidgets.QTableWidgetItem(str(pair))
            item_pair.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_pairs.setItem(row, 1, item_pair)

            # -----------------------------------------------------
            # Number of ticks
            # -----------------------------------------------------
            item_n = QtWidgets.QTableWidgetItem(str(n_ticks))
            item_n.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_pairs.setItem(row, 2, item_n)

            # -----------------------------------------------------
            # Style defaults:
            #   1. explicit tick style
            #   2. Show Pairs style
            #   3. standard atom/pair color
            # -----------------------------------------------------
            tick_style = tick_styles.get(pair, {}) or {}
            pair_style = pair_styles.get(pair, {}) or {}

            color = _safe_mpl_color(
                tick_style.get("color", "") or pair_style.get("color", ""),
                standard_pair_color(pair, row),
            )

            try:
                linewidth = float(tick_style.get("linewidth", 0.7))
            except Exception:
                linewidth = 0.7

            try:
                alpha = float(tick_style.get("alpha", 0.30))
            except Exception:
                alpha = 0.30

            # Color button
            btn_color = QtWidgets.QPushButton("Color")
            btn_color.setProperty("color", color)
            btn_color.setStyleSheet(f"background-color: {color};")
            btn_color.clicked.connect(lambda _=None, r=row: self._choose_color(r))
            self.tbl_pairs.setCellWidget(row, 3, btn_color)

            # Width spinbox
            spin_width = QtWidgets.QDoubleSpinBox()
            spin_width.setRange(0.1, 10.0)
            spin_width.setDecimals(2)
            spin_width.setSingleStep(0.1)
            spin_width.setValue(linewidth)
            self.tbl_pairs.setCellWidget(row, 4, spin_width)

            # Alpha spinbox
            spin_alpha = QtWidgets.QDoubleSpinBox()
            spin_alpha.setRange(0.01, 1.0)
            spin_alpha.setDecimals(2)
            spin_alpha.setSingleStep(0.05)
            spin_alpha.setValue(alpha)
            spin_alpha.setToolTip("Opacity: 0.30 means 70% transparent.")
            self.tbl_pairs.setCellWidget(row, 5, spin_alpha)

    def _select_all(self) -> None:
        for row in range(self.tbl_pairs.rowCount()):
            it = self.tbl_pairs.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.Checked)

    def _select_none(self) -> None:
        for row in range(self.tbl_pairs.rowCount()):
            it = self.tbl_pairs.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.Unchecked)

    def apply_to_owner(self) -> None:
        selected_pairs = set()
        all_pairs = set()

        tick_styles = dict(getattr(self.owner, "_tick_pair_style", {}) or {})

        rmin = float(self.spin_tick_rmin.value())
        rmax = float(self.spin_tick_rmax.value())

        if rmax <= rmin:
            rmax = rmin + 0.001

        self.owner._tick_range_min = rmin
        self.owner._tick_range_max = rmax

        for row in range(self.tbl_pairs.rowCount()):
            it_chk = self.tbl_pairs.item(row, 0)
            it_pair = self.tbl_pairs.item(row, 1)

            if it_pair is None:
                continue

            pair = str(it_pair.text()).strip()
            if not pair:
                continue

            all_pairs.add(pair)

            if it_chk is not None and it_chk.checkState() == Qt.Checked:
                selected_pairs.add(pair)

            # Extract style widgets
            btn_color = self.tbl_pairs.cellWidget(row, 3)
            spin_width = self.tbl_pairs.cellWidget(row, 4)
            spin_alpha = self.tbl_pairs.cellWidget(row, 5)

            color = ""
            if btn_color is not None:
                color = _safe_mpl_color(
                    btn_color.property("color"),
                    standard_pair_color(pair, row),
                )

            linewidth = 0.7
            if spin_width is not None:
                linewidth = float(spin_width.value())

            alpha = 0.30
            if spin_alpha is not None:
                alpha = float(spin_alpha.value())

            tick_styles[pair] = {
                "color": color,
                "linewidth": linewidth,
                "alpha": alpha,
            }

        # If all pairs are selected, store None = no pair filter.
        if selected_pairs == all_pairs:
            self.owner._tick_filter_pairs = None
        else:
            self.owner._tick_filter_pairs = selected_pairs

        self.owner._tick_filter_direction_mode = str(
            self.combo_direction.currentData() or "all"
        )
        self.owner._tick_filter_direction_text = str(self.le_contains.text()).strip()

        self.owner._tick_pair_style = tick_styles

class WarrenDisplayOptionsDialog(QtWidgets.QDialog):
    """Popup for Warren plot display options."""

    def __init__(self, owner: "MainWindow"):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Warren plot display options")
        self.resize(850, 600)

        layout = QtWidgets.QVBoxLayout(self)

        # -------------------------
        # Y mode + search
        # -------------------------
        top = QtWidgets.QHBoxLayout()

        self.combo_y = QtWidgets.QComboBox()
        self.combo_y.addItem("Warren: sqrt(<ΔL²>)", "warren")
        self.combo_y.addItem("Microstrain: sqrt(<ΔL²>)/L", "microstrain")

        current_y = getattr(owner, "_warren_y_mode", "warren")
        idx = self.combo_y.findData(current_y)
        if idx >= 0:
            self.combo_y.setCurrentIndex(idx)

        self.le_search = QtWidgets.QLineEdit()
        self.le_search.setPlaceholderText("Search reflection/direction, e.g. 200, 110, h00, hh0...")

        self.chk_show_sparse = QtWidgets.QCheckBox("Show sparse directions (≤5 points)")
        self.chk_show_sparse.setChecked(False)
        self.chk_show_sparse.setToolTip(
            "When off, directions with 5 or fewer points are hidden from this table."
        )

        top.addWidget(QtWidgets.QLabel("Y-axis:"))
        top.addWidget(self.combo_y)
        top.addSpacing(15)
        top.addWidget(QtWidgets.QLabel("Search:"))
        top.addWidget(self.le_search, 1)
        top.addWidget(self.chk_show_sparse)

        layout.addLayout(top)

        # -------------------------
        # Direction table
        # -------------------------
        self.tbl = QtWidgets.QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["Show", "Direction", "Family", "First L / Å", "Max Y", "Color"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

        layout.addWidget(self.tbl, 1)

        # -------------------------
        # Selection buttons
        # -------------------------
        sel_row = QtWidgets.QHBoxLayout()
        self.btn_default = QtWidgets.QPushButton("Default: min / middle / max")
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")

        sel_row.addWidget(self.btn_default)
        sel_row.addWidget(self.btn_all)
        sel_row.addWidget(self.btn_none)
        sel_row.addStretch(1)

        layout.addLayout(sel_row)

        # -------------------------
        # Style controls
        # -------------------------
        gb_style = QtWidgets.QGroupBox("Style and range")
        form = QtWidgets.QFormLayout(gb_style)

        st = getattr(owner, "_warren_style", {}) or {}

        self.combo_line = QtWidgets.QComboBox()
        self.combo_line.addItems(["-", "--", "-.", ":", "None"])
        self.combo_line.setCurrentText(str(st.get("linestyle", "-")))

        self.combo_marker = QtWidgets.QComboBox()
        self.combo_marker.addItems(["o", "s", "^", "v", "D", "x", "+", ".", "None"])
        self.combo_marker.setCurrentText(str(st.get("marker", "o")))

        self.spin_lw = QtWidgets.QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 10.0)
        self.spin_lw.setSingleStep(0.2)
        self.spin_lw.setValue(float(st.get("linewidth", 1.5)))

        self.spin_ms = QtWidgets.QDoubleSpinBox()
        self.spin_ms.setRange(0.1, 30.0)
        self.spin_ms.setSingleStep(0.5)
        self.spin_ms.setValue(float(st.get("markersize", 5.0)))

        self.le_xmin = QtWidgets.QLineEdit(str(st.get("xmin", "")))
        self.le_xmax = QtWidgets.QLineEdit(str(st.get("xmax", "")))
        self.le_ymin = QtWidgets.QLineEdit(str(st.get("ymin", "")))
        self.le_ymax = QtWidgets.QLineEdit(str(st.get("ymax", "")))

        form.addRow("Line style:", self.combo_line)
        form.addRow("Marker:", self.combo_marker)
        form.addRow("Line width:", self.spin_lw)
        form.addRow("Marker size:", self.spin_ms)
        form.addRow("X min:", self.le_xmin)
        form.addRow("X max:", self.le_xmax)
        form.addRow("Y min:", self.le_ymin)
        form.addRow("Y max:", self.le_ymax)

        layout.addWidget(gb_style)

        # -------------------------
        # Bottom buttons
        # -------------------------
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")

        bottom.addWidget(self.btn_apply)
        bottom.addWidget(self.btn_cancel)

        layout.addLayout(bottom)

        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_default.clicked.connect(self._select_default)
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)

        self.le_search.textChanged.connect(self._apply_search_filter)
        self.combo_y.currentIndexChanged.connect(lambda _=None: self._recompute_max_y_column())
        self.chk_show_sparse.toggled.connect(lambda _=None: self._populate())
        
        # Double-click the color cell to change color.
        self.tbl.cellDoubleClicked.connect(self._on_cell_double_clicked)
        
        self._populate()

    def _family_text(self, d: Tuple[int, int, int]) -> str:
        return self.owner._warren_family_text(d)

    def _dir_label(self, d: Tuple[int, int, int]) -> str:
        return self.owner._warren_label(d)

    def _compact_dir(self, d: Tuple[int, int, int]) -> str:
        try:
            h, k, l = d
            return f"{h}{k}{l}".replace("-", "")
        except Exception:
            return ""

    def _max_y_for_dir(self, d: Tuple[int, int, int]) -> float:
        cache = getattr(self.owner, "_warren_display_cache", {}) or {}
        info = cache.get(d, {}) or {}

        mode = str(self.combo_y.currentData() or "warren")

        if mode == "warren":
            return float(info.get("max_warren", float("nan")))

        return float(info.get("max_microstrain", float("nan")))

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """
        Double-click color column to choose a color.

        This avoids creating one QPushButton per row, which is very slow.
        """
        if col != 5:
            return

        it = self.tbl.item(row, 5)
        if it is None:
            return

        old = _safe_mpl_color(it.data(Qt.UserRole), "#1f77b4")

        color = QtWidgets.QColorDialog.getColor(
            QColor(old),
            self,
            "Choose Warren direction color",
        )

        if not color.isValid():
            return

        color_hex = color.name()

        it.setData(Qt.UserRole, color_hex)
        it.setBackground(QColor(color_hex))
        it.setToolTip(f"Double-click to change color\n{color_hex}")


    def _choose_color(self, row: int) -> None:
        btn = self.tbl.cellWidget(row, 5)
        if btn is None:
            return

        old = _safe_mpl_color(btn.property("color"), "#1f77b4")
        col = QtWidgets.QColorDialog.getColor(QColor(old), self, "Choose Warren direction color")

        if not col.isValid():
            return

        color = col.name()
        btn.setProperty("color", color)
        btn.setStyleSheet(f"background-color: {color};")

    def _populate(self) -> None:
        """
        Populate Warren direction table.

        Performance notes:
        - Uses cached Warren metadata from owner._warren_display_cache.
        - Does NOT create QPushButton widgets per row.
        - Hides sparse directions by default.
        - Disables table updates while filling.
        """
        t0 = time.perf_counter()

        cache = getattr(self.owner, "_warren_display_cache", {}) or {}

        keys = sorted(
            cache.keys(),
            key=lambda d: cache.get(d, {}).get("first_L", float("inf")),
        )

        # Hide sparse directions by default.
        show_sparse = bool(self.chk_show_sparse.isChecked())

        if not show_sparse:
            keys = [
                d for d in keys
                if int((cache.get(d, {}) or {}).get("n_points", 0)) > 5
            ]

        selected = set(getattr(self.owner, "_warren_selected_keys", []) or [])

        if not selected:
            selected = set(self.owner._default_warren_keys_min_mid_max())

        self.tbl.setUpdatesEnabled(False)
        self.tbl.blockSignals(True)

        try:
            self.tbl.setRowCount(0)
            self.tbl.setRowCount(len(keys))

            for row, d in enumerate(keys):
                info = cache.get(d, {}) or {}

                first_L = float(info.get("first_L", float("nan")))
                max_y = self._max_y_for_dir(d)
                n_points = int(info.get("n_points", 0))

                # Show checkbox
                chk = QtWidgets.QTableWidgetItem("")
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                chk.setCheckState(Qt.Checked if d in selected else Qt.Unchecked)
                chk.setData(Qt.UserRole, d)
                self.tbl.setItem(row, 0, chk)

                # Direction label, now spaced: <2 0 0>
                it_dir = QtWidgets.QTableWidgetItem(str(info.get("label", self._dir_label(d))))
                it_dir.setData(Qt.UserRole, d)
                it_dir.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl.setItem(row, 1, it_dir)

                # Family
                it_family = QtWidgets.QTableWidgetItem(str(info.get("family", self._family_text(d))))
                it_family.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl.setItem(row, 2, it_family)

                # First L
                it_L = QtWidgets.QTableWidgetItem("" if not np.isfinite(first_L) else f"{first_L:.5g}")
                it_L.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_L.setToolTip(f"N points = {n_points}")
                self.tbl.setItem(row, 3, it_L)

                # Max Y
                it_y = QtWidgets.QTableWidgetItem("" if not np.isfinite(max_y) else f"{max_y:.5g}")
                it_y.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_y.setToolTip(f"N points = {n_points}")
                self.tbl.setItem(row, 4, it_y)

                # Color cell as normal item, NOT QPushButton.
                dst = getattr(self.owner, "_warren_dir_style", {}) or {}
                color = _safe_mpl_color(
                    (dst.get(d, {}) or {}).get("color", ""),
                    _COLOR_CYCLE[row % len(_COLOR_CYCLE)],
                )

                it_color = QtWidgets.QTableWidgetItem("")
                it_color.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_color.setData(Qt.UserRole, color)
                it_color.setBackground(QColor(color))
                it_color.setToolTip(f"Double-click to change color\n{color}")
                self.tbl.setItem(row, 5, it_color)

        finally:
            self.tbl.blockSignals(False)
            self.tbl.setUpdatesEnabled(True)

        self._apply_search_filter()

        print(
            f"[WARREN OPTIONS] rows={len(keys)} "
            f"populate={time.perf_counter() - t0:.3f} s"
        )

    def _recompute_max_y_column(self) -> None:
        for row in range(self.tbl.rowCount()):
            it_dir = self.tbl.item(row, 1)
            if it_dir is None:
                continue

            d = it_dir.data(Qt.UserRole)
            if not d:
                continue

            max_y = self._max_y_for_dir(d)
            it_y = self.tbl.item(row, 4)

            if it_y is not None:
                it_y.setText("" if not np.isfinite(max_y) else f"{max_y:.5g}")

    def _apply_search_filter(self) -> None:
        text = str(self.le_search.text() or "").strip().lower()

        for row in range(self.tbl.rowCount()):
            it_dir = self.tbl.item(row, 1)
            it_family = self.tbl.item(row, 2)

            if it_dir is None:
                self.tbl.setRowHidden(row, False)
                continue

            d = it_dir.data(Qt.UserRole)
            label = str(it_dir.text()).lower()
            family = str(it_family.text()).lower() if it_family else ""
            compact = self._compact_dir(d).lower() if d else ""

            if not text:
                show = True
            else:
                show = (
                    text in label
                    or text in family
                    or text in compact
                )

            self.tbl.setRowHidden(row, not show)

    def _select_all(self) -> None:
        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            if it is not None and not self.tbl.isRowHidden(row):
                it.setCheckState(Qt.Checked)

    def _select_none(self) -> None:
        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.Unchecked)

    def _select_default(self) -> None:
        default_keys = set(self.owner._default_warren_keys_min_mid_max())

        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            it_dir = self.tbl.item(row, 1)

            if it is None or it_dir is None:
                continue

            d = it_dir.data(Qt.UserRole)
            it.setCheckState(Qt.Checked if d in default_keys else Qt.Unchecked)

    def apply_to_owner(self) -> None:
        selected = []
        dir_style = dict(getattr(self.owner, "_warren_dir_style", {}) or {})

        for row in range(self.tbl.rowCount()):
            it_chk = self.tbl.item(row, 0)
            it_dir = self.tbl.item(row, 1)

            if it_chk is None or it_dir is None:
                continue

            d = it_dir.data(Qt.UserRole)

            if d is None:
                continue

            if it_chk.checkState() == Qt.Checked:
                selected.append(d)

            it_color = self.tbl.item(row, 5)

            if it_color is not None:
                color = _safe_mpl_color(
                    it_color.data(Qt.UserRole),
                    _COLOR_CYCLE[row % len(_COLOR_CYCLE)],
                )
            else:
                color = _COLOR_CYCLE[row % len(_COLOR_CYCLE)]

            dir_style[d] = {"color": color}

        self.owner._warren_selected_keys = selected
        self.owner._warren_y_mode = str(self.combo_y.currentData() or "warren")

        self.owner._warren_style = {
            "linestyle": "None" if self.combo_line.currentText() == "None" else self.combo_line.currentText(),
            "marker": "None" if self.combo_marker.currentText() == "None" else self.combo_marker.currentText(),
            "linewidth": float(self.spin_lw.value()),
            "markersize": float(self.spin_ms.value()),
            "xmin": self.le_xmin.text().strip(),
            "xmax": self.le_xmax.text().strip(),
            "ymin": self.le_ymin.text().strip(),
            "ymax": self.le_ymax.text().strip(),
        }

        self.owner._warren_dir_style = dir_style

class LocalDynamicsDisplayOptionsDialog(QtWidgets.QDialog):
    """Popup for local dynamics display options."""

    def __init__(self, owner: "MainWindow"):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Local dynamics display options")
        self.resize(800, 560)

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Lambdas + deltas", "both")
        self.combo_mode.addItem("Lambdas only", "lambda")
        self.combo_mode.addItem("Deltas only", "delta")

        mode = getattr(owner, "_local_display_mode", "both")
        idx = self.combo_mode.findData(mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

        top.addWidget(QtWidgets.QLabel("Show:"))
        top.addWidget(self.combo_mode)
        top.addStretch(1)

        layout.addLayout(top)

        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Show", "Pair", "Color", "N points"])
        self.tbl.verticalHeader().setVisible(False)

        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

        layout.addWidget(self.tbl, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all")
        self.btn_none = QtWidgets.QPushButton("Select none")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        gb_style = QtWidgets.QGroupBox("Style and range")
        form = QtWidgets.QFormLayout(gb_style)

        st = getattr(owner, "_local_style", {}) or {}

        self.combo_lam_marker = QtWidgets.QComboBox()
        self.combo_lam_marker.addItems(["o", "s", "^", "v", "D", "x", "+", ".", "None"])
        self.combo_lam_marker.setCurrentText(str(st.get("lambda_marker", "o")))

        self.combo_del_marker = QtWidgets.QComboBox()
        self.combo_del_marker.addItems(["x", "o", "s", "^", "v", "D", "+", ".", "None"])
        self.combo_del_marker.setCurrentText(str(st.get("delta_marker", "x")))

        self.combo_lam_line = QtWidgets.QComboBox()
        self.combo_lam_line.addItems(["-", "--", "-.", ":", "None"])
        self.combo_lam_line.setCurrentText(str(st.get("lambda_linestyle", "-")))

        self.combo_del_line = QtWidgets.QComboBox()
        self.combo_del_line.addItems(["--", "-", "-.", ":", "None"])
        self.combo_del_line.setCurrentText(str(st.get("delta_linestyle", "--")))

        self.spin_lw = QtWidgets.QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 10.0)
        self.spin_lw.setSingleStep(0.2)
        self.spin_lw.setValue(float(st.get("linewidth", 1.5)))

        self.spin_ms = QtWidgets.QDoubleSpinBox()
        self.spin_ms.setRange(0.1, 30.0)
        self.spin_ms.setSingleStep(0.5)
        self.spin_ms.setValue(float(st.get("markersize", 5.0)))

        self.le_xmin = QtWidgets.QLineEdit(str(st.get("xmin", "")))
        self.le_xmax = QtWidgets.QLineEdit(str(st.get("xmax", "")))
        self.le_ymin = QtWidgets.QLineEdit(str(st.get("ymin", "")))
        self.le_ymax = QtWidgets.QLineEdit(str(st.get("ymax", "")))

        form.addRow("Lambda marker:", self.combo_lam_marker)
        form.addRow("Delta marker:", self.combo_del_marker)
        form.addRow("Lambda line:", self.combo_lam_line)
        form.addRow("Delta line:", self.combo_del_line)
        form.addRow("Line width:", self.spin_lw)
        form.addRow("Marker size:", self.spin_ms)
        form.addRow("X min:", self.le_xmin)
        form.addRow("X max:", self.le_xmax)
        form.addRow("Y min:", self.le_ymin)
        form.addRow("Y max:", self.le_ymax)

        layout.addWidget(gb_style)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")

        bottom.addWidget(self.btn_apply)
        bottom.addWidget(self.btn_cancel)
        layout.addLayout(bottom)

        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._populate()

    def _choose_color(self, row: int) -> None:
        btn = self.tbl.cellWidget(row, 2)
        if btn is None:
            return

        old = _safe_mpl_color(btn.property("color"), _COLOR_CYCLE[row % len(_COLOR_CYCLE)])
        col = QtWidgets.QColorDialog.getColor(QColor(old), self, "Choose pair color")

        if not col.isValid():
            return

        color = col.name()
        btn.setProperty("color", color)
        btn.setStyleSheet(f"background-color: {color};")

    def _populate(self) -> None:
        trends = getattr(self.owner, "_local_trends", {}) or {}
        selected_pairs = getattr(self.owner, "_local_selected_pairs", None)
        pair_style = getattr(self.owner, "_local_pair_style", {}) or {}

        pairs = sorted(trends.keys())
        self.tbl.setRowCount(len(pairs))

        for row, pair in enumerate(pairs):
            data = trends.get(pair, {}) or {}
            n = len(data.get("r", []) or [])

            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)

            if selected_pairs is None:
                checked = True
            else:
                checked = pair in selected_pairs

            chk.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.tbl.setItem(row, 0, chk)

            it_pair = QtWidgets.QTableWidgetItem(pair)
            it_pair.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(row, 1, it_pair)

            color = _safe_mpl_color(
                (pair_style.get(pair, {}) or {}).get("color", ""),
                standard_pair_color(pair, row),
            )

            btn_color = QtWidgets.QPushButton("Color")
            btn_color.setProperty("color", color)
            btn_color.setStyleSheet(f"background-color: {color};")
            btn_color.clicked.connect(lambda _=None, r=row: self._choose_color(r))
            self.tbl.setCellWidget(row, 2, btn_color)

            it_n = QtWidgets.QTableWidgetItem(str(n))
            it_n.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(row, 3, it_n)

    def _select_all(self) -> None:
        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.Checked)

    def _select_none(self) -> None:
        for row in range(self.tbl.rowCount()):
            it = self.tbl.item(row, 0)
            if it is not None:
                it.setCheckState(Qt.Unchecked)

    def apply_to_owner(self) -> None:
        selected = []
        dir_style = dict(getattr(self.owner, "_warren_dir_style", {}) or {})
    
        for row in range(self.tbl.rowCount()):
            it_chk = self.tbl.item(row, 0)
            it_dir = self.tbl.item(row, 1)
    
            if it_chk is None or it_dir is None:
                continue
            
            d = it_dir.data(Qt.UserRole)
    
            if d is None:
                continue
            
            if it_chk.checkState() == Qt.Checked:
                selected.append(d)
    
            it_color = self.tbl.item(row, 5)
    
            if it_color is not None:
                color = _safe_mpl_color(
                    it_color.data(Qt.UserRole),
                    _COLOR_CYCLE[row % len(_COLOR_CYCLE)],
                )
            else:
                color = _COLOR_CYCLE[row % len(_COLOR_CYCLE)]
    
            dir_style[d] = {"color": color}
    
        self.owner._warren_selected_keys = selected
        self.owner._warren_y_mode = str(self.combo_y.currentData() or "warren")
    
        self.owner._warren_style = {
            "linestyle": "None" if self.combo_line.currentText() == "None" else self.combo_line.currentText(),
            "marker": "None" if self.combo_marker.currentText() == "None" else self.combo_marker.currentText(),
            "linewidth": float(self.spin_lw.value()),
            "markersize": float(self.spin_ms.value()),
            "xmin": self.le_xmin.text().strip(),
            "xmax": self.le_xmax.text().strip(),
            "ymin": self.le_ymin.text().strip(),
            "ymax": self.le_ymax.text().strip(),
        }
    
        self.owner._warren_dir_style = dir_style

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Fitting – Real-time Monitor")

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(
                int(geo.width() * 0.95),
                int(geo.height() * 0.95),
            )
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        else:
            self.resize(1400, 900)

        # -------------------------
        # Top controls - Define top_bar first
        # -------------------------
        top_bar = QtWidgets.QHBoxLayout()
        
        # Create all the buttons first
        btn_new = QtWidgets.QPushButton("New input")
        btn_load = QtWidgets.QPushButton("Load input")
        btn_save = QtWidgets.QPushButton("Save input")
        btn_save_as = QtWidgets.QPushButton("Save as…")
        btn_update = QtWidgets.QPushButton("Update input")
        
        self.btn_calculate = QtWidgets.QPushButton("Calculate")
        self.btn_calculate.setToolTip(
            "Calculate the PDF once using the current input values, without refinement."
        )

        self.btn_run = QtWidgets.QPushButton("▶ Run")
        self.btn_run_cs = QtWidgets.QPushButton("Run CS grid search")
        self.btn_run_cs.setToolTip(
                    "Run discrete crystallite-shape grid search. "
                    "Each shape candidate is tested by refining continuous PDF parameters."
                )
        self.btn_pause = QtWidgets.QPushButton("⏸ Pause")
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop/Save")
        self.btn_clear = QtWidgets.QPushButton("Clear graphs")

        
        
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        self.chk_update_pdf = QtWidgets.QCheckBox("Update PDF during refinement")
        self.chk_update_pdf.setChecked(True)
        
        self.chk_fullgrid_updates = QtWidgets.QCheckBox("Compute full-grid Rwp in updates")
        self.chk_fullgrid_updates.setChecked(False)
        self.chk_fullgrid_updates.setToolTip("More accurate, but slower. When off, Rwp uses the current refinement grid.")
        
        self.chk_zoom_last = QtWidgets.QCheckBox("Zoom last")
        self.chk_zoom_last.setChecked(True)
        self.spin_zoom_last = QtWidgets.QSpinBox()
        self.spin_zoom_last.setRange(1, 500)
        self.spin_zoom_last.setValue(5)
        self.spin_zoom_last.setSuffix(" it")
        self.spin_zoom_last.setToolTip("Keep x-axis focused on last N iterations (nfev).")
        
        self.spin_update_sec = QtWidgets.QDoubleSpinBox()
        self.spin_update_sec.setRange(0.05, 10.0)
        self.spin_update_sec.setSingleStep(0.1)
        self.spin_update_sec.setValue(0.5)
        self.spin_update_sec.setSuffix(" s")
        self.spin_update_sec.setToolTip("How often to push progress updates to the GUI.")
        
        # Now add widgets to top_bar
        # Keep file/input controls and technical options in the top row.
        for w in (btn_new, btn_load, btn_save, btn_save_as, btn_update):
            top_bar.addWidget(w)

        top_bar.addWidget(self.btn_clear)

        top_bar.addStretch(1)

        top_bar.addWidget(QtWidgets.QLabel("Update:"))
        top_bar.addWidget(self.spin_update_sec)
        top_bar.addWidget(self.chk_update_pdf)
        top_bar.addWidget(self.chk_fullgrid_updates)

        top_bar.addSpacing(10)

        top_bar.addWidget(self.chk_zoom_last)
        top_bar.addWidget(self.spin_zoom_last)
        
        # Add the VIEW PDF PLOT button to top_bar
        self.btn_view_pdf = QtWidgets.QPushButton("VIEW PDF PLOT")
        self.btn_view_pdf.setFixedSize(220, 58)
        self.btn_view_pdf.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #2c3e50;
                color: white;
                border: 2px solid #34495e;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #1a252f;
            }
        """)

        # Larger action buttons for the main run/calculate row.
        big_action_style = """
        QPushButton {
            font-size: 15px;
            font-weight: bold;
            padding: 8px 18px;
            min-height: 42px;
            border: 1px solid #9aa4ad;
            border-radius: 8px;
            background-color: #f4f6f7;
        }
        QPushButton:hover {
            background-color: #e8eef2;
        }
        QPushButton:pressed {
            background-color: #d6e0e6;
        }
        QPushButton:disabled {
            color: #9a9a9a;
            background-color: #eeeeee;
            border: 1px solid #cccccc;
        }
        """

        for btn, width in (
            (self.btn_calculate, 135),
            (self.btn_run, 115),
            (self.btn_run_cs, 190),
            (self.btn_pause, 115),
            (self.btn_stop, 150),
        ):
            btn.setMinimumSize(width, 50)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            btn.setStyleSheet(big_action_style)
        
        
        # Create top_wrap widget with the top_bar layout
        top_wrap = QtWidgets.QWidget()
        top_wrap.setLayout(top_bar)

        # Main action row: calculate/run/pause/stop + VIEW PDF PLOT
        view_bar = QtWidgets.QHBoxLayout()
        view_bar.setContentsMargins(0, 8, 0, 8)
        view_bar.setSpacing(12)
        
        view_bar.addStretch(1)
        
        view_bar.addWidget(self.btn_calculate)
        view_bar.addWidget(self.btn_run)
        view_bar.addWidget(self.btn_run_cs)
        view_bar.addWidget(self.btn_pause)
        view_bar.addWidget(self.btn_stop)
        
        view_bar.addSpacing(25)
        
        view_bar.addWidget(self.btn_view_pdf)
        
        view_bar.addStretch(1)
        
        view_wrap = QtWidgets.QWidget()
        view_wrap.setLayout(view_bar)



        # -------------------------
        # Initialize instance variables
        # -------------------------
        self._current_path: Optional[str] = None
        self._worker: Optional[FitWorker] = None
        self._sync_block = False
        self._save_on_stop = False

        self._last_structure_view_path: str = ""
        self._last_structure_crystal_path: str = ""
        self._last_builder_structure_path_for_view: str = ""

        self._pdf_dialog: Optional[PdfPlotDialog] = None
        self._pdf_detached = False
        self._returning_pdf_to_main = False

        # -------------------------
        # LEFT: Builder/Raw tabs
        # RIGHT: Results tabs
        # -------------------------
        self.raw_editor = QtWidgets.QPlainTextEdit()
        self.raw_editor.setPlaceholderText("Load an input file, or build one in the Builder tab…")
        self.builder = InputBuilder()

        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self.builder, "Builder")
        self.left_tabs.addTab(self.raw_editor, "Raw input")

        self.builder.changed.connect(self._on_builder_changed)
        self.raw_editor.textChanged.connect(self._on_raw_changed)

        # Results tabs
        self.tabs = QtWidgets.QTabWidget()

        self.plot_rwp = MplPlot(title="Rwp vs iteration")
        self.rwp_line = None
        self.rwp_x: List[float] = []
        self.rwp_y: List[float] = []
        self._rwp_best_by_nfev: Dict[int, float] = {}

        self._rwp_display_eps = 0.01  # percent Rwp; smaller changes are displayed as flat

        self.plot_warren = MplPlot(title="Warren plot (directional strain)")

        # Warren plot state/controls
        self._warren_raw: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
        self._warren_grouped: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
        self._warren_current_keys: List[Tuple[int, int, int]] = []

        self._warren_selected_keys: List[Tuple[int, int, int]] = []
        self._warren_y_mode = "warren"  # "warren" or "microstrain"

        self._warren_style: Dict[str, Any] = {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.5,
            "markersize": 5.0,
            "xmin": "",
            "xmax": "",
            "ymin": "",
            "ymax": "",
        }

        self._warren_dir_style: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._warren_dialog_window = None

        # Cached Warren display data.
        # This avoids rebuilding NumPy arrays every time the dialog/search/plot is refreshed.
        self._warren_display_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

        self.combo_warren_dir = QtWidgets.QComboBox()
        self.combo_warren_dir.setToolTip(
            "Choose Warren direction family. '<first 3 PDF peaks>' plots the first three "
            "distance families."
        )

        self.combo_warren_y = QtWidgets.QComboBox()
        self.combo_warren_y.addItem("Warren: sqrt(<ΔL²>)", "warren")
        self.combo_warren_y.addItem("Microstrain: sqrt(<ΔL²>)/L", "microstrain")
        self.combo_warren_y.setToolTip("Choose whether to plot Warren broadening or microstrain.")

        self.btn_save_warren_txt = QtWidgets.QPushButton("Save Warren .txt")
        self.btn_save_warren_txt.setToolTip(
            "Save the currently displayed Warren data as tab-separated text."
        )

        self.btn_warren_options = QtWidgets.QPushButton("Display options")
        self.btn_warren_options.setToolTip("Choose Warren directions, search reflections, and edit plot style.")

        self.btn_warren_window = QtWidgets.QPushButton("Open window")
        self.btn_warren_window.setToolTip("Open Warren plot in a separate window.")

        try:
            self.plot_warren.toolbar.addSeparator()
            self.plot_warren.toolbar.addWidget(self.btn_warren_options)
            self.plot_warren.toolbar.addWidget(self.btn_warren_window)

            self.plot_warren.toolbar.addSeparator()
            self.plot_warren.toolbar.addWidget(self.btn_save_warren_txt)
        except Exception:
            pass
        
        self.btn_warren_options.clicked.connect(self.open_warren_display_options)
        self.btn_warren_window.clicked.connect(self.open_warren_plot_window)
        self.btn_save_warren_txt.clicked.connect(self.save_warren_txt)

        self.plot_size = MplPlot(title="Crystallite size distribution")
        self.param_tabs = ParamEvolutionTabs()

        # Crystallite-shape grid-search plots
        self._cs_results: List[Dict[str, Any]] = []

        self.cs_grid_widget = QtWidgets.QWidget()
        _cs_layout = QtWidgets.QVBoxLayout(self.cs_grid_widget)
        _cs_layout.setContentsMargins(0, 0, 0, 0)

        self.cs_grid_tabs = QtWidgets.QTabWidget()

        self.plot_cs_diameter = MplPlot(title="Final Rwp vs diameter")
        self.plot_cs_height = MplPlot(title="Final Rwp vs axis length")
        self.plot_cs_map = MplPlot(title="CS grid search Rwp map")

        self.cs_grid_tabs.addTab(self.plot_cs_diameter, "Rwp vs diameter")
        self.cs_grid_tabs.addTab(self.plot_cs_height, "Rwp vs axis length")
        self.cs_grid_tabs.addTab(self.plot_cs_map, "Rwp map")

        _cs_layout.addWidget(self.cs_grid_tabs, 1)

        # Local lambda / delta trend plot
        self.plot_local_trends = MplPlot(title="Local dynamics trends")
        self._local_trends: Dict[str, Any] = {}
        self._last_refined_params: Dict[str, Any] = {}

        self._local_selected_pairs = None  # None means all pairs
        self._local_display_mode = "both"  # "both", "lambda", "delta"

        self._local_style: Dict[str, Any] = {
            "lambda_marker": "o",
            "delta_marker": "x",
            "lambda_linestyle": "-",
            "delta_linestyle": "--",
            "linewidth": 1.5,
            "markersize": 5.0,
            "xmin": "",
            "xmax": "",
            "ymin": "",
            "ymax": "",
        }

        self._local_pair_style: Dict[str, Dict[str, Any]] = {}
        self._local_dialog_window = None

        self.combo_local_pair = QtWidgets.QComboBox()
        self.combo_local_pair.setToolTip("Select pair type for lambda/delta trend plot.")

        self.combo_local_mode = QtWidgets.QComboBox()
        self.combo_local_mode.addItem("λ used + δ continuation", "combined")
        self.combo_local_mode.setToolTip(
            "Plot lambda coefficients where defined, then delta1/r + delta2/r² "
            "where lambdas are not defined."
        )

        self.btn_local_10A = QtWidgets.QPushButton("10 Å")
        self.btn_local_10A.setToolTip("Recompute local trends only up to 10 Å.")

        self.btn_local_full = QtWidgets.QPushButton("Full range")
        self.btn_local_full.setToolTip(
            "Recompute local trends up to the nanocrystal size d if available; "
            "otherwise up to r_max."
        )

        # ---------------------------------------------------------
        # Local dynamics toolbar buttons
        # ---------------------------------------------------------
        self.btn_local_options = QtWidgets.QPushButton("Display options")
        self.btn_local_options.setToolTip(
            "Choose local-dynamics pairs, mode, colors, markers, line styles, and plot range."
        )

        self.btn_local_window = QtWidgets.QPushButton("Open window")
        self.btn_local_window.setToolTip(
            "Open the local dynamics plot in a separate window."
        )

        try:
            self.plot_local_trends.toolbar.addSeparator()
            self.plot_local_trends.toolbar.addWidget(self.btn_local_options)
            self.plot_local_trends.toolbar.addWidget(self.btn_local_window)

            self.plot_local_trends.toolbar.addSeparator()
            self.plot_local_trends.toolbar.addWidget(QtWidgets.QLabel("Compute:"))
            self.plot_local_trends.toolbar.addWidget(self.btn_local_10A)
            self.plot_local_trends.toolbar.addWidget(self.btn_local_full)

        except Exception:
            pass
        
        self.btn_local_options.clicked.connect(self.open_local_display_options)
        self.btn_local_window.clicked.connect(self.open_local_plot_window)
        self.btn_local_10A.clicked.connect(self.compute_local_trends_10A)
        self.btn_local_full.clicked.connect(self.compute_local_trends_full_range)
       

        # Cache for real-time local trend plotting from GUI edits
        self._local_trend_model = None
        self._local_trend_model_key = None

        self._local_trend_timer = QtCore.QTimer(self)
        self._local_trend_timer.setSingleShot(True)
        self._local_trend_timer.timeout.connect(self._update_local_trends_from_current_gui)

        self.structure_view = StructureViewer()

        self.tabs.addTab(self.plot_rwp, "Rwp")
        self.tabs.addTab(self.cs_grid_widget, "CS grid")
        self.tabs.addTab(self.plot_warren, "Warren")
        self.tabs.addTab(self.plot_size, "Size dist.")
        self.tabs.addTab(self.param_tabs, "Params")
        self.tabs.addTab(self.plot_local_trends, "Local trends")
        self.tabs.addTab(self.structure_view, "Structure")

        # -------------------------
        # Horizontal splitter: LEFT vs RIGHT (default 50/50)
        # -------------------------
        top_split = QtWidgets.QSplitter(Qt.Horizontal)
        top_split.addWidget(self.left_tabs)
        top_split.addWidget(self.tabs)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        top_split.setHandleWidth(8)
        try:
            top_split.setSizes([700, 700])  # 50/50 of 1400px width
        except Exception:
            pass

        # -------------------------
        # Vertical splitter: TOP area vs PDF area
        # Default: PDF area gets ~55-60% of the screen height.
        # -------------------------
        self.main_split = QtWidgets.QSplitter(Qt.Vertical)
        self.main_split.addWidget(top_split)
        self.main_split.setStretchFactor(0, 2)
        self.main_split.setStretchFactor(1, 3)
        self.main_split.setHandleWidth(8)
        try:
            # For a 900 px window: top ~350 px, PDF plot ~550 px
            self.main_split.setSizes([350, 550])
        except Exception:
            pass

        # -------------------------
        # Central layout
        # -------------------------
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addWidget(top_wrap)          # fixed-height toolbar row
        v.addWidget(view_wrap)         # centered VIEW PDF PLOT button row
        v.addWidget(self.main_split, 1)     # resizable area
        self.setCentralWidget(central)

        # -------------------------
        # PDF plot exists ONLY for the pop-up window, not in the main layout
        # -------------------------
        self.plot_pdf = MplPlot(title="PDF fit and residuals")
        self.pdf_line_exp = None
        self.pdf_line_fit = None
        self.pdf_line_res = None
        self.pdf_rwp_text = None
        self._pdf_best_rwp = float("inf")
        # Pair contribution display state
        self._pdf_pair_contrib = {}
        self._pdf_pair_lines = {}
        self._pdf_pair_style = {}
        self._pdf_pair_sum_line = None
        self._pdf_pair_sum_visible = False
        self._pair_display_dialog = None

        # PDF tick-marker state
        self._pdf_tick_data = {}
        self._pdf_tick_collections = []
        self._pdf_tick_lookup = []
        self._pdf_tick_annotation = None
        self._pdf_tick_motion_cid = None

        # Tick-marker filter state
        # None means all pairs.
        self._tick_filter_pairs = None
        self._tick_filter_direction_mode = "all"
        self._tick_filter_direction_text = ""

        self._tick_range_min = DEFAULT_TICK_MARKER_MIN_R
        self._tick_range_max = DEFAULT_TICK_MARKER_MAX_R

        # Per-pair tick-marker style.
        self._tick_pair_style: Dict[str, Dict[str, Any]] = {}

        # Main-curve display styles
        self._show_gfit = True
        self._gexp_style = {
            "linestyle": "-",
            "marker": "None",
            "linewidth": 1.0,
            "markersize": 0.0,
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "color": "black",
        }

        self.spin_res_offset = QtWidgets.QDoubleSpinBox()
        self.spin_res_offset.setRange(0.0, 5.0)
        self.spin_res_offset.setSingleStep(0.1)
        self.spin_res_offset.setValue(1.0)
        self.spin_res_offset.setToolTip("Residual offset factor")
        self.spin_res_offset.setMaximumWidth(90)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale")
        self.btn_pdf_maximize = QtWidgets.QPushButton("Maximize")
        self.btn_pdf_maximize.setToolTip("Maximize the PDF plot window.")

        self.btn_pdf_fullscreen = QtWidgets.QPushButton("Full screen")
        self.btn_pdf_fullscreen.setToolTip("Show the PDF plot window in full-screen mode.")

        self.btn_pdf_restore = QtWidgets.QPushButton("Restore")
        self.btn_pdf_restore.setToolTip("Restore the PDF plot window to normal size.")
        self.btn_autoscale.setToolTip("Autoscale the PDF fit/residual plot.")

        self.btn_detach_pdf = QtWidgets.QPushButton("Close PDF plot")
        self.btn_detach_pdf.setToolTip("Close/hide the PDF plot window.")

        # Pair-contribution popup button.
        # This replaces the old small "Pairs" toolbar menu.
        self.btn_show_pairs = QtWidgets.QPushButton("Show Pairs")
        self.btn_show_pairs.setToolTip("Open pair-contribution display settings.")
        self.btn_show_pairs.clicked.connect(self.open_pair_display_dialog)

        self.btn_tick_markers = QtWidgets.QCheckBox("Tick markers")
        self.btn_tick_markers.setToolTip(
            "Enable/disable PDF tick markers. "
            "When enabled, a popup opens to choose range, pair filters, direction filters, and style."
        )

        self.chk_show_shape_factor = QtWidgets.QCheckBox("Show shape factor γ(r)")
        self.chk_show_shape_factor.setChecked(False)
        self.chk_show_shape_factor.setToolTip(
            "Show the isotropic average shape factor γ(r) overlaid on the PDF plot.\n"
            "For spherical model: γ(r) = 1 - 1.5(r/d) + 0.5(r/d)³\n"
            "For cylinder/disk: direction-averaged common volume\n"
            "For finite shape: direction-averaged finite coordination ratio"
        )
        

        try:
            # Put Show Pairs directly after the Matplotlib save button.
            self._insert_widget_after_toolbar_action(
                self.plot_pdf.toolbar,
                "save",
                self.btn_show_pairs,
            )

            # Tick marker controls in the PDF plot window.
            self.plot_pdf.toolbar.addWidget(self.btn_tick_markers)
            self.plot_pdf.toolbar.addWidget(self.chk_show_shape_factor)

            self.plot_pdf.toolbar.addSeparator()
            self.plot_pdf.toolbar.addWidget(QtWidgets.QLabel("Residual offset:"))
            self.plot_pdf.toolbar.addWidget(self.spin_res_offset)

            self.plot_pdf.toolbar.addSeparator()
            self.plot_pdf.toolbar.addWidget(self.btn_autoscale)

            self.plot_pdf.toolbar.addSeparator()
            self.plot_pdf.toolbar.addWidget(self.btn_pdf_maximize)
            self.plot_pdf.toolbar.addWidget(self.btn_pdf_fullscreen)
            self.plot_pdf.toolbar.addWidget(self.btn_pdf_restore)

            self.plot_pdf.toolbar.addSeparator()
            self.plot_pdf.toolbar.addWidget(self.btn_detach_pdf)

        except Exception:
            pass
        
        self.pdf_container = QtWidgets.QWidget()
        pdf_layout = QtWidgets.QVBoxLayout(self.pdf_container)
        pdf_layout.setContentsMargins(0, 0, 0, 0)
        pdf_layout.setSpacing(4)
        pdf_layout.addWidget(self.plot_pdf, 1)

        # -------------------------
        # Wiring
        # -------------------------
        btn_new.clicked.connect(self.new_input)
        btn_load.clicked.connect(self.load_input)
        btn_save.clicked.connect(self.save_input)
        btn_save_as.clicked.connect(self.save_input_as)
        btn_update.clicked.connect(self.update_input_from_builder)

        self.btn_calculate.clicked.connect(self.calculate_once)
        self.btn_run.clicked.connect(self.run_fit)
        self.btn_run_cs.clicked.connect(self.run_cs_grid_search)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_and_save)
        self.btn_clear.clicked.connect(self.clear_graphs)

        self.btn_view_pdf.clicked.connect(self.detach_pdf_plot)
        self.btn_tick_markers.toggled.connect(self._on_tick_markers_toggled)
        self.btn_autoscale.clicked.connect(self.autoscale_pdf)
        self.chk_show_shape_factor.toggled.connect(self._toggle_shape_factor_display)

        self.btn_pdf_maximize.clicked.connect(self.maximize_pdf_plot)
        self.btn_pdf_fullscreen.clicked.connect(self.fullscreen_pdf_plot)
        self.btn_pdf_restore.clicked.connect(self.restore_pdf_plot)

        self.btn_detach_pdf.clicked.connect(self.attach_pdf_plot)

        self.spin_res_offset.valueChanged.connect(lambda _: self._refresh_residual_offset())
        self.spin_zoom_last.valueChanged.connect(lambda _: self._on_zoom_changed())
        self.chk_zoom_last.toggled.connect(lambda _: self._on_zoom_changed())


        self._add_shortcuts()
        self._reset_progress_series()

        self._refinable_allowed: set[str] = set()
        self._warren_crystal_system: Optional[str] = None
        self._on_zoom_changed()
        self._update_run_button_modes()

    #------------------------------------------------------
    #TICK MARKERS
    #------------------------------------------------------

    def open_tick_marker_filter_dialog(self) -> None:
        """
        Open tick-marker filter dialog without toggling marker visibility.
        """
        tick_data = getattr(self, "_pdf_tick_data", {}) or {}
        pairs = tick_data.get("pairs", {}) if isinstance(tick_data, dict) else {}

        if not pairs:
            QtWidgets.QMessageBox.information(
                self,
                "No tick data",
                "No tick-marker data are available yet. Use Calculate or Run first.",
            )
            return

        dlg = TickMarkerFilterDialog(self)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            dlg.apply_to_owner()

            if self.btn_tick_markers.isChecked():
                self._refresh_pdf_tick_markers()

    def _tick_data_max_r(self) -> float:
        tick_data = getattr(self, "_pdf_tick_data", {}) or {}
        pairs = tick_data.get("pairs", {}) if isinstance(tick_data, dict) else {}

        mx = 0.0

        for pdata in pairs.values():
            try:
                rr = np.asarray(pdata.get("r", []), dtype=float)
                rr = rr[np.isfinite(rr)]
                if rr.size:
                    mx = max(mx, float(np.max(rr)))
            except Exception:
                pass

        return mx


    def _calculate_tick_markers_to_r(self, max_r: float) -> bool:
        """
        Calculate tick-marker metadata up to max_r.

        This replaces the old separate 'Full tick range' button.
        """
        try:
            max_r = float(max_r)

            if not np.isfinite(max_r) or max_r <= 0.0:
                max_r = DEFAULT_TICK_MARKER_MAX_R

            txt = self._compose_input_text_for_run()

            if not txt.strip():
                QtWidgets.QMessageBox.warning(
                    self,
                    "No input",
                    "Please load or create an input file first.",
                )
                return False

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inp",
                mode="w",
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.flush()
            tmp.close()

            try:
                config = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            params = dict(config.get("initial", {}) or {})

            if isinstance(getattr(self, "_last_refined_params", None), dict):
                params.update(self._last_refined_params)

            constraints = config.get("constraints", {}) or {}

            if constraints:
                try:
                    params = apply_constraints(params, constraints)
                except Exception:
                    pass

            self.statusBar().showMessage(
                f"Calculating tick markers up to {max_r:.3g} Å..."
            )
            QtWidgets.QApplication.processEvents()

            r_dummy = np.linspace(
                float(config.get("r_min", 0.0)),
                max_r,
                10,
            )

            pdf_model = PDFCalculator(
                config,
                config["structure_file"],
                r_dummy,
                pair_cutoff=max_r,
            )

            tick_data = pdf_model.build_pdf_tick_data(
                params,
                max_r=max_r,
                max_ticks_per_pair=None,
            )

            self._set_pdf_tick_data(tick_data)

            self.statusBar().showMessage(
                f"Tick markers calculated up to {max_r:.3g} Å."
            )

            return True

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Tick marker calculation failed",
                str(e),
            )
            return False
    

    # -------------------------
    # Shortcuts
    # -------------------------
    def _add_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.load_input)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_input)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.run_fit)
        QShortcut(QKeySequence("F11"), self, activated=self.fullscreen_pdf_plot)

    def _on_zoom_changed(self) -> None:
        n = int(self.spin_zoom_last.value()) if self.chk_zoom_last.isChecked() else 0
        self.param_tabs.set_zoom_last_n(n)

    # -------------------------
    # Builder <-> Raw sync
    # -------------------------
    def _parse_raw_to_dict(self) -> Optional[Dict[str, Any]]:
        txt = self.raw_editor.toPlainText()
        if not txt.strip():
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".inp", mode="w", encoding="utf-8")
        tmp.write(txt)
        tmp.flush()
        tmp.close()
        try:
            return read_input_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _commit_pending_gui_edits(self) -> None:
        try:
            fw = QtWidgets.QApplication.focusWidget()
            if fw is not None:
                fw.clearFocus()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    def _apply_builder_into_raw(self, *, commit_edits: bool = True) -> None:
        if self._sync_block:
            return

        if commit_edits:
            self._commit_pending_gui_edits()
        self._sync_block = True
        try:
            import configparser

            cfg_raw = configparser.RawConfigParser()
            cfg_raw.optionxform = str
            txt = self.raw_editor.toPlainText()
            if txt.strip():
                cfg_raw.read_string(txt)

            updates = self.builder.to_updates()

            deletes = updates.pop("__delete__", {}) if isinstance(updates, dict) else {}
            del_sections = updates.pop("__delete_sections__", []) if isinstance(updates, dict) else []

            # IMPORTANT:
            # Remove all old lambda keys before writing current GUI lambda keys.
            # Otherwise deleted lambda rows remain in the raw input file.
            self._remove_existing_lambda_keys(cfg_raw)

            # First write/update all current Builder values.
            for sec, kv in updates.items():
                if sec not in cfg_raw:
                    cfg_raw[sec] = {}
                if isinstance(kv, dict):
                    for k, v in kv.items():
                        if isinstance(v, bool):
                            cfg_raw[sec][str(k)] = "true" if v else "false"
                        elif isinstance(v, (tuple, list)) and len(v) == 2:
                            lo = "" if v[0] in (None, "") else str(v[0])
                            hi = "" if v[1] in (None, "") else str(v[1])
                            cfg_raw[sec][str(k)] = f"{lo}, {hi}"
                        else:
                            cfg_raw[sec][str(k)] = str(v)

            # Then apply key deletions.
            # Deletions must come AFTER updates so disabled/model-incompatible
            # parameters cannot be re-added.
            if isinstance(deletes, dict):
                for sec, keys in deletes.items():
                    if not sec or sec not in cfg_raw:
                        continue
                    for k in keys or []:
                        try:
                            cfg_raw[sec].pop(str(k), None)
                        except Exception:
                            pass

            # Finally remove whole sections, e.g. contrast_factors.
            for sec in del_sections or []:
                try:
                    if sec in cfg_raw:
                        cfg_raw.remove_section(sec)
                except Exception:
                    pass            

            with tempfile.NamedTemporaryFile(delete=False, suffix=".inp", mode="w", encoding="utf-8") as tf:
                cfg_raw.write(tf)
                tmp_path = tf.name

            with open(tmp_path, "r", encoding="utf-8") as f:
                out_txt = f.read()

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            self.raw_editor.setPlainText(out_txt)
        finally:
            self._sync_block = False

    def _compose_input_text_for_run(self) -> str:
        """Return an INI text for running a fit without mutating raw editor."""
        self._commit_pending_gui_edits()
        import configparser
        import io

        cfg_raw = configparser.RawConfigParser()
        cfg_raw.optionxform = str

        txt = self.raw_editor.toPlainText()
        if txt.strip():
            cfg_raw.read_string(txt)

        updates = self.builder.to_updates()

        deletes = updates.pop("__delete__", {}) if isinstance(updates, dict) else {}
        del_sections = updates.pop("__delete_sections__", []) if isinstance(updates, dict) else []

        # Remove old lambda keys before writing current lambda keys.
        self._remove_existing_lambda_keys(cfg_raw)

        # First write/update all current Builder values.
        for sec, kv in updates.items():
            if sec not in cfg_raw:
                cfg_raw[sec] = {}
            if isinstance(kv, dict):
                for k, v in kv.items():
                    if isinstance(v, bool):
                        cfg_raw[sec][str(k)] = "true" if v else "false"
                    elif isinstance(v, (tuple, list)) and len(v) == 2:
                        lo = "" if v[0] in (None, "") else str(v[0])
                        hi = "" if v[1] in (None, "") else str(v[1])
                        cfg_raw[sec][str(k)] = f"{lo}, {hi}"
                    else:
                        cfg_raw[sec][str(k)] = str(v)

        # Then apply key deletions.
        if isinstance(deletes, dict):
            for sec, keys in deletes.items():
                if not sec or sec not in cfg_raw:
                    continue
                for k in keys or []:
                    try:
                        cfg_raw[sec].pop(str(k), None)
                    except Exception:
                        pass

        # Finally remove whole sections.
        for sec in del_sections or []:
            try:
                if sec in cfg_raw:
                    cfg_raw.remove_section(sec)
            except Exception:
                pass

        buf = io.StringIO()
        cfg_raw.write(buf)
        return buf.getvalue()

    def update_input_from_builder(self) -> None:
        self._apply_builder_into_raw()
        self.statusBar().showMessage("Updated raw input from Builder.")

    def _on_builder_changed(self) -> None:
        if self._sync_block:
            return

        if self.left_tabs.currentWidget() is self.builder:
            # Auto-sync Builder -> raw input immediately.
            # commit_edits=False avoids stealing focus while the user is typing in a table.
            self._apply_builder_into_raw(commit_edits=False)
            # Optional autosave to disk.
            # Enable this only if you really want every Builder change to overwrite the file.
            if self._current_path:
                try:
                    with open(self._current_path, "w", encoding="utf-8") as f:
                        f.write(self.raw_editor.toPlainText())
                except Exception:
                    pass

            self.statusBar().showMessage("Builder changed: raw input updated.")

            # Do not parse the raw input and refresh structure view on every
            # small Builder edit. Only do it when the structure path changes.
            try:
                structure_path = str(
                    self.builder.le_structure.text()
                    or ""
                ).strip()

                if structure_path != getattr(
                    self,
                    "_last_builder_structure_path_for_view",
                    "",
                ):
                    self._last_builder_structure_path_for_view = structure_path

                    cfg = self._parse_raw_to_dict()

                    if cfg:
                        self._refresh_structure_view(cfg)

            except Exception:
                pass

        self._update_run_button_modes()

            # Do NOT compute shells / local trends here.
            # Shell generation must happen only when Run is clicked.

    def _on_raw_changed(self) -> None:
        if self._sync_block:
            return
        if self.left_tabs.currentWidget() is self.raw_editor:
            cfg = self._parse_raw_to_dict()
            if cfg is None:
                return
            self._sync_block = True
            try:
                self.builder.set_from_config(cfg)
            finally:
                self._sync_block = False
            self._refresh_structure_view(cfg)


        self._update_run_button_modes()
            # Do NOT compute shells / local trends here.
            # Shell generation must happen only when Run is clicked.

    def _refresh_structure_view(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        """
        Refresh the Structure tab, but do NOT reload/reparse the CIF unless the
        structure file path actually changed.

        This avoids expensive CIF parsing and 3D redraws on every small GUI edit.
        """
        try:
            if cfg is None:
                cfg = self._parse_raw_to_dict()

            if not cfg:
                self.structure_view.clear()
                self._warren_crystal_system = None
                self._last_structure_view_path = ""
                self._last_structure_crystal_path = ""
                return

            files = cfg.get("files", {}) if isinstance(cfg.get("files", {}), dict) else {}

            path = str(
                files.get(
                    "structure_file",
                    cfg.get("structure_file", ""),
                )
                or ""
            ).strip()

            if not path:
                self.structure_view.clear()
                self._warren_crystal_system = None
                self._last_structure_view_path = ""
                self._last_structure_crystal_path = ""
                return

            try:
                path_abs = os.path.abspath(path)
            except Exception:
                path_abs = path

            # ---------------------------------------------------------
            # Expensive operation: only reload structure if path changed.
            # ---------------------------------------------------------
            if path_abs != getattr(self, "_last_structure_view_path", ""):
                self.structure_view.load_structure(path)
                self._last_structure_view_path = path_abs

            # Cheap display update. Keep this allowed on every change.
            try:
                self.structure_view.set_biso_by_species(
                    cfg.get("biso_by_species", {}) or {}
                )
            except Exception:
                pass

            # ---------------------------------------------------------
            # Expensive-ish SpacegroupAnalyzer: only redo if path changed.
            # ---------------------------------------------------------
            if path_abs != getattr(self, "_last_structure_crystal_path", ""):
                self._warren_crystal_system = None

                try:
                    if path and os.path.exists(path):
                        if str(path).lower().endswith(".cif"):
                            structure, _asu = read_cif_asu_sites(path)
                        else:
                            structure = Structure.from_file(path)

                        if structure is not None:
                            sga = SpacegroupAnalyzer(structure, symprec=1e-3)
                            self._warren_crystal_system = str(
                                sga.get_crystal_system() or ""
                            ).lower()

                except Exception:
                    self._warren_crystal_system = None

                self._last_structure_crystal_path = path_abs

        except Exception:
            pass

    # -------------------------
    # File operations
    # -------------------------
    def new_input(self):
        try:
            if self._worker is not None and self._worker.isRunning():
                QtWidgets.QMessageBox.information(self, "Refinement running", "Stop the running refinement first.")
                return
        except Exception:
            pass

        self._current_path = None
        self._sync_block = True
        try:
            self.raw_editor.setPlainText("")
        finally:
            self._sync_block = False

        try:
            self.builder.reset_defaults()
        except Exception:
            pass

        self.left_tabs.setCurrentWidget(self.builder)
        self.clear_graphs()
        self.structure_view.clear()
        self.statusBar().showMessage("New input (unsaved). Use 'Save as…' to write to disk.")

        self._update_run_button_modes()

    def load_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open input file", "", "Input files (*.inp *.ini *.txt);;All files (*)"
        )
        if not path:
            return

        # Set current path early (important if callbacks fire during builder load)
        self._current_path = path

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._sync_block = True
                try:
                    self.raw_editor.setPlainText(f.read())
                finally:
                    self._sync_block = False

            cfg = read_input_file(path)

            # Resolve relative paths relative to the input file location
            base_dir = os.path.dirname(os.path.abspath(path))
            try:
                if isinstance(cfg, dict):
                    files = cfg.get("files", {}) if isinstance(cfg.get("files", {}), dict) else {}
                    sf = str(files.get("structure_file", cfg.get("structure_file", "")) or "").strip()
                    gf = str(files.get("gr_data_file", cfg.get("gr_data_file", "")) or "").strip()

                    if sf and not os.path.isabs(sf):
                        sf = os.path.normpath(os.path.join(base_dir, sf))
                    if gf and not os.path.isabs(gf):
                        gf = os.path.normpath(os.path.join(base_dir, gf))

                    if files:
                        files["structure_file"] = sf
                        files["gr_data_file"] = gf
                        cfg["files"] = files
                    else:
                        cfg["structure_file"] = sf
                        cfg["gr_data_file"] = gf
            except Exception:
                pass

            # Load into builder (this should NOT overwrite input values anymore, given the builder fix)
            self._sync_block = True
            try:
                self.builder.set_from_config(cfg)
            finally:
                self._sync_block = False

            self.left_tabs.setCurrentWidget(self.builder)

            # Make raw input consistent with resolved paths (optional but robust)
            self._apply_builder_into_raw()

            self._refresh_structure_view(cfg)
            self.statusBar().showMessage(f"Loaded: {path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

        self._update_run_button_modes()


    def _apply_pdf_base_styles(self):
        """Apply user-selected styles for G_exp and G_fit."""
        try:
            if self.pdf_line_fit is not None:
                self.pdf_line_fit.set_visible(bool(getattr(self, "_show_gfit", True)))
                self.pdf_line_fit.set_color("red")
                self.pdf_line_fit.set_linewidth(1.8)

            if self.pdf_line_exp is not None:
                st = getattr(self, "_gexp_style", {}) or {}

                self.pdf_line_exp.set_linestyle(st.get("linestyle", "None"))

                marker = st.get("marker", "o")
                self.pdf_line_exp.set_marker("" if marker == "None" else marker)

                self.pdf_line_exp.set_linewidth(float(st.get("linewidth", 1.0)))
                self.pdf_line_exp.set_markersize(float(st.get("markersize", 4.0)))
                self.pdf_line_exp.set_color(st.get("color", "black"))
                self.pdf_line_exp.set_markerfacecolor(st.get("markerfacecolor", "none"))
                self.pdf_line_exp.set_markeredgecolor(st.get("markeredgecolor", "black"))
        except Exception:
            pass


    def save_input(self):
        self._commit_pending_gui_edits()
        if not self._current_path:
            return self.save_input_as()

        if self.left_tabs.currentWidget() is self.builder:
            self._apply_builder_into_raw()
        else:
            cfg = self._parse_raw_to_dict()
            if cfg is not None:
                self._sync_block = True
                try:
                    self.builder.set_from_config(cfg)
                finally:
                    self._sync_block = False

        try:
            with open(self._current_path, "w", encoding="utf-8") as f:
                f.write(self.raw_editor.toPlainText())
            self.statusBar().showMessage(f"Saved: {self._current_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def save_input_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save input file", self._current_path or "input.inp",
            "Input files (*.inp *.ini *.txt);;All files (*)"
        )
        if not path:
            return
        self._current_path = path
        self.save_input()

    # -------------------------
    # Fit lifecycle
    # -------------------------
    def calculate_once(self):
        """
        Calculate the PDF once from the current input values, without refinement.

        This:
        - reads the current Builder/Raw input
        - builds PDFCalculator
        - evaluates G(r)
        - updates the PDF plot
        - shows all numeric parameters in the Params tab
        """
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Refinement running",
                "Stop the running refinement before using Calculate.",
            )
            return

        # Compose current input text without necessarily saving it to disk.
        txt = self.raw_editor.toPlainText()
        if self.left_tabs.currentWidget() is self.builder:
            txt = self._compose_input_text_for_run()

        if not txt.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "No input",
                "Please load or create an input file first.",
            )
            return

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".inp",
            mode="w",
            encoding="utf-8",
        )

        tmp.write(txt)
        tmp.flush()
        tmp.close()

        try:
            self.statusBar().showMessage("Calculating PDF from input values...")
            QtWidgets.QApplication.processEvents()

            config = read_input_file(tmp.name)

            # Load experimental G(r)
            r_exp, G_exp = np.loadtxt(config["gr_data_file"], unpack=True)

            r_min = float(config.get("r_min", 1.0))
            r_max = float(config.get("r_max", float(np.max(r_exp))))
            r_extension = float(config.get("r_extension", 1.2))
            pair_cutoff = r_max * r_extension

            mask = (r_exp >= r_min) & (r_exp <= r_max)
            r_exp = r_exp[mask]
            G_exp = G_exp[mask]

            # Build model
            pdf_model = PDFCalculator(
                config,
                config["structure_file"],
                r_exp,
                pair_cutoff=pair_cutoff,
            )

            # Input parameters
            params = dict(config.get("initial", {}) or {})

            # Apply user constraints, if present
            constraints = config.get("constraints", {}) or {}
            if constraints:
                params = apply_constraints(params, constraints)

            # Apply lattice constraints for display/reporting
            params_display = apply_model_lattice_constraints_to_params(
                params,
                pdf_model,
            )

            refcfg = config.get("refinement", {}) or {}

            def _cfg_bool_local(val, default=False):
                if val is None:
                    return bool(default)
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(int(val))
                if isinstance(val, str):
                    s = val.strip().lower()
                    if s in ("1", "true", "yes", "y", "on"):
                        return True
                    if s in ("0", "false", "no", "n", "off"):
                        return False
                return bool(default)

            compute_pair_contrib = _cfg_bool_local(
                refcfg.get("compute_pair_contributions_on_finish", False),
                False,
            )

            compute_warren = _cfg_bool_local(
                refcfg.get("compute_warren_on_finish", False),
                False,
            )

            compute_local_trends = _cfg_bool_local(
                refcfg.get("compute_local_trends_on_finish", False),
                False,
            )


            # Evaluate once.
            #
            # The cylinder/disk shape-factor plotting calculation is
            # comparatively expensive. Compute it only when its display
            # checkbox is enabled.
            calculate_gamma_avg = bool(
                self.chk_show_shape_factor.isChecked()
            )

            t_evaluate = time.perf_counter()

            print(
                "[CALCULATE] Starting PDF evaluation "
                f"(shape factor={calculate_gamma_avg})..."
            )

            result = pdf_model.evaluate(
                params,
                return_contributions=compute_pair_contrib,
                compute_gamma_avg=calculate_gamma_avg,
            )

            print(
                "[CALCULATE] PDF evaluation completed in "
                f"{time.perf_counter() - t_evaluate:.3f} s"
            )

            r = np.asarray(result["r"], dtype=float)
            G_fit = np.asarray(result["G_r"], dtype=float)

            # Do NOT build tick markers during Calculate.
            # For large CIFs this can be expensive and freezes the GUI.
            # Tick markers will be calculated only when the user enables the Tick markers checkbox.
            tick_data = {}

            if G_fit.shape != G_exp.shape:
                # Fallback in case the model r-grid differs
                G_exp_use = np.interp(r, r_exp, G_exp)
            else:
                G_exp_use = G_exp

            res = G_exp_use - G_fit

            den = float(np.sum(G_exp_use * G_exp_use))
            num = float(np.sum(res * res))
            rwp = 100.0 * float(np.sqrt(num / den)) if den > 0.0 else float("nan")

            # Clear previous progress plots and show parameters
            self._reset_progress_series()
            self.param_tabs.clear_all()

            # ---------------------------------------------------------
            # Show only refinable parameters in the Params tab.
            #
            # Do not create one Matplotlib canvas for every fixed atomic
            # coordinate, occupancy, and Biso parameter. Large structures
            # can contain hundreds or thousands of these parameters.
            # ---------------------------------------------------------
            refinable_flags = config.get("refinable", {}) or {}

            refinable_names = {
                str(name)
                for name, enabled in refinable_flags.items()
                if bool(enabled)
            }

            numeric_params = {}

            for name in sorted(refinable_names):
                if name not in params_display:
                    continue

                try:
                    numeric_params[name] = float(
                        params_display[name]
                    )
                except Exception:
                    pass

            self._refinable_allowed = set(numeric_params.keys())

            for name in sorted(numeric_params.keys()):
                self.param_tabs.ensure_param(name)

            self.param_tabs.push(
                1,
                numeric_params,
                allowed=self._refinable_allowed,
            )

            # Update PDF plot with shape factor
            gamma_avg = result.get("gamma_avg", None)
            self._update_pdf_plot_with_shape_factor(
                r,
                G_exp_use,
                G_fit,
                res,
                gamma_avg=gamma_avg,
                title_suffix="(calculated from input)",
                rwp=rwp,
            )

            self._set_pdf_tick_data(tick_data)

            

            # Optional pair contributions
            if compute_pair_contrib:
                try:
                    self._set_pdf_pair_contributions(
                        result.get("G_contrib", {}) or {}
                    )
                except Exception:
                    pass
            
            # Size distribution: cheap, always compute if possible.
            try:
                size_dist = self._build_size_distribution_from_params(params)

                if isinstance(size_dist, dict):
                    self._plot_size_dist(size_dist)

            except Exception:
                pass
            
            
            # Optional Warren plot from Calculate.
            if compute_warren:
                try:
                    self.statusBar().showMessage("Computing Warren plot from input values...")
                    QtWidgets.QApplication.processEvents()

                    max_L = pdf_model._plot_distance_limit_from_params(
                        params,
                        fallback=5.0,
                    )

                    warren = pdf_model.build_warren_plot(
                        params,
                        max_L=max_L,
                    )

                    if isinstance(warren, dict) and warren:
                        self._plot_warren(warren)

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Warren calculation failed",
                        str(e),
                    )


            # Optional local trends from Calculate.
            if compute_local_trends:
                try:
                    self.statusBar().showMessage("Computing local dynamics trends from input values...")
                    QtWidgets.QApplication.processEvents()

                    local_trends = pdf_model.build_local_trends(
                        params,
                        max_r_plot=10.0,
                    )

                    if isinstance(local_trends, dict):
                        self._plot_local_trends(local_trends)

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Local dynamics calculation failed",
                        str(e),
                    )

            # ---------------------------------------------------------
            # Update the structure view only if the structure path changed.
            #
            # Rebuilding PyVista sphere glyphs and silhouettes can be
            # expensive and is unrelated to calculating the PDF.
            # ---------------------------------------------------------
            try:
                structure_path = str(
                    config.get("structure_file", "")
                    or ""
                ).strip()

                if structure_path:
                    try:
                        structure_path_abs = os.path.abspath(
                            structure_path
                        )
                    except Exception:
                        structure_path_abs = structure_path

                    previous_path = str(
                        getattr(
                            self,
                            "_last_structure_view_path",
                            "",
                        )
                        or ""
                    )

                    if structure_path_abs != previous_path:
                        self._refresh_structure_view(config)

            except Exception:
                pass

            # Open the PDF window so the user immediately sees the calculation.
            try:
                self.detach_pdf_plot()
            except Exception:
                pass

            self.statusBar().showMessage(
                f"Calculated PDF from input values. Rwp = {rwp:.3f}%"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Calculate failed",
                str(e),
            )

        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _current_shape_mode(self) -> str:
        """
        Return current crystallite-shape mode.
    
        Prefer the Builder state when the Builder exists, because the raw editor may
        not yet have been synchronized after pressing Apply in the shape dialog.
        """
        try:
            if hasattr(self, "builder") and self.builder is not None:
                spec = getattr(self.builder, "_crystallite_shape_spec", {}) or {}
    
                use_shape_tab = (
                    hasattr(self.builder, "combo_size_model")
                    and self.builder.combo_size_model.currentIndex() == 2
                )
                
                if use_shape_tab and spec:
                    return "finite_shape"
        except Exception:
            pass
        
        try:
            txt = self.raw_editor.toPlainText()
    
            if self.left_tabs.currentWidget() is self.builder:
                txt = self._compose_input_text_for_run()
    
            if not txt.strip():
                return "conventional"
    
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inp",
                mode="w",
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.flush()
            tmp.close()
    
            try:
                cfg = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
                
            shape_cfg = cfg.get("crystallite_shape", {}) or {}
            return str(shape_cfg.get("mode", "conventional")).strip().lower()
    
        except Exception:
            return "conventional"

    def _update_run_button_modes(self) -> None:
        running = bool(self._worker is not None and self._worker.isRunning())

        if running:
            self.btn_run.setEnabled(False)
            self.btn_run_cs.setEnabled(False)
            return

        mode = self._current_shape_mode()
        finite = mode == "finite_shape"

        search_mode = self._current_shape_search_mode()

        self.btn_run.setEnabled(not finite)
        self.btn_run_cs.setEnabled(finite)

        if finite and search_mode == "predefined":
            self.btn_run_cs.setText("Run finite-shape fit")
            self.btn_run_cs.setToolTip(
                "Run one refinement using the pre-defined crystallite dimensions."
            )
        else:
            self.btn_run_cs.setText("Run CS grid search")
            self.btn_run_cs.setToolTip(
                "Run discrete crystallite-shape grid search. "
                "Each shape candidate is tested by refining continuous PDF parameters."
            )

    def run_cs_grid_search(self):
        """
        Run finite crystallite-shape discrete grid search.
        """
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Already running",
                "A refinement/grid search is already running.",
            )
            return

        txt = self.raw_editor.toPlainText()

        if self.left_tabs.currentWidget() is self.builder:
            txt = self._compose_input_text_for_run()

        if not txt.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "No input",
                "Please load or create an input file first.",
            )
            return

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".inp",
            mode="w",
            encoding="utf-8",
        )
        tmp.write(txt)
        tmp.flush()
        tmp.close()

        from .cs_grid_worker import CSGridSearchWorker

        self._cs_results = []

        try:
            self.plot_cs_diameter.clear()
            self.plot_cs_height.clear()
            self.plot_cs_map.clear()
        except Exception:
            pass

        self._worker = CSGridSearchWorker(
            input_path=tmp.name,
            source_path=self._current_path,
            progress_every_sec=float(self.spin_update_sec.value()),
            update_pdf=bool(self.chk_update_pdf.isChecked()),
            ask_after_each_shape=bool(getattr(self, "_ask_after_each_shape", False)),
        )

        self._worker.progress.connect(self.on_cs_grid_progress)
        self._worker.finished_ok.connect(self.on_cs_grid_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.status.connect(lambda s: self.statusBar().showMessage(s))

        self.btn_calculate.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run_cs.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("⏸ Pause")

        self.tabs.setCurrentWidget(self.cs_grid_widget)

        self._worker.start()

    @QtCore.Slot(dict)
    def on_cs_grid_progress(self, payload: Dict[str, Any]):
        """
        Receive live progress or completed shape result from CSGridSearchWorker.
        """
        # ---------------------------------------------------------
        # Live inner-refinement update
        # ---------------------------------------------------------
        if "nfev" in payload:
            try:
                nfev = int(payload.get("nfev", 0))
                rwp = float(payload.get("rwp", np.nan))

                params = payload.get("params", {}) or {}

                self.param_tabs.push(
                    nfev,
                    params,
                    allowed=self._refinable_allowed if self._refinable_allowed else None,
                )

                # Rwp plot.
                if nfev > 0 and np.isfinite(rwp):
                    key = int(nfev)
                    self._rwp_best_by_nfev[key] = float(payload.get("best_rwp", rwp))

                    xs = sorted(self._rwp_best_by_nfev.keys())
                    ys = [self._rwp_best_by_nfev[x] for x in xs]

                    self.rwp_x = xs
                    self.rwp_y = ys

                    if self.rwp_line is not None:
                        self.rwp_line.set_data(self.rwp_x, self.rwp_y)
                        self.plot_rwp.ax.relim()
                        self.plot_rwp.ax.autoscale_view(True, True, True)
                        self.plot_rwp.canvas.draw_idle()

                # PDF plot.
                if "r" in payload and "G_fit" in payload and "G_exp" in payload:
                    r = np.asarray(payload["r"], dtype=float)
                    G_exp = np.asarray(payload["G_exp"], dtype=float)
                    G_fit = np.asarray(payload["G_fit"], dtype=float)
                    res = np.asarray(payload.get("residual", G_exp - G_fit), dtype=float)

                    shape_txt = (
                        f"D={payload.get('diameter_cells', '?')}, "
                        f"H={payload.get('height_cells', '?')}, "
                        f"nfev={payload.get('nfev', '?')}"
                    )

                    self._update_pdf_plot(
                        r,
                        G_exp,
                        G_fit,
                        res,
                        title_suffix=f"(finite shape: {shape_txt})",
                        rwp=rwp,
                    )

                    try:
                        self.detach_pdf_plot()
                    except Exception:
                        pass

                self.statusBar().showMessage(
                    f"Finite-shape refinement "
                    f"{int(payload.get('shape_index', 0)) + 1}/"
                    f"{int(payload.get('shape_total', 1))}: "
                    f"D={payload.get('diameter_cells')}, "
                    f"H={payload.get('height_cells')}, "
                    f"nfev={payload.get('nfev')}, "
                    f"Rwp={rwp:.3f}%"
                )

            except Exception:
                pass

            return

        # ---------------------------------------------------------
        # Completed candidate result
        # ---------------------------------------------------------
        try:
            row = {
                "diameter_cells": float(payload.get("diameter_cells", np.nan)),
                "height_cells": float(payload.get("height_cells", np.nan)),
                "rwp": float(payload.get("rwp", np.nan)),
                "params": payload.get("params", {}) or {},
            }

            self._cs_results.append(row)

            self._update_cs_grid_plots()

            best = payload.get("best", None)

            if isinstance(best, dict):
                self.statusBar().showMessage(
                    f"Finite-shape candidate "
                    f"{int(payload.get('index', 0)) + 1}/"
                    f"{int(payload.get('n_total', 0))}: "
                    f"D={payload.get('diameter_cells')}, "
                    f"H={payload.get('height_cells')}, "
                    f"Rwp={float(payload.get('rwp', np.nan)):.3f} %, "
                    f"best={float(best.get('rwp', np.nan)):.3f} %"
                )

        except Exception:
            pass

    # ---------------------------------------------------------
    # Existing completed-shape update below this point
    # ---------------------------------------------------------

    @QtCore.Slot(dict)
    def on_cs_grid_finished(self, result: Dict[str, Any]):
        """
        Called when finite-shape fit / CS grid search finishes.
        """
        self.btn_calculate.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self._worker = None
        self._update_run_button_modes()

        try:
            self._update_cs_grid_plots()
        except Exception:
            pass

        best = result.get("best", None)

        if isinstance(best, dict):
            rwp_val = float(best.get("rwp", np.nan))

            reply = QtWidgets.QMessageBox.question(
                self,
                "Finite-shape result",
                "Best crystallite shape:\n\n"
                f"Diameter = {best.get('diameter_cells')}\n"
                f"Axis length = {best.get('height_cells')}\n"
                f"Rwp = {rwp_val:.3f} %\n\n"
                "Save these refined parameters and shape dimensions to the input file?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )

            if reply == QtWidgets.QMessageBox.Yes:
                self._save_cs_best_result_to_input(best, result)

        self.statusBar().showMessage(
            f"Finite-shape run finished in {float(result.get('elapsed_s', 0.0)):.2f} s"
        )

    def _save_cs_best_result_to_input(self, best: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Save best finite-shape result to the current input file.

        Saves:
            - refined continuous parameters into [initial_values]
            - refine flags into [refinable_parameters]
            - best finite-shape dimensions into [crystallite_shape]
        """
        try:
            target_path = self._current_path

            if not target_path:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save finite-shape refined input file",
                    "input_finite_shape_refined.inp",
                    "Input files (*.inp *.ini *.txt);;All files (*)",
                )

                if not path:
                    return

                target_path = path

            try:
                self._apply_builder_into_raw()
            except Exception:
                pass

            import configparser
            import io

            cfg_raw = configparser.RawConfigParser()
            cfg_raw.optionxform = str

            raw_txt = self.raw_editor.toPlainText()

            if raw_txt.strip():
                cfg_raw.read_string(raw_txt)

            for sec in ("initial_values", "refinable_parameters", "crystallite_shape"):
                if sec not in cfg_raw:
                    cfg_raw[sec] = {}

            params = best.get("params", {}) or {}

            for k, v in params.items():
                kk = str(k).strip()

                if not kk:
                    continue

                cfg_raw["initial_values"][kk] = str(v)

            refinable_flags = result.get("refinable_flags", {}) or {}

            for k, v in refinable_flags.items():
                kk = str(k).strip()

                if not kk:
                    continue

                cfg_raw["refinable_parameters"][kk] = "true" if bool(v) else "false"

            cfg_raw["crystallite_shape"]["mode"] = "finite_shape"
            cfg_raw["crystallite_shape"]["search_mode"] = "predefined"
            cfg_raw["crystallite_shape"]["diameter_cells"] = str(best.get("diameter_cells"))
            cfg_raw["crystallite_shape"]["height_cells"] = str(best.get("height_cells"))
            cfg_raw["crystallite_shape"]["scan_diameter"] = "false"
            cfg_raw["crystallite_shape"]["scan_height"] = "false"
            cfg_raw["crystallite_shape"]["diameter_infinite"] = "false"

            try:
                shape_spec = getattr(self.builder, "_crystallite_shape_spec", {}) or {}

                for key in (
                    "shape_type",
                    "axis_h",
                    "axis_k",
                    "axis_l",
                    "base1_h",
                    "base1_k",
                    "base1_l",
                    "base2_h",
                    "base2_k",
                    "base2_l",
                    "diameter_min_cells",
                    "diameter_max_cells",
                    "height_min_cells",
                    "height_max_cells",
                    "step_cells",
                ):
                    if key in shape_spec:
                        cfg_raw["crystallite_shape"][key] = str(shape_spec[key])

            except Exception:
                pass

            buf = io.StringIO()
            cfg_raw.write(buf)
            out_txt = buf.getvalue()

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(out_txt)

            self._current_path = target_path

            self._sync_block = True
            try:
                self.raw_editor.setPlainText(out_txt)
            finally:
                self._sync_block = False

            try:
                cfg_new = read_input_file(target_path)

                self._sync_block = True
                try:
                    self.builder.set_from_config(cfg_new)
                finally:
                    self._sync_block = False

                self._refresh_structure_view(cfg_new)

            except Exception:
                pass

            self.statusBar().showMessage(
                f"Saved finite-shape result to: {target_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Save finite-shape result failed",
                str(e),
            )

    def run_fit(self):
        """
        Run a full least-squares refinement in the background worker.
        """
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Already running",
                "A refinement is already running.",
            )
            return

        txt = self.raw_editor.toPlainText()

        if self.left_tabs.currentWidget() is self.builder:
            txt = self._compose_input_text_for_run()

        if not txt.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "No input",
                "Please load or paste an input file first.",
            )
            return

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".inp",
            mode="w",
            encoding="utf-8",
        )
        tmp.write(txt)
        tmp.flush()
        tmp.close()

        update_pdf = True
        fullgrid_rwp = bool(self.chk_fullgrid_updates.isChecked())
        progress_sec = float(self.spin_update_sec.value())

        self._reset_progress_series()
        self.param_tabs.clear_all()
        self._save_on_stop = False

        self._refinable_allowed = set()

        try:
            cfg0 = read_input_file(tmp.name)
            refinable = dict(cfg0.get("refinable", {}))

            self._refinable_allowed = {
                str(k)
                for k, v in refinable.items()
                if bool(v)
            }

            for name in sorted(self._refinable_allowed):
                self.param_tabs.ensure_param(name)

            self._refresh_structure_view(cfg0)

        except Exception:
            self._refinable_allowed = set()

        self._worker = FitWorker(
            input_path=tmp.name,
            source_path=self._current_path,
            update_pdf=update_pdf,
            fullgrid_rwp=fullgrid_rwp,
            progress_every_sec=progress_sec,
            activate_tick_markers=bool(self.btn_tick_markers.isChecked()),
        )

        self._worker.progress.connect(self.on_progress)
        self._worker.finished_ok.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.status.connect(lambda s: self.statusBar().showMessage(s))

        self.btn_calculate.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run_cs.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("⏸ Pause")

        self._worker.start()


    def toggle_pause(self):
        if self._worker is None:
            return
        if self._worker.is_paused:
            self._worker.resume()
            self.btn_pause.setText("⏸ Pause")
            self.statusBar().showMessage("Resumed")
        else:
            self._worker.pause()
            self.btn_pause.setText("▶ Resume")
            self.statusBar().showMessage("Paused")

    def stop_and_save(self):
        if self._worker is None or (not self._worker.isRunning()):
            return

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Stop refinement")
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Stop the refinement now?")
        msg.setInformativeText("Choose whether to stop immediately, or stop and save the current best parameters to the input file.")
        btn_stop = msg.addButton("Stop", QtWidgets.QMessageBox.AcceptRole)
        btn_stop_save = msg.addButton("Stop && Save", QtWidgets.QMessageBox.DestructiveRole)
        btn_cancel = msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        msg.setDefaultButton(btn_cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_cancel:
            return

        save_best = (clicked == btn_stop_save)
        self._save_on_stop = bool(save_best)

        try:
            self._worker.request_stop(save_best=save_best)
        except TypeError:
            self._worker.request_stop()

    # -------------------------
    # Plot updates
    # -------------------------
    def _reset_progress_series(self):
        self._pdf_best_rwp = float("inf")
        self._rwp_best_by_nfev = {}
        self.rwp_x.clear()
        self.rwp_y.clear()
        self.plot_rwp.ax.clear()
        self.plot_rwp.ax.set_title("Minimization (Rwp vs nfev)")
        self.plot_rwp.ax.set_xlabel("nfev")
        self.plot_rwp.ax.set_ylabel("Rwp (%)")
        (self.rwp_line,) = self.plot_rwp.ax.plot([], [])
        self.plot_rwp.canvas.draw_idle()

    def clear_graphs(self):
        self._reset_progress_series()

        self._warren_raw = {}
        self._warren_grouped = {}
        self._warren_current_keys = []

        try:
            self.combo_warren_dir.blockSignals(True)
            self.combo_warren_dir.clear()
            self.combo_warren_dir.blockSignals(False)
        except Exception:
            pass

        self.plot_warren.clear()
        self.plot_size.clear()

        try:
            self._local_trends = {}
            self._local_selected_pairs = None
            self._local_pair_style = {}
            self._local_display_mode = "both"

            try:
                self.combo_local_pair.blockSignals(True)
                self.combo_local_pair.clear()
                self.combo_local_pair.blockSignals(False)
            except Exception:
                pass
            
            self.plot_local_trends.clear()
        except Exception:
            pass

        self.param_tabs.clear_all()

        self.plot_pdf.clear()
        self.pdf_line_exp = None
        self.pdf_line_fit = None
        self.pdf_line_res = None
        self.pdf_rwp_text = None

        # Clear tick markers.
        self._clear_pdf_tick_markers()
        self._pdf_tick_data = {}

        # Clear pair contributions.
        self._pdf_pair_contrib = {}
        self._pdf_pair_lines = {}
        self._pdf_pair_style = {}
        self._pdf_pair_sum_line = None
        self._pdf_pair_sum_visible = False

        try:
            if self._pair_display_dialog is not None:
                self._pair_display_dialog.close()
                self._pair_display_dialog = None
        except Exception:
            pass

    def _toggle_shape_factor_display(self, checked):
        """
        Show/hide the DShaper-style negative shape term on the main G(r) axis.

        The plotted curve is not dimensionless gamma(r). It is:

            -scale * 4*pi*rho0*r*gamma(r)

        so it has the same units as G(r).
        """
        try:
            if hasattr(self, "pdf_line_gamma") and self.pdf_line_gamma is not None:
                self.pdf_line_gamma.set_visible(bool(checked))

                self.plot_pdf.ax.relim(visible_only=True)
                self.plot_pdf.ax.autoscale_view(True, True, True)

                try:
                    self.plot_pdf.ax.legend(loc="best")
                except Exception:
                    pass

                self.plot_pdf.canvas.draw_idle()

        except Exception:
            pass

        
    def autoscale_pdf(self):
        self.plot_pdf.ax.relim()
        self.plot_pdf.ax.autoscale(True, axis="both", tight=True)
        self.plot_pdf.canvas.draw_idle()

    def toggle_pdf_window(self):
        """Move PDF plot between main GUI and floating window."""
        if self._pdf_detached:
            self.attach_pdf_plot()
        else:
            self.detach_pdf_plot()


    def detach_pdf_plot(self):
        """Show the PDF plot/control panel in a separate pop-up window."""
        try:
            if self._pdf_dialog is None:
                self._pdf_dialog = PdfPlotDialog(self)

            lay = self._pdf_dialog.layout()
            if lay is not None and self.pdf_container.parent() is not self._pdf_dialog:
                lay.addWidget(self.pdf_container)

            self._pdf_detached = True

            self._pdf_dialog.show()
            self._pdf_dialog.raise_()
            self._pdf_dialog.activateWindow()

            self.statusBar().showMessage("PDF plot opened in separate window.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open PDF plot failed", str(e))


    def attach_pdf_plot(self):
        """Hide the PDF plot window. Do NOT return it to the main GUI."""
        try:
            self._returning_pdf_to_main = True

            self._pdf_detached = False

            if self._pdf_dialog is not None:
                self._pdf_dialog.hide()

            self.statusBar().showMessage("PDF plot window closed.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Close PDF plot failed", str(e))
        finally:
            self._returning_pdf_to_main = False

    def maximize_pdf_plot(self) -> None:
        """Maximize the floating PDF plot window."""
        try:
            self.detach_pdf_plot()

            if self._pdf_dialog is not None:
                self._pdf_dialog.showMaximized()
                self._pdf_dialog.raise_()
                self._pdf_dialog.activateWindow()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Maximize PDF plot failed",
                str(e),
            )


    def fullscreen_pdf_plot(self) -> None:
        """Show the floating PDF plot window in full-screen mode."""
        try:
            self.detach_pdf_plot()

            if self._pdf_dialog is not None:
                self._pdf_dialog.showFullScreen()
                self._pdf_dialog.raise_()
                self._pdf_dialog.activateWindow()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Full-screen PDF plot failed",
                str(e),
            )


    def restore_pdf_plot(self) -> None:
        """Restore the floating PDF plot window to normal size."""
        try:
            self.detach_pdf_plot()

            if self._pdf_dialog is None:
                return

            self._pdf_dialog.showNormal()

            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()

                w = int(geo.width() * 0.85)
                h = int(geo.height() * 0.85)

                self._pdf_dialog.resize(w, h)
                self._pdf_dialog.move(
                    geo.center().x() - w // 2,
                    geo.center().y() - h // 2,
                )
            else:
                self._pdf_dialog.resize(1200, 750)

            self._pdf_dialog.raise_()
            self._pdf_dialog.activateWindow()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Restore PDF plot failed",
                str(e),
            )


    
    def _refresh_residual_offset(self):
        pass

    # -------------------------
    # Slots from worker
    # -------------------------
    @QtCore.Slot(dict)
    def on_progress(self, payload: Dict[str, Any]):
        try:
            nfev = int(payload.get("nfev", 0))
        except Exception:
            nfev = 0

        try:
            y = float(payload.get("best_rwp", payload.get("rwp", np.nan)))
        except Exception:
            y = float("nan")

        if nfev > 0:
            self._rwp_best_by_nfev[nfev] = y

        xs = sorted(self._rwp_best_by_nfev.keys())

        # Build monotonic best-Rwp series first.
        raw_ys: List[float] = []
        ybest = float("inf")
        for x in xs:
            yy = float(self._rwp_best_by_nfev.get(x, float("nan")))
            if np.isfinite(yy) and yy < ybest:
                ybest = yy
            raw_ys.append(ybest if np.isfinite(ybest) else yy)

        # Display smoothing:
        # if the change is below 0.01 % Rwp, keep the previous displayed value.
        ys: List[float] = []
        last_display = None
        eps = float(getattr(self, "_rwp_display_eps", 0.01))

        for yy in raw_ys:
            if not np.isfinite(yy):
                ys.append(yy)
                continue
            
            if last_display is None:
                disp = yy
            elif abs(yy - last_display) < eps:
                disp = last_display
            else:
                disp = yy

            ys.append(disp)
            last_display = disp

        self.rwp_x = xs
        self.rwp_y = ys

        if self.rwp_line is not None:
            self.rwp_line.set_data(self.rwp_x, self.rwp_y)
            self.plot_rwp.ax.relim()
            self.plot_rwp.ax.autoscale_view(True, True, True)

            # Avoid visually amplifying tiny numerical Rwp changes.
            finite_y_all = [float(v) for v in self.rwp_y if np.isfinite(v)]
            if finite_y_all:
                y0, y1 = min(finite_y_all), max(finite_y_all)
                min_span = 0.05  # percent Rwp
                if (y1 - y0) < min_span:
                    yc = 0.5 * (y0 + y1)
                    self.plot_rwp.ax.set_ylim(yc - min_span / 2.0, yc + min_span / 2.0)

            if self.chk_zoom_last.isChecked() and len(self.rwp_x) >= 2:
                n = int(self.spin_zoom_last.value())
                xmax = float(self.rwp_x[-1])
                xmin = max(0.0, xmax - float(n))
                self.plot_rwp.ax.set_xlim(xmin, xmax)
                yy = [yy for xx, yy in zip(self.rwp_x, self.rwp_y) if xx >= xmin]
                if yy:
                    y0, y1 = min(yy), max(yy)
                    if np.isfinite(y0) and np.isfinite(y1):
                        min_span = 0.05  # percent Rwp
                        span = y1 - y0

                        if span < min_span:
                            yc = 0.5 * (y0 + y1)
                            self.plot_rwp.ax.set_ylim(yc - min_span / 2.0, yc + min_span / 2.0)
                        else:
                            pad = 0.05 * span
                            self.plot_rwp.ax.set_ylim(y0 - pad, y1 + pad)

            self.plot_rwp.canvas.draw_idle()

        try:
            self.param_tabs.push(
                nfev,
                payload.get("params", {}) or {},
                allowed=self._refinable_allowed if self._refinable_allowed else None,
            )
        except Exception:
            pass

        if "r" in payload and "G_fit" in payload and "G_exp" in payload:
            try:
                r = np.asarray(payload["r"], dtype=float)
                G_exp = np.asarray(payload["G_exp"], dtype=float)
                G_fit = np.asarray(payload["G_fit"], dtype=float)
                res = np.asarray(payload.get("residual", G_exp - G_fit), dtype=float)

                # IMPORTANT:
                # payload["G_fit"] corresponds to the CURRENT trial parameters.
                # Therefore we must compare using payload["rwp"], not payload["best_rwp"].
                try:
                    rwp_current = float(payload.get("rwp", np.nan))
                except Exception:
                    rwp_current = float("nan")

                # If rwp was not provided or is invalid, compute it from the displayed arrays.
                if not np.isfinite(rwp_current):
                    den = float(np.sum(G_exp * G_exp))
                    num = float(np.sum((G_exp - G_fit) ** 2))
                    rwp_current = 100.0 * float(np.sqrt(num / den)) if den > 0 else float("nan")

                # Only update the main PDF plot when the CURRENT Rwp improves.
                # This makes the PDF plot behave like the Rwp-best plot.
                if np.isfinite(rwp_current) and rwp_current < self._pdf_best_rwp - 1e-6:
                    self._pdf_best_rwp = float(rwp_current)

                    gamma_avg = payload.get("gamma_avg", None)
                    self._update_pdf_plot_with_shape_factor(
                        r,
                        G_exp,
                        G_fit,
                        res,
                        gamma_avg=gamma_avg,
                        title_suffix=(
                            f"(best so far, stage={payload.get('stage','')}, "
                            f"nfev={payload.get('nfev','')})"
                        ),
                        rwp=rwp_current,
                    )

            except Exception:
                pass

    def _update_pdf_plot(
        self,
        r: np.ndarray,
        G_exp: np.ndarray,
        G_fit: np.ndarray,
        res: np.ndarray,
        *,
        title_suffix: str = "",
        rwp: Optional[float] = None,
    ) -> None:
        ax = self.plot_pdf.ax
        if self.pdf_line_exp is None:
            ax.clear()
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("G(r)")
            (self.pdf_line_exp,) = ax.plot(
                r,
                G_exp,
                label="G_exp",
                linestyle="-",
                marker="",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=0.0,
                linewidth=1.0,
                color="black",
            )

            (self.pdf_line_fit,) = ax.plot(
                r,
                G_fit,
                label="G_fit",
                linestyle="-",
                linewidth=1.8,
                color="red",
            )

            (self.pdf_line_res,) = ax.plot(
                r,
                res,
                label="residual (offset)",
                linestyle="-",
                linewidth=1.2,
                color="green",
            )
            ax.legend(loc="best")
        else:
            self.pdf_line_exp.set_data(r, G_exp)
            self.pdf_line_fit.set_data(r, G_fit)

        amp = float(np.nanmax(G_exp) - np.nanmin(G_exp)) if G_exp.size else 1.0
        off = -float(self.spin_res_offset.value()) * 0.25 * amp
        if self.pdf_line_res is not None:
            self.pdf_line_res.set_data(r, res + off)

        ax.set_title("PDF fit and residuals " + title_suffix)
        # Display current Rwp on PDF plot, top-right.
        try:
            if rwp is None or not np.isfinite(float(rwp)):
                den = float(np.sum(G_exp * G_exp))
                num = float(np.sum((G_exp - G_fit) ** 2))
                rwp_val = 100.0 * float(np.sqrt(num / den)) if den > 0 else float("nan")
            else:
                rwp_val = float(rwp)
        
            if self.pdf_rwp_text is None or getattr(self.pdf_rwp_text, "axes", None) is not ax:
                self.pdf_rwp_text = ax.text(
                    0.985,
                    0.985,
                    "",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="0.7",
                        alpha=0.75,
                    ),
                )
        
            if np.isfinite(rwp_val):
                self.pdf_rwp_text.set_text(f"Rwp = {rwp_val:.3f}%")
            else:
                self.pdf_rwp_text.set_text("Rwp = n/a")
        
        except Exception:
            pass

        self._apply_pdf_base_styles()
        self._refresh_pdf_pair_curves()

        ax.relim(visible_only=True)
        ax.autoscale_view(True, True, True)

        # Tick markers depend on current y-limits, so redraw them after autoscale.
        if getattr(self, "btn_tick_markers", None) is not None:
            if self.btn_tick_markers.isChecked():
                self._refresh_pdf_tick_markers()

        self.plot_pdf.canvas.draw_idle()


    def _update_pdf_plot_with_shape_factor(
        self,
        r: np.ndarray,
        G_exp: np.ndarray,
        G_fit: np.ndarray,
        res: np.ndarray,
        gamma_avg: np.ndarray = None,
        *,
        title_suffix: str = "",
        rwp: Optional[float] = None,
    ) -> None:
        """
        Update PDF plot with optional DShaper-style shape term.

        Important:
        gamma_avg is badly named here. It is not dimensionless gamma(r).
        It is the G(r)-scale shape term:

            -scale * 4*pi*rho0*r*gamma(r)

        Therefore it is plotted on the same axis as G(r).
        """
        ax = self.plot_pdf.ax

        r = np.asarray(r, dtype=float)
        G_exp = np.asarray(G_exp, dtype=float)
        G_fit = np.asarray(G_fit, dtype=float)
        res = np.asarray(res, dtype=float)

        if gamma_avg is not None:
            gamma_avg = np.asarray(gamma_avg, dtype=float)
            self._current_gamma_avg = gamma_avg
            self._current_r = r

        if self.pdf_line_exp is None:
            ax.clear()
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("G(r)")

            self.pdf_line_gamma = None
            self.pdf_rwp_text = None

            # Experimental data
            (self.pdf_line_exp,) = ax.plot(
                r,
                G_exp,
                label="G_exp",
                linestyle="-",
                marker="",
                linewidth=1.0,
                color="black",
            )

            # Fitted data
            (self.pdf_line_fit,) = ax.plot(
                r,
                G_fit,
                label="G_fit",
                linestyle="-",
                linewidth=1.8,
                color="red",
            )

            # Residual
            (self.pdf_line_res,) = ax.plot(
                r,
                res,
                label="residual (offset)",
                linestyle="-",
                linewidth=1.2,
                color="green",
            )

            # DShaper-style negative shape term, same axis as G(r)
            if gamma_avg is not None:
                (self.pdf_line_gamma,) = ax.plot(
                    r,
                    gamma_avg,
                    label=r"$-s\,4\pi\rho_0 r\,\gamma(r)$ shape term",
                    linestyle="--",
                    linewidth=2.0,
                    color="blue",
                    alpha=0.7,
                    visible=bool(self.chk_show_shape_factor.isChecked()),
                )

        else:
            self.pdf_line_exp.set_data(r, G_exp)
            self.pdf_line_fit.set_data(r, G_fit)

            if self.pdf_line_res is not None:
                pass

            # This is the exact place where your snippet belongs.
            # It updates the existing shape-term line when new data arrive.
            if gamma_avg is not None:
                if not hasattr(self, "pdf_line_gamma"):
                    self.pdf_line_gamma = None

                if self.pdf_line_gamma is None or getattr(self.pdf_line_gamma, "axes", None) is not ax:
                    (self.pdf_line_gamma,) = ax.plot(
                        r,
                        gamma_avg,
                        label=r"$-s\,4\pi\rho_0 r\,\gamma(r)$ shape term",
                        linestyle="--",
                        linewidth=2.0,
                        color="blue",
                        alpha=0.7,
                        visible=bool(self.chk_show_shape_factor.isChecked()),
                    )
                else:
                    self.pdf_line_gamma.set_data(r, gamma_avg)
                    self.pdf_line_gamma.set_visible(bool(self.chk_show_shape_factor.isChecked()))

            else:
                if hasattr(self, "pdf_line_gamma") and self.pdf_line_gamma is not None:
                    self.pdf_line_gamma.set_visible(False)

        amp = float(np.nanmax(G_exp) - np.nanmin(G_exp)) if G_exp.size else 1.0
        off = -float(self.spin_res_offset.value()) * 0.25 * amp

        if self.pdf_line_res is not None:
            self.pdf_line_res.set_data(r, res + off)

        ax.set_title("PDF fit and residuals " + title_suffix)

        # Display current Rwp on PDF plot, top-right.
        try:
            if rwp is None or not np.isfinite(float(rwp)):
                den = float(np.sum(G_exp * G_exp))
                num = float(np.sum((G_exp - G_fit) ** 2))
                rwp_val = 100.0 * float(np.sqrt(num / den)) if den > 0 else float("nan")
            else:
                rwp_val = float(rwp)

            if self.pdf_rwp_text is None or getattr(self.pdf_rwp_text, "axes", None) is not ax:
                self.pdf_rwp_text = ax.text(
                    0.985,
                    0.985,
                    "",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="0.7",
                        alpha=0.75,
                    ),
                )

            if np.isfinite(rwp_val):
                self.pdf_rwp_text.set_text(f"Rwp = {rwp_val:.3f}%")
            else:
                self.pdf_rwp_text.set_text("Rwp = n/a")

        except Exception:
            pass

        self._apply_pdf_base_styles()
        self._refresh_pdf_pair_curves()

        try:
            ax.legend(loc="best")
        except Exception:
            pass

        ax.relim(visible_only=True)
        ax.autoscale_view(True, True, True)

        # Tick markers depend on current y-limits, so redraw them after autoscale.
        if getattr(self, "btn_tick_markers", None) is not None:
            if self.btn_tick_markers.isChecked():
                self._refresh_pdf_tick_markers()

        self.plot_pdf.canvas.draw_idle()

        
    @QtCore.Slot(dict)
    def on_finished(self, result: Dict[str, Any]):
        self.btn_calculate.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._update_run_button_modes()

        if "r_full" in result and "G_exp_full" in result and "G_fit_full" in result:
            gamma_avg = result.get("gamma_avg", None)
            self._update_pdf_plot_with_shape_factor(
                np.asarray(result["r_full"], float),
                np.asarray(result["G_exp_full"], float),
                np.asarray(result["G_fit_full"], float),
                np.asarray(
                    result.get(
                        "res_full",
                        np.asarray(result["G_exp_full"], float) - np.asarray(result["G_fit_full"], float),
                    ),
                    float,
                ),
                gamma_avg=gamma_avg,
                title_suffix="(final)",
            )

            try:
                self._set_pdf_pair_contributions(result.get("G_contrib", {}) or {})
            except Exception:
                pass
            
        warren = result.get("warren_by_dir")
        if isinstance(warren, dict) and len(warren) > 0:
            self._plot_warren(warren)

        size = result.get("size_distribution")
        if isinstance(size, dict) and "x" in size and "pdf" in size:
            self._plot_size_dist(size)

        local_trends = result.get("local_trends")
        if isinstance(local_trends, dict):
            self._plot_local_trends(local_trends)

        tick_data = result.get("tick_data")
        if isinstance(tick_data, dict):
            self._set_pdf_tick_data(tick_data)

        try:
            self.plot_pdf.canvas.draw()
            self.plot_warren.canvas.draw()
            self.plot_size.canvas.draw()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        refined = result.get("refined_params")
        if isinstance(refined, dict):
            self._last_refined_params = dict(refined)

        stopped = bool(result.get("stopped", False))
        save_on_stop = bool(result.get("save_on_stop", self._save_on_stop))
        input_path = result.get("input_path")
        refinable_flags = result.get("refinable_flags") or {}

        target_path = self._current_path or input_path

        def _apply_to_path(path: str) -> None:
            run_inp = str(input_path) if input_path is not None else ""
            dst = str(path)

            if run_inp and os.path.exists(run_inp) and os.path.abspath(run_inp) != os.path.abspath(dst):
                try:
                    with open(run_inp, "r", encoding="utf-8") as src:
                        run_txt = src.read()
                    dst_dir = os.path.dirname(dst) or "."
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".inp", mode="w", encoding="utf-8", dir=dst_dir) as tf:
                        tf.write(run_txt)
                        tmp_dst = tf.name
                    os.replace(tmp_dst, dst)
                except Exception:
                    pass

            update_input_file_with_refined_params(dst, refined, refinable_flags)

            with open(dst, "r", encoding="utf-8") as f:
                self._sync_block = True
                try:
                    self.raw_editor.setPlainText(f.read())
                finally:
                    self._sync_block = False

            try:
                cfg_new = read_input_file(dst)
                self._sync_block = True
                try:
                    self.builder.set_from_config(cfg_new)
                finally:
                    self._sync_block = False
                self._refresh_structure_view(cfg_new)
            except Exception:
                pass

            self._current_path = dst
            self.statusBar().showMessage(f"Saved refined parameters to: {dst}")

        if refined and target_path:
            if stopped and save_on_stop:
                try:
                    if self._current_path is None:
                        path, _ = QtWidgets.QFileDialog.getSaveFileName(
                            self, "Save refined input file", "input_refined.inp",
                            "Input files (*.inp *.ini *.txt);;All files (*)"
                        )
                        if not path:
                            return
                        target_path = path
                    _apply_to_path(target_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Apply failed", str(e))
            elif not stopped:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Save refined parameters",
                    "Refinement finished. Save refined parameters and replace the input file on disk?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    try:
                        if self._current_path is None:
                            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                                self, "Save refined input file", "input_refined.inp",
                                "Input files (*.inp *.ini *.txt);;All files (*)"
                            )
                            if not path:
                                return
                            target_path = path
                        _apply_to_path(target_path)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "Apply failed", str(e))

        self.statusBar().showMessage(f"Finished in {result.get('elapsed_s', 0):.2f} s")

    def _insert_widget_after_toolbar_action(self, toolbar, text_contains: str, widget):
        """
        Insert a widget immediately after a Matplotlib toolbar action.

        Used to place 'Show Pairs' immediately after the save icon.
        """
        try:
            actions = toolbar.actions()
            text_contains = str(text_contains).lower()

            target_idx = None
            for i, act in enumerate(actions):
                txt = str(act.text() or "").lower()
                tip = str(act.toolTip() or "").lower()
                if text_contains in txt or text_contains in tip:
                    target_idx = i

            if target_idx is None:
                toolbar.addWidget(widget)
                return

            before = actions[target_idx + 1] if target_idx + 1 < len(actions) else None
            toolbar.insertWidget(before, widget)
        except Exception:
            try:
                toolbar.addWidget(widget)
            except Exception:
                pass
    

    def _rebuild_warren_display_cache(self) -> None:
        """
        Precompute Warren plot arrays and metadata once.

        This makes display-option dialogs and redraws much faster because they
        no longer repeatedly convert lists to NumPy arrays and recompute max values.
        """
        cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

        grouped = getattr(self, "_warren_grouped", {}) or {}

        for d, arr in grouped.items():
            try:
                a = np.asarray(arr, dtype=float)

                if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
                    continue

                L = np.asarray(a[:, 0], dtype=float)
                dL2 = np.maximum(np.asarray(a[:, 1], dtype=float), 0.0)

                finite = np.isfinite(L) & np.isfinite(dL2) & (L > 0)

                if not np.any(finite):
                    continue

                L = L[finite]
                dL2 = dL2[finite]

                order = np.argsort(L)
                L = L[order]
                dL2 = dL2[order]

                y_warren = np.sqrt(dL2)
                y_micro = y_warren / np.maximum(L, 1e-12)

                cache[d] = {
                    "L": L,
                    "dL2": dL2,
                    "y_warren": y_warren,
                    "y_microstrain": y_micro,
                    "n_points": int(L.size),
                    "first_L": float(L[0]) if L.size else float("nan"),
                    "max_warren": float(np.nanmax(y_warren)) if y_warren.size else float("nan"),
                    "max_microstrain": float(np.nanmax(y_micro)) if y_micro.size else float("nan"),
                    "label": self._warren_label(d),
                    "family": self._warren_family_text(d),
                }

            except Exception:
                continue

        self._warren_display_cache = cache

    def _canonical_warren_direction(self, direction: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Canonicalize a Warren direction for display, using crystal-system-dependent
        symmetry equivalence.

        This affects only GUI grouping/labels, not the physics.

        Rules:
        - Always reduce by gcd.
        - Always treat opposite directions as equivalent: [hkl] ~ [-h,-k,-l]
        - For cubic: also permute freely -> family <hkl>
        - For tetragonal / hexagonal / trigonal: a,b are equivalent in-plane, but c is special
        - For orthorhombic / monoclinic / triclinic: no permutation, only sign normalization
        """
        try:
            h, k, l = [int(x) for x in direction]
        except Exception:
            return (0, 0, 0)

        if h == 0 and k == 0 and l == 0:
            return (0, 0, 0)

        from math import gcd

        # Reduce by gcd first
        g = gcd(gcd(abs(h), abs(k)), abs(l))
        if g > 0:
            h //= g
            k //= g
            l //= g

        # Make opposite directions equivalent: first nonzero index positive
        for v in (h, k, l):
            if v != 0:
                if v < 0:
                    h, k, l = -h, -k, -l
                break

        cs = (self._warren_crystal_system or "").lower()

        # Cubic: full permutation symmetry
        if cs == "cubic":
            vals = sorted((abs(h), abs(k), abs(l)), reverse=True)
            return tuple(vals)

        # Tetragonal / hexagonal / trigonal:
        # a and b are symmetry-equivalent in plane, but c is distinct
        if cs in ("tetragonal", "hexagonal", "trigonal"):
            ab = sorted((abs(h), abs(k)), reverse=True)
            l_abs = abs(l)
            return (ab[0], ab[1], l_abs)

        # Orthorhombic / monoclinic / triclinic:
        # do not permute axes
        return (h, k, l)


    def _warren_family_text(self, d: Tuple[int, int, int]) -> str:
        h, k, l = d
        ah, ak, al = abs(h), abs(k), abs(l)

        if ah != 0 and ak == 0 and al == 0:
            return "h00"
        if ah == 0 and ak != 0 and al == 0:
            return "0k0"
        if ah == 0 and ak == 0 and al != 0:
            return "00l"
        if ah == ak and al == 0 and ah != 0:
            return "hh0"
        if ah != 0 and ak == 0 and al != 0:
            return "h0l"
        if ah == 0 and ak != 0 and al != 0:
            return "0kl"
        if ah == ak == al and ah != 0:
            return "hhh"
        return "hkl"


    def _warren_label(self, direction: Tuple[int, int, int]) -> str:
        h, k, l = direction
        return f"<{h} {k} {l}>"

    def _group_warren_directions(
        self,
        strain_by_dir: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
    ) -> Dict[Tuple[int, int, int], List[Tuple[float, float]]]:
        """
        Group raw Warren data by equivalent direction family and merge duplicate L values.

        Input values are expected as:
            direction -> [(L, dL2), ...]

        Output:
            canonical_direction -> [(L, average_dL2), ...]
        """
        grouped_tmp: Dict[Tuple[int, int, int], Dict[float, List[float]]] = {}

        for direction, arr in (strain_by_dir or {}).items():
            if not isinstance(direction, tuple) or len(direction) != 3:
                continue
            if direction == (0, 0, 0):
                continue

            canon = self._canonical_warren_direction(direction)
            if canon == (0, 0, 0):
                continue

            grouped_tmp.setdefault(canon, {})

            for item in arr or []:
                try:
                    L = float(item[0])
                    dL2 = float(item[1])
                except Exception:
                    continue

                if not np.isfinite(L) or not np.isfinite(dL2) or L <= 0:
                    continue

                # Merge equivalent rows at the same distance.
                key_L = round(L, 5)
                grouped_tmp[canon].setdefault(key_L, []).append(dL2)

        grouped: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}

        for canon, by_L in grouped_tmp.items():
            rows = []
            for L_key, vals in by_L.items():
                if not vals:
                    continue
                rows.append((float(L_key), float(np.mean(vals))))
            rows.sort(key=lambda x: x[0])
            if rows:
                grouped[canon] = rows

        return grouped

    def _plot_warren(self, strain_by_dir: Dict[Tuple[int, int, int], List[Tuple[float, float]]]):
        """
        Store Warren data, group equivalent directions, populate direction selector,
        and plot the default view: first 3 PDF-peak direction families.
        """
        self._warren_raw = strain_by_dir or {}
        self._warren_grouped = self._group_warren_directions(self._warren_raw)

        # New: cache all display arrays/metadata once.
        self._rebuild_warren_display_cache()

        # Default selection uses the cache and ignores directions with <= 5 points.
        self._warren_selected_keys = self._default_warren_keys_min_mid_max()

        # Sort direction families by first distance. This corresponds to the first PDF peaks.
        sorted_dirs = sorted(
            self._warren_grouped.keys(),
            key=lambda d: self._warren_grouped[d][0][0] if self._warren_grouped.get(d) else float("inf"),
        )

        try:
            self.combo_warren_dir.blockSignals(True)
            self.combo_warren_dir.clear()

            self.combo_warren_dir.addItem("First 3 PDF peaks", "__first3__")

            for d in sorted_dirs:
                first_L = self._warren_grouped[d][0][0]
                self.combo_warren_dir.addItem(f"{self._warren_label(d)}  first L={first_L:.4g} Å", d)

            self.combo_warren_dir.blockSignals(False)
        except Exception:
            pass

        self._refresh_warren_plot()

    def _refresh_warren_plot(self) -> None:
        self._draw_warren_on_ax(self.plot_warren.ax)
        self.plot_warren.canvas.draw_idle()

    def save_warren_txt(self) -> None:
        """Save currently displayed Warren/microstrain data to a tab-separated text file."""
        if not getattr(self, "_warren_current_keys", None):
            QtWidgets.QMessageBox.information(
                self,
                "No Warren data",
                "No Warren data is currently displayed.",
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Warren plot data",
            "warren_plot.txt",
            "Text files (*.txt);;All files (*)",
        )

        if not path:
            return

        try:
            try:
                y_mode = getattr(self, "_warren_y_mode", "warren")
            except Exception:
                y_mode = "microstrain"

            with open(path, "w", encoding="utf-8") as f:
                f.write("# Warren/microstrain plot data\n")

                if y_mode == "warren":
                    f.write("# Mode: Warren plot\n")
                    f.write("# Columns: direction\tL_A\tsqrt_dL2_A\n")
                else:
                    f.write("# Mode: Microstrain\n")
                    f.write("# Columns: direction\tL_A\tmicrostrain\n")

                for d in self._warren_current_keys:
                    arr = self._warren_grouped.get(d, [])
                    label = self._warren_label(d)

                    for L, dL2 in arr:
                        L = float(L)
                        dL2 = float(dL2)

                        sqrt_dL2 = float(np.sqrt(max(dL2, 0.0)))
                        microstrain = float(sqrt_dL2 / max(L, 1e-12))

                        if y_mode == "warren":
                            f.write(f"{label}\t{L:.10g}\t{sqrt_dL2:.10g}\n")
                        else:
                            f.write(f"{label}\t{L:.10g}\t{microstrain:.10g}\n")

            self.statusBar().showMessage(f"Saved Warren data: {path}")

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                str(e),
            )

    def open_pair_display_dialog(self):
        if self._pair_display_dialog is None:
            self._pair_display_dialog = PairDisplayDialog(self)
        else:
            # Rebuild because pair list may have changed after a new refinement.
            self._pair_display_dialog.close()
            self._pair_display_dialog = PairDisplayDialog(self)

        self._pair_display_dialog.show()
        self._pair_display_dialog.raise_()
        self._pair_display_dialog.activateWindow()


    def _set_pdf_pair_contributions(self, pair_contrib: Dict[str, Any]) -> None:
        """
        Store weighted additive pair contributions.
        """
        self._pdf_pair_contrib = {}

        try:
            for k, v in (pair_contrib or {}).items():
                arr = np.asarray(v, dtype=float)
                if arr.size > 0 and np.any(np.isfinite(arr)):
                    self._pdf_pair_contrib[str(k)] = arr
        except Exception:
            self._pdf_pair_contrib = {}

        # Initialize default styles for new pairs, preserving existing user styles.
        if not hasattr(self, "_pdf_pair_style"):
            self._pdf_pair_style = {}

        for pair in self._pdf_pair_contrib:
            self._pdf_pair_style.setdefault(
                pair,
                {
                    "visible": False,
                    "color": "",
                    "linestyle": "--",
                    "marker": "None",
                    "linewidth": 1.2,
                    "alpha": 0.85,
                },
            )

        # Remove styles for pairs no longer present.
        for pair in list(self._pdf_pair_style.keys()):
            if pair not in self._pdf_pair_contrib:
                self._pdf_pair_style.pop(pair, None)

        self._refresh_pdf_pair_curves()

    def _rebuild_pdf_pair_menu(self) -> None:
        """Rebuild checkable pair menu."""
        try:
            self.menu_pdf_pairs.clear()
            self._pdf_pair_actions = {}

            if not self._pdf_pair_contrib:
                act = self.menu_pdf_pairs.addAction("No pair contributions")
                act.setEnabled(False)
                return

            # Convenience actions
            act_all = self.menu_pdf_pairs.addAction("Select all")
            act_none = self.menu_pdf_pairs.addAction("Select none")
            self.menu_pdf_pairs.addSeparator()

            def _select_all():
                for a in self._pdf_pair_actions.values():
                    a.setChecked(True)
                self._refresh_pdf_pair_curves()

            def _select_none():
                for a in self._pdf_pair_actions.values():
                    a.setChecked(False)
                self._refresh_pdf_pair_curves()

            act_all.triggered.connect(_select_all)
            act_none.triggered.connect(_select_none)

            for pair in sorted(self._pdf_pair_contrib.keys()):
                act = self.menu_pdf_pairs.addAction(pair)
                act.setCheckable(True)
                act.setChecked(False)
                act.toggled.connect(lambda _=None: self._refresh_pdf_pair_curves())
                self._pdf_pair_actions[pair] = act

        except Exception:
            pass


    def _refresh_pdf_pair_curves(self) -> None:
        """Show/hide selected pair contribution curves on the PDF plot."""
        ax = self.plot_pdf.ax

        # Remove old pair lines
        for line in list(getattr(self, "_pdf_pair_lines", {}).values()):
            try:
                line.remove()
            except Exception:
                pass
        self._pdf_pair_lines = {}

        if getattr(self, "_pdf_pair_sum_line", None) is not None:
            try:
                self._pdf_pair_sum_line.remove()
            except Exception:
                pass
            self._pdf_pair_sum_line = None

        if not getattr(self, "_pdf_pair_contrib", None):
            self.plot_pdf.canvas.draw_idle()
            return

        if self.pdf_line_fit is None:
            self.plot_pdf.canvas.draw_idle()
            return

        try:
            r = np.asarray(self.pdf_line_fit.get_xdata(), dtype=float)
        except Exception:
            self.plot_pdf.canvas.draw_idle()
            return

        styles = getattr(self, "_pdf_pair_style", {}) or {}

        selected = [
            pair for pair, st in styles.items()
            if bool(st.get("visible", False))
        ]

        if not selected:
            try:
                ax.legend(loc="best")
            except Exception:
                pass
            self.plot_pdf.canvas.draw_idle()
            return

        # Plot selected individual pair contributions
        for pair in selected:
            y = np.asarray(self._pdf_pair_contrib.get(pair, []), dtype=float)
            if y.size != r.size:
                continue

            st = styles.get(pair, {})
            marker = st.get("marker", "None")
            linestyle = st.get("linestyle", "--")

            kwargs = {
                "linestyle": "None" if linestyle == "None" else linestyle,
                "marker": "" if marker == "None" else marker,
                "linewidth": float(st.get("linewidth", 1.2)),
                "alpha": float(st.get("alpha", 0.85)),
                "label": f"{pair} contribution",
            }

            color = _safe_mpl_color(
                st.get("color", ""),
                standard_pair_color(pair),
            )

            kwargs["color"] = color

            line, = ax.plot(r, y, **kwargs)

            # If user did not choose a color, Matplotlib selected one.
            # Store it so tick markers can match it.
            if not color:
                self._pdf_pair_style.setdefault(pair, {})
                self._pdf_pair_style[pair]["color"] = line.get_color()

            self._pdf_pair_lines[pair] = line

        # Plot sum of selected contributions
        if bool(getattr(self, "_pdf_pair_sum_visible", False)):
            ysum = np.zeros_like(r, dtype=float)
            any_valid = False

            for pair in selected:
                y = np.asarray(self._pdf_pair_contrib.get(pair, []), dtype=float)
                if y.size == r.size:
                    ysum += y
                    any_valid = True

            if any_valid:
                self._pdf_pair_sum_line, = ax.plot(
                    r,
                    ysum,
                    linestyle="-",
                    linewidth=2.3,
                    alpha=0.95,
                    color="tab:red",
                    label="sum selected pair contributions",
                )

        try:
            ax.legend(loc="best")
            ax.relim(visible_only=True)
            ax.autoscale_view(True, True, True)
        except Exception:
            pass

        self.plot_pdf.canvas.draw_idle()

    def _on_tick_markers_toggled(self, checked: bool) -> None:
        """
    Single-control tick-marker workflow.

    - Checked:
        open popup
        user chooses range/filter/style
        calculate tick data if needed
        draw markers

    - Unchecked:
        remove markers
    """
        try:
            if not checked:
                self._clear_pdf_tick_markers()
                self.plot_pdf.canvas.draw_idle()
                return

            # Ensure at least default tick data exist before opening the dialog.
            tick_data = getattr(self, "_pdf_tick_data", {}) or {}
            pairs = tick_data.get("pairs", {}) if isinstance(tick_data, dict) else {}

            if not pairs:
                ok = self._calculate_tick_markers_to_r(
                    float(getattr(self, "_tick_range_max", DEFAULT_TICK_MARKER_MAX_R))
                )

                if not ok:
                    self.btn_tick_markers.blockSignals(True)
                    self.btn_tick_markers.setChecked(False)
                    self.btn_tick_markers.blockSignals(False)
                    self._clear_pdf_tick_markers()
                    return

            dlg = TickMarkerFilterDialog(self)

            if dlg.exec() != QtWidgets.QDialog.Accepted:
                self.btn_tick_markers.blockSignals(True)
                self.btn_tick_markers.setChecked(False)
                self.btn_tick_markers.blockSignals(False)
                self._clear_pdf_tick_markers()
                return

            dlg.apply_to_owner()

            required_max = float(getattr(self, "_tick_range_max", DEFAULT_TICK_MARKER_MAX_R))
            available_max = self._tick_data_max_r()

            # If the user asks beyond currently loaded tick data, calculate more.
            if available_max + 1e-9 < required_max:
                ok = self._calculate_tick_markers_to_r(required_max)

                if not ok:
                    self.btn_tick_markers.blockSignals(True)
                    self.btn_tick_markers.setChecked(False)
                    self.btn_tick_markers.blockSignals(False)
                    self._clear_pdf_tick_markers()
                    return

            self._refresh_pdf_tick_markers()
            self.plot_pdf.canvas.draw_idle()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Tick markers failed",
                str(e),
            )

            try:
                self.btn_tick_markers.blockSignals(True)
                self.btn_tick_markers.setChecked(False)
                self.btn_tick_markers.blockSignals(False)
                self._clear_pdf_tick_markers()
            except Exception:
                pass

    def _set_pdf_tick_data(self, tick_data: dict) -> None:
        """
        Store tick-marker metadata and refresh if the toggle is active.
        """
        self._pdf_tick_data = tick_data or {}

        if getattr(self, "btn_tick_markers", None) is not None:
            if self.btn_tick_markers.isChecked():
                self._refresh_pdf_tick_markers()


    def _clear_pdf_tick_markers(self) -> None:
        """
        Remove tick-marker artists and hover annotation.
        """
        try:
            for coll in getattr(self, "_pdf_tick_collections", []):
                try:
                    coll.remove()
                except Exception:
                    pass

            self._pdf_tick_collections = []
            self._pdf_tick_lookup = []

            if self._pdf_tick_annotation is not None:
                try:
                    self._pdf_tick_annotation.remove()
                except Exception:
                    pass
                self._pdf_tick_annotation = None

            if self._pdf_tick_motion_cid is not None:
                try:
                    self.plot_pdf.canvas.mpl_disconnect(self._pdf_tick_motion_cid)
                except Exception:
                    pass
                self._pdf_tick_motion_cid = None

        except Exception:
            pass

    def _parse_tick_direction(
        self,
        direction: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Parse a direction string like '[1 0 4]' into integer h, k, l.
        """
        try:
            vals = re.findall(r"-?\d+", str(direction))

            if len(vals) < 3:
                return None, None, None

            return int(vals[0]), int(vals[1]), int(vals[2])

        except Exception:
            return None, None, None

    def _tick_direction_allowed(self, direction: str) -> bool:
        """
        Return True if a tick direction passes the current direction filter.
        """
        mode = str(getattr(self, "_tick_filter_direction_mode", "all") or "all")
        text = str(getattr(self, "_tick_filter_direction_text", "") or "").strip()

        if mode == "all":
            return True

        h, k, l = self._parse_tick_direction(direction)

        if h is None or k is None or l is None:
            return False

        if mode == "h00":
            return h != 0 and k == 0 and l == 0

        if mode == "0k0":
            return h == 0 and k != 0 and l == 0

        if mode == "00l":
            return h == 0 and k == 0 and l != 0

        if mode == "hh0":
            return h != 0 and k != 0 and abs(h) == abs(k) and l == 0

        if mode == "h0l":
            return h != 0 and k == 0 and l != 0

        if mode == "0kl":
            return h == 0 and k != 0 and l != 0

        if mode == "contains":
            if not text:
                return True

            return text.lower() in str(direction).lower()

        return True
    
    def _tick_style_for_pair(
        self,
        pair: str,
        row: int,
        colors: List[str],
    ) -> Dict[str, Any]:
        """
        Return tick marker style for a pair.

        Priority:
          1. Explicit tick-marker style from popup
          2. Standard atom/pair color
          3. Matplotlib fallback cycle

        This no longer depends on Show Pairs being calculated.
        """
        tick_styles = getattr(self, "_tick_pair_style", {}) or {}
        tick_style = tick_styles.get(pair, {}) or {}

        fallback = standard_pair_color(pair, row)

        color = _safe_mpl_color(
            tick_style.get("color", ""),
            fallback,
        )

        try:
            linewidth = float(tick_style.get("linewidth", 0.7))
        except Exception:
            linewidth = 0.7

        try:
            alpha = float(tick_style.get("alpha", 0.30))
        except Exception:
            alpha = 0.30

        alpha = max(0.01, min(alpha, 1.0))
        linewidth = max(0.1, linewidth)

        return {
            "color": color,
            "linewidth": linewidth,
            "alpha": alpha,
        }

    def _refresh_pdf_tick_markers(self) -> None:
        """
        Draw full-height vertical PDF tick markers inside the plot frame.

        Efficient design:
        - one LineCollection per pair
        - one hover annotation
        - only visible x-range is drawn
        - no individual Line2D per tick
        """
        self._clear_pdf_tick_markers()

        if getattr(self, "btn_tick_markers", None) is None:
            return

        if not self.btn_tick_markers.isChecked():
            return

        tick_data = getattr(self, "_pdf_tick_data", {}) or {}
        pairs = tick_data.get("pairs", {}) if isinstance(tick_data, dict) else {}

        if not pairs:
            return

        ax = self.plot_pdf.ax

        try:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        except Exception:
            return

        if (
            not np.isfinite(xmin)
            or not np.isfinite(xmax)
            or not np.isfinite(ymin)
            or not np.isfinite(ymax)
            or xmin == xmax
            or ymin == ymax
        ):
            return

        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        allowed_pairs = getattr(self, "_tick_filter_pairs", None)

        sorted_pairs = sorted(pairs.keys())

        if allowed_pairs is not None:
            sorted_pairs = [p for p in sorted_pairs if p in allowed_pairs]

        self._pdf_tick_lookup = []

        for row, pair in enumerate(sorted_pairs):
            pdata = pairs.get(pair, {}) or {}

            rr_all = np.asarray(pdata.get("r", []), dtype=float)
            shells_all = list(pdata.get("shell", []))
            directions_all = list(pdata.get("direction", []))
            mults_all = list(pdata.get("multiplicity", []))

            if rr_all.size == 0:
                continue

            tick_rmin = float(getattr(self, "_tick_range_min", DEFAULT_TICK_MARKER_MIN_R))
            tick_rmax = float(getattr(self, "_tick_range_max", DEFAULT_TICK_MARKER_MAX_R))

            draw_min = max(float(xmin), tick_rmin)
            draw_max = min(float(xmax), tick_rmax)

            keep = (
                np.isfinite(rr_all)
                & (rr_all >= draw_min)
                & (rr_all <= draw_max)
            )

            if directions_all:
                direction_keep = np.array(
                    [
                        self._tick_direction_allowed(
                            directions_all[i] if i < len(directions_all) else ""
                        )
                        for i in range(rr_all.size)
                    ],
                    dtype=bool,
                )

                keep = keep & direction_keep

            if not np.any(keep):
                continue

            keep_idx = np.where(keep)[0]
            rr = rr_all[keep_idx]

            shells = [
                shells_all[i] if i < len(shells_all) else None
                for i in keep_idx
            ]
            directions = [
                directions_all[i] if i < len(directions_all) else ""
                for i in keep_idx
            ]
            mults = [
                mults_all[i] if i < len(mults_all) else None
                for i in keep_idx
            ]

            if rr.size == 0:
                continue

            segments = [
                [(float(x), float(ymin)), (float(x), float(ymax))]
                for x in rr
            ]

            style = self._tick_style_for_pair(
                pair,
                row,
                colors,
            )

            coll = LineCollection(
                segments,
                colors=style["color"],
                linewidths=style["linewidth"],
                alpha=style["alpha"],
                clip_on=True,
                zorder=1.1,
            )

            ax.add_collection(coll, autolim=False)
            self._pdf_tick_collections.append(coll)

            order = np.argsort(rr)

            self._pdf_tick_lookup.append(
                {
                    "pair": pair,
                    "r": rr[order],
                    "shell": [shells[i] for i in order],
                    "direction": [directions[i] for i in order],
                    "multiplicity": [mults[i] for i in order],
                }
            )

        if not self._pdf_tick_lookup:
            self.plot_pdf.canvas.draw_idle()
            return

        self._pdf_tick_annotation = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="0.5",
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="->", color="0.4"),
            zorder=20,
        )
        self._pdf_tick_annotation.set_visible(False)

        self._pdf_tick_motion_cid = self.plot_pdf.canvas.mpl_connect(
            "motion_notify_event",
            self._on_pdf_tick_hover,
        )

        self.plot_pdf.canvas.draw_idle()        

    def _on_pdf_tick_hover(self, event) -> None:
        """
        Hover callback for full-height PDF tick markers.

        Efficient:
        - uses sorted r arrays
        - searches nearest x only
        - does not use Matplotlib picking on thousands of artists
        """
        ann = getattr(self, "_pdf_tick_annotation", None)

        if ann is None:
            return

        ax = self.plot_pdf.ax

        if event.inaxes is not ax or event.xdata is None:
            if ann.get_visible():
                ann.set_visible(False)
                self.plot_pdf.canvas.draw_idle()
            return

        x = float(event.xdata)

        best = None
        best_dx_pix = float("inf")

        # Pixel threshold for selecting nearest tick.
        x_threshold_pix = 7.0

        lookup = getattr(self, "_pdf_tick_lookup", []) or []

        for rowdata in lookup:
            rr = np.asarray(rowdata.get("r", []), dtype=float)

            if rr.size == 0:
                continue

            idx = int(np.searchsorted(rr, x))

            candidates = []
            if 0 <= idx < rr.size:
                candidates.append(idx)
            if 0 <= idx - 1 < rr.size:
                candidates.append(idx - 1)

            for ci in candidates:
                x_tick = float(rr[ci])

                try:
                    x_pix = ax.transData.transform((x_tick, 0.0))[0]
                    dx_pix = abs(float(event.x) - float(x_pix))
                except Exception:
                    continue

                if dx_pix < best_dx_pix and dx_pix <= x_threshold_pix:
                    best_dx_pix = dx_pix
                    best = (rowdata, ci)

        if best is None:
            if ann.get_visible():
                ann.set_visible(False)
                self.plot_pdf.canvas.draw_idle()
            return

        rowdata, i = best

        pair = rowdata.get("pair", "")
        rr = float(rowdata["r"][i])
        shell = rowdata["shell"][i]
        direction = rowdata["direction"][i]
        mult = rowdata["multiplicity"][i]

        y_anchor = event.ydata

        if y_anchor is None or not np.isfinite(y_anchor):
            ymin, ymax = ax.get_ylim()
            y_anchor = 0.5 * (ymin + ymax)

        txt = (
            f"Pair: {pair}\n"
            f"r = {rr:.5g} Å\n"
            f"Shell: {shell}\n"
            f"Direction: {direction}\n"
            f"Multiplicity: {mult}"
        )

        ann.xy = (rr, y_anchor)
        ann.set_text(txt)
        ann.set_visible(True)

        self.plot_pdf.canvas.draw_idle()


    def calculate_full_tick_markers(self) -> None:
        """
        Calculate full-range PDF tick-marker metadata on demand.

        This is intentionally separate from normal Calculate/Run, because full-range
        tick metadata can be large for high r_max.
        """
        try:
            txt = self._compose_input_text_for_run()

            if not txt.strip():
                QtWidgets.QMessageBox.warning(
                    self,
                    "No input",
                    "Please load or create an input file first.",
                )
                return

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inp",
                mode="w",
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.flush()
            tmp.close()

            try:
                config = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            # Use current input parameters, overridden by last refined parameters if available.
            params = dict(config.get("initial", {}) or {})

            if isinstance(getattr(self, "_last_refined_params", None), dict):
                params.update(self._last_refined_params)

            constraints = config.get("constraints", {}) or {}
            if constraints:
                try:
                    params = apply_constraints(params, constraints)
                except Exception:
                    pass

            # Full tick range: use r_max.
            full_max_r = float(config.get("r_max", 0.0))

            if not np.isfinite(full_max_r) or full_max_r <= 0.0:
                try:
                    # Fallback from current PDF plot x-axis.
                    _xmin, _xmax = self.plot_pdf.ax.get_xlim()
                    full_max_r = float(_xmax)
                except Exception:
                    full_max_r = DEFAULT_TICK_MARKER_MAX_R

            full_max_r = max(float(full_max_r), DEFAULT_TICK_MARKER_MAX_R)

            self.statusBar().showMessage(
                f"Calculating full-range tick markers up to {full_max_r:.3g} Å..."
            )
            QtWidgets.QApplication.processEvents()

            # Build a model with enough pair cutoff for full tick range.
            # This will normally load from your shell cache if it already exists.
            r_dummy = np.linspace(
                float(config.get("r_min", 0.0)),
                full_max_r,
                10,
            )

            pdf_model = PDFCalculator(
                config,
                config["structure_file"],
                r_dummy,
                pair_cutoff=full_max_r,
            )

            tick_data = pdf_model.build_pdf_tick_data(
                params,
                max_r=full_max_r,
                max_ticks_per_pair=None,
            )

            self._set_pdf_tick_data(tick_data)

            # Turn markers on automatically after full-range calculation.
            if not self.btn_tick_markers.isChecked():
                self.btn_tick_markers.setChecked(True)
            else:
                self._refresh_pdf_tick_markers()

            self.statusBar().showMessage(
                f"Full-range tick markers calculated up to {full_max_r:.3g} Å."
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Full tick range failed",
                str(e),
            )


    def _schedule_local_trends_update(self) -> None:
        """Debounce GUI edits so lambda trend plot updates in near real time."""
        try:
            self._local_trend_timer.start(350)
        except Exception:
            pass
    
    def _get_local_trend_model(
        self,
        cfg: Dict[str, Any],
        *,
        pair_cutoff: Optional[float] = None,
    ):
        """
        Build/reuse a PDFCalculator only for shell distances used in the local trend plot.

        For default local trends, use pair_cutoff ≈ 10 Å.
        For full-range local trends, use pair_cutoff ≈ d or r_max.
        """
        try:
            structure_file = str(cfg.get("structure_file", "") or "").strip()

            if not structure_file or not os.path.exists(structure_file):
                return None

            r_max = float(cfg.get("r_max", 0.0))
            r_extension = float(cfg.get("r_extension", 1.2))

            if pair_cutoff is None:
                pair_cutoff_use = r_max * r_extension
            else:
                pair_cutoff_use = float(pair_cutoff)

            if not np.isfinite(pair_cutoff_use) or pair_cutoff_use <= 0.0:
                pair_cutoff_use = 10.0

            key = (
                os.path.abspath(structure_file),
                round(float(pair_cutoff_use), 8),
            )

            if (
                getattr(self, "_local_trend_model_key", None) == key
                and getattr(self, "_local_trend_model", None) is not None
            ):
                return self._local_trend_model

            r_dummy = np.linspace(
                float(cfg.get("r_min", 0.0)),
                min(float(cfg.get("r_max", 10.0)), float(pair_cutoff_use)),
                10,
            )

            model = PDFCalculator(
                cfg,
                structure_file,
                r_dummy,
                pair_cutoff=pair_cutoff_use,
            )

            self._local_trend_model = model
            self._local_trend_model_key = key

            return model

        except Exception:
            return None


    def _read_current_cfg_for_local_trends(self) -> Optional[Dict[str, Any]]:
        """Read current GUI/raw input into a config dict."""
        try:
            txt = self._compose_input_text_for_run()

            if not txt.strip():
                return None

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inp",
                mode="w",
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.flush()
            tmp.close()

            try:
                cfg = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            return cfg

        except Exception:
            return None

    def _params_for_local_trends(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use current input parameters, overridden by last refined parameters if available.
        """
        params = dict(cfg.get("initial", {}) or {})

        if isinstance(getattr(self, "_last_refined_params", None), dict):
            params.update(self._last_refined_params)

        return params

    def _full_local_trend_limit(self, params: Dict[str, Any], cfg: Dict[str, Any]) -> float:
        """
        Full range for local trends:
        - use particle diameter d if available
        - otherwise use r_max
        """
        try:
            d = float(params.get("d", params.get("D", 0.0)))

            if np.isfinite(d) and d > 0.0:
                return d

        except Exception:
            pass

        try:
            rmax = float(cfg.get("r_max", 10.0))

            if np.isfinite(rmax) and rmax > 0.0:
                return rmax

        except Exception:
            pass

        return 10.0

    def _compute_local_trends_with_limit(self, max_r_plot: float) -> None:
        """
        Recompute local trends on demand with a chosen distance range.
        """
        try:
            cfg = self._read_current_cfg_for_local_trends()

            if not cfg:
                return

            params = self._params_for_local_trends(cfg)

            max_r_plot = float(max_r_plot)
            if not np.isfinite(max_r_plot) or max_r_plot <= 0.0:
                max_r_plot = 10.0

            # Use a shell model only as large as needed.
            model = self._get_local_trend_model(
                cfg,
                pair_cutoff=max_r_plot,
            )

            if model is None:
                return

            self.statusBar().showMessage(
                f"Computing local trends up to {max_r_plot:.3g} Å..."
            )
            QtWidgets.QApplication.processEvents()

            trends = self._build_local_trends_from_model(
                model,
                params,
                max_r_plot=max_r_plot,
            )

            if isinstance(trends, dict):
                self._plot_local_trends(trends)

            self.statusBar().showMessage(
                f"Local trends computed up to {max_r_plot:.3g} Å."
            )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Local trends failed",
                str(e),
            )

    def compute_local_trends_10A(self) -> None:
        """Button slot: recompute local trends up to 10 Å."""
        self._compute_local_trends_with_limit(10.0)

    def compute_local_trends_full_range(self) -> None:
        """Button slot: recompute local trends up to d or r_max."""
        cfg = self._read_current_cfg_for_local_trends()

        if not cfg:
            return

        params = self._params_for_local_trends(cfg)
        max_r_plot = self._full_local_trend_limit(params, cfg)

        self._compute_local_trends_with_limit(max_r_plot)

    def _update_local_trends_from_current_gui(self) -> None:
        """
        Recompute local trends from current GUI values, without running refinement.
        This is triggered when lambdas/deltas are edited.
        """
        try:
            txt = self._compose_input_text_for_run()
            if not txt.strip():
                return

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".inp", mode="w", encoding="utf-8")
            tmp.write(txt)
            tmp.flush()
            tmp.close()

            try:
                cfg = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            model = self._get_local_trend_model(
                cfg,
                pair_cutoff=10.0,
            )

            if model is None:
                return

            params = dict(cfg.get("initial", {}) or {})

            trends = self._build_local_trends_from_model(
                model,
                params,
                max_r_plot=10.0,
            )

            if isinstance(trends, dict):
                self._plot_local_trends(trends)

        except Exception:
            pass


    def _build_local_trends_from_model(
        self,
        pdf_model,
        params: Dict[str, Any],
        *,
        max_r_plot: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Build local trends from current GUI params.

        Default range is 10 Å. Longer ranges are computed only when the user
        clicks the Full range button.
        """
        try:
            return pdf_model.build_local_trends(
                params or {},
                max_r_plot=float(max_r_plot),
            )
        except Exception:
            return {}

    def _plot_local_trends(self, trends: Dict[str, Any]) -> None:
        """
        Store and display local dynamics trends.

        Display settings are now controlled by LocalDynamicsDisplayOptionsDialog.
        """
        self._local_trends = trends or {}

        # If selected pairs refer to old data, clean them.
        try:
            if self._local_selected_pairs is not None:
                valid = set(self._local_trends.keys())
                self._local_selected_pairs = {
                    p for p in self._local_selected_pairs
                    if p in valid
                }

                # If nothing remains selected, return to default = all pairs.
                if not self._local_selected_pairs:
                    self._local_selected_pairs = None
        except Exception:
            self._local_selected_pairs = None

        self._refresh_local_trends_plot()


    def _draw_local_trends_on_ax(self, ax) -> None:
        ax.clear()

        trends = getattr(self, "_local_trends", {}) or {}

        max_range = None
        try:
            for _pair_data in trends.values():
                if isinstance(_pair_data, dict) and "max_r_plot" in _pair_data:
                    max_range = float(_pair_data["max_r_plot"])
                    break
        except Exception:
            max_range = None

        if max_range is not None and np.isfinite(max_range):
            ax.set_title(
                f"Local dynamics: λ coefficients and δ continuation "
                f"(up to {max_range:.3g} Å)"
            )
        else:
            ax.set_title("Local dynamics: λ coefficients and δ continuation")

        ax.set_xlabel("Pair distance r (Å)")
        ax.set_ylabel(r"$\lambda_k$ or $\delta_1/r + \delta_2/r^2$")

        if not trends:
            ax.text(
                0.5,
                0.5,
                "No local trend data available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(False)
            return

        selected_pairs = getattr(self, "_local_selected_pairs", None)

        if selected_pairs is None:
            pairs = sorted(trends.keys())
        else:
            pairs = [p for p in sorted(trends.keys()) if p in selected_pairs]

        display_mode = getattr(self, "_local_display_mode", "both")
        style = getattr(self, "_local_style", {}) or {}
        pair_style = getattr(self, "_local_pair_style", {}) or {}

        lam_marker = style.get("lambda_marker", "o")
        del_marker = style.get("delta_marker", "x")
        lam_line = style.get("lambda_linestyle", "-")
        del_line = style.get("delta_linestyle", "--")

        lam_marker = "" if lam_marker == "None" else lam_marker
        del_marker = "" if del_marker == "None" else del_marker
        lam_line = "None" if lam_line == "None" else lam_line
        del_line = "None" if del_line == "None" else del_line

        plotted_any = False

        for row, pair in enumerate(pairs):
            data = trends.get(pair, {}) or {}

            try:
                r = np.asarray(data.get("r", []), dtype=float)
                lam = np.asarray(data.get("lambda", []), dtype=float)
                delta_eff = np.asarray(data.get("delta_eff", []), dtype=float)
            except Exception:
                continue

            if r.size == 0 or lam.size != r.size or delta_eff.size != r.size:
                continue

            order = np.argsort(r)
            r = r[order]
            lam = lam[order]
            delta_eff = delta_eff[order]

            finite_r = np.isfinite(r)
            lam_mask = finite_r & np.isfinite(lam)
            delta_mask = finite_r & (~np.isfinite(lam)) & np.isfinite(delta_eff)

            color = _safe_mpl_color(
                (pair_style.get(pair, {}) or {}).get("color", ""),
                standard_pair_color(pair, row),
            )

            # Lambdas and deltas now use the SAME color for the same pair.

            color = _safe_mpl_color(
                (pair_style.get(pair, {}) or {}).get("color", ""),
                standard_pair_color(pair, row),
            )

            if display_mode in ("both", "lambda") and np.any(lam_mask):
                ax.plot(
                    r[lam_mask],
                    lam[lam_mask],
                    marker=lam_marker,
                    linestyle=lam_line,
                    color=color,
                    linewidth=float(style.get("linewidth", 1.5)),
                    markersize=float(style.get("markersize", 5.0)),
                    label=f"{pair} λ",
                )
                plotted_any = True

            if display_mode in ("both", "delta") and np.any(delta_mask):
                ax.plot(
                    r[delta_mask],
                    delta_eff[delta_mask],
                    marker=del_marker,
                    linestyle=del_line,
                    color=color,
                    linewidth=float(style.get("linewidth", 1.5)),
                    markersize=float(style.get("markersize", 5.0)),
                    label=f"{pair} δ",
                )
                plotted_any = True

        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No lambda or delta values available for selected display mode.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(False)
            return

        ax.legend(loc="best")
        ax.grid(True, linestyle=":")

        try:
            self._apply_axis_limits_from_style(ax, style)
        except Exception:
            pass

    
    def _build_size_distribution_from_params(self, params: Dict[str, Any]):
        """
        Build size distribution from current parameters.

        Priority:
        1. Cylinder thickness distribution if cylinder model is active.
        2. Conventional spherical d/d_std distribution.
        """
        try:
            cyl_D = float(params.get("cyl_diameter", 0.0))
            cyl_t = float(params.get("cyl_thickness", 0.0))
            cyl_std = float(params.get("cyl_thickness_std", 0.0))

            if np.isfinite(cyl_D) and cyl_D > 0 and np.isfinite(cyl_t) and cyl_t > 0:
                if np.isfinite(cyl_std) and cyl_std > 0:
                    sig2 = np.log(1.0 + (cyl_std * cyl_std) / (cyl_t * cyl_t))
                    sig = float(np.sqrt(sig2))
                    mu = float(np.log(cyl_t) - 0.5 * sig2)

                    x = np.linspace(
                        max(1e-6, cyl_t * 0.05),
                        cyl_t * 3.0,
                        200,
                    )

                    pdf = (
                        1.0 / (x * sig * np.sqrt(2.0 * np.pi))
                    ) * np.exp(
                        -0.5 * ((np.log(x) - mu) / sig) ** 2
                    )

                    return {
                        "x": x,
                        "pdf": pdf,
                        "mean": float(cyl_t),
                        "std": float(cyl_std),
                        "mu": float(mu),
                        "sigma": float(sig),
                        "label": "Cylinder thickness",
                        "xlabel": "Thickness t (Å)",
                    }

        except Exception:
            pass

        try:
            d = float(params.get("d", np.nan))
            dstd = float(params.get("d_std", np.nan))

            mu = sig = mean = std = None

            if np.isfinite(d) and d > 0:
                if np.isfinite(dstd) and dstd > 0:
                    sig2 = np.log(1.0 + (dstd * dstd) / (d * d))
                    sig = float(np.sqrt(sig2))
                    mu = float(np.log(d) - 0.5 * sig2)
                    mean = d
                    std = dstd

            if mu is not None and sig is not None:
                x = np.linspace(
                    max(1e-6, float(mean) * 0.05),
                    float(mean) * 3.0,
                    200,
                )

                pdf = (
                    1.0 / (x * sig * np.sqrt(2.0 * np.pi))
                ) * np.exp(
                    -0.5 * ((np.log(x) - mu) / sig) ** 2
                )

                return {
                    "x": x,
                    "pdf": pdf,
                    "mean": float(mean),
                    "std": float(std),
                    "mu": float(mu),
                    "sigma": float(sig),
                    "label": "Spherical crystallite size",
                    "xlabel": "D (Å)",
                }

        except Exception:
            pass

        return None

    def _plot_size_dist(self, size: Dict[str, Any]):
        ax = self.plot_size.ax
        ax.clear()
        label = str(size.get("label", "Crystallite size"))
        xlabel = str(size.get("xlabel", "D"))

        ax.set_title(f"{label} distribution (lognormal)")
        if not size or "x" not in size or "pdf" not in size or len(size.get("x", [])) == 0:
            ax.text(0.5, 0.5, "No size-distribution data available for this model/parameter set.",
                    transform=ax.transAxes, ha="center", va="center")
            ax.grid(False)
            self.plot_size.canvas.draw_idle()
            return
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PDF")
        ax.plot(size["x"], size["pdf"], marker="o", linestyle="-")
        if "mean" in size and "std" in size:
            ax.text(0.02, 0.95, f"mean={size['mean']:.3g}, std={size['std']:.3g}",
                    transform=ax.transAxes, va="top")
        ax.grid(True, linestyle=":")
        self.plot_size.canvas.draw_idle()

    @QtCore.Slot(str)
    def on_failed(self, message: str):
        self.btn_calculate.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self._worker = None
        self._update_run_button_modes()

        QtWidgets.QMessageBox.critical(self, "Run failed", message)

    def closeEvent(self, event):
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.request_stop()
                self._worker.wait(2000)
        except Exception:
            pass

        try:
            self._returning_pdf_to_main = True
            if self._pdf_dialog is not None:
                self._pdf_dialog.hide()
        except Exception:
            pass
        finally:
            self._returning_pdf_to_main = False

        super().closeEvent(event)

    def _remove_existing_lambda_keys(self, cfg_raw) -> None:
        """Remove all lambda/lam keys from raw config before writing current GUI lambdas.
        This makes 'Remove selected' in the LambdaListWidget actually remove
        the parameter from the raw input file.
        """
        for sec in ("initial_values", "refinable_parameters", "bounds"):
            if sec not in cfg_raw:
                continue
            for k in list(cfg_raw[sec].keys()):
                lk = str(k).strip().lower()
                if lk.startswith("lambda") or lk.startswith("lam"):
                    try:
                        cfg_raw[sec].pop(k, None)
                    except Exception:
                        pass
    
    def _warren_y_values(
        self,
        arr: List[Tuple[float, float]],
        y_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(arr, dtype=float)

        if a.ndim != 2 or a.shape[1] != 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        L = a[:, 0]
        dL2 = np.maximum(a[:, 1], 0.0)

        mode = y_mode or getattr(self, "_warren_y_mode", "warren")

        if mode == "warren":
            y = np.sqrt(dL2)
        else:
            y = np.sqrt(dL2) / np.maximum(L, 1e-12)

        return L, y


    def _default_warren_keys_min_mid_max(self) -> List[Tuple[int, int, int]]:
        """
        Default Warren display:
        plot smallest, middle, and largest Warren/microstrain maxima.

        Important:
        ignore directions with <= 5 points because those are usually not
        representative and can dominate the min/max selection incorrectly.
        """
        cache = getattr(self, "_warren_display_cache", {}) or {}

        y_mode = getattr(self, "_warren_y_mode", "warren")
        max_key = "max_warren" if y_mode == "warren" else "max_microstrain"

        rows = []

        for d, info in cache.items():
            try:
                n_points = int(info.get("n_points", 0))

                # User requested more than 5 points.
                if n_points <= 5:
                    continue

                y_max = float(info.get(max_key, float("nan")))

                if not np.isfinite(y_max):
                    continue

                rows.append((y_max, d))

            except Exception:
                continue

        # Fallback: if no direction has > 5 points, use anything valid.
        if not rows:
            for d, info in cache.items():
                try:
                    y_max = float(info.get(max_key, float("nan")))

                    if np.isfinite(y_max):
                        rows.append((y_max, d))

                except Exception:
                    continue

        if not rows:
            return []

        rows.sort(key=lambda x: x[0])

        if len(rows) <= 3:
            return [d for _y, d in rows]

        return [
            rows[0][1],
            rows[len(rows) // 2][1],
            rows[-1][1],
        ]

    def open_warren_display_options(self) -> None:
        if not getattr(self, "_warren_grouped", None):
            QtWidgets.QMessageBox.information(
                self,
                "No Warren data",
                "No Warren data are available yet.",
            )
            return

        dlg = WarrenDisplayOptionsDialog(self)

        if dlg.exec() == QtWidgets.QDialog.Accepted:
            dlg.apply_to_owner()
            self._refresh_warren_plot()


    def open_warren_plot_window(self) -> None:
        """
        Open Warren plot in a separate window.

        The separate window also has Display options and Refresh buttons.
        """
        if not getattr(self, "_warren_display_cache", None):
            QtWidgets.QMessageBox.information(
                self,
                "No Warren data",
                "No Warren data are available yet.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Warren plot")
        dlg.setWindowFlags(
            dlg.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        dlg.resize(1100, 750)

        plot = MplPlot(title="Warren plot")

        btn_options = QtWidgets.QPushButton("Display options")
        btn_refresh = QtWidgets.QPushButton("Refresh")

        try:
            plot.toolbar.addSeparator()
            plot.toolbar.addWidget(btn_options)
            plot.toolbar.addWidget(btn_refresh)
        except Exception:
            pass

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(plot)

        def redraw():
            self._draw_warren_on_ax(plot.ax)
            plot.canvas.draw_idle()

        def open_options():
            opt = WarrenDisplayOptionsDialog(self)

            if opt.exec() == QtWidgets.QDialog.Accepted:
                opt.apply_to_owner()

                # Refresh main plot and this separate plot.
                self._refresh_warren_plot()
                redraw()

        btn_options.clicked.connect(open_options)
        btn_refresh.clicked.connect(redraw)

        redraw()

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        self._warren_dialog_window = dlg


    def _apply_axis_limits_from_style(self, ax, style: Dict[str, Any]) -> None:
        def _float_or_none(x):
            try:
                s = str(x).strip()
                if not s:
                    return None
                return float(s)
            except Exception:
                return None

        xmin = _float_or_none(style.get("xmin", ""))
        xmax = _float_or_none(style.get("xmax", ""))
        ymin = _float_or_none(style.get("ymin", ""))
        ymax = _float_or_none(style.get("ymax", ""))

        if xmin is not None or xmax is not None:
            old_xmin, old_xmax = ax.get_xlim()
            ax.set_xlim(
                xmin if xmin is not None else old_xmin,
                xmax if xmax is not None else old_xmax,
            )

        if ymin is not None or ymax is not None:
            old_ymin, old_ymax = ax.get_ylim()
            ax.set_ylim(
                ymin if ymin is not None else old_ymin,
                ymax if ymax is not None else old_ymax,
            )


    def _draw_warren_on_ax(self, ax) -> None:
        ax.clear()

        y_mode = getattr(self, "_warren_y_mode", "warren")
        cache = getattr(self, "_warren_display_cache", {}) or {}

        if y_mode == "warren":
            ax.set_ylabel(r"$\sqrt{\langle \Delta L^2 \rangle}$ (Å)")
            ax.set_title("Warren plot")
            y_key = "y_warren"
        else:
            ax.set_ylabel(r"$\sqrt{\langle \Delta L^2 \rangle}/L$")
            ax.set_title("Microstrain plot")
            y_key = "y_microstrain"

        ax.set_xlabel("L (Å)")

        if not cache:
            ax.text(
                0.5,
                0.5,
                "No Warren-strain data available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(False)
            self._warren_current_keys = []
            return

        keys = list(getattr(self, "_warren_selected_keys", []) or [])

        if not keys:
            keys = self._default_warren_keys_min_mid_max()

        # Keep only keys that still exist in the cache.
        keys = [d for d in keys if d in cache]

        self._warren_current_keys = list(keys)

        if not keys:
            ax.text(
                0.5,
                0.5,
                "No selected Warren directions.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(False)
            return

        style = getattr(self, "_warren_style", {}) or {}
        dir_style = getattr(self, "_warren_dir_style", {}) or {}

        marker = style.get("marker", "o")
        linestyle = style.get("linestyle", "-")

        marker = "" if marker == "None" else marker
        linestyle = "None" if linestyle == "None" else linestyle

        plotted_any = False

        for i, d in enumerate(keys):
            info = cache.get(d, {}) or {}

            L = np.asarray(info.get("L", []), dtype=float)
            y = np.asarray(info.get(y_key, []), dtype=float)

            if L.size == 0 or y.size == 0:
                continue

            color = _safe_mpl_color(
                (dir_style.get(d, {}) or {}).get("color", ""),
                _COLOR_CYCLE[i % len(_COLOR_CYCLE)],
            )

            ax.plot(
                L,
                y,
                marker=marker,
                linestyle=linestyle,
                linewidth=float(style.get("linewidth", 1.5)),
                markersize=float(style.get("markersize", 5.0)),
                color=color,
                label=info.get("label", self._warren_label(d)),
            )

            plotted_any = True

        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No plottable Warren data for selected directions.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(False)
            return

        ax.legend(loc="best")
        ax.grid(True, linestyle=":")

        try:
            self._apply_axis_limits_from_style(ax, style)
        except Exception:
            pass

    def open_local_display_options(self) -> None:
        if not getattr(self, "_local_trends", None):
            QtWidgets.QMessageBox.information(
                self,
                "No local dynamics data",
                "No local dynamics data are available yet.",
            )
            return
    
        dlg = LocalDynamicsDisplayOptionsDialog(self)
    
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            dlg.apply_to_owner()
            self._refresh_local_trends_plot()
    
    def _refresh_local_trends_plot(self) -> None:
        self._draw_local_trends_on_ax(self.plot_local_trends.ax)
        self.plot_local_trends.canvas.draw_idle()


    def open_local_plot_window(self) -> None:
        if not getattr(self, "_local_trends", None):
            QtWidgets.QMessageBox.information(
                self,
                "No local dynamics data",
                "No local dynamics data are available yet.",
            )
            return
    
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Local dynamics trends")
        dlg.setWindowFlags(
            dlg.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        dlg.resize(1000, 700)
    
        plot = MplPlot(title="Local dynamics trends")
    
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(plot)
    
        self._draw_local_trends_on_ax(plot.ax)
        plot.canvas.draw_idle()
    
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
    
        self._local_dialog_window = dlg

    def _update_cs_grid_plots(self) -> None:
        """
        Update CS grid-search diagnostic plots.

        Plots:
            1. best Rwp vs diameter
            2. best Rwp vs axis length
            3. Rwp heatmap over diameter/axis length
        """
        rows = list(getattr(self, "_cs_results", []) or [])

        if not rows:
            return

        d = np.asarray([r["diameter_cells"] for r in rows], dtype=float)
        h = np.asarray([r["height_cells"] for r in rows], dtype=float)
        rwp = np.asarray([r["rwp"] for r in rows], dtype=float)

        # ---------------------------------------------------------
        # Rwp vs diameter
        # ---------------------------------------------------------
        ax = self.plot_cs_diameter.ax
        ax.clear()
        ax.set_title("Final Rwp vs diameter")
        ax.set_xlabel("Diameter / unit cells")
        ax.set_ylabel("Best Rwp (%)")
        ax.grid(True, linestyle=":")

        finite_d = np.isfinite(d)

        if np.any(finite_d):
            unique_d = np.asarray(sorted(set(d[finite_d])), dtype=float)
            y = []

            for dv in unique_d:
                mask = d == dv
                y.append(float(np.nanmin(rwp[mask])))

            ax.plot(unique_d, y, marker="o", linestyle="-")
        else:
            ax.text(
                0.5,
                0.5,
                "Diameter treated as infinite.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        self.plot_cs_diameter.canvas.draw_idle()

        # ---------------------------------------------------------
        # Rwp vs height
        # ---------------------------------------------------------
        ax = self.plot_cs_height.ax
        ax.clear()
        ax.set_title("Final Rwp vs axis length")
        ax.set_xlabel("Axis length / unit cells")
        ax.set_ylabel("Best Rwp (%)")
        ax.grid(True, linestyle=":")

        unique_h = np.asarray(sorted(set(h)), dtype=float)
        y = []

        for hv in unique_h:
            mask = h == hv
            y.append(float(np.nanmin(rwp[mask])))

        ax.plot(unique_h, y, marker="o", linestyle="-")

        self.plot_cs_height.canvas.draw_idle()

        # ---------------------------------------------------------
        # Heatmap
        # ---------------------------------------------------------
        ax = self.plot_cs_map.ax
        ax.clear()
        ax.set_title("CS grid search Rwp map")
        ax.set_xlabel("Diameter / unit cells")
        ax.set_ylabel("Axis length / unit cells")

        if np.any(finite_d):
            unique_d = np.asarray(sorted(set(d[finite_d])), dtype=float)
            unique_h = np.asarray(sorted(set(h)), dtype=float)

            Z = np.full((len(unique_h), len(unique_d)), np.nan, dtype=float)

            for row in rows:
                dv = float(row["diameter_cells"])
                hv = float(row["height_cells"])
                rv = float(row["rwp"])

                if not np.isfinite(dv):
                    continue

                i_arr = np.where(unique_h == hv)[0]
                j_arr = np.where(unique_d == dv)[0]

                if i_arr.size == 0 or j_arr.size == 0:
                    continue

                i = int(i_arr[0])
                j = int(j_arr[0])

                if np.isnan(Z[i, j]) or rv < Z[i, j]:
                    Z[i, j] = rv


            if len(unique_d) < 2 or len(unique_h) < 2:
                ax.scatter(d, h, c=rwp, s=80)
                ax.set_title("Finite-shape fit result")
                ax.set_xlabel("Diameter / unit cells")
                ax.set_ylabel("Axis length / unit cells")

                try:
                    for dv, hv, rv in zip(d, h, rwp):
                        ax.text(float(dv), float(hv), f"{float(rv):.3f}%", ha="center", va="bottom")
                except Exception:
                    pass
                
                self.plot_cs_map.canvas.draw_idle()
                return


            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[
                    float(unique_d.min()),
                    float(unique_d.max()),
                    float(unique_h.min()),
                    float(unique_h.max()),
                ],
                interpolation="nearest",
            )

            ax.figure.colorbar(im, ax=ax, label="Rwp (%)")

        else:
            ax.text(
                0.5,
                0.5,
                "Diameter infinite: heatmap not applicable.\nUse Rwp vs axis length.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        self.plot_cs_map.canvas.draw_idle()

    def _current_shape_search_mode(self) -> str:
        try:
            if hasattr(self, "builder") and self.builder is not None:
                spec = getattr(self.builder, "_crystallite_shape_spec", {}) or {}

                if spec:
                    return str(spec.get("search_mode", "grid_search")).strip().lower()
        except Exception:
            pass

        try:
            txt = self.raw_editor.toPlainText()

            if self.left_tabs.currentWidget() is self.builder:
                txt = self._compose_input_text_for_run()

            if not txt.strip():
                return "grid_search"

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inp",
                mode="w",
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.flush()
            tmp.close()

            try:
                cfg = read_input_file(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            shape_cfg = cfg.get("crystallite_shape", {}) or {}
            return str(shape_cfg.get("search_mode", "grid_search")).strip().lower()

        except Exception:
            return "grid_search"

    

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.showMaximized()
    app.exec()