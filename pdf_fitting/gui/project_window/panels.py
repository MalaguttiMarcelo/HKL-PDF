"""
Editor panels of the project window, one per tree node type.

Every panel has ``load(win, project, dataset, phase)`` and emits ``edited``
whenever it changes the project model in place.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.gui.cif_utils import clean_el_symbol
from pdf_fitting.gui.param_widgets import LambdaListWidget, SiteParamTableWidget
from pdf_fitting.gui.plots import MplPlot
from pdf_fitting.project.model import Parameter, safe_id

from . import sections as S
from .widgets import Card, DictEditor, ParamSection, SettingsForm, fmt_value, parse_number

PHASE_COLORS = ["#1d4ed8", "#c2410c", "#15803d", "#7c3aed", "#b91c1c", "#0e7490", "#a16207", "#be185d"]


def phase_color(index: int) -> str:
    return PHASE_COLORS[index % len(PHASE_COLORS)]


class Panel(QtWidgets.QWidget):
    edited = QtCore.Signal()
    navigate = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(12)
        self.win = None
        self.project = None
        self.dataset = None
        self.phase = None

    def load(self, win, project, dataset=None, phase=None):
        self.win, self.project, self.dataset, self.phase = win, project, dataset, phase

    def finish(self):
        self.lay.addStretch(1)


def S_safe(label: str) -> str:
    from pdf_fitting.structure_handler import safe_param_key

    return safe_param_key(label)


def resolve_info(project, phase):
    path = project.structure_path(phase)
    return S.structure_info(path)


# =============================================================================
# Project level
# =============================================================================

SETTINGS_SPEC_STRATEGY = [
    ("staged_refinement", "Staged refinement", "bool", None, False, "Refine geometry, Biso, shape and strain in stages before the final stage with everything."),
    ("max_nfev_final", "Max evaluations (final stage)", "int", (1, 100000), 60, "Function-evaluation budget of the final stage."),
    ("max_nfev_stage", "Max evaluations (other stages)", "int", (1, 100000), 30, ""),
    ("ftol", "ftol", "float", (0.0, 1.0, 8), 1e-3, "Relative change of the cost that stops the fit."),
    ("xtol", "xtol", "float", (0.0, 1.0, 8), 1e-3, "Relative change of the parameters that stops the fit."),
    ("gtol", "gtol", "float", (0.0, 1.0, 8), 1e-3, "Gradient tolerance."),
    ("refinement_r_step_final", "r step, final stage (Å)", "float", (0.0, 10.0, 4), 0.0, "Coarse r grid used while refining (0 = full grid)."),
    ("refinement_r_step", "r step, other stages (Å)", "float", (0.0, 10.0, 4), 0.0, "0 = automatic."),
    ("loss", "Loss function", "choice", [("linear", "Linear (least squares)"), ("soft_l1", "Soft L1"), ("huber", "Huber"), ("cauchy", "Cauchy"), ("arctan", "Arctan")], "linear", "Robust loss for outliers."),
    ("progress_every_sec", "Progress update every (s)", "float", (0.0, 60.0, 2), 0.5, ""),
]
SETTINGS_SPEC_BACKEND = [
    ("shell_backend", "Shell generation", "choice", [("cpu", "CPU"), ("cuda", "CUDA (GPU)"), ("auto", "Automatic")], "cpu", ""),
    ("pdf_backend", "PDF accumulation", "choice", [("cpu", "CPU"), ("cuda", "CUDA (GPU)"), ("auto", "Automatic")], "cpu", ""),
    ("gpu_device", "GPU device", "int", (0, 16), 0, ""),
    ("gpu_memory_fraction", "GPU memory fraction", "float", (0.05, 0.95, 2), 0.7, ""),
    ("numba_threads", "Numba threads", "int", (1, 512), max(1, os.cpu_count() or 1), ""),
]
SETTINGS_SPEC_OUTPUT = [
    ("compute_pair_contributions_on_finish", "Pair contributions", "bool", None, True, "Compute partial pair contributions after the fit."),
    ("compute_warren_on_finish", "Warren analysis", "bool", None, False, ""),
    ("compute_local_trends_on_finish", "Local trends", "bool", None, False, ""),
    ("export_finite_coordination", "Export finite coordination", "bool", None, False, ""),
    ("compute_esd", "esds and correlation matrix", "bool", None, False,
     "After the refinement, compute estimated standard deviations and the correlation matrix "
     "(one extra model evaluation pair per refined parameter)."),
    ("esd_nyquist_correction", "Correct esds for correlated PDF points", "bool", None, False,
     "Count the data at the Nyquist spacing π/Qmax instead of every point: esds × √(N / N_eff)."),
]


class SettingsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.forms = []
        for title, spec in (
            ("Strategy and tolerances", SETTINGS_SPEC_STRATEGY),
            ("Compute backend", SETTINGS_SPEC_BACKEND),
            ("When a refinement finishes", SETTINGS_SPEC_OUTPUT),
        ):
            card = Card(title)
            form = SettingsForm(spec)
            form.edited.connect(self.edited)
            card.add(form)
            self.lay.addWidget(card)
            self.forms.append(form)
        self._known = {s[0] for spec in (SETTINGS_SPEC_STRATEGY, SETTINGS_SPEC_BACKEND, SETTINGS_SPEC_OUTPUT) for s in spec}
        card = Card("Other settings", "advanced keys kept from the input file")
        self.extra = DictEditor()
        self.extra.edited.connect(self._extra_edited)
        card.add(self.extra)
        self.lay.addWidget(card)
        self._extra: Dict[str, Any] = {}
        self.finish()

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project)
        for f in self.forms:
            f.bind(project.refinement)
        self._extra = {k: v for k, v in project.refinement.items() if k not in self._known}
        self.extra.bind(self._extra)

    def _extra_edited(self):
        for k in [k for k in self.project.refinement if k not in self._known]:
            self.project.refinement.pop(k, None)
        self.project.refinement.update(self._extra)
        self.edited.emit()


class ConstraintsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Constraints", "parameter = expression")
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("Parameter", "Expression", "Status"))
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 260)
        self.table.setColumnWidth(2, 200)
        self.table.itemChanged.connect(self._changed)
        card.add(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        b_add, b_del = QtWidgets.QPushButton("Add constraint"), QtWidgets.QPushButton("Remove selected")
        b_add.clicked.connect(self._add)
        b_del.clicked.connect(self._remove)
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        hint = QtWidgets.QLabel(
            "Names are dataset.phase.parameter (e.g. ds1.calcite.biso_O) or dataset.parameter "
            "(e.g. ds1.qdamp). Expressions may use + − × ÷, ** and abs, min, max, sqrt, log, exp. "
            "A constrained parameter is not refined; it follows its expression."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        card.add(hint)
        self.lay.addWidget(card, 1)
        self._block = False

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project)
        self._block = True
        try:
            items = list(project.constraints.items())
            self.table.setRowCount(len(items))
            for r, (k, v) in enumerate(items):
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
            self._validate()
        finally:
            self._block = False

    def _validate(self):
        names = {full for full, *_ in self.project.iter_parameters(active_only=False)}
        token = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*(?:\.[A-Za-z_][A-Za-z0-9_\-]*)+")
        for r in range(self.table.rowCount()):
            t = self.table.item(r, 0)
            e = self.table.item(r, 1)
            target = t.text().strip() if t else ""
            expr = e.text().strip() if e else ""
            unknown = [n for n in [target] + token.findall(expr) if n and n not in names]
            status = "OK" if not unknown else "Unknown: " + ", ".join(unknown)
            it = QtWidgets.QTableWidgetItem(status)
            it.setFlags(Qt.ItemIsEnabled)
            it.setForeground(QtGui.QColor("#15803d" if not unknown else "#b91c1c"))
            self.table.setItem(r, 2, it)

    def _changed(self, _item=None):
        if self._block:
            return
        self._block = True
        try:
            self.project.constraints.clear()
            for r in range(self.table.rowCount()):
                t, e = self.table.item(r, 0), self.table.item(r, 1)
                k = t.text().strip() if t else ""
                v = e.text().strip() if e else ""
                if k and v:
                    self.project.constraints[k] = v
            self._validate()
        finally:
            self._block = False
        self.edited.emit()

    def _add(self):
        self._block = True
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(""))
        self._block = False
        self.table.editItem(self.table.item(r, 0))

    def _remove(self):
        for r in sorted({i.row() for i in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(r)
        self._changed()


class RefinedPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Everything being refined")
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter by name or location…")
        self.filter.textChanged.connect(self._apply_filter)
        card.add_header_widget(self.filter)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("Name", "Where", "Value", "Min", "Max", "Esd", ""))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 260)
        self.table.setMinimumHeight(220)
        card.add(self.table, 1)
        self.lay.addWidget(card, 1)

        # --- esds and correlations ------------------------------------
        card = Card("esds and correlation matrix", "least squares at the current values; enable in Settings to compute after each refinement")
        row = QtWidgets.QHBoxLayout()
        self.btn_esd = QtWidgets.QPushButton("Compute esds now")
        self.btn_esd.setObjectName("primary")
        self.btn_esd.setToolTip("Jacobian at the current parameter values (2 model evaluations per refined parameter); no refinement.")
        self.btn_esd.clicked.connect(lambda: self.win.compute_esds())
        row.addWidget(self.btn_esd)
        row.addWidget(QtWidgets.QLabel("Show |ρ| ≥"))
        self.sp_thr = QtWidgets.QDoubleSpinBox()
        self.sp_thr.setRange(0.0, 1.0)
        self.sp_thr.setSingleStep(0.05)
        self.sp_thr.setValue(0.8)
        self.sp_thr.valueChanged.connect(lambda _v: self._show_uncertainty())
        row.addWidget(self.sp_thr)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        self.lbl_unc = QtWidgets.QLabel("No esd analysis yet.")
        self.lbl_unc.setObjectName("muted")
        self.lbl_unc.setWordWrap(True)
        self.lbl_unc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card.add(self.lbl_unc)
        split = QtWidgets.QSplitter(Qt.Horizontal)
        from pdf_fitting.gui.plots import MplPlot

        self.plot_corr = MplPlot()
        self.plot_corr.setMinimumHeight(380)
        split.addWidget(self.plot_corr)
        self.tbl_corr = QtWidgets.QTableWidget(0, 3)
        self.tbl_corr.setHorizontalHeaderLabels(("Parameter 1", "Parameter 2", "ρ"))
        self.tbl_corr.verticalHeader().setVisible(False)
        self.tbl_corr.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_corr.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        split.addWidget(self.tbl_corr)
        split.setSizes([600, 400])
        card.add(split, 1)
        self.lay.addWidget(card, 1)

    def _show_uncertainty(self):
        unc = (self.project.last_uncertainty or {}) if self.project is not None else {}
        ax = self.plot_corr.ax
        self.plot_corr.fig.clear()
        ax = self.plot_corr.ax = self.plot_corr.fig.add_subplot(111)
        self.tbl_corr.setRowCount(0)
        names = unc.get("names") or []
        if not names:
            self.lbl_unc.setText(unc.get("message") or "No esd analysis yet. Press “Compute esds now”, or enable it in Settings › When a refinement finishes.")
            ax.text(0.5, 0.5, "No correlation matrix", transform=ax.transAxes, ha="center", va="center", color="#6b7280")
            ax.set_axis_off()
            self.plot_corr.canvas.draw_idle()
            return
        from pdf_fitting.uncertainty import UncertaintyResult

        res = UncertaintyResult.from_json(unc)
        parts = [f"χ²red = {res.chi2_red:.4g}", res.message]
        if res.at_bound:
            parts.append("at a bound (esd not meaningful): " + ", ".join(res.at_bound))
        for sg in res.singular:
            combo = " ".join(f"{c:+.2f}·{n}" for n, c in sg["params"])
            parts.append(f"⚠ undetermined combination (relative singular value {sg['sv'] or 0:.1e}): {combo}")
        self.lbl_unc.setText(" · ".join(parts))
        short = [n.split(".", 1)[-1] if n.count(".") >= 1 else n for n in res.names]
        C = np.where(np.isfinite(res.correlation), res.correlation, 0.0)
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        n = len(short)
        fs = 8 if n <= 20 else (6 if n <= 40 else 4)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=90, fontsize=fs)
        ax.set_yticklabels(short, fontsize=fs)
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(C[i, j]) > 0.6 else "black")
        self.plot_corr.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.plot_corr.canvas.draw_idle()
        rows = res.strong_correlations(float(self.sp_thr.value()))
        self.tbl_corr.setRowCount(len(rows))
        for r, (a, b, c) in enumerate(rows):
            for j, t in enumerate((a, b, f"{c:+.3f}")):
                self.tbl_corr.setItem(r, j, QtWidgets.QTableWidgetItem(t))

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project)
        self._show_uncertainty()
        rows = []
        for full, p, ds, ph in project.iter_parameters():
            if not p.refine:
                continue
            local = full.rsplit(".", 1)[-1]
            if ph is None:
                where, node = f"{ds.name} › Instrument", f"ds:{ds.id}:instr"
            else:
                sec = S.section_of(local)
                sec_name = {"overview": "Phase", "lat": "Lattice", "atoms": "Atoms", "micro": "Microstructure", "dyn": "Local dynamics", "shape": "Shape & stacking", "saxs": "SAXS", "other": "Phase"}[sec]
                sec_id = {"overview": "", "other": ""}.get(sec, sec)
                where = f"{ds.name} › {ph.name} › {sec_name}"
                node = f"ph:{ds.id}:{ph.id}" + (f":{sec_id}" if sec_id else "")
            rows.append((full, where, p, node))
        self.table.setRowCount(len(rows))
        for r, (full, where, p, node) in enumerate(rows):
            for c, text in enumerate((full, where, fmt_value(p.value), fmt_value(p.min), fmt_value(p.max), "" if p.esd is None else f"{float(p.esd):.3g}")):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
            btn = QtWidgets.QPushButton("Go to")
            btn.clicked.connect(lambda _=False, n=node: self.navigate.emit(n))
            self.table.setCellWidget(r, 6, btn)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text):
        text = text.strip().lower()
        for r in range(self.table.rowCount()):
            a, b = self.table.item(r, 0), self.table.item(r, 1)
            hay = ((a.text() if a else "") + " " + (b.text() if b else "")).lower()
            self.table.setRowHidden(r, bool(text) and text not in hay)


# =============================================================================
# Dataset level
# =============================================================================


class DatasetPanel(Panel):
    """
    Classic PDF fit plot of the whole dataset (tick markers of every phase,
    pair contributions, curve styles, residual offset, γ(r), pop-out) and the
    phase table with scales and weight fractions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("G(r) — observed, calculated, difference", "tick markers, pairs and styles in the plot toolbar")
        for text, slot in (("Pop out", "pop_out"), ("Maximize", "maximize_pdf_plot"), ("Full screen", "fullscreen_pdf_plot")):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _=False, s=slot: self._plot_action(s))
            card.add_header_widget(btn)
        self.plot_host_area = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_host_area)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        # Shown while the plot is in its own window.
        self.placeholder = QtWidgets.QWidget()
        ph_lay = QtWidgets.QVBoxLayout(self.placeholder)
        ph_lay.addStretch(1)
        msg = QtWidgets.QLabel("The plot is in its own window.")
        msg.setAlignment(Qt.AlignCenter)
        msg.setObjectName("muted")
        ph_lay.addWidget(msg)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_back = QtWidgets.QPushButton("Bring the plot back here")
        self.btn_back.setObjectName("primary")
        self.btn_back.clicked.connect(self._dock_back)
        self.btn_show = QtWidgets.QPushButton("Show the plot window")
        self.btn_show.setToolTip("Bring the separate plot window to the front (e.g. when it is minimized or hidden).")
        self.btn_show.clicked.connect(self._show_window)
        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_show)
        btn_row.addStretch(1)
        ph_lay.addLayout(btn_row)
        ph_lay.addStretch(1)
        self.placeholder.setVisible(False)
        self.plot_layout.addWidget(self.placeholder)
        self.plot_host_area.setMinimumHeight(460)
        card.add(self.plot_host_area, 1)
        self.lay.addWidget(card, 3)

        card = Card("Phases in this dataset", "weight fractions are derived from the scale factors")
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(("Active", "Phase", "CIF", "Scale", "Refine", "Atom %", "Weight %", ""))
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.itemChanged.connect(self._changed)
        card.add(self.table)
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add phase from CIF…")
        self.btn_add.clicked.connect(lambda: self.win.add_phase(self.dataset))
        row.addWidget(self.btn_add)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        self.lay.addWidget(card, 1)
        self._block = False
        self._host = None

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset)
        self._fill_table()
        host = win.dataset_host(dataset)
        if host is not self._host:
            if self._host is not None:
                self._host.pdf_container.setParent(None)
            self._host = host
        host.dock_into(self.plot_layout, self.placeholder)
        win.push_dataset_result(dataset)

    def _dock_back(self):
        if self._host is not None:
            self._host.dock_into(self.plot_layout, self.placeholder)
            self._host.dock_back()
            self.win.push_dataset_result(self.dataset)

    def _show_window(self):
        if self._host is not None:
            self._host.pop_out(force_show=True)

    def _plot_action(self, name):
        if self._host is None:
            return
        if name == "pop_out":
            self._host.pop_out()
        else:
            self.placeholder.setVisible(True)
            getattr(self._host.win, name)()

    def _fractions(self):
        """Weight fractions from the current scales (same formula as the model)."""
        raw = {}
        for ph in self.dataset.active_phases():
            info = resolve_info(self.project, ph)
            scale = ph.parameters.get("scale", Parameter(1.0)).value
            try:
                scale = float(scale)
            except (TypeError, ValueError):
                scale = 0.0
            z = info.z_avg if info and info.ok else 1.0
            m = info.m_avg if info and info.ok else 1.0
            atoms = max(scale, 0.0) / max(z * z, 1e-12)
            raw[ph.id] = (atoms, atoms * m)
        n = sum(v[0] for v in raw.values()) or 1.0
        w = sum(v[1] for v in raw.values()) or 1.0
        return {k: (100 * v[0] / n, 100 * v[1] / w) for k, v in raw.items()}

    def _fill_table(self):
        self._block = True
        try:
            fr = self._fractions()
            phases = self.dataset.phases
            self.table.setRowCount(len(phases))
            for r, ph in enumerate(phases):
                it = QtWidgets.QTableWidgetItem("")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if ph.active else Qt.Unchecked)
                self.table.setItem(r, 0, it)
                it = QtWidgets.QTableWidgetItem("● " + ph.name)
                it.setForeground(QtGui.QColor(phase_color(r)))
                it.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(r, 1, it)
                it = QtWidgets.QTableWidgetItem(os.path.basename(ph.cif))
                it.setFlags(Qt.ItemIsEnabled)
                it.setToolTip(ph.cif)
                self.table.setItem(r, 2, it)
                p = ph.parameters.setdefault("scale", Parameter(1.0))
                self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(fmt_value(p.value)))
                it = QtWidgets.QTableWidgetItem("")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if p.refine else Qt.Unchecked)
                self.table.setItem(r, 4, it)
                a, w = fr.get(ph.id, (float("nan"), float("nan")))
                for c, val in ((5, a), (6, w)):
                    it = QtWidgets.QTableWidgetItem("—" if not ph.active or math.isnan(val) else f"{val:.1f}")
                    it.setFlags(Qt.ItemIsEnabled)
                    self.table.setItem(r, c, it)
                btn = QtWidgets.QPushButton("Open")
                btn.clicked.connect(lambda _=False, pid=ph.id: self.navigate.emit(f"ph:{self.dataset.id}:{pid}"))
                self.table.setCellWidget(r, 7, btn)
            self.table.setFixedHeight(self.table.horizontalHeader().height() + sum(self.table.rowHeight(i) for i in range(len(phases))) + 6)
        finally:
            self._block = False

    def _changed(self, item):
        if self._block:
            return
        ph = self.dataset.phases[item.row()]
        p = ph.parameters.setdefault("scale", Parameter(1.0))
        if item.column() == 0:
            ph.active = item.checkState() == Qt.Checked
        elif item.column() == 3:
            v = parse_number(item.text())
            if isinstance(v, float):
                p.value = v
        elif item.column() == 4:
            p.refine = item.checkState() == Qt.Checked
        else:
            return
        self._fill_table()
        self.edited.emit()


class DataPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Measured data", "X-ray G(r): first column r (Å), second column G(r)")
        grid = QtWidgets.QGridLayout()
        self.le_name = QtWidgets.QLineEdit()
        self.le_file = QtWidgets.QLineEdit()
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        self.sp_rmin = QtWidgets.QDoubleSpinBox()
        self.sp_rmax = QtWidgets.QDoubleSpinBox()
        self.sp_rext = QtWidgets.QDoubleSpinBox()
        for sp, lo, hi, dec in ((self.sp_rmin, 0, 1000, 3), (self.sp_rmax, 0, 1000, 3), (self.sp_rext, 1, 3, 3)):
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(0.1)
        grid.addWidget(QtWidgets.QLabel("Dataset name"), 0, 0)
        grid.addWidget(self.le_name, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("G(r) file"), 1, 0)
        grid.addWidget(self.le_file, 1, 1)
        grid.addWidget(btn, 1, 2)
        grid.addWidget(QtWidgets.QLabel("r min (Å)"), 2, 0)
        grid.addWidget(self.sp_rmin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("r max (Å)"), 3, 0)
        grid.addWidget(self.sp_rmax, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Pair-generation extension (× r max)"), 4, 0)
        grid.addWidget(self.sp_rext, 4, 1)
        grid.setColumnStretch(1, 1)
        card.body_layout.addLayout(grid)
        self.lay.addWidget(card)
        card = Card("Data preview")
        self.plot = MplPlot()
        self.plot.setMinimumHeight(300)
        card.add(self.plot, 1)
        self.lay.addWidget(card, 1)
        self.le_name.editingFinished.connect(self._save)
        self.le_file.editingFinished.connect(self._save)
        for sp in (self.sp_rmin, self.sp_rmax, self.sp_rext):
            sp.valueChanged.connect(lambda _v: self._save())
        self._block = False

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset)
        self._block = True
        try:
            self.le_name.setText(dataset.name)
            self.le_file.setText(dataset.data_file)
            self.sp_rmin.setValue(float(dataset.r_min))
            self.sp_rmax.setValue(float(dataset.r_max))
            self.sp_rext.setValue(float(dataset.r_extension))
        finally:
            self._block = False
        self._preview()

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select G(r) data", os.path.dirname(self.project.resolve_path(self.dataset.data_file) or ""), "G(r) data (*.gr *.xy *.dat *.txt);;All files (*)")
        if path:
            self.le_file.setText(path)
            self._save()

    def _save(self):
        if self._block or self.dataset is None:
            return
        self.dataset.name = self.le_name.text().strip() or self.dataset.name
        self.dataset.data_file = self.le_file.text().strip()
        self.dataset.r_min = self.sp_rmin.value()
        self.dataset.r_max = self.sp_rmax.value()
        self.dataset.r_extension = self.sp_rext.value()
        self._preview()
        self.edited.emit()

    def _preview(self):
        ax = self.plot.ax
        ax.clear()
        path = self.project.resolve_path(self.dataset.data_file)
        try:
            data = np.loadtxt(path, unpack=True)
            r, G = data[0], data[1]
            ax.plot(r, G, color="#9ca3af", lw=0.8, label="Full file")
            m = (r >= self.dataset.r_min) & (r <= self.dataset.r_max)
            ax.plot(r[m], G[m], color="#111827", lw=1.0, label="Fitted range")
            ax.legend(fontsize=8)
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("G(r)")
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, f"Cannot read data file:\n{exc}", ha="center", va="center", transform=ax.transAxes, color="#b91c1c", wrap=True)
        self.plot.canvas.draw_idle()


class InstrumentPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sec = ParamSection("Instrument resolution", "shared by every phase of this dataset")
        self.sec.edited.connect(self.edited)
        self.lay.addWidget(self.sec)
        self.finish()

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset)
        names = list(S.INSTRUMENT) + [n for n in dataset.parameters if n not in S.INSTRUMENT]
        self.sec.set_rows(dataset.parameters, names, candidates=S.INSTRUMENT)


# =============================================================================
# Phase level
# =============================================================================


class PhaseOverviewPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Phase")
        grid = QtWidgets.QGridLayout()
        self.le_name = QtWidgets.QLineEdit()
        self.le_cif = QtWidgets.QLineEdit()
        self.le_cif.setReadOnly(True)
        btn_cif = QtWidgets.QPushButton("Replace CIF…")
        btn_cif.clicked.connect(self._replace_cif)
        self.chk_active = QtWidgets.QCheckBox("Include this phase in the fit")
        self.lbl_info = QtWidgets.QLabel()
        self.lbl_info.setObjectName("muted")
        self.lbl_info.setWordWrap(True)
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        grid.addWidget(self.le_name, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Structure (CIF)"), 1, 0)
        grid.addWidget(self.le_cif, 1, 1)
        grid.addWidget(btn_cif, 1, 2)
        grid.addWidget(self.chk_active, 2, 1)
        grid.addWidget(self.lbl_info, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        card.body_layout.addLayout(grid)
        self.lay.addWidget(card)

        self.scale = ParamSection("Scale", "weight fraction is derived from the scales of all phases", allow_remove=False)
        self.scale.edited.connect(self.edited)
        self.lay.addWidget(self.scale)

        card = Card("Shell generation")
        self.form = SettingsForm([
            ("force_p1_shells", "Force P1 shells", "bool", None, False, "Generate neighbor rows from every site instead of the asymmetric unit."),
        ], columns=1)
        self.form.edited.connect(self.edited)
        card.add(self.form)
        self.lay.addWidget(card)

        self.other = ParamSection("Other parameters", "parameters not shown in the other sections")
        self.other.edited.connect(self.edited)
        self.lay.addWidget(self.other)
        self.finish()

        self.le_name.editingFinished.connect(self._save)
        self.chk_active.toggled.connect(lambda _v: self._save())
        self._block = False

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        self._block = True
        try:
            self.le_name.setText(phase.name)
            self.le_cif.setText(phase.cif)
            self.chk_active.setChecked(phase.active)
            info = resolve_info(project, phase)
            if info is None:
                self.lbl_info.setText("CIF file not found.")
            elif not info.ok:
                self.lbl_info.setText(f"Could not read CIF: {info.error}")
            else:
                self.lbl_info.setText(f"{info.formula} · space group {info.spacegroup} · elements {', '.join(info.elements)} · {len(info.sites)} site(s) in the asymmetric unit")
        finally:
            self._block = False
        phase.parameters.setdefault("scale", Parameter(1.0))
        self.scale.set_rows(phase.parameters, ["scale"])
        self.form.bind(phase.settings)
        others = [n for n in phase.parameters if S.section_of(n) == "other"]
        self.other.set_rows(phase.parameters, others)
        self.other.setVisible(bool(others))

    def _save(self):
        if self._block or self.phase is None:
            return
        self.phase.name = self.le_name.text().strip() or self.phase.name
        self.phase.active = self.chk_active.isChecked()
        self.edited.emit()

    def _replace_cif(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CIF", os.path.dirname(self.project.resolve_path(self.phase.cif) or ""), "CIF files (*.cif);;All files (*)")
        if not path:
            return
        if (self.phase.sections or {}).get("structure_edit"):
            if QtWidgets.QMessageBox.question(
                self, "Replace CIF",
                "This phase has sites added / deleted in the structure editor. "
                "Replacing the CIF discards those edits. Continue?",
            ) != QtWidgets.QMessageBox.Yes:
                return
            self.phase.sections.pop("structure_edit", None)
        self.phase.cif = path
        self.load(self.win, self.project, self.dataset, self.phase)
        self.edited.emit()


class LatticePanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sec = ParamSection("Unit cell")
        self.sec.edited.connect(self._edited)
        self.lay.addWidget(self.sec)
        self.finish()

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        info = resolve_info(project, phase)
        locks: Dict[str, str] = {}
        defaults = {}
        if info and info.ok:
            defaults = dict(info.lattice)
            for k, v in info.constraints.items():
                locks[k] = f"= {v}" if isinstance(v, str) else "symmetry"
            self.sec.note_label.setText(f"constrained by {info.spacegroup} symmetry")
        # Locked parameters show the value they will take.
        for k, v in locks.items():
            if k in phase.parameters:
                src = info.constraints[k]
                if isinstance(src, str) and src in phase.parameters:
                    phase.parameters[k].value = phase.parameters[src].value
                elif not isinstance(src, str):
                    phase.parameters[k].value = float(src)
                phase.parameters[k].refine = False
        self.sec.set_rows(phase.parameters, list(S.LATTICE), candidates=S.LATTICE, locks=locks, defaults=defaults)

    def _edited(self):
        self.load(self.win, self.project, self.dataset, self.phase)
        self.edited.emit()


class AtomsPanel(Panel):
    MAX_TABLE_SITES = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Atomic sites (asymmetric unit)", "Rx/Ry/Rz/Rocc refine flags · fixed and tied coordinates are handled by symmetry")
        self.table = SiteParamTableWidget()
        self.table.setMinimumHeight(150)
        self.table.changed.connect(self._table_edited)
        self.table.refineFlagToggled.connect(self._sync_element_box)
        self.lbl_big = QtWidgets.QLabel()
        self.lbl_big.setObjectName("muted")
        self.lbl_big.setWordWrap(True)
        card.add(self.table, 1)
        card.add(self.lbl_big)
        self.lay.addWidget(card, 1)

        row = QtWidgets.QHBoxLayout()
        self.box_occ, self.lay_occ = self._element_box("Refine occupancy by element")
        self.box_xyz, self.lay_xyz = self._element_box("Refine fractional coordinates by element")
        row.addWidget(self.box_occ, 1)
        row.addWidget(self.box_xyz, 1)
        self.lay.addLayout(row)

        self.limits = SettingsForm([
            ("occ_min", "Occupancy minimum", "float", (0.0, 1.0, 3), 0.0, "Lower limit for every refined occupancy (per-site Occ Min overrides)."),
            ("occ_max", "Occupancy maximum", "float", (0.0, 1.0, 3), 1.0, "Upper limit for every refined occupancy."),
            ("xyz_max_shift", "Max coordinate shift (Å)", "float", (0.0, 5.0, 3), 0.3, "Largest shift of a refined coordinate from the CIF value (0 = no limit)."),
            ("biso_by_site", "One Biso per site", "bool", None, False, "Unticked: one Biso per element (recommended for large structures)."),
        ])
        self.limits.edited.connect(self._limits_edited)
        card = Card("Limits and Biso mode")
        card.add(self.limits)
        self.lay.addWidget(card)

        self.biso = ParamSection("Biso per element", "Å²")
        self.biso.edited.connect(self.edited)
        self.lay.addWidget(self.biso)

        # Anisotropic displacement parameters (β), per site.
        self.adp = ParamSection(
            "Anisotropic displacement (β)",
            "exp(−Σ h_i h_j β_ij) · site symmetry: tied components follow the free ones, fixed ones are 0",
            allow_remove=False,
        )
        self.adp.edited.connect(self.edited)
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("Site"))
        self.cmb_adp_site = QtWidgets.QComboBox()
        self.cmb_adp_site.setMinimumWidth(160)
        arow.addWidget(self.cmb_adp_site)
        self.btn_aniso = QtWidgets.QPushButton("Make anisotropic")
        self.btn_aniso.setToolTip("Create β11…β23 for the site from its isotropic B (β = B/4 · G*), "
                                  "so the fit is unchanged; the free components get the site's Biso refine flag.")
        self.btn_aniso.clicked.connect(lambda: self._make_anisotropic(True))
        self.btn_iso = QtWidgets.QPushButton("Make isotropic")
        self.btn_iso.setToolTip("Remove the β parameters of the site (its B_eq goes to the site Biso when Biso is per site).")
        self.btn_iso.clicked.connect(lambda: self._make_anisotropic(False))
        arow.addWidget(self.btn_aniso)
        arow.addWidget(self.btn_iso)
        arow.addStretch(1)
        self.adp.body_layout.insertLayout(0, arow)
        self.lbl_adp = QtWidgets.QLabel("")
        self.lbl_adp.setObjectName("muted")
        self.lbl_adp.setWordWrap(True)
        self.adp.add(self.lbl_adp)
        self.lay.addWidget(self.adp)
        self.finish()
        self._records: List[Dict[str, Any]] = []
        self._checks = {"occ": {}, "xyz": {}}
        self._loading = False

    # ------------------------------------------------------------------
    # Anisotropic displacement
    # ------------------------------------------------------------------
    def _adp_context(self):
        """(PhaseStructure, ops, current lattice) of the phase, cached per load."""
        from pdf_fitting.project.structure_tools import phase_structure
        from pdf_fitting.structure_handler import symmetry_operations

        path = self.project.resolve_path(self.project.structure_path(self.phase))
        local = {k: p.value for k, p in self.phase.parameters.items()}
        ps = phase_structure(path, local)
        return ps, symmetry_operations(ps.structure), ps.structure.lattice

    def _fill_adp(self):
        import re as _re

        from pdf_fitting.structure_handler import ADP_COMPONENTS, adp_site_rotations, beta_constraints, beta_matrix

        params = self.phase.parameters
        keys = []
        for rec in self._records:
            from pdf_fitting.structure_handler import safe_param_key

            k = safe_param_key(rec.get("label", ""))
            if k and k not in [x[0] for x in keys]:
                keys.append((k, rec.get("label", ""), rec.get("el", "")))
        cur = self.cmb_adp_site.currentData()
        self.cmb_adp_site.blockSignals(True)
        self.cmb_adp_site.clear()
        for k, label, el in keys:
            self.cmb_adp_site.addItem(f"{label} ({el})", k)
        i = self.cmb_adp_site.findData(cur)
        self.cmb_adp_site.setCurrentIndex(max(i, 0))
        self.cmb_adp_site.blockSignals(False)
        aniso = sorted({n.lower().split("_", 1)[1] for n in params if S.is_beta(n)})
        names, locks, notes = [], {}, []
        if aniso:
            try:
                ps, ops, lat = self._adp_context()
                G = lat.matrix @ lat.matrix.T
                ref = {k: ps.reference[i] for i, k in enumerate(ps.group_keys)}
            except Exception as exc:  # noqa: BLE001
                ps = None
                notes.append(f"structure not available: {exc}")
            for key in aniso:
                cons = None
                if ps is not None and key in ref:
                    cons = beta_constraints(adp_site_rotations(ref[key], ops))
                vals = {}
                for c in ADP_COMPONENTS:
                    name = next((n for n in params if n.lower() == f"beta{c}_{key}"), None)
                    if name is None:
                        continue
                    names.append(name)
                    vals[c] = params[name].value or 0.0
                    if cons is not None:
                        kind, terms = cons[c]
                        if kind == "fixed":
                            locks[name] = "0 (symmetry)"
                        elif kind == "tied":
                            locks[name] = "= " + " + ".join(f"{w:g}·β{t}" for t, w in terms)
                if ps is not None and cons is not None:
                    b = []
                    for c in ADP_COMPONENTS:
                        kind, terms = cons[c]
                        b.append(vals.get(c, 0.0) if kind == "free" else (0.0 if kind == "fixed" else sum(w * vals.get(t, 0.0) for t, w in terms)))
                    beq = 4.0 / 3.0 * float(np.trace(beta_matrix(b) @ G))
                    notes.append(f"{key}: B_eq = {beq:.4g} Å²")
        self.adp.set_rows(params, names, locks=locks)
        self.lbl_adp.setText(" · ".join(notes) if notes else "No anisotropic sites. Choose a site and press “Make anisotropic”.")

    def _make_anisotropic(self, on: bool):
        from pdf_fitting.project.model import Parameter as _P
        from pdf_fitting.structure_handler import ADP_COMPONENTS, adp_site_rotations, beta_constraints, beta_matrix, symmetrize_beta

        key = self.cmb_adp_site.currentData()
        if not key:
            return
        params = self.phase.parameters
        el = next((r.get("el", "") for r in self._records if S_safe(r.get("label", "")) == key), "")
        if on:
            try:
                ps, ops, lat = self._adp_context()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Anisotropic displacement", f"Structure not available: {exc}")
                return
            ref = {k: ps.reference[i] for i, k in enumerate(ps.group_keys)}
            if key not in ref:
                QtWidgets.QMessageBox.warning(self, "Anisotropic displacement", f"Site {key} not found in the structure.")
                return
            lc = {n.lower(): n for n in params}
            bname = lc.get(f"biso_{key}") or lc.get(f"biso_{el.lower()}")
            B = float(params[bname].value) if bname else 0.5
            refine = bool(params[bname].refine) if bname else False
            Gs = np.linalg.inv(lat.matrix @ lat.matrix.T)
            rots = adp_site_rotations(ref[key], ops)
            b6 = symmetrize_beta(np.array([Gs[0, 0], Gs[1, 1], Gs[2, 2], Gs[0, 1], Gs[0, 2], Gs[1, 2]]) * B / 4.0, rots)
            source = "B = " + f"{B:.4g} Å²"
            try:
                from pdf_fitting.project.edited_cif import cif_aniso_betas

                cif_b = cif_aniso_betas(self.project.resolve_path(self.project.structure_path(self.phase)))
                lab = next((l for l in cif_b if S_safe(l) == key), None)
                if lab is not None:
                    b6 = symmetrize_beta(np.asarray(cif_b[lab], dtype=float), rots)
                    source = "the CIF anisotropic values"
            except Exception as exc:  # noqa: BLE001
                print(f"[ADP] CIF aniso values not read: {exc}")
            cons = beta_constraints(rots)
            for c, v in zip(ADP_COMPONENTS, b6):
                name = f"beta{c}_{key}"
                p = params.setdefault(name, _P(value=float(v)))
                p.value = float(v)
                p.refine = refine and cons[c][0] == "free"
            self.win.statusBar().showMessage(f"{key}: anisotropic from {source} (the next Calculate rebuilds the shells once).", 8000)
        else:
            removed = [n for n in list(params) if S.is_beta(n) and n.lower().endswith("_" + key)]
            if not removed:
                return
            if self.phase.settings.get("biso_by_site"):
                try:
                    ps, ops, lat = self._adp_context()
                    G = lat.matrix @ lat.matrix.T
                    vals = [params[f"beta{c}_{key}"].value if f"beta{c}_{key}" in params else 0.0 for c in ADP_COMPONENTS]
                    beq = 4.0 / 3.0 * float(np.trace(beta_matrix(vals) @ G))
                    params.setdefault(f"biso_{key}", _P(value=beq)).value = beq
                except Exception:  # noqa: BLE001
                    pass
            for n in removed:
                params.pop(n, None)
        self._fill_adp()
        self.edited.emit()

    def _element_box(self, title):
        box = Card(title)
        lay = QtWidgets.QHBoxLayout()
        lay.addStretch(1)
        box.body_layout.addLayout(lay)
        return box, lay

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        self._loading = True
        try:
            info = resolve_info(project, phase)
            self._records = list(info.sites) if info and info.ok else []
            elements = info.elements if info and info.ok else []
            params = phase.parameters
            initial = {k: p.value for k, p in params.items()}
            refinable = {k: p.refine for k, p in params.items()}
            bounds = {k: (p.min, p.max) for k, p in params.items() if p.min is not None or p.max is not None}
            use = {k: True for k in params if k.startswith("biso_") and not self._is_element_biso(k, elements)}
            big = len(self._records) > self.MAX_TABLE_SITES
            self.table.setVisible(not big)
            self.lbl_big.setVisible(big)
            if big:
                self.lbl_big.setText(f"{len(self._records):,} sites: the site table is hidden for speed. Use the element boxes below to refine occupancies or coordinates of whole elements.")
                self.table.set_sites([], {}, {}, {}, {})
            else:
                self.table.set_sites(self._records, initial, refinable, bounds=bounds, use_flags=use)
                # Size the table to its rows (up to ~14 rows, then scroll).
                rows_h = sum(self.table.rowHeight(r) for r in range(min(self.table.rowCount(), 14)))
                self.table.setFixedHeight(self.table.horizontalHeader().height() + rows_h + 24)
            self.limits.bind(phase.settings)
            self.table.set_biso_controls_enabled(bool(phase.settings.get("biso_by_site", False)))

            for kind, lay, key in (("occ", self.lay_occ, "refine_occ_species"), ("xyz", self.lay_xyz, "refine_xyz_species")):
                while lay.count():
                    it = lay.takeAt(0)
                    if it.widget():
                        it.widget().deleteLater()
                selected = {s.lower() for s in re.split(r"[,;\s]+", str(phase.settings.get(key, "") or "")) if s}
                self._checks[kind] = {}
                for el in elements:
                    chk = QtWidgets.QCheckBox(el)
                    chk.setChecked(el.lower() in selected)
                    chk.toggled.connect(lambda v, k=kind, e=el: self._element_toggled(k, e, v))
                    lay.addWidget(chk)
                    self._checks[kind][el] = chk
                lay.addStretch(1)
                for el in elements:
                    if el.lower() in selected:
                        self.table.set_refine_flags_for_element(el, True, kind)

            label = (getattr(self, "focus", {}) or {}).get("label")
            if label and not big:
                for r in range(self.table.rowCount()):
                    it = self.table.item(r, 1)
                    if it is not None and it.text().strip() == str(label):
                        self.table.selectRow(r)
                        self.table.scrollToItem(it)
                        break

            bnames = [f"biso_{el}" for el in elements]
            existing = [k for k in params if k.startswith("biso_") and self._is_element_biso(k, elements)]
            names = list(dict.fromkeys(existing + bnames))
            self.biso.set_rows(params, names, candidates=bnames, defaults={n: 0.5 for n in bnames})
            self._fill_adp()
        finally:
            self._loading = False

    @staticmethod
    def _is_element_biso(key, elements):
        raw = key[5:]
        return any(raw.lower() == el.lower() for el in elements)

    def _table_edited(self):
        if self._loading or self.phase is None:
            return
        ini, ref, bnd, use = self.table.extract_all()
        params = self.phase.parameters
        for k, v in ini.items():
            if k.startswith("biso_") and not use.get(k, False):
                continue
            p = params.setdefault(k, Parameter())
            v2 = parse_number(v) if isinstance(v, str) else v
            p.value = 0.0 if v2 is None else v2
            p.refine = bool(ref.get(k, False))
            lo, hi = None, None
            if k in bnd:
                parts = [s.strip() for s in str(bnd[k]).split(",", 1)]
                if len(parts) == 2:
                    lo, hi = parse_number(parts[0]), parse_number(parts[1])
            p.min = lo if isinstance(lo, float) else None
            p.max = hi if isinstance(hi, float) else None
        self.edited.emit()

    def _element_toggled(self, kind, element, checked):
        if self._loading:
            return
        key = "refine_occ_species" if kind == "occ" else "refine_xyz_species"
        selected = [el for el, chk in self._checks[kind].items() if chk.isChecked()]
        self.phase.settings[key] = ", ".join(selected)
        self.table.set_refine_flags_for_element(element, checked, kind)
        self.edited.emit()

    def _sync_element_box(self, element, kind):
        state = self.table.element_refine_state(element, kind)
        chk = next((c for e, c in self._checks.get(kind, {}).items() if e.lower() == element.lower()), None)
        if state is None or chk is None:
            return
        chk.blockSignals(True)
        chk.setChecked(bool(state))
        chk.blockSignals(False)
        key = "refine_occ_species" if kind == "occ" else "refine_xyz_species"
        self.phase.settings[key] = ", ".join(e for e, c in self._checks[kind].items() if c.isChecked())

    def _limits_edited(self):
        self.table.set_biso_controls_enabled(bool(self.phase.settings.get("biso_by_site", False)))
        self.edited.emit()


class MicroPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.size = ParamSection("Crystallite size", "lognormal distribution (d, d_std)")
        self.cyl = ParamSection("Cylinder / disk size", "hkl-dependent common volume")
        card = Card("Microstrain model")
        self.combo = QtWidgets.QComboBox()
        for value, text in (("isotropic", "Isotropic"), ("wilkens", "Wilkens (dislocations)"), ("pah", "PAH (anisotropic)")):
            self.combo.addItem(text, value)
        self.combo.currentIndexChanged.connect(self._model_changed)
        card.add(self.combo)
        self.strain = ParamSection("Microstrain parameters")
        self.contrast = ParamSection("Dislocation contrast factors", "cubic A/B or the invariants E_i of the Laue class")
        self.lbl_terms = QtWidgets.QLabel("")
        self.lbl_terms.setObjectName("muted")
        self.lbl_terms.setWordWrap(True)
        self.lbl_terms.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.contrast.add(self.lbl_terms)
        crow = QtWidgets.QHBoxLayout()
        self.btn_import_cf = QtWidgets.QPushButton("Import contrast factors…")
        self.btn_import_cf.setToolTip(
            "Read coefficients from a text or JSON file: 'EdgeE1 = …', 'ScrewE2 = …', 'E1 = …' (edge and screw alike), "
            "CEdgeA/B, CScrewA/B, burgers_mag, rho, fE, Re. Intended also for DISCO results."
        )
        self.btn_import_cf.clicked.connect(self._import_contrast)
        crow.addWidget(self.btn_import_cf)
        crow.addStretch(1)
        self.contrast.body_layout.addLayout(crow)
        self._sg_cache: Dict[str, Any] = {}
        for w in (self.size, self.cyl):
            self.lay.addWidget(w)
        self.lay.addWidget(card)
        for w in (self.strain, self.contrast):
            self.lay.addWidget(w)
        for w in (self.size, self.cyl, self.strain, self.contrast):
            w.edited.connect(self.edited)
        self.finish()
        self._loading = False

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        self._loading = True
        try:
            params = phase.parameters
            self.size.set_rows(params, list(S.SIZE), candidates=S.SIZE)
            self.cyl.set_rows(params, list(S.CYLINDER), candidates=S.CYLINDER)
            model = str(phase.settings.get("microstrain_model", "isotropic")).lower()
            idx = self.combo.findData(model)
            self.combo.setCurrentIndex(max(idx, 0))
            self._fill_strain()
        finally:
            self._loading = False

    def _fill_strain(self):
        params = self.phase.parameters
        model = self.combo.currentData()
        names = list(S.STRAIN_BY_MODEL.get(model, ()))
        self.strain.set_rows(params, names, candidates=names)
        from pdf_fitting.project.contrast_factors import invariant_note, parameter_names

        contrast = [n for n in params if S.is_contrast(n)]
        sg = self._space_group()
        cands = parameter_names(sg)
        self.contrast.set_rows(params, contrast + [c for c in cands if c not in contrast], candidates=cands)
        self.lbl_terms.setText(invariant_note(sg) if sg else "Space group not available: invariant names E1…E3 offered.")
        self.contrast.setVisible(model in ("wilkens", "pah"))

    def _space_group(self):
        from pdf_fitting.project.contrast_factors import phase_space_group

        try:
            path = self.project.resolve_path(self.project.structure_path(self.phase))
        except Exception:  # noqa: BLE001
            return None
        key = (path, tuple(round(float(self.phase.parameters[k].value), 6) for k in ("a", "b", "c", "alpha", "beta", "gamma") if k in self.phase.parameters))
        if key not in self._sg_cache:
            try:
                local = {k: p.value for k, p in self.phase.parameters.items()}
                self._sg_cache[key] = phase_space_group(path, local)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[MICRO] space group of {self.phase.name} not available: {exc}")
                self._sg_cache[key] = None
        return self._sg_cache[key]

    def _import_contrast(self):
        from pdf_fitting.project.contrast_factors import read_contrast_file
        from pdf_fitting.project.model import Parameter as _P

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import contrast factors", os.path.dirname(self.project.path or "") if getattr(self.project, "path", None) else "",
            "Contrast factors (*.txt *.dat *.json *.inp *.out);;All files (*)",
        )
        if not path:
            return
        try:
            values = read_contrast_file(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import contrast factors", str(exc))
            return
        params = self.phase.parameters
        lc = {k.lower(): k for k in params}
        lines = []
        for name, v in values.items():
            key = lc.get(name.lower(), name)
            old = params[key].value if key in params else None
            lines.append(f"{key}: {'(new)' if old is None else fmt_value(old)} → {v:.6g}")
        if QtWidgets.QMessageBox.question(self, "Import contrast factors", f"Set these parameters of {self.phase.name}?" + chr(10) * 2 + chr(10).join(lines)) != QtWidgets.QMessageBox.Yes:
            return
        for name, v in values.items():
            key = lc.get(name.lower(), name)
            params.setdefault(key, _P(value=float(v))).value = float(v)
        self._fill_strain()
        self.edited.emit()

    def _model_changed(self, _i):
        if self._loading:
            return
        self.phase.settings["microstrain_model"] = self.combo.currentData()
        self._fill_strain()
        self.edited.emit()


class DynamicsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.lay.addWidget(self.banner)
        self.deltas = ParamSection("Correlated motion (δ1 / δ2)", "global and per element pair")
        self.deltas.edited.connect(self.edited)
        self.lay.addWidget(self.deltas)
        card = Card("Pair-dependent λ coefficients", "per element pair and coordination shell")
        self.lambdas = LambdaListWidget()
        self.lambdas.setMinimumHeight(240)
        self.lambdas.changed.connect(self._lambdas_edited)
        card.add(self.lambdas, 1)
        self.lay.addWidget(card, 1)
        self._loading = False

    def load(self, win, project, dataset=None, phase=None, focus_element: str = ""):
        super().load(win, project, dataset, phase)
        focus = getattr(self, "focus", {}) or {}
        focus_pair = str(focus.get("pair") or "")
        focus_k = focus.get("k")
        self._loading = True
        try:
            info = resolve_info(project, phase)
            pairs = info.pairs if info and info.ok else []
            params = phase.parameters
            cands = ["delta1", "delta2"] + [f"delta{i}_{p}" for p in pairs for i in (1, 2)]
            present = [n for n in params if S.is_dynamics(n)]
            order = list(dict.fromkeys(["delta1", "delta2"] + sorted(present) + cands))
            if focus_element:
                fe = focus_element.lower()
                order = sorted(order, key=lambda n: 0 if fe in n.lower().split("_")[-1].split("-") else 1)
                self.banner.setText(f"Opened from the structure view: pairs with {focus_element} are listed first.")
            self.banner.setVisible(bool(focus_element))
            if focus_pair:
                order = sorted(order, key=lambda n: 0 if n.lower().endswith("_" + focus_pair) else 1)
                d = focus.get("distance")
                self.banner.setText(
                    f"From the structure view: pair <b>{focus_pair}</b>"
                    + (f", shell k = <b>{focus_k}</b>" if focus_k is not None else "")
                    + (f" (distance {float(d):.4f} Å)" if d is not None else "")
                    + ". Its δ rows are listed first and its λ coefficients are selected below; add λ "
                    + "coefficients up to this k if they are missing."
                )
                self.banner.setVisible(True)
            self.deltas.set_rows(params, order, candidates=cands)
            lam = {k: p for k, p in params.items() if S.is_lambda(k)}
            self.lambdas.set_pairs(pairs, commit_current=False)
            self.lambdas.set_lambdas(
                {k: p.value for k, p in lam.items()},
                {k: p.refine for k, p in lam.items()},
                {k: (fmt_value(p.min), fmt_value(p.max)) for k, p in lam.items()},
            )
            if focus_pair:
                idx = self.lambdas.combo_pair.findText(focus_pair)
                if idx >= 0:
                    self.lambdas.combo_pair.setCurrentIndex(idx)
                if focus_k is not None:
                    tbl = self.lambdas.tbl
                    for r in range(tbl.rowCount()):
                        it = tbl.item(r, 0)
                        if it is not None and it.text().strip() == str(focus_k):
                            tbl.selectRow(r)
                            tbl.scrollToItem(it)
                            break
        finally:
            self._loading = False

    def _lambdas_edited(self):
        if self._loading or self.phase is None:
            return
        ini, ref, bnd = self.lambdas.extract_all()
        params = self.phase.parameters
        for k in [k for k in params if S.is_lambda(k) and k not in ini]:
            params.pop(k)
        for k, v in ini.items():
            p = params.setdefault(k, Parameter())
            p.value = v if not isinstance(v, str) else (parse_number(v) or 0.0)
            p.refine = bool(ref.get(k, False))
            lo, hi = bnd.get(k, ("", ""))
            lo, hi = parse_number(lo), parse_number(hi)
            p.min = lo if isinstance(lo, float) else None
            p.max = hi if isinstance(hi, float) else None
        self.edited.emit()


class ShapePanel(Panel):
    """Finite crystallite shape and stacking-fault model (classic builder and 3D views)."""

    STACK_SECTIONS = ("layer_stacking", "stacking_search", "stacking_vector_search", "stacking_refinement", "stacking_faults")

    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("Crystallite shape and stacking faults")
        self.lbl_summary = QtWidgets.QLabel()
        self.lbl_summary.setWordWrap(True)
        card.add(self.lbl_summary)
        row = QtWidgets.QHBoxLayout()
        self.btn_builder = QtWidgets.QPushButton("Open crystallite & stacking builder…")
        self.btn_builder.setObjectName("primary")
        self.btn_builder.setToolTip("Disks, cylinders, prisms, ellipsoids, layer sequences, stacking search and refinement — with 3D views.")
        self.btn_builder.clicked.connect(self._open_builder)
        self.btn_bulk = QtWidgets.QPushButton("Use conventional (bulk) model")
        self.btn_bulk.clicked.connect(self._use_bulk)
        self.btn_grid = QtWidgets.QPushButton("Run crystallite-shape grid search…")
        self.btn_grid.setToolTip("Classic CS grid search for this phase alone; results appear in Results › CS grid.")
        self.btn_grid.clicked.connect(self._run_grid)
        for b in (self.btn_builder, self.btn_bulk, self.btn_grid):
            row.addWidget(b)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        self.lay.addWidget(card)

        self.stack_params = ParamSection("Stacking parameters")
        self.stack_params.edited.connect(self.edited)
        self.lay.addWidget(self.stack_params)

        self.advanced = QtWidgets.QGroupBox("Advanced: raw settings")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        adv = QtWidgets.QVBoxLayout(self.advanced)
        self.tabs = QtWidgets.QTabWidget()
        self.shape = DictEditor()
        self.shape.edited.connect(self.edited)
        self.tabs.addTab(self.shape, "Crystallite shape")
        self.stack_editors = {}
        for name in self.STACK_SECTIONS:
            ed = DictEditor()
            ed.edited.connect(self.edited)
            self.stack_editors[name] = ed
            self.tabs.addTab(ed, name.replace("_", " ").capitalize())
        adv.addWidget(self.tabs)
        self.tabs.setVisible(False)
        self.advanced.toggled.connect(self.tabs.setVisible)
        self.lay.addWidget(self.advanced)
        self.finish()

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        shape = phase.sections.setdefault("crystallite_shape", {})
        mode = str(shape.get("mode", "conventional") or "conventional")
        stacking = phase.sections.get("layer_stacking") or {}
        seq = stacking.get("sequence") if isinstance(stacking, dict) else None
        parts = [f"Model: <b>{'finite crystallite' if mode == 'finite_shape' else 'conventional (bulk, size from d / cylinder)'}</b>"]
        if mode == "finite_shape":
            desc = ", ".join(f"{k} = {v}" for k, v in shape.items() if k not in ("mode",) and not isinstance(v, (dict, list)))[:300]
            if desc:
                parts.append(desc)
        if isinstance(seq, list) and seq:
            parts.append(f"Layer stacking: {len(seq)} layer(s) in the sequence")
        self.lbl_summary.setText("<br>".join(parts))
        self.btn_bulk.setEnabled(mode == "finite_shape")
        stack = [n for n in phase.parameters if S.is_stacking(n)]
        self.stack_params.set_rows(phase.parameters, stack)
        self.stack_params.setVisible(bool(stack))
        self.shape.bind(shape)
        for name, ed in self.stack_editors.items():
            ed.bind(phase.sections.setdefault(name, {}))

    def _open_builder(self):
        host = self.win.phase_host(self.dataset, self.phase)
        host.user_session = True
        # Modal. On "accept" the Builder emits changed; the window merges the
        # new shape / stacking state into this phase (cancel changes nothing).
        host.win.builder.open_crystallite_shape_dialog()

    def _use_bulk(self):
        self.phase.sections["crystallite_shape"] = {"mode": "conventional"}
        self.load(self.win, self.project, self.dataset, self.phase)
        self.edited.emit()

    def _run_grid(self):
        host = self.win.phase_host(self.dataset, self.phase)
        host.user_session = True  # applying a grid candidate updates the phase
        host.grid_used = True     # show the CS grid tab in Results
        self.win.select_node(f"ph:{self.dataset.id}:{self.phase.id}:results")
        try:
            host.win.run_cs_grid_search()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Grid search failed", str(exc))


class SaxsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(parent)
        card = Card("SAXS envelope", "low-r particle-shape term of this phase")
        self.form = SettingsForm([
            ("saxs_enabled", "Enable SAXS term", "bool", None, False, ""),
            ("saxs_model", "Model", "choice", [("sphere", "Sphere"), ("cylinder", "Cylinder")], "sphere", ""),
            ("saxs_apply_qdamp", "Apply Qdamp", "bool", None, False, ""),
            ("saxs_apply_qmax", "Apply Qmax termination", "bool", None, True, ""),
        ])
        self.form.edited.connect(self.edited)
        card.add(self.form)
        self.lay.addWidget(card)
        self.params = ParamSection("SAXS parameters", "0 for diameter/height: use the crystallite size model")
        self.params.edited.connect(self.edited)
        self.lay.addWidget(self.params)
        self.finish()

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        self.form.bind(phase.settings)
        self.params.set_rows(phase.parameters, list(S.SAXS), candidates=S.SAXS)


class ResultsPanel(Panel):
    """Classic output tabs of this phase (the PDF fit is on the dataset page)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QtWidgets.QTabWidget()
        self.classic_page = QtWidgets.QWidget()
        self.classic_layout = QtWidgets.QVBoxLayout(self.classic_page)
        self.classic_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self.classic_page, "Warren · size · local trends · shape factors")
        self.p_hist = MplPlot()
        self.tabs.addTab(self.p_hist, "Parameter history")
        self.tabs.setMinimumHeight(620)
        self.lay.addWidget(self.tabs, 1)
        self._host = None

    def load(self, win, project, dataset=None, phase=None):
        super().load(win, project, dataset, phase)
        host = win.phase_host(dataset, phase)
        tabs = host.output_tabs
        if host is not self._host:
            if self._host is not None:
                self._host.output_tabs.setParent(None)
            self._host = host
            # Progress tabs of the classic window are not fed here.
            for title in ("Rwp", "Params"):
                for i in range(tabs.count()):
                    if tabs.tabText(i) == title:
                        tabs.removeTab(i)
                        break
            self.classic_layout.addWidget(tabs, 1)
            tabs.setVisible(True)
        # CS grid results only after a grid search was run for this phase.
        for i in range(tabs.count()):
            if tabs.tabText(i) == "CS grid":
                tabs.setTabVisible(i, bool(getattr(host, "grid_used", False)))
        win.push_phase_result(dataset, phase)
        self._plot_history(win, dataset, phase)

    def _plot_history(self, win, dataset, phase):
        ax = self.p_hist.ax
        ax.clear()
        hist = win.history if win else []
        prefix = f"{dataset.id}.{phase.id}."
        names = sorted({k for _n, p in hist for k in p if k.startswith(prefix)})
        refined = [k for k in names if phase.parameters.get(k[len(prefix):], Parameter()).refine]
        if hist and refined:
            n = np.array([h[0] for h in hist])
            for k in refined[:12]:
                y = np.array([float(h[1].get(k, np.nan)) for h in hist])
                finite = y[np.isfinite(y)]
                y0 = finite[0] if finite.size else 1.0
                ax.plot(n, y / y0 if y0 else y, lw=1.0, label=k[len(prefix):])
            ax.set_xlabel("Function evaluation")
            ax.set_ylabel("Value / start value")
            ax.legend(fontsize=8, ncols=2)
        else:
            ax.text(0.5, 0.5, "Run a refinement to see parameter histories", ha="center", va="center", transform=ax.transAxes, color="#4b5563")
        self.p_hist.canvas.draw_idle()
