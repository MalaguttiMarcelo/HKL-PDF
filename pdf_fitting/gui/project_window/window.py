"""
Main window for multi-dataset, multi-phase projects.

Layout: toolbar and status chips on top with a small progress window at the
top right (Rwp and weight fractions vs. iteration, common to all phases);
below it the project tree on the left and the editor of the selected node.
"""

from __future__ import annotations

import copy
import math
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pdf_fitting.project import Dataset, Parameter, Phase, Project, safe_id

from . import panels as P
from . import sections as S
from .theme import apply_theme, tokens, watch_scheme
from .widgets import scroll_page

APP_NAME = "HKL-PDF"
PROJECT_FILTER = "HKL-PDF project (*.hklpdf.json *.json);;Legacy input (*.inp);;All files (*)"

SECTION_ITEMS = (
    ("structure", "Structure (3D)"),
    ("lat", "Lattice"),
    ("atoms", "Atoms & occupancy"),
    ("micro", "Microstructure"),
    ("dyn", "Local dynamics"),
    ("shape", "Shape & stacking"),
    ("saxs", "SAXS"),
    ("peaks", "Peak widths (free fit)"),
    ("results", "Results"),
)


class ProgressCanvas(QtWidgets.QFrame):
    """Rwp and weight fraction vs. function evaluation (all phases)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(300)
        self.setMaximumWidth(520)
        self.setFixedHeight(168)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Refinement progress · all phases")
        t.setObjectName("cardTitle")
        top.addWidget(t)
        top.addStretch(1)
        self.combo_ds = QtWidgets.QComboBox()
        self.combo_ds.setToolTip("Dataset whose phase fractions are shown")
        self.combo_ds.currentIndexChanged.connect(lambda _i: self.redraw())
        top.addWidget(self.combo_ds)
        lay.addLayout(top)
        self.fig = Figure(figsize=(5, 1.4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)
        self.ax_rwp = self.fig.add_subplot(1, 2, 1)
        self.ax_frac = self.fig.add_subplot(1, 2, 2)
        self.reset()

    def reset(self):
        self.rwp: List[tuple] = []
        self.frac: Dict[str, Dict[str, List[tuple]]] = {}
        self.names: Dict[str, Dict[str, str]] = {}
        self.redraw()

    def set_datasets(self, project: Project):
        current = self.combo_ds.currentData()
        self.combo_ds.blockSignals(True)
        self.combo_ds.clear()
        for ds in project.datasets:
            self.combo_ds.addItem(ds.name, ds.id)
        idx = self.combo_ds.findData(current)
        self.combo_ds.setCurrentIndex(max(idx, 0))
        self.combo_ds.blockSignals(False)
        self.names = {ds.id: {ph.id: ph.name for ph in ds.phases} for ds in project.datasets}
        self._order = {ds.id: [ph.id for ph in ds.phases] for ds in project.datasets}

    def add(self, nfev: int, rwp: float, fractions: Dict[str, Dict[str, Dict[str, float]]]):
        if math.isfinite(rwp):
            self.rwp.append((nfev, rwp))
        for ds_id, phases in (fractions or {}).items():
            for pid, f in phases.items():
                self.frac.setdefault(ds_id, {}).setdefault(pid, []).append((nfev, f.get("weight_fraction", np.nan)))

    def redraw(self):
        t = tokens()
        self.fig.set_facecolor(t["base"])
        for ax in (self.ax_rwp, self.ax_frac):
            ax.clear()
            ax.set_facecolor(t["base"])
            ax.tick_params(labelsize=7, colors=t["muted"])
            for spine in ax.spines.values():
                spine.set_color(t["border"])
        self.ax_rwp.set_title("Rwp (%)", fontsize=8, color=t["text"])
        self.ax_frac.set_title("Weight fraction (%)", fontsize=8, color=t["text"])
        if self.rwp:
            x, y = zip(*self.rwp)
            self.ax_rwp.plot(x, y, color="#1d4ed8", lw=1.2)
            self.ax_rwp.text(0.98, 0.9, f"{y[-1]:.2f}", transform=self.ax_rwp.transAxes, ha="right", fontsize=8, color=t["text"])
        ds_id = self.combo_ds.currentData()
        order = getattr(self, "_order", {}).get(ds_id, [])
        for pid, pts in (self.frac.get(ds_id) or {}).items():
            i = order.index(pid) if pid in order else 0
            x, y = zip(*pts)
            self.ax_frac.plot(x, y, color=P.phase_color(i), lw=1.2, label=self.names.get(ds_id, {}).get(pid, pid))
        if self.frac.get(ds_id):
            self.ax_frac.legend(fontsize=6, loc="best")
            self.ax_frac.set_ylim(-2, 102)
        self.canvas.draw_idle()


class ProjectWindow(QtWidgets.QMainWindow):
    def __init__(self, project: Optional[Project] = None):
        super().__init__()
        self.project = project or Project(name="Untitled project")
        self.dirty = False
        self.results: Dict[str, Dict[str, Any]] = {}
        self.history: List[tuple] = []
        self.current_node = "settings"
        self._thread: Optional[QtCore.QThread] = None
        self._worker = None
        self._studios: Dict[str, Any] = {}
        self._classic = None
        # Hidden classic windows lending their plots / builders (see legacy_bridge).
        self._ds_hosts: Dict[str, Any] = {}
        self._ph_hosts: Dict[str, Any] = {}
        self._gen = 0              # increments whenever the project changes
        self._host_gen: Dict[int, int] = {}
        self._result_version = 0
        self._host_result: Dict[int, int] = {}
        self._rwp_before_run = float("nan")
        self._theme = apply_theme(self)
        watch_scheme(self._on_scheme_changed)
        self._build_ui()
        self._load_project(self.project, mark_clean=True)
        # Start maximized (main() also calls showMaximized()).
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _on_scheme_changed(self):
        self._theme = apply_theme(self)
        self.progress_canvas.redraw()
        self.rebuild_tree()
        self.show_node(self.current_node)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.resize(1500, 950)
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header: toolbar + chips (left), progress window (right).
        header = QtWidgets.QWidget()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        title_row = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel("PDF")
        logo.setFixedSize(28, 28)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("background:#1d4ed8;color:white;border-radius:6px;font-weight:700;font-size:11px;")
        title_row.addWidget(logo)
        app_lbl = QtWidgets.QLabel(APP_NAME)
        app_lbl.setStyleSheet("font-weight:700;font-size:15px;")
        title_row.addWidget(app_lbl)
        self.lbl_project = QtWidgets.QLabel()
        self.lbl_project.setObjectName("muted")
        title_row.addWidget(self.lbl_project)
        title_row.addStretch(1)
        left.addLayout(title_row)

        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        st = self.style()
        self.act_open = self._action("Open…", self.open_project, QtGui.QKeySequence.Open, st.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.act_save = self._action("Save", self.save_project, QtGui.QKeySequence.Save, st.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.act_save_as = self._action("Save as…", self.save_project_as, QtGui.QKeySequence.SaveAs)
        self.act_add_ds = self._action("+ Dataset", self.add_dataset)
        self.act_add_ph = self._action("+ Phase (CIF)", lambda: self.add_phase(None))
        self.act_import_ph = self._action("+ Phase from .inp", lambda: self.import_phase_from_inp(None))
        self.act_calc = self._action("Calculate", self.calculate, "F5", st.standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.act_refine = self._action("Refine", self.refine, "F6", st.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.act_stop = self._action("Stop", self.stop, "Esc", st.standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.act_stop.setEnabled(False)
        for a in (self.act_open, self.act_save):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        for a in (self.act_add_ds, self.act_add_ph, self.act_import_ph):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        for a in (self.act_calc, self.act_refine, self.act_stop):
            self.toolbar.addAction(a)
        # Let the toolbar shrink (Qt adds an overflow » button) so the window
        # can always be maximized, whatever the screen size.
        self.toolbar.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.setMinimumWidth(160)
        left.addWidget(self.toolbar)

        chips = QtWidgets.QHBoxLayout()
        self.chip_state = QtWidgets.QLabel("Ready")
        self.chip_state.setObjectName("chip")
        self.chip_rwp = QtWidgets.QLabel("Rwp —")
        self.chip_rwp.setObjectName("chip")
        self.chip_refined = QtWidgets.QLabel()
        self.chip_refined.setObjectName("chip")
        self.chip_backend = QtWidgets.QLabel()
        self.chip_backend.setObjectName("chip")
        for c in (self.chip_state, self.chip_rwp, self.chip_refined, self.chip_backend):
            chips.addWidget(c)
        # Chips keep their natural size but may shrink (never below 40 px)
        # so the window can still be maximized on small screens.
        for w in (self.chip_state, self.chip_rwp, self.chip_refined, self.chip_backend, self.lbl_project):
            w.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
            w.setMinimumWidth(40)
        chips.addStretch(1)
        left.addLayout(chips)
        hl.addLayout(left, 1)
        self.progress_canvas = ProgressCanvas()
        hl.addWidget(self.progress_canvas)
        root.addWidget(header)

        # Body: tree | editor
        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.setHandleWidth(4)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(("Project", "Refined"))
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.setMinimumWidth(200)
        self.tree.currentItemChanged.connect(self._tree_selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        split.addWidget(self.tree)

        editor = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(editor)
        el.setContentsMargins(18, 12, 18, 12)
        el.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        self.lbl_crumbs = QtWidgets.QLabel()
        self.lbl_crumbs.setObjectName("crumbs")
        self.lbl_title = QtWidgets.QLabel()
        self.lbl_title.setObjectName("pageTitle")
        self.lbl_sub = QtWidgets.QLabel()
        self.lbl_sub.setObjectName("muted")
        for w in (self.lbl_crumbs, self.lbl_title, self.lbl_sub):
            # Long dataset / file names must not set the window's minimum width.
            w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            w.setMinimumWidth(0)
            titles.addWidget(w)
        head.addLayout(titles, 1)
        self.btn_structure = QtWidgets.QPushButton("View structure")
        self.btn_structure.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
        self.btn_structure.clicked.connect(self.view_structure)
        head.addWidget(self.btn_structure, 0, Qt.AlignBottom)
        el.addLayout(head)

        self.stack = QtWidgets.QStackedWidget()
        el.addWidget(self.stack, 1)
        split.addWidget(editor)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 1200])
        root.addWidget(split, 1)

        self.panels: Dict[str, P.Panel] = {
            "settings": P.SettingsPanel(),
            "constraints": P.ConstraintsPanel(),
            "refined": P.RefinedPanel(),
            "dataset": P.DatasetPanel(),
            "data": P.DataPanel(),
            "instr": P.InstrumentPanel(),
            "overview": P.PhaseOverviewPanel(),
            "lat": P.LatticePanel(),
            "atoms": P.AtomsPanel(),
            "micro": P.MicroPanel(),
            "dyn": P.DynamicsPanel(),
            "shape": P.ShapePanel(),
            "saxs": P.SaxsPanel(),
            "results": P.ResultsPanel(),
        }
        from .structure_page import StructurePanel

        self.panels["structure"] = StructurePanel()
        from .peaks_page import PeaksPanel

        self.panels["peaks"] = PeaksPanel()
        self._pages = {}
        for key, panel in self.panels.items():
            panel.edited.connect(self._on_edited)
            panel.navigate.connect(self.select_node)
            # Every page scrolls, so no page can push the window's minimum
            # size beyond the screen.
            page = scroll_page(panel) if key not in ("refined", "constraints") else panel
            self._pages[key] = page
            self.stack.addWidget(page)

        self._build_menus()
        self.statusBar().showMessage("Ready")

    def _action(self, text, slot, shortcut=None, icon=None):
        act = QtGui.QAction(text, self)
        if icon is not None:
            act.setIcon(icon)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
        act.triggered.connect(lambda _=False: slot())
        return act

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self._action("New project", self.new_project, QtGui.QKeySequence.New))
        m.addAction(self.act_open)
        m.addAction(self.act_save)
        m.addAction(self.act_save_as)
        m.addSeparator()
        m.addAction(self._action("Quit", self.close, QtGui.QKeySequence.Quit))
        m = mb.addMenu("&Project")
        for a in (self.act_add_ds, self.act_add_ph, self.act_import_ph):
            m.addAction(a)
        m.addSeparator()
        for a in (self.act_calc, self.act_refine, self.act_stop):
            m.addAction(a)
        m = mb.addMenu("&View")
        m.addAction(self._action("Classic window (single phase)", self.open_classic))

    # ------------------------------------------------------------------
    # Project / tree
    # ------------------------------------------------------------------

    def _load_project(self, project: Project, *, mark_clean: bool, auto_calc: bool = False):
        self.project = project
        self.results = {}
        self.history = []
        self.progress_canvas.reset()
        self.progress_canvas.set_datasets(project)
        self.rebuild_tree()
        first = f"ds:{project.datasets[0].id}" if project.datasets else "settings"
        self.select_node(first)
        self.dirty = not mark_clean
        self._update_title()
        self._update_chips()
        if hasattr(self, "chip_rwp") and hasattr(self, "panels"):
            self._restore_saved_results(auto_calc=auto_calc)

    def _restore_saved_results(self, auto_calc: bool = False):
        """Show the fit saved in the project file; otherwise calculate it."""
        from pdf_fitting.project.saved_results import decode_dataset_result

        peaks = getattr(self, "panels", {}).get("peaks")
        if peaks is not None:
            peaks._state = {}
        restored = 0
        for ds in self.project.datasets:
            enc = (self.project.last_results or {}).get(ds.id)
            if not enc:
                continue
            try:
                self.results[ds.id] = decode_dataset_result(enc)
                restored += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[PROJECT] saved fit of {ds.name} could not be read: {exc}")
        if restored:
            self._result_version += 1
            rw = [d.get("rwp") for d in self.results.values() if d.get("rwp") == d.get("rwp")]
            if rw:
                self.chip_rwp.setText(f"Rwp {float(np.sqrt(np.mean(np.square(rw)))):.2f} % (saved)")
            self._push_all_results()
            self.show_node(self.current_node)
            self.statusBar().showMessage("Showing the fit saved with the project. Calculate to update it.", 8000)
        elif auto_calc and self.project.datasets:
            QtCore.QTimer.singleShot(300, lambda: self._start("calculate", auto=True))

    def _update_title(self):
        name = self.project.name
        path = self.project.path or "not saved"
        self.lbl_project.setText(f"{name} · {os.path.basename(path)}")
        self.setWindowTitle(f"{'*' if self.dirty else ''}{name} — {APP_NAME}")

    def _update_chips(self):
        n = sum(1 for _f, p, _d, _ph in self.project.iter_parameters() if p.refine)
        self.chip_refined.setText(f"{n} parameter(s) flagged to refine")
        r = self.project.refinement
        self.chip_backend.setText(f"Shells {str(r.get('shell_backend', 'cpu')).upper()} · PDF {str(r.get('pdf_backend', 'cpu')).upper()}")

    def _count(self, params) -> int:
        return sum(1 for p in params.values() if p.refine)

    def _section_count(self, phase: Phase, sec: str) -> int:
        n = 0
        for name, p in phase.parameters.items():
            if not p.refine:
                continue
            s = S.section_of(name)
            s = {"overview": "", "other": ""}.get(s, s)
            if s == sec:
                n += 1
        return n

    def rebuild_tree(self):
        # Plain recursive walk (a QTreeWidgetItemIterator still alive when
        # clear() deletes the items crashes Qt).
        expanded = set()

        def walk(item):
            if item.isExpanded():
                expanded.add(item.data(0, Qt.UserRole))
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        first_build = self.tree.topLevelItemCount() == 0

        self.tree.blockSignals(True)
        self.tree.setCurrentItem(None)
        self.tree.clear()

        def add(parent, text, node, count=0, bold=False, color=None):
            item = QtWidgets.QTreeWidgetItem([text, str(count) if count else ""])
            item.setData(0, Qt.UserRole, node)
            if bold:
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
            if color:
                pm = QtGui.QPixmap(10, 10)
                pm.fill(Qt.transparent)
                painter = QtGui.QPainter(pm)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                painter.setBrush(QtGui.QColor(color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(0, 0, 10, 10)
                painter.end()
                item.setIcon(0, QtGui.QIcon(pm))
            item.setForeground(1, QtGui.QColor(tokens()["accent"]))
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            return item

        add(None, "Refinement settings", "settings")
        add(None, "Constraints", "constraints", len(self.project.constraints))
        add(None, "Refined parameters", "refined",
            sum(1 for _f, p, _d, _ph in self.project.iter_parameters() if p.refine))
        for ds in self.project.datasets:
            n_ds = self._count(ds.parameters) + sum(self._count(ph.parameters) for ph in ds.active_phases())
            ds_item = add(None, ds.name, f"ds:{ds.id}", n_ds, bold=True)
            add(ds_item, "Data & range", f"ds:{ds.id}:data")
            add(ds_item, "Instrument", f"ds:{ds.id}:instr", self._count(ds.parameters))
            for i, ph in enumerate(ds.phases):
                label = ph.name if ph.active else f"{ph.name} (inactive)"
                ph_item = add(ds_item, label, f"ph:{ds.id}:{ph.id}", self._count(ph.parameters), color=P.phase_color(i))
                if not ph.active:
                    ph_item.setForeground(0, QtGui.QColor("#9ca3af"))
                for sec, text in SECTION_ITEMS:
                    add(ph_item, text, f"ph:{ds.id}:{ph.id}:{sec}", self._section_count(ph, sec) if sec not in ("results", "structure", "peaks") else 0)
            if first_build or f"ds:{ds.id}" in expanded:
                ds_item.setExpanded(True)
        for item in self._tree_items():
            if item.data(0, Qt.UserRole) in expanded:
                item.setExpanded(True)
            if item.data(0, Qt.UserRole) == self.current_node:
                self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)

    def _tree_items(self):
        """All tree items as a list (no QTreeWidgetItemIterator, see rebuild_tree)."""
        out = []

        def walk(item):
            out.append(item)
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        return out

    def _refresh_counts(self):
        """Update refined counts in place (cheap; keeps the tree as it is)."""
        for item in self._tree_items():
            node = item.data(0, Qt.UserRole) or ""
            parts = node.split(":")
            n = 0
            try:
                if node == "refined":
                    n = sum(1 for _f, p, _d, _ph in self.project.iter_parameters() if p.refine)
                elif node == "constraints":
                    n = len(self.project.constraints)
                elif parts[0] == "ds":
                    ds = self.project.dataset(parts[1])
                    if len(parts) == 2:
                        n = self._count(ds.parameters) + sum(self._count(ph.parameters) for ph in ds.active_phases())
                    elif parts[2] == "instr":
                        n = self._count(ds.parameters)
                elif parts[0] == "ph":
                    ph = self.project.dataset(parts[1]).phase(parts[2])
                    n = self._count(ph.parameters) if len(parts) == 3 else (self._section_count(ph, parts[3]) if parts[3] not in ("results", "structure", "peaks") else 0)
            except KeyError:
                n = 0
            item.setText(1, str(n) if n else "")

    def _tree_selection_changed(self, current, _previous):
        if current is not None:
            self.show_node(current.data(0, Qt.UserRole))

    def select_node(self, node: str):
        for item in self._tree_items():
            if item.data(0, Qt.UserRole) == node:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                return
        self.show_node(node)

    def _resolve(self, node: str):
        parts = (node or "settings").split(":")
        ds = ph = None
        key = parts[0]
        if parts[0] == "ds":
            ds = self.project.dataset(parts[1])
            key = "dataset" if len(parts) == 2 else parts[2]
        elif parts[0] == "ph":
            ds = self.project.dataset(parts[1])
            ph = ds.phase(parts[2])
            key = "overview" if len(parts) == 3 else parts[3]
        return key, ds, ph

    def show_node(self, node: str, **kwargs):
        try:
            key, ds, ph = self._resolve(node)
        except KeyError:
            node, key, ds, ph = "settings", "settings", None, None
        self.current_node = node
        panel = self.panels[key]
        # Focus hints from links (e.g. a site label or pair / shell to show).
        panel.focus = dict(getattr(self, "_pending_focus", None) or {})
        self._pending_focus = None
        try:
            panel.load(self, self.project, ds, ph, **kwargs)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Panel error", traceback.format_exc(limit=3))
        self.stack.setCurrentWidget(self._pages[key])

        titles = {
            "settings": ("Refinement settings", "Strategy, tolerances and compute backend for the whole project."),
            "constraints": ("Constraints", "Tie parameters within a phase, across phases or across datasets."),
            "refined": ("Refined parameters", "Every parameter flagged for refinement, in all datasets and phases."),
            "dataset": ("", "Fit, phases and weight fractions of this dataset."),
            "data": ("Data & range", "Measured G(r) file and fitted r range."),
            "instr": ("Instrument", "Resolution parameters shared by every phase of this dataset."),
        }
        sec_names = dict(SECTION_ITEMS)
        crumbs = self.project.name
        if ph is not None:
            info = P.resolve_info(self.project, ph)
            sg = info.spacegroup if info and info.ok else ""
            title = sec_names.get(key, ph.name) if key != "overview" else ph.name
            sub = f"{ph.name} · {sg}" if key != "overview" else f"{os.path.basename(ph.cif)} · {sg}"
            crumbs = f"{self.project.name} › {ds.name}" + (f" › {ph.name}" if key != "overview" else "")
        else:
            title, sub = titles.get(key, ("", ""))
            if ds is not None:
                crumbs = f"{self.project.name} › {ds.name}" if key != "dataset" else self.project.name
                if key == "dataset":
                    title = ds.name
        self.lbl_crumbs.setText(crumbs)
        self.lbl_title.setText(title)
        self.lbl_sub.setText(sub)
        self.btn_structure.setVisible(ph is not None and key != "structure")

    def _on_edited(self):
        self._gen += 1
        self.dirty = True
        self._update_title()
        self._update_chips()
        self._refresh_counts()
        if self.current_node.startswith("ph:") and self.current_node.count(":") == 2:
            # Phase name / active state may have changed.
            self.rebuild_tree()

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        node = item.data(0, Qt.UserRole) if item else ""
        parts = node.split(":") if node else []
        if parts and parts[0] == "ds" and len(parts) == 2:
            ds = self.project.dataset(parts[1])
            menu.addAction("Add phase from CIF…", lambda: self.add_phase(ds))
            menu.addAction("Add phase from .inp…", lambda: self.import_phase_from_inp(ds))
            menu.addAction("Duplicate dataset (same phases)…", lambda: self.duplicate_dataset(ds))
            menu.addSeparator()
            menu.addAction("Remove dataset", lambda: self.remove_dataset(ds))
        elif parts and parts[0] == "ph" and len(parts) == 3:
            ds = self.project.dataset(parts[1])
            ph = ds.phase(parts[2])
            menu.addAction("View structure", self.view_structure)
            menu.addAction("Duplicate phase", lambda: self.duplicate_phase(ds, ph))
            if len(self.project.datasets) > 1:
                menu.addAction("Copy phase to the other datasets", lambda: self.copy_phase_to_datasets(ds, ph))
            menu.addSeparator()
            menu.addAction("Remove phase", lambda: self.remove_phase(ds, ph))
        else:
            menu.addAction("Add dataset…", self.add_dataset)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        ans = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Save the changes to the current project first?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
        )
        if ans == QtWidgets.QMessageBox.Cancel:
            return False
        if ans == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return True

    def new_project(self):
        if not self._confirm_discard():
            return
        self._load_project(Project(name="Untitled project"), mark_clean=True)

    def open_project(self, path: Optional[str] = None):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project or legacy input", self._last_dir(), PROJECT_FILTER)
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot open", f"{path}\n\n{exc}")
            return
        imported = path.lower().endswith(".inp")
        self._remember_dir(path)
        self._load_project(project, mark_clean=not imported, auto_calc=True)
        if imported:
            self.statusBar().showMessage("Imported a legacy .inp file. Save it as a .json project to keep the new layout.", 10000)

    def save_project(self) -> bool:
        if not self.project.path:
            return self.save_project_as()
        try:
            from pdf_fitting.project.saved_results import encode_dataset_result

            ids = {ds.id for ds in self.project.datasets}
            saved = {}
            for ds_id, d in self.results.items():
                if ds_id in ids:
                    try:
                        saved[ds_id] = encode_dataset_result(d)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[PROJECT] fit of {ds_id} not saved: {exc}")
            self.project.last_results = saved
            self.project.save()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.project.path}", 5000)
        return True

    def save_project_as(self) -> bool:
        base = safe_id(self.project.name, "project") + ".hklpdf.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", os.path.join(self._last_dir(), base), "HKL-PDF project (*.hklpdf.json)")
        if not path:
            return False
        if not path.lower().endswith(".json"):
            path += ".hklpdf.json"
        self.project.path = path
        self._remember_dir(path)
        ok = self.save_project()
        self._update_title()
        return ok

    def _last_dir(self) -> str:
        if self.project.path:
            return os.path.dirname(self.project.path)
        return str(QtCore.QSettings("HKL-PDF", "project_window").value("last_dir", os.getcwd()))

    def _remember_dir(self, path):
        QtCore.QSettings("HKL-PDF", "project_window").setValue("last_dir", os.path.dirname(os.path.abspath(path)))

    # ------------------------------------------------------------------
    # Datasets and phases
    # ------------------------------------------------------------------

    def _current_dataset(self) -> Optional[Dataset]:
        parts = self.current_node.split(":")
        if parts[0] in ("ds", "ph") and len(parts) > 1:
            try:
                return self.project.dataset(parts[1])
            except KeyError:
                return None
        return self.project.datasets[0] if self.project.datasets else None

    def add_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select G(r) data for the new dataset", self._last_dir(), "G(r) data (*.gr *.xy *.dat *.txt);;All files (*)")
        if not path:
            return
        self._remember_dir(path)
        try:
            r = np.loadtxt(path, unpack=True)[0]
            r_lo, r_hi = float(np.min(r)), float(np.max(r))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot read data", str(exc))
            return
        ds = Dataset(
            id=self.project.new_dataset_id(),
            name=os.path.splitext(os.path.basename(path))[0],
            data_file=path,
            r_min=max(1.0, r_lo),
            r_max=min(30.0, r_hi),
        )
        template = self.project.datasets[0] if self.project.datasets else None
        if template is not None:
            ds.parameters = copy.deepcopy(template.parameters)
            ds.r_min, ds.r_max, ds.r_extension = template.r_min, min(template.r_max, r_hi), template.r_extension
            if template.phases and QtWidgets.QMessageBox.question(
                self, "Copy phases", f"Copy the phases of “{template.name}” into the new dataset?",
            ) == QtWidgets.QMessageBox.Yes:
                ds.phases = copy.deepcopy(template.phases)
        else:
            ds.parameters = {"qdamp": Parameter(0.04), "delta_broad": Parameter(0.0), "qmax": Parameter(0.0)}
        self.project.datasets.append(ds)
        self._structure_changed(f"ds:{ds.id}")

    def duplicate_dataset(self, ds: Dataset):
        new = copy.deepcopy(ds)
        new.id = self.project.new_dataset_id()
        new.name = ds.name + " (copy)"
        self.project.datasets.append(new)
        self._structure_changed(f"ds:{new.id}:data")

    def remove_dataset(self, ds: Dataset):
        if QtWidgets.QMessageBox.question(self, "Remove dataset", f"Remove “{ds.name}” and its phases from the project?") != QtWidgets.QMessageBox.Yes:
            return
        self.project.datasets.remove(ds)
        self.results.pop(ds.id, None)
        self._structure_changed("settings")

    def _new_phase_from_cif(self, ds: Dataset, path: str) -> Optional[Phase]:
        info = S.structure_info(path)
        if info is None or not info.ok:
            QtWidgets.QMessageBox.critical(self, "Cannot read CIF", (info.error if info else "File not found"))
            return None
        name = os.path.splitext(os.path.basename(path))[0]
        ph = Phase(id=self.project.new_phase_id(ds, name), name=name, cif=path)
        ph.parameters["scale"] = Parameter(1.0, refine=True)
        for k in S.LATTICE:
            ph.parameters[k] = Parameter(float(info.lattice[k]), refine=(k == "a"))
        for el in info.elements:
            ph.parameters[f"biso_{el}"] = Parameter(0.5, min=0.0, max=10.0)
        ph.settings["microstrain_model"] = "isotropic"
        return ph

    def add_phase(self, ds: Optional[Dataset]):
        ds = ds or self._current_dataset()
        if ds is None:
            QtWidgets.QMessageBox.information(self, "No dataset", "Add a dataset (G(r) data) first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Add phase to {ds.name}", self._last_dir(), "CIF files (*.cif);;All files (*)")
        if not path:
            return
        self._remember_dir(path)
        ph = self._new_phase_from_cif(ds, path)
        if ph is None:
            return
        ds.phases.append(ph)
        self._structure_changed(f"ph:{ds.id}:{ph.id}")

    def import_phase_from_inp(self, ds: Optional[Dataset]):
        ds = ds or self._current_dataset()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add phase from a legacy .inp", self._last_dir(), "Legacy input (*.inp)")
        if not path:
            return
        self._remember_dir(path)
        try:
            src = Project.from_inp(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Cannot read input", str(exc))
            return
        if ds is None:
            # Empty project: take the whole legacy project.
            self._load_project(src, mark_clean=False)
            return
        ph = copy.deepcopy(src.datasets[0].phases[0])
        ph.id = self.project.new_phase_id(ds, ph.name)
        ds.phases.append(ph)
        self._structure_changed(f"ph:{ds.id}:{ph.id}")

    def duplicate_phase(self, ds: Dataset, ph: Phase):
        new = copy.deepcopy(ph)
        new.name = ph.name + " (copy)"
        new.id = self.project.new_phase_id(ds, new.name)
        ds.phases.append(new)
        self._structure_changed(f"ph:{ds.id}:{new.id}")

    def copy_phase_to_datasets(self, ds: Dataset, ph: Phase):
        n = 0
        for other in self.project.datasets:
            if other is ds:
                continue
            new = copy.deepcopy(ph)
            new.id = self.project.new_phase_id(other, ph.name)
            other.phases.append(new)
            n += 1
        self._structure_changed(self.current_node)
        self.statusBar().showMessage(f"Copied {ph.name} to {n} dataset(s).", 5000)

    def remove_phase(self, ds: Dataset, ph: Phase):
        if QtWidgets.QMessageBox.question(self, "Remove phase", f"Remove “{ph.name}” from {ds.name}?") != QtWidgets.QMessageBox.Yes:
            return
        ds.phases.remove(ph)
        self._structure_changed(f"ds:{ds.id}")

    def _structure_changed(self, node: str):
        self._gen += 1
        self.dirty = True
        self.progress_canvas.set_datasets(self.project)
        self.current_node = node
        self.rebuild_tree()
        self.select_node(node)
        self._update_title()
        self._update_chips()

    # ------------------------------------------------------------------
    # Calculation / refinement
    # ------------------------------------------------------------------

    def last_result_for(self, ds_id: str):
        return self.results.get(ds_id)

    def _validate_for_run(self, quiet: bool = False) -> bool:
        if not self.project.datasets:
            if not quiet:
                QtWidgets.QMessageBox.information(self, "Nothing to calculate", "Add a dataset and at least one phase first.")
            return False
        problems = []
        for ds in self.project.datasets:
            if not os.path.exists(self.project.resolve_path(ds.data_file)):
                problems.append(f"{ds.name}: data file not found")
            if not ds.active_phases():
                problems.append(f"{ds.name}: no active phase")
            for ph in ds.active_phases():
                if not os.path.exists(self.project.resolve_path(ph.cif)):
                    problems.append(f"{ds.name} › {ph.name}: CIF not found")
        if problems:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Cannot run", "\n".join(problems))
            return False
        return True

    def calculate(self):
        self._start("calculate")

    def refine(self):
        self._start("refine")

    def _start(self, mode: str, *, auto: bool = False):
        if self._thread is not None or getattr(self, "_external_stop", None) is not None or not self._validate_for_run(quiet=auto):
            return
        from .worker import ProjectWorker

        self._rwp_before_run = float("nan")
        prev = [d.get("rwp") for d in self.results.values() if d.get("rwp") is not None]
        if prev and len(prev) == len(self.project.datasets):
            self._rwp_before_run = float(np.sqrt(np.mean(np.square(prev)))) if len(prev) > 1 else float(prev[0])
        if mode == "refine":
            self.history = []
            self.progress_canvas.reset()
            self.progress_canvas.set_datasets(self.project)
        self._thread = QtCore.QThread(self)
        self._worker = ProjectWorker(self.project, mode)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_done)
        self._set_running(True, {"calculate": "Calculating…", "esd": "Computing esds…"}.get(mode, "Refining…"))
        if not auto:
            self._pop_out_fit_plots()
        self._thread.start()

    def _pop_out_fit_plots(self):
        """Show the fit of the current (or first) dataset in its own window."""
        ds = self._current_dataset()
        if ds is None:
            return
        try:
            host = self.dataset_host(ds)
            if self.current_node == f"ds:{ds.id}":
                panel = self.panels["dataset"]
                host.dock_into(panel.plot_layout, panel.placeholder)
            host.pop_out()
            host.win._pdf_dialog.setWindowTitle(f"PDF fit — {ds.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[PROJECT] could not pop out the fit plot: {exc}")

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self.chip_state.setText("Stopping…")
        elif getattr(self, "_external_stop", None) is not None:
            self._external_stop()
            self.chip_state.setText("Stopping…")

    def set_external_job(self, running: bool, text: str = "Working…", stop=None):
        """A page-level background job (e.g. the free-peak fit): busy chip,
        busy cursor, run actions disabled, Stop routed to ``stop``."""
        self._external_stop = stop if running else None
        self._set_running(running, text if running else "Ready")
        if running:
            QtWidgets.QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _set_running(self, running: bool, text: str = "Ready"):
        for a in (self.act_calc, self.act_refine, self.act_open, self.act_add_ds, self.act_add_ph, self.act_import_ph):
            a.setEnabled(not running)
        self.act_stop.setEnabled(running)
        self.chip_state.setObjectName("chipOk" if running else "chip")
        self.chip_state.setText(text)
        self.chip_state.style().unpolish(self.chip_state)
        self.chip_state.style().polish(self.chip_state)

    def _on_status(self, text: str):
        self.statusBar().showMessage(text)

    def _on_progress(self, payload: Dict[str, Any]):
        nfev = payload.get("nfev", 0)
        self.history.append((nfev, payload.get("params", {})))
        self.progress_canvas.add(nfev, payload.get("best_rwp", payload.get("rwp", np.nan)), payload.get("fractions", {}))
        # Redraw at most every 0.3 s.
        if not getattr(self, "_redraw_pending", False):
            self._redraw_pending = True

            def _redraw():
                self._redraw_pending = False
                self.progress_canvas.redraw()

            QtCore.QTimer.singleShot(300, _redraw)
        self.chip_state.setText(f"Refining · stage “{payload.get('stage', '')}” · evaluation {nfev}")
        for ds_id, (r, g_exp, g_fit) in (payload.get("curves") or {}).items():
            host = self._ds_hosts.get(ds_id)
            if host is not None:
                try:
                    name = self.project.dataset(ds_id).name
                except KeyError:
                    name = ds_id
                host.update_live(r, g_exp, g_fit, f"{name} — refining, evaluation {nfev}")
        rwp = payload.get("best_rwp", np.nan)
        if rwp == rwp:
            self.chip_rwp.setText(f"Rwp {rwp:.2f} %")

    def _on_finished(self, summary: Dict[str, Any]):
        rwp = summary.get("rwp", np.nan)
        self.chip_rwp.setText(f"Rwp {rwp:.2f} %" if rwp == rwp else "Rwp —")
        if summary.get("mode") == "refine":
            # Show the final fit first, so the user can judge it.
            self._show_final_fit(summary)
            stopped = bool(summary.get("stopped"))
            before = self._rwp_before_run
            change = f"  (before: {before:.2f} %)" if before == before else ""
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Refinement stopped" if stopped else "Refinement finished")
            box.setText(
                (f"The refinement was stopped. Best Rwp reached: {rwp:.2f} %." if stopped
                 else f"The refinement finished with Rwp {rwp:.2f} %.") + change
            )
            box.setInformativeText(
                "Look at the fit. Keep the refined parameters, or discard them and "
                "return to the values before the refinement?"
            )
            keep = box.addButton("Keep results", QtWidgets.QMessageBox.AcceptRole)
            box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
            box.setDefaultButton(keep)
            box.exec()
            if box.clickedButton() is not keep:
                self.statusBar().showMessage("Refinement discarded; parameters unchanged.", 10000)
                # Put the plots back to the last kept result.
                self._result_version += 1
                self._push_all_results()
                self._refresh_counts()
                self.show_node(self.current_node)
                return
        self._result_version += 1
        for d in summary.get("datasets", []):
            self.results[d["id"]] = d
        unc = summary.get("uncertainty")
        esd = self._store_uncertainty(unc, summary)
        if summary.get("mode") == "refine":
            self.project.apply_values(summary.get("params", {}), refinable=summary.get("refinable"), esd=esd)
            self._gen += 1
            fractions = {d["id"]: d.get("fractions", {}) for d in summary.get("datasets", [])}
            self.progress_canvas.add(len(self.history) and self.history[-1][0] + 1, rwp, fractions)
            self.progress_canvas.redraw()
            self.dirty = True
            self._update_title()
            msg = "Refinement stopped; best parameters kept." if summary.get("stopped") else "Refinement finished; parameters kept."
        elif summary.get("mode") == "esd":
            msg = "esds computed." if esd else ("esd calculation: " + str((unc or {}).get("message", "no result")))
        else:
            msg = "Calculation finished."
        if unc and unc.get("names"):
            msg += f" esds: χ²red {unc.get('chi2_red') or float('nan'):.3g}, {unc.get('message', '')}."
        self.statusBar().showMessage(f"{msg} Rwp {rwp:.2f} % · {summary.get('elapsed_s', 0):.1f} s", 12000)
        self._refresh_counts()
        self._update_chips()
        self._push_all_results()
        self.show_node(self.current_node)
        self._apply_pending_marks()

    def compute_esds(self):
        """esds and correlations at the current parameter values (no refinement)."""
        if not any(p.refine for _f, p, _d, _ph in self.project.iter_parameters()):
            QtWidgets.QMessageBox.information(self, "esds", "Flag the parameters to analyse as refined first.")
            return
        self._start("esd")

    def _store_uncertainty(self, unc, summary):
        """Keep the esd analysis (window + project file); return {name: esd} or None."""
        mode = summary.get("mode")
        if not unc or not unc.get("names"):
            if mode == "refine" and not summary.get("stopped"):
                # A refinement without esds: the old esds no longer apply.
                for name in (summary.get("refinable") or {}):
                    p = self.project.find_parameter(name)
                    if p is not None and (summary.get("refinable") or {}).get(name):
                        p.esd = None
            return None
        self.project.last_uncertainty = dict(unc)
        esd = {n: e for n, e in zip(unc["names"], unc["esd"]) if e is not None}
        for n, (_v, e) in (unc.get("dependent") or {}).items():
            if e is not None:
                esd[n] = e
        if mode == "esd":
            for name, e in esd.items():
                p = self.project.find_parameter(name)
                if p is not None:
                    p.esd = e
            self.dirty = True
            self._update_title()
        return esd

    def _show_final_fit(self, summary: Dict[str, Any]):
        """Draw the final fit in the (popped-out) plots before asking keep / discard."""
        for d in summary.get("datasets", []):
            host = self._ds_hosts.get(d["id"])
            if host is None:
                continue
            try:
                name = self.project.dataset(d["id"]).name
            except KeyError:
                name = d["id"]
            host.update_live(d["r"], d["G_exp"], d["G_fit"], f"{name} — final fit, Rwp {d['rwp']:.2f} %")
        QtWidgets.QApplication.processEvents()

    def _on_failed(self, text: str):
        print(text)
        QtWidgets.QMessageBox.critical(self, "Run failed", text[-3000:])

    def _thread_done(self):
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._set_running(False)

    # ------------------------------------------------------------------
    # Classic hosts (plots with tick markers, output tabs, shape builder)
    # ------------------------------------------------------------------

    def _phase_local_values(self, ds: Dataset, ph: Phase) -> Dict[str, Any]:
        out = {k: p.value for k, p in ds.parameters.items()}
        out.update({k: p.value for k, p in ph.parameters.items()})
        return out

    def _make_host(self, factory):
        self.statusBar().showMessage("Preparing plot tools (first time only)…")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            return factory()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

    def _ensure_host_config(self, host, ds: Dataset, ph: Optional[Phase]):
        if ph is None or self._host_gen.get(id(host)) == self._gen:
            return
        host.load_config(self.project.phase_legacy_config(ds, ph), params=self._phase_local_values(ds, ph))
        self._host_gen[id(host)] = self._gen

    def _tick_entries(self, ds: Dataset):
        entries = []
        for i, ph in enumerate(ds.phases):
            if not ph.active:
                continue
            entries.append({
                "cfg": self.project.phase_legacy_config(ds, ph),
                "params": self._phase_local_values(ds, ph),
                "name": ph.name,
                "color": P.phase_color(i),
            })
        return entries

    def dataset_host(self, ds: Dataset):
        from .legacy_bridge import DatasetPlotHost

        host = self._ds_hosts.get(ds.id)
        if host is None:
            host = self._make_host(lambda: DatasetPlotHost(ds.name, lambda d=ds: self._tick_entries(d), self))
            self._ds_hosts[ds.id] = host
        first = next(iter(ds.active_phases()), None)
        self._ensure_host_config(host, ds, first)
        return host

    def phase_host(self, ds: Dataset, ph: Phase):
        from .legacy_bridge import LegacyHost

        key = f"{ds.id}:{ph.id}"
        host = self._ph_hosts.get(key)
        if host is None:
            host = self._make_host(lambda: LegacyHost(f"{ds.name} › {ph.name}", self))
            host.configChanged.connect(lambda d=ds, p=ph, h=host: self._host_config_changed(d, p, h))
            self._ph_hosts[key] = host
        self._ensure_host_config(host, ds, ph)
        return host

    def merge_from_host(self, ds: Dataset, ph: Phase, host) -> bool:
        """
        Copy the classic Builder's shape / stacking state into the phase:
        crystallite-shape and stacking sections, phase settings and the
        stacking / size parameters. Returns True if anything changed.
        """
        import json
        from pdf_fitting.project.model import PHASE_SECTIONS

        try:
            cfg = host.compose_config()
        except Exception as exc:  # noqa: BLE001
            print(f"[PROJECT] Could not read the classic builder state: {exc}")
            return False
        tmp = Project.from_legacy_config(cfg).datasets[0].phases[0]
        before = json.dumps(ph.to_json(), sort_keys=True, default=str)

        for section in PHASE_SECTIONS:
            if section == "contrast_factors":
                continue
            value = tmp.sections.get(section, {})
            if value or section in ph.sections:
                ph.sections[section] = copy.deepcopy(value or {})
        # Phase settings (microstrain model, SAXS, element refinement) are
        # owned by the project window: the classic Builder may re-derive them
        # (e.g. switch to PAH when PAH parameters exist), so never copy them.
        shape_names = set(S.SIZE) | set(S.CYLINDER)
        for name, p in tmp.parameters.items():
            if S.is_stacking(name) or name in shape_names:
                ph.parameters[name] = copy.deepcopy(p)
        for name in [n for n in ph.parameters if S.is_stacking(n) and n not in tmp.parameters]:
            ph.parameters.pop(name)

        changed = json.dumps(ph.to_json(), sort_keys=True, default=str) != before
        if changed:
            self._gen += 1
            self._host_gen[id(host)] = self._gen  # the host already holds this state
            self.dirty = True
            self._update_title()
            self._refresh_counts()
        return changed

    def _host_config_changed(self, ds: Dataset, ph: Phase, host):
        # Only after an action started from the project window (builder
        # dialog, CS grid search / applying a candidate). Loading a phase into
        # the hidden builder must not rewrite the project.
        if not getattr(host, "user_session", False):
            return
        if self.merge_from_host(ds, ph, host) and self.current_node == f"ph:{ds.id}:{ph.id}:shape":
            self.show_node(self.current_node)

    def push_dataset_result(self, ds: Dataset):
        res = self.results.get(ds.id)
        host = self._ds_hosts.get(ds.id)
        if res is None or host is None or self._host_result.get(id(host)) == self._result_version:
            return
        contrib = {}
        for ph in ds.phases:
            pr = res["phases"].get(ph.id)
            if pr:
                for pair, g in (pr.get("G_contrib") or {}).items():
                    contrib[f"{ph.name}: {pair}"] = g
        host.show_result({
            "r_full": res["r"],
            "G_exp_full": res["G_exp"],
            "G_fit_full": res["G_fit"],
            "res_full": res["G_exp"] - res["G_fit"],
            "G_contrib": contrib,
            "gamma_avg": None,
        })
        host.win.plot_pdf.ax.set_title(f"{ds.name} — Rwp {res['rwp']:.2f} %", fontsize=10)
        host.win.plot_pdf.canvas.draw_idle()
        self._host_result[id(host)] = self._result_version
        if getattr(host, "mark_state", None):
            host.distance_marks = []  # the redraw removed them
            self._draw_marks(ds, host)

    def push_phase_result(self, ds: Dataset, ph: Phase):
        res = self.results.get(ds.id)
        host = self._ph_hosts.get(f"{ds.id}:{ph.id}")
        if res is None or host is None or self._host_result.get(id(host)) == self._result_version:
            return
        pr = res["phases"].get(ph.id)
        if pr is None:
            return
        payload = {
            "r_full": res["r"],
            "G_exp_full": res["G_exp"],
            "G_fit_full": res["G_fit"],
            "res_full": res["G_exp"] - res["G_fit"],
            "G_contrib": pr.get("G_contrib") or {},
            "gamma_avg": pr.get("gamma_avg"),
        }
        for key in ("warren_by_dir", "size_distribution", "local_trends"):
            if pr.get(key):
                payload[key] = pr[key]
        host.show_result(payload)
        host.win.plot_pdf.ax.set_title(f"{ds.name} — {ph.name} (total fit, Rwp {res['rwp']:.2f} %)", fontsize=10)
        host.win.plot_pdf.canvas.draw_idle()
        self._host_result[id(host)] = self._result_version

    # ------------------------------------------------------------------
    # Structure viewer / classic window
    # ------------------------------------------------------------------

    def navigate_to(self, node: str, **focus):
        """Open a node with focus hints (site label, pair, shell k, ...)."""
        self._pending_focus = focus
        if node == self.current_node:
            self.show_node(node)
        else:
            self.select_node(node)

    def _push_all_results(self):
        """Send the stored results to every open fit / phase plot."""
        for ds in self.project.datasets:
            if ds.id in self._ds_hosts:
                self.push_dataset_result(ds)
            for ph in ds.phases:
                if f"{ds.id}:{ph.id}" in self._ph_hosts:
                    self.push_phase_result(ds, ph)

    def _apply_pending_marks(self):
        pending = getattr(self, "_pending_marks", None)
        self._pending_marks = None
        if pending and pending[0].id in self.results:
            ds, marks, pairs = pending
            self.mark_distances_on_fit(ds, marks, pairs=pairs)

    def mark_distances_on_fit(self, ds: Dataset, marks, clear: bool = False, pairs=None):
        """
        Show distances on the dataset's PDF fit: an exact marker for each
        distance plus the classic tick markers of the given pairs
        ("<phase name>: <pair>" keys), over the fitted r range.
        Calculates the fit first if there is none yet.
        """
        if not clear and ds.id not in self.results:
            self._pending_marks = (ds, list(marks), list(pairs or []))
            if self._thread is None:
                self.statusBar().showMessage("No fit calculated yet — calculating first, then marking…", 8000)
                self.calculate()
            return
        host = self.dataset_host(ds)
        host.mark_state = None if clear else (list(marks), list(pairs or []))
        self.push_dataset_result(ds)
        self._draw_marks(ds, host, clear=clear, pop=not clear)

    @staticmethod
    def _pair_key(key: str):
        """('phase', frozenset(elements)) of a '<phase>: a-b' tick key (order/case free)."""
        phase, _, pair = str(key).rpartition(": ")
        return phase, frozenset(x.strip().lower() for x in pair.split("-"))

    def _draw_marks(self, ds: Dataset, host, *, clear: bool = False, pop: bool = False):
        """Distance lines + tick markers of the marked pairs on the dataset plot."""
        state = getattr(host, "mark_state", None)
        marks, pairs = state if (state and not clear) else ([], [])
        w = host.win
        if not clear and pairs:
            try:
                have = w._tick_data_max_r() if hasattr(w, "_tick_data_max_r") else 0.0
                tick_keys = set((getattr(w, "_pdf_tick_data", {}) or {}).get("pairs", {}) or {})
                wanted = {self._pair_key(p) for p in pairs}
                matched = {k for k in tick_keys if self._pair_key(k) in wanted}
                if have + 1e-9 < float(ds.r_max) or not matched:
                    host._calculate_merged_ticks(float(ds.r_max) * 1.02)
                    tick_keys = set((getattr(w, "_pdf_tick_data", {}) or {}).get("pairs", {}) or {})
                    matched = {k for k in tick_keys if self._pair_key(k) in wanted}
                if not matched:  # phase label differs: match the elements only
                    els = {e for _ph, e in wanted}
                    matched = {k for k in tick_keys if self._pair_key(k)[1] in els}
                if not matched:
                    print(f"[PROJECT] no tick data for {sorted(pairs)}; available: {sorted(tick_keys)[:12]}")
                w._tick_filter_pairs = matched or set(pairs)
                w._tick_range_min = float(ds.r_min)
                w._tick_range_max = float(ds.r_max)
                w.btn_tick_markers.blockSignals(True)
                w.btn_tick_markers.setChecked(True)
                w.btn_tick_markers.blockSignals(False)
                w._refresh_pdf_tick_markers()
            except Exception as exc:  # noqa: BLE001
                print(f"[PROJECT] tick markers for marks failed: {exc}")
        elif clear:
            try:
                w._tick_filter_pairs = None
                w.btn_tick_markers.blockSignals(True)
                w.btn_tick_markers.setChecked(False)
                w.btn_tick_markers.blockSignals(False)
                w._clear_pdf_tick_markers()
            except Exception:  # noqa: BLE001
                pass
        ax = host.win.plot_pdf.ax
        for artist in getattr(host, "distance_marks", []):
            try:
                artist.remove()
            except Exception:  # noqa: BLE001
                pass
        host.distance_marks = []
        if marks:
            for d, label in marks:
                line = ax.axvline(float(d), color="#7c3aed", lw=2.0, ls="-", alpha=0.9, zorder=5)
                text = ax.annotate(
                    f"{label}\n{float(d):.3f} Å", xy=(float(d), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -4), textcoords="offset points", fontsize=8, color="#7c3aed", va="top", rotation=90,
                )
                host.distance_marks += [line, text]
            if pop:
                host.pop_out()
                # The pop-out window may redraw the axes: draw the ticks again.
                QtCore.QTimer.singleShot(200, lambda h=host: self._refresh_ticks(h))
        host.win.plot_pdf.canvas.draw_idle()

    @staticmethod
    def _refresh_ticks(host):
        try:
            if host.win.btn_tick_markers.isChecked():
                host.win._refresh_pdf_tick_markers()
                host.win.plot_pdf.canvas.draw_idle()
        except Exception:  # noqa: BLE001
            pass

    def view_structure(self):
        parts = self.current_node.split(":")
        if parts[0] == "ph":
            self.select_node(f"ph:{parts[1]}:{parts[2]}:structure")
        return

    def _view_structure_window_unused(self):
        parts = self.current_node.split(":")
        if parts[0] != "ph":
            return
        ds = self.project.dataset(parts[1])
        ph = ds.phase(parts[2])
        path = self.project.resolve_path(ph.cif)
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "CIF not found", path)
            return
        try:
            self.statusBar().showMessage("Opening structure viewer…")
            QtWidgets.QApplication.processEvents()
            from pdf_fitting.gui.structure_studio import StructureStudioWindow

            key = f"{ds.id}:{ph.id}"
            studio = self._studios.get(key)
            if studio is None:
                studio = StructureStudioWindow(self)
                self._studios[key] = studio
            lattice = {}
            for k in S.LATTICE:
                if k in ph.parameters:
                    try:
                        lattice[k] = float(ph.parameters[k].value)
                    except (TypeError, ValueError):
                        pass
            studio.load_structure(path, lattice_params=lattice, input_path=self.project.path or "")
            studio.setWindowTitle(f"Structure — {ph.name} ({ds.name})")
            studio.show_and_raise()
            self.statusBar().showMessage("Structure viewer opened.", 4000)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Structure viewer failed", str(exc))

    def open_classic(self):
        try:
            from pdf_fitting.gui.mainwindow import MainWindow

            if self._classic is None:
                self._classic = MainWindow()
            self._classic.show()
            self._classic.raise_()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Classic window failed", str(exc))

    def closeEvent(self, event):
        if self._thread is not None:
            if QtWidgets.QMessageBox.question(self, "Refinement running", "Stop the running calculation and quit?") != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.stop()
            self._thread.quit()
            self._thread.wait(5000)
        if not self._confirm_discard():
            event.ignore()
            return
        for studio in self._studios.values():
            try:
                studio.shutdown()
            except Exception:  # noqa: BLE001
                pass
        for host in list(self._ds_hosts.values()) + list(self._ph_hosts.values()):
            host.shutdown()
        try:
            self.panels["structure"].shutdown()
        except Exception:  # noqa: BLE001
            pass
        event.accept()
