"""
Structure page of a phase: the GPU Structure Studio embedded in the project
window, showing the phase as currently parameterised, with a "Refinement
links" tab:

- picked atoms (last three), distance and angle;
- the selected site: label, Wyckoff multiplicity, current x/y/z/occ/Biso and
  which coordinates symmetry leaves free / tied / fixed;
- neighbor table (pair, distance, count, λ shell k) — mark distances on the
  PDF fit;
- jump to the site in Atoms & occupancy, or to the pair / shell in Local
  dynamics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.project import structure_tools as T

from .structure_edit import EditController
from .widgets import Card, fmt_value


_GL_OK = None


def opengl_available() -> bool:
    """
    True if Qt can create an OpenGL context here. VTK aborts the whole
    program (native crash) when it cannot, e.g. offscreen or in some remote
    sessions, so the 3D viewer is only created after this check.
    """
    global _GL_OK
    if _GL_OK is not None:
        return _GL_OK
    platform = QtGui.QGuiApplication.platformName().lower()
    if platform in ("offscreen", "minimal"):
        _GL_OK = False
        return _GL_OK
    try:
        ctx = QtGui.QOpenGLContext()
        surface = QtGui.QOffscreenSurface()
        surface.create()
        _GL_OK = bool(ctx.create() and surface.isValid() and ctx.makeCurrent(surface))
        if _GL_OK:
            ctx.doneCurrent()
    except Exception:  # noqa: BLE001
        _GL_OK = False
    return _GL_OK


class LinksTab(QtWidgets.QWidget):
    """Selection, measurements, neighbors and links to the parameters."""

    def __init__(self, page: "StructurePanel", editor: QtWidgets.QWidget):
        super().__init__()
        self.page = page
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        hint = QtWidgets.QLabel("Click atoms in the 3D view. When a pair (or triplet) is complete, the next click starts a new one.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Measure:"))
        self.rb_distance = QtWidgets.QRadioButton("Distance (2 atoms)")
        self.rb_angle = QtWidgets.QRadioButton("Angle (3 atoms)")
        self.rb_distance.setChecked(True)
        mode_row.addWidget(self.rb_distance)
        mode_row.addWidget(self.rb_angle)
        mode_row.addStretch(1)
        lay.addLayout(mode_row)
        self.rb_distance.toggled.connect(lambda _v: self.page.clear_picks())

        self.lbl_picks = QtWidgets.QLabel("No atom picked.")
        self.lbl_picks.setWordWrap(True)
        self.lbl_picks.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_picks)
        self.lbl_measure = QtWidgets.QLabel("")
        self.lbl_measure.setObjectName("banner")
        self.lbl_measure.setVisible(False)
        lay.addWidget(self.lbl_measure)
        btn_clear = QtWidgets.QPushButton("Clear picks")
        btn_clear.clicked.connect(self.page.clear_picks)
        lay.addWidget(btn_clear, 0, Qt.AlignLeft)

        box = QtWidgets.QGroupBox("Selected site")
        bl = QtWidgets.QVBoxLayout(box)
        self.lbl_site = QtWidgets.QLabel("—")
        self.lbl_site.setWordWrap(True)
        self.lbl_site.setTextFormat(Qt.RichText)
        bl.addWidget(self.lbl_site)
        bl.addWidget(editor)
        row = QtWidgets.QHBoxLayout()
        self.btn_atoms = QtWidgets.QPushButton("Edit in Atoms && occupancy")
        self.btn_atoms.clicked.connect(self.page.go_to_atoms)
        self.btn_pair = QtWidgets.QPushButton("Edit pair in Local dynamics")
        self.btn_pair.clicked.connect(self.page.go_to_pair)
        row.addWidget(self.btn_atoms)
        row.addWidget(self.btn_pair)
        bl.addLayout(row)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Neighbors of the selected atom")
        nl = QtWidgets.QVBoxLayout(box)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("up to"))
        self.sp_rmax = QtWidgets.QDoubleSpinBox()
        self.sp_rmax.setRange(1.0, 15.0)
        self.sp_rmax.setValue(4.0)
        self.sp_rmax.setSuffix(" Å")
        self.sp_rmax.valueChanged.connect(lambda _v: self.page.refresh_neighbors())
        top.addWidget(self.sp_rmax)
        top.addStretch(1)
        nl.addLayout(top)
        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(("Neighbor", "Pair", "Distance (Å)", "Count", "Shell k"))
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.setMinimumHeight(220)
        self.tbl.itemDoubleClicked.connect(lambda _it: self.page.go_to_neighbor_row())
        nl.addWidget(self.tbl, 1)
        row = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Mark selected on PDF fit")
        b1.setToolTip("Show the selected distances and the tick markers of their pairs on the PDF fit "
                      "(calculates the fit first if needed).")
        b1.clicked.connect(self.page.mark_selected_on_fit)
        b2 = QtWidgets.QPushButton("Clear marks")
        b2.clicked.connect(self.page.clear_fit_marks)
        b3 = QtWidgets.QPushButton("Pair dynamics of row")
        b3.setToolTip("Open the δ / λ entry of the selected row's pair and shell (double-click does the same).")
        b3.clicked.connect(self.page.go_to_neighbor_row)
        for b in (b1, b2, b3):
            row.addWidget(b)
        nl.addLayout(row)
        lay.addWidget(box, 1)


class StructurePanel(QtWidgets.QWidget):
    """Hosts one Structure Studio per phase (created on first use)."""

    edited = QtCore.Signal()
    navigate = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.win = None
        self.project = None
        self.dataset = None
        self.phase = None
        self.focus = {}
        self._studios: Dict[str, Any] = {}
        self._links: Dict[str, LinksTab] = {}
        self._popped: Dict[str, bool] = {}
        self._loaded_gen: Dict[str, int] = {}
        self._ps: Dict[str, T.PhaseStructure] = {}
        self._picks: Dict[str, List[Dict[str, Any]]] = {}
        self._edits: Dict[str, EditController] = {}
        self._ctx: Dict[str, tuple] = {}  # key -> (dataset, phase)
        self._studio = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self._run_preview)
        self._preview_dataset = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        bar = QtWidgets.QHBoxLayout()
        self.lbl_info = QtWidgets.QLabel()
        self.lbl_info.setObjectName("muted")
        self.lbl_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        bar.addWidget(self.lbl_info, 1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh from parameters")
        self.btn_refresh.setToolTip("Rebuild the structure from the current lattice, x/y/z and occupancy parameters.")
        self.btn_refresh.clicked.connect(lambda: self._load_structure(force=True))
        self.btn_pop = QtWidgets.QPushButton("Pop out")
        self.btn_pop.clicked.connect(self.toggle_pop)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(self.btn_pop)
        lay.addLayout(bar)
        self.placeholder = QtWidgets.QLabel("The structure is in its own window. Close it or press “Dock back” to return it here.")
        self.placeholder.setObjectName("banner")
        self.placeholder.setVisible(False)
        lay.addWidget(self.placeholder)
        self.host_area = QtWidgets.QWidget()
        self.host_layout = QtWidgets.QVBoxLayout(self.host_area)
        self.host_layout.setContentsMargins(0, 0, 0, 0)
        self.host_area.setMinimumHeight(380)
        lay.addWidget(self.host_area, 1)

    # ------------------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.dataset.id}:{self.phase.id}"

    def load(self, win, project, dataset=None, phase=None):
        self.win, self.project, self.dataset, self.phase = win, project, dataset, phase
        key = self.key
        if not opengl_available():
            self.lbl_info.setText(
                "The 3D viewer needs OpenGL, which is not available in this session "
                "(e.g. remote desktop without GPU acceleration)."
            )
            self.btn_pop.setEnabled(False)
            self.btn_refresh.setEnabled(False)
            return
        studio = self._studios.get(key)
        if studio is None:
            studio = self._create_studio(key)
        if self._studio is not None and self._studio is not studio and not self._popped.get(self._key_of(self._studio)):
            self._studio.setParent(None)
            self._studio.hide()
        self._studio = studio
        if self._popped.get(key):
            self.placeholder.setVisible(True)
            self.btn_pop.setText("Dock back")
        else:
            self._dock(studio)
        self._load_structure()

    def _key_of(self, studio) -> str:
        return next((k for k, s in self._studios.items() if s is studio), "")

    def _create_studio(self, key: str):
        from pdf_fitting.gui.structure_studio import StructureStudioWindow

        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            studio = StructureStudioWindow(None)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        controller = EditController(self, self.dataset, self.phase)
        self._edits[key] = controller
        self._ctx[key] = (self.dataset, self.phase)
        links = LinksTab(self, controller.editor)
        links_scroll = QtWidgets.QScrollArea()
        links_scroll.setWidgetResizable(True)
        links_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        links_scroll.setWidget(links)
        studio.controls.insertTab(0, links_scroll, "Refinement links")
        studio.splitter.setStretchFactor(0, 0)
        studio.splitter.setStretchFactor(1, 1)
        studio.controls.setCurrentIndex(0)
        # Every pick in the 3D view comes through select_atom_record.
        original = studio.select_atom_record

        def _select(record, _orig=original, _key=key):
            _orig(record)
            self._on_pick(_key, record)

        studio.select_atom_record = _select
        studio.closed.connect(lambda k=key: self._studio_closed(k))
        self._studios[key] = studio
        self._links[key] = links
        self._picks[key] = []
        return studio

    def _dock(self, studio):
        studio.setWindowFlags(Qt.Widget)
        self.host_layout.addWidget(studio, 1)
        studio.show()
        QtCore.QTimer.singleShot(0, lambda s=studio: self._fit_splitter(s))
        self.placeholder.setVisible(False)
        self.btn_pop.setText("Pop out")

    def toggle_pop(self):
        studio, key = self._studio, self.key
        if studio is None:
            return
        if self._popped.get(key):
            self._popped[key] = False
            self._dock(studio)
            return
        self._popped[key] = True
        studio.setParent(None)
        studio.setWindowFlags(Qt.Window)
        studio.setWindowTitle(f"Structure — {self.phase.name} ({self.dataset.name})")
        studio.resize(1400, 900)
        studio.show()
        studio.raise_()
        self.placeholder.setVisible(True)
        self.btn_pop.setText("Dock back")

    def _studio_closed(self, key):
        # Closing the popped-out window docks it back.
        if self._popped.get(key) and self._studios.get(key) is self._studio:
            self._popped[key] = False
            QtCore.QTimer.singleShot(0, lambda: self._dock(self._studio))

    # ------------------------------------------------------------------
    def _local_params(self) -> Dict[str, Any]:
        out = {k: p.value for k, p in self.dataset.parameters.items()}
        out.update({k: p.value for k, p in self.phase.parameters.items()})
        return out

    def _push_adp(self, studio, ps, phase=None):
        """Cartesian U of every anisotropic site of the phase → studio ellipsoids."""
        phase = phase or self.phase
        try:
            studio.set_adp_ellipsoids(adp_u_by_site(ps, {k: p.value for k, p in phase.parameters.items()}))
        except Exception as exc:  # noqa: BLE001
            print(f"[ADP] ellipsoids not shown: {exc}")

    @staticmethod
    def _fit_splitter(studio):
        """Controls ~1/3 of the width (at most 440 px); the 3D view gets the rest."""
        try:
            total = max(studio.splitter.width(), studio.width(), 900)
            left = int(min(440, max(340, 0.33 * total)))
            studio.splitter.setSizes([left, max(total - left, 300)])
        except RuntimeError:
            pass

    def _load_structure(self, force: bool = False):
        key = self.key
        gen = getattr(self.win, "_gen", 0)
        if not force and self._loaded_gen.get(key) == gen and key in self._ps:
            self._update_info()
            return
        path = self.project.structure_path(self.phase)
        if not os.path.exists(path):
            self.lbl_info.setText(f"CIF not found: {path}")
            return
        try:
            ps = T.phase_structure(path, self._local_params())
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"Could not build the structure: {exc}")
            return
        studio = self._studio
        first = key not in self._ps
        self._ps[key] = ps
        cfg_stem = os.path.splitext(self.project.path)[0] if self.project.path else ""
        sidecar_input = f"{cfg_stem}.{self.dataset.id}.{self.phase.id}.hklpdf" if cfg_stem else None
        try:
            if first:
                studio.set_structure(ps.structure, path, input_path=sidecar_input)
                self._push_adp(studio, ps)
                if not self._popped.get(key):
                    QtCore.QTimer.singleShot(0, lambda s_=studio: self._fit_splitter(s_))
            else:
                studio.update_structure(ps.structure)
                self._push_adp(studio, ps)
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"Viewer error: {exc}")
            return
        self._loaded_gen[key] = gen
        self._update_info()
        if not first:
            self._remap_picks(key)

    def _update_info(self):
        ps = self._ps.get(self.key)
        if ps is None:
            return
        lat = ps.structure.lattice
        self.lbl_info.setText(
            f"{self.phase.name}: {len(ps.structure)} atoms in the conventional cell, "
            f"{len(ps.group_labels)} site(s) in the asymmetric unit · "
            f"a={lat.a:.5f} b={lat.b:.5f} c={lat.c:.5f} Å · α={lat.alpha:.3f} β={lat.beta:.3f} γ={lat.gamma:.3f}° "
            "(current parameters)"
        )

    # ------------------------------------------------------------------
    # Hooks for the edit controller (stage 2)
    # ------------------------------------------------------------------
    def _params_for(self, key: str) -> Dict[str, Any]:
        ds, ph = self._ctx[key]
        out = {k: p.value for k, p in ds.parameters.items()}
        out.update({k: p.value for k, p in ph.parameters.items()})
        return out

    def rebuild_after_edit(self, key: str):
        """Rebuild the structure of one phase from its parameters (keeps camera, picks)."""
        ds, ph = self._ctx[key]
        path = self.project.structure_path(ph)
        try:
            ps = T.phase_structure(path, self._params_for(key))
        except Exception as exc:  # noqa: BLE001
            self.flash(f"Could not rebuild the structure: {exc}")
            return
        self._ps[key] = ps
        studio = self._studios.get(key)
        if studio is not None:
            studio.update_structure(ps.structure)
            self._push_adp(studio, ps, ph)
        self._remap_picks(key)
        if key == self.key:
            self._update_info()

    def _remap_picks(self, key: str):
        studio = self._studios.get(key)
        picks = self._picks.get(key) or []
        if studio is None or not picks:
            return
        lookup = {tuple(r["key"]): r for r in (studio.atom_records or [])}
        self._picks[key] = [dict(lookup.get(tuple(p["key"]), p)) for p in picks]
        try:
            studio.highlight_records(self._picks[key])
        except Exception:  # noqa: BLE001
            pass
        controller = self._edits.get(key)
        if controller is not None and self._picks[key]:
            controller.select(self._picks[key][-1])
        if key == self.key:
            self._show_picks()
            self.refresh_neighbors()

    def clear_picks_for(self, key: str):
        self._picks[key] = []
        studio = self._studios.get(key)
        if studio is not None and hasattr(studio, "highlight_records"):
            try:
                studio.highlight_records([])
            except Exception:  # noqa: BLE001
                pass
        if key == self.key:
            self._show_picks()
            links = self._links.get(key)
            if links is not None:
                links.tbl.setRowCount(0)

    def structure_changed(self, dataset, phase):
        """Sites were added / deleted: refresh the tree and dependent pages."""
        if self.win is not None:
            self.win.rebuild_tree()
            self.win._update_chips()

    def mark_loaded(self, key: str):
        self._loaded_gen[key] = getattr(self.win, "_gen", 0)

    def request_preview(self, dataset):
        self._preview_dataset = dataset
        self._preview_timer.start()

    def _run_preview(self):
        if self.win is None:
            return
        if self.win._thread is not None:
            self._preview_timer.start()  # a run is in progress; try again shortly
            return
        self.win.calculate()

    def flash(self, message: str):
        if self.win is not None:
            self.win.statusBar().showMessage(message, 6000)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), message)

    # ------------------------------------------------------------------
    # Picks and measurements
    # ------------------------------------------------------------------
    def _on_pick(self, key: str, record: Dict[str, Any]):
        picks = self._picks.setdefault(key, [])
        links = self._links.get(key)
        group_size = 3 if (links is not None and links.rb_angle.isChecked()) else 2
        if len(picks) >= group_size:
            # The previous pair / triplet is complete: start a new one.
            picks.clear()
        picks.append(dict(record))
        # Highlight every kept pick (pair / triplet) with distance lines.
        studio = self._studios.get(key)
        if studio is not None and hasattr(studio, "highlight_records"):
            try:
                studio.highlight_records(picks)
            except Exception as exc:  # noqa: BLE001
                print(f"[STRUCTURE] highlight failed: {exc}")
        controller = self._edits.get(key)
        if controller is not None:
            controller.select(record)
        if key == self.key:
            self._show_picks()
            self.refresh_neighbors()

    def clear_picks(self):
        self._picks[self.key] = []
        if self.key in self._edits:
            self._edits[self.key].select(None)
        studio = self._studios.get(self.key)
        if studio is not None and hasattr(studio, "highlight_records"):
            studio.highlight_records([])
        self._show_picks()
        links = self._links[self.key]
        links.tbl.setRowCount(0)

    def _show_picks(self):
        links = self._links[self.key]
        ps = self._ps.get(self.key)
        picks = self._picks.get(self.key, [])
        if not picks or ps is None:
            links.lbl_picks.setText("No atom picked.")
            links.lbl_site.setText("—")
            links.lbl_measure.setVisible(False)
            links.btn_pair.setEnabled(False)
            return
        names = [f"{p.get('label', '?')} ({p.get('element', '')})" for p in picks]
        links.lbl_picks.setText("Picks: " + "  →  ".join(names))
        pos = [p["position"] for p in picks]
        parts = []
        angle_mode = links.rb_angle.isChecked()
        if len(picks) >= 2:
            parts.append(f"distance {names[0]} – {names[1]}: <b>{T.distance(pos[0], pos[1]):.4f} Å</b>")
        if len(picks) == 3:
            parts.append(f"distance {names[1]} – {names[2]}: <b>{T.distance(pos[1], pos[2]):.4f} Å</b>")
            parts.append(f"angle {names[0]} – {names[1]} – {names[2]}: <b>{T.angle(pos[0], pos[1], pos[2]):.2f}°</b>")
        elif angle_mode and len(picks) < 3:
            parts.append(f"pick {3 - len(picks)} more atom(s) for the angle")
        links.lbl_measure.setText("<br>".join(parts))
        links.lbl_measure.setVisible(bool(parts))
        links.btn_pair.setEnabled(len(picks) >= 2)

        last = picks[-1]
        site = int(last.get("base_site_index", 0))
        g = ps.group_of_site(site)
        key = ps.group_keys[g]
        params = self.phase.parameters
        cons = ps.constraints(g)
        rows = []
        for letter in "xyz":
            p = params.get(f"{letter}_{key}")
            val = fmt_value(p.value) if p is not None else f"{ps.reference[g]['xyz'.index(letter)]:.6g} (CIF)"
            flag = " · refined" if p is not None and p.refine else ""
            rows.append(f"{letter} = {val} <i>({cons[letter]}{flag})</i>")
        occ = params.get(f"occ_{key}")
        biso = params.get(f"biso_{key}") or params.get(f"biso_{last.get('element', '')}")
        links.lbl_site.setText(
            f"<b>{ps.group_labels[g]}</b> · {last.get('element', '')} · multiplicity {ps.multiplicity(g)} in the cell<br>"
            + "<br>".join(rows)
            + f"<br>occ = {fmt_value(occ.value) if occ else '1 (CIF)'} · Biso = {fmt_value(biso.value) if biso else '—'}"
        )

    def refresh_neighbors(self):
        links = self._links.get(self.key)
        ps = self._ps.get(self.key)
        picks = self._picks.get(self.key, [])
        if links is None or ps is None or not picks:
            return
        site = int(picks[-1].get("base_site_index", 0))
        try:
            rows = T.neighbor_table(ps, site, float(links.sp_rmax.value()))
        except Exception as exc:  # noqa: BLE001
            rows = []
            links.lbl_picks.setText(f"Neighbor search failed: {exc}")
        links.tbl.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate((row["label"], row["pair"], f"{row['distance']:.4f}", str(row["count"]), "" if row["k"] is None else str(row["k"]))):
                it = QtWidgets.QTableWidgetItem(text)
                it.setData(Qt.UserRole, row)
                links.tbl.setItem(r, c, it)
        self._neighbor_rows = rows

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------
    def go_to_atoms(self):
        picks = self._picks.get(self.key, [])
        ps = self._ps.get(self.key)
        if not picks or ps is None:
            return
        label = ps.label_of_site(int(picks[-1].get("base_site_index", 0)))
        self.win.navigate_to(f"ph:{self.dataset.id}:{self.phase.id}:atoms", label=label)

    def go_to_pair(self):
        picks = self._picks.get(self.key, [])
        ps = self._ps.get(self.key)
        if len(picks) < 2 or ps is None:
            return
        a, b = picks[0], picks[1]
        pair = T.pair_name(a.get("element", ""), b.get("element", ""))
        d = T.distance(a["position"], b["position"])
        shells = T.pair_shell_distances(ps, *pair.split("-"), rmax=d + 0.05)
        k = int(np.argmin(np.abs(np.asarray(shells) - d))) if shells else None
        self.win.navigate_to(f"ph:{self.dataset.id}:{self.phase.id}:dyn", pair=pair, k=k, distance=d)

    def go_to_neighbor_row(self):
        links = self._links.get(self.key)
        items = links.tbl.selectedItems() if links else []
        if not items:
            return
        row = items[0].data(Qt.UserRole)
        self.win.navigate_to(f"ph:{self.dataset.id}:{self.phase.id}:dyn", pair=row["pair"], k=row["k"], distance=row["distance"])

    def mark_selected_on_fit(self):
        links = self._links.get(self.key)
        if links is None:
            return
        # One entry per selected row (each row has several selected cells).
        rows = {i.row(): i.data(Qt.UserRole) for i in links.tbl.selectedItems()}
        if not rows:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Select neighbor rows first.")
            return
        marks = [(r["distance"], f"{self.phase.name}: {r['pair']} k={r['k']}") for r in rows.values()]
        pairs = sorted({f"{self.phase.name}: {r['pair']}" for r in rows.values()})
        self.win.mark_distances_on_fit(self.dataset, marks, pairs=pairs)

    def clear_fit_marks(self):
        self.win.mark_distances_on_fit(self.dataset, [], clear=True)

    def shutdown(self):
        for controller in self._edits.values():
            if controller.drag is not None:
                controller.drag.uninstall()
        for studio in self._studios.values():
            try:
                studio.shutdown()
            except Exception:  # noqa: BLE001
                pass


def adp_u_by_site(ps, params):
    """
    {site index: Cartesian U (Å²)} for the sites of a PhaseStructure whose
    group has beta<ij>_<key> parameters: β_site = R β_ASU Rᵀ (R maps the
    ASU position onto the site), U = Aᵀ (β_site / 2π²) A.
    """
    import numpy as _np

    from pdf_fitting.structure_handler import (
        ADP_COMPONENTS, adp_site_rotations, beta_constraints, beta_matrix, same_mod1, symmetry_operations,
    )

    lc = {str(k).lower(): v for k, v in (params or {}).items()}
    keys = [k for k in ps.group_keys if any(f"beta{c}_{k.lower()}" in lc for c in ADP_COMPONENTS)]
    if not keys:
        return {}
    st = ps.structure
    ops = symmetry_operations(st)
    A = st.lattice.matrix
    frac = st.frac_coords
    out = {}
    for key in keys:
        g = ps.group_keys.index(key)
        p0 = _np.asarray(ps.reference[g], dtype=float)
        cons = beta_constraints(adp_site_rotations(p0, ops))
        free = {c: float(lc.get(f"beta{c}_{key.lower()}", 0.0)) for c in ADP_COMPONENTS if cons[c][0] == "free"}
        b = [free[c] if cons[c][0] == "free" else (0.0 if cons[c][0] == "fixed" else sum(w * free[t] for t, w in cons[c][1]))
             for c in ADP_COMPONENTS]
        B = beta_matrix(b)
        for s in _np.flatnonzero(ps.site_group == g):
            R = next((R for R, t in ops if same_mod1(R @ p0 + t, frac[s])), _np.eye(3))
            out[int(s)] = A.T @ (R @ B @ R.T / (2.0 * _np.pi ** 2)) @ A
    return out
