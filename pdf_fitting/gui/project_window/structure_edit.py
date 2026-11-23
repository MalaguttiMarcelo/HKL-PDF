"""
Stage 2 of the structure editor: symmetry-preserving edits.

* SiteEditor: numeric fields for the selected site's x / y / z (only the
  coordinates the site symmetry leaves free are editable; tied ones follow,
  fixed ones are locked), occupancy and Biso. Values are the same project
  parameters the refinement uses (x_<site>, y_<site>, z_<site>, occ_<site>,
  biso_<site> or biso_<Element>).
* DragController: constrained drag in the 3D view. The mouse displacement is
  projected onto the free directions of the dragged atom (its symmetry copy's
  rotation R_s and the special-position basis B), so every symmetry-equivalent
  atom moves consistently and special positions are kept.
* Undo / redo per phase, "Reset site to CIF", and an optional live G(r)
  preview (recalculates after each change; off by default).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.project.model import Parameter
from pdf_fitting.structure_handler import same_mod1, site_coordinate_basis


def _spin(lo, hi, decimals, step) -> QtWidgets.QDoubleSpinBox:
    sp = QtWidgets.QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setKeyboardTracking(False)
    sp.setMinimumWidth(110)
    return sp


class SiteEditor(QtWidgets.QWidget):
    """Numeric editing of the selected site (symmetry-aware)."""

    def __init__(self, controller: "EditController"):
        super().__init__()
        self.c = controller
        self._block = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(6)

        self.site_part = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.site_part)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.spins = {}
        self.status = {}
        for row, letter in enumerate("xyz"):
            sp = _spin(-2.0, 3.0, 6, 0.001)
            sp.valueChanged.connect(lambda _v, a=letter: self._coord_changed(a))
            st = QtWidgets.QLabel()
            st.setObjectName("muted")
            grid.addWidget(QtWidgets.QLabel(letter), row, 0)
            grid.addWidget(sp, row, 1)
            grid.addWidget(st, row, 2)
            self.spins[letter] = sp
            self.status[letter] = st
        self.sp_occ = _spin(0.0, 1.0, 4, 0.01)
        self.sp_occ.valueChanged.connect(lambda _v: self._scalar_changed("occ"))
        self.sp_biso = _spin(0.0, 20.0, 4, 0.05)
        self.sp_biso.valueChanged.connect(lambda _v: self._scalar_changed("biso"))
        self.lbl_biso = QtWidgets.QLabel()
        self.lbl_biso.setObjectName("muted")
        grid.addWidget(QtWidgets.QLabel("occ"), 3, 0)
        grid.addWidget(self.sp_occ, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Biso"), 4, 0)
        grid.addWidget(self.sp_biso, 4, 1)
        grid.addWidget(self.lbl_biso, 4, 2)
        grid.setColumnStretch(2, 1)
        lay.addWidget(self.site_part)

        row = QtWidgets.QHBoxLayout()
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_redo = QtWidgets.QPushButton("Redo")
        self.btn_reset = QtWidgets.QPushButton("Reset site to CIF")
        self.btn_undo.clicked.connect(controller.undo)
        self.btn_redo.clicked.connect(controller.redo)
        self.btn_reset.clicked.connect(controller.reset_site)
        for b in (self.btn_undo, self.btn_redo, self.btn_reset):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add site…")
        self.btn_add.setToolTip("Add a new site; the space group generates all its symmetry-equivalent atoms.")
        self.btn_delete = QtWidgets.QPushButton("Delete site")
        self.btn_delete.setToolTip("Delete the selected site and all its symmetry-equivalent atoms.")
        self.btn_export = QtWidgets.QPushButton("Export CIF…")
        self.btn_export.setToolTip("Write the current structure (lattice, x/y/z, occupancies, Biso) as a CIF.")
        self.btn_add.clicked.connect(controller.add_site)
        self.btn_delete.clicked.connect(controller.delete_site)
        self.btn_export.clicked.connect(controller.export_cif)
        for b in (self.btn_add, self.btn_delete, self.btn_export):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.chk_drag = QtWidgets.QCheckBox("Drag atoms in the 3D view (moves only along symmetry-free directions)")
        self.chk_drag.toggled.connect(controller.set_drag_enabled)
        self.chk_preview = QtWidgets.QCheckBox("Live G(r) preview (recalculate after each change)")
        self.chk_preview.setChecked(False)
        lay.addWidget(self.chk_drag)
        lay.addWidget(self.chk_preview)
        self.site_part.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_delete.setEnabled(False)

    # ------------------------------------------------------------------
    def show_site(self, info: Optional[Dict[str, Any]]):
        self._block = True
        try:
            has_site = info is not None
            self.site_part.setEnabled(has_site)
            self.btn_reset.setEnabled(has_site)
            self.btn_delete.setEnabled(has_site)
            if info is None:
                for letter in "xyz":
                    self.status[letter].setText("")
                return
            for letter in "xyz":
                sp = self.spins[letter]
                sp.setValue(float(info["coords"][letter]))
                state = info["constraints"][letter]
                sp.setEnabled(state == "free")
                self.status[letter].setText(
                    {"free": "free", "tied": "tied (follows the free coordinate)", "fixed": "fixed by symmetry"}[state]
                    + (" · refined" if info["refined"].get(letter) else "")
                )
            self.sp_occ.setValue(float(info["occ"]))
            self.sp_biso.setValue(float(info["biso"]))
            self.lbl_biso.setText(info["biso_label"])
            self.sp_biso.setEnabled(info["biso_name"] is not None)
        finally:
            self._block = False
        self.update_history_buttons()

    def update_history_buttons(self):
        self.btn_undo.setEnabled(self.c.can_undo())
        self.btn_redo.setEnabled(self.c.can_redo())

    def _coord_changed(self, _axis):
        if self._block:
            return
        free = {a: self.spins[a].value() for a in "xyz"}
        self.c.set_coordinates(free, source="field")

    def _scalar_changed(self, which):
        if self._block:
            return
        if which == "occ":
            self.c.set_scalar("occ", self.sp_occ.value())
        else:
            self.c.set_scalar("biso", self.sp_biso.value())


class EditController(QtCore.QObject):
    """Applies edits of the structure page to the phase parameters."""

    def __init__(self, page, dataset, phase):
        super().__init__()
        self.page = page
        # Bound to one phase: a popped-out viewer keeps editing its own phase.
        self.ds = dataset
        self.ph = phase
        self.key = f"{dataset.id}:{phase.id}"
        self.editor = SiteEditor(self)
        self._undo: Dict[str, List[Dict[str, Optional[Parameter]]]] = {}
        self._redo: Dict[str, List[Dict[str, Optional[Parameter]]]] = {}
        self._site: Optional[Dict[str, Any]] = None  # current site context
        self.drag = None

    # ------------------------------------------------------------------
    # Context of the selected site
    # ------------------------------------------------------------------
    def select(self, record: Optional[Dict[str, Any]]):
        page = self.page
        ps = page._ps.get(self.key)
        if record is None or ps is None:
            self._site = None
            self.editor.show_site(None)
            return
        site = int(record.get("base_site_index", 0))
        g = ps.group_of_site(site)
        self._site = {"site": site, "group": g, "key": ps.group_keys[g], "element": record.get("element", "")}
        self.editor.show_site(self._site_info())

    def _site_info(self) -> Optional[Dict[str, Any]]:
        if self._site is None:
            return None
        page = self.page
        ps = page._ps[self.key]
        g, key = self._site["group"], self._site["key"]
        params = self.ph.parameters
        p_cur = self._current_position(g)
        coords = {letter: p_cur[i] for i, letter in enumerate("xyz")}
        refined = {letter: bool(params.get(f"{letter}_{key}") and params[f"{letter}_{key}"].refine) for letter in "xyz"}
        occ = params.get(f"occ_{key}")
        occ_val = float(occ.value) if occ is not None else float(sum(ps.structure[self._site["site"]].species.values()))
        el = self._site["element"]
        biso_name = None
        if f"biso_{key}" in params:
            biso_name, label = f"biso_{key}", "this site"
        else:
            for name in params:
                if name.lower() == f"biso_{el}".lower():
                    biso_name, label = name, f"all {el} sites (one Biso per element)"
                    break
            else:
                label = "no Biso parameter"
        biso_val = float(params[biso_name].value) if biso_name else 0.0
        return {
            "coords": coords,
            "constraints": ps.constraints(g),
            "refined": refined,
            "occ": occ_val,
            "biso": biso_val,
            "biso_name": biso_name,
            "biso_label": label,
        }

    def _current_position(self, g: int) -> np.ndarray:
        page = self.page
        ps = page._ps[self.key]
        key = ps.group_keys[g]
        params = self.ph.parameters
        p = np.array(ps.reference[g], dtype=float)
        for i, letter in enumerate("xyz"):
            par = params.get(f"{letter}_{key}")
            if par is not None:
                try:
                    p[i] = float(par.value)
                except (TypeError, ValueError):
                    pass
        return p

    # ------------------------------------------------------------------
    # Undo / redo (snapshots of the touched parameters)
    # ------------------------------------------------------------------
    def _snapshot(self, names) -> Dict[str, Optional[Parameter]]:
        params = self.ph.parameters
        return {n: (Parameter(**vars(params[n])) if n in params else None) for n in names}

    def _push_undo(self, names):
        key = self.key
        self._undo.setdefault(key, []).append(self._snapshot(names))
        del self._undo[key][:-200]
        self._redo[key] = []

    def can_undo(self) -> bool:
        return bool(self._undo.get(self.key))

    def can_redo(self) -> bool:
        return bool(self._redo.get(self.key))

    def _restore(self, snap):
        params = self.ph.parameters
        for name, p in snap.items():
            if p is None:
                params.pop(name, None)
            else:
                params[name] = Parameter(**vars(p))

    def undo(self):
        key = self.key
        if not self._undo.get(key):
            return
        snap = self._undo[key].pop()
        if snap.get("__full__"):
            self._redo.setdefault(key, []).append(self._full_snapshot())
            self._restore_full(snap)
            self._after_structure_change()
            return
        self._redo.setdefault(key, []).append(self._snapshot(snap.keys()))
        self._restore(snap)
        self._after_edit(final=True)

    def redo(self):
        key = self.key
        if not self._redo.get(key):
            return
        snap = self._redo[key].pop()
        if snap.get("__full__"):
            self._undo.setdefault(key, []).append(self._full_snapshot())
            self._restore_full(snap)
            self._after_structure_change()
            return
        self._undo.setdefault(key, []).append(self._snapshot(snap.keys()))
        self._restore(snap)
        self._after_edit(final=True)

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------
    def _coord_names(self, key):
        return [f"{letter}_{key}" for letter in "xyz"]

    def set_coordinates(self, values: Dict[str, float], *, source: str = "field", push_undo: bool = True, final: bool = True):
        """
        Set the selected site's coordinates. Only the free coordinates of
        ``values`` are used; tied and fixed ones follow from the site
        symmetry (delta = B @ shift of the free coordinates).
        """
        if self._site is None:
            return
        page = self.page
        ps = page._ps[self.key]
        g, key = self._site["group"], self._site["key"]
        p0 = self._current_position(g)
        B, free = site_coordinate_basis(p0, ps.ops)
        if not free:
            return
        shift = np.array([float(values["xyz"[a]]) - p0[a] for a in free])
        if np.allclose(shift, 0.0, atol=1e-12):
            return
        new = p0 + B @ shift
        if push_undo:
            self._push_undo(self._coord_names(key))
        params = self.ph.parameters
        for i, letter in enumerate("xyz"):
            name = f"{letter}_{key}"
            if name not in params:
                params[name] = Parameter(value=float(ps.reference[g][i]))
            params[name].value = float(new[i])
        self._after_edit(final=final)

    def set_scalar(self, which: str, value: float):
        if self._site is None:
            return
        info = self._site_info()
        key = self._site["key"]
        params = self.ph.parameters
        if which == "occ":
            name = f"occ_{key}"
            self._push_undo([name])
            params.setdefault(name, Parameter(value=float(info["occ"]), min=0.0, max=1.0)).value = float(value)
        else:
            name = info["biso_name"]
            if name is None:
                return
            self._push_undo([name])
            params[name].value = float(value)
        self._after_edit(final=True)

    def reset_site(self):
        if self._site is None:
            return
        page = self.page
        ps = page._ps[self.key]
        g, key = self._site["group"], self._site["key"]
        names = self._coord_names(key) + [f"occ_{key}"]
        self._push_undo(names)
        params = self.ph.parameters
        for i, letter in enumerate("xyz"):
            name = f"{letter}_{key}"
            if name in params:
                params[name].value = float(ps.reference[g][i])
        if f"occ_{key}" in params:
            base = ps.structure[self._site["site"]]
            try:
                from pdf_fitting.gui.cif_utils import read_cif_asu_sites

                _s, asu = read_cif_asu_sites(ps.cif_path)
                from pdf_fitting.structure_handler import safe_param_key

                occ0 = next((float(r.get("occ", 1.0)) for r in asu if safe_param_key(r.get("label", "")) == key), 1.0)
            except Exception:  # noqa: BLE001
                occ0 = float(sum(base.species.values()))
            params[f"occ_{key}"].value = occ0
        self._after_edit(final=True)

    def _after_edit(self, *, final: bool):
        """Rebuild the view, keep the selection, mark the project edited."""
        page = self.page
        page.rebuild_after_edit(self.key)
        if self._site is not None:
            self.editor.show_site(self._site_info())
        self.editor.update_history_buttons()
        if final:
            page.edited.emit()
            page.mark_loaded(self.key)
            if self.editor.chk_preview.isChecked():
                page.request_preview(self.ds)

    # ------------------------------------------------------------------
    # Stage 3: add / delete whole orbits, export CIF
    # ------------------------------------------------------------------
    def _edit_section(self) -> Dict[str, Any]:
        """The phase's stored site list (created from the CIF on first edit)."""
        from pdf_fitting.project.edited_cif import base_sites

        sections = self.ph.sections
        edit = sections.get("structure_edit")
        if not edit or edit.get("sites") is None:
            project = self.page.project
            base = project.resolve_path(self.ph.cif)
            edit = {"base_cif": self.ph.cif, "sites": base_sites(base)}
            sections["structure_edit"] = edit
        return edit

    def _full_snapshot(self) -> Dict[str, Any]:
        import copy

        return {
            "__full__": True,
            "params": {k: Parameter(**vars(p)) for k, p in self.ph.parameters.items()},
            "section": copy.deepcopy((self.ph.sections or {}).get("structure_edit")),
        }

    def _push_full_undo(self):
        self._undo.setdefault(self.key, []).append(self._full_snapshot())
        del self._undo[self.key][:-200]
        self._redo[self.key] = []

    def _restore_full(self, snap):
        import copy

        self.ph.parameters.clear()
        self.ph.parameters.update({k: Parameter(**vars(p)) for k, p in snap["params"].items()})
        if snap["section"] is None:
            self.ph.sections.pop("structure_edit", None)
        else:
            self.ph.sections["structure_edit"] = copy.deepcopy(snap["section"])

    def _after_structure_change(self):
        page = self.page
        page.clear_picks_for(self.key)
        self._site = None
        self.editor.show_site(None)
        page.rebuild_after_edit(self.key)
        page.edited.emit()
        page.mark_loaded(self.key)
        page.structure_changed(self.ds, self.ph)
        self.editor.update_history_buttons()
        if self.editor.chk_preview.isChecked():
            page.request_preview(self.ds)

    def add_site(self):
        page = self.page
        ps = page._ps.get(self.key)
        if ps is None:
            return
        picks = page._picks.get(self.key) or []
        dlg = AddSiteDialog(self, ps, picks, self._edit_section_preview())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        site = dlg.result_site()
        edit_sites_now = self._edit_section_preview()
        check = self._overlap_check(site)
        if check is not None:
            msg, mult, exact = check
            if exact:
                QtWidgets.QMessageBox.warning(page, "Add site", f"{site['label']} cannot be added: {msg}")
                return
            if QtWidgets.QMessageBox.question(
                page, "Add site",
                f"The new site {site['label']} generates {mult} atom(s) in the cell.\n{msg}\n\nAdd it anyway?",
            ) != QtWidgets.QMessageBox.Yes:
                return
        self._push_full_undo()
        edit = self._edit_section()
        edit["sites"].append(site)
        # One Biso per element: give a new element its Biso parameter.
        params = self.ph.parameters
        uses_element_biso = any(
            name.lower().startswith("biso_") and name[5:].lower() in {s["element"].lower() for s in edit_sites_now}
            for name in params
        )
        if uses_element_biso and not any(n.lower() == f"biso_{site['element']}".lower() for n in params):
            params[f"biso_{site['element']}"] = Parameter(
                value=float(site.get("biso") or 0.5), min=0.0, max=10.0
            )
        self._after_structure_change()
        page.flash(f"Added {site['label']} ({site['element']}).")

    def _edit_section_preview(self) -> List[Dict[str, Any]]:
        from pdf_fitting.project.edited_cif import base_sites, copy_sites

        edit = (self.ph.sections or {}).get("structure_edit")
        if edit and edit.get("sites") is not None:
            return copy_sites(edit["sites"])
        return base_sites(self.page.project.resolve_path(self.ph.cif))

    def _orbit(self, f) -> np.ndarray:
        """Distinct fractional positions generated from f by the space group."""
        ps = self.page._ps[self.key]
        pts: List[np.ndarray] = []
        for R, t in ps.ops:
            q = np.mod(R @ np.asarray(f, dtype=float) + t, 1.0)
            if not any(same_mod1(q, p, 1.0e-3) for p in pts):
                pts.append(q)
        return np.asarray(pts, dtype=float)

    def _overlap_check(self, site):
        """
        (message, multiplicity, exact) for the new orbit, or None if it keeps
        at least 0.5 Å from every existing atom. ``exact`` means coincidence
        (< 0.01 Å): such a site cannot be added.
        """
        ps = self.page._ps[self.key]
        pts = self._orbit([site["x"], site["y"], site["z"]])
        mult = len(pts)
        existing = np.asarray(ps.structure.frac_coords, dtype=float)
        if existing.size == 0:
            return None
        d = ps.structure.lattice.get_all_distances(pts, existing)
        dmin = float(np.min(d))
        if dmin < 0.01:
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            return (f"It coincides with an existing atom ({ps.label_of_site(int(j))}).", mult, True)
        if dmin < 0.5:
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            return (f"It comes {dmin:.3f} Å from {ps.label_of_site(int(j))}.", mult, False)
        return None

    def delete_site(self):
        if self._site is None:
            return
        page = self.page
        ps = page._ps.get(self.key)
        g, key = self._site["group"], self._site["key"]
        label = ps.group_labels[g]
        if QtWidgets.QMessageBox.question(
            page, "Delete site",
            f"Delete site {label} and its {ps.multiplicity(g)} symmetry-equivalent atom(s)?\n"
            "Its x/y/z, occupancy and site Biso parameters are removed too (Undo restores them).",
        ) != QtWidgets.QMessageBox.Yes:
            return
        from pdf_fitting.structure_handler import safe_param_key

        self._push_full_undo()
        edit = self._edit_section()
        edit["sites"] = [s for s in edit["sites"] if safe_param_key(s["label"]) != key]
        for name in [f"{p}_{key}" for p in ("x", "y", "z", "occ", "biso")]:
            self.ph.parameters.pop(name, None)
        self._after_structure_change()
        page.flash(f"Deleted {label}.")

    def export_cif(self):
        """Write the current structure (lattice, x/y/z, occ, Biso) as a CIF."""
        import os

        from pdf_fitting.project.edited_cif import base_sites, cif_text_with_sites
        from pdf_fitting.structure_handler import safe_param_key

        page = self.page
        ps = page._ps.get(self.key)
        if ps is None:
            return
        project = page.project
        source = project.structure_path(self.ph)
        params = self.ph.parameters
        lookup = {k.lower(): p for k, p in params.items()}
        sites = []
        for s in base_sites(source):
            key = safe_param_key(s["label"])
            row = dict(s)
            for letter in "xyz":
                p = lookup.get(f"{letter}_{key}")
                if p is not None:
                    row[letter] = float(p.value)
            p = lookup.get(f"occ_{key}")
            if p is not None:
                row["occ"] = float(p.value)
            p = lookup.get(f"biso_{key}") or lookup.get(f"biso_{s['element'].lower()}")
            if p is not None:
                row["biso"] = float(p.value)
            b = [lookup.get(f"beta{c}_{key}") for c in ("11", "22", "33", "12", "13", "23")]
            if any(x is not None for x in b):
                row["beta"] = [float(x.value) if x is not None else 0.0 for x in b]
            sites.append(row)
        lat = ps.structure.lattice
        lattice = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), lat.parameters))
        default = os.path.join(
            os.path.dirname(project.path) if project.path else os.getcwd(),
            f"{self.ph.name}_refined.cif",
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(page, "Export CIF", default, "CIF files (*.cif)")
        if not path:
            return
        text = cif_text_with_sites(source, sites, lattice=lattice)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        page.flash(f"Exported {os.path.basename(path)} ({len(sites)} site(s) in the asymmetric unit).")

    # ------------------------------------------------------------------
    # Drag
    # ------------------------------------------------------------------
    def set_drag_enabled(self, on: bool):
        studio = self.page._studios.get(self.key)
        if studio is None:
            return
        if self.drag is None or self.drag.studio is not studio:
            if self.drag is not None:
                self.drag.uninstall()
            self.drag = DragController(studio, self)
        if on:
            self.drag.install()
        else:
            self.drag.uninstall()


class DragController:
    """
    Constrained drag of atoms in the VTK view.

    Left-press on an atom starts a drag (camera rotation is suppressed);
    the mouse displacement at the atom's depth is converted to a Cartesian
    displacement and solved for the shift of the free coordinates:
        Δcart = Lᵀ R_s B s   →   s = lstsq(Lᵀ R_s B, Δcart)
    """

    def __init__(self, studio, controller: EditController):
        self.studio = studio
        self.c = controller
        self._tags: List[int] = []
        self._state: Optional[Dict[str, Any]] = None
        self._last_update = 0.0

    @property
    def iren(self):
        return self.studio.plotter.iren.interactor

    def install(self):
        if self._tags:
            return
        for event, cb in (
            ("LeftButtonPressEvent", self._press),
            ("MouseMoveEvent", self._move),
            ("LeftButtonReleaseEvent", self._release),
        ):
            self._tags.append(self.iren.AddObserver(event, cb, 10.0))

    def uninstall(self):
        for tag in self._tags:
            try:
                self.iren.RemoveObserver(tag)
            except Exception:  # noqa: BLE001
                pass
        self._tags = []
        self._state = None

    def _abort(self, index: int):
        try:
            self.iren.GetCommand(self._tags[index]).SetAbortFlag(1)
        except Exception:  # noqa: BLE001
            pass

    def _renderer(self):
        return self.studio.plotter.renderer

    def _display_to_world(self, x, y, z):
        ren = self._renderer()
        ren.SetDisplayPoint(float(x), float(y), float(z))
        ren.DisplayToWorld()
        w = ren.GetWorldPoint()
        return np.array(w[:3]) / (w[3] if w[3] else 1.0)

    def _world_to_display(self, p):
        ren = self._renderer()
        ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        ren.WorldToDisplay()
        return np.array(ren.GetDisplayPoint())

    def _atom_under_cursor(self, x, y) -> Optional[Dict[str, Any]]:
        """
        First atom sphere hit by the mouse ray (nearest to the camera).
        Bonds and other meshes are ignored, unlike geometric pickers.
        """
        records = self.studio.atom_records or []
        if not records:
            return None
        near = self._display_to_world(x, y, 0.0)
        far = self._display_to_world(x, y, 1.0)
        direction = far - near
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            return None
        direction /= length
        centers = np.asarray([r["position"] for r in records], dtype=float)
        radii = np.asarray([float(r.get("radius", 0.4)) for r in records], dtype=float)
        rel = centers - near[None, :]
        t = rel @ direction
        perp = np.linalg.norm(rel - t[:, None] * direction[None, :], axis=1)
        hit = np.flatnonzero((perp <= radii) & (t > 0))
        if hit.size == 0:
            return None
        return records[int(hit[np.argmin(t[hit])])]

    def _press(self, obj, _event):
        x, y = self.iren.GetEventPosition()
        record = self._atom_under_cursor(x, y)
        if record is None:
            return  # empty space: normal camera rotation
        # The drag also selects the atom (pick list, site editor).
        self.studio.select_atom_record(record)
        page = self.c.page
        ps = page._ps.get(self.c.key)
        if ps is None or self.c._site is None:
            return
        g = self.c._site["group"]
        p_cur = self.c._current_position(g)
        B, free = site_coordinate_basis(p_cur, ps.ops)
        self._abort(0)
        if not free:
            page.flash("This site is fixed by symmetry; it cannot move without lowering the symmetry.")
            return
        # Rotation of the dragged symmetry copy relative to the ASU position.
        frac_site = np.asarray(ps.structure[int(record["base_site_index"])].frac_coords, dtype=float)
        R_s = np.eye(3)
        for R, t in ps.ops:
            if same_mod1(R @ p_cur + t, frac_site):
                R_s = R
                break
        L = np.asarray(ps.structure.lattice.matrix, dtype=float)
        A = L.T @ R_s @ B
        depth = self._world_to_display(record["position"])[2]
        start_world = self._display_to_world(x, y, depth)
        self.c._push_undo(self.c._coord_names(self.c._site["key"]))
        self._state = {
            "A": A,
            "free": list(free),
            "p_start": p_cur.copy(),
            "depth": depth,
            "start_world": start_world,
        }

    def _move(self, obj, _event):
        if self._state is None:
            return
        self._abort(1)
        now = time.perf_counter()
        if now - self._last_update < 0.04:
            return
        self._last_update = now
        self._apply_drag(final=False)

    def _release(self, obj, _event):
        if self._state is None:
            return
        self._abort(2)
        self._apply_drag(final=True)
        self._state = None

    def _apply_drag(self, *, final: bool):
        st = self._state
        x, y = self.iren.GetEventPosition()
        world = self._display_to_world(x, y, st["depth"])
        d_cart = world - st["start_world"]
        s, *_ = np.linalg.lstsq(st["A"], d_cart, rcond=None)
        values = {letter: st["p_start"][i] for i, letter in enumerate("xyz")}
        for j, axis in enumerate(st["free"]):
            values["xyz"[axis]] = st["p_start"][axis] + float(s[j])
        self.c.set_coordinates(values, source="drag", push_undo=False, final=final)


class AddSiteDialog(QtWidgets.QDialog):
    """New site: element, label, fractional position, occupancy, Biso."""

    def __init__(self, controller: EditController, ps, picks, current_sites):
        super().__init__(controller.page)
        self.setWindowTitle("Add site")
        self._sites = current_sites
        self._ps = ps
        self._picks = picks
        form = QtWidgets.QFormLayout(self)

        self.cb_el = QtWidgets.QComboBox()
        self.cb_el.setEditable(True)
        present = []
        for site in current_sites:
            if site["element"] not in present:
                present.append(site["element"])
        try:
            from pymatgen.core.periodic_table import Element

            others = [e.symbol for e in Element if e.symbol not in present and e.Z <= 103]
        except Exception:  # noqa: BLE001
            others = []
        self.cb_el.addItems(present + others)
        self.cb_el.currentTextChanged.connect(self._suggest_label)
        self.le_label = QtWidgets.QLineEdit()
        self.spins = {}
        pos_row = QtWidgets.QHBoxLayout()
        for letter in "xyz":
            sp = _spin(-1.0, 2.0, 6, 0.01)
            self.spins[letter] = sp
            pos_row.addWidget(QtWidgets.QLabel(letter))
            pos_row.addWidget(sp)
        btns = QtWidgets.QHBoxLayout()
        b_pick = QtWidgets.QPushButton("Last picked atom")
        b_mid = QtWidgets.QPushButton("Midpoint of the picked pair")
        b_pick.clicked.connect(self._from_pick)
        b_mid.clicked.connect(self._from_midpoint)
        b_pick.setEnabled(bool(picks))
        b_mid.setEnabled(len(picks) >= 2)
        btns.addWidget(b_pick)
        btns.addWidget(b_mid)
        self.sp_occ = _spin(0.0, 1.0, 4, 0.05)
        self.sp_occ.setValue(1.0)
        self.sp_biso = _spin(0.0, 20.0, 4, 0.1)
        self.sp_biso.setValue(0.5)

        form.addRow("Element", self.cb_el)
        form.addRow("Label", self.le_label)
        form.addRow("Fractional position", pos_row)
        form.addRow("Take position from", btns)
        form.addRow("Occupancy", self.sp_occ)
        form.addRow("Biso (Å²)", self.sp_biso)
        note = QtWidgets.QLabel(
            "The space group generates every symmetry-equivalent atom. A position on a special "
            "position gives fewer copies; you are warned if the new atoms overlap existing ones."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addRow(note)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
        self._suggest_label(self.cb_el.currentText())
        if len(picks) >= 2:
            self._from_midpoint()
        elif picks:
            self._from_pick()

    def _suggest_label(self, element):
        from pdf_fitting.project.edited_cif import unique_label

        element = (element or "X").strip() or "X"
        self.le_label.setText(unique_label(self._sites, element))

    def _frac_of(self, record):
        lat = self._ps.structure.lattice
        return np.asarray(lat.get_fractional_coords(np.asarray(record["position"], dtype=float)), dtype=float)

    def _set_pos(self, f):
        for i, letter in enumerate("xyz"):
            self.spins[letter].setValue(float(f[i]))

    def _from_pick(self):
        if self._picks:
            self._set_pos(self._frac_of(self._picks[-1]))

    def _from_midpoint(self):
        if len(self._picks) >= 2:
            self._set_pos(0.5 * (self._frac_of(self._picks[0]) + self._frac_of(self._picks[1])))

    def _accept(self):
        from pdf_fitting.structure_handler import safe_param_key

        label = self.le_label.text().strip()
        element = self.cb_el.currentText().strip()
        if not label or not element:
            QtWidgets.QMessageBox.warning(self, "Add site", "Element and label are required.")
            return
        if safe_param_key(label) in {safe_param_key(s["label"]) for s in self._sites}:
            QtWidgets.QMessageBox.warning(self, "Add site", f"The label {label} is already used.")
            return
        self.accept()

    def result_site(self) -> Dict[str, Any]:
        return {
            "label": self.le_label.text().strip(),
            "element": self.cb_el.currentText().strip(),
            "x": float(self.spins["x"].value()),
            "y": float(self.spins["y"].value()),
            "z": float(self.spins["z"].value()),
            "occ": float(self.sp_occ.value()),
            "biso": float(self.sp_biso.value()),
        }
