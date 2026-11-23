"""
"Peak widths" page of a phase: free-peak (Pawley-like) fit and the
direction-resolved Warren-type σ² analysis, producing starting values for
the conventional refinement (Biso, λ per pair/shell, δ_g or Wilkens ρ / fE)
and DISCO-like estimates of the contrast-factor anisotropy.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from pdf_fitting.gui.plots import MplPlot
from pdf_fitting.project.model import Parameter

from .widgets import Card, SettingsForm, fmt_value

MODE_CHOICES = [("shared", "Shared by all directions"), ("direction", "Refined per direction"), ("fixed", "Fixed (phase value)")]

DEFAULT_W: Dict[str, Any] = {
    "preset": "shared_dyn", "static_model": "direction", "strain_form": "wilkens", "B_mode": "shared",
    "d1_mode": "fixed", "d2_mode": "fixed", "n_lambda": 3, "min_peaks": 2, "fit_Re": False,
    "max_rel_esd": 0.25, "use_merged": True,
}

PRESETS: Dict[str, Any] = {
    "shared_dyn": ("Shared dynamics, independent strain per direction", {
        "static_model": "direction", "B_mode": "shared", "d1_mode": "fixed", "d2_mode": "fixed"}),
    "independent": ("Each direction independent (own B and strain)", {
        "static_model": "direction", "B_mode": "direction", "d1_mode": "fixed", "d2_mode": "fixed"}),
    "invariant": ("Constrained by contrast-factor invariants (E_i)", {
        "static_model": "invariant", "B_mode": "shared", "d1_mode": "fixed", "d2_mode": "fixed"}),
    "model_cf": ("Constrained by the phase contrast factors (ρ, fE)", {
        "static_model": "model_cf", "B_mode": "shared", "d1_mode": "fixed", "d2_mode": "fixed"}),
    "isotropic": ("Isotropic strain (δ_g)", {
        "static_model": "isotropic", "B_mode": "shared", "d1_mode": "fixed", "d2_mode": "fixed"}),
}


class BusySpinner(QtWidgets.QWidget):
    """Small rotating arc shown while a background job runs."""

    def __init__(self, parent=None, size: int = 18):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self):
        self._angle = 0
        self.show()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight), 2.4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        m = 3
        rect = QtCore.QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)
        p.drawArc(rect, -self._angle * 16, 270 * 16)
        p.end()


class _Job(QtCore.QObject):
    status = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def run(self):
        try:
            self.done.emit(self.fn(self.status.emit))
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class PeaksPanel(QtWidgets.QWidget):
    edited = QtCore.Signal()
    navigate = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.win = self.project = self.dataset = self.phase = None
        self.focus = {}
        self._state: Dict[str, Dict[str, Any]] = {}   # per phase key: fitter, result, warren
        self._thread = None
        self._job = None
        self._popout: Optional["DirectionsWindow"] = None
        self._stop_event: Optional[threading.Event] = None
        self._job_t0 = 0.0
        self._job_text = ""
        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._show_elapsed)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Free-peak fit: peak positions come from the structure; each peak group in the r range gets a "
            "free width σ and amplitude (everything else is the full model). The sample widths "
            "σ² − δ_broad²r² are then fitted per direction of the pair vector with "
            "σ² = (B/4π²)(1 − λ) + static(r, direction): starting values for the refinement and an "
            "estimate of the contrast-factor anisotropy."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        lay.addWidget(intro)

        # --- Free-peak fit -------------------------------------------------
        card = Card("1 · Free-peak fit")
        self.fp_settings: Dict[str, Any] = {"r_lo": 2.0, "r_hi": 15.0, "merge_tol": 0.0, "min_weight": 0.02, "max_nfev": 500}
        self.fp_form = SettingsForm([
            ("r_lo", "r min (Å)", "float", (0.5, 200.0, 3), 2.0, "Lower end of the fitted range."),
            ("r_hi", "r max (Å)", "float", (1.0, 200.0, 3), 15.0, "Upper end: longer ranges give more peaks per direction."),
            ("merge_tol", "Merge peaks closer than (Å)", "float", (0.0, 2.0, 4), 0.0, "0 = automatic (half the median peak width)."),
            ("min_weight", "Ignore peaks weaker than (fraction)", "float", (0.0, 0.5, 3), 0.02, "Relative to the strongest peak; they keep the model width."),
            ("max_nfev", "Max function evaluations", "int", (10, 20000), 500, "Each evaluation is one model calculation; the Jacobian is analytic."),
        ])
        card.add(self.fp_form)
        row = QtWidgets.QHBoxLayout()
        self.btn_fp = QtWidgets.QPushButton("Run free-peak fit")
        self.btn_fp.setObjectName("primary")
        self.btn_fp.clicked.connect(self.run_free_peaks)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        self.spinner = BusySpinner()
        self.lbl_fp = QtWidgets.QLabel("")
        self.lbl_fp.setObjectName("muted")
        self.lbl_fp.setWordWrap(True)
        row.addWidget(self.btn_fp)
        row.addWidget(self.btn_stop)
        row.addWidget(self.spinner)
        row.addWidget(self.lbl_fp, 1)
        card.body_layout.addLayout(row)
        self.plot_fit = MplPlot()
        self.plot_fit.setMinimumHeight(320)
        card.add(self.plot_fit)
        self.tbl = QtWidgets.QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(("Use", "#", "r (Å)", "Peak(s): pair ⟨uvw⟩ (k)", "σ (Å)", "± esd", "σ model", "Amplitude", "Flags"))
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.tbl.setMinimumHeight(260)
        card.add(self.tbl)
        lay.addWidget(card)

        # --- Width analysis ----------------------------------------------
        card = Card("2 · Width analysis per direction (dynamic + static)")
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel("Preset"))
        self.cmb_preset = QtWidgets.QComboBox()
        for key, (label, _vals) in PRESETS.items():
            self.cmb_preset.addItem(label, key)
        self.cmb_preset.setToolTip("Sets the constraint options below; they can then be changed one by one.")
        self.cmb_preset.activated.connect(self._apply_preset)
        prow.addWidget(self.cmb_preset, 1)
        card.body_layout.addLayout(prow)
        self.w_form = SettingsForm([
            ("static_model", "Static (microstrain) part", "choice", [
                ("direction", "Independent per direction (S_d free)"),
                ("invariant", "Contrast-factor invariants (Popa / DISCO E_i)"),
                ("model_cf", "Contrast factors of the phase (ρ, fE)"),
                ("isotropic", "Isotropic δ_g·r"),
            ], "direction", "How the static σ² depends on the direction of the pair vector."),
            ("strain_form", "Strain r-dependence", "choice", [
                ("wilkens", "Wilkens f*(r/Re)·r²  (S = ρb²C/4π)"),
                ("gaussian", "Gaussian ε²·r²"),
            ], "wilkens", "Per-direction and invariant models; the phase-contrast model is always Wilkens."),
            ("B_mode", "B (Debye–Waller)", "choice", MODE_CHOICES, "shared", "Per direction = anisotropic B along each direction."),
            ("n_lambda", "Free λ for the first shells", "int", (0, 12), 3, "Per element pair; beyond them λ = δ1/r + δ2/r²."),
            ("d1_mode", "δ1", "choice", MODE_CHOICES, "fixed", ""),
            ("d2_mode", "δ2", "choice", MODE_CHOICES, "fixed", ""),
            ("min_peaks", "Peaks for a direction's own parameters", "int", (1, 20), 2,
             "Directions with fewer peaks share one 'other' set (per-direction parameters only)."),
            ("fit_Re", "Refine Re (Wilkens)", "bool", None, False, "Strongly correlated with ρ; normally fixed."),
            ("max_rel_esd", "Skip peaks with esd/σ above", "float", (0.01, 2.0, 3), 0.25, ""),
            ("use_merged", "Use merged peaks", "bool", None, True, "Merged peaks are modelled as weighted averages of their members."),
        ])
        card.add(self.w_form)
        row = QtWidgets.QHBoxLayout()
        self.btn_w = QtWidgets.QPushButton("Fit widths")
        self.btn_w.setObjectName("primary")
        self.btn_w.clicked.connect(self.run_warren)
        self.btn_pop = QtWidgets.QPushButton("Directions window…")
        self.btn_pop.setToolTip("Pop-out window: pick the directions to inspect.")
        self.btn_pop.clicked.connect(self.open_directions)
        self.btn_apply = QtWidgets.QPushButton("Apply as starting values…")
        self.btn_apply.clicked.connect(self.apply_values)
        self.btn_micro = QtWidgets.QPushButton("Load microstructure model")
        self.btn_micro.setToolTip(
            "Take ρ, fE, Re, the Burgers vector and the contrast factors (cubic A/B or EdgeE*/ScrewE*) from the "
            "Microstructure page, plus Biso / λ from the phase, and fit the widths with the phase contrast factors."
        )
        self.btn_micro.clicked.connect(self.load_micro_model)
        row.addWidget(self.btn_micro)
        row.addWidget(self.btn_w)
        row.addWidget(self.btn_pop)
        row.addWidget(self.btn_apply)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        self.lbl_w = QtWidgets.QLabel("")
        self.lbl_w.setObjectName("muted")
        self.lbl_w.setWordWrap(True)
        card.add(self.lbl_w)
        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(QtWidgets.QLabel("Show"))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.currentIndexChanged.connect(lambda _i: self._draw_inline())
        vrow.addWidget(self.cmb_view, 1)
        card.body_layout.addLayout(vrow)
        self.plot_w = MplPlot()
        self.plot_w.setMinimumHeight(420)
        card.add(self.plot_w)
        self.tbl_w = QtWidgets.QTableWidget(0, 4)
        self.tbl_w.setHorizontalHeaderLabels(("Parameter", "Value", "± esd", "Note"))
        self.tbl_w.verticalHeader().setVisible(False)
        self.tbl_w.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.tbl_w.setMinimumHeight(240)
        card.add(self.tbl_w)
        lay.addWidget(card)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    @property
    def key(self):
        return f"{self.dataset.id}:{self.phase.id}"

    def load(self, win, project, dataset=None, phase=None):
        self.win, self.project, self.dataset, self.phase = win, project, dataset, phase
        if self.key not in self._state:
            self._state[self.key] = self._restore_saved() or {"fp": dict(self.fp_settings), "w": dict(DEFAULT_W)}
        st = self._state[self.key]
        st["fp"]["r_hi"] = min(st["fp"].get("r_hi", 15.0), float(dataset.r_max))
        st["fp"]["r_lo"] = max(st["fp"].get("r_lo", 2.0), float(dataset.r_min))
        self.fp_form.bind(st["fp"])
        for k, v in DEFAULT_W.items():
            st["w"].setdefault(k, v)
        self.w_form.bind(st["w"])
        i = self.cmb_preset.findData(st["w"].get("preset"))
        self.cmb_preset.setCurrentIndex(max(i, 0))
        self._show_free_peaks()
        self._show_warren()

    def _busy(self, on: bool, text: str = ""):
        for b in (self.btn_fp, self.btn_w, self.btn_apply, self.btn_pop, self.btn_micro):
            b.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        if on:
            self.spinner.start()
            self._job_t0 = time.perf_counter()
            self._job_text = text or "Running…"
            self._elapsed_timer.start()
            self._show_elapsed()
        else:
            self.spinner.stop()
            self._elapsed_timer.stop()
        if self.win is not None:
            self.win.set_external_job(on, "Free-peak fit…", stop=self.request_stop)

    def _show_elapsed(self):
        self.lbl_fp.setText(f"{self._job_text} {time.perf_counter() - self._job_t0:.0f} s")

    def _on_job_status(self, msg: str):
        self._job_text = msg
        self._show_elapsed()
        if self.win is not None:
            self.win.statusBar().showMessage(msg)

    def request_stop(self):
        if self._stop_event is not None:
            self._stop_event.set()
            self._job_text = "Stopping…"
            self._show_elapsed()

    def _start(self, fn, on_done):
        if self._thread is not None or (self.win is not None and self.win._thread is not None):
            QtWidgets.QMessageBox.information(self, "Busy", "Another calculation is running.")
            return
        self._thread = QtCore.QThread(self)
        self._job = _Job(fn)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.status.connect(self._on_job_status, Qt.QueuedConnection)
        # Bound slots of this (main-thread) widget: Qt queues them to the GUI
        # thread. Plain functions / lambdas would run in the worker thread
        # and touch widgets from there (crash).
        self._on_done = on_done
        self._job.done.connect(self._job_done, Qt.QueuedConnection)
        self._job.failed.connect(self._job_failed, Qt.QueuedConnection)
        for sig in (self._job.done, self._job.failed):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(self._job_finished)
        self._busy(True, "Preparing…")
        self._thread.start()

    @QtCore.Slot(object)
    def _job_done(self, out):
        self._elapsed_timer.stop()
        cb, self._on_done = getattr(self, "_on_done", None), None
        if cb is not None:
            cb(out)

    @QtCore.Slot(str)
    def _job_failed(self, text):
        self._elapsed_timer.stop()
        self._on_done = None
        print(text)
        QtWidgets.QMessageBox.critical(self, "Free-peak analysis failed", text[-3000:])

    def _job_finished(self):
        self._thread.deleteLater()
        self._thread = None
        self._job = None
        self._busy(False)
        st = self._state.get(self.key, {})
        self.btn_apply.setEnabled(st.get("warren") is not None)
        self.btn_pop.setEnabled(st.get("warren") is not None)

    # ------------------------------------------------------------------
    # Free-peak fit
    # ------------------------------------------------------------------
    def run_free_peaks(self):
        from pdf_fitting.project.free_peaks import FreePeakFitter

        st = self._state[self.key]
        cfg = dict(st["fp"])
        project, ds, ph, key = self.project, self.dataset, self.phase, self.key

        from pdf_fitting.project.free_peaks import FreePeakStopped
        from pdf_fitting.project.warren_fit import start_from_phase

        self._stop_event = stop_event = threading.Event()

        def job(status):
            try:
                fitter = FreePeakFitter(
                    project, ds, ph, r_lo=cfg["r_lo"], r_hi=cfg["r_hi"],
                    merge_tol=cfg["merge_tol"] or None, min_weight=cfg["min_weight"], status=status,
                    stop_event=stop_event,
                )
                status(f"Free-peak fit of {len(fitter.clusters)} peak group(s)…")
                result = fitter.fit(max_nfev=int(cfg["max_nfev"]))
                start = start_from_phase(fitter.calc, fitter.params, result.clusters)
            except FreePeakStopped:
                return key, None, None, None
            return key, fitter, result, start

        def done(out):
            k, fitter, result, start = out
            if result is None:
                self.win.statusBar().showMessage("Free-peak fit stopped; previous results kept.", 8000)
                self._show_free_peaks()
                return
            s = self._state[k]
            s["fitter"], s["result"], s["warren"], s["start"] = fitter, result, None, start
            s["exclude"] = set()
            self._persist(k)
            if k == self.key:
                self._show_free_peaks()
                self._show_warren()
            self.win.statusBar().showMessage(f"Free-peak fit: Rwp {result.rwp_start:.2f} → {result.rwp:.2f} % in {result.elapsed_s:.1f} s", 8000)

        self._start(job, done)

    def _show_free_peaks(self):
        st = self._state.get(self.key, {})
        res = st.get("result")
        ax = self.plot_fit.ax
        ax.clear()
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        if res is None:
            self.lbl_fp.setText("Not run yet.")
            ax.text(0.5, 0.5, "Run the free-peak fit", transform=ax.transAxes, ha="center", va="center", color="#6b7280")
            self.plot_fit.canvas.draw_idle()
            self.tbl.blockSignals(False)
            return
        n_ok = sum(c.reliable for c in res.clusters)
        self.lbl_fp.setText(
            f"{len(res.clusters)} peak group(s), {n_ok} well determined · merge tolerance {res.merge_tol:.3f} Å · "
            f"Rwp {res.rwp_start:.2f} → {res.rwp:.2f} % · {res.nfev} evaluations, {res.elapsed_s:.0f} s · {res.message}"
            + (" · loaded from the project file" if st.get("fitter") is None else "")
        )
        ax.plot(res.r, res.G_exp, color="#111827", lw=0.9, label="Observed")
        ax.plot(res.r, res.G_start, color="#9ca3af", lw=0.9, ls="--", label="Model (start)")
        ax.plot(res.r, res.G_fit, color="#dc2626", lw=1.0, label="Free-peak fit")
        off = float(np.min(res.G_exp)) - 0.3 * float(np.ptp(res.G_exp))
        ax.plot(res.r, res.G_exp - res.G_fit + off, color="#15803d", lw=0.8, label="Difference")
        for c in res.clusters:
            ax.axvline(c.r, color="#c2410c" if c.merged else "#1d4ed8", lw=0.5, alpha=0.35)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("G(r)")
        ax.legend(fontsize=8, ncols=4, loc="upper right")
        self.plot_fit.canvas.draw_idle()

        excl = st.setdefault("exclude", set())
        self.tbl.setRowCount(len(res.clusters))
        for i, c in enumerate(res.clusters):
            use = QtWidgets.QTableWidgetItem("")
            use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            use.setCheckState(Qt.Unchecked if c.index in excl else Qt.Checked)
            self.tbl.setItem(i, 0, use)
            flags = []
            if c.merged:
                flags.append("merged")
            if not c.reliable:
                flags.append("unreliable")
            vals = (str(c.index), f"{c.r:.4f}", c.label, f"{c.sigma:.4f}", f"{c.sigma_esd:.4f}", f"{c.sigma0:.4f}", f"{c.amp:.3f} ± {c.amp_esd:.3f}", ", ".join(flags))
            for j, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if not c.reliable:
                    it.setForeground(QtGui.QColor("#9ca3af"))
                self.tbl.setItem(i, j, it)
        self.tbl.blockSignals(False)
        try:
            self.tbl.itemChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.tbl.itemChanged.connect(self._use_changed)

    def _use_changed(self, item):
        if item.column() != 0:
            return
        st = self._state[self.key]
        idx = st["result"].clusters[item.row()].index
        if item.checkState() == Qt.Checked:
            st["exclude"].discard(idx)
        else:
            st["exclude"].add(idx)
        self._persist(self.key)

    # ------------------------------------------------------------------
    # Width analysis
    # ------------------------------------------------------------------
    def _apply_preset(self, _i=None):
        key = self.cmb_preset.currentData()
        st = self._state.get(self.key)
        if st is None or key not in PRESETS:
            return
        st["w"].update(PRESETS[key][1])
        st["w"]["preset"] = key
        self.w_form.bind(st["w"])

    def run_warren(self):
        from pdf_fitting.project.warren_fit import WarrenOptions, fit_warren, start_from_phase

        st = self._state.get(self.key, {})
        res = st.get("result")
        if res is None:
            QtWidgets.QMessageBox.information(self, "Width analysis", "Run the free-peak fit first.")
            return
        w = st["w"]
        opt = WarrenOptions(
            static_model=w["static_model"], strain_form=w["strain_form"], B_mode=w["B_mode"],
            d1_mode=w["d1_mode"], d2_mode=w["d2_mode"], n_lambda=int(w["n_lambda"]),
            fit_Re=bool(w["fit_Re"]), min_peaks=int(w["min_peaks"]),
            max_rel_esd=float(w["max_rel_esd"]), use_merged=bool(w["use_merged"]),
            exclude=sorted(st.get("exclude", set())),
        )
        try:
            start = st.get("start")
            if start is None:
                start = st["start"] = start_from_phase(st["fitter"].calc, st["fitter"].params, res.clusters)
            st["warren"] = fit_warren(res.clusters, elements=res.elements, start=start, options=opt)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Width analysis", str(exc))
            return
        self._persist(self.key)
        self._show_warren()

    # ------------------------------------------------------------------
    # Saving in the project file (phase.sections["free_peaks"][dataset id])
    # ------------------------------------------------------------------
    def _persist(self, key):
        from pdf_fitting.project.warren_fit import start_to_json

        ds_id, ph_id = key.split(":", 1)
        try:
            phase = self.project.dataset(ds_id).phase(ph_id)
        except Exception:  # noqa: BLE001
            phase = self.phase if key == self.key else None
        st = self._state.get(key)
        if phase is None or st is None or st.get("result") is None:
            return
        try:
            entry = {
                "fp": dict(st["fp"]),
                "w": dict(st["w"]),
                "exclude": sorted(int(i) for i in st.get("exclude", set())),
                "result": st["result"].to_json(),
                "start": start_to_json(st["start"]) if st.get("start") is not None else None,
                "warren_done": st.get("warren") is not None,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[PEAKS] could not save the free-peak result: {exc}")
            return
        phase.sections.setdefault("free_peaks", {})[ds_id] = _json_safe(entry)
        if self.win is not None:
            self.win.dirty = True
            try:
                self.win._update_title()
            except Exception:  # noqa: BLE001
                pass

    def _restore_saved(self):
        from pdf_fitting.project.free_peaks import FreePeakResult
        from pdf_fitting.project.warren_fit import WarrenOptions, fit_warren, start_from_json

        saved = ((self.phase.sections or {}).get("free_peaks") or {}).get(self.dataset.id)
        if not saved or not saved.get("result"):
            return None
        try:
            st = {
                "fp": {**self.fp_settings, **(saved.get("fp") or {})},
                "w": {**DEFAULT_W, **(saved.get("w") or {})},
                "exclude": set(int(i) for i in saved.get("exclude") or []),
                "result": FreePeakResult.from_json(saved["result"]),
                "fitter": None,
                "start": start_from_json(saved["start"]) if saved.get("start") else None,
                "warren": None,
            }
            if saved.get("warren_done") and st["start"] is not None:
                w = st["w"]
                opt = WarrenOptions(
                    static_model=w["static_model"], strain_form=w["strain_form"], B_mode=w["B_mode"],
                    d1_mode=w["d1_mode"], d2_mode=w["d2_mode"], n_lambda=int(w["n_lambda"]),
                    fit_Re=bool(w["fit_Re"]), min_peaks=int(w["min_peaks"]),
                    max_rel_esd=float(w["max_rel_esd"]), use_merged=bool(w["use_merged"]),
                    exclude=sorted(st["exclude"]),
                )
                try:
                    st["warren"] = fit_warren(st["result"].clusters, elements=st["result"].elements, start=st["start"], options=opt)
                except Exception as exc:  # noqa: BLE001
                    print(f"[PEAKS] saved width analysis could not be redone: {exc}")
            return st
        except Exception as exc:  # noqa: BLE001
            print(f"[PEAKS] saved free-peak result could not be read: {exc}")
            return None

    def load_micro_model(self):
        from pdf_fitting.project.contrast_factors import has_cubic, has_family
        from pdf_fitting.project.warren_fit import refresh_start_from_params, start_from_phase

        st = self._state.get(self.key, {})
        res = st.get("result")
        if res is None:
            QtWidgets.QMessageBox.information(self, "Width analysis", "Run the free-peak fit first.")
            return
        if st.get("start") is None:
            if st.get("fitter") is None:
                QtWidgets.QMessageBox.information(self, "Width analysis", "Run the free-peak fit again (no saved model data).")
                return
            st["start"] = start_from_phase(st["fitter"].calc, st["fitter"].params, res.clusters)
        params = {k: p.value for k, p in self.dataset.parameters.items()}
        params.update({k: p.value for k, p in self.phase.parameters.items()})
        st["start"] = refresh_start_from_params(st["start"], params, res.clusters)
        sd = st["start"]
        kind = "cubic A/B" if has_cubic(params) else ("invariants E_i" if (has_family(params, "edge") or has_family(params, "screw")) else "none")
        msg = (f"Microstructure model loaded: ρ = {sd.get('rho', 0):.4g} Å⁻², fE = {sd.get('fE', 0):.3f}, Re = {sd.get('Re', 0):.4g} Å, "
               f"b = {sd.get('burgers', 0):.4g} Å, contrast factors: {kind}.")
        if kind != "none" and sd.get("burgers", 0) > 0:
            st["w"].update(static_model="model_cf", preset="model_cf")
            self.w_form.bind(st["w"])
            i = self.cmb_preset.findData("model_cf")
            self.cmb_preset.setCurrentIndex(max(i, 0))
            self.run_warren()
        else:
            msg += " No contrast factors set: the width model was not changed."
            self._persist(self.key)
        self.win.statusBar().showMessage(msg, 12000)
        self.lbl_w.setText(msg + ("  ·  " + self.lbl_w.text() if self.lbl_w.text() else ""))

    def _view_items(self, wr):
        from .direction_plots import family_title

        items = [("Overview: all directions", "overview")]
        if wr is not None and wr.options.static_model != "isotropic":
            items.append(("Contrast factors (S_d vs direction)", "contrast"))
        if wr is not None:
            for fam in wr.family_order():
                info = wr.family_info[fam]
                extra = "" if info.own else " · shared"
                items.append((f"{family_title(wr, fam)} · {len(info.clusters)} peak(s){extra}", fam))
        return items

    def _draw_inline(self):
        from .direction_plots import plot_contrast, plot_directions, plot_overview

        wr = self._state.get(self.key, {}).get("warren") if self.phase is not None else None
        what = self.cmb_view.currentData()
        fig = self.plot_w.fig
        try:
            if what == "contrast":
                plot_contrast(fig, wr)
            elif what in (None, "overview"):
                plot_overview(fig, wr)
            else:
                plot_directions(fig, wr, [what])
        except Exception:  # noqa: BLE001
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, traceback.format_exc()[-400:], transform=ax.transAxes, ha="center", va="center", fontsize=7)
        if fig.axes:
            self.plot_w.ax = fig.axes[0]
        self.plot_w.canvas.draw_idle()

    def _show_warren(self):
        from pdf_fitting.project.warren_fit import hkl_label, starting_values

        st = self._state.get(self.key, {})
        wr = st.get("warren")
        self.tbl_w.setRowCount(0)
        self.btn_apply.setEnabled(wr is not None)
        self.btn_pop.setEnabled(wr is not None)
        cur = self.cmb_view.currentData()
        self.cmb_view.blockSignals(True)
        self.cmb_view.clear()
        for text, data in self._view_items(wr):
            self.cmb_view.addItem(text, data)
        i = self.cmb_view.findData(cur)
        self.cmb_view.setCurrentIndex(max(i, 0))
        self.cmb_view.blockSignals(False)
        self._draw_inline()
        if self._popout is not None:
            self._popout.set_result(wr, f"{self.phase.name} · {self.dataset.name}")
        if wr is None:
            self.lbl_w.setText("")
            return
        big = [
            f"{wr.names[i]}/{wr.names[j]} {wr.correlation[i, j]:+.2f}"
            for i in range(len(wr.names)) for j in range(i)
            if np.isfinite(wr.correlation[i, j]) and abs(wr.correlation[i, j]) > 0.9
        ]
        text = f"{len(wr.used)} peaks, {len(wr.names)} parameters · reduced χ² {wr.chi2_red:.3g} (esds scaled by it)"
        if big:
            text += " · strong correlations: " + ", ".join(big[:8]) + (" …" if len(big) > 8 else "")
        for wmsg in wr.warnings:
            text += " · ⚠ " + wmsg
        self.lbl_w.setText(text)

        rows: List[tuple] = []
        for n, v, e in wr.table():
            rows.append((n, f"{v:.6g}", f"{e:.3g}", ""))
        if wr.options.static_model != "isotropic":
            rows.append(("— per direction —", "", "", "S_d = ρb²C_d/4π (Wilkens) or ε² (Gaussian)"))
            for fam in wr.family_order():
                info = wr.family_info[fam]
                note = hkl_label(info.hkl)
                if wr.cf_axis == "H":
                    note += f" · H = {info.x_cf:.3f}"
                elif wr.cf_axis == "delta":
                    note += f" · δ = {info.x_cf:.1f}°"
                if np.isfinite(info.rho):
                    note += f" · ρ (model C) = {info.rho:.3g} ± {info.rho_esd:.2g}"
                if not info.own and wr.options.static_model == "direction":
                    note += " · shares 'other'"
                rows.append((f"S {fam}", f"{info.S:.4g}", f"{info.S_esd:.2g}", note))
        if wr.inv_names and len(wr.inv_values):
            rows.append(("— invariants —", "", "", wr.inv_source))
            for n, v, e in zip(wr.inv_names, wr.inv_values, wr.inv_esd):
                rows.append((f"Ẽ{n[1:]} = ρb²E{n[1:]}/4π", f"{v:.4g}", f"{e:.2g}",
                             f"E{n[1:]}/E1 = {v / wr.inv_values[0]:.3f}" if wr.inv_values[0] else ""))
        cf = wr.cf_fit
        if cf.get("rho") is not None:
            rows.append(("— contrast factors of the phase —", "", "", cf.get("source", "")))
            rows.append(("ρ (Å⁻²)", f"{cf['rho']:.4g}", f"{cf.get('rho_esd', float('nan')):.2g}", ""))
            rows.append(("fE", f"{cf.get('fE', float('nan')):.3f}", f"{cf.get('fE_esd', float('nan')):.2g}",
                         "" if np.isfinite(cf.get("fE_esd", float("nan"))) else "kept (not refinable here)"))
        sv = starting_values(wr)
        rows.append(("— suggested starting values —", "", "", "applied with 'Apply as starting values…'"))
        for n, v, note, on in sv:
            rows.append((n, f"{v:.6g}", "", note + ("" if on else " (optional)")))
        self.tbl_w.setRowCount(len(rows))
        bold = QtGui.QFont()
        bold.setBold(True)
        for r, vals in enumerate(rows):
            for j, t in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(t)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if vals[0].startswith("—"):
                    it.setFont(bold)
                self.tbl_w.setItem(r, j, it)

    def open_directions(self):
        st = self._state.get(self.key, {})
        wr = st.get("warren")
        if self._popout is None:
            self._popout = DirectionsWindow(None)
            self._popout.destroyed.connect(lambda *_: setattr(self, "_popout", None))
        self._popout.set_result(wr, f"{self.phase.name} · {self.dataset.name}")
        self._popout.show()
        self._popout.raise_()
        self._popout.activateWindow()

    def apply_values(self):
        from pdf_fitting.project.warren_fit import starting_values

        st = self._state.get(self.key, {})
        wr = st.get("warren")
        if wr is None:
            return
        params = self.phase.parameters
        lc = {k.lower(): k for k in params}
        changes = []
        for name, value, note, on in starting_values(wr):
            if not np.isfinite(value):
                continue
            if name.startswith("biso_"):
                el = name[5:]
                targets = [k for k in params if k.lower() == name.lower()]
                if not targets:
                    # Site Biso: every site of that element.
                    targets = [k for k in params if k.lower().startswith("biso_") and k[5:].rstrip("0123456789").lower() == el.lower()]
                if not targets:
                    targets = [name]
            else:
                targets = [lc.get(name.lower(), name)]
            for t in targets:
                old = params[t].value if t in params else None
                changes.append((t, old, float(value), note, on))
        dlg = ApplyDialog(self, self.phase.name, changes)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = dlg.selected()
        for t, _o, v, _n, _on in chosen:
            params.setdefault(t, Parameter(value=float(v))).value = float(v)
        if chosen:
            self.edited.emit()
        self.win.statusBar().showMessage(f"Applied {len(chosen)} starting value(s) to {self.phase.name}.", 8000)


# -----------------------------------------------------------------------------
# Pop-out window and apply dialog
# -----------------------------------------------------------------------------


class DirectionsWindow(QtWidgets.QWidget):
    """Pick one or more directions (or the overview / contrast plot) to inspect."""

    MAX_ROWS = 6

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Peak widths per direction")
        self.resize(1300, 850)
        self.wr = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setObjectName("muted")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        split = QtWidgets.QSplitter(Qt.Horizontal)
        lay.addWidget(split, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel("Select what to show (Ctrl / Shift for several directions):")
        hint.setWordWrap(True)
        ll.addWidget(hint)
        self.lst = QtWidgets.QListWidget()
        self.lst.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.lst.itemSelectionChanged.connect(self._redraw)
        ll.addWidget(self.lst, 1)
        split.addWidget(left)
        right = QtWidgets.QSplitter(Qt.Vertical)
        self.plot = MplPlot()
        right.addWidget(self.plot)
        self.tbl = QtWidgets.QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(("Direction", "r (Å)", "k", "σ²_sample", "± esd", "model", "dynamic", "static", "(obs − model)/esd"))
        self.tbl.verticalHeader().setVisible(False)
        right.addWidget(self.tbl)
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 5)

    def set_result(self, wr, title: str = ""):
        from .direction_plots import family_title

        keep = {it.data(Qt.UserRole) for it in self.lst.selectedItems()}
        self.wr = wr
        self.setWindowTitle(f"Peak widths per direction — {title}" if title else "Peak widths per direction")
        self.lst.blockSignals(True)
        self.lst.clear()
        entries = [("Overview: all directions", "overview")]
        if wr is not None and wr.options.static_model != "isotropic":
            entries.append(("Contrast factors", "contrast"))
        if wr is not None:
            for fam in wr.family_order():
                info = wr.family_info[fam]
                entries.append((f"{family_title(wr, fam)} · {len(info.clusters)} peak(s)" + ("" if info.own else " · shared"), fam))
        for text, data in entries:
            it = QtWidgets.QListWidgetItem(text)
            it.setData(Qt.UserRole, data)
            self.lst.addItem(it)
            if data in keep:
                it.setSelected(True)
        if not self.lst.selectedItems() and self.lst.count():
            self.lst.item(0).setSelected(True)
        self.lst.blockSignals(False)
        if wr is not None:
            o = wr.options
            self.lbl.setText(
                f"{len(wr.used)} peaks · reduced χ² {wr.chi2_red:.3g} · static: {o.static_model} ({o.strain_form}) · "
                f"B {o.B_mode} · δ1 {o.d1_mode} · δ2 {o.d2_mode} · σ² shown without the instrumental part"
            )
        else:
            self.lbl.setText("No width analysis yet.")
        self._redraw()

    def _redraw(self):
        from .direction_plots import plot_contrast, plot_directions, plot_overview

        sel = [it.data(Qt.UserRole) for it in self.lst.selectedItems()]
        fig = self.plot.fig
        fams = [s for s in sel if s not in ("overview", "contrast")][: self.MAX_ROWS]
        try:
            if fams:
                plot_directions(fig, self.wr, fams)
            elif "contrast" in sel:
                plot_contrast(fig, self.wr)
            else:
                plot_overview(fig, self.wr)
        except Exception:  # noqa: BLE001
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, traceback.format_exc()[-400:], transform=ax.transAxes, ha="center", va="center", fontsize=7)
        if fig.axes:
            self.plot.ax = fig.axes[0]
        self.plot.canvas.draw_idle()
        self._fill_table(fams or (self.wr.family_order() if self.wr is not None else []))

    def _fill_table(self, fams):
        wr = self.wr
        self.tbl.setRowCount(0)
        if wr is None:
            return
        rows = []
        for fam in fams:
            for i in sorted(wr.family_info[fam].clusters, key=lambda j: wr.r[j]):
                ks = sorted({m["k"] for m in wr.members[i] if m["family"] == fam})
                o, e, mdl = wr.observed_sample[i], wr.observed_esd[i], wr.predicted_sample[i]
                lbl = fam + (" (merged)" if len(wr.cluster_families[i]) > 1 else "")
                rows.append((lbl, f"{wr.r[i]:.4f}", ",".join(map(str, ks)), f"{o:.5g}", f"{e:.2g}", f"{mdl:.5g}",
                             f"{wr.thermal[i]:.5g}", f"{wr.static[i]:.5g}", f"{(o - mdl) / e:+.2f}"))
        self.tbl.setRowCount(len(rows))
        for r, vals in enumerate(rows):
            for j, t in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(t)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl.setItem(r, j, it)
        self.tbl.resizeColumnsToContents()


class ApplyDialog(QtWidgets.QDialog):
    """Checklist of the suggested parameter values."""

    def __init__(self, parent, phase_name: str, changes):
        super().__init__(parent)
        self.setWindowTitle("Apply starting values")
        self.resize(780, 460)
        self.changes = changes
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(f"Set the ticked parameters of {phase_name}:"))
        self.tbl = QtWidgets.QTableWidget(len(changes), 4)
        self.tbl.setHorizontalHeaderLabels(("Parameter", "Current", "New", "Note"))
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        for r, (t, old, new, note, on) in enumerate(changes):
            it = QtWidgets.QTableWidgetItem(t)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.tbl.setItem(r, 0, it)
            for j, text in enumerate((fmt_value(old) if old is not None else "(new)", f"{new:.6g}", note), start=1):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setFlags(Qt.ItemIsEnabled)
                self.tbl.setItem(r, j, cell)
        self.tbl.resizeColumnToContents(0)
        lay.addWidget(self.tbl, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected(self):
        return [c for r, c in enumerate(self.changes) if self.tbl.item(r, 0).checkState() == Qt.Checked]


def _json_safe(obj):
    """NaN/inf → None, numpy scalars → Python (the project file is strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
