"""
Background calculation / refinement of a whole Project.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
from PySide6 import QtCore

from pdf_fitting.project import Project


def _rwp(G_exp, G_fit) -> float:
    G_exp = np.asarray(G_exp, float)
    den = float(np.sum(G_exp * G_exp))
    if den <= 0:
        return float("nan")
    return 100.0 * float(np.sqrt(np.sum((G_exp - np.asarray(G_fit, float)) ** 2) / den))


def _phase_extras(pe, local: Dict[str, Any], r, settings: Dict[str, Any], status=None) -> Dict[str, Any]:
    """Classic per-phase outputs: shape factor, size distribution, Warren, local trends."""
    from .legacy_bridge import size_distribution

    model = pe.model
    calc = getattr(model, "pdf_calculator", None) or model
    out: Dict[str, Any] = {}
    try:
        out["gamma_avg"] = np.asarray(calc.compute_isotropic_shape_factor(local, r), float)
    except Exception:
        out["gamma_avg"] = None
    out["size_distribution"] = size_distribution(local)
    if settings.get("compute_warren_on_finish"):
        if status:
            status(f"Warren plot: {pe.phase.name}…")
        try:
            max_L = calc._plot_distance_limit_from_params(local, fallback=5.0)
            w = calc.build_warren_plot(local, max_L=max_L)
            if isinstance(w, dict) and all(isinstance(k, tuple) and len(k) == 3 for k in w):
                out["warren_by_dir"] = w
        except Exception:
            pass
    if settings.get("compute_local_trends_on_finish"):
        if status:
            status(f"Local trends: {pe.phase.name}…")
        try:
            out["local_trends"] = calc.build_local_trends(local, max_r_plot=10.0)
        except Exception:
            pass
    return out


def summarize_result(pm, params: Dict[str, Any], settings: Optional[Dict[str, Any]] = None, status=None) -> Dict[str, Any]:
    """Full-grid evaluation with phase contributions and classic outputs, per dataset."""
    settings = settings or {}
    out = pm.evaluate(params, return_contributions=True)
    datasets = []
    for e, d in zip(pm.datasets, out["datasets"]):
        phases = {}
        for pe in e.phases:
            res = d["phases"].get(pe.phase.id)
            if res is None:
                continue
            local = pe.local_params(params)
            phases[pe.phase.id] = {
                "G_r": np.asarray(res.get("G_r"), float),
                "G_ab": {k: np.asarray(v, float) for k, v in (res.get("G_ab") or {}).items()},
                "G_contrib": {k: np.asarray(v, float) for k, v in (res.get("G_contrib") or {}).items()},
                "weights": dict(res.get("weights") or {}),
                "local_params": dict(local),
                **_phase_extras(pe, local, e.r_exp, settings, status),
            }
        datasets.append(
            {
                "id": e.dataset.id,
                "r": np.asarray(e.r_exp, float),
                "G_exp": np.asarray(e.G_exp, float),
                "G_fit": np.asarray(d["G_r"], float),
                "rwp": _rwp(e.G_exp, d["G_r"]),
                "phases": phases,
                "fractions": d.get("fractions", {}),
            }
        )
    return {
        "rwp": _rwp(pm.G_exp, out["G_r"]),
        "datasets": datasets,
        "params": dict(params),
    }


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class ProjectWorker(QtCore.QObject):
    """mode "calculate" or "refine". Works on a deep copy of the project."""

    status = QtCore.Signal(str)
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, project: Project, mode: str = "refine", parent=None):
        super().__init__(parent)
        self.project = copy.deepcopy(project)
        self.project.path = project.path
        self.mode = mode
        self.stop_event = threading.Event()
        self._best: Optional[Dict[str, Any]] = None
        self._last_curves = 0.0

    def _uncertainty(self, engine, pm, params):
        """esds + correlations at the engine's current values (None when stopped / failed)."""
        from pdf_fitting.uncertainty import EsdStopped, compute_uncertainties, nyquist_points

        n_eff = None
        if _truthy(self.project.refinement.get("esd_nyquist_correction", False)):
            spans = []
            for e in pm.datasets:
                r = np.asarray(e.r_exp, float)
                key = f"{e.dataset.id}.qmax"
                qmax = params.get(key, e.dataset.parameters["qmax"].value if "qmax" in e.dataset.parameters else 0.0)
                try:
                    spans.append((float(r.max() - r.min()), float(qmax)))
                except (TypeError, ValueError):
                    pass
            n_eff = nyquist_points(spans)
        try:
            self.status.emit("Computing esds and correlations…")
            res = compute_uncertainties(engine, n_eff=n_eff, status=self.status.emit, stop_event=self.stop_event)
            return res.to_json()
        except EsdStopped:
            return {"message": "esd calculation stopped."}
        except Exception as exc:  # noqa: BLE001
            return {"message": f"esd calculation failed: {exc}"}

    def stop(self):
        self.stop_event.set()

    @QtCore.Slot()
    def run(self):
        try:
            from pdf_fitting.project.multiphase import ProjectModel

            t0 = time.perf_counter()
            pm = ProjectModel(
                self.project,
                status_callback=self.status.emit,
                stop_event=self.stop_event,
            )
            initial, refinable, bounds = self.project.flat_tables()

            if self.mode == "calculate":
                self.status.emit("Calculating G(r)…")
                summary = summarize_result(pm, initial, self.project.refinement, self.status.emit)
                summary.update(mode="calculate", elapsed_s=time.perf_counter() - t0)
                self.finished.emit(summary)
                return

            from pdf_fitting.fit_engine import FitEngine, FitStopped

            config = {
                "initial": dict(initial),
                "refinable": dict(refinable),
                "bounds": dict(bounds),
                # Every evaluation reports progress; the window throttles redraws.
                "refinement": {
                    "progress_every_nfev": 1,
                    **self.project.refinement,
                    # Curves for the live fit plot (sent at most every 0.5 s).
                    "update_pdf_during_refinement": True,
                },
                "constraints": dict(self.project.constraints),
                "r_max": pm.r_max,
            }

            def on_progress(payload):
                params = payload.get("params") or {}
                fractions = {}
                for e in pm.datasets:
                    try:
                        fractions[e.dataset.id] = pm.phase_fractions(params, e)
                    except Exception:
                        pass
                small = {
                    "nfev": int(payload.get("nfev", 0)),
                    "stage": payload.get("stage", ""),
                    "rwp": float(payload.get("rwp", np.nan)),
                    "best_rwp": float(payload.get("best_rwp", np.nan)),
                    "elapsed_s": float(payload.get("elapsed_s", 0.0)),
                    "fractions": fractions,
                    "params": dict(params),
                }
                now = time.perf_counter()
                if "r" in payload and "G_fit" in payload and now - self._last_curves >= 0.5:
                    self._last_curves = now
                    curves = {}
                    t = np.asarray(payload["r"], float)
                    g_exp = np.asarray(payload["G_exp"], float)
                    g_fit = np.asarray(payload["G_fit"], float)
                    for e, mask, r_real in pm.split_virtual(t):
                        curves[e.dataset.id] = (r_real, g_exp[mask], g_fit[mask])
                    small["curves"] = curves
                cost = float(payload.get("cost", np.inf))
                if self._best is None or cost < self._best["cost"]:
                    self._best = {"cost": cost, "params": dict(params)}
                self.progress.emit(small)

            engine = FitEngine(
                pm.r_exp,
                pm.G_exp,
                pm,
                config,
                progress_callback=on_progress,
                stop_event=self.stop_event,
            )
            if self.mode == "esd":
                # esds at the current values, no refinement.
                summary = summarize_result(pm, initial, self.project.refinement, self.status.emit)
                summary.update(mode="esd", refinable=dict(config["refinable"]), elapsed_s=time.perf_counter() - t0)
                summary["uncertainty"] = self._uncertainty(engine, pm, initial)
                self.finished.emit(summary)
                return

            self.status.emit("Running refinement…")
            stopped = False
            try:
                result = engine.run_refinement()
                final = result if isinstance(result, dict) else dict(getattr(result, "params", {}) or {})
            except FitStopped:
                stopped = True
                final = dict((self._best or {}).get("params") or config["initial"])

            self.status.emit("Evaluating final model…")
            summary = summarize_result(pm, final, self.project.refinement, self.status.emit)
            summary.update(
                mode="refine",
                stopped=stopped,
                refinable=dict(config["refinable"]),
                elapsed_s=time.perf_counter() - t0,
            )
            if not stopped and _truthy(self.project.refinement.get("compute_esd", False)):
                summary["uncertainty"] = self._uncertainty(engine, pm, final)
            self.finished.emit(summary)

        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
