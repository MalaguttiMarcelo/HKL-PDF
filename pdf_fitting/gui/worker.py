from __future__ import annotations

import time
import traceback
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PySide6 import QtCore

from pdf_fitting.io_handler import read_input_file
from pdf_fitting.fit_engine import FitEngine, FitStopped
from pdf_fitting.models.gr_model import PDFCalculator, lattice_matrix, apply_lattice_constraints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _cfg_bool(val, default: bool = False) -> bool:
    """
    Robust bool parser.

    Important:
        bool("false") is True in Python, so do not use bool(raw_string).
    """
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


class FitWorker(QtCore.QThread):
    """Runs the refinement in a background thread and emits progress payloads."""

    progress = QtCore.Signal(dict)        # incremental updates
    finished_ok = QtCore.Signal(dict)     # final results
    failed = QtCore.Signal(str)           # error text
    status = QtCore.Signal(str)           # status messages

    def __init__(
        self,
        *,
        input_path: str,
        source_path: Optional[str] = None,
        update_pdf: bool,
        fullgrid_rwp: bool,
        progress_every_sec: float = 0.5,
        activate_tick_markers: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        

        self.input_path = input_path
        self.source_path = source_path

        self.update_pdf = bool(update_pdf)
        self.fullgrid_rwp = bool(fullgrid_rwp)
        self.progress_every_sec = float(progress_every_sec)
        self.activate_tick_markers = bool(activate_tick_markers)


        # For FitEngine pause/stop interface.
        import threading
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self._save_on_stop = False
        self._best_cost = float("inf")
        self._best_params = None

    @property
    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def request_stop(self, *, save_best: bool = False):
        self._save_on_stop = bool(save_best)
        self.stop_event.set()
        self.status.emit("Stop requested…")

    def pause(self):
        self.pause_event.set()
        self.status.emit("Pausing…")

    def resume(self):
        self.pause_event.clear()
        self.status.emit("Resuming…")

    def _build_local_trends(
        self,
        pdf_model,
        params: Dict[str, Any],
        *,
        max_r_plot: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Build local trends using PDFCalculator.build_local_trends().

        Default is intentionally limited to 10 Å so finalization after refinement
        does not spend time scanning long-range shell tables.
        """
        try:
            return pdf_model.build_local_trends(
                params or {},
                max_r_plot=float(max_r_plot),
            )
        except Exception:
            return {}

    def _build_size_distribution(self, params: Dict[str, Any]):
        """
        Build crystallite size distribution.

        Cheap diagnostic. Keep it always enabled.
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

        return None

    def run(self):
        t0 = time.perf_counter()

        # These are initialized here so the FitStopped handler can use them safely.
        config = None
        pdf_model = None
        r_exp = None
        G_exp = None

        try:
            self.status.emit("Loading configuration…")
            config = read_input_file(self.input_path)

            if "refinement" not in config or config["refinement"] is None:
                config["refinement"] = {}

            refcfg = config.get("refinement", {}) or {}

            # Configure FitEngine live-update rate and whether to include PDF arrays in progress payload.
            config["refinement"]["progress_every_sec"] = self.progress_every_sec
            config["refinement"]["progress_every_nfev"] = 0
            config["refinement"]["update_pdf_during_refinement"] = bool(self.update_pdf)
            config["refinement"]["fullgrid_rwp"] = bool(self.fullgrid_rwp)

            # Optional final-output controls.
            compute_pair_contrib = _cfg_bool(
                refcfg.get("compute_pair_contributions_on_finish", False),
                False,
            )

            compute_warren_on_finish = _cfg_bool(
                refcfg.get("compute_warren_on_finish", False),
                False,
            )

            compute_local_trends_on_finish = _cfg_bool(
                refcfg.get("compute_local_trends_on_finish", False),
                False,
            )

            # Load experimental data.
            self.status.emit("Loading experimental data…")
            r_exp, G_exp = np.loadtxt(config["gr_data_file"], unpack=True)

            r_min = float(config.get("r_min", config.get("range", {}).get("r_min", 1.0)))
            r_max = float(config.get("r_max", config.get("range", {}).get("r_max", float(np.max(r_exp)))))
            r_extension = float(config.get("r_extension", config.get("pair_generation", {}).get("r_extension", 1.0)))
            pair_cutoff = r_max * r_extension

            mask = (r_exp >= r_min) & (r_exp <= r_max)
            r_exp = r_exp[mask]
            G_exp = G_exp[mask]

            # Build model.
            self.status.emit("Initializing PDF model…")
            pdf_model = PDFCalculator(
                config,
                config["structure_file"],
                r_exp,
                pair_cutoff=pair_cutoff,
            )

            # Optional status note.
            try:
                structure = pdf_model.structure_handler.structure
                crystal_system = SpacegroupAnalyzer(structure).get_crystal_system().lower()
                if crystal_system != "cubic":
                    self.status.emit(
                        "Non-cubic structure detected. Using invariant contrast-factor form if coefficients are provided."
                    )
            except Exception:
                pass

            # Progress callback: forward to GUI, optionally compute full-grid Rwp occasionally.
            last_fullgrid = 0.0

            def progress_cb(payload: Dict[str, Any]):
                nonlocal last_fullgrid

                if self.fullgrid_rwp:
                    now = time.perf_counter()

                    if (now - last_fullgrid) >= max(1.5, self.progress_every_sec):
                        try:
                            params = payload.get("params", {}) or {}
                            G_fit_full = np.asarray(pdf_model.evaluate(params)["G_r"], dtype=float)

                            num = float(np.sum((G_exp - G_fit_full) ** 2))
                            den = float(np.sum(G_exp ** 2))

                            if den > 0:
                                payload["rwp_full"] = 100.0 * float(np.sqrt(num / den))

                            last_fullgrid = now

                        except Exception:
                            pass

                # Compute the plotted shape/SAXS term during progress only
                # when PDF updates are enabled. The final curve is calculated
                # separately after refinement.
                if self.update_pdf and "r" in payload:
                    try:
                        params = payload.get("params", {}) or {}

                        gamma_avg = pdf_model.compute_isotropic_shape_factor(
                            params,
                            np.asarray(payload["r"], dtype=float),
                        )

                        payload["gamma_avg"] = gamma_avg

                    except Exception:
                        pass

                # Track best parameters seen so far for Stop & Save.
                try:
                    cost = float(payload.get("cost", float("inf")))
                    params = payload.get("params", None)

                    if params is not None and cost < self._best_cost:
                        self._best_cost = cost
                        self._best_params = dict(params)

                except Exception:
                    pass

                self.progress.emit(payload)

            # Run refinement.
            self.status.emit("Running refinement…")
            engine = FitEngine(
                r_exp,
                G_exp,
                pdf_model,
                config,
                progress_callback=progress_cb,
                stop_event=self.stop_event,
                pause_event=self.pause_event,
            )

            refined_params = engine.run_refinement()

            # Use best-seen parameters if available.
            if self._best_params is not None:
                refined_params_best = self._best_params
            else:
                refined_params_best = refined_params

            if refined_params_best is None:
                refined_params_best = config["initial"]

            # ------------------------------------------------------------------
            # Final outputs
            # ------------------------------------------------------------------
            self.status.emit("Preparing final outputs…")
            t_final = time.perf_counter()

            G_contrib = {}
            G_ab = {}

            # Reuse final full-grid PDF already computed by FitEngine, if available.
            if (
                getattr(engine, "final_model_result", None) is not None
                and getattr(engine, "final_G_fit", None) is not None
            ):
                results = engine.final_model_result

                r_full = np.asarray(engine.final_r, dtype=float)
                G_exp_full = np.asarray(engine.final_G_exp, dtype=float)
                G_fit_full = np.asarray(engine.final_G_fit, dtype=float)
                res_full = np.asarray(engine.final_residual, dtype=float)

            else:
                # Fallback only. This should rarely run.
                results = pdf_model.evaluate(
                    refined_params_best,
                    return_contributions=False,
                )

                r_full = r_exp
                G_exp_full = G_exp
                G_fit_full = np.asarray(results["G_r"], dtype=float)
                res_full = G_exp_full - G_fit_full

            # Optional pair contributions.
            if compute_pair_contrib:
                self.status.emit("Computing pair contributions…")
                t_pair = time.perf_counter()

                try:
                    contrib_results = pdf_model.evaluate(
                        refined_params_best,
                        return_contributions=True,
                    )

                    G_contrib = contrib_results.get("G_contrib", {}) or {}
                    G_ab = contrib_results.get("G_ab", {}) or {}

                    
                    print(
                        f"[FINAL TIMING] pair contributions: "
                        f"{time.perf_counter() - t_pair:.3f} s"
                    )

                except Exception:
                    G_contrib = {}
                    G_ab = {}

            # Optional Warren plot.
            warren_by_dir: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}

            if compute_warren_on_finish:
                self.status.emit("Computing Warren plot…")
                t_warren = time.perf_counter()

                try:
                    max_L = pdf_model._plot_distance_limit_from_params(
                        refined_params_best,
                        fallback=5.0,
                    )

                    w = pdf_model.build_warren_plot(
                        refined_params_best,
                        max_L=max_L,
                    )

                    if isinstance(w, dict) and w:
                        ok_keys = all(
                            isinstance(k, tuple) and len(k) == 3
                            for k in w.keys()
                        )

                        if ok_keys:
                            warren_by_dir = w

                    print(
                        f"[FINAL TIMING] Warren plot: "
                        f"{time.perf_counter() - t_warren:.3f} s"
                    )

                except Exception:
                    warren_by_dir = {}

            # Cheap size distribution.
            size_dist = self._build_size_distribution(refined_params_best)

            # Optional local lambda/delta trends.
            local_trends = {}

            if compute_local_trends_on_finish:
                self.status.emit("Computing local trends up to 10 Å…")
                t_local = time.perf_counter()

                try:
                    # Standard/default local trends are only up to 10 Å.
                    # This avoids scanning long-range shell tables after refinement.
                    local_trends = self._build_local_trends(
                        pdf_model,
                        refined_params_best,
                        max_r_plot=10.0,
                    )

                    print(
                        f"[FINAL TIMING] local trends up to 10 Å: "
                        f"{time.perf_counter() - t_local:.3f} s"
                    )

                except Exception:
                    local_trends = {}

            # Do NOT build tick markers automatically.
            # For large CIFs this is expensive.
            # Tick markers are calculated on demand when the user enables the Tick markers checkbox.
            tick_data = {}

            print(
                f"[FINAL TIMING] total final outputs: "
                f"{time.perf_counter() - t_final:.3f} s"
            )


            elapsed = time.perf_counter() - t0

            # Compute isotropic shape factor for final output
            try:
                gamma_avg = pdf_model.compute_isotropic_shape_factor(refined_params_best, r_full)
            except Exception:
                gamma_avg = None
                
            self.finished_ok.emit(
                {
                    "input_path": self.input_path,
                    "source_path": self.source_path,
                    "refined_params": refined_params_best,
                    "refinable_flags": config.get("refinable", {}),
                    "elapsed_s": elapsed,
                    "r_full": r_full,
                    "G_exp_full": G_exp_full,
                    "G_fit_full": G_fit_full,
                    "res_full": res_full,
                    "G_contrib": G_contrib,
                    "G_ab": G_ab,
                    "warren_by_dir": warren_by_dir,
                    "size_distribution": size_dist,
                    "local_trends": local_trends,
                    "tick_data": tick_data,
                    "gamma_avg": gamma_avg,
                }
            )

        except FitStopped:
            elapsed = time.perf_counter() - t0

            best = self._best_params

            out = {
                "input_path": self.input_path,
                "source_path": self.source_path,
                "elapsed_s": elapsed,
                "stopped": True,
                "save_on_stop": bool(getattr(self, "_save_on_stop", False)),
            }

            if best is not None and pdf_model is not None and config is not None and r_exp is not None and G_exp is not None:
                try:
                    results = pdf_model.evaluate(best, return_contributions=False)
                    G_fit = np.asarray(results["G_r"], dtype=float)

                    out.update(
                        {
                            "refined_params": best,
                            "refinable_flags": config.get("refinable", {}),
                            "r_full": r_exp,
                            "G_exp_full": G_exp,
                            "G_fit_full": G_fit,
                            "res_full": G_exp - G_fit,
                        }
                    )

                except Exception:
                    out["refined_params"] = best

            self.finished_ok.emit(out)

        except Exception:
            self.failed.emit(traceback.format_exc())