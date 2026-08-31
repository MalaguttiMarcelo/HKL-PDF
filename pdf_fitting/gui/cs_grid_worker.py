from __future__ import annotations

import copy
import os
import time
import traceback
from typing import Optional, Dict, Any, List

import numpy as np
from PySide6 import QtCore

from pdf_fitting.io_handler import read_input_file
from pdf_fitting.fit_engine import FitEngine, FitStopped
from pdf_fitting.models.gr_model import PDFCalculator
from pdf_fitting.models.crystallite_shapes import CrystalliteShapeSpec, iter_shape_scan
from pdf_fitting.models.finite_shape_shells import (
    get_or_build_finite_shell_table,
    export_coordination_per_direction_csv,
    export_gamma_functions_by_direction,
)



def _cfg_bool(val, default: bool = False) -> bool:
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


class CSGridSearchWorker(QtCore.QThread):
    """
    Discrete crystallite-shape grid-search worker.

    For each finite shape:
        1. build/load finite shell table
        2. replace PDFCalculator shell table
        3. run normal continuous least-squares refinement
        4. report final Rwp
    """

    progress = QtCore.Signal(dict)
    finished_ok = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(
        self,
        *,
        input_path: str,
        source_path: Optional[str] = None,
        progress_every_sec: float = 0.5,
        update_pdf: bool = True,
        ask_after_each_shape: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.input_path = input_path
        self.source_path = source_path
        self.progress_every_sec = float(progress_every_sec)
        self.update_pdf = bool(update_pdf)
        self.ask_after_each_shape = bool(ask_after_each_shape)

        import threading
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self._best = None

    @property
    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def request_stop(self, *, save_best: bool = False):
        self.stop_event.set()
        self.status.emit("CS grid search stop requested…")

    def pause(self):
        self.pause_event.set()
        self.status.emit("Pausing CS grid search…")

    def resume(self):
        self.pause_event.clear()
        self.status.emit("Resuming CS grid search…")

    def _prepare_config_for_finite_shape(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove conventional spherical size parameters from config.

        In finite_shape mode, size is represented by the finite shell table.
        Therefore d, d_std, and d_mu must not appear in the active parameter set.
        """
        cfg = copy.deepcopy(config)

        cfg.setdefault("initial", {})
        cfg.setdefault("refinable", {})
        cfg.setdefault("bounds", {})

        for k in ("d", "d_std", "d_mu"):
            cfg["initial"].pop(k, None)
            cfg["refinable"].pop(k, None)
            cfg["bounds"].pop(k, None)

        return cfg

    def _compute_rwp(self, G_exp: np.ndarray, G_fit: np.ndarray) -> float:
        res = np.asarray(G_exp, dtype=float) - np.asarray(G_fit, dtype=float)
        den = float(np.sum(np.asarray(G_exp, dtype=float) ** 2))
        num = float(np.sum(res * res))
        return 100.0 * float(np.sqrt(num / den)) if den > 0.0 else float("nan")

    def run(self):
        t0 = time.perf_counter()

        try:
            self.status.emit("Loading CS grid-search configuration…")
            base_config = read_input_file(self.input_path)

            shape_cfg = base_config.get("crystallite_shape", {}) or {}

            if str(shape_cfg.get("mode", "conventional")).strip().lower() != "finite_shape":
                raise RuntimeError("CS grid search requires [crystallite_shape] mode = finite_shape")

            shape_spec = CrystalliteShapeSpec.from_dict(shape_cfg)

            candidates = list(iter_shape_scan(shape_spec))

            single_candidate = len(candidates) == 1

            if not candidates:
                raise RuntimeError("No crystallite-shape candidates were generated.")

            # Reject infinite diameter for now in this first CPU finite backend.
            for c in candidates:
                if not np.isfinite(float(c.diameter_cells)):
                    raise RuntimeError(
                        "Infinite-diameter slab mode is not implemented in the first CPU backend. "
                        "Please enable 'Scan/refine diameter' or use a finite diameter value."
                    )

            # Experimental data
            r_exp, G_exp = np.loadtxt(base_config["gr_data_file"], unpack=True)

            r_min = float(base_config.get("r_min", 1.0))
            r_max = float(base_config.get("r_max", float(np.max(r_exp))))
            r_extension = float(base_config.get("r_extension", 1.2))
            pair_cutoff = r_max * r_extension

            m = (r_exp >= r_min) & (r_exp <= r_max)
            r_exp = np.asarray(r_exp[m], dtype=float)
            G_exp = np.asarray(G_exp[m], dtype=float)

            # Base finite config
            finite_config = self._prepare_config_for_finite_shape(base_config)

            # Build one model. We will swap its shell table per candidate.
            self.status.emit("Initializing base PDF model for CS grid search…")
            pdf_model = PDFCalculator(
                finite_config,
                finite_config["structure_file"],
                r_exp,
                pair_cutoff=pair_cutoff,
            )

            structure = pdf_model.structure_handler.structure

            results: List[Dict[str, Any]] = []

            for idx, cand in enumerate(candidates):
                if self.stop_event.is_set():
                    break

                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.05)

                if single_candidate:
                    self.status.emit(
                        f"Finite-shape fit: "
                        f"D={cand.diameter_cells:g}, H={cand.height_cells:g}"
                    )
                else:
                    self.status.emit(
                        f"CS grid {idx + 1}/{len(candidates)}: "
                        f"D={cand.diameter_cells:g}, H={cand.height_cells:g}"
                    )

                # Build/load finite shell table for this candidate.
                t_shell = time.perf_counter()
         

                finite_shell_table = get_or_build_finite_shell_table(
                    structure,
                    pair_cutoff=float(pair_cutoff),
                    shape_spec=cand,
                    boundary_mode="atom",
                )

                try:
                    export_coord = _cfg_bool(
                        (base_config.get("refinement", {}) or {}).get(
                            "export_finite_coordination",
                            False,
                        ),
                        False,
                    )
                
                    if export_coord:
                        # Use the real input/config directory when available.
                        # self.input_path is often a temporary file created by the GUI.
                        if self.source_path:
                            config_dir = os.path.dirname(os.path.abspath(self.source_path))
                        else:
                            config_dir = os.path.dirname(os.path.abspath(self.input_path))
                
                        shape_type = str(cand.shape_type).strip() or "shape"
                
                        shape_tag = (
                            f"{shape_type}_"
                            f"D{float(cand.diameter_cells):g}_"
                            f"H{float(cand.height_cells):g}"
                        )
                
                        shape_tag = (
                            shape_tag
                            .replace(" ", "_")
                            .replace("/", "_")
                            .replace("\\", "_")
                        )
                
                        gamma_output_dir = os.path.join(
                            config_dir,
                            "finite_gamma_functions",
                            shape_tag,
                        )
                
                        export_gamma_functions_by_direction(
                            finite_shell_table,
                            structure,
                            gamma_output_dir,
                            write_details=True,
                        )
                
                        self.status.emit(
                            f"Exported direction-resolved gamma functions: {gamma_output_dir}"
                        )
                
                except Exception:
                    pass

                shell_s = time.perf_counter() - t_shell

                pdf_model.replace_shell_table(finite_shell_table)

                # Use a fresh config per candidate so each shape starts from same input.
                cfg_i = copy.deepcopy(finite_config)

                # Make progress less noisy inside inner refinement unless user configured otherwise.
                cfg_i.setdefault("refinement", {})
                cfg_i["refinement"]["progress_every_sec"] = self.progress_every_sec
                cfg_i["refinement"]["update_pdf_during_refinement"] = bool(self.update_pdf)

                def progress_cb(payload: Dict[str, Any]):
                    payload = dict(payload)
                    payload["cs_grid"] = True
                    payload["shape_index"] = int(idx)
                    payload["shape_total"] = int(len(candidates))
                    payload["diameter_cells"] = float(cand.diameter_cells)
                    payload["height_cells"] = float(cand.height_cells)
                    self.progress.emit(payload)

                engine = FitEngine(
                    r_exp,
                    G_exp,
                    pdf_model,
                    cfg_i,
                    progress_callback=progress_cb,
                    stop_event=self.stop_event,
                    pause_event=self.pause_event,
                )

                try:
                    refined_params = engine.run_refinement()
                except FitStopped:
                    break

                if refined_params is None:
                    refined_params = cfg_i["initial"]

                if getattr(engine, "final_G_fit", None) is not None:
                    G_fit = np.asarray(engine.final_G_fit, dtype=float)
                else:
                    model_result = pdf_model.evaluate(refined_params)
                    G_fit = np.asarray(model_result["G_r"], dtype=float)

                rwp = self._compute_rwp(G_exp, G_fit)

                row = {
                    "index": int(idx),
                    "n_total": int(len(candidates)),
                    "diameter_cells": float(cand.diameter_cells),
                    "height_cells": float(cand.height_cells),
                    "rwp": float(rwp),
                    "params": dict(refined_params),
                    "shell_build_s": float(shell_s),
                }

                results.append(row)

                if self._best is None or (
                    np.isfinite(rwp)
                    and float(rwp) < float(self._best.get("rwp", float("inf")))
                ):
                    self._best = dict(row)

                payload = dict(row)
                payload["best"] = dict(self._best) if self._best is not None else None
                self.progress.emit(payload)

            self.finished_ok.emit(
                {
                    "elapsed_s": float(time.perf_counter() - t0),
                    "results": results,
                    "best": self._best,
                    "input_path": self.input_path,
                    "source_path": self.source_path,
                    "refinable_flags": finite_config.get("refinable", {}),
                }
            )

        except Exception:
            self.failed.emit(traceback.format_exc())