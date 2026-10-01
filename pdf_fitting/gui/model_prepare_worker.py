from __future__ import annotations

import copy
import os
import time
import traceback
from typing import Any, Dict

import numpy as np

from PySide6 import QtCore


class ModelPrepareWorker(QtCore.QThread):
    """
    Build and warm a PDFCalculator in a background thread.

    The expensive imports, CIF parsing, shell-cache loading, feature-cache
    loading, and first Numba call happen here instead of when Calculate is
    pressed.
    """

    prepared = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        generation: int,
        parent=None,
    ):
        super().__init__(parent)

        self.config = copy.deepcopy(config)
        self.generation = int(generation)

    def run(self) -> None:
        total_start = time.perf_counter()

        try:
            import_start = time.perf_counter()

            from pdf_fitting.fit_engine import apply_constraints
            from pdf_fitting.models.gr_model import PDFCalculator

            import_seconds = (
                time.perf_counter()
                - import_start
            )

            config = copy.deepcopy(
                self.config
            )

            structure_file = os.path.abspath(
                str(config["structure_file"])
            )

            gr_data_file = os.path.abspath(
                str(config["gr_data_file"])
            )

            config["structure_file"] = structure_file
            config["gr_data_file"] = gr_data_file

            self.status.emit(
                "Preparing the numerical PDF model in the background..."
            )

            data_start = time.perf_counter()

            r_exp, G_exp = np.loadtxt(
                gr_data_file,
                unpack=True,
            )

            r_min = float(
                config.get(
                    "r_min",
                    1.0,
                )
            )

            r_max = float(
                config.get(
                    "r_max",
                    np.max(r_exp),
                )
            )

            r_extension = float(
                config.get(
                    "r_extension",
                    1.2,
                )
            )

            pair_cutoff = (
                r_max
                * r_extension
            )

            mask = (
                (r_exp >= r_min)
                & (r_exp <= r_max)
            )

            r_exp = np.asarray(
                r_exp[mask],
                dtype=float,
            )

            G_exp = np.asarray(
                G_exp[mask],
                dtype=float,
            )

            data_seconds = (
                time.perf_counter()
                - data_start
            )

            model_start = time.perf_counter()

            model = PDFCalculator(
                config,
                structure_file,
                r_exp,
                pair_cutoff=pair_cutoff,
            )

            model_seconds = (
                time.perf_counter()
                - model_start
            )

            params = dict(
                config.get(
                    "initial",
                    {},
                )
                or {}
            )

            constraints = (
                config.get(
                    "constraints",
                    {},
                )
                or {}
            )

            if constraints:
                params = apply_constraints(
                    params,
                    constraints,
                )

            self.status.emit(
                "Warming the PDF kernels in the background..."
            )

            evaluation_start = time.perf_counter()

            result = model.evaluate(
                params,
                return_contributions=False,
                compute_gamma_avg=False,
            )

            evaluation_seconds = (
                time.perf_counter()
                - evaluation_start
            )

            self.prepared.emit(
                {
                    "generation": self.generation,
                    "config": config,
                    "model": model,
                    "params": params,
                    "r_exp": r_exp,
                    "G_exp": G_exp,
                    "pair_cutoff": float(pair_cutoff),
                    "result": result,
                    "import_s": float(import_seconds),
                    "data_s": float(data_seconds),
                    "model_s": float(model_seconds),
                    "evaluation_s": float(evaluation_seconds),
                    "total_s": float(
                        time.perf_counter()
                        - total_start
                    ),
                }
            )

        except Exception:
            self.failed.emit(
                traceback.format_exc()
            )