"""
Import the heavy libraries in a background thread after the window is shown,
so the first Structure (3D) / Calculate does not wait for them (PyVista + VTK
alone take ~10 s from a network drive). Imports only: no Qt widgets are
created off the GUI thread. If the user is faster, the import lock simply
makes the GUI thread wait for the rest.
"""

from __future__ import annotations

import importlib
import threading
import time

MODULES = (
    "scipy.optimize",
    "scipy.sparse",
    "pdf_fitting.models.model_factory",
    "pdf_fitting.fit_engine",
    "pdf_fitting.project.multiphase",
    "pdf_fitting.gui.structure_studio",
    "pyvista",
    "vtk",
    "pyvistaqt",
    # GPU stack (~11 s to import from a network drive); skipped silently
    # when CuPy is not installed.
    "cupy",
    "cupyx.scipy.special",
    "pdf_fitting.models.gpu.generic_stacked_pdf",
    "pdf_fitting.models.stochastic_stacked_pdf_model",
)

_started = False


def _run(verbose: bool):
    t0 = time.perf_counter()
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            if verbose and not name.startswith("cupy"):
                print(f"[PRELOAD] {name}: {exc}")
    if verbose:
        print(f"[PRELOAD] libraries ready in {time.perf_counter() - t0:.1f} s", flush=True)


def start_background_preload(verbose: bool = True) -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_run, args=(verbose,), name="hklpdf-preload", daemon=True).start()
