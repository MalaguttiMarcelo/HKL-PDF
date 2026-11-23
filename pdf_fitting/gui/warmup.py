"""
``hkl-pdf --warmup``: compile the Numba kernels once and store them in the
user cache, so the first Calculate after installing is not slowed down by
compilation. Run by the installer; safe to run again at any time.

Evaluates the bundled examples on a short r range: Fe–Cr (Wilkens strain,
spherical size) and calcite (PAH strain, several species). Exit code 0 even
when an example is missing — warming up is only an optimisation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

EXAMPLES = (
    ("1-FeCr-ball-milled", "input_fecr_paper_2.hklpdf.json"),
    ("2-Calcite-ball-milled", "calcite_150min.hklpdf.json"),
)


def _examples_dir() -> Path:
    base = getattr(sys, "_MEIPASS", None)  # installed program
    if base:
        return Path(base) / "examples"
    return Path(__file__).resolve().parents[2] / "examples"


def run_warmup(r_max: float = 20.0) -> int:
    from pdf_fitting.project.model import Project
    from pdf_fitting.project.multiphase import ProjectModel

    t0 = time.perf_counter()
    for folder, name in EXAMPLES:
        path = _examples_dir() / folder / name
        if not path.is_file():
            print(f"[WARMUP] example not found: {path}")
            continue
        t = time.perf_counter()
        try:
            project = Project.load(str(path))
            for ds in project.datasets:
                ds.r_max = min(float(ds.r_max), r_max)
            model = ProjectModel(project)
            params = project.flat_tables()[0]
            model.evaluate(params)
            model.evaluate(params)  # second call: the steady-state kernels
            print(f"[WARMUP] {folder}: {time.perf_counter() - t:.1f} s")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARMUP] {folder} failed: {exc}")
    print(f"[WARMUP] done in {time.perf_counter() - t0:.1f} s (Numba cache: {os.environ.get('NUMBA_CACHE_DIR', 'default')})")
    return 0
