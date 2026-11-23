# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller specification of the Windows program (folder build, not one
# file). Build with installer/build.ps1, which also creates the installer.
#
# Only modules that HKL-PDF imports are bundled (PyInstaller follows the
# imports; package hooks add their data files). The previous spec collected
# *everything* from scipy, pymatgen, VTK, pandas, ... including their tests,
# which doubled the size and the number of files to load at start-up.

import importlib.util
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(SPECPATH)

datas = [(os.path.join(ROOT, "examples"), "examples")]
datas += collect_data_files("pymatgen")          # element data, YAML/JSON tables
datas += collect_data_files("pyvista")

hiddenimports = collect_submodules("pdf_fitting")  # some modules are imported by name
hiddenimports += ["matplotlib.backends.backend_qtagg", "PySide6.QtNetwork"]

# The whole Python standard library (a few MB). The downloaded GPU support
# (CuPy) imports standard modules from compiled code that PyInstaller cannot
# see (e.g. graphlib); with the full library any CuPy version finds them.
import sys as _sys

_SKIP_STDLIB = ("tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3", "ensurepip", "venv", "pydoc_data")
for _name in sorted(getattr(_sys, "stdlib_module_names", ())):
    if _name.startswith("_") or _name in _SKIP_STDLIB or _name.startswith(_SKIP_STDLIB):
        continue
    hiddenimports.append(_name)
    try:
        _spec = importlib.util.find_spec(_name)
    except (ImportError, ValueError):
        continue
    if _spec is not None and _spec.submodule_search_locations:
        try:
            hiddenimports += [m for m in collect_submodules(_name) if ".test" not in m and ".tests" not in m]
        except Exception:  # noqa: BLE001
            pass

# List of the GPU packages that "Install NVIDIA GPU support" downloads.
datas += [(os.path.join(ROOT, "pdf_fitting", "gpu_packages.json"), "pdf_fitting")]

# HKL-PDF itself is kept as .py files: Numba's on-disk kernel cache needs the
# source files.
collection_mode = {"pdf_fitting": "py"}

# NVIDIA GPU support is NOT shipped: it is downloaded on request
# (pdf_fitting/gpu_install.py) into the user folder. CuPy is analysed here
# only so that the standard-library modules it needs are bundled; its own
# files are kept as plain files and deleted by build.ps1.
GPU = importlib.util.find_spec("cupy") is not None
if GPU:
    for pkg in ("cupy", "cupyx", "cupy_backends", "cuda"):
        if importlib.util.find_spec(pkg) is not None:
            collection_mode[pkg] = "py"
    hiddenimports += collect_submodules("cupy") + collect_submodules("cupyx.scipy.special")
    hiddenimports += collect_submodules("cupy_backends")
    if importlib.util.find_spec("cuda") is not None:
        hiddenimports += collect_submodules("cuda.pathfinder")

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "IPython",
        "jupyter_client",
        "notebook",
        "pytest",
        "sphinx",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "graphify",
    ],
    noarchive=False,
    optimize=0,
    module_collection_mode=collection_mode,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HKL-PDF",
    icon=os.path.join(ROOT, "installer", "HKL-PDF.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HKL-PDF",
)
