# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = []

packages_to_collect = [
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "pymatgen",
    "pandas",
    "joblib",
    "matplotlib",
    "pyvista",
    "pyvistaqt",
    "vtkmodules",
    "qtpy",
]

for package_name in packages_to_collect:
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )

    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

datas += [
    (
        "examples",
        "examples",
    ),
    (
        "docs",
        "docs",
    ),
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(
    a.pure
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HKL-PDF",
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