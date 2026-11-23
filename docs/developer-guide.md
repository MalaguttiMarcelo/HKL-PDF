# Developer Guide

## Main package structure

```text
pdf_fitting/
├── fit_engine.py            least-squares driver, parameter pipeline
├── uncertainty.py           esds and correlation matrix
├── io_handler.py            .inp reader and writer
├── structure_handler.py     shells, site symmetry, coordinates, ADPs
├── user_paths.py            user data and cache folders
├── project/                 Qt-free project model
│   ├── model.py             Project / Dataset / Phase / Parameter, JSON
│   ├── multiphase.py        one calculator per phase, combined residual
│   ├── saved_results.py     compact storage of the last fit
│   ├── edited_cif.py        derived CIFs from structure edits, ADP loops
│   ├── structure_tools.py   orbits and symmetry helpers for the editor
│   ├── free_peaks.py        free-peak fit
│   ├── warren_fit.py        width analysis per direction
│   └── contrast_factors.py  Laue-class invariants, contrast-factor files
├── models/
│   ├── gr_model.py          PDFCalculator
│   ├── numba_kernels.py     parallel CPU kernels
│   ├── direction_utils.py   shell directions
│   ├── crystallite_shapes.py, finite_shape_shells.py
│   ├── model_factory.py     chooses the model for a configuration
│   ├── layer_stacking.py    stacking specification
│   ├── dynamic_stacked_model.py          deterministic stacking
│   ├── stochastic_stacked_pdf_model.py   stochastic stacking
│   ├── compiled_stacked_* / compiled_stochastic_*   topology compilers
│   ├── fast_unique.py       exact row grouping (optional GPU)
│   └── gpu/                 CuPy backends
└── gui/
    ├── main.py              entry point (project or classic window)
    ├── preload.py           background imports
    ├── project_window/      project window, panels, structure page, peaks page
    ├── structure_studio/    3D structure view
    ├── finite_crystallite_builder/   shape and stacking builder
    └── mainwindow.py, builder.py, worker.py, ...   classic window
```

## Calculation flow

1. `Project.load()` reads a JSON project, or `read_input_file()` an INI
   file (converted to a one-phase project).
2. `PDFCalculator` creates a `StructureHandler`.
3. `StructureHandler` generates or loads packed shell arrays.
4. `PDFCalculator.evaluate()` calculates distances, widths, partial PDFs,
   weights, baseline, damping, and termination effects.
5. `FitEngine` sends residuals and its forward-difference Jacobian to
   `scipy.optimize.least_squares`. Multi-phase projects combine one
   calculator per phase (`project/multiphase.py`).
6. GUI workers execute calculations outside the Qt main thread.
7. Progress dictionaries are emitted to the GUI.

## Packed shell table

Important arrays include:

```text
shell_frac
shell_mult
shell_mult_eff
shell_pair_id
shell_alpha_id
shell_beta_id
shell_hkl
shell_direction
shell_upair_id
shell_k_within_upair
pair_offsets
```

Shell arrays are sorted by ordered pair ID for fast Numba accumulation.

## Numba kernels

The numerical kernels provide:

- pair-parallel Gaussian accumulation;
- shell-block Gaussian accumulation;
- pair-resolved gamma histograms;
- finite-pair counting;
- shell-generation post-processing.

Avoid Python objects, dictionaries, and strings inside hot kernels.

## Cache versions

When a cache format or meaning changes, update the corresponding version
constant.

Examples include:

```python
CACHE_VERSION = "..."
FINITE_CATALOG_CACHE_VERSION = "..."
FINITE_SHELL_CACHE_VERSION = "..."
```

This prevents incompatible old caches from being loaded.

## Adding a parameter

1. Add a default if appropriate.
2. Add GUI metadata and a tooltip.
3. Parse it in the model.
4. Define safe bounds.
5. Add it to staged grouping if required.
6. Document its units and equation.
7. Add tests.

## Adding a shape

1. Extend `CrystalliteShapeSpec`.
2. Add the geometry to `shape_mask()`.
3. Update the shape dialog.
4. Add a preview outline.
5. Include new fields in the cache hash.
6. Test orientation and boundary inclusion.

## Tests

```bash
pytest
```

The tests are in `tests/`. They cover multi-phase projects, saved
results, shell directions, ADPs, esds, and the width analysis.

## Formatting

```bash
ruff check .
ruff format .
```

## Building the Windows installer

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
```

Requirements: Python 3.14 (`py -3.14`) and Inno Setup 6. Build on a local
disk; the script uses `%LOCALAPPDATA%\HKL-PDF-build` by default.

| File | Role |
|---|---|
| `installer/requirements-build.txt` | Pinned package versions of the build environment |
| `HKL-PDF.spec` | PyInstaller folder build; bundles only imported modules |
| `launcher.py` | Entry point of the executable; sets the cache folders |
| `installer/build.ps1` | Environment → GPU manifest → PyInstaller → remove CUDA → warm-up test → installer |
| `installer/make_gpu_manifest.py` | Writes `pdf_fitting/gpu_packages.json` (wheel URLs, sizes, SHA-256) from the pinned versions |
| `installer/HKL-PDF.iss` | Inno Setup script: per-user install, optional GPU download task |
| `pdf_fitting/gpu_install.py` | Downloads, checks, unpacks and activates the GPU wheels; `check()` GPU test |

Notes:

- `pdf_fitting` is bundled as `.py` files, because Numba's on-disk kernel
  cache needs the source files.
- No CUDA library is shipped. CuPy is analysed during the build (so the
  standard-library modules it needs are bundled) and then removed. On
  request (`--install-gpu`, the installer task, or **File › NVIDIA GPU
  support…**) the wheels listed in `gpu_packages.json` are downloaded from
  PyPI, checked against their SHA-256, and unpacked into
  `%LOCALAPPDATA%\HKL-PDF\gpu\<python tag>`. That folder is put on
  `sys.path` at start-up. Only CuPy, `cuda-pathfinder`, the CUDA runtime,
  NVRTC and cuBLAS are downloaded (about 730 MB); cuSPARSE, cuSOLVER,
  cuFFT and cuRAND are not used.
- The wheels must match the program's Python version (`cp314`).
  `make_gpu_manifest.py` uses the build environment's Python.
- `HKL-PDF.exe --gpu-check` tests the GPU support (exit code 0 = works).
- `HKL-PDF.exe --warmup` evaluates two bundled examples to fill the Numba
  cache. The installer runs it, and the build script uses it as a smoke
  test.
- Regenerate `requirements-build.txt` from the tested development
  environment (`pip freeze`) when upgrading packages.

## Start-up and background mode

- `gui/splash.py` imports every heavy library behind a loading screen
  before the main window opens.
- `gui/resident.py` keeps one HKL-PDF process per user (local socket). A
  second start hands its request to the running copy and exits. With
  **Keep HKL-PDF loaded after closing** (default on), closing the last
  window leaves the process in the tray. `--background` (used by **Start at
  Windows login**, default off) loads without a window.

## Build

```bash
python -m build
```

## Current implementation limitations

- X-ray weighting uses atomic-number approximations.
- Site-labelled Biso values are averaged by element.
- The main fit uses finite-difference Jacobians (the free-peak fit has an
  analytic one).
- Peak widths with anisotropic ADPs are calculated on the CPU.
- Infinite-diameter finite slabs are not implemented.
- AMD GPUs are not supported.
