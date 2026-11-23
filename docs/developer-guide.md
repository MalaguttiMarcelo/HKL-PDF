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

The maintained tests are in `tests/`. They cover multi-phase projects,
saved results, shell directions, ADPs, esds, and the width analysis.

The older tests in `pdf_fitting/test/` target an earlier stacking API and
are not part of the default test run.

## Formatting

```bash
ruff check .
ruff format .
```

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
