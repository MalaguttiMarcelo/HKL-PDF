# Developer Guide

## Main package structure

```text
pdf_fitting/
├── fit_engine.py
├── io_handler.py
├── structure_handler.py
├── user_paths.py
├── models/
│   ├── gr_model.py
│   ├── numba_kernels.py
│   ├── microstrain_utils.py
│   ├── crystallite_shapes.py
│   └── finite_shape_shells.py
└── gui/
    ├── main.py
    ├── mainwindow.py
    ├── builder.py
    ├── worker.py
    ├── cs_grid_worker.py
    ├── plots.py
    ├── structure_view.py
    └── fast_shape_viewer.py
```

## Calculation flow

1. `read_input_file()` parses the INI file.
2. `PDFCalculator` creates a `StructureHandler`.
3. `StructureHandler` generates or loads packed shell arrays.
4. `PDFCalculator.evaluate()` calculates distances, widths, partial PDFs,
   weights, baseline, damping, and termination effects.
5. `FitEngine` sends residuals to `scipy.optimize.least_squares`.
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

Recommended test areas:

- input parsing;
- safe constraints;
- lattice symmetry;
- shell multiplicity;
- finite coordination;
- PDF normalization;
- Qmax behaviour;
- bounds;
- worker stopping and pausing.

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
- Site coordinates and occupancies do not yet regenerate packed shells during
  refinement.
- Analytic Jacobians are not implemented.
- Infinite-diameter finite slabs are not implemented.