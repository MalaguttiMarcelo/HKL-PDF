# Changelog

All notable changes to HKL-PDF are documented in this file.

The project follows the principles of
[Keep a Changelog](https://keepachangelog.com/).

## 0.2.0 — 2026-11-23

### Added

- Project window: several datasets and several phases per dataset, a
  project tree, per-phase scale factors and weight fractions, and namespaced
  constraints between datasets and phases.
- JSON project files (`*.hklpdf.json`). Old `.inp` files still open. The
  last calculated fit and esds are saved in the project and shown when it is
  opened.
- Structure editor (Structure 3D page): measurements, neighbour table, tick
  markers on the fit plot, symmetry-preserving site moves, adding and
  deleting orbits, CIF export.
- Refinement of site coordinates and occupancies with site-symmetry
  constraints.
- Anisotropic displacement parameters ($\beta_{ij}$) with site-symmetry
  constraints, CIF import and export, and thermal ellipsoids.
- Esds and correlation matrix of refined parameters, with an optional
  correction for correlated PDF points (off by default).
- Free-peak fit and peak-width analysis per crystallographic direction,
  including contrast-factor invariants per Laue class and import of
  contrast factors from a file.
- Layer stacking: deterministic sequences and stochastic A/B and
  multi-state Markov stacking faults, in finite crystallites or as bulk.
- NVIDIA GPU (CuPy) evaluation of stacking models, including Wilkens and
  PAH microstrain.
- Session and disk caches for compiled stacking topologies.
- Loading screen that imports every heavy library before the main window
  opens, and an application icon.
- One running copy per user; a second start opens its window in the
  running copy. Optional background mode keeps HKL-PDF loaded after the
  last window closes (default on); optional start at Windows login (default
  off).
- Windows installer: per-user installation without administrator rights,
  Numba kernels prepared during installation, and a one-step build script
  (`installer/build.ps1`). NVIDIA GPU support is not shipped. It is
  downloaded on request (installer option or **File › NVIDIA GPU
  support…**), with pinned checksums.
- Refinement progress plots moved to their own page at the top of the
  project tree; compact one-row header.
- Debug timing output is off unless `PDF_FITTING_TIMING=1` is set.
- Finite-shape gamma and coordination export.
- On-demand PDF tick-marker generation.
- Element-grouped Biso controls for large structures.

### Changed

- The project window is the default; the previous window opens with
  `--classic`.
- Relative paths in `.inp` files are read relative to the input file.
- The fit uses its own forward-difference Jacobian with absolute steps.
- Per-shell calculations (distances, correlations, strain, widths) run in
  parallel Numba kernels.
- Shell directions are found as exact small-integer directions; shell caches
  were rebuilt.
- Expensive diagnostics are calculated only when requested.
- Contrast factors are stored as normal refinement parameters.
- The `[contrast_factors]` section is deprecated.
- Large structures use packed NumPy shell tables rather than Python shell
  dictionaries.

### Fixed

- Stochastic A/B state parameters (`stack_state_a_*`, `stack_state_b_*`)
  were ignored. Refinements of stochastic stacking made before this fix
  should be repeated.
- Crashes when a free-peak fit finished and when a new file was opened.
- Missing tick markers for a pair selected in the structure editor.
- A popped-out plot window could open off screen.

### Known limitations

- Atomic-number scattering approximations are currently used.
- Site-specific Biso values are averaged by element (use anisotropic ADPs
  for site-specific displacements).
- Peak widths with anisotropic ADPs are calculated on the CPU.
- Infinite-diameter finite slabs are not implemented.
- AMD GPUs are not supported.

## 0.1.0

Initial public release.

### Added

- Conventional PDF fitting
- Optional missing-low-$Q$/SAXS belly model
- Spherical and cylinder/disk size models
- Finite crystallite-shape support
- Isotropic, Wilkens, and PAH microstrain models
- PySide6 GUI
- Numba-accelerated PDF calculation

### Fixed

- Separated the conventional PDF baseline from optional SAXS belly mode
- Added an explicit `saxs_enabled` switch
- Corrected Qmax kernel normalization
- Reduced low-$r$ boundary artifacts by evaluating Qmax on an internal grid