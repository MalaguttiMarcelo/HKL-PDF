# Changelog

All notable changes to HKL-PDF are documented in this file.

The project follows the principles of
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added

- Documentation for installation, input files, equations, fitting models, and
  development.
- Finite-shape gamma and coordination export.
- On-demand PDF tick-marker generation.
- Element-grouped Biso controls for large structures.

### Changed

- Expensive diagnostics are calculated only when requested.
- Contrast factors are stored as normal refinement parameters.
- The `[contrast_factors]` section is deprecated.
- Large structures use packed NumPy shell tables rather than Python shell
  dictionaries.

### Known limitations

- Atomic-number scattering approximations are currently used.
- Site-specific Biso values are averaged by element.
- Site coordinates and occupancies are not yet active in packed-shell
  refinement.
- Analytic Jacobians are not implemented.
- Infinite-diameter finite slabs are not implemented.

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