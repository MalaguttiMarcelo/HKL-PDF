# HKL-PDF Documentation

HKL-PDF is a Python application for calculating and refining atomic pair
distribution function data, $G(r)$, from crystallographic structures.

The application combines a PySide6 graphical interface with SciPy
least-squares refinement, pymatgen structure handling, and Numba-accelerated
numerical kernels.

> [!IMPORTANT]
> HKL-PDF is research software. Always inspect the residual, parameter
> correlations, refinement bounds, and physical validity of the model before
> reporting results.

## Main features

- Project window with several datasets and several phases per dataset
- JSON project files that also store the last fit and esds
- Interactive structure editor with symmetry-preserving site edits
- Refinement of site coordinates and occupancies
- Anisotropic displacement parameters ($\beta_{ij}$) with site-symmetry
  constraints and thermal ellipsoids
- Esds and correlation matrix of refined parameters (optional)
- Free-peak fit and peak-width analysis per crystallographic direction
- Deterministic and stochastic (Markov) layer stacking and stacking faults
- Optional NVIDIA GPU acceleration (CuPy/CUDA)
- PDF calculation from CIF and other pymatgen-supported structure files
- Conventional periodic-crystal PDF fitting
- Symmetry-constrained lattice parameters
- Weighted and robust least-squares refinement
- Automatic staged refinement
- Element-level and site-labelled isotropic displacement parameters
- Pair-specific local correlation parameters
- Spherical crystallite-size distributions
- Direction-dependent cylinder and disk models
- Explicit finite crystallite shapes
- Finite-shape grid searches
- Isotropic, Wilkens, and PAH microstrain models
- Cubic and non-cubic contrast factors
- Optional missing-low-$Q$/SAXS belly model
- Finite-$Q_{\max}$ termination effects
- Pair contributions and PDF tick markers
- Warren and local-dynamics plots
- Interactive structure and crystallite-shape viewers
- Shell and finite-shape caching
- Real-time refinement monitoring

## Getting started

1. [Install HKL-PDF](installation.md).
2. Follow the [quick-start guide](quick-start.md).
3. Read the [input-file reference](input-format.md).
4. Review the [parameter reference](parameters.md).
5. Use the [troubleshooting guide](troubleshooting.md) if a calculation fails.

## User documentation

- [Installation](installation.md)
- [Quick start](quick-start.md)
- [Graphical interface](gui.md)
- [Project files (JSON)](project-format.md)
- [Input-file format (.inp)](input-format.md)
- [Parameter reference](parameters.md)
- [Structure editor](structure-editor.md)
- [GPU acceleration and performance](gpu.md)
- [Troubleshooting](troubleshooting.md)

## Theory and fitting

- [Model equations](equations.md)
- [Conventional PDF fitting](conventional-fitting.md)
- [Crystallite-size modelling](microcrystallin.md)
- [Microstrain](microstrain.md)
- [SAXS and missing-low-$Q$ model](saxs-model.md)
- [Finite crystallite shapes](finite-shapes.md)
- [Layer stacking and stacking faults](stacking-faults.md)
- [Anisotropic displacement parameters](adp.md)
- [Esds and correlation matrix](uncertainties.md)
- [Free-peak fit and width analysis per direction](free-peaks.md)

## Model reference

- [Conventional PDF model](models/conventional.md)
- [Spherical size model](models/spherical-size.md)
- [Cylinder and disk model](models/cylinder-disk.md)
- [Finite-shape model](models/finite-shape.md)
- [Microstrain models](models/microstrain.md)
- [SAXS belly model](models/saxs-belly.md)

## Development

- [Developer guide](developer-guide.md)
- [Changelog](changelog.md)

## Typical workflow

A normal HKL-PDF analysis follows these steps:

1. Prepare a crystallographic structure, normally as a CIF file.
2. Prepare experimental $G(r)$ data as a two-column text file.
3. Create a project (or open an `.inp` file) and add the dataset and
   phases.
4. Select a fitting range.
5. Calculate the initial PDF without refinement.
6. Compare the calculated PDF with the experimental data.
7. Refine a small set of physically meaningful parameters.
8. Inspect $R_\mathrm{wp}$, the residual, and parameter evolution.
9. Add size, local-dynamics, or microstrain parameters only when required.
10. Save the refined parameters and preserve the original input.

## Supported model families

### Conventional periodic model

The conventional model calculates crystallographic pair shells from a
periodic structure. Analytical functions describe particle size and
instrumental effects.

This is the recommended starting model for most refinements.

### Spherical crystallite model

A spherical common-volume function attenuates the PDF with increasing pair
distance. Monodisperse and lognormal size distributions are supported.

### Cylinder and disk model

A direction-dependent common-volume function describes cylindrical or
disk-like crystallites. The cylinder axis is specified as a direct
crystallographic direction.

### Finite crystallite model

Explicit finite particles are created from translated unit cells. Atom-pair
coordination is counted inside the selected particle boundary.

A single predefined shape or a discrete diameter/height grid search can be
used.

### Microstrain models

HKL-PDF provides:

- isotropic distance-dependent broadening;
- Wilkens dislocation microstrain;
- PAH directional microstrain.

### Layer stacking

Layered materials can be built from a layer and a deterministic or random
(Markov) sequence of stacking vectors, inside a finite crystallite or as
bulk.

### SAXS belly model

An optional smooth negative term can represent a missing-low-$Q$ or
particle-shape contribution independently of the main PDF peak envelope.

## Input data

Experimental data must contain at least two numeric columns:

```text
r    G(r)
```

For example:

```text
1.0000   -0.1532
1.0100   -0.1498
1.0200   -0.1421
```

The first column is the pair distance $r$, normally in Å. The second column
is the experimental reduced PDF $G(r)$.

## Citation

Citation metadata are provided in the repository-level
[`CITATION.cff`](../CITATION.cff) file.

## License

See the repository-level [`LICENSE`](../LICENSE) file.