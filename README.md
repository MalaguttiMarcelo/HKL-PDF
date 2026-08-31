# PDF Fitting

A Python application for fitting atomic pair distribution function data using
crystal structures, crystallite-size models, microstrain models, and optional
missing-low-Q/SAXS baseline modelling.

> Status: early research software. Validate fitted parameters independently
> before using them in publications.

## Features

- PDF calculation from CIF structures
- Least-squares refinement with SciPy
- Symmetry-constrained lattice refinement
- Isotropic displacement parameters
- Spherical crystallite-size distributions
- Cylinder/disk common-volume models
- Finite crystallite-shape calculations
- Optional missing-low-Q/SAXS belly modelling
- Isotropic, Wilkens, and PAH microstrain models
- Pair contributions and PDF tick markers
- Warren plots
- PySide6 graphical interface
- Numba-accelerated shell accumulation

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- A working OpenGL installation for PyVista/VTK visualization

## Installation

Clone the repository:

```bash
git clone https://github.com/MalaguttiMarcelo/HKL-PDF.git
cd pdf-fitting