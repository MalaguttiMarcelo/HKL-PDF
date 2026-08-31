# Installation

## Requirements

HKL-PDF requires Python 3.10 or newer.

The main dependencies include:

- NumPy
- SciPy
- pandas
- pymatgen
- Numba
- joblib
- PySide6
- Matplotlib
- PyVista
- PyVistaQt
- VTK
- qtpy

A working OpenGL installation is recommended for the structure and
crystallite-shape viewers.

## Clone the repository

```bash
git clone https://github.com/MalaguttiMarcelo/HKL-PDF.git
cd HKL-PDF
```

Replace the repository URL if you publish the project under a different
GitHub account or repository name.

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install HKL-PDF

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

Editable mode is recommended while developing HKL-PDF because changes to the
Python source are available without reinstalling the package.

## Install visualization dependencies

If the visualization packages were not installed automatically, run:

```bash
python -m pip install pyvista pyvistaqt vtk qtpy
```

## Start the GUI

```bash
python -m pdf_fitting.gui
```

The project may also provide a launcher or installed command, depending on
the package configuration.

## First calculation

The first PDF calculation may be slower because:

1. Numba compiles the numerical kernels.
2. HKL-PDF generates the crystallographic shell table.
3. The shell table is written to the user cache.

Later calculations with the same structure and pair cutoff should be faster.

## Cache location

HKL-PDF creates writable user-cache directories.

On Windows, they are normally located under:

```text
%LOCALAPPDATA%\HKL-PDF
```

If `LOCALAPPDATA` is unavailable, the fallback is:

```text
~/HKL-PDF
```

Important cache subdirectories include:

```text
shell_cache
finite_shape_cache
```

## Verify the installation

Test the package import:

```bash
python -c "import pdf_fitting; print('HKL-PDF imported successfully')"
```

Test the GUI:

```bash
python -m pdf_fitting.gui
```

## Development tools

Optional development tools can be installed with:

```bash
python -m pip install pytest ruff build
```

Run the tests:

```bash
pytest
```

Build source and wheel distributions:

```bash
python -m build
```

The generated packages are written to `dist/`.

## OpenGL problems

If the structure viewer does not open:

1. update the graphics driver;
2. update PyVista and VTK;
3. verify that OpenGL is available;
4. avoid running the viewer through a remote session without GPU support.

```bash
python -m pip install --upgrade pyvista pyvistaqt vtk
```