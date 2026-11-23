# Installation

## Windows installer

The simplest way on Windows is the installer from the
[Releases page](https://github.com/MalaguttiMarcelo/HKL-PDF/releases):

1. Download and run `HKL-PDF-Setup-<version>.exe`. No administrator rights
   are needed; HKL-PDF is installed for the current user in
   `%LOCALAPPDATA%\Programs\HKL-PDF`.
2. Only on computers with an NVIDIA graphics card: tick **Download and
   install NVIDIA GPU support** (about 700 MB from the Python Package
   Index). It can also be added later with **File › NVIDIA GPU support…**.
3. The installer prepares the numerical kernels at the end (about a
   minute).

Messages of the installed program are written to
`%LOCALAPPDATA%\HKL-PDF\hkl-pdf.log`. Attach this file to bug reports.

To build the installer yourself, see
[Building the installer](developer-guide.md#building-the-windows-installer).

The rest of this page describes installation from the Python source.

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

## Optional: NVIDIA GPU support

GPU calculations need CuPy built for your CUDA version, for example:

```bash
python -m pip install cupy-cuda12x
```

See [GPU acceleration](gpu.md).

## Start the GUI

```bash
python -m pdf_fitting.gui
```

After `pip install`, the `hkl-pdf` command starts the same window. Add
`--classic` to open the single-phase window.

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
cache/stochastic_topology
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