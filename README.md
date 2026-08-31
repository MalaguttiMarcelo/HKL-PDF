# HKL-PDF

HKL-PDF is a Python application for calculating and refining atomic pair
distribution function data, $G(r)$, using crystallographic structures,
crystallite-size models, microstrain models, and optional
missing-low-$Q$/SAXS baseline modelling.

> [!IMPORTANT]
> HKL-PDF is early-stage research software. Validate fitted parameters,
> residuals, parameter correlations, and physical assumptions independently
> before using the results in publications.

## Features

- PDF calculation from CIF structures
- Least-squares refinement with SciPy
- Symmetry-constrained lattice refinement
- Element-level and site-labelled isotropic displacement parameters
- Spherical crystallite-size distributions
- Direction-dependent cylinder/disk common-volume models
- Explicit finite crystallite-shape calculations
- Finite-shape diameter and height grid searches
- Optional missing-low-$Q$/SAXS belly modelling
- Isotropic, Wilkens, and PAH microstrain models
- Cubic and non-cubic contrast factors
- Pair-specific local correlation parameters
- Pair contributions and PDF tick markers
- Warren and local-dynamics plots
- PySide6 graphical interface
- PyVista/VTK structure and crystallite-shape visualization
- Numba-accelerated shell generation and PDF calculation
- Cached shell and finite-shape calculations
- Real-time refinement monitoring

## Documentation

The full documentation is available in the [`docs`](docs/index.md) directory.

Useful pages include:

- [Installation](docs/installation.md)
- [Quick start](docs/quick-start.md)
- [Graphical interface](docs/gui.md)
- [Input-file format](docs/input-format.md)
- [Parameter reference](docs/parameters.md)
- [Model equations](docs/equations.md)
- [Troubleshooting](docs/troubleshooting.md)

## Installation

There are two ways to use HKL-PDF:

1. run the prebuilt Windows executable;
2. install and run the Python source code.

## Option 1: Windows executable

The prebuilt executable does not require the user to install Python or the
Python dependencies separately.

### From a GitHub release

The recommended method is:

1. Open the
   [HKL-PDF Releases page](https://github.com/MalaguttiMarcelo/HKL-PDF/releases).
2. Open the latest release.
3. Download the Windows executable or release archive.
4. If an archive was downloaded, extract it to a writable folder.
5. Double-click `HKL-PDF.exe`.

Do not run the executable directly from inside a ZIP archive. Extract all
files first so that any required bundled libraries remain beside the
executable.

### From a cloned repository

If the compiled `dist` directory is included in the repository, clone it:

```bash
git clone https://github.com/MalaguttiMarcelo/HKL-PDF.git
cd HKL-PDF
```

Then open the `dist` directory and double-click:

```text
HKL-PDF.exe
```

Depending on the PyInstaller build format, the executable may instead be
located at:

```text
dist/HKL-PDF/HKL-PDF.exe
```

From Windows PowerShell, it can be started with one of the following commands:

```powershell
.\dist\HKL-PDF.exe
```

or:

```powershell
.\dist\HKL-PDF\HKL-PDF.exe
```

Use the path that exists in the downloaded distribution.

> [!NOTE]
> The Windows executable only runs on Windows. Linux and macOS users should
> install and run HKL-PDF from the Python source.

### Windows security warning

Because the executable may not be digitally signed, Windows SmartScreen may
display a warning.

Only continue if the executable was downloaded from the official HKL-PDF
GitHub repository or its associated Zenodo release.

If SmartScreen appears:

1. click **More info**;
2. verify that the file came from the official repository;
3. click **Run anyway** if you trust the download.

## Option 2: Install from Python source

### Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- A working OpenGL installation for PyVista/VTK visualization

The main Python dependencies include:

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

### Clone the repository

```bash
git clone https://github.com/MalaguttiMarcelo/HKL-PDF.git
cd HKL-PDF
```

### Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the package

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install HKL-PDF and its declared dependencies:

```bash
python -m pip install -e .
```

If the optional visualization dependencies were not installed automatically:

```bash
python -m pip install pyvista pyvistaqt vtk qtpy
```

### Start the application

```bash
python -m pdf_fitting.gui
```

## Quick start

1. Start HKL-PDF.
2. Select a crystallographic structure file, normally a CIF.
3. Select an experimental two-column $G(r)$ data file.
4. Enter the fitting range.
5. Select the parameters to refine.
6. Give the refinable parameters physically meaningful bounds.
7. Click **Calculate** to inspect the initial model.
8. Click **Run** to start conventional refinement.
9. For an explicit finite crystallite, use **Run finite-shape fit** or
   **Run CS grid search**.
10. Inspect $R_\mathrm{wp}$, the residual, and parameter evolution before
    saving the refined values.

## Experimental data format

The experimental PDF file must contain at least two numeric columns:

```text
r        G(r)
1.0000   -0.1532
1.0100   -0.1498
1.0200   -0.1421
```

The first column is the pair distance $r$, normally in Å. The second column
is the experimental reduced PDF $G(r)$.

## Examples

Example input files and their associated structure and experimental data are
provided in the [`examples`](examples) directory.

The examples demonstrate:

1. ball-milled Fe-Cr using spherical size, local correlations, and Wilkens
   microstrain;
2. ball-milled calcite using spherical size, pair-specific local correlations,
   and PAH microstrain;
3. cylindrical Pd using a direction-dependent cylinder/disk envelope and an
   optional SAXS belly.

Open the `README.md` file inside each example directory for model-specific
instructions.

> [!NOTE]
> Example input files should use relative paths. If a file still contains an
> absolute path from the development computer, update `[files]` before running
> it.

## First-run performance

The first calculation may take longer because HKL-PDF:

- compiles Numba numerical kernels;
- generates crystallographic pair shells;
- creates reusable shell-cache files.

Subsequent calculations with the same structure and pair cutoff should be
faster.

## User cache

HKL-PDF stores generated cache files in a writable user directory.

On Windows, this is normally:

```text
%LOCALAPPDATA%\HKL-PDF
```

Important cache directories include:

```text
shell_cache
finite_shape_cache
```

These generated caches do not need to be committed to Git.

## Building the executable

The repository contains a PyInstaller specification file:

```text
HKL-PDF.spec
```

Install PyInstaller in the build environment:

```bash
python -m pip install pyinstaller
```

Build the application from the repository root:

```bash
pyinstaller --clean HKL-PDF.spec
```

The generated executable or application directory will be written to:

```text
dist/
```

The exact layout depends on whether the PyInstaller specification creates a
one-file or one-folder build.

## Troubleshooting

### The application starts slowly the first time

This is normally caused by Numba compilation or shell-cache generation.

### The structure viewer does not appear

Update the visualization dependencies and graphics driver:

```bash
python -m pip install --upgrade pyvista pyvistaqt vtk qtpy
```

### The input contains invalid file paths

Update the `[files]` section:

```ini
[files]
structure_file = path/to/structure.cif
gr_data_file = path/to/experimental_gr.xy
```

See the complete [troubleshooting guide](docs/troubleshooting.md) for more
information.

## Citation

If you use HKL-PDF in research, cite the software using the metadata in
[`CITATION.cff`](CITATION.cff).

After a Zenodo DOI has been created, the DOI and citation instructions should
also be added here.

For reproducible research, cite the DOI corresponding to the exact software
release used for the analysis.

## License

HKL-PDF is distributed under the terms stated in the
[`LICENSE`](LICENSE) file.

## Contributing

Bug reports, documentation corrections, and code contributions are welcome.

Before contributing, read [`CONTRIBUTING.md`](CONTRIBUTING.md).