# HKL-PDF

HKL-PDF is a Python application for calculating and refining atomic pair
distribution function data, $G(r)$, using crystallographic structures,
crystallite-size models, microstrain models, and optional
missing-low-$Q$/SAXS baseline modelling.

## Download

**Windows:** [download the installer from the latest release](https://github.com/MalaguttiMarcelo/HKL-PDF/releases/latest)
(`HKL-PDF-Setup-<version>.exe` under *Assets*), double-click it, and follow
the steps. No Python, terminal, or administrator rights are needed. See
[Option 1](#option-1-windows-installer) below for details.

Linux and macOS: [install from the Python source](#option-2-install-from-python-source).

> [!IMPORTANT]
> HKL-PDF is early-stage research software. Validate fitted parameters,
> residuals, parameter correlations, and physical assumptions independently
> before using the results in publications.

## Features

- Project window with several datasets and several phases per dataset
- JSON project files that also keep the last fit and esds
- Interactive structure editor with symmetry-preserving site edits
- Refinement of site coordinates and occupancies
- Anisotropic displacement parameters ($\beta_{ij}$) with thermal ellipsoids
- Esds and correlation matrix of refined parameters (optional)
- Free-peak fit and peak-width analysis per crystallographic direction
- Deterministic and stochastic (Markov) layer stacking and stacking faults
- Optional NVIDIA GPU acceleration through CuPy
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
- [Project files (JSON)](docs/project-format.md)
- [Input-file format (.inp)](docs/input-format.md)
- [Structure editor](docs/structure-editor.md)
- [Anisotropic ADPs](docs/adp.md)
- [Esds and correlation matrix](docs/uncertainties.md)
- [Free-peak fit and widths per direction](docs/free-peaks.md)
- [Layer stacking and stacking faults](docs/stacking-faults.md)
- [GPU acceleration and performance](docs/gpu.md)
- [Parameter reference](docs/parameters.md)
- [Model equations](docs/equations.md)
- [Troubleshooting](docs/troubleshooting.md)

## Installation

There are two ways to use HKL-PDF:

1. run the prebuilt Windows executable;
2. install and run the Python source code.

## Option 1: Windows installer

1. Open the
   [HKL-PDF Releases page](https://github.com/MalaguttiMarcelo/HKL-PDF/releases).
2. Download `HKL-PDF-Setup-<version>.exe` from the latest release.
3. Run it. No administrator rights are needed: HKL-PDF is installed for the
   current user in `%LOCALAPPDATA%\Programs\HKL-PDF` (an "all users"
   installation can be chosen instead).
4. Only on computers with an NVIDIA graphics card: tick **Download and
   install NVIDIA GPU support**. The installer then downloads CuPy and the
   needed CUDA libraries (about 700 MB) from the Python Package Index. GPU
   support can also be added later with **File › NVIDIA GPU support…** in
   HKL-PDF.
5. At the end, the installer prepares the numerical kernels (about a
   minute), so the first calculation is not slowed down.

Start HKL-PDF from the Start menu. The examples are reachable from the
**HKL-PDF examples** Start-menu entry.

### Windows security warning

The installer is not digitally signed, so Windows SmartScreen may show a
warning. Only continue if the file came from the official HKL-PDF GitHub
repository: click **More info**, then **Run anyway**.

### Fast start: keep HKL-PDF loaded

Loading the scientific libraries takes time. By default, closing the HKL-PDF
window keeps the program loaded in the background (tray icon near the
clock), so the next start opens a window in about a second. It stays loaded
until **Quit HKL-PDF** (File menu or tray icon) or a restart of the
computer.

Both options are in the **File** menu and in the tray-icon menu:

| Option | Default | Effect |
|---|---|---|
| Keep HKL-PDF loaded after closing | on | Closing the last window leaves HKL-PDF in the tray |
| Start HKL-PDF at Windows login | off | Loads HKL-PDF in the background at login, so the first start after a restart is fast too |

Keeping HKL-PDF loaded uses memory (roughly 0.5–1 GB) while it waits.

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

### Optional: NVIDIA GPU support

```bash
python -m pip install cupy-cuda12x
```

Choose the CuPy package that matches your CUDA version. Then select **CUDA**
in **Refinement settings › Backends**. See
[GPU acceleration](docs/gpu.md).

### Start the application

```bash
python -m pdf_fitting.gui                 # project window
python -m pdf_fitting.gui project.json    # open a project or .inp file
python -m pdf_fitting.gui --classic       # previous single-phase window
```

## Quick start

1. Start HKL-PDF.
2. Click **+ Dataset** and select an experimental two-column $G(r)$ file.
3. Click **+ Phase (CIF)** and select the structure. Repeat for more phases.
4. Set the fitting range in **Data & range** and `qmax` in **Instrument**.
5. Tick **Refine** for the parameters to refine and give them physically
   meaningful bounds.
6. Press **Calculate** (F5) to inspect the starting model.
7. Press **Refine** (F6).
8. Inspect $R_\mathrm{wp}$, the residual, and the parameter evolution. Turn
   on esds in **Refinement settings** to see the correlation matrix.
9. Save the project (Ctrl+S) as a `.json` file next to the data.

Older `.inp` files can be opened directly.

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

Example projects (`*.hklpdf.json`), input files (`*.inp`), and their
structure and experimental data are provided in the
[`examples`](examples) directory. Paths are relative, so the examples run
from any location.

The examples demonstrate:

1. ball-milled Fe-Cr using spherical size, local correlations, and Wilkens
   microstrain;
2. ball-milled calcite using spherical size, pair-specific local correlations,
   and PAH microstrain;
3. cylindrical Pd using a direction-dependent cylinder/disk envelope and an
   optional SAXS belly.

Open the `README.md` file inside each example directory for model-specific
instructions.

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
cache/stochastic_topology
edited_structures
```

These generated caches do not need to be committed to Git.

## Building the Windows installer

The installer is built by one script, from a build environment on a local
disk (building from a network drive is very slow):

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
```

It needs Python 3.14 (`py -3.14`) and
[Inno Setup 6](https://jrsoftware.org/isinfo.php). The script:

1. creates `%LOCALAPPDATA%\HKL-PDF-build\venv` with the pinned packages of
   `installer/requirements-build.txt`;
2. builds the program folder with PyInstaller (`HKL-PDF.spec`; a folder
   build, not a single file, so nothing is unpacked at every start);
3. removes CuPy and every CUDA library from the program folder (GPU
   support is downloaded by the user on request; the list of exact files
   and checksums is `pdf_fitting/gpu_packages.json`);
4. checks the built program with `HKL-PDF.exe --warmup`;
5. writes `installer-output/HKL-PDF-Setup-<version>.exe`.

Options: `-Recreate` rebuilds the environment, `-NoInstaller` stops after
the program folder, `-BuildRoot <dir>` changes the build folder, `-TestGpu`
also downloads the GPU support and tests it with the built program.

### Publishing a release

Do not commit the installer. Attach it to a GitHub Release, where each file
may be up to 2 GB (files committed to the repository are limited to
100 MB). Create the release on GitHub, or with the GitHub CLI:

```powershell
gh release create v<version> installer-output\HKL-PDF-Setup-<version>.exe --title "HKL-PDF <version>" --notes-file docs\changelog.md
```

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