# Graphical Interface

HKL-PDF opens the **project window**, which supports several datasets and
several phases per dataset. The previous single-phase window is still
available as the **classic window**.

```bash
python -m pdf_fitting.gui                    # project window
python -m pdf_fitting.gui project.json       # ... and open a project
python -m pdf_fitting.gui old_input.inp      # ... or import an .inp file
python -m pdf_fitting.gui --classic          # classic single-phase window
```

The classic window can also be opened from **View › Classic window**.

## Start-up and background mode

A loading screen shows which library is being loaded. All heavy libraries
are imported before the main window opens, so later actions do not wait.

By default, closing the window keeps HKL-PDF loaded in the background,
with a tray icon near the clock. The next start, including double-clicking
a project, opens a window in about a second. HKL-PDF stays loaded until
**File › Quit HKL-PDF** (or **Quit HKL-PDF** in the tray-icon menu) or a
restart of the computer.

| Option (File menu and tray menu) | Default |
|---|---|
| Keep HKL-PDF loaded after closing | on |
| Start HKL-PDF at Windows login | off |

**File › Close window** closes only the window; **File › Quit HKL-PDF**
also unloads the program.

## Layout

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ HKL-PDF · project │ +Dataset · +Phase · Calculate · Refine · Stop │ chips│
├──────────────────────┬──────────────────────────────────────────────────┤
│ Project tree         │ Editor page for the selected tree node           │
│                      │                                                  │
└──────────────────────┴──────────────────────────────────────────────────┘
```

- The **toolbar** holds the dataset, phase, and calculation actions.
  **Open** and **Save** are in the **File** menu.
  Shortcuts: **F5** Calculate, **F6** Refine, **Esc** Stop, **Ctrl+O** Open,
  **Ctrl+S** Save.
- The **chips** show the worker state, the current $R_\mathrm{wp}$, the
  number of parameters flagged for refinement, and the selected shell and
  PDF backends (CPU or CUDA).
- The **project tree** on the left selects the editor page on the right.
  The number beside a node is the count of parameters flagged for
  refinement below it.

## Refinement progress

The first node of the tree, **Refinement progress**, plots the best
$R_\mathrm{wp}$ and the phase weight fractions against the
function-evaluation count, for all phases. When **Refine** is pressed, the
window switches to this page. The node is highlighted while the refinement
runs and shows the current $R_\mathrm{wp}$ beside it.

## Project tree

```text
Refinement progress
Refinement settings
Constraints
Refined parameters
<Dataset>
├── Data & range
├── Instrument
└── <Phase>
    ├── Structure (3D)
    ├── Lattice
    ├── Atoms & occupancy
    ├── Microstructure
    ├── Local dynamics
    ├── Shape & stacking
    ├── SAXS
    ├── Peak widths (free fit)
    └── Results
```

Right-click a dataset or phase for more actions: add a phase from a CIF or
an `.inp` file, duplicate a dataset with the same phases, duplicate a phase,
copy a phase to the other datasets, or remove it.

Phases can be made inactive. An inactive phase stays in the project but is
not calculated.

## Parameter grid

Every page that holds parameters uses the same grid:

| Column | Meaning |
|---|---|
| Value | Current value |
| Refine | Include the parameter in the least-squares fit |
| Min / Max | Bounds used during refinement |
| Esd | Estimated standard deviation from the last esd calculation |

## Pages

### Refinement settings

These settings apply to the whole project:

- **Strategy and tolerances:** staged refinement, evaluation budgets,
  `ftol`, `xtol`, `gtol`, the coarse refinement $r$ step, and the robust
  loss function.
- **Backends:** CPU or CUDA for shell generation and PDF accumulation, the
  GPU device, the GPU memory fraction, and the number of Numba threads.
  See [GPU acceleration](gpu.md).
- **After the fit:** pair contributions, Warren analysis, local trends,
  export of the finite coordination, and the
  [esd and correlation-matrix calculation](uncertainties.md).

### Constraints

Constraints tie one parameter to an expression of others. Names are
namespaced by dataset and phase, for example:

```text
ds1.fe_bcc_300k.re = ds1.fe_bcc_300k.d
```

See [Project files](project-format.md#constraints).

### Refined parameters

This page lists every refined parameter with its value, bounds, and esd.

After an esd calculation, the page also shows:

- the correlation-matrix heat map;
- a table of the strongest correlations.

**Compute esds now** calculates them for the current parameter values
without refining again.

### Dataset pages

- **Dataset node:** the fit plot of the dataset (all phases).
- **Data & range:** the experimental file, $r_\mathrm{min}$,
  $r_\mathrm{max}$, and the pair-generation extension.
- **Instrument:** `qdamp`, `delta_broad`, `qmax`, and the $Q_\mathrm{max}$
  termination options. These are shared by every phase of the dataset.

### Phase pages

- **Overview:** the scale factor and the derived weight fraction.
- **Structure (3D):** the interactive structure editor.
  See [Structure editor](structure-editor.md).
- **Lattice:** the cell parameters. Symmetry constraints such as $b = a$
  are applied automatically.
- **Atoms & occupancy:** coordinates, occupancies, Biso, and anisotropic
  displacement parameters. See [Anisotropic ADPs](adp.md).
- **Microstructure:** crystallite size, cylinder or disk size, the
  microstrain model, and contrast factors. Laue-class invariant terms are
  listed by name, and **Import contrast factors…** reads them from a file.
  See [Microstrain](microstrain.md).
- **Local dynamics:** the correlated-motion parameters
  $\delta_1$, $\delta_2$, and the per-pair $\lambda$ values.
- **Shape & stacking:** opens the crystallite and stacking builder, switches
  back to the conventional (bulk) model, or starts a crystallite-shape grid
  search. See [Stacking faults](stacking-faults.md).
- **SAXS:** the optional missing-low-$Q$ term. See
  [SAXS model](saxs-model.md).
- **Peak widths (free fit):** fits every peak freely and analyses the
  widths per crystallographic direction. See
  [Free-peak fit](free-peaks.md).
- **Results:** the plots of the phase: fit, pair contributions, Warren,
  local trends, size distribution, and CS grid.

## Fit plot

The fit plot shows the experimental and calculated $G(r)$ and the offset
residual. It can also show:

- pair contributions and their sums;
- shell tick markers, filtered by pair, direction, and $r$ range;
- the shape or SAXS term.

When a neighbour is selected in the structure editor, the matching pair is
marked on the fit plot with tick markers.

The plot can be popped out into its own window. When it is popped out, its
usual place shows **Bring the plot back here** and **Show the plot
window** buttons.

## Saved fits

When a project is saved, the last calculated curves and the last esd
results are stored in the project file. They are shown again when the
project is opened, so a refinement does not need to be repeated just to
see it. Press **Calculate** to update them.

## Performance recommendations

For large calculations:

- calculate pair contributions, Warren plots, and local trends only when
  needed;
- leave tick markers off until required;
- use a coarse refinement $r$ step and a fine final calculation;
- consider the CUDA backends for large structures and stacking models.

## Classic window

The classic window keeps the original layout: a **Builder** and a raw-input
editor on the left, and plots on the right. It reads and writes single-phase
`.inp` files. Its actions are **Calculate**, **Run**,
**Run finite-shape fit**, **Run CS grid search**, **Pause**, and
**Stop/Save**.

Projects opened from `.inp` files in the project window are converted to a
one-dataset, one-phase project. Save them as `.json` to keep the
multi-phase features.
