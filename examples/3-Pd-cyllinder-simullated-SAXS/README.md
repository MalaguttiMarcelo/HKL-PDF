# Pd Cylinder/Disk with SAXS Belly Example

This example demonstrates PDF fitting of an FCC palladium crystallite using:

- a direction-dependent cylinder/disk common-volume model;
- an optional missing-low-$Q$/SAXS belly;
- conventional periodic PDF shell generation;
- cubic symmetry constraints;
- isotropic microstrain mode.

The experimental file name indicates a cylindrical Pd dataset with nominal
dimensions of approximately 250 Å in diameter and 101.15 Å in height. The
values in the input file are the current fitting values and may differ from
the nominal dimensions.

## Required files

The input currently refers to:

```text
config/Pd.cif
data/cylinder_d250_h101.15_Biso0p3_SAXS.xy
```

Before running the example, update the paths in `[files]` if the repository is
stored in a different location.

For a portable repository layout, relative paths are recommended. For example,
if the input file is inside an example subdirectory:

```ini
[files]
structure_file = ../../config/Pd.cif
gr_data_file = ../../data/cylinder_d250_h101.15_Biso0p3_SAXS.xy
```

Adjust the number of `../` components to match the actual location of the
example input file.

## Model summary

The calculation uses conventional periodic shell generation:

```ini
[crystallite_shape]
mode = conventional
```

The main crystallite envelope is a cylinder/disk model:

```ini
[initial_values]
cyl_diameter = 189.88674471622278
cyl_thickness = 125.23731045807084
cyl_thickness_std = 0.0

cyl_axis_h = 0.0
cyl_axis_k = 0.0
cyl_axis_l = 1.0
```

Both `cyl_diameter` and `cyl_thickness` are refined:

```ini
[refinable_parameters]
cyl_diameter = true
cyl_thickness = true
cyl_thickness_std = false
```

The cylinder axis is the direct crystallographic direction

$
[0\ 0\ 1].
$

Because the thickness standard deviation is zero, this example uses a
monodisperse cylinder thickness.

## SAXS belly configuration

The missing-low-$Q$/SAXS contribution is enabled:

```ini
[refinement]
saxs_enabled = true
saxs_model = same_as_size_model
saxs_apply_qdamp = false
saxs_apply_qmax = false
```

The `same_as_size_model` option makes the SAXS belly use the average
common-volume function of the main cylinder/disk model.

The independent SAXS dimensions are set to zero:

```ini
[initial_values]
saxs_scale = 1.0
saxs_diameter = 0.0
saxs_diameter_std = 0.0
saxs_height = 0.0
saxs_height_std = 0.0
```

Therefore, no separate SAXS particle dimensions are used. The SAXS term
follows the main cylinder/disk dimensions.

The belly contribution is

$
G_{\mathrm{SAXS}}(r)=
-s\,s_{\mathrm{SAXS}}\,
4\pi\rho_0r\,
\gamma_{\mathrm{cyl}}(r),
$

where $s$ is the PDF scale, $s_{\mathrm{SAXS}}$ is `saxs_scale`, and
$\gamma_{\mathrm{cyl}}(r)$ is the powder-averaged cylinder common-volume
function.

In this example:

- `saxs_apply_qdamp = false` prevents `qdamp` from damping the SAXS belly;
- `saxs_apply_qmax = false` prevents the finite-$Q_{\max}$ filter from being
  applied to the SAXS belly;
- the structural PDF is still calculated using the configured `qmax`.

## Structural parameters

Palladium is treated as cubic. Only `a` is independently refined:

```ini
[refinable_parameters]
a = true
b = false
c = false
alpha = false
beta = false
gamma = false
```

Cubic symmetry imposes:

$
b=a,\qquad c=a,
$

and

$
\alpha=\beta=\gamma=90^\circ.
$

The isotropic displacement parameter is fixed:

```ini
[initial_values]
biso_Pd = 0.3

[refinable_parameters]
biso_Pd = false

[bounds]
biso_Pd = 0, 10
```

## Instrument settings

The example uses:

```ini
[initial_values]
qdamp = 0.0
delta_broad = 0.0
qmax = 40.0
qmax_zeros = 5.0
```

The selected fitting interval is:

```ini
[range]
r_min = 2.0
r_max = 200.0
```

Pair shells are generated beyond the fitted range using:

```ini
[pair_generation]
r_extension = 1.2
```

This gives an approximate pair cutoff of

$
r_{\mathrm{cut}}
=
r_{\max}r_{\mathrm{extension}}
=
200\times1.2
=
240\ \text{Å}.
$

## Refined parameters

The active refinable parameters are:

- `scale`;
- cubic lattice parameter `a`;
- `cyl_diameter`;
- `cyl_thickness`.

All other parameters are fixed in this example.

## Running the example

Start HKL-PDF:

```bash
python -m pdf_fitting.gui
```

Then:

1. click **Load input**;
2. select the example input file;
3. verify that the Pd CIF and experimental PDF paths are valid;
4. click **Calculate** to inspect the initial model;
5. click **Run** to start the conventional refinement;
6. inspect the fitted PDF, residual, $R_\mathrm{wp}$, and parameter
   evolution;
7. save the refined input only after checking that the dimensions and lattice
   parameter are physically reasonable.

Because this is a conventional cylinder/disk model, use **Run**, not the
finite-shape grid-search button.

## Expected behaviour

During refinement:

- `scale` changes the overall PDF amplitude;
- `a` changes the positions of the Pd pair peaks;
- `cyl_diameter` controls correlations perpendicular to the selected axis;
- `cyl_thickness` controls correlations parallel to the selected axis;
- the SAXS belly follows the average envelope of the same cylinder/disk model.

The cylinder dimensions can be correlated with scale and the SAXS belly.
Interpret the result using the complete residual and parameter evolution, not
only the final $R_\mathrm{wp}$.

## Notes

- The dataset extends to a large $r_{\max}$, so initial shell generation may
  take time.
- Later calculations should be faster after the shell cache has been created.
- `staged_refinement` is disabled, so all selected parameters are refined
  together.
- `delta_g = 0` and is fixed, so no isotropic microstrain broadening is fitted.
- Pair contributions, Warren plots, and local-dynamics plots are disabled for
  faster finalization.