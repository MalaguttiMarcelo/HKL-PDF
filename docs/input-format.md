# Input-File Format

HKL-PDF uses an INI-style text file.

## Files

The `[files]` section is required.

```ini
[files]
structure_file = structure.cif
gr_data_file = experimental_gr.dat
```

## Refinement range

```ini
[range]
r_min = 1.0
r_max = 30.0
```

Only experimental points inside this interval are fitted.

## Pair generation

```ini
[pair_generation]
r_extension = 1.2
```

The shell-generation cutoff is approximately

$$
r_{\mathrm{cut}} =
r_{\max}r_{\mathrm{extension}}.
$$

A larger value generates more pairs and requires more memory.

## Initial values

All model parameters belong in `[initial_values]`.

```ini
[initial_values]
scale = 1.0
a = 3.5
b = 3.5
c = 3.5
alpha = 90
beta = 90
gamma = 90
qdamp = 0.04
qmax = 25
d = 80
d_std = 5
biso_Fe = 0.5
```

Parameters used by the refinement engine must have numeric values.

## Refinable parameters

```ini
[refinable_parameters]
scale = true
a = true
d = true
biso_Fe = true
qmax = false
```

The older section name `[refinable_flags]` is also accepted.

## Bounds

Bounds are entered as `minimum, maximum`.

```ini
[bounds]
scale = 0, 10
a = 3.0, 4.0
d = 1, 1000
biso_Fe = 0, 10
```

Open bounds are allowed:

```ini
[bounds]
parameter1 = , 10
parameter2 = 0,
```

HKL-PDF applies default safety bounds to several parameter families:

- `a`, `b`, `c`: greater than or equal to 0.1 Å;
- lattice angles: 10–170°;
- `biso_*`: 0–10 Å²;
- `saxs_scale`: 0–5;
- SAXS dimensions: non-negative safety ranges.

Explicit user bounds override these defaults.

## Constraints

```ini
[constraints]
b = a
c = a
re = d
parameter3 = 2 * parameter1 + sqrt(parameter2)
```

A constrained parameter is derived and is not independently refined.

Allowed functions are:

- `abs`
- `min`
- `max`
- `sqrt`
- `log`
- `log10`
- `exp`

Basic arithmetic and powers are supported. Circular or unresolved
dependencies produce an error.

## Refinement settings

```ini
[refinement]
staged_refinement = true
loss = soft_l1
f_scale = 0.05

max_nfev_stage = 25
max_nfev_final = 60

ftol = 0.001
xtol = 0.001
gtol = 0.001

refinement_r_step = 0.05
refinement_r_step_final = 0.0

progress_every_sec = 0.5
update_pdf_during_refinement = true

microstrain_model = isotropic

compute_pair_contributions_on_finish = false
compute_warren_on_finish = false
compute_local_trends_on_finish = false
export_finite_coordination = false
```

A final refinement step of zero uses the full experimental grid.

## Conventional shape mode

```ini
[crystallite_shape]
mode = conventional
```

## Fixed finite shape

```ini
[crystallite_shape]
mode = finite_shape
search_mode = predefined
shape_type = cylinder

diameter_cells = 12
height_cells = 4

axis_h = 0
axis_k = 0
axis_l = 1

base1_h = 1
base1_k = 0
base1_l = 0

base2_h = 0
base2_k = 1
base2_l = 0
```

## Finite-shape grid search

```ini
[crystallite_shape]
mode = finite_shape
search_mode = grid_search
shape_type = cylinder

diameter_min_cells = 5
diameter_max_cells = 20
height_min_cells = 2
height_max_cells = 10
step_cells = 1

scan_diameter = true
scan_height = true
diameter_infinite = false
```

## Pair-specific parameters

Preferred lambda syntax:

```ini
[initial_values]
lambda_fe-o_0 = 0.10
lambda_fe-o_1 = 0.05
delta1_fe-o = 0.25
delta2_fe-o = 0.00
```

Legacy global lambdas remain supported:

```ini
lambda_0 = 0.10
lambda_1 = 0.05
```

## Contrast factors

Contrast factors are normal parameters:

```ini
[initial_values]
CEdgeA = 0.265280
CEdgeB = -0.355950
CScrewA = 0.307288
CScrewB = -0.819979
burgers_mag = 2.48
```

Non-cubic invariant coefficients use names such as:

```ini
EdgeE1 = 0.1
EdgeE2 = 0.2
ScrewE1 = 0.1
ScrewE2 = 0.2
```

The old `[contrast_factors]` section is deprecated.