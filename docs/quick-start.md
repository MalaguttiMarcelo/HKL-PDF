# Quick Start

## 1. Prepare the experimental PDF

The experimental data file must contain two numeric columns:

```text
r    G(r)
```

For example:

```text
1.0000  -0.1532
1.0100  -0.1498
1.0200  -0.1421
```

The first column is normally $r$ in Å. The second column is the
experimental reduced PDF $G(r)$.

## 2. Prepare the structure

A CIF file is recommended. Other structure formats supported by pymatgen may
also be used.

## 3. Start HKL-PDF

```bash
python -m pdf_fitting.gui
```

## 4. Create an input file

In the **Builder** tab:

1. select the structure file;
2. select the experimental $G(r)$ file;
3. enter `r_min` and `r_max`;
4. choose the crystallite-size model;
5. choose the microstrain model;
6. enter initial parameter values;
7. select the parameters to refine;
8. assign physically meaningful bounds;
9. save the input file.

## 5. Calculate the initial model

Click **Calculate** before starting a refinement.

This performs one PDF calculation without changing the parameters. Check:

- peak positions;
- peak widths;
- total scale;
- high-$r$ damping;
- baseline behaviour;
- the displayed $R_\mathrm{wp}$.

Do not begin refinement from a clearly unreasonable model.

## 6. Refine gradually

A recommended order is:

1. `scale`;
2. independent lattice parameters;
3. element-level `biso_*`;
4. crystallite-size parameters;
5. local correlated-motion parameters;
6. microstrain parameters;
7. contrast-factor coefficients only when physically justified.

Enable staged refinement with:

```ini
[refinement]
staged_refinement = true
```

## 7. Inspect the result

After refinement, inspect:

- $R_\mathrm{wp}$;
- the residual;
- parameter evolution;
- parameters that reached their bounds;
- the physical plausibility of the result;
- pair contributions, if calculated;
- Warren and local-dynamics plots, if requested.

## Minimal input

```ini
[files]
structure_file = structure.cif
gr_data_file = experimental_gr.dat

[range]
r_min = 1.0
r_max = 30.0

[pair_generation]
r_extension = 1.2

[initial_values]
scale = 1.0
a = 3.50
alpha = 90
beta = 90
gamma = 90
qdamp = 0.04
qmax = 25.0
d = 80.0
d_std = 0.0
biso_Fe = 0.5

[refinable_parameters]
scale = true
a = true
qdamp = false
qmax = false
d = true
d_std = false
biso_Fe = true

[bounds]
scale = 0.0, 10.0
a = 3.0, 4.0
d = 1.0, 1000.0
biso_Fe = 0.0, 10.0

[refinement]
staged_refinement = true
loss = soft_l1
f_scale = 0.05
max_nfev_stage = 25
max_nfev_final = 60
ftol = 0.001
xtol = 0.001
gtol = 0.001

[crystallite_shape]
mode = conventional
```

## Good practice

- Refine as few parameters as possible.
- Use bounds based on physical knowledge.
- Keep `qmax` consistent with the experimental data reduction.
- Avoid activating multiple competing broadening models unnecessarily.
- Preserve the original input before replacing it with refined values.