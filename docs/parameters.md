# Parameter Reference

Units assume distances are expressed in Å.

## Structural parameters

| Parameter | Unit | Description |
|---|---:|---|
| `scale` | – | Overall PDF scale. |
| `a`, `b`, `c` | Å | Unit-cell lengths. |
| `alpha`, `beta`, `gamma` | degrees | Unit-cell angles. |

Symmetry determines which lattice parameters are independent.

## Instrumental parameters

| Parameter | Unit | Description |
|---|---:|---|
| `qdamp` | Å$^{-1}$ | Real-space damping due to reciprocal-space resolution. |
| `qmax` | Å$^{-1}$ | Maximum momentum transfer for termination effects. |
| `qmax_zeros` | – | Number of kernel zero intervals retained. |
| `delta_broad` | – | Additional distance-dependent peak broadening. |

`qbroad` and `eta` are deprecated by the current GUI.

## Spherical crystallite size

| Parameter | Unit | Description |
|---|---:|---|
| `d` | Å | Mean spherical particle diameter. |
| `d_std` | Å | Real-space diameter standard deviation. |
| `d_mu` | – | Optional legacy direct lognormal location. |

A non-positive `d` disables spherical size damping.

## Cylinder and disk size

| Parameter | Unit | Description |
|---|---:|---|
| `cyl_diameter` | Å | Cylinder or disk diameter. |
| `cyl_thickness` | Å | Axial thickness or height. |
| `cyl_thickness_std` | Å | Thickness-distribution standard deviation. |
| `cyl_axis_h` | – | First index of the direct-space cylinder axis. |
| `cyl_axis_k` | – | Second index of the direct-space cylinder axis. |
| `cyl_axis_l` | – | Third index of the direct-space cylinder axis. |
| `cyl_mu` | – | Optional direct lognormal thickness location. |
| `cyl_sigma` | – | Optional direct lognormal thickness width. |

The model is active only when both diameter and thickness are positive.

## Isotropic displacement parameters

Element-level parameters use names such as:

```text
biso_C
biso_O
biso_Fe
biso_Co
```

Site-labelled parameters use names such as:

```text
biso_c1
biso_o2
biso_fe3
```

Default safety bounds are 0–10 Å².

> Current implementation: site-labelled Biso values are averaged by element
> before shell broadening. Fully independent site-to-site Biso broadening is
> not yet implemented.

## Correlated motion

| Parameter | Description |
|---|---|
| `delta1` | Global $1/r$ correlation coefficient. |
| `delta2` | Global $1/r^2$ correlation coefficient. |
| `delta1_A-B` | Pair-specific `delta1`. |
| `delta2_A-B` | Pair-specific `delta2`. |
| `lambda_A-B_k` | Correlation coefficient for pair A–B and distance rank $k$. |
| `lambda_k` | Legacy global shell coefficient. |

## Isotropic microstrain

| Parameter | Description |
|---|---|
| `delta_g` | Distance-proportional isotropic Gaussian broadening. |

## Wilkens parameters

| Parameter | Description |
|---|---|
| `rho` | Dislocation-density parameter used by the Wilkens expression. |
| `re` or `Re` | Effective Wilkens outer radius. |
| `fe` or `fE` | Edge fraction. Screw fraction is $1-f_E$. |
| `burgers_mag` | Burgers-vector magnitude in Å. |

If `burgers_mag` is absent, the compatibility default is

$
b=\frac{\sqrt{3}}{2}a.
$

Provide `burgers_mag` when this assumption is not appropriate.

## PAH parameters

| Parameter | Description |
|---|---|
| `pah_a` | Linear-in-distance PAH variance coefficient. |
| `pah_b` | Quadratic-in-distance PAH variance coefficient. |
| `fe` | Edge/screw mixing fraction when contrast factors are used. |

## Cubic contrast factors

| Parameter | Description |
|---|---|
| `CEdgeA` | Cubic edge A coefficient. |
| `CEdgeB` | Cubic edge B coefficient. |
| `CScrewA` | Cubic screw A coefficient. |
| `CScrewB` | Cubic screw B coefficient. |

## Non-cubic contrast factors

Non-cubic systems use invariant coefficients:

```text
EdgeE1, EdgeE2, ...
ScrewE1, ScrewE2, ...
```

The number of coefficients depends on space-group symmetry.

## SAXS belly parameters

| Parameter | Unit | Description |
|---|---:|---|
| `saxs_scale` | – | Independent amplitude of the SAXS belly. |
| `saxs_diameter` | Å | Independent SAXS diameter. |
| `saxs_diameter_std` | Å | Diameter-distribution standard deviation. |
| `saxs_height` | Å | Cylinder/disk SAXS height. |
| `saxs_height_std` | Å | SAXS height standard deviation. |

Activation and model selection belong in `[refinement]`:

```ini
saxs_enabled = true
saxs_model = sphere
saxs_apply_qdamp = false
saxs_apply_qmax = true
```

## Atomic-site parameters

The GUI can write:

```text
x_site
y_site
z_site
occ_site
biso_site
```

> Current limitation: refined `x_*`, `y_*`, `z_*`, and `occ_*` values are not
> yet used to regenerate packed pair vectors during every model evaluation.