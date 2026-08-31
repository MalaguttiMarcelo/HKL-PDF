# Crystallite-Size Modelling

Finite particle size reduces PDF intensity at large pair distances because
fewer atom pairs remain inside a particle.

HKL-PDF provides three size approaches.

## Spherical model

Parameters:

```ini
[initial_values]
d = 80
d_std = 5
```

This model is fast and suitable for approximately isotropic particles.

## Cylinder and disk model

Parameters:

```ini
[initial_values]
cyl_diameter = 100
cyl_thickness = 20
cyl_thickness_std = 2
cyl_axis_h = 0
cyl_axis_k = 0
cyl_axis_l = 1
```

The common-volume factor depends on the direction of every crystallographic
pair relative to the selected axis.

## Explicit finite shapes

Finite mode counts atom pairs inside a selected shape.

Supported geometry definitions include:

- sphere;
- ellipsoid;
- cylinder;
- disk;
- prism.

Finite mode supports either one fixed particle or a diameter/height grid
search.

## Lognormal size distributions

For a real-space mean $\bar D$ and standard deviation $s_D$,

$$
\sigma_{\ln}^{2} =
\ln\left(
1+\frac{s_D^2}{\bar D^2}
\right),
$$

$$
\mu_{\ln} =
\ln(\bar D)-\frac{\sigma_{\ln}^{2}}{2}.
$$

A zero standard deviation represents a monodisperse size.

## Model selection

Use:

- a sphere for approximately equiaxed particles;
- a cylinder for rods or wires;
- a disk for platelets;
- a finite shape for strong directional truncation or explicit faceting.

## Correlations

Particle dimensions may correlate with:

- `qdamp`;
- Biso;
- isotropic microstrain;
- Wilkens or PAH parameters;
- the SAXS belly;
- scale.

Use independent microscopy or diffraction information to choose initial
values and bounds whenever possible.