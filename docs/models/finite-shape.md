# Finite Crystallite-Shape Model

Finite-shape mode constructs explicit finite coordination tables for a
selected crystallite geometry.

## Example

```ini
[crystallite_shape]
mode = finite_shape
search_mode = predefined
shape_type = cylinder

diameter_cells = 20.0
height_cells = 5.0

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

## Pair catalogue

A shape-independent all-site catalogue stores:

- center-site index;
- neighbour-site index;
- periodic image;
- fractional pair vector;
- pair distance;
- pair IDs;
- species IDs;
- $hkl$;
- reduced direction.

## Shape mask

Each translated atom is tested against the shape in Cartesian coordinates.

## Finite coordination

A catalogue pair contributes only when its center and neighbour are both
inside the particle.

The effective finite multiplicity is

$$
m_{\mathrm{finite}}^{\mathrm{eff}}=
\frac{N_{\mathrm{finite\ pairs}}}
{N_{\alpha,\mathrm{inside}}}.
$$

## Baseline gamma

$$
\gamma_{\mathrm{finite}}=
\frac{
m_{\mathrm{finite}}^{\mathrm{eff}}
}{
m_{\mathrm{bulk}}^{\mathrm{eff}}
}.
$$

Peak amplitudes use finite coordination directly. Gamma is used for the
finite-particle baseline.

## Grid search

Grid-search mode evaluates a set of discrete diameter and height candidates.
Each candidate receives a continuous least-squares refinement.

## Cache

The pair catalogue and finite shell tables are cached in the user data
directory.