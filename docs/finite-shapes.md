# Finite Crystallite Shapes

Finite-shape mode counts atom pairs inside an explicit particle boundary.

## Supported shapes

- sphere;
- ellipsoid;
- cylinder;
- disk;
- rectangular prism.

## Fixed shape

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

## Grid search

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

Each candidate receives a normal continuous least-squares refinement. Final
candidate quality is compared using $R_\mathrm{wp}$.

## Orientation

The axis and base vectors are direct crystallographic directions. They are
converted into Cartesian vectors using the current lattice and then
orthonormalized.

The height follows the selected axis. Base dimensions follow the selected
base directions.

## Atom-level boundary mode

The current finite-shape worker uses atom-level inclusion. Every translated
atom is tested against the shape boundary.

For a pair to contribute, both the center atom and corresponding neighbour
must be inside the particle.

## Finite effective coordination

$$
m_{\mathrm{finite}}^{\mathrm{eff}} =
\frac{N_{\mathrm{finite\ pairs}}}
{N_{\alpha,\mathrm{inside}}}.
$$

The finite baseline ratio is

$$
\gamma_{\mathrm{finite}} =
\frac{
m_{\mathrm{finite}}^{\mathrm{eff}}
}{
m_{\mathrm{bulk}}^{\mathrm{eff}}
}.
$$

Finite coordination controls peak amplitudes. The gamma ratio controls the
finite-particle baseline.

## Caching

HKL-PDF caches:

1. the all-site periodic pair catalogue;
2. the shape-specific finite shell table.

## Direction-resolved export

Enable:

```ini
[refinement]
export_finite_coordination = true
```

HKL-PDF writes direction-resolved gamma functions and detailed coordination
tables.

## Limitations

- Infinite-diameter slab mode is not implemented.
- Large particles can require substantial memory.
- Grid-search cost increases with the number of candidates.
- Finite dimensions are measured in crystallographic direction units.