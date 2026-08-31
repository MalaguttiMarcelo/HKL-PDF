# Spherical Crystallite-Size Model

The spherical model applies the common volume of two identical spheres
separated by distance $r$.

## Monodisperse sphere

For diameter $D$,

$$
\gamma(r)=
1-\frac{3r}{2D}
+\frac{r^3}{2D^3},
\qquad r<D,
$$

and

$$
\gamma(r)=0,\qquad r\ge D.
$$

The PDF is

$$
G(r)=
s\,4\pi\rho_0r\,
\gamma(r)[g(r)-1].
$$

## Input

```ini
[crystallite_shape]
mode = conventional

[initial_values]
d = 100.0
d_std = 10.0
```

- `d` is the real-space mean diameter in Å.
- `d_std` is the real-space standard deviation in Å.

If `d <= 0`, spherical size damping is disabled.

## Distribution

When `d_std > 0`, HKL-PDF uses a lognormal-averaged shape function.

When `d_std = 0`, the model is monodisperse.

## Advantages

- fast;
- stable;
- only one or two parameters;
- suitable for approximately equiaxed particles.

## Limitations

The model cannot represent rods, platelets, faceting, or direction-dependent
pair truncation.