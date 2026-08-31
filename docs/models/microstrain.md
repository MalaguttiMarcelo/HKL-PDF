# Microstrain Models

HKL-PDF supports isotropic, Wilkens, and PAH microstrain models.

## Isotropic model

```ini
[refinement]
microstrain_model = isotropic

[initial_values]
delta_g = 0.001
```

The broadening variance is

$
\sigma_{\mathrm{iso}}^2=
(\delta_gL)^2.
$

## Wilkens model

```ini
[refinement]
microstrain_model = wilkens

[initial_values]
rho = 0.0002
re = 150
fe = 0.5
burgers_mag = 2.5
```

The model uses

$
\langle\varepsilon^2(L)\rangle=
\frac{\rho b^2}{4\pi}
C_{hkl}f^\ast(L/R_e),
$

$
\sigma_{\mathrm{W}}^2=
L^2\langle\varepsilon^2(L)\rangle.
$

Reduced crystallographic directions are used so that parallel orders share
the same directional contrast factor.

## Cubic contrast factors

$
C_{hkl}=A+B
\frac{h^2k^2+k^2l^2+l^2h^2}
{(h^2+k^2+l^2)^2}.
$

Separate edge and screw values are mixed with `fe`.

## Non-cubic invariant expansion

$
C_{hkl}=
\sum_iE_iX_i(h,k,l).
$

The number and form of invariant terms depend on space-group symmetry.

## PAH model

```ini
[refinement]
microstrain_model = pah

[initial_values]
pah_a = 0.0001
pah_b = 0.00001
fe = 0.5
```

The variance is

$
\sigma_{\mathrm{PAH}}^2=
I_{hkl}
(a_{\mathrm{PAH}}L+b_{\mathrm{PAH}}L^2).
$

Negative trial variances are clipped to zero.