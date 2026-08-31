# Microstrain

Microstrain broadens PDF peaks as pair distance increases.

HKL-PDF provides three primary models.

## Isotropic microstrain

```ini
[refinement]
microstrain_model = isotropic

[initial_values]
delta_g = 0.001
```

The variance contribution is

$
\sigma_g^2=(\delta_gL)^2.
$

This model has no crystallographic direction dependence.

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
C_{hkl}
f^\ast(L/R_e).
$

The peak-variance contribution is

$
\sigma_{\mathrm{W}}^2=
L^2\langle\varepsilon^2(L)\rangle.
$

For cubic materials, contrast factors may use:

```ini
CEdgeA = 0.265280
CEdgeB = -0.355950
CScrewA = 0.307288
CScrewB = -0.819979
```

Non-cubic materials use symmetry-dependent `EdgeE*` and `ScrewE*`
coefficients.

## PAH model

```ini
[refinement]
microstrain_model = pah

[initial_values]
pah_a = 0.0001
pah_b = 0.00001
fe = 0.5
```

The implemented variance is

$
\sigma_{\mathrm{PAH}}^2=
I_{hkl}
(a_{\mathrm{PAH}}L+b_{\mathrm{PAH}}L^2).
$

## Warren plot

The Warren view can display

$
\sqrt{\langle\Delta L^2\rangle}
$

or

$
\frac{\sqrt{\langle\Delta L^2\rangle}}{L}.
$

Directions are reduced and grouped according to crystal-system symmetry.

## Recommendations

- Supply `burgers_mag` when the default BCC-like assumption is inappropriate.
- Use contrast factors from a physically justified calculation or source.
- Avoid refining many invariant coefficients from PDF data alone.
- Do not activate several competing microstrain models simultaneously.
- Inspect directional Warren curves, not only $R_\mathrm{wp}$.