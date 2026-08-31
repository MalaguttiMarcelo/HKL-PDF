# Cylinder and Disk Crystallite-Size Model

The cylinder/disk common-volume model uses a direction-dependent particle
envelope.

It becomes active when both `cyl_diameter` and `cyl_thickness` are positive.

## Input

```ini
[crystallite_shape]
mode = conventional

[initial_values]
cyl_diameter = 200.0
cyl_thickness = 40.0
cyl_thickness_std = 5.0
cyl_axis_h = 0.0
cyl_axis_k = 0.0
cyl_axis_l = 1.0
```

## Geometry

For pair length $L$ and angle $\phi$ to the axis,

$$
z=L|\cos\phi|,
\qquad
r_\perp=L|\sin\phi|.
$$

The radial overlap is

$$
\gamma_{\mathrm{radial}}=
\frac{2}{\pi}
\left[
\cos^{-1}(x)-x\sqrt{1-x^2}
\right],
\qquad
x=\frac{r_\perp}{D}.
$$

For thickness $T$,

$$
\gamma_{\mathrm{axial}}=
\max\left(1-\frac{z}{T},0\right).
$$

The total common-volume factor is

$$
\gamma_{\mathrm{cyl}}=
\gamma_{\mathrm{radial}}
\gamma_{\mathrm{axial}}.
$$

## Thickness distribution

A positive `cyl_thickness_std` activates a lognormal thickness average.

Direct lognormal parameters `cyl_mu` and `cyl_sigma` are also recognized.

## Interpretation

- large diameter and small thickness: disk or platelet;
- small diameter and large thickness: rod-like cylinder;
- similar dimensions: cylindrical nanoparticle.

## Limitations

The model assumes a circular cross-section and does not represent explicit
faceting.