# SAXS and Missing-Low-Q Belly Model

HKL-PDF supports two PDF baseline modes:

1. corrected conventional PDF mode;
2. independent missing-low-$Q$/SAXS belly mode.

The model is selected with `saxs_enabled`.

## Conventional mode

```ini
[refinement]
saxs_enabled = false
```

The required conventional or shape-dependent PDF baseline remains active.

## SAXS belly mode

```ini
[refinement]
saxs_enabled = true
saxs_model = sphere
saxs_apply_qdamp = false
saxs_apply_qmax = true
```

The belly term is

$$
G_{\mathrm{belly}}(r)=
-s\,s_{\mathrm{SAXS}}\,
4\pi\rho_0r\,
\gamma_{\mathrm{SAXS}}(r).
$$

## Independent dimensions

```ini
[initial_values]
saxs_scale = 1.0
saxs_diameter = 100.0
saxs_diameter_std = 5.0
saxs_height = 20.0
saxs_height_std = 2.0
```

Sphere mode uses the diameter parameters. Cylinder and disk modes also use
height parameters.

## Same-as-size mode

```ini
saxs_model = same_as_size_model
```

This uses the common-volume function of the active main size model.

## Damping and termination

The belly receives `qdamp` only when:

```ini
saxs_apply_qdamp = true
```

Finite-$Q_{\max}$ termination is controlled with:

```ini
saxs_apply_qmax = true
```

## Caution

The SAXS belly can correlate strongly with scale, particle dimensions, and
low-$r$ corrections. It should not be used as an arbitrary background
function.