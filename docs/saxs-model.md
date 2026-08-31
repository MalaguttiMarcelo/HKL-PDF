# SAXS and Missing-Low-Q Model

HKL-PDF supports two baseline modes:

1. conventional PDF mode;
2. independent missing-low-$Q$/SAXS belly mode.

## Conventional mode

```ini
[refinement]
saxs_enabled = false
```

The structural PDF contains its required conventional or shape-dependent
negative baseline.

## SAXS belly mode

```ini
[refinement]
saxs_enabled = true
saxs_model = sphere
saxs_apply_qdamp = false
saxs_apply_qmax = true
```

The independent belly term is

$
G_{\mathrm{SAXS}}(r)=
-s\,s_{\mathrm{SAXS}}\,
4\pi\rho_0r\,
\gamma_{\mathrm{SAXS}}(r).
$

## Sphere model

```ini
[refinement]
saxs_model = sphere

[initial_values]
saxs_scale = 1.0
saxs_diameter = 100
saxs_diameter_std = 5
```

## Cylinder or disk model

```ini
[refinement]
saxs_model = cylinder

[initial_values]
saxs_scale = 1.0
saxs_diameter = 100
saxs_diameter_std = 5
saxs_height = 20
saxs_height_std = 2
```

## Same as the main size model

```ini
[refinement]
saxs_model = same_as_size_model
```

The common-volume function of the active size model is reused.

## Fallback behaviour

If all independent SAXS dimensions are zero, HKL-PDF uses the main
crystallite-size model as the SAXS envelope.

## Damping

The peak contribution always receives `qdamp`.

The SAXS belly receives it only when:

```ini
saxs_apply_qdamp = true
```

Finite-$Q_{\max}$ termination is controlled independently:

```ini
saxs_apply_qmax = true
```

## Caution

The SAXS belly is not a generic polynomial background. It is a
particle-shape-related term and can correlate strongly with scale, particle
size, and low-$r$ data corrections.