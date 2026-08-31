# Conventional PDF Fitting

Conventional mode uses a periodic crystal structure together with analytical
size, strain, damping, and termination corrections.

Select it with:

```ini
[crystallite_shape]
mode = conventional
```

## Recommended refinement order

### 1. Scale

Refine `scale` first.

### 2. Lattice parameters

Refine only the independent lattice parameters allowed by symmetry.

| Crystal system | Independent parameters |
|---|---|
| Cubic | `a` |
| Tetragonal | `a`, `c` |
| Orthorhombic | `a`, `b`, `c` |
| Hexagonal/trigonal | `a`, `c` |
| Monoclinic | `a`, `b`, `c`, `beta` |
| Triclinic | all six |

### 3. Displacement parameters

Begin with one `biso_Element` per element. Use conservative bounds such as
0–10 Å².

### 4. Crystallite size

Choose one main model:

- spherical `d` and `d_std`; or
- cylinder/disk dimensions.

### 5. Local correlations

Add pair lambdas or delta parameters only when needed to describe short-range
peak widths.

### 6. Microstrain

Choose one primary microstrain model:

- isotropic;
- Wilkens;
- PAH.

## Staged refinement

```ini
[refinement]
staged_refinement = true
```

HKL-PDF automatically groups refinable parameters into approximately:

1. lattice;
2. scale and Biso;
3. size and shape;
4. microstrain;
5. other parameters;
6. all parameters together.

## Robust loss

The default robust loss is `soft_l1`:

```ini
[refinement]
loss = soft_l1
f_scale = 0.05
```

Ordinary least squares can be selected with:

```ini
loss = linear
```

## Refinement grids

Early stages can use a coarser $r$ grid:

```ini
refinement_r_step = 0.05
```

The final stage can use the full grid:

```ini
refinement_r_step_final = 0.0
```

## Diagnosing residuals

- Incorrect peak positions usually indicate lattice or structure errors.
- Incorrect amplitudes may indicate scale, composition, or normalization
  problems.
- Incorrect high-$r$ decay may indicate size, `qdamp`, or microstrain.
- Incorrect low-$r$ widths may indicate Biso or local correlations.
- Oscillatory residuals may indicate an inconsistent `qmax`.
- A sloping baseline may indicate an inappropriate size or SAXS model.

A lower $R_\mathrm{wp}$ does not automatically make a more complicated model
physically correct.