# Troubleshooting

## The first calculation is slow

The first run may compile Numba kernels and generate a shell cache. Later runs
with the same structure and cutoff should be faster.

## Shell generation uses too much memory

Reduce:

- `r_max`;
- `r_extension`;
- finite-particle dimensions;
- the number of grid-search candidates.

## No parameters are refined

Check that:

- every parameter exists in `[initial_values]`;
- the corresponding refine flag is true;
- the parameter is not derived in `[constraints]`;
- the selected GUI model has not removed it.

## SciPy reports an infeasible initial point

An initial value is outside its bounds. Correct the initial value or change
the bounds.

## Invalid unit-cell volume

Check:

- positive `a`, `b`, and `c`;
- valid lattice angles;
- symmetry-consistent values;
- bounds that prevent degenerate geometry.

## Constraints cannot be resolved

Check for:

- misspelled parameter names;
- missing dependencies;
- circular definitions;
- unsupported functions.

## The first PDF peaks are suppressed

Check that `qmax` matches the experimental data reduction. An incorrect
cutoff or very coarse $r$ grid can strongly affect low-$r$ peaks.

## The fit improves but parameters are not physical

This usually indicates parameter correlation or over-parameterization.

Try:

- refining fewer parameters;
- using staged refinement;
- tightening physically justified bounds;
- disabling competing broadening models;
- checking the initial structure.

## Site Biso parameters behave similarly

Site-labelled Biso values are currently averaged by element. Prefer
element-level Biso, or use [anisotropic ADPs](adp.md) for site-specific
displacements.

## A refined coordinate does not move

Coordinates fixed by the site symmetry cannot be refined. Only the free
coordinates are refined; tied coordinates follow them. Check also the
**Max coordinate shift** limit on the **Atoms & occupancy** page.

## The GPU is not used

- Check that `shell_backend` or `pdf_backend` is set to `cuda`.
- Check that CuPy imports: `python -c "import cupy"`.
- Read the console: HKL-PDF prints why it falls back to the CPU.
- Peak widths with anisotropic ADPs are always calculated on the CPU.

See [GPU acceleration](gpu.md).

## A stacking model takes long to start

The first compilation of a stochastic stacking ensemble can take about
20 seconds. It is cached for the session and on disk; later runs with the
same stacking setup start in a few seconds.

## Esds look too small

PDF points are correlated. Turn on **Correct esds for correlated PDF
points** in the refinement settings. See [Esds](uncertainties.md).

## A project opened on another computer cannot find its files

Use relative paths for the data and CIF files, and keep them next to the
project file. Relative paths in `.inp` and `.json` files are read relative to
the file's own folder.

## Finite-shape calculations are slow

- Start with one predefined shape.
- Reduce the particle dimensions.
- Reduce `r_max`.
- Use a larger grid-search step.
- Allow caches to finish building.
- Disable detailed gamma export unless required.

## Tick markers are slow

Use a smaller tick-marker range and select fewer pair types.

## PyVista or VTK errors

Update the visualization packages:

```bash
python -m pip install --upgrade pyvista pyvistaqt vtk qtpy
```

Also update the graphics driver and verify OpenGL support.

## Windows `wglMakeCurrent failed`

This is normally an OpenGL or VTK window-lifecycle problem. Close viewer
windows before terminating Python and use current VTK/PyVista releases.

## The refinement stops improving

Possible causes include:

- parameters at bounds;
- an incorrect structure;
- overly correlated parameters;
- unsuitable robust-loss settings;
- insufficient `max_nfev`;
- overly loose termination tolerances.

Inspect the residual and parameter-evolution plots before increasing the
iteration limit.