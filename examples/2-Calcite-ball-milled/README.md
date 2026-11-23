# Calcite ball-milled example

PDF fitting of a calcite sample ball-milled for 150 minutes, with spherical
crystallite size, pair-specific local correlations, and PAH microstrain.

## Files

| File | Contents |
|---|---|
| `Calcite.cif` | Calcite structure |
| `5-CaCO3-150min.integrated.subtracted_bkg.corr.2th.qmax20.xy` | Experimental $G(r)$ ($Q_\mathrm{max}$ = 20 Å⁻¹) |
| `calcite_150min.hklpdf.json` | Project for the project window (includes the saved fit) |
| `Calcite_150min.inp` | Same model as an `.inp` file: conventional fit, 2–150 Å |

All paths are relative, so the example runs from any location.

## Running

```bash
python -m pdf_fitting.gui examples/2-Calcite-ball-milled/calcite_150min.hklpdf.json
```

Press **Calculate** (F5) to recalculate the saved fit, or **Refine** (F6).

## What to try

- **Atoms & occupancy:** make the O site anisotropic and refine its free
  $\beta_{ij}$ components; view the ellipsoids in **Structure (3D)**.
- **Microstructure:** the trigonal contrast-factor invariants are listed by
  name; **Import contrast factors…** reads them from a file.
