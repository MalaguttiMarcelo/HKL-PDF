# FeCr ball-milled example

PDF fitting of a ball-milled Fe–Cr sample with a body-centred cubic Fe
structural model, spherical crystallite size, local correlations, and Wilkens
dislocation microstrain.

## Files

| File | Contents |
|---|---|
| `Fe_bcc_300K.cif` | BCC Fe structure |
| `FeCr_ID31.xy` | Experimental $G(r)$ |
| `input_fecr_paper_2.hklpdf.json` | Project for the project window (includes the saved fit) |
| `input_FeCr_paper_2.inp` | Same model as an `.inp` file: conventional fit, 2–150 Å |
| `input_FeCr_paper_3.inp` | Variant with an explicit finite disk-shaped crystallite |

All paths are relative, so the example runs from any location.

## Running

```bash
python -m pdf_fitting.gui examples/1-FeCr-ball-milled/input_fecr_paper_2.hklpdf.json
```

The saved fit is shown when the project opens. Press **Calculate** (F5) to
recalculate it, or **Refine** (F6) to refine.

## What to try

- **Peak widths (free fit)** on the phase: free-peak fit and the width
  analysis per direction.
- **Refinement settings › esds and correlation matrix**, then
  **Refined parameters › Compute esds now**.
- **Structure (3D)**: select an Fe atom, then a neighbour, and use
  **Mark selected on PDF fit**.
