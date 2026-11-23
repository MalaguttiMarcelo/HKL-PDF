# Project Files

The project window saves projects as JSON files, normally named
`*.hklpdf.json`. A project holds one or more **datasets**; each dataset
holds one or more **phases**.

Older single-phase `.inp` files can still be opened. They become a project
with one dataset and one phase. See [Input-file format](input-format.md) for
the `.inp` format.

## Structure

```json
{
  "format": "hkl-pdf-project",
  "version": 1,
  "name": "FeCr ball milled",
  "refinement": { "max_nfev_final": 60, "pdf_backend": "cpu" },
  "constraints": { "ds1.fe.re": "ds1.fe.d" },
  "datasets": [
    {
      "id": "ds1",
      "name": "FeCr_ID31",
      "data_file": "FeCr_ID31.xy",
      "r_min": 2.0,
      "r_max": 150.0,
      "r_extension": 1.2,
      "parameters": {
        "qdamp": { "value": 0.0115, "refine": true, "min": 0.0, "max": 0.1 },
        "qmax":  { "value": 20.0 }
      },
      "phases": [
        {
          "id": "fe",
          "name": "Fe_bcc_300K",
          "cif": "Fe_bcc_300K.cif",
          "active": true,
          "parameters": {
            "scale": { "value": 1.02, "refine": true },
            "a":     { "value": 2.8665, "refine": true, "esd": 0.0001 }
          },
          "settings": {},
          "sections": {}
        }
      ]
    }
  ],
  "last_results": {},
  "last_uncertainty": {}
}
```

### Top level

| Key | Meaning |
|---|---|
| `format` | Always `hkl-pdf-project` |
| `version` | Format version (currently 1) |
| `name` | Project name |
| `refinement` | Project-wide settings (see [Parameter reference](parameters.md#refinement-settings)) |
| `constraints` | Map of target parameter → expression |
| `datasets` | List of datasets |
| `last_results` | Last calculated curves, stored compactly (optional) |
| `last_uncertainty` | Last esds and correlation matrix (optional) |

### Dataset

| Key | Meaning |
|---|---|
| `id` | Short identifier used in namespaced names |
| `data_file` | Experimental $G(r)$ file |
| `r_min`, `r_max` | Fitting range (Å) |
| `r_extension` | Pair-generation cutoff factor |
| `parameters` | Instrument parameters shared by every phase |
| `phases` | List of phases |

### Phase

| Key | Meaning |
|---|---|
| `id` | Short identifier used in namespaced names |
| `cif` | Structure file |
| `active` | Inactive phases are kept but not calculated |
| `parameters` | Phase parameters |
| `settings` | Phase-level options (strain model, Biso mode, …) |
| `sections` | Shape, stacking, structure edits, free-peak results, … |

### Parameter

| Key | Meaning |
|---|---|
| `value` | Value |
| `refine` | `true` to refine (omitted when false) |
| `min`, `max` | Bounds (optional) |
| `esd` | Last esd (optional) |

## Paths

Relative paths in `data_file` and `cif` are taken relative to the folder of
the project file. Projects can therefore be moved or shared together with
their data.

## Namespaced names

Inside a project, every parameter has a full name of the form

```text
<dataset id>.<phase id>.<parameter>     phase parameter
<dataset id>.<parameter>                instrument parameter
```

for example `ds1.fe.a` or `ds1.qdamp`. The full names are used in
constraints and in the **Refined parameters** page.

## Constraints

A constraint sets a parameter from an expression of other parameters:

```json
"constraints": {
  "ds1.fe.re": "ds1.fe.d",
  "ds1.cr.a":  "ds1.fe.a * 1.002"
}
```

The constrained parameter is not refined itself. Its esd is obtained by
propagation from the parameters it depends on.

## Several phases

Each phase has its own scale factor. Weight fractions are derived from the
scale factors and shown on the phase overview page and in the progress
panel.

## Several datasets

Datasets are refined together. Each dataset has its own instrument
parameters and its own phases. Use **Duplicate dataset (same phases)** to
start a series, and constraints to share parameters between datasets.

## Saved results

`last_results` stores the last calculated curves of every dataset (as
compressed float32 arrays), so the fit appears immediately when the project
is opened. `last_uncertainty` stores the last esds and correlation matrix.
Both are optional and are replaced at the next calculation.
