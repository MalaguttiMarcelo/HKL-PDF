# Graphical Interface

## Builder and raw input

The left side contains:

- **Builder** — structured input editor;
- **Raw input** — generated INI text.

The Builder updates the raw input while preserving parameters that are not
currently exposed in a visible table.

## Main actions

### Calculate

Calculates the PDF once without refinement.

### Run

Starts conventional least-squares refinement.

### Run finite-shape fit

Runs one predefined finite shape.

### Run CS grid search

Runs a discrete crystallite-shape search.

### Pause

Temporarily pauses the worker.

### Stop/Save

Stops the current calculation. The best parameters seen by the worker can
optionally be saved.

## Parameter tables

A normal parameter table contains:

- parameter name;
- value;
- lower bound;
- upper bound;
- refinement checkbox.

Local-dynamics tables also contain a **Use?** checkbox. When disabled, the
parameter is removed from the generated input.

## Size models

The GUI provides:

- spherical;
- cylinder/disk;
- finite crystallite shape.

The finite-shape dialog contains:

- shape type;
- fixed or grid-search mode;
- dimensions;
- orientation vectors;
- OpenGL preview;
- XYZ export.

## Microstrain models

Available models are:

- isotropic;
- Wilkens;
- PAH.

The contrast-factor panel becomes available for directional strain models.

## Biso modes

The GUI supports:

- one Biso per element;
- site-labelled Biso parameters.

Element-level Biso is recommended for large structures.

## PDF window

The PDF window can display:

- experimental $G(r)$;
- calculated $G(r)$;
- offset residual;
- $R_\mathrm{wp}$;
- pair contributions;
- pair-contribution sums;
- shell tick markers;
- the shape or SAXS term.

Tick markers can be filtered by:

- pair type;
- crystallographic direction;
- minimum and maximum $r$;
- colour, width, and transparency.

## Result tabs

- **Rwp** — best $R_\mathrm{wp}$ versus function evaluations.
- **CS grid** — finite-shape candidate comparison.
- **Warren** — direction-resolved strain.
- **Size dist.** — size distribution.
- **Params** — evolution of refinable parameters.
- **Local trends** — lambda and delta trends.
- **Structure** — VTK structure viewer.

## Performance recommendations

For large calculations:

- disable live PDF updates;
- disable full-grid live $R_\mathrm{wp}$;
- calculate pair contributions only when needed;
- calculate Warren plots only when needed;
- calculate local trends only when needed;
- leave tick markers disabled until required.