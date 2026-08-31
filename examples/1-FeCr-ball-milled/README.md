# FeCr ball-milled example

This example demonstrates PDF fitting of a ball-milled FeCr sample using a
body-centred cubic Fe structural model.

## Files

- `data/Fe_bcc_300K.cif`: BCC Fe structure.
- `data/FeCr_ID31.xy`: experimental PDF data.
- `FeCr-conventional.inp`: conventional PDF fitting.

## Conventional fit

The conventional fit uses:

```ini
[refinement]
saxs_enabled = false

In the public input, replace the absolute paths with relative paths:

```ini
[files]
structure_file = data/Fe_bcc_300K.cif
gr_data_file = data/FeCr_ID31.xy