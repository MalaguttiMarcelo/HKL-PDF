# Structure Editor

Every phase has a **Structure (3D)** page. It shows the structure in a 3D
view and edits the same site parameters that the refinement uses
(`x_<site>`, `y_<site>`, `z_<site>`, `occ_<site>`, `biso_<site>`,
`beta<ij>_<site>`).

The original CIF file is never modified. Edited sites are stored in the
project. The model reads a derived CIF in the user cache, in which only the
atom-site loop is replaced. Use **Export CIF…** to write the edited
structure to a file.

## View

- Rotate, pan, and zoom with the mouse.
- **Repeat** shows several unit cells.
- **Atom scale** changes the atom size.
- **Thermal ellipsoids (50 %)** draws the anisotropic displacement
  ellipsoids of anisotropic sites (see [Anisotropic ADPs](adp.md)).
- **Pop out** moves the view to its own window; **Dock back** returns it.
- **Refresh from parameters** redraws the structure from the current
  parameter values, for example after a refinement.

## Selecting and measuring

Click atoms in the 3D view.

- **Distance (2 atoms)** or **Angle (3 atoms)** selects the measurement.
  When a pair or triplet is complete, the next click starts a new one.
- **Clear picks** removes the current selection.

The **Neighbors of the selected atom** table lists the neighbours up to a
chosen distance:

- **Mark selected on PDF fit** marks the matching pair with tick markers on
  the fit plot;
- **Clear marks** removes them;
- **Pair dynamics of row** opens the matching pair on the
  **Local dynamics** page.

With one atom selected, **Edit in Atoms & occupancy** opens its row on the
Atoms page. With two atoms selected, **Edit pair in Local dynamics** opens
the pair's correlated-motion entry.

## Editing sites

Edits preserve the space-group symmetry:

- a site moves with its whole orbit of equivalent positions;
- coordinates fixed by the site symmetry cannot change;
- coordinates tied by the site symmetry move together.

Available edits:

- numeric fields for coordinates, occupancy, and Biso of the selected site;
- **Drag atoms in the 3D view** moves the atom only along its
  symmetry-free directions;
- **Add site…** adds a new orbit at the last picked atom, at the midpoint
  of a picked pair, or at typed coordinates;
- **Delete site** removes a whole orbit;
- **Reset site to CIF** restores the original values;
- **Undo** and **Redo**.

**Live G(r) preview** recalculates the PDF after each change. It is off by
default because a calculation can take a few seconds for large structures.

## Refining coordinates and occupancies

Coordinates and occupancies refined on the **Atoms & occupancy** page use
the same site-symmetry constraints as the editor. The settings on that page
limit them:

| Setting | Meaning |
|---|---|
| Occupancy minimum / maximum | Limits for every refined occupancy |
| Max coordinate shift (Å) | Largest shift of a refined coordinate from the CIF value (0 = no limit) |
| One Biso per site | Otherwise one Biso per element (recommended for large structures) |
