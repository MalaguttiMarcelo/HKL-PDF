from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)


def _integer_cell_count(
    value: Any,
    *,
    label: str,
) -> int:
    """
    Convert a prism dimension in cells to an exact positive integer.

    A periodic crystallographic supercell must use integer repetitions.
    """
    try:
        numeric_value = float(
            value
        )
    except Exception as exc:
        raise ValueError(
            f"{label} must be a positive integer number of unit cells."
        ) from exc

    if not np.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{label} must be finite."
        )

    rounded_value = int(
        round(
            numeric_value
        )
    )

    if rounded_value < 1:
        raise ValueError(
            f"{label} must be at least 1."
        )

    if not np.isclose(
        numeric_value,
        float(
            rounded_value
        ),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            f"{label} is {numeric_value:g} cells.\n\n"
            "A periodic CIF export requires integer prism dimensions.\n"
            f"Use {rounded_value} cells or another exact integer value."
        )

    return rounded_value


def _direction_tuple(
    h: Any,
    k: Any,
    l: Any,
    *,
    label: str,
) -> Tuple[int, int, int]:
    direction = (
        int(
            h
        ),
        int(
            k
        ),
        int(
            l
        ),
    )

    if direction == (
        0,
        0,
        0,
    ):
        raise ValueError(
            f"{label} cannot be [0 0 0]."
        )

    return direction


def build_prism_periodic_supercell(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
) -> Tuple[
    Structure,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Build a periodic supercell matching the finite-prism dimensions.

    New periodic CIF axes
    ---------------------
    new a:
        prism base direction 1 × axis1_cells

    new b:
        prism base direction 2 × axis2_cells

    new c:
        prism axis direction × axis3_cells

    Notes
    -----
    This creates a crystallographically valid periodic supercell. It does not
    atom-clip a finite particle, because atom clipping at a boundary would
    produce an invalid periodic CIF when repeated.

    The exported structure is intended to be used later with the conventional
    cylinder/disk common-volume model.
    """
    if structure is None:
        raise ValueError(
            "A source structure is required."
        )

    shape_type = str(
        shape_spec.shape_type
        or ""
    ).strip().lower()

    if shape_type not in {
        "prism",
        "rectangular_prism",
    }:
        raise ValueError(
            "Periodic prism CIF export is available only for prism shapes."
        )

    axis1_cells = _integer_cell_count(
        shape_spec.axis1_cells,
        label="Prism base axis 1",
    )

    axis2_cells = _integer_cell_count(
        shape_spec.axis2_cells,
        label="Prism base axis 2",
    )

    axis3_cells = _integer_cell_count(
        shape_spec.axis3_cells,
        label="Prism axis 3",
    )

    base1 = _direction_tuple(
        shape_spec.base1_h,
        shape_spec.base1_k,
        shape_spec.base1_l,
        label="Prism base direction 1",
    )

    base2 = _direction_tuple(
        shape_spec.base2_h,
        shape_spec.base2_k,
        shape_spec.base2_l,
        label="Prism base direction 2",
    )

    axis3 = _direction_tuple(
        shape_spec.axis_h,
        shape_spec.axis_k,
        shape_spec.axis_l,
        label="Prism / shape axis",
    )

    transformation = np.asarray(
        [
            [
                axis1_cells * base1[0],
                axis1_cells * base1[1],
                axis1_cells * base1[2],
            ],
            [
                axis2_cells * base2[0],
                axis2_cells * base2[1],
                axis2_cells * base2[2],
            ],
            [
                axis3_cells * axis3[0],
                axis3_cells * axis3[1],
                axis3_cells * axis3[2],
            ],
        ],
        dtype=np.int64,
    )

    determinant = int(
        round(
            float(
                np.linalg.det(
                    transformation.astype(
                        float
                    )
                )
            )
        )
    )

    if determinant == 0:
        raise ValueError(
            "The selected prism base directions and prism axis are not "
            "linearly independent.\n\n"
            f"Base direction 1: {base1}\n"
            f"Base direction 2: {base2}\n"
            f"Prism axis:      {axis3}\n\n"
            "Choose three independent crystallographic directions."
        )

    used_base2 = base2

    # Pymatgen requires a positive-volume transformation matrix.
    #
    # Reversing b preserves the same crystallographic axis physically,
    # but converts a left-handed basis to a right-handed basis.
    if determinant < 0:
        transformation[
            1,
            :
        ] *= -1

        used_base2 = (
            -base2[0],
            -base2[1],
            -base2[2],
        )

        determinant = abs(
            determinant
        )

    supercell = structure.copy()

    supercell.make_supercell(
        transformation,
        to_unit_cell=True,
    )

    information = {
        "axis1_cells": int(
            axis1_cells
        ),
        "axis2_cells": int(
            axis2_cells
        ),
        "axis3_cells": int(
            axis3_cells
        ),
        "base1_direction": base1,
        "base2_direction": used_base2,
        "axis_direction": axis3,
        "transformation_matrix": transformation.copy(),
        "determinant": int(
            determinant
        ),
        "number_of_atoms": int(
            len(
                supercell.sites
            )
        ),
    }

    return (
        supercell,
        transformation,
        information,
    )


def export_prism_periodic_cif(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    path: str,
) -> Dict[
    str,
    Any,
]:
    """
    Build and write a periodic prism-supercell CIF.
    """
    periodic_structure, transformation, information = (
        build_prism_periodic_supercell(
            structure=structure,
            shape_spec=shape_spec,
        )
    )

    writer = CifWriter(
        periodic_structure,
    )

    writer.write_file(
        path
    )

    information[
        "path"
    ] = str(
        path
    )

    information[
        "lattice_a"
    ] = float(
        periodic_structure.lattice.a
    )

    information[
        "lattice_b"
    ] = float(
        periodic_structure.lattice.b
    )

    information[
        "lattice_c"
    ] = float(
        periodic_structure.lattice.c
    )

    information[
        "lattice_alpha"
    ] = float(
        periodic_structure.lattice.alpha
    )

    information[
        "lattice_beta"
    ] = float(
        periodic_structure.lattice.beta
    )

    information[
        "lattice_gamma"
    ] = float(
        periodic_structure.lattice.gamma
    )

    return information