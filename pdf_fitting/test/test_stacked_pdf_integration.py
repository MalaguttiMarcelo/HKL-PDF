from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    default_ab_sequence,
)

from pdf_fitting.models.stacked_finite_shells import (
    build_stacked_finite_shell_table,
)


def make_structure():
    lattice = Lattice.from_parameters(
        3.0,
        3.5,
        20.0,
        90.0,
        90.0,
        90.0,
    )

    return Structure(
        lattice=lattice,
        species=[
            "C",
            "O",
        ],
        coords=[
            [
                0.25,
                0.25,
                0.50,
            ],
            [
                0.75,
                0.75,
                0.50,
            ],
        ],
        coords_are_cartesian=False,
    )


def test_stacked_shell_table_has_pdf_metadata():
    structure = make_structure()

    shape_spec = CrystalliteShapeSpec(
        shape_type="prism",
        axis1_cells=3.0,
        axis2_cells=3.0,
        axis3_cells=5.0,
        diameter_cells=3.0,
        height_cells=5.0,
    )

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=8.0,
        shape_spec=shape_spec,
        stack_spec=stack_spec,
        analyze_supercell_symmetry=True,
    )

    shell_table = result.shell_table

    assert shell_table[
        "geometry_mode"
    ] == "stacked_supercell"

    assert shell_table[
        "shell_cart"
    ].shape == shell_table[
        "shell_frac"
    ].shape

    assert shell_table[
        "stacking_volume"
    ] > 0.0

    assert shell_table[
        "bulk_atom_count"
    ] > 0

    assert shell_table[
        "stacking_bulk_number_density"
    ] > 0.0

    expected_density = (
        shell_table[
            "bulk_atom_count"
        ]
        / shell_table[
            "stacking_volume"
        ]
    )

    assert np.isclose(
        shell_table[
            "stacking_bulk_number_density"
        ],
        expected_density,
    )

    reconstructed = (
        shell_table[
            "shell_frac"
        ].astype(float)
        @ shell_table[
            "stacking_lattice_matrix"
        ]
    )

    assert np.allclose(
        reconstructed,
        shell_table[
            "shell_cart"
        ],
        atol=1.0e-5,
    )