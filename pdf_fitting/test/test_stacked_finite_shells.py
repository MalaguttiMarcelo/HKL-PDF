from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    default_a_sequence,
    default_ab_sequence,
    default_abc_sequence,
)

from pdf_fitting.models.stacked_finite_shells import (
    build_stacked_finite_shell_table,
    validate_stacked_shell_table,
)


def make_two_atom_layer() -> Structure:
    """
    Small P1 layer suitable for fast finite/bulk shell tests.
    """
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


def make_prism_shape(
    number_of_layers: int,
) -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec(
        shape_type="prism",
        axis1_cells=3.0,
        axis2_cells=3.0,
        axis3_cells=float(
            number_of_layers
        ),
        diameter_cells=3.0,
        height_cells=float(
            number_of_layers
        ),
        axis_h=0,
        axis_k=0,
        axis_l=1,
        base1_h=1,
        base1_k=0,
        base1_l=0,
        base2_h=0,
        base2_k=1,
        base2_l=0,
    )


def assert_basic_shell_table(
    shell_table,
):
    validate_stacked_shell_table(
        shell_table
    )

    number_of_shells = int(
        shell_table[
            "shell_frac"
        ].shape[0]
    )

    assert number_of_shells > 0

    assert shell_table[
        "geometry_mode"
    ] == "stacked_supercell"

    assert shell_table[
        "shell_frac"
    ].shape == (
        number_of_shells,
        3,
    )

    assert shell_table[
        "shell_direction"
    ].shape == (
        number_of_shells,
        3,
    )

    assert shell_table[
        "shell_hkl"
    ].shape == (
        number_of_shells,
        3,
    )

    assert np.all(
        shell_table[
            "shell_mult"
        ] > 0
    )

    assert np.all(
        shell_table[
            "shell_bulk_mult"
        ] > 0
    )

    assert np.all(
        shell_table[
            "shell_mult_eff"
        ] > 0.0
    )

    assert np.all(
        shell_table[
            "shell_bulk_mult_eff"
        ] > 0.0
    )

    assert np.all(
        np.isfinite(
            shell_table[
                "shell_gamma"
            ]
        )
    )

    assert int(
        shell_table[
            "pair_offsets"
        ][
            -1
        ]
    ) == number_of_shells


def test_a_stack_shell_table():
    structure = make_two_atom_layer()

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=1.0,
        sequence=default_a_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=7.0,
        shape_spec=make_prism_shape(
            3
        ),
        stack_spec=stack_spec,
    )

    assert_basic_shell_table(
        result.shell_table
    )

    assert (
        result.finite_particle.number_of_layer_positions
        == 3
    )

    assert (
        result.bulk_supercell.number_of_sequence_layers
        == 1
    )

    assert (
        result.shell_table[
            "number_of_layer_positions"
        ]
        == 3
    )


def test_ab_stack_shell_table():
    structure = make_two_atom_layer()

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=8.0,
        shape_spec=make_prism_shape(
            5
        ),
        stack_spec=stack_spec,
    )

    assert_basic_shell_table(
        result.shell_table
    )

    assert (
        result.bulk_supercell.number_of_sequence_layers
        == 2
    )

    assert (
        result.finite_particle.number_of_layer_positions
        == 5
    )

    assert set(
        result.finite_particle.atom_sequence_ids.tolist()
    ) == {
        0,
        1,
    }


def test_abc_stack_shell_table():
    structure = make_two_atom_layer()

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=3.5,
        repeat_period=3.0,
        sequence=default_abc_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=8.0,
        shape_spec=make_prism_shape(
            7
        ),
        stack_spec=stack_spec,
    )

    assert_basic_shell_table(
        result.shell_table
    )

    assert (
        result.bulk_supercell.number_of_sequence_layers
        == 3
    )

    assert set(
        result.finite_particle.atom_sequence_ids.tolist()
    ) == {
        0,
        1,
        2,
    }


def test_shell_fractional_vectors_match_cartesian_vectors():
    structure = make_two_atom_layer()

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=8.0,
        shape_spec=make_prism_shape(
            5
        ),
        stack_spec=stack_spec,
    )

    shell_table = result.shell_table

    reconstructed = (
        shell_table[
            "shell_frac"
        ].astype(
            float
        )
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


def test_stacking_distance_changes_pair_geometry():
    structure = make_two_atom_layer()

    shape_spec = make_prism_shape(
        5
    )

    first_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    second_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=5.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    first = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=9.0,
        shape_spec=shape_spec,
        stack_spec=first_spec,
    )

    second = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=9.0,
        shape_spec=shape_spec,
        stack_spec=second_spec,
    )

    assert not np.allclose(
        first.bulk_supercell.lattice_matrix,
        second.bulk_supercell.lattice_matrix,
    )

    assert not np.isclose(
        first.bulk_supercell.period_normal_height,
        second.bulk_supercell.period_normal_height,
    )

    first_distances = np.sqrt(
        np.sum(
            first.shell_table[
                "shell_cart"
            ].astype(
                float
            ) ** 2,
            axis=1,
        )
    )

    second_distances = np.sqrt(
        np.sum(
            second.shell_table[
                "shell_cart"
            ].astype(
                float
            ) ** 2,
            axis=1,
        )
    )

    # The shell arrays can have different lengths because changing spacing may
    # move shells across the pair cutoff. Compare their rounded distance sets.
    first_set = set(
        np.rint(
            first_distances
            * 1.0e5
        ).astype(
            int
        ).tolist()
    )

    second_set = set(
        np.rint(
            second_distances
            * 1.0e5
        ).astype(
            int
        ).tolist()
    )

    assert first_set != second_set


def test_disabled_sequence_row_is_periodic_vacancy():
    structure = make_two_atom_layer()

    sequence = default_ab_sequence()
    sequence[1].enabled = False

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=sequence,
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=9.0,
        shape_spec=make_prism_shape(
            6
        ),
        stack_spec=stack_spec,
    )

    assert_basic_shell_table(
        result.shell_table
    )

    assert (
        result.bulk_supercell.number_of_sequence_layers
        == 1
    )

    assert np.isclose(
        result.bulk_supercell.period_normal_height,
        8.0,
    )

    assert (
        result.finite_particle.number_of_layer_positions
        == 6
    )

    assert (
        result.finite_particle.number_of_enabled_layers
        == 3
    )


def test_all_directions_are_recalculated():
    structure = make_two_atom_layer()

    stack_spec = LayerStackSpec(
        enabled=True,
        nominal_spacing=4.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacked_finite_shell_table(
        structure=structure,
        pair_cutoff=8.0,
        shape_spec=make_prism_shape(
            5
        ),
        stack_spec=stack_spec,
    )

    directions = np.asarray(
        result.shell_table[
            "shell_direction"
        ],
        dtype=int,
    )

    assert directions.ndim == 2
    assert directions.shape[1] == 3

    assert np.any(
        np.any(
            directions != 0,
            axis=1,
        )
    )

    unique_directions = np.asarray(
        result.shell_table[
            "strain_unique_dirs"
        ],
        dtype=int,
    )

    assert unique_directions.ndim == 2
    assert unique_directions.shape[1] == 3