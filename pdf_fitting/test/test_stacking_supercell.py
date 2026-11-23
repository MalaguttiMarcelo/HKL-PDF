from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    default_a_sequence,
    default_ab_sequence,
    default_abc_sequence,
)

from pdf_fitting.models.stacking_supercell import (
    build_stacking_supercell,
)


def make_single_layer_structure() -> Structure:
    """
    One carbon atom in an a-b layer with vacuum along c.
    """
    lattice = Lattice.from_parameters(
        3.0,
        4.0,
        20.0,
        90.0,
        90.0,
        90.0,
    )

    return Structure(
        lattice=lattice,
        species=["C"],
        coords=[
            [
                0.25,
                0.50,
                0.50,
            ]
        ],
        coords_are_cartesian=False,
    )


def test_a_stack_supercell():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=1.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=default_a_sequence(),
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    assert result.number_of_atoms == 1
    assert result.number_of_sequence_layers == 1

    assert np.allclose(
        result.vector_a,
        structure.lattice.matrix[0],
    )

    assert np.allclose(
        result.vector_b,
        structure.lattice.matrix[1],
    )

    assert np.isclose(
        np.dot(
            result.vector_c_stack,
            result.stack_normal,
        ),
        6.0,
    )

    assert np.isclose(
        result.period_normal_height,
        6.0,
    )


def test_ab_stack_supercell():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=2.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    assert result.number_of_atoms == 2
    assert result.number_of_sequence_layers == 2

    assert np.isclose(
        result.period_normal_height,
        12.0,
    )

    assert np.isclose(
        np.dot(
            result.vector_c_stack,
            result.stack_normal,
        ),
        12.0,
    )

    assert set(
        result.atom_sequence_indices.tolist()
    ) == {
        0,
        1,
    }


def test_abc_stack_supercell():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=5.0,
        repeat_period=3.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=default_abc_sequence(),
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    assert result.number_of_atoms == 3
    assert result.number_of_sequence_layers == 3

    assert np.isclose(
        result.period_normal_height,
        15.0,
    )

    assert set(
        result.atom_sequence_indices.tolist()
    ) == {
        0,
        1,
        2,
    }


def test_repeat_shift_enters_c_vector():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=2.0,
        repeat_shift_u=0.25,
        repeat_shift_v=-0.50,
        sequence=default_ab_sequence(),
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    expected = (
        0.25
        * structure.lattice.matrix[0]
        - 0.50
        * structure.lattice.matrix[1]
        + 12.0
        * result.stack_normal
    )

    assert np.allclose(
        result.vector_c_stack,
        expected,
    )


def test_fractional_coordinates_are_wrapped():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    assert np.all(
        result.atom_frac >= 0.0
    )

    assert np.all(
        result.atom_frac < 1.0
    )

    assert np.allclose(
        result.atom_cart,
        result.atom_frac
        @ result.lattice_matrix,
    )


def test_disabled_sequence_row_is_a_periodic_vacancy():
    structure = make_single_layer_structure()

    sequence = default_ab_sequence()
    sequence[1].enabled = False

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=2.0,
        sequence=sequence,
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
    )

    assert result.number_of_atoms == 1
    assert result.number_of_sequence_layers == 1

    # The period remains two layer spacings even though B produces no atoms.
    assert np.isclose(
        result.period_normal_height,
        12.0,
    )


def test_lattice_override_rebuilds_in_plane_vectors():
    structure = make_single_layer_structure()

    spec = LayerStackSpec(
        enabled=True,
        vector_mode="absolute",
        nominal_spacing=6.0,
        repeat_period=2.0,
        sequence=default_ab_sequence(),
    )

    refined_lattice = Lattice.from_parameters(
        3.2,
        4.3,
        21.0,
        90.0,
        90.0,
        90.0,
    )

    result = build_stacking_supercell(
        structure=structure,
        stack_spec=spec,
        lattice_matrix=refined_lattice.matrix,
    )

    assert np.allclose(
        result.vector_a,
        refined_lattice.matrix[0],
    )

    assert np.allclose(
        result.vector_b,
        refined_lattice.matrix[1],
    )

    assert np.isclose(
        np.dot(
            result.vector_c_stack,
            result.stack_normal,
        ),
        12.0,
    )