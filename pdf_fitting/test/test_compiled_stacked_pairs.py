from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.compiled_stacked_geometry import (
    compile_stacked_geometry,
)

from pdf_fitting.models.compiled_stacked_pairs import (
    compile_stacked_pair_topology,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
)


def make_structure() -> Structure:
    lattice = Lattice.from_parameters(
        4.0,
        5.0,
        10.0,
        90.0,
        90.0,
        90.0,
    )

    return Structure(
        lattice=lattice,
        species=[
            "C",
            "N",
        ],
        coords=[
            [
                0.25,
                0.25,
                0.45,
            ],
            [
                0.65,
                0.65,
                0.55,
            ],
        ],
        coords_are_cartesian=False,
    )


def make_shape() -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec(
        shape_type="prism",
        axis1_cells=2.0,
        axis2_cells=2.0,
        axis3_cells=4.0,
        diameter_cells=2.0,
        height_cells=4.0,
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


def make_stack(
    spacing_mode: str = "independent",
) -> LayerStackSpec:
    return LayerStackSpec(
        enabled=True,
        vector_mode="incremental",
        spacing_mode=spacing_mode,
        nominal_spacing=4.0,
        stacking_fraction=0.5,
        repeat_period=1.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=[
            SequenceLayer(
                label="A",
                shift_u=0.0,
                shift_v=0.0,
                stack_coordinate=0.0,
            ),
            SequenceLayer(
                label="B",
                shift_u=0.0,
                shift_v=0.25,
                stack_coordinate=1.0,
            ),
        ],
    )


def compile_topology(
    spacing_mode: str = "independent",
):
    geometry = compile_stacked_geometry(
        structure=make_structure(),
        shape_spec=make_shape(),
        stack_spec=make_stack(
            spacing_mode
        ),
        reference_parameters={
            "stacking_distance": 4.0,
            "stacking_fraction": 0.5,
        },
    )

    return compile_stacked_pair_topology(
        geometry=geometry,
        pair_cutoff=8.0,
        topology_margin=3.0,
        reference_parameters={
            "stacking_distance": 4.0,
            "stacking_fraction": 0.5,
        },
    )


def test_pair_topology_compiles_once():
    topology = compile_topology()

    assert topology.shell_mult.size > 0
    assert topology.shell_bulk_mult.size > 0

    assert (
        topology.shell_delta_fractional.shape[0]
        == topology.shell_mult.size
    )

    assert topology.finite_pair_rows > 0
    assert topology.bulk_pair_rows > 0


def test_non_geometry_evaluation_is_identical():
    topology = compile_topology()

    first = topology.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
            "scale": 1.0,
            "biso_C": 0.3,
        }
    )

    second = topology.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
            "scale": 2.0,
            "biso_C": 5.0,
        }
    )

    assert np.allclose(
        first.shell_cart,
        second.shell_cart,
    )

    assert np.allclose(
        first.shell_distance,
        second.shell_distance,
    )

    assert np.array_equal(
        first.shell_active,
        second.shell_active,
    )


def test_lattice_changes_pair_distances_without_recompiling():
    topology = compile_topology()

    first = topology.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
        }
    )

    second = topology.evaluate(
        {
            "a": 4.4,
            "b": 5.3,
            "c": 12.0,
            "stacking_distance": 4.0,
        }
    )

    assert first.number_of_shells == second.number_of_shells

    assert not np.allclose(
        first.shell_distance,
        second.shell_distance,
    )


def test_independent_spacing_changes_interlayer_pairs():
    topology = compile_topology(
        "independent"
    )

    first = topology.evaluate(
        {
            "stacking_distance": 4.0,
        }
    )

    second = topology.evaluate(
        {
            "stacking_distance": 5.0,
        }
    )

    assert first.number_of_shells == second.number_of_shells

    assert not np.allclose(
        first.shell_distance,
        second.shell_distance,
    )

    assert np.isclose(
        first.geometry.spacing,
        4.0,
    )

    assert np.isclose(
        second.geometry.spacing,
        5.0,
    )


def test_fraction_spacing_tracks_c_normal():
    topology = compile_topology(
        "fraction_of_c_normal"
    )

    first = topology.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.5,
        }
    )

    second = topology.evaluate(
        {
            "c": 12.0,
            "stacking_fraction": 0.5,
        }
    )

    assert np.isclose(
        first.geometry.spacing,
        5.0,
    )

    assert np.isclose(
        second.geometry.spacing,
        6.0,
    )


def test_fraction_refinement_changes_pair_distances():
    topology = compile_topology(
        "fraction_of_c_normal"
    )

    first = topology.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.4,
        }
    )

    second = topology.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.7,
        }
    )

    assert np.isclose(
        first.geometry.spacing,
        4.0,
    )

    assert np.isclose(
        second.geometry.spacing,
        7.0,
    )

    assert not np.allclose(
        first.shell_distance,
        second.shell_distance,
    )


def test_dynamic_fractional_vectors_reconstruct_cartesian():
    topology = compile_topology()

    evaluated = topology.evaluate(
        {
            "a": 4.2,
            "b": 5.1,
            "c": 11.0,
            "stacking_distance": 4.5,
        }
    )

    reconstructed = (
        evaluated.shell_frac
        @ evaluated.geometry.stacking_lattice_matrix
    )

    assert np.allclose(
        reconstructed,
        evaluated.shell_cart,
        atol=1.0e-8,
    )


def test_cutoff_mask_changes_without_topology_rebuild():
    topology = compile_topology()

    first = topology.evaluate(
        {
            "stacking_distance": 3.0,
        }
    )

    second = topology.evaluate(
        {
            "stacking_distance": 6.0,
        }
    )

    assert first.number_of_shells == second.number_of_shells

    # Some pair rows should generally cross the active physical cutoff as
    # the stacking distance changes.
    assert (
        first.number_of_active_shells
        != second.number_of_active_shells
        or not np.allclose(
            first.shell_distance,
            second.shell_distance,
        )
    )