from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.compiled_stacked_geometry import (
    compile_stacked_geometry,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
)


def make_corrugated_layer() -> Structure:
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
            "C",
        ],
        coords=[
            [
                0.25,
                0.25,
                0.45,
            ],
            [
                0.25,
                0.25,
                0.55,
            ],
        ],
        coords_are_cartesian=False,
    )


def make_shape(
    number_of_layers: int,
) -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec(
        shape_type="prism",
        axis1_cells=1.5,
        axis2_cells=1.5,
        axis3_cells=float(
            number_of_layers
        ),
        diameter_cells=1.5,
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


def make_stack(
    *,
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
            )
        ],
    )


def test_compilation_keeps_integer_layer_count():
    structure = make_corrugated_layer()

    compiled = compile_stacked_geometry(
        structure=structure,
        shape_spec=make_shape(
            5
        ),
        stack_spec=make_stack(),
    )

    first = compiled.evaluate(
        {
            "stacking_distance": 4.0,
        }
    )

    second = compiled.evaluate(
        {
            "stacking_distance": 6.0,
        }
    )

    assert first.number_of_layers == 5
    assert second.number_of_layers == 5

    assert np.unique(
        first.atom_layer_indices
    ).size == 5

    assert np.unique(
        second.atom_layer_indices
    ).size == 5


def test_independent_spacing_moves_layers():
    structure = make_corrugated_layer()

    compiled = compile_stacked_geometry(
        structure=structure,
        shape_spec=make_shape(
            5
        ),
        stack_spec=make_stack(
            spacing_mode="independent"
        ),
    )

    first = compiled.evaluate(
        {
            "stacking_distance": 4.0,
        }
    )

    second = compiled.evaluate(
        {
            "stacking_distance": 6.0,
        }
    )

    assert np.isclose(
        first.spacing,
        4.0,
    )

    assert np.isclose(
        second.spacing,
        6.0,
    )

    assert np.isclose(
        first.center_span,
        16.0,
    )

    assert np.isclose(
        second.center_span,
        24.0,
    )


def test_parent_c_changes_intralayer_thickness():
    structure = make_corrugated_layer()

    compiled = compile_stacked_geometry(
        structure=structure,
        shape_spec=make_shape(
            3
        ),
        stack_spec=make_stack(
            spacing_mode="independent"
        ),
    )

    first = compiled.evaluate(
        {
            "c": 10.0,
            "stacking_distance": 4.0,
        }
    )

    second = compiled.evaluate(
        {
            "c": 20.0,
            "stacking_distance": 4.0,
        }
    )

    # Two source atoms differ by fractional z = 0.1.
    # Their intralayer separation therefore doubles from 1 Å to 2 Å.
    first_layer = np.where(
        first.atom_layer_indices == 0
    )[0]

    second_layer = np.where(
        second.atom_layer_indices == 0
    )[0]

    first_source_0 = first_layer[
        first.atom_source_site_indices[
            first_layer
        ] == 0
    ][0]

    first_source_1 = first_layer[
        first.atom_source_site_indices[
            first_layer
        ] == 1
    ][0]

    second_source_0 = second_layer[
        second.atom_source_site_indices[
            second_layer
        ] == 0
    ][0]

    second_source_1 = second_layer[
        second.atom_source_site_indices[
            second_layer
        ] == 1
    ][0]

    first_distance = np.linalg.norm(
        first.atom_cart[
            first_source_1
        ]
        - first.atom_cart[
            first_source_0
        ]
    )

    second_distance = np.linalg.norm(
        second.atom_cart[
            second_source_1
        ]
        - second.atom_cart[
            second_source_0
        ]
    )

    assert np.isclose(
        first_distance,
        1.0,
        atol=1.0e-8,
    )

    assert np.isclose(
        second_distance,
        2.0,
        atol=1.0e-8,
    )

    # Independent spacing remains unchanged.
    assert np.isclose(
        first.spacing,
        4.0,
    )

    assert np.isclose(
        second.spacing,
        4.0,
    )


def test_fraction_spacing_follows_c_normal():
    structure = make_corrugated_layer()

    compiled = compile_stacked_geometry(
        structure=structure,
        shape_spec=make_shape(
            3
        ),
        stack_spec=make_stack(
            spacing_mode="fraction_of_c_normal"
        ),
    )

    first = compiled.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.5,
        }
    )

    second = compiled.evaluate(
        {
            "c": 20.0,
            "stacking_fraction": 0.5,
        }
    )

    assert np.isclose(
        first.spacing,
        5.0,
    )

    assert np.isclose(
        second.spacing,
        10.0,
    )


def test_fraction_can_refine_independently():
    structure = make_corrugated_layer()

    compiled = compile_stacked_geometry(
        structure=structure,
        shape_spec=make_shape(
            3
        ),
        stack_spec=make_stack(
            spacing_mode="fraction_of_c_normal"
        ),
    )

    first = compiled.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.4,
        }
    )

    second = compiled.evaluate(
        {
            "c": 10.0,
            "stacking_fraction": 0.7,
        }
    )

    assert np.isclose(
        first.spacing,
        4.0,
    )

    assert np.isclose(
        second.spacing,
        7.0,
    )