from __future__ import annotations

import numpy as np

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.dynamic_stacked_model import (
    DynamicStackedPdfModel,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
)


class FakePdfCalculator:
    """
    Minimal PDFCalculator substitute for wrapper tests.
    """

    def __init__(self):
        self.shell_table = None
        self.update_calls = 0
        self.last_update = None

        self.constraints = {}
        self.structure_handler = None

    def replace_shell_table(
        self,
        shell_table,
        *,
        device_shell_table=None,
    ):
        self.shell_table = shell_table

    def update_dynamic_stacked_geometry(
        self,
        **kwargs,
    ):
        self.update_calls += 1
        self.last_update = kwargs

    def evaluate(
        self,
        parameters,
        **kwargs,
    ):
        shell_cart = np.asarray(
            self.last_update[
                "shell_cart"
            ],
            dtype=float,
        )

        distances = np.sqrt(
            np.sum(
                shell_cart
                * shell_cart,
                axis=1,
            )
        )

        return {
            "r": distances,
            "G_r": distances.copy(),
        }

    def build_warren_plot(
        self,
        parameters,
        **kwargs,
    ):
        return {}

    def build_local_trends(
        self,
        parameters,
        **kwargs,
    ):
        return {}

    def compute_isotropic_shape_factor(
        self,
        parameters,
        r_grid,
    ):
        return np.zeros_like(
            r_grid,
            dtype=float,
        )

    def build_shape_factor_directions(
        self,
        parameters,
        **kwargs,
    ):
        return {}

    def build_pdf_tick_data(
        self,
        parameters,
        **kwargs,
    ):
        return {
            "pairs": {},
        }


def make_structure():
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


def make_shape():
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


def make_stack():
    return LayerStackSpec(
        enabled=True,
        vector_mode="incremental",
        spacing_mode="independent",
        nominal_spacing=4.0,
        repeat_period=1.0,
        repeat_shift_u=-0.25,
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
                shift_u=0.25,
                shift_v=0.0,
                stack_coordinate=1.0,
            ),
        ],
    )


def make_model():
    calculator = FakePdfCalculator()

    model = DynamicStackedPdfModel(
        pdf_calculator=calculator,
        source_structure=make_structure(),
        shape_spec=make_shape(),
        stack_spec=make_stack(),
        pair_cutoff=8.0,
        topology_margin=3.0,
        reference_parameters={
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
        },
    )

    return model, calculator


def test_model_initializes_shell_table():
    model, calculator = make_model()

    assert calculator.shell_table is not None

    assert calculator.shell_table[
        "geometry_mode"
    ] == "dynamic_stacked_supercell"

    assert model.geometry_update_count == 1

    assert calculator.update_calls == 1


def test_non_geometry_parameters_do_not_update_geometry():
    model, calculator = make_model()

    initial_count = model.geometry_update_count
    initial_calls = calculator.update_calls

    model.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
            "scale": 2.0,
            "biso_C": 4.0,
        }
    )

    assert (
        model.geometry_update_count
        == initial_count
    )

    assert (
        calculator.update_calls
        == initial_calls
    )


def test_lattice_change_updates_geometry():
    model, calculator = make_model()

    initial_count = model.geometry_update_count

    first = model.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
        }
    )

    second = model.evaluate(
        {
            "a": 4.3,
            "b": 5.2,
            "c": 11.0,
            "stacking_distance": 4.0,
        }
    )

    assert (
        model.geometry_update_count
        == initial_count + 1
    )

    assert not np.allclose(
        first[
            "G_r"
        ],
        second[
            "G_r"
        ],
    )


def test_spacing_change_updates_geometry():
    model, calculator = make_model()

    initial_count = model.geometry_update_count

    first = model.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
        }
    )

    second = model.evaluate(
        {
            "a": 4.0,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 5.0,
        }
    )

    assert (
        model.geometry_update_count
        == initial_count + 1
    )

    assert not np.allclose(
        first[
            "G_r"
        ],
        second[
            "G_r"
        ],
    )


def test_repeated_identical_geometry_is_cached():
    model, calculator = make_model()

    model.evaluate(
        {
            "a": 4.2,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
        }
    )

    count_after_first = (
        model.geometry_update_count
    )

    model.evaluate(
        {
            "a": 4.2,
            "b": 5.0,
            "c": 10.0,
            "stacking_distance": 4.0,
            "scale": 3.0,
        }
    )

    assert (
        model.geometry_update_count
        == count_after_first
    )