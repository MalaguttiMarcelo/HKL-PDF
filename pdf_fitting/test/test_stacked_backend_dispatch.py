from __future__ import annotations

from pdf_fitting.models.crystallite_shapes import (
    shape_spec_from_config,
)

from pdf_fitting.models.layer_stacking import (
    stacking_spec_from_config,
    validate_fixed_stacking_refinement_config,
)


def make_config():
    return {
        "crystallite_shape": {
            "mode": "finite_shape",
            "shape_type": "prism",
            "axis1_cells": 2.0,
            "axis2_cells": 2.0,
            "axis3_cells": 5.0,
        },
        "layer_stacking": {
            "enabled": True,
            "schema_version": 3,
            "model": "periodic_sequence",
            "definition_mode": (
                "repeating_explicit_sequence"
            ),
            "nominal_spacing": 6.0,
            "repeat_period": 3.0,
            "repeat_shift_u": 0.0,
            "repeat_shift_v": 0.0,
            "sequence": [
                {
                    "enabled": True,
                    "label": "A",
                    "shift_u": 0.0,
                    "shift_v": 0.0,
                    "stack_coordinate": 0.0,
                    "normal_delta": 0.0,
                },
                {
                    "enabled": True,
                    "label": "B",
                    "shift_u": 1.0 / 3.0,
                    "shift_v": 2.0 / 3.0,
                    "stack_coordinate": 1.0,
                    "normal_delta": 0.0,
                },
                {
                    "enabled": True,
                    "label": "C",
                    "shift_u": 2.0 / 3.0,
                    "shift_v": 1.0 / 3.0,
                    "stack_coordinate": 2.0,
                    "normal_delta": 0.0,
                },
            ],
        },
        "refinable": {
            "scale": True,
            "delta_g": True,
            "a": False,
            "b": False,
            "c": False,
            "stacking_distance": False,
        },
    }


def test_shape_spec_from_config():
    shape = shape_spec_from_config(
        make_config()
    )

    assert shape.shape_type == "prism"
    assert shape.axis1_cells == 2.0
    assert shape.axis2_cells == 2.0
    assert shape.axis3_cells == 5.0


def test_stacking_spec_from_config():
    stack = stacking_spec_from_config(
        make_config()
    )

    assert stack.enabled
    assert stack.nominal_spacing == 6.0
    assert stack.repeat_period == 3.0
    assert len(stack.sequence) == 3


def test_fixed_geometry_validation_passes():
    invalid = (
        validate_fixed_stacking_refinement_config(
            make_config()
        )
    )

    assert invalid == []


def test_fixed_geometry_validation_catches_lattice():
    config = make_config()

    config["refinable"]["a"] = True
    config["refinable"]["gamma"] = True
    config["refinable"]["stacking_distance"] = True

    invalid = (
        validate_fixed_stacking_refinement_config(
            config
        )
    )

    assert invalid == [
        "a",
        "gamma",
        "stacking_distance",
    ]


def test_fixed_geometry_validation_catches_coordinates():
    config = make_config()

    config["refinable"]["x_c1"] = True
    config["refinable"]["occ_c1"] = True

    invalid = (
        validate_fixed_stacking_refinement_config(
            config
        )
    )

    assert invalid == [
        "occ_c1",
        "x_c1",
    ]