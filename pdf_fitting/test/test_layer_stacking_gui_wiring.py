from __future__ import annotations

import json
import os

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

from PySide6 import QtWidgets

from pdf_fitting.gui.builder import (
    InputBuilder,
)


def get_application():
    application = QtWidgets.QApplication.instance()

    if application is None:
        application = QtWidgets.QApplication([])

    return application


def abc_stacking_spec():
    return {
        "schema_version": 3,
        "enabled": True,
        "model": "periodic_sequence",
        "definition_mode": (
            "repeating_explicit_sequence"
        ),
        "basis_mode": "ab_plane",
        "nominal_spacing": 6.0,
        "refine_nominal_spacing": False,
        "nominal_spacing_min": 5.0,
        "nominal_spacing_max": 7.0,
        "repeat_period": 3.0,
        "repeat_shift_u": 0.0,
        "repeat_shift_v": 0.0,
        "show_transition_vectors": True,
        "color_layers": True,
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
    }


def finite_shape_spec():
    return {
        "mode": "finite_shape",
        "schema_version": 3,
        "search_mode": "predefined",
        "shape_type": "prism",
        "axis1_cells": 2.0,
        "axis2_cells": 2.0,
        "axis3_cells": 5.0,
        "diameter_cells": 2.0,
        "height_cells": 5.0,
        "axis_h": 0,
        "axis_k": 0,
        "axis_l": 1,
        "base1_h": 1,
        "base1_k": 0,
        "base1_l": 0,
        "base2_h": 0,
        "base2_k": 1,
        "base2_l": 0,
    }


def test_builder_exports_periodic_stacking():
    application = get_application()
    _ = application

    builder = InputBuilder()

    builder.gb_size.setChecked(
        True
    )

    builder.combo_size_model.setCurrentIndex(
        2
    )

    builder._crystallite_shape_spec = (
        finite_shape_spec()
    )

    builder._layer_stacking_spec = (
        abc_stacking_spec()
    )

    updates = builder.to_updates()

    assert "layer_stacking" in updates
    assert "stacking_faults" not in updates

    stacking = updates[
        "layer_stacking"
    ]

    assert stacking[
        "enabled"
    ] is True

    assert stacking[
        "repeat_period"
    ] == 3.0

    sequence = json.loads(
        stacking[
            "sequence_json"
        ]
    )

    assert len(sequence) == 3

    assert [
        row["label"]
        for row in sequence
    ] == [
        "A",
        "B",
        "C",
    ]

    assert updates[
        "initial_values"
    ][
        "stacking_distance"
    ] == 6.0

    # Fixed-geometry stage: spacing refinement is forced off.
    assert updates[
        "refinable_parameters"
    ][
        "stacking_distance"
    ] is False

    assert "stacking_faults" in updates[
        "__delete_sections__"
    ]

    builder.deleteLater()


def test_builder_loads_periodic_stacking():
    application = get_application()
    _ = application

    builder = InputBuilder()

    config = {
        "structure_file": "",
        "gr_data_file": "",
        "r_min": 1.0,
        "r_max": 10.0,
        "r_extension": 1.2,
        "refinement": {},
        "initial": {
            "scale": 1.0,
        },
        "refinable": {
            "scale": False,
        },
        "bounds": {},
        "constraints": {},
        "crystallite_shape": finite_shape_spec(),
        "layer_stacking": abc_stacking_spec(),
    }

    builder.set_from_config(
        config
    )

    assert builder._crystallite_shape_spec[
        "mode"
    ] == "finite_shape"

    assert builder._layer_stacking_spec[
        "enabled"
    ] is True

    assert builder._layer_stacking_spec[
        "nominal_spacing"
    ] == 6.0

    assert len(
        builder._layer_stacking_spec[
            "sequence"
        ]
    ) == 3

    assert (
        builder.combo_size_model.currentIndex()
        == 2
    )

    builder.deleteLater()