from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    absolute_sequence_layers,
    effective_repeat_translation,
)


@dataclass
class StackingTransition:
    index: int
    transition_id: str
    source_label: str
    target_label: str

    du: float
    dv: float
    h: float

    closing: bool = False

    @property
    def suffix(
        self,
    ) -> str:
        if self.closing:
            return "close"

        return str(
            self.index
        )

    @property
    def parameter_du(
        self,
    ) -> str:
        return f"stack_du_{self.suffix}"

    @property
    def parameter_dv(
        self,
    ) -> str:
        return f"stack_dv_{self.suffix}"

    @property
    def parameter_h(
        self,
    ) -> str:
        return f"stack_h_{self.suffix}"


def transition_parameter_names(
    number_of_rows: int,
) -> List[str]:
    number_of_rows = max(
        1,
        int(
            number_of_rows
        ),
    )

    names = []

    for index in range(
        1,
        number_of_rows,
    ):
        names.extend(
            [
                f"stack_du_{index}",
                f"stack_dv_{index}",
                f"stack_h_{index}",
            ]
        )

    names.extend(
        [
            "stack_du_close",
            "stack_dv_close",
            "stack_h_close",
        ]
    )

    return names


def transitions_from_stack_spec(
    stack_spec: LayerStackSpec,
) -> List[StackingTransition]:
    spec = stack_spec.normalized()

    if spec.vector_mode != "incremental":
        raise ValueError(
            "Transition refinement requires incremental vector mode."
        )

    sequence = list(
        spec.sequence
    )

    if not sequence:
        raise ValueError(
            "The stacking sequence contains no rows."
        )

    absolute_layers = absolute_sequence_layers(
        spec
    )

    repeat_u, repeat_v, repeat_s = (
        effective_repeat_translation(
            spec
        )
    )

    transitions = []

    for index in range(
        1,
        len(sequence),
    ):
        previous = absolute_layers[
            index - 1
        ]

        current = absolute_layers[
            index
        ]

        delta_s = (
            float(
                current.stack_coordinate
            )
            - float(
                previous.stack_coordinate
            )
        )

        delta_z = (
            float(
                current.normal_delta
            )
            - float(
                previous.normal_delta
            )
        )

        transitions.append(
            StackingTransition(
                index=index,
                transition_id=f"layer:{sequence[index].uid}",
                source_label=str(
                    sequence[index - 1].label
                ),
                target_label=str(
                    sequence[index].label
                ),
                du=float(
                    current.shift_u
                    - previous.shift_u
                ),
                dv=float(
                    current.shift_v
                    - previous.shift_v
                ),
                h=float(
                    delta_s
                    * spec.nominal_spacing
                    + delta_z
                ),
                closing=False,
            )
        )

    final_layer = absolute_layers[
        -1
    ]

    closing_h = (
        float(
            repeat_s
        )
        * spec.nominal_spacing
        - (
            float(
                final_layer.stack_coordinate
            )
            * spec.nominal_spacing
            + float(
                final_layer.normal_delta
            )
        )
    )

    transitions.append(
        StackingTransition(
            index=len(sequence),
            transition_id="closure",
            source_label=str(
                sequence[-1].label
            ),
            target_label=str(
                sequence[0].label
            ),
            du=float(
                repeat_u
                - final_layer.shift_u
            ),
            dv=float(
                repeat_v
                - final_layer.shift_v
            ),
            h=float(
                closing_h
            ),
            closing=True,
        )
    )

    return transitions


def evaluate_transition_values(
    stack_spec: LayerStackSpec,
    parameters: Mapping[str, Any] | None,
) -> Tuple[
    List[StackingTransition],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    parameters = dict(
        parameters
        or {}
    )

    transitions = transitions_from_stack_spec(
        stack_spec
    )

    du = np.asarray(
        [
            float(
                parameters.get(
                    transition.parameter_du,
                    transition.du,
                )
            )
            for transition in transitions
        ],
        dtype=float,
    )

    dv = np.asarray(
        [
            float(
                parameters.get(
                    transition.parameter_dv,
                    transition.dv,
                )
            )
            for transition in transitions
        ],
        dtype=float,
    )

    h = np.asarray(
        [
            float(
                parameters.get(
                    transition.parameter_h,
                    transition.h,
                )
            )
            for transition in transitions
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            du
        )
    ):
        raise ValueError(
            "A stacking Δu parameter is non-finite."
        )

    if not np.all(
        np.isfinite(
            dv
        )
    ):
        raise ValueError(
            "A stacking Δv parameter is non-finite."
        )

    if (
        not np.all(
            np.isfinite(
                h
            )
        )
        or np.any(
            h <= 0.0
        )
    ):
        raise ValueError(
            "Every stacking transition height must be positive and finite."
        )

    return (
        transitions,
        du,
        dv,
        h,
    )


def apply_transition_parameters_to_stack_spec(
    stack_spec: LayerStackSpec,
    parameters: Mapping[str, Any] | None,
) -> LayerStackSpec:
    """
    Apply generic transition parameters to a LayerStackSpec.

    This converts refined stack_du_*, stack_dv_* and stack_h_* values back to
    the existing incremental sequence/repeat representation used by the GUI.
    """
    parameters = dict(
        parameters
        or {}
    )

    candidate = LayerStackSpec.from_dict(
        stack_spec.to_dict()
    )

    if candidate.vector_mode != "incremental":
        return candidate

    if (
        candidate.spacing_mode == "independent"
        and "stacking_distance" in parameters
    ):
        candidate.nominal_spacing = max(
            float(
                parameters[
                    "stacking_distance"
                ]
            ),
            1.0e-12,
        )

    (
        transitions,
        transition_du,
        transition_dv,
        transition_h,
    ) = evaluate_transition_values(
        candidate,
        parameters,
    )

    number_of_rows = len(
        candidate.sequence
    )

    if (
        number_of_rows <= 0
        or len(
            transitions
        )
        != number_of_rows
    ):
        return candidate

    spacing = max(
        float(
            candidate.nominal_spacing
        ),
        1.0e-12,
    )

    # Rows 1...N-1 directly store incremental transitions.
    for row in range(
        1,
        number_of_rows,
    ):
        transition_index = (
            row
            - 1
        )

        candidate.sequence[
            row
        ].shift_u = float(
            transition_du[
                transition_index
            ]
        )

        candidate.sequence[
            row
        ].shift_v = float(
            transition_dv[
                transition_index
            ]
        )

        candidate.sequence[
            row
        ].stack_coordinate = 1.0

        candidate.sequence[
            row
        ].normal_delta = (
            float(
                transition_h[
                    transition_index
                ]
            )
            - spacing
        )

    absolute_layers = absolute_sequence_layers(
        candidate
    )

    final_layer = absolute_layers[
        -1
    ]

    closure_du = float(
        transition_du[
            -1
        ]
    )

    closure_dv = float(
        transition_dv[
            -1
        ]
    )

    closure_h = float(
        transition_h[
            -1
        ]
    )

    candidate.repeat_shift_u = (
        float(
            final_layer.shift_u
        )
        + closure_du
    )

    candidate.repeat_shift_v = (
        float(
            final_layer.shift_v
        )
        + closure_dv
    )

    candidate.repeat_period = max(
        (
            closure_h
            + float(
                final_layer.normal_delta
            )
        )
        / spacing,
        1.0e-12,
    )

    return candidate.normalized()