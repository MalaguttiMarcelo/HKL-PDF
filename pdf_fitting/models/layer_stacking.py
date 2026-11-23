from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence
from uuid import uuid4

import ast
import json
import numpy as np


_EPSILON = 1.0e-12


def _new_uid() -> str:
    return uuid4().hex


def _as_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        return False

    return bool(default)

def _decode_structured_ini_value(
    value: Any,
    default,
):
    """
    Decode a list/matrix stored in an INI string.

    Supports both valid JSON:

        [{"label":"A","du":0.0,"dv":0.0,"h":3.8}]
        [[0.0,1.0],[1.0,0.0]]

    and older Python-literal text:

        [{'label': 'A', 'du': 0.0, 'dv': 0.0, 'h': 3.8}]
        [[0.0, 1.0], [1.0, 0.0]]

    ast.literal_eval is safe for literals and does not execute code.
    """
    if value is None:
        return default

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return list(
            value
        )

    if not isinstance(
        value,
        str,
    ):
        return default

    text = value.strip()

    if not text:
        return default

    try:
        return json.loads(
            text
        )
    except Exception:
        pass

    try:
        return ast.literal_eval(
            text
        )
    except Exception:
        return default

@dataclass
class SequenceLayer:
    """
    One row in an infinitely repeated stacking sequence.

    shift_u and shift_v are absolute fractional translations along the
    input structure's a and b vectors.

    stack_coordinate is dimensionless. Its physical contribution along
    the stacking normal is:

        stack_coordinate * nominal_spacing

    normal_delta is an optional additional displacement in Angstrom.
    """

    uid: str = field(
        default_factory=_new_uid
    )

    enabled: bool = True
    label: str = "A"

    shift_u: float = 0.0
    shift_v: float = 0.0

    stack_coordinate: float = 0.0
    normal_delta: float = 0.0

    def normal_offset(
        self,
        nominal_spacing: float,
        *,
        repeat_index: int = 0,
        repeat_period: float = 1.0,
    ) -> float:
        realized_coordinate = (
            float(self.stack_coordinate)
            + float(repeat_index)
            * float(repeat_period)
        )

        return (
            realized_coordinate
            * float(nominal_spacing)
            + float(self.normal_delta)
        )



    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
        *,
        default_coordinate: float = 0.0,
        nominal_spacing: float = 1.0,
    ) -> "SequenceLayer":
        values = dict(values or {})

        if "stack_coordinate" in values:
            stack_coordinate = float(
                values.get(
                    "stack_coordinate",
                    default_coordinate,
                )
            )

            normal_delta = float(
                values.get(
                    "normal_delta",
                    0.0,
                )
            )

        elif "normal_offset" in values:
            spacing = max(
                abs(float(nominal_spacing)),
                _EPSILON,
            )

            stack_coordinate = float(
                values.get(
                    "normal_offset",
                    0.0,
                )
            ) / spacing

            normal_delta = 0.0

        else:
            stack_coordinate = float(
                default_coordinate
            )

            normal_delta = 0.0

        return cls(
            uid=str(
                values.get(
                    "uid",
                    "",
                )
                or _new_uid()
            ),
            enabled=_as_bool(
                values.get(
                    "enabled",
                    True,
                ),
                True,
            ),
            label=str(
                values.get(
                    "label",
                    "A",
                )
                or "A"
            ),
            shift_u=float(
                values.get(
                    "shift_u",
                    0.0,
                )
            ),
            shift_v=float(
                values.get(
                    "shift_v",
                    0.0,
                )
            ),
            stack_coordinate=stack_coordinate,
            normal_delta=normal_delta,
        )


@dataclass
class RealizedLayer:
    """
    One temporary layer generated from the infinite periodic sequence.

    Realized layers are used by the preview and particle builder. They are
    not saved as the authoritative stacking definition.
    """

    realized_index: int
    sequence_index: int
    repeat_index: int

    label: str

    shift_u: float
    shift_v: float

    stack_coordinate: float
    normal_offset: float

    enabled: bool = True
    atom_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_a_sequence() -> List[SequenceLayer]:
    return [
        SequenceLayer(
            label="A",
            shift_u=0.0,
            shift_v=0.0,
            stack_coordinate=0.0,
            normal_delta=0.0,
        )
    ]


def default_ab_sequence() -> List[SequenceLayer]:
    return [
        SequenceLayer(
            label="A",
            shift_u=0.0,
            shift_v=0.0,
            stack_coordinate=0.0,
            normal_delta=0.0,
        ),
        SequenceLayer(
            label="B",
            shift_u=1.0 / 3.0,
            shift_v=2.0 / 3.0,
            stack_coordinate=1.0,
            normal_delta=0.0,
        ),
    ]


def default_abc_sequence() -> List[SequenceLayer]:
    return [
        SequenceLayer(
            label="A",
            shift_u=0.0,
            shift_v=0.0,
            stack_coordinate=0.0,
            normal_delta=0.0,
        ),
        SequenceLayer(
            label="B",
            shift_u=1.0 / 3.0,
            shift_v=2.0 / 3.0,
            stack_coordinate=1.0,
            normal_delta=0.0,
        ),
        SequenceLayer(
            label="C",
            shift_u=2.0 / 3.0,
            shift_v=1.0 / 3.0,
            stack_coordinate=2.0,
            normal_delta=0.0,
        ),
    ]

@dataclass
class StackingMarkovState:
    """
    One emitted transition vector in the generic N-state stochastic model.

    The current state emits this vector before the next state is selected
    from the Markov transition matrix.
    """

    uid: str = field(
        default_factory=_new_uid
    )

    label: str = "State"

    du: float = 0.0
    dv: float = 0.0
    h: float = 3.8

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
        *,
        default_label: str = "State",
    ) -> "StackingMarkovState":
        values = dict(
            values
            or {}
        )

        return cls(
            uid=str(
                values.get(
                    "uid",
                    "",
                )
                or _new_uid()
            ),
            label=str(
                values.get(
                    "label",
                    default_label,
                )
                or default_label
            ),
            du=float(
                values.get(
                    "du",
                    0.0,
                )
            ),
            dv=float(
                values.get(
                    "dv",
                    0.0,
                )
            ),
            h=max(
                float(
                    values.get(
                        "h",
                        3.8,
                    )
                ),
                _EPSILON,
            ),
        ) 

@dataclass
class LayerStackSpec:
    """
    Infinite periodic stacking specification.

    Both supported definition modes repeat indefinitely:

        repeating_motif
        repeating_explicit_sequence

    The distinction controls how the model is presented to the user. The
    mathematical realization is periodic in both cases.
    """

    enabled: bool = False

    schema_version: int = 3
    model: str = "periodic_sequence"

    # ---------------------------------------------------------
    # TOPAS-like correlated A/B stochastic stacking model.
    #
    # These settings are used only when:
    #
    #     model = "correlated_ab_disorder"
    #
    # The two vectors are emitted by the CURRENT state:
    #
    #     current A -> use state_a vector
    #     current B -> use state_b vector
    #
    # The next state is then chosen with p_alternate.
    # ---------------------------------------------------------
    p_alternate: float = 1.0

    p_grid_enabled: bool = False
    p_grid_minimum: float = 0.0
    p_grid_maximum: float = 1.0
    p_grid_step: float = 0.05

    ensemble_samples: int = 20
    ensemble_seed: int = 12345
    ensemble_start_state: str = "A"

    # Stochastic preview only.
    preview_realization_index: int = 0

    # Current-state transition vectors:
    #
    # State A emits this vector before choosing A/B next state.
    state_a_du: float = 0.0
    state_a_dv: float = 0.0
    state_a_h: float = 3.8

    # State B emits this vector before choosing A/B next state.
    state_b_du: float = 0.0
    state_b_dv: float = 0.0
    state_b_h: float = 3.8

    # ---------------------------------------------------------
    # Generic arbitrary-state Markov stacking.
    #
    # Used when:
    #
    #     model = "generic_markov"
    #
    # Every state emits its own du/dv/h transition vector.
    # The next state is selected from markov_transition_matrix.
    # ---------------------------------------------------------
    markov_states: List[
        StackingMarkovState
    ] = field(
        default_factory=list
    )

    markov_transition_matrix: List[
        List[float]
    ] = field(
        default_factory=list
    )

    markov_probability_controls: List[
        List[
            Dict[str, Any]
        ]
    ] = field(
        default_factory=list
    )

    markov_start_state: int = 0

    definition_mode: str = "repeating_motif"

    # Interpretation of sequence-row u, v, stack_coordinate and normal_delta.
    #
    # incremental:
    #     Every row after row zero stores the displacement from the previous
    #     row. Row zero is the reference layer.
    #
    # absolute:
    #     Every row stores its absolute placement relative to sequence row zero.
    vector_mode: str = "incremental"

    # ---------------------------------------------------------
    # Interlayer-spacing model
    # ---------------------------------------------------------
    #
    # independent:
    #     The spacing is an independent distance in Å.
    #
    # fraction_of_c_normal:
    #     The spacing is:
    #
    #         stacking_fraction
    #         * abs(dot(parent_c, normal(parent_a, parent_b)))
    #
    #     This constrains the spacing to the component of the parent c
    #     vector normal to the layer plane.
    spacing_mode: str = "independent"

    nominal_spacing: float = 5.0

    refine_nominal_spacing: bool = False
    nominal_spacing_min: float = 0.1
    nominal_spacing_max: float = 100.0

    stacking_fraction: float = 1.0

    refine_stacking_fraction: bool = False
    stacking_fraction_min: float = 0.001
    stacking_fraction_max: float = 100.0

    basis_mode: str = "ab_plane"

    repeat_period: float = 2.0
    repeat_shift_u: float = 0.0
    repeat_shift_v: float = 0.0

    sequence: List[SequenceLayer] = field(
        default_factory=default_ab_sequence
    )

    show_transition_vectors: bool = True
    color_layers: bool = True

    def normalized(self) -> "LayerStackSpec":

        model = str(
            self.model
            or "periodic_sequence"
        ).strip().lower()

        model_aliases = {
            "periodic": "periodic_sequence",
            "sequence": "periodic_sequence",
            "explicit": "periodic_sequence",
            "explicit_sequence": "periodic_sequence",
            "markov": "correlated_ab_disorder",
            "markov_ab": "correlated_ab_disorder",
            "topas_d02": "correlated_ab_disorder",
            "topas_d02_2": "correlated_ab_disorder",
            "correlated_ab": "correlated_ab_disorder",
            "stochastic_ab": "correlated_ab_disorder",
            "generic_markov": "generic_markov",
            "markov_n_state": "generic_markov",
            "multistate_markov": "generic_markov",
            "stochastic_multistate": "generic_markov",
        }

        model = model_aliases.get(
            model,
            model,
        )

        if model not in {
            "periodic_sequence",
            "correlated_ab_disorder",
            "generic_markov",
        }:
            model = "periodic_sequence"

        p_alternate = float(
            self.p_alternate
        )

        p_alternate = min(
            1.0,
            max(
                0.0,
                p_alternate,
            ),
        )

        p_grid_minimum = min(
            1.0,
            max(
                0.0,
                float(
                    self.p_grid_minimum
                ),
            ),
        )

        p_grid_maximum = min(
            1.0,
            max(
                0.0,
                float(
                    self.p_grid_maximum
                ),
            ),
        )

        if p_grid_maximum < p_grid_minimum:
            p_grid_minimum, p_grid_maximum = (
                p_grid_maximum,
                p_grid_minimum,
            )

        p_grid_step = max(
            float(
                self.p_grid_step
            ),
            1.0e-6,
        )

        ensemble_samples = max(
            1,
            int(
                self.ensemble_samples
            ),
        )

        ensemble_start_state = str(
            self.ensemble_start_state
            or "A"
        ).strip().upper()

        if ensemble_start_state not in {
            "A",
            "B",
        }:
            ensemble_start_state = "A"

        state_a_h = max(
            float(
                self.state_a_h
            ),
            _EPSILON,
        )

        state_b_h = max(
            float(
                self.state_b_h
            ),
            _EPSILON,
        )

        spacing = max(
            abs(float(self.nominal_spacing)),
            _EPSILON,
        )

        minimum = max(
            float(self.nominal_spacing_min),
            _EPSILON,
        )

        maximum = max(
            float(self.nominal_spacing_max),
            minimum,
        )

        spacing_mode = str(
            self.spacing_mode
            or "independent"
        ).strip().lower()

        spacing_mode_aliases = {
            "distance": "independent",
            "independent_distance": "independent",
            "independent_angstrom": "independent",
            "fraction": "fraction_of_c_normal",
            "fraction_of_c": "fraction_of_c_normal",
            "c_fraction": "fraction_of_c_normal",
        }

        spacing_mode = spacing_mode_aliases.get(
            spacing_mode,
            spacing_mode,
        )

        if spacing_mode not in {
            "independent",
            "fraction_of_c_normal",
        }:
            spacing_mode = "independent"

        stacking_fraction = max(
            abs(
                float(
                    self.stacking_fraction
                )
            ),
            _EPSILON,
        )

        stacking_fraction_minimum = max(
            float(
                self.stacking_fraction_min
            ),
            _EPSILON,
        )

        stacking_fraction_maximum = max(
            float(
                self.stacking_fraction_max
            ),
            stacking_fraction_minimum,
        )

        mode = str(
            self.definition_mode
            or "repeating_motif"
        ).strip().lower()

        if mode not in {
            "repeating_motif",
            "repeating_explicit_sequence",
        }:
            mode = "repeating_motif"

        vector_mode = str(
            self.vector_mode
            or "incremental"
        ).strip().lower()

        if vector_mode not in {
            "incremental",
            "absolute",
        }:
            vector_mode = "incremental"

        sequence: List[SequenceLayer] = []

        for row_index, layer in enumerate(
            self.sequence
        ):
            if isinstance(layer, SequenceLayer):
                source = layer
            else:
                source = SequenceLayer.from_dict(
                    layer,
                    default_coordinate=float(
                        row_index
                    ),
                    nominal_spacing=spacing,
                )

            sequence.append(
                SequenceLayer(
                    uid=str(
                        source.uid
                        or _new_uid()
                    ),
                    enabled=bool(
                        source.enabled
                    ),
                    label=str(
                        source.label
                        or f"L{row_index}"
                    ),
                    shift_u=float(
                        source.shift_u
                    ),
                    shift_v=float(
                        source.shift_v
                    ),
                    stack_coordinate=float(
                        source.stack_coordinate
                    ),
                    normal_delta=float(
                        source.normal_delta
                    ),
                )
            )

        if not sequence:
            sequence = default_a_sequence()

        repeat_period = float(
            self.repeat_period
        )

        if abs(repeat_period) <= _EPSILON:
            coordinates = [
                float(layer.stack_coordinate)
                for layer in sequence
            ]

            if len(coordinates) >= 2:
                differences = np.diff(
                    np.asarray(
                        sorted(coordinates),
                        dtype=float,
                    )
                )

                positive = differences[
                    differences > _EPSILON
                ]

                step = (
                    float(np.median(positive))
                    if positive.size
                    else 1.0
                )

                repeat_period = (
                    max(coordinates)
                    - min(coordinates)
                    + step
                )
            else:
                repeat_period = 1.0

        if repeat_period < 0.0:
            repeat_period = abs(
                repeat_period
            )

        markov_states = []

        for state_index, raw_state in enumerate(
            self.markov_states
            or []
        ):
            if isinstance(
                raw_state,
                StackingMarkovState,
            ):
                state = raw_state
            else:
                state = StackingMarkovState.from_dict(
                    raw_state,
                    default_label=f"S{state_index}",
                )

            markov_states.append(
                StackingMarkovState(
                    uid=str(
                        state.uid
                        or _new_uid()
                    ),
                    label=str(
                        state.label
                        or f"S{state_index}"
                    ),
                    du=float(
                        state.du
                    ),
                    dv=float(
                        state.dv
                    ),
                    h=max(
                        float(
                            state.h
                        ),
                        _EPSILON,
                    ),
                )
            )

        if (
            model == "generic_markov"
            and not markov_states
        ):
            markov_states = default_generic_markov_states()

        number_of_markov_states = max(
            1,
            len(
                markov_states
            ),
        )

        raw_markov_matrix = np.asarray(
            self.markov_transition_matrix,
            dtype=float,
        )

        if (
            model == "generic_markov"
            and raw_markov_matrix.shape
            != (
                number_of_markov_states,
                number_of_markov_states,
            )
        ):
            raw_markov_matrix = np.asarray(
                default_generic_markov_matrix(),
                dtype=float,
            )

        markov_matrix = normalize_markov_matrix(
            raw_markov_matrix,
            number_of_states=number_of_markov_states,
        )

        markov_probability_controls = (
            normalize_markov_probability_controls(
                self.markov_probability_controls,
                number_of_states=number_of_markov_states,
                transition_matrix=markov_matrix,
            )
        )

        markov_start_state = max(
            0,
            min(
                int(
                    self.markov_start_state
                ),
                max(
                    0,
                    len(
                        markov_states
                    )
                    - 1,
                ),
            ),
        )


        return LayerStackSpec(
            enabled=bool(
                self.enabled
            ),
            schema_version=3,
            model=model,
            p_alternate=p_alternate,
            p_grid_enabled=bool(
                self.p_grid_enabled
            ),
            p_grid_minimum=p_grid_minimum,
            p_grid_maximum=p_grid_maximum,
            p_grid_step=p_grid_step,
            ensemble_samples=ensemble_samples,
            ensemble_seed=int(
                self.ensemble_seed
            ),
            ensemble_start_state=ensemble_start_state,
            preview_realization_index=max(
                0,
                int(
                    self.preview_realization_index
                ),
            ),
            state_a_du=float(
                self.state_a_du
            ),
            state_a_dv=float(
                self.state_a_dv
            ),
            state_a_h=state_a_h,
            state_b_du=float(
                self.state_b_du
            ),
            state_b_dv=float(
                self.state_b_dv
            ),
            state_b_h=state_b_h,
            markov_states=markov_states,
            markov_transition_matrix=markov_matrix,
            markov_probability_controls=(
                markov_probability_controls
            ),
            markov_start_state=markov_start_state,
            definition_mode=mode,
            vector_mode=vector_mode,
            spacing_mode=spacing_mode,
            nominal_spacing=spacing,
            refine_nominal_spacing=bool(
                self.refine_nominal_spacing
            ),
            nominal_spacing_min=minimum,
            nominal_spacing_max=maximum,
            stacking_fraction=stacking_fraction,
            refine_stacking_fraction=bool(
                self.refine_stacking_fraction
            ),
            stacking_fraction_min=(
                stacking_fraction_minimum
            ),
            stacking_fraction_max=(
                stacking_fraction_maximum
            ),
            basis_mode="ab_plane",
            repeat_period=max(
                repeat_period,
                _EPSILON,
            ),
            repeat_shift_u=float(
                self.repeat_shift_u
            ),
            repeat_shift_v=float(
                self.repeat_shift_v
            ),
            sequence=sequence,
            show_transition_vectors=bool(
                self.show_transition_vectors
            ),
            color_layers=bool(
                self.color_layers
            ),
        )

    def enabled_sequence(
        self,
    ) -> List[SequenceLayer]:
        """
        Return enabled rows from the normalized periodic sequence.

        Disabled rows retain their positions in the complete periodic
        sequence, but they do not generate atoms.
        """
        normalized = self.normalized()

        return [
            layer
            for layer in normalized.sequence
            if bool(
                layer.enabled
            )
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        normalized = self.normalized()

        return {
            "enabled": normalized.enabled,
            "schema_version": 3,
            "model": normalized.model,
            "p_alternate": normalized.p_alternate,
            "p_same_state": (
                1.0
                - normalized.p_alternate
            ),
            "p_grid_enabled": normalized.p_grid_enabled,
            "p_grid_minimum": normalized.p_grid_minimum,
            "p_grid_maximum": normalized.p_grid_maximum,
            "p_grid_step": normalized.p_grid_step,
            "ensemble_samples": normalized.ensemble_samples,
            "ensemble_seed": normalized.ensemble_seed,
            "ensemble_start_state": normalized.ensemble_start_state,
            "preview_realization_index": (
                normalized.preview_realization_index
            ),
            "state_a_du": normalized.state_a_du,
            "state_a_dv": normalized.state_a_dv,
            "state_a_h": normalized.state_a_h,
            "state_b_du": normalized.state_b_du,
            "state_b_dv": normalized.state_b_dv,
            "state_b_h": normalized.state_b_h,
            "markov_states": [
                state.to_dict()
                for state in normalized.markov_states
            ],
            "markov_transition_matrix": (
                normalized.markov_transition_matrix
            ),
            "markov_probability_controls": (
                normalized.markov_probability_controls
            ),
            "markov_start_state": (
                normalized.markov_start_state
            ),
            "definition_mode": (
                normalized.definition_mode
            ),
            "vector_mode": (
                normalized.vector_mode
            ),
            "basis_mode": "ab_plane",
            "spacing_mode": (
                normalized.spacing_mode
            ),
            "nominal_spacing": (
                normalized.nominal_spacing
            ),
            "refine_nominal_spacing": (
                normalized.refine_nominal_spacing
            ),
            "nominal_spacing_min": (
                normalized.nominal_spacing_min
            ),
            "nominal_spacing_max": (
                normalized.nominal_spacing_max
            ),
            "stacking_fraction": (
                normalized.stacking_fraction
            ),
            "refine_stacking_fraction": (
                normalized.refine_stacking_fraction
            ),
            "stacking_fraction_min": (
                normalized.stacking_fraction_min
            ),
            "stacking_fraction_max": (
                normalized.stacking_fraction_max
            ),
            "repeat_period": (
                normalized.repeat_period
            ),
            "repeat_shift_u": (
                normalized.repeat_shift_u
            ),
            "repeat_shift_v": (
                normalized.repeat_shift_v
            ),
            "sequence": [
                layer.to_dict()
                for layer in normalized.sequence
            ],
            "show_transition_vectors": (
                normalized.show_transition_vectors
            ),
            "color_layers": (
                normalized.color_layers
            ),
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "LayerStackSpec":
        values = dict(values or {})

        spacing = float(
            values.get(
                "nominal_spacing",
                values.get(
                    "stacking_distance",
                    5.0,
                ),
            )
        )

        raw_sequence = values.get(
            "sequence",
            values.get(
                "layers",
                [],
            ),
        )

        sequence: List[SequenceLayer] = []

        if isinstance(
            raw_sequence,
            Sequence,
        ) and not isinstance(
            raw_sequence,
            (str, bytes),
        ):
            for row_index, raw_layer in enumerate(
                raw_sequence
            ):
                if isinstance(
                    raw_layer,
                    SequenceLayer,
                ):
                    sequence.append(
                        raw_layer
                    )

                elif isinstance(
                    raw_layer,
                    Mapping,
                ):
                    sequence.append(
                        SequenceLayer.from_dict(
                            raw_layer,
                            default_coordinate=float(
                                row_index
                            ),
                            nominal_spacing=spacing,
                        )
                    )

        if not sequence:
            sequence = default_ab_sequence()

        repeat_period = values.get(
            "repeat_period",
            None,
        )

        if repeat_period is None:
            coordinates = [
                float(layer.stack_coordinate)
                for layer in sequence
            ]

            if len(coordinates) >= 2:
                repeat_period = (
                    max(coordinates)
                    - min(coordinates)
                    + 1.0
                )
            else:
                repeat_period = 1.0

        raw_markov_states = _decode_structured_ini_value(
            values.get(
                "markov_states",
                values.get(
                    "markov_states_json",
                    [],
                ),
            ),
            [],
        )

        markov_states = []

        if isinstance(
            raw_markov_states,
            Sequence,
        ) and not isinstance(
            raw_markov_states,
            (
                str,
                bytes,
            ),
        ):
            for state_index, raw_state in enumerate(
                raw_markov_states
            ):
                if isinstance(
                    raw_state,
                    StackingMarkovState,
                ):
                    markov_states.append(
                        raw_state
                    )

                elif isinstance(
                    raw_state,
                    Mapping,
                ):
                    markov_states.append(
                        StackingMarkovState.from_dict(
                            raw_state,
                            default_label=f"S{state_index}",
                        )
                    )

        return cls(
            enabled=_as_bool(
                values.get(
                    "enabled",
                    False,
                )
            ),
            model=str(
                values.get(
                    "model",
                    "periodic_sequence",
                )
                or "periodic_sequence"
            ),
            p_alternate=float(
                values.get(
                    "p_alternate",
                    values.get(
                        "p_alt",
                        values.get(
                            "p_fault",
                            1.0,
                        ),
                    ),
                )
            ),
            p_grid_enabled=_as_bool(
                values.get(
                    "p_grid_enabled",
                    values.get(
                        "p_search_enabled",
                        False,
                    ),
                )
            ),
            p_grid_minimum=float(
                values.get(
                    "p_grid_minimum",
                    values.get(
                        "p_search_minimum",
                        0.0,
                    ),
                )
            ),
            p_grid_maximum=float(
                values.get(
                    "p_grid_maximum",
                    values.get(
                        "p_search_maximum",
                        1.0,
                    ),
                )
            ),
            p_grid_step=float(
                values.get(
                    "p_grid_step",
                    values.get(
                        "p_search_step",
                        0.05,
                    ),
                )
            ),
            ensemble_samples=max(
                1,
                int(
                    values.get(
                        "ensemble_samples",
                        20,
                    )
                ),
            ),
            ensemble_seed=int(
                values.get(
                    "ensemble_seed",
                    12345,
                )
            ),
            ensemble_start_state=str(
                values.get(
                    "ensemble_start_state",
                    "A",
                )
                or "A"
            ),
            preview_realization_index=max(
                0,
                int(
                    values.get(
                        "preview_realization_index",
                        0,
                    )
                ),
            ),
            state_a_du=float(
                values.get(
                    "state_a_du",
                    0.0,
                )
            ),
            state_a_dv=float(
                values.get(
                    "state_a_dv",
                    0.0,
                )
            ),
            state_a_h=float(
                values.get(
                    "state_a_h",
                    3.8,
                )
            ),
            state_b_du=float(
                values.get(
                    "state_b_du",
                    0.0,
                )
            ),
            state_b_dv=float(
                values.get(
                    "state_b_dv",
                    0.0,
                )
            ),
            state_b_h=float(
                values.get(
                    "state_b_h",
                    3.8,
                )
            ),
            markov_states=markov_states,
            markov_transition_matrix=_decode_structured_ini_value(
                values.get(
                    "markov_transition_matrix",
                    values.get(
                        "markov_transition_matrix_json",
                        [],
                    ),
                ),
                [],
            ),
            markov_probability_controls=(
                _decode_structured_ini_value(
                    values.get(
                        "markov_probability_controls",
                        values.get(
                            "markov_probability_controls_json",
                            [],
                        ),
                    ),
                    [],
                )
            ),
            markov_start_state=int(
                values.get(
                    "markov_start_state",
                    0,
                )
            ),
            definition_mode=str(
                values.get(
                    "definition_mode",
                    "repeating_motif",
                )
                or "repeating_motif"
            ),
            vector_mode=str(
                values.get(
                    "vector_mode",
                    "incremental",
                )
                or "incremental"
            ),
            spacing_mode=str(
                values.get(
                    "spacing_mode",
                    "independent",
                )
                or "independent"
            ),
            nominal_spacing=spacing,
            refine_nominal_spacing=_as_bool(
                values.get(
                    "refine_nominal_spacing",
                    values.get(
                        "refine_stacking_distance",
                        False,
                    ),
                )
            ),
            nominal_spacing_min=float(
                values.get(
                    "nominal_spacing_min",
                    values.get(
                        "stacking_distance_min",
                        0.1,
                    ),
                )
            ),
            nominal_spacing_max=float(
                values.get(
                    "nominal_spacing_max",
                    values.get(
                        "stacking_distance_max",
                        100.0,
                    ),
                )
            ),
            stacking_fraction=float(
                values.get(
                    "stacking_fraction",
                    1.0,
                )
            ),
            refine_stacking_fraction=_as_bool(
                values.get(
                    "refine_stacking_fraction",
                    False,
                )
            ),
            stacking_fraction_min=float(
                values.get(
                    "stacking_fraction_min",
                    0.001,
                )
            ),
            stacking_fraction_max=float(
                values.get(
                    "stacking_fraction_max",
                    100.0,
                )
            ),
            repeat_period=float(
                repeat_period
            ),
            repeat_shift_u=float(
                values.get(
                    "repeat_shift_u",
                    0.0,
                )
            ),
            repeat_shift_v=float(
                values.get(
                    "repeat_shift_v",
                    0.0,
                )
            ),
            sequence=sequence,
            show_transition_vectors=_as_bool(
                values.get(
                    "show_transition_vectors",
                    values.get(
                        "show_transition_arrows",
                        True,
                    ),
                ),
                True,
            ),
            color_layers=_as_bool(
                values.get(
                    "color_layers",
                    True,
                ),
                True,
            ),
        ).normalized()

def absolute_sequence_layers(
    stack_spec: LayerStackSpec,
) -> List[SequenceLayer]:
    """
    Return absolute placements for all rows in one sequence period.

    Incremental mode
    ----------------
    Row zero is the reference layer and is placed at:

        u = 0
        v = 0
        s = 0
        local z = 0

    Every later row is added cumulatively to the preceding row:

        absolute[j] = absolute[j - 1] + incremental[j]

    Absolute mode
    -------------
    Sequence rows are copied without cumulative conversion.
    """
    spec = stack_spec.normalized()

    if spec.vector_mode == "absolute":
        return [
            SequenceLayer.from_dict(
                layer.to_dict(),
                default_coordinate=float(
                    row
                ),
                nominal_spacing=spec.nominal_spacing,
            )
            for row, layer in enumerate(
                spec.sequence
            )
        ]

    absolute_layers: List[SequenceLayer] = []

    cumulative_u = 0.0
    cumulative_v = 0.0
    cumulative_s = 0.0
    cumulative_normal_delta = 0.0

    for row, layer in enumerate(
        spec.sequence
    ):
        if row > 0:
            cumulative_u += float(
                layer.shift_u
            )

            cumulative_v += float(
                layer.shift_v
            )

            cumulative_s += float(
                layer.stack_coordinate
            )

            cumulative_normal_delta += float(
                layer.normal_delta
            )

        absolute_layers.append(
            SequenceLayer(
                uid=str(
                    layer.uid
                ),
                enabled=bool(
                    layer.enabled
                ),
                label=str(
                    layer.label
                ),
                shift_u=float(
                    cumulative_u
                ),
                shift_v=float(
                    cumulative_v
                ),
                stack_coordinate=float(
                    cumulative_s
                ),
                normal_delta=float(
                    cumulative_normal_delta
                ),
            )
        )

    return absolute_layers


def effective_repeat_translation(
    stack_spec: LayerStackSpec,
) -> tuple[float, float, float]:
    """
    Return the translation from row zero in one period to row zero in the
    next period.

    Returns
    -------
    repeat_u
        Total lateral a-shift applied after one complete sequence period.

    repeat_v
        Total lateral b-shift applied after one complete sequence period.

    repeat_s
        Total stacking-coordinate period.

    In incremental mode, ``repeat_period`` is interpreted as the final closing
    Δs from the last sequence row to row zero of the next period.

    Example
    -------
    For six rows with cumulative coordinates 0, 1, 2, 3, 4, 5 and closing
    Δs = 1:

        effective repeat s = 5 + 1 = 6
    """
    spec = stack_spec.normalized()

    if spec.vector_mode == "absolute":
        return (
            float(
                spec.repeat_shift_u
            ),
            float(
                spec.repeat_shift_v
            ),
            float(
                spec.repeat_period
            ),
        )

    absolute_layers = absolute_sequence_layers(
        spec
    )

    if not absolute_layers:
        return (
            float(
                spec.repeat_shift_u
            ),
            float(
                spec.repeat_shift_v
            ),
            max(
                float(
                    spec.repeat_period
                ),
                _EPSILON,
            ),
        )

    last_layer = absolute_layers[
        -1
    ]

    closing_delta_s = max(
        float(
            spec.repeat_period
        ),
        _EPSILON,
    )

    return (
        # These are total lateral shifts between complete periods.
        # A value of zero makes every period restart at the same lateral
        # registry.
        float(
            spec.repeat_shift_u
        ),
        float(
            spec.repeat_shift_v
        ),
        float(
            last_layer.stack_coordinate
        )
        + closing_delta_s,
    )




def d02_reference_parameters(
    stack_spec: LayerStackSpec,
) -> Dict[str, float]:
    """
    Return the reference parameters of a two-row incremental D02 motif.

    The generated transitions are:

        A -> B:
            u_ab
            v_ab
            spacing + delta_h

        B -> A:
            u_repeat - u_ab
            v_repeat - v_ab
            spacing - delta_h
    """
    spec = stack_spec.normalized()

    if spec.vector_mode != "incremental":
        raise ValueError(
            "Dynamic D02 refinement currently requires "
            "vector_mode='incremental'."
        )

    if len(spec.sequence) != 2:
        raise ValueError(
            "Dynamic D02 refinement currently requires exactly "
            "two sequence rows."
        )

    absolute_layers = absolute_sequence_layers(
        spec
    )

    repeat_u, repeat_v, repeat_s = (
        effective_repeat_translation(
            spec
        )
    )

    first_layer = absolute_layers[0]
    second_layer = absolute_layers[1]

    return {
        "u_ab": float(
            second_layer.shift_u
            - first_layer.shift_u
        ),
        "v_ab": float(
            second_layer.shift_v
            - first_layer.shift_v
        ),
        "u_repeat": float(
            repeat_u
        ),
        "v_repeat": float(
            repeat_v
        ),
        "delta_h": float(
            second_layer.normal_delta
            - first_layer.normal_delta
        ),
        "repeat_s": float(
            repeat_s
        ),
    }


def evaluate_stacking_spacing(
    stack_spec: LayerStackSpec,
    lattice_matrix: np.ndarray,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> float:
    """
    Evaluate the physical interlayer distance for the current lattice.

    Parameters
    ----------
    stack_spec
        Periodic stacking definition.

    lattice_matrix
        Current parent lattice matrix using row vectors:

            Cartesian = fractional @ lattice_matrix

    parameters
        Current refinement parameters.

        Independent mode reads:

            stacking_distance

        Fraction mode reads:

            stacking_fraction

        Missing values fall back to the values stored in stack_spec.
    """
    spec = stack_spec.normalized()

    parameters = dict(
        parameters
        or {}
    )

    lattice_matrix = np.asarray(
        lattice_matrix,
        dtype=float,
    )

    if lattice_matrix.shape != (
        3,
        3,
    ):
        raise ValueError(
            "lattice_matrix must have shape (3, 3)."
        )

    if spec.spacing_mode == "independent":
        spacing = float(
            parameters.get(
                "stacking_distance",
                spec.nominal_spacing,
            )
        )

        if (
            not np.isfinite(
                spacing
            )
            or spacing <= _EPSILON
        ):
            raise ValueError(
                "stacking_distance must be positive and finite."
            )

        return float(
            spacing
        )

    vector_a = lattice_matrix[0]
    vector_b = lattice_matrix[1]
    vector_c = lattice_matrix[2]

    normal = np.cross(
        vector_a,
        vector_b,
    )

    normal_length = float(
        np.linalg.norm(
            normal
        )
    )

    if normal_length <= _EPSILON:
        raise ValueError(
            "The parent a and b vectors are collinear."
        )

    normal /= normal_length

    c_normal = abs(
        float(
            np.dot(
                vector_c,
                normal,
            )
        )
    )

    if c_normal <= _EPSILON:
        raise ValueError(
            "The parent c vector has no component normal to the "
            "a-b layer plane."
        )

    fraction = float(
        parameters.get(
            "stacking_fraction",
            spec.stacking_fraction,
        )
    )

    if (
        not np.isfinite(
            fraction
        )
        or fraction <= _EPSILON
    ):
        raise ValueError(
            "stacking_fraction must be positive and finite."
        )

    return float(
        fraction
        * c_normal
    )


def layer_basis(
    lattice_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the a vector, b vector and unit normal to the a-b plane.
    """
    matrix = np.asarray(
        lattice_matrix,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "The lattice matrix must have shape (3, 3)."
        )

    vector_u = matrix[0]
    vector_v = matrix[1]

    normal = np.cross(
        vector_u,
        vector_v,
    )

    length = float(
        np.linalg.norm(normal)
    )

    if length <= _EPSILON:
        raise ValueError(
            "The lattice a and b vectors are collinear."
        )

    normal /= length

    return (
        vector_u,
        vector_v,
        normal,
    )


def realized_layer_translation(
    layer: RealizedLayer,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    vector_u, vector_v, normal = layer_basis(
        lattice_matrix
    )

    return (
        float(layer.shift_u) * vector_u
        + float(layer.shift_v) * vector_v
        + float(layer.normal_offset) * normal
    )


def realize_periodic_sequence(
    stack_spec: LayerStackSpec,
    *,
    minimum_normal_offset: float,
    maximum_normal_offset: float,
    padding_repeats: int = 1,
) -> List[RealizedLayer]:
    """
    Realize enough copies of an infinite periodic sequence to cover a
    requested physical normal-offset range.
    """
    spec = stack_spec.normalized()

    if minimum_normal_offset > maximum_normal_offset:
        minimum_normal_offset, maximum_normal_offset = (
            maximum_normal_offset,
            minimum_normal_offset,
        )

    spacing = max(
        abs(float(spec.nominal_spacing)),
        _EPSILON,
    )

    period_coordinate = max(
        abs(float(spec.repeat_period)),
        _EPSILON,
    )

    period_angstrom = (
        period_coordinate
        * spacing
    )

    maximum_delta = max(
        [
            abs(float(layer.normal_delta))
            for layer in spec.sequence
        ]
        or [0.0]
    )

    expanded_minimum = (
        float(minimum_normal_offset)
        - maximum_delta
        - period_angstrom
    )

    expanded_maximum = (
        float(maximum_normal_offset)
        + maximum_delta
        + period_angstrom
    )

    minimum_repeat = int(
        np.floor(
            expanded_minimum
            / period_angstrom
        )
    ) - max(
        0,
        int(padding_repeats),
    )

    maximum_repeat = int(
        np.ceil(
            expanded_maximum
            / period_angstrom
        )
    ) + max(
        0,
        int(padding_repeats),
    )

    realized: List[RealizedLayer] = []
    sequence_length = len(
        spec.sequence
    )

    for repeat_index in range(
        minimum_repeat,
        maximum_repeat + 1,
    ):
        for sequence_index, layer in enumerate(
            spec.sequence
        ):
            if not layer.enabled:
                continue

            stack_coordinate = (
                float(layer.stack_coordinate)
                + repeat_index
                * float(spec.repeat_period)
            )

            normal_offset = (
                stack_coordinate
                * spacing
                + float(layer.normal_delta)
            )

            if (
                normal_offset
                < expanded_minimum
                - period_angstrom
                or normal_offset
                > expanded_maximum
                + period_angstrom
            ):
                continue

            realized.append(
                RealizedLayer(
                    realized_index=(
                        repeat_index
                        * sequence_length
                        + sequence_index
                    ),
                    sequence_index=sequence_index,
                    repeat_index=repeat_index,
                    label=str(
                        layer.label
                    ),
                    shift_u=(
                        float(layer.shift_u)
                        + repeat_index
                        * float(spec.repeat_shift_u)
                    ),
                    shift_v=(
                        float(layer.shift_v)
                        + repeat_index
                        * float(spec.repeat_shift_v)
                    ),
                    stack_coordinate=stack_coordinate,
                    normal_offset=normal_offset,
                    enabled=True,
                    atom_count=0,
                )
            )

    realized.sort(
        key=lambda layer: (
            layer.normal_offset,
            layer.realized_index,
        )
    )

    return realized

def selected_layer_indices(
    number_of_layers: int,
) -> List[int]:
    """
    Return exactly N consecutive realized-layer indices.

    Layer zero is always included and remains the sequence reference.

    Examples
    --------
    N = 1:
        0

    N = 2:
        -1, 0

    N = 3:
        -1, 0, 1

    N = 4:
        -2, -1, 0, 1
    """
    number_of_layers = max(
        1,
        int(
            round(
                float(
                    number_of_layers
                )
            )
        ),
    )

    first_index = -(
        number_of_layers
        // 2
    )

    return list(
        range(
            first_index,
            first_index
            + number_of_layers,
        )
    )


def realize_n_layers(
    stack_spec: LayerStackSpec,
    *,
    number_of_layers: int,
    center_stack: bool = True,
) -> List[RealizedLayer]:
    """
    Realize exactly N consecutive positions from an infinite periodic stack.

    Editable incremental vectors are converted to absolute placements before
    periodic copies are generated.
    """
    spec = stack_spec.normalized()

    indices = selected_layer_indices(
        number_of_layers
    )

    absolute_layers = absolute_sequence_layers(
        spec
    )

    sequence_length = len(
        absolute_layers
    )

    if sequence_length <= 0:
        return []

    (
        repeat_shift_u,
        repeat_shift_v,
        repeat_period_s,
    ) = effective_repeat_translation(
        spec
    )

    realized: List[RealizedLayer] = []

    for realized_index in indices:
        repeat_index, sequence_index = divmod(
            int(
                realized_index
            ),
            int(
                sequence_length
            ),
        )

        editable_layer = spec.sequence[
            sequence_index
        ]

        absolute_layer = absolute_layers[
            sequence_index
        ]

        shift_u = (
            float(
                absolute_layer.shift_u
            )
            + float(
                repeat_index
            )
            * float(
                repeat_shift_u
            )
        )

        shift_v = (
            float(
                absolute_layer.shift_v
            )
            + float(
                repeat_index
            )
            * float(
                repeat_shift_v
            )
        )

        stack_coordinate = (
            float(
                absolute_layer.stack_coordinate
            )
            + float(
                repeat_index
            )
            * float(
                repeat_period_s
            )
        )

        normal_delta = float(
            absolute_layer.normal_delta
        )

        normal_offset = (
            stack_coordinate
            * float(
                spec.nominal_spacing
            )
            + normal_delta
        )

        realized.append(
            RealizedLayer(
                realized_index=int(
                    realized_index
                ),
                sequence_index=int(
                    sequence_index
                ),
                repeat_index=int(
                    repeat_index
                ),
                label=str(
                    editable_layer.label
                    or f"L{sequence_index}"
                ),
                shift_u=float(
                    shift_u
                ),
                shift_v=float(
                    shift_v
                ),
                stack_coordinate=float(
                    stack_coordinate
                ),
                normal_offset=float(
                    normal_offset
                ),
                enabled=bool(
                    editable_layer.enabled
                ),
                atom_count=0,
            )
        )

    if (
        center_stack
        and realized
    ):
        offsets = np.asarray(
            [
                layer.normal_offset
                for layer in realized
            ],
            dtype=float,
        )

        normal_center = 0.5 * (
            float(
                np.min(
                    offsets
                )
            )
            + float(
                np.max(
                    offsets
                )
            )
        )

        for layer in realized:
            layer.normal_offset -= normal_center

    return realized



def stack_thickness_metrics(
    realized_layers: Sequence[RealizedLayer],
    *,
    fallback_spacing: float,
) -> Dict[str, float]:
    """
    Calculate physical thickness information for a realized finite stack.

    center_span
        Distance between the first and last layer centers.

    boundary_thickness
        Nominal thickness obtained by placing a boundary half one local
        interlayer spacing outside each outer layer center.

    The boundary thickness is for reporting and discrete sphere/ellipsoid
    cross-sections. It is not used to clip atoms along the stacking direction.
    """
    layers = list(
        realized_layers
        or []
    )

    fallback_spacing = max(
        abs(
            float(
                fallback_spacing
            )
        ),
        _EPSILON,
    )

    if not layers:
        return {
            "center_span": 0.0,
            "boundary_thickness": 0.0,
            "lower_boundary": 0.0,
            "upper_boundary": 0.0,
            "minimum_center": 0.0,
            "maximum_center": 0.0,
        }

    offsets = np.asarray(
        sorted(
            float(
                layer.normal_offset
            )
            for layer in layers
        ),
        dtype=float,
    )

    minimum_center = float(
        offsets[0]
    )

    maximum_center = float(
        offsets[-1]
    )

    center_span = max(
        0.0,
        maximum_center
        - minimum_center,
    )

    if offsets.size == 1:
        lower_gap = fallback_spacing
        upper_gap = fallback_spacing

    else:
        differences = np.diff(
            offsets
        )

        positive_differences = differences[
            differences > _EPSILON
        ]

        typical_gap = (
            float(
                np.median(
                    positive_differences
                )
            )
            if positive_differences.size
            else fallback_spacing
        )

        lower_gap = float(
            differences[0]
        )

        upper_gap = float(
            differences[-1]
        )

        if lower_gap <= _EPSILON:
            lower_gap = typical_gap

        if upper_gap <= _EPSILON:
            upper_gap = typical_gap

    lower_boundary = (
        minimum_center
        - 0.5
        * lower_gap
    )

    upper_boundary = (
        maximum_center
        + 0.5
        * upper_gap
    )

    boundary_thickness = max(
        0.0,
        upper_boundary
        - lower_boundary,
    )

    return {
        "center_span": float(
            center_span
        ),
        "boundary_thickness": float(
            boundary_thickness
        ),
        "lower_boundary": float(
            lower_boundary
        ),
        "upper_boundary": float(
            upper_boundary
        ),
        "minimum_center": float(
            minimum_center
        ),
        "maximum_center": float(
            maximum_center
        ),
    }

def default_generic_markov_states() -> List[
    StackingMarkovState
]:
    """
    Default three-state model:

        D+ = positive disorder
        Z  = near-zero / normal stacking
        D- = negative disorder
    """
    return [
        StackingMarkovState(
            label="D+",
            du=0.04255,
            dv=0.0,
            h=4.04294,
        ),
        StackingMarkovState(
            label="Z",
            du=0.00403,
            dv=0.01269,
            h=3.72673,
        ),
        StackingMarkovState(
            label="D-",
            du=-0.04255,
            dv=0.0,
            h=4.04294,
        ),
    ]

def default_generic_markov_matrix() -> List[
    List[float]
]:
    """
    Default transition matrix for:

        D+ -> Z
        Z  -> D+ or D-
        D- -> Z
    """
    return [
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
    ]


def normalize_markov_probability_controls(
    controls,
    *,
    number_of_states: int,
    transition_matrix,
) -> List[List[Dict[str, Any]]]:
    """
    Return one probability-grid control for every transition-matrix cell.

    Output shape:

        number_of_states x number_of_states

    Every cell contains:

        {
            "enabled": bool,
            "minimum": float,
            "maximum": float,
            "step": float,
        }
    """
    number_of_states = max(
        1,
        int(
            number_of_states
        ),
    )

    transition_matrix = normalize_markov_matrix(
        transition_matrix,
        number_of_states=number_of_states,
    )

    if isinstance(
        controls,
        str,
    ):
        controls = _decode_structured_ini_value(
            controls,
            [],
        )

    if not isinstance(
        controls,
        Sequence,
    ) or isinstance(
        controls,
        (
            str,
            bytes,
        ),
    ):
        controls = []

    output = []

    for source_index in range(
        number_of_states
    ):
        output_row = []

        if (
            source_index < len(
                controls
            )
            and isinstance(
                controls[
                    source_index
                ],
                Sequence,
            )
            and not isinstance(
                controls[
                    source_index
                ],
                (
                    str,
                    bytes,
                ),
            )
        ):
            source_controls = controls[
                source_index
            ]
        else:
            source_controls = []

        for target_index in range(
            number_of_states
        ):
            current_probability = float(
                transition_matrix[
                    source_index
                ][
                    target_index
                ]
            )

            if (
                target_index
                < len(
                    source_controls
                )
                and isinstance(
                    source_controls[
                        target_index
                    ],
                    Mapping,
                )
            ):
                raw_control = dict(
                    source_controls[
                        target_index
                    ]
                )
            else:
                raw_control = {}

            minimum = float(
                raw_control.get(
                    "minimum",
                    current_probability,
                )
            )

            maximum = float(
                raw_control.get(
                    "maximum",
                    current_probability,
                )
            )

            if maximum < minimum:
                minimum, maximum = (
                    maximum,
                    minimum,
                )

            minimum = min(
                1.0,
                max(
                    0.0,
                    minimum,
                ),
            )

            maximum = min(
                1.0,
                max(
                    0.0,
                    maximum,
                ),
            )

            step = max(
                float(
                    raw_control.get(
                        "step",
                        0.05,
                    )
                ),
                1.0e-8,
            )

            output_row.append(
                {
                    "enabled": _as_bool(
                        raw_control.get(
                            "enabled",
                            False,
                        ),
                        False,
                    ),
                    "minimum": float(
                        minimum
                    ),
                    "maximum": float(
                        maximum
                    ),
                    "step": float(
                        step
                    ),
                }
            )

        output.append(
            output_row
        )

    return output

def normalize_markov_matrix(
    matrix,
    *,
    number_of_states: int,
) -> List[List[float]]:
    """
    Return a valid square row-stochastic Markov matrix.

    Each row is normalized so that:

        sum(row) == 1.0

    Empty/invalid rows become identity rows.
    """
    number_of_states = max(
        1,
        int(
            number_of_states
        ),
    )

    array = np.asarray(
        matrix,
        dtype=float,
    )

    if array.shape != (
        number_of_states,
        number_of_states,
    ):
        array = np.eye(
            number_of_states,
            dtype=float,
        )

    array = np.maximum(
        array,
        0.0,
    )

    row_sums = np.sum(
        array,
        axis=1,
    )

    for row in range(
        number_of_states
    ):
        if row_sums[row] <= _EPSILON:
            array[
                row,
                :
            ] = 0.0

            array[
                row,
                row,
            ] = 1.0

    row_sums = np.sum(
        array,
        axis=1,
        keepdims=True,
    )

    array = array / np.maximum(
        row_sums,
        _EPSILON,
    )

    return array.tolist()


def stacking_spec_from_config(
    config: Mapping[str, Any] | None,
) -> LayerStackSpec:
    """
    Read the periodic stacking specification from an application config.

    Preferred location
    ------------------
    config["layer_stacking"]

    Compatibility locations
    -----------------------
    config["crystallite_shape"]["stacking"]
    config["stacking"]

    The old transition-based ``stacking_faults`` representation is not
    converted automatically because its transitions do not uniquely define
    the new absolute periodic sequence.
    """
    config = dict(
        config
        or {}
    )

    raw = config.get(
        "layer_stacking",
        None,
    )

    if not isinstance(
        raw,
        Mapping,
    ):
        shape_values = config.get(
            "crystallite_shape",
            {},
        )

        if isinstance(
            shape_values,
            Mapping,
        ):
            raw = shape_values.get(
                "stacking",
                None,
            )

    if not isinstance(
        raw,
        Mapping,
    ):
        raw = config.get(
            "stacking",
            None,
        )

    if not isinstance(
        raw,
        Mapping,
    ):
        return LayerStackSpec(
            enabled=False
        ).normalized()

    return LayerStackSpec.from_dict(
        raw
    )


def validate_fixed_stacking_refinement_config(
    config: Mapping[str, Any] | None,
) -> List[str]:
    """
    Return geometry-changing parameters that are incorrectly marked refinable
    while the fixed stacked-shell backend is active.

    This check is temporary. These parameters will become valid after dynamic
    stacked-shell rebuilding is implemented.
    """
    config = dict(
        config
        or {}
    )

    stack_spec = stacking_spec_from_config(
        config
    )

    if not stack_spec.enabled:
        return []

    refinable = config.get(
        "refinable",
        {},
    )

    if not isinstance(
        refinable,
        Mapping,
    ):
        return []

    forbidden_names = {
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "stacking_distance",
    }

    invalid = []

    for name, enabled in refinable.items():
        name_text = str(
            name
        ).strip()

        name_lower = name_text.lower()

        is_geometry_parameter = (
            name_lower in forbidden_names
            or name_lower.startswith("x_")
            or name_lower.startswith("y_")
            or name_lower.startswith("z_")
            or name_lower.startswith("occ_")
        )

        if (
            is_geometry_parameter
            and _as_bool(
                enabled,
                False,
            )
        ):
            invalid.append(
                name_text
            )

    return sorted(
        invalid,
        key=str.lower,
    )

