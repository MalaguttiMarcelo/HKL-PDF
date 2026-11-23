from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from pymatgen.core import Lattice
from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    RealizedLayer,
    layer_basis,
    stack_thickness_metrics,
)

from pdf_fitting.models.stacking_disorder import (
    STATE_A,
    STATE_B,
    state_name,
)

from pdf_fitting.models.compiled_stacked_geometry import (
    _centered_source_layer_cartesian,
    _lattice_matrix_from_parameters,
    _reference_lateral_mask,
)


_EPSILON = 1.0e-12


@dataclass
class CompiledStochasticLayerDescriptor:
    """
    One fixed layer position inside one frozen Markov realization.

    coefficient_a / coefficient_b count how many A-state and B-state
    transition vectors have been emitted before this layer.

    Example
    -------
    States:
        A B B A

    Layer coefficients:
        layer 0: A_count=0, B_count=0
        layer 1: A_count=1, B_count=0
        layer 2: A_count=1, B_count=1
        layer 3: A_count=1, B_count=2

    because current layer state emits the transition vector.
    """

    realization_index: int
    layer_index: int
    state: int

    transition_coefficients: np.ndarray

    label: str


@dataclass
class CompiledStochasticRealizationTopology:
    """
    Topology for one finite frozen stochastic stack realization.

    Atom topology is fixed:

        source site identity;
        in-plane parent-cell translations;
        layer membership.

    Cartesian coordinates are evaluated later using refined lattice and
    refined state-A/state-B vectors.
    """

    realization_index: int

    layers: List[
        CompiledStochasticLayerDescriptor
    ]

    atom_source_site_indices: np.ndarray
    atom_layer_descriptor_indices: np.ndarray
    atom_cell_a: np.ndarray
    atom_cell_b: np.ndarray


@dataclass
class EvaluatedStochasticRealizationGeometry:
    """
    Current Cartesian geometry for one frozen stochastic realization.
    """

    realization_index: int

    parent_lattice_matrix: np.ndarray
    stack_normal: np.ndarray

    state_a_du: float
    state_a_dv: float
    state_a_h: float

    state_b_du: float
    state_b_dv: float
    state_b_h: float

    layer_origins: np.ndarray
    layer_normal_offsets: np.ndarray

    atom_cart: np.ndarray
    atom_source_site_indices: np.ndarray
    atom_layer_indices: np.ndarray
    atom_sequence_indices: np.ndarray

    atom_cell_a: np.ndarray
    atom_cell_b: np.ndarray

    atom_elements: List[str]

    realized_layers: List[
        RealizedLayer
    ]

    center_span: float
    boundary_thickness: float

    @property
    def number_of_atoms(
        self,
    ) -> int:
        return int(
            self.atom_cart.shape[0]
        )

    @property
    def number_of_layers(
        self,
    ) -> int:
        return int(
            self.layer_origins.shape[0]
        )


@dataclass
class CompiledStochasticEnsembleGeometry:
    """
    Compiled frozen Markov ensemble.

    The state paths are fixed during one P(alternate) candidate.

    Continuous refinement changes only:

        lattice parameters;
        state-A vector;
        state-B vector.

    The path topology, state identities, source atoms, and lateral envelope
    membership stay fixed.
    """

    source_structure: Structure
    shape_spec: CrystalliteShapeSpec
    stack_spec: LayerStackSpec

    state_paths: np.ndarray

    source_fractional: np.ndarray
    source_elements: List[str]

    realizations: List[
        CompiledStochasticRealizationTopology
    ]

    reference_lattice_matrix: np.ndarray

    @property
    def ensemble_samples(
        self,
    ) -> int:
        return int(
            len(
                self.realizations
            )
        )

    @property
    def number_of_layers(
        self,
    ) -> int:
        if self.state_paths.ndim != 2:
            return 0

        return int(
            self.state_paths.shape[1]
        )

    def evaluate_realization(
        self,
        realization_index: int,
        parameters: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> EvaluatedStochasticRealizationGeometry:
        """
        Evaluate one frozen realization using current continuous parameters.
        """
        parameters = dict(
            parameters
            or {}
        )

        realization_index = int(
            realization_index
        )

        if (
            realization_index < 0
            or realization_index
            >= len(
                self.realizations
            )
        ):
            raise IndexError(
                "Stochastic realization index is outside the compiled ensemble."
            )

        topology = self.realizations[
            realization_index
        ]

        lattice_matrix = _lattice_matrix_from_parameters(
            self.source_structure,
            parameters,
        )

        vector_a, vector_b, normal = layer_basis(
            lattice_matrix
        )

        state_a_du = float(
            parameters.get(
                "stack_state_a_du",
                self.stack_spec.state_a_du,
            )
        )

        state_a_dv = float(
            parameters.get(
                "stack_state_a_dv",
                self.stack_spec.state_a_dv,
            )
        )

        state_a_h = max(
            float(
                parameters.get(
                    "stack_state_a_h",
                    self.stack_spec.state_a_h,
                )
            ),
            _EPSILON,
        )

        state_b_du = float(
            parameters.get(
                "stack_state_b_du",
                self.stack_spec.state_b_du,
            )
        )

        state_b_dv = float(
            parameters.get(
                "stack_state_b_dv",
                self.stack_spec.state_b_dv,
            )
        )

        state_b_h = max(
            float(
                parameters.get(
                    "stack_state_b_h",
                    self.stack_spec.state_b_h,
                )
            ),
            _EPSILON,
        )

        layers = topology.layers

        state_du, state_dv, state_h, _labels = (
            _generic_state_vectors(
                self.stack_spec,
                parameters,
            )
        )

        transition_coefficients = np.asarray(
            [
                layer.transition_coefficients
                for layer in layers
            ],
            dtype=float,
        )

        layer_u = (
            transition_coefficients
            @ state_du
        )

        layer_v = (
            transition_coefficients
            @ state_dv
        )

        layer_h_uncentered = (
            transition_coefficients
            @ state_h
        )

        if layer_h_uncentered.size:
            center_h = 0.5 * (
                float(
                    np.min(
                        layer_h_uncentered
                    )
                )
                + float(
                    np.max(
                        layer_h_uncentered
                    )
                )
            )
        else:
            center_h = 0.0

        layer_h = (
            layer_h_uncentered
            - center_h
        )

        layer_origins = (
            layer_u[
                :,
                None
            ]
            * vector_a[
                None,
                :
            ]
            + layer_v[
                :,
                None
            ]
            * vector_b[
                None,
                :
            ]
            + layer_h[
                :,
                None
            ]
            * normal[
                None,
                :
            ]
        )

        source_cart = _centered_source_layer_cartesian(
            self.source_fractional,
            lattice_matrix,
            normal,
        )

        source_indices = np.asarray(
            topology.atom_source_site_indices,
            dtype=np.int64,
        )

        layer_indices = np.asarray(
            topology.atom_layer_descriptor_indices,
            dtype=np.int64,
        )

        atom_cell_a = np.asarray(
            topology.atom_cell_a,
            dtype=float,
        )

        atom_cell_b = np.asarray(
            topology.atom_cell_b,
            dtype=float,
        )

        atom_cart = (
            source_cart[
                source_indices
            ]
            + atom_cell_a[
                :,
                None
            ]
            * vector_a[
                None,
                :
            ]
            + atom_cell_b[
                :,
                None
            ]
            * vector_b[
                None,
                :
            ]
            + layer_origins[
                layer_indices
            ]
        )

        realized_layers = []

        for local_index, layer in enumerate(
            layers
        ):
            state = int(
                layer.state
            )

            realized_layers.append(
                RealizedLayer(
                    realized_index=int(
                        layer.layer_index
                    ),
                    sequence_index=state,
                    repeat_index=0,
                    label=str(
                        layer.label
                    ),
                    shift_u=float(
                        layer_u[
                            local_index
                        ]
                    ),
                    shift_v=float(
                        layer_v[
                            local_index
                        ]
                    ),
                    stack_coordinate=float(
                        local_index
                    ),
                    normal_offset=float(
                        layer_h[
                            local_index
                        ]
                    ),
                    enabled=True,
                    atom_count=0,
                )
            )

        for layer_descriptor_index in layer_indices:
            realized_layers[
                int(
                    layer_descriptor_index
                )
            ].atom_count += 1

        thickness = stack_thickness_metrics(
            realized_layers,
            fallback_spacing=0.5
            * (
                state_a_h
                + state_b_h
            ),
        )

        atom_layer_index = np.asarray(
            [
                layers[
                    int(
                        local_index
                    )
                ].layer_index
                for local_index in layer_indices
            ],
            dtype=np.int32,
        )

        atom_sequence_index = np.asarray(
            [
                layers[
                    int(
                        local_index
                    )
                ].state
                for local_index in layer_indices
            ],
            dtype=np.int32,
        )

        atom_elements = [
            self.source_elements[
                int(
                    source_index
                )
            ]
            for source_index in source_indices
        ]

        return EvaluatedStochasticRealizationGeometry(
            realization_index=realization_index,
            parent_lattice_matrix=np.asarray(
                lattice_matrix,
                dtype=float,
            ),
            stack_normal=np.asarray(
                normal,
                dtype=float,
            ),
            state_a_du=float(
                state_a_du
            ),
            state_a_dv=float(
                state_a_dv
            ),
            state_a_h=float(
                state_a_h
            ),
            state_b_du=float(
                state_b_du
            ),
            state_b_dv=float(
                state_b_dv
            ),
            state_b_h=float(
                state_b_h
            ),
            layer_origins=np.asarray(
                layer_origins,
                dtype=float,
            ),
            layer_normal_offsets=np.asarray(
                layer_h,
                dtype=float,
            ),
            atom_cart=np.asarray(
                atom_cart,
                dtype=float,
            ),
            atom_source_site_indices=np.asarray(
                source_indices,
                dtype=np.int32,
            ),
            atom_layer_indices=atom_layer_index,
            atom_sequence_indices=atom_sequence_index,
            atom_cell_a=np.asarray(
                topology.atom_cell_a,
                dtype=np.int32,
            ),
            atom_cell_b=np.asarray(
                topology.atom_cell_b,
                dtype=np.int32,
            ),
            atom_elements=atom_elements,
            realized_layers=realized_layers,
            center_span=float(
                thickness[
                    "center_span"
                ]
            ),
            boundary_thickness=float(
                thickness[
                    "boundary_thickness"
                ]
            ),
        )

    def evaluate_all(
        self,
        parameters: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> List[
        EvaluatedStochasticRealizationGeometry
    ]:
        """
        Evaluate every frozen stochastic realization.
        """
        return [
            self.evaluate_realization(
                realization_index,
                parameters,
            )
            for realization_index in range(
                self.ensemble_samples
            )
        ]


def _layer_descriptors_from_states(
    *,
    realization_index: int,
    states: np.ndarray,
    number_of_states: int,
    labels: List[str],
) -> List[
    CompiledStochasticLayerDescriptor
]:
    """
    Build cumulative per-state transition coefficients.

    For each layer, transition_coefficients[i] contains how many transitions
    emitted by state i occurred before that layer.
    """
    states = np.asarray(
        states,
        dtype=np.int16,
    ).reshape(
        -1
    )

    number_of_states = max(
        1,
        int(
            number_of_states
        ),
    )

    if states.size <= 0:
        raise ValueError(
            "A stochastic realization must contain at least one layer."
        )

    if not np.all(
        (states >= 0)
        & (states < number_of_states)
    ):
        raise ValueError(
            "State path contains an invalid state index."
        )

    labels = list(
        labels
        or []
    )

    while len(labels) < number_of_states:
        labels.append(
            f"S{len(labels)}"
        )

    output = []

    counts = np.zeros(
        number_of_states,
        dtype=np.int32,
    )

    for layer_index, state in enumerate(
        states
    ):
        state = int(
            state
        )

        output.append(
            CompiledStochasticLayerDescriptor(
                realization_index=int(
                    realization_index
                ),
                layer_index=int(
                    layer_index
                ),
                state=state,
                transition_coefficients=np.asarray(
                    counts,
                    dtype=np.int32,
                ).copy(),
                label=str(
                    labels[
                        state
                    ]
                ),
            )
        )

        if layer_index < states.size - 1:
            counts[
                state
            ] += 1

    return output

def _generic_state_vectors(
    stack_spec: LayerStackSpec,
    parameters: Mapping[
        str,
        Any,
    ],
):
    """
    Return generic state-vector arrays.

    Returned arrays have shape:

        (number_of_states,)
    """
    parameters = dict(
        parameters
        or {}
    )

    states = list(
        stack_spec.markov_states
        or []
    )

    if not states:
        # Two-state A/B model: refined as stack_state_a_* / stack_state_b_*.
        states = [
            {
                "uid": "a",
                "label": "A",
                "du": stack_spec.state_a_du,
                "dv": stack_spec.state_a_dv,
                "h": stack_spec.state_a_h,
            },
            {
                "uid": "b",
                "label": "B",
                "du": stack_spec.state_b_du,
                "dv": stack_spec.state_b_dv,
                "h": stack_spec.state_b_h,
            },
        ]

    labels = []
    du_values = []
    dv_values = []
    h_values = []

    for state_index, state in enumerate(
        states
    ):
        if isinstance(
            state,
            dict,
        ):
            uid = str(
                state.get(
                    "uid",
                    state_index,
                )
            )

            label = str(
                state.get(
                    "label",
                    f"S{state_index}",
                )
            )

            default_du = float(
                state.get(
                    "du",
                    0.0,
                )
            )

            default_dv = float(
                state.get(
                    "dv",
                    0.0,
                )
            )

            default_h = float(
                state.get(
                    "h",
                    3.8,
                )
            )

        else:
            uid = str(
                state.uid
            )

            label = str(
                state.label
            )

            default_du = float(
                state.du
            )

            default_dv = float(
                state.dv
            )

            default_h = float(
                state.h
            )

        du_values.append(
            float(
                parameters.get(
                    f"stack_state_{uid}_du",
                    default_du,
                )
            )
        )

        dv_values.append(
            float(
                parameters.get(
                    f"stack_state_{uid}_dv",
                    default_dv,
                )
            )
        )

        h_values.append(
            max(
                float(
                    parameters.get(
                        f"stack_state_{uid}_h",
                        default_h,
                    )
                ),
                _EPSILON,
            )
        )

        labels.append(
            label
        )

    return (
        np.asarray(
            du_values,
            dtype=float,
        ),
        np.asarray(
            dv_values,
            dtype=float,
        ),
        np.asarray(
            h_values,
            dtype=float,
        ),
        labels,
    )


def _reference_state_vectors(
    stack_spec: LayerStackSpec,
    parameters: Mapping[
        str,
        Any,
    ],
):
    """
    Return reference vectors used only to compile lateral envelope membership.
    """
    return (
        float(
            parameters.get(
                "stack_state_a_du",
                stack_spec.state_a_du,
            )
        ),
        float(
            parameters.get(
                "stack_state_a_dv",
                stack_spec.state_a_dv,
            )
        ),
        max(
            float(
                parameters.get(
                    "stack_state_a_h",
                    stack_spec.state_a_h,
                )
            ),
            _EPSILON,
        ),
        float(
            parameters.get(
                "stack_state_b_du",
                stack_spec.state_b_du,
            )
        ),
        float(
            parameters.get(
                "stack_state_b_dv",
                stack_spec.state_b_dv,
            )
        ),
        max(
            float(
                parameters.get(
                    "stack_state_b_h",
                    stack_spec.state_b_h,
                )
            ),
            _EPSILON,
        ),
    )


def _reference_layer_origins(
    *,
    layers: List[
        CompiledStochasticLayerDescriptor
    ],
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    normal: np.ndarray,
    state_du: np.ndarray,
    state_dv: np.ndarray,
    state_h: np.ndarray,
) -> np.ndarray:
    """
    Reference Cartesian layer origins used to compile fixed lateral membership
    for an arbitrary N-state Markov stacking model.
    """
    transition_coefficients = np.asarray(
        [
            layer.transition_coefficients
            for layer in layers
        ],
        dtype=float,
    )

    state_du = np.asarray(
        state_du,
        dtype=float,
    ).reshape(
        -1
    )

    state_dv = np.asarray(
        state_dv,
        dtype=float,
    ).reshape(
        -1
    )

    state_h = np.asarray(
        state_h,
        dtype=float,
    ).reshape(
        -1
    )

    if (
        transition_coefficients.ndim != 2
        or transition_coefficients.shape[1]
        != state_du.size
    ):
        raise ValueError(
            "Layer transition coefficients do not match "
            "the number of Markov states."
        )

    layer_u = (
        transition_coefficients
        @ state_du
    )

    layer_v = (
        transition_coefficients
        @ state_dv
    )

    layer_h = (
        transition_coefficients
        @ state_h
    )

    if layer_h.size:
        layer_h = (
            layer_h
            - 0.5
            * (
                float(
                    np.min(
                        layer_h
                    )
                )
                + float(
                    np.max(
                        layer_h
                    )
                )
            )
        )

    return (
        layer_u[
            :,
            None
        ]
        * vector_a[
            None,
            :
        ]
        + layer_v[
            :,
            None
        ]
        * vector_b[
            None,
            :
        ]
        + layer_h[
            :,
            None
        ]
        * normal[
            None,
            :
        ]
    )


def _compile_one_realization(
    *,
    realization_index: int,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    states: np.ndarray,
    reference_parameters: Mapping[
        str,
        Any,
    ],
    reference_lattice: np.ndarray,
    source_fractional: np.ndarray,
    source_cart: np.ndarray,
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    normal: np.ndarray,
) -> CompiledStochasticRealizationTopology:
    """
    Compile atom membership for one frozen stochastic realization.
    """
    state_du, state_dv, state_h, state_labels = (
        _generic_state_vectors(
            stack_spec,
            reference_parameters,
        )
    )

    layers = _layer_descriptors_from_states(
        realization_index=realization_index,
        states=states,
        number_of_states=int(
            state_du.size
        ),
        labels=state_labels,
    )

    reference_origins = _reference_layer_origins(
        layers=layers,
        vector_a=vector_a,
        vector_b=vector_b,
        normal=normal,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )

    layer_coefficients = np.asarray(
        [
            layer.transition_coefficients
            for layer in layers
        ],
        dtype=float,
    )

    max_shift_u = float(
        np.max(
            np.abs(
                layer_coefficients
                @ state_du
            )
        )
    )

    max_shift_v = float(
        np.max(
            np.abs(
                layer_coefficients
                @ state_dv
            )
        )
    )


    axis1_cells = max(
        float(
            getattr(
                shape_spec,
                "axis1_cells",
                shape_spec.diameter_cells,
            )
        ),
        1.0,
    )

    axis2_cells = max(
        float(
            getattr(
                shape_spec,
                "axis2_cells",
                shape_spec.diameter_cells,
            )
        ),
        1.0,
    )

    count_a = int(
        np.ceil(
            axis1_cells
            + max_shift_u
            + 2.0
        )
    )

    count_b = int(
        np.ceil(
            axis2_cells
            + max_shift_v
            + 2.0
        )
    )

    source_rows = []
    layer_rows = []
    cell_a_rows = []
    cell_b_rows = []

    for layer_descriptor_index, layer in enumerate(
        layers
    ):
        layer_origin = reference_origins[
            layer_descriptor_index
        ]

        for cell_a in range(
            -count_a,
            count_a + 1,
        ):
            for cell_b in range(
                -count_b,
                count_b + 1,
            ):
                positions = (
                    source_cart
                    + cell_a
                    * vector_a
                    + cell_b
                    * vector_b
                    + layer_origin[
                        None,
                        :
                    ]
                )

                inside = _reference_lateral_mask(
                    positions,
                    shape_spec=shape_spec,
                    lattice_matrix=reference_lattice,
                )

                inside_indices = np.flatnonzero(
                    inside
                )

                for source_site_index in inside_indices:
                    source_rows.append(
                        int(
                            source_site_index
                        )
                    )

                    layer_rows.append(
                        int(
                            layer_descriptor_index
                        )
                    )

                    cell_a_rows.append(
                        int(
                            cell_a
                        )
                    )

                    cell_b_rows.append(
                        int(
                            cell_b
                        )
                    )

    if not source_rows:
        raise RuntimeError(
            "The stochastic realization contains no atoms after finite-shape clipping."
        )

    return CompiledStochasticRealizationTopology(
        realization_index=int(
            realization_index
        ),
        layers=layers,
        atom_source_site_indices=np.asarray(
            source_rows,
            dtype=np.int32,
        ),
        atom_layer_descriptor_indices=np.asarray(
            layer_rows,
            dtype=np.int32,
        ),
        atom_cell_a=np.asarray(
            cell_a_rows,
            dtype=np.int32,
        ),
        atom_cell_b=np.asarray(
            cell_b_rows,
            dtype=np.int32,
        ),
    )


def compile_stochastic_ensemble_geometry(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    state_paths: np.ndarray,
    reference_parameters: Optional[
        Mapping[str, Any]
    ] = None,
) -> CompiledStochasticEnsembleGeometry:
    """
    Compile all frozen finite stochastic realizations.

    Parameters
    ----------
    structure
        Single-layer source structure.

    shape_spec
        Finite crystallite dimensions and orientation.

    stack_spec
        Must have:

            model = "correlated_ab_disorder"

    state_paths
        Frozen Markov states of shape:

            (ensemble_samples, number_of_layers)

        State encoding:

            0 = A
            1 = B

    reference_parameters
        Initial continuous vector/lattice values used only to decide which
        source atoms belong to the finite lateral envelope.

    Notes
    -----
    This function intentionally does not compile pair topology. That is the
    responsibility of the next module:

        compiled_stochastic_pairs.py
    """
    if structure is None:
        raise ValueError(
            "A source layer structure is required."
        )

    stack_spec = stack_spec.normalized()

    if not stack_spec.enabled:
        raise ValueError(
            "The stochastic stacking specification is disabled."
        )

    if (
        stack_spec.model
        not in {
            "correlated_ab_disorder",
            "generic_markov",
        }
    ):
        raise ValueError(
            "compile_stochastic_ensemble_geometry requires either "
            "model='correlated_ab_disorder' or "
            "model='generic_markov'."
        )

    state_paths = np.asarray(
        state_paths,
        dtype=np.int16,
    )

    if state_paths.ndim != 2:
        raise ValueError(
            "state_paths must have shape "
            "(ensemble_samples, number_of_layers)."
        )

    if state_paths.shape[0] <= 0:
        raise ValueError(
            "state_paths must contain at least one realization."
        )

    if state_paths.shape[1] <= 0:
        raise ValueError(
            "state_paths must contain at least one layer."
        )

    if stack_spec.model == "generic_markov":
        number_of_states = max(
            1,
            len(
                stack_spec.markov_states
            ),
        )
    else:
        number_of_states = 2

    if not np.all(
        (state_paths >= 0)
        & (
            state_paths
            < number_of_states
        )
    ):
        raise ValueError(
            "state_paths contain a state index outside the "
            "configured Markov-state range."
        )
    reference_parameters = dict(
        reference_parameters
        or {}
    )

    reference_lattice = _lattice_matrix_from_parameters(
        structure,
        reference_parameters,
    )

    vector_a, vector_b, normal = layer_basis(
        reference_lattice
    )

    source_fractional = np.asarray(
        structure.frac_coords,
        dtype=float,
    )

    source_elements = [
        str(
            site.species
        )
        for site in structure.sites
    ]

    source_cart = _centered_source_layer_cartesian(
        source_fractional,
        reference_lattice,
        normal,
    )

    realizations = []

    for realization_index in range(
        state_paths.shape[0]
    ):
        topology = _compile_one_realization(
            realization_index=realization_index,
            structure=structure,
            shape_spec=shape_spec,
            stack_spec=stack_spec,
            states=state_paths[
                realization_index
            ],
            reference_parameters=reference_parameters,
            reference_lattice=reference_lattice,
            source_fractional=source_fractional,
            source_cart=source_cart,
            vector_a=vector_a,
            vector_b=vector_b,
            normal=normal,
        )

        realizations.append(
            topology
        )

    print(
        "[STOCHASTIC GEOMETRY] "
        f"compiled_realizations={len(realizations):,}, "
        f"layers_per_realization={state_paths.shape[1]:,}, "
        f"atoms_min={min(len(item.atom_source_site_indices) for item in realizations):,}, "
        f"atoms_max={max(len(item.atom_source_site_indices) for item in realizations):,}"
    )

    return CompiledStochasticEnsembleGeometry(
        source_structure=structure,
        shape_spec=CrystalliteShapeSpec.from_dict(
            shape_spec.to_dict()
        ),
        stack_spec=stack_spec,
        state_paths=np.asarray(
            state_paths,
            dtype=np.int16,
        ),
        source_fractional=np.asarray(
            source_fractional,
            dtype=float,
        ),
        source_elements=list(
            source_elements
        ),
        realizations=realizations,
        reference_lattice_matrix=np.asarray(
            reference_lattice,
            dtype=float,
        ),
    )