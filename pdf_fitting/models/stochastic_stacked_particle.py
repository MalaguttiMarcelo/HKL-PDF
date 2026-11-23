from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import time

from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    RealizedLayer,
    layer_basis,
)

from pdf_fitting.models.stacked_particle import (
    StackedParticle,
    _centered_base_layer,
    _conservative_in_plane_range,
    _lateral_mask_for_layer,
)

from pdf_fitting.models.stacking_disorder import (
    STATE_A,
    STATE_B,
    state_name,
)

from pdf_fitting.models.compiled_stochastic_geometry import (
    _generic_state_vectors,
)


_EPSILON = 1.0e-10

def stochastic_realized_layers(
    *,
    states: np.ndarray,
    stack_spec: LayerStackSpec,
    parameters=None,
    center_stack: bool = True,
) -> List[RealizedLayer]:
    """
    Build finite realized layers from either:

        correlated A/B stochastic stacking;
        generic N-state Markov stacking.

    The current state emits its transition vector before the next state is
    selected.
    """
    parameters = dict(
        parameters
        or {}
    )

    spec = stack_spec.normalized()

    states = np.asarray(
        states,
        dtype=np.int16,
    ).reshape(
        -1
    )

    if states.size <= 0:
        raise ValueError(
            "The stochastic stack requires at least one layer."
        )

    state_du, state_dv, state_h, state_labels = (
        _generic_state_vectors(
            spec,
            parameters,
        )
    )

    state_du = np.asarray(
        state_du,
        dtype=np.float64,
    )

    state_dv = np.asarray(
        state_dv,
        dtype=np.float64,
    )

    state_h = np.asarray(
        state_h,
        dtype=np.float64,
    )

    number_of_states = int(
        state_du.size
    )

    if number_of_states <= 0:
        raise ValueError(
            "The stochastic stacking model contains no states."
        )

    if not np.all(
        (
            states >= 0
        )
        & (
            states < number_of_states
        )
    ):
        raise ValueError(
            "The stochastic preview path contains a state index outside "
            "the configured Markov-state range."
        )

    if (
        not np.all(
            np.isfinite(
                state_du
            )
        )
        or not np.all(
            np.isfinite(
                state_dv
            )
        )
        or not np.all(
            np.isfinite(
                state_h
            )
        )
        or np.any(
            state_h <= 0.0
        )
    ):
        raise ValueError(
            "Every stochastic state vector must be finite, and every "
            "state height must be positive."
        )

    layer_u = np.zeros(
        states.size,
        dtype=np.float64,
    )

    layer_v = np.zeros(
        states.size,
        dtype=np.float64,
    )

    layer_h = np.zeros(
        states.size,
        dtype=np.float64,
    )

    for layer_index in range(
        states.size - 1
    ):
        current_state = int(
            states[
                layer_index
            ]
        )

        layer_u[
            layer_index + 1
        ] = (
            layer_u[
                layer_index
            ]
            + state_du[
                current_state
            ]
        )

        layer_v[
            layer_index + 1
        ] = (
            layer_v[
                layer_index
            ]
            + state_dv[
                current_state
            ]
        )

        layer_h[
            layer_index + 1
        ] = (
            layer_h[
                layer_index
            ]
            + state_h[
                current_state
            ]
        )

    if center_stack:
        normal_center = 0.5 * (
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

        layer_h -= normal_center

    realized = []

    for layer_index, state_index_value in enumerate(
        states
    ):
        state_index = int(
            state_index_value
        )

        if (
            0
            <= state_index
            < len(
                state_labels
            )
        ):
            label = str(
                state_labels[
                    state_index
                ]
            )
        else:
            label = f"S{state_index}"

        realized.append(
            RealizedLayer(
                realized_index=int(
                    layer_index
                ),
                sequence_index=state_index,
                repeat_index=0,
                label=label,
                shift_u=float(
                    layer_u[
                        layer_index
                    ]
                ),
                shift_v=float(
                    layer_v[
                        layer_index
                    ]
                ),
                stack_coordinate=float(
                    layer_index
                ),
                normal_offset=float(
                    layer_h[
                        layer_index
                    ]
                ),
                enabled=True,
                atom_count=0,
            )
        )

    return realized


def build_stochastic_stacked_particle(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    states: np.ndarray,
    parameters=None,
    max_atoms: Optional[int] = None,
) -> StackedParticle:
    """
    Build one finite atomistic Markov A/B stacked particle.

    This uses the same lateral clipping/envelope rules as build_stacked_particle,
    but the layer origins are generated from a stochastic non-periodic path.
    """
    if structure is None:
        raise ValueError(
            "A source layer structure is required."
        )

    lattice = np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )

    realized_layers = stochastic_realized_layers(
        states=states,
        stack_spec=stack_spec,
        parameters=parameters,
        center_stack=True,
    )

    if not realized_layers:
        raise RuntimeError(
            "The stochastic layer path generated no layers."
        )

    offsets = np.asarray(
        [
            layer.normal_offset
            for layer in realized_layers
        ],
        dtype=float,
    )

    if offsets.size <= 1:
        fallback_h = 0.5 * (
            float(
                stack_spec.state_a_h
            )
            + float(
                stack_spec.state_b_h
            )
        )

        stack_center_span = 0.0
        stack_boundary_thickness = fallback_h

    else:
        differences = np.diff(
            np.sort(
                offsets
            )
        )

        positive = differences[
            differences
            > _EPSILON
        ]

        typical = (
            float(
                np.median(
                    positive
                )
            )
            if positive.size
            else 0.5
            * (
                float(
                    stack_spec.state_a_h
                )
                + float(
                    stack_spec.state_b_h
                )
            )
        )

        stack_center_span = float(
            np.max(
                offsets
            )
            - np.min(
                offsets
            )
        )

        stack_boundary_thickness = (
            stack_center_span
            + typical
        )

    vector_u, vector_v, normal = layer_basis(
        lattice
    )

    base_atoms, base_elements = _centered_base_layer(
        structure,
        normal,
    )

    (
        u_min,
        u_max,
        v_min,
        v_max,
    ) = _conservative_in_plane_range(
        lattice_matrix=lattice,
        shape_spec=shape_spec,
        realized_layers=realized_layers,
    )

    all_atoms = []
    all_elements: List[str] = []

    all_layer_ids = []
    all_layer_indices = []
    all_sequence_ids = []

    layer_origins = []
    layer_indices = []
    layer_sequence_ids = []

    candidate_count = 0
    clipped_count = 0

    for layer_id, realized in enumerate(
        realized_layers
    ):
        origin = (
            float(
                realized.shift_u
            )
            * vector_u
            + float(
                realized.shift_v
            )
            * vector_v
            + float(
                realized.normal_offset
            )
            * normal
        )

        layer_origins.append(
            origin
        )

        layer_indices.append(
            int(
                realized.realized_index
            )
        )

        layer_sequence_ids.append(
            int(
                realized.sequence_index
            )
        )

        layer_atom_count = 0

        for cell_u in range(
            u_min,
            u_max
            + 1,
        ):
            for cell_v in range(
                v_min,
                v_max
                + 1,
            ):
                cell_translation = (
                    cell_u
                    * vector_u
                    + cell_v
                    * vector_v
                )

                positions = (
                    base_atoms
                    + cell_translation[
                        None,
                        :
                    ]
                    + origin[
                        None,
                        :
                    ]
                )

                candidate_count += int(
                    positions.shape[0]
                )

                inside = _lateral_mask_for_layer(
                    positions,
                    layer=realized,
                    shape_spec=shape_spec,
                    lattice_matrix=lattice,
                    stack_boundary_thickness=(
                        stack_boundary_thickness
                    ),
                )

                inside = np.asarray(
                    inside,
                    dtype=bool,
                )

                if not np.any(
                    inside
                ):
                    continue

                selected_positions = positions[
                    inside
                ]

                selected_site_indices = np.flatnonzero(
                    inside
                )

                count = int(
                    selected_positions.shape[0]
                )

                layer_atom_count += count
                clipped_count += count

                all_atoms.append(
                    selected_positions
                )

                all_elements.extend(
                    base_elements[
                        site_index
                    ]
                    for site_index in selected_site_indices
                )

                all_layer_ids.extend(
                    [layer_id]
                    * count
                )

                all_layer_indices.extend(
                    [
                        realized.realized_index
                    ]
                    * count
                )

                all_sequence_ids.extend(
                    [
                        realized.sequence_index
                    ]
                    * count
                )

        realized.atom_count = int(
            layer_atom_count
        )

    if all_atoms:
        atom_cart = np.vstack(
            all_atoms
        ).astype(
            np.float32
        )
    else:
        atom_cart = np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

    atom_layer_ids = np.asarray(
        all_layer_ids,
        dtype=np.int32,
    )

    atom_layer_indices = np.asarray(
        all_layer_indices,
        dtype=np.int32,
    )

    atom_sequence_ids = np.asarray(
        all_sequence_ids,
        dtype=np.int32,
    )

    layer_origins_array = np.asarray(
        layer_origins,
        dtype=np.float32,
    ).reshape(
        -1,
        3,
    )

    if layer_origins_array.shape[0] >= 2:
        transition_starts = layer_origins_array[
            :-1
        ]

        transition_ends = layer_origins_array[
            1:
        ]
    else:
        transition_starts = np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

        transition_ends = np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

    return StackedParticle(
        atom_cart=atom_cart,
        atom_elements=list(
            all_elements
        ),
        atom_layer_ids=atom_layer_ids,
        atom_layer_indices=atom_layer_indices,
        atom_sequence_ids=atom_sequence_ids,
        layer_origins=layer_origins_array,
        layer_indices=np.asarray(
            layer_indices,
            dtype=np.int32,
        ),
        layer_sequence_ids=np.asarray(
            layer_sequence_ids,
            dtype=np.int32,
        ),
        realized_layers=realized_layers,
        transition_starts=transition_starts,
        transition_ends=transition_ends,
        stack_center=np.zeros(
            3,
            dtype=np.float32,
        ),
        stack_normal=np.asarray(
            normal,
            dtype=np.float32,
        ),
        number_of_layer_positions=int(
            len(
                realized_layers
            )
        ),
        number_of_enabled_layers=int(
            len(
                realized_layers
            )
        ),
        stack_center_span=float(
            stack_center_span
        ),
        stack_boundary_thickness=float(
            stack_boundary_thickness
        ),
        n_candidate_atoms=int(
            candidate_count
        ),
        n_clipped_atoms=int(
            clipped_count
        ),
    )