from __future__ import annotations

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np
import time

from pymatgen.core import Lattice

from pdf_fitting.models.compiled_stacked_geometry import (
    _lattice_matrix_from_parameters,
)

from pdf_fitting.models.compiled_stochastic_pairs import (
    CompiledStochasticPairTopology,
    StochasticPairRows,
    evaluate_stochastic_pair_rows,
)

from pdf_fitting.models.layer_stacking import (
    layer_basis,
)

from pdf_fitting.models.compiled_stochastic_geometry import (
    _generic_state_vectors,
)

_EPSILON = 1.0e-12
_DISTANCE_SCALE = 1.0e5


def _safe_species_label(
    value: str,
) -> str:
    text = str(
        value
        or ""
    ).strip()

    if not text:
        return "x"

    if len(text) == 1:
        return text.lower()

    return (
        text[0].upper()
        + text[1:].lower()
    ).lower()


def _reduce_direction(
    h: int,
    k: int,
    l: int,
) -> Tuple[int, int, int]:
    h = int(h)
    k = int(k)
    l = int(l)

    divisor = int(
        np.gcd(
            np.gcd(
                abs(h),
                abs(k),
            ),
            abs(l),
        )
    )

    if divisor == 0:
        return (
            0,
            0,
            0,
        )

    h //= divisor
    k //= divisor
    l //= divisor

    for value in (
        h,
        k,
        l,
    ):
        if value != 0:
            if value < 0:
                h = -h
                k = -k
                l = -l

            break

    return (
        h,
        k,
        l,
    )


def _legacy_hkl_and_direction(
    cartesian_vectors: np.ndarray,
    lattice_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign diagnostic hkl/direction values using an effective statistical
    stacking lattice.

    The stochastic model does not have an exact periodic c vector. The
    effective lattice is used only for shell labels, directions, Warren plots,
    and diagnostics. PDF distances always use shell_cart directly.
    """
    cartesian_vectors = np.asarray(
        cartesian_vectors,
        dtype=np.float64,
    )

    if cartesian_vectors.size == 0:
        return (
            np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.int16,
            ),
            np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.int16,
            ),
        )

    lattice = Lattice(
        np.asarray(
            lattice_matrix,
            dtype=np.float64,
        )
    )

    reciprocal_transpose = np.asarray(
        lattice.reciprocal_lattice.matrix.T,
        dtype=np.float64,
    )

    inverse_reciprocal_transpose = np.linalg.inv(
        reciprocal_transpose
    )

    hkl_float = (
        cartesian_vectors
        @ inverse_reciprocal_transpose.T
    )

    hkl_int = np.rint(
        hkl_float
    ).astype(
        np.int32
    )

    direction = np.zeros_like(
        hkl_int,
        dtype=np.int32,
    )

    for row in range(
        hkl_int.shape[0]
    ):
        direction[
            row
        ] = _reduce_direction(
            int(
                hkl_int[
                    row,
                    0,
                ]
            ),
            int(
                hkl_int[
                    row,
                    1,
                ]
            ),
            int(
                hkl_int[
                    row,
                    2,
                ]
            ),
        )

    int16_min = np.iinfo(
        np.int16
    ).min

    int16_max = np.iinfo(
        np.int16
    ).max

    return (
        np.clip(
            hkl_int,
            int16_min,
            int16_max,
        ).astype(
            np.int16
        ),
        np.clip(
            direction,
            int16_min,
            int16_max,
        ).astype(
            np.int16
        ),
    )


def _state_vectors_from_parameters(
    topology: CompiledStochasticPairTopology,
    parameters: Mapping[
        str,
        Any,
    ],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return the current transition vectors for every stochastic state.
    """
    (
        state_du,
        state_dv,
        state_h,
        _state_labels,
    ) = _generic_state_vectors(
        topology.geometry.stack_spec,
        parameters,
    )

    return (
        np.asarray(
            state_du,
            dtype=np.float64,
        ),
        np.asarray(
            state_dv,
            dtype=np.float64,
        ),
        np.asarray(
            state_h,
            dtype=np.float64,
        ),
    )

def _effective_bulk_transition(
    topology: CompiledStochasticPairTopology,
    parameters: Mapping[
        str,
        Any,
    ],
):
    """
    Calculate the effective transition vector from all configured stochastic
    states and the frozen bulk transition coefficients.
    """
    (
        state_du,
        state_dv,
        state_h,
    ) = _state_vectors_from_parameters(
        topology,
        parameters,
    )

    coefficients = np.asarray(
        topology.bulk_rows.transition_coefficients,
        dtype=np.float64,
    )

    if (
        coefficients.ndim != 2
        or coefficients.shape[1] != state_du.size
    ):
        raise ValueError(
            "The bulk transition coefficient matrix does not match "
            "the configured stochastic-state count."
        )

    state_weight = np.sum(
        np.abs(
            coefficients
        ),
        axis=0,
    )

    total_weight = float(
        np.sum(
            state_weight
        )
    )

    if total_weight <= _EPSILON:
        state_probability = np.full(
            state_du.size,
            1.0
            / max(
                1,
                state_du.size,
            ),
            dtype=np.float64,
        )

    else:
        state_probability = (
            state_weight
            / total_weight
        )

    effective_du = float(
        np.dot(
            state_probability,
            state_du,
        )
    )

    effective_dv = float(
        np.dot(
            state_probability,
            state_dv,
        )
    )

    effective_h = max(
        float(
            np.dot(
                state_probability,
                state_h,
            )
        ),
        _EPSILON,
    )

    return (
        effective_du,
        effective_dv,
        effective_h,
        state_probability,
    )

def _effective_stacking_lattice(
    topology: CompiledStochasticPairTopology,
    parameters: Mapping[
        str,
        Any,
    ],
):
    lattice_matrix = _lattice_matrix_from_parameters(
        topology.geometry.source_structure,
        parameters,
    )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    (
        effective_du,
        effective_dv,
        effective_h,
        state_probability,
    ) = _effective_bulk_transition(
        topology,
        parameters,
    )

    vector_c = (
        effective_du
        * vector_a
        + effective_dv
        * vector_b
        + effective_h
        * normal
    )

    stacking_lattice_matrix = np.asarray(
        [
            vector_a,
            vector_b,
            vector_c,
        ],
        dtype=np.float64,
    )

    volume = abs(
        float(
            np.linalg.det(
                stacking_lattice_matrix
            )
        )
    )

    if (
        not np.isfinite(
            volume
        )
        or volume <= _EPSILON
    ):
        raise ValueError(
            "The effective stochastic stacking lattice has invalid volume."
        )

    inverse = np.linalg.inv(
        stacking_lattice_matrix
    )

    return (
        np.asarray(
            lattice_matrix,
            dtype=np.float64,
        ),
        np.asarray(
            stacking_lattice_matrix,
            dtype=np.float64,
        ),
        np.asarray(
            inverse,
            dtype=np.float64,
        ),
        float(
            volume
        ),
        float(
            effective_du
        ),
        float(
            effective_dv
        ),
        float(
            effective_h
        ),
        np.asarray(
            state_probability,
            dtype=np.float64,
        ),
    )


def _make_group_key(
    *,
    alpha_id: np.ndarray,
    beta_id: np.ndarray,
    distance: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Group by:

        ordered element pair;
        current pair distance;
        reduced effective stacking direction.
    """
    alpha_id = np.asarray(
        alpha_id,
        dtype=np.int16,
    )

    beta_id = np.asarray(
        beta_id,
        dtype=np.int16,
    )

    distance = np.asarray(
        distance,
        dtype=np.float64,
    )

    direction = np.asarray(
        direction,
        dtype=np.int16,
    )

    key = np.zeros(
        distance.size,
        dtype=[
            (
                "alpha",
                np.int16,
            ),
            (
                "beta",
                np.int16,
            ),
            (
                "distance",
                np.int32,
            ),
            (
                "dh",
                np.int16,
            ),
            (
                "dk",
                np.int16,
            ),
            (
                "dl",
                np.int16,
            ),
        ],
    )

    key[
        "alpha"
    ] = alpha_id

    key[
        "beta"
    ] = beta_id

    key[
        "distance"
    ] = np.rint(
        distance
        * _DISTANCE_SCALE
    ).astype(
        np.int32
    )

    key[
        "dh"
    ] = direction[
        :,
        0,
    ]

    key[
        "dk"
    ] = direction[
        :,
        1,
    ]

    key[
        "dl"
    ] = direction[
        :,
        2,
    ]

    return key


def _key_tuple(
    value,
):
    return (
        int(
            value[
                "alpha"
            ]
        ),
        int(
            value[
                "beta"
            ]
        ),
        int(
            value[
                "distance"
            ]
        ),
        int(
            value[
                "dh"
            ]
        ),
        int(
            value[
                "dk"
            ]
        ),
        int(
            value[
                "dl"
            ]
        ),
    )


def _group_rows(
    *,
    rows: StochasticPairRows,
    cartesian: np.ndarray,
    distances: np.ndarray,
    directions: np.ndarray,
):
    """
    Group weighted directed rows into current stochastic shells.
    """
    cartesian = np.asarray(
        cartesian,
        dtype=np.float64,
    )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    directions = np.asarray(
        directions,
        dtype=np.int16,
    )

    valid = (
        np.isfinite(
            distances
        )
        & (
            distances
            > 1.0e-8
        )
    )

    if not np.any(
        valid
    ):
        return {
            "key": np.zeros(
                0,
                dtype=[
                    (
                        "alpha",
                        np.int16,
                    ),
                    (
                        "beta",
                        np.int16,
                    ),
                    (
                        "distance",
                        np.int32,
                    ),
                    (
                        "dh",
                        np.int16,
                    ),
                    (
                        "dk",
                        np.int16,
                    ),
                    (
                        "dl",
                        np.int16,
                    ),
                ],
            ),
            "weight": np.zeros(
                0,
                dtype=np.float64,
            ),
            "first_index": np.zeros(
                0,
                dtype=np.int64,
            ),
            "cartesian": np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            ),
            "distance": np.zeros(
                0,
                dtype=np.float64,
            ),
            "alpha_id": np.zeros(
                0,
                dtype=np.int16,
            ),
            "beta_id": np.zeros(
                0,
                dtype=np.int16,
            ),
            "direction": np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.int16,
            ),
        }

    alpha_id = np.asarray(
        rows.alpha_id,
        dtype=np.int16,
    )[
        valid
    ]

    beta_id = np.asarray(
        rows.beta_id,
        dtype=np.int16,
    )[
        valid
    ]

    cartesian_valid = cartesian[
        valid
    ]

    distances_valid = distances[
        valid
    ]

    directions_valid = directions[
        valid
    ]

    weights_valid = np.asarray(
        rows.row_weight,
        dtype=np.float64,
    )[
        valid
    ]

    key = _make_group_key(
        alpha_id=alpha_id,
        beta_id=beta_id,
        distance=distances_valid,
        direction=directions_valid,
    )

    (
        unique_key,
        first_index,
        inverse,
    ) = np.unique(
        key,
        return_index=True,
        return_inverse=True,
    )

    grouped_weight = np.bincount(
        inverse.astype(
            np.int64
        ),
        weights=weights_valid,
        minlength=unique_key.size,
    ).astype(
        np.float64
    )

    return {
        "key": unique_key,
        "weight": grouped_weight,
        "first_index": first_index.astype(
            np.int64,
            copy=False,
        ),
        "cartesian": cartesian_valid[
            first_index
        ].astype(
            np.float64,
            copy=False,
        ),
        "distance": distances_valid[
            first_index
        ].astype(
            np.float64,
            copy=False,
        ),
        "alpha_id": alpha_id[
            first_index
        ].astype(
            np.int16,
            copy=False,
        ),
        "beta_id": beta_id[
            first_index
        ].astype(
            np.int16,
            copy=False,
        ),
        "direction": directions_valid[
            first_index
        ].astype(
            np.int16,
            copy=False,
        ),
    }


def _rank_shells_within_upairs(
    shell_upair_id: np.ndarray,
    distance_integer: np.ndarray,
) -> np.ndarray:
    shell_upair_id = np.asarray(
        shell_upair_id,
        dtype=np.int32,
    )

    distance_integer = np.asarray(
        distance_integer,
        dtype=np.int64,
    )

    number_of_shells = int(
        shell_upair_id.size
    )

    output = np.zeros(
        number_of_shells,
        dtype=np.int32,
    )

    if number_of_shells <= 0:
        return output

    order = np.lexsort(
        (
            distance_integer,
            shell_upair_id,
        )
    )

    upair_sorted = shell_upair_id[
        order
    ]

    distance_sorted = distance_integer[
        order
    ]

    new_upair = np.ones(
        number_of_shells,
        dtype=bool,
    )

    new_distance = np.ones(
        number_of_shells,
        dtype=bool,
    )

    if number_of_shells > 1:
        new_upair[
            1:
        ] = (
            upair_sorted[
                1:
            ]
            != upair_sorted[
                :-1
            ]
        )

        new_distance[
            1:
        ] = (
            new_upair[
                1:
            ]
            | (
                distance_sorted[
                    1:
                ]
                != distance_sorted[
                    :-1
                ]
            )
        )

    cumulative = (
        np.cumsum(
            new_distance
        ).astype(
            np.int32
        )
        - 1
    )

    reset = np.maximum.accumulate(
        np.where(
            new_upair,
            cumulative,
            0,
        )
    ).astype(
        np.int32
    )

    output[
        order
    ] = (
        cumulative
        - reset
    )

    return output


def build_stochastic_ensemble_shell_table(
    *,
    topology: CompiledStochasticPairTopology,
    parameters: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build one PDFCalculator-compatible shell table for a frozen stochastic
    ensemble.

    This function is intentionally called every time the continuous state
    vectors or lattice parameters change.

    It performs:

        1. evaluate finite stochastic pair rows;
        2. evaluate stochastic bulk rows;
        3. group current pair vectors by pair/distance/direction;
        4. calculate finite and bulk effective coordination;
        5. calculate discrete finite gamma;
        6. return a normal packed shell table.

    The PDF backend can then use its normal Gaussian accumulation kernels.
    """
    parameters = dict(
        parameters
        or {}
    )

    build_start = time.perf_counter()

    (
        parent_lattice,
        stacking_lattice,
        stacking_inverse,
        stacking_volume,
        effective_du,
        effective_dv,
        effective_h,
        state_probability,
    ) = _effective_stacking_lattice(
        topology,
        parameters,
    )

    (
        state_du,
        state_dv,
        state_h,
    ) = _state_vectors_from_parameters(
        topology,
        parameters,
    )

    (
        finite_cart,
        finite_distance,
    ) = evaluate_stochastic_pair_rows(
        topology.finite_rows,
        lattice_matrix=parent_lattice,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )

    (
        bulk_cart,
        bulk_distance,
    ) = evaluate_stochastic_pair_rows(
        topology.bulk_rows,
        lattice_matrix=parent_lattice,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )
    finite_active = (
        np.isfinite(
            finite_distance
        )
        & (
            finite_distance
            > 1.0e-8
        )
        & (
            finite_distance
            <= float(
                topology.pair_cutoff
            )
            + 1.0e-10
        )
    )

    bulk_active = (
        np.isfinite(
            bulk_distance
        )
        & (
            bulk_distance
            > 1.0e-8
        )
        & (
            bulk_distance
            <= float(
                topology.pair_cutoff
            )
            + 1.0e-10
        )
    )

    finite_cart = finite_cart[
        finite_active
    ]

    finite_distance = finite_distance[
        finite_active
    ]

    bulk_cart = bulk_cart[
        bulk_active
    ]

    bulk_distance = bulk_distance[
        bulk_active
    ]

    finite_rows = StochasticPairRows(
        source_alpha=topology.finite_rows.source_alpha[
            finite_active
        ],
        source_beta=topology.finite_rows.source_beta[
            finite_active
        ],
        delta_fractional=topology.finite_rows.delta_fractional[
            finite_active
        ],
        delta_a=topology.finite_rows.delta_a[
            finite_active
        ],
        delta_b=topology.finite_rows.delta_b[
            finite_active
        ],
        transition_coefficients=(
            topology
            .finite_rows
            .transition_coefficients[
                finite_active
            ]
        ),
        reference_cart=topology.finite_rows.reference_cart[
            finite_active
        ],
        reference_distance=topology.finite_rows.reference_distance[
            finite_active
        ],
        alpha_id=topology.finite_rows.alpha_id[
            finite_active
        ],
        beta_id=topology.finite_rows.beta_id[
            finite_active
        ],
        realization_index=topology.finite_rows.realization_index[
            finite_active
        ],
        row_weight=topology.finite_rows.row_weight[
            finite_active
        ],
    )


    bulk_rows = StochasticPairRows(
        source_alpha=topology.bulk_rows.source_alpha[
            bulk_active
        ],
        source_beta=topology.bulk_rows.source_beta[
            bulk_active
        ],
        delta_fractional=topology.bulk_rows.delta_fractional[
            bulk_active
        ],
        delta_a=topology.bulk_rows.delta_a[
            bulk_active
        ],
        delta_b=topology.bulk_rows.delta_b[
            bulk_active
        ],
        transition_coefficients=(
            topology
            .bulk_rows
            .transition_coefficients[
                bulk_active
            ]
        ),
        reference_cart=topology.bulk_rows.reference_cart[
            bulk_active
        ],
        reference_distance=topology.bulk_rows.reference_distance[
            bulk_active
        ],
        alpha_id=topology.bulk_rows.alpha_id[
            bulk_active
        ],
        beta_id=topology.bulk_rows.beta_id[
            bulk_active
        ],
        realization_index=topology.bulk_rows.realization_index[
            bulk_active
        ],
        row_weight=topology.bulk_rows.row_weight[
            bulk_active
        ],
    )

    if finite_rows.number_of_rows <= 0:
        raise RuntimeError(
            "No active finite stochastic pair rows remain at the current parameters."
        )

    if bulk_rows.number_of_rows <= 0:
        raise RuntimeError(
            "No active stochastic bulk pair rows remain at the current parameters."
        )

    finite_hkl, finite_direction = _hkl_and_direction(
        finite_cart,
        stacking_lattice,
    )

    bulk_hkl, bulk_direction = _hkl_and_direction(
        bulk_cart,
        stacking_lattice,
    )

    finite_grouped = _group_rows(
        rows=finite_rows,
        cartesian=finite_cart,
        distances=finite_distance,
        directions=finite_direction,
    )

    bulk_grouped = _group_rows(
        rows=bulk_rows,
        cartesian=bulk_cart,
        distances=bulk_distance,
        directions=bulk_direction,
    )

    if finite_grouped[
        "key"
    ].size <= 0:
        raise RuntimeError(
            "Finite stochastic shell grouping produced no shells."
        )

    bulk_lookup = {
        _key_tuple(
            key
        ): float(
            weight
        )
        for key, weight in zip(
            bulk_grouped[
                "key"
            ],
            bulk_grouped[
                "weight"
            ],
        )
    }

    finite_center_weight = np.asarray(
        topology.finite_center_species_weight,
        dtype=np.float64,
    )

    bulk_center_weight = np.asarray(
        topology.bulk_center_species_weight,
        dtype=np.float64,
    )

    shell_alpha_id = np.asarray(
        finite_grouped[
            "alpha_id"
        ],
        dtype=np.int16,
    )

    shell_beta_id = np.asarray(
        finite_grouped[
            "beta_id"
        ],
        dtype=np.int16,
    )

    shell_cart = np.asarray(
        finite_grouped[
            "cartesian"
        ],
        dtype=np.float64,
    )

    shell_distance = np.asarray(
        finite_grouped[
            "distance"
        ],
        dtype=np.float64,
    )

    shell_direction = np.asarray(
        finite_grouped[
            "direction"
        ],
        dtype=np.int16,
    )

    shell_hkl = np.asarray(
        finite_hkl[
            finite_grouped[
                "first_index"
            ]
        ],
        dtype=np.int16,
    )

    finite_weight = np.asarray(
        finite_grouped[
            "weight"
        ],
        dtype=np.float64,
    )

    bulk_weight = np.zeros(
        finite_weight.size,
        dtype=np.float64,
    )

    for index, key in enumerate(
        finite_grouped[
            "key"
        ]
    ):
        bulk_weight[
            index
        ] = float(
            bulk_lookup.get(
                _key_tuple(
                    key
                ),
                0.0,
            )
        )

    shell_mult_eff = np.zeros(
        finite_weight.size,
        dtype=np.float32,
    )

    shell_bulk_mult_eff = np.zeros(
        finite_weight.size,
        dtype=np.float32,
    )

    shell_gamma = np.zeros(
        finite_weight.size,
        dtype=np.float32,
    )

    for index in range(
        finite_weight.size
    ):
        alpha_id = int(
            shell_alpha_id[
                index
            ]
        )

        finite_denominator = (
            float(
                finite_center_weight[
                    alpha_id
                ]
            )
            if (
                0
                <= alpha_id
                < finite_center_weight.size
            )
            else 0.0
        )

        bulk_denominator = (
            float(
                bulk_center_weight[
                    alpha_id
                ]
            )
            if (
                0
                <= alpha_id
                < bulk_center_weight.size
            )
            else 0.0
        )

        finite_effective = 0.0
        bulk_effective = 0.0

        if finite_denominator > _EPSILON:
            finite_effective = (
                finite_weight[
                    index
                ]
                / finite_denominator
            )

        if bulk_denominator > _EPSILON:
            bulk_effective = (
                bulk_weight[
                    index
                ]
                / bulk_denominator
            )

        gamma = 0.0

        if bulk_effective > _EPSILON:
            gamma = (
                finite_effective
                / bulk_effective
            )

        shell_mult_eff[
            index
        ] = np.float32(
            finite_effective
        )

        shell_bulk_mult_eff[
            index
        ] = np.float32(
            bulk_effective
        )

        shell_gamma[
            index
        ] = np.float32(
            gamma
        )

    unique_species = list(
        topology.unique_species
    )

    safe_species = [
        _safe_species_label(
            species
        )
        for species in unique_species
    ]

    pair_values = sorted(
        {
            (
                safe_species[
                    int(
                        alpha_id
                    )
                ],
                safe_species[
                    int(
                        beta_id
                    )
                ],
            )
            for alpha_id, beta_id in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        }
    )

    pair_list = list(
        pair_values
    )

    pair_index = {
        pair: index
        for index, pair in enumerate(
            pair_list
        )
    }

    upair_values = sorted(
        {
            tuple(
                sorted(
                    pair
                )
            )
            for pair in pair_list
        }
    )

    upair_list = list(
        upair_values
    )

    upair_index = {
        pair: index
        for index, pair in enumerate(
            upair_list
        )
    }

    shell_pair_id = np.asarray(
        [
            pair_index[
                (
                    safe_species[
                        int(
                            alpha_id
                        )
                    ],
                    safe_species[
                        int(
                            beta_id
                        )
                    ],
                )
            ]
            for alpha_id, beta_id in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        ],
        dtype=np.int32,
    )

    shell_upair_id = np.asarray(
        [
            upair_index[
                tuple(
                    sorted(
                        (
                            safe_species[
                                int(
                                    alpha_id
                                )
                            ],
                            safe_species[
                                int(
                                    beta_id
                                )
                            ],
                        )
                    )
                )
            ]
            for alpha_id, beta_id in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        ],
        dtype=np.int32,
    )

    distance_integer = np.rint(
        shell_distance
        * _DISTANCE_SCALE
    ).astype(
        np.int64
    )

    shell_k_within_upair = _rank_shells_within_upairs(
        shell_upair_id,
        distance_integer,
    )

    shell_k_rank = np.arange(
        shell_pair_id.size,
        dtype=np.int32,
    )

    shell_frac = (
        shell_cart
        @ stacking_inverse
    ).astype(
        np.float32
    )

    shell_mult = np.maximum(
        0,
        np.rint(
            finite_weight
        ).astype(
            np.int64
        ),
    )

    shell_bulk_mult = np.maximum(
        0,
        np.rint(
            bulk_weight
        ).astype(
            np.int64
        ),
    )

    int32_max = np.iinfo(
        np.int32
    ).max

    shell_mult = np.minimum(
        shell_mult,
        int32_max,
    ).astype(
        np.int32
    )

    shell_bulk_mult = np.minimum(
        shell_bulk_mult,
        int32_max,
    ).astype(
        np.int32
    )

    order = np.argsort(
        shell_pair_id,
        kind="mergesort",
    )

    shell_frac = shell_frac[
        order
    ]

    shell_cart = shell_cart[
        order
    ].astype(
        np.float32
    )

    shell_mult = shell_mult[
        order
    ]

    shell_bulk_mult = shell_bulk_mult[
        order
    ]

    shell_mult_eff = shell_mult_eff[
        order
    ]

    shell_bulk_mult_eff = shell_bulk_mult_eff[
        order
    ]

    shell_gamma = shell_gamma[
        order
    ]

    shell_pair_id = shell_pair_id[
        order
    ]

    shell_alpha_id = shell_alpha_id[
        order
    ]

    shell_beta_id = shell_beta_id[
        order
    ]

    shell_hkl = shell_hkl[
        order
    ]

    shell_direction = shell_direction[
        order
    ]

    shell_k_rank = shell_k_rank[
        order
    ]

    shell_upair_id = shell_upair_id[
        order
    ]

    shell_k_within_upair = shell_k_within_upair[
        order
    ]

    number_of_pairs = int(
        len(
            pair_list
        )
    )

    pair_counts = np.bincount(
        shell_pair_id,
        minlength=number_of_pairs,
    ).astype(
        np.int32
    )

    pair_offsets = np.zeros(
        number_of_pairs + 1,
        dtype=np.int32,
    )

    pair_offsets[
        1:
    ] = np.cumsum(
        pair_counts
    )

    pair_alpha_id = np.zeros(
        number_of_pairs,
        dtype=np.int16,
    )

    pair_beta_id = np.zeros(
        number_of_pairs,
        dtype=np.int16,
    )

    for pair_id in range(
        number_of_pairs
    ):
        rows = np.where(
            shell_pair_id
            == pair_id
        )[0]

        if rows.size <= 0:
            continue

        first = int(
            rows[
                0
            ]
        )

        pair_alpha_id[
            pair_id
        ] = shell_alpha_id[
            first
        ]

        pair_beta_id[
            pair_id
        ] = shell_beta_id[
            first
        ]

    try:
        (
            strain_unique_dirs,
            strain_inv_dirs,
        ) = np.unique(
            shell_direction.astype(
                np.int32,
                copy=False,
            ),
            axis=0,
            return_inverse=True,
        )

        strain_unique_dirs = strain_unique_dirs.astype(
            np.int32,
            copy=False,
        )

        strain_inv_dirs = strain_inv_dirs.astype(
            np.int32,
            copy=False,
        )

    except Exception:
        strain_unique_dirs = np.zeros(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

        strain_inv_dirs = np.zeros(
            0,
            dtype=np.int32,
        )

    source_atom_count = int(
        len(
            topology.geometry.source_structure.sites
        )
    )

    effective_bulk_atoms = max(
        1.0,
        float(
            source_atom_count
        ),
    )

    stochastic_bulk_density = (
        float(
            source_atom_count
        )
        / max(
            stacking_volume,
            _EPSILON,
        )
    )

    out = {
        "geometry_mode": "stochastic_ensemble",
        "shell_frac": shell_frac,
        "shell_cart": shell_cart,
        "shell_mult": shell_mult,
        "shell_bulk_mult": shell_bulk_mult,
        "shell_mult_eff": shell_mult_eff,
        "shell_finite_mult_eff": shell_mult_eff.copy(),
        "shell_bulk_mult_eff": shell_bulk_mult_eff,
        "shell_gamma": shell_gamma,
        "shell_pair_id": shell_pair_id,
        "shell_alpha_id": shell_alpha_id,
        "shell_beta_id": shell_beta_id,
        "shell_hkl": shell_hkl,
        "shell_direction": shell_direction,
        "shell_k_rank": shell_k_rank,
        "shell_upair_id": shell_upair_id,
        "shell_k_within_upair": shell_k_within_upair,
        "pair_offsets": pair_offsets,
        "pair_alpha_id": pair_alpha_id,
        "pair_beta_id": pair_beta_id,
        "unique_species": unique_species,
        "pair_list": pair_list,
        "upair_list": upair_list,
        "strain_unique_dirs": strain_unique_dirs,
        "strain_inv_dirs": strain_inv_dirs,
        "stacking_lattice_matrix": stacking_lattice,
        "stacking_inverse_lattice_matrix": stacking_inverse,
        "stacking_volume": float(
            stacking_volume
        ),
        "stacking_bulk_number_density": float(
            stochastic_bulk_density
        ),
        "bulk_atom_count": int(
            round(
                effective_bulk_atoms
            )
        ),
        "stacking_nominal_spacing": float(
            effective_h
        ),
        "stacking_repeat_period": 1.0,
        "stacking_period_normal_height": float(
            effective_h
        ),
        "finite_atom_count": int(
            round(
                np.sum(
                    topology.finite_center_species_weight
                )
                * max(
                    1,
                    topology.geometry.ensemble_samples,
                )
            )
        ),
        "number_of_layer_positions": int(
            topology.geometry.number_of_layers
        ),
        "stochastic_model": "correlated_ab_disorder",
        "stochastic_ensemble_samples": int(
            topology.geometry.ensemble_samples
        ),
        "stochastic_bulk_layers": int(
            topology.bulk_layers
        ),
        "stochastic_bulk_realizations": int(
            topology.bulk_realizations
        ),
        "stochastic_effective_du": float(
            effective_du
        ),
        "stochastic_effective_dv": float(
            effective_dv
        ),
        "stochastic_effective_h": float(
            effective_h
        ),
        "stochastic_state_probability": np.asarray(
            state_probability,
            dtype=np.float64,
        ),
    }

    print(
        "[STOCHASTIC SHELLS] "
        f"finite_rows={finite_rows.number_of_rows:,}, "
        f"bulk_rows={bulk_rows.number_of_rows:,}, "
        f"shells={shell_frac.shape[0]:,}, "
        f"effective_h={effective_h:.6g} Å, "
        f"time={time.perf_counter() - build_start:.3f} s"
    )

    return out


def _hkl_and_direction(cartesian, lattice_matrix):
    """hkl as before; direction = exact reciprocal direction parallel to r (direction_utils)."""
    from pdf_fitting.models.direction_utils import reciprocal_directions

    # hkl as the legacy code (nearest reciprocal-lattice point, h ≈ G·u/2π),
    # without its slow per-row direction reduction (replaced just below).
    cart = np.asarray(cartesian, dtype=np.float64).reshape(-1, 3)
    if cart.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=np.int16)
        return empty, empty.copy()
    A = np.asarray(lattice_matrix, dtype=np.float64)
    recip_T = (2.0 * np.pi * np.linalg.inv(A).T).T  # pymatgen reciprocal_lattice.matrix.T
    hkl = np.rint(cart @ np.linalg.inv(recip_T).T).astype(np.int16)
    direction = reciprocal_directions(cart, A)
    return hkl, direction.astype(np.int16, copy=False)
