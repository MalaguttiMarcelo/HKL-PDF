from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

import time

import numpy as np

from pdf_fitting.models.fast_unique import unique_rows as _fast_unique_rows

try:
    from numba import (
        njit,
        prange,
    )

    NUMBA_AVAILABLE = True

except Exception:
    NUMBA_AVAILABLE = False
    njit = None
    prange = range

from pymatgen.core import Lattice

from pdf_fitting.models.compiled_stacked_geometry import (
    _lattice_matrix_from_parameters,
)

from pdf_fitting.models.compiled_stochastic_pairs import (
    CompiledStochasticPairTopology,
    StochasticPairRows,
    evaluate_stochastic_pair_rows,
)

from pdf_fitting.models.compiled_stochastic_geometry import (
    _generic_state_vectors,
)

from pdf_fitting.models.layer_stacking import (
    layer_basis,
)


_EPSILON = 1.0e-12
_DESCRIPTOR_SCALE = 1.0e10


@dataclass
class CompiledStochasticShellTopology:
    """
    Fixed stochastic shell descriptors.

    The key design rule is that arbitrary Markov-state geometry is represented
    by one coefficient matrix:

        transition_coefficients.shape == (n_shells, n_states)

    A/B stacking is simply the special case n_states == 2.
    """

    pair_topology: CompiledStochasticPairTopology

    source_alpha: np.ndarray
    source_beta: np.ndarray

    delta_fractional: np.ndarray
    delta_a: np.ndarray
    delta_b: np.ndarray

    transition_coefficients: np.ndarray

    shell_mult_eff: np.ndarray
    shell_bulk_mult_eff: np.ndarray
    shell_gamma: np.ndarray

    shell_alpha_id: np.ndarray
    shell_beta_id: np.ndarray

    shell_hkl: np.ndarray
    shell_direction: np.ndarray

    shell_pair_id: np.ndarray
    shell_upair_id: np.ndarray
    shell_k_rank: np.ndarray
    shell_k_within_upair: np.ndarray

    pair_offsets: np.ndarray
    pair_alpha_id: np.ndarray
    pair_beta_id: np.ndarray

    unique_species: list
    pair_list: list
    upair_list: list

    strain_unique_dirs: np.ndarray
    strain_inv_dirs: np.ndarray

    reference_distance: np.ndarray

    stacking_lattice_matrix_reference: np.ndarray
    stacking_inverse_lattice_matrix_reference: np.ndarray
    stacking_volume_reference: float
    stacking_bulk_number_density_reference: float

    finite_atom_count_average: float
    bulk_atom_count_reference: int

    @property
    def number_of_shells(
        self,
    ) -> int:
        return int(
            self.source_alpha.size
        )


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


def _packed_view(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.ascontiguousarray(
        matrix,
        dtype=np.int64,
    )

    packed_dtype = np.dtype(
        (
            np.void,
            matrix.dtype.itemsize
            * matrix.shape[1],
        )
    )

    return matrix.view(
        packed_dtype
    ).reshape(
        -1
    )


def _descriptor_matrix(
    rows: StochasticPairRows,
) -> np.ndarray:
    """
    Descriptor independent of continuously refined geometry.

    Current distance is deliberately excluded. Distances are recalculated
    on every model evaluation from lattice/state vectors.
    """
    coefficients = np.asarray(
        rows.transition_coefficients,
        dtype=np.int64,
    )

    return np.ascontiguousarray(
        np.column_stack(
            (
                np.asarray(
                    rows.source_alpha,
                    dtype=np.int64,
                ),
                np.asarray(
                    rows.source_beta,
                    dtype=np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_fractional,
                        dtype=np.float64,
                    )[
                        :,
                        0,
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64
                ),
                np.rint(
                    np.asarray(
                        rows.delta_fractional,
                        dtype=np.float64,
                    )[
                        :,
                        1,
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64
                ),
                np.rint(
                    np.asarray(
                        rows.delta_fractional,
                        dtype=np.float64,
                    )[
                        :,
                        2,
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64
                ),
                np.rint(
                    np.asarray(
                        rows.delta_a,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64
                ),
                np.rint(
                    np.asarray(
                        rows.delta_b,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64
                ),
                coefficients,
            )
        ),
        dtype=np.int64,
    )


def _state_vectors(
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
    Return generic state vector arrays:

        state_du.shape == (n_states,)
        state_dv.shape == (n_states,)
        state_h.shape  == (n_states,)
    """
    state_du, state_dv, state_h, _labels = (
        _generic_state_vectors(
            topology.geometry.stack_spec,
            parameters,
        )
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


_STATE_WEIGHT_CACHE = {}


def _bulk_state_weight(topology, n_states: int) -> np.ndarray:
    coefficients = topology.bulk_rows.transition_coefficients
    key = (id(coefficients), getattr(coefficients, "shape", None))
    cached = _STATE_WEIGHT_CACHE.get(key)
    if cached is not None and cached[0] is coefficients:
        weight = cached[1]
    else:
        arr = np.asarray(coefficients, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != n_states:
            raise ValueError(
                "Bulk transition coefficient matrix does not match "
                "the configured Markov-state count."
            )
        weight = np.sum(np.abs(arr), axis=0)
        if len(_STATE_WEIGHT_CACHE) > 8:
            _STATE_WEIGHT_CACHE.clear()
        _STATE_WEIGHT_CACHE[key] = (coefficients, weight)
    if weight.size != n_states:
        raise ValueError(
            "Bulk transition coefficient matrix does not match "
            "the configured Markov-state count."
        )
    return weight


def _effective_stacking_lattice(
    topology: CompiledStochasticPairTopology,
    parameters: Mapping[
        str,
        Any,
    ],
):
    """
    Build a statistical effective lattice for diagnostics, hkl labels,
    shell fractional coordinates, and number-density reporting.

    The PDF itself always uses exact Cartesian pair vectors.
    """
    parent_lattice = _lattice_matrix_from_parameters(
        topology.geometry.source_structure,
        parameters,
    )

    vector_a, vector_b, normal = layer_basis(
        parent_lattice
    )

    state_du, state_dv, state_h = _state_vectors(
        topology,
        parameters,
    )

    # Σ|C| over the bulk rows is fixed for a compiled topology: computed once
    # (it was a full pass over millions of rows on every evaluation).
    state_weight = _bulk_state_weight(topology, state_du.size)
    total_weight = float(
        np.sum(
            state_weight
        )
    )

    if total_weight <= _EPSILON:
        state_probability = np.full(
            state_du.size,
            1.0 / max(
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

    vector_c = (
        effective_du
        * vector_a
        + effective_dv
        * vector_b
        + effective_h
        * normal
    )

    stacking_lattice = np.asarray(
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
                stacking_lattice
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
            "The effective stochastic stacking lattice is singular."
        )

    stacking_inverse = np.linalg.inv(
        stacking_lattice
    )

    return (
        np.asarray(
            parent_lattice,
            dtype=np.float64,
        ),
        np.asarray(
            stacking_lattice,
            dtype=np.float64,
        ),
        np.asarray(
            stacking_inverse,
            dtype=np.float64,
        ),
        float(
            volume
        ),
        float(
            effective_h
        ),
        np.asarray(
            state_probability,
            dtype=np.float64,
        ),
    )


def _reduce_direction(
    h: int,
    k: int,
    l: int,
) -> Tuple[
    int,
    int,
    int,
]:
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

if NUMBA_AVAILABLE:
    @njit(
        cache=True,
        parallel=True,
    )
    def _reduce_directions_numba(
        hkl: np.ndarray,
    ) -> np.ndarray:
        number_of_rows = int(
            hkl.shape[0]
        )

        output = np.zeros(
            (
                number_of_rows,
                3,
            ),
            dtype=np.int32,
        )

        for index in prange(
            number_of_rows
        ):
            h = int(
                hkl[
                    index,
                    0,
                ]
            )

            k = int(
                hkl[
                    index,
                    1,
                ]
            )

            l = int(
                hkl[
                    index,
                    2,
                ]
            )

            first = abs(
                h
            )

            second = abs(
                k
            )

            while second != 0:
                first, second = (
                    second,
                    first % second,
                )

            divisor = int(
                first
            )

            first = abs(
                divisor
            )

            second = abs(
                l
            )

            while second != 0:
                first, second = (
                    second,
                    first % second,
                )

            divisor = int(
                first
            )

            if divisor == 0:
                continue

            h //= divisor
            k //= divisor
            l //= divisor

            if h != 0:
                if h < 0:
                    h = -h
                    k = -k
                    l = -l

            elif k != 0:
                if k < 0:
                    h = -h
                    k = -k
                    l = -l

            elif l < 0:
                h = -h
                k = -k
                l = -l

            output[
                index,
                0,
            ] = h

            output[
                index,
                1,
            ] = k

            output[
                index,
                2,
            ] = l

        return output

else:
    _reduce_directions_numba = None

def _legacy_hkl_and_direction(
    cartesian: np.ndarray,
    stacking_lattice: np.ndarray,
):
    cartesian = np.asarray(
        cartesian,
        dtype=np.float64,
    )

    if cartesian.size == 0:
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
        stacking_lattice
    )

    reciprocal_transpose = np.asarray(
        lattice.reciprocal_lattice.matrix.T,
        dtype=np.float64,
    )

    inverse_reciprocal_transpose = np.linalg.inv(
        reciprocal_transpose
    )

    hkl = np.rint(
        cartesian
        @ inverse_reciprocal_transpose.T
    ).astype(
        np.int32
    )

    if _reduce_directions_numba is not None:
        direction = _reduce_directions_numba(
            hkl
        )

    else:
        direction = np.zeros_like(
            hkl,
            dtype=np.int32,
        )

        for index in range(
            hkl.shape[0]
        ):
            direction[
                index
            ] = _reduce_direction(
                int(
                    hkl[
                        index,
                        0,
                    ]
                ),
                int(
                    hkl[
                        index,
                        1,
                    ]
                ),
                int(
                    hkl[
                        index,
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
            hkl,
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


def _rank_shells_within_upairs(
    shell_upair_id: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    shell_upair_id = np.asarray(
        shell_upair_id,
        dtype=np.int32,
    )

    distance_key = np.rint(
        np.asarray(
            distance,
            dtype=np.float64,
        )
        * 1.0e5
    ).astype(
        np.int64
    )

    output = np.zeros(
        shell_upair_id.size,
        dtype=np.int32,
    )

    if shell_upair_id.size <= 0:
        return output

    order = np.lexsort(
        (
            distance_key,
            shell_upair_id,
        )
    )

    pair_sorted = shell_upair_id[
        order
    ]

    distance_sorted = distance_key[
        order
    ]

    new_pair = np.ones(
        order.size,
        dtype=bool,
    )

    new_distance = np.ones(
        order.size,
        dtype=bool,
    )

    if order.size > 1:
        new_pair[
            1:
        ] = (
            pair_sorted[
                1:
            ]
            != pair_sorted[
                :-1
            ]
        )

        new_distance[
            1:
        ] = (
            new_pair[
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
            new_pair,
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


def _pair_metadata(
    *,
    unique_species: list,
    shell_alpha_id: np.ndarray,
    shell_beta_id: np.ndarray,
    reference_distance: np.ndarray,
):
    safe_species = [
        _safe_species_label(
            species
        )
        for species in unique_species
    ]

    pair_list = sorted(
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

    pair_index = {
        pair: index
        for index, pair in enumerate(
            pair_list
        )
    }

    upair_list = sorted(
        {
            tuple(
                sorted(
                    pair
                )
            )
            for pair in pair_list
        }
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

    shell_k_rank = np.arange(
        shell_pair_id.size,
        dtype=np.int32,
    )

    shell_k_within_upair = _rank_shells_within_upairs(
        shell_upair_id,
        reference_distance,
    )

    return (
        pair_list,
        upair_list,
        shell_pair_id,
        shell_upair_id,
        shell_k_rank,
        shell_k_within_upair,
    )


def compile_stochastic_shell_topology(
    *,
    pair_topology: CompiledStochasticPairTopology,
    reference_parameters: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> CompiledStochasticShellTopology:
    """
    Compile finite and bulk rows into fixed stochastic shell descriptors.

    This operation is performed once per frozen stochastic ensemble.
    """
    reference_parameters = dict(
        reference_parameters
        or {}
    )

    start_time = time.perf_counter()

    finite_rows = pair_topology.finite_rows
    bulk_rows = pair_topology.bulk_rows

    finite_matrix = _descriptor_matrix(
        finite_rows
    )

    bulk_matrix = _descriptor_matrix(
        bulk_rows
    )

    finite_packed = _packed_view(
        finite_matrix
    )

    bulk_packed = _packed_view(
        bulk_matrix
    )

    finite_unique_start = time.perf_counter()

    (
        unique_finite,
        finite_first,
        finite_inverse,
    ) = _fast_unique_rows(
        finite_matrix,
        return_index=True,
        return_inverse=True,
    )

    finite_unique_seconds = (
        time.perf_counter()
        - finite_unique_start
    )

    print(
        "[STOCHASTIC COMPILED SHELLS] Finite grouping: "
        f"input_rows={finite_rows.number_of_rows:,}, "
        f"unique_rows={unique_finite.size:,}, "
        f"descriptor_size="
        f"{finite_matrix.nbytes / (1024.0 ** 3):.3f} GiB, "
        f"time={finite_unique_seconds:.3f} s",
        flush=True,
    )

    finite_weight = np.bincount(
        finite_inverse.astype(
            np.int64
        ),
        weights=np.asarray(
            finite_rows.row_weight,
            dtype=np.float64,
        ),
        minlength=unique_finite.size,
    ).astype(
        np.float64
    )

    bulk_unique_start = time.perf_counter()

    (
        unique_bulk,
        bulk_first,
        bulk_inverse,
    ) = _fast_unique_rows(
        bulk_matrix,
        return_index=True,
        return_inverse=True,
    )

    bulk_unique_seconds = (
        time.perf_counter()
        - bulk_unique_start
    )

    print(
        "[STOCHASTIC COMPILED SHELLS] Bulk grouping: "
        f"input_rows={bulk_rows.number_of_rows:,}, "
        f"unique_rows={unique_bulk.size:,}, "
        f"descriptor_size="
        f"{bulk_matrix.nbytes / (1024.0 ** 3):.3f} GiB, "
        f"time={bulk_unique_seconds:.3f} s",
        flush=True,
    )

    bulk_weight = np.bincount(
        bulk_inverse.astype(
            np.int64
        ),
        weights=np.asarray(
            bulk_rows.row_weight,
            dtype=np.float64,
        ),
        minlength=unique_bulk.size,
    ).astype(
        np.float64
    )

    bulk_order = np.argsort(
        unique_bulk,
        kind="mergesort",
    )

    sorted_bulk = unique_bulk[
        bulk_order
    ]

    sorted_bulk_weight = bulk_weight[
        bulk_order
    ]

    bulk_positions = np.searchsorted(
        sorted_bulk,
        unique_finite,
        side="left",
    )

    matched = (
        bulk_positions
        < sorted_bulk.size
    )

    matched_indices = np.flatnonzero(
        matched
    )

    if matched_indices.size:
        matched[
            matched_indices
        ] &= (
            sorted_bulk[
                bulk_positions[
                    matched_indices
                ]
            ]
            == unique_finite[
                matched_indices
            ]
        )

    retained = np.flatnonzero(
        matched
    )

    if retained.size <= 0:
        raise RuntimeError(
            "No finite stochastic descriptors match the stochastic bulk descriptors."
        )

    finite_rep = finite_first[
        retained
    ].astype(
        np.int64,
        copy=False,
    )

    source_alpha = finite_rows.source_alpha[
        finite_rep
    ].astype(
        np.int32,
        copy=False,
    )

    source_beta = finite_rows.source_beta[
        finite_rep
    ].astype(
        np.int32,
        copy=False,
    )

    delta_fractional = finite_rows.delta_fractional[
        finite_rep
    ].astype(
        np.float64,
        copy=False,
    )

    delta_a = finite_rows.delta_a[
        finite_rep
    ].astype(
        np.float64,
        copy=False,
    )

    delta_b = finite_rows.delta_b[
        finite_rep
    ].astype(
        np.float64,
        copy=False,
    )

    transition_coefficients = finite_rows.transition_coefficients[
        finite_rep
    ].astype(
        np.int32,
        copy=False,
    )

    finite_shell_weight = finite_weight[
        retained
    ].astype(
        np.float64,
        copy=False,
    )

    bulk_shell_weight = sorted_bulk_weight[
        bulk_positions[
            retained
        ]
    ].astype(
        np.float64,
        copy=False,
    )

    source_species_id = np.asarray(
        pair_topology.source_species_id,
        dtype=np.int16,
    )

    shell_alpha_id = source_species_id[
        source_alpha
    ]

    shell_beta_id = source_species_id[
        source_beta
    ]

    (
        reference_parent_lattice,
        reference_stacking_lattice,
        reference_stacking_inverse,
        reference_stacking_volume,
        _reference_h,
        _state_probability,
    ) = _effective_stacking_lattice(
        pair_topology,
        reference_parameters,
    )

    reference_rows = StochasticPairRows(
        source_alpha=source_alpha,
        source_beta=source_beta,
        delta_fractional=delta_fractional,
        delta_a=delta_a,
        delta_b=delta_b,
        transition_coefficients=transition_coefficients,
        reference_cart=np.zeros(
            (
                source_alpha.size,
                3,
            ),
            dtype=np.float64,
        ),
        reference_distance=np.zeros(
            source_alpha.size,
            dtype=np.float64,
        ),
        alpha_id=shell_alpha_id,
        beta_id=shell_beta_id,
        realization_index=np.zeros(
            source_alpha.size,
            dtype=np.int32,
        ),
        row_weight=finite_shell_weight,
    )

    state_du, state_dv, state_h = _state_vectors(
        pair_topology,
        reference_parameters,
    )

    (
        reference_cart,
        reference_distance,
    ) = evaluate_stochastic_pair_rows(
        reference_rows,
        lattice_matrix=reference_parent_lattice,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )

    shell_hkl, shell_direction = _hkl_and_direction(
        reference_cart,
        reference_stacking_lattice,
    )

    finite_center_weight = np.asarray(
        pair_topology.finite_center_species_weight,
        dtype=np.float64,
    )

    bulk_center_weight = np.asarray(
        pair_topology.bulk_center_species_weight,
        dtype=np.float64,
    )

    shell_mult_eff = np.zeros(
        finite_shell_weight.size,
        dtype=np.float32,
    )

    shell_bulk_mult_eff = np.zeros(
        bulk_shell_weight.size,
        dtype=np.float32,
    )

    shell_gamma = np.zeros(
        finite_shell_weight.size,
        dtype=np.float32,
    )

    for index in range(
        finite_shell_weight.size
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
                finite_shell_weight[
                    index
                ]
                / finite_denominator
            )

        if bulk_denominator > _EPSILON:
            bulk_effective = (
                bulk_shell_weight[
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

    (
        pair_list,
        upair_list,
        shell_pair_id,
        shell_upair_id,
        shell_k_rank,
        shell_k_within_upair,
    ) = _pair_metadata(
        unique_species=list(
            pair_topology.unique_species
        ),
        shell_alpha_id=shell_alpha_id,
        shell_beta_id=shell_beta_id,
        reference_distance=reference_distance,
    )

    order = np.argsort(
        shell_pair_id,
        kind="mergesort",
    )

    source_alpha = source_alpha[
        order
    ]

    source_beta = source_beta[
        order
    ]

    delta_fractional = delta_fractional[
        order
    ]

    delta_a = delta_a[
        order
    ]

    delta_b = delta_b[
        order
    ]

    transition_coefficients = transition_coefficients[
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

    reference_distance = reference_distance[
        order
    ]

    shell_pair_id = shell_pair_id[
        order
    ]

    shell_upair_id = shell_upair_id[
        order
    ]

    shell_k_rank = shell_k_rank[
        order
    ]

    shell_k_within_upair = shell_k_within_upair[
        order
    ]

    pair_counts = np.bincount(
        shell_pair_id,
        minlength=len(
            pair_list
        ),
    ).astype(
        np.int32
    )

    pair_offsets = np.zeros(
        len(
            pair_list
        )
        + 1,
        dtype=np.int32,
    )

    pair_offsets[
        1:
    ] = np.cumsum(
        pair_counts
    )

    pair_alpha_id = np.zeros(
        len(
            pair_list
        ),
        dtype=np.int16,
    )

    pair_beta_id = np.zeros(
        len(
            pair_list
        ),
        dtype=np.int16,
    )

    for pair_id in range(
        len(
            pair_list
        )
    ):
        first_row = int(
            pair_offsets[
                pair_id
            ]
        )

        stop_row = int(
            pair_offsets[
                pair_id
                + 1
            ]
        )

        if stop_row <= first_row:
            continue

        pair_alpha_id[
            pair_id
        ] = shell_alpha_id[
            first_row
        ]

        pair_beta_id[
            pair_id
        ] = shell_beta_id[
            first_row
        ]

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

    source_atom_count = int(
        len(
            pair_topology.geometry.source_structure.sites
        )
    )

    stacking_density = (
        float(
            source_atom_count
        )
        / max(
            reference_stacking_volume,
            _EPSILON,
        )
    )

    compiled = CompiledStochasticShellTopology(
        pair_topology=pair_topology,
        source_alpha=source_alpha,
        source_beta=source_beta,
        delta_fractional=delta_fractional,
        delta_a=delta_a,
        delta_b=delta_b,
        transition_coefficients=transition_coefficients,
        shell_mult_eff=shell_mult_eff,
        shell_bulk_mult_eff=shell_bulk_mult_eff,
        shell_gamma=shell_gamma,
        shell_alpha_id=shell_alpha_id,
        shell_beta_id=shell_beta_id,
        shell_hkl=shell_hkl,
        shell_direction=shell_direction,
        shell_pair_id=shell_pair_id,
        shell_upair_id=shell_upair_id,
        shell_k_rank=shell_k_rank,
        shell_k_within_upair=shell_k_within_upair,
        pair_offsets=pair_offsets,
        pair_alpha_id=pair_alpha_id,
        pair_beta_id=pair_beta_id,
        unique_species=list(
            pair_topology.unique_species
        ),
        pair_list=pair_list,
        upair_list=upair_list,
        strain_unique_dirs=strain_unique_dirs.astype(
            np.int32,
            copy=False,
        ),
        strain_inv_dirs=strain_inv_dirs.astype(
            np.int32,
            copy=False,
        ),
        reference_distance=reference_distance,
        stacking_lattice_matrix_reference=reference_stacking_lattice,
        stacking_inverse_lattice_matrix_reference=reference_stacking_inverse,
        stacking_volume_reference=float(
            reference_stacking_volume
        ),
        stacking_bulk_number_density_reference=float(
            stacking_density
        ),
        finite_atom_count_average=float(
            np.sum(
                finite_center_weight
            )
        ),
        bulk_atom_count_reference=source_atom_count,
    )

    print(
        "[STOCHASTIC COMPILED SHELLS] "
        f"finite_rows={finite_rows.number_of_rows:,}, "
        f"bulk_rows={bulk_rows.number_of_rows:,}, "
        f"fixed_shells={compiled.number_of_shells:,}, "
        f"states={compiled.transition_coefficients.shape[1]}, "
        f"time={time.perf_counter() - start_time:.3f} s",
        flush=True,
    )

    return compiled


def evaluate_stochastic_shell_geometry(
    topology: CompiledStochasticShellTopology,
    parameters: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> Dict[
    str,
    Any,
]:
    """
    Evaluate Cartesian shell geometry from fixed descriptors.

    No topology reconstruction and no shell regrouping occur here.
    """
    parameters = dict(
        parameters
        or {}
    )

    pair_topology = topology.pair_topology

    (
        parent_lattice,
        stacking_lattice,
        stacking_inverse,
        stacking_volume,
        effective_h,
        state_probability,
    ) = _effective_stacking_lattice(
        pair_topology,
        parameters,
    )

    state_du, state_dv, state_h = _state_vectors(
        pair_topology,
        parameters,
    )

    rows = StochasticPairRows(
        source_alpha=topology.source_alpha,
        source_beta=topology.source_beta,
        delta_fractional=topology.delta_fractional,
        delta_a=topology.delta_a,
        delta_b=topology.delta_b,
        transition_coefficients=topology.transition_coefficients,
        reference_cart=np.zeros(
            (
                topology.number_of_shells,
                3,
            ),
            dtype=np.float64,
        ),
        reference_distance=np.zeros(
            topology.number_of_shells,
            dtype=np.float64,
        ),
        alpha_id=topology.shell_alpha_id,
        beta_id=topology.shell_beta_id,
        realization_index=np.zeros(
            topology.number_of_shells,
            dtype=np.int32,
        ),
        row_weight=np.ones(
            topology.number_of_shells,
            dtype=np.float64,
        ),
    )

    shell_cart, shell_distance = evaluate_stochastic_pair_rows(
        rows,
        lattice_matrix=parent_lattice,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )

    shell_active = (
        np.isfinite(
            shell_distance
        )
        & (
            shell_distance
            > 1.0e-8
        )
        & (
            shell_distance
            <= float(
                pair_topology.pair_cutoff
            )
            + 1.0e-10
        )
    )

    shell_frac = (
        shell_cart
        @ stacking_inverse
    ).astype(
        np.float32
    )

    source_atom_count = int(
        len(
            pair_topology.geometry.source_structure.sites
        )
    )

    number_density = (
        float(
            source_atom_count
        )
        / max(
            stacking_volume,
            _EPSILON,
        )
    )

    return {
        "shell_cart": np.asarray(
            shell_cart,
            dtype=np.float32,
        ),
        "shell_frac": shell_frac,
        "shell_distance": np.asarray(
            shell_distance,
            dtype=np.float64,
        ),
        "shell_active": np.asarray(
            shell_active,
            dtype=bool,
        ),
        "shell_mult_eff": np.asarray(
            topology.shell_mult_eff,
            dtype=np.float32,
        ),
        "shell_bulk_mult_eff": np.asarray(
            topology.shell_bulk_mult_eff,
            dtype=np.float32,
        ),
        "shell_gamma": np.asarray(
            topology.shell_gamma,
            dtype=np.float32,
        ),
        "stacking_lattice_matrix": np.asarray(
            stacking_lattice,
            dtype=np.float64,
        ),
        "stacking_inverse_lattice_matrix": np.asarray(
            stacking_inverse,
            dtype=np.float64,
        ),
        "stacking_volume": float(
            stacking_volume
        ),
        "stacking_bulk_number_density": float(
            number_density
        ),
        "stacking_spacing": float(
            effective_h
        ),
        "stacking_period_height": float(
            effective_h
        ),
        "state_probability": np.asarray(
            state_probability,
            dtype=np.float64,
        ),
    }


def initial_stochastic_shell_table(
    topology: CompiledStochasticShellTopology,
    reference_parameters: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> Dict[
    str,
    Any,
]:
    """
    Build the initial PDFCalculator-compatible shell table.
    """
    evaluated = evaluate_stochastic_shell_geometry(
        topology,
        reference_parameters,
    )

    shell_mult = np.maximum(
        0,
        np.rint(
            topology.shell_mult_eff
        ).astype(
            np.int64
        ),
    ).astype(
        np.int32
    )

    shell_bulk_mult = np.maximum(
        0,
        np.rint(
            topology.shell_bulk_mult_eff
        ).astype(
            np.int64
        ),
    ).astype(
        np.int32
    )

    return {
        "geometry_mode": "stochastic_ensemble",
        "shell_frac": evaluated[
            "shell_frac"
        ],
        "shell_cart": evaluated[
            "shell_cart"
        ],
        "shell_mult": shell_mult,
        "shell_bulk_mult": shell_bulk_mult,
        "shell_mult_eff": topology.shell_mult_eff,
        "shell_finite_mult_eff": topology.shell_mult_eff.copy(),
        "shell_bulk_mult_eff": topology.shell_bulk_mult_eff,
        "shell_gamma": topology.shell_gamma,
        "shell_pair_id": topology.shell_pair_id,
        "shell_alpha_id": topology.shell_alpha_id,
        "shell_beta_id": topology.shell_beta_id,
        "shell_hkl": topology.shell_hkl,
        "shell_direction": topology.shell_direction,
        "shell_k_rank": topology.shell_k_rank,
        "shell_upair_id": topology.shell_upair_id,
        "shell_k_within_upair": topology.shell_k_within_upair,
        "pair_offsets": topology.pair_offsets,
        "pair_alpha_id": topology.pair_alpha_id,
        "pair_beta_id": topology.pair_beta_id,
        "unique_species": topology.unique_species,
        "pair_list": topology.pair_list,
        "upair_list": topology.upair_list,
        "strain_unique_dirs": topology.strain_unique_dirs,
        "strain_inv_dirs": topology.strain_inv_dirs,
        "stacking_lattice_matrix": evaluated[
            "stacking_lattice_matrix"
        ],
        "stacking_inverse_lattice_matrix": evaluated[
            "stacking_inverse_lattice_matrix"
        ],
        "stacking_volume": float(
            evaluated[
                "stacking_volume"
            ]
        ),
        "stacking_bulk_number_density": float(
            evaluated[
                "stacking_bulk_number_density"
            ]
        ),
        "bulk_atom_count": int(
            topology.bulk_atom_count_reference
        ),
        "stacking_nominal_spacing": float(
            evaluated[
                "stacking_spacing"
            ]
        ),
        "stacking_repeat_period": 1.0,
        "stacking_period_normal_height": float(
            evaluated[
                "stacking_period_height"
            ]
        ),
        "finite_atom_count": int(
            round(
                topology.finite_atom_count_average
            )
        ),
        "number_of_layer_positions": int(
            topology.pair_topology.geometry.number_of_layers
        ),
        "stochastic_model": str(
            topology.pair_topology.geometry.stack_spec.model
        ),
    }


print(
    "[COMPILED STOCHASTIC SHELLS] Module loaded:",
    __file__,
    flush=True,
)


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
