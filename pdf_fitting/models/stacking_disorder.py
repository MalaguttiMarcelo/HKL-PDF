from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np


STATE_A = np.int8(0)
STATE_B = np.int8(1)


def _as_probability(
    value,
    default: float,
) -> float:
    try:
        value = float(
            value
        )
    except Exception:
        value = float(
            default
        )

    return float(
        min(
            1.0,
            max(
                0.0,
                value,
            ),
        )
    )


def p_same_state(
    p_alternate: float,
) -> float:
    return float(
        1.0
        - _as_probability(
            p_alternate,
            1.0,
        )
    )


def state_name(
    state: int,
) -> str:
    return (
        "B"
        if int(state) == int(STATE_B)
        else "A"
    )


@dataclass(frozen=True)
class CorrelatedABPathSet:
    """
    Frozen random numbers and state paths for TOPAS-like A/B disorder.

    ``uniform_randoms`` is the authoritative frozen stochastic basis.

    For one realization with N layers, there are N - 1 transitions.

    At transition n:
        - current state emits its own vector;
        - random number u[n] chooses whether the next state alternates.

    Therefore changing p_alternate does not regenerate random numbers.
    It only changes the threshold applied to the same frozen uniforms.
    """

    number_of_layers: int
    ensemble_samples: int
    seed: int
    start_state: int

    uniform_randoms: np.ndarray
    states: np.ndarray

    p_alternate: float

    def summary_dict(
        self,
    ) -> Dict[str, Any]:
        transitions_per_realization = max(
            0,
            int(
                self.number_of_layers
            )
            - 1,
        )

        return {
            "number_of_layers": int(
                self.number_of_layers
            ),
            "ensemble_samples": int(
                self.ensemble_samples
            ),
            "seed": int(
                self.seed
            ),
            "start_state": state_name(
                self.start_state
            ),
            "p_alternate": float(
                self.p_alternate
            ),
            "p_same_state": p_same_state(
                self.p_alternate
            ),
            "transitions_per_realization": (
                transitions_per_realization
            ),
            "total_transitions": int(
                self.ensemble_samples
                * transitions_per_realization
            ),
        }


def frozen_uniform_randoms(
    *,
    number_of_layers: int,
    ensemble_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Return reproducible uniforms of shape:

        (ensemble_samples, number_of_layers - 1)
    """
    number_of_layers = max(
        1,
        int(
            number_of_layers
        ),
    )

    ensemble_samples = max(
        1,
        int(
            ensemble_samples
        ),
    )

    number_of_transitions = max(
        0,
        number_of_layers
        - 1,
    )

    generator = np.random.default_rng(
        int(
            seed
        )
    )

    return generator.random(
        (
            ensemble_samples,
            number_of_transitions,
        ),
        dtype=np.float64,
    )


def states_from_uniforms(
    uniform_randoms: np.ndarray,
    *,
    p_alternate: float,
    start_state: str = "A",
) -> np.ndarray:
    """
    Convert frozen uniforms into TOPAS-compatible A/B state paths.

    p_alternate = 1:
        A B A B ...

    p_alternate = 0:
        A A A A ...
        or B B B B ...

    p_alternate = 0.5:
        random next-state selection.
    """
    uniform_randoms = np.asarray(
        uniform_randoms,
        dtype=np.float64,
    )

    if uniform_randoms.ndim != 2:
        raise ValueError(
            "uniform_randoms must have shape "
            "(ensemble_samples, number_of_transitions)."
        )

    ensemble_samples = int(
        uniform_randoms.shape[0]
    )

    number_of_transitions = int(
        uniform_randoms.shape[1]
    )

    number_of_layers = (
        number_of_transitions
        + 1
    )

    start_text = str(
        start_state
        or "A"
    ).strip().upper()

    initial_state = (
        STATE_B
        if start_text == "B"
        else STATE_A
    )

    p_alternate = _as_probability(
        p_alternate,
        1.0,
    )

    states = np.empty(
        (
            ensemble_samples,
            number_of_layers,
        ),
        dtype=np.int8,
    )

    states[
        :,
        0
    ] = initial_state

    for transition_index in range(
        number_of_transitions
    ):
        current = states[
            :,
            transition_index
        ]

        alternate = (
            uniform_randoms[
                :,
                transition_index
            ]
            < p_alternate
        )

        next_state = current.copy()

        next_state[
            alternate
        ] = (
            1
            - current[
                alternate
            ]
        )

        states[
            :,
            transition_index
            + 1
        ] = next_state

    return states


def generate_correlated_ab_paths(
    *,
    number_of_layers: int,
    ensemble_samples: int,
    p_alternate: float,
    seed: int,
    start_state: str = "A",
    uniform_randoms: Optional[np.ndarray] = None,
) -> CorrelatedABPathSet:
    """
    Build frozen random paths.

    If ``uniform_randoms`` is supplied, it is reused. This is required for
    stable comparison across a P_alt grid and for continuous vector refinement.
    """
    number_of_layers = max(
        1,
        int(
            number_of_layers
        ),
    )

    ensemble_samples = max(
        1,
        int(
            ensemble_samples
        ),
    )

    number_of_transitions = max(
        0,
        number_of_layers
        - 1,
    )

    if uniform_randoms is None:
        uniform_randoms = frozen_uniform_randoms(
            number_of_layers=number_of_layers,
            ensemble_samples=ensemble_samples,
            seed=int(
                seed
            ),
        )

    uniform_randoms = np.asarray(
        uniform_randoms,
        dtype=np.float64,
    )

    expected_shape = (
        ensemble_samples,
        number_of_transitions,
    )

    if uniform_randoms.shape != expected_shape:
        raise ValueError(
            "uniform_randoms shape mismatch: "
            f"expected {expected_shape}, "
            f"received {uniform_randoms.shape}."
        )

    p_alternate = _as_probability(
        p_alternate,
        1.0,
    )

    states = states_from_uniforms(
        uniform_randoms,
        p_alternate=p_alternate,
        start_state=start_state,
    )

    start_text = str(
        start_state
        or "A"
    ).strip().upper()

    initial_state = (
        STATE_B
        if start_text == "B"
        else STATE_A
    )

    return CorrelatedABPathSet(
        number_of_layers=number_of_layers,
        ensemble_samples=ensemble_samples,
        seed=int(
            seed
        ),
        start_state=int(
            initial_state
        ),
        uniform_randoms=uniform_randoms,
        states=states,
        p_alternate=p_alternate,
    )


def state_transition_statistics(
    states: np.ndarray,
) -> Dict[str, int]:
    """
    Count A->A, A->B, B->A and B->B transitions for diagnostics.
    """
    states = np.asarray(
        states,
        dtype=np.int8,
    )

    out = {
        "A_to_A": 0,
        "A_to_B": 0,
        "B_to_A": 0,
        "B_to_B": 0,
    }

    if (
        states.ndim != 2
        or states.shape[1] <= 1
    ):
        return out

    current = states[
        :,
        :-1
    ].reshape(
        -1
    )

    next_state = states[
        :,
        1:
    ].reshape(
        -1
    )

    out[
        "A_to_A"
    ] = int(
        np.count_nonzero(
            (
                current
                == STATE_A
            )
            & (
                next_state
                == STATE_A
            )
        )
    )

    out[
        "A_to_B"
    ] = int(
        np.count_nonzero(
            (
                current
                == STATE_A
            )
            & (
                next_state
                == STATE_B
            )
        )
    )

    out[
        "B_to_A"
    ] = int(
        np.count_nonzero(
            (
                current
                == STATE_B
            )
            & (
                next_state
                == STATE_A
            )
        )
    )

    out[
        "B_to_B"
    ] = int(
        np.count_nonzero(
            (
                current
                == STATE_B
            )
            & (
                next_state
                == STATE_B
            )
        )
    )

    return out


def p_alternate_grid(
    *,
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    """
    Inclusive grid for P_alt.

    This is intentionally separate from continuous least-squares refinement.
    """
    minimum = _as_probability(
        minimum,
        0.0,
    )

    maximum = _as_probability(
        maximum,
        1.0,
    )

    if maximum < minimum:
        minimum, maximum = (
            maximum,
            minimum,
        )

    step = max(
        float(
            step
        ),
        1.0e-6,
    )

    count = int(
        np.floor(
            (
                maximum
                - minimum
            )
            / step
            + 0.5
        )
    ) + 1

    values = (
        minimum
        + step
        * np.arange(
            count,
            dtype=float,
        )
    )

    values = values[
        values
        <= maximum
        + 1.0e-10
    ]

    if (
        values.size == 0
        or abs(
            float(
                values[
                    -1
                ]
            )
            - maximum
        )
        > 1.0e-8
    ):
        values = np.concatenate(
            (
                values,
                np.asarray(
                    [
                        maximum
                    ],
                    dtype=float,
                ),
            )
        )

    return np.unique(
        np.round(
            values,
            12,
        )
    )

@dataclass(frozen=True)
class GenericMarkovPathSet:
    """
    Frozen arbitrary-state Markov paths.

    states has shape:

        (ensemble_samples, number_of_layers)
    """

    number_of_layers: int
    ensemble_samples: int
    seed: int
    start_state: int

    uniform_randoms: np.ndarray
    states: np.ndarray

    transition_matrix: np.ndarray


def normalize_generic_markov_matrix(
    matrix,
    *,
    number_of_states: int,
) -> np.ndarray:
    """
    Convert arbitrary user input to a valid row-stochastic matrix.
    """
    number_of_states = max(
        1,
        int(
            number_of_states
        ),
    )

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.shape != (
        number_of_states,
        number_of_states,
    ):
        matrix = np.eye(
            number_of_states,
            dtype=float,
        )

    matrix = np.maximum(
        matrix,
        0.0,
    )

    row_sums = np.sum(
        matrix,
        axis=1,
    )

    for state_index in range(
        number_of_states
    ):
        if row_sums[
            state_index
        ] <= 1.0e-12:
            matrix[
                state_index,
                :
            ] = 0.0

            matrix[
                state_index,
                state_index,
            ] = 1.0

    row_sums = np.sum(
        matrix,
        axis=1,
        keepdims=True,
    )

    matrix /= np.maximum(
        row_sums,
        1.0e-12,
    )

    matrix[
        :,
        -1
    ] += (
        1.0
        - np.sum(
            matrix,
            axis=1,
        )
    )

    return np.asarray(
        matrix,
        dtype=np.float64,
    )

def generic_markov_matrix_grid(
    *,
    transition_matrix,
    probability_controls,
    maximum_candidates: int = 100000,
    row_sum_tolerance: float = 1.0e-8,
) -> List[np.ndarray]:
    """
    Generate valid row-stochastic transition matrices from per-cell grids.

    Each enabled matrix cell is scanned over:

        minimum
        maximum
        step

    Disabled cells remain fixed at their current matrix value.

    Only combinations for which each row sums to 1 are retained.
    """
    transition_matrix = np.asarray(
        transition_matrix,
        dtype=np.float64,
    )

    if (
        transition_matrix.ndim != 2
        or transition_matrix.shape[0]
        != transition_matrix.shape[1]
    ):
        raise ValueError(
            "The generic Markov transition matrix must be square."
        )

    number_of_states = int(
        transition_matrix.shape[0]
    )

    transition_matrix = normalize_generic_markov_matrix(
        transition_matrix,
        number_of_states=number_of_states,
    )

    from pdf_fitting.models.layer_stacking import (
        normalize_markov_probability_controls,
    )

    controls = normalize_markov_probability_controls(
        probability_controls,
        number_of_states=number_of_states,
        transition_matrix=transition_matrix.tolist(),
    )

    maximum_candidates = max(
        1,
        int(
            maximum_candidates
        ),
    )

    def inclusive_values(
        minimum: float,
        maximum: float,
        step: float,
    ) -> np.ndarray:
        minimum = float(
            minimum
        )

        maximum = float(
            maximum
        )

        if maximum < minimum:
            minimum, maximum = (
                maximum,
                minimum,
            )

        step = max(
            float(
                step
            ),
            1.0e-12,
        )

        count = int(
            np.floor(
                (
                    maximum
                    - minimum
                )
                / step
                + 0.5
            )
        ) + 1

        values = (
            minimum
            + step
            * np.arange(
                count,
                dtype=np.float64,
            )
        )

        values = values[
            values
            <= maximum
            + 1.0e-12
        ]

        if (
            values.size == 0
            or abs(
                float(
                    values[
                        -1
                    ]
                )
                - maximum
            )
            > 1.0e-10
        ):
            values = np.concatenate(
                (
                    values,
                    np.asarray(
                        [
                            maximum
                        ],
                        dtype=np.float64,
                    ),
                )
            )

        return np.unique(
            np.round(
                values,
                12,
            )
        )

    row_candidates = []

    for source_index in range(
        number_of_states
    ):
        value_sets = []

        for target_index in range(
            number_of_states
        ):
            control = controls[
                source_index
            ][
                target_index
            ]

            if bool(
                control.get(
                    "enabled",
                    False,
                )
            ):
                values = inclusive_values(
                    control.get(
                        "minimum",
                        transition_matrix[
                            source_index,
                            target_index,
                        ],
                    ),
                    control.get(
                        "maximum",
                        transition_matrix[
                            source_index,
                            target_index,
                        ],
                    ),
                    control.get(
                        "step",
                        0.05,
                    ),
                )
            else:
                values = np.asarray(
                    [
                        transition_matrix[
                            source_index,
                            target_index,
                        ]
                    ],
                    dtype=np.float64,
                )

            value_sets.append(
                values
            )

        current_rows = [
            np.zeros(
                number_of_states,
                dtype=np.float64,
            )
        ]

        for target_index, values in enumerate(
            value_sets
        ):
            expanded_rows = []

            for partial_row in current_rows:
                for probability in values:
                    candidate_row = partial_row.copy()

                    candidate_row[
                        target_index
                    ] = float(
                        probability
                    )

                    expanded_rows.append(
                        candidate_row
                    )

                    if len(
                        expanded_rows
                    ) > maximum_candidates:
                        raise ValueError(
                            "The generic Markov probability grid is too large. "
                            "Reduce the number of scanned probabilities or "
                            "increase their step sizes."
                        )

            current_rows = expanded_rows

        valid_rows = []

        for candidate_row in current_rows:
            if abs(
                float(
                    np.sum(
                        candidate_row
                    )
                )
                - 1.0
            ) <= float(
                row_sum_tolerance
            ):
                valid_rows.append(
                    candidate_row
                )

        if not valid_rows:
            raise ValueError(
                "No valid probability combinations were generated for "
                f"Markov row {source_index}. Each row must sum to 1. "
                "Check the fixed probabilities and the selected minimum, "
                "maximum, and step values."
            )

        row_candidates.append(
            valid_rows
        )

    matrices = [
        np.zeros(
            (
                number_of_states,
                number_of_states,
            ),
            dtype=np.float64,
        )
    ]

    for source_index, candidate_rows in enumerate(
        row_candidates
    ):
        expanded_matrices = []

        for partial_matrix in matrices:
            for candidate_row in candidate_rows:
                candidate_matrix = partial_matrix.copy()

                candidate_matrix[
                    source_index,
                    :
                ] = candidate_row

                expanded_matrices.append(
                    candidate_matrix
                )

                if len(
                    expanded_matrices
                ) > maximum_candidates:
                    raise ValueError(
                        "The complete generic Markov probability grid is too "
                        "large. Reduce the number of enabled probability grids."
                    )

        matrices = expanded_matrices

    unique_matrices = {}

    for matrix in matrices:
        key = tuple(
            np.round(
                matrix.reshape(
                    -1
                ),
                12,
            )
        )

        unique_matrices[
            key
        ] = matrix

    return list(
        unique_matrices.values()
    )


def generate_generic_markov_paths(
    *,
    number_of_layers: int,
    ensemble_samples: int,
    transition_matrix,
    seed: int,
    start_state: int = 0,
    uniform_randoms: Optional[np.ndarray] = None,
) -> GenericMarkovPathSet:
    """
    Generate reproducible paths for an arbitrary N-state Markov model.

    The current state emits its vector, then the matrix row for the current
    state selects the next state.
    """
    number_of_layers = max(
        1,
        int(
            number_of_layers
        ),
    )

    ensemble_samples = max(
        1,
        int(
            ensemble_samples
        ),
    )

    transition_matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    if transition_matrix.ndim != 2:
        raise ValueError(
            "transition_matrix must be a two-dimensional array."
        )

    if (
        transition_matrix.shape[0]
        != transition_matrix.shape[1]
    ):
        raise ValueError(
            "transition_matrix must be square."
        )

    number_of_states = int(
        transition_matrix.shape[0]
    )

    if number_of_states <= 0:
        raise ValueError(
            "transition_matrix must contain at least one state."
        )

    transition_matrix = normalize_generic_markov_matrix(
        transition_matrix,
        number_of_states=number_of_states,
    )

    start_state = max(
        0,
        min(
            int(
                start_state
            ),
            number_of_states - 1,
        ),
    )

    number_of_transitions = max(
        0,
        number_of_layers - 1,
    )

    expected_shape = (
        ensemble_samples,
        number_of_transitions,
    )

    if uniform_randoms is None:
        generator = np.random.default_rng(
            int(
                seed
            )
        )

        uniform_randoms = generator.random(
            expected_shape,
            dtype=np.float64,
        )

    uniform_randoms = np.asarray(
        uniform_randoms,
        dtype=np.float64,
    )

    if uniform_randoms.shape != expected_shape:
        raise ValueError(
            "uniform_randoms shape mismatch: "
            f"expected {expected_shape}, "
            f"received {uniform_randoms.shape}."
        )

    states = np.empty(
        (
            ensemble_samples,
            number_of_layers,
        ),
        dtype=np.int16,
    )

    states[
        :,
        0
    ] = np.int16(
        start_state
    )

    cumulative = np.cumsum(
        transition_matrix,
        axis=1,
    )

    cumulative[
        :,
        -1
    ] = 1.0

    for transition_index in range(
        number_of_transitions
    ):
        current_states = states[
            :,
            transition_index
        ].astype(
            np.int64,
            copy=False,
        )

        random_values = uniform_randoms[
            :,
            transition_index
        ]

        next_states = np.empty(
            ensemble_samples,
            dtype=np.int16,
        )

        for current_state in range(
            number_of_states
        ):
            mask = (
                current_states
                == current_state
            )

            if not np.any(
                mask
            ):
                continue

            next_states[
                mask
            ] = np.searchsorted(
                cumulative[
                    current_state
                ],
                random_values[
                    mask
                ],
                side="right",
            ).astype(
                np.int16,
                copy=False,
            )

        states[
            :,
            transition_index + 1
        ] = next_states

    return GenericMarkovPathSet(
        number_of_layers=number_of_layers,
        ensemble_samples=ensemble_samples,
        seed=int(
            seed
        ),
        start_state=start_state,
        uniform_randoms=uniform_randoms,
        states=states,
        transition_matrix=transition_matrix,
    )



def path_has_valid_closure(
    states: np.ndarray,
    transition_matrix: np.ndarray,
) -> bool:
    states = np.asarray(
        states,
        dtype=np.int32,
    ).reshape(
        -1
    )

    transition_matrix = np.asarray(
        transition_matrix,
        dtype=np.float64,
    )

    if states.size <= 0:
        return False

    initial_state = int(
        states[
            0
        ]
    )

    final_state = int(
        states[
            -1
        ]
    )

    return bool(
        transition_matrix[
            final_state,
            initial_state,
        ]
        > 0.0
    )


def correlated_ab_transition_matrix(
    p_alternate: float,
) -> np.ndarray:
    p_alternate = min(
        1.0,
        max(
            0.0,
            float(
                p_alternate
            ),
        ),
    )

    p_same = (
        1.0
        - p_alternate
    )

    return np.asarray(
        [
            [
                p_same,
                p_alternate,
            ],
            [
                p_alternate,
                p_same,
            ],
        ],
        dtype=np.float64,
    )

def generate_cyclic_generic_markov_states(
    *,
    number_of_layers: int,
    ensemble_samples: int,
    transition_matrix,
    seed: int,
    start_state: int = 0,
    regenerate_invalid: bool = True,
    maximum_attempts_per_realization: int = 10000,
) -> np.ndarray:
    number_of_layers = max(
        2,
        int(
            number_of_layers
        ),
    )

    ensemble_samples = max(
        1,
        int(
            ensemble_samples
        ),
    )

    transition_matrix = np.asarray(
        transition_matrix,
        dtype=np.float64,
    )

    output = []

    for realization_index in range(
        ensemble_samples
    ):
        accepted = None

        for attempt_index in range(
            maximum_attempts_per_realization
        ):
            current_seed = (
                int(
                    seed
                )
                + realization_index
                * 1_000_003
                + attempt_index
            )

            path_set = generate_generic_markov_paths(
                number_of_layers=number_of_layers,
                ensemble_samples=1,
                transition_matrix=transition_matrix,
                seed=current_seed,
                start_state=start_state,
            )

            candidate = np.asarray(
                path_set.states[
                    0
                ],
                dtype=np.int16,
            )

            if path_has_valid_closure(
                candidate,
                transition_matrix,
            ):
                accepted = candidate
                break

            if not regenerate_invalid:
                raise RuntimeError(
                    "Generated Markov path has a forbidden periodic "
                    "closure transition."
                )

        if accepted is None:
            raise RuntimeError(
                "Could not generate a cyclic Markov path after "
                f"{maximum_attempts_per_realization} attempts."
            )

        output.append(
            accepted
        )

    return np.asarray(
        output,
        dtype=np.int16,
    )


def generate_cyclic_correlated_ab_states(
    *,
    number_of_layers: int,
    ensemble_samples: int,
    p_alternate: float,
    seed: int,
    start_state: str = "A",
    regenerate_invalid: bool = True,
    maximum_attempts_per_realization: int = 10000,
) -> np.ndarray:
    transition_matrix = correlated_ab_transition_matrix(
        p_alternate
    )

    if (
        abs(
            float(
                p_alternate
            )
            - 1.0
        )
        <= 1.0e-12
        and int(
            number_of_layers
        )
        % 2
        != 0
    ):
        raise ValueError(
            "Perfect alternating A/B bulk stacking requires an even "
            "number of realization layers."
        )
    
    start_index = (
        1
        if str(
            start_state
            or "A"
        ).strip().upper()
        == "B"
        else 0
    )

    return generate_cyclic_generic_markov_states(
        number_of_layers=number_of_layers,
        ensemble_samples=ensemble_samples,
        transition_matrix=transition_matrix,
        seed=seed,
        start_state=start_index,
        regenerate_invalid=regenerate_invalid,
        maximum_attempts_per_realization=(
            maximum_attempts_per_realization
        ),
    )

