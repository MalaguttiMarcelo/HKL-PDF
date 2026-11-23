from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import os
import re
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np

from pdf_fitting.models.fast_unique import unique_rows as _fast_unique_rows
from scipy.spatial import cKDTree

from pdf_fitting.models.compiled_stochastic_geometry import (
    CompiledStochasticEnsembleGeometry,
    CompiledStochasticLayerDescriptor,
    EvaluatedStochasticRealizationGeometry,
    _generic_state_vectors,
    _layer_descriptors_from_states,
    _reference_layer_origins,
)

from pdf_fitting.models.compiled_stacked_geometry import (
    _centered_source_layer_cartesian,
    _lattice_matrix_from_parameters,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    layer_basis,
)


_EPSILON = 1.0e-12
_DESCRIPTOR_SCALE = 1.0e10


def _available_memory_bytes() -> Optional[int]:
    """
    Return currently available physical memory.

    Windows uses GlobalMemoryStatusEx without requiring psutil. Other
    platforms use psutil when it is available.
    """
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatusEx(
                ctypes.Structure
            ):
                _fields_ = [
                    (
                        "dwLength",
                        ctypes.c_ulong,
                    ),
                    (
                        "dwMemoryLoad",
                        ctypes.c_ulong,
                    ),
                    (
                        "ullTotalPhys",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullAvailPhys",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullTotalPageFile",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullAvailPageFile",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullTotalVirtual",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullAvailVirtual",
                        ctypes.c_ulonglong,
                    ),
                    (
                        "ullAvailExtendedVirtual",
                        ctypes.c_ulonglong,
                    ),
                ]

            status = MemoryStatusEx()

            status.dwLength = ctypes.sizeof(
                MemoryStatusEx
            )

            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(
                    status
                )
            )

            if success:
                return int(
                    status.ullAvailPhys
                )

        except Exception:
            pass

    try:
        import importlib

        psutil = importlib.import_module(
            "psutil"
        )

        return int(
            psutil.virtual_memory().available
        )

    except Exception:
        return None

    
def _resolve_stochastic_topology_workers(
    requested_workers: int,
    *,
    number_of_jobs: int,
    estimated_bytes_per_worker: int,
) -> int:
    """
    Select the number of concurrent stochastic-topology jobs.

    requested_workers
    -----------------
    -1
        Force all available logical CPUs.

    0
        Automatic resource selection. Use all available CPUs when memory
        permits, otherwise reduce concurrency to avoid paging.

    positive integer
        Use at most that many workers.

    Notes
    -----
    Each worker can temporarily hold:

        pair-index arrays;
        directed pair descriptors;
        Cartesian pair vectors;
        descriptor matrices;
        np.unique sorting buffers.

    Therefore one worker may require several GB for a large crystallite.
    """
    logical_cpus = max(
        1,
        int(
            os.cpu_count()
            or 1
        ),
    )

    number_of_jobs = max(
        1,
        int(
            number_of_jobs
        ),
    )

    maximum_cpu_workers = min(
        logical_cpus,
        number_of_jobs,
    )

    requested_workers = int(
        requested_workers
    )

    if requested_workers < 0:
        return maximum_cpu_workers

    if requested_workers > 0:
        return max(
            1,
            min(
                requested_workers,
                maximum_cpu_workers,
            ),
        )

    available_memory = _available_memory_bytes()

    if (
        available_memory is None
        or estimated_bytes_per_worker <= 0
    ):
        return maximum_cpu_workers

    # Leave 25% of currently available RAM for:
    #
    #     Qt and plotting;
    #     the source structure;
    #     already completed realization arrays;
    #     final concatenation;
    #     NumPy sorting temporaries.
    usable_memory = int(
        0.75
        * float(
            available_memory
        )
    )

    memory_workers = max(
        1,
        int(
            usable_memory
            // int(
                estimated_bytes_per_worker
            )
        ),
    )

    return max(
        1,
        min(
            maximum_cpu_workers,
            memory_workers,
        ),
    )

def _estimate_topology_worker_bytes(
    *,
    atom_count: int,
    observed_directed_rows: Optional[int] = None,
) -> int:
    """
    Estimate peak memory required by one realization worker.

    If an observed row count is available, use it. Otherwise use a
    conservative atom-count estimate until the first realization completes.
    """
    atom_count = max(
        1,
        int(
            atom_count
        ),
    )

    if observed_directed_rows is not None:
        estimated_rows = max(
            1,
            int(
                observed_directed_rows
            ),
        )

    else:
        # Conservative initial estimate for a dense finite-pair topology.
        # This is only used to select concurrency before the first job.
        estimated_rows = max(
            atom_count
            * 1000,
            1_000_000,
        )

    # Approximately:
    #
    #     persistent row arrays       ~105 bytes/row
    #     descriptor matrix           ~72 bytes/row for two states
    #     sorting/index temporaries    ~40-80 bytes/row
    #
    # Use 256 bytes per directed row as a conservative peak estimate.
    bytes_per_row = 256

    minimum_worker_bytes = 512 * 1024 * 1024

    return max(
        minimum_worker_bytes,
        int(
            estimated_rows
            * bytes_per_row
        ),
    )


@dataclass
class StochasticPairRows:
    """
    Directed stochastic pair descriptors.

    Each row represents the directed pair vector:

        beta - alpha

    The pair vector is evaluated as:

        delta_fractional @ parent_lattice
        + delta_a * parent_a
        + delta_b * parent_b
        + sum_s transition_coefficients[s] * state_vector[s]

    where each state vector is:

        state_du[s] * parent_a
        + state_dv[s] * parent_b
        + state_h[s] * stack_normal

    The coefficient matrix has shape:

        (
            number_of_pair_rows,
            number_of_markov_states,
        )

    Correlated A/B stacking is the two-state special case.
    """

    source_alpha: np.ndarray
    source_beta: np.ndarray

    delta_fractional: np.ndarray
    delta_a: np.ndarray
    delta_b: np.ndarray

    transition_coefficients: np.ndarray

    reference_cart: np.ndarray
    reference_distance: np.ndarray

    alpha_id: np.ndarray
    beta_id: np.ndarray

    realization_index: np.ndarray

    row_weight: np.ndarray

    @property
    def number_of_rows(
        self,
    ) -> int:
        return int(
            self.source_alpha.size
        )


@dataclass
class CompiledStochasticPairTopology:
    """
    Compiled pair topology for a frozen stochastic ensemble.

    finite_rows:
        Directed pair rows from finite particles.

    bulk_rows:
        Directed pair rows from long open stochastic stacks using only
        central/interior layer atoms as bulk centers.

    The bulk stack deliberately has no periodic closure. This avoids adding
    nonphysical correlations between the final and initial random layers.
    """

    geometry: CompiledStochasticEnsembleGeometry

    pair_cutoff: float
    topology_cutoff: float

    finite_rows: StochasticPairRows
    bulk_rows: StochasticPairRows

    unique_species: List[str]
    source_species_id: np.ndarray

    finite_center_species_weight: np.ndarray
    bulk_center_species_weight: np.ndarray

    bulk_layers: int
    bulk_center_layer_count: int
    bulk_realizations: int

    reference_lattice_matrix: np.ndarray

    build_seconds: float

    @property
    def finite_pair_rows(
        self,
    ) -> int:
        return int(
            self.finite_rows.number_of_rows
        )

    @property
    def bulk_pair_rows(
        self,
    ) -> int:
        return int(
            self.bulk_rows.number_of_rows
        )


def _clean_element_symbol(
    value: str,
) -> str:
    match = re.match(
        r"([A-Za-z]{1,2})",
        str(
            value
            or ""
        ).strip(),
    )

    if match is None:
        return "X"

    symbol = match.group(
        1
    )

    if len(symbol) == 1:
        return symbol.upper()

    return (
        symbol[0].upper()
        + symbol[1].lower()
    )


def _empty_rows() -> StochasticPairRows:
    return StochasticPairRows(
        source_alpha=np.zeros(
            0,
            dtype=np.int32,
        ),
        source_beta=np.zeros(
            0,
            dtype=np.int32,
        ),
        delta_fractional=np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float64,
        ),
        delta_a=np.zeros(
            0,
            dtype=np.float64,
        ),
        delta_b=np.zeros(
            0,
            dtype=np.float64,
        ),
        transition_coefficients=np.zeros(
            (
                0,
                0,
            ),
            dtype=np.int32,
        ),
        reference_cart=np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float64,
        ),
        reference_distance=np.zeros(
            0,
            dtype=np.float64,
        ),
        alpha_id=np.zeros(
            0,
            dtype=np.int16,
        ),
        beta_id=np.zeros(
            0,
            dtype=np.int16,
        ),
        realization_index=np.zeros(
            0,
            dtype=np.int32,
        ),
        row_weight=np.zeros(
            0,
            dtype=np.float64,
        ),
    )


def _source_species_metadata(
    geometry: CompiledStochasticEnsembleGeometry,
) -> Dict[str, Any]:
    source_symbols = [
        _clean_element_symbol(
            value
        )
        for value in geometry.source_elements
    ]

    unique_species = sorted(
        set(
            source_symbols
        )
    )

    species_index = {
        symbol: index
        for index, symbol in enumerate(
            unique_species
        )
    }

    source_species_id = np.asarray(
        [
            species_index[
                symbol
            ]
            for symbol in source_symbols
        ],
        dtype=np.int16,
    )

    return {
        "unique_species": unique_species,
        "source_species_id": source_species_id,
    }

def _group_identical_state_paths(
    state_paths: np.ndarray,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    """
    Group identical stochastic state paths.

    Identical state paths generate identical stochastic layer topology.
    They only need to be compiled once. Their statistical contribution is
    preserved through the multiplicity value.
    """
    state_paths = np.ascontiguousarray(
        state_paths,
    )

    if state_paths.ndim != 2:
        raise ValueError(
            "state_paths must have shape "
            "(number_of_realizations, number_of_layers)."
        )

    grouped: Dict[
        bytes,
        Dict[
            str,
            Any,
        ],
    ] = {}

    for realization_index in range(
        state_paths.shape[0]
    ):
        path = np.ascontiguousarray(
            state_paths[
                realization_index
            ]
        )

        key = path.tobytes()

        if key not in grouped:
            grouped[
                key
            ] = {
                "representative_index": int(
                    realization_index
                ),
                "multiplicity": 0,
                "member_indices": [],
            }

        grouped[
            key
        ][
            "multiplicity"
        ] += 1

        grouped[
            key
        ][
            "member_indices"
        ].append(
            int(
                realization_index
            )
        )

    groups = list(
        grouped.values()
    )

    groups.sort(
        key=lambda item: int(
            item[
                "representative_index"
            ]
        )
    )

    return groups

def _take_stochastic_pair_rows(
    rows: StochasticPairRows,
    indices: np.ndarray,
    grouped_weight: np.ndarray,
) -> StochasticPairRows:
    """
    Select representative stochastic pair rows after descriptor grouping.
    """
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    grouped_weight = np.asarray(
        grouped_weight,
        dtype=np.float64,
    )

    return StochasticPairRows(
        source_alpha=np.asarray(
            rows.source_alpha[
                indices
            ],
            dtype=np.int32,
        ),
        source_beta=np.asarray(
            rows.source_beta[
                indices
            ],
            dtype=np.int32,
        ),
        delta_fractional=np.asarray(
            rows.delta_fractional[
                indices
            ],
            dtype=np.float64,
        ),
        delta_a=np.asarray(
            rows.delta_a[
                indices
            ],
            dtype=np.float64,
        ),
        delta_b=np.asarray(
            rows.delta_b[
                indices
            ],
            dtype=np.float64,
        ),
        transition_coefficients=np.asarray(
            rows.transition_coefficients[
                indices
            ],
            dtype=np.int32,
        ),
        reference_cart=np.asarray(
            rows.reference_cart[
                indices
            ],
            dtype=np.float64,
        ),
        reference_distance=np.asarray(
            rows.reference_distance[
                indices
            ],
            dtype=np.float64,
        ),
        alpha_id=np.asarray(
            rows.alpha_id[
                indices
            ],
            dtype=np.int16,
        ),
        beta_id=np.asarray(
            rows.beta_id[
                indices
            ],
            dtype=np.int16,
        ),
        realization_index=np.asarray(
            rows.realization_index[
                indices
            ],
            dtype=np.int32,
        ),
        row_weight=grouped_weight,
    )

def _collapse_stochastic_pair_rows(
    rows: StochasticPairRows,
) -> StochasticPairRows:
    """
    Collapse duplicate descriptors before different realizations are joined.

    This is the important memory reduction step. The original code joins all
    raw pair rows and later calls np.unique() on the complete finite ensemble.

    Here, duplicate descriptors are reduced inside each realization first.
    Their row weights are summed so the statistical result is preserved.
    """
    number_of_rows = int(
        rows.number_of_rows
    )

    if number_of_rows <= 1:
        return rows

    transition_coefficients = np.asarray(
        rows.transition_coefficients,
        dtype=np.int64,
    )

    if transition_coefficients.ndim != 2:
        raise ValueError(
            "transition_coefficients must be two-dimensional."
        )

    descriptor = np.ascontiguousarray(
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
                        0
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_fractional,
                        dtype=np.float64,
                    )[
                        :,
                        1
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_fractional,
                        dtype=np.float64,
                    )[
                        :,
                        2
                    ]
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_a,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_b,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                transition_coefficients,
            )
        ),
        dtype=np.int64,
    )

    packed_dtype = np.dtype(
        (
            np.void,
            descriptor.dtype.itemsize
            * descriptor.shape[1],
        )
    )

    packed = descriptor.view(
        packed_dtype
    ).reshape(
        -1
    )

    row_weight = np.asarray(
        rows.row_weight,
        dtype=np.float64,
    )

    weights_are_constant = bool(
        row_weight.size > 0
        and np.all(
            row_weight
            == row_weight[
                0
            ]
        )
    )

    if weights_are_constant:
        (
            _unique_packed,
            first_index,
            counts,
        ) = _fast_unique_rows(
            descriptor,
            return_index=True,
            return_counts=True,
        )

        grouped_weight = (
            counts.astype(
                np.float64,
                copy=False,
            )
            * float(
                row_weight[
                    0
                ]
            )
        )

    else:
        (
            _unique_packed,
            first_index,
            inverse,
        ) = _fast_unique_rows(
            descriptor,
            return_index=True,
            return_inverse=True,
        )

        grouped_weight = np.bincount(
            inverse.astype(
                np.int64,
                copy=False,
            ),
            weights=row_weight,
            minlength=first_index.size,
        ).astype(
            np.float64,
            copy=False,
        )

    return _take_stochastic_pair_rows(
        rows,
        first_index,
        grouped_weight,
    )


def _append_directed_pairs(
    *,
    first: np.ndarray,
    second: np.ndarray,
    positions: np.ndarray,
    source_index: np.ndarray,
    layer_descriptor_index: np.ndarray,
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    layer_descriptors: List[
        CompiledStochasticLayerDescriptor
    ],
    source_fractional: np.ndarray,
    source_species_id: np.ndarray,
    realization_index: int,
    row_weight: float,
    output: Dict[str, list],
) -> None:
    """
    Expand unordered pairs into directed rows and append descriptors.
    """
    if first.size == 0:
        return

    first = np.asarray(
        first,
        dtype=np.int64,
    )

    second = np.asarray(
        second,
        dtype=np.int64,
    )

    directed_center = np.concatenate(
        (
            first,
            second,
        )
    )

    directed_neighbor = np.concatenate(
        (
            second,
            first,
        )
    )

    cart_forward = (
        positions[
            second
        ]
        - positions[
            first
        ]
    )

    directed_cart = np.vstack(
        (
            cart_forward,
            -cart_forward,
        )
    )

    source_alpha = source_index[
        directed_center
    ].astype(
        np.int32,
        copy=False,
    )

    source_beta = source_index[
        directed_neighbor
    ].astype(
        np.int32,
        copy=False,
    )

    layer_alpha = layer_descriptor_index[
        directed_center
    ].astype(
        np.int32,
        copy=False,
    )

    layer_beta = layer_descriptor_index[
        directed_neighbor
    ].astype(
        np.int32,
        copy=False,
    )

    delta_fractional = (
        source_fractional[
            source_beta
        ]
        - source_fractional[
            source_alpha
        ]
    )

    delta_a = (
        cell_a[
            directed_neighbor
        ].astype(
            np.float64,
            copy=False,
        )
        - cell_a[
            directed_center
        ].astype(
            np.float64,
            copy=False,
        )
    )

    delta_b = (
        cell_b[
            directed_neighbor
        ].astype(
            np.float64,
            copy=False,
        )
        - cell_b[
            directed_center
        ].astype(
            np.float64,
            copy=False,
        )
    )

    layer_coefficients = np.asarray(
        [
            layer.transition_coefficients
            for layer in layer_descriptors
        ],
        dtype=np.int32,
    )

    transition_coefficients = (
        layer_coefficients[
            layer_beta
        ]
        - layer_coefficients[
            layer_alpha
        ]
    ).astype(
        np.int32,
        copy=False,
    )

    distances = np.sqrt(
        np.einsum(
            "ij,ij->i",
            directed_cart,
            directed_cart,
        )
    )

    output[
        "source_alpha"
    ].append(
        source_alpha
    )

    output[
        "source_beta"
    ].append(
        source_beta
    )

    output[
        "delta_fractional"
    ].append(
        delta_fractional.astype(
            np.float64,
            copy=False,
        )
    )

    output[
        "delta_a"
    ].append(
        delta_a
    )

    output[
        "delta_b"
    ].append(
        delta_b
    )

    output[
        "transition_coefficients"
    ].append(
        transition_coefficients
    )

    output[
        "reference_cart"
    ].append(
        directed_cart.astype(
            np.float64,
            copy=False,
        )
    )

    output[
        "reference_distance"
    ].append(
        distances.astype(
            np.float64,
            copy=False,
        )
    )

    output[
        "alpha_id"
    ].append(
        source_species_id[
            source_alpha
        ].astype(
            np.int16,
            copy=False,
        )
    )

    output[
        "beta_id"
    ].append(
        source_species_id[
            source_beta
        ].astype(
            np.int16,
            copy=False,
        )
    )

    output[
        "realization_index"
    ].append(
        np.full(
            source_alpha.size,
            int(
                realization_index
            ),
            dtype=np.int32,
        )
    )

    output[
        "row_weight"
    ].append(
        np.full(
            source_alpha.size,
            float(
                row_weight
            ),
            dtype=np.float64,
        )
    )


def _finalize_rows(
    output: Dict[str, list],
) -> StochasticPairRows:
    if not output[
        "source_alpha"
    ]:
        return _empty_rows()

    return StochasticPairRows(
        source_alpha=np.concatenate(
            output[
                "source_alpha"
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        source_beta=np.concatenate(
            output[
                "source_beta"
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        delta_fractional=np.vstack(
            output[
                "delta_fractional"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        delta_a=np.concatenate(
            output[
                "delta_a"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        delta_b=np.concatenate(
            output[
                "delta_b"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        transition_coefficients=np.vstack(
            output[
                "transition_coefficients"
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        reference_cart=np.vstack(
            output[
                "reference_cart"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        reference_distance=np.concatenate(
            output[
                "reference_distance"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        alpha_id=np.concatenate(
            output[
                "alpha_id"
            ]
        ).astype(
            np.int16,
            copy=False,
        ),
        beta_id=np.concatenate(
            output[
                "beta_id"
            ]
        ).astype(
            np.int16,
            copy=False,
        ),
        realization_index=np.concatenate(
            output[
                "realization_index"
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        row_weight=np.concatenate(
            output[
                "row_weight"
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
    )


def _new_row_collector() -> Dict[str, list]:
    return {
        "source_alpha": [],
        "source_beta": [],
        "delta_fractional": [],
        "delta_a": [],
        "delta_b": [],
        "transition_coefficients": [],
        "reference_cart": [],
        "reference_distance": [],
        "alpha_id": [],
        "beta_id": [],
        "realization_index": [],
        "row_weight": [],
    }


def _finite_pairs_for_realization(
    *,
    evaluated: EvaluatedStochasticRealizationGeometry,
    topology,
    source_fractional: np.ndarray,
    source_species_id: np.ndarray,
    topology_cutoff: float,
    ensemble_samples: int,
    realization_multiplicity: int = 1,
) -> Tuple[
    StochasticPairRows,
    np.ndarray,
]:
    """
    Build directed finite-particle pair descriptors for one realization.

    Returns:
        pair rows;
        species counts for center atoms in this realization.
    """
    positions = np.asarray(
        evaluated.atom_cart,
        dtype=np.float64,
    )

    if positions.shape[0] <= 1:
        return (
            _empty_rows(),
            np.zeros(
                int(
                    np.max(
                        source_species_id
                    )
                )
                + 1,
                dtype=np.float64,
            ),
        )

    tree = cKDTree(
        positions
    )

    unordered_pairs = tree.query_pairs(
        r=float(
            topology_cutoff
        ),
        output_type="ndarray",
    )

    unordered_pairs = np.asarray(
        unordered_pairs,
        dtype=np.int32,
    ).reshape(
        -1,
        2,
    )

    if unordered_pairs.size == 0:
        return (
            _empty_rows(),
            np.zeros(
                int(
                    np.max(
                        source_species_id
                    )
                )
                + 1,
                dtype=np.float64,
            ),
        )

    source_index = np.asarray(
        topology.atom_source_site_indices,
        dtype=np.int32,
    )

    layer_descriptor_index = np.asarray(
        topology.atom_layer_descriptor_indices,
        dtype=np.int32,
    )

    cell_a = np.asarray(
        topology.atom_cell_a,
        dtype=np.int32,
    )

    cell_b = np.asarray(
        topology.atom_cell_b,
        dtype=np.int32,
    )

    collector = _new_row_collector()

    _append_directed_pairs(
        first=unordered_pairs[
            :,
            0,
        ],
        second=unordered_pairs[
            :,
            1,
        ],
        positions=positions,
        source_index=source_index,
        layer_descriptor_index=layer_descriptor_index,
        cell_a=cell_a,
        cell_b=cell_b,
        layer_descriptors=topology.layers,
        source_fractional=source_fractional,
        source_species_id=source_species_id,
        realization_index=int(
            topology.realization_index
        ),
        row_weight=(
            float(
                realization_multiplicity
            )
            / float(
                ensemble_samples
            )
        ),
        output=collector,
    )

    pair_rows = _finalize_rows(
        collector
    )

    center_species_count = np.bincount(
        source_species_id[
            source_index
        ].astype(
            np.int64
        ),
        minlength=int(
            np.max(
                source_species_id
            )
        )
        + 1,
    ).astype(
        np.float64
    )

    center_species_count *= (
        float(
            realization_multiplicity
        )
        / float(
            ensemble_samples
        )
    )

    return (
        pair_rows,
        center_species_count,
    )


def _bulk_realization_descriptors(
    *,
    realization_index: int,
    states: np.ndarray,
    stack_spec: LayerStackSpec,
    reference_parameters: Mapping[
        str,
        Any,
    ],
) -> List[
    CompiledStochasticLayerDescriptor
]:
    """
    Create long open-stack descriptors for bulk normalization.

    There is intentionally no periodic closure.
    """
    state_du, _state_dv, _state_h, labels = (
        _generic_state_vectors(
            stack_spec,
            reference_parameters,
        )
    )

    return _layer_descriptors_from_states(
        realization_index=realization_index,
        states=states,
        number_of_states=int(
            state_du.size
        ),
        labels=labels,
    )


def _reference_origins_for_descriptors(
    *,
    layers: List[
        CompiledStochasticLayerDescriptor
    ],
    stack_spec: LayerStackSpec,
    reference_parameters: Mapping[
        str,
        Any,
    ],
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """
    Build reference origins for a generic N-state Markov path.

    Each layer carries a vector of transition counts:

        transition_coefficients[layer, state]

    and each state emits one vector:

        state_du[state] * a
        state_dv[state] * b
        state_h[state]  * normal
    """

    state_du, state_dv, state_h, _labels = (
        _generic_state_vectors(
            stack_spec,
            reference_parameters,
        )
    )

    return _reference_layer_origins(
        layers=layers,
        vector_a=vector_a,
        vector_b=vector_b,
        normal=normal,
        state_du=state_du,
        state_dv=state_dv,
        state_h=state_h,
    )


def _bulk_pairs_for_path(
    *,
    realization_index: int,
    states: np.ndarray,
    source_fractional: np.ndarray,
    source_cart: np.ndarray,
    source_species_id: np.ndarray,
    stack_spec: LayerStackSpec,
    reference_parameters: Mapping[
        str,
        Any,
    ],
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    normal: np.ndarray,
    topology_cutoff: float,
    ensemble_samples: int,
    interior_margin_layers: int,
    max_center_layers: int,
    realization_multiplicity: int = 1,
) -> Tuple[
    StochasticPairRows,
    np.ndarray,
    int,
]:
    """
    Build open-stack bulk pair rows for one frozen realization.

    This is intentionally non-periodic along the stacking direction.
    In-plane periodicity is represented by explicit parent a/b translations.

    Every pair descriptor uses generic transition coefficients:

        shape = (number_of_pair_rows, number_of_states)
    """
    states = np.asarray(
        states,
        dtype=np.int16,
    ).reshape(
        -1
    )

    layers = _bulk_realization_descriptors(
        realization_index=realization_index,
        states=states,
        stack_spec=stack_spec,
        reference_parameters=reference_parameters,
    )

    number_of_layers = int(
        len(
            layers
        )
    )

    number_of_species = int(
        np.max(
            source_species_id
        )
    ) + 1

    if number_of_layers <= 0:
        return (
            _empty_rows(),
            np.zeros(
                number_of_species,
                dtype=np.float64,
            ),
            0,
        )

    interior_start = max(
        0,
        int(
            interior_margin_layers
        ),
    )

    interior_stop = min(
        number_of_layers,
        number_of_layers
        - int(
            interior_margin_layers
        ),
    )

    center_layer_indices = np.arange(
        interior_start,
        interior_stop,
        dtype=np.int32,
    )

    if center_layer_indices.size == 0:
        center_layer_indices = np.arange(
            number_of_layers,
            dtype=np.int32,
        )

    if (
        max_center_layers > 0
        and center_layer_indices.size
        > int(
            max_center_layers
        )
    ):
        sample_indices = np.linspace(
            0,
            center_layer_indices.size - 1,
            int(
                max_center_layers
            ),
            dtype=np.int64,
        )

        center_layer_indices = center_layer_indices[
            sample_indices
        ]

    reference_origins = _reference_origins_for_descriptors(
        layers=layers,
        stack_spec=stack_spec,
        reference_parameters=reference_parameters,
        vector_a=vector_a,
        vector_b=vector_b,
        normal=normal,
    )

    source_count = int(
        source_fractional.shape[0]
    )

    a_length = max(
        float(
            np.linalg.norm(
                vector_a
            )
        ),
        _EPSILON,
    )

    b_length = max(
        float(
            np.linalg.norm(
                vector_b
            )
        ),
        _EPSILON,
    )

    n_a = max(
        1,
        int(
            np.ceil(
                float(
                    topology_cutoff
                )
                / a_length
            )
        )
        + 1,
    )

    n_b = max(
        1,
        int(
            np.ceil(
                float(
                    topology_cutoff
                )
                / b_length
            )
        )
        + 1,
    )

    translations_a = range(
        -n_a,
        n_a + 1,
    )

    translations_b = range(
        -n_b,
        n_b + 1,
    )

    neighbor_positions = []
    neighbor_source_index = []
    neighbor_layer_index = []
    neighbor_cell_a = []
    neighbor_cell_b = []

    for layer_index in range(
        number_of_layers
    ):
        origin = reference_origins[
            layer_index
        ]

        for cell_a in translations_a:
            for cell_b in translations_b:
                translation = (
                    cell_a
                    * vector_a
                    + cell_b
                    * vector_b
                )

                neighbor_positions.append(
                    source_cart
                    + translation[
                        None,
                        :
                    ]
                    + origin[
                        None,
                        :
                    ]
                )

                neighbor_source_index.append(
                    np.arange(
                        source_count,
                        dtype=np.int32,
                    )
                )

                neighbor_layer_index.append(
                    np.full(
                        source_count,
                        layer_index,
                        dtype=np.int32,
                    )
                )

                neighbor_cell_a.append(
                    np.full(
                        source_count,
                        cell_a,
                        dtype=np.int32,
                    )
                )

                neighbor_cell_b.append(
                    np.full(
                        source_count,
                        cell_b,
                        dtype=np.int32,
                    )
                )

    if not neighbor_positions:
        return (
            _empty_rows(),
            np.zeros(
                number_of_species,
                dtype=np.float64,
            ),
            0,
        )

    neighbor_positions = np.vstack(
        neighbor_positions
    ).astype(
        np.float64,
        copy=False,
    )

    neighbor_source_index = np.concatenate(
        neighbor_source_index
    ).astype(
        np.int32,
        copy=False,
    )

    neighbor_layer_index = np.concatenate(
        neighbor_layer_index
    ).astype(
        np.int32,
        copy=False,
    )

    neighbor_cell_a = np.concatenate(
        neighbor_cell_a
    ).astype(
        np.int32,
        copy=False,
    )

    neighbor_cell_b = np.concatenate(
        neighbor_cell_b
    ).astype(
        np.int32,
        copy=False,
    )

    center_positions = []
    center_source_index = []
    center_layer_index = []

    for layer_index in center_layer_indices:
        origin = reference_origins[
            int(
                layer_index
            )
        ]

        center_positions.append(
            source_cart
            + origin[
                None,
                :
            ]
        )

        center_source_index.append(
            np.arange(
                source_count,
                dtype=np.int32,
            )
        )

        center_layer_index.append(
            np.full(
                source_count,
                int(
                    layer_index
                ),
                dtype=np.int32,
            )
        )

    center_positions = np.vstack(
        center_positions
    ).astype(
        np.float64,
        copy=False,
    )

    center_source_index = np.concatenate(
        center_source_index
    ).astype(
        np.int32,
        copy=False,
    )

    center_layer_index = np.concatenate(
        center_layer_index
    ).astype(
        np.int32,
        copy=False,
    )

    tree = cKDTree(
        neighbor_positions
    )

    center_to_neighbor = tree.query_ball_point(
        center_positions,
        r=float(
            topology_cutoff
        ),
        workers=1,
    )

    layer_coefficients = np.asarray(
        [
            layer.transition_coefficients
            for layer in layers
        ],
        dtype=np.int32,
    )

    source_fractional_wrapped = np.asarray(
        source_fractional,
        dtype=np.float64,
    ).copy()

    source_fractional_wrapped[
        :,
        0,
    ] %= 1.0

    source_fractional_wrapped[
        :,
        1,
    ] %= 1.0

    collector = _new_row_collector()

    row_weight = (
        float(
            realization_multiplicity
        )
        / (
            float(
                ensemble_samples
            )
            * float(
                max(
                    1,
                    center_layer_indices.size,
                )
            )
        )
    )

    for center_row, neighbor_rows in enumerate(
        center_to_neighbor
    ):
        if not neighbor_rows:
            continue

        alpha_position = center_positions[
            center_row
        ]

        alpha_source = int(
            center_source_index[
                center_row
            ]
        )

        alpha_layer = int(
            center_layer_index[
                center_row
            ]
        )

        neighbor_rows = np.asarray(
            neighbor_rows,
            dtype=np.int64,
        )

        beta_position = neighbor_positions[
            neighbor_rows
        ]

        vectors = (
            beta_position
            - alpha_position[
                None,
                :
            ]
        )

        distances = np.sqrt(
            np.einsum(
                "ij,ij->i",
                vectors,
                vectors,
            )
        )

        valid = (
            np.isfinite(
                distances
            )
            & (
                distances
                > 1.0e-8
            )
            & (
                distances
                <= float(
                    topology_cutoff
                )
                + 1.0e-10
            )
        )

        if not np.any(
            valid
        ):
            continue

        neighbor_rows = neighbor_rows[
            valid
        ]

        vectors = vectors[
            valid
        ]

        distances = distances[
            valid
        ]

        beta_source = neighbor_source_index[
            neighbor_rows
        ].astype(
            np.int32,
            copy=False,
        )

        beta_layer = neighbor_layer_index[
            neighbor_rows
        ].astype(
            np.int32,
            copy=False,
        )

        beta_cell_a = neighbor_cell_a[
            neighbor_rows
        ].astype(
            np.float64,
            copy=False,
        )

        beta_cell_b = neighbor_cell_b[
            neighbor_rows
        ].astype(
            np.float64,
            copy=False,
        )

        delta_fractional = (
            source_fractional_wrapped[
                beta_source
            ]
            - source_fractional_wrapped[
                alpha_source
            ]
        )

        transition_coefficients = (
            layer_coefficients[
                beta_layer
            ]
            - layer_coefficients[
                alpha_layer
            ]
        ).astype(
            np.int32,
            copy=False,
        )

        number = int(
            beta_source.size
        )

        collector[
            "source_alpha"
        ].append(
            np.full(
                number,
                alpha_source,
                dtype=np.int32,
            )
        )

        collector[
            "source_beta"
        ].append(
            beta_source
        )

        collector[
            "delta_fractional"
        ].append(
            delta_fractional
        )

        collector[
            "delta_a"
        ].append(
            beta_cell_a
        )

        collector[
            "delta_b"
        ].append(
            beta_cell_b
        )

        collector[
            "transition_coefficients"
        ].append(
            transition_coefficients
        )

        collector[
            "reference_cart"
        ].append(
            vectors
        )

        collector[
            "reference_distance"
        ].append(
            distances
        )

        collector[
            "alpha_id"
        ].append(
            np.full(
                number,
                int(
                    source_species_id[
                        alpha_source
                    ]
                ),
                dtype=np.int16,
            )
        )

        collector[
            "beta_id"
        ].append(
            source_species_id[
                beta_source
            ].astype(
                np.int16,
                copy=False,
            )
        )

        collector[
            "realization_index"
        ].append(
            np.full(
                number,
                int(
                    realization_index
                ),
                dtype=np.int32,
            )
        )

        collector[
            "row_weight"
        ].append(
            np.full(
                number,
                row_weight,
                dtype=np.float64,
            )
        )

    pair_rows = _finalize_rows(
        collector
    )

    center_species_count = np.bincount(
        source_species_id[
            center_source_index
        ].astype(
            np.int64
        ),
        minlength=number_of_species,
    ).astype(
        np.float64
    )

    center_species_count *= (
        float(
            realization_multiplicity
        )
        / (
            float(
                ensemble_samples
            )
            * float(
                max(
                    1,
                    center_layer_indices.size,
                )
            )
        )
    )

    return (
        pair_rows,
        center_species_count,
        int(
            center_layer_indices.size
        ),
    )

def _concatenate_rows(
    rows: List[
        StochasticPairRows
    ],
) -> StochasticPairRows:
    rows = [
        row
        for row in rows
        if row.number_of_rows > 0
    ]

    if not rows:
        return _empty_rows()

    return StochasticPairRows(
        source_alpha=np.concatenate(
            [
                row.source_alpha
                for row in rows
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        source_beta=np.concatenate(
            [
                row.source_beta
                for row in rows
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        delta_fractional=np.vstack(
            [
                row.delta_fractional
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        delta_a=np.concatenate(
            [
                row.delta_a
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        delta_b=np.concatenate(
            [
                row.delta_b
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        transition_coefficients=np.vstack(
            [
                row.transition_coefficients
                for row in rows
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        reference_cart=np.vstack(
            [
                row.reference_cart
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        reference_distance=np.concatenate(
            [
                row.reference_distance
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
        alpha_id=np.concatenate(
            [
                row.alpha_id
                for row in rows
            ]
        ).astype(
            np.int16,
            copy=False,
        ),
        beta_id=np.concatenate(
            [
                row.beta_id
                for row in rows
            ]
        ).astype(
            np.int16,
            copy=False,
        ),
        realization_index=np.concatenate(
            [
                row.realization_index
                for row in rows
            ]
        ).astype(
            np.int32,
            copy=False,
        ),
        row_weight=np.concatenate(
            [
                row.row_weight
                for row in rows
            ]
        ).astype(
            np.float64,
            copy=False,
        ),
    )


def _bulk_paths_from_uniforms(
    *,
    geometry: CompiledStochasticEnsembleGeometry,
    bulk_layers: int,
):
    """
    Generate bulk paths using the SAME stochastic model as the finite stack.

    This is essential: finite/bulk gamma normalization is invalid if the
    finite stack uses generic Markov paths while bulk uses A/B paths.
    """
    from pdf_fitting.models.stacking_disorder import (
        generate_correlated_ab_paths,
        generate_generic_markov_paths,
    )

    stack_spec = geometry.stack_spec.normalized()

    bulk_seed = int(
        stack_spec.ensemble_seed
    ) + 1_000_003

    if stack_spec.model == "generic_markov":
        return generate_generic_markov_paths(
            number_of_layers=int(
                bulk_layers
            ),
            ensemble_samples=int(
                geometry.ensemble_samples
            ),
            transition_matrix=np.asarray(
                stack_spec.markov_transition_matrix,
                dtype=float,
            ),
            seed=bulk_seed,
            start_state=int(
                stack_spec.markov_start_state
            ),
        ).states

    return generate_correlated_ab_paths(
        number_of_layers=int(
            bulk_layers
        ),
        ensemble_samples=int(
            geometry.ensemble_samples
        ),
        p_alternate=float(
            stack_spec.p_alternate
        ),
        seed=bulk_seed,
        start_state=str(
            stack_spec.ensemble_start_state
        ),
    ).states


def compile_stochastic_pair_topology(
    *,
    geometry: CompiledStochasticEnsembleGeometry,
    pair_cutoff: float,
    topology_margin: float = 1.0,
    reference_parameters: Optional[
        Mapping[str, Any]
    ] = None,
    bulk_layers: int = 100,
    bulk_max_center_layers: int = 8,
    cpu_workers: int = 0,
    collapse_realizations: bool = True,
    deduplicate_paths: bool = True,
) -> CompiledStochasticPairTopology:
    """
    Compile finite and open-bulk pair rows for a frozen stochastic ensemble.

    Parameters
    ----------
    geometry
        Frozen finite ensemble topology from
        compile_stochastic_ensemble_geometry().

    pair_cutoff
        PDF pair cutoff.

    topology_margin
        Extra distance margin retained during pair-topology construction.
        This keeps pair topology valid for moderate continuous vector/lattice
        refinement.

    bulk_layers
        Long open stochastic stack used for bulk normalization.
        Typical TOPAS-compatible value: 100.

    bulk_max_center_layers
        Maximum number of interior bulk layers sampled per realization.
        This limits topology size while retaining representative bulk
        statistics. Use 8 initially; increase for final validation.

    Notes
    -----
    The stochastic bulk is deliberately non-periodic along stacking direction.
    Only interior layers act as centers, avoiding artificial end-to-start
    periodic correlations.
    """
    pair_cutoff = float(
        pair_cutoff
    )

    topology_margin = max(
        float(
            topology_margin
        ),
        0.0,
    )

    topology_cutoff = (
        pair_cutoff
        + topology_margin
    )

    if pair_cutoff <= 0.0:
        raise ValueError(
            "pair_cutoff must be positive."
        )

    if geometry is None:
        raise ValueError(
            "A compiled stochastic geometry is required."
        )

    reference_parameters = dict(
        reference_parameters
        or {}
    )

    requested_cpu_workers = int(
        cpu_workers
    )

    start_time = time.perf_counter()

    species_metadata = _source_species_metadata(
        geometry
    )

    unique_species = list(
        species_metadata[
            "unique_species"
        ]
    )

    source_species_id = np.asarray(
        species_metadata[
            "source_species_id"
        ],
        dtype=np.int16,
    )

    number_of_species = int(
        len(
            unique_species
        )
    )

    finite_rows_all = []

    finite_center_species_weight = np.zeros(
        number_of_species,
        dtype=np.float64,
    )


    if deduplicate_paths:
        finite_path_groups = _group_identical_state_paths(
            geometry.state_paths
        )

    else:
        finite_path_groups = [
            {
                "representative_index": int(
                    realization_index
                ),
                "multiplicity": 1,
                "member_indices": [
                    int(
                        realization_index
                    )
                ],
            }
            for realization_index in range(
                geometry.ensemble_samples
            )
        ]

    finite_atom_counts = [
        int(
            realization.atom_source_site_indices.size
        )
        for realization in geometry.realizations
    ]

    maximum_finite_atom_count = max(
        finite_atom_counts
        or [
            1
        ]
    )

    estimated_worker_bytes = _estimate_topology_worker_bytes(
        atom_count=maximum_finite_atom_count,
    )

    finite_cpu_workers = _resolve_stochastic_topology_workers(
        requested_cpu_workers,
        number_of_jobs=len(
            finite_path_groups
        ),
        estimated_bytes_per_worker=(
            estimated_worker_bytes
        ),
    )

    available_memory = _available_memory_bytes()

    print(
        "[STOCHASTIC CPU] "
        f"logical_cpus={os.cpu_count() or 1}, "
        f"requested_workers={requested_cpu_workers}, "
        f"finite_workers={finite_cpu_workers}, "
        f"unique_finite_jobs={len(finite_path_groups)}, "
        f"estimated_worker_memory="
        f"{estimated_worker_bytes / (1024.0 ** 3):.2f} GiB, "
        f"available_memory="
        + (
            f"{available_memory / (1024.0 ** 3):.2f} GiB"
            if available_memory is not None
            else "unknown"
        ),
        flush=True,
    )

    print(
        "[STOCHASTIC PAIRS] Finite path grouping: "
        f"input_paths={geometry.ensemble_samples:,}, "
        f"unique_paths={len(finite_path_groups):,}, "
        f"duplicates="
        f"{geometry.ensemble_samples - len(finite_path_groups):,}",
        flush=True,
    )

    def compile_finite_group(
        group: Dict[
            str,
            Any,
        ],
    ) -> Dict[
        str,
        Any,
    ]:
        realization_index = int(
            group[
                "representative_index"
            ]
        )

        realization_multiplicity = int(
            group[
                "multiplicity"
            ]
        )

        realization_topology = geometry.realizations[
            realization_index
        ]

        evaluated = geometry.evaluate_realization(
            realization_index,
            reference_parameters,
        )

        (
            pair_rows,
            center_species_count,
        ) = _finite_pairs_for_realization(
            evaluated=evaluated,
            topology=realization_topology,
            source_fractional=geometry.source_fractional,
            source_species_id=source_species_id,
            topology_cutoff=topology_cutoff,
            ensemble_samples=geometry.ensemble_samples,
            realization_multiplicity=(
                realization_multiplicity
            ),
        )

        raw_row_count = int(
            pair_rows.number_of_rows
        )

        collapse_seconds = 0.0

        if (
            collapse_realizations
            and raw_row_count > 0
        ):
            collapse_start = time.perf_counter()

            pair_rows = _collapse_stochastic_pair_rows(
                pair_rows
            )

            collapse_seconds = (
                time.perf_counter()
                - collapse_start
            )

        return {
            "representative_index": realization_index,
            "member_indices": list(
                group[
                    "member_indices"
                ]
            ),
            "multiplicity": realization_multiplicity,
            "atom_count": int(
                evaluated.number_of_atoms
            ),
            "raw_row_count": raw_row_count,
            "collapsed_row_count": int(
                pair_rows.number_of_rows
            ),
            "collapse_seconds": float(
                collapse_seconds
            ),
            "pair_rows": pair_rows,
            "center_species_count": (
                center_species_count
            ),
        }

    finite_results = []

    if (
        finite_cpu_workers <= 1
        or len(
            finite_path_groups
        )
        <= 1
    ):
        for group in finite_path_groups:
            finite_results.append(
                compile_finite_group(
                    group
                )
            )

    else:
        with ThreadPoolExecutor(
            max_workers=finite_cpu_workers,
            thread_name_prefix="stochastic-finite",
        ) as executor:
            future_to_group = {
                executor.submit(
                    compile_finite_group,
                    group,
                ): group
                for group in finite_path_groups
            }

            for future in as_completed(
                future_to_group
            ):
                finite_results.append(
                    future.result()
                )

    finite_results.sort(
        key=lambda result: int(
            result[
                "representative_index"
            ]
        )
    )

    for result in finite_results:
        finite_rows_all.append(
            result[
                "pair_rows"
            ]
        )

        finite_center_species_weight += result[
            "center_species_count"
        ]

        print(
            "[STOCHASTIC PAIRS] Finite realization group: "
            f"representative="
            f"{result['representative_index'] + 1}, "
            f"members={result['member_indices']}, "
            f"multiplicity={result['multiplicity']}, "
            f"atoms={result['atom_count']:,}, "
            f"raw_rows={result['raw_row_count']:,}, "
            f"collapsed_rows={result['collapsed_row_count']:,}, "
            f"collapse_time={result['collapse_seconds']:.3f} s",
            flush=True,
        )

    finite_rows = _concatenate_rows(
        finite_rows_all
    )

    if finite_rows.number_of_rows <= 0:
        raise RuntimeError(
            "The stochastic finite ensemble generated no pair rows."
        )

    reference_lattice = _lattice_matrix_from_parameters(
        geometry.source_structure,
        reference_parameters,
    )

    vector_a, vector_b, normal = layer_basis(
        reference_lattice
    )

    source_cart = _centered_source_layer_cartesian(
        geometry.source_fractional,
        reference_lattice,
        normal,
    )

    (
        state_du,
        state_dv,
        state_h,
        _state_labels,
    ) = _generic_state_vectors(
        geometry.stack_spec,
        reference_parameters,
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

    if (
        state_h.size <= 0
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
            "Every Markov stacking state must have a positive, "
            "finite transition height."
        )

    minimum_height = max(
        float(
            np.min(
                state_h
            )
        ),
        _EPSILON,
    )

    interior_margin_layers = max(
        2,
        int(
            np.ceil(
                topology_cutoff
                / minimum_height
            )
        )
        + 1,
    )

    bulk_layers = max(
        int(
            bulk_layers
        ),
        2
        * interior_margin_layers
        + 3,
    )

    bulk_paths = _bulk_paths_from_uniforms(
        geometry=geometry,
        bulk_layers=bulk_layers,
    )

    bulk_rows_all = []

    bulk_center_species_weight = np.zeros(
        number_of_species,
        dtype=np.float64,
    )

    selected_center_layer_count = 0

    print(
        "[STOCHASTIC PAIRS] Compiling open stochastic bulk reference: "
        f"bulk_layers={bulk_layers}, "
        f"interior_margin_layers={interior_margin_layers}, "
        f"max_center_layers={bulk_max_center_layers}",
        flush=True,
    )

    if deduplicate_paths:
        bulk_path_groups = _group_identical_state_paths(
            bulk_paths
        )

    else:
        bulk_path_groups = [
            {
                "representative_index": int(
                    realization_index
                ),
                "multiplicity": 1,
                "member_indices": [
                    int(
                        realization_index
                    )
                ],
            }
            for realization_index in range(
                bulk_paths.shape[0]
            )
        ]

    print(
        "[STOCHASTIC PAIRS] Bulk path grouping: "
        f"input_paths={bulk_paths.shape[0]:,}, "
        f"unique_paths={len(bulk_path_groups):,}, "
        f"duplicates="
        f"{bulk_paths.shape[0] - len(bulk_path_groups):,}",
        flush=True,
    )

    bulk_source_count = int(
        geometry.source_fractional.shape[0]
    )

    estimated_bulk_atoms = max(
        1,
        int(
            bulk_layers
        )
        * max(
            1,
            bulk_source_count,
        ),
    )

    estimated_bulk_worker_bytes = (
        _estimate_topology_worker_bytes(
            atom_count=estimated_bulk_atoms,
        )
    )

    bulk_cpu_workers = (
        _resolve_stochastic_topology_workers(
            requested_cpu_workers,
            number_of_jobs=len(
                bulk_path_groups
            ),
            estimated_bytes_per_worker=(
                estimated_bulk_worker_bytes
            ),
        )
    )

    print(
        "[STOCHASTIC CPU] "
        f"bulk_workers={bulk_cpu_workers}, "
        f"unique_bulk_jobs={len(bulk_path_groups)}, "
        f"estimated_bulk_worker_memory="
        f"{estimated_bulk_worker_bytes / (1024.0 ** 3):.2f} GiB",
        flush=True,
    )

    def compile_bulk_group(
        group: Dict[
            str,
            Any,
        ],
    ) -> Dict[
        str,
        Any,
    ]:
        realization_index = int(
            group[
                "representative_index"
            ]
        )

        realization_multiplicity = int(
            group[
                "multiplicity"
            ]
        )

        (
            pair_rows,
            center_species_count,
            center_layer_count,
        ) = _bulk_pairs_for_path(
            realization_index=realization_index,
            states=bulk_paths[
                realization_index
            ],
            source_fractional=geometry.source_fractional,
            source_cart=source_cart,
            source_species_id=source_species_id,
            stack_spec=geometry.stack_spec,
            reference_parameters=reference_parameters,
            vector_a=vector_a,
            vector_b=vector_b,
            normal=normal,
            topology_cutoff=topology_cutoff,
            ensemble_samples=bulk_paths.shape[0],
            interior_margin_layers=interior_margin_layers,
            max_center_layers=int(
                bulk_max_center_layers
            ),
            realization_multiplicity=(
                realization_multiplicity
            ),
        )

        raw_row_count = int(
            pair_rows.number_of_rows
        )

        collapse_seconds = 0.0

        if (
            collapse_realizations
            and raw_row_count > 0
        ):
            collapse_start = time.perf_counter()

            pair_rows = _collapse_stochastic_pair_rows(
                pair_rows
            )

            collapse_seconds = (
                time.perf_counter()
                - collapse_start
            )

        return {
            "representative_index": realization_index,
            "member_indices": list(
                group[
                    "member_indices"
                ]
            ),
            "multiplicity": realization_multiplicity,
            "center_layer_count": int(
                center_layer_count
            ),
            "weighted_center_layer_count": int(
                center_layer_count
                * realization_multiplicity
            ),
            "raw_row_count": raw_row_count,
            "collapsed_row_count": int(
                pair_rows.number_of_rows
            ),
            "collapse_seconds": float(
                collapse_seconds
            ),
            "pair_rows": pair_rows,
            "center_species_count": (
                center_species_count
            ),
        }

    bulk_results = []

    if (
        bulk_cpu_workers <= 1
        or len(
            bulk_path_groups
        )
        <= 1
    ):
        for group in bulk_path_groups:
            bulk_results.append(
                compile_bulk_group(
                    group
                )
            )

    else:
        with ThreadPoolExecutor(
            max_workers=bulk_cpu_workers,
            thread_name_prefix="stochastic-bulk",
        ) as executor:
            future_to_group = {
                executor.submit(
                    compile_bulk_group,
                    group,
                ): group
                for group in bulk_path_groups
            }

            for future in as_completed(
                future_to_group
            ):
                bulk_results.append(
                    future.result()
                )

    bulk_source_count = int(
        geometry.source_fractional.shape[0]
    )

    estimated_bulk_atoms = max(
        1,
        int(
            bulk_layers
        )
        * max(
            1,
            bulk_source_count
        ),
    )

    estimated_bulk_worker_bytes = _estimate_topology_worker_bytes(
        atom_count=estimated_bulk_atoms,
    )

    bulk_cpu_workers = _resolve_stochastic_topology_workers(
        requested_cpu_workers,
        number_of_jobs=len(
            bulk_path_groups
        ),
        estimated_bytes_per_worker=(
            estimated_bulk_worker_bytes
        ),
    )

    print(
        "[STOCHASTIC CPU] "
        f"bulk_workers={bulk_cpu_workers}, "
        f"unique_bulk_jobs={len(bulk_path_groups)}, "
        f"estimated_bulk_worker_memory="
        f"{estimated_bulk_worker_bytes / (1024.0 ** 3):.2f} GiB",
        flush=True,
    )

    bulk_results.sort(
        key=lambda result: int(
            result[
                "representative_index"
            ]
        )
    )

    for result in bulk_results:
        bulk_rows_all.append(
            result[
                "pair_rows"
            ]
        )

        bulk_center_species_weight += result[
            "center_species_count"
        ]

        selected_center_layer_count += int(
            result[
                "weighted_center_layer_count"
            ]
        )

        print(
            "[STOCHASTIC PAIRS] Bulk realization group: "
            f"representative="
            f"{result['representative_index'] + 1}, "
            f"members={result['member_indices']}, "
            f"multiplicity={result['multiplicity']}, "
            f"center_layers={result['center_layer_count']}, "
            f"raw_rows={result['raw_row_count']:,}, "
            f"collapsed_rows={result['collapsed_row_count']:,}, "
            f"collapse_time={result['collapse_seconds']:.3f} s",
            flush=True,
        )

    bulk_rows = _concatenate_rows(
        bulk_rows_all
    )

    if bulk_rows.number_of_rows <= 0:
        raise RuntimeError(
            "The stochastic bulk ensemble generated no pair rows."
        )

    elapsed = time.perf_counter() - start_time

    print(
        "[STOCHASTIC PAIRS] Completed stochastic pair topology: "
        f"finite_rows={finite_rows.number_of_rows:,}, "
        f"bulk_rows={bulk_rows.number_of_rows:,}, "
        f"time={elapsed:.3f} s"
    )

    return CompiledStochasticPairTopology(
        geometry=geometry,
        pair_cutoff=pair_cutoff,
        topology_cutoff=topology_cutoff,
        finite_rows=finite_rows,
        bulk_rows=bulk_rows,
        unique_species=unique_species,
        source_species_id=source_species_id,
        finite_center_species_weight=np.asarray(
            finite_center_species_weight,
            dtype=np.float64,
        ),
        bulk_center_species_weight=np.asarray(
            bulk_center_species_weight,
            dtype=np.float64,
        ),
        bulk_layers=int(
            bulk_layers
        ),
        bulk_center_layer_count=int(
            selected_center_layer_count
        ),
        bulk_realizations=int(
            bulk_paths.shape[0]
        ),
        reference_lattice_matrix=np.asarray(
            reference_lattice,
            dtype=np.float64,
        ),
        build_seconds=float(
            elapsed
        ),
    )


def evaluate_stochastic_pair_rows(
    rows: StochasticPairRows,
    *,
    lattice_matrix: np.ndarray,
    state_du: np.ndarray,
    state_dv: np.ndarray,
    state_h: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluate Cartesian pair vectors for an arbitrary N-state stacking model.
    """
    lattice_matrix = np.asarray(
        lattice_matrix,
        dtype=np.float64,
    )

    state_du = np.asarray(
        state_du,
        dtype=np.float64,
    ).reshape(
        -1
    )

    state_dv = np.asarray(
        state_dv,
        dtype=np.float64,
    ).reshape(
        -1
    )

    state_h = np.asarray(
        state_h,
        dtype=np.float64,
    ).reshape(
        -1
    )

    coefficients = np.asarray(
        rows.transition_coefficients,
        dtype=np.float64,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            "transition_coefficients must have shape "
            "(number_of_rows, number_of_states)."
        )

    if coefficients.shape[1] != state_du.size:
        raise ValueError(
            "State-vector count does not match transition coefficient count."
        )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    delta_u = (
        np.asarray(
            rows.delta_a,
            dtype=np.float64,
        )
        + coefficients
        @ state_du
    )

    delta_v = (
        np.asarray(
            rows.delta_b,
            dtype=np.float64,
        )
        + coefficients
        @ state_dv
    )

    delta_h = (
        coefficients
        @ state_h
    )

    cartesian = (
        np.asarray(
            rows.delta_fractional,
            dtype=np.float64,
        )
        @ lattice_matrix
        + delta_u[
            :,
            None
        ]
        * vector_a[
            None,
            :
        ]
        + delta_v[
            :,
            None
        ]
        * vector_b[
            None,
            :
        ]
        + delta_h[
            :,
            None
        ]
        * normal[
            None,
            :
        ]
    )

    distances = np.sqrt(
        np.einsum(
            "ij,ij->i",
            cartesian,
            cartesian,
        )
    )

    return (
        np.asarray(
            cartesian,
            dtype=np.float64,
        ),
        np.asarray(
            distances,
            dtype=np.float64,
        ),
    )

def descriptor_matrix(
    rows: StochasticPairRows,
) -> np.ndarray:
    """
    Return an integer-quantized generic stochastic descriptor matrix.

    The final columns contain the complete transition-coefficient vector:

        transition_coefficients.shape
            == (number_of_rows, number_of_states)

    Therefore this works for both:

        correlated A/B stacking, where number_of_states == 2;
        generic Markov stacking, where number_of_states >= 1.
    """
    transition_coefficients = np.asarray(
        rows.transition_coefficients,
        dtype=np.int64,
    )

    if transition_coefficients.ndim != 2:
        raise ValueError(
            "transition_coefficients must have shape "
            "(number_of_rows, number_of_states)."
        )

    if (
        transition_coefficients.shape[0]
        != rows.number_of_rows
    ):
        raise ValueError(
            "transition_coefficients row count does not match "
            "the stochastic pair-row count."
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
                    np.int64,
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
                    np.int64,
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
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_a,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                np.rint(
                    np.asarray(
                        rows.delta_b,
                        dtype=np.float64,
                    )
                    * _DESCRIPTOR_SCALE
                ).astype(
                    np.int64,
                ),
                transition_coefficients,
            )
        ),
        dtype=np.int64,
    )