from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import re
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np

from scipy.spatial import cKDTree

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True

except Exception:
    NUMBA_AVAILABLE = False
    njit = None
    prange = range


from pymatgen.core import Lattice, Structure

from pdf_fitting.models.compiled_stacked_geometry import (
    CompiledStackedGeometry,
    EvaluatedStackedGeometry,
)

from pdf_fitting.models.layer_stacking import (
    absolute_sequence_layers,
    layer_basis,
)


from pdf_fitting.models.stacking_transitions import (
    evaluate_transition_values,
)

import os

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

_EPSILON = 1.0e-12

_DESCRIPTOR_SCALE = 1.0e10


@dataclass
class EvaluatedStackedPairs:
    """
    Dynamic shell geometry evaluated from a fixed compiled topology.
    """

    geometry: EvaluatedStackedGeometry

    shell_cart: np.ndarray
    shell_frac: np.ndarray
    shell_distance: np.ndarray

    shell_active: np.ndarray

    shell_mult: np.ndarray
    shell_bulk_mult: np.ndarray

    shell_mult_eff: np.ndarray
    shell_bulk_mult_eff: np.ndarray
    shell_gamma: np.ndarray

    shell_alpha_id: np.ndarray
    shell_beta_id: np.ndarray

    shell_hkl: np.ndarray
    shell_direction: np.ndarray

    unique_species: List[str]

    pair_cutoff: float
    topology_cutoff: float

    @property
    def number_of_shells(self) -> int:
        return int(
            self.shell_cart.shape[0]
        )

    @property
    def number_of_active_shells(self) -> int:
        return int(
            np.count_nonzero(
                self.shell_active
            )
        )


@dataclass
class CompiledStackedPairTopology:
    """
    Pair topology compiled once for one finite stacking candidate.

    The shell descriptors are independent of the current lattice metric.

    Each shell vector is evaluated as:

        delta_source_fractional @ parent_lattice
        + delta_a * current_a
        + delta_b * current_b
        + (
            delta_s * current_spacing
            + delta_normal
          ) * current_layer_normal
    """

    geometry: CompiledStackedGeometry

    pair_cutoff: float
    topology_cutoff: float

    shell_source_alpha: np.ndarray
    shell_source_beta: np.ndarray

    shell_delta_fractional: np.ndarray

    shell_delta_a: np.ndarray
    shell_delta_b: np.ndarray

    shell_transition_coefficients: np.ndarray

    reference_transition_du: np.ndarray
    reference_transition_dv: np.ndarray
    reference_transition_h: np.ndarray



    shell_mult: np.ndarray
    shell_bulk_mult: np.ndarray

    shell_alpha_id: np.ndarray
    shell_beta_id: np.ndarray

    finite_species_counts: np.ndarray
    bulk_species_counts: np.ndarray

    unique_species: List[str]

    shell_hkl: np.ndarray
    shell_direction: np.ndarray

    reference_distance: np.ndarray

    finite_pair_rows: int
    bulk_pair_rows: int

    def evaluate(
        self,
        parameters: Mapping[str, Any] | None = None,
    ) -> EvaluatedStackedPairs:
        """
        Reevaluate pair vectors using arbitrary periodic stacking transitions.
        """
        parameters = dict(
            parameters
            or {}
        )

        evaluated_geometry = self.geometry.evaluate(
            parameters
        )

        lattice_matrix = np.asarray(
            evaluated_geometry.parent_lattice_matrix,
            dtype=float,
        )

        vector_a, vector_b, normal = layer_basis(
            lattice_matrix
        )

        (
            _transitions,
            transition_du,
            transition_dv,
            transition_h,
        ) = evaluate_transition_values(
            self.geometry.stack_spec,
            parameters,
        )

        transition_du = np.asarray(
            transition_du,
            dtype=float,
        )

        transition_dv = np.asarray(
            transition_dv,
            dtype=float,
        )

        transition_h = np.asarray(
            transition_h,
            dtype=float,
        )

        coefficients = np.asarray(
            self.shell_transition_coefficients,
            dtype=float,
        )

        if coefficients.ndim != 2:
            raise ValueError(
                "shell_transition_coefficients must be two-dimensional."
            )

        if coefficients.shape[1] != transition_du.size:
            raise ValueError(
                "The transition coefficient count does not match the "
                "current stacking sequence."
            )

        current_delta_a = (
            np.asarray(
                self.shell_delta_a,
                dtype=float,
            )
            + coefficients
            @ transition_du
        )

        current_delta_b = (
            np.asarray(
                self.shell_delta_b,
                dtype=float,
            )
            + coefficients
            @ transition_dv
        )

        current_delta_normal = (
            coefficients
            @ transition_h
        )

        shell_cart = (
            np.asarray(
                self.shell_delta_fractional,
                dtype=float,
            )
            @ lattice_matrix
            + current_delta_a[
                :,
                None
            ]
            * vector_a[
                None,
                :
            ]
            + current_delta_b[
                :,
                None
            ]
            * vector_b[
                None,
                :
            ]
            + current_delta_normal[
                :,
                None
            ]
            * normal[
                None,
                :
            ]
        )

        shell_distance = np.sqrt(
            np.einsum(
                "ij,ij->i",
                shell_cart,
                shell_cart,
            )
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
                    self.pair_cutoff
                )
                + 1.0e-10
            )
        )

        stacking_inverse = np.asarray(
            evaluated_geometry.stacking_inverse_lattice_matrix,
            dtype=float,
        )

        shell_frac = (
            shell_cart
            @ stacking_inverse
        )

        number_of_shells = int(
            shell_distance.size
        )

        shell_mult_eff = np.zeros(
            number_of_shells,
            dtype=np.float32,
        )

        shell_bulk_mult_eff = np.zeros(
            number_of_shells,
            dtype=np.float32,
        )

        shell_gamma = np.zeros(
            number_of_shells,
            dtype=np.float32,
        )

        alpha_id = np.asarray(
            self.shell_alpha_id,
            dtype=np.int64,
        )

        finite_species_counts = np.asarray(
            self.finite_species_counts,
            dtype=np.float64,
        )

        bulk_species_counts = np.asarray(
            self.bulk_species_counts,
            dtype=np.float64,
        )

        valid_alpha = (
            (alpha_id >= 0)
            & (
                alpha_id
                < finite_species_counts.size
            )
            & (
                alpha_id
                < bulk_species_counts.size
            )
        )

        finite_denominator = np.zeros(
            number_of_shells,
            dtype=np.float64,
        )

        bulk_denominator = np.zeros(
            number_of_shells,
            dtype=np.float64,
        )

        finite_denominator[
            valid_alpha
        ] = finite_species_counts[
            alpha_id[
                valid_alpha
            ]
        ]

        bulk_denominator[
            valid_alpha
        ] = bulk_species_counts[
            alpha_id[
                valid_alpha
            ]
        ]

        shell_mult_float = np.asarray(
            self.shell_mult,
            dtype=np.float64,
        )

        shell_bulk_mult_float = np.asarray(
            self.shell_bulk_mult,
            dtype=np.float64,
        )

        valid_finite = (
            shell_active
            & (
                finite_denominator
                > 0.0
            )
        )

        valid_bulk = (
            shell_active
            & (
                bulk_denominator
                > 0.0
            )
        )

        shell_mult_eff[
            valid_finite
        ] = (
            shell_mult_float[
                valid_finite
            ]
            / finite_denominator[
                valid_finite
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        shell_bulk_mult_eff[
            valid_bulk
        ] = (
            shell_bulk_mult_float[
                valid_bulk
            ]
            / bulk_denominator[
                valid_bulk
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        valid_gamma = (
            valid_finite
            & valid_bulk
            & (
                shell_bulk_mult_eff
                > 0.0
            )
        )

        shell_gamma[
            valid_gamma
        ] = (
            shell_mult_eff[
                valid_gamma
            ]
            / shell_bulk_mult_eff[
                valid_gamma
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        return EvaluatedStackedPairs(
            geometry=evaluated_geometry,
            shell_cart=np.asarray(
                shell_cart,
                dtype=np.float64,
            ),
            shell_frac=np.asarray(
                shell_frac,
                dtype=np.float64,
            ),
            shell_distance=np.asarray(
                shell_distance,
                dtype=np.float64,
            ),
            shell_active=np.asarray(
                shell_active,
                dtype=bool,
            ),
            shell_mult=np.asarray(
                self.shell_mult,
                dtype=np.int32,
            ),
            shell_bulk_mult=np.asarray(
                self.shell_bulk_mult,
                dtype=np.int32,
            ),
            shell_mult_eff=shell_mult_eff,
            shell_bulk_mult_eff=(
                shell_bulk_mult_eff
            ),
            shell_gamma=shell_gamma,
            shell_alpha_id=np.asarray(
                self.shell_alpha_id,
                dtype=np.int16,
            ),
            shell_beta_id=np.asarray(
                self.shell_beta_id,
                dtype=np.int16,
            ),
            shell_hkl=np.asarray(
                self.shell_hkl,
                dtype=np.int16,
            ),
            shell_direction=np.asarray(
                self.shell_direction,
                dtype=np.int16,
            ),
            unique_species=list(
                self.unique_species
            ),
            pair_cutoff=float(
                self.pair_cutoff
            ),
            topology_cutoff=float(
                self.topology_cutoff
            ),
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




def _quantize(
    value: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.rint(
        np.asarray(
            value,
            dtype=float,
        )
        * float(
            scale
        )
    ).astype(
        np.int64,
    )

def _make_descriptor_matrix(
    *,
    source_alpha: np.ndarray,
    source_beta: np.ndarray,
    delta_fractional: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    transition_coefficients: np.ndarray,
) -> np.ndarray:
    transition_coefficients = np.asarray(
        transition_coefficients,
        dtype=np.int64,
    )

    if transition_coefficients.ndim != 2:
        raise ValueError(
            "transition_coefficients must be two-dimensional."
        )

    return np.ascontiguousarray(
        np.column_stack(
            (
                np.asarray(
                    source_alpha,
                    dtype=np.int64,
                ),
                np.asarray(
                    source_beta,
                    dtype=np.int64,
                ),
                _quantize(
                    np.asarray(
                        delta_fractional,
                        dtype=float,
                    )[
                        :,
                        0
                    ],
                    _DESCRIPTOR_SCALE,
                ),
                _quantize(
                    np.asarray(
                        delta_fractional,
                        dtype=float,
                    )[
                        :,
                        1
                    ],
                    _DESCRIPTOR_SCALE,
                ),
                _quantize(
                    np.asarray(
                        delta_fractional,
                        dtype=float,
                    )[
                        :,
                        2
                    ],
                    _DESCRIPTOR_SCALE,
                ),
                _quantize(
                    delta_a,
                    _DESCRIPTOR_SCALE,
                ),
                _quantize(
                    delta_b,
                    _DESCRIPTOR_SCALE,
                ),
                transition_coefficients,
            )
        ),
        dtype=np.int64,
    )


def _unique_descriptor_matrix(
    matrix: np.ndarray,
):
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

    packed = matrix.view(
        packed_dtype
    ).reshape(
        -1
    )

    unique_packed, first_index, counts = np.unique(
        packed,
        return_index=True,
        return_counts=True,
    )

    unique_matrix = matrix[
        first_index
    ]

    return (
        unique_matrix,
        np.asarray(
            counts,
            dtype=np.int64,
        ),
        unique_packed,
    )


def _decode_descriptor_matrix(
    matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    matrix = np.asarray(
        matrix,
        dtype=np.int64,
    )

    return {
        "source_alpha": matrix[
            :,
            0
        ].astype(
            np.int32,
        ),
        "source_beta": matrix[
            :,
            1
        ].astype(
            np.int32,
        ),
        "delta_fractional": (
            matrix[
                :,
                2:5
            ].astype(
                np.float64
            )
            / _DESCRIPTOR_SCALE
        ),
        "delta_a": (
            matrix[
                :,
                5
            ].astype(
                np.float64
            )
            / _DESCRIPTOR_SCALE
        ),
        "delta_b": (
            matrix[
                :,
                6
            ].astype(
                np.float64
            )
            / _DESCRIPTOR_SCALE
        ),
        "transition_coefficients": matrix[
            :,
            7:
        ].astype(
            np.int32,
        ),
    }



if NUMBA_AVAILABLE:
    @njit(
        cache=True,
        fastmath=True,
        parallel=True,
    )
    def _count_cell_list_pairs_numba(
        positions,
        sorted_atom_indices,
        cell_offsets,
        nx,
        ny,
        nz,
        cutoff_squared,
        tolerance_squared,
    ):
        """
        Count unordered finite-particle pairs within cutoff.

        Every spatial cell checks:

            - pairs inside itself;
            - pairs in the 13 forward neighboring cells.

        This guarantees every unordered pair is counted exactly once.
        """
        number_of_cells = int(
            cell_offsets.size - 1
        )

        counts = np.zeros(
            number_of_cells,
            dtype=np.int64,
        )

        yz_size = ny * nz

        for cell_id in prange(
            number_of_cells
        ):
            start = cell_offsets[
                cell_id
            ]

            stop = cell_offsets[
                cell_id + 1
            ]

            if stop <= start:
                continue

            cell_x = cell_id // yz_size
            remainder = cell_id - cell_x * yz_size
            cell_y = remainder // nz
            cell_z = remainder - cell_y * nz

            count = 0

            # -----------------------------------------------------
            # Pairs inside the current cell.
            # -----------------------------------------------------
            for first_position in range(
                start,
                stop,
            ):
                atom_i = sorted_atom_indices[
                    first_position
                ]

                xi = positions[
                    atom_i,
                    0,
                ]

                yi = positions[
                    atom_i,
                    1,
                ]

                zi = positions[
                    atom_i,
                    2,
                ]

                for second_position in range(
                    first_position + 1,
                    stop,
                ):
                    atom_j = sorted_atom_indices[
                        second_position
                    ]

                    dx = positions[
                        atom_j,
                        0,
                    ] - xi

                    dy = positions[
                        atom_j,
                        1,
                    ] - yi

                    dz = positions[
                        atom_j,
                        2,
                    ] - zi

                    distance_squared = (
                        dx * dx
                        + dy * dy
                        + dz * dz
                    )

                    if (
                        distance_squared
                        > tolerance_squared
                        and distance_squared
                        <= cutoff_squared
                    ):
                        count += 1

            # -----------------------------------------------------
            # Pairs with forward neighboring cells.
            #
            # Forward neighbor rule:
            #
            # dx > 0
            # OR
            # dx == 0 and dy > 0
            # OR
            # dx == 0 and dy == 0 and dz > 0
            #
            # This visits every cell-cell pair exactly once.
            # -----------------------------------------------------
            for shift_x in range(
                0,
                2,
            ):
                neighbor_x = cell_x + shift_x

                if (
                    neighbor_x < 0
                    or neighbor_x >= nx
                ):
                    continue

                for shift_y in range(
                    -1,
                    2,
                ):
                    neighbor_y = cell_y + shift_y

                    if (
                        neighbor_y < 0
                        or neighbor_y >= ny
                    ):
                        continue

                    for shift_z in range(
                        -1,
                        2,
                    ):
                        neighbor_z = cell_z + shift_z

                        if (
                            neighbor_z < 0
                            or neighbor_z >= nz
                        ):
                            continue

                        # Skip current cell. It was already handled above.
                        if (
                            shift_x == 0
                            and shift_y == 0
                            and shift_z == 0
                        ):
                            continue

                        # Only retain forward cell offsets.
                        if shift_x == 0:
                            if shift_y < 0:
                                continue

                            if (
                                shift_y == 0
                                and shift_z <= 0
                            ):
                                continue

                        neighbor_cell_id = (
                            neighbor_x
                            * yz_size
                            + neighbor_y
                            * nz
                            + neighbor_z
                        )

                        neighbor_start = cell_offsets[
                            neighbor_cell_id
                        ]

                        neighbor_stop = cell_offsets[
                            neighbor_cell_id + 1
                        ]

                        if neighbor_stop <= neighbor_start:
                            continue

                        for first_position in range(
                            start,
                            stop,
                        ):
                            atom_i = sorted_atom_indices[
                                first_position
                            ]

                            xi = positions[
                                atom_i,
                                0,
                            ]

                            yi = positions[
                                atom_i,
                                1,
                            ]

                            zi = positions[
                                atom_i,
                                2,
                            ]

                            for second_position in range(
                                neighbor_start,
                                neighbor_stop,
                            ):
                                atom_j = sorted_atom_indices[
                                    second_position
                                ]

                                dx = positions[
                                    atom_j,
                                    0,
                                ] - xi

                                dy = positions[
                                    atom_j,
                                    1,
                                ] - yi

                                dz = positions[
                                    atom_j,
                                    2,
                                ] - zi

                                distance_squared = (
                                    dx * dx
                                    + dy * dy
                                    + dz * dz
                                )

                                if (
                                    distance_squared
                                    > tolerance_squared
                                    and distance_squared
                                    <= cutoff_squared
                                ):
                                    count += 1

            counts[
                cell_id
            ] = count

        return counts


    @njit(
        cache=True,
        fastmath=True,
        parallel=True,
    )
    def _fill_cell_list_pairs_numba(
        positions,
        sorted_atom_indices,
        cell_offsets,
        pair_offsets,
        nx,
        ny,
        nz,
        cutoff_squared,
        tolerance_squared,
    ):
        """
        Fill the exact unordered pair array counted by
        _count_cell_list_pairs_numba().
        """
        number_of_cells = int(
            cell_offsets.size - 1
        )

        total_pairs = int(
            pair_offsets[
                -1
            ]
        )

        output = np.empty(
            (
                total_pairs,
                2,
            ),
            dtype=np.int32,
        )

        yz_size = ny * nz

        for cell_id in prange(
            number_of_cells
        ):
            start = cell_offsets[
                cell_id
            ]

            stop = cell_offsets[
                cell_id + 1
            ]

            if stop <= start:
                continue

            output_position = pair_offsets[
                cell_id
            ]

            cell_x = cell_id // yz_size
            remainder = cell_id - cell_x * yz_size
            cell_y = remainder // nz
            cell_z = remainder - cell_y * nz

            # -----------------------------------------------------
            # Same-cell pairs.
            # -----------------------------------------------------
            for first_position in range(
                start,
                stop,
            ):
                atom_i = sorted_atom_indices[
                    first_position
                ]

                xi = positions[
                    atom_i,
                    0,
                ]

                yi = positions[
                    atom_i,
                    1,
                ]

                zi = positions[
                    atom_i,
                    2,
                ]

                for second_position in range(
                    first_position + 1,
                    stop,
                ):
                    atom_j = sorted_atom_indices[
                        second_position
                    ]

                    dx = positions[
                        atom_j,
                        0,
                    ] - xi

                    dy = positions[
                        atom_j,
                        1,
                    ] - yi

                    dz = positions[
                        atom_j,
                        2,
                    ] - zi

                    distance_squared = (
                        dx * dx
                        + dy * dy
                        + dz * dz
                    )

                    if (
                        distance_squared
                        > tolerance_squared
                        and distance_squared
                        <= cutoff_squared
                    ):
                        output[
                            output_position,
                            0,
                        ] = atom_i

                        output[
                            output_position,
                            1,
                        ] = atom_j

                        output_position += 1

            # -----------------------------------------------------
            # Forward neighboring cells.
            # -----------------------------------------------------
            for shift_x in range(
                0,
                2,
            ):
                neighbor_x = cell_x + shift_x

                if (
                    neighbor_x < 0
                    or neighbor_x >= nx
                ):
                    continue

                for shift_y in range(
                    -1,
                    2,
                ):
                    neighbor_y = cell_y + shift_y

                    if (
                        neighbor_y < 0
                        or neighbor_y >= ny
                    ):
                        continue

                    for shift_z in range(
                        -1,
                        2,
                    ):
                        neighbor_z = cell_z + shift_z

                        if (
                            neighbor_z < 0
                            or neighbor_z >= nz
                        ):
                            continue

                        if (
                            shift_x == 0
                            and shift_y == 0
                            and shift_z == 0
                        ):
                            continue

                        if shift_x == 0:
                            if shift_y < 0:
                                continue

                            if (
                                shift_y == 0
                                and shift_z <= 0
                            ):
                                continue

                        neighbor_cell_id = (
                            neighbor_x
                            * yz_size
                            + neighbor_y
                            * nz
                            + neighbor_z
                        )

                        neighbor_start = cell_offsets[
                            neighbor_cell_id
                        ]

                        neighbor_stop = cell_offsets[
                            neighbor_cell_id + 1
                        ]

                        if neighbor_stop <= neighbor_start:
                            continue

                        for first_position in range(
                            start,
                            stop,
                        ):
                            atom_i = sorted_atom_indices[
                                first_position
                            ]

                            xi = positions[
                                atom_i,
                                0,
                            ]

                            yi = positions[
                                atom_i,
                                1,
                            ]

                            zi = positions[
                                atom_i,
                                2,
                            ]

                            for second_position in range(
                                neighbor_start,
                                neighbor_stop,
                            ):
                                atom_j = sorted_atom_indices[
                                    second_position
                                ]

                                dx = positions[
                                    atom_j,
                                    0,
                                ] - xi

                                dy = positions[
                                    atom_j,
                                    1,
                                ] - yi

                                dz = positions[
                                    atom_j,
                                    2,
                                ] - zi

                                distance_squared = (
                                    dx * dx
                                    + dy * dy
                                    + dz * dz
                                )

                                if (
                                    distance_squared
                                    > tolerance_squared
                                    and distance_squared
                                    <= cutoff_squared
                                ):
                                    output[
                                        output_position,
                                        0,
                                    ] = atom_i

                                    output[
                                        output_position,
                                        1,
                                    ] = atom_j

                                    output_position += 1

        return output


else:
    _count_cell_list_pairs_numba = None
    _fill_cell_list_pairs_numba = None


def _finite_unordered_pairs_parallel_cell_list(
    positions: np.ndarray,
    *,
    topology_cutoff: float,
) -> np.ndarray:
    """
    Build unordered finite-particle pairs using a Numba-parallel cell list.

    The cell edge is equal to ``topology_cutoff``. Therefore any valid pair
    must belong either to the same spatial cell or one of the 26 neighboring
    cells.

    The implementation evaluates only the 13 forward neighboring cells, so
    each unordered atom pair appears exactly once.
    """
    positions = np.ascontiguousarray(
        np.asarray(
            positions,
            dtype=np.float64,
        )
    )

    atom_count = int(
        positions.shape[0]
    )

    if atom_count <= 1:
        return np.zeros(
            (
                0,
                2,
            ),
            dtype=np.int32,
        )

    topology_cutoff = float(
        topology_cutoff
    )

    if (
        not np.isfinite(
            topology_cutoff
        )
        or topology_cutoff <= 0.0
    ):
        raise ValueError(
            "topology_cutoff must be positive and finite."
        )

    lower = np.min(
        positions,
        axis=0,
    )

    upper = np.max(
        positions,
        axis=0,
    )

    extent = np.maximum(
        upper - lower,
        0.0,
    )

    cell_counts_xyz = np.maximum(
        1,
        np.floor(
            extent
            / topology_cutoff
        ).astype(
            np.int64
        )
        + 1,
    )

    nx = int(
        cell_counts_xyz[
            0
        ]
    )

    ny = int(
        cell_counts_xyz[
            1
        ]
    )

    nz = int(
        cell_counts_xyz[
            2
        ]
    )

    cell_coordinates = np.floor(
        (
            positions
            - lower[
                None,
                :
            ]
        )
        / topology_cutoff
    ).astype(
        np.int64
    )

    cell_coordinates[
        :,
        0,
    ] = np.clip(
        cell_coordinates[
            :,
            0,
        ],
        0,
        nx - 1,
    )

    cell_coordinates[
        :,
        1,
    ] = np.clip(
        cell_coordinates[
            :,
            1,
        ],
        0,
        ny - 1,
    )

    cell_coordinates[
        :,
        2,
    ] = np.clip(
        cell_coordinates[
            :,
            2,
        ],
        0,
        nz - 1,
    )

    cell_id = (
        cell_coordinates[
            :,
            0,
        ]
        * ny
        * nz
        + cell_coordinates[
            :,
            1,
        ]
        * nz
        + cell_coordinates[
            :,
            2,
        ]
    ).astype(
        np.int64,
        copy=False,
    )

    sorted_atom_indices = np.argsort(
        cell_id,
        kind="stable",
    ).astype(
        np.int32,
        copy=False,
    )

    number_of_cells = int(
        nx
        * ny
        * nz
    )

    atoms_per_cell = np.bincount(
        cell_id,
        minlength=number_of_cells,
    ).astype(
        np.int64,
        copy=False,
    )

    cell_offsets = np.zeros(
        number_of_cells + 1,
        dtype=np.int64,
    )

    cell_offsets[
        1:
    ] = np.cumsum(
        atoms_per_cell
    )

    cutoff_squared = (
        topology_cutoff
        * topology_cutoff
    )

    tolerance_squared = 1.0e-16

    counts = _count_cell_list_pairs_numba(
        positions,
        sorted_atom_indices,
        cell_offsets,
        nx,
        ny,
        nz,
        cutoff_squared,
        tolerance_squared,
    )

    pair_offsets = np.zeros(
        counts.size + 1,
        dtype=np.int64,
    )

    pair_offsets[
        1:
    ] = np.cumsum(
        counts
    )

    unordered_pairs = _fill_cell_list_pairs_numba(
        positions,
        sorted_atom_indices,
        cell_offsets,
        pair_offsets,
        nx,
        ny,
        nz,
        cutoff_squared,
        tolerance_squared,
    )

    return np.asarray(
        unordered_pairs,
        dtype=np.int32,
    )

def _layer_transition_coefficients(
    layers,
    number_of_transitions: int,
) -> np.ndarray:
    """
    Return signed transition counts from realized layer zero to every layer.

    Each column corresponds to one transition in the periodic sequence,
    including the final closure transition.
    """
    number_of_transitions = int(
        number_of_transitions
    )

    if number_of_transitions <= 0:
        raise ValueError(
            "number_of_transitions must be positive."
        )

    coefficients = np.zeros(
        (
            len(
                layers
            ),
            number_of_transitions,
        ),
        dtype=np.int32,
    )

    for layer_index, layer in enumerate(
        layers
    ):
        realized_index = int(
            layer.realized_index
        )

        if realized_index > 0:
            for step_index in range(
                realized_index
            ):
                coefficients[
                    layer_index,
                    step_index
                    % number_of_transitions,
                ] += 1

        elif realized_index < 0:
            for step_index in range(
                realized_index,
                0,
            ):
                coefficients[
                    layer_index,
                    step_index
                    % number_of_transitions,
                ] -= 1

    return coefficients


def _finite_descriptor_rows(
    compiled_geometry: CompiledStackedGeometry,
    reference_geometry: EvaluatedStackedGeometry,
    *,
    topology_cutoff: float,
) -> Dict[str, np.ndarray]:
    positions = np.asarray(
        reference_geometry.atom_cart,
        dtype=float,
    )

    print(
        "[DYNAMIC FINITE PAIRS] "
        f"finite_atoms={positions.shape[0]:,}, "
        f"topology_cutoff={float(topology_cutoff):.6g} Å"
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
        raise RuntimeError(
            "The finite stacked geometry produced no pair topology."
        )

    first = unordered_pairs[
        :,
        0
    ]

    second = unordered_pairs[
        :,
        1
    ]

    center = np.concatenate(
        (
            first,
            second,
        )
    )

    neighbor = np.concatenate(
        (
            second,
            first,
        )
    )

    source_indices = np.asarray(
        compiled_geometry.atom_source_site_indices,
        dtype=np.int32,
    )

    layer_descriptor_indices = np.asarray(
        compiled_geometry.atom_layer_descriptor_indices,
        dtype=np.int32,
    )

    cell_a = np.asarray(
        compiled_geometry.atom_cell_a,
        dtype=np.int32,
    )

    cell_b = np.asarray(
        compiled_geometry.atom_cell_b,
        dtype=np.int32,
    )

    source_alpha = source_indices[
        center
    ]

    source_beta = source_indices[
        neighbor
    ]

    source_fractional = np.asarray(
        compiled_geometry.source_fractional,
        dtype=float,
    ).copy()

    source_fractional[
        :,
        0
    ] %= 1.0

    source_fractional[
        :,
        1
    ] %= 1.0

    delta_fractional = (
        source_fractional[
            source_beta
        ]
        - source_fractional[
            source_alpha
        ]
    )

    center_layer = layer_descriptor_indices[
        center
    ]

    neighbor_layer = layer_descriptor_indices[
        neighbor
    ]

    number_of_transitions = len(
        compiled_geometry.stack_spec.sequence
    )

    layer_coefficients = (
        _layer_transition_coefficients(
            compiled_geometry.layers,
            number_of_transitions,
        )
    )

    transition_coefficients = (
        layer_coefficients[
            neighbor_layer
        ]
        - layer_coefficients[
            center_layer
        ]
    )

    # Only in-plane unit-cell translations remain static.
    # Layer translations are represented by transition_coefficients.
    delta_a = (
        cell_a[
            neighbor
        ].astype(
            np.float64,
            copy=False,
        )
        - cell_a[
            center
        ].astype(
            np.float64,
            copy=False,
        )
    )

    delta_b = (
        cell_b[
            neighbor
        ].astype(
            np.float64,
            copy=False,
        )
        - cell_b[
            center
        ].astype(
            np.float64,
            copy=False,
        )
    )

    return {
        "source_alpha": source_alpha,
        "source_beta": source_beta,
        "delta_fractional": delta_fractional,
        "delta_a": delta_a,
        "delta_b": delta_b,
        "transition_coefficients": np.asarray(
            transition_coefficients,
            dtype=np.int32,
        ),
        "number_of_rows": int(
            center.size
        ),
    }


def _compile_bulk_atoms(
    compiled_geometry: CompiledStackedGeometry,
    reference_geometry: EvaluatedStackedGeometry,
) -> Dict[str, np.ndarray]:
    spec = compiled_geometry.stack_spec.normalized()

    absolute_sequence = absolute_sequence_layers(
        spec
    )

    enabled_rows = [
        (
            row,
            layer,
        )
        for row, layer in enumerate(
            absolute_sequence
        )
        if bool(
            layer.enabled
        )
    ]

    source_fractional = np.asarray(
        compiled_geometry.source_fractional,
        dtype=float,
    ).copy()

    source_fractional[
        :,
        0
    ] %= 1.0

    source_fractional[
        :,
        1
    ] %= 1.0

    source_count = int(
        source_fractional.shape[0]
    )

    sequence_indices = []
    source_indices = []
    shift_u = []
    shift_v = []
    stack_s = []
    normal_delta = []

    for sequence_index, layer in enabled_rows:
        for source_index in range(
            source_count
        ):
            sequence_indices.append(
                sequence_index
            )

            source_indices.append(
                source_index
            )

            shift_u.append(
                float(
                    layer.shift_u
                )
            )

            shift_v.append(
                float(
                    layer.shift_v
                )
            )

            stack_s.append(
                float(
                    layer.stack_coordinate
                )
            )

            normal_delta.append(
                float(
                    layer.normal_delta
                )
            )

    source_indices = np.asarray(
        source_indices,
        dtype=np.int32,
    )

    shift_u = np.asarray(
        shift_u,
        dtype=float,
    )

    shift_v = np.asarray(
        shift_v,
        dtype=float,
    )

    stack_s = np.asarray(
        stack_s,
        dtype=float,
    )

    normal_delta = np.asarray(
        normal_delta,
        dtype=float,
    )

    lattice_matrix = np.asarray(
        reference_geometry.parent_lattice_matrix,
        dtype=float,
    )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    base_cart = (
        source_fractional
        @ lattice_matrix
    )

    base_normal = (
        base_cart
        @ normal
    )

    base_cart -= (
        0.5
        * (
            float(
                np.min(
                    base_normal
                )
            )
            + float(
                np.max(
                    base_normal
                )
            )
        )
        * normal[
            None,
            :
        ]
    )

    positions = (
        base_cart[
            source_indices
        ]
        + shift_u[
            :,
            None
        ]
        * vector_a[
            None,
            :
        ]
        + shift_v[
            :,
            None
        ]
        * vector_b[
            None,
            :
        ]
        + (
            stack_s
            * reference_geometry.spacing
            + normal_delta
        )[
            :,
            None
        ]
        * normal[
            None,
            :
        ]
    )

    fractional_stack = (
        positions
        @ reference_geometry.stacking_inverse_lattice_matrix
    )

    fractional_stack %= 1.0

    return {
        "sequence_index": np.asarray(
            sequence_indices,
            dtype=np.int32,
        ),
        "source_index": source_indices,
        "shift_u": shift_u,
        "shift_v": shift_v,
        "stack_s": stack_s,
        "normal_delta": normal_delta,
        "frac_stack": fractional_stack,
    }


def _bulk_descriptor_rows(
    compiled_geometry: CompiledStackedGeometry,
    reference_geometry: EvaluatedStackedGeometry,
    *,
    topology_cutoff: float,
) -> Dict[str, np.ndarray]:
    bulk_atoms = _compile_bulk_atoms(
        compiled_geometry,
        reference_geometry,
    )

    source_structure = compiled_geometry.source_structure

    source_species = [
        site.species
        for site in source_structure.sites
    ]

    species = [
        source_species[
            int(source_index)
        ]
        for source_index in bulk_atoms[
            "source_index"
        ]
    ]

    bulk_structure = Structure(
        lattice=Lattice(
            reference_geometry.stacking_lattice_matrix
        ),
        species=species,
        coords=bulk_atoms[
            "frac_stack"
        ],
        coords_are_cartesian=False,
        to_unit_cell=False,
        validate_proximity=False,
    )

    (
        center,
        neighbor,
        images,
        _distance,
    ) = bulk_structure.get_neighbor_list(
        r=float(
            topology_cutoff
        ),
        numerical_tol=1.0e-8,
        exclude_self=True,
    )

    center = np.asarray(
        center,
        dtype=np.int32,
    )

    neighbor = np.asarray(
        neighbor,
        dtype=np.int32,
    )

    images = np.asarray(
        images,
        dtype=np.int32,
    ).reshape(
        -1,
        3,
    )

    if center.size == 0:
        raise RuntimeError(
            "The periodic stacked geometry produced no pair topology."
        )

    source_alpha = bulk_atoms[
        "source_index"
    ][
        center
    ]

    source_beta = bulk_atoms[
        "source_index"
    ][
        neighbor
    ]

    source_fractional = np.asarray(
        compiled_geometry.source_fractional,
        dtype=float,
    ).copy()

    source_fractional[
        :,
        0
    ] %= 1.0

    source_fractional[
        :,
        1
    ] %= 1.0

    delta_fractional = (
        source_fractional[
            source_beta
        ]
        - source_fractional[
            source_alpha
        ]
    )

    number_of_transitions = len(
        compiled_geometry.stack_spec.sequence
    )

    sequence_index = np.asarray(
        bulk_atoms[
            "sequence_index"
        ],
        dtype=np.int32,
    )

    prefix_coefficients = np.zeros(
        (
            number_of_transitions,
            number_of_transitions,
        ),
        dtype=np.int32,
    )

    for sequence_row in range(
        number_of_transitions
    ):
        if sequence_row > 0:
            prefix_coefficients[
                sequence_row,
                :sequence_row,
            ] = 1

    transition_coefficients = (
        prefix_coefficients[
            sequence_index[
                neighbor
            ]
        ]
        - prefix_coefficients[
            sequence_index[
                center
            ]
        ]
    )

    # Moving by one stacking-supercell c image traverses one complete
    # sequence period, so every transition is crossed once.
    transition_coefficients += (
        images[
            :,
            2
        ][
            :,
            None
        ]
    )

    delta_a = images[
        :,
        0
    ].astype(
        np.float64
    )

    delta_b = images[
        :,
        1
    ].astype(
        np.float64
    )

    return {
        "source_alpha": source_alpha,
        "source_beta": source_beta,
        "delta_fractional": delta_fractional,
        "delta_a": delta_a,
        "delta_b": delta_b,
        "transition_coefficients": np.asarray(
            transition_coefficients,
            dtype=np.int32,
        ),
        "number_of_rows": int(
            center.size
        ),
    }

def _reduce_direction(
    h: int,
    k: int,
    l: int,
) -> Tuple[int, int, int]:
    divisor = int(
        np.gcd(
            np.gcd(
                abs(
                    int(h)
                ),
                abs(
                    int(k)
                ),
            ),
            abs(
                int(l)
            ),
        )
    )

    if divisor == 0:
        return (
            0,
            0,
            0,
        )

    h = int(
        h
    ) // divisor

    k = int(
        k
    ) // divisor

    l = int(
        l
    ) // divisor

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


def _legacy_reference_hkl_and_direction(
    shell_cart: np.ndarray,
    stacking_lattice_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lattice = Lattice(
        stacking_lattice_matrix
    )

    reciprocal_transpose = np.asarray(
        lattice.reciprocal_lattice.matrix.T,
        dtype=float,
    )

    inverse_reciprocal_transpose = np.linalg.inv(
        reciprocal_transpose
    )

    hkl_float = (
        shell_cart
        @ inverse_reciprocal_transpose.T
    )

    hkl_int = np.rint(
        hkl_float
    ).astype(
        np.int32,
    )

    directions = np.zeros_like(
        hkl_int,
        dtype=np.int32,
    )

    for row in range(
        hkl_int.shape[0]
    ):
        directions[
            row
        ] = _reduce_direction(
            hkl_int[
                row,
                0,
            ],
            hkl_int[
                row,
                1,
            ],
            hkl_int[
                row,
                2,
            ],
        )

    integer_minimum = np.iinfo(
        np.int16
    ).min

    integer_maximum = np.iinfo(
        np.int16
    ).max

    return (
        np.clip(
            hkl_int,
            integer_minimum,
            integer_maximum,
        ).astype(
            np.int16,
        ),
        np.clip(
            directions,
            integer_minimum,
            integer_maximum,
        ).astype(
            np.int16,
        ),
    )


def _species_metadata(
    compiled_geometry: CompiledStackedGeometry,
) -> Dict[str, Any]:
    source_symbols = [
        _clean_element_symbol(
            value
        )
        for value in compiled_geometry.source_elements
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

    finite_source = np.asarray(
        compiled_geometry.atom_source_site_indices,
        dtype=np.int32,
    )

    finite_species_counts = np.bincount(
        source_species_id[
            finite_source
        ].astype(
            np.int64
        ),
        minlength=len(
            unique_species
        ),
    ).astype(
        np.int64,
    )

    enabled_sequence_count = sum(
        1
        for layer in compiled_geometry.stack_spec.sequence
        if bool(
            layer.enabled
        )
    )

    source_species_counts = np.bincount(
        source_species_id.astype(
            np.int64
        ),
        minlength=len(
            unique_species
        ),
    ).astype(
        np.int64,
    )

    bulk_species_counts = (
        source_species_counts
        * int(
            enabled_sequence_count
        )
    )

    return {
        "unique_species": unique_species,
        "source_species_id": source_species_id,
        "finite_species_counts": (
            finite_species_counts
        ),
        "bulk_species_counts": bulk_species_counts,
    }


def compile_stacked_pair_topology(
    *,
    geometry: CompiledStackedGeometry,
    pair_cutoff: float,
    topology_margin: float = 1.0,
    reference_parameters: Optional[
        Mapping[str, Any]
    ] = None,
    shell_backend: str = "cpu",
    gpu_device: int = 0,
    gpu_memory_fraction: float = 0.70,
) -> CompiledStackedPairTopology:
    """
    Compile finite and periodic pair topology once.

    Parameters
    ----------
    geometry
        Compiled Phase-A finite stacking geometry.

    pair_cutoff
        Physical PDF pair cutoff.

    topology_margin
        Extra reference-distance margin used when compiling candidate pairs.

        Pair topology remains valid only while geometry changes are small
        enough that a pair initially farther than topology_cutoff cannot move
        inside pair_cutoff.
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

    if pair_cutoff <= 0.0:
        raise ValueError(
            "pair_cutoff must be positive."
        )

    topology_cutoff = (
        pair_cutoff
        + topology_margin
    )

    compile_start = time.perf_counter()

    reference_parameters = dict(
        reference_parameters
        or {}
    )

    reference_geometry_start = time.perf_counter()

    reference_geometry = geometry.evaluate(
        reference_parameters
    )

    reference_geometry_seconds = (
        time.perf_counter()
        - reference_geometry_start
    )

    finite_rows_start = time.perf_counter()

    finite_rows = _finite_descriptor_rows(
        geometry,
        reference_geometry,
        topology_cutoff=topology_cutoff,
    )

    finite_rows_seconds = (
        time.perf_counter()
        - finite_rows_start
    )

    bulk_rows_start = time.perf_counter()

    bulk_rows = _bulk_descriptor_rows(
        geometry,
        reference_geometry,
        topology_cutoff=topology_cutoff,
    )

    bulk_rows_seconds = (
        time.perf_counter()
        - bulk_rows_start
    )

    descriptor_keys_start = time.perf_counter()

    finite_matrix = _make_descriptor_matrix(
        source_alpha=finite_rows[
            "source_alpha"
        ],
        source_beta=finite_rows[
            "source_beta"
        ],
        delta_fractional=finite_rows[
            "delta_fractional"
        ],
        delta_a=finite_rows[
            "delta_a"
        ],
        delta_b=finite_rows[
            "delta_b"
        ],
        transition_coefficients=finite_rows[
            "transition_coefficients"
        ],
    )

    bulk_matrix = _make_descriptor_matrix(
        source_alpha=bulk_rows[
            "source_alpha"
        ],
        source_beta=bulk_rows[
            "source_beta"
        ],
        delta_fractional=bulk_rows[
            "delta_fractional"
        ],
        delta_a=bulk_rows[
            "delta_a"
        ],
        delta_b=bulk_rows[
            "delta_b"
        ],
        transition_coefficients=bulk_rows[
            "transition_coefficients"
        ],
    )

    descriptor_keys_seconds = (
        time.perf_counter()
        - descriptor_keys_start
    )

    finite_unique_start = time.perf_counter()

    (
        finite_unique,
        finite_counts,
        finite_unique_packed,
    ) = _unique_descriptor_matrix(
        finite_matrix
    )

    finite_unique_seconds = (
        time.perf_counter()
        - finite_unique_start
    )

    bulk_unique_start = time.perf_counter()

    (
        bulk_unique,
        bulk_counts,
        bulk_unique_packed,
    ) = _unique_descriptor_matrix(
        bulk_matrix
    )

    bulk_unique_seconds = (
        time.perf_counter()
        - bulk_unique_start
    )

    descriptor_match_start = time.perf_counter()

    bulk_position = np.searchsorted(
        bulk_unique_packed,
        finite_unique_packed,
        side="left",
    )

    matched = (
        bulk_position
        < bulk_unique_packed.size
    )

    matched_indices = np.flatnonzero(
        matched
    )

    if matched_indices.size:
        matched[
            matched_indices
        ] &= (
            bulk_unique_packed[
                bulk_position[
                    matched_indices
                ]
            ]
            == finite_unique_packed[
                matched_indices
            ]
        )

    retained_matrix = finite_unique[
        matched
    ]

    retained_finite_counts = finite_counts[
        matched
    ]

    retained_bulk_counts = bulk_counts[
        bulk_position[
            matched
        ]
    ]

    descriptor_match_seconds = (
        time.perf_counter()
        - descriptor_match_start
    )

    if retained_matrix.shape[0] == 0:
        raise RuntimeError(
            "No finite pair descriptors matched the periodic bulk topology."
        )

    decoded = _decode_descriptor_matrix(
        retained_matrix
    )

    species_metadata = _species_metadata(
        geometry
    )

    source_species_id = species_metadata[
        "source_species_id"
    ]

    shell_alpha_id = source_species_id[
        decoded[
            "source_alpha"
        ]
    ]

    shell_beta_id = source_species_id[
        decoded[
            "source_beta"
        ]
    ]

    (
        _reference_transitions,
        reference_transition_du,
        reference_transition_dv,
        reference_transition_h,
    ) = evaluate_transition_values(
        geometry.stack_spec,
        reference_parameters,
    )

    temporary_topology = CompiledStackedPairTopology(
        geometry=geometry,
        pair_cutoff=pair_cutoff,
        topology_cutoff=topology_cutoff,
        shell_source_alpha=decoded[
            "source_alpha"
        ],
        shell_source_beta=decoded[
            "source_beta"
        ],
        shell_delta_fractional=decoded[
            "delta_fractional"
        ],
        shell_delta_a=decoded[
            "delta_a"
        ],
        shell_delta_b=decoded[
            "delta_b"
        ],
        shell_mult=np.asarray(
            retained_finite_counts,
            dtype=np.int32,
        ),
        shell_bulk_mult=np.asarray(
            retained_bulk_counts,
            dtype=np.int32,
        ),
        shell_alpha_id=np.asarray(
            shell_alpha_id,
            dtype=np.int16,
        ),
        shell_beta_id=np.asarray(
            shell_beta_id,
            dtype=np.int16,
        ),
        finite_species_counts=np.asarray(
            species_metadata[
                "finite_species_counts"
            ],
            dtype=np.int64,
        ),
        bulk_species_counts=np.asarray(
            species_metadata[
                "bulk_species_counts"
            ],
            dtype=np.int64,
        ),
        unique_species=list(
            species_metadata[
                "unique_species"
            ]
        ),
        shell_hkl=np.zeros(
            (
                retained_matrix.shape[
                    0
                ],
                3,
            ),
            dtype=np.int16,
        ),
        shell_direction=np.zeros(
            (
                retained_matrix.shape[
                    0
                ],
                3,
            ),
            dtype=np.int16,
        ),
        reference_distance=np.zeros(
            retained_matrix.shape[
                0
            ],
            dtype=np.float64,
        ),
        finite_pair_rows=int(
            finite_rows[
                "number_of_rows"
            ]
        ),
        bulk_pair_rows=int(
            bulk_rows[
                "number_of_rows"
            ]
        ),
        shell_transition_coefficients=np.asarray(
            decoded[
                "transition_coefficients"
            ],
            dtype=np.int32,
        ),
        reference_transition_du=np.asarray(
            reference_transition_du,
            dtype=np.float64,
        ),
        reference_transition_dv=np.asarray(
            reference_transition_dv,
            dtype=np.float64,
        ),
        reference_transition_h=np.asarray(
            reference_transition_h,
            dtype=np.float64,
        ),
    )

    final_reference_start = time.perf_counter()

    evaluated_reference = temporary_topology.evaluate(
        reference_parameters
    )

    shell_hkl, shell_direction = (
        _reference_hkl_and_direction(
            evaluated_reference.shell_cart,
            evaluated_reference.geometry.stacking_lattice_matrix,
        )
    )

    final_reference_seconds = (
        time.perf_counter()
        - final_reference_start
    )

    temporary_topology.shell_hkl = shell_hkl
    temporary_topology.shell_direction = (
        shell_direction
    )
    temporary_topology.reference_distance = (
        evaluated_reference.shell_distance
    )

    total_seconds = (
        time.perf_counter()
        - compile_start
    )

    print(
        "[DYNAMIC TOPOLOGY PROFILE] "
        f"reference_geometry={reference_geometry_seconds:.3f} s, "
        f"finite_pairs={finite_rows_seconds:.3f} s, "
        f"bulk_pairs={bulk_rows_seconds:.3f} s, "
        f"descriptor_keys={descriptor_keys_seconds:.3f} s, "
        f"finite_unique={finite_unique_seconds:.3f} s, "
        f"bulk_unique={bulk_unique_seconds:.3f} s, "
        f"descriptor_match={descriptor_match_seconds:.3f} s, "
        f"reference_hkl={final_reference_seconds:.3f} s, "
        f"finite_rows={int(finite_rows['number_of_rows']):,}, "
        f"bulk_rows={int(bulk_rows['number_of_rows']):,}, "
        f"shells={retained_matrix.shape[0]:,}, "
        f"total={total_seconds:.3f} s"
    )

    return temporary_topology


def _reference_hkl_and_direction(cartesian, lattice_matrix):
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
