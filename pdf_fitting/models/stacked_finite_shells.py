from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import re
import time

import numpy as np

from scipy.spatial import cKDTree

from pymatgen.core import Lattice, Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
)

from pdf_fitting.models.stacked_particle import (
    StackedParticle,
    build_stacked_particle,
)

from pdf_fitting.models.stacking_supercell import (
    StackingSupercell,
    build_stacking_supercell,
)


_DISTANCE_SCALE = 1.0e5
_ZERO_TOLERANCE = 1.0e-8


@dataclass
class StackedFiniteShellResult:
    """
    Fixed-geometry result for a finite stacked crystallite.

    shell_table
        PDFCalculator-compatible packed shell table.

    finite_particle
        Explicit N-layer finite particle used to calculate finite pair counts.

    bulk_supercell
        Infinite periodic stacking-supercell reference used to calculate bulk
        pair counts and crystallographic directions.

    build_seconds
        Total wall-clock time.

    finite_pair_rows
        Number of directed finite-particle pair rows before grouping.

    bulk_pair_rows
        Number of directed periodic-supercell pair rows before grouping.
    """

    shell_table: Dict[str, Any]
    finite_particle: StackedParticle
    bulk_supercell: StackingSupercell

    build_seconds: float

    finite_pair_rows: int
    bulk_pair_rows: int

    def to_summary_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "number_of_finite_atoms": int(
                self.finite_particle.atom_cart.shape[0]
            ),
            "number_of_layer_positions": int(
                self.finite_particle.number_of_layer_positions
            ),
            "number_of_enabled_layers": int(
                self.finite_particle.number_of_enabled_layers
            ),
            "number_of_bulk_atoms": int(
                self.bulk_supercell.number_of_atoms
            ),
            "number_of_shells": int(
                self.shell_table["shell_frac"].shape[0]
            ),
            "finite_pair_rows": int(
                self.finite_pair_rows
            ),
            "bulk_pair_rows": int(
                self.bulk_pair_rows
            ),
            "build_seconds": float(
                self.build_seconds
            ),
            "stacking_geometry_hash": str(
                self.bulk_supercell.geometry_hash
            ),
        }


def clean_element_symbol(
    value: str,
) -> str:
    """
    Extract a clean chemical element symbol.

    Examples
    --------
    C
    C1
    Fe2+
    O0.5
    """
    match = re.match(
        r"([A-Za-z]{1,2})",
        str(
            value
            or ""
        ).strip(),
    )

    if match is None:
        raise ValueError(
            f"Could not extract an element symbol from {value!r}."
        )

    symbol = match.group(
        1
    )

    if len(symbol) == 1:
        return symbol.upper()

    return (
        symbol[0].upper()
        + symbol[1].lower()
    )


def lambda_safe_species_label(
    value: str,
) -> str:
    """
    Return the same kind of pair label used by the existing PDF code.
    """
    symbol = clean_element_symbol(
        value
    )

    text = str(
        symbol
    ).strip()

    text = text.replace(
        "+",
        "plus",
    )

    text = text.replace(
        "-",
        "minus",
    )

    text = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        text,
    )

    return text.lower()


def _reduce_direction(
    h: int,
    k: int,
    l: int,
) -> Tuple[int, int, int]:
    """
    Reduce an integer direction and make the first non-zero index positive.

    Opposite pair vectors therefore belong to the same direction family.
    """
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


def _legacy_pair_hkl_and_direction(
    cartesian_vectors: np.ndarray,
    stacking_lattice_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate nearest crystallographic indices and reduced directions using
    the stacking-supercell reciprocal lattice.

    This follows the convention used by the existing shell generators:

        reciprocal_T = reciprocal_lattice.matrix.T
        hkl_float = cart @ inverse(reciprocal_T).T

    The stacking supercell, rather than the parent CIF cell, defines the
    reciprocal basis.
    """
    cartesian_vectors = np.asarray(
        cartesian_vectors,
        dtype=float,
    )

    if cartesian_vectors.shape[0] == 0:
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
            stacking_lattice_matrix,
            dtype=float,
        )
    )

    reciprocal_transpose = np.asarray(
        lattice.reciprocal_lattice.matrix.T,
        dtype=float,
    )

    inverse_reciprocal_transpose = np.linalg.inv(
        reciprocal_transpose
    )

    hkl_float = (
        cartesian_vectors
        @ inverse_reciprocal_transpose.T
    )

    hkl_int32 = np.rint(
        hkl_float
    ).astype(
        np.int32,
    )

    directions_int32 = np.zeros_like(
        hkl_int32,
        dtype=np.int32,
    )

    for row in range(
        hkl_int32.shape[0]
    ):
        direction = _reduce_direction(
            int(
                hkl_int32[
                    row,
                    0,
                ]
            ),
            int(
                hkl_int32[
                    row,
                    1,
                ]
            ),
            int(
                hkl_int32[
                    row,
                    2,
                ]
            ),
        )

        directions_int32[
            row,
            :
        ] = direction

    int16_min = np.iinfo(
        np.int16
    ).min

    int16_max = np.iinfo(
        np.int16
    ).max

    hkl = np.clip(
        hkl_int32,
        int16_min,
        int16_max,
    ).astype(
        np.int16,
    )

    direction = np.clip(
        directions_int32,
        int16_min,
        int16_max,
    ).astype(
        np.int16,
    )

    return (
        hkl,
        direction,
    )


def _finite_directed_pair_rows(
    particle: StackedParticle,
    *,
    pair_cutoff: float,
) -> Dict[str, np.ndarray]:
    """
    Generate all directed finite-particle pair rows within pair_cutoff.

    cKDTree first returns each unordered pair once. Every pair is then expanded
    into both directed rows:

        i -> j
        j -> i

    This matches the ordered-pair normalization used by PDFCalculator.
    """
    positions = np.asarray(
        particle.atom_cart,
        dtype=float,
    )

    atom_count = int(
        positions.shape[0]
    )

    empty = {
        "center": np.zeros(
            0,
            dtype=np.int32,
        ),
        "neighbor": np.zeros(
            0,
            dtype=np.int32,
        ),
        "cart": np.zeros(
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
    }

    if atom_count <= 1:
        return empty

    tree = cKDTree(
        positions
    )

    unordered_pairs = tree.query_pairs(
        r=float(
            pair_cutoff
        ),
        output_type="ndarray",
    )

    unordered_pairs = np.asarray(
        unordered_pairs,
        dtype=np.int64,
    ).reshape(
        -1,
        2,
    )

    if unordered_pairs.shape[0] == 0:
        return empty

    first = unordered_pairs[
        :,
        0,
    ]

    second = unordered_pairs[
        :,
        1,
    ]

    forward_vectors = (
        positions[
            second
        ]
        - positions[
            first
        ]
    )

    distances = np.sqrt(
        np.einsum(
            "ij,ij->i",
            forward_vectors,
            forward_vectors,
        )
    )

    valid = (
        np.isfinite(
            distances
        )
        & (
            distances
            > _ZERO_TOLERANCE
        )
        & (
            distances
            <= float(
                pair_cutoff
            )
            + 1.0e-10
        )
    )

    first = first[
        valid
    ]

    second = second[
        valid
    ]

    forward_vectors = forward_vectors[
        valid
    ]

    distances = distances[
        valid
    ]

    number_of_unordered_pairs = int(
        first.size
    )

    center = np.empty(
        2
        * number_of_unordered_pairs,
        dtype=np.int32,
    )

    neighbor = np.empty(
        2
        * number_of_unordered_pairs,
        dtype=np.int32,
    )

    cartesian = np.empty(
        (
            2
            * number_of_unordered_pairs,
            3,
        ),
        dtype=np.float64,
    )

    directed_distances = np.empty(
        2
        * number_of_unordered_pairs,
        dtype=np.float64,
    )

    center[
        0::2
    ] = first.astype(
        np.int32,
    )

    neighbor[
        0::2
    ] = second.astype(
        np.int32,
    )

    cartesian[
        0::2
    ] = forward_vectors

    directed_distances[
        0::2
    ] = distances

    center[
        1::2
    ] = second.astype(
        np.int32,
    )

    neighbor[
        1::2
    ] = first.astype(
        np.int32,
    )

    cartesian[
        1::2
    ] = -forward_vectors

    directed_distances[
        1::2
    ] = distances

    return {
        "center": center,
        "neighbor": neighbor,
        "cart": cartesian,
        "distance": directed_distances,
    }


def _bulk_periodic_pair_rows(
    supercell: StackingSupercell,
    *,
    pair_cutoff: float,
) -> Dict[str, np.ndarray]:
    """
    Generate directed periodic pair rows from the stacking supercell.

    Every supercell atom acts as a center atom. Neighbor atoms may belong to
    periodic images of the stacking supercell.
    """
    structure = supercell.structure

    (
        center_indices,
        neighbor_indices,
        image_vectors,
        distances,
    ) = structure.get_neighbor_list(
        r=float(
            pair_cutoff
        ),
        numerical_tol=_ZERO_TOLERANCE,
        exclude_self=True,
    )

    center_indices = np.asarray(
        center_indices,
        dtype=np.int32,
    )

    neighbor_indices = np.asarray(
        neighbor_indices,
        dtype=np.int32,
    )

    image_vectors = np.asarray(
        image_vectors,
        dtype=np.int32,
    ).reshape(
        -1,
        3,
    )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    if center_indices.size == 0:
        return {
            "center": center_indices,
            "neighbor": neighbor_indices,
            "image": image_vectors,
            "frac": np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            ),
            "cart": np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            ),
            "distance": distances,
        }

    fractional = np.asarray(
        structure.frac_coords,
        dtype=np.float64,
    )

    pair_fractional = (
        fractional[
            neighbor_indices
        ]
        + image_vectors.astype(
            np.float64
        )
        - fractional[
            center_indices
        ]
    )

    pair_cartesian = (
        pair_fractional
        @ supercell.lattice_matrix
    )

    recalculated_distances = np.sqrt(
        np.einsum(
            "ij,ij->i",
            pair_cartesian,
            pair_cartesian,
        )
    )

    valid = (
        np.isfinite(
            recalculated_distances
        )
        & (
            recalculated_distances
            > _ZERO_TOLERANCE
        )
        & (
            recalculated_distances
            <= float(
                pair_cutoff
            )
            + 1.0e-10
        )
    )

    return {
        "center": center_indices[
            valid
        ],
        "neighbor": neighbor_indices[
            valid
        ],
        "image": image_vectors[
            valid
        ],
        "frac": pair_fractional[
            valid
        ],
        "cart": pair_cartesian[
            valid
        ],
        "distance": recalculated_distances[
            valid
        ],
    }


def _species_metadata(
    finite_particle: StackedParticle,
    bulk_supercell: StackingSupercell,
) -> Dict[str, Any]:
    """
    Build one common element/species index shared by finite and bulk rows.
    """
    finite_symbols = [
        clean_element_symbol(
            value
        )
        for value in finite_particle.atom_elements
    ]

    bulk_symbols = [
        clean_element_symbol(
            value
        )
        for value in bulk_supercell.atom_elements
    ]

    unique_species = sorted(
        set(
            finite_symbols
        )
        | set(
            bulk_symbols
        )
    )

    if not unique_species:
        raise RuntimeError(
            "No chemical species were found in the stacked geometry."
        )

    species_index = {
        symbol: index
        for index, symbol in enumerate(
            unique_species
        )
    }

    finite_species_id = np.asarray(
        [
            species_index[
                symbol
            ]
            for symbol in finite_symbols
        ],
        dtype=np.int16,
    )

    bulk_species_id = np.asarray(
        [
            species_index[
                symbol
            ]
            for symbol in bulk_symbols
        ],
        dtype=np.int16,
    )

    finite_species_counts = np.bincount(
        finite_species_id.astype(
            np.int64
        ),
        minlength=len(
            unique_species
        ),
    ).astype(
        np.int64,
    )

    bulk_species_counts = np.bincount(
        bulk_species_id.astype(
            np.int64
        ),
        minlength=len(
            unique_species
        ),
    ).astype(
        np.int64,
    )

    lambda_labels = [
        lambda_safe_species_label(
            symbol
        )
        for symbol in unique_species
    ]

    return {
        "unique_species": unique_species,
        "species_index": species_index,
        "lambda_labels": lambda_labels,
        "finite_species_id": finite_species_id,
        "bulk_species_id": bulk_species_id,
        "finite_species_counts": finite_species_counts,
        "bulk_species_counts": bulk_species_counts,
    }


def _build_group_key(
    *,
    alpha_id: np.ndarray,
    beta_id: np.ndarray,
    distance: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Build the common finite/bulk grouping key.

    Grouping preserves:

        ordered chemical pair
        distance bin
        reduced stacking-supercell direction
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
        np.int32,
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
) -> Tuple[int, int, int, int, int, int]:
    """
    Convert one structured scalar into a normal dictionary key.
    """
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


def _rank_shells_within_unordered_pairs(
    shell_upair_id: np.ndarray,
    distance_integer: np.ndarray,
) -> np.ndarray:
    """
    Rank unique distances within each unordered chemical pair.
    """
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

    ranks = np.zeros(
        number_of_shells,
        dtype=np.int32,
    )

    if number_of_shells == 0:
        return ranks

    order = np.lexsort(
        (
            distance_integer,
            shell_upair_id,
        )
    )

    sorted_upair = shell_upair_id[
        order
    ]

    sorted_distance = distance_integer[
        order
    ]

    new_pair = np.ones(
        number_of_shells,
        dtype=bool,
    )

    new_distance = np.ones(
        number_of_shells,
        dtype=bool,
    )

    if number_of_shells > 1:
        new_pair[
            1:
        ] = (
            sorted_upair[
                1:
            ]
            != sorted_upair[
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
                sorted_distance[
                    1:
                ]
                != sorted_distance[
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

    sorted_rank = (
        cumulative
        - reset
    )

    ranks[
        order
    ] = sorted_rank

    return ranks


def _assemble_shell_table(
    *,
    finite_rows: Dict[str, np.ndarray],
    bulk_rows: Dict[str, np.ndarray],
    finite_particle: StackedParticle,
    bulk_supercell: StackingSupercell,
    species_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Group finite and bulk pair rows and assemble a PDFCalculator shell table.
    """
    finite_center = np.asarray(
        finite_rows["center"],
        dtype=np.int32,
    )

    finite_neighbor = np.asarray(
        finite_rows["neighbor"],
        dtype=np.int32,
    )

    finite_cart = np.asarray(
        finite_rows["cart"],
        dtype=np.float64,
    )

    finite_distance = np.asarray(
        finite_rows["distance"],
        dtype=np.float64,
    )

    bulk_center = np.asarray(
        bulk_rows["center"],
        dtype=np.int32,
    )

    bulk_neighbor = np.asarray(
        bulk_rows["neighbor"],
        dtype=np.int32,
    )

    bulk_cart = np.asarray(
        bulk_rows["cart"],
        dtype=np.float64,
    )

    bulk_distance = np.asarray(
        bulk_rows["distance"],
        dtype=np.float64,
    )

    if finite_distance.size == 0:
        raise RuntimeError(
            "The finite stacked particle produced no pair rows. "
            "Increase the particle size or pair cutoff."
        )

    if bulk_distance.size == 0:
        raise RuntimeError(
            "The periodic stacking supercell produced no bulk pair rows."
        )

    finite_species_id = np.asarray(
        species_metadata[
            "finite_species_id"
        ],
        dtype=np.int16,
    )

    bulk_species_id = np.asarray(
        species_metadata[
            "bulk_species_id"
        ],
        dtype=np.int16,
    )

    finite_alpha = finite_species_id[
        finite_center
    ]

    finite_beta = finite_species_id[
        finite_neighbor
    ]

    bulk_alpha = bulk_species_id[
        bulk_center
    ]

    bulk_beta = bulk_species_id[
        bulk_neighbor
    ]

    finite_hkl, finite_direction = (
        _pair_hkl_and_direction(
            finite_cart,
            bulk_supercell.lattice_matrix,
        )
    )

    bulk_hkl, bulk_direction = (
        _pair_hkl_and_direction(
            bulk_cart,
            bulk_supercell.lattice_matrix,
        )
    )

    finite_key = _build_group_key(
        alpha_id=finite_alpha,
        beta_id=finite_beta,
        distance=finite_distance,
        direction=finite_direction,
    )

    bulk_key = _build_group_key(
        alpha_id=bulk_alpha,
        beta_id=bulk_beta,
        distance=bulk_distance,
        direction=bulk_direction,
    )

    (
        unique_finite_key,
        finite_first_index,
        finite_inverse,
        finite_counts,
    ) = np.unique(
        finite_key,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    (
        unique_bulk_key,
        bulk_first_index,
        bulk_counts,
    ) = np.unique(
        bulk_key,
        return_index=True,
        return_counts=True,
    )

    bulk_lookup = {
        _key_tuple(
            key
        ): (
            int(
                count
            ),
            int(
                first_index
            ),
        )
        for (
            key,
            count,
            first_index,
        ) in zip(
            unique_bulk_key,
            bulk_counts,
            bulk_first_index,
        )
    }

    number_of_finite_groups = int(
        unique_finite_key.size
    )

    finite_species_counts = np.asarray(
        species_metadata[
            "finite_species_counts"
        ],
        dtype=np.float64,
    )

    bulk_species_counts = np.asarray(
        species_metadata[
            "bulk_species_counts"
        ],
        dtype=np.float64,
    )

    retained_indices = []
    retained_bulk_counts = []
    retained_bulk_first_indices = []

    for group_index in range(
        number_of_finite_groups
    ):
        lookup_value = bulk_lookup.get(
            _key_tuple(
                unique_finite_key[
                    group_index
                ]
            )
        )

        if lookup_value is None:
            continue

        retained_indices.append(
            group_index
        )

        retained_bulk_counts.append(
            lookup_value[0]
        )

        retained_bulk_first_indices.append(
            lookup_value[1]
        )

    if not retained_indices:
        raise RuntimeError(
            "Finite stacked shells could not be matched to the periodic "
            "stacking-supercell bulk reference."
        )

    retained_indices = np.asarray(
        retained_indices,
        dtype=np.int64,
    )

    retained_bulk_counts = np.asarray(
        retained_bulk_counts,
        dtype=np.int64,
    )

    retained_bulk_first_indices = np.asarray(
        retained_bulk_first_indices,
        dtype=np.int64,
    )

    representative_finite_rows = finite_first_index[
        retained_indices
    ]

    finite_raw_counts = finite_counts[
        retained_indices
    ].astype(
        np.int64,
    )

    retained_keys = unique_finite_key[
        retained_indices
    ]

    shell_alpha_id = retained_keys[
        "alpha"
    ].astype(
        np.int16,
    )

    shell_beta_id = retained_keys[
        "beta"
    ].astype(
        np.int16,
    )

    distance_integer = retained_keys[
        "distance"
    ].astype(
        np.int64,
    )

    shell_direction = np.column_stack(
        (
            retained_keys[
                "dh"
            ],
            retained_keys[
                "dk"
            ],
            retained_keys[
                "dl"
            ],
        )
    ).astype(
        np.int16,
    )

    # Use the representative finite pair vector. It belongs to the same
    # periodic bulk shell but preserves the exact finite-particle geometry.
    shell_cartesian = finite_cart[
        representative_finite_rows
    ]

    shell_fractional = (
        shell_cartesian
        @ bulk_supercell.inverse_lattice_matrix
    ).astype(
        np.float32,
    )

    shell_hkl = finite_hkl[
        representative_finite_rows
    ].astype(
        np.int16,
    )

    shell_mult_eff = np.zeros(
        retained_indices.size,
        dtype=np.float32,
    )

    shell_bulk_mult_eff = np.zeros(
        retained_indices.size,
        dtype=np.float32,
    )

    shell_gamma = np.zeros(
        retained_indices.size,
        dtype=np.float32,
    )

    for shell_index in range(
        retained_indices.size
    ):
        alpha_id = int(
            shell_alpha_id[
                shell_index
            ]
        )

        finite_denominator = (
            float(
                finite_species_counts[
                    alpha_id
                ]
            )
            if (
                0
                <= alpha_id
                < finite_species_counts.size
            )
            else 0.0
        )

        bulk_denominator = (
            float(
                bulk_species_counts[
                    alpha_id
                ]
            )
            if (
                0
                <= alpha_id
                < bulk_species_counts.size
            )
            else 0.0
        )

        finite_effective = 0.0
        bulk_effective = 0.0

        if finite_denominator > 0.0:
            finite_effective = (
                float(
                    finite_raw_counts[
                        shell_index
                    ]
                )
                / finite_denominator
            )

        if bulk_denominator > 0.0:
            bulk_effective = (
                float(
                    retained_bulk_counts[
                        shell_index
                    ]
                )
                / bulk_denominator
            )

        gamma = 0.0

        if bulk_effective > 0.0:
            gamma = (
                finite_effective
                / bulk_effective
            )

        shell_mult_eff[
            shell_index
        ] = np.float32(
            finite_effective
        )

        shell_bulk_mult_eff[
            shell_index
        ] = np.float32(
            bulk_effective
        )

        shell_gamma[
            shell_index
        ] = np.float32(
            gamma
        )

    int32_max = np.iinfo(
        np.int32
    ).max

    shell_mult = np.minimum(
        finite_raw_counts,
        int32_max,
    ).astype(
        np.int32,
    )

    shell_bulk_mult = np.minimum(
        retained_bulk_counts,
        int32_max,
    ).astype(
        np.int32,
    )

    unique_species = list(
        species_metadata[
            "unique_species"
        ]
    )

    lambda_labels = list(
        species_metadata[
            "lambda_labels"
        ]
    )

    ordered_pair_values = sorted(
        {
            (
                lambda_labels[
                    int(
                        alpha_id
                    )
                ],
                lambda_labels[
                    int(
                        beta_id
                    )
                ],
            )
            for (
                alpha_id,
                beta_id,
            ) in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        }
    )

    pair_list = list(
        ordered_pair_values
    )

    pair_index = {
        pair: index
        for index, pair in enumerate(
            pair_list
        )
    }

    unordered_pair_values = sorted(
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
        unordered_pair_values
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
                    lambda_labels[
                        int(
                            alpha_id
                        )
                    ],
                    lambda_labels[
                        int(
                            beta_id
                        )
                    ],
                )
            ]
            for (
                alpha_id,
                beta_id,
            ) in zip(
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
                            lambda_labels[
                                int(
                                    alpha_id
                                )
                            ],
                            lambda_labels[
                                int(
                                    beta_id
                                )
                            ],
                        )
                    )
                )
            ]
            for (
                alpha_id,
                beta_id,
            ) in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        ],
        dtype=np.int32,
    )

    shell_k_within_upair = (
        _rank_shells_within_unordered_pairs(
            shell_upair_id,
            distance_integer,
        )
    )

    shell_k_rank = np.arange(
        shell_pair_id.size,
        dtype=np.int32,
    )

    order = np.argsort(
        shell_pair_id,
        kind="mergesort",
    )

    shell_fractional = shell_fractional[
        order
    ]

    shell_cartesian = shell_cartesian[
        order
    ].astype(
        np.float32,
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

    distance_integer = distance_integer[
        order
    ]

    number_of_pairs = len(
        pair_list
    )

    pair_counts = np.bincount(
        shell_pair_id,
        minlength=number_of_pairs,
    ).astype(
        np.int32,
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

        if rows.size == 0:
            continue

        first_row = int(
            rows[
                0
            ]
        )

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

        strain_unique_dirs = (
            strain_unique_dirs.astype(
                np.int32,
                copy=False,
            )
        )

        strain_inv_dirs = (
            strain_inv_dirs.astype(
                np.int32,
                copy=False,
            )
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

    return {
        "geometry_mode": "stacked_supercell",
        "shell_frac": shell_fractional,
        "shell_cart": shell_cartesian,
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
        "stacking_lattice_matrix": np.asarray(
            bulk_supercell.lattice_matrix,
            dtype=np.float64,
        ),
        "stacking_inverse_lattice_matrix": np.asarray(
            bulk_supercell.inverse_lattice_matrix,
            dtype=np.float64,
        ),
        "stacking_reciprocal_lattice_matrix": np.asarray(
            bulk_supercell.reciprocal_lattice_matrix,
            dtype=np.float64,
        ),
        "stacking_geometry_hash": str(
            bulk_supercell.geometry_hash
        ),
        "stacking_nominal_spacing": float(
            bulk_supercell.nominal_spacing
        ),
        "stacking_repeat_period": float(
            bulk_supercell.repeat_period
        ),
        "stacking_period_normal_height": float(
            bulk_supercell.period_normal_height
        ),
        "stacking_volume": float(
            bulk_supercell.volume
        ),
        "stacking_bulk_number_density": float(
            bulk_supercell.number_of_atoms
            / max(
                bulk_supercell.volume,
                1.0e-12,
            )
        ),
        "stacking_symmetry_symbol": (
            bulk_supercell.symmetry_symbol
            if bulk_supercell.symmetry_symbol is not None
            else "P1"
        ),
        "stacking_symmetry_number": int(
            bulk_supercell.symmetry_number
            if bulk_supercell.symmetry_number is not None
            else 1
        ),
        "number_of_layer_positions": int(
            finite_particle.number_of_layer_positions
        ),
        "number_of_enabled_layers": int(
            finite_particle.number_of_enabled_layers
        ),
        "finite_atom_count": int(
            finite_particle.atom_cart.shape[0]
        ),
        "bulk_atom_count": int(
            bulk_supercell.number_of_atoms
        ),
        "finite_species_counts": np.asarray(
            species_metadata[
                "finite_species_counts"
            ],
            dtype=np.int64,
        ),
        "bulk_species_counts": np.asarray(
            species_metadata[
                "bulk_species_counts"
            ],
            dtype=np.int64,
        ),
        "distance_integer": np.asarray(
            distance_integer,
            dtype=np.int64,
        ),
    }


def validate_stacked_shell_table(
    shell_table: Dict[str, Any],
) -> None:
    """
    Validate the packed shell table before it is passed to PDFCalculator.
    """
    required = (
        "shell_frac",
        "shell_mult",
        "shell_bulk_mult",
        "shell_mult_eff",
        "shell_bulk_mult_eff",
        "shell_gamma",
        "shell_pair_id",
        "shell_alpha_id",
        "shell_beta_id",
        "shell_hkl",
        "shell_direction",
        "shell_k_rank",
        "shell_upair_id",
        "shell_k_within_upair",
        "pair_offsets",
        "pair_alpha_id",
        "pair_beta_id",
        "unique_species",
        "pair_list",
        "upair_list",
        "strain_unique_dirs",
        "strain_inv_dirs",
        "stacking_lattice_matrix",
    )

    for key in required:
        if key not in shell_table:
            raise KeyError(
                f"Stacked shell table is missing {key!r}."
            )

    number_of_shells = int(
        np.asarray(
            shell_table[
                "shell_frac"
            ]
        ).shape[0]
    )

    shell_arrays = (
        "shell_mult",
        "shell_bulk_mult",
        "shell_mult_eff",
        "shell_bulk_mult_eff",
        "shell_gamma",
        "shell_pair_id",
        "shell_alpha_id",
        "shell_beta_id",
        "shell_hkl",
        "shell_direction",
        "shell_k_rank",
        "shell_upair_id",
        "shell_k_within_upair",
    )

    for key in shell_arrays:
        array = np.asarray(
            shell_table[
                key
            ]
        )

        if array.shape[0] != number_of_shells:
            raise ValueError(
                f"{key!r} has {array.shape[0]} rows but "
                f"shell_frac has {number_of_shells}."
            )

    if number_of_shells <= 0:
        raise ValueError(
            "The stacked shell table contains no shells."
        )

    if not np.all(
        np.isfinite(
            shell_table[
                "shell_frac"
            ]
        )
    ):
        raise ValueError(
            "shell_frac contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            shell_table[
                "shell_mult_eff"
            ]
        )
    ):
        raise ValueError(
            "shell_mult_eff contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            shell_table[
                "shell_gamma"
            ]
        )
    ):
        raise ValueError(
            "shell_gamma contains non-finite values."
        )

    pair_offsets = np.asarray(
        shell_table[
            "pair_offsets"
        ],
        dtype=np.int64,
    )

    if pair_offsets.ndim != 1:
        raise ValueError(
            "pair_offsets must be one-dimensional."
        )

    if pair_offsets.size == 0:
        raise ValueError(
            "pair_offsets is empty."
        )

    if int(
        pair_offsets[
            -1
        ]
    ) != number_of_shells:
        raise ValueError(
            "The final pair offset does not equal the shell count."
        )


def build_stacked_finite_shell_table(
    *,
    structure: Structure,
    pair_cutoff: float,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    analyze_supercell_symmetry: bool = False,
) -> StackedFiniteShellResult:
    """
    Build one fixed-geometry finite stacked shell table.

    This is the Phase-3 CPU reference implementation.

    Important
    ---------
    The stacking distance and parent lattice are fixed for this build. If
    either changes, this function must be called again so that:

        stacking supercell
        finite atom positions
        pair distances
        hkl values
        reduced directions
        shell grouping

    are all recalculated.
    """
    if structure is None:
        raise ValueError(
            "A source structure is required."
        )

    pair_cutoff = float(
        pair_cutoff
    )

    if (
        not np.isfinite(
            pair_cutoff
        )
        or pair_cutoff <= 0.0
    ):
        raise ValueError(
            "pair_cutoff must be positive and finite."
        )

    stack_spec = stack_spec.normalized()

    if not stack_spec.enabled:
        raise ValueError(
            "The stacking specification is disabled."
        )

    start_time = time.perf_counter()

    print(
        "[STACKED SHELLS] Building periodic stacking supercell..."
    )

    bulk_supercell = build_stacking_supercell(
        structure=structure,
        stack_spec=stack_spec,
        analyze_symmetry=(
            analyze_supercell_symmetry
        ),
    )

    print(
        "[STACKED SHELLS] Building finite complete-layer particle..."
    )

    finite_particle = build_stacked_particle(
        structure=structure,
        shape_spec=shape_spec,
        stack_spec=stack_spec,
        max_atoms=None,
    )

    if finite_particle.atom_cart.shape[0] <= 1:
        raise RuntimeError(
            "The finite stacked particle contains too few atoms."
        )

    print(
        "[STACKED SHELLS] Generating finite particle pairs: "
        f"atoms={finite_particle.atom_cart.shape[0]:,}, "
        f"cutoff={pair_cutoff:.6g} Å"
    )

    finite_rows = _finite_directed_pair_rows(
        finite_particle,
        pair_cutoff=pair_cutoff,
    )

    print(
        "[STACKED SHELLS] Generating periodic bulk pairs: "
        f"supercell_atoms={bulk_supercell.number_of_atoms:,}, "
        f"cutoff={pair_cutoff:.6g} Å"
    )

    bulk_rows = _bulk_periodic_pair_rows(
        bulk_supercell,
        pair_cutoff=pair_cutoff,
    )

    finite_pair_rows = int(
        finite_rows[
            "distance"
        ].size
    )

    bulk_pair_rows = int(
        bulk_rows[
            "distance"
        ].size
    )

    print(
        "[STACKED SHELLS] Pair rows: "
        f"finite={finite_pair_rows:,}, "
        f"bulk={bulk_pair_rows:,}"
    )

    species_metadata = _species_metadata(
        finite_particle,
        bulk_supercell,
    )

    shell_table = _assemble_shell_table(
        finite_rows=finite_rows,
        bulk_rows=bulk_rows,
        finite_particle=finite_particle,
        bulk_supercell=bulk_supercell,
        species_metadata=species_metadata,
    )

    validate_stacked_shell_table(
        shell_table
    )

    elapsed = float(
        time.perf_counter()
        - start_time
    )

    print(
        "[STACKED SHELLS] Built fixed-geometry shell table: "
        f"shells={shell_table['shell_frac'].shape[0]:,}, "
        f"finite_atoms={finite_particle.atom_cart.shape[0]:,}, "
        f"bulk_atoms={bulk_supercell.number_of_atoms:,}, "
        f"time={elapsed:.3f} s"
    )

    return StackedFiniteShellResult(
        shell_table=shell_table,
        finite_particle=finite_particle,
        bulk_supercell=bulk_supercell,
        build_seconds=elapsed,
        finite_pair_rows=finite_pair_rows,
        bulk_pair_rows=bulk_pair_rows,
    )


def _pair_hkl_and_direction(cartesian, lattice_matrix):
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
