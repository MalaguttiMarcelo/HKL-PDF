from __future__ import annotations

from dataclasses import dataclass

from typing import (
    Any,
    Mapping,
)

import numpy as np

from pymatgen.core import (
    Lattice,
    Structure,
)

from pdf_fitting.models.compiled_stacked_geometry import (
    _centered_source_layer_cartesian,
    _lattice_matrix_from_parameters,
)

from pdf_fitting.models.compiled_stochastic_geometry import (
    _generic_state_vectors,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    layer_basis,
)


@dataclass
class BulkStackingRealization:
    realization_index: int
    state_path: np.ndarray
    structure: Structure

    parent_lattice_matrix: np.ndarray
    realization_lattice_matrix: np.ndarray

    layer_origins: np.ndarray
    atom_state_indices: np.ndarray

    weight: float


def build_bulk_stacking_realization(
    *,
    source_structure: Structure,
    stack_spec: LayerStackSpec,
    state_path: np.ndarray,
    realization_index: int,
    parameters: Mapping[
        str,
        Any,
    ],
    weight: float,
) -> BulkStackingRealization:
    parameters = dict(
        parameters
        or {}
    )

    state_path = np.asarray(
        state_path,
        dtype=np.int32,
    ).reshape(
        -1
    )

    if state_path.size < 2:
        raise ValueError(
            "A periodic bulk realization requires at least two layers."
        )

    parent_lattice = _lattice_matrix_from_parameters(
        source_structure,
        parameters,
    )

    vector_a, vector_b, normal = layer_basis(
        parent_lattice
    )

    (
        state_du,
        state_dv,
        state_h,
        _state_labels,
    ) = _generic_state_vectors(
        stack_spec,
        parameters,
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

    if not np.all(
        (
            state_path >= 0
        )
        & (
            state_path < state_du.size
        )
    ):
        raise ValueError(
            "The realization contains an invalid state index."
        )

    number_of_layers = int(
        state_path.size
    )

    layer_u = np.zeros(
        number_of_layers,
        dtype=np.float64,
    )

    layer_v = np.zeros(
        number_of_layers,
        dtype=np.float64,
    )

    layer_h = np.zeros(
        number_of_layers,
        dtype=np.float64,
    )

    for layer_index in range(
        1,
        number_of_layers
    ):
        previous_state = int(
            state_path[
                layer_index - 1
            ]
        )

        layer_u[
            layer_index
        ] = (
            layer_u[
                layer_index - 1
            ]
            + state_du[
                previous_state
            ]
        )

        layer_v[
            layer_index
        ] = (
            layer_v[
                layer_index - 1
            ]
            + state_dv[
                previous_state
            ]
        )

        layer_h[
            layer_index
        ] = (
            layer_h[
                layer_index - 1
            ]
            + state_h[
                previous_state
            ]
        )

    final_state = int(
        state_path[
            -1
        ]
    )

    total_u = float(
        layer_u[
            -1
        ]
        + state_du[
            final_state
        ]
    )

    total_v = float(
        layer_v[
            -1
        ]
        + state_dv[
            final_state
        ]
    )

    total_h = float(
        layer_h[
            -1
        ]
        + state_h[
            final_state
        ]
    )

    if total_h <= 1.0e-12:
        raise ValueError(
            "The periodic realization has a non-positive stack height."
        )

    realization_c = (
        total_u
        * vector_a
        + total_v
        * vector_b
        + total_h
        * normal
    )

    realization_lattice_matrix = np.asarray(
        [
            vector_a,
            vector_b,
            realization_c,
        ],
        dtype=np.float64,
    )

    realization_inverse = np.linalg.inv(
        realization_lattice_matrix
    )

    source_fractional = np.asarray(
        source_structure.frac_coords,
        dtype=np.float64,
    )

    source_cartesian = _centered_source_layer_cartesian(
        source_fractional,
        parent_lattice,
        normal,
    )

    atom_cartesian_rows = []
    atom_species = []
    atom_state_indices = []

    layer_origins = np.zeros(
        (
            number_of_layers,
            3,
        ),
        dtype=np.float64,
    )

    for layer_index in range(
        number_of_layers
    ):
        origin = (
            layer_u[
                layer_index
            ]
            * vector_a
            + layer_v[
                layer_index
            ]
            * vector_b
            + layer_h[
                layer_index
            ]
            * normal
        )

        layer_origins[
            layer_index
        ] = origin

        atom_cartesian_rows.append(
            source_cartesian
            + origin[
                None,
                :
            ]
        )

        atom_species.extend(
            site.species
            for site in source_structure.sites
        )

        atom_state_indices.extend(
            [
                int(
                    state_path[
                        layer_index
                    ]
                )
            ]
            * len(
                source_structure.sites
            )
        )

    atom_cartesian = np.vstack(
        atom_cartesian_rows
    )

    atom_fractional = (
        atom_cartesian
        @ realization_inverse
    )

    atom_fractional %= 1.0

    realization_structure = Structure(
        lattice=Lattice(
            realization_lattice_matrix
        ),
        species=atom_species,
        coords=atom_fractional,
        coords_are_cartesian=False,
        to_unit_cell=False,
        validate_proximity=False,
        site_properties={
            "stack_state_index": (
                atom_state_indices
            ),
        },
    )

    return BulkStackingRealization(
        realization_index=int(
            realization_index
        ),
        state_path=np.asarray(
            state_path,
            dtype=np.int16,
        ),
        structure=realization_structure,
        parent_lattice_matrix=np.asarray(
            parent_lattice,
            dtype=np.float64,
        ),
        realization_lattice_matrix=(
            realization_lattice_matrix
        ),
        layer_origins=layer_origins,
        atom_state_indices=np.asarray(
            atom_state_indices,
            dtype=np.int16,
        ),
        weight=float(
            weight
        ),
    )

def export_bulk_realization(
    realization: BulkStackingRealization,
    *,
    cif_path: str,
    manifest_path: str,
) -> None:
    import json
    import os

    output_directory = os.path.dirname(
        os.path.abspath(
            cif_path
        )
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    realization.structure.to(
        filename=cif_path
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "realization_index": int(
                    realization.realization_index
                ),
                "weight": float(
                    realization.weight
                ),
                "state_path": [
                    int(
                        value
                    )
                    for value in realization.state_path
                ],
                "parent_lattice_matrix": (
                    realization
                    .parent_lattice_matrix
                    .tolist()
                ),
                "realization_lattice_matrix": (
                    realization
                    .realization_lattice_matrix
                    .tolist()
                ),
            },
            file,
            indent=2,
        )

def validate_bulk_stacking_realization(
    realization: BulkStackingRealization,
    *,
    minimum_distance: float = 0.25,
) -> None:
    """
    Reject malformed periodic realizations containing overlapping atoms.
    """
    structure = realization.structure

    if len(
        structure.sites
    ) <= 1:
        raise ValueError(
            "The bulk realization contains too few atoms."
        )

    (
        center_indices,
        neighbor_indices,
        image_vectors,
        distances,
    ) = structure.get_neighbor_list(
        r=max(
            float(
                minimum_distance
            ),
            0.5,
        ),
        numerical_tol=1.0e-10,
        exclude_self=True,
    )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    finite_distances = distances[
        np.isfinite(
            distances
        )
        & (
            distances
            > 1.0e-10
        )
    ]

    if finite_distances.size == 0:
        return

    nearest_distance = float(
        np.min(
            finite_distances
        )
    )

    if nearest_distance < float(
        minimum_distance
    ):
        bad_index = int(
            np.argmin(
                distances
            )
        )

        raise ValueError(
            "Bulk stacking realization contains overlapping or nearly "
            "overlapping atoms: "
            f"realization={realization.realization_index}, "
            f"minimum_distance={nearest_distance:.8g} Å, "
            f"center={int(center_indices[bad_index])}, "
            f"neighbor={int(neighbor_indices[bad_index])}, "
            f"image={tuple(int(v) for v in image_vectors[bad_index])}."
        )