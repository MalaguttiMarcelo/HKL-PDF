from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    build_orientation_basis,
    shape_dimension_lengths,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    RealizedLayer,
    layer_basis,
    realize_n_layers,
    realized_layer_translation,
    stack_thickness_metrics,
)


_EPSILON = 1.0e-10


@dataclass
class StackedParticle:
    atom_cart: np.ndarray
    atom_elements: List[str]

    atom_layer_ids: np.ndarray
    atom_layer_indices: np.ndarray
    atom_sequence_ids: np.ndarray

    layer_origins: np.ndarray
    layer_indices: np.ndarray
    layer_sequence_ids: np.ndarray

    realized_layers: List[RealizedLayer]

    transition_starts: np.ndarray
    transition_ends: np.ndarray

    stack_center: np.ndarray
    stack_normal: np.ndarray

    number_of_layer_positions: int
    number_of_enabled_layers: int

    stack_center_span: float
    stack_boundary_thickness: float

    n_candidate_atoms: int
    n_clipped_atoms: int


def _shape_dimensions(
    *,
    shape_spec: CrystalliteShapeSpec,
    lattice_matrix: np.ndarray,
):
    """
    Return shape geometry in the oriented shape basis.

    Axis 3 is not used as an atom-level clipping boundary when stacking is
    enabled. It represents the number of complete layer positions.
    """
    e1, e2, e3 = build_orientation_basis(
        lattice_matrix,
        axis_hkl=(
            shape_spec.axis_h,
            shape_spec.axis_k,
            shape_spec.axis_l,
        ),
        base1_hkl=(
            shape_spec.base1_h,
            shape_spec.base1_k,
            shape_spec.base1_l,
        ),
        base2_hkl=(
            shape_spec.base2_h,
            shape_spec.base2_k,
            shape_spec.base2_l,
        ),
    )

    base1_length, base2_length, axis_length = (
        shape_dimension_lengths(
            lattice_matrix,
            axis_hkl=(
                shape_spec.axis_h,
                shape_spec.axis_k,
                shape_spec.axis_l,
            ),
            base1_hkl=(
                shape_spec.base1_h,
                shape_spec.base1_k,
                shape_spec.base1_l,
            ),
            base2_hkl=(
                shape_spec.base2_h,
                shape_spec.base2_k,
                shape_spec.base2_l,
            ),
        )
    )

    axis1_cells = float(
        getattr(
            shape_spec,
            "axis1_cells",
            shape_spec.diameter_cells,
        )
    )

    axis2_cells = float(
        getattr(
            shape_spec,
            "axis2_cells",
            shape_spec.diameter_cells,
        )
    )

    shape_type = str(
        shape_spec.shape_type
    ).strip().lower()

    if shape_type in {
        "sphere",
        "spherical",
        "cylinder",
        "disk",
        "disk-like",
        "disk_like",
    }:
        axis2_cells = axis1_cells

    length_1 = max(
        axis1_cells
        * float(
            base1_length
        ),
        _EPSILON,
    )

    length_2 = max(
        axis2_cells
        * float(
            base2_length
        ),
        _EPSILON,
    )

    half_1 = 0.5 * length_1
    half_2 = 0.5 * length_2

    return {
        "shape_type": shape_type,
        "e1": np.asarray(
            e1,
            dtype=float,
        ),
        "e2": np.asarray(
            e2,
            dtype=float,
        ),
        "e3": np.asarray(
            e3,
            dtype=float,
        ),
        "half_1": float(
            half_1
        ),
        "half_2": float(
            half_2
        ),
        "base1_length": float(
            base1_length
        ),
        "base2_length": float(
            base2_length
        ),
        "axis_length": float(
            axis_length
        ),
    }


def _centered_base_layer(
    structure: Structure,
    normal: np.ndarray,
) -> tuple[np.ndarray, List[str]]:
    """
    Return the atoms from one structural layer centered around the origin.

    The input CIF is assumed to contain one layer. Fractional a and b
    coordinates are wrapped into one in-plane cell.
    """
    lattice = np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )

    fractional = np.asarray(
        structure.frac_coords,
        dtype=float,
    ).copy()

    fractional[:, 0] %= 1.0
    fractional[:, 1] %= 1.0

    cartesian = (
        fractional
        @ lattice
    )

    lower = np.min(
        cartesian,
        axis=0,
    )

    upper = np.max(
        cartesian,
        axis=0,
    )

    center = 0.5 * (
        lower
        + upper
    )

    normal_coordinates = (
        cartesian
        @ normal
    )

    normal_center = 0.5 * (
        float(
            np.min(
                normal_coordinates
            )
        )
        + float(
            np.max(
                normal_coordinates
            )
        )
    )

    center += (
        normal_center
        - float(
            np.dot(
                center,
                normal,
            )
        )
    ) * normal

    cartesian -= center[
        None,
        :
    ]

    elements = [
        str(site.species)
        for site in structure.sites
    ]

    return (
        cartesian,
        elements,
    )


def _conservative_in_plane_range(
    *,
    lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    realized_layers: List[RealizedLayer],
) -> tuple[int, int, int, int]:
    """
    Build a conservative in-plane a/b replication range.

    Exact lateral inclusion is determined afterward.
    """
    vector_u, vector_v, _normal = layer_basis(
        lattice_matrix
    )

    geometry = _shape_dimensions(
        shape_spec=shape_spec,
        lattice_matrix=lattice_matrix,
    )

    maximum_half_width = max(
        float(
            geometry["half_1"]
        ),
        float(
            geometry["half_2"]
        ),
        1.0,
    )

    basis = np.asarray(
        [
            vector_u,
            vector_v,
        ],
        dtype=float,
    )

    inverse = np.linalg.pinv(
        basis
    )

    cartesian_extent = np.asarray(
        [
            maximum_half_width,
            maximum_half_width,
            maximum_half_width,
        ],
        dtype=float,
    )

    coefficient_extent = (
        cartesian_extent
        @ np.abs(
            inverse
        )
    )

    maximum_shift_u = max(
        [
            abs(
                float(
                    layer.shift_u
                )
            )
            for layer in realized_layers
        ]
        or [0.0]
    )

    maximum_shift_v = max(
        [
            abs(
                float(
                    layer.shift_v
                )
            )
            for layer in realized_layers
        ]
        or [0.0]
    )

    count_u = max(
        1,
        int(
            np.ceil(
                coefficient_extent[0]
                + maximum_shift_u
                + 2.0
            )
        ),
    )

    count_v = max(
        1,
        int(
            np.ceil(
                coefficient_extent[1]
                + maximum_shift_v
                + 2.0
            )
        ),
    )

    return (
        -count_u,
        count_u,
        -count_v,
        count_v,
    )


def _lateral_mask_for_layer(
    positions: np.ndarray,
    *,
    layer: RealizedLayer,
    shape_spec: CrystalliteShapeSpec,
    lattice_matrix: np.ndarray,
    stack_boundary_thickness: float,
) -> np.ndarray:
    """
    Return a lateral atom mask for one complete layer.

    No atom-level upper/lower stacking-direction boundary is applied.

    Cylinder/disk/prism
        All selected layers use the same lateral cross-section.

    Sphere/ellipsoid
        A single lateral cross-section is calculated from the layer center.
        All atoms in the layer use that same cross-section.
    """
    positions = np.asarray(
        positions,
        dtype=float,
    )

    geometry = _shape_dimensions(
        shape_spec=shape_spec,
        lattice_matrix=lattice_matrix,
    )

    shape_type = str(
        geometry["shape_type"]
    )

    e1 = geometry["e1"]
    e2 = geometry["e2"]

    x = (
        positions
        @ e1
    )

    y = (
        positions
        @ e2
    )

    half_1 = max(
        float(
            geometry["half_1"]
        ),
        _EPSILON,
    )

    half_2 = max(
        float(
            geometry["half_2"]
        ),
        _EPSILON,
    )

    if shape_type in {
        "prism",
        "rectangular_prism",
    }:
        return (
            np.abs(
                x
            )
            <= half_1
        ) & (
            np.abs(
                y
            )
            <= half_2
        )

    if shape_type in {
        "ellipsoid",
        "ellipsoidal",
    }:
        stack_half_thickness = max(
            0.5
            * float(
                stack_boundary_thickness
            ),
            _EPSILON,
        )

        normalized_layer_position = (
            float(
                layer.normal_offset
            )
            / stack_half_thickness
        )

        remaining = max(
            0.0,
            1.0
            - normalized_layer_position
            * normalized_layer_position,
        )

        if remaining <= _EPSILON:
            return np.zeros(
                positions.shape[0],
                dtype=bool,
            )

        scale = float(
            np.sqrt(
                remaining
            )
        )

        layer_half_1 = max(
            half_1
            * scale,
            _EPSILON,
        )

        layer_half_2 = max(
            half_2
            * scale,
            _EPSILON,
        )

        return (
            (
                x
                / layer_half_1
            ) ** 2
            + (
                y
                / layer_half_2
            ) ** 2
        ) <= 1.0

    if shape_type in {
        "sphere",
        "spherical",
    }:
        radius = max(
            0.5 * (
                half_1
                + half_2
            ),
            _EPSILON,
        )

        stack_half_thickness = max(
            0.5
            * float(
                stack_boundary_thickness
            ),
            _EPSILON,
        )

        normalized_layer_position = (
            float(
                layer.normal_offset
            )
            / stack_half_thickness
        )

        remaining = max(
            0.0,
            1.0
            - normalized_layer_position
            * normalized_layer_position,
        )

        if remaining <= _EPSILON:
            return np.zeros(
                positions.shape[0],
                dtype=bool,
            )

        layer_radius = max(
            radius
            * float(
                np.sqrt(
                    remaining
                )
            ),
            _EPSILON,
        )

        return (
            x * x
            + y * y
        ) <= (
            layer_radius
            * layer_radius
        )

    # Cylinder and disk use a constant circular AB cross-section.
    radius = max(
        0.5 * (
            half_1
            + half_2
        ),
        _EPSILON,
    )

    return (
        x * x
        + y * y
    ) <= (
        radius
        * radius
    )


def build_stacked_particle(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    max_atoms: Optional[int] = None,
) -> StackedParticle:
    """
    Build a finite particle containing exactly N complete layer positions.

    When stacking is enabled:

        N = round(shape_spec.axis3_cells)

    No atom is clipped by an upper or lower stack-axis boundary. The shape
    acts only as a lateral AB envelope. Sphere and ellipsoid are represented
    by one complete lateral slice per layer.
    """
    lattice = np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )

    stack_spec = stack_spec.normalized()

    number_of_layers = max(
        1,
        int(
            round(
                float(
                    getattr(
                        shape_spec,
                        "axis3_cells",
                        shape_spec.height_cells,
                    )
                )
            )
        ),
    )

    realized_layers = realize_n_layers(
        stack_spec,
        number_of_layers=number_of_layers,
        center_stack=True,
    )

    thickness = stack_thickness_metrics(
        realized_layers,
        fallback_spacing=(
            stack_spec.nominal_spacing
        ),
    )

    stack_center_span = float(
        thickness["center_span"]
    )

    stack_boundary_thickness = float(
        thickness[
            "boundary_thickness"
        ]
    )

    _, _, normal = layer_basis(
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

    vector_u = lattice[0]
    vector_v = lattice[1]

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

    enabled_layer_count = 0

    for layer_id, realized in enumerate(
        realized_layers
    ):
        origin = realized_layer_translation(
            realized,
            lattice,
        )

        layer_origins.append(
            origin
        )

        layer_indices.append(
            realized.realized_index
        )

        layer_sequence_ids.append(
            realized.sequence_index
        )

        if not realized.enabled:
            realized.atom_count = 0
            continue

        enabled_layer_count += 1
        layer_atom_count = 0

        for cell_u in range(
            u_min,
            u_max + 1,
        ):
            for cell_v in range(
                v_min,
                v_max + 1,
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

                number_selected = int(
                    selected_positions.shape[0]
                )

                layer_atom_count += (
                    number_selected
                )

                clipped_count += (
                    number_selected
                )

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
                    * number_selected
                )

                all_layer_indices.extend(
                    [
                        realized.realized_index
                    ]
                    * number_selected
                )

                all_sequence_ids.extend(
                    [
                        realized.sequence_index
                    ]
                    * number_selected
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
            (0, 3),
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

    output_elements = list(
        all_elements
    )

    if (
        max_atoms is not None
        and atom_cart.shape[0]
        > int(max_atoms)
    ):
        selected = np.linspace(
            0,
            atom_cart.shape[0] - 1,
            int(max_atoms),
            dtype=np.int64,
        )

        atom_cart = atom_cart[
            selected
        ]

        atom_layer_ids = atom_layer_ids[
            selected
        ]

        atom_layer_indices = atom_layer_indices[
            selected
        ]

        atom_sequence_ids = atom_sequence_ids[
            selected
        ]

        output_elements = [
            output_elements[
                index
            ]
            for index in selected
        ]

    layer_origins_array = np.asarray(
        layer_origins,
        dtype=np.float32,
    ).reshape(
        -1,
        3,
    )

    if layer_origins_array.shape[0] >= 2:
        transition_starts = (
            layer_origins_array[:-1]
        )

        transition_ends = (
            layer_origins_array[1:]
        )
    else:
        transition_starts = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

        transition_ends = np.zeros(
            (0, 3),
            dtype=np.float32,
        )

    return StackedParticle(
        atom_cart=atom_cart,
        atom_elements=output_elements,
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
            number_of_layers
        ),
        number_of_enabled_layers=int(
            enabled_layer_count
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