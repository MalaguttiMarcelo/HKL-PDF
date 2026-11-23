from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from pymatgen.core import Lattice
from pymatgen.core.structure import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    build_orientation_basis,
    shape_dimension_lengths,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    RealizedLayer,
    absolute_sequence_layers,
    effective_repeat_translation,
    evaluate_stacking_spacing,
    layer_basis,
    selected_layer_indices,
    stack_thickness_metrics,
)



_EPSILON = 1.0e-12


@dataclass
class CompiledLayerDescriptor:
    """
    One selected complete layer in the finite particle.

    The descriptor is topological and does not contain final Cartesian
    coordinates.
    """

    realized_index: int
    sequence_index: int
    repeat_index: int

    enabled: bool
    label: str

    shift_u: float
    shift_v: float

    stack_coordinate: float
    normal_delta: float


@dataclass
class EvaluatedStackedGeometry:
    """
    Current Cartesian realization of one compiled finite stacking candidate.
    """

    parent_lattice_matrix: np.ndarray
    stacking_lattice_matrix: np.ndarray
    stacking_inverse_lattice_matrix: np.ndarray

    stack_normal: np.ndarray

    spacing: float
    period_height: float

    atom_cart: np.ndarray
    atom_frac_stack: np.ndarray

    atom_elements: List[str]

    atom_source_site_indices: np.ndarray
    atom_layer_indices: np.ndarray
    atom_sequence_indices: np.ndarray
    atom_cell_a: np.ndarray
    atom_cell_b: np.ndarray

    layer_origins: np.ndarray
    layer_normal_offsets: np.ndarray

    center_span: float
    boundary_thickness: float

    number_density: float

    @property
    def number_of_atoms(self) -> int:
        return int(
            self.atom_cart.shape[0]
        )

    @property
    def number_of_layers(self) -> int:
        return int(
            self.layer_origins.shape[0]
        )

from pdf_fitting.models.stacking_transitions import (
    evaluate_transition_values,
)

def evaluate_generic_layer_components(
    *,
    layers: List[CompiledLayerDescriptor],
    stack_spec: LayerStackSpec,
    parameters: Mapping[str, Any],
):
    """
    Evaluate arbitrary periodic transition vectors.

    Every realized layer is represented by the cumulative sum of transition
    vectors from realized layer zero to that layer.
    """
    (
        transitions,
        transition_du,
        transition_dv,
        transition_h,
    ) = evaluate_transition_values(
        stack_spec,
        parameters,
    )

    number_of_transitions = len(
        transitions
    )

    if number_of_transitions <= 0:
        raise ValueError(
            "The stacking sequence contains no transitions."
        )

    period_du = float(
        np.sum(
            transition_du
        )
    )

    period_dv = float(
        np.sum(
            transition_dv
        )
    )

    period_h = float(
        np.sum(
            transition_h
        )
    )

    prefix_du = np.concatenate(
        (
            np.asarray(
                [0.0],
                dtype=float,
            ),
            np.cumsum(
                transition_du[:-1]
            ),
        )
    )

    prefix_dv = np.concatenate(
        (
            np.asarray(
                [0.0],
                dtype=float,
            ),
            np.cumsum(
                transition_dv[:-1]
            ),
        )
    )

    prefix_h = np.concatenate(
        (
            np.asarray(
                [0.0],
                dtype=float,
            ),
            np.cumsum(
                transition_h[:-1]
            ),
        )
    )

    layer_u = np.zeros(
        len(
            layers
        ),
        dtype=float,
    )

    layer_v = np.zeros(
        len(
            layers
        ),
        dtype=float,
    )

    layer_h = np.zeros(
        len(
            layers
        ),
        dtype=float,
    )

    for index, layer in enumerate(
        layers
    ):
        sequence_index = int(
            layer.sequence_index
        )

        repeat_index = int(
            layer.repeat_index
        )

        layer_u[index] = (
            repeat_index
            * period_du
            + prefix_du[
                sequence_index
            ]
        )

        layer_v[index] = (
            repeat_index
            * period_dv
            + prefix_dv[
                sequence_index
            ]
        )

        layer_h[index] = (
            repeat_index
            * period_h
            + prefix_h[
                sequence_index
            ]
        )

    return (
        layer_u,
        layer_v,
        layer_h,
        period_du,
        period_dv,
        period_h,
        transition_du,
        transition_dv,
        transition_h,
    )

@dataclass
class CompiledStackedGeometry:
    """
    Compiled finite stacking candidate.

    Compilation fixes:

        stacking sequence;
        integer layer count;
        AB envelope membership;
        source-site identities;
        in-plane cell translations;
        layer identities.

    Evaluation changes:

        parent lattice;
        intralayer source-atom positions;
        stacking normal;
        interlayer spacing;
        stacking-cell matrix;
        final Cartesian atom coordinates.

    No neighbor pairs are compiled in Phase A. Pair topology is Phase B.
    """

    source_structure: Structure
    shape_spec: CrystalliteShapeSpec
    stack_spec: LayerStackSpec

    source_fractional: np.ndarray
    source_elements: List[str]

    layers: List[CompiledLayerDescriptor]

    atom_source_site_indices: np.ndarray
    atom_layer_descriptor_indices: np.ndarray
    atom_cell_a: np.ndarray
    atom_cell_b: np.ndarray

    reference_lattice_matrix: np.ndarray
    reference_spacing: float


    def evaluate(
        self,
        parameters: Mapping[str, Any] | None = None,
    ) -> EvaluatedStackedGeometry:
        """
        Evaluate the compiled topology for the current lattice and arbitrary
        periodic stacking-transition parameters.
        """
        parameters = dict(
            parameters
            or {}
        )

        lattice_matrix = _lattice_matrix_from_parameters(
            self.source_structure,
            parameters,
        )

        vector_a, vector_b, normal = layer_basis(
            lattice_matrix
        )

        # Retained as the nominal/reference spacing used by the existing
        # LayerStackSpec representation and for reporting.
        spacing = evaluate_stacking_spacing(
            self.stack_spec,
            lattice_matrix,
            parameters=parameters,
        )

        (
            layer_u,
            layer_v,
            layer_height,
            dynamic_repeat_u,
            dynamic_repeat_v,
            dynamic_repeat_height,
            _transition_du,
            _transition_dv,
            _transition_h,
        ) = evaluate_generic_layer_components(
            layers=self.layers,
            stack_spec=self.stack_spec,
            parameters=parameters,
        )

        layer_u = np.asarray(
            layer_u,
            dtype=float,
        )

        layer_v = np.asarray(
            layer_v,
            dtype=float,
        )

        layer_height = np.asarray(
            layer_height,
            dtype=float,
        )

        if layer_height.size:
            normal_center = 0.5 * (
                float(
                    np.min(
                        layer_height
                    )
                )
                + float(
                    np.max(
                        layer_height
                    )
                )
            )
        else:
            normal_center = 0.0

        layer_normal_offsets = (
            layer_height
            - normal_center
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
            + layer_normal_offsets[
                :,
                None
            ]
            * normal[
                None,
                :
            ]
        )

        base_atom_cart = _centered_source_layer_cartesian(
            self.source_fractional,
            lattice_matrix,
            normal,
        )

        source_indices = np.asarray(
            self.atom_source_site_indices,
            dtype=np.int64,
        )

        layer_descriptor_indices = np.asarray(
            self.atom_layer_descriptor_indices,
            dtype=np.int64,
        )

        atom_cart = (
            base_atom_cart[
                source_indices
            ]
            + np.asarray(
                self.atom_cell_a,
                dtype=float,
            )[
                :,
                None
            ]
            * vector_a[
                None,
                :
            ]
            + np.asarray(
                self.atom_cell_b,
                dtype=float,
            )[
                :,
                None
            ]
            * vector_b[
                None,
                :
            ]
            + layer_origins[
                layer_descriptor_indices
            ]
        )

        repeat_u = float(
            dynamic_repeat_u
        )

        repeat_v = float(
            dynamic_repeat_v
        )

        period_height = float(
            dynamic_repeat_height
        )

        if (
            not np.isfinite(
                period_height
            )
            or period_height <= _EPSILON
        ):
            raise ValueError(
                "The evaluated stacking period height must be positive "
                "and finite."
            )

        stacking_vector_c = (
            repeat_u
            * vector_a
            + repeat_v
            * vector_b
            + period_height
            * normal
        )

        stacking_lattice_matrix = np.asarray(
            [
                vector_a,
                vector_b,
                stacking_vector_c,
            ],
            dtype=float,
        )

        determinant = float(
            np.linalg.det(
                stacking_lattice_matrix
            )
        )

        if (
            not np.isfinite(
                determinant
            )
            or abs(
                determinant
            )
            <= _EPSILON
        ):
            raise ValueError(
                "The evaluated stacking-cell lattice is singular."
            )

        stacking_inverse = np.linalg.inv(
            stacking_lattice_matrix
        )

        atom_frac_stack = (
            atom_cart
            @ stacking_inverse
        )

        atom_layer_indices = np.asarray(
            [
                self.layers[
                    index
                ].realized_index
                for index in layer_descriptor_indices
            ],
            dtype=np.int32,
        )

        atom_sequence_indices = np.asarray(
            [
                self.layers[
                    index
                ].sequence_index
                for index in layer_descriptor_indices
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

        evaluated_layers = [
            RealizedLayer(
                realized_index=int(
                    layer.realized_index
                ),
                sequence_index=int(
                    layer.sequence_index
                ),
                repeat_index=int(
                    layer.repeat_index
                ),
                label=str(
                    layer.label
                ),
                shift_u=float(
                    layer_u[
                        index
                    ]
                ),
                shift_v=float(
                    layer_v[
                        index
                    ]
                ),
                stack_coordinate=float(
                    layer_height[
                        index
                    ]
                    / max(
                        spacing,
                        _EPSILON,
                    )
                ),
                normal_offset=float(
                    layer_normal_offsets[
                        index
                    ]
                ),
                enabled=bool(
                    layer.enabled
                ),
                atom_count=0,
            )
            for index, layer in enumerate(
                self.layers
            )
        ]

        thickness = stack_thickness_metrics(
            evaluated_layers,
            fallback_spacing=spacing,
        )

        volume = abs(
            determinant
        )

        enabled_sequence_rows = sum(
            1
            for layer in self.stack_spec.sequence
            if bool(
                layer.enabled
            )
        )

        enabled_atoms_per_period = (
            int(
                enabled_sequence_rows
            )
            * len(
                self.source_structure.sites
            )
        )

        number_density = (
            float(
                enabled_atoms_per_period
            )
            / max(
                volume,
                _EPSILON,
            )
        )

        return EvaluatedStackedGeometry(
            parent_lattice_matrix=np.asarray(
                lattice_matrix,
                dtype=float,
            ),
            stacking_lattice_matrix=np.asarray(
                stacking_lattice_matrix,
                dtype=float,
            ),
            stacking_inverse_lattice_matrix=np.asarray(
                stacking_inverse,
                dtype=float,
            ),
            stack_normal=np.asarray(
                normal,
                dtype=float,
            ),
            spacing=float(
                spacing
            ),
            period_height=float(
                period_height
            ),
            atom_cart=np.asarray(
                atom_cart,
                dtype=float,
            ),
            atom_frac_stack=np.asarray(
                atom_frac_stack,
                dtype=float,
            ),
            atom_elements=atom_elements,
            atom_source_site_indices=np.asarray(
                source_indices,
                dtype=np.int32,
            ),
            atom_layer_indices=atom_layer_indices,
            atom_sequence_indices=np.asarray(
                atom_sequence_indices,
                dtype=np.int32,
            ),
            atom_cell_a=np.asarray(
                self.atom_cell_a,
                dtype=np.int32,
            ),
            atom_cell_b=np.asarray(
                self.atom_cell_b,
                dtype=np.int32,
            ),
            layer_origins=np.asarray(
                layer_origins,
                dtype=float,
            ),
            layer_normal_offsets=np.asarray(
                layer_normal_offsets,
                dtype=float,
            ),
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
            number_density=float(
                number_density
            ),
        )
    
def _lattice_matrix_from_parameters(
    structure: Structure,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """
    Build the current parent lattice matrix from refinement parameters.
    """
    lattice = structure.lattice

    a = float(
        parameters.get(
            "a",
            lattice.a,
        )
    )

    b = float(
        parameters.get(
            "b",
            lattice.b,
        )
    )

    c = float(
        parameters.get(
            "c",
            lattice.c,
        )
    )

    alpha = float(
        parameters.get(
            "alpha",
            lattice.alpha,
        )
    )

    beta = float(
        parameters.get(
            "beta",
            lattice.beta,
        )
    )

    gamma = float(
        parameters.get(
            "gamma",
            lattice.gamma,
        )
    )

    current_lattice = Lattice.from_parameters(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
    )

    return np.asarray(
        current_lattice.matrix,
        dtype=float,
    )


def _centered_source_layer_cartesian(
    source_fractional: np.ndarray,
    lattice_matrix: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """
    Evaluate source-layer atoms with the current parent lattice.

    The layer is centered only along the current stacking normal. This removes
    the arbitrary vacuum-position offset in the source CIF while preserving
    intralayer thickness and corrugation controlled by parent c/angles.
    """
    source_fractional = np.asarray(
        source_fractional,
        dtype=float,
    )

    wrapped = source_fractional.copy()

    wrapped[
        :,
        0
    ] %= 1.0

    wrapped[
        :,
        1
    ] %= 1.0

    cartesian = (
        wrapped
        @ lattice_matrix
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

    return (
        cartesian
        - normal_center
        * normal[
            None,
            :
        ]
    )


def _compile_layers(
    stack_spec: LayerStackSpec,
    number_of_layers: int,
) -> List[CompiledLayerDescriptor]:
    """
    Compile selected layer identities and cumulative sequence shifts.
    """
    spec = stack_spec.normalized()

    indices = selected_layer_indices(
        number_of_layers
    )

    absolute_sequence = absolute_sequence_layers(
        spec
    )

    sequence_length = len(
        absolute_sequence
    )

    if sequence_length <= 0:
        raise ValueError(
            "The stacking sequence contains no rows."
        )

    (
        repeat_u,
        repeat_v,
        repeat_s,
    ) = effective_repeat_translation(
        spec
    )

    compiled_layers: List[
        CompiledLayerDescriptor
    ] = []

    for realized_index in indices:
        repeat_index, sequence_index = divmod(
            int(
                realized_index
            ),
            int(
                sequence_length
            ),
        )

        editable_layer = spec.sequence[
            sequence_index
        ]

        absolute_layer = absolute_sequence[
            sequence_index
        ]

        compiled_layers.append(
            CompiledLayerDescriptor(
                realized_index=int(
                    realized_index
                ),
                sequence_index=int(
                    sequence_index
                ),
                repeat_index=int(
                    repeat_index
                ),
                enabled=bool(
                    editable_layer.enabled
                ),
                label=str(
                    editable_layer.label
                ),
                shift_u=(
                    float(
                        absolute_layer.shift_u
                    )
                    + repeat_index
                    * float(
                        repeat_u
                    )
                ),
                shift_v=(
                    float(
                        absolute_layer.shift_v
                    )
                    + repeat_index
                    * float(
                        repeat_v
                    )
                ),
                stack_coordinate=(
                    float(
                        absolute_layer.stack_coordinate
                    )
                    + repeat_index
                    * float(
                        repeat_s
                    )
                ),
                normal_delta=float(
                    absolute_layer.normal_delta
                ),
            )
        )

    return compiled_layers


def _reference_lateral_mask(
    positions: np.ndarray,
    *,
    shape_spec: CrystalliteShapeSpec,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compile fixed AB-envelope membership at the reference lattice.

    For Phase A, cylinder/disk/prism base membership is supported directly.
    Sphere/ellipsoid use their maximum AB projection; layer-dependent slicing
    remains handled by the existing GUI particle builder until Phase B.
    """
    positions = np.asarray(
        positions,
        dtype=float,
    )

    e1, e2, _e3 = build_orientation_basis(
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

    base1_length, base2_length, _axis_length = (
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

    half_1 = max(
        0.5
        * axis1_cells
        * float(
            base1_length
        ),
        _EPSILON,
    )

    half_2 = max(
        0.5
        * axis2_cells
        * float(
            base2_length
        ),
        _EPSILON,
    )

    x = positions @ e1
    y = positions @ e2

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
        return (
            (
                x
                / half_1
            ) ** 2
            + (
                y
                / half_2
            ) ** 2
        ) <= 1.0

    radius = 0.5 * (
        half_1
        + half_2
    )

    return (
        x * x
        + y * y
    ) <= (
        radius
        * radius
    )


def compile_stacked_geometry(
    *,
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    reference_parameters: Optional[
        Mapping[str, Any]
    ] = None,
) -> CompiledStackedGeometry:
    """
    Compile one discrete finite stacking candidate.

    This function is called once for each:

        stacking sequence;
        AB envelope;
        integer number of layers.
    """
    if structure is None:
        raise ValueError(
            "A source layer structure is required."
        )

    stack_spec = stack_spec.normalized()

    if not stack_spec.enabled:
        raise ValueError(
            "The stacking specification is disabled."
        )

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

    reference_parameters = dict(
        reference_parameters
        or {}
    )

    reference_lattice = _lattice_matrix_from_parameters(
        structure,
        reference_parameters,
    )

    reference_spacing = evaluate_stacking_spacing(
        stack_spec,
        reference_lattice,
        parameters=reference_parameters,
    )

    layers = _compile_layers(
        stack_spec,
        number_of_layers,
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

    (
        reference_layer_u,
        reference_layer_v,
        reference_layer_height,
        _reference_repeat_u,
        _reference_repeat_v,
        _reference_repeat_height,
        _reference_transition_du,
        _reference_transition_dv,
        _reference_transition_h,
    ) = evaluate_generic_layer_components(
        layers=layers,
        stack_spec=stack_spec,
        parameters=reference_parameters,
    )

    reference_layer_u = np.asarray(
        reference_layer_u,
        dtype=float,
    )

    reference_layer_v = np.asarray(
        reference_layer_v,
        dtype=float,
    )

    reference_layer_height = np.asarray(
        reference_layer_height,
        dtype=float,
    )

    reference_height_center = 0.5 * (
        float(
            np.min(
                reference_layer_height
            )
        )
        + float(
            np.max(
                reference_layer_height
            )
        )
    )

    centered_reference_height = (
        reference_layer_height
        - reference_height_center
    )

    reference_origins = (
        reference_layer_u[
            :,
            None
        ]
        * vector_a[
            None,
            :
        ]
        + reference_layer_v[
            :,
            None
        ]
        * vector_b[
            None,
            :
        ]
        + centered_reference_height[
            :,
            None
        ]
        * normal[
            None,
            :
        ]
    )

    maximum_shift_u = max(
        [
            abs(
                float(
                    value
                )
            )
            for value in reference_layer_u
        ]
        or [
            0.0
        ]
    )

    maximum_shift_v = max(
        [
            abs(
                float(
                    value
                )
            )
            for value in reference_layer_v
        ]
        or [
            0.0
        ]
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
            + maximum_shift_u
            + 2.0
        )
    )

    count_b = int(
        np.ceil(
            axis2_cells
            + maximum_shift_v
            + 2.0
        )
    )

    source_site_rows = []
    layer_descriptor_rows = []
    cell_a_rows = []
    cell_b_rows = []

    for layer_descriptor_index, layer in enumerate(
        layers
    ):
        if not layer.enabled:
            continue

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

                for source_site_index in np.flatnonzero(
                    inside
                ):
                    source_site_rows.append(
                        int(
                            source_site_index
                        )
                    )

                    layer_descriptor_rows.append(
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

    if not source_site_rows:
        raise RuntimeError(
            "The compiled finite stacking geometry contains no atoms."
        )

    return CompiledStackedGeometry(
        source_structure=structure,
        shape_spec=CrystalliteShapeSpec.from_dict(
            shape_spec.to_dict()
        ),
        stack_spec=stack_spec,
        source_fractional=np.asarray(
            source_fractional,
            dtype=float,
        ),
        source_elements=list(
            source_elements
        ),
        layers=layers,
        atom_source_site_indices=np.asarray(
            source_site_rows,
            dtype=np.int32,
        ),
        atom_layer_descriptor_indices=np.asarray(
            layer_descriptor_rows,
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
        reference_lattice_matrix=np.asarray(
            reference_lattice,
            dtype=float,
        ),
        reference_spacing=float(
            reference_spacing
        ),
    )