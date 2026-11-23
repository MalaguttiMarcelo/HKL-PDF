from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    direct_hkl_to_cart,
)
from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    layer_basis,
)


_EPSILON = 1.0e-10


def _safe_integer_count(
    value: Any,
    *,
    label: str,
) -> int:
    """
    Return a positive integer prism cell count.

    The exported periodic CIF must have a well-defined unit-cell extent.
    """
    try:
        numeric_value = float(
            value
        )
    except Exception as exc:
        raise ValueError(
            f"{label} must be a positive number."
        ) from exc

    if not np.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{label} must be finite."
        )

    rounded = int(
        round(
            numeric_value
        )
    )

    if rounded < 1:
        raise ValueError(
            f"{label} must be at least 1."
        )

    if not np.isclose(
        numeric_value,
        float(
            rounded
        ),
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise ValueError(
            f"{label} must be an integer for periodic CIF export.\n\n"
            f"Current value: {numeric_value:g}\n"
            f"Nearest integer: {rounded}"
        )

    return rounded


def _direction(
    h: Any,
    k: Any,
    l: Any,
    *,
    label: str,
) -> Tuple[int, int, int]:
    output = (
        int(
            h
        ),
        int(
            k
        ),
        int(
            l
        ),
    )

    if output == (
        0,
        0,
        0,
    ):
        raise ValueError(
            f"{label} cannot be [0 0 0]."
        )

    return output


def _cell_vectors_from_prism(
    *,
    lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    period_vector: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    """
    Construct the periodic CIF lattice from the current prism dimensions.

    New CIF axes:
        a = base direction 1 × axis1_cells
        b = base direction 2 × axis2_cells
        c = current realization repeat vector

    For a non-stacked particle, c is the prism-axis vector multiplied by
    axis3_cells.

    For stacked deterministic/stochastic particles, c is derived from the
    displayed realization, including its actual lateral drift and total stack
    height.
    """
    shape_type = str(
        shape_spec.shape_type
        or ""
    ).strip().lower()

    if shape_type not in {
        "prism",
        "rectangular_prism",
    }:
        raise ValueError(
            "Periodic realization CIF export is available only for prism shapes."
        )

    axis1_cells = _safe_integer_count(
        shape_spec.axis1_cells,
        label="Prism base axis 1",
    )

    axis2_cells = _safe_integer_count(
        shape_spec.axis2_cells,
        label="Prism base axis 2",
    )

    base1 = _direction(
        shape_spec.base1_h,
        shape_spec.base1_k,
        shape_spec.base1_l,
        label="Prism base direction 1",
    )

    base2 = _direction(
        shape_spec.base2_h,
        shape_spec.base2_k,
        shape_spec.base2_l,
        label="Prism base direction 2",
    )

    axis3 = _direction(
        shape_spec.axis_h,
        shape_spec.axis_k,
        shape_spec.axis_l,
        label="Prism axis",
    )

    vector_a = (
        float(
            axis1_cells
        )
        * direct_hkl_to_cart(
            *base1,
            lattice_matrix,
        )
    )

    vector_b = (
        float(
            axis2_cells
        )
        * direct_hkl_to_cart(
            *base2,
            lattice_matrix,
        )
    )

    vector_c = np.asarray(
        period_vector,
        dtype=float,
    )

    if (
        np.linalg.norm(
            vector_a
        )
        <= _EPSILON
        or np.linalg.norm(
            vector_b
        )
        <= _EPSILON
        or np.linalg.norm(
            vector_c
        )
        <= _EPSILON
    ):
        raise ValueError(
            "One of the exported CIF lattice vectors has zero length."
        )

    lattice_out = np.asarray(
        [
            vector_a,
            vector_b,
            vector_c,
        ],
        dtype=float,
    )

    determinant = float(
        np.linalg.det(
            lattice_out
        )
    )

    if abs(
        determinant
    ) <= _EPSILON:
        raise ValueError(
            "The requested prism/realization lattice is singular.\n\n"
            "The selected base directions and stacking axis must define "
            "three independent directions."
        )

    # CIF/Pymatgen expects a right-handed lattice.
    #
    # Reverse b if the selected directions form a left-handed basis.
    if determinant < 0.0:
        lattice_out[
            1,
            :
        ] *= -1.0

        determinant = -determinant

        base2 = (
            -base2[0],
            -base2[1],
            -base2[2],
        )

    information = {
        "axis1_cells": int(
            axis1_cells
        ),
        "axis2_cells": int(
            axis2_cells
        ),
        "base1_direction": base1,
        "base2_direction": base2,
        "axis_direction": axis3,
        "volume": float(
            determinant
        ),
    }

    return (
        lattice_out,
        information,
    )


def _stochastic_period_vector(
    *,
    source_lattice_matrix: np.ndarray,
    states: np.ndarray,
    stack_spec: LayerStackSpec,
    parameters: Dict[str, Any],
) -> np.ndarray:
    """
    Build the periodic repeat vector of the currently displayed stochastic path.

    Every state emits its vector. The full selected realization is therefore
    made periodic by using the sum of all emitted state vectors as the new
    CIF c vector.
    """
    from pdf_fitting.models.compiled_stochastic_geometry import (
        _generic_state_vectors,
    )

    states = np.asarray(
        states,
        dtype=np.int32,
    ).reshape(
        -1
    )

    if states.size == 0:
        raise ValueError(
            "The stochastic realization contains no layers."
        )

    vector_a, vector_b, normal = layer_basis(
        source_lattice_matrix
    )

    state_du, state_dv, state_h, _labels = (
        _generic_state_vectors(
            stack_spec,
            parameters,
        )
    )

    state_du = np.asarray(
        state_du,
        dtype=float,
    )

    state_dv = np.asarray(
        state_dv,
        dtype=float,
    )

    state_h = np.asarray(
        state_h,
        dtype=float,
    )

    if not np.all(
        (
            states >= 0
        )
        & (
            states < state_du.size
        )
    ):
        raise ValueError(
            "The current stochastic realization contains an invalid state index."
        )

    total_du = float(
        np.sum(
            state_du[
                states
            ]
        )
    )

    total_dv = float(
        np.sum(
            state_dv[
                states
            ]
        )
    )

    total_h = float(
        np.sum(
            state_h[
                states
            ]
        )
    )

    if total_h <= _EPSILON:
        raise ValueError(
            "The current stochastic realization has non-positive total height."
        )

    return (
        total_du
        * vector_a
        + total_dv
        * vector_b
        + total_h
        * normal
    )


def _nonstacked_period_vector(
    *,
    source_lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    include_closing_gap: bool = True,
) -> np.ndarray:
    """
    Return the periodic c vector for a non-stacked prism.

    For a non-stacked finite prism there is no explicit layer-stacking
    closing transition. Therefore the prism axis size defines the displayed
    cell thickness.

    When include_closing_gap is True, one additional parent-cell translation
    is retained after the displayed prism thickness. This prevents the last
    structural slice from being placed directly against the first slice of
    the next periodic CIF repeat.
    """
    axis3_cells = _safe_integer_count(
        shape_spec.axis3_cells,
        label="Prism axis 3",
    )

    axis3 = _direction(
        shape_spec.axis_h,
        shape_spec.axis_k,
        shape_spec.axis_l,
        label="Prism axis",
    )

    repeat_cells = int(
        axis3_cells
    )

    if include_closing_gap:
        repeat_cells += 1

    period_vector = (
        float(
            repeat_cells
        )
        * direct_hkl_to_cart(
            *axis3,
            source_lattice_matrix,
        )
    )

    if (
        np.linalg.norm(
            period_vector
        )
        <= _EPSILON
    ):
        raise ValueError(
            "The non-stacked prism produced an invalid periodic c vector."
        )

    return np.asarray(
        period_vector,
        dtype=float,
    )


def _deterministic_period_vector(
    *,
    source_lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
) -> np.ndarray:
    """
    Return the periodic c vector for the currently displayed deterministic
    realization.

    The exported cell must include the closing transition from the final
    displayed layer to the first layer of the next repeated realization.

    For N displayed layers, this is calculated as:

        origin(layer N)
        -
        origin(layer 0)

    where layer N is the first layer immediately after the N displayed
    positions.

    This guarantees that the CIF repeat preserves:

        - the spacing after the last displayed layer;
        - the lateral closing translation;
        - the correct sequence-row transition;
        - the correct repeat-period behavior.
    """
    from pdf_fitting.models.layer_stacking import (
        realize_n_layers,
        realized_layer_translation,
    )

    number_of_layers = _safe_integer_count(
        shape_spec.axis3_cells,
        label="Number of complete layers",
    )

    stack_spec = stack_spec.normalized()

    # Build one extra layer. The extra layer is the first layer of the
    # next periodic copy after the currently displayed realization.
    realized_layers = realize_n_layers(
        stack_spec,
        number_of_layers=(
            number_of_layers
            + 1
        ),
        center_stack=False,
    )

    if len(
        realized_layers
    ) < (
        number_of_layers
        + 1
    ):
        raise RuntimeError(
            "Could not construct the closing layer needed for "
            "periodic CIF export."
        )

    first_layer = realized_layers[
        0
    ]

    next_first_layer = realized_layers[
        number_of_layers
    ]

    first_origin = realized_layer_translation(
        first_layer,
        source_lattice_matrix,
    )

    next_first_origin = realized_layer_translation(
        next_first_layer,
        source_lattice_matrix,
    )

    period_vector = (
        next_first_origin
        - first_origin
    )

    if (
        not np.all(
            np.isfinite(
                period_vector
            )
        )
        or np.linalg.norm(
            period_vector
        )
        <= _EPSILON
    ):
        raise ValueError(
            "The deterministic realization produced an invalid "
            "periodic closing vector."
        )

    return np.asarray(
        period_vector,
        dtype=float,
    )
def build_periodic_structure_from_displayed_particle(
    *,
    atom_cart: np.ndarray,
    atom_elements,
    source_lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    period_vector: np.ndarray,
) -> Tuple[
    Structure,
    Dict[str, Any],
]:
    """
    Construct a periodic P1 Structure containing the displayed finite prism.

    The displayed finite particle is wrapped into the new prism/realization
    cell. Repeating the exported CIF repeats that exact finite realization.
    """
    atom_cart = np.asarray(
        atom_cart,
        dtype=float,
    )

    if (
        atom_cart.ndim != 2
        or atom_cart.shape[1] != 3
        or atom_cart.shape[0] == 0
    ):
        raise ValueError(
            "The displayed realization contains no atoms to export."
        )

    if len(
        atom_elements
    ) != atom_cart.shape[0]:
        raise ValueError(
            "The displayed realization has inconsistent atom labels."
        )

    lattice_matrix, information = _cell_vectors_from_prism(
        lattice_matrix=source_lattice_matrix,
        shape_spec=shape_spec,
        period_vector=period_vector,
    )

    lattice = Lattice(
        lattice_matrix
    )

    inverse = np.linalg.inv(
        lattice_matrix
    )

    fractional = (
        atom_cart
        @ inverse
    )

    fractional %= 1.0

    fractional[
        np.isclose(
            fractional,
            1.0,
            atol=1.0e-10,
        )
    ] = 0.0

    periodic_structure = Structure(
        lattice=lattice,
        species=list(
            atom_elements
        ),
        coords=fractional,
        coords_are_cartesian=False,
        to_unit_cell=True,
        validate_proximity=False,
    )

    information.update(
        {
            "number_of_atoms": int(
                len(
                    periodic_structure.sites
                )
            ),
            "lattice_a": float(
                periodic_structure.lattice.a
            ),
            "lattice_b": float(
                periodic_structure.lattice.b
            ),
            "lattice_c": float(
                periodic_structure.lattice.c
            ),
            "lattice_alpha": float(
                periodic_structure.lattice.alpha
            ),
            "lattice_beta": float(
                periodic_structure.lattice.beta
            ),
            "lattice_gamma": float(
                periodic_structure.lattice.gamma
            ),
            "lattice_matrix": lattice_matrix,
        }
    )

    return (
        periodic_structure,
        information,
    )


def export_displayed_prism_realization_cif(
    *,
    atom_cart: np.ndarray,
    atom_elements,
    source_lattice_matrix: np.ndarray,
    shape_spec: CrystalliteShapeSpec,
    period_vector: np.ndarray,
    path: str,
) -> Dict[str, Any]:
    """
    Export a periodic CIF made from the current displayed finite prism.
    """
    periodic_structure, information = (
        build_periodic_structure_from_displayed_particle(
            atom_cart=atom_cart,
            atom_elements=atom_elements,
            source_lattice_matrix=source_lattice_matrix,
            shape_spec=shape_spec,
            period_vector=period_vector,
        )
    )

    writer = CifWriter(
        periodic_structure,
        symprec=None,
    )

    writer.write_file(
        path
    )

    information[
        "path"
    ] = str(
        path
    )

    return information