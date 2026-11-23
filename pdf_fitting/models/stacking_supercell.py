from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Any, Dict, List, Optional

import json
import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
    absolute_sequence_layers,
    effective_repeat_translation,
    layer_basis,
)


_EPSILON = 1.0e-12


@dataclass
class StackingSupercell:
    """
    Periodic bulk reference generated from one stacking sequence period.

    This object represents the infinite periodic stacking law. It is not the
    finite N-layer crystallite.

    Coordinate conventions
    ----------------------
    lattice_matrix
        Row-vector lattice matrix:

            Cartesian = fractional @ lattice_matrix

    atom_frac
        Atom coordinates wrapped into the periodic stacking supercell.

    atom_cart
        Cartesian coordinates corresponding to atom_frac.

    atom_cart_unwrapped
        Cartesian coordinates before periodic wrapping. These are useful for
        debugging the stacking sequence.

    atom_sequence_indices
        Sequence-table row that generated each atom.

    atom_source_site_indices
        Input structural-layer site that generated each atom.
    """

    structure: Structure

    lattice_matrix: np.ndarray
    inverse_lattice_matrix: np.ndarray
    reciprocal_lattice_matrix: np.ndarray

    vector_a: np.ndarray
    vector_b: np.ndarray
    vector_c_stack: np.ndarray
    stack_normal: np.ndarray

    repeat_period: float
    repeat_shift_u: float
    repeat_shift_v: float

    nominal_spacing: float
    period_normal_height: float

    atom_frac: np.ndarray
    atom_cart: np.ndarray
    atom_cart_unwrapped: np.ndarray

    atom_sequence_indices: np.ndarray
    atom_source_site_indices: np.ndarray
    atom_labels: List[str]
    atom_elements: List[str]

    normal_phase_shift: float

    volume: float

    geometry_hash: str

    symmetry_symbol: Optional[str] = None
    symmetry_number: Optional[int] = None

    @property
    def number_of_atoms(self) -> int:
        return int(
            self.atom_frac.shape[0]
        )

    @property
    def number_of_sequence_layers(self) -> int:
        if self.atom_sequence_indices.size == 0:
            return 0

        return int(
            np.unique(
                self.atom_sequence_indices
            ).size
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Return lightweight serializable diagnostic information.
        """
        return {
            "number_of_atoms": self.number_of_atoms,
            "number_of_sequence_layers": (
                self.number_of_sequence_layers
            ),
            "repeat_period": float(
                self.repeat_period
            ),
            "repeat_shift_u": float(
                self.repeat_shift_u
            ),
            "repeat_shift_v": float(
                self.repeat_shift_v
            ),
            "nominal_spacing": float(
                self.nominal_spacing
            ),
            "period_normal_height": float(
                self.period_normal_height
            ),
            "volume": float(
                self.volume
            ),
            "normal_phase_shift": float(
                self.normal_phase_shift
            ),
            "geometry_hash": str(
                self.geometry_hash
            ),
            "symmetry_symbol": self.symmetry_symbol,
            "symmetry_number": self.symmetry_number,
            "lattice_matrix": (
                self.lattice_matrix.tolist()
            ),
        }


def _clean_element_symbol(
    value: str,
) -> str:
    """
    Extract a display element symbol from a pymatgen species string.
    """
    text = str(
        value
        or ""
    ).strip()

    if not text:
        return "X"

    first = text[0].upper()

    if (
        len(text) > 1
        and text[1].islower()
    ):
        return (
            first
            + text[1].lower()
        )

    return first


def _validate_lattice_matrix(
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        lattice_matrix,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "The lattice matrix must have shape (3, 3)."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            "The lattice matrix contains non-finite values."
        )

    determinant = float(
        np.linalg.det(
            matrix
        )
    )

    if abs(determinant) <= _EPSILON:
        raise ValueError(
            "The lattice matrix is singular."
        )

    return matrix


def _center_input_layer_along_normal(
    structure: Structure,
    lattice_matrix: np.ndarray,
    stack_normal: np.ndarray,
) -> np.ndarray:
    """
    Return one input structural layer in Cartesian coordinates.

    Fractional a and b coordinates are wrapped into one in-plane cell. The
    layer is centered only along the stacking normal.

    It is deliberately not centered in the a-b plane because a and b define
    the periodic in-plane origin of the stacking supercell.
    """
    fractional = np.asarray(
        structure.frac_coords,
        dtype=float,
    ).copy()

    if (
        fractional.ndim != 2
        or fractional.shape[1] != 3
    ):
        raise ValueError(
            "The structure fractional coordinates must have shape (N, 3)."
        )

    if fractional.shape[0] == 0:
        raise ValueError(
            "The input structure contains no atoms."
        )

    fractional[:, 0] %= 1.0
    fractional[:, 1] %= 1.0

    cartesian = (
        fractional
        @ lattice_matrix
    )

    normal_coordinates = (
        cartesian
        @ stack_normal
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

    cartesian = (
        cartesian
        - normal_center
        * stack_normal[
            None,
            :
        ]
    )

    return np.asarray(
        cartesian,
        dtype=float,
    )


def _largest_gap_phase_shift(
    normal_offsets: np.ndarray,
    period_height: float,
) -> float:
    """
    Choose a periodic normal-origin shift that places the supercell boundary
    in the largest gap between layer centers.

    For a regular A stack this places the layer in the middle of the cell.

    For a regular AB stack this places A and B away from the upper/lower
    supercell boundaries.

    The result is a global Cartesian displacement along the stacking normal.
    It does not change the periodic stacking geometry.
    """
    offsets = np.asarray(
        normal_offsets,
        dtype=float,
    ).reshape(
        -1
    )

    period_height = max(
        abs(
            float(
                period_height
            )
        ),
        _EPSILON,
    )

    if offsets.size == 0:
        return 0.0

    residues = np.mod(
        offsets,
        period_height,
    )

    residues.sort()

    if residues.size == 1:
        boundary = (
            residues[0]
            + 0.5
            * period_height
        ) % period_height

        return float(
            -boundary
        )

    next_values = np.concatenate(
        (
            residues[1:],
            residues[:1]
            + period_height,
        )
    )

    gaps = (
        next_values
        - residues
    )

    largest_gap_index = int(
        np.argmax(
            gaps
        )
    )

    boundary = (
        residues[
            largest_gap_index
        ]
        + 0.5
        * gaps[
            largest_gap_index
        ]
    ) % period_height

    return float(
        -boundary
    )


def _site_property_keys(
    structure: Structure,
) -> List[str]:
    keys = set()

    for site in structure.sites:
        properties = getattr(
            site,
            "properties",
            {},
        ) or {}

        keys.update(
            properties.keys()
        )

    return sorted(
        str(key)
        for key in keys
    )


def _geometry_hash(
    *,
    structure: Structure,
    stack_spec: LayerStackSpec,
    lattice_matrix: np.ndarray,
) -> str:
    """
    Build a stable hash for the periodic stacking-supercell geometry.

    This will later be used by the stacked pair/shell cache.
    """
    structure_rows = []

    for site in structure.sites:
        structure_rows.append(
            {
                "species": str(
                    site.species
                ),
                "fractional": [
                    round(
                        float(value),
                        12,
                    )
                    for value in site.frac_coords
                ],
            }
        )

    payload = {
        "version": (
            "stacking_supercell_v1"
        ),
        "lattice_matrix": [
            [
                round(
                    float(value),
                    12,
                )
                for value in row
            ]
            for row in np.asarray(
                lattice_matrix,
                dtype=float,
            )
        ],
        "structure": structure_rows,
        "stacking": (
            stack_spec.normalized().to_dict()
        ),
    }

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return md5(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def build_stacking_supercell(
    *,
    structure: Structure,
    stack_spec: LayerStackSpec,
    lattice_matrix: Optional[np.ndarray] = None,
    analyze_symmetry: bool = False,
    symmetry_tolerance: float = 1.0e-3,
) -> StackingSupercell:
    """
    Construct one periodic stacking supercell.

    Parameters
    ----------
    structure
        Input structure containing one structural layer.

    stack_spec
        Infinite periodic stacking definition.

    lattice_matrix
        Optional current/refined parent lattice matrix. If omitted, the
        structure's current lattice matrix is used.

        Supplying this argument is important for the later dynamic backend:
        the stacking supercell can then be rebuilt from the current lattice
        parameters without modifying the original Structure.

    analyze_symmetry
        If True, run pymatgen SpacegroupAnalyzer on the generated stacking
        supercell. This is deliberately optional because symmetry analysis can
        be expensive and should not occur during normal GUI preview updates.

    Returns
    -------
    StackingSupercell
        Periodic bulk reference containing one complete stacking period.
    """
    if structure is None:
        raise ValueError(
            "A structure is required."
        )

    spec = stack_spec.normalized()

    if lattice_matrix is None:
        parent_matrix = np.asarray(
            structure.lattice.matrix,
            dtype=float,
        )
    else:
        parent_matrix = np.asarray(
            lattice_matrix,
            dtype=float,
        )

    parent_matrix = _validate_lattice_matrix(
        parent_matrix
    )

    vector_a, vector_b, stack_normal = layer_basis(
        parent_matrix
    )

    nominal_spacing = max(
        abs(
            float(
                spec.nominal_spacing
            )
        ),
        _EPSILON,
    )

    absolute_layers = absolute_sequence_layers(
        spec
    )

    (
        effective_repeat_u,
        effective_repeat_v,
        effective_repeat_period,
    ) = effective_repeat_translation(
        spec
    )

    repeat_period = max(
        abs(
            float(
                effective_repeat_period
            )
        ),
        _EPSILON,
    )

    period_normal_height = (
        repeat_period
        * nominal_spacing
    )

    vector_c_stack = (
        float(
            effective_repeat_u
        )
        * vector_a
        + float(
            effective_repeat_v
        )
        * vector_b
        + period_normal_height
        * stack_normal
    )


    stack_matrix = np.asarray(
        [
            vector_a,
            vector_b,
            vector_c_stack,
        ],
        dtype=float,
    )

    stack_matrix = _validate_lattice_matrix(
        stack_matrix
    )

    inverse_stack_matrix = np.linalg.inv(
        stack_matrix
    )

    stack_lattice = Lattice(
        stack_matrix
    )

    reciprocal_matrix = np.asarray(
        stack_lattice
        .reciprocal_lattice_crystallographic
        .matrix,
        dtype=float,
    )

    base_atom_cart = _center_input_layer_along_normal(
        structure,
        parent_matrix,
        stack_normal,
    )

    enabled_rows = [
        (
            sequence_index,
            sequence_layer,
        )
        for (
            sequence_index,
            sequence_layer,
        ) in enumerate(
            absolute_layers
        )
        if bool(
            sequence_layer.enabled
        )
    ]

    if not enabled_rows:
        raise ValueError(
            "The stacking sequence contains no enabled layers."
        )

    layer_normal_offsets = np.asarray(
        [
            float(
                layer.stack_coordinate
            )
            * nominal_spacing
            + float(
                layer.normal_delta
            )
            for (
                _sequence_index,
                layer,
            ) in enabled_rows
        ],
        dtype=float,
    )

    normal_phase_shift = _largest_gap_phase_shift(
        layer_normal_offsets,
        period_normal_height,
    )

    species = []

    atom_cart_unwrapped_rows = []
    atom_sequence_indices = []
    atom_source_site_indices = []
    atom_labels: List[str] = []
    atom_elements: List[str] = []

    original_property_keys = _site_property_keys(
        structure
    )

    site_properties: Dict[
        str,
        List[Any],
    ] = {
        key: []
        for key in original_property_keys
    }

    site_properties.update(
        {
            "stack_sequence_index": [],
            "stack_sequence_label": [],
            "stack_source_site_index": [],
        }
    )

    for sequence_index, sequence_layer in enabled_rows:
        layer_translation = (
            float(
                sequence_layer.shift_u
            )
            * vector_a
            + float(
                sequence_layer.shift_v
            )
            * vector_b
            + (
                float(
                    sequence_layer.stack_coordinate
                )
                * nominal_spacing
                + float(
                    sequence_layer.normal_delta
                )
                + normal_phase_shift
            )
            * stack_normal
        )

        layer_atoms = (
            base_atom_cart
            + layer_translation[
                None,
                :
            ]
        )

        for source_site_index, site in enumerate(
            structure.sites
        ):
            species.append(
                site.species
            )

            atom_cart_unwrapped_rows.append(
                layer_atoms[
                    source_site_index
                ]
            )

            atom_sequence_indices.append(
                int(
                    sequence_index
                )
            )

            atom_source_site_indices.append(
                int(
                    source_site_index
                )
            )

            atom_labels.append(
                str(
                    sequence_layer.label
                    or f"L{sequence_index}"
                )
            )

            atom_elements.append(
                _clean_element_symbol(
                    str(
                        site.species
                    )
                )
            )

            original_properties = getattr(
                site,
                "properties",
                {},
            ) or {}

            for property_key in original_property_keys:
                site_properties[
                    property_key
                ].append(
                    original_properties.get(
                        property_key,
                        None,
                    )
                )

            site_properties[
                "stack_sequence_index"
            ].append(
                int(
                    sequence_index
                )
            )

            site_properties[
                "stack_sequence_label"
            ].append(
                str(
                    sequence_layer.label
                    or f"L{sequence_index}"
                )
            )

            site_properties[
                "stack_source_site_index"
            ].append(
                int(
                    source_site_index
                )
            )

    atom_cart_unwrapped = np.asarray(
        atom_cart_unwrapped_rows,
        dtype=float,
    ).reshape(
        -1,
        3,
    )

    atom_frac_unwrapped = (
        atom_cart_unwrapped
        @ inverse_stack_matrix
    )

    atom_frac = np.mod(
        atom_frac_unwrapped,
        1.0,
    )

    # Remove floating-point values infinitesimally below/above cell edges.
    atom_frac[
        np.isclose(
            atom_frac,
            0.0,
            atol=1.0e-12,
        )
    ] = 0.0

    atom_frac[
        np.isclose(
            atom_frac,
            1.0,
            atol=1.0e-12,
        )
    ] = 0.0

    atom_cart = (
        atom_frac
        @ stack_matrix
    )

    stack_structure = Structure(
        lattice=stack_lattice,
        species=species,
        coords=atom_frac,
        coords_are_cartesian=False,
        to_unit_cell=False,
        validate_proximity=False,
        site_properties=site_properties,
    )

    symmetry_symbol: Optional[str] = None
    symmetry_number: Optional[int] = None

    if analyze_symmetry:
        try:
            analyzer = SpacegroupAnalyzer(
                stack_structure,
                symprec=float(
                    symmetry_tolerance
                ),
            )

            symmetry_symbol = str(
                analyzer.get_space_group_symbol()
            )

            symmetry_number = int(
                analyzer.get_space_group_number()
            )

        except Exception:
            symmetry_symbol = "P1"
            symmetry_number = 1

    geometry_hash = _geometry_hash(
        structure=structure,
        stack_spec=spec,
        lattice_matrix=parent_matrix,
    )

    volume = abs(
        float(
            np.linalg.det(
                stack_matrix
            )
        )
    )

    return StackingSupercell(
        structure=stack_structure,
        lattice_matrix=np.asarray(
            stack_matrix,
            dtype=float,
        ),
        inverse_lattice_matrix=np.asarray(
            inverse_stack_matrix,
            dtype=float,
        ),
        reciprocal_lattice_matrix=np.asarray(
            reciprocal_matrix,
            dtype=float,
        ),
        vector_a=np.asarray(
            vector_a,
            dtype=float,
        ),
        vector_b=np.asarray(
            vector_b,
            dtype=float,
        ),
        vector_c_stack=np.asarray(
            vector_c_stack,
            dtype=float,
        ),
        stack_normal=np.asarray(
            stack_normal,
            dtype=float,
        ),
        repeat_period=float(
            repeat_period
        ),
        repeat_shift_u=float(
            effective_repeat_u
        ),
        repeat_shift_v=float(
            effective_repeat_v
        ),
        nominal_spacing=float(
            nominal_spacing
        ),
        period_normal_height=float(
            period_normal_height
        ),
        atom_frac=np.asarray(
            atom_frac,
            dtype=float,
        ),
        atom_cart=np.asarray(
            atom_cart,
            dtype=float,
        ),
        atom_cart_unwrapped=np.asarray(
            atom_cart_unwrapped,
            dtype=float,
        ),
        atom_sequence_indices=np.asarray(
            atom_sequence_indices,
            dtype=np.int32,
        ),
        atom_source_site_indices=np.asarray(
            atom_source_site_indices,
            dtype=np.int32,
        ),
        atom_labels=list(
            atom_labels
        ),
        atom_elements=list(
            atom_elements
        ),
        normal_phase_shift=float(
            normal_phase_shift
        ),
        volume=float(
            volume
        ),
        geometry_hash=str(
            geometry_hash
        ),
        symmetry_symbol=symmetry_symbol,
        symmetry_number=symmetry_number,
    )


def rebuild_stacking_supercell(
    previous: StackingSupercell,
    *,
    source_structure: Structure,
    stack_spec: LayerStackSpec,
    lattice_matrix: np.ndarray,
    analyze_symmetry: bool = False,
    symmetry_tolerance: float = 1.0e-3,
) -> StackingSupercell:
    """
    Rebuild a stacking supercell for a new lattice or stacking distance.

    The ``previous`` argument makes the intended use explicit and will later
    allow topology caches to be reused. In this phase the geometry is rebuilt
    completely for correctness.
    """
    _ = previous

    return build_stacking_supercell(
        structure=source_structure,
        stack_spec=stack_spec,
        lattice_matrix=lattice_matrix,
        analyze_symmetry=analyze_symmetry,
        symmetry_tolerance=symmetry_tolerance,
    )