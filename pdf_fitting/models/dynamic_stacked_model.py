from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import hashlib
import json
import os
import threading
import time

import re

import numpy as np


from pdf_fitting.models.compiled_stacked_geometry import (
    CompiledLayerDescriptor,
    CompiledStackedGeometry,
    compile_stacked_geometry,
)


from pdf_fitting.models.compiled_stacked_pairs import (
    CompiledStackedPairTopology,
    EvaluatedStackedPairs,
    compile_stacked_pair_topology,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)



from pdf_fitting.user_paths import (
    user_cache_dir,
)

from pdf_fitting.models.gpu.backend import (
    resolve_backend,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    layer_basis,
)

from pdf_fitting.models.stacking_transitions import (
    evaluate_transition_values,
    transition_parameter_names,
)

_DYNAMIC_STACKED_TOPOLOGY_CACHE: Dict[
    str,
    Dict[str, Any],
] = {}

_DYNAMIC_STACKED_TOPOLOGY_CACHE_LOCK = threading.RLock()

_DYNAMIC_STACKED_TOPOLOGY_CACHE_MAX_ENTRIES = 4

_DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_VERSION = (
    "dynamic_stacked_topology_disk_v6_exact_directions"
)

_DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_DIR = user_cache_dir(
    "dynamic_stacked_topology_cache"
)


def _rounded_float(
    value: Any,
    default: float,
) -> float:
    try:
        return round(
            float(
                value
            ),
            10,
        )
    except Exception:
        return float(
            default
        )


def _stacking_topology_cache_key(
    *,
    source_structure,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: LayerStackSpec,
    pair_cutoff: float,
    topology_margin: float,
    reference_parameters: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Build an in-memory cache key for compiled stacked topology.

    The topology is independent of continuously refined parameters:

        a
        b
        c
        alpha
        beta
        gamma
        stacking_distance
        stacking_fraction
        scale
        biso_*
        qdamp
        delta_g
        delta_broad
        delta1_*
        delta2_*
        lambda_*
        SAXS parameters

    Those parameters are evaluated dynamically after topology is loaded.

    The cache key only contains static/discrete topology information:

        source fractional coordinates and species;
        finite shape type, dimensions, and orientation;
        layer count;
        periodic stacking sequence;
        repeat translation;
        pair cutoff;
        topology margin.

    ``reference_parameters`` is intentionally excluded. It remains an
    argument only because the caller has this information available.
    """
    _ = reference_parameters

    structure_rows = []

    for site in source_structure.sites:
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

    stack_dict = stack_spec.to_dict()

    for key in (
        "refine_nominal_spacing",
        "nominal_spacing_min",
        "nominal_spacing_max",
        "refine_stacking_fraction",
        "stacking_fraction_min",
        "stacking_fraction_max",
    ):
        stack_dict.pop(
            key,
            None,
        )

    sequence = []

    for layer in stack_dict.get(
        "sequence",
        [],
    ):
        if not isinstance(
            layer,
            Mapping,
        ):
            continue

        sequence.append(
            {
                "enabled": bool(
                    layer.get(
                        "enabled",
                        True,
                    )
                ),
                "label": str(
                    layer.get(
                        "label",
                        "",
                    )
                ),
                "shift_u": _rounded_float(
                    layer.get(
                        "shift_u",
                        0.0,
                    ),
                    0.0,
                ),
                "shift_v": _rounded_float(
                    layer.get(
                        "shift_v",
                        0.0,
                    ),
                    0.0,
                ),
                "stack_coordinate": _rounded_float(
                    layer.get(
                        "stack_coordinate",
                        0.0,
                    ),
                    0.0,
                ),
                "normal_delta": _rounded_float(
                    layer.get(
                        "normal_delta",
                        0.0,
                    ),
                    0.0,
                ),
            }
        )

    stack_dict[
        "sequence"
    ] = sequence

    payload = {
        "cache_version": (
            "dynamic_stacked_topology_memory_v5_generic_transitions"
        ),
        "structure": structure_rows,
        "shape_spec": shape_spec.to_dict(),
        "stack_spec": stack_dict,
        "pair_cutoff": round(
            float(
                pair_cutoff
            ),
            8,
        ),
        "topology_margin": round(
            float(
                topology_margin
            ),
            8,
        ),
    }

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def clear_dynamic_stacked_topology_memory_cache(
) -> None:
    """
    Clear all in-memory compiled stacked topologies.

    This is mainly useful for development, testing, or manual memory release.
    """
    with _DYNAMIC_STACKED_TOPOLOGY_CACHE_LOCK:
        _DYNAMIC_STACKED_TOPOLOGY_CACHE.clear()


def _store_dynamic_stacked_topology_cache(
    cache_key: str,
    value: Dict[str, Any],
) -> None:
    """
    Store one compiled dynamic stacked topology.

    Keep only a small number of entries because each topology can contain
    large NumPy arrays.
    """
    with _DYNAMIC_STACKED_TOPOLOGY_CACHE_LOCK:
        _DYNAMIC_STACKED_TOPOLOGY_CACHE[
            cache_key
        ] = value

        while (
            len(
                _DYNAMIC_STACKED_TOPOLOGY_CACHE
            )
            > _DYNAMIC_STACKED_TOPOLOGY_CACHE_MAX_ENTRIES
        ):
            oldest_key = next(
                iter(
                    _DYNAMIC_STACKED_TOPOLOGY_CACHE
                )
            )

            _DYNAMIC_STACKED_TOPOLOGY_CACHE.pop(
                oldest_key,
                None,
            )

def _dynamic_stacked_topology_disk_paths(
    cache_key: str,
) -> tuple[str, str]:
    key = str(
        cache_key
    ).strip()

    array_path = os.path.join(
        _DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_DIR,
        f"dynamic_stacked_topology_{key}.npz",
    )

    metadata_path = os.path.join(
        _DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_DIR,
        f"dynamic_stacked_topology_{key}.json",
    )

    return (
        array_path,
        metadata_path,
    )


def _save_dynamic_stacked_topology_disk_cache(
    *,
    cache_key: str,
    compiled_geometry: CompiledStackedGeometry,
    compiled_pairs: CompiledStackedPairTopology,
    shell_order: np.ndarray,
) -> None:
    """
    Save immutable stacked topology arrays.

    The current evaluated Cartesian geometry is intentionally not saved because
    it must be recalculated from the active refinement parameters.
    """
    os.makedirs(
        _DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_DIR,
        exist_ok=True,
    )

    array_path, metadata_path = (
        _dynamic_stacked_topology_disk_paths(
            cache_key
        )
    )

    metadata = {
        "cache_version": (
            _DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_VERSION
        ),
        "cache_key": str(
            cache_key
        ),
        "shape_spec": (
            compiled_geometry.shape_spec.to_dict()
        ),
        "stack_spec": (
            compiled_geometry.stack_spec.to_dict()
        ),
        "source_elements": list(
            compiled_geometry.source_elements
        ),
        "layers": [
            {
                "realized_index": int(
                    layer.realized_index
                ),
                "sequence_index": int(
                    layer.sequence_index
                ),
                "repeat_index": int(
                    layer.repeat_index
                ),
                "enabled": bool(
                    layer.enabled
                ),
                "label": str(
                    layer.label
                ),
                "shift_u": float(
                    layer.shift_u
                ),
                "shift_v": float(
                    layer.shift_v
                ),
                "stack_coordinate": float(
                    layer.stack_coordinate
                ),
                "normal_delta": float(
                    layer.normal_delta
                ),
            }
            for layer in compiled_geometry.layers
        ],
        "reference_spacing": float(
            compiled_geometry.reference_spacing
        ),
        "pair_cutoff": float(
            compiled_pairs.pair_cutoff
        ),
        "topology_cutoff": float(
            compiled_pairs.topology_cutoff
        ),
        "unique_species": list(
            compiled_pairs.unique_species
        ),
        "finite_pair_rows": int(
            compiled_pairs.finite_pair_rows
        ),
        "bulk_pair_rows": int(
            compiled_pairs.bulk_pair_rows
        ),
    }

    temporary_array_path = (
        array_path
        + ".tmp"
    )

    temporary_metadata_path = (
        metadata_path
        + ".tmp"
    )

    try:
        with open(
            temporary_array_path,
            "wb",
        ) as file:
            np.savez(
                file,
                source_fractional=np.asarray(
                    compiled_geometry.source_fractional,
                    dtype=np.float64,
                ),
                atom_source_site_indices=np.asarray(
                    compiled_geometry.atom_source_site_indices,
                    dtype=np.int32,
                ),
                atom_layer_descriptor_indices=np.asarray(
                    compiled_geometry.atom_layer_descriptor_indices,
                    dtype=np.int32,
                ),
                atom_cell_a=np.asarray(
                    compiled_geometry.atom_cell_a,
                    dtype=np.int32,
                ),
                atom_cell_b=np.asarray(
                    compiled_geometry.atom_cell_b,
                    dtype=np.int32,
                ),
                reference_lattice_matrix=np.asarray(
                    compiled_geometry.reference_lattice_matrix,
                    dtype=np.float64,
                ),
                shell_source_alpha=np.asarray(
                    compiled_pairs.shell_source_alpha,
                    dtype=np.int32,
                ),
                shell_source_beta=np.asarray(
                    compiled_pairs.shell_source_beta,
                    dtype=np.int32,
                ),
                shell_delta_fractional=np.asarray(
                    compiled_pairs.shell_delta_fractional,
                    dtype=np.float64,
                ),
                shell_delta_a=np.asarray(
                    compiled_pairs.shell_delta_a,
                    dtype=np.float64,
                ),
                shell_delta_b=np.asarray(
                    compiled_pairs.shell_delta_b,
                    dtype=np.float64,
                ),

                shell_mult=np.asarray(
                    compiled_pairs.shell_mult,
                    dtype=np.int32,
                ),
                shell_bulk_mult=np.asarray(
                    compiled_pairs.shell_bulk_mult,
                    dtype=np.int32,
                ),
                shell_alpha_id=np.asarray(
                    compiled_pairs.shell_alpha_id,
                    dtype=np.int16,
                ),
                shell_beta_id=np.asarray(
                    compiled_pairs.shell_beta_id,
                    dtype=np.int16,
                ),
                finite_species_counts=np.asarray(
                    compiled_pairs.finite_species_counts,
                    dtype=np.int64,
                ),
                bulk_species_counts=np.asarray(
                    compiled_pairs.bulk_species_counts,
                    dtype=np.int64,
                ),
                shell_hkl=np.asarray(
                    compiled_pairs.shell_hkl,
                    dtype=np.int16,
                ),
                shell_direction=np.asarray(
                    compiled_pairs.shell_direction,
                    dtype=np.int16,
                ),
                reference_distance=np.asarray(
                    compiled_pairs.reference_distance,
                    dtype=np.float64,
                ),
                shell_order=np.asarray(
                    shell_order,
                    dtype=np.int64,
                ),
                shell_transition_coefficients=np.asarray(
                    compiled_pairs.shell_transition_coefficients,
                    dtype=np.int32,
                ),
                reference_transition_du=np.asarray(
                    compiled_pairs.reference_transition_du,
                    dtype=np.float64,
                ),
                reference_transition_dv=np.asarray(
                    compiled_pairs.reference_transition_dv,
                    dtype=np.float64,
                ),
                reference_transition_h=np.asarray(
                    compiled_pairs.reference_transition_h,
                    dtype=np.float64,
                ),
            )

        with open(
            temporary_metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        os.replace(
            temporary_array_path,
            array_path,
        )

        os.replace(
            temporary_metadata_path,
            metadata_path,
        )

        print(
            "[DYNAMIC STACKED CACHE] "
            f"Saved disk topology cache: {array_path}"
        )

    except Exception as exc:
        print(
            "[DYNAMIC STACKED CACHE] "
            f"Could not save disk topology cache: {exc}"
        )

        for path in (
            temporary_array_path,
            temporary_metadata_path,
        ):
            try:
                if os.path.exists(
                    path
                ):
                    os.remove(
                        path
                    )
            except Exception:
                pass


def _load_dynamic_stacked_topology_disk_cache(
    *,
    cache_key: str,
    source_structure,
) -> Optional[
    Dict[str, Any]
]:
    """
    Load immutable compiled stacked topology from disk.

    Current Cartesian shell vectors, shell distances, active masks, density,
    and finite gamma values are not loaded. They are recalculated dynamically.
    """
    array_path, metadata_path = (
        _dynamic_stacked_topology_disk_paths(
            cache_key
        )
    )

    if not (
        os.path.exists(
            array_path
        )
        and os.path.exists(
            metadata_path
        )
    ):
        return None

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        if (
            metadata.get(
                "cache_version"
            )
            != _DYNAMIC_STACKED_TOPOLOGY_DISK_CACHE_VERSION
        ):
            return None

        if str(
            metadata.get(
                "cache_key",
                "",
            )
        ) != str(
            cache_key
        ):
            return None

        arrays = np.load(
            array_path,
            allow_pickle=False,
        )

        shape_spec = CrystalliteShapeSpec.from_dict(
            metadata.get(
                "shape_spec",
                {},
            )
        )

        stack_spec = LayerStackSpec.from_dict(
            metadata.get(
                "stack_spec",
                {},
            )
        )

        layers = [
            CompiledLayerDescriptor(
                realized_index=int(
                    row[
                        "realized_index"
                    ]
                ),
                sequence_index=int(
                    row[
                        "sequence_index"
                    ]
                ),
                repeat_index=int(
                    row[
                        "repeat_index"
                    ]
                ),
                enabled=bool(
                    row[
                        "enabled"
                    ]
                ),
                label=str(
                    row[
                        "label"
                    ]
                ),
                shift_u=float(
                    row[
                        "shift_u"
                    ]
                ),
                shift_v=float(
                    row[
                        "shift_v"
                    ]
                ),
                stack_coordinate=float(
                    row[
                        "stack_coordinate"
                    ]
                ),
                normal_delta=float(
                    row[
                        "normal_delta"
                    ]
                ),
            )
            for row in metadata.get(
                "layers",
                [],
            )
        ]

        compiled_geometry = CompiledStackedGeometry(
            source_structure=source_structure,
            shape_spec=shape_spec,
            stack_spec=stack_spec,
            source_fractional=np.asarray(
                arrays[
                    "source_fractional"
                ],
                dtype=np.float64,
            ),
            source_elements=list(
                metadata.get(
                    "source_elements",
                    [],
                )
            ),
            layers=layers,
            atom_source_site_indices=np.asarray(
                arrays[
                    "atom_source_site_indices"
                ],
                dtype=np.int32,
            ),
            atom_layer_descriptor_indices=np.asarray(
                arrays[
                    "atom_layer_descriptor_indices"
                ],
                dtype=np.int32,
            ),
            atom_cell_a=np.asarray(
                arrays[
                    "atom_cell_a"
                ],
                dtype=np.int32,
            ),
            atom_cell_b=np.asarray(
                arrays[
                    "atom_cell_b"
                ],
                dtype=np.int32,
            ),
            reference_lattice_matrix=np.asarray(
                arrays[
                    "reference_lattice_matrix"
                ],
                dtype=np.float64,
            ),
            reference_spacing=float(
                metadata[
                    "reference_spacing"
                ]
            ),
        )

        compiled_pairs = CompiledStackedPairTopology(
            geometry=compiled_geometry,
            pair_cutoff=float(
                metadata[
                    "pair_cutoff"
                ]
            ),
            topology_cutoff=float(
                metadata[
                    "topology_cutoff"
                ]
            ),
            shell_source_alpha=np.asarray(
                arrays[
                    "shell_source_alpha"
                ],
                dtype=np.int32,
            ),
            shell_source_beta=np.asarray(
                arrays[
                    "shell_source_beta"
                ],
                dtype=np.int32,
            ),
            shell_delta_fractional=np.asarray(
                arrays[
                    "shell_delta_fractional"
                ],
                dtype=np.float64,
            ),
            shell_delta_a=np.asarray(
                arrays[
                    "shell_delta_a"
                ],
                dtype=np.float64,
            ),
            shell_delta_b=np.asarray(
                arrays[
                    "shell_delta_b"
                ],
                dtype=np.float64,
            ),
            shell_transition_coefficients=np.asarray(
                arrays[
                    "shell_transition_coefficients"
                ],
                dtype=np.int32,
            ),
            reference_transition_du=np.asarray(
                arrays[
                    "reference_transition_du"
                ],
                dtype=np.float64,
            ),
            reference_transition_dv=np.asarray(
                arrays[
                    "reference_transition_dv"
                ],
                dtype=np.float64,
            ),
            reference_transition_h=np.asarray(
                arrays[
                    "reference_transition_h"
                ],
                dtype=np.float64,
            ),
            shell_mult=np.asarray(
                arrays[
                    "shell_mult"
                ],
                dtype=np.int32,
            ),
            shell_bulk_mult=np.asarray(
                arrays[
                    "shell_bulk_mult"
                ],
                dtype=np.int32,
            ),
            shell_alpha_id=np.asarray(
                arrays[
                    "shell_alpha_id"
                ],
                dtype=np.int16,
            ),
            shell_beta_id=np.asarray(
                arrays[
                    "shell_beta_id"
                ],
                dtype=np.int16,
            ),
            finite_species_counts=np.asarray(
                arrays[
                    "finite_species_counts"
                ],
                dtype=np.int64,
            ),
            bulk_species_counts=np.asarray(
                arrays[
                    "bulk_species_counts"
                ],
                dtype=np.int64,
            ),
            unique_species=list(
                metadata.get(
                    "unique_species",
                    [],
                )
            ),
            shell_hkl=np.asarray(
                arrays[
                    "shell_hkl"
                ],
                dtype=np.int16,
            ),
            shell_direction=np.asarray(
                arrays[
                    "shell_direction"
                ],
                dtype=np.int16,
            ),
            reference_distance=np.asarray(
                arrays[
                    "reference_distance"
                ],
                dtype=np.float64,
            ),
            finite_pair_rows=int(
                metadata[
                    "finite_pair_rows"
                ]
            ),
            bulk_pair_rows=int(
                metadata[
                    "bulk_pair_rows"
                ]
            ),
        )

        shell_order = np.asarray(
            arrays[
                "shell_order"
            ],
            dtype=np.int64,
        )

        reference_evaluated = compiled_pairs.evaluate(
            {
                "a": float(
                    source_structure.lattice.a
                ),
                "b": float(
                    source_structure.lattice.b
                ),
                "c": float(
                    source_structure.lattice.c
                ),
                "alpha": float(
                    source_structure.lattice.alpha
                ),
                "beta": float(
                    source_structure.lattice.beta
                ),
                "gamma": float(
                    source_structure.lattice.gamma
                ),
            }
        )

        (
            initial_shell_table,
            rebuilt_shell_order,
        ) = _initial_shell_table(
            compiled_pairs,
            reference_evaluated,
        )

        if not np.array_equal(
            shell_order,
            rebuilt_shell_order,
        ):
            shell_order = rebuilt_shell_order

        print(
            "[DYNAMIC STACKED CACHE] "
            f"Disk hit: loaded compiled topology from {array_path}"
        )

        return {
            "compiled_geometry": compiled_geometry,
            "compiled_pairs": compiled_pairs,
            "shell_order": shell_order,
            "initial_shell_table": initial_shell_table,
        }

    except Exception as exc:
        print(
            "[DYNAMIC STACKED CACHE] "
            f"Could not load disk topology cache: {exc}"
        )

        return None

    
def _lambda_safe_label(
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
        return "x"

    symbol = match.group(
        1
    )

    if len(symbol) == 1:
        symbol = symbol.upper()
    else:
        symbol = (
            symbol[0].upper()
            + symbol[1].lower()
        )

    return symbol.lower()


def _rank_within_unordered_pair(
    shell_upair_id: np.ndarray,
    reference_distance: np.ndarray,
) -> np.ndarray:
    """
    Rank unique reference distances inside each unordered pair channel.

    Directionally distinct shell descriptors at the same distance receive
    the same lambda-shell rank.
    """
    shell_upair_id = np.asarray(
        shell_upair_id,
        dtype=np.int32,
    )

    distance_key = np.rint(
        np.asarray(
            reference_distance,
            dtype=float,
        )
        * 1.0e5
    ).astype(
        np.int64,
    )

    number_of_shells = int(
        shell_upair_id.size
    )

    output = np.zeros(
        number_of_shells,
        dtype=np.int32,
    )

    if number_of_shells == 0:
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


def _initial_shell_table(
    topology: CompiledStackedPairTopology,
    evaluated: EvaluatedStackedPairs,
) -> tuple[Dict[str, Any], np.ndarray]:
    """
    Build the static packed shell-table metadata used by PDFCalculator.

    Returns
    -------
    shell_table
        Packed shell table sorted by ordered pair ID.

    shell_order
        Mapping from topology descriptor order to packed shell-table order.
        Dynamic arrays must be reordered with this array before updating
        PDFCalculator.
    """
    unique_species = list(
        topology.unique_species
    )

    safe_labels = [
        _lambda_safe_label(
            species
        )
        for species in unique_species
    ]

    shell_alpha_id = np.asarray(
        topology.shell_alpha_id,
        dtype=np.int16,
    )

    shell_beta_id = np.asarray(
        topology.shell_beta_id,
        dtype=np.int16,
    )

    pair_values = sorted(
        {
            (
                safe_labels[
                    int(alpha_id)
                ],
                safe_labels[
                    int(beta_id)
                ],
            )
            for alpha_id, beta_id in zip(
                shell_alpha_id,
                shell_beta_id,
            )
        }
    )

    pair_list = list(
        pair_values
    )

    pair_index = {
        pair: index
        for index, pair in enumerate(
            pair_list
        )
    }

    unordered_values = sorted(
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
        unordered_values
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
                    safe_labels[
                        int(alpha_id)
                    ],
                    safe_labels[
                        int(beta_id)
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
                            safe_labels[
                                int(alpha_id)
                            ],
                            safe_labels[
                                int(beta_id)
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

    shell_k_within_upair = (
        _rank_within_unordered_pair(
            shell_upair_id,
            topology.reference_distance,
        )
    )

    shell_k_rank = np.arange(
        shell_pair_id.size,
        dtype=np.int32,
    )

    shell_order = np.argsort(
        shell_pair_id,
        kind="mergesort",
    )

    shell_pair_id = shell_pair_id[
        shell_order
    ]

    shell_upair_id = shell_upair_id[
        shell_order
    ]

    shell_k_within_upair = (
        shell_k_within_upair[
            shell_order
        ]
    )

    shell_k_rank = shell_k_rank[
        shell_order
    ]

    shell_alpha_id = shell_alpha_id[
        shell_order
    ]

    shell_beta_id = shell_beta_id[
        shell_order
    ]

    shell_mult = np.asarray(
        topology.shell_mult,
        dtype=np.int32,
    )[
        shell_order
    ]

    shell_bulk_mult = np.asarray(
        topology.shell_bulk_mult,
        dtype=np.int32,
    )[
        shell_order
    ]

    shell_hkl = np.asarray(
        topology.shell_hkl,
        dtype=np.int16,
    )[
        shell_order
    ]

    shell_direction = np.asarray(
        topology.shell_direction,
        dtype=np.int16,
    )[
        shell_order
    ]

    shell_frac = np.asarray(
        evaluated.shell_frac,
        dtype=np.float32,
    )[
        shell_order
    ]

    shell_cart = np.asarray(
        evaluated.shell_cart,
        dtype=np.float32,
    )[
        shell_order
    ]

    shell_mult_eff = np.asarray(
        evaluated.shell_mult_eff,
        dtype=np.float32,
    )[
        shell_order
    ]

    shell_bulk_mult_eff = np.asarray(
        evaluated.shell_bulk_mult_eff,
        dtype=np.float32,
    )[
        shell_order
    ]

    shell_gamma = np.asarray(
        evaluated.shell_gamma,
        dtype=np.float32,
    )[
        shell_order
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
            rows[0]
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

    stacking_matrix = np.asarray(
        evaluated.geometry.stacking_lattice_matrix,
        dtype=float,
    )

    stacking_volume = abs(
        float(
            np.linalg.det(
                stacking_matrix
            )
        )
    )

    enabled_sequence_rows = sum(
        1
        for layer in topology.geometry.stack_spec.sequence
        if bool(
            layer.enabled
        )
    )

    bulk_atom_count = (
        enabled_sequence_rows
        * len(
            topology.geometry.source_structure.sites
        )
    )

    shell_table = {
        "geometry_mode": (
            "dynamic_stacked_supercell"
        ),
        "shell_frac": shell_frac,
        "shell_cart": shell_cart,
        "shell_mult": shell_mult,
        "shell_bulk_mult": shell_bulk_mult,
        "shell_mult_eff": shell_mult_eff,
        "shell_finite_mult_eff": (
            shell_mult_eff.copy()
        ),
        "shell_bulk_mult_eff": (
            shell_bulk_mult_eff
        ),
        "shell_gamma": shell_gamma,
        "shell_pair_id": shell_pair_id,
        "shell_alpha_id": shell_alpha_id,
        "shell_beta_id": shell_beta_id,
        "shell_hkl": shell_hkl,
        "shell_direction": shell_direction,
        "shell_k_rank": shell_k_rank,
        "shell_upair_id": shell_upair_id,
        "shell_k_within_upair": (
            shell_k_within_upair
        ),
        "pair_offsets": pair_offsets,
        "pair_alpha_id": pair_alpha_id,
        "pair_beta_id": pair_beta_id,
        "unique_species": unique_species,
        "pair_list": pair_list,
        "upair_list": upair_list,
        "strain_unique_dirs": strain_unique_dirs,
        "strain_inv_dirs": strain_inv_dirs,
        "stacking_lattice_matrix": (
            stacking_matrix
        ),
        "stacking_inverse_lattice_matrix": np.asarray(
            evaluated.geometry.stacking_inverse_lattice_matrix,
            dtype=float,
        ),
        "stacking_volume": float(
            stacking_volume
        ),
        "stacking_bulk_number_density": float(
            evaluated.geometry.number_density
        ),
        "bulk_atom_count": int(
            bulk_atom_count
        ),
        "stacking_nominal_spacing": float(
            evaluated.geometry.spacing
        ),
        "stacking_repeat_period": float(
            topology.geometry.stack_spec.repeat_period
        ),
        "stacking_period_normal_height": float(
            evaluated.geometry.period_height
        ),
        "finite_atom_count": int(
            evaluated.geometry.number_of_atoms
        ),
        "number_of_layer_positions": int(
            evaluated.geometry.number_of_layers
        ),
    }

    return (
        shell_table,
        np.asarray(
            shell_order,
            dtype=np.int64,
        ),
    )


class DynamicStackedPdfModel:
    """
    Dynamic finite-stacked PDF model.

    Static/discrete compilation occurs once:

        sequence topology;
        complete-layer count;
        AB envelope;
        finite atom identities;
        finite pair topology;
        periodic bulk pair topology;
        pair IDs and directions.

    Geometry evaluation occurs when geometry parameters change:

        parent lattice;
        stacking normal;
        interlayer spacing/fraction;
        atom Cartesian coordinates;
        shell vectors/distances;
        stacking-cell metric and density.

    The underlying PDFCalculator handles all PDF physics after receiving
    updated geometry arrays.
    """

    def __init__(
        self,
        *,
        pdf_calculator,
        source_structure,
        shape_spec: CrystalliteShapeSpec,
        stack_spec: LayerStackSpec,
        pair_cutoff: float,
        reference_parameters: Optional[
            Mapping[str, Any]
        ] = None,
        topology_margin: float = 1.0,
    ):
        self.pdf_calculator = pdf_calculator
        self.source_structure = source_structure
        self.shape_spec = (
            CrystalliteShapeSpec.from_dict(
                shape_spec.to_dict()
            )
        )
        self.stack_spec = (
            stack_spec.normalized()
        )
        self.pair_cutoff = float(
            pair_cutoff
        )
        self.topology_margin = float(
            topology_margin
        )

        self.reference_parameters = dict(
            reference_parameters
            or {}
        )


        self._topology_cache_key = (
            _stacking_topology_cache_key(
                source_structure=(
                    self.source_structure
                ),
                shape_spec=self.shape_spec,
                stack_spec=self.stack_spec,
                pair_cutoff=self.pair_cutoff,
                topology_margin=self.topology_margin,
                reference_parameters=(
                    self.reference_parameters
                ),
            )
        )

        cached_topology = None

        with _DYNAMIC_STACKED_TOPOLOGY_CACHE_LOCK:
            cached_topology = (
                _DYNAMIC_STACKED_TOPOLOGY_CACHE.get(
                    self._topology_cache_key,
                    None,
                )
            )

        if cached_topology is None:
            cached_topology = (
                _load_dynamic_stacked_topology_disk_cache(
                    cache_key=self._topology_cache_key,
                    source_structure=self.source_structure,
                )
            )

            if cached_topology is not None:
                _store_dynamic_stacked_topology_cache(
                    self._topology_cache_key,
                    cached_topology,
                )

        if cached_topology is not None:
            self.compiled_geometry = cached_topology[
                "compiled_geometry"
            ]

            self.compiled_pairs = cached_topology[
                "compiled_pairs"
            ]

            self._shell_order = cached_topology[
                "shell_order"
            ]

            initial_shell_table = cached_topology[
                "initial_shell_table"
            ]

            print(
                "[DYNAMIC STACKED CACHE] "
                "Memory hit: reusing compiled geometry and pair topology."
            )

        else:
            topology_build_start = time.perf_counter()

            geometry_build_start = time.perf_counter()

            self.compiled_geometry = (
                compile_stacked_geometry(
                    structure=self.source_structure,
                    shape_spec=self.shape_spec,
                    stack_spec=self.stack_spec,
                    reference_parameters=(
                        self.reference_parameters
                    ),
                )
            )

            geometry_build_seconds = (
                time.perf_counter()
                - geometry_build_start
            )

            pair_topology_build_start = (
                time.perf_counter()
            )

            refinement_config = (
                self.pdf_calculator.config.get(
                    "refinement",
                    {},
                )
                or {}
            )

            self.compiled_pairs = (
                compile_stacked_pair_topology(
                    geometry=self.compiled_geometry,
                    pair_cutoff=self.pair_cutoff,
                    topology_margin=self.topology_margin,
                    reference_parameters=(
                        self.reference_parameters
                    ),
                    shell_backend=str(
                        refinement_config.get(
                            "shell_backend",
                            "cpu",
                        )
                    ),
                    gpu_device=int(
                        refinement_config.get(
                            "gpu_device",
                            0,
                        )
                    ),
                    gpu_memory_fraction=float(
                        refinement_config.get(
                            "gpu_memory_fraction",
                            0.70,
                        )
                    ),
                )
            )

            pair_topology_build_seconds = (
                time.perf_counter()
                - pair_topology_build_start
            )

            reference_evaluated = (
                self.compiled_pairs.evaluate(
                    self.reference_parameters
                )
            )

            (
                initial_shell_table,
                self._shell_order,
            ) = _initial_shell_table(
                self.compiled_pairs,
                reference_evaluated,
            )

            total_topology_build_seconds = (
                time.perf_counter()
                - topology_build_start
            )

            _store_dynamic_stacked_topology_cache(
                self._topology_cache_key,
                {
                    "compiled_geometry": (
                        self.compiled_geometry
                    ),
                    "compiled_pairs": (
                        self.compiled_pairs
                    ),
                    "shell_order": (
                        self._shell_order
                    ),
                    "initial_shell_table": (
                        initial_shell_table
                    ),
                },
            )

            _save_dynamic_stacked_topology_disk_cache(
                cache_key=self._topology_cache_key,
                compiled_geometry=self.compiled_geometry,
                compiled_pairs=self.compiled_pairs,
                shell_order=self._shell_order,
            )

            print(
                "[DYNAMIC STACKED CACHE] "
                "Memory miss: compiled new topology. "
                f"geometry={geometry_build_seconds:.3f} s, "
                f"pairs={pair_topology_build_seconds:.3f} s, "
                f"total={total_topology_build_seconds:.3f} s"
            )

        self.pdf_calculator.replace_shell_table(
            initial_shell_table
        )


        self._last_geometry_key = None
        self._geometry_update_count = 0
        self._last_geometry = None
        self._last_parameters = dict(
            self.reference_parameters
        )
        self._last_evaluated_pairs = None

        self._dynamic_cuda_geometry_backend = None
        self._dynamic_cuda_geometry_active = False

        self._dynamic_cuda_pdf_backend = None
        self._dynamic_cuda_pdf_active = False

        self._initialize_dynamic_cuda_pdf_backend()

        self._apply_geometry(
            self.reference_parameters,
            force=True,
        )

    def _initialize_dynamic_cuda_pdf_backend(
        self,
    ) -> None:
        """
        Resident GPU backend for arbitrary transition coefficients
        (gpu.generic_stacked_pdf): shell vectors = Δfrac·L + (Δa + C·du)·â
        + (Δb + C·dv)·b̂ + (C·h)·n̂, then σ, amplitudes, Gaussians, finite-size
        and SAXS γ on the device.
        """
        self._dynamic_cuda_pdf_backend = None
        self._dynamic_cuda_pdf_active = False
        refinement_config = (
            self.pdf_calculator.config.get("refinement", {}) or {}
        )
        requested_backend = str(refinement_config.get("pdf_backend", "cpu") or "cpu").strip().lower()
        if requested_backend not in {"cuda", "gpu", "cupy", "auto"}:
            return
        try:
            from pdf_fitting.models.gpu.generic_stacked_pdf import GenericStackedCudaPdfBackend

            pairs = self.compiled_pairs
            calc = self.pdf_calculator
            alpha = np.asarray(pairs.shell_alpha_id, dtype=np.int64)
            fin = np.asarray(pairs.finite_species_counts, dtype=np.float64)
            blk = np.asarray(pairs.bulk_species_counts, dtype=np.float64)
            ok = (alpha >= 0) & (alpha < fin.size) & (alpha < blk.size)
            fden = np.zeros(alpha.size)
            bden = np.zeros(alpha.size)
            fden[ok] = fin[alpha[ok]]
            bden[ok] = blk[alpha[ok]]
            mult = np.asarray(pairs.shell_mult, dtype=np.float64)
            bmult = np.asarray(pairs.shell_bulk_mult, dtype=np.float64)
            mult_eff = np.where(fden > 0, mult / np.maximum(fden, 1e-30), 0.0)
            bulk_eff = np.where(bden > 0, bmult / np.maximum(bden, 1e-30), 0.0)
            # The calculator's shell table is the topology reordered by
            # _shell_order (sorted by ordered pair): upload in that order.
            order = np.asarray(getattr(self, "_shell_order", np.arange(alpha.size)), dtype=np.int64)
            self._dynamic_cuda_pdf_backend = GenericStackedCudaPdfBackend(
                delta_fractional=np.asarray(pairs.shell_delta_fractional)[order],
                delta_a=np.asarray(pairs.shell_delta_a)[order],
                delta_b=np.asarray(pairs.shell_delta_b)[order],
                transition_coefficients=np.asarray(pairs.shell_transition_coefficients)[order],
                shell_pair_id=np.asarray(calc._shell_pair_id, dtype=np.int32),
                shell_alpha_id=np.asarray(calc._shell_alpha_id, dtype=np.int32),
                shell_beta_id=np.asarray(calc._shell_beta_id, dtype=np.int32),
                shell_upair_id=np.asarray(calc._shell_upair_id, dtype=np.int32),
                shell_k_rank=np.asarray(calc._shell_k_rank, dtype=np.int32),
                shell_k_within_upair=np.asarray(calc._shell_k_within_upair, dtype=np.int32),
                shell_mult_eff=mult_eff[order],
                shell_bulk_mult_eff=bulk_eff[order],
                species_count=len(calc._unique_species_for_shells),
                pair_weights=np.asarray(calc._w_ordered, dtype=np.float32),
                pair_offsets=np.asarray(calc._pair_offsets, dtype=np.int32),
                pair_cutoff=float(pairs.pair_cutoff),
                device_id=int(refinement_config.get("gpu_device", 0) or 0),
            )
            self._dynamic_cuda_pdf_active = True
            print(
                f"[DYNAMIC STACKED GPU] {alpha.size:,} shells resident on the GPU "
                f"({self._dynamic_cuda_pdf_backend.memory_bytes() / 2**20:.0f} MiB)."
            )
        except Exception as exc:  # noqa: BLE001
            self._dynamic_cuda_pdf_backend = None
            self._dynamic_cuda_pdf_active = False
            print(f"[DYNAMIC STACKED GPU] GPU backend unavailable ({exc}); using the CPU.")

    def _initialize_dynamic_cuda_geometry_backend(
        self,
    ) -> None:
        """
        Generic transition geometry currently uses CPU evaluation.
        """
        self._dynamic_cuda_geometry_backend = None
        self._dynamic_cuda_geometry_active = False


    def __getattr__(
        self,
        name: str,
    ):
        """
        Proxy methods and attributes not implemented by this wrapper.
        """
        calculator = object.__getattribute__(
            self,
            "pdf_calculator",
        )

        return getattr(
            calculator,
            name,
        )

    @property
    def geometry_update_count(
        self,
    ) -> int:
        return int(
            self._geometry_update_count
        )

    def _geometry_key(
        self,
        parameters: Mapping[str, Any],
    ) -> Tuple[Any, ...]:
        """
        Return a cache key for every parameter that changes stacked geometry.
        """
        parameters = dict(
            parameters
            or {}
        )

        lattice = self.source_structure.lattice

        values = [
            float(
                parameters.get(
                    "a",
                    lattice.a,
                )
            ),
            float(
                parameters.get(
                    "b",
                    lattice.b,
                )
            ),
            float(
                parameters.get(
                    "c",
                    lattice.c,
                )
            ),
            float(
                parameters.get(
                    "alpha",
                    lattice.alpha,
                )
            ),
            float(
                parameters.get(
                    "beta",
                    lattice.beta,
                )
            ),
            float(
                parameters.get(
                    "gamma",
                    lattice.gamma,
                )
            ),
        ]

        (
            _transitions,
            transition_du,
            transition_dv,
            transition_h,
        ) = evaluate_transition_values(
            self.stack_spec,
            parameters,
        )

        values.extend(
            float(
                value
            )
            for value in transition_du
        )

        values.extend(
            float(
                value
            )
            for value in transition_dv
        )

        values.extend(
            float(
                value
            )
            for value in transition_h
        )

        return tuple(
            round(
                value,
                12,
            )
            for value in values
        )

    
    def _apply_geometry(
        self,
        parameters: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        geometry_key = self._geometry_key(
            parameters
        )

        if (
            not force
            and geometry_key
            == self._last_geometry_key
        ):
            return

        if (
            self._dynamic_cuda_geometry_active
            and self._dynamic_cuda_geometry_backend
            is not None
        ):
            evaluated_geometry = self.compiled_geometry.evaluate(
                parameters
            )

            lattice_matrix = np.asarray(
                evaluated_geometry.parent_lattice_matrix,
                dtype=float,
            )

            vector_a, vector_b, stack_normal = layer_basis(
                lattice_matrix
            )

            cuda_arrays = (
                self._dynamic_cuda_geometry_backend.evaluate(
                    lattice_matrix=lattice_matrix,
                    vector_a=vector_a,
                    vector_b=vector_b,
                    stack_normal=stack_normal,
                    spacing=float(
                        evaluated_geometry.spacing
                    ),
                    stacking_inverse_lattice_matrix=(
                        evaluated_geometry
                        .stacking_inverse_lattice_matrix
                    ),
                )
            )

            evaluated = EvaluatedStackedPairs(
                geometry=evaluated_geometry,
                shell_cart=np.asarray(
                    cuda_arrays[
                        "shell_cart"
                    ],
                    dtype=np.float64,
                ),
                shell_frac=np.asarray(
                    cuda_arrays[
                        "shell_frac"
                    ],
                    dtype=np.float64,
                ),
                shell_distance=np.asarray(
                    cuda_arrays[
                        "shell_distance"
                    ],
                    dtype=np.float64,
                ),
                shell_active=np.asarray(
                    cuda_arrays[
                        "shell_active"
                    ],
                    dtype=bool,
                ),
                shell_mult=np.asarray(
                    self.compiled_pairs.shell_mult,
                    dtype=np.int32,
                ),
                shell_bulk_mult=np.asarray(
                    self.compiled_pairs.shell_bulk_mult,
                    dtype=np.int32,
                ),
                shell_mult_eff=np.asarray(
                    cuda_arrays[
                        "shell_mult_eff"
                    ],
                    dtype=np.float32,
                ),
                shell_bulk_mult_eff=np.asarray(
                    cuda_arrays[
                        "shell_bulk_mult_eff"
                    ],
                    dtype=np.float32,
                ),
                shell_gamma=np.asarray(
                    cuda_arrays[
                        "shell_gamma"
                    ],
                    dtype=np.float32,
                ),
                shell_alpha_id=np.asarray(
                    self.compiled_pairs.shell_alpha_id,
                    dtype=np.int16,
                ),
                shell_beta_id=np.asarray(
                    self.compiled_pairs.shell_beta_id,
                    dtype=np.int16,
                ),
                shell_hkl=np.asarray(
                    self.compiled_pairs.shell_hkl,
                    dtype=np.int16,
                ),
                shell_direction=np.asarray(
                    self.compiled_pairs.shell_direction,
                    dtype=np.int16,
                ),
                unique_species=list(
                    self.compiled_pairs.unique_species
                ),
                pair_cutoff=float(
                    self.compiled_pairs.pair_cutoff
                ),
                topology_cutoff=float(
                    self.compiled_pairs.topology_cutoff
                ),
            )

        else:
            evaluated = self.compiled_pairs.evaluate(
                parameters
            )

        self._last_evaluated_pairs = evaluated

        order = self._shell_order

        stacking_matrix = np.asarray(
            evaluated.geometry.stacking_lattice_matrix,
            dtype=float,
        )

        stacking_volume = abs(
            float(
                np.linalg.det(
                    stacking_matrix
                )
            )
        )

        self.pdf_calculator.update_dynamic_stacked_geometry(
            shell_cart=np.asarray(
                evaluated.shell_cart,
                dtype=np.float32,
            )[
                order
            ],
            shell_frac=np.asarray(
                evaluated.shell_frac,
                dtype=np.float32,
            )[
                order
            ],
            shell_mult_eff=np.asarray(
                evaluated.shell_mult_eff,
                dtype=np.float32,
            )[
                order
            ],
            shell_bulk_mult_eff=np.asarray(
                evaluated.shell_bulk_mult_eff,
                dtype=np.float32,
            )[
                order
            ],
            shell_gamma=np.asarray(
                evaluated.shell_gamma,
                dtype=np.float32,
            )[
                order
            ],
            shell_active=np.asarray(
                evaluated.shell_active,
                dtype=bool,
            )[
                order
            ],
            stacking_lattice_matrix=(
                stacking_matrix
            ),
            stacking_inverse_lattice_matrix=np.asarray(
                evaluated.geometry.stacking_inverse_lattice_matrix,
                dtype=float,
            ),
            stacking_volume=float(
                stacking_volume
            ),
            stacking_bulk_number_density=float(
                evaluated.geometry.number_density
            ),
            stacking_spacing=float(
                evaluated.geometry.spacing
            ),
            stacking_period_height=float(
                evaluated.geometry.period_height
            ),
        )

        self._last_geometry_key = (
            geometry_key
        )

        self._geometry_update_count += 1

    @property
    def current_stacking_distance(
        self,
    ) -> float:
        if self._last_evaluated_pairs is None:
            return float(
                self.stack_spec.nominal_spacing
            )

        return float(
            self._last_evaluated_pairs.geometry.spacing
        )

    @property
    def current_stack_thickness(
        self,
    ) -> float:
        if self._last_evaluated_pairs is None:
            return 0.0

        return float(
            self._last_evaluated_pairs.geometry.boundary_thickness
        )

    def _gpu_lambda_delta_arrays(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
    ):
        parameters = dict(
            parameters
            or {}
        )

        global_lambda_values = []

        lambda_index = 0

        while (
            f"lambda_{lambda_index}"
            in parameters
        ):
            global_lambda_values.append(
                float(
                    parameters[
                        f"lambda_{lambda_index}"
                    ]
                )
            )

            lambda_index += 1

        pair_lambda_items = []

        for key, value in parameters.items():
            name = str(
                key
            ).strip()

            match = re.match(
                r"(?i)^lambda_([A-Za-z0-9]+)-([A-Za-z0-9]+)_(\d+)$",
                name,
            )

            if match is None:
                match = re.match(
                    r"(?i)^lambda_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)$",
                    name,
                )

            if match is None:
                continue

            first = match.group(
                1
            ).lower()

            second = match.group(
                2
            ).lower()

            first, second = sorted(
                (
                    first,
                    second,
                )
            )

            upair_id = self.pdf_calculator._upair_index.get(
                (
                    first,
                    second,
                ),
                None,
            )

            if upair_id is None:
                continue

            pair_lambda_items.append(
                (
                    int(
                        upair_id
                    ),
                    int(
                        match.group(
                            3
                        )
                    ),
                    float(
                        value
                    ),
                )
            )

        pair_lambda_values = None

        if pair_lambda_items:
            maximum_k = (
                max(
                    item[
                        1
                    ]
                    for item in pair_lambda_items
                )
                + 1
            )

            pair_lambda_values = np.full(
                (
                    len(
                        self.pdf_calculator
                        ._upair_list
                    ),
                    maximum_k,
                ),
                np.nan,
                dtype=np.float32,
            )

            for upair_id, shell_k, value in pair_lambda_items:
                pair_lambda_values[
                    upair_id,
                    shell_k,
                ] = np.float32(
                    value
                )

        global_delta1 = float(
            parameters.get(
                "delta1",
                0.0,
            )
        )

        global_delta2 = float(
            parameters.get(
                "delta2",
                0.0,
            )
        )

        pair_delta1 = np.full(
            len(
                self.pdf_calculator
                ._upair_list
            ),
            np.nan,
            dtype=np.float32,
        )

        pair_delta2 = np.full(
            len(
                self.pdf_calculator
                ._upair_list
            ),
            np.nan,
            dtype=np.float32,
        )

        for upair_id, pair in enumerate(
            self.pdf_calculator._upair_list
        ):
            first, second = pair

            delta1_dash = (
                f"delta1_{first}-{second}"
            )

            delta2_dash = (
                f"delta2_{first}-{second}"
            )

            delta1_underscore = (
                f"delta1_{first}_{second}"
            )

            delta2_underscore = (
                f"delta2_{first}_{second}"
            )

            if delta1_dash in parameters:
                pair_delta1[
                    upair_id
                ] = np.float32(
                    parameters[
                        delta1_dash
                    ]
                )

            elif delta1_underscore in parameters:
                pair_delta1[
                    upair_id
                ] = np.float32(
                    parameters[
                        delta1_underscore
                    ]
                )

            if delta2_dash in parameters:
                pair_delta2[
                    upair_id
                ] = np.float32(
                    parameters[
                        delta2_dash
                    ]
                )

            elif delta2_underscore in parameters:
                pair_delta2[
                    upair_id
                ] = np.float32(
                    parameters[
                        delta2_underscore
                    ]
                )

        return (
            np.asarray(
                global_lambda_values,
                dtype=np.float32,
            ),
            pair_lambda_values,
            float(
                global_delta1
            ),
            float(
                global_delta2
            ),
            pair_delta1,
            pair_delta2,
        )

    def _gpu_saxs_parameters(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
        *,
        lattice_matrix: np.ndarray,
    ) -> Dict[
        str,
        Any,
    ]:
        parameters = dict(
            parameters
            or {}
        )

        saxs_enabled = bool(
            self.pdf_calculator._saxs_enabled(
                parameters
            )
        )

        saxs_model = str(
            self.pdf_calculator._saxs_model_name(
                parameters
            )
            or "same"
        ).strip().lower()

        axis_h = int(
            round(
                float(
                    parameters.get(
                        "cyl_axis_h",
                        0.0,
                    )
                )
            )
        )

        axis_k = int(
            round(
                float(
                    parameters.get(
                        "cyl_axis_k",
                        0.0,
                    )
                )
            )
        )

        axis_l = int(
            round(
                float(
                    parameters.get(
                        "cyl_axis_l",
                        1.0,
                    )
                )
            )
        )

        if (
            axis_h == 0
            and axis_k == 0
            and axis_l == 0
        ):
            axis_l = 1

        cylinder_axis = np.asarray(
            [
                axis_h,
                axis_k,
                axis_l,
            ],
            dtype=float,
        ) @ np.asarray(
            lattice_matrix,
            dtype=float,
        )

        return {
            "saxs_enabled": saxs_enabled,
            "saxs_model": saxs_model,
            "saxs_scale": float(
                self.pdf_calculator._saxs_scale(
                    parameters
                )
            ),
            "saxs_diameter": float(
                parameters.get(
                    "saxs_diameter",
                    0.0,
                )
            ),
            "saxs_diameter_std": float(
                parameters.get(
                    "saxs_diameter_std",
                    0.0,
                )
            ),
            "saxs_height": float(
                parameters.get(
                    "saxs_height",
                    0.0,
                )
            ),
            "saxs_height_std": float(
                parameters.get(
                    "saxs_height_std",
                    0.0,
                )
            ),
            "saxs_apply_qdamp": bool(
                self.pdf_calculator._saxs_apply_qdamp(
                    parameters
                )
            ),
            "qdamp": float(
                parameters.get(
                    "qdamp",
                    0.0,
                )
            ),
            "cylinder_axis_cart": cylinder_axis,
        }

    def _can_use_dynamic_cuda_pdf(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
        kwargs: Mapping[
            str,
            Any,
        ],
    ) -> bool:
        """GPU path for plain G(r) when no CPU-only feature is active."""
        if not self._dynamic_cuda_pdf_active or self._dynamic_cuda_pdf_backend is None:
            return False
        for key, value in (kwargs or {}).items():
            if key != "r_override" and value:
                return False
        p = {str(k).lower(): v for k, v in dict(parameters or {}).items()}
        if getattr(self.pdf_calculator, "_adp", None) is not None:
            return False
        if float(p.get("cyl_diameter", 0.0) or 0.0) > 0.0:
            return False
        return True

    def evaluate(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        if not self._can_use_dynamic_cuda_pdf(
            parameters,
            kwargs,
        ):
            self._apply_geometry(
                parameters
            )

            return self.pdf_calculator.evaluate(
                parameters,
                **kwargs,
            )

        evaluated_geometry = self.compiled_geometry.evaluate(
            parameters
        )

        lattice_matrix = np.asarray(
            evaluated_geometry.parent_lattice_matrix,
            dtype=float,
        )

        vector_a, vector_b, stack_normal = layer_basis(
            lattice_matrix
        )

        r_override = kwargs.get(
            "r_override",
            None,
        )

        if r_override is None:
            r_requested = np.asarray(
                self.pdf_calculator.r,
                dtype=float,
            )
        else:
            r_requested = np.asarray(
                r_override,
                dtype=float,
            )

        qmax = float(
            parameters.get(
                "qmax",
                self.pdf_calculator.config.get(
                    "qmax",
                    0.0,
                ),
            )
        )

        qmax_zeros = int(
            parameters.get(
                "qmax_zeros",
                self.pdf_calculator.config.get(
                    "qmax_zeros",
                    5,
                ),
            )
        )

        qmax_pad = str(
            parameters.get(
                "qmax_pad",
                self.pdf_calculator.config.get(
                    "qmax_pad",
                    "zero",
                ),
            )
        ).strip().lower()

        use_internal_qmax_grid = False
        r_gpu = r_requested

        if (
            qmax > 0.0
            and r_requested.size > 1
        ):
            step = float(
                np.median(
                    np.diff(
                        r_requested
                    )
                )
            )

            if (
                np.isfinite(
                    step
                )
                and step > 0.0
            ):
                qmax_width = (
                    float(
                        qmax_zeros
                    )
                    * np.pi
                    / qmax
                )

                r_gpu = np.arange(
                    0.0,
                    float(
                        r_requested[
                            -1
                        ]
                    )
                    + qmax_width
                    + 0.5
                    * step,
                    step,
                    dtype=np.float32,
                )

                use_internal_qmax_grid = True

        (
            global_lambda_values,
            pair_lambda_values,
            global_delta1,
            global_delta2,
            pair_delta1,
            pair_delta2,
        ) = self._gpu_lambda_delta_arrays(
            parameters
        )

        lattice_volume = abs(
            float(
                np.linalg.det(
                    evaluated_geometry
                    .parent_lattice_matrix
                )
            )
        )

        (
            rho0,
            density_by_species,
        ) = (
            self.pdf_calculator
            ._current_number_density_and_species_densities(
                lattice_volume
            )
        )

        biso_by_species = (
            self.pdf_calculator
            ._biso_values_vector_from_params(
                parameters
            )
        )

        saxs_parameters = self._gpu_saxs_parameters(
            parameters,
            lattice_matrix=lattice_matrix,
        )

        (
            _transitions,
            transition_du,
            transition_dv,
            transition_h,
        ) = evaluate_transition_values(
            self.stack_spec,
            parameters,
        )
        from pdf_fitting.models.gpu.generic_stacked_pdf import configure_strain_from_calculator

        configure_strain_from_calculator(
            self._dynamic_cuda_pdf_backend, self.pdf_calculator, parameters,
            float(np.linalg.norm(np.asarray(lattice_matrix)[0])),
        )
        self._dynamic_cuda_pdf_backend.set_state_vectors(transition_du, transition_dv, transition_h)
        self._dynamic_cuda_pdf_backend.update_pair_weights(
            np.asarray(self.pdf_calculator._w_ordered, dtype=np.float32)
        )
        gpu_result = (
            self._dynamic_cuda_pdf_backend.evaluate(
                lattice_matrix=lattice_matrix,
                vector_a=vector_a,
                vector_b=vector_b,
                stack_normal=stack_normal,
                spacing=float(
                    evaluated_geometry.spacing
                ),
                global_lambda_values=global_lambda_values,
                pair_lambda_values=pair_lambda_values,
                global_delta1=global_delta1,
                global_delta2=global_delta2,
                pair_delta1=pair_delta1,
                pair_delta2=pair_delta2,
                biso_by_species=biso_by_species,
                density_by_species=density_by_species,
                delta_g=float(
                    parameters.get(
                        "delta_g",
                        0.0,
                    )
                ),
                delta_broad=float(
                    parameters.get(
                        "delta_broad",
                        0.0,
                    )
                ),
                r_grid=r_gpu,
                scale=float(
                    parameters.get(
                        "scale",
                        1.0,
                    )
                ),
                rho0=float(
                    rho0
                ),
                **saxs_parameters,
            )
        )

        G_gpu = np.asarray(
            gpu_result[
                "G_r"
            ],
            dtype=float,
        )

        if qmax > 0.0:
            from pdf_fitting.models.gr_model import (
                apply_qmax_termination,
            )

            G_gpu = apply_qmax_termination(
                G_gpu,
                np.asarray(
                    r_gpu,
                    dtype=float,
                ),
                qmax,
                n_zeros=qmax_zeros,
                pad_mode=qmax_pad,
                use_sine_kernel=True,
                normalize=False,
            )

        if use_internal_qmax_grid:
            G_requested = np.interp(
                r_requested,
                r_gpu,
                G_gpu,
            )

            g_requested = np.interp(
                r_requested,
                r_gpu,
                np.asarray(
                    gpu_result[
                        "g_r"
                    ],
                    dtype=float,
                ),
            )

        else:
            G_requested = G_gpu
            g_requested = np.asarray(
                gpu_result[
                    "g_r"
                ],
                dtype=float,
            )

        self._last_geometry_key = self._geometry_key(
            parameters
        )

        self._geometry_update_count += 1

        return {
            "r": np.asarray(
                r_requested,
                dtype=float,
            ),
            "G_r": np.asarray(
                G_requested,
                dtype=float,
            ),
            "g_r": np.asarray(
                g_requested,
                dtype=float,
            ),
            "g_ab": {},
            "G_ab": {},
            "G_contrib": {},
            "weights": {},
            "gamma_avg": None,
        }


    
    def build_warren_plot(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_warren_plot(
            parameters,
            **kwargs,
        )

    def build_local_trends(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_local_trends(
            parameters,
            **kwargs,
        )

    def compute_isotropic_shape_factor(
        self,
        parameters,
        r_grid,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return (
            self.pdf_calculator
            .compute_isotropic_shape_factor(
                parameters,
                r_grid,
            )
        )

    def build_shape_factor_directions(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return (
            self.pdf_calculator
            .build_shape_factor_directions(
                parameters,
                **kwargs,
            )
        )

    def build_pdf_tick_data(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_pdf_tick_data(
            parameters,
            **kwargs,
        )