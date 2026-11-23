from __future__ import annotations

import copy
import json
import os

from typing import (
    Any,
    Mapping,
)

import numpy as np

from pdf_fitting.models.bulk_stacking_realization import (
    BulkStackingRealization,
    export_bulk_realization,
)


def _realization_paths(
    *,
    realization_directory: str,
    realization_index: int,
    geometry_key: str,
):
    realization_directory = os.path.abspath(
        realization_directory
    )

    realization_subdirectory = os.path.join(
        realization_directory,
        (
            f"realization_"
            f"{int(realization_index):04d}"
        ),
    )

    os.makedirs(
        realization_subdirectory,
        exist_ok=True,
    )

    stem = (
        "geometry_"
        + str(
            geometry_key
        )[
            :24
        ]
    )

    cif_path = os.path.join(
        realization_subdirectory,
        stem + ".cif",
    )

    manifest_path = os.path.join(
        realization_subdirectory,
        stem + ".json",
    )

    return (
        cif_path,
        manifest_path,
    )


def _manifest_matches(
    manifest_path: str,
    geometry_key: str,
) -> bool:
    if not os.path.exists(
        manifest_path
    ):
        return False

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(
                file
            )

        return str(
            manifest.get(
                "geometry_key",
                "",
            )
        ) == str(
            geometry_key
        )

    except Exception:
        return False


def _write_realization_bundle(
    *,
    realization: BulkStackingRealization,
    cif_path: str,
    manifest_path: str,
    geometry_key: str,
) -> None:
    export_bulk_realization(
        realization,
        cif_path=cif_path,
        manifest_path=manifest_path,
    )

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(
                file
            )

    except Exception:
        manifest = {}

    manifest[
        "geometry_key"
    ] = str(
        geometry_key
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )


def _realization_pdf_config(
    *,
    base_config: Mapping[
        str,
        Any,
    ],
    cif_path: str,
) -> dict:
    config = copy.deepcopy(
        dict(
            base_config
            or {}
        )
    )

    config[
        "structure_file"
    ] = os.path.abspath(
        cif_path
    )

    config[
        "layer_stacking"
    ] = {
        "enabled": False,
    }

    config[
        "stochastic_ensemble"
    ] = {
        "use_stacking_as_bulk": False,
    }

    # The generated realization is already a bulk periodic structure.
    # Disable finite-shape replacement and analytical crystallite envelopes.
    shape_config = dict(
        config.get(
            "crystallite_shape",
            {},
        )
        or {}
    )

    shape_config[
        "mode"
    ] = "conventional"

    shape_config[
        "search_mode"
    ] = "predefined"

    config[
        "crystallite_shape"
    ] = shape_config

    refinement_config = dict(
        config.get(
            "refinement",
            {},
        )
        or {}
    )

    refinement_config[
        "defer_cuda_pdf_backend"
    ] = False

    # Random realization structures are P1 approximants. Do not allow
    # SpacegroupAnalyzer to detect accidental realization-dependent
    # symmetry and reduce the center-site set.
    refinement_config[
        "force_p1_shells"
    ] = True

    config[
        "refinement"
    ] = refinement_config

    return config

def open_bulk_realization_calculator(
    *,
    base_config: Mapping[
        str,
        Any,
    ],
    realization: BulkStackingRealization,
    realization_directory: str,
    geometry_key: str,
    pair_cutoff: float,
    r_grid: np.ndarray,
    export_cif: bool = True,
):
    """
    Create one PDFCalculator from the in-memory periodic realization.

    CIF files are optional diagnostic exports. They are never parsed back
    into the numerical model.
    """
    if export_cif:
        (
            cif_path,
            manifest_path,
        ) = _realization_paths(
            realization_directory=realization_directory,
            realization_index=(
                realization.realization_index
            ),
            geometry_key=geometry_key,
        )

        bundle_valid = bool(
            os.path.exists(
                cif_path
            )
            and _manifest_matches(
                manifest_path,
                geometry_key,
            )
        )

        if not bundle_valid:
            _write_realization_bundle(
                realization=realization,
                cif_path=cif_path,
                manifest_path=manifest_path,
                geometry_key=geometry_key,
            )

    config = copy.deepcopy(
        dict(
            base_config
            or {}
        )
    )

    config[
        "layer_stacking"
    ] = {
        "enabled": False,
    }

    config[
        "stochastic_ensemble"
    ] = {
        "use_stacking_as_bulk": False,
    }

    # No finite crystallite envelope is applied to a bulk realization.
    # This does not disable the independent SAXS belly.
    config[
        "crystallite_shape"
    ] = {
        "mode": "conventional",
        "search_mode": "predefined",
    }

    refinement_config = dict(
        config.get(
            "refinement",
            {},
        )
        or {}
    )

    refinement_config[
        "defer_cuda_pdf_backend"
    ] = False

    # Random realization supercells must be treated as P1. Accidental
    # realization-dependent symmetry must not change shell normalization.
    refinement_config[
        "force_p1_shells"
    ] = True

    # Preserve the user's SAXS settings.
    config[
        "refinement"
    ] = refinement_config

    from pdf_fitting.models.gr_model import (
        PDFCalculator,
    )

    calculator = PDFCalculator(
        config,
        realization.structure,
        np.asarray(
            r_grid,
            dtype=float,
        ),
        pair_cutoff=float(
            pair_cutoff
        ),
    )

    return calculator