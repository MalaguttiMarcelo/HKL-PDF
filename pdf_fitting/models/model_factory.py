from __future__ import annotations

import copy

from typing import (
    Any,
    Mapping,
)

import numpy as np


def _status(
    callback,
    message: str,
) -> None:
    if callback is None:
        print(
            f"[MODEL FACTORY] {message}",
            flush=True,
        )
        return

    try:
        callback(
            str(
                message
            )
        )
    except Exception:
        print(
            f"[MODEL FACTORY] {message}",
            flush=True,
        )


def build_pdf_model(
    *,
    config: Mapping[
        str,
        Any,
    ],
    r_exp: np.ndarray,
    pair_cutoff: float,
    stop_event=None,
    status_callback=None,
):
    """
    Build the correct PDF model for the current configuration.

    Selection order
    ---------------
    1. Stochastic stacking used as bulk:
           StreamingStackingBulkModel

    2. Existing finite stochastic stacking:
           StochasticStackedPdfModel

    3. Existing deterministic dynamic stacking:
           DynamicStackedPdfModel

    4. Existing non-stacked finite crystallite:
           PDFCalculator + finite shell table

    5. Conventional single-structure model:
           PDFCalculator
    """
    config = copy.deepcopy(
        dict(
            config
            or {}
        )
    )

    r_exp = np.asarray(
        r_exp,
        dtype=float,
    )

    pair_cutoff = float(
        pair_cutoff
    )

    if "structure_file" not in config:
        raise KeyError(
            "The model configuration has no structure_file."
        )

    from pdf_fitting.models.crystallite_shapes import (
        CrystalliteShapeSpec,
    )

    from pdf_fitting.models.gr_model import (
        PDFCalculator,
    )

    from pdf_fitting.models.layer_stacking import (
        stacking_spec_from_config,
    )

    from pdf_fitting.models.stochastic_ensemble_spec import (
        stochastic_ensemble_spec_from_config,
    )

    shape_config = dict(
        config.get(
            "crystallite_shape",
            {},
        )
        or {}
    )

    shape_mode = str(
        shape_config.get(
            "mode",
            "conventional",
        )
        or "conventional"
    ).strip().lower()

    shape_search_mode = str(
        shape_config.get(
            "search_mode",
            "predefined",
        )
        or "predefined"
    ).strip().lower()

    shape_spec = CrystalliteShapeSpec.from_dict(
        shape_config
    )

    stack_spec = stacking_spec_from_config(
        config
    )

    ensemble_spec = stochastic_ensemble_spec_from_config(
        config
    )

    stochastic_stacking = bool(
        stack_spec.enabled
        and stack_spec.model
        in {
            "correlated_ab_disorder",
            "generic_markov",
        }
    )

    # ---------------------------------------------------------
    # Stochastic stacking used as periodic bulk.
    #
    # Do not initialize a conventional PDFCalculator for the source layer.
    # The streaming model generates and opens one realization calculator at
    # a time.
    # ---------------------------------------------------------
    if (
        stochastic_stacking
        and ensemble_spec.use_stacking_as_bulk
    ):
        _status(
            status_callback,
            "Initializing streaming stochastic bulk model...",
        )

        from pymatgen.core import (
            Structure,
        )

        structure_file = str(
            config[
                "structure_file"
            ]
        )

        try:
            if structure_file.lower().endswith(
                ".cif"
            ):
                from pdf_fitting.gui.cif_utils import (
                    read_cif_asu_sites,
                )

                source_structure, _asu = read_cif_asu_sites(
                    structure_file
                )

                if source_structure is None:
                    source_structure = Structure.from_file(
                        structure_file
                    )

            else:
                source_structure = Structure.from_file(
                    structure_file
                )

        except Exception:
            source_structure = Structure.from_file(
                structure_file
            )

        from pdf_fitting.models.streaming_stacking_bulk_model import (
            StreamingStackingBulkModel,
        )

        return StreamingStackingBulkModel(
            config=config,
            source_structure=source_structure,
            stack_spec=stack_spec,
            ensemble_spec=ensemble_spec,
            pair_cutoff=pair_cutoff,
            r_grid=r_exp,
            reference_parameters=(
                config.get(
                    "initial",
                    {},
                )
                or {}
            ),
            stop_event=stop_event,
            status_callback=status_callback,
        )

    # ---------------------------------------------------------
    # Every remaining model starts with the ordinary PDFCalculator.
    # ---------------------------------------------------------
    pdf_config = copy.deepcopy(
        config
    )

    pdf_config[
        "refinement"
    ] = dict(
        config.get(
            "refinement",
            {},
        )
        or {}
    )

    if stack_spec.enabled:
        pdf_config[
            "refinement"
        ][
            "defer_cuda_pdf_backend"
        ] = True

    _status(
        status_callback,
        "Initializing base PDF calculator...",
    )

    pdf_model = PDFCalculator(
        pdf_config,
        pdf_config[
            "structure_file"
        ],
        r_exp,
        pair_cutoff=pair_cutoff,
    )

    source_structure = (
        pdf_model
        .structure_handler
        .structure
    )

    topology_margin = float(
        (
            config.get(
                "refinement",
                {},
            )
            or {}
        ).get(
            "stacked_topology_margin",
            3.0,
        )
    )

    # ---------------------------------------------------------
    # Existing finite stochastic stacking backend.
    # ---------------------------------------------------------
    if stochastic_stacking:
        if shape_mode != "finite_shape":
            raise ValueError(
                "The finite stochastic stacking backend requires "
                "[crystallite_shape] mode = finite_shape, or enable "
                "'Use stacking as bulk'."
            )

        if shape_search_mode == "grid_search":
            raise ValueError(
                "A finite stochastic grid search must be run through the "
                "finite-shape grid-search worker."
            )

        _status(
            status_callback,
            "Initializing existing finite stochastic stacking model...",
        )

        from pdf_fitting.models.stochastic_stacked_pdf_model import (
            StochasticStackedPdfModel,
        )

        return StochasticStackedPdfModel(
            pdf_calculator=pdf_model,
            source_structure=source_structure,
            shape_spec=shape_spec,
            stack_spec=stack_spec,
            pair_cutoff=pair_cutoff,
            reference_parameters=(
                config.get(
                    "initial",
                    {},
                )
                or {}
            ),
            topology_margin=topology_margin,
        )

    # ---------------------------------------------------------
    # Existing deterministic stacked backend.
    # ---------------------------------------------------------
    if stack_spec.enabled:
        if shape_mode != "finite_shape":
            raise ValueError(
                "The deterministic stacked backend requires "
                "[crystallite_shape] mode = finite_shape."
            )

        if shape_search_mode == "grid_search":
            raise ValueError(
                "A deterministic finite-shape grid search must be run "
                "through the finite-shape grid-search worker."
            )

        _status(
            status_callback,
            "Initializing deterministic dynamic stacked model...",
        )

        from pdf_fitting.models.dynamic_stacked_model import (
            DynamicStackedPdfModel,
        )

        return DynamicStackedPdfModel(
            pdf_calculator=pdf_model,
            source_structure=source_structure,
            shape_spec=shape_spec,
            stack_spec=stack_spec,
            pair_cutoff=pair_cutoff,
            reference_parameters=(
                config.get(
                    "initial",
                    {},
                )
                or {}
            ),
            topology_margin=topology_margin,
        )

    # ---------------------------------------------------------
    # Existing non-stacked finite crystallite.
    # ---------------------------------------------------------
    if shape_mode == "finite_shape":
        if shape_search_mode == "grid_search":
            raise ValueError(
                "A finite-shape grid search must be run through the "
                "finite-shape grid-search worker."
            )

        _status(
            status_callback,
            "Loading or building finite crystallite shell table...",
        )

        from pdf_fitting.models.finite_shape_shells import (
            get_or_build_finite_shell_result,
        )

        finite_result = get_or_build_finite_shell_result(
            source_structure,
            pair_cutoff=pair_cutoff,
            shape_spec=shape_spec,
            refinement_config=(
                config.get(
                    "refinement",
                    {},
                )
                or {}
            ),
            boundary_mode="atom",
        )

        pdf_model.replace_shell_table(
            finite_result.shell_table,
            device_shell_table=(
                finite_result.device_shell_table
            ),
        )

        return pdf_model

    # ---------------------------------------------------------
    # Conventional single-structure model.
    # ---------------------------------------------------------
    return pdf_model