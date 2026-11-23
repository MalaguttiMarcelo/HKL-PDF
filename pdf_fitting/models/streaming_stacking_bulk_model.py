from __future__ import annotations

import copy
import gc
import hashlib
import json
import os

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

import numpy as np

from pdf_fitting.models.bulk_stacking_realization import (
    build_bulk_stacking_realization,
    validate_bulk_stacking_realization,
)

from pdf_fitting.models.bulk_realization_shells import (
    open_bulk_realization_calculator,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
)

from pdf_fitting.models.stacking_disorder import (
    generate_correlated_ab_paths,
    generate_cyclic_correlated_ab_states,
    generate_cyclic_generic_markov_states,
    generate_generic_markov_paths,
)

from pdf_fitting.models.stochastic_ensemble_spec import (
    StochasticEnsembleSpec,
)


class StreamingStackingBulkModel:
    """
    Sequentially evaluate frozen stochastic stacking realizations as periodic
    bulk approximants.

    Important
    ---------
    - Markov paths are generated once and remain frozen.
    - All realizations share the same parent-layer lattice parameters.
    - All realizations share the same stacking-state vectors.
    - The realization periodic c vector is derived from the frozen path and
      shared stacking vectors.
    - Only one realization calculator is retained at a time.
    - Finite-particle gamma and crystallite-shape form factors are disabled.
    """

    def __init__(
        self,
        *,
        config: Mapping[
            str,
            Any,
        ],
        source_structure,
        stack_spec: LayerStackSpec,
        ensemble_spec: StochasticEnsembleSpec,
        pair_cutoff: float,
        r_grid: np.ndarray,
        reference_parameters: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        stop_event=None,
        status_callback=None,
    ):
        self.config = copy.deepcopy(
            dict(
                config
                or {}
            )
        )

        self.source_structure = (
            source_structure
        )

        self.stack_spec = stack_spec.normalized()

        self.ensemble_spec = ensemble_spec

        self.pair_cutoff = float(
            pair_cutoff
        )

        self.r = np.asarray(
            r_grid,
            dtype=float,
        )

        self.r_min = float(
            self.config.get(
                "r_min",
                (
                    self.r[0]
                    if self.r.size
                    else 0.0
                ),
            )
        )

        self.r_max = float(
            self.config.get(
                "r_max",
                (
                    self.r[-1]
                    if self.r.size
                    else 0.0
                ),
            )
        )

        self.reference_parameters = dict(
            reference_parameters
            or {}
        )

        self.stop_event = stop_event
        self.status_callback = status_callback

        self.constraints = {}

        self._last_parameters = dict(
            self.reference_parameters
        )

        self._last_result = None

        self._last_realization_calculator = None
        self._last_realization_key = None

        self._geometry_update_count = 0

        self.realization_directory = (
            self._resolve_realization_directory()
        )

        self.state_paths = (
            self._build_frozen_state_paths()
        )

        (
            self.unique_state_paths,
            self.realization_weights,
            self.realization_members,
        ) = self._deduplicate_state_paths(
            self.state_paths
        )

        self._status(
            "Bulk stacking ensemble prepared: "
            f"{self.state_paths.shape[0]} realization(s), "
            f"{self.unique_state_paths.shape[0]} unique path(s), "
            f"{self.state_paths.shape[1]} layers per path."
        )

    def _status(
        self,
        message: str,
    ) -> None:
        callback = self.status_callback

        if callback is None:
            print(
                f"[STREAMING STACKING BULK] {message}",
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
                f"[STREAMING STACKING BULK] {message}",
                flush=True,
            )

    def _check_stopped(
        self,
    ) -> None:
        if (
            self.stop_event is not None
            and self.stop_event.is_set()
        ):
            try:
                from pdf_fitting.fit_engine import (
                    FitStopped,
                )

                raise FitStopped(
                    "Bulk stacking evaluation stopped by user."
                )

            except ImportError:
                raise RuntimeError(
                    "Bulk stacking evaluation stopped by user."
                )

    def _resolve_realization_directory(
        self,
    ) -> str:
        configured_directory = str(
            self.ensemble_spec.realization_directory
            or "stacking_realizations"
        ).strip()

        if os.path.isabs(
            configured_directory
        ):
            output_directory = configured_directory

        else:
            source_path = str(
                self.config.get(
                    "structure_file",
                    "",
                )
                or ""
            ).strip()

            if source_path:
                base_directory = os.path.dirname(
                    os.path.abspath(
                        source_path
                    )
                )
            else:
                base_directory = os.getcwd()

            output_directory = os.path.join(
                base_directory,
                configured_directory,
            )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        return output_directory

    def _build_frozen_state_paths(
        self,
    ) -> np.ndarray:
        number_of_layers = max(
            2,
            int(
                self.ensemble_spec.realization_period_layers
            ),
        )

        ensemble_samples = max(
            1,
            int(
                self.stack_spec.ensemble_samples
            ),
        )

        require_cyclic = bool(
            self.ensemble_spec.require_cyclic_paths
        )

        regenerate_invalid = bool(
            self.ensemble_spec.regenerate_invalid_cycles
        )

        if self.stack_spec.model == "generic_markov":
            transition_matrix = np.asarray(
                self.stack_spec.markov_transition_matrix,
                dtype=float,
            )

            if require_cyclic:
                states = generate_cyclic_generic_markov_states(
                    number_of_layers=number_of_layers,
                    ensemble_samples=ensemble_samples,
                    transition_matrix=transition_matrix,
                    seed=int(
                        self.stack_spec.ensemble_seed
                    ),
                    start_state=int(
                        self.stack_spec.markov_start_state
                    ),
                    regenerate_invalid=regenerate_invalid,
                )

            else:
                states = generate_generic_markov_paths(
                    number_of_layers=number_of_layers,
                    ensemble_samples=ensemble_samples,
                    transition_matrix=transition_matrix,
                    seed=int(
                        self.stack_spec.ensemble_seed
                    ),
                    start_state=int(
                        self.stack_spec.markov_start_state
                    ),
                ).states

        else:
            if require_cyclic:
                states = generate_cyclic_correlated_ab_states(
                    number_of_layers=number_of_layers,
                    ensemble_samples=ensemble_samples,
                    p_alternate=float(
                        self.stack_spec.p_alternate
                    ),
                    seed=int(
                        self.stack_spec.ensemble_seed
                    ),
                    start_state=str(
                        self.stack_spec.ensemble_start_state
                    ),
                    regenerate_invalid=regenerate_invalid,
                )

            else:
                states = generate_correlated_ab_paths(
                    number_of_layers=number_of_layers,
                    ensemble_samples=ensemble_samples,
                    p_alternate=float(
                        self.stack_spec.p_alternate
                    ),
                    seed=int(
                        self.stack_spec.ensemble_seed
                    ),
                    start_state=str(
                        self.stack_spec.ensemble_start_state
                    ),
                ).states

        states = np.asarray(
            states,
            dtype=np.int16,
        )

        if states.ndim != 2:
            raise ValueError(
                "Frozen stochastic paths must have shape "
                "(number_of_realizations, number_of_layers)."
            )

        return states

    @staticmethod
    def _deduplicate_state_paths(
        state_paths: np.ndarray,
    ):
        state_paths = np.ascontiguousarray(
            state_paths,
            dtype=np.int16,
        )

        grouped = {}

        for realization_index in range(
            state_paths.shape[0]
        ):
            path = np.ascontiguousarray(
                state_paths[
                    realization_index
                ],
                dtype=np.int16,
            )

            key = path.tobytes()

            if key not in grouped:
                grouped[
                    key
                ] = {
                    "path": path.copy(),
                    "members": [],
                }

            grouped[
                key
            ][
                "members"
            ].append(
                int(
                    realization_index
                )
            )

        grouped_values = list(
            grouped.values()
        )

        unique_paths = np.asarray(
            [
                item[
                    "path"
                ]
                for item in grouped_values
            ],
            dtype=np.int16,
        )

        members = [
            list(
                item[
                    "members"
                ]
            )
            for item in grouped_values
        ]

        weights = np.asarray(
            [
                len(
                    item[
                        "members"
                    ]
                )
                for item in grouped_values
            ],
            dtype=np.float64,
        )

        weights /= max(
            float(
                np.sum(
                    weights
                )
            ),
            1.0e-12,
        )

        return (
            unique_paths,
            weights,
            members,
        )

    def _geometry_key(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
        state_path: np.ndarray,
    ) -> str:
        parameters = dict(
            parameters
            or {}
        )

        lattice = self.source_structure.lattice

        geometry_parameters = {
            "a": float(
                parameters.get(
                    "a",
                    lattice.a,
                )
            ),
            "b": float(
                parameters.get(
                    "b",
                    lattice.b,
                )
            ),
            "c": float(
                parameters.get(
                    "c",
                    lattice.c,
                )
            ),
            "alpha": float(
                parameters.get(
                    "alpha",
                    lattice.alpha,
                )
            ),
            "beta": float(
                parameters.get(
                    "beta",
                    lattice.beta,
                )
            ),
            "gamma": float(
                parameters.get(
                    "gamma",
                    lattice.gamma,
                )
            ),
        }

        if self.stack_spec.model == "generic_markov":
            for state_index, state in enumerate(
                self.stack_spec.markov_states
            ):
                uid = str(
                    state.uid
                )

                geometry_parameters[
                    f"state_{uid}_du"
                ] = float(
                    parameters.get(
                        f"stack_state_{uid}_du",
                        state.du,
                    )
                )

                geometry_parameters[
                    f"state_{uid}_dv"
                ] = float(
                    parameters.get(
                        f"stack_state_{uid}_dv",
                        state.dv,
                    )
                )

                geometry_parameters[
                    f"state_{uid}_h"
                ] = float(
                    parameters.get(
                        f"stack_state_{uid}_h",
                        state.h,
                    )
                )

        else:
            for name, default in (
                (
                    "stack_state_a_du",
                    self.stack_spec.state_a_du,
                ),
                (
                    "stack_state_a_dv",
                    self.stack_spec.state_a_dv,
                ),
                (
                    "stack_state_a_h",
                    self.stack_spec.state_a_h,
                ),
                (
                    "stack_state_b_du",
                    self.stack_spec.state_b_du,
                ),
                (
                    "stack_state_b_dv",
                    self.stack_spec.state_b_dv,
                ),
                (
                    "stack_state_b_h",
                    self.stack_spec.state_b_h,
                ),
            ):
                geometry_parameters[
                    name
                ] = float(
                    parameters.get(
                        name,
                        default,
                    )
                )

        payload = {
            "version": "streaming_stacking_bulk_v1",
            "geometry_parameters": geometry_parameters,
            "state_path": [
                int(
                    value
                )
                for value in np.asarray(
                    state_path,
                    dtype=np.int16,
                )
            ],
            "pair_cutoff": float(
                self.pair_cutoff
            ),
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        ).hexdigest()

    def _local_parameters(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Return parameters used by one periodic bulk realization.

        Parent-layer lattice and stacking-vector parameters have already
        been applied while constructing the realization Structure.

        The independent SAXS parameters remain active.
        """
        local_parameters = dict(
            parameters
            or {}
        )

        # The generated realization already contains the evaluated parent
        # lattice and stacking-vector geometry.
        for parameter_name in (
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "stacking_distance",
            "stacking_fraction",
        ):
            local_parameters.pop(
                parameter_name,
                None,
            )

        # Disable ordinary crystallite-size envelopes. Keep SAXS parameters.
        for parameter_name in (
            "d",
            "d_std",
            "d_mu",
            "cyl_diameter",
            "cyl_thickness",
            "cyl_thickness_std",
            "cyl_axis_h",
            "cyl_axis_k",
            "cyl_axis_l",
            "cyl_mu",
            "cyl_sigma",
        ):
            local_parameters.pop(
                parameter_name,
                None,
            )

        # Atomic positions and stacking vectors are already encoded in the
        # generated realization Structure.
        for parameter_name in list(
            local_parameters.keys()
        ):
            parameter_name_lower = str(
                parameter_name
            ).strip().lower()

            if parameter_name_lower.startswith(
                (
                    "x_",
                    "y_",
                    "z_",
                    "occ_",
                    "stack_du_",
                    "stack_dv_",
                    "stack_h_",
                    "stack_u_",
                    "stack_v_",
                    "stack_delta_",
                    "stack_state_",
                )
            ):
                local_parameters.pop(
                    parameter_name,
                    None,
                )

        # Average unscaled realization PDFs, then apply the physical phase
        # scale once after ensemble normalization.
        local_parameters[
            "scale"
        ] = 1.0

        # Qmax termination is linear and is applied once after averaging.
        local_parameters[
            "qmax"
        ] = 0.0

        return local_parameters

    
    def _release_last_calculator(
        self,
    ) -> None:
        self._last_realization_calculator = None
        self._last_realization_key = None

        gc.collect()

    def _open_realization_calculator(
        self,
        *,
        realization,
        geometry_key: str,
        parameters: Mapping[
            str,
            Any,
        ],
    ):
        if (
            self._last_realization_calculator is not None
            and self._last_realization_key == geometry_key
        ):
            return self._last_realization_calculator

        self._release_last_calculator()

        calculator = open_bulk_realization_calculator(
            base_config=self.config,
            realization=realization,
            realization_directory=(
                self.realization_directory
            ),
            geometry_key=geometry_key,
            pair_cutoff=self.pair_cutoff,
            r_grid=self.r,
            export_cif=bool(
                self.ensemble_spec.export_realization_cifs
            ),
        )

        self._last_realization_calculator = calculator
        self._last_realization_key = geometry_key

        return calculator

    def evaluate(
        self,
        parameters,
        **kwargs,
    ):
        """
        Stream periodic realizations and construct one ensemble-normalized PDF.

        The component g(r) functions are averaged first. One common ensemble
        number density is then used to construct G(r). This avoids averaging
        realization-specific negative baselines.
        """
        parameters = dict(
            parameters
            or {}
        )

        self._last_parameters = dict(
            parameters
        )

        number_of_realizations = int(
            self.unique_state_paths.shape[0]
        )

        if number_of_realizations <= 0:
            raise RuntimeError(
                "The stochastic bulk ensemble contains no realizations."
            )

        average_g = np.zeros(
            self.r.size,
            dtype=np.float64,
        )

        # Dimensionless SAXS gamma averaged across realizations.
        average_saxs_gamma = np.zeros(
            self.r.size,
            dtype=np.float64,
        )

        weighted_atom_count = 0.0
        weighted_volume = 0.0
        total_weight = 0.0

        realization_results = {}

        local_parameters = self._local_parameters(
            parameters
        )

        phase_scale = float(
            parameters.get(
                "scale",
                1.0,
            )
        )

        refinement_config = dict(
            self.config.get(
                "refinement",
                {},
            )
            or {}
        )

        saxs_enabled = bool(
            refinement_config.get(
                "saxs_enabled",
                False,
            )
        )

        saxs_scale = float(
            parameters.get(
                "saxs_scale",
                1.0,
            )
        )

        for realization_index in range(
            number_of_realizations
        ):
            self._check_stopped()

            state_path = self.unique_state_paths[
                realization_index
            ]

            weight = float(
                self.realization_weights[
                    realization_index
                ]
            )

            if weight <= 0.0:
                continue

            self._status(
                "Evaluating bulk stacking realization "
                f"{realization_index + 1}/{number_of_realizations}."
            )

            realization = build_bulk_stacking_realization(
                source_structure=self.source_structure,
                stack_spec=self.stack_spec,
                state_path=state_path,
                realization_index=realization_index,
                parameters=parameters,
                weight=weight,
            )

            validate_bulk_stacking_realization(
                realization,
                minimum_distance=0.10,
            )

            geometry_key = self._geometry_key(
                parameters,
                state_path,
            )

            calculator = self._open_realization_calculator(
                realization=realization,
                geometry_key=geometry_key,
                parameters=parameters,
            )

            component_result = calculator.evaluate(
                local_parameters,
                return_contributions=False,
                compute_gamma_avg=False,
            )

            component_r = np.asarray(
                component_result[
                    "r"
                ],
                dtype=float,
            )

            component_g = np.asarray(
                component_result[
                    "g_r"
                ],
                dtype=np.float64,
            )

            if (
                component_r.shape
                != self.r.shape
                or not np.allclose(
                    component_r,
                    self.r,
                    rtol=0.0,
                    atol=1.0e-10,
                )
            ):
                component_g = np.interp(
                    self.r,
                    component_r,
                    component_g,
                )

            if (
                component_g.size != self.r.size
                or not np.all(
                    np.isfinite(
                        component_g
                    )
                )
            ):
                raise ValueError(
                    "A bulk stacking realization returned invalid g(r): "
                    f"realization={realization_index}."
                )

            average_g += (
                weight
                * component_g
            )

            atom_count = int(
                len(
                    realization.structure.sites
                )
            )

            volume = abs(
                float(
                    realization.structure.lattice.volume
                )
            )

            if volume <= 1.0e-12:
                raise ValueError(
                    "A bulk stacking realization has invalid volume: "
                    f"realization={realization_index}, "
                    f"volume={volume}."
                )

            weighted_atom_count += (
                weight
                * float(
                    atom_count
                )
            )

            weighted_volume += (
                weight
                * volume
            )

            # -----------------------------------------------------
            # SAXS gamma
            #
            # compute_isotropic_shape_factor() returns:
            #
            #   -scale * saxs_scale * 4*pi*rho_j*r*gamma_j(r)
            #
            # With local scale=1, divide by saxs_scale*4*pi*rho_j*r
            # to recover dimensionless gamma_j(r), then average gamma_j.
            # -----------------------------------------------------
            if saxs_enabled:
                shape_parameters = dict(
                    local_parameters
                )

                shape_parameters[
                    "scale"
                ] = 1.0

                shape_parameters[
                    "qmax"
                ] = 0.0

                shape_parameters[
                    "qdamp"
                ] = 0.0

                shape_parameters[
                    "saxs_apply_qdamp"
                ] = False

                shape_term = np.asarray(
                    calculator.compute_isotropic_shape_factor(
                        shape_parameters,
                        self.r,
                    ),
                    dtype=np.float64,
                )

                realization_density = (
                    float(
                        atom_count
                    )
                    / volume
                )

                denominator = (
                    max(
                        abs(
                            saxs_scale
                        ),
                        1.0e-30,
                    )
                    * 4.0
                    * np.pi
                    * realization_density
                    * np.maximum(
                        self.r,
                        1.0e-12,
                    )
                )

                realization_gamma = (
                    -shape_term
                    / denominator
                )

                if self.r.size > 0:
                    realization_gamma[
                        np.abs(
                            self.r
                        )
                        <= 1.0e-12
                    ] = 1.0

                realization_gamma = np.clip(
                    realization_gamma,
                    0.0,
                    1.0,
                )

                average_saxs_gamma += (
                    weight
                    * realization_gamma
                )

            total_weight += weight

            realization_results[
                str(
                    realization_index
                )
            ] = {
                "weight": weight,
                "members": list(
                    self.realization_members[
                        realization_index
                    ]
                ),
                "atom_count": atom_count,
                "volume": volume,
                "density": (
                    float(
                        atom_count
                    )
                    / volume
                ),
            }

            self._release_last_calculator()

        if total_weight <= 0.0:
            raise RuntimeError(
                "The bulk stacking ensemble has no positive realization "
                "weight."
            )

        average_g /= total_weight

        if saxs_enabled:
            average_saxs_gamma /= total_weight

        if weighted_volume <= 1.0e-12:
            raise RuntimeError(
                "The stochastic bulk ensemble has invalid total volume."
            )

        ensemble_density = (
            weighted_atom_count
            / weighted_volume
        )

        norm = (
            4.0
            * np.pi
            * ensemble_density
        )

        r_values = np.asarray(
            self.r,
            dtype=np.float64,
        )

        qdamp = float(
            parameters.get(
                "qdamp",
                0.0,
            )
        )

        if qdamp > 0.0:
            damping = np.exp(
                -0.5
                * qdamp
                * qdamp
                * r_values
                * r_values
            )
        else:
            damping = np.ones_like(
                r_values
            )

        if saxs_enabled:
            # Structural positive peak term.
            peak_term = (
                phase_scale
                * norm
                * r_values
                * average_g
            )

            peak_term *= damping

            # Independent SAXS / missing-low-Q belly.
            belly_term = (
                -phase_scale
                * saxs_scale
                * norm
                * r_values
                * average_saxs_gamma
            )

            saxs_apply_qdamp = bool(
                refinement_config.get(
                    "saxs_apply_qdamp",
                    False,
                )
            )

            if saxs_apply_qdamp:
                belly_term *= damping

            total_g = (
                peak_term
                + belly_term
            )

        else:
            # Conventional bulk PDF:
            #
            #   G(r) = scale * 4*pi*rho*r*(g(r) - 1)
            #
            total_g = (
                phase_scale
                * norm
                * r_values
                * (
                    average_g
                    - 1.0
                )
            )

            total_g *= damping

        # Apply qmax once to the complete averaged ensemble.
        qmax = float(
            parameters.get(
                "qmax",
                self.config.get(
                    "qmax",
                    0.0,
                ),
            )
        )

        if (
            qmax > 0.0
            and r_values.size > 1
        ):
            from pdf_fitting.models.gr_model import (
                apply_qmax_termination,
            )

            qmax_zeros = int(
                parameters.get(
                    "qmax_zeros",
                    self.config.get(
                        "qmax_zeros",
                        5,
                    ),
                )
            )

            qmax_pad = str(
                parameters.get(
                    "qmax_pad",
                    self.config.get(
                        "qmax_pad",
                        "zero",
                    ),
                )
            ).strip().lower()

            total_g = apply_qmax_termination(
                total_g,
                r_values,
                qmax,
                n_zeros=qmax_zeros,
                pad_mode=qmax_pad,
                use_sine_kernel=True,
                normalize=False,
            )

        if not np.all(
            np.isfinite(
                total_g
            )
        ):
            raise ValueError(
                "The ensemble-normalized bulk PDF contains non-finite values."
            )

        maximum_absolute_g = float(
            np.max(
                np.abs(
                    total_g
                )
            )
        )

        if maximum_absolute_g > 1.0e8:
            raise ValueError(
                "The ensemble-normalized bulk PDF has an unphysical "
                f"amplitude: max_abs_G={maximum_absolute_g:.8g}."
            )

        output = {
            "r": r_values.copy(),
            "G_r": np.asarray(
                total_g,
                dtype=np.float64,
            ),
            "g_r": np.asarray(
                average_g,
                dtype=np.float64,
            ),
            "g_ab": {},
            "G_ab": {},
            "G_contrib": {},
            "weights": {},
            "gamma_avg": None,
            "ensemble_density": float(
                ensemble_density
            ),
            "realizations": realization_results,
        }

        self._last_result = output
        self._geometry_update_count += 1

        return output

    
    @property
    def geometry_update_count(
        self,
    ) -> int:
        return int(
            self._geometry_update_count
        )

    @property
    def current_stacking_distance(
        self,
    ) -> float:
        try:
            from pdf_fitting.models.compiled_stochastic_geometry import (
                _generic_state_vectors,
            )

            _du, _dv, state_h, _labels = (
                _generic_state_vectors(
                    self.stack_spec,
                    self._last_parameters,
                )
            )

            return float(
                np.mean(
                    np.asarray(
                        state_h,
                        dtype=float,
                    )
                )
            )

        except Exception:
            return 0.0

    @property
    def current_stack_thickness(
        self,
    ) -> float:
        return float(
            self.current_stacking_distance
            * max(
                1,
                int(
                    self.ensemble_spec
                    .realization_period_layers
                ),
            )
        )

    def compute_isotropic_shape_factor(
        self,
        parameters,
        r_grid,
    ):
        """
        Bulk stacking has no finite crystallite shape term.
        """
        return np.zeros_like(
            np.asarray(
                r_grid,
                dtype=float,
            )
        )

    def build_shape_factor_directions(
        self,
        parameters,
        **kwargs,
    ):
        return {
            "mode": "bulk_stacking",
            "isotropic": True,
            "directions": {},
            "metadata": {
                "message": (
                    "Crystallite-shape factors are disabled in "
                    "stacking-as-bulk mode."
                ),
            },
        }

    def _diagnostic_from_first_realization(
        self,
        method_name: str,
        parameters,
        **kwargs,
    ):
        """
        Diagnostics are evaluated from the first representative realization.

        The total PDF remains the weighted ensemble average.
        """
        if self.unique_state_paths.shape[0] <= 0:
            return {}

        state_path = self.unique_state_paths[
            0
        ]

        realization = build_bulk_stacking_realization(
            source_structure=self.source_structure,
            stack_spec=self.stack_spec,
            state_path=state_path,
            realization_index=0,
            parameters=parameters,
            weight=float(
                self.realization_weights[
                    0
                ]
            ),
        )

        geometry_key = self._geometry_key(
            parameters,
            state_path,
        )

        calculator = self._open_realization_calculator(
            realization=realization,
            geometry_key=geometry_key,
            parameters=parameters,
        )

        local_parameters = self._local_parameters(
            parameters
        )

        try:
            method = getattr(
                calculator,
                method_name,
            )

            return method(
                local_parameters,
                **kwargs,
            )

        finally:
            self._release_last_calculator()

    def build_warren_plot(
        self,
        parameters,
        **kwargs,
    ):
        return self._diagnostic_from_first_realization(
            "build_warren_plot",
            parameters,
            **kwargs,
        )

    def build_local_trends(
        self,
        parameters,
        **kwargs,
    ):
        return self._diagnostic_from_first_realization(
            "build_local_trends",
            parameters,
            **kwargs,
        )

    def build_pdf_tick_data(
        self,
        parameters,
        **kwargs,
    ):
        return self._diagnostic_from_first_realization(
            "build_pdf_tick_data",
            parameters,
            **kwargs,
        )

    def _plot_distance_limit_from_params(
        self,
        parameters,
        fallback: float = 5.0,
    ) -> float:
        return float(
            fallback
        )