from __future__ import annotations

import configparser
import copy
import json
import os
import tempfile
import time
import traceback

from typing import Any, Dict, List, Optional

import numpy as np

from PySide6 import QtCore

from pdf_fitting.fit_engine import FitEngine, FitStopped
from pdf_fitting.io_handler import read_input_file
from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    iter_shape_scan,
)
from pdf_fitting.models.dynamic_stacked_model import (
    DynamicStackedPdfModel,
)
from pdf_fitting.models.stochastic_stacked_pdf_model import (
    StochasticStackedPdfModel,
)

from pdf_fitting.models.finite_shape_shells import (
    export_gamma_functions_by_direction,
    get_or_build_finite_shell_result,
)
from pdf_fitting.models.gr_model import PDFCalculator
from pdf_fitting.models.layer_stacking import (
    stacking_spec_from_config,
)
from pdf_fitting.models.stacking_search import (
    StackingSearchSpec,
    apply_lateral_candidate,
    stacking_search_spec_from_config,
)

from pdf_fitting.models.stacking_disorder import (
    generic_markov_matrix_grid,
    p_alternate_grid,
)


def _cfg_bool(
    value,
    default: bool = False,
) -> bool:
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(int(value))

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            return True

        if text in {
            "0",
            "false",
            "no",
            "n",
            "off",
        }:
            return False

    return bool(default)


class CSGridSearchWorker(QtCore.QThread):
    progress = QtCore.Signal(dict)
    finished_ok = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(
        self,
        *,
        input_path: str,
        source_path: Optional[str] = None,
        progress_every_sec: float = 0.5,
        update_pdf: bool = True,
        ask_after_each_shape: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.input_path = str(input_path)
        self.source_path = source_path
        self.progress_every_sec = float(
            progress_every_sec
        )
        self.update_pdf = bool(update_pdf)
        self.ask_after_each_shape = bool(
            ask_after_each_shape
        )

        import threading

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self._best = None
        self._best_visual = None
        self._save_on_stop = False

    @property
    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def request_stop(
        self,
        *,
        save_best: bool = False,
    ) -> None:
        self._save_on_stop = bool(save_best)
        self.stop_event.set()

        if save_best:
            self.status.emit(
                "Finite-shape stop-and-save requested..."
            )
        else:
            self.status.emit(
                "Finite-shape stop requested..."
            )

    def pause(self) -> None:
        self.pause_event.set()
        self.status.emit(
            "Pausing finite-shape search..."
        )

    def resume(self) -> None:
        self.pause_event.clear()
        self.status.emit(
            "Resuming finite-shape search..."
        )

    def _wait_if_paused(self) -> bool:
        while self.pause_event.is_set():
            if self.stop_event.is_set():
                return False

            time.sleep(0.05)

        return not self.stop_event.is_set()

    def _prepare_config_for_finite_shape(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        output = copy.deepcopy(config)

        output.setdefault(
            "initial",
            {},
        )
        output.setdefault(
            "refinable",
            {},
        )
        output.setdefault(
            "bounds",
            {},
        )

        for parameter_name in (
            "d",
            "d_std",
            "d_mu",
        ):
            output["initial"].pop(
                parameter_name,
                None,
            )
            output["refinable"].pop(
                parameter_name,
                None,
            )
            output["bounds"].pop(
                parameter_name,
                None,
            )

        return output

    @staticmethod
    def _compute_rwp(
        experimental: np.ndarray,
        calculated: np.ndarray,
    ) -> float:
        experimental = np.asarray(
            experimental,
            dtype=float,
        )

        calculated = np.asarray(
            calculated,
            dtype=float,
        )

        residual = experimental - calculated

        denominator = float(
            np.sum(
                experimental
                * experimental
            )
        )

        numerator = float(
            np.sum(
                residual
                * residual
            )
        )

        if denominator <= 0.0:
            return float("nan")

        return 100.0 * float(
            np.sqrt(
                numerator
                / denominator
            )
        )

    def _candidate_detail_root(
        self,
        search_spec: StackingSearchSpec,
    ) -> str:
        if self.source_path:
            base_directory = os.path.dirname(
                os.path.abspath(
                    self.source_path
                )
            )
        else:
            base_directory = os.path.dirname(
                os.path.abspath(
                    self.input_path
                )
            )

        output_directory = str(
            search_spec.detail_directory
            or "stacking_search_results"
        ).strip()

        if not os.path.isabs(
            output_directory
        ):
            output_directory = os.path.join(
                base_directory,
                output_directory,
            )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        return output_directory

    @staticmethod
    def _set_parser_value(
        parser,
        section: str,
        key: str,
        value,
    ) -> None:
        if not parser.has_section(section):
            parser.add_section(section)

        if isinstance(value, bool):
            text = (
                "true"
                if value
                else "false"
            )
        elif (
            isinstance(value, (tuple, list))
            and len(value) == 2
        ):
            lower = (
                ""
                if value[0] is None
                else str(value[0])
            )
            upper = (
                ""
                if value[1] is None
                else str(value[1])
            )
            text = f"{lower}, {upper}"
        else:
            text = str(value)

        parser.set(
            section,
            str(key),
            text,
        )

    def _write_candidate_input(
        self,
        *,
        destination: str,
        candidate_config: Dict[str, Any],
        candidate_shape_spec,
        candidate_stack_spec,
        candidate_search_spec,
        refined_parameters: Dict[str, Any],
    ) -> None:
        parser = configparser.RawConfigParser()
        parser.optionxform = str

        parser.read(
            self.input_path,
            encoding="utf-8-sig",
        )

        for section in (
            "initial_values",
            "refinable_parameters",
            "bounds",
            "crystallite_shape",
            "layer_stacking",
            "stacking_search",
        ):
            if not parser.has_section(section):
                parser.add_section(section)

        parameter_values = dict(
            candidate_config.get(
                "initial",
                {},
            )
            or {}
        )

        parameter_values.update(
            refined_parameters
            or {}
        )

        for key, value in parameter_values.items():
            self._set_parser_value(
                parser,
                "initial_values",
                str(key),
                value,
            )

        for key, value in (
            candidate_config.get(
                "refinable",
                {},
            )
            or {}
        ).items():
            self._set_parser_value(
                parser,
                "refinable_parameters",
                str(key),
                bool(value),
            )

        for key, value in (
            candidate_config.get(
                "bounds",
                {},
            )
            or {}
        ).items():
            self._set_parser_value(
                parser,
                "bounds",
                str(key),
                value,
            )

        shape_dictionary = (
            candidate_shape_spec.to_dict()
        )

        shape_dictionary["mode"] = (
            "finite_shape"
        )

        shape_dictionary["search_mode"] = (
            "predefined"
        )

        shape_dictionary[
            "scan_diameter"
        ] = False

        shape_dictionary[
            "scan_height"
        ] = False

        for key, value in shape_dictionary.items():
            self._set_parser_value(
                parser,
                "crystallite_shape",
                str(key),
                value,
            )

        stack_dictionary = (
            candidate_stack_spec.to_dict()
        )

        sequence = stack_dictionary.pop(
            "sequence",
            [],
        )

        for key, value in stack_dictionary.items():
            key_text = str(
                key
            )

            if (
                key_text
                in {
                    "markov_states",
                    "markov_transition_matrix",
                    "markov_probability_controls",
                }
                and isinstance(
                    value,
                    (
                        list,
                        tuple,
                    ),
                )
            ):
                value_to_write = json.dumps(
                    value,
                    separators=(
                        ",",
                        ":",
                    ),
                )

            else:
                value_to_write = value

            self._set_parser_value(
                parser,
                "layer_stacking",
                key_text,
                value_to_write,
            )

        self._set_parser_value(
            parser,
            "layer_stacking",
            "sequence_json",
            json.dumps(
                sequence,
                separators=(",", ":"),
            ),
        )

        stored_search = StackingSearchSpec.from_dict(
            candidate_search_spec.to_dict()
        )

        stored_search.spacing_treatment = "fixed"
        stored_search.lateral_treatment = "fixed"

        for key, value in stored_search.to_dict().items():
            self._set_parser_value(
                parser,
                "stacking_search",
                str(key),
                value,
            )

        with open(
            destination,
            "w",
            encoding="utf-8",
        ) as output_file:
            parser.write(
                output_file
            )

    def _write_candidate_detail(
        self,
        *,
        root_directory: str,
        cs_index: int,
        candidate_index: int,
        row: Dict[str, Any],
        r: np.ndarray,
        experimental: np.ndarray,
        calculated: np.ndarray,
        residual: np.ndarray,
        candidate_config: Dict[str, Any],
        candidate_shape_spec,
        candidate_stack_spec,
        candidate_search_spec,
        refined_parameters: Dict[str, Any],
        best_per_cs: bool,
    ) -> Dict[str, str]:
        cs_directory = os.path.join(
            root_directory,
            f"cs_{int(cs_index):04d}",
        )

        os.makedirs(
            cs_directory,
            exist_ok=True,
        )

        if best_per_cs:
            stem = (
                f"best_candidate_"
                f"{int(candidate_index):06d}"
            )
        else:
            stem = (
                f"candidate_"
                f"{int(candidate_index):06d}"
            )

        detail_path = os.path.join(
            cs_directory,
            stem + ".npz",
        )

        input_path = os.path.join(
            cs_directory,
            stem + ".inp",
        )

        metadata_path = os.path.join(
            cs_directory,
            stem + ".json",
        )

        np.savez(
            detail_path,
            r=np.asarray(
                r,
                dtype=float,
            ),
            G_exp=np.asarray(
                experimental,
                dtype=float,
            ),
            G_fit=np.asarray(
                calculated,
                dtype=float,
            ),
            residual=np.asarray(
                residual,
                dtype=float,
            ),
        )

        self._write_candidate_input(
            destination=input_path,
            candidate_config=candidate_config,
            candidate_shape_spec=(
                candidate_shape_spec
            ),
            candidate_stack_spec=(
                candidate_stack_spec
            ),
            candidate_search_spec=(
                candidate_search_spec
            ),
            refined_parameters=(
                refined_parameters
            ),
        )

        serializable_row = {}

        for key, value in row.items():
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                serializable_row[
                    key
                ] = value

            elif isinstance(
                value,
                np.ndarray,
            ):
                serializable_row[
                    key
                ] = value.tolist()

            elif isinstance(
                value,
                (
                    list,
                    tuple,
                    dict,
                ),
            ):
                try:
                    json.dumps(
                        value
                    )

                    serializable_row[
                        key
                    ] = value

                except Exception:
                    pass
        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                serializable_row,
                output_file,
                indent=2,
            )

        return {
            "detail_file": detail_path,
            "candidate_input_file": input_path,
            "metadata_file": metadata_path,
        }

    def run(self) -> None:
        start_time = time.perf_counter()

        try:
            self.status.emit(
                "Loading finite-shape search configuration..."
            )

            base_config = read_input_file(
                self.input_path
            )

            shape_config = (
                base_config.get(
                    "crystallite_shape",
                    {},
                )
                or {}
            )

            if (
                str(
                    shape_config.get(
                        "mode",
                        "conventional",
                    )
                ).strip().lower()
                != "finite_shape"
            ):
                raise RuntimeError(
                    "Finite-shape search requires "
                    "[crystallite_shape] mode = finite_shape."
                )

            shape_specification = (
                CrystalliteShapeSpec.from_dict(
                    shape_config
                )
            )

            stack_specification = (
                stacking_spec_from_config(
                    base_config
                )
            )

            from pdf_fitting.models.stochastic_ensemble_spec import (
                stochastic_ensemble_spec_from_config,
            )

            stochastic_ensemble_spec = (
                stochastic_ensemble_spec_from_config(
                    base_config
                )
            )

            search_specification = (
                stacking_search_spec_from_config(
                    base_config
                )
            )

            shape_candidates = list(
                iter_shape_scan(
                    shape_specification
                )
            )

            if (
                stochastic_ensemble_spec.use_stacking_as_bulk
            ):
                shape_candidates = [
                    shape_specification
                ]
                
            if not shape_candidates:
                raise RuntimeError(
                    "No crystallite-shape candidates were generated."
                )

            if stack_specification.enabled:
                for candidate in shape_candidates:
                    candidate.axis3_cells = float(
                        max(
                            1,
                            int(
                                round(
                                    candidate.axis3_cells
                                )
                            ),
                        )
                    )

                    candidate.height_cells = float(
                        candidate.axis3_cells
                    )

            for candidate in shape_candidates:
                if not np.isfinite(
                    float(
                        candidate.diameter_cells
                    )
                ):
                    raise RuntimeError(
                        "Infinite-diameter slab mode is not implemented."
                    )

            if (
                not stack_specification.enabled
                and (
                    search_specification.spacing_treatment
                    != "fixed"
                    or search_specification.lateral_treatment
                    != "fixed"
                )
            ):
                raise RuntimeError(
                    "A stacking search was requested, but periodic "
                    "layer stacking is disabled."
                )

            if (
                stack_specification.spacing_mode
                == "fraction_of_c_normal"
                and search_specification.spacing_treatment
                == "grid"
            ):
                raise RuntimeError(
                    "The current spacing grid is expressed in Angstrom. "
                    "Use independent spacing mode for a spacing grid."
                )

            if stack_specification.enabled:
                spacing_values = (
                    search_specification.spacing_values(
                        stack_specification.nominal_spacing
                    )
                )

                lateral_values = (
                    search_specification.lateral_values(
                        stack_specification.repeat_shift_u,
                        stack_specification.repeat_shift_v,
                    )
                )
            else:
                spacing_values = np.asarray(
                    [
                        float(
                            stack_specification.nominal_spacing
                        )
                    ],
                    dtype=float,
                )

                lateral_values = [
                    {
                        "u": float(
                            stack_specification.repeat_shift_u
                        ),
                        "v": float(
                            stack_specification.repeat_shift_v
                        ),
                        "sample_index": 0,
                    }
                ]

            # -----------------------------------------------------
            # Outer stochastic grids
            #
            # correlated_ab_disorder:
            #     scan P(alternate), if enabled.
            #
            # generic_markov:
            #     scan transition-probability matrices generated from
            #     markov_probability_controls.
            #
            # Transition probabilities are discrete outer-grid variables.
            # They are not normal least-squares parameters because changing
            # them changes the frozen stochastic paths/topology.
            # -----------------------------------------------------

            # -----------------------------------------------------
            # P(alternate) candidates
            # -----------------------------------------------------
            if (
                stack_specification.enabled
                and stack_specification.model
                == "correlated_ab_disorder"
                and stack_specification.p_grid_enabled
            ):
                p_alt_values = p_alternate_grid(
                    minimum=float(
                        stack_specification.p_grid_minimum
                    ),
                    maximum=float(
                        stack_specification.p_grid_maximum
                    ),
                    step=float(
                        stack_specification.p_grid_step
                    ),
                )

            else:
                # P_alt is not used by generic_markov, but retaining one
                # placeholder keeps the common candidate loop simple.
                p_alt_values = np.asarray(
                    [
                        float(
                            stack_specification.p_alternate
                        )
                    ],
                    dtype=float,
                )

            # -----------------------------------------------------
            # Generic Markov transition-matrix candidates
            # -----------------------------------------------------
            if (
                stack_specification.enabled
                and stack_specification.model
                == "generic_markov"
            ):
                markov_matrix_candidates = (
                    generic_markov_matrix_grid(
                        transition_matrix=np.asarray(
                            stack_specification
                            .markov_transition_matrix,
                            dtype=float,
                        ),
                        probability_controls=(
                            stack_specification
                            .markov_probability_controls
                        ),
                        maximum_candidates=int(
                            (
                                base_config.get(
                                    "refinement",
                                    {},
                                )
                                or {}
                            ).get(
                                "maximum_markov_grid_candidates",
                                100000,
                            )
                        ),
                    )
                )

                if not markov_matrix_candidates:
                    raise RuntimeError(
                        "The generic Markov probability grid produced "
                        "no valid transition matrices. Every transition-"
                        "matrix row must sum to 1."
                    )

            else:
                # Deterministic and correlated-A/B models do not use the
                # generic transition-matrix grid.
                markov_matrix_candidates = [
                    None
                ]

            # Preserve the original lateral-vector count for reporting.
            lateral_vector_count = len(
                lateral_values
            )

            # -----------------------------------------------------
            # Combine lateral, P_alt and Markov-matrix candidates
            # into one flat outer-candidate list.
            # -----------------------------------------------------
            expanded_lateral_values = []

            for p_alt_index, p_alt_value in enumerate(
                p_alt_values
            ):
                for (
                    markov_matrix_index,
                    markov_matrix,
                ) in enumerate(
                    markov_matrix_candidates
                ):
                    for lateral_candidate in lateral_values:
                        expanded_candidate = dict(
                            lateral_candidate
                        )

                        expanded_candidate[
                            "p_alternate"
                        ] = float(
                            p_alt_value
                        )

                        expanded_candidate[
                            "p_alt_index"
                        ] = int(
                            p_alt_index
                        )

                        expanded_candidate[
                            "markov_matrix_index"
                        ] = int(
                            markov_matrix_index
                        )

                        if markov_matrix is None:
                            expanded_candidate[
                                "markov_transition_matrix"
                            ] = None

                        else:
                            expanded_candidate[
                                "markov_transition_matrix"
                            ] = np.asarray(
                                markov_matrix,
                                dtype=float,
                            ).tolist()

                        expanded_lateral_values.append(
                            expanded_candidate
                        )

            lateral_values = expanded_lateral_values

            total_candidate_count = (
                len(
                    shape_candidates
                )
                * int(
                    spacing_values.size
                )
                * len(
                    lateral_values
                )
            )

            self.status.emit(
                "Search candidates: "
                f"{len(shape_candidates)} crystallite sizes × "
                f"{int(spacing_values.size)} spacings × "
                f"{lateral_vector_count} lateral vectors × "
                f"{int(p_alt_values.size)} P(alternate) values × "
                f"{len(markov_matrix_candidates)} Markov matrices = "
                f"{total_candidate_count} refinements."
            )

        
            experimental_r, experimental_g = np.loadtxt(
                base_config["gr_data_file"],
                unpack=True,
            )

            r_minimum = float(
                base_config.get(
                    "r_min",
                    1.0,
                )
            )

            r_maximum = float(
                base_config.get(
                    "r_max",
                    np.max(experimental_r),
                )
            )

            pair_extension = float(
                base_config.get(
                    "r_extension",
                    1.2,
                )
            )

            pair_cutoff = (
                r_maximum
                * pair_extension
            )

            use_points = (
                (experimental_r >= r_minimum)
                & (experimental_r <= r_maximum)
            )

            experimental_r = np.asarray(
                experimental_r[use_points],
                dtype=float,
            )

            experimental_g = np.asarray(
                experimental_g[use_points],
                dtype=float,
            )

            finite_config = (
                self._prepare_config_for_finite_shape(
                    base_config
                )
            )

            base_pdf_config = copy.deepcopy(
                finite_config
            )

            if stack_specification.enabled:
                base_pdf_config.setdefault(
                    "refinement",
                    {},
                )

                base_pdf_config["refinement"][
                    "defer_cuda_pdf_backend"
                ] = True

            self.status.emit(
                "Initializing base PDF model..."
            )

            pdf_model = PDFCalculator(
                base_pdf_config,
                base_pdf_config["structure_file"],
                experimental_r,
                pair_cutoff=pair_cutoff,
            )

            structure = (
                pdf_model
                .structure_handler
                .structure
            )

            results: List[Dict[str, Any]] = []
            best_by_cs: Dict[int, Dict[str, Any]] = {}

            detail_root = None

            if (
                search_specification.detail_storage
                != "none"
            ):
                detail_root = (
                    self._candidate_detail_root(
                        search_specification
                    )
                )

            completed_index = 0

            stop_requested = False

            for cs_index, shape_candidate in enumerate(
                shape_candidates
            ):
                if stop_requested:
                    break

                for spacing_index, spacing_value in enumerate(
                    spacing_values
                ):
                    if stop_requested:
                        break

                    for lateral_index, lateral_candidate in enumerate(
                        lateral_values
                    ):
                        if self.stop_event.is_set():
                            stop_requested = True
                            break

                        if not self._wait_if_paused():
                            stop_requested = True
                            break

                        candidate_index = int(
                            completed_index
                        )

                        completed_index += 1

                        candidate_stack_spec = (
                            apply_lateral_candidate(
                                stack_specification,
                                search_specification,
                                u=float(
                                    lateral_candidate["u"]
                                ),
                                v=float(
                                    lateral_candidate["v"]
                                ),
                            )
                        )

                        # P_alt is fixed for this outer stochastic-grid
                        # candidate. Calling normalized() protects the
                        # 0 <= P_alt <= 1 range.
                        candidate_stack_spec.p_alternate = float(
                            lateral_candidate.get(
                                "p_alternate",
                                stack_specification.p_alternate,
                            )
                        )

                        # For generic Markov stacking, install the selected
                        # transition-matrix candidate before normalization and
                        # before frozen paths/topology are generated.
                        if (
                            candidate_stack_spec.model
                            == "generic_markov"
                        ):
                            selected_markov_matrix = (
                                lateral_candidate.get(
                                    "markov_transition_matrix",
                                    None,
                                )
                            )

                            if selected_markov_matrix is None:
                                raise RuntimeError(
                                    "A generic Markov candidate has no "
                                    "transition matrix."
                                )

                            candidate_stack_spec.markov_transition_matrix = [
                                [
                                    float(
                                        probability
                                    )
                                    for probability in row
                                ]
                                for row in selected_markov_matrix
                            ]

                        candidate_stack_spec = (
                            candidate_stack_spec.normalized()
                        )

                        if (
                            candidate_stack_spec.spacing_mode
                            == "independent"
                        ):
                            candidate_stack_spec.nominal_spacing = float(
                                spacing_value
                            )

                        candidate_config = copy.deepcopy(
                            finite_config
                        )

                        candidate_config.setdefault(
                            "refinement",
                            {},
                        )
                        candidate_config.setdefault(
                            "initial",
                            {},
                        )
                        candidate_config.setdefault(
                            "refinable",
                            {},
                        )
                        candidate_config.setdefault(
                            "bounds",
                            {},
                        )

                        candidate_config["refinement"][
                            "progress_every_sec"
                        ] = self.progress_every_sec

                        candidate_config["refinement"][
                            "progress_every_nfev"
                        ] = 1

                        candidate_config["refinement"][
                            "update_pdf_during_refinement"
                        ] = bool(
                            self.update_pdf
                        )

                        candidate_config[
                            "layer_stacking"
                        ] = candidate_stack_spec.to_dict()

                        if (
                            candidate_stack_spec.spacing_mode
                            == "independent"
                        ):
                            candidate_config["initial"][
                                "stacking_distance"
                            ] = float(
                                spacing_value
                            )

                            if (
                                search_specification.spacing_treatment
                                == "continuous"
                            ):
                                candidate_config["refinable"][
                                    "stacking_distance"
                                ] = True

                                candidate_config["bounds"][
                                    "stacking_distance"
                                ] = (
                                    float(
                                        search_specification.spacing_minimum
                                    ),
                                    float(
                                        search_specification.spacing_maximum
                                    ),
                                )

                            else:
                                candidate_config["refinable"][
                                    "stacking_distance"
                                ] = False

                        if (
                            candidate_stack_spec.model
                            == "generic_markov"
                        ):
                            stochastic_status = (
                                "Markov matrix "
                                f"{int(lateral_candidate.get('markov_matrix_index', 0)) + 1}/"
                                f"{len(markov_matrix_candidates)}"
                            )

                        elif (
                            candidate_stack_spec.model
                            == "correlated_ab_disorder"
                        ):
                            stochastic_status = (
                                "P_alt="
                                f"{float(lateral_candidate.get('p_alternate', 1.0)):g}"
                            )

                        else:
                            stochastic_status = (
                                "deterministic stacking"
                            )

                        self.status.emit(
                            "Candidate "
                            f"{candidate_index + 1}/"
                            f"{total_candidate_count}: "
                            f"axis1={shape_candidate.axis1_cells:g}, "
                            f"axis2={shape_candidate.axis2_cells:g}, "
                            f"layers={shape_candidate.axis3_cells:g}, "
                            f"spacing={float(spacing_value):g} Å, "
                            f"u={float(lateral_candidate['u']):g}, "
                            f"v={float(lateral_candidate['v']):g}, "
                            f"{stochastic_status}"
                        )

                        shell_start = time.perf_counter()

                        if stack_specification.enabled:
                            topology_margin = float(
                                (
                                    base_config.get(
                                        "refinement",
                                        {},
                                    )
                                    or {}
                                ).get(
                                    "stacked_topology_margin",
                                    3.0,
                                )
                            )

                            if (
                                stochastic_ensemble_spec
                                .use_stacking_as_bulk
                            ):
                                from pdf_fitting.models.streaming_stacking_bulk_model import (
                                    StreamingStackingBulkModel,
                                )

                                model_for_fit = (
                                    StreamingStackingBulkModel(
                                        config=candidate_config,
                                        source_structure=structure,
                                        stack_spec=(
                                            candidate_stack_spec
                                        ),
                                        ensemble_spec=(
                                            stochastic_ensemble_spec
                                        ),
                                        pair_cutoff=float(
                                            pair_cutoff
                                        ),
                                        r_grid=experimental_r,
                                        reference_parameters=(
                                            candidate_config.get(
                                                "initial",
                                                {},
                                            )
                                            or {}
                                        ),
                                    )
                                )

                                backend_label = (
                                    "streaming_stochastic_bulk"
                                )

                            else:
                                model_for_fit = (
                                    StochasticStackedPdfModel(
                                        pdf_calculator=pdf_model,
                                        source_structure=structure,
                                        shape_spec=shape_candidate,
                                        stack_spec=candidate_stack_spec,
                                        pair_cutoff=float(
                                            pair_cutoff
                                        ),
                                        topology_margin=topology_margin,
                                        reference_parameters=(
                                            candidate_config.get(
                                                "initial",
                                                {},
                                            )
                                            or {}
                                        ),
                                    )
                                )

                                backend_label = (
                                    "stochastic_ensemble_"
                                    + str(
                                        candidate_stack_spec.model
                                    )
                                )
                                
                            if (
                                candidate_stack_spec.model
                                in {
                                    "correlated_ab_disorder",
                                    "generic_markov",
                                }
                            ):
                                model_for_fit = (
                                    StochasticStackedPdfModel(
                                        pdf_calculator=pdf_model,
                                        source_structure=structure,
                                        shape_spec=shape_candidate,
                                        stack_spec=candidate_stack_spec,
                                        pair_cutoff=float(
                                            pair_cutoff
                                        ),
                                        topology_margin=topology_margin,
                                        reference_parameters=(
                                            candidate_config.get(
                                                "initial",
                                                {},
                                            )
                                            or {}
                                        ),
                                    )
                                )

                                backend_label = (
                                    "stochastic_ensemble_"
                                    + str(
                                        candidate_stack_spec.model
                                    )
                                )

                            else:
                                model_for_fit = (
                                    DynamicStackedPdfModel(
                                        pdf_calculator=pdf_model,
                                        source_structure=structure,
                                        shape_spec=shape_candidate,
                                        stack_spec=candidate_stack_spec,
                                        pair_cutoff=float(
                                            pair_cutoff
                                        ),
                                        topology_margin=topology_margin,
                                        reference_parameters=(
                                            candidate_config.get(
                                                "initial",
                                                {},
                                            )
                                            or {}
                                        ),
                                    )
                                )

                                backend_label = (
                                    "dynamic_stacked"
                                )

                        else:
                            finite_result = (
                                get_or_build_finite_shell_result(
                                    structure,
                                    pair_cutoff=float(
                                        pair_cutoff
                                    ),
                                    shape_spec=shape_candidate,
                                    refinement_config=(
                                        base_config.get(
                                            "refinement",
                                            {},
                                        )
                                        or {}
                                    ),
                                    boundary_mode="atom",
                                )
                            )

                            pdf_model.replace_shell_table(
                                finite_result.shell_table,
                                device_shell_table=(
                                    finite_result
                                    .device_shell_table
                                ),
                            )

                            model_for_fit = pdf_model
                            backend_label = (
                                finite_result.backend
                            )

                        shell_seconds = (
                            time.perf_counter()
                            - shell_start
                        )

                        def progress_callback(
                            payload: Dict[str, Any],
                        ) -> None:
                            output = dict(
                                payload
                            )

                            parameter_values = dict(
                                output.get(
                                    "params",
                                    {},
                                )
                                or {}
                            )

                            refinable_names = {
                                str(parameter_name)
                                for parameter_name, enabled
                                in (
                                    candidate_config.get(
                                        "refinable",
                                        {},
                                    )
                                    or {}
                                ).items()
                                if bool(enabled)
                            }

                            output["params"] = {
                                parameter_name: parameter_values[
                                    parameter_name
                                ]
                                for parameter_name in refinable_names
                                if parameter_name in parameter_values
                            }

                            output["cs_grid"] = True

                            output["shape_index"] = int(
                                cs_index
                            )

                            output["shape_total"] = int(
                                len(
                                    shape_candidates
                                )
                            )

                            output["cs_index"] = int(
                                cs_index
                            )

                            output["index"] = int(
                                candidate_index
                            )

                            output["n_total"] = int(
                                total_candidate_count
                            )

                            output["stacking_candidate_index"] = int(
                                candidate_index
                            )

                            output["diameter_cells"] = float(
                                shape_candidate.diameter_cells
                            )

                            output["height_cells"] = float(
                                shape_candidate.height_cells
                            )

                            output["axis1_cells"] = float(
                                shape_candidate.axis1_cells
                            )

                            output["axis2_cells"] = float(
                                shape_candidate.axis2_cells
                            )

                            output["axis3_cells"] = float(
                                shape_candidate.axis3_cells
                            )

                            output["number_of_layers"] = (
                                int(
                                    round(
                                        shape_candidate.axis3_cells
                                    )
                                )
                                if stack_specification.enabled
                                else None
                            )

                            output["stacking_distance_grid"] = float(
                                spacing_value
                            )

                            output["stacking_u"] = float(
                                lateral_candidate["u"]
                            )

                            output["stacking_v"] = float(
                                lateral_candidate["v"]
                            )

                            output["p_alternate"] = float(
                                lateral_candidate.get(
                                    "p_alternate",
                                    candidate_stack_spec.p_alternate,
                                )
                            )

                            output["p_same_state"] = float(
                                1.0
                                - output["p_alternate"]
                            )

                            output["p_alt_index"] = int(
                                lateral_candidate.get(
                                    "p_alt_index",
                                    0,
                                )
                            )

                            output["markov_matrix_index"] = int(
                                lateral_candidate.get(
                                    "markov_matrix_index",
                                    0,
                                )
                            )

                            output["markov_state_labels"] = [
                                str(
                                    state.label
                                )
                                for state in (
                                    candidate_stack_spec.markov_states
                                    or []
                                )
                            ]

                            selected_markov_matrix = (
                                lateral_candidate.get(
                                    "markov_transition_matrix",
                                    None,
                                )
                            )

                            if selected_markov_matrix is None:
                                output[
                                    "markov_transition_matrix"
                                ] = None

                            else:
                                output[
                                    "markov_transition_matrix"
                                ] = [
                                    [
                                        float(
                                            probability
                                        )
                                        for probability in row
                                    ]
                                    for row in selected_markov_matrix
                                ]

                            self.progress.emit(
                                output
                            )

                        fit_start_time = time.perf_counter()

                        engine = FitEngine(
                            experimental_r,
                            experimental_g,
                            model_for_fit,
                            candidate_config,
                            progress_callback=progress_callback,
                            stop_event=self.stop_event,
                            pause_event=self.pause_event,
                        )

                        try:
                            refined_parameters = (
                                engine.run_refinement()
                            )

                        except FitStopped:
                            stop_requested = True
                            break

                        fit_elapsed_seconds = float(
                            time.perf_counter()
                            - fit_start_time
                        )
                        

                        if refined_parameters is None:
                            refined_parameters = dict(
                                candidate_config[
                                    "initial"
                                ]
                            )

                        if (
                            getattr(
                                engine,
                                "final_G_fit",
                                None,
                            )
                            is not None
                        ):
                            calculated_g = np.asarray(
                                engine.final_G_fit,
                                dtype=float,
                            )
                        else:
                            model_result = (
                                model_for_fit.evaluate(
                                    refined_parameters
                                )
                            )

                            calculated_g = np.asarray(
                                model_result["G_r"],
                                dtype=float,
                            )

                        rwp = self._compute_rwp(
                            experimental_g,
                            calculated_g,
                        )

                        fit_r = np.asarray(
                            (
                                engine.final_r
                                if getattr(
                                    engine,
                                    "final_r",
                                    None,
                                )
                                is not None
                                else experimental_r
                            ),
                            dtype=float,
                        )

                        fit_experimental = np.asarray(
                            (
                                engine.final_G_exp
                                if getattr(
                                    engine,
                                    "final_G_exp",
                                    None,
                                )
                                is not None
                                else experimental_g
                            ),
                            dtype=float,
                        )

                        fit_residual = np.asarray(
                            (
                                engine.final_residual
                                if getattr(
                                    engine,
                                    "final_residual",
                                    None,
                                )
                                is not None
                                else (
                                    fit_experimental
                                    - calculated_g
                                )
                            ),
                            dtype=float,
                        )

                        if stack_specification.enabled:
                            model_for_fit.evaluate(
                                refined_parameters
                            )

                            final_spacing = float(
                                model_for_fit
                                .current_stacking_distance
                            )

                            physical_thickness = float(
                                model_for_fit
                                .current_stack_thickness
                            )

                        else:
                            final_spacing = None
                            physical_thickness = None

                        result_row = {
                            "index": int(
                                candidate_index
                            ),
                            "n_total": int(
                                total_candidate_count
                            ),
                            "cs_index": int(
                                cs_index
                            ),
                            "spacing_index": int(
                                spacing_index
                            ),
                            "lateral_index": int(
                                lateral_index
                            ),
                            "stacking_candidate_index": int(
                                candidate_index
                            ),
                            "lateral_sample_index": int(
                                lateral_candidate[
                                    "sample_index"
                                ]
                            ),
                            "p_alt_index": int(
                                lateral_candidate.get(
                                    "p_alt_index",
                                    0,
                                )
                            ),
                            "p_alternate": (
                                float(
                                    lateral_candidate.get(
                                        "p_alternate",
                                        candidate_stack_spec.p_alternate,
                                    )
                                )
                                if (
                                    candidate_stack_spec.model
                                    == "correlated_ab_disorder"
                                )
                                else None
                            ),
                            "p_same_state": (
                                float(
                                    1.0
                                    - float(
                                        lateral_candidate.get(
                                            "p_alternate",
                                            candidate_stack_spec.p_alternate,
                                        )
                                    )
                                )
                                if (
                                    candidate_stack_spec.model
                                    == "correlated_ab_disorder"
                                )
                                else None
                            ),
                            "markov_matrix_index": int(
                                lateral_candidate.get(
                                    "markov_matrix_index",
                                    0,
                                )
                            ),
                            "markov_transition_matrix": (
                                None
                                if lateral_candidate.get(
                                    "markov_transition_matrix",
                                    None,
                                )
                                is None
                                else [
                                    [
                                        float(
                                            probability
                                        )
                                        for probability in row
                                    ]
                                    for row in lateral_candidate[
                                        "markov_transition_matrix"
                                    ]
                                ]
                            ),
                            "markov_state_labels": [
                                str(
                                    state.label
                                )
                                for state in (
                                    candidate_stack_spec.markov_states
                                    or []
                                )
                            ],
                            "diameter_cells": float(
                                shape_candidate.diameter_cells
                            ),
                            "height_cells": float(
                                shape_candidate.height_cells
                            ),
                            "axis1_cells": float(
                                shape_candidate.axis1_cells
                            ),
                            "axis2_cells": float(
                                shape_candidate.axis2_cells
                            ),
                            "axis3_cells": float(
                                shape_candidate.axis3_cells
                            ),
                            "number_of_layers": (
                                int(
                                    round(
                                        shape_candidate.axis3_cells
                                    )
                                )
                                if stack_specification.enabled
                                else None
                            ),
                            "stacking_spacing_mode": (
                                candidate_stack_spec.spacing_mode
                                if stack_specification.enabled
                                else None
                            ),
                            "stacking_distance_grid": float(
                                spacing_value
                            ),
                            "stacking_distance": (
                                final_spacing
                            ),
                            "stacking_u": float(
                                lateral_candidate["u"]
                            ),
                            "stacking_v": float(
                                lateral_candidate["v"]
                            ),
                            "physical_stack_thickness": (
                                physical_thickness
                            ),
                            "rwp": float(rwp),
                            "params": dict(
                                refined_parameters
                            ),
                            "initial_params": dict(
                                candidate_config.get(
                                    "initial",
                                    {},
                                )
                                or {}
                            ),
                            "refinable_flags": dict(
                                candidate_config.get(
                                    "refinable",
                                    {},
                                )
                                or {}
                            ),
                            "fit_elapsed_s": float(
                                fit_elapsed_seconds
                            ),
                            "shell_build_s": float(
                                shell_seconds
                            ),
                            "backend": backend_label,
                            "detail_file": "",
                            "candidate_input_file": "",
                            "metadata_file": "",
                        }

                        previous_cs_best = best_by_cs.get(
                            int(cs_index)
                        )

                        is_best_for_cs = bool(
                            previous_cs_best is None
                            or float(rwp)
                            < float(
                                previous_cs_best[
                                    "rwp"
                                ]
                            )
                        )

                        if is_best_for_cs:
                            best_by_cs[
                                int(cs_index)
                            ] = result_row

                        if (
                            detail_root is not None
                            and (
                                search_specification.detail_storage
                                == "all"
                                or (
                                    search_specification.detail_storage
                                    == "best_per_cs"
                                    and is_best_for_cs
                                )
                            )
                        ):
                            detail_paths = (
                                self._write_candidate_detail(
                                    root_directory=(
                                        detail_root
                                    ),
                                    cs_index=cs_index,
                                    candidate_index=(
                                        candidate_index
                                    ),
                                    row=result_row,
                                    r=fit_r,
                                    experimental=(
                                        fit_experimental
                                    ),
                                    calculated=calculated_g,
                                    residual=fit_residual,
                                    candidate_config=(
                                        candidate_config
                                    ),
                                    candidate_shape_spec=(
                                        shape_candidate
                                    ),
                                    candidate_stack_spec=(
                                        candidate_stack_spec
                                    ),
                                    candidate_search_spec=(
                                        search_specification
                                    ),
                                    refined_parameters=(
                                        refined_parameters
                                    ),
                                    best_per_cs=(
                                        search_specification
                                        .detail_storage
                                        == "best_per_cs"
                                    ),
                                )
                            )

                            result_row.update(
                                detail_paths
                            )

                        results.append(
                            result_row
                        )

                        previous_global_best = (
                            float(
                                self._best.get(
                                    "rwp",
                                    float("inf"),
                                )
                            )
                            if self._best is not None
                            else float("inf")
                        )

                        is_new_global_best = bool(
                            self._best is None
                            or (
                                np.isfinite(rwp)
                                and float(rwp)
                                < previous_global_best
                            )
                        )

                        if is_new_global_best:
                            self._best = dict(
                                result_row
                            )

                            self._best_visual = {
                                "r": fit_r,
                                "G_exp": fit_experimental,
                                "G_fit": calculated_g,
                                "residual": fit_residual,
                                "rwp": float(rwp),
                                "index": int(
                                    candidate_index
                                ),
                                "n_total": int(
                                    total_candidate_count
                                ),
                                "cs_index": int(
                                    cs_index
                                ),
                                "diameter_cells": float(
                                    shape_candidate
                                    .diameter_cells
                                ),
                                "height_cells": float(
                                    shape_candidate
                                    .height_cells
                                ),
                                "axis1_cells": float(
                                    shape_candidate.axis1_cells
                                ),
                                "axis2_cells": float(
                                    shape_candidate.axis2_cells
                                ),
                                "axis3_cells": float(
                                    shape_candidate.axis3_cells
                                ),
                                "number_of_layers": (
                                    result_row[
                                        "number_of_layers"
                                    ]
                                ),
                                "stacking_distance_grid": float(
                                    spacing_value
                                ),
                                "stacking_distance": (
                                    final_spacing
                                ),
                                "stacking_u": float(
                                    lateral_candidate["u"]
                                ),
                                "stacking_v": float(
                                    lateral_candidate["v"]
                                ),
                                "physical_stack_thickness": (
                                    physical_thickness
                                ),
                                "markov_matrix_index": int(
                                    lateral_candidate.get(
                                        "markov_matrix_index",
                                        0,
                                    )
                                ),
                                "markov_transition_matrix": (
                                    None
                                    if lateral_candidate.get(
                                        "markov_transition_matrix",
                                        None,
                                    )
                                    is None
                                    else [
                                        [
                                            float(
                                                probability
                                            )
                                            for probability in row
                                        ]
                                        for row in lateral_candidate[
                                            "markov_transition_matrix"
                                        ]
                                    ]
                                ),
                                "markov_state_labels": [
                                    str(
                                        state.label
                                    )
                                    for state in (
                                        candidate_stack_spec.markov_states
                                        or []
                                    )
                                ],
                                "stacking_distance_grid": float(
                                    spacing_value
                                ),
                                "stacking_u": float(
                                    lateral_candidate["u"]
                                ),
                                "stacking_v": float(
                                    lateral_candidate["v"]
                                ),
                                "initial_params": dict(
                                    candidate_config.get(
                                        "initial",
                                        {},
                                    )
                                    or {}
                                ),
                                "params": dict(
                                    refined_parameters
                                ),
                                "refinable_flags": dict(
                                    candidate_config.get(
                                        "refinable",
                                        {},
                                    )
                                    or {}
                                ),
                                "fit_elapsed_s": float(
                                    fit_elapsed_seconds
                                ),
                            }

                        output_payload = dict(
                            result_row
                        )

                        output_payload[
                            "is_new_global_best"
                        ] = bool(
                            is_new_global_best
                        )

                        output_payload["best"] = (
                            dict(self._best)
                            if self._best is not None
                            else None
                        )

                        if is_new_global_best:
                            output_payload.update(
                                self._best_visual
                            )

                        self.progress.emit(
                            output_payload
                        )

            self.finished_ok.emit(
                {
                    "elapsed_s": float(
                        time.perf_counter()
                        - start_time
                    ),
                    "results": results,
                    "best": self._best,
                    "best_visual": self._best_visual,
                    "best_by_cs": {
                        int(key): dict(value)
                        for key, value in (
                            best_by_cs.items()
                        )
                    },
                    "input_path": self.input_path,
                    "source_path": self.source_path,
                    "refinable_flags": finite_config.get(
                        "refinable",
                        {},
                    ),
                    "stopped": bool(
                        self.stop_event.is_set()
                    ),
                    "save_on_stop": bool(
                        self._save_on_stop
                    ),
                    "detail_root": detail_root,
                    "total_candidates": int(
                        total_candidate_count
                    ),
                    "completed_candidates": int(
                        len(results)
                    ),
                }
            )

        except Exception:
            error_text = traceback.format_exc()

            print(
                "[CS GRID WORKER ERROR]\n"
                + error_text,
                flush=True,
            )

            self.failed.emit(
                error_text
            )