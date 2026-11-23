from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
    default_ab_sequence,
)


def _as_bool(
    value: Any,
) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    return bool(value)


def _default_sequence_dicts() -> List[
    Dict[str, Any]
]:
    return [
        layer.to_dict()
        for layer in default_ab_sequence()
    ]


@dataclass
class FiniteCrystalliteState:
    """
    Serializable GUI state.

    When stacking is disabled, axis3_cells is a conventional geometric
    dimension measured in oriented unit-cell lengths.

    When stacking is enabled, axis3_cells is the integer number of complete
    stacking-layer positions. The physical thickness is derived from the
    realized sequence and nominal stacking distance.

    The stacking sequence itself is infinite and periodic.
    """

    shape_type: str = "cylinder"

    axis1_cells: float = 10.0
    axis2_cells: float = 10.0
    axis3_cells: float = 5.0

    search_mode: str = "predefined"

    scan_axis1: bool = True
    axis1_min_cells: float = 5.0
    axis1_max_cells: float = 20.0
    axis1_step_cells: float = 1.0

    scan_axis2: bool = False
    axis2_min_cells: float = 5.0
    axis2_max_cells: float = 20.0
    axis2_step_cells: float = 1.0

    scan_axis3: bool = True
    axis3_min_cells: float = 1.0
    axis3_max_cells: float = 15.0
    axis3_step_cells: float = 1.0

    axis_h: int = 0
    axis_k: int = 0
    axis_l: int = 1

    base1_h: int = 1
    base1_k: int = 0
    base1_l: int = 0

    base2_h: int = 0
    base2_k: int = 1
    base2_l: int = 0

    show_shape_surface: bool = True
    show_shape_axes: bool = True
    shape_opacity: float = 0.18

    show_atoms: bool = True
    show_layer_origins: bool = True
    all_atom_preview_limit: int = 100000
    orthographic_projection: bool = True

    show_layer_labels: bool = True
    show_parent_unit_cell: bool = False
    show_lattice_vectors: bool = True
    show_periodic_closure_vector: bool = True

    preview_view_preset: str = "ac_section"
    preview_view_mode: str = "3d"

    preview_max_layer_labels: int = 50
    preview_label_font_size: int = 12
    preview_atom_size: float = 10.0
    preview_layer_origin_size: float = 15.0

    stacking_enabled: bool = False

    stacking_model: str = "periodic_sequence"

    stacking_p_alternate: float = 1.0
    stacking_p_grid_enabled: bool = False
    stacking_p_grid_minimum: float = 0.0
    stacking_p_grid_maximum: float = 1.0
    stacking_p_grid_step: float = 0.05

    stacking_ensemble_samples: int = 20
    stacking_ensemble_seed: int = 12345
    stacking_ensemble_start_state: str = "A"
    stacking_preview_realization_index: int = 0

    stacking_state_a_du: float = 0.0
    stacking_state_a_dv: float = 0.0
    stacking_state_a_h: float = 3.8

    stacking_state_b_du: float = 0.0
    stacking_state_b_dv: float = 0.0
    stacking_state_b_h: float = 3.8

    stacking_definition_mode: str = (
        "repeating_motif"
    )

    stacking_vector_mode: str = "incremental"

    stacking_nominal_spacing: float = 5.0

    refine_stacking_nominal_spacing: bool = False
    stacking_nominal_spacing_min: float = 0.1
    stacking_nominal_spacing_max: float = 100.0

    stacking_repeat_period: float = 2.0
    stacking_repeat_shift_u: float = 0.0
    stacking_repeat_shift_v: float = 0.0

    stacking_sequence: List[
        Dict[str, Any]
    ] = field(
        default_factory=_default_sequence_dicts
    )

    show_transition_vectors: bool = True
    color_stacking_layers: bool = True
    show_generated_layers: bool = True

    stacking_spacing_mode: str = "independent"

    stacking_fraction: float = 1.0
    refine_stacking_fraction: bool = False
    stacking_fraction_min: float = 0.001
    stacking_fraction_max: float = 100.0


    def number_of_complete_layers(
        self,
    ) -> int:
        """
        Return the discrete stack size represented by axis3_cells.
        """
        return max(
            1,
            int(
                round(
                    float(
                        self.axis3_cells
                    )
                )
            ),
        )

    
    def sequence_layers(
        self,
    ) -> List[SequenceLayer]:
        layers: List[
            SequenceLayer
        ] = []

        for row, value in enumerate(
            self.stacking_sequence
        ):
            if isinstance(
                value,
                SequenceLayer,
            ):
                layers.append(
                    value
                )

            elif isinstance(
                value,
                Mapping,
            ):
                layers.append(
                    SequenceLayer.from_dict(
                        value,
                        default_coordinate=float(
                            row
                        ),
                        nominal_spacing=(
                            self.stacking_nominal_spacing
                        ),
                    )
                )

        if not layers:
            layers = default_ab_sequence()

        return layers

    def stacking_spec(
        self,
    ) -> LayerStackSpec:
        return LayerStackSpec(
            enabled=bool(
                self.stacking_enabled
            ),
            model=str(
                self.stacking_model
            ),
            p_alternate=float(
                self.stacking_p_alternate
            ),
            p_grid_enabled=bool(
                self.stacking_p_grid_enabled
            ),
            p_grid_minimum=float(
                self.stacking_p_grid_minimum
            ),
            p_grid_maximum=float(
                self.stacking_p_grid_maximum
            ),
            p_grid_step=float(
                self.stacking_p_grid_step
            ),
            ensemble_samples=int(
                self.stacking_ensemble_samples
            ),
            ensemble_seed=int(
                self.stacking_ensemble_seed
            ),
            ensemble_start_state=str(
                self.stacking_ensemble_start_state
            ),
            preview_realization_index=int(
                self.stacking_preview_realization_index
            ),
            state_a_du=float(
                self.stacking_state_a_du
            ),
            state_a_dv=float(
                self.stacking_state_a_dv
            ),
            state_a_h=float(
                self.stacking_state_a_h
            ),
            state_b_du=float(
                self.stacking_state_b_du
            ),
            state_b_dv=float(
                self.stacking_state_b_dv
            ),
            state_b_h=float(
                self.stacking_state_b_h
            ),
            definition_mode=str(
                self.stacking_definition_mode
            ),
            vector_mode=str(
                self.stacking_vector_mode
            ),
            nominal_spacing=float(
                self.stacking_nominal_spacing
            ),
            refine_nominal_spacing=bool(
                self.refine_stacking_nominal_spacing
            ),
            nominal_spacing_min=float(
                self.stacking_nominal_spacing_min
            ),
            nominal_spacing_max=float(
                self.stacking_nominal_spacing_max
            ),
            repeat_period=float(
                self.stacking_repeat_period
            ),
            repeat_shift_u=float(
                self.stacking_repeat_shift_u
            ),
            repeat_shift_v=float(
                self.stacking_repeat_shift_v
            ),
            sequence=self.sequence_layers(),
            show_transition_vectors=bool(
                self.show_transition_vectors
            ),
            color_layers=bool(
                self.color_stacking_layers
            ),
            spacing_mode=str(
                self.stacking_spacing_mode
            ),
            stacking_fraction=float(
                self.stacking_fraction
            ),
            refine_stacking_fraction=bool(
                self.refine_stacking_fraction
            ),
            stacking_fraction_min=float(
                self.stacking_fraction_min
            ),
            stacking_fraction_max=float(
                self.stacking_fraction_max
            ),
        ).normalized()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        output = asdict(
            self
        )

        stacking_keys = (
            "stacking_enabled",
            "stacking_vector_mode",
            "stacking_definition_mode",
            "stacking_nominal_spacing",
            "refine_stacking_nominal_spacing",
            "stacking_nominal_spacing_min",
            "stacking_nominal_spacing_max",
            "stacking_spacing_mode",
            "stacking_fraction",
            "refine_stacking_fraction",
            "stacking_fraction_min",
            "stacking_fraction_max",
            "stacking_repeat_period",
            "stacking_repeat_shift_u",
            "stacking_repeat_shift_v",
            "stacking_sequence",
            "show_transition_vectors",
            "color_stacking_layers",
        )

        for key in stacking_keys:
            output.pop(
                key,
                None,
            )

        output["stacking"] = (
            self.stacking_spec().to_dict()
        )

        return output

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "FiniteCrystalliteState":
        values = dict(
            values
            or {}
        )

        state = cls()

        simple_fields = (
            "shape_type",
            "axis1_cells",
            "axis2_cells",
            "axis3_cells",
            "search_mode",
            "scan_axis1",
            "axis1_min_cells",
            "axis1_max_cells",
            "axis1_step_cells",
            "scan_axis2",
            "axis2_min_cells",
            "axis2_max_cells",
            "axis2_step_cells",
            "scan_axis3",
            "axis3_min_cells",
            "axis3_max_cells",
            "axis3_step_cells",
            "axis_h",
            "axis_k",
            "axis_l",
            "base1_h",
            "base1_k",
            "base1_l",
            "base2_h",
            "base2_k",
            "base2_l",
            "show_shape_surface",
            "show_shape_axes",
            "shape_opacity",
            "show_atoms",
            "show_layer_origins",
            "all_atom_preview_limit",
            "orthographic_projection",
            "show_layer_labels",
            "show_parent_unit_cell",
            "show_lattice_vectors",
            "show_periodic_closure_vector",
            "preview_view_preset",
            "preview_view_mode",
            "preview_max_layer_labels",
            "preview_label_font_size",
            "preview_atom_size",
            "preview_layer_origin_size",
            "show_generated_layers",
        )

        for key in simple_fields:
            if key not in values:
                continue

            if not hasattr(
                state,
                key,
            ):
                continue

            old_value = getattr(
                state,
                key,
            )
            value = values[
                key
            ]

            try:
                if isinstance(
                    old_value,
                    bool,
                ):
                    value = _as_bool(
                        value
                    )

                elif isinstance(
                    old_value,
                    int,
                ):
                    value = int(
                        value
                    )

                elif isinstance(
                    old_value,
                    float,
                ):
                    value = float(
                        value
                    )

                else:
                    value = str(
                        value
                    )

                setattr(
                    state,
                    key,
                    value,
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        if "axis1_cells" not in values:
            state.axis1_cells = float(
                values.get(
                    "diameter_cells",
                    state.axis1_cells,
                )
            )

        if "axis2_cells" not in values:
            state.axis2_cells = float(
                state.axis1_cells
            )

        if "axis3_cells" not in values:
            state.axis3_cells = float(
                values.get(
                    "height_cells",
                    state.axis3_cells,
                )
            )

        raw_stacking = values.get(
            "stacking",
            values.get(
                "stacking_faults",
                {},
            ),
        )

        if isinstance(
            raw_stacking,
            Mapping,
        ):
            spec = LayerStackSpec.from_dict(
                raw_stacking
            )

            state.stacking_enabled = (
                spec.enabled
            )

            state.stacking_model = (
                spec.model
            )

            state.stacking_p_alternate = (
                spec.p_alternate
            )

            state.stacking_p_grid_enabled = (
                spec.p_grid_enabled
            )

            state.stacking_p_grid_minimum = (
                spec.p_grid_minimum
            )

            state.stacking_p_grid_maximum = (
                spec.p_grid_maximum
            )

            state.stacking_p_grid_step = (
                spec.p_grid_step
            )

            state.stacking_ensemble_samples = (
                spec.ensemble_samples
            )

            state.stacking_ensemble_seed = (
                spec.ensemble_seed
            )

            state.stacking_ensemble_start_state = (
                spec.ensemble_start_state
            )

            state.stacking_preview_realization_index = (
                spec.preview_realization_index
            )

            state.stacking_state_a_du = (
                spec.state_a_du
            )

            state.stacking_state_a_dv = (
                spec.state_a_dv
            )

            state.stacking_state_a_h = (
                spec.state_a_h
            )

            state.stacking_state_b_du = (
                spec.state_b_du
            )

            state.stacking_state_b_dv = (
                spec.state_b_dv
            )

            state.stacking_state_b_h = (
                spec.state_b_h
            )
            
            state.stacking_definition_mode = (
                spec.definition_mode
            )

            state.stacking_vector_mode = (
                spec.vector_mode
            )

            state.stacking_nominal_spacing = (
                spec.nominal_spacing
            )

            state.refine_stacking_nominal_spacing = (
                spec.refine_nominal_spacing
            )

            state.stacking_nominal_spacing_min = (
                spec.nominal_spacing_min
            )

            state.stacking_nominal_spacing_max = (
                spec.nominal_spacing_max
            )

            state.stacking_repeat_period = (
                spec.repeat_period
            )

            state.stacking_repeat_shift_u = (
                spec.repeat_shift_u
            )

            state.stacking_repeat_shift_v = (
                spec.repeat_shift_v
            )

            state.stacking_sequence = [
                layer.to_dict()
                for layer in spec.sequence
            ]

            state.show_transition_vectors = (
                spec.show_transition_vectors
            )

            state.color_stacking_layers = (
                spec.color_layers
            )

            state.stacking_spacing_mode = (
                spec.spacing_mode
            )

            state.stacking_fraction = (
                spec.stacking_fraction
            )

            state.refine_stacking_fraction = (
                spec.refine_stacking_fraction
            )

            state.stacking_fraction_min = (
                spec.stacking_fraction_min
            )

            state.stacking_fraction_max = (
                spec.stacking_fraction_max
            )

        return state