from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np


def _as_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

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


def inclusive_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    minimum = float(minimum)
    maximum = float(maximum)
    step = float(step)

    if maximum < minimum:
        minimum, maximum = (
            maximum,
            minimum,
        )

    if (
        not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError(
            "Grid-search step must be positive and finite."
        )

    count = int(
        np.floor(
            (
                maximum
                - minimum
            )
            / step
            + 0.5
        )
    ) + 1

    values = (
        minimum
        + step
        * np.arange(
            count,
            dtype=float,
        )
    )

    values = values[
        values
        <= maximum
        + 1.0e-10
    ]

    return np.round(
        values,
        12,
    )


@dataclass
class StackingSearchSpec:
    spacing_treatment: str = "fixed"

    spacing_minimum: float = 3.0
    spacing_maximum: float = 5.0
    spacing_step: float = 0.1

    lateral_treatment: str = "fixed"

    lateral_target: str = "repeat_translation"
    sequence_row: int = 0

    scan_u: bool = True
    u_minimum: float = -0.5
    u_maximum: float = 0.5
    u_step: float = 0.1

    scan_v: bool = True
    v_minimum: float = -0.5
    v_maximum: float = 0.5
    v_step: float = 0.1

    random_samples: int = 25
    random_seed: int = 12345

    detail_storage: str = "none"
    detail_directory: str = "stacking_search_results"

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "StackingSearchSpec":
        values = dict(
            values
            or {}
        )

        spacing_treatment = str(
            values.get(
                "spacing_treatment",
                "fixed",
            )
        ).strip().lower()

        if spacing_treatment not in {
            "fixed",
            "continuous",
            "grid",
        }:
            spacing_treatment = "fixed"

        lateral_treatment = str(
            values.get(
                "lateral_treatment",
                "fixed",
            )
        ).strip().lower()

        if lateral_treatment not in {
            "fixed",
            "grid",
            "random",
        }:
            lateral_treatment = "fixed"

        lateral_target = str(
            values.get(
                "lateral_target",
                "repeat_translation",
            )
        ).strip().lower()

        if lateral_target not in {
            "repeat_translation",
            "sequence_row",
        }:
            lateral_target = "repeat_translation"

        detail_storage = str(
            values.get(
                "detail_storage",
                "none",
            )
        ).strip().lower()

        if detail_storage not in {
            "none",
            "best_per_cs",
            "all",
        }:
            detail_storage = "none"

        return cls(
            spacing_treatment=spacing_treatment,
            spacing_minimum=float(
                values.get(
                    "spacing_minimum",
                    3.0,
                )
            ),
            spacing_maximum=float(
                values.get(
                    "spacing_maximum",
                    5.0,
                )
            ),
            spacing_step=float(
                values.get(
                    "spacing_step",
                    0.1,
                )
            ),
            lateral_treatment=lateral_treatment,
            lateral_target=lateral_target,
            sequence_row=max(
                0,
                int(
                    values.get(
                        "sequence_row",
                        0,
                    )
                ),
            ),
            scan_u=_as_bool(
                values.get(
                    "scan_u",
                    True,
                ),
                True,
            ),
            u_minimum=float(
                values.get(
                    "u_minimum",
                    -0.5,
                )
            ),
            u_maximum=float(
                values.get(
                    "u_maximum",
                    0.5,
                )
            ),
            u_step=float(
                values.get(
                    "u_step",
                    0.1,
                )
            ),
            scan_v=_as_bool(
                values.get(
                    "scan_v",
                    True,
                ),
                True,
            ),
            v_minimum=float(
                values.get(
                    "v_minimum",
                    -0.5,
                )
            ),
            v_maximum=float(
                values.get(
                    "v_maximum",
                    0.5,
                )
            ),
            v_step=float(
                values.get(
                    "v_step",
                    0.1,
                )
            ),
            random_samples=max(
                1,
                int(
                    values.get(
                        "random_samples",
                        25,
                    )
                ),
            ),
            random_seed=int(
                values.get(
                    "random_seed",
                    12345,
                )
            ),
            detail_storage=detail_storage,
            detail_directory=str(
                values.get(
                    "detail_directory",
                    "stacking_search_results",
                )
            ).strip()
            or "stacking_search_results",
        )

    def spacing_values(
        self,
        fixed_spacing: float,
    ) -> np.ndarray:
        if self.spacing_treatment == "grid":
            return inclusive_grid(
                self.spacing_minimum,
                self.spacing_maximum,
                self.spacing_step,
            )

        return np.asarray(
            [
                float(
                    fixed_spacing
                )
            ],
            dtype=float,
        )

    def lateral_values(
        self,
        fixed_u: float,
        fixed_v: float,
    ) -> List[Dict[str, float]]:
        if self.lateral_treatment == "fixed":
            return [
                {
                    "u": float(
                        fixed_u
                    ),
                    "v": float(
                        fixed_v
                    ),
                    "sample_index": 0,
                }
            ]

        if self.lateral_treatment == "grid":
            if self.scan_u:
                u_values = inclusive_grid(
                    self.u_minimum,
                    self.u_maximum,
                    self.u_step,
                )
            else:
                u_values = np.asarray(
                    [
                        float(
                            fixed_u
                        )
                    ],
                    dtype=float,
                )

            if self.scan_v:
                v_values = inclusive_grid(
                    self.v_minimum,
                    self.v_maximum,
                    self.v_step,
                )
            else:
                v_values = np.asarray(
                    [
                        float(
                            fixed_v
                        )
                    ],
                    dtype=float,
                )

            output = []
            sample_index = 0

            for u_value in u_values:
                for v_value in v_values:
                    output.append(
                        {
                            "u": float(
                                u_value
                            ),
                            "v": float(
                                v_value
                            ),
                            "sample_index": int(
                                sample_index
                            ),
                        }
                    )

                    sample_index += 1

            return output

        generator = np.random.default_rng(
            int(
                self.random_seed
            )
        )

        output = [
            {
                "u": float(
                    fixed_u
                ),
                "v": float(
                    fixed_v
                ),
                "sample_index": 0,
            }
        ]

        for sample_index in range(
            1,
            int(
                self.random_samples
            ),
        ):
            if self.scan_u:
                u_value = generator.uniform(
                    min(
                        self.u_minimum,
                        self.u_maximum,
                    ),
                    max(
                        self.u_minimum,
                        self.u_maximum,
                    ),
                )
            else:
                u_value = fixed_u

            if self.scan_v:
                v_value = generator.uniform(
                    min(
                        self.v_minimum,
                        self.v_maximum,
                    ),
                    max(
                        self.v_minimum,
                        self.v_maximum,
                    ),
                )
            else:
                v_value = fixed_v

            output.append(
                {
                    "u": float(
                        u_value
                    ),
                    "v": float(
                        v_value
                    ),
                    "sample_index": int(
                        sample_index
                    ),
                }
            )

        return output

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "spacing_treatment": self.spacing_treatment,
            "spacing_minimum": float(
                self.spacing_minimum
            ),
            "spacing_maximum": float(
                self.spacing_maximum
            ),
            "spacing_step": float(
                self.spacing_step
            ),
            "lateral_treatment": self.lateral_treatment,
            "lateral_target": self.lateral_target,
            "sequence_row": int(
                self.sequence_row
            ),
            "scan_u": bool(
                self.scan_u
            ),
            "u_minimum": float(
                self.u_minimum
            ),
            "u_maximum": float(
                self.u_maximum
            ),
            "u_step": float(
                self.u_step
            ),
            "scan_v": bool(
                self.scan_v
            ),
            "v_minimum": float(
                self.v_minimum
            ),
            "v_maximum": float(
                self.v_maximum
            ),
            "v_step": float(
                self.v_step
            ),
            "random_samples": int(
                self.random_samples
            ),
            "random_seed": int(
                self.random_seed
            ),
            "detail_storage": self.detail_storage,
            "detail_directory": self.detail_directory,
        }


def stacking_search_spec_from_config(
    config: Mapping[str, Any] | None,
) -> StackingSearchSpec:
    config = dict(
        config
        or {}
    )

    return StackingSearchSpec.from_dict(
        config.get(
            "stacking_search",
            {},
        )
    )

def apply_lateral_candidate(
    stack_spec,
    search_spec: StackingSearchSpec,
    *,
    u: float,
    v: float,
):
    from pdf_fitting.models.layer_stacking import (
        LayerStackSpec,
    )

    candidate = LayerStackSpec.from_dict(
        stack_spec.to_dict()
    )

    if search_spec.lateral_target == "sequence_row":
        row = int(
            search_spec.sequence_row
        )

        if row < 0 or row >= len(candidate.sequence):
            raise IndexError(
                "The selected stacking sequence row is outside "
                "the current sequence."
            )

        candidate.sequence[row].shift_u = float(u)
        candidate.sequence[row].shift_v = float(v)

    else:
        candidate.repeat_shift_u = float(u)
        candidate.repeat_shift_v = float(v)

    return candidate.normalized()


def stacking_candidate_count(
    search_spec: StackingSearchSpec,
    *,
    fixed_spacing: float,
    fixed_u: float,
    fixed_v: float,
) -> Dict[str, int]:
    spacing_count = int(
        search_spec.spacing_values(
            fixed_spacing
        ).size
    )

    lateral_count = len(
        search_spec.lateral_values(
            fixed_u,
            fixed_v,
        )
    )

    return {
        "spacing": spacing_count,
        "lateral": lateral_count,
        "total": spacing_count * lateral_count,
    }

