from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


TREATMENT_FIXED = "fixed"
TREATMENT_CONTINUOUS = "continuous"
TREATMENT_GRID = "grid"

VALID_TREATMENTS = {
    TREATMENT_FIXED,
    TREATMENT_CONTINUOUS,
    TREATMENT_GRID,
}


def _as_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if value is None:
        return bool(
            default
        )

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, float),
    ):
        return bool(
            value
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    return bool(
        default
    )


@dataclass
class ComponentControl:
    enabled: bool = False
    minimum: float = -0.5
    maximum: float = 0.5
    step: float = 0.05

    def normalized(
        self,
    ) -> "ComponentControl":
        minimum = float(
            self.minimum
        )

        maximum = float(
            self.maximum
        )

        if maximum < minimum:
            minimum, maximum = (
                maximum,
                minimum,
            )

        step = abs(
            float(
                self.step
            )
        )

        if step <= 1.0e-12:
            step = 0.05

        return ComponentControl(
            enabled=bool(
                self.enabled
            ),
            minimum=minimum,
            maximum=maximum,
            step=step,
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        value = self.normalized()

        return {
            "enabled": value.enabled,
            "minimum": value.minimum,
            "maximum": value.maximum,
            "step": value.step,
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
        *,
        normal_component: bool = False,
    ) -> "ComponentControl":
        values = dict(
            values
            or {}
        )

        if normal_component:
            default_minimum = 0.001
            default_maximum = 20.0
            default_step = 0.05
        else:
            default_minimum = -0.5
            default_maximum = 0.5
            default_step = 0.05

        return cls(
            enabled=_as_bool(
                values.get(
                    "enabled",
                    False,
                )
            ),
            minimum=float(
                values.get(
                    "minimum",
                    default_minimum,
                )
            ),
            maximum=float(
                values.get(
                    "maximum",
                    default_maximum,
                )
            ),
            step=float(
                values.get(
                    "step",
                    default_step,
                )
            ),
        ).normalized()


@dataclass
class TransitionControl:
    transition_id: str
    label: str = ""

    u: ComponentControl = field(
        default_factory=ComponentControl
    )

    v: ComponentControl = field(
        default_factory=ComponentControl
    )

    h: ComponentControl = field(
        default_factory=lambda: ComponentControl(
            minimum=0.001,
            maximum=20.0,
            step=0.05,
        )
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "label": self.label,
            "u": self.u.to_dict(),
            "v": self.v.to_dict(),
            "h": self.h.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "TransitionControl":
        values = dict(
            values
            or {}
        )

        return cls(
            transition_id=str(
                values.get(
                    "transition_id",
                    "",
                )
            ),
            label=str(
                values.get(
                    "label",
                    "",
                )
            ),
            u=ComponentControl.from_dict(
                values.get(
                    "u",
                    {},
                )
            ),
            v=ComponentControl.from_dict(
                values.get(
                    "v",
                    {},
                )
            ),
            h=ComponentControl.from_dict(
                values.get(
                    "h",
                    {},
                ),
                normal_component=True,
            ),
        )


@dataclass
class StackingRefinementState:
    treatment: str = TREATMENT_FIXED

    transitions: List[
        TransitionControl
    ] = field(
        default_factory=list
    )

    def normalized(
        self,
    ) -> "StackingRefinementState":
        treatment = str(
            self.treatment
            or TREATMENT_FIXED
        ).strip().lower()

        if treatment not in VALID_TREATMENTS:
            treatment = TREATMENT_FIXED

        transition_map = {}

        for transition in self.transitions:
            if not isinstance(
                transition,
                TransitionControl,
            ):
                transition = (
                    TransitionControl.from_dict(
                        transition
                    )
                )

            transition_id = str(
                transition.transition_id
            ).strip()

            if not transition_id:
                continue

            transition_map[
                transition_id
            ] = transition

        return StackingRefinementState(
            treatment=treatment,
            transitions=list(
                transition_map.values()
            ),
        )

    def control_map(
        self,
    ) -> Dict[str, TransitionControl]:
        normalized = self.normalized()

        return {
            transition.transition_id: transition
            for transition in normalized.transitions
        }

    def transition_map(
        self,
    ) -> Dict[str, TransitionControl]:
        """
        Compatibility alias for control_map().
        """
        return self.control_map()
    
    def to_dict(
        self,
    ) -> Dict[str, Any]:
        normalized = self.normalized()

        return {
            "treatment": normalized.treatment,
            "transitions": [
                transition.to_dict()
                for transition in normalized.transitions
            ],
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "StackingRefinementState":
        values = dict(
            values
            or {}
        )

        transitions = []

        for value in values.get(
            "transitions",
            [],
        ):
            if isinstance(
                value,
                Mapping,
            ):
                transitions.append(
                    TransitionControl.from_dict(
                        value
                    )
                )

        return cls(
            treatment=str(
                values.get(
                    "treatment",
                    TREATMENT_FIXED,
                )
            ),
            transitions=transitions,
        ).normalized()

# -------------------------------------------------------------------------
# Compatibility aliases used by the finite-crystallite dialog.
#
# These aliases let older and newer GUI code use either naming convention.
# This block is at module level, so it must start at column zero.
# -------------------------------------------------------------------------
STACKING_TREATMENT_FIXED = TREATMENT_FIXED
STACKING_TREATMENT_CONTINUOUS = TREATMENT_CONTINUOUS
STACKING_TREATMENT_GRID = TREATMENT_GRID

StackingComponentControl = ComponentControl
StackingTransitionControl = TransitionControl
StackingRefinementSpec = StackingRefinementState