from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple


@dataclass
class StackingTransition:
    """
    One deterministic layer-to-layer transition.

    u, v:
        Fractional shift in the layer a/b basis.

    spacing_multiplier:
        Multiplier applied to stacking_distance.

        Normally this is 1.0.
    """

    name: str = "A"
    u: float = 0.0
    v: float = 0.0
    spacing_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Dict[str, Any] | None,
        *,
        default_name: str,
    ) -> "StackingTransition":
        values = dict(
            values
            or {}
        )

        return cls(
            name=str(
                values.get(
                    "name",
                    default_name,
                )
                or default_name
            ).strip()
            or default_name,
            u=float(
                values.get(
                    "u",
                    0.0,
                )
                or 0.0
            ),
            v=float(
                values.get(
                    "v",
                    0.0,
                )
                or 0.0
            ),
            spacing_multiplier=float(
                values.get(
                    "spacing_multiplier",
                    1.0,
                )
                or 1.0
            ),
        )


@dataclass
class StackingFaultSpec:
    """
    Deterministic explicit layer-stacking specification.

    Version 1 intentionally supports:

        - one layer motif from the CIF;
        - two possible transitions A and B;
        - explicit deterministic transition sequence;
        - optional cyclic sequence repetition;
        - discrete scan over number of layers;
        - continuous stacking_distance in Å.

    Probability/Markov stacking is deliberately not included yet.
    """

    enabled: bool = False
    model: str = "explicit_sequence"

    # Layer stacking is initially defined along the loaded CIF c direction.
    # This will later be generalized to arbitrary [h k l].
    stack_h: int = 0
    stack_k: int = 0
    stack_l: int = 1

    # Continuous interlayer separation in Å.
    stacking_distance: float = 5.0
    refine_stacking_distance: bool = False
    stacking_distance_min: float = 0.1
    stacking_distance_max: float = 100.0

    # Current finite stack.
    n_layers: int = 1

    # Discrete layer-count scan.
    scan_layers: bool = False
    layer_min: int = 1
    layer_max: int = 10
    layer_step: int = 1

    transition_a: StackingTransition = field(
        default_factory=lambda: StackingTransition(
            name="A",
            u=0.0,
            v=0.0,
            spacing_multiplier=1.0,
        )
    )

    transition_b: StackingTransition = field(
        default_factory=lambda: StackingTransition(
            name="B",
            u=0.0,
            v=0.0,
            spacing_multiplier=1.0,
        )
    )

    # Transition labels, not layer labels.
    # A stack with N layers uses N - 1 transitions.
    sequence: List[str] = field(
        default_factory=lambda: ["A"]
    )

    repeat_sequence: bool = True

    # GUI-only options.
    show_transition_arrows: bool = True
    color_layers: bool = True
    show_possible_transitions: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)

        out["sequence"] = " ".join(
            self.normalized_sequence()
        )

        return out

    @staticmethod
    def _to_bool(
        value: Any,
        default: bool = False,
    ) -> bool:
        if value is None:
            return bool(default)

        if isinstance(
            value,
            bool,
        ):
            return bool(value)

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return bool(
                int(value)
            )

        text = str(
            value
        ).strip().lower()

        if text in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        ):
            return True

        if text in (
            "0",
            "false",
            "no",
            "n",
            "off",
        ):
            return False

        return bool(default)

    @staticmethod
    def parse_sequence(
        value: Any,
    ) -> List[str]:
        """
        Parse a sequence entered as:

            A A B A
            A,A,B,A
            a b a b

        Only A and B are retained.
        """
        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raw_values = value
        else:
            raw_values = str(
                value
                or ""
            ).replace(
                ",",
                " ",
            ).split()

        out = []

        for item in raw_values:
            label = str(
                item
            ).strip().upper()

            if label in (
                "A",
                "B",
            ):
                out.append(
                    label
                )

        return out

    @classmethod
    def from_dict(
        cls,
        values: Dict[str, Any] | None,
    ) -> "StackingFaultSpec":
        values = dict(
            values
            or {}
        )

        transition_a = StackingTransition.from_dict(
            {
                "name": "A",
                "u": values.get(
                    "transition_a_u",
                    0.0,
                ),
                "v": values.get(
                    "transition_a_v",
                    0.0,
                ),
                "spacing_multiplier": values.get(
                    "transition_a_spacing_multiplier",
                    1.0,
                ),
            },
            default_name="A",
        )

        transition_b = StackingTransition.from_dict(
            {
                "name": "B",
                "u": values.get(
                    "transition_b_u",
                    0.0,
                ),
                "v": values.get(
                    "transition_b_v",
                    0.0,
                ),
                "spacing_multiplier": values.get(
                    "transition_b_spacing_multiplier",
                    1.0,
                ),
            },
            default_name="B",
        )

        sequence = cls.parse_sequence(
            values.get(
                "sequence",
                "A",
            )
        )

        if not sequence:
            sequence = ["A"]

        layer_step = max(
            1,
            int(
                float(
                    values.get(
                        "layer_step",
                        1,
                    )
                    or 1
                )
            ),
        )

        layer_min = max(
            1,
            int(
                float(
                    values.get(
                        "layer_min",
                        1,
                    )
                    or 1
                )
            ),
        )

        layer_max = max(
            layer_min,
            int(
                float(
                    values.get(
                        "layer_max",
                        10,
                    )
                    or 10
                )
            ),
        )

        return cls(
            enabled=cls._to_bool(
                values.get(
                    "enabled",
                    False,
                )
            ),
            model=str(
                values.get(
                    "model",
                    "explicit_sequence",
                )
                or "explicit_sequence"
            ).strip().lower(),
            stack_h=int(
                float(
                    values.get(
                        "stack_h",
                        0,
                    )
                    or 0
                )
            ),
            stack_k=int(
                float(
                    values.get(
                        "stack_k",
                        0,
                    )
                    or 0
                )
            ),
            stack_l=int(
                float(
                    values.get(
                        "stack_l",
                        1,
                    )
                    or 1
                )
            ),
            stacking_distance=float(
                values.get(
                    "stacking_distance",
                    5.0,
                )
                or 5.0
            ),
            refine_stacking_distance=cls._to_bool(
                values.get(
                    "refine_stacking_distance",
                    False,
                )
            ),
            stacking_distance_min=float(
                values.get(
                    "stacking_distance_min",
                    0.1,
                )
                or 0.1
            ),
            stacking_distance_max=float(
                values.get(
                    "stacking_distance_max",
                    100.0,
                )
                or 100.0
            ),
            n_layers=max(
                1,
                int(
                    float(
                        values.get(
                            "n_layers",
                            1,
                        )
                        or 1
                    )
                ),
            ),
            scan_layers=cls._to_bool(
                values.get(
                    "scan_layers",
                    False,
                )
            ),
            layer_min=layer_min,
            layer_max=layer_max,
            layer_step=layer_step,
            transition_a=transition_a,
            transition_b=transition_b,
            sequence=sequence,
            repeat_sequence=cls._to_bool(
                values.get(
                    "repeat_sequence",
                    True,
                ),
                True,
            ),
            show_transition_arrows=cls._to_bool(
                values.get(
                    "show_transition_arrows",
                    True,
                ),
                True,
            ),
            color_layers=cls._to_bool(
                values.get(
                    "color_layers",
                    True,
                ),
                True,
            ),
            show_possible_transitions=cls._to_bool(
                values.get(
                    "show_possible_transitions",
                    False,
                ),
                False,
            ),
        )

    def normalized_sequence(
        self,
    ) -> List[str]:
        out = []

        for label in self.sequence:
            value = str(
                label
            ).strip().upper()

            if value in (
                "A",
                "B",
            ):
                out.append(
                    value
                )

        return out or ["A"]

    def transition_for_label(
        self,
        label: str,
    ) -> StackingTransition:
        label = str(
            label
        ).strip().upper()

        if label == "B":
            return self.transition_b

        return self.transition_a

    def transitions_for_layers(
        self,
        n_layers: int | None = None,
    ) -> List[StackingTransition]:
        """
        Return the N-1 transition objects needed for N layers.
        """
        if n_layers is None:
            n_layers = self.n_layers

        n_layers = max(
            1,
            int(
                n_layers
            ),
        )

        n_transitions = max(
            0,
            n_layers - 1,
        )

        if n_transitions == 0:
            return []

        labels = self.normalized_sequence()

        out = []

        for index in range(
            n_transitions
        ):
            if index < len(labels):
                label = labels[index]

            elif self.repeat_sequence:
                label = labels[
                    index
                    % len(
                        labels
                    )
                ]

            else:
                label = labels[-1]

            out.append(
                self.transition_for_label(
                    label
                )
            )

        return out

    def layer_values_for_scan(
        self,
    ) -> List[int]:
        if not self.scan_layers:
            return [
                max(
                    1,
                    int(
                        self.n_layers
                    ),
                )
            ]

        return list(
            range(
                int(
                    self.layer_min
                ),
                int(
                    self.layer_max
                ) + 1,
                max(
                    1,
                    int(
                        self.layer_step
                    ),
                ),
            )
        )