from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

from typing import (
    Any,
    Dict,
    Mapping,
)


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
        (
            int,
            float,
        ),
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
        "y",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
        "n",
    }:
        return False

    return bool(
        default
    )


@dataclass
class StochasticEnsembleSpec:
    """
    Settings controlling how frozen stochastic realizations are evaluated.

    use_stacking_as_bulk=False
        Retain the existing atomistic finite-stacking implementation.

    use_stacking_as_bulk=True
        Treat every frozen realization as a periodic bulk approximant and
        stream realization models sequentially.
    """

    use_stacking_as_bulk: bool = False

    realization_period_layers: int = 64

    require_cyclic_paths: bool = True
    regenerate_invalid_cycles: bool = True

    realization_cache_size: int = 1

    export_realization_cifs: bool = True

    realization_directory: str = (
        "stacking_realizations"
    )

    skip_hkl_when_unused: bool = True

    @classmethod
    def from_dict(
        cls,
        values: Mapping[
            str,
            Any,
        ]
        | None,
    ) -> "StochasticEnsembleSpec":
        values = dict(
            values
            or {}
        )

        return cls(
            use_stacking_as_bulk=_as_bool(
                values.get(
                    "use_stacking_as_bulk",
                    False,
                ),
                False,
            ),
            realization_period_layers=max(
                2,
                int(
                    values.get(
                        "realization_period_layers",
                        64,
                    )
                ),
            ),
            require_cyclic_paths=_as_bool(
                values.get(
                    "require_cyclic_paths",
                    True,
                ),
                True,
            ),
            regenerate_invalid_cycles=_as_bool(
                values.get(
                    "regenerate_invalid_cycles",
                    True,
                ),
                True,
            ),
            realization_cache_size=max(
                1,
                int(
                    values.get(
                        "realization_cache_size",
                        1,
                    )
                ),
            ),
            export_realization_cifs=_as_bool(
                values.get(
                    "export_realization_cifs",
                    True,
                ),
                True,
            ),
            realization_directory=str(
                values.get(
                    "realization_directory",
                    "stacking_realizations",
                )
                or "stacking_realizations"
            ).strip(),
            skip_hkl_when_unused=_as_bool(
                values.get(
                    "skip_hkl_when_unused",
                    True,
                ),
                True,
            ),
        )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        return asdict(
            self
        )


def stochastic_ensemble_spec_from_config(
    config: Mapping[
        str,
        Any,
    ]
    | None,
) -> StochasticEnsembleSpec:
    config = dict(
        config
        or {}
    )

    return StochasticEnsembleSpec.from_dict(
        config.get(
            "stochastic_ensemble",
            {},
        )
    )