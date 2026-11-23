from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pymatgen.core import Lattice
from pymatgen.core import Structure

from pdf_fitting.models.compiled_stacked_geometry import (
    compile_stacked_geometry,
)

from pdf_fitting.models.compiled_stacked_pairs import (
    compile_stacked_pair_topology,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.gpu.backend import (
    cuda_available,
)

from pdf_fitting.models.gpu.dynamic_stacked_pdf import (
    DynamicStackedCudaPdfBackend,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
    SequenceLayer,
    layer_basis,
)


pytestmark = pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA/CuPy is not available.",
)


def _structure() -> Structure:
    lattice = Lattice.from_parameters(
        5.0,
        5.0,
        8.0,
        90.0,
        90.0,
        90.0,
    )

    return Structure(
        lattice,
        [
            "C",
            "C",
            "N",
        ],
        [
            (0.00, 0.00, 0.00),
            (0.25, 0.25, 0.00),
            (0.50, 0.50, 0.05),
        ],
        coords_are_cartesian=False,
        to_unit_cell=False,
    )


def _shape() -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec(
        shape_type="cylinder",
        search_mode="predefined",
        axis1_cells=3.0,
        axis2_cells=3.0,
        axis3_cells=4.0,
        diameter_cells=3.0,
        height_cells=4.0,
        axis_h=0,
        axis_k=0,
        axis_l=1,
        base1_h=1,
        base1_k=0,
        base1_l=0,
        base2_h=0,
        base2_k=1,
        base2_l=0,
    )


def _stack() -> LayerStackSpec:
    return LayerStackSpec(
        enabled=True,
        definition_mode="repeating_motif",
        vector_mode="absolute",
        spacing_mode="independent",
        nominal_spacing=3.20,
        repeat_period=1.0,
        repeat_shift_u=0.0,
        repeat_shift_v=0.0,
        sequence=[
            SequenceLayer(
                enabled=True,
                label="A",
                shift_u=0.0,
                shift_v=0.0,
                stack_coordinate=0.0,
                normal_delta=0.0,
            ),
        ],
    ).normalized()


def _reference_parameters() -> dict:
    return {
        "a": 5.00,
        "b": 5.00,
        "c": 8.00,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "stacking_distance": 3.20,
    }


def _trial_parameters() -> dict:
    return {
        "a": 5.07,
        "b": 4.96,
        "c": 7.91,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 89.40,
        "stacking_distance": 3.35,
    }


def _pair_ids_and_offsets(
    topology,
):
    pair_values = sorted(
        {
            (
                int(alpha_id),
                int(beta_id),
            )
            for alpha_id, beta_id in zip(
                topology.shell_alpha_id,
                topology.shell_beta_id,
            )
        }
    )

    pair_index = {
        pair: index
        for index, pair in enumerate(
            pair_values
        )
    }

    shell_pair_id = np.asarray(
        [
            pair_index[
                (
                    int(alpha_id),
                    int(beta_id),
                )
            ]
            for alpha_id, beta_id in zip(
                topology.shell_alpha_id,
                topology.shell_beta_id,
            )
        ],
        dtype=np.int32,
    )

    shell_order = np.argsort(
        shell_pair_id,
        kind="mergesort",
    )

    shell_pair_id = shell_pair_id[
        shell_order
    ]

    pair_counts = np.bincount(
        shell_pair_id,
        minlength=len(
            pair_values
        ),
    ).astype(
        np.int32,
    )

    pair_offsets = np.zeros(
        len(
            pair_values
        )
        + 1,
        dtype=np.int32,
    )

    pair_offsets[
        1:
    ] = np.cumsum(
        pair_counts
    )

    pair_weights = np.full(
        len(
            pair_values
        ),
        1.0
        / max(
            len(
                pair_values
            ),
            1,
        ),
        dtype=np.float32,
    )

    return (
        shell_order,
        shell_pair_id,
        pair_offsets,
        pair_weights,
    )


def _cpu_reference_gpu_algorithm(
    *,
    topology,
    cpu_evaluated,
    shell_order: np.ndarray,
    shell_pair_id: np.ndarray,
    pair_offsets: np.ndarray,
    pair_weights: np.ndarray,
    r_grid: np.ndarray,
    correlation: np.ndarray,
    biso_by_species: np.ndarray,
    density_by_species: np.ndarray,
    delta_g: float,
    delta_broad: float,
    scale: float,
    rho0: float,
):
    r_grid = np.asarray(
        r_grid,
        dtype=np.float64,
    )

    shell_order = np.asarray(
        shell_order,
        dtype=np.int64,
    )

    shell_pair_id = np.asarray(
        shell_pair_id,
        dtype=np.int32,
    )

    pair_offsets = np.asarray(
        pair_offsets,
        dtype=np.int32,
    )

    pair_weights = np.asarray(
        pair_weights,
        dtype=np.float64,
    )

    r_ij = np.asarray(
        cpu_evaluated.shell_distance,
        dtype=np.float64,
    )[
        shell_order
    ]

    active = np.asarray(
        cpu_evaluated.shell_active,
        dtype=bool,
    )[
        shell_order
    ]

    shell_mult_eff = np.asarray(
        cpu_evaluated.shell_mult_eff,
        dtype=np.float64,
    )[
        shell_order
    ]

    shell_bulk_mult_eff = np.asarray(
        cpu_evaluated.shell_bulk_mult_eff,
        dtype=np.float64,
    )[
        shell_order
    ]

    shell_gamma = np.asarray(
        cpu_evaluated.shell_gamma,
        dtype=np.float64,
    )[
        shell_order
    ]

    shell_alpha_id = np.asarray(
        topology.shell_alpha_id,
        dtype=np.int32,
    )[
        shell_order
    ]

    shell_beta_id = np.asarray(
        topology.shell_beta_id,
        dtype=np.int32,
    )[
        shell_order
    ]

    shell_biso = (
        0.5
        * (
            biso_by_species[
                shell_alpha_id
            ]
            + biso_by_species[
                shell_beta_id
            ]
        )
    )

    sigma_squared = (
        (
            1.0
            / (
                4.0
                * np.pi
                * np.pi
            )
        )
        * shell_biso
        * correlation
        + (
            float(
                delta_g
            )
            * r_ij
        ) ** 2
        + (
            float(
                delta_broad
            )
            * r_ij
        ) ** 2
    )

    sigma = np.sqrt(
        np.maximum(
            sigma_squared,
            1.0e-24,
        )
    )

    beta_density = np.maximum(
        density_by_species[
            shell_beta_id
        ],
        1.0e-30,
    )

    amplitude = (
        shell_mult_eff
        / (
            4.0
            * np.pi
            * np.maximum(
                r_ij
                * r_ij,
                1.0e-24,
            )
            * beta_density
        )
    )

    pair_count = int(
        pair_weights.size
    )

    g_by_pair = np.zeros(
        (
            pair_count,
            r_grid.size,
        ),
        dtype=np.float64,
    )

    inv_sqrt_2pi = 1.0 / math.sqrt(
        2.0
        * np.pi
    )

    for shell_index in range(
        r_ij.size
    ):
        if not active[
            shell_index
        ]:
            continue

        pair_id = int(
            shell_pair_id[
                shell_index
            ]
        )

        width = float(
            sigma[
                shell_index
            ]
        )

        if width <= 0.0:
            continue

        amp = float(
            amplitude[
                shell_index
            ]
        )

        if amp == 0.0:
            continue

        center = float(
            r_ij[
                shell_index
            ]
        )

        lower = center - 5.0 * width
        upper = center + 5.0 * width

        first = int(
            np.searchsorted(
                r_grid,
                lower,
                side="left",
            )
        )

        last = int(
            np.searchsorted(
                r_grid,
                upper,
                side="left",
            )
        )

        if last <= first:
            continue

        x = (
            r_grid[
                first:last
            ]
            - center
        ) / width

        g_by_pair[
            pair_id,
            first:last,
        ] += (
            amp
            * inv_sqrt_2pi
            / width
            * np.exp(
                -0.5
                * x
                * x
            )
        )

    g_total = (
        pair_weights[
            :,
            None,
        ]
        * g_by_pair
    ).sum(
        axis=0
    )

    gamma_numerator = np.zeros(
        (
            pair_count,
            r_grid.size,
        ),
        dtype=np.float64,
    )

    gamma_denominator = np.zeros(
        (
            pair_count,
            r_grid.size,
        ),
        dtype=np.float64,
    )

    for shell_index in range(
        r_ij.size
    ):
        if not active[
            shell_index
        ]:
            continue

        if r_ij[
            shell_index
        ] <= 0.0:
            continue

        weight = float(
            shell_bulk_mult_eff[
                shell_index
            ]
        )

        if weight <= 0.0:
            continue

        if (
            r_ij[
                shell_index
            ]
            < r_grid[
                0
            ]
            or r_ij[
                shell_index
            ]
            > r_grid[
                -1
            ]
        ):
            continue

        right = int(
            np.searchsorted(
                r_grid,
                r_ij[
                    shell_index
                ],
                side="left",
            )
        )

        right = min(
            max(
                right,
                0,
            ),
            r_grid.size - 1,
        )

        left = max(
            right - 1,
            0,
        )

        distance_left = abs(
            float(
                r_ij[
                    shell_index
                ]
            )
            - float(
                r_grid[
                    left
                ]
            )
        )

        distance_right = abs(
            float(
                r_ij[
                    shell_index
                ]
            )
            - float(
                r_grid[
                    right
                ]
            )
        )

        grid_index = (
            left
            if distance_left <= distance_right
            else right
        )

        pair_id = int(
            shell_pair_id[
                shell_index
            ]
        )

        gamma_numerator[
            pair_id,
            grid_index,
        ] += (
            float(
                shell_gamma[
                    shell_index
                ]
            )
            * weight
        )

        gamma_denominator[
            pair_id,
            grid_index,
        ] += weight

    gamma_by_pair = np.zeros(
        (
            pair_count,
            r_grid.size,
        ),
        dtype=np.float64,
    )

    for pair_id in range(
        pair_count
    ):
        mask = (
            gamma_denominator[
                pair_id
            ]
            > 0.0
        )

        if not np.any(
            mask
        ):
            continue

        known_r = r_grid[
            mask
        ]

        known_gamma = (
            gamma_numerator[
                pair_id,
                mask,
            ]
            / gamma_denominator[
                pair_id,
                mask,
            ]
        )

        if known_r[
            0
        ] > 0.0:
            known_r = np.concatenate(
                (
                    np.asarray(
                        [
                            0.0,
                        ],
                        dtype=np.float64,
                    ),
                    known_r,
                )
            )

            known_gamma = np.concatenate(
                (
                    np.asarray(
                        [
                            1.0,
                        ],
                        dtype=np.float64,
                    ),
                    known_gamma,
                )
            )

        else:
            known_gamma[
                0
            ] = 1.0

        gamma_by_pair[
            pair_id,
            :
        ] = np.interp(
            r_grid,
            known_r,
            known_gamma,
            left=1.0,
            right=0.0,
        )

    gamma_base = (
        pair_weights[
            :,
            None,
        ]
        * gamma_by_pair
    ).sum(
        axis=0
    )

    g_pdf = (
        float(
            scale
        )
        * float(
            4.0
            * np.pi
            * rho0
        )
        * r_grid
        * (
            g_total
            - gamma_base
        )
    )

    return {
        "G_r": g_pdf,
        "g_r": g_total,
        "gamma_base": gamma_base,
    }

def _cpu_sphere_gamma(
    r: np.ndarray,
    *,
    diameter: float,
    diameter_std: float,
) -> np.ndarray:
    r = np.asarray(
        r,
        dtype=np.float64,
    )

    diameter = float(
        diameter
    )

    diameter_std = float(
        diameter_std
    )

    if diameter <= 0.0:
        return np.ones_like(
            r,
            dtype=np.float64,
        )

    if diameter_std < 1.0e-8:
        gamma = np.zeros_like(
            r,
            dtype=np.float64,
        )

        inside = r < diameter

        ratio = (
            r[
                inside
            ]
            / diameter
        )

        gamma[
            inside
        ] = (
            1.0
            - 1.5
            * ratio
            + 0.5
            * ratio
            * ratio
            * ratio
        )

        return gamma

    sigma_squared = np.log(
        (
            diameter_std
            * diameter_std
        )
        / (
            diameter
            * diameter
        )
        + 1.0
    )

    sigma = float(
        np.sqrt(
            sigma_squared
        )
    )

    mu = float(
        np.log(
            diameter
        )
        - 0.5
        * sigma_squared
    )

    from scipy.special import erfc

    log_r = np.log(
        np.maximum(
            r,
            1.0e-8,
        )
    )

    a_term = erfc(
        (
            -mu
            - 3.0
            * sigma_squared
            + log_r
        )
        / (
            np.sqrt(
                2.0
            )
            * sigma
        )
    )

    b_term = erfc(
        (
            -mu
            + log_r
        )
        / (
            np.sqrt(
                2.0
            )
            * sigma
        )
    )

    c_term = erfc(
        (
            -mu
            - 2.0
            * sigma_squared
            + log_r
        )
        / (
            np.sqrt(
                2.0
            )
            * sigma
        )
    )

    exp_a = np.exp(
        -3.0
        * mu
        - 4.5
        * sigma_squared
    )

    exp_b = np.exp(
        -mu
        - 2.5
        * sigma_squared
    )

    return (
        0.5
        * a_term
        + 0.25
        * r
        * r
        * r
        * b_term
        * exp_a
        - 0.75
        * r
        * c_term
        * exp_b
    )

def _cpu_pair_gamma_grid(
    *,
    r_grid: np.ndarray,
    r_ij: np.ndarray,
    shell_gamma: np.ndarray,
    shell_weight: np.ndarray,
    shell_active: np.ndarray,
    shell_pair_id: np.ndarray,
    pair_count: int,
) -> np.ndarray:
    r_grid = np.asarray(
        r_grid,
        dtype=np.float64,
    )

    r_ij = np.asarray(
        r_ij,
        dtype=np.float64,
    )

    shell_gamma = np.asarray(
        shell_gamma,
        dtype=np.float64,
    )

    shell_weight = np.asarray(
        shell_weight,
        dtype=np.float64,
    )

    shell_active = np.asarray(
        shell_active,
        dtype=bool,
    )

    shell_pair_id = np.asarray(
        shell_pair_id,
        dtype=np.int32,
    )

    pair_count = int(
        pair_count
    )

    gamma_by_pair = np.zeros(
        (
            pair_count,
            r_grid.size,
        ),
        dtype=np.float64,
    )

    for pair_id in range(
        pair_count
    ):
        valid = (
            shell_active
            & (
                shell_pair_id
                == pair_id
            )
            & np.isfinite(
                r_ij
            )
            & (
                r_ij
                > 0.0
            )
            & (
                r_ij
                >= r_grid[
                    0
                ]
            )
            & (
                r_ij
                <= r_grid[
                    -1
                ]
            )
            & np.isfinite(
                shell_gamma
            )
            & np.isfinite(
                shell_weight
            )
            & (
                shell_weight
                > 0.0
            )
        )

        if not np.any(
            valid
        ):
            continue

        distance = r_ij[
            valid
        ]

        gamma = shell_gamma[
            valid
        ]

        weight = shell_weight[
            valid
        ]

        numerator = np.zeros(
            r_grid.size,
            dtype=np.float64,
        )

        denominator = np.zeros(
            r_grid.size,
            dtype=np.float64,
        )

        for distance_value, gamma_value, weight_value in zip(
            distance,
            gamma,
            weight,
        ):
            right = int(
                np.searchsorted(
                    r_grid,
                    distance_value,
                    side="left",
                )
            )

            right = min(
                max(
                    right,
                    0,
                ),
                r_grid.size - 1,
            )

            left = max(
                right - 1,
                0,
            )

            if abs(
                distance_value
                - r_grid[
                    left
                ]
            ) <= abs(
                distance_value
                - r_grid[
                    right
                ]
            ):
                grid_index = left
            else:
                grid_index = right

            numerator[
                grid_index
            ] += (
                gamma_value
                * weight_value
            )

            denominator[
                grid_index
            ] += weight_value

        populated = denominator > 0.0

        if not np.any(
            populated
        ):
            continue

        known_r = r_grid[
            populated
        ]

        known_gamma = (
            numerator[
                populated
            ]
            / denominator[
                populated
            ]
        )

        if known_r[
            0
        ] > 0.0:
            known_r = np.concatenate(
                (
                    np.asarray(
                        [
                            0.0,
                        ],
                        dtype=np.float64,
                    ),
                    known_r,
                )
            )

            known_gamma = np.concatenate(
                (
                    np.asarray(
                        [
                            1.0,
                        ],
                        dtype=np.float64,
                    ),
                    known_gamma,
                )
            )

        else:
            known_gamma[
                0
            ] = 1.0

        gamma_by_pair[
            pair_id,
            :
        ] = np.interp(
            r_grid,
            known_r,
            known_gamma,
            left=1.0,
            right=0.0,
        )

    return gamma_by_pair


def _cpu_cylinder_shell_gamma(
    *,
    cart_vectors: np.ndarray,
    r_ij: np.ndarray,
    shell_active: np.ndarray,
    diameter: float,
    thickness: float,
    thickness_std: float,
    axis_cart: np.ndarray,
) -> np.ndarray:
    """
    CPU reference for DynamicStackedCudaPdfBackend._cylinder_shell_gamma().
    """
    from scipy.special import erf
    from scipy.special import erfc

    cart_vectors = np.asarray(
        cart_vectors,
        dtype=np.float64,
    )

    r_ij = np.asarray(
        r_ij,
        dtype=np.float64,
    )

    shell_active = np.asarray(
        shell_active,
        dtype=bool,
    )

    diameter = float(
        diameter
    )

    thickness = float(
        thickness
    )

    thickness_std = float(
        thickness_std
    )

    axis_cart = np.asarray(
        axis_cart,
        dtype=np.float64,
    )

    if (
        diameter <= 0.0
        or thickness <= 0.0
    ):
        return np.ones_like(
            r_ij,
            dtype=np.float64,
        )

    axis_norm = float(
        np.linalg.norm(
            axis_cart
        )
    )

    if axis_norm <= 1.0e-12:
        axis_cart = np.asarray(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        axis_norm = 1.0

    axis_cart = (
        axis_cart
        / axis_norm
    )

    projected = np.abs(
        cart_vectors
        @ axis_cart
    )

    radial = np.sqrt(
        np.maximum(
            r_ij
            * r_ij
            - projected
            * projected,
            0.0,
        )
    )

    gamma = np.zeros_like(
        r_ij,
        dtype=np.float64,
    )

    valid = (
        shell_active
        & np.isfinite(
            r_ij
        )
        & (
            r_ij
            > 0.0
        )
        & (
            radial
            < diameter
        )
    )

    if not np.any(
        valid
    ):
        return gamma

    radial_valid = radial[
        valid
    ]

    projected_valid = projected[
        valid
    ]

    x = np.clip(
        radial_valid
        / diameter,
        0.0,
        1.0,
    )

    radial_factor = (
        2.0
        / np.pi
        * (
            np.arccos(
                x
            )
            - x
            * np.sqrt(
                np.maximum(
                    1.0
                    - x
                    * x,
                    0.0,
                )
            )
        )
    )

    if thickness_std <= 1.0e-12:
        axial_factor = np.maximum(
            1.0
            - projected_valid
            / thickness,
            0.0,
        )

        gamma[
            valid
        ] = (
            radial_factor
            * axial_factor
        )

        return gamma

    sigma_squared = np.log(
        1.0
        + (
            thickness_std
            * thickness_std
        )
        / (
            thickness
            * thickness
        )
    )

    sigma = float(
        np.sqrt(
            sigma_squared
        )
    )

    mu = float(
        np.log(
            thickness
        )
        - 0.5
        * sigma_squared
    )

    z = projected_valid

    z_zero = z <= 1.0e-30

    z_safe = np.maximum(
        z,
        1.0e-30,
    )

    log_z = np.log(
        z_safe
    )

    sqrt_two_sigma = (
        np.sqrt(
            2.0
        )
        * sigma
    )

    bracket = (
        np.exp(
            mu
            + 0.5
            * sigma_squared
        )
        * (
            1.0
            + erf(
                (
                    mu
                    + sigma_squared
                    - log_z
                )
                / sqrt_two_sigma
            )
        )
        + z
        * (
            erfc(
                (
                    mu
                    - log_z
                )
                / sqrt_two_sigma
            )
            - 2.0
        )
    )

    bracket[
        z_zero
    ] = (
        2.0
        * np.exp(
            mu
            + 0.5
            * sigma_squared
        )
    )

    radial_bracket = (
        0.5
        * np.pi
        * radial_factor
        * diameter
    )

    gamma_valid = (
        np.exp(
            -mu
            - 0.5
            * sigma_squared
        )
        / (
            diameter
            * np.pi
        )
        * bracket
        * radial_bracket
    )

    gamma[
        valid
    ] = np.clip(
        gamma_valid,
        0.0,
        1.0,
    )

    return gamma


def _cpu_saxs_cylinder_gamma(
    *,
    cart_vectors: np.ndarray,
    r_ij: np.ndarray,
    shell_active: np.ndarray,
    shell_pair_id: np.ndarray,
    shell_bulk_mult_eff: np.ndarray,
    pair_weights: np.ndarray,
    r_grid: np.ndarray,
    diameter: float,
    diameter_std: float,
    height: float,
    height_std: float,
    axis_cart: np.ndarray,
) -> np.ndarray:
    diameter = float(
        diameter
    )

    diameter_std = float(
        diameter_std
    )

    height = float(
        height
    )

    height_std = float(
        height_std
    )

    pair_weights = np.asarray(
        pair_weights,
        dtype=np.float64,
    )

    if (
        diameter <= 0.0
        or height <= 0.0
    ):
        return np.zeros_like(
            r_grid,
            dtype=np.float64,
        )

    def weighted_gamma(
        shell_gamma: np.ndarray,
    ) -> np.ndarray:
        gamma_by_pair = _cpu_pair_gamma_grid(
            r_grid=r_grid,
            r_ij=r_ij,
            shell_gamma=shell_gamma,
            shell_weight=shell_bulk_mult_eff,
            shell_active=shell_active,
            shell_pair_id=shell_pair_id,
            pair_count=pair_weights.size,
        )

        return (
            pair_weights[
                :,
                None,
            ]
            * gamma_by_pair
        ).sum(
            axis=0
        )

    if diameter_std <= 1.0e-12:
        shell_gamma = _cpu_cylinder_shell_gamma(
            cart_vectors=cart_vectors,
            r_ij=r_ij,
            shell_active=shell_active,
            diameter=diameter,
            thickness=height,
            thickness_std=height_std,
            axis_cart=axis_cart,
        )

        return weighted_gamma(
            shell_gamma
        )

    sigma_squared = np.log(
        1.0
        + (
            diameter_std
            * diameter_std
        )
        / (
            diameter
            * diameter
        )
    )

    sigma = float(
        np.sqrt(
            sigma_squared
        )
    )

    mu = float(
        np.log(
            diameter
        )
        - 0.5
        * sigma_squared
    )

    diameter_minimum = max(
        1.0e-6,
        diameter
        * 0.05,
    )

    diameter_maximum = (
        diameter
        + 4.0
        * diameter_std
    )

    if diameter_maximum <= diameter_minimum:
        diameter_maximum = (
            diameter
            * 3.0
        )

    diameter_grid = np.linspace(
        diameter_minimum,
        diameter_maximum,
        15,
        dtype=np.float64,
    )

    probability = (
        1.0
        / (
            diameter_grid
            * sigma
            * np.sqrt(
                2.0
                * np.pi
            )
        )
        * np.exp(
            -0.5
            * (
                (
                    np.log(
                        diameter_grid
                    )
                    - mu
                )
                / sigma
            ) ** 2
        )
    )

    weights = probability / max(
        float(
            np.sum(
                probability
            )
        ),
        1.0e-30,
    )

    total = np.zeros_like(
        r_grid,
        dtype=np.float64,
    )

    for diameter_value, weight in zip(
        diameter_grid,
        weights,
    ):
        shell_gamma = _cpu_cylinder_shell_gamma(
            cart_vectors=cart_vectors,
            r_ij=r_ij,
            shell_active=shell_active,
            diameter=float(
                diameter_value
            ),
            thickness=height,
            thickness_std=height_std,
            axis_cart=axis_cart,
        )

        total += (
            float(
                weight
            )
            * weighted_gamma(
                shell_gamma
            )
        )

    return total


def test_dynamic_stacked_cuda_core_matches_numpy_reference() -> None:
    structure = _structure()
    shape = _shape()
    stack = _stack()

    reference = _reference_parameters()
    parameters = _trial_parameters()

    geometry = compile_stacked_geometry(
        structure=structure,
        shape_spec=shape,
        stack_spec=stack,
        reference_parameters=reference,
    )

    topology = compile_stacked_pair_topology(
        geometry=geometry,
        pair_cutoff=8.0,
        topology_margin=1.0,
        reference_parameters=reference,
        shell_backend="cpu",
    )

    cpu_evaluated = topology.evaluate(
        parameters
    )

    lattice_matrix = np.asarray(
        cpu_evaluated.geometry.parent_lattice_matrix,
        dtype=float,
    )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    (
        shell_order,
        shell_pair_id,
        pair_offsets,
        pair_weights,
    ) = _pair_ids_and_offsets(
        topology
    )

    r_grid = np.linspace(
        0.0,
        8.0,
        801,
        dtype=np.float32,
    )

    correlation = np.linspace(
        0.75,
        1.00,
        topology.shell_mult.size,
        dtype=np.float32,
    )[
        shell_order
    ]

    biso_by_species = np.asarray(
        [
            0.20,
            0.45,
        ],
        dtype=np.float32,
    )

    density_by_species = np.asarray(
        [
            0.70,
            1.10,
        ],
        dtype=np.float32,
    )

    delta_g = 0.002
    delta_broad = 0.001
    scale = 0.85
    rho0 = 0.31

    gpu = DynamicStackedCudaPdfBackend(
        shell_delta_fractional=(
            topology.shell_delta_fractional[
                shell_order
            ]
        ),
        shell_delta_a=(
            topology.shell_delta_a[
                shell_order
            ]
        ),
        shell_delta_b=(
            topology.shell_delta_b[
                shell_order
            ]
        ),
        shell_delta_s=(
            topology.shell_delta_s[
                shell_order
            ]
        ),
        shell_delta_normal=(
            topology.shell_delta_normal[
                shell_order
            ]
        ),
        shell_pair_id=shell_pair_id,
        shell_alpha_id=(
            topology.shell_alpha_id[
                shell_order
            ]
        ),
        shell_beta_id=(
            topology.shell_beta_id[
                shell_order
            ]
        ),
        shell_mult=(
            topology.shell_mult[
                shell_order
            ]
        ),
        shell_bulk_mult=(
            topology.shell_bulk_mult[
                shell_order
            ]
        ),
        finite_species_counts=(
            topology.finite_species_counts
        ),
        bulk_species_counts=(
            topology.bulk_species_counts
        ),
        pair_weights=pair_weights,
        pair_offsets=pair_offsets,
        pair_cutoff=float(
            topology.pair_cutoff
        ),
        device_id=0,
    )

    gpu_result = gpu.evaluate(
        lattice_matrix=lattice_matrix,
        vector_a=vector_a,
        vector_b=vector_b,
        stack_normal=normal,
        spacing=float(
            cpu_evaluated.geometry.spacing
        ),
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        r_grid=r_grid,
        scale=scale,
        rho0=rho0,
    )

    cpu_result = _cpu_reference_gpu_algorithm(
        topology=topology,
        cpu_evaluated=cpu_evaluated,
        shell_order=shell_order,
        shell_pair_id=shell_pair_id,
        pair_offsets=pair_offsets,
        pair_weights=pair_weights,
        r_grid=r_grid,
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        scale=scale,
        rho0=rho0,
    )

    np.testing.assert_allclose(
        gpu_result[
            "g_r"
        ],
        cpu_result[
            "g_r"
        ],
        rtol=2.0e-4,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_result[
            "gamma_base"
        ],
        cpu_result[
            "gamma_base"
        ],
        rtol=2.0e-4,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        gpu_result[
            "G_r"
        ],
        cpu_result[
            "G_r"
        ],
        rtol=3.0e-4,
        atol=3.0e-5,
    )


def test_dynamic_stacked_cuda_saxs_sphere_matches_numpy() -> None:
    structure = _structure()
    shape = _shape()
    stack = _stack()

    reference = _reference_parameters()
    parameters = _trial_parameters()

    geometry = compile_stacked_geometry(
        structure=structure,
        shape_spec=shape,
        stack_spec=stack,
        reference_parameters=reference,
    )

    topology = compile_stacked_pair_topology(
        geometry=geometry,
        pair_cutoff=8.0,
        topology_margin=1.0,
        reference_parameters=reference,
        shell_backend="cpu",
    )

    cpu_evaluated = topology.evaluate(
        parameters
    )

    lattice_matrix = np.asarray(
        cpu_evaluated.geometry.parent_lattice_matrix,
        dtype=float,
    )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    (
        shell_order,
        shell_pair_id,
        pair_offsets,
        pair_weights,
    ) = _pair_ids_and_offsets(
        topology
    )

    r_grid = np.linspace(
        0.0,
        8.0,
        801,
        dtype=np.float32,
    )

    correlation = np.linspace(
        0.75,
        1.00,
        topology.shell_mult.size,
        dtype=np.float32,
    )[
        shell_order
    ]

    biso_by_species = np.asarray(
        [
            0.20,
            0.45,
        ],
        dtype=np.float32,
    )

    density_by_species = np.asarray(
        [
            0.70,
            1.10,
        ],
        dtype=np.float32,
    )

    delta_g = 0.002
    delta_broad = 0.001
    scale = 0.85
    rho0 = 0.31

    saxs_scale = 1.25
    saxs_diameter = 6.40
    saxs_diameter_std = 0.75

    gpu = DynamicStackedCudaPdfBackend(
        shell_delta_fractional=(
            topology.shell_delta_fractional[
                shell_order
            ]
        ),
        shell_delta_a=(
            topology.shell_delta_a[
                shell_order
            ]
        ),
        shell_delta_b=(
            topology.shell_delta_b[
                shell_order
            ]
        ),
        shell_delta_s=(
            topology.shell_delta_s[
                shell_order
            ]
        ),
        shell_delta_normal=(
            topology.shell_delta_normal[
                shell_order
            ]
        ),
        shell_pair_id=shell_pair_id,
        shell_alpha_id=(
            topology.shell_alpha_id[
                shell_order
            ]
        ),
        shell_beta_id=(
            topology.shell_beta_id[
                shell_order
            ]
        ),
        shell_mult=(
            topology.shell_mult[
                shell_order
            ]
        ),
        shell_bulk_mult=(
            topology.shell_bulk_mult[
                shell_order
            ]
        ),
        finite_species_counts=(
            topology.finite_species_counts
        ),
        bulk_species_counts=(
            topology.bulk_species_counts
        ),
        pair_weights=pair_weights,
        pair_offsets=pair_offsets,
        pair_cutoff=float(
            topology.pair_cutoff
        ),
        device_id=0,
    )

    gpu_result = gpu.evaluate(
        lattice_matrix=lattice_matrix,
        vector_a=vector_a,
        vector_b=vector_b,
        stack_normal=normal,
        spacing=float(
            cpu_evaluated.geometry.spacing
        ),
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        r_grid=r_grid,
        scale=scale,
        rho0=rho0,
        saxs_enabled=True,
        saxs_model="sphere",
        saxs_scale=saxs_scale,
        saxs_diameter=saxs_diameter,
        saxs_diameter_std=saxs_diameter_std,
        saxs_height=0.0,
        saxs_height_std=0.0,
        saxs_apply_qdamp=False,
        qdamp=0.0,
    )

    cpu_core = _cpu_reference_gpu_algorithm(
        topology=topology,
        cpu_evaluated=cpu_evaluated,
        shell_order=shell_order,
        shell_pair_id=shell_pair_id,
        pair_offsets=pair_offsets,
        pair_weights=pair_weights,
        r_grid=r_grid,
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        scale=scale,
        rho0=rho0,
    )

    r_float = np.asarray(
        r_grid,
        dtype=np.float64,
    )

    gamma_saxs = _cpu_sphere_gamma(
        r_float,
        diameter=saxs_diameter,
        diameter_std=saxs_diameter_std,
    )

    cpu_peak = (
        float(
            scale
        )
        * float(
            4.0
            * np.pi
            * rho0
        )
        * r_float
        * np.asarray(
            cpu_core[
                "g_r"
            ],
            dtype=np.float64,
        )
    )

    cpu_belly = (
        -float(
            scale
        )
        * float(
            saxs_scale
        )
        * float(
            4.0
            * np.pi
            * rho0
        )
        * r_float
        * gamma_saxs
    )

    cpu_saxs_result = (
        cpu_peak
        + cpu_belly
    )

    np.testing.assert_allclose(
        gpu_result[
            "gamma_saxs"
        ],
        gamma_saxs,
        rtol=3.0e-4,
        atol=3.0e-5,
    )

    np.testing.assert_allclose(
        gpu_result[
            "G_r"
        ],
        cpu_saxs_result,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

def test_dynamic_stacked_cuda_saxs_cylinder_matches_numpy() -> None:
    structure = _structure()
    shape = _shape()
    stack = _stack()

    reference = _reference_parameters()
    parameters = _trial_parameters()

    geometry = compile_stacked_geometry(
        structure=structure,
        shape_spec=shape,
        stack_spec=stack,
        reference_parameters=reference,
    )

    topology = compile_stacked_pair_topology(
        geometry=geometry,
        pair_cutoff=8.0,
        topology_margin=1.0,
        reference_parameters=reference,
        shell_backend="cpu",
    )

    cpu_evaluated = topology.evaluate(
        parameters
    )

    lattice_matrix = np.asarray(
        cpu_evaluated.geometry.parent_lattice_matrix,
        dtype=float,
    )

    vector_a, vector_b, normal = layer_basis(
        lattice_matrix
    )

    (
        shell_order,
        shell_pair_id,
        pair_offsets,
        pair_weights,
    ) = _pair_ids_and_offsets(
        topology
    )

    r_grid = np.linspace(
        0.0,
        8.0,
        801,
        dtype=np.float32,
    )

    correlation = np.linspace(
        0.75,
        1.00,
        topology.shell_mult.size,
        dtype=np.float32,
    )[
        shell_order
    ]

    biso_by_species = np.asarray(
        [
            0.20,
            0.45,
        ],
        dtype=np.float32,
    )

    density_by_species = np.asarray(
        [
            0.70,
            1.10,
        ],
        dtype=np.float32,
    )

    delta_g = 0.002
    delta_broad = 0.001
    scale = 0.85
    rho0 = 0.31

    saxs_scale = 1.35
    saxs_diameter = 6.50
    saxs_diameter_std = 0.85
    saxs_height = 3.10
    saxs_height_std = 0.25

    cylinder_axis_cart = np.asarray(
        normal,
        dtype=np.float32,
    )

    gpu = DynamicStackedCudaPdfBackend(
        shell_delta_fractional=(
            topology.shell_delta_fractional[
                shell_order
            ]
        ),
        shell_delta_a=(
            topology.shell_delta_a[
                shell_order
            ]
        ),
        shell_delta_b=(
            topology.shell_delta_b[
                shell_order
            ]
        ),
        shell_delta_s=(
            topology.shell_delta_s[
                shell_order
            ]
        ),
        shell_delta_normal=(
            topology.shell_delta_normal[
                shell_order
            ]
        ),
        shell_pair_id=shell_pair_id,
        shell_alpha_id=(
            topology.shell_alpha_id[
                shell_order
            ]
        ),
        shell_beta_id=(
            topology.shell_beta_id[
                shell_order
            ]
        ),
        shell_mult=(
            topology.shell_mult[
                shell_order
            ]
        ),
        shell_bulk_mult=(
            topology.shell_bulk_mult[
                shell_order
            ]
        ),
        finite_species_counts=(
            topology.finite_species_counts
        ),
        bulk_species_counts=(
            topology.bulk_species_counts
        ),
        pair_weights=pair_weights,
        pair_offsets=pair_offsets,
        pair_cutoff=float(
            topology.pair_cutoff
        ),
        device_id=0,
    )

    gpu_result = gpu.evaluate(
        lattice_matrix=lattice_matrix,
        vector_a=vector_a,
        vector_b=vector_b,
        stack_normal=normal,
        spacing=float(
            cpu_evaluated.geometry.spacing
        ),
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        r_grid=r_grid,
        scale=scale,
        rho0=rho0,
        saxs_enabled=True,
        saxs_model="cylinder",
        saxs_scale=saxs_scale,
        saxs_diameter=saxs_diameter,
        saxs_diameter_std=saxs_diameter_std,
        saxs_height=saxs_height,
        saxs_height_std=saxs_height_std,
        saxs_apply_qdamp=False,
        qdamp=0.0,
        cylinder_axis_cart=cylinder_axis_cart,
    )

    cpu_core = _cpu_reference_gpu_algorithm(
        topology=topology,
        cpu_evaluated=cpu_evaluated,
        shell_order=shell_order,
        shell_pair_id=shell_pair_id,
        pair_offsets=pair_offsets,
        pair_weights=pair_weights,
        r_grid=r_grid,
        correlation=correlation,
        biso_by_species=biso_by_species,
        density_by_species=density_by_species,
        delta_g=delta_g,
        delta_broad=delta_broad,
        scale=scale,
        rho0=rho0,
    )

    r_float = np.asarray(
        r_grid,
        dtype=np.float64,
    )

    shell_cart = np.asarray(
        cpu_evaluated.shell_cart,
        dtype=np.float64,
    )[
        shell_order
    ]

    shell_distance = np.asarray(
        cpu_evaluated.shell_distance,
        dtype=np.float64,
    )[
        shell_order
    ]

    shell_active = np.asarray(
        cpu_evaluated.shell_active,
        dtype=bool,
    )[
        shell_order
    ]

    shell_bulk_mult_eff = np.asarray(
        cpu_evaluated.shell_bulk_mult_eff,
        dtype=np.float64,
    )[
        shell_order
    ]

    gamma_saxs = _cpu_saxs_cylinder_gamma(
        cart_vectors=shell_cart,
        r_ij=shell_distance,
        shell_active=shell_active,
        shell_pair_id=shell_pair_id,
        shell_bulk_mult_eff=shell_bulk_mult_eff,
        pair_weights=pair_weights,
        r_grid=r_float,
        diameter=saxs_diameter,
        diameter_std=saxs_diameter_std,
        height=saxs_height,
        height_std=saxs_height_std,
        axis_cart=cylinder_axis_cart,
    )

    cpu_peak = (
        float(
            scale
        )
        * float(
            4.0
            * np.pi
            * rho0
        )
        * r_float
        * np.asarray(
            cpu_core[
                "g_r"
            ],
            dtype=np.float64,
        )
    )

    cpu_belly = (
        -float(
            scale
        )
        * float(
            saxs_scale
        )
        * float(
            4.0
            * np.pi
            * rho0
        )
        * r_float
        * gamma_saxs
    )

    cpu_saxs_result = (
        cpu_peak
        + cpu_belly
    )

    np.testing.assert_allclose(
        gpu_result[
            "gamma_saxs"
        ],
        gamma_saxs,
        rtol=7.0e-4,
        atol=7.0e-5,
    )

    np.testing.assert_allclose(
        gpu_result[
            "G_r"
        ],
        cpu_saxs_result,
        rtol=8.0e-4,
        atol=8.0e-5,
    )