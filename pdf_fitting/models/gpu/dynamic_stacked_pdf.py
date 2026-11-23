from __future__ import annotations

from typing import Any, Dict, Optional

import time

import numpy as np

from cupyx.scipy.special import (
    erf,
    erfc,
)

from .backend import (
    get_cupy,
    select_device,
)


_INV_4PI2 = np.float32(
    1.0
    / (
        4.0
        * np.pi
        * np.pi
    )
)

_CUDA_DYNAMIC_STACKED_GAUSSIAN_SOURCE = r"""
extern "C" __global__
void accumulate_dynamic_stacked_gaussians(
    const float* r_grid,
    const int n_r,
    const float* shell_distance,
    const float* sigma,
    const float* amplitude,
    const int* pair_id,
    const unsigned char* active,
    const int n_shell,
    const int n_pair,
    float* output
) {
    const int shell_index =
        blockDim.x
        * blockIdx.x
        + threadIdx.x;

    if (shell_index >= n_shell) {
        return;
    }

    if (active[shell_index] == 0) {
        return;
    }

    const float center = shell_distance[
        shell_index
    ];

    const float width = sigma[
        shell_index
    ];

    const float amp = amplitude[
        shell_index
    ];

    if (
        !(width > 0.0f)
        || !(amp != 0.0f)
    ) {
        return;
    }

    const int pair = pair_id[
        shell_index
    ];

    if (
        pair < 0
        || pair >= n_pair
    ) {
        return;
    }

    const float lower = (
        center
        - 5.0f
        * width
    );

    const float upper = (
        center
        + 5.0f
        * width
    );

    int left = 0;
    int right = n_r;

    while (left < right) {
        const int middle = (
            left
            + right
        ) >> 1;

        if (
            r_grid[middle]
            < lower
        ) {
            left = middle + 1;
        }
        else {
            right = middle;
        }
    }

    const int first_index = left;

    left = 0;
    right = n_r;

    while (left < right) {
        const int middle = (
            left
            + right
        ) >> 1;

        if (
            r_grid[middle]
            < upper
        ) {
            left = middle + 1;
        }
        else {
            right = middle;
        }
    }

    const int last_index = left;

    const float inverse_width = (
        1.0f
        / width
    );

    const float prefactor = (
        amp
        * 0.3989422804014327f
        * inverse_width
    );

    const long long output_offset = (
        ((long long) pair)
        * ((long long) n_r)
    );

    for (
        int grid_index = first_index;
        grid_index < last_index;
        ++grid_index
    ) {
        const float normalized_distance = (
            r_grid[grid_index]
            - center
        )
        * inverse_width;

        const float contribution = (
            prefactor
            * __expf(
                -0.5f
                * normalized_distance
                * normalized_distance
            )
        );

        atomicAdd(
            &output[
                output_offset
                + grid_index
            ],
            contribution
        );
    }
}
"""


class DynamicStackedCudaPdfBackend:
    """
    Resident CUDA backend for dynamic stacked PDF calculations.

    This backend retains immutable stacked topology arrays on GPU and computes
    dynamic shell geometry, broadening, Gaussian accumulation, and finite-shape
    gamma baselines without downloading full shell arrays to CPU.

    GPU-3c initial scope:

        - dynamic lattice / spacing geometry;
        - isotropic microstrain;
        - Biso;
        - delta_g;
        - delta_broad;
        - precomputed lambda/delta correlation;
        - finite stacked gamma baseline;
        - Gaussian pair accumulation.

    Qmax termination remains on CPU in PDFCalculator.

    SAXS support will be added to this class after the core Gaussian PDF path
    is validated against the existing CPU model.
    """

    def __init__(
        self,
        *,
        shell_delta_fractional: np.ndarray,
        shell_delta_a: np.ndarray,
        shell_delta_b: np.ndarray,
        shell_delta_s: np.ndarray,
        shell_delta_normal: np.ndarray,
        shell_pair_id: np.ndarray,
        shell_alpha_id: np.ndarray,
        shell_beta_id: np.ndarray,
        shell_upair_id: np.ndarray,
        shell_k_rank: np.ndarray,
        shell_k_within_upair: np.ndarray,
        shell_mult: np.ndarray,
        shell_bulk_mult: np.ndarray,
        finite_species_counts: np.ndarray,
        bulk_species_counts: np.ndarray,
        pair_weights: np.ndarray,
        pair_offsets: np.ndarray,
        pair_cutoff: float,
        device_id: int = 0,
    ) -> None:
        self.cp = get_cupy()

        self.device_id = int(
            device_id
        )

        self.pair_cutoff = float(
            pair_cutoff
        )

        select_device(
            self.device_id
        )

        self._gaussian_kernel = self.cp.RawKernel(
            _CUDA_DYNAMIC_STACKED_GAUSSIAN_SOURCE,
            "accumulate_dynamic_stacked_gaussians",
            options=(
                "--use_fast_math",
            ),
        )

        self.shell_count = int(
            np.asarray(
                shell_delta_a
            ).size
        )

        self.pair_count = int(
            np.asarray(
                pair_weights
            ).size
        )

        with self.cp.cuda.Device(
            self.device_id
        ):
            self.d_shell_delta_fractional = self.cp.asarray(
                np.ascontiguousarray(
                    shell_delta_fractional,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_delta_a = self.cp.asarray(
                np.ascontiguousarray(
                    shell_delta_a,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_delta_b = self.cp.asarray(
                np.ascontiguousarray(
                    shell_delta_b,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_delta_s = self.cp.asarray(
                np.ascontiguousarray(
                    shell_delta_s,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_delta_normal = self.cp.asarray(
                np.ascontiguousarray(
                    shell_delta_normal,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_pair_id = self.cp.asarray(
                np.ascontiguousarray(
                    shell_pair_id,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_alpha_id = self.cp.asarray(
                np.ascontiguousarray(
                    shell_alpha_id,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_beta_id = self.cp.asarray(
                np.ascontiguousarray(
                    shell_beta_id,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_upair_id = self.cp.asarray(
                np.ascontiguousarray(
                    shell_upair_id,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_k_rank = self.cp.asarray(
                np.ascontiguousarray(
                    shell_k_rank,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_k_within_upair = self.cp.asarray(
                np.ascontiguousarray(
                    shell_k_within_upair,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

            self.d_shell_mult = self.cp.asarray(
                np.ascontiguousarray(
                    shell_mult,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_shell_bulk_mult = self.cp.asarray(
                np.ascontiguousarray(
                    shell_bulk_mult,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_finite_species_counts = self.cp.asarray(
                np.ascontiguousarray(
                    finite_species_counts,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_bulk_species_counts = self.cp.asarray(
                np.ascontiguousarray(
                    bulk_species_counts,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_pair_weights = self.cp.asarray(
                np.ascontiguousarray(
                    pair_weights,
                    dtype=np.float32,
                ),
                dtype=self.cp.float32,
            )

            self.d_pair_offsets = self.cp.asarray(
                np.ascontiguousarray(
                    pair_offsets,
                    dtype=np.int32,
                ),
                dtype=self.cp.int32,
            )

        self.d_shell_cart = None
        self.d_shell_distance = None
        self.d_shell_active = None
        self.d_shell_mult_eff = None
        self.d_shell_bulk_mult_eff = None
        self.d_shell_gamma = None
        self.d_sigma = None
        self.d_amplitude = None
        self.d_r_grid = None
        self.d_output_pair = None

        self._r_grid_shape = None

    def _ensure_output(
        self,
        r_grid: np.ndarray,
    ) -> None:
        cp = self.cp

        r_grid = np.ascontiguousarray(
            r_grid,
            dtype=np.float32,
        )

        output_shape = (
            self.pair_count,
            int(
                r_grid.size
            ),
        )

        if (
            self.d_r_grid is None
            or self._r_grid_shape != tuple(
                r_grid.shape
            )
        ):
            self.d_r_grid = cp.asarray(
                r_grid,
                dtype=cp.float32,
            )

            self._r_grid_shape = tuple(
                r_grid.shape
            )

        else:
            self.d_r_grid.set(
                r_grid
            )

        if (
            self.d_output_pair is None
            or self.d_output_pair.shape
            != output_shape
        ):
            self.d_output_pair = cp.zeros(
                output_shape,
                dtype=cp.float32,
            )

        else:
            self.d_output_pair.fill(
                cp.float32(
                    0.0
                )
            )

    def _evaluate_geometry(
        self,
        *,
        lattice_matrix: np.ndarray,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        stack_normal: np.ndarray,
        spacing: float,
    ) -> None:
        cp = self.cp

        d_lattice = cp.asarray(
            np.ascontiguousarray(
                lattice_matrix,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_vector_a = cp.asarray(
            np.ascontiguousarray(
                vector_a,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_vector_b = cp.asarray(
            np.ascontiguousarray(
                vector_b,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_normal = cp.asarray(
            np.ascontiguousarray(
                stack_normal,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        normal_offset = (
            self.d_shell_delta_s
            * cp.float32(
                spacing
            )
            + self.d_shell_delta_normal
        )

        self.d_shell_cart = (
            self.d_shell_delta_fractional
            @ d_lattice
            + self.d_shell_delta_a[
                :,
                None,
            ]
            * d_vector_a[
                None,
                :,
            ]
            + self.d_shell_delta_b[
                :,
                None,
            ]
            * d_vector_b[
                None,
                :,
            ]
            + normal_offset[
                :,
                None,
            ]
            * d_normal[
                None,
                :,
            ]
        ).astype(
            cp.float32,
            copy=False,
        )

        self.d_shell_distance = cp.sqrt(
            cp.sum(
                self.d_shell_cart
                * self.d_shell_cart,
                axis=1,
            )
        ).astype(
            cp.float32,
            copy=False,
        )

        self.d_shell_active = (
            cp.isfinite(
                self.d_shell_distance
            )
            & (
                self.d_shell_distance
                > cp.float32(
                    1.0e-8
                )
            )
            & (
                self.d_shell_distance
                <= cp.float32(
                    self.pair_cutoff
                    + 1.0e-10
                )
            )
        )

        finite_denominator = self.d_finite_species_counts[
            self.d_shell_alpha_id
        ]

        bulk_denominator = self.d_bulk_species_counts[
            self.d_shell_alpha_id
        ]

        valid_finite = (
            self.d_shell_active
            & (
                finite_denominator
                > cp.float32(
                    0.0
                )
            )
        )

        valid_bulk = (
            self.d_shell_active
            & (
                bulk_denominator
                > cp.float32(
                    0.0
                )
            )
        )

        self.d_shell_mult_eff = cp.where(
            valid_finite,
            self.d_shell_mult
            / cp.maximum(
                finite_denominator,
                cp.float32(
                    1.0e-30
                ),
            ),
            cp.float32(
                0.0
            ),
        ).astype(
            cp.float32,
            copy=False,
        )

        self.d_shell_bulk_mult_eff = cp.where(
            valid_bulk,
            self.d_shell_bulk_mult
            / cp.maximum(
                bulk_denominator,
                cp.float32(
                    1.0e-30
                ),
            ),
            cp.float32(
                0.0
            ),
        ).astype(
            cp.float32,
            copy=False,
        )

        self.d_shell_gamma = cp.where(
            self.d_shell_bulk_mult_eff
            > cp.float32(
                0.0
            ),
            self.d_shell_mult_eff
            / self.d_shell_bulk_mult_eff,
            cp.float32(
                0.0
            ),
        ).astype(
            cp.float32,
            copy=False,
        )

    def _build_correlation(
        self,
        *,
        global_lambda_values: np.ndarray,
        pair_lambda_values: Optional[
            np.ndarray
        ],
        global_delta1: float,
        global_delta2: float,
        pair_delta1: np.ndarray,
        pair_delta2: np.ndarray,
    ):
        """
        Build lambda/delta correlation entirely on GPU.

        Priority matches the CPU PDFCalculator path:

            pair lambda
            global lambda
            pair delta1/delta2
            global delta1/delta2
        """
        cp = self.cp

        correlation = cp.empty(
            self.shell_count,
            dtype=cp.float32,
        )

        remaining = cp.ones(
            self.shell_count,
            dtype=cp.bool_,
        )

        if (
            pair_lambda_values is not None
            and pair_lambda_values.size > 0
        ):
            d_pair_lambda = cp.asarray(
                np.ascontiguousarray(
                    pair_lambda_values,
                    dtype=np.float32,
                ),
                dtype=cp.float32,
            )

            valid_pair = (
                self.d_shell_k_within_upair
                < d_pair_lambda.shape[
                    1
                ]
            )

            lambda_value = cp.full(
                self.shell_count,
                cp.nan,
                dtype=cp.float32,
            )

            lambda_value[
                valid_pair
            ] = d_pair_lambda[
                self.d_shell_upair_id[
                    valid_pair
                ],
                self.d_shell_k_within_upair[
                    valid_pair
                ],
            ]

            has_pair_lambda = cp.isfinite(
                lambda_value
            )

            correlation[
                has_pair_lambda
            ] = (
                cp.float32(
                    1.0
                )
                - lambda_value[
                    has_pair_lambda
                ]
            )

            remaining = ~has_pair_lambda

        global_lambda_values = np.asarray(
            global_lambda_values,
            dtype=np.float32,
        )

        if (
            global_lambda_values.size > 0
        ):
            d_global_lambda = cp.asarray(
                global_lambda_values,
                dtype=cp.float32,
            )

            has_global_lambda = (
                remaining
                & (
                    self.d_shell_k_rank
                    < d_global_lambda.size
                )
            )

            correlation[
                has_global_lambda
            ] = (
                cp.float32(
                    1.0
                )
                - d_global_lambda[
                    self.d_shell_k_rank[
                        has_global_lambda
                    ]
                ]
            )

            remaining = (
                remaining
                & ~has_global_lambda
            )

        d_pair_delta1 = cp.asarray(
            np.ascontiguousarray(
                pair_delta1,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_pair_delta2 = cp.asarray(
            np.ascontiguousarray(
                pair_delta2,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        distance = cp.maximum(
            self.d_shell_distance,
            cp.float32(
                1.0e-12
            ),
        )

        delta1_value = cp.full(
            self.shell_count,
            cp.float32(
                global_delta1
            ),
            dtype=cp.float32,
        )

        delta2_value = cp.full(
            self.shell_count,
            cp.float32(
                global_delta2
            ),
            dtype=cp.float32,
        )

        pair_delta1_value = d_pair_delta1[
            self.d_shell_upair_id
        ]

        pair_delta2_value = d_pair_delta2[
            self.d_shell_upair_id
        ]

        has_pair_delta1 = cp.isfinite(
            pair_delta1_value
        )

        has_pair_delta2 = cp.isfinite(
            pair_delta2_value
        )

        delta1_value[
            has_pair_delta1
        ] = pair_delta1_value[
            has_pair_delta1
        ]

        delta2_value[
            has_pair_delta2
        ] = pair_delta2_value[
            has_pair_delta2
        ]

        correlation[
            remaining
        ] = (
            cp.float32(
                1.0
            )
            - delta1_value[
                remaining
            ]
            / distance[
                remaining
            ]
            - delta2_value[
                remaining
            ]
            / (
                distance[
                    remaining
                ]
                * distance[
                    remaining
                ]
            )
        )

        return cp.maximum(
            correlation,
            cp.float32(
                0.0
            ),
        ).astype(
            cp.float32,
            copy=False,
        )
    
    def _prepare_sigma_and_amplitude(
        self,
        *,
        correlation: np.ndarray,
        biso_by_species: np.ndarray,
        density_by_species: np.ndarray,
        delta_g: float,
        delta_broad: float,
    ) -> None:
        cp = self.cp

        if isinstance(
            correlation,
            cp.ndarray,
        ):
            d_correlation = correlation.astype(
                cp.float32,
                copy=False,
            )
        else:
            d_correlation = cp.asarray(
                np.ascontiguousarray(
                    correlation,
                    dtype=np.float32,
                ),
                dtype=cp.float32,
            )

        d_biso = cp.asarray(
            np.ascontiguousarray(
                biso_by_species,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_density = cp.asarray(
            np.ascontiguousarray(
                density_by_species,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        shell_biso = (
            cp.float32(
                0.5
            )
            * (
                d_biso[
                    self.d_shell_alpha_id
                ]
                + d_biso[
                    self.d_shell_beta_id
                ]
            )
        )

        sigma_squared = (
            cp.float32(
                _INV_4PI2
            )
            * shell_biso
            * d_correlation
            + (
                cp.float32(
                    delta_g
                )
                * self.d_shell_distance
            ) ** 2
            + (
                cp.float32(
                    delta_broad
                )
                * self.d_shell_distance
            ) ** 2
        )

        self.d_sigma = cp.sqrt(
            cp.maximum(
                sigma_squared,
                cp.float32(
                    1.0e-24
                ),
            )
        ).astype(
            cp.float32,
            copy=False,
        )

        neighbor_density = cp.maximum(
            d_density[
                self.d_shell_beta_id
            ],
            cp.float32(
                1.0e-30
            ),
        )

        distance_squared = cp.maximum(
            self.d_shell_distance
            * self.d_shell_distance,
            cp.float32(
                1.0e-24
            ),
        )

        self.d_amplitude = (
            self.d_shell_mult_eff
            / (
                cp.float32(
                    4.0
                    * np.pi
                )
                * distance_squared
                * neighbor_density
            )
        ).astype(
            cp.float32,
            copy=False,
        )

    def _accumulate_gaussians(
        self,
    ) -> None:
        """
        Accumulate shell Gaussians directly into pair-resolved g(r).

        This uses one CUDA thread per shell and atomic additions into the
        corresponding ordered-pair PDF row.

        Unlike the old CuPy matrix implementation, this does not create an
        intermediate array shaped:

            number_of_shells_in_pair × number_of_r_points

        Therefore it is suitable for large dynamic stacked particles.
        """
        cp = self.cp

        if self.d_output_pair is None:
            raise RuntimeError(
                "GPU pair-output array is not initialized."
            )

        if self.d_shell_active is None:
            raise RuntimeError(
                "Dynamic stacked shell activity mask is not initialized."
            )

        if self.d_shell_distance is None:
            raise RuntimeError(
                "Dynamic stacked shell distances are not initialized."
            )

        if self.d_sigma is None:
            raise RuntimeError(
                "Dynamic stacked shell widths are not initialized."
            )

        if self.d_amplitude is None:
            raise RuntimeError(
                "Dynamic stacked shell amplitudes are not initialized."
            )

        active_uint8 = self.d_shell_active.astype(
            cp.uint8,
            copy=False,
        )

        threads = 256

        blocks = (
            self.shell_count
            + threads
            - 1
        ) // threads

        self._gaussian_kernel(
            (
                blocks,
            ),
            (
                threads,
            ),
            (
                self.d_r_grid,
                np.int32(
                    self.d_r_grid.size
                ),
                self.d_shell_distance,
                self.d_sigma,
                self.d_amplitude,
                self.d_shell_pair_id,
                active_uint8,
                np.int32(
                    self.shell_count
                ),
                np.int32(
                    self.pair_count
                ),
                self.d_output_pair,
            ),
        )


    def _finite_gamma_by_pair(
        self,
    ):
        """
        Build finite gamma(r) on GPU using nearest-grid histograms.

        This matches the existing CPU method conceptually:

            shell gamma values are weighted by shell_bulk_mult_eff;
            values are grouped by pair and nearest r-grid point;
            missing values are interpolated later.
        """
        cp = self.cp

        n_r = int(
            self.d_r_grid.size
        )

        gamma_numerator = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        gamma_denominator = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        r_min = self.d_r_grid[
            0
        ]

        r_max = self.d_r_grid[
            -1
        ]

        valid = (
            self.d_shell_active
            & (
                self.d_shell_distance
                > cp.float32(
                    0.0
                )
            )
            & (
                self.d_shell_distance
                >= r_min
            )
            & (
                self.d_shell_distance
                <= r_max
            )
            & (
                self.d_shell_bulk_mult_eff
                > cp.float32(
                    0.0
                )
            )
        )

        if not bool(
            cp.any(
                valid
            ).item()
        ):
            return cp.zeros(
                (
                    self.pair_count,
                    n_r,
                ),
                dtype=cp.float32,
            )

        valid_indices = cp.flatnonzero(
            valid
        )

        valid_pair = self.d_shell_pair_id[
            valid_indices
        ]

        valid_r = self.d_shell_distance[
            valid_indices
        ]

        valid_gamma = self.d_shell_gamma[
            valid_indices
        ]

        valid_weight = self.d_shell_bulk_mult_eff[
            valid_indices
        ]

        right = cp.searchsorted(
            self.d_r_grid,
            valid_r,
            side="left",
        )

        right = cp.clip(
            right,
            0,
            n_r - 1,
        )

        left = cp.maximum(
            right - 1,
            0,
        )

        use_left = (
            cp.abs(
                valid_r
                - self.d_r_grid[
                    left
                ]
            )
            <= cp.abs(
                valid_r
                - self.d_r_grid[
                    right
                ]
            )
        )

        grid_index = cp.where(
            use_left,
            left,
            right,
        ).astype(
            cp.int32,
            copy=False,
        )

        flat_index = (
            valid_pair.astype(
                cp.int64
            )
            * np.int64(
                n_r
            )
            + grid_index.astype(
                cp.int64
            )
        )

        cp.add.at(
            gamma_numerator.ravel(),
            flat_index,
            valid_gamma
            * valid_weight,
        )

        cp.add.at(
            gamma_denominator.ravel(),
            flat_index,
            valid_weight,
        )

        gamma_by_pair = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        for pair_id in range(
            self.pair_count
        ):
            denominator = gamma_denominator[
                pair_id
            ]

            numerator = gamma_numerator[
                pair_id
            ]

            valid_grid = denominator > cp.float32(
                0.0
            )

            if not bool(
                cp.any(
                    valid_grid
                ).item()
            ):
                continue

            known_index = cp.flatnonzero(
                valid_grid
            )

            known_r = self.d_r_grid[
                known_index
            ]

            known_gamma = (
                numerator[
                    known_index
                ]
                / denominator[
                    known_index
                ]
            )

            if int(
                known_index[
                    0
                ].item()
            ) > 0:
                known_r = cp.concatenate(
                    (
                        cp.asarray(
                            [
                                0.0
                            ],
                            dtype=cp.float32,
                        ),
                        known_r,
                    )
                )

                known_gamma = cp.concatenate(
                    (
                        cp.asarray(
                            [
                                1.0
                            ],
                            dtype=cp.float32,
                        ),
                        known_gamma,
                    )
                )

            else:
                known_gamma[
                    0
                ] = cp.float32(
                    1.0
                )

            gamma_by_pair[
                pair_id,
                :
            ] = cp.interp(
                self.d_r_grid,
                known_r,
                known_gamma,
                left=cp.float32(
                    1.0
                ),
                right=cp.float32(
                    0.0
                ),
            )

        return gamma_by_pair

    def _sphere_gamma(
        self,
        *,
        diameter: float,
        diameter_std: float,
    ):
        """
        GPU equivalent of PDFCalculator.shape_function().

        Returns a dimensionless gamma(r) evaluated on self.d_r_grid.
        """
        cp = self.cp

        r = self.d_r_grid.astype(
            cp.float32,
            copy=False,
        )

        diameter = float(
            diameter
        )

        diameter_std = float(
            diameter_std
        )

        if diameter <= 0.0:
            return cp.ones_like(
                r,
                dtype=cp.float32,
            )

        if diameter_std < 1.0e-8:
            gamma = cp.zeros_like(
                r,
                dtype=cp.float32,
            )

            mask = r < cp.float32(
                diameter
            )

            ratio = (
                r[
                    mask
                ]
                / cp.float32(
                    diameter
                )
            )

            gamma[
                mask
            ] = (
                cp.float32(
                    1.0
                )
                - cp.float32(
                    1.5
                )
                * ratio
                + cp.float32(
                    0.5
                )
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

        log_r = cp.log(
            cp.maximum(
                r,
                cp.float32(
                    1.0e-8
                ),
            )
        )

        sqrt_two_sigma = cp.float32(
            np.sqrt(
                2.0
            )
            * sigma
        )

        a_term = erfc(
            (
                -cp.float32(
                    mu
                )
                - cp.float32(
                    3.0
                    * sigma_squared
                )
                + log_r
            )
            / sqrt_two_sigma
        )

        b_term = erfc(
            (
                -cp.float32(
                    mu
                )
                + log_r
            )
            / sqrt_two_sigma
        )

        c_term = erfc(
            (
                -cp.float32(
                    mu
                )
                - cp.float32(
                    2.0
                    * sigma_squared
                )
                + log_r
            )
            / sqrt_two_sigma
        )

        exp_a = cp.float32(
            np.exp(
                -3.0
                * mu
                - 4.5
                * sigma_squared
            )
        )

        exp_b = cp.float32(
            np.exp(
                -mu
                - 2.5
                * sigma_squared
            )
        )

        gamma = (
            cp.float32(
                0.5
            )
            * a_term
            + cp.float32(
                0.25
            )
            * r
            * r
            * r
            * b_term
            * exp_a
            - cp.float32(
                0.75
            )
            * r
            * c_term
            * exp_b
        )

        return cp.asarray(
            gamma,
            dtype=cp.float32,
        )

    def _cylinder_shell_gamma(
        self,
        *,
        diameter: float,
        thickness: float,
        thickness_std: float,
        axis_cart: np.ndarray,
    ):
        """
        GPU equivalent of the monodisperse/lognormal cylinder common-volume
        shell factor used by the CPU PDF model.

        Returns one gamma value per dynamic shell.
        """
        cp = self.cp

        diameter = float(
            diameter
        )

        thickness = float(
            thickness
        )

        thickness_std = float(
            thickness_std
        )

        if (
            diameter <= 0.0
            or thickness <= 0.0
        ):
            return cp.ones(
                self.shell_count,
                dtype=cp.float32,
            )

        axis_cart = np.asarray(
            axis_cart,
            dtype=np.float32,
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
                dtype=np.float32,
            )

            axis_norm = 1.0

        axis_cart = (
            axis_cart
            / np.float32(
                axis_norm
            )
        )

        d_axis = cp.asarray(
            axis_cart,
            dtype=cp.float32,
        )

        distance = self.d_shell_distance

        projected = cp.abs(
            self.d_shell_cart
            @ d_axis
        )

        radial_squared = cp.maximum(
            distance
            * distance
            - projected
            * projected,
            cp.float32(
                0.0
            ),
        )

        radial = cp.sqrt(
            radial_squared
        )

        gamma = cp.zeros(
            self.shell_count,
            dtype=cp.float32,
        )

        valid = (
            self.d_shell_active
            & (
                distance
                > cp.float32(
                    0.0
                )
            )
            & (
                radial
                < cp.float32(
                    diameter
                )
            )
        )

        if not bool(
            cp.any(
                valid
            ).item()
        ):
            return gamma

        radial_valid = radial[
            valid
        ]

        projected_valid = projected[
            valid
        ]

        x = cp.clip(
            radial_valid
            / cp.float32(
                diameter
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            ),
        )

        radial_factor = (
            cp.float32(
                2.0
                / np.pi
            )
            * (
                cp.arccos(
                    x
                )
                - x
                * cp.sqrt(
                    cp.maximum(
                        cp.float32(
                            1.0
                        )
                        - x
                        * x,
                        cp.float32(
                            0.0
                        ),
                    )
                )
            )
        )

        if thickness_std <= 1.0e-12:
            axial_factor = cp.maximum(
                cp.float32(
                    1.0
                )
                - projected_valid
                / cp.float32(
                    thickness
                ),
                cp.float32(
                    0.0
                ),
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

        z_zero = z <= cp.float32(
            1.0e-30
        )

        z_safe = cp.maximum(
            z,
            cp.float32(
                1.0e-30
            ),
        )

        log_z = cp.log(
            z_safe
        )

        sqrt_two_sigma = cp.float32(
            np.sqrt(
                2.0
            )
            * sigma
        )

        bracket = (
            cp.float32(
                np.exp(
                    mu
                    + 0.5
                    * sigma_squared
                )
            )
            * (
                cp.float32(
                    1.0
                )
                + erf(
                    (
                        cp.float32(
                            mu
                            + sigma_squared
                        )
                        - log_z
                    )
                    / sqrt_two_sigma
                )
            )
            + z
            * (
                erfc(
                    (
                        cp.float32(
                            mu
                        )
                        - log_z
                    )
                    / sqrt_two_sigma
                )
                - cp.float32(
                    2.0
                )
            )
        )

        bracket = cp.where(
            z_zero,
            cp.float32(
                2.0
                * np.exp(
                    mu
                    + 0.5
                    * sigma_squared
                )
            ),
            bracket,
        )

        gamma_valid = (
            cp.float32(
                np.exp(
                    -mu
                    - 0.5
                    * sigma_squared
                )
                / (
                    diameter
                    * np.pi
                )
            )
            * bracket
            * (
                cp.float32(
                    0.5
                )
                * np.pi
                * radial_factor
                * cp.float32(
                    diameter
                )
            )
        )

        gamma[
            valid
        ] = cp.clip(
            gamma_valid,
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            ),
        )

        return gamma

    def _shell_gamma_to_pair_grid(
        self,
        shell_gamma,
        shell_weight,
    ):
        """
        Build pair-resolved gamma(r) using the current dynamic shell distances.

        This is used by:

            finite stacked baseline;
            cylinder/disk SAXS gamma;
            cylinder/disk main shape gamma.
        """
        cp = self.cp

        n_r = int(
            self.d_r_grid.size
        )

        numerator = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        denominator = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        valid = (
            self.d_shell_active
            & (
                self.d_shell_distance
                > cp.float32(
                    0.0
                )
            )
            & (
                self.d_shell_distance
                >= self.d_r_grid[
                    0
                ]
            )
            & (
                self.d_shell_distance
                <= self.d_r_grid[
                    -1
                ]
            )
            & (
                shell_weight
                > cp.float32(
                    0.0
                )
            )
        )

        if not bool(
            cp.any(
                valid
            ).item()
        ):
            return cp.zeros(
                (
                    self.pair_count,
                    n_r,
                ),
                dtype=cp.float32,
            )

        shell_index = cp.flatnonzero(
            valid
        )

        pair_id = self.d_shell_pair_id[
            shell_index
        ]

        distance = self.d_shell_distance[
            shell_index
        ]

        gamma = shell_gamma[
            shell_index
        ]

        weight = shell_weight[
            shell_index
        ]

        right = cp.searchsorted(
            self.d_r_grid,
            distance,
            side="left",
        )

        right = cp.clip(
            right,
            0,
            n_r - 1,
        )

        left = cp.maximum(
            right - 1,
            0,
        )

        grid_index = cp.where(
            cp.abs(
                distance
                - self.d_r_grid[
                    left
                ]
            )
            <= cp.abs(
                distance
                - self.d_r_grid[
                    right
                ]
            ),
            left,
            right,
        ).astype(
            cp.int32,
            copy=False,
        )

        flat_index = (
            pair_id.astype(
                cp.int64
            )
            * np.int64(
                n_r
            )
            + grid_index.astype(
                cp.int64
            )
        )

        cp.add.at(
            numerator.ravel(),
            flat_index,
            gamma
            * weight,
        )

        cp.add.at(
            denominator.ravel(),
            flat_index,
            weight,
        )

        gamma_by_pair = cp.zeros(
            (
                self.pair_count,
                n_r,
            ),
            dtype=cp.float32,
        )

        for pair_id_value in range(
            self.pair_count
        ):
            den = denominator[
                pair_id_value
            ]

            num = numerator[
                pair_id_value
            ]

            valid_grid = den > cp.float32(
                0.0
            )

            if not bool(
                cp.any(
                    valid_grid
                ).item()
            ):
                continue

            grid_positions = cp.flatnonzero(
                valid_grid
            )

            known_r = self.d_r_grid[
                grid_positions
            ]

            known_gamma = (
                num[
                    grid_positions
                ]
                / den[
                    grid_positions
                ]
            )

            if int(
                grid_positions[
                    0
                ].item()
            ) > 0:
                known_r = cp.concatenate(
                    (
                        cp.asarray(
                            [
                                0.0,
                            ],
                            dtype=cp.float32,
                        ),
                        known_r,
                    )
                )

                known_gamma = cp.concatenate(
                    (
                        cp.asarray(
                            [
                                1.0,
                            ],
                            dtype=cp.float32,
                        ),
                        known_gamma,
                    )
                )

            else:
                known_gamma[
                    0
                ] = cp.float32(
                    1.0
                )

            gamma_by_pair[
                pair_id_value,
                :
            ] = cp.interp(
                self.d_r_grid,
                known_r,
                known_gamma,
                left=cp.float32(
                    1.0
                ),
                right=cp.float32(
                    0.0
                ),
            )

        return gamma_by_pair

    def _weighted_pair_gamma(
        self,
        gamma_by_pair,
    ):
        """
        Return Faber-Ziman weighted total gamma(r).
        """
        return (
            self.d_pair_weights[
                :,
                None,
            ]
            * gamma_by_pair
        ).sum(
            axis=0,
            dtype=self.cp.float32,
        )

    def _saxs_cylinder_gamma(
        self,
        *,
        diameter: float,
        diameter_std: float,
        height: float,
        height_std: float,
        axis_cart: np.ndarray,
    ):
        """
        Compute the powder-weighted cylinder/disk SAXS gamma(r) on GPU.

        Diameter distribution uses the same 15-point lognormal quadrature
        convention currently used by the CPU implementation.
        """
        cp = self.cp

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

        if (
            diameter <= 0.0
            or height <= 0.0
        ):
            return None

        if diameter_std <= 1.0e-12:
            shell_gamma = self._cylinder_shell_gamma(
                diameter=diameter,
                thickness=height,
                thickness_std=height_std,
                axis_cart=axis_cart,
            )

            gamma_by_pair = self._shell_gamma_to_pair_grid(
                shell_gamma,
                self.d_shell_bulk_mult_eff,
            )

            return self._weighted_pair_gamma(
                gamma_by_pair
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
            dtype=np.float32,
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

        total_gamma = cp.zeros(
            self.d_r_grid.size,
            dtype=cp.float32,
        )

        for diameter_value, weight in zip(
            diameter_grid,
            weights,
        ):
            shell_gamma = self._cylinder_shell_gamma(
                diameter=float(
                    diameter_value
                ),
                thickness=height,
                thickness_std=height_std,
                axis_cart=axis_cart,
            )

            gamma_by_pair = self._shell_gamma_to_pair_grid(
                shell_gamma,
                self.d_shell_bulk_mult_eff,
            )

            total_gamma += cp.float32(
                weight
            ) * self._weighted_pair_gamma(
                gamma_by_pair
            )

        return total_gamma
    
    def evaluate(
        self,
        *,
        lattice_matrix: np.ndarray,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        stack_normal: np.ndarray,
        spacing: float,
        global_lambda_values: np.ndarray,
        pair_lambda_values: Optional[
            np.ndarray
        ],
        global_delta1: float,
        global_delta2: float,
        pair_delta1: np.ndarray,
        pair_delta2: np.ndarray,
        biso_by_species: np.ndarray,
        density_by_species: np.ndarray,
        delta_g: float,
        delta_broad: float,
        r_grid: np.ndarray,
        scale: float,
        rho0: float,
        saxs_enabled: bool = False,
        saxs_model: str = "same",
        saxs_scale: float = 1.0,
        saxs_diameter: float = 0.0,
        saxs_diameter_std: float = 0.0,
        saxs_height: float = 0.0,
        saxs_height_std: float = 0.0,
        saxs_apply_qdamp: bool = False,
        qdamp: float = 0.0,
        cylinder_axis_cart: Optional[
            np.ndarray
        ] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate the core dynamic stacked PDF entirely on GPU.

        This method does not yet apply Qmax termination. The CPU caller must
        retain the existing Qmax treatment for numerical consistency.
        """
        evaluation_start = time.perf_counter()

        cp = self.cp

        with cp.cuda.Device(
            self.device_id
        ):
            self._ensure_output(
                r_grid
            )

            self._evaluate_geometry(
                lattice_matrix=lattice_matrix,
                vector_a=vector_a,
                vector_b=vector_b,
                stack_normal=stack_normal,
                spacing=float(
                    spacing
                ),
            )

            correlation = self._build_correlation(
                global_lambda_values=(
                    global_lambda_values
                ),
                pair_lambda_values=(
                    pair_lambda_values
                ),
                global_delta1=float(
                    global_delta1
                ),
                global_delta2=float(
                    global_delta2
                ),
                pair_delta1=pair_delta1,
                pair_delta2=pair_delta2,
            )

            self._prepare_sigma_and_amplitude(
                correlation=correlation,
                biso_by_species=biso_by_species,
                density_by_species=density_by_species,
                delta_g=float(
                    delta_g
                ),
                delta_broad=float(
                    delta_broad
                ),
            )

            self._accumulate_gaussians()

            weighted_g = (
                self.d_pair_weights[
                    :,
                    None,
                ]
                * self.d_output_pair
            ).sum(
                axis=0,
                dtype=cp.float32,
            )

            gamma_by_pair = self._finite_gamma_by_pair()

            gamma_base = (
                self.d_pair_weights[
                    :,
                    None,
                ]
                * gamma_by_pair
            ).sum(
                axis=0,
                dtype=cp.float32,
            )

            r_device = self.d_r_grid

            g_total = weighted_g

            norm = cp.float32(
                4.0
                * np.pi
                * rho0
            )

            g_peak = (
                cp.float32(
                    scale
                )
                * norm
                * r_device
                * g_total
            ).astype(
                cp.float32,
                copy=False,
            )

            if bool(
                saxs_enabled
            ):
                model_name = str(
                    saxs_model
                    or "same"
                ).strip().lower()

                if model_name in (
                    "sphere",
                    "spherical",
                ):
                    if float(
                        saxs_diameter
                    ) > 0.0:
                        gamma_saxs = self._sphere_gamma(
                            diameter=float(
                                saxs_diameter
                            ),
                            diameter_std=float(
                                saxs_diameter_std
                            ),
                        )
                    else:
                        gamma_saxs = gamma_base

                elif model_name in (
                    "cylinder",
                    "cyl",
                    "disk",
                    "disc",
                ):
                    if cylinder_axis_cart is None:
                        axis_cart = np.asarray(
                            stack_normal,
                            dtype=np.float32,
                        )
                    else:
                        axis_cart = np.asarray(
                            cylinder_axis_cart,
                            dtype=np.float32,
                        )

                    gamma_saxs = self._saxs_cylinder_gamma(
                        diameter=float(
                            saxs_diameter
                        ),
                        diameter_std=float(
                            saxs_diameter_std
                        ),
                        height=float(
                            saxs_height
                        ),
                        height_std=float(
                            saxs_height_std
                        ),
                        axis_cart=axis_cart,
                    )

                    if gamma_saxs is None:
                        gamma_saxs = gamma_base

                else:
                    gamma_saxs = gamma_base

                g_belly = (
                    -cp.float32(
                        scale
                    )
                    * cp.float32(
                        saxs_scale
                    )
                    * norm
                    * r_device
                    * gamma_saxs
                ).astype(
                    cp.float32,
                    copy=False,
                )

                if (
                    bool(
                        saxs_apply_qdamp
                    )
                    and float(
                        qdamp
                    ) > 0.0
                ):
                    g_belly *= cp.exp(
                        cp.float32(
                            -0.5
                        )
                        * cp.float32(
                            qdamp
                            * qdamp
                        )
                        * r_device
                        * r_device
                    )

                if float(
                    qdamp
                ) > 0.0:
                    g_peak *= cp.exp(
                        cp.float32(
                            -0.5
                        )
                        * cp.float32(
                            qdamp
                            * qdamp
                        )
                        * r_device
                        * r_device
                    )

                g_pdf = (
                    g_peak
                    + g_belly
                ).astype(
                    cp.float32,
                    copy=False,
                )

            else:
                g_baseline = (
                    -cp.float32(
                        scale
                    )
                    * norm
                    * r_device
                    * gamma_base
                ).astype(
                    cp.float32,
                    copy=False,
                )

                g_pdf = (
                    g_peak
                    + g_baseline
                ).astype(
                    cp.float32,
                    copy=False,
                )

                if float(
                    qdamp
                ) > 0.0:
                    g_pdf *= cp.exp(
                        cp.float32(
                            -0.5
                        )
                        * cp.float32(
                            qdamp
                            * qdamp
                        )
                        * r_device
                        * r_device
                    )

            cp.cuda.get_current_stream().synchronize()

            gpu_seconds = (
                time.perf_counter()
                - evaluation_start
            )

            transfer_start = time.perf_counter()

            result = {
                "G_r": cp.asnumpy(
                    g_pdf
                ),
                "g_r": cp.asnumpy(
                    g_total
                ),
                "gamma_base": cp.asnumpy(
                    gamma_base
                ),
                "gamma_saxs": (
                    None
                    if not bool(
                        saxs_enabled
                    )
                    else cp.asnumpy(
                        gamma_saxs
                    )
                ),
            }

            transfer_seconds = (
                time.perf_counter()
                - transfer_start
            )

            print(
                "[DYNAMIC CUDA PDF] "
                f"gpu_core={gpu_seconds:.4f} s, "
                f"host_transfer={transfer_seconds:.4f} s, "
                f"shells={self.shell_count:,}, "
                f"pairs={self.pair_count:,}, "
                f"r_points={self.d_r_grid.size:,}"
            )

            return result