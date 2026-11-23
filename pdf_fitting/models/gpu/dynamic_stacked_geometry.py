from __future__ import annotations

from typing import Dict, Any

import time

import numpy as np

from .backend import (
    get_cupy,
    select_device,
)


class DynamicStackedCudaGeometryBackend:
    """
    GPU-resident dynamic stacked shell-geometry evaluator.

    This backend stores topology-only descriptor arrays on CUDA:

        shell_delta_fractional
        shell_delta_a
        shell_delta_b
        shell_delta_s
        shell_delta_normal
        shell_alpha_id
        shell_mult
        shell_bulk_mult
        finite_species_counts
        bulk_species_counts

    For each model evaluation it computes:

        shell_cart
        shell_distance
        shell_active
        shell_frac
        shell_mult_eff
        shell_bulk_mult_eff
        shell_gamma

    This first implementation returns evaluated arrays to the CPU because the
    existing PDFCalculator CPU path still owns Gaussian accumulation.

    Later GPU-3b/GPU-3c will retain those arrays on CUDA and use them directly
    for sigma/Aij/G(r) accumulation.
    """

    def __init__(
        self,
        *,
        shell_delta_fractional: np.ndarray,
        shell_delta_a: np.ndarray,
        shell_delta_b: np.ndarray,
        shell_delta_s: np.ndarray,
        shell_delta_normal: np.ndarray,
        shell_alpha_id: np.ndarray,
        shell_mult: np.ndarray,
        shell_bulk_mult: np.ndarray,
        finite_species_counts: np.ndarray,
        bulk_species_counts: np.ndarray,
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

        shell_delta_fractional = np.ascontiguousarray(
            shell_delta_fractional,
            dtype=np.float32,
        )

        shell_delta_a = np.ascontiguousarray(
            shell_delta_a,
            dtype=np.float32,
        )

        shell_delta_b = np.ascontiguousarray(
            shell_delta_b,
            dtype=np.float32,
        )

        shell_delta_s = np.ascontiguousarray(
            shell_delta_s,
            dtype=np.float32,
        )

        shell_delta_normal = np.ascontiguousarray(
            shell_delta_normal,
            dtype=np.float32,
        )

        shell_alpha_id = np.ascontiguousarray(
            shell_alpha_id,
            dtype=np.int32,
        )

        shell_mult = np.ascontiguousarray(
            shell_mult,
            dtype=np.float32,
        )

        shell_bulk_mult = np.ascontiguousarray(
            shell_bulk_mult,
            dtype=np.float32,
        )

        finite_species_counts = np.ascontiguousarray(
            finite_species_counts,
            dtype=np.float32,
        )

        bulk_species_counts = np.ascontiguousarray(
            bulk_species_counts,
            dtype=np.float32,
        )

        shell_count = int(
            shell_delta_a.size
        )

        if (
            shell_delta_fractional.ndim != 2
            or shell_delta_fractional.shape[1] != 3
            or shell_delta_fractional.shape[0] != shell_count
        ):
            raise ValueError(
                "shell_delta_fractional must have shape "
                "(n_shell, 3)."
            )

        for name, value in (
            (
                "shell_delta_b",
                shell_delta_b,
            ),
            (
                "shell_delta_s",
                shell_delta_s,
            ),
            (
                "shell_delta_normal",
                shell_delta_normal,
            ),
            (
                "shell_alpha_id",
                shell_alpha_id,
            ),
            (
                "shell_mult",
                shell_mult,
            ),
            (
                "shell_bulk_mult",
                shell_bulk_mult,
            ),
        ):
            if int(
                value.size
            ) != shell_count:
                raise ValueError(
                    f"{name} does not match shell count."
                )

        with self.cp.cuda.Device(
            self.device_id
        ):
            self.d_shell_delta_fractional = self.cp.asarray(
                shell_delta_fractional,
                dtype=self.cp.float32,
            )

            self.d_shell_delta_a = self.cp.asarray(
                shell_delta_a,
                dtype=self.cp.float32,
            )

            self.d_shell_delta_b = self.cp.asarray(
                shell_delta_b,
                dtype=self.cp.float32,
            )

            self.d_shell_delta_s = self.cp.asarray(
                shell_delta_s,
                dtype=self.cp.float32,
            )

            self.d_shell_delta_normal = self.cp.asarray(
                shell_delta_normal,
                dtype=self.cp.float32,
            )

            self.d_shell_alpha_id = self.cp.asarray(
                shell_alpha_id,
                dtype=self.cp.int32,
            )

            self.d_shell_mult = self.cp.asarray(
                shell_mult,
                dtype=self.cp.float32,
            )

            self.d_shell_bulk_mult = self.cp.asarray(
                shell_bulk_mult,
                dtype=self.cp.float32,
            )

            self.d_finite_species_counts = self.cp.asarray(
                finite_species_counts,
                dtype=self.cp.float32,
            )

            self.d_bulk_species_counts = self.cp.asarray(
                bulk_species_counts,
                dtype=self.cp.float32,
            )

        self.shell_count = shell_count

        self.d_shell_cart = None
        self.d_shell_distance = None
        self.d_shell_active = None
        self.d_shell_frac = None
        self.d_shell_mult_eff = None
        self.d_shell_bulk_mult_eff = None
        self.d_shell_gamma = None

    def evaluate(
        self,
        *,
        lattice_matrix: np.ndarray,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        stack_normal: np.ndarray,
        spacing: float,
        stacking_inverse_lattice_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate dynamic stacked shell arrays on CUDA.

        The returned arrays are host NumPy arrays in this GPU-3a version.
        """
        evaluation_start = time.perf_counter()

        cp = self.cp

        lattice_matrix = np.ascontiguousarray(
            lattice_matrix,
            dtype=np.float32,
        )

        vector_a = np.ascontiguousarray(
            vector_a,
            dtype=np.float32,
        )

        vector_b = np.ascontiguousarray(
            vector_b,
            dtype=np.float32,
        )

        stack_normal = np.ascontiguousarray(
            stack_normal,
            dtype=np.float32,
        )

        stacking_inverse_lattice_matrix = np.ascontiguousarray(
            stacking_inverse_lattice_matrix,
            dtype=np.float32,
        )

        if lattice_matrix.shape != (
            3,
            3,
        ):
            raise ValueError(
                "lattice_matrix must have shape (3, 3)."
            )

        if stacking_inverse_lattice_matrix.shape != (
            3,
            3,
        ):
            raise ValueError(
                "stacking_inverse_lattice_matrix must have "
                "shape (3, 3)."
            )

        if vector_a.shape != (
            3,
        ):
            raise ValueError(
                "vector_a must have shape (3,)."
            )

        if vector_b.shape != (
            3,
        ):
            raise ValueError(
                "vector_b must have shape (3,)."
            )

        if stack_normal.shape != (
            3,
        ):
            raise ValueError(
                "stack_normal must have shape (3,)."
            )

        with cp.cuda.Device(
            self.device_id
        ):
            d_lattice_matrix = cp.asarray(
                lattice_matrix,
                dtype=cp.float32,
            )

            d_vector_a = cp.asarray(
                vector_a,
                dtype=cp.float32,
            )

            d_vector_b = cp.asarray(
                vector_b,
                dtype=cp.float32,
            )

            d_stack_normal = cp.asarray(
                stack_normal,
                dtype=cp.float32,
            )

            d_stacking_inverse = cp.asarray(
                stacking_inverse_lattice_matrix,
                dtype=cp.float32,
            )

            normal_offsets = (
                self.d_shell_delta_s
                * cp.float32(
                    spacing
                )
                + self.d_shell_delta_normal
            )

            self.d_shell_cart = (
                self.d_shell_delta_fractional
                @ d_lattice_matrix
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
                + normal_offsets[
                    :,
                    None,
                ]
                * d_stack_normal[
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

            self.d_shell_frac = (
                self.d_shell_cart
                @ d_stacking_inverse
            ).astype(
                cp.float32,
                copy=False,
            )

            finite_denominator = cp.zeros(
                self.shell_count,
                dtype=cp.float32,
            )

            bulk_denominator = cp.zeros(
                self.shell_count,
                dtype=cp.float32,
            )

            valid_alpha = (
                self.d_shell_alpha_id
                >= 0
            ) & (
                self.d_shell_alpha_id
                < self.d_finite_species_counts.size
            ) & (
                self.d_shell_alpha_id
                < self.d_bulk_species_counts.size
            )

            finite_denominator[
                valid_alpha
            ] = self.d_finite_species_counts[
                self.d_shell_alpha_id[
                    valid_alpha
                ]
            ]

            bulk_denominator[
                valid_alpha
            ] = self.d_bulk_species_counts[
                self.d_shell_alpha_id[
                    valid_alpha
                ]
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

            self.d_shell_mult_eff = cp.zeros(
                self.shell_count,
                dtype=cp.float32,
            )

            self.d_shell_bulk_mult_eff = cp.zeros(
                self.shell_count,
                dtype=cp.float32,
            )

            self.d_shell_gamma = cp.zeros(
                self.shell_count,
                dtype=cp.float32,
            )

            self.d_shell_mult_eff[
                valid_finite
            ] = (
                self.d_shell_mult[
                    valid_finite
                ]
                / finite_denominator[
                    valid_finite
                ]
            )

            self.d_shell_bulk_mult_eff[
                valid_bulk
            ] = (
                self.d_shell_bulk_mult[
                    valid_bulk
                ]
                / bulk_denominator[
                    valid_bulk
                ]
            )

            valid_gamma = (
                valid_finite
                & valid_bulk
                & (
                    self.d_shell_bulk_mult_eff
                    > cp.float32(
                        0.0
                    )
                )
            )

            self.d_shell_gamma[
                valid_gamma
            ] = (
                self.d_shell_mult_eff[
                    valid_gamma
                ]
                / self.d_shell_bulk_mult_eff[
                    valid_gamma
                ]
            )

            cp.cuda.get_current_stream().synchronize()

            gpu_compute_seconds = (
                time.perf_counter()
                - evaluation_start
            )

            transfer_start = time.perf_counter()

            result = {
                "shell_cart": cp.asnumpy(
                    self.d_shell_cart
                ),
                "shell_distance": cp.asnumpy(
                    self.d_shell_distance
                ),
                "shell_active": cp.asnumpy(
                    self.d_shell_active
                ),
                "shell_frac": cp.asnumpy(
                    self.d_shell_frac
                ),
                "shell_mult_eff": cp.asnumpy(
                    self.d_shell_mult_eff
                ),
                "shell_bulk_mult_eff": cp.asnumpy(
                    self.d_shell_bulk_mult_eff
                ),
                "shell_gamma": cp.asnumpy(
                    self.d_shell_gamma
                ),
            }

            del d_lattice_matrix
            del d_vector_a
            del d_vector_b
            del d_stack_normal
            del d_stacking_inverse
            del normal_offsets
            del finite_denominator
            del bulk_denominator
            del valid_alpha
            del valid_finite
            del valid_bulk
            del valid_gamma

        transfer_seconds = (
            time.perf_counter()
            - transfer_start
        )

        print(
            "[DYNAMIC CUDA GEOMETRY] "
            f"gpu_compute={gpu_compute_seconds:.4f} s, "
            f"host_transfer={transfer_seconds:.4f} s, "
            f"shells={self.shell_count:,}"
        )

        return result