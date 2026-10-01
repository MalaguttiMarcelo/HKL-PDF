from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .backend import (
    get_cupy,
    select_device,
)


_CUDA_GAUSSIAN_SOURCE = r"""
extern "C" __global__
void accumulate_gaussians(
    const float* r_grid,
    const int n_r,
    const float* r_ij,
    const float* sigma,
    const float* amplitude,
    const int* pair_id,
    const int n_shell,
    const int n_pair,
    float* output
) {
    const int shell_index =
        blockDim.x * blockIdx.x + threadIdx.x;

    if (shell_index >= n_shell) {
        return;
    }

    const float center = r_ij[shell_index];
    const float width = sigma[shell_index];

    if (!(width > 0.0f)) {
        return;
    }

    const int pair = pair_id[shell_index];

    if (pair < 0 || pair >= n_pair) {
        return;
    }

    const float lower = center - 5.0f * width;
    const float upper = center + 5.0f * width;

    int left = 0;
    int right = n_r;

    while (left < right) {
        const int middle = (left + right) >> 1;

        if (r_grid[middle] < lower) {
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
        const int middle = (left + right) >> 1;

        if (r_grid[middle] < upper) {
            left = middle + 1;
        }
        else {
            right = middle;
        }
    }

    const int last_index = left;

    const float inverse_width = 1.0f / width;

    const float prefactor =
        amplitude[shell_index]
        * 0.3989422804014327f
        * inverse_width;

    const long long pair_offset =
        ((long long) pair)
        * ((long long) n_r);

    for (
        int grid_index = first_index;
        grid_index < last_index;
        ++grid_index
    ) {
        const float normalized_distance =
            (r_grid[grid_index] - center)
            * inverse_width;

        const float contribution =
            prefactor
            * __expf(
                -0.5f
                * normalized_distance
                * normalized_distance
            );

        atomicAdd(
            &output[
                pair_offset
                + grid_index
            ],
            contribution
        );
    }
}
"""


class CudaResidentPdfBackend:
    """
    Resident CUDA backend for the shell-dependent PDF calculations.

    Static shell arrays stay on the GPU. For each PDF evaluation this backend
    calculates:

        Cartesian shell vectors
        shell distances r_ij
        directional Chkl
        PAH or Wilkens strain variance
        sigma
        A_ij
        Gaussian accumulation

    The correlation array is still prepared by the existing CPU code and copied
    to the GPU. This preserves the existing lambda/delta behavior.
    """

    def __init__(
        self,
        *,
        shell_frac: np.ndarray,
        shell_pair_id: np.ndarray,
        shell_alpha_id: np.ndarray,
        shell_beta_id: np.ndarray,
        shell_mult_eff: np.ndarray,
        strain_unique_dirs: Optional[np.ndarray],
        strain_inv_dirs: Optional[np.ndarray],
        n_pairs: int,
        device_id: int = 0,
    ):
        self.cp = get_cupy()
        self.device_id = int(device_id)

        select_device(
            self.device_id
        )

        self.kernel = self.cp.RawKernel(
            _CUDA_GAUSSIAN_SOURCE,
            "accumulate_gaussians",
            options=(
                "--use_fast_math",
            ),
        )

        self._feature_matrix_host_identity = None
        self._feature_matrix_device = None

        self._r_grid = None
        self._r_grid_shape = None
        self._output = None

        self.replace_shell_table(
            shell_frac=shell_frac,
            shell_pair_id=shell_pair_id,
            shell_alpha_id=shell_alpha_id,
            shell_beta_id=shell_beta_id,
            shell_mult_eff=shell_mult_eff,
            strain_unique_dirs=strain_unique_dirs,
            strain_inv_dirs=strain_inv_dirs,
            n_pairs=n_pairs,
        )

    def replace_shell_table(
        self,
        *,
        shell_frac: np.ndarray,
        shell_pair_id: np.ndarray,
        shell_alpha_id: np.ndarray,
        shell_beta_id: np.ndarray,
        shell_mult_eff: np.ndarray,
        strain_unique_dirs: Optional[np.ndarray],
        strain_inv_dirs: Optional[np.ndarray],
        n_pairs: int,
    ) -> None:
        cp = self.cp

        with cp.cuda.Device(
            self.device_id
        ):
            self.d_shell_frac = cp.asarray(
                np.ascontiguousarray(
                    shell_frac,
                    dtype=np.float32,
                ),
                dtype=cp.float32,
            )

            self.d_shell_pair_id = cp.asarray(
                np.ascontiguousarray(
                    shell_pair_id,
                    dtype=np.int32,
                ),
                dtype=cp.int32,
            )

            self.d_shell_alpha_id = cp.asarray(
                np.ascontiguousarray(
                    shell_alpha_id,
                    dtype=np.int32,
                ),
                dtype=cp.int32,
            )

            self.d_shell_beta_id = cp.asarray(
                np.ascontiguousarray(
                    shell_beta_id,
                    dtype=np.int32,
                ),
                dtype=cp.int32,
            )

            self.d_shell_mult_eff = cp.asarray(
                np.ascontiguousarray(
                    shell_mult_eff,
                    dtype=np.float32,
                ),
                dtype=cp.float32,
            )

            if strain_unique_dirs is not None:
                self.d_strain_unique_dirs = cp.asarray(
                    np.ascontiguousarray(
                        strain_unique_dirs,
                        dtype=np.int32,
                    ),
                    dtype=cp.int32,
                )
            else:
                self.d_strain_unique_dirs = None

            if strain_inv_dirs is not None:
                self.d_strain_inv_dirs = cp.asarray(
                    np.ascontiguousarray(
                        strain_inv_dirs,
                        dtype=np.int32,
                    ),
                    dtype=cp.int32,
                )
            else:
                self.d_strain_inv_dirs = None

        self.n_shell = int(
            self.d_shell_frac.shape[0]
        )

        self.n_pairs = int(
            n_pairs
        )

        self.d_cart = None
        self.d_r_ij = None
        self.d_sigma = None
        self.d_amplitude = None
        self.d_strain_l2 = None

        self._feature_matrix_host_identity = None
        self._feature_matrix_device = None
        self._output = None

    def compute_distances(
        self,
        lattice_matrix: np.ndarray,
        *,
        return_cartesian: bool = False,
    ):
        """
        Calculate shell Cartesian vectors and distances on the GPU.

        The host distance array is returned because the existing CPU
        lambda/delta correlation logic still consumes r_ij.
        """
        cp = self.cp

        lattice_matrix = np.ascontiguousarray(
            lattice_matrix,
            dtype=np.float32,
        )

        with cp.cuda.Device(
            self.device_id
        ):
            d_matrix = cp.asarray(
                lattice_matrix,
                dtype=cp.float32,
            )

            self.d_cart = (
                self.d_shell_frac
                @ d_matrix.T
            )

            self.d_r_ij = cp.sqrt(
                cp.sum(
                    self.d_cart
                    * self.d_cart,
                    axis=1,
                )
            ).astype(
                cp.float32,
                copy=False,
            )

            r_host = cp.asnumpy(
                self.d_r_ij
            )

            if return_cartesian:
                cart_host = cp.asnumpy(
                    self.d_cart
                )

                return (
                    r_host,
                    cart_host,
                )

        return r_host

    def _device_feature_matrix(
        self,
        feature_matrix: np.ndarray,
    ):
        cp = self.cp

        identity = (
            id(feature_matrix),
            tuple(feature_matrix.shape),
            str(feature_matrix.dtype),
        )

        if (
            self._feature_matrix_device is None
            or self._feature_matrix_host_identity != identity
        ):
            self._feature_matrix_device = cp.asarray(
                np.ascontiguousarray(
                    feature_matrix,
                    dtype=np.float32,
                ),
                dtype=cp.float32,
            )

            self._feature_matrix_host_identity = identity

        return self._feature_matrix_device

    def _invariant_chkl(
        self,
        *,
        feature_matrix: np.ndarray,
        edge_coefficients: np.ndarray,
        screw_coefficients: np.ndarray,
        edge_present: bool,
        screw_present: bool,
        edge_fraction: float,
    ):
        cp = self.cp

        d_feature_matrix = self._device_feature_matrix(
            feature_matrix
        )

        d_edge = cp.asarray(
            np.ascontiguousarray(
                edge_coefficients,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        d_screw = cp.asarray(
            np.ascontiguousarray(
                screw_coefficients,
                dtype=np.float32,
            ),
            dtype=cp.float32,
        )

        edge_values = (
            d_feature_matrix
            @ d_edge
        )

        screw_values = (
            d_feature_matrix
            @ d_screw
        )

        if edge_present and not screw_present:
            screw_values = edge_values

        elif screw_present and not edge_present:
            edge_values = screw_values

        chkl = (
            cp.float32(edge_fraction)
            * edge_values
            + cp.float32(
                1.0
                - float(edge_fraction)
            )
            * screw_values
        )

        return cp.maximum(
            chkl,
            cp.float32(0.0),
        ).astype(
            cp.float32,
            copy=False,
        )

    def _cubic_chkl(
        self,
        *,
        edge_fraction: float,
        cubic_coefficients: Tuple[
            float,
            float,
            float,
            float,
        ],
    ):
        cp = self.cp

        if self.d_strain_unique_dirs is None:
            return cp.zeros(
                0,
                dtype=cp.float32,
            )

        edge_a, edge_b, screw_a, screw_b = [
            float(value)
            for value in cubic_coefficients
        ]

        directions = self.d_strain_unique_dirs.astype(
            cp.float32,
            copy=False,
        )

        h = directions[:, 0]
        k = directions[:, 1]
        l = directions[:, 2]

        h2 = h * h
        k2 = k * k
        l2 = l * l

        denominator = (
            h2
            + k2
            + l2
        ) ** 2

        numerator = (
            h2 * k2
            + k2 * l2
            + l2 * h2
        )

        mixed_a = (
            float(edge_fraction)
            * edge_a
            + (
                1.0
                - float(edge_fraction)
            )
            * screw_a
        )

        mixed_b = (
            float(edge_fraction)
            * edge_b
            + (
                1.0
                - float(edge_fraction)
            )
            * screw_b
        )

        output = cp.zeros_like(
            denominator,
            dtype=cp.float32,
        )

        valid = denominator > 0.0

        output[valid] = (
            cp.float32(mixed_a)
            + cp.float32(mixed_b)
            * numerator[valid]
            / denominator[valid]
        )

        return cp.maximum(
            output,
            cp.float32(0.0),
        )

    def _wilkens_fstar(
        self,
        distance,
        Re: float,
    ):
        cp = self.cp

        x = (
            distance
            / cp.float32(
                max(
                    float(Re),
                    1.0e-30,
                )
            )
        )

        result = cp.empty_like(
            x,
            dtype=cp.float32,
        )

        low_mask = x <= 1.0

        if bool(
            cp.any(
                low_mask
            ).item()
        ):
            low_x = cp.maximum(
                x[low_mask],
                cp.float32(1.0e-30),
            )

            result[low_mask] = (
                -cp.log(
                    low_x
                )
                + cp.float32(
                    7.0 / 4.0
                    - np.log(2.0)
                )
                + low_x * low_x
                / cp.float32(6.0)
                - cp.float32(
                    32.0
                    / (
                        225.0
                        * np.pi
                    )
                )
                * low_x
                * low_x
                * low_x
            )

        high_mask = ~low_mask

        if bool(
            cp.any(
                high_mask
            ).item()
        ):
            high_x = x[
                high_mask
            ]

            result[high_mask] = (
                cp.float32(
                    512.0
                    / (
                        90.0
                        * np.pi
                    )
                )
                / high_x
                - (
                    cp.float32(
                        11.0 / 24.0
                    )
                    + cp.float32(0.25)
                    * cp.log(
                        cp.float32(2.0)
                        * high_x
                    )
                )
                / (
                    high_x
                    * high_x
                )
            )

        return result

    def prepare_sigma_amplitude(
        self,
        *,
        correlation: np.ndarray,
        biso_by_species: np.ndarray,
        density_by_species: np.ndarray,
        delta_g: float,
        delta_broad: float,
        directional_mode: str = "none",
        feature_matrix: Optional[np.ndarray] = None,
        edge_coefficients: Optional[np.ndarray] = None,
        screw_coefficients: Optional[np.ndarray] = None,
        edge_present: bool = False,
        screw_present: bool = False,
        edge_fraction: float = 0.5,
        cubic_coefficients: Optional[
            Tuple[
                float,
                float,
                float,
                float,
            ]
        ] = None,
        rho: float = 0.0,
        Re: float = 100.0,
        burgers_magnitude: float = 0.0,
        pah_a: float = 0.0,
        pah_b: float = 0.0,
        shell_amplitude_factor: Optional[np.ndarray] = None,
    ) -> None:
        """
        Calculate directional strain, sigma and A_ij on the GPU.
        """
        cp = self.cp

        if self.d_r_ij is None:
            raise RuntimeError(
                "compute_distances() must be called before "
                "prepare_sigma_amplitude()."
            )

        correlation = np.ascontiguousarray(
            correlation,
            dtype=np.float32,
        )

        biso_by_species = np.ascontiguousarray(
            biso_by_species,
            dtype=np.float32,
        )

        density_by_species = np.ascontiguousarray(
            density_by_species,
            dtype=np.float32,
        )

        if correlation.size != self.n_shell:
            raise ValueError(
                "The correlation array does not match the shell count."
            )

        with cp.cuda.Device(
            self.device_id
        ):
            d_correlation = cp.asarray(
                correlation,
                dtype=cp.float32,
            )

            d_biso = cp.asarray(
                biso_by_species,
                dtype=cp.float32,
            )

            d_density = cp.asarray(
                density_by_species,
                dtype=cp.float32,
            )

            shell_biso = (
                cp.float32(0.5)
                * (
                    d_biso[
                        self.d_shell_alpha_id
                    ]
                    + d_biso[
                        self.d_shell_beta_id
                    ]
                )
            )

            self.d_strain_l2 = cp.zeros(
                self.n_shell,
                dtype=cp.float32,
            )

            mode = str(
                directional_mode
                or "none"
            ).strip().lower()

            if (
                mode in (
                    "pah",
                    "wilkens",
                )
                and self.d_strain_inv_dirs is not None
            ):
                if cubic_coefficients is not None:
                    chkl_unique = self._cubic_chkl(
                        edge_fraction=edge_fraction,
                        cubic_coefficients=cubic_coefficients,
                    )

                elif (
                    feature_matrix is not None
                    and edge_coefficients is not None
                    and screw_coefficients is not None
                ):
                    chkl_unique = self._invariant_chkl(
                        feature_matrix=feature_matrix,
                        edge_coefficients=edge_coefficients,
                        screw_coefficients=screw_coefficients,
                        edge_present=bool(
                            edge_present
                        ),
                        screw_present=bool(
                            screw_present
                        ),
                        edge_fraction=edge_fraction,
                    )

                else:
                    chkl_unique = None

                if chkl_unique is not None:
                    strain_factor = chkl_unique[
                        self.d_strain_inv_dirs
                    ].astype(
                        cp.float32,
                        copy=False,
                    )

                    if mode == "pah":
                        self.d_strain_l2 = (
                            strain_factor
                            * (
                                cp.float32(
                                    pah_a
                                )
                                * self.d_r_ij
                                + cp.float32(
                                    pah_b
                                )
                                * self.d_r_ij
                                * self.d_r_ij
                            )
                        )

                    elif mode == "wilkens":
                        fstar_values = self._wilkens_fstar(
                            self.d_r_ij,
                            Re,
                        )

                        prefactor = (
                            cp.float32(
                                rho
                            )
                            * cp.float32(
                                burgers_magnitude
                            )
                            * cp.float32(
                                burgers_magnitude
                            )
                            / cp.float32(
                                4.0
                                * np.pi
                            )
                        )

                        epsilon_squared = (
                            prefactor
                            * strain_factor
                            * fstar_values
                        )

                        self.d_strain_l2 = (
                            self.d_r_ij
                            * self.d_r_ij
                            * epsilon_squared
                        )

                    self.d_strain_l2 = cp.maximum(
                        self.d_strain_l2,
                        cp.float32(0.0),
                    )

            sigma_squared = (
                cp.float32(
                    1.0
                    / (
                        4.0
                        * np.pi
                        * np.pi
                    )
                )
                * shell_biso
                * d_correlation
                + (
                    cp.float32(
                        delta_g
                    )
                    * self.d_r_ij
                ) ** 2
                + (
                    cp.float32(
                        delta_broad
                    )
                    * self.d_r_ij
                ) ** 2
                + self.d_strain_l2
            )

            self.d_sigma = cp.sqrt(
                cp.maximum(
                    sigma_squared,
                    cp.float32(1.0e-24),
                )
            ).astype(
                cp.float32,
                copy=False,
            )

            neighbor_density = d_density[
                self.d_shell_beta_id
            ]

            distance_squared = cp.maximum(
                self.d_r_ij
                * self.d_r_ij,
                cp.float32(1.0e-24),
            )

            neighbor_density = cp.maximum(
                neighbor_density,
                cp.float32(1.0e-30),
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

            if shell_amplitude_factor is not None:
                self.d_amplitude *= cp.asarray(
                    np.ascontiguousarray(
                        shell_amplitude_factor,
                        dtype=np.float32,
                    ),
                    dtype=cp.float32,
                )

    def accumulate(
        self,
        r_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Accumulate prepared shell Gaussians into pair-resolved g(r).
        """
        cp = self.cp

        if (
            self.d_r_ij is None
            or self.d_sigma is None
            or self.d_amplitude is None
        ):
            raise RuntimeError(
                "GPU shell physics has not been prepared."
            )

        r_grid = np.ascontiguousarray(
            r_grid,
            dtype=np.float32,
        )

        with cp.cuda.Device(
            self.device_id
        ):
            if (
                self._r_grid is None
                or self._r_grid_shape != r_grid.shape
            ):
                self._r_grid = cp.asarray(
                    r_grid,
                    dtype=cp.float32,
                )

                self._r_grid_shape = tuple(
                    r_grid.shape
                )

            else:
                self._r_grid.set(
                    r_grid
                )

            output_shape = (
                self.n_pairs,
                int(
                    r_grid.size
                ),
            )

            if (
                self._output is None
                or self._output.shape != output_shape
            ):
                self._output = cp.zeros(
                    output_shape,
                    dtype=cp.float32,
                )
            else:
                self._output.fill(
                    cp.float32(0.0)
                )

            threads = 256

            blocks = (
                self.n_shell
                + threads
                - 1
            ) // threads

            self.kernel(
                (
                    blocks,
                ),
                (
                    threads,
                ),
                (
                    self._r_grid,
                    np.int32(
                        r_grid.size
                    ),
                    self.d_r_ij,
                    self.d_sigma,
                    self.d_amplitude,
                    self.d_shell_pair_id,
                    np.int32(
                        self.n_shell
                    ),
                    np.int32(
                        self.n_pairs
                    ),
                    self._output,
                ),
            )

            cp.cuda.get_current_stream().synchronize()

            return cp.asnumpy(
                self._output
            )