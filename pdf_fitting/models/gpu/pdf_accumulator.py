from __future__ import annotations

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


class CudaPdfAccumulator:
    """
    CUDA Gaussian shell accumulator.

    Static pair IDs remain on the GPU. Current distances, widths and amplitudes
    are copied for each model evaluation.
    """

    def __init__(
        self,
        shell_pair_id: np.ndarray,
        n_pairs: int,
        *,
        device_id: int = 0,
    ):
        cp = get_cupy()

        select_device(
            device_id
        )

        self.cp = cp
        self.device_id = int(
            device_id
        )

        self.kernel = cp.RawKernel(
            _CUDA_GAUSSIAN_SOURCE,
            "accumulate_gaussians",
            options=(
                "--use_fast_math",
            ),
        )

        self.shell_pair_id = None
        self.n_shell = 0
        self.n_pairs = 0

        self._r_grid = None
        self._r_grid_shape = None

        self._r_ij = None
        self._sigma = None
        self._amplitude = None
        self._output = None

        self.replace_shell_table(
            shell_pair_id,
            n_pairs,
        )

    def replace_shell_table(
        self,
        shell_pair_id: np.ndarray,
        n_pairs: int,
    ) -> None:
        cp = self.cp

        with cp.cuda.Device(
            self.device_id
        ):
            self.shell_pair_id = cp.asarray(
                np.ascontiguousarray(
                    shell_pair_id,
                    dtype=np.int32,
                ),
                dtype=cp.int32,
            )

        self.n_shell = int(
            self.shell_pair_id.size
        )

        self.n_pairs = int(
            n_pairs
        )

        self._r_ij = None
        self._sigma = None
        self._amplitude = None
        self._output = None

    def accumulate(
        self,
        r_grid: np.ndarray,
        r_ij: np.ndarray,
        sigma: np.ndarray,
        amplitude: np.ndarray,
    ) -> np.ndarray:
        cp = self.cp

        r_grid = np.ascontiguousarray(
            r_grid,
            dtype=np.float32,
        )

        r_ij = np.ascontiguousarray(
            r_ij,
            dtype=np.float32,
        )

        sigma = np.ascontiguousarray(
            sigma,
            dtype=np.float32,
        )

        amplitude = np.ascontiguousarray(
            amplitude,
            dtype=np.float32,
        )

        if (
            r_ij.size != self.n_shell
            or sigma.size != self.n_shell
            or amplitude.size != self.n_shell
        ):
            raise ValueError(
                "CUDA PDF arrays do not match the current shell-table length."
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

            if (
                self._r_ij is None
                or self._r_ij.size != self.n_shell
            ):
                self._r_ij = cp.empty(
                    self.n_shell,
                    dtype=cp.float32,
                )

                self._sigma = cp.empty(
                    self.n_shell,
                    dtype=cp.float32,
                )

                self._amplitude = cp.empty(
                    self.n_shell,
                    dtype=cp.float32,
                )

            self._r_ij.set(
                r_ij
            )

            self._sigma.set(
                sigma
            )

            self._amplitude.set(
                amplitude
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
                    0.0
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
                    self._r_ij,
                    self._sigma,
                    self._amplitude,
                    self.shell_pair_id,
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

            result = cp.asnumpy(
                self._output
            )

        return result