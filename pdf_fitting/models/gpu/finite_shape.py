from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    build_orientation_basis,
    shape_dimension_lengths,
)

from .backend import (
    get_cupy,
    gpu_memory_info,
    select_device,
)


_SHAPE_SPHERE = 0
_SHAPE_CYLINDER = 1
_SHAPE_DISK = 2
_SHAPE_ELLIPSOID = 3
_SHAPE_PRISM = 4


_CUDA_FINITE_COORDINATION_SOURCE = r"""
extern "C" __global__
void count_finite_coordination(
    const unsigned char* site_cell_mask,
    const int n_cells,
    const int n_side,
    const int* center_site,
    const int* neighbor_site,
    const int* image,
    const int row_offset,
    const int n_rows,
    long long* output_counts
) {
    const int local_row = blockIdx.x;

    if (local_row >= n_rows) {
        return;
    }

    const int row = row_offset + local_row;
    const int tid = threadIdx.x;

    const int center = center_site[row];
    const int neighbor = neighbor_site[row];

    const int dx = image[3 * row + 0];
    const int dy = image[3 * row + 1];
    const int dz = image[3 * row + 2];

    const int plane = n_side * n_side;

    unsigned long long local_count = 0ULL;

    for (
        int cell = tid;
        cell < n_cells;
        cell += blockDim.x
    ) {
        const int ix = cell / plane;
        const int remainder = cell - ix * plane;
        const int iy = remainder / n_side;
        const int iz = remainder - iy * n_side;

        const int ix2 = ix + dx;
        const int iy2 = iy + dy;
        const int iz2 = iz + dz;

        if (
            ix2 < 0
            || ix2 >= n_side
            || iy2 < 0
            || iy2 >= n_side
            || iz2 < 0
            || iz2 >= n_side
        ) {
            continue;
        }

        const int cell2 = (
            (ix2 * n_side + iy2)
            * n_side
            + iz2
        );

        const unsigned char center_inside = site_cell_mask[
            center * n_cells + cell
        ];

        const unsigned char neighbor_inside = site_cell_mask[
            neighbor * n_cells + cell2
        ];

        if (
            center_inside != 0
            && neighbor_inside != 0
        ) {
            local_count += 1ULL;
        }
    }

    __shared__ unsigned long long shared_counts[256];

    shared_counts[tid] = local_count;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared_counts[tid] += shared_counts[
                tid + stride
            ];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output_counts[row] = (
            (long long)
            shared_counts[0]
        );
    }
}
"""


class CudaFiniteShapeBackend:
    """
    CUDA implementation of the expensive finite crystallite construction.

    This first implementation performs on the GPU:

        1. atom-in-shape mask construction;
        2. direct atomistic finite coordination counting.

    The final shell grouping remains on CPU for now, using the existing
    build_shell_table_from_finite_counts() implementation. This ensures the
    CUDA result uses the exact same shell-table format as the CPU backend.

    The direct method is exact integer counting and is deliberately retained as
    the validation reference for a future FFT implementation.
    """

    def __init__(
        self,
        *,
        device_id: int = 0,
        memory_fraction: float = 0.70,
    ) -> None:
        self.cp = get_cupy()

        self.device_id = int(
            device_id
        )

        self.memory_fraction = float(
            memory_fraction
        )

        select_device(
            self.device_id
        )

        self._coordination_kernel = self.cp.RawKernel(
            _CUDA_FINITE_COORDINATION_SOURCE,
            "count_finite_coordination",
            options=(
                "--use_fast_math",
            ),
        )

    def _check_memory(
        self,
        estimated_bytes: int,
    ) -> None:
        memory = gpu_memory_info(
            self.device_id
        )

        free_bytes = int(
            memory["free"]
        )

        allowed_bytes = int(
            max(
                0.10,
                min(
                    self.memory_fraction,
                    0.95,
                ),
            )
            * free_bytes
        )

        if int(estimated_bytes) > allowed_bytes:
            raise MemoryError(
                "CUDA finite-shape construction does not fit within "
                "the configured GPU memory allowance. "
                f"Estimated={estimated_bytes / 1024 ** 3:.2f} GiB, "
                f"allowed={allowed_bytes / 1024 ** 3:.2f} GiB."
            )

    @staticmethod
    def _shape_code(
        shape_type: str,
    ) -> int:
        value = str(
            shape_type
            or ""
        ).strip().lower()

        if value in (
            "sphere",
            "spherical",
        ):
            return _SHAPE_SPHERE

        if value in (
            "cylinder",
        ):
            return _SHAPE_CYLINDER

        if value in (
            "disk",
            "disk-like",
            "disk_like",
        ):
            return _SHAPE_DISK

        if value in (
            "ellipsoid",
            "ellipsoidal",
        ):
            return _SHAPE_ELLIPSOID

        if value in (
            "prism",
            "rectangular_prism",
        ):
            return _SHAPE_PRISM

        return _SHAPE_CYLINDER

    def _shape_dimensions(
        self,
        shape_spec: CrystalliteShapeSpec,
        lattice_matrix: np.ndarray,
    ):
        """
        Return the geometry needed by the GPU atom-in-shape calculation.

        Returns
        -------
        shape_code
        e1, e2, e3
        radius_or_half_length_1
        radius_or_half_length_2
        half_length_3
        grid_n
        """
        shape_type = str(
            shape_spec.shape_type
        ).strip().lower()

        axis1_cells = float(
            getattr(
                shape_spec,
                "axis1_cells",
                shape_spec.diameter_cells,
            )
        )

        axis2_cells = float(
            getattr(
                shape_spec,
                "axis2_cells",
                shape_spec.diameter_cells,
            )
        )

        axis3_cells = float(
            getattr(
                shape_spec,
                "axis3_cells",
                shape_spec.height_cells,
            )
        )

        if shape_type in (
            "sphere",
            "spherical",
        ):
            axis2_cells = axis1_cells
            axis3_cells = axis1_cells

        elif shape_type in (
            "cylinder",
            "disk",
            "disk-like",
            "disk_like",
        ):
            axis2_cells = axis1_cells

        e1, e2, e3 = build_orientation_basis(
            lattice_matrix,
            axis_hkl=(
                int(
                    shape_spec.axis_h
                ),
                int(
                    shape_spec.axis_k
                ),
                int(
                    shape_spec.axis_l
                ),
            ),
            base1_hkl=(
                int(
                    shape_spec.base1_h
                ),
                int(
                    shape_spec.base1_k
                ),
                int(
                    shape_spec.base1_l
                ),
            ),
            base2_hkl=(
                int(
                    shape_spec.base2_h
                ),
                int(
                    shape_spec.base2_k
                ),
                int(
                    shape_spec.base2_l
                ),
            ),
        )

        base1_len, base2_len, axis_len = (
            shape_dimension_lengths(
                lattice_matrix,
                axis_hkl=(
                    int(
                        shape_spec.axis_h
                    ),
                    int(
                        shape_spec.axis_k
                    ),
                    int(
                        shape_spec.axis_l
                    ),
                ),
                base1_hkl=(
                    int(
                        shape_spec.base1_h
                    ),
                    int(
                        shape_spec.base1_k
                    ),
                    int(
                        shape_spec.base1_l
                    ),
                ),
                base2_hkl=(
                    int(
                        shape_spec.base2_h
                    ),
                    int(
                        shape_spec.base2_k
                    ),
                    int(
                        shape_spec.base2_l
                    ),
                ),
            )
        )

        length_1 = (
            axis1_cells
            * float(
                base1_len
            )
        )

        length_2 = (
            axis2_cells
            * float(
                base2_len
            )
        )

        length_3 = (
            axis3_cells
            * float(
                axis_len
            )
        )

        if shape_type in (
            "sphere",
            "spherical",
        ):
            # This matches the existing CPU shape_mask() sphere convention.
            radius = 0.25 * (
                length_1
                + length_2
            )

            half_1 = radius
            half_2 = radius
            half_3 = radius

        else:
            half_1 = 0.5 * length_1
            half_2 = 0.5 * length_2
            half_3 = 0.5 * length_3

        max_dim_cells = max(
            axis1_cells,
            axis2_cells,
            axis3_cells,
            1.0,
        )

        grid_n = int(
            np.ceil(
                max_dim_cells
            )
        ) + 3

        return (
            self._shape_code(
                shape_type
            ),
            np.asarray(
                e1,
                dtype=np.float32,
            ),
            np.asarray(
                e2,
                dtype=np.float32,
            ),
            np.asarray(
                e3,
                dtype=np.float32,
            ),
            float(
                half_1
            ),
            float(
                half_2
            ),
            float(
                half_3
            ),
            int(
                grid_n
            ),
        )

    def build_atom_mask(
        self,
        *,
        structure,
        shape_spec: CrystalliteShapeSpec,
    ) -> Dict[str, Any]:
        """
        Construct the finite atom-in-shape mask on CUDA.

        This matches the CPU atom-boundary implementation:

            atom_cart =
                (cell_translation + site_fractional_coordinate)
                @ lattice_matrix

        Then the atom is tested against sphere/cylinder/disk/ellipsoid/prism.

        Returns a dictionary compatible with the existing CPU shell-table
        assembly code, plus a CUDA device mask under:

            device_site_cell_mask
        """
        cp = self.cp

        lattice_matrix = np.asarray(
            structure.lattice.matrix,
            dtype=np.float32,
        )

        (
            shape_code,
            e1,
            e2,
            e3,
            half_1,
            half_2,
            half_3,
            grid_n,
        ) = self._shape_dimensions(
            shape_spec,
            lattice_matrix,
        )

        values = np.arange(
            -grid_n,
            grid_n + 1,
            dtype=np.int32,
        )

        aa, bb, cc = np.meshgrid(
            values,
            values,
            values,
            indexing="ij",
        )

        cell_trans = np.column_stack(
            (
                aa.ravel(),
                bb.ravel(),
                cc.ravel(),
            )
        ).astype(
            np.int32,
            copy=False,
        )

        frac_sites = np.asarray(
            structure.frac_coords,
            dtype=np.float32,
        )

        n_sites = int(
            frac_sites.shape[0]
        )

        n_cells = int(
            cell_trans.shape[0]
        )

        # Mask plus temporary Cartesian chunks.
        # This conservative estimate keeps enough GPU memory available.
        estimated_bytes = (
            n_sites
            * n_cells
            * 2
            + min(
                n_cells,
                262144,
            )
            * 3
            * 4
            * 4
        )

        self._check_memory(
            int(
                estimated_bytes
            )
        )

        print(
            "[CUDA FINITE] Building atom mask: "
            f"sites={n_sites:,}, "
            f"cells={n_cells:,}, "
            f"grid_n={grid_n}"
        )

        t0 = time.perf_counter()

        with cp.cuda.Device(
            self.device_id
        ):
            d_trans = cp.asarray(
                cell_trans,
                dtype=cp.float32,
            )

            d_frac_sites = cp.asarray(
                frac_sites,
                dtype=cp.float32,
            )

            d_matrix = cp.asarray(
                lattice_matrix,
                dtype=cp.float32,
            )

            d_e1 = cp.asarray(
                e1,
                dtype=cp.float32,
            )

            d_e2 = cp.asarray(
                e2,
                dtype=cp.float32,
            )

            d_e3 = cp.asarray(
                e3,
                dtype=cp.float32,
            )

            d_mask = cp.zeros(
                (
                    n_sites,
                    n_cells,
                ),
                dtype=cp.uint8,
            )

            d_cell_inside = cp.zeros(
                n_cells,
                dtype=cp.bool_,
            )

            # Chunk cells so large finite particles do not allocate one
            # enormous temporary n_cells x 3 Cartesian array per site.
            chunk_cells = min(
                n_cells,
                262144,
            )

            for site_index in range(
                n_sites
            ):
                site_frac = d_frac_sites[
                    site_index
                ]

                for start in range(
                    0,
                    n_cells,
                    chunk_cells,
                ):
                    stop = min(
                        start
                        + chunk_cells,
                        n_cells,
                    )

                    frac = (
                        d_trans[
                            start:stop
                        ]
                        + site_frac[
                            None,
                            :
                        ]
                    )

                    cart = frac @ d_matrix

                    x = cart @ d_e1
                    y = cart @ d_e2
                    z = cart @ d_e3

                    if shape_code == _SHAPE_SPHERE:
                        inside = (
                            x * x
                            + y * y
                            + z * z
                        ) <= (
                            np.float32(
                                half_1
                            )
                            * np.float32(
                                half_1
                            )
                        )

                    elif shape_code in (
                        _SHAPE_CYLINDER,
                        _SHAPE_DISK,
                    ):
                        radius = np.float32(
                            0.5
                            * (
                                half_1
                                + half_2
                            )
                        )

                        inside = (
                            x * x
                            + y * y
                            <= radius * radius
                        ) & (
                            cp.abs(
                                z
                            )
                            <= np.float32(
                                half_3
                            )
                        )

                    elif shape_code == _SHAPE_ELLIPSOID:
                        inside = (
                            (
                                x
                                / np.float32(
                                    max(
                                        half_1,
                                        1.0e-12,
                                    )
                                )
                            ) ** 2
                            + (
                                y
                                / np.float32(
                                    max(
                                        half_2,
                                        1.0e-12,
                                    )
                                )
                            ) ** 2
                            + (
                                z
                                / np.float32(
                                    max(
                                        half_3,
                                        1.0e-12,
                                    )
                                )
                            ) ** 2
                        ) <= np.float32(
                            1.0
                        )

                    elif shape_code == _SHAPE_PRISM:
                        inside = (
                            (
                                cp.abs(
                                    x
                                )
                                <= np.float32(
                                    half_1
                                )
                            )
                            & (
                                cp.abs(
                                    y
                                )
                                <= np.float32(
                                    half_2
                                )
                            )
                            & (
                                cp.abs(
                                    z
                                )
                                <= np.float32(
                                    half_3
                                )
                            )
                        )

                    else:
                        radius = np.float32(
                            0.5
                            * (
                                half_1
                                + half_2
                            )
                        )

                        inside = (
                            x * x
                            + y * y
                            <= radius * radius
                        ) & (
                            cp.abs(
                                z
                            )
                            <= np.float32(
                                half_3
                            )
                        )

                    d_mask[
                        site_index,
                        start:stop,
                    ] = inside.astype(
                        cp.uint8
                    )

                    d_cell_inside[
                        start:stop
                    ] |= inside

            # We only need per-site atom counts on the host for
            # n_alpha_inside. Do not download the complete mask here.
            site_counts = cp.asnumpy(
                cp.sum(
                    d_mask,
                    axis=1,
                    dtype=cp.int64,
                )
            )

            cell_inside = cp.asnumpy(
                d_cell_inside
            )

        site_species = [
            str(
                next(
                    iter(site.species)
                )
            ).strip()
            for site in structure.sites
        ]

        # Match finite_shape_shells.clean_element_name().
        import re

        cleaned_species = []

        for value in site_species:
            match = re.match(
                r"([A-Za-z]{1,2})",
                value,
            )

            if match is None:
                cleaned_species.append(
                    str(
                        value
                    )
                )
            else:
                symbol = match.group(
                    1
                )

                cleaned_species.append(
                    symbol.upper()
                    if len(symbol) == 1
                    else (
                        symbol[0].upper()
                        + symbol[1:].lower()
                    )
                )

        unique_species = sorted(
            set(
                cleaned_species
            )
        )

        species_index = {
            species: index
            for index, species in enumerate(
                unique_species
            )
        }

        site_species_id = np.asarray(
            [
                species_index[
                    species
                ]
                for species in cleaned_species
            ],
            dtype=np.int16,
        )

        n_alpha_inside = np.zeros(
            len(
                unique_species
            ),
            dtype=np.int64,
        )

        for site_index, species_id in enumerate(
            site_species_id
        ):
            n_alpha_inside[
                int(
                    species_id
                )
            ] += int(
                site_counts[
                    site_index
                ]
            )

        print(
            "[CUDA FINITE] Atom mask built in "
            f"{time.perf_counter() - t0:.3f} s; "
            f"inside cells={int(np.sum(cell_inside)):,}"
        )

        return {
            "boundary_mode": "atom",
            "grid_n": int(
                grid_n
            ),
            "cell_trans": cell_trans,
            "cell_inside": cell_inside.astype(
                np.bool_,
                copy=False,
            ),
            "site_cell_mask": None,
            "device_site_cell_mask": d_mask,
            "n_cells_inside": int(
                np.sum(
                    cell_inside
                )
            ),
            "n_alpha_inside": n_alpha_inside,
            "unique_species_mask": unique_species,
        }

    def count_finite_coordination(
        self,
        *,
        catalog: Dict[str, Any],
        mask_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Count finite atomistic pair coordination on CUDA.

        This is the direct GPU equivalent of:

            _count_finite_coordination_numba(...)

        Every catalogue row gets one exact integer count.
        """
        cp = self.cp

        if str(
            mask_data.get(
                "boundary_mode",
                "",
            )
        ).strip().lower() != "atom":
            raise NotImplementedError(
                "CUDA finite construction currently supports "
                "boundary_mode='atom' only."
            )

        d_mask = mask_data.get(
            "device_site_cell_mask",
            None,
        )

        if d_mask is None:
            raise RuntimeError(
                "CUDA finite atom mask is missing."
            )

        center_site = np.ascontiguousarray(
            catalog["center_site"],
            dtype=np.int32,
        )

        neighbor_site = np.ascontiguousarray(
            catalog["neighbor_site"],
            dtype=np.int32,
        )

        image = np.ascontiguousarray(
            catalog["image"],
            dtype=np.int32,
        )

        n_rows = int(
            center_site.size
        )

        n_cells = int(
            d_mask.shape[1]
        )

        grid_n = int(
            mask_data["grid_n"]
        )

        n_side = int(
            2
            * grid_n
            + 1
        )

        if n_rows <= 0:
            return np.zeros(
                0,
                dtype=np.int64,
            )

        # Device catalogue arrays plus output.
        estimated_bytes = (
            n_rows
            * (
                4
                + 4
                + 12
                + 8
            )
        )

        self._check_memory(
            int(
                estimated_bytes
            )
        )

        print(
            "[CUDA FINITE] Counting atomistic coordination: "
            f"catalog_rows={n_rows:,}, "
            f"grid_cells={n_cells:,}"
        )

        t0 = time.perf_counter()

        with cp.cuda.Device(
            self.device_id
        ):
            d_center = cp.asarray(
                center_site,
                dtype=cp.int32,
            )

            d_neighbor = cp.asarray(
                neighbor_site,
                dtype=cp.int32,
            )

            d_image = cp.asarray(
                image.reshape(
                    -1
                ),
                dtype=cp.int32,
            )

            d_counts = cp.zeros(
                n_rows,
                dtype=cp.int64,
            )

            threads = 256

            # Chunk block launches on Windows to avoid one extremely large
            # launch and to allow regular CUDA stream synchronization.
            max_rows_per_launch = 16384

            for row_offset in range(
                0,
                n_rows,
                max_rows_per_launch,
            ):
                rows_this_launch = min(
                    max_rows_per_launch,
                    n_rows
                    - row_offset,
                )

                self._coordination_kernel(
                    (
                        rows_this_launch,
                    ),
                    (
                        threads,
                    ),
                    (
                        d_mask,
                        np.int32(
                            n_cells
                        ),
                        np.int32(
                            n_side
                        ),
                        d_center,
                        d_neighbor,
                        d_image,
                        np.int32(
                            row_offset
                        ),
                        np.int32(
                            rows_this_launch
                        ),
                        d_counts,
                    ),
                )

            cp.cuda.get_current_stream().synchronize()

            counts = cp.asnumpy(
                d_counts
            )

            # Release large temporary catalogue arrays. The finite mask remains
            # resident until this method returns, because it belongs to
            # mask_data and is needed only during this construction step.
            del d_center
            del d_neighbor
            del d_image
            del d_counts

        print(
            "[CUDA FINITE] Coordination counts completed in "
            f"{time.perf_counter() - t0:.3f} s"
        )

        return np.asarray(
            counts,
            dtype=np.int64,
        )

    def build_mask_and_counts(
        self,
        *,
        structure,
        shape_spec: CrystalliteShapeSpec,
        catalog: Dict[str, Any],
    ):
        """
        Build the CUDA atom-in-shape mask and calculate exact atomistic
        finite coordination counts for every catalogue row.

        Returns
        -------
        mask_data
            Dictionary compatible with the existing CPU finite-shell table
            assembly function:

                build_shell_table_from_finite_counts(...)

        counts
            Integer finite coordination count for each catalogue row.
        """
        mask_data = self.build_atom_mask(
            structure=structure,
            shape_spec=shape_spec,
        )

        counts = self.count_finite_coordination(
            catalog=catalog,
            mask_data=mask_data,
        )

        return (
            mask_data,
            counts,
        )