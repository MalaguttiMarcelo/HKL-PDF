from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .backend import (
    get_cupy,
    gpu_memory_info,
    select_device,
)


def _check_gpu_memory(
    estimated_bytes: int,
    *,
    device_id: int,
    memory_fraction: float,
) -> None:
    """
    Raise MemoryError when the estimated temporary requirement is too large.

    This is intentionally conservative because CuPy sorting requires temporary
    workspace in addition to the explicitly allocated arrays.
    """
    memory = gpu_memory_info(
        device_id
    )

    free_bytes = int(
        memory["free"]
    )

    allowed_bytes = int(
        max(
            0.10,
            min(
                float(memory_fraction),
                0.95,
            ),
        )
        * free_bytes
    )

    if int(estimated_bytes) > allowed_bytes:
        raise MemoryError(
            "CUDA shell grouping does not fit within the configured GPU "
            f"memory allowance. Estimated={estimated_bytes / 1024 ** 3:.2f} GiB, "
            f"allowed={allowed_bytes / 1024 ** 3:.2f} GiB."
        )


def group_shell_rows_cuda(
    *,
    lambda_alpha_id: np.ndarray,
    lambda_beta_id: np.ndarray,
    distance_integer: np.ndarray,
    direction_h: np.ndarray,
    direction_k: np.ndarray,
    direction_l: np.ndarray,
    alpha_weight_row: np.ndarray,
    device_id: int = 0,
    memory_fraction: float = 0.70,
) -> Dict[str, np.ndarray]:
    """
    Group shell candidate rows on CUDA.

    Grouping key:

        lambda alpha id
        lambda beta id
        integer distance bin
        reduced direction h
        reduced direction k
        reduced direction l

    Returns
    -------
    {
        "first_idx": representative original row for every group,
        "counts": raw multiplicity for every group,
        "shell_mult_eff": weighted multiplicity for every group,
        "lambda_alpha": grouped lambda-alpha ids,
        "lambda_beta": grouped lambda-beta ids,
        "distance_integer": grouped distance bins,
        "direction_h": grouped h direction,
        "direction_k": grouped k direction,
        "direction_l": grouped l direction,
    }
    """
    cp = get_cupy()

    select_device(
        device_id
    )

    lambda_alpha_id = np.ascontiguousarray(
        lambda_alpha_id,
        dtype=np.int32,
    )

    lambda_beta_id = np.ascontiguousarray(
        lambda_beta_id,
        dtype=np.int32,
    )

    distance_integer = np.ascontiguousarray(
        distance_integer,
        dtype=np.int32,
    )

    direction_h = np.ascontiguousarray(
        direction_h,
        dtype=np.int32,
    )

    direction_k = np.ascontiguousarray(
        direction_k,
        dtype=np.int32,
    )

    direction_l = np.ascontiguousarray(
        direction_l,
        dtype=np.int32,
    )

    alpha_weight_row = np.ascontiguousarray(
        alpha_weight_row,
        dtype=np.float32,
    )

    row_count = int(
        distance_integer.size
    )

    if row_count <= 0:
        return {
            "first_idx": np.zeros(0, dtype=np.int64),
            "counts": np.zeros(0, dtype=np.int32),
            "shell_mult_eff": np.zeros(0, dtype=np.float32),
            "lambda_alpha": np.zeros(0, dtype=np.int32),
            "lambda_beta": np.zeros(0, dtype=np.int32),
            "distance_integer": np.zeros(0, dtype=np.int32),
            "direction_h": np.zeros(0, dtype=np.int32),
            "direction_k": np.zeros(0, dtype=np.int32),
            "direction_l": np.zeros(0, dtype=np.int32),
        }

    input_arrays = (
        lambda_alpha_id,
        lambda_beta_id,
        distance_integer,
        direction_h,
        direction_k,
        direction_l,
        alpha_weight_row,
    )

    if any(
        array.size != row_count
        for array in input_arrays
    ):
        raise ValueError(
            "All CUDA shell-grouping input arrays must have the same length."
        )

    # Conservative estimate:
    # inputs + sorting index + sorted temporaries + group IDs + sort workspace.
    estimated_bytes = int(
        row_count
        * 100
    )

    _check_gpu_memory(
        estimated_bytes,
        device_id=device_id,
        memory_fraction=memory_fraction,
    )

    with cp.cuda.Device(
        int(device_id)
    ):
        d_alpha = cp.asarray(
            lambda_alpha_id,
            dtype=cp.int32,
        )

        d_beta = cp.asarray(
            lambda_beta_id,
            dtype=cp.int32,
        )

        d_distance = cp.asarray(
            distance_integer,
            dtype=cp.int32,
        )

        d_h = cp.asarray(
            direction_h,
            dtype=cp.int32,
        )

        d_k = cp.asarray(
            direction_k,
            dtype=cp.int32,
        )

        d_l = cp.asarray(
            direction_l,
            dtype=cp.int32,
        )

        d_weight = cp.asarray(
            alpha_weight_row,
            dtype=cp.float32,
        )

        # CuPy lexsort requires a stacked 2D CuPy array.
        #
        # lexsort uses the final row as the primary key. Therefore this
        # produces the same ordering as the CPU structured key:
        #
        #   lambda_alpha_id
        #   lambda_beta_id
        #   distance_integer
        #   direction_h
        #   direction_k
        #   direction_l
        sort_keys = cp.stack(
            (
                d_l,
                d_k,
                d_h,
                d_distance,
                d_beta,
                d_alpha,
            ),
            axis=0,
        )

        order = cp.lexsort(
            sort_keys
        )

        alpha_sorted = d_alpha[
            order
        ]

        beta_sorted = d_beta[
            order
        ]

        distance_sorted = d_distance[
            order
        ]

        h_sorted = d_h[
            order
        ]

        k_sorted = d_k[
            order
        ]

        l_sorted = d_l[
            order
        ]

        weight_sorted = d_weight[
            order
        ]

        new_group = cp.empty(
            row_count,
            dtype=cp.bool_,
        )

        new_group[0] = True

        if row_count > 1:
            new_group[1:] = (
                (alpha_sorted[1:] != alpha_sorted[:-1])
                | (beta_sorted[1:] != beta_sorted[:-1])
                | (distance_sorted[1:] != distance_sorted[:-1])
                | (h_sorted[1:] != h_sorted[:-1])
                | (k_sorted[1:] != k_sorted[:-1])
                | (l_sorted[1:] != l_sorted[:-1])
            )

        first_sorted_positions = cp.flatnonzero(
            new_group
        ).astype(
            cp.int64,
            copy=False,
        )

        group_count = int(
            first_sorted_positions.size
        )

        group_id_sorted = (
            cp.cumsum(
                new_group,
                dtype=cp.int32,
            )
            - cp.int32(1)
        )

        group_end_positions = cp.empty(
            group_count,
            dtype=cp.int64,
        )

        if group_count > 1:
            group_end_positions[:-1] = (
                first_sorted_positions[1:]
            )

        group_end_positions[-1] = row_count

        counts = (
            group_end_positions
            - first_sorted_positions
        ).astype(
            cp.int32,
            copy=False,
        )

        shell_mult_eff = cp.bincount(
            group_id_sorted,
            weights=weight_sorted,
            minlength=group_count,
        ).astype(
            cp.float32,
            copy=False,
        )

        first_original_rows = order[
            first_sorted_positions
        ].astype(
            cp.int64,
            copy=False,
        )

        result = {
            "first_idx": cp.asnumpy(
                first_original_rows
            ),
            "counts": cp.asnumpy(
                counts
            ),
            "shell_mult_eff": cp.asnumpy(
                shell_mult_eff
            ),
            "lambda_alpha": cp.asnumpy(
                alpha_sorted[
                    first_sorted_positions
                ]
            ),
            "lambda_beta": cp.asnumpy(
                beta_sorted[
                    first_sorted_positions
                ]
            ),
            "distance_integer": cp.asnumpy(
                distance_sorted[
                    first_sorted_positions
                ]
            ),
            "direction_h": cp.asnumpy(
                h_sorted[
                    first_sorted_positions
                ]
            ),
            "direction_k": cp.asnumpy(
                k_sorted[
                    first_sorted_positions
                ]
            ),
            "direction_l": cp.asnumpy(
                l_sorted[
                    first_sorted_positions
                ]
            ),
        }

        del d_alpha
        del d_beta
        del d_distance
        del d_h
        del d_k
        del d_l
        del d_weight
        del sort_keys
        del order
        del alpha_sorted
        del beta_sorted
        del distance_sorted
        del h_sorted
        del k_sorted
        del l_sorted
        del weight_sorted
        del new_group
        del first_sorted_positions
        del group_id_sorted
        del group_end_positions
        del counts
        del shell_mult_eff
        del first_original_rows

        cp.get_default_memory_pool().free_all_blocks()

    return result


def unique_int3_cuda(
    rows: np.ndarray,
    *,
    device_id: int = 0,
    memory_fraction: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CUDA equivalent of:

        np.unique(rows, axis=0, return_inverse=True)

    for an integer array shaped (N, 3).
    """
    cp = get_cupy()

    select_device(
        device_id
    )

    rows = np.ascontiguousarray(
        rows,
        dtype=np.int32,
    )

    if (
        rows.ndim != 2
        or rows.shape[1] != 3
    ):
        raise ValueError(
            "unique_int3_cuda expects an array shaped (N, 3)."
        )

    row_count = int(
        rows.shape[0]
    )

    if row_count <= 0:
        return (
            np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.int32,
            ),
            np.zeros(
                0,
                dtype=np.int32,
            ),
        )

    estimated_bytes = int(
        row_count
        * 55
    )

    _check_gpu_memory(
        estimated_bytes,
        device_id=device_id,
        memory_fraction=memory_fraction,
    )

    with cp.cuda.Device(
        int(device_id)
    ):
        d_rows = cp.asarray(
            rows,
            dtype=cp.int32,
        )

        # CuPy lexsort expects a stacked 2D array in some versions.
        # The final row is the primary key, giving h, k, l ordering.
        sort_keys = cp.stack(
            (
                d_rows[:, 2],
                d_rows[:, 1],
                d_rows[:, 0],
            ),
            axis=0,
        )

        order = cp.lexsort(
            sort_keys
        )

        sorted_rows = d_rows[
            order
        ]

        new_group = cp.empty(
            row_count,
            dtype=cp.bool_,
        )

        new_group[0] = True

        if row_count > 1:
            new_group[1:] = cp.any(
                sorted_rows[1:]
                != sorted_rows[:-1],
                axis=1,
            )

        first_sorted_positions = cp.flatnonzero(
            new_group
        ).astype(
            cp.int64,
            copy=False,
        )

        unique_rows = sorted_rows[
            first_sorted_positions
        ]

        group_id_sorted = (
            cp.cumsum(
                new_group,
                dtype=cp.int32,
            )
            - cp.int32(1)
        )

        inverse = cp.empty(
            row_count,
            dtype=cp.int32,
        )

        inverse[
            order
        ] = group_id_sorted

        result_unique = cp.asnumpy(
            unique_rows
        )

        result_inverse = cp.asnumpy(
            inverse
        )

        del d_rows
        del sort_keys
        del order
        del sorted_rows
        del new_group
        del first_sorted_positions
        del unique_rows
        del group_id_sorted
        del inverse

        cp.get_default_memory_pool().free_all_blocks()

    return (
        result_unique.astype(
            np.int32,
            copy=False,
        ),
        result_inverse.astype(
            np.int32,
            copy=False,
        ),
    )