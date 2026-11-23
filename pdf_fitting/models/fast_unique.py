"""
Exact, faster replacement for ``np.unique(packed_rows, return_index=...,
return_inverse=..., return_counts=...)`` where ``packed_rows`` is an int64
matrix viewed as one np.void per row (the stacking topology compilers use
this to group millions of pair descriptors).

NumPy orders np.void values by memcmp of their bytes. For little-endian
int64 cells that is the order of each cell's bytes, least significant first;
byte-swapping every cell and reading it as uint64 gives numbers with exactly
that order. A stable column-wise lexsort of the swapped matrix therefore
reproduces np.unique's sorted keys, first-occurrence indices, inverse and
counts — bit for bit — so code that later searchsorted()s the packed keys
keeps working.

On an NVIDIA GPU (CuPy) the sort runs on the device when enabled with
``set_gpu(True)``; GPU use is serialised (compilers run in thread pools).
"""

from __future__ import annotations

import sys
import threading
from typing import Tuple

import numpy as np

_GPU = {"enabled": False}
_GPU_LOCK = threading.Lock()
_MIN_GPU_ROWS = 200_000


def set_gpu(enabled: bool) -> None:
    _GPU["enabled"] = bool(enabled)


def _packed_view(matrix: np.ndarray) -> np.ndarray:
    m = np.ascontiguousarray(matrix, dtype=np.int64)
    return m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).reshape(-1)


def _unique_gpu(keys_u64: np.ndarray):
    """(order-independent) first indices, inverse, counts computed on the device."""
    import cupy as cp

    with _GPU_LOCK:
        d = cp.asarray(keys_u64)
        n = d.shape[0]
        order = cp.lexsort(d.T[::-1])
        sk = d[order]
        new_group = cp.empty(n, dtype=cp.bool_)
        new_group[0] = True
        new_group[1:] = cp.any(sk[1:] != sk[:-1], axis=1)
        starts = cp.flatnonzero(new_group)
        first = order[starts]
        gid = cp.cumsum(new_group, dtype=cp.int64) - 1
        inv = cp.empty(n, dtype=cp.int64)
        inv[order] = gid
        counts = cp.diff(cp.concatenate([starts, cp.asarray([n], dtype=starts.dtype)]))
        out = (cp.asnumpy(first), cp.asnumpy(inv), cp.asnumpy(counts))
        del d, order, sk, new_group, starts, gid
        cp.get_default_memory_pool().free_all_blocks()
        return out


def unique_rows(
    matrix: np.ndarray,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Tuple:
    """
    Same outputs, in the same order, as
    np.unique(_packed_view(matrix), return_index=..., return_inverse=..., return_counts=...).
    """
    m = np.ascontiguousarray(matrix, dtype=np.int64)
    if m.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    n = m.shape[0]
    packed = _packed_view(m)
    if n == 0:
        out = [packed[:0]]
        if return_index:
            out.append(np.zeros(0, dtype=np.intp))
        if return_inverse:
            out.append(np.zeros(0, dtype=np.intp))
        if return_counts:
            out.append(np.zeros(0, dtype=np.intp))
        return tuple(out) if len(out) > 1 else out[0]
    if sys.byteorder != "little":
        res = np.unique(packed, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
        return res
    if not (_GPU["enabled"] and n >= _MIN_GPU_ROWS):
        # np.unique on the packed rows is faster than a CPU lexsort.
        return np.unique(packed, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    keys = m.byteswap().view(np.uint64)
    try:
        first, inv, counts = _unique_gpu(keys)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAST UNIQUE] GPU grouping unavailable ({exc}); using the CPU.")
        _GPU["enabled"] = False
        return np.unique(packed, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    out = [packed[first]]
    if return_index:
        out.append(first.astype(np.intp, copy=False))
    if return_inverse:
        out.append(inv.astype(np.intp, copy=False))
    if return_counts:
        out.append(counts.astype(np.intp, copy=False))
    return tuple(out) if len(out) > 1 else out[0]
