# -*- coding: utf-8 -*-
"""Numba kernels for fast G(r) accumulation.

Design goals
------------
- Keep the hot-path free of Python objects (no dicts, no strings).
- Avoid race conditions: parallelize over pair blocks, not shells.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Numba is required for the optimized G(r) kernels. Install numba>=0.57."
    ) from exc

# Avoid np.pi inside jitted code; keep constants module-level
_PI = 3.141592653589793
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * _PI)


@njit(cache=True, fastmath=True)
def _searchsorted_left(a: np.ndarray, x: float) -> int:
    """Numba-friendly np.searchsorted(a, x, side='left')."""
    lo = 0
    hi = a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, fastmath=True, parallel=True)
def accumulate_gaussians_by_pair(
    r_grid: np.ndarray,       # (n_r,)
    r_ij: np.ndarray,         # (n_shell,)
    sigma: np.ndarray,        # (n_shell,)
    A_ij: np.ndarray,         # (n_shell,)
    pair_offsets: np.ndarray, # (n_pair+1,)
    out_g: np.ndarray,        # (n_pair, n_r)
) -> None:
    """Accumulate shell Gaussians into out_g[pair, :].

    Shell arrays MUST be sorted by pair_id such that shells for pair p live in
    the slice [pair_offsets[p], pair_offsets[p+1]).
    """
    n_pair = pair_offsets.size - 1

    for pid in prange(n_pair):
        s0 = pair_offsets[pid]
        s1 = pair_offsets[pid + 1]
        if s1 <= s0:
            continue

        for s in range(s0, s1):
            rij = r_ij[s]
            sig = sigma[s]
            if sig <= 0.0:
                continue

            lo = rij - 5.0 * sig
            hi = rij + 5.0 * sig
            i0 = _searchsorted_left(r_grid, lo)
            i1 = _searchsorted_left(r_grid, hi)
            if i1 <= i0:
                continue

            pref = A_ij[s] * _INV_SQRT_2PI / sig
            inv_sig = 1.0 / sig

            for i in range(i0, i1):
                x = (r_grid[i] - rij) * inv_sig
                out_g[pid, i] += pref * math.exp(-0.5 * x * x)


@njit(cache=True, fastmath=True, parallel=True)
def accumulate_gaussians_by_shell_blocks(
    r_grid: np.ndarray,        # (n_r,)
    r_ij: np.ndarray,          # (n_shell,)
    sigma: np.ndarray,         # (n_shell,)
    A_ij: np.ndarray,          # (n_shell,)
    shell_pair_id: np.ndarray, # (n_shell,) int32 pair id for each shell (ordered pairs)
    out_partial: np.ndarray,   # (n_block, n_pair, n_r) thread-block partial sums
) -> None:
    """Accumulate shell Gaussians into per-block partial buffers (race-free).

    This kernel parallelizes over *shell blocks* rather than pair blocks, which is
    much more efficient when n_pair is small (e.g., single-element structures).
    Each parallel worker writes into its own out_partial[b, :, :] slice, so no
    atomics are needed. The caller should reduce:
        out_g = out_partial.sum(axis=0)
    to obtain out_g[pair, :].

    Parameters
    ----------
    shell_pair_id
        Pair id for each shell (ordered pairs, 0..n_pair-1).
    out_partial
        Zero-initialized output array shaped (n_block, n_pair, n_r).
        n_block should typically be numba.get_num_threads().
    """
    n_block = out_partial.shape[0]
    n_shell = r_ij.size
    if n_block <= 0 or n_shell <= 0:
        return

    # Static block partitioning for predictable work split
    block_size = (n_shell + n_block - 1) // n_block

    for b in prange(n_block):
        s0 = b * block_size
        s1 = s0 + block_size
        if s1 > n_shell:
            s1 = n_shell
        if s0 >= s1:
            continue

        for s in range(s0, s1):
            rij = r_ij[s]
            sig = sigma[s]
            if sig <= 0.0:
                continue

            lo = rij - 5.0 * sig
            hi = rij + 5.0 * sig
            i0 = _searchsorted_left(r_grid, lo)
            i1 = _searchsorted_left(r_grid, hi)
            if i1 <= i0:
                continue

            pid = shell_pair_id[s]
            pref = A_ij[s] * _INV_SQRT_2PI / sig
            inv_sig = 1.0 / sig

            for i in range(i0, i1):
                x = (r_grid[i] - rij) * inv_sig
                out_partial[b, pid, i] += pref * math.exp(-0.5 * x * x)


@njit(cache=True, fastmath=True, parallel=True)
def accumulate_pair_gamma_histogram_by_pair(
    r_grid: np.ndarray,
    r_ij: np.ndarray,
    shell_gamma: np.ndarray,
    weights: np.ndarray,
    pair_offsets: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> None:
    """
    Fast pair-resolved histogram of shell-dependent gamma values.

    Parallel over pair channels. No races because each pid writes only to
    numerator[pid, :] and denominator[pid, :].
    """
    n_pair = pair_offsets.size - 1
    n_r = r_grid.size

    if n_r <= 0:
        return

    r_min = r_grid[0]
    r_max = r_grid[n_r - 1]

    for pid in prange(n_pair):
        s0 = pair_offsets[pid]
        s1 = pair_offsets[pid + 1]

        for s in range(s0, s1):
            rr = r_ij[s]

            if rr < r_min or rr > r_max:
                continue

            gg = shell_gamma[s]
            ww = weights[s]

            if not np.isfinite(rr):
                continue

            if not np.isfinite(gg):
                continue

            if not np.isfinite(ww) or ww <= 0.0:
                continue

            idx_right = _searchsorted_left(r_grid, rr)

            if idx_right >= n_r:
                idx_right = n_r - 1

            idx_left = idx_right - 1

            if idx_left < 0:
                idx_left = 0

            d_left = abs(rr - r_grid[idx_left])
            d_right = abs(rr - r_grid[idx_right])

            if d_left <= d_right:
                idx = idx_left
            else:
                idx = idx_right

            numerator[pid, idx] += gg * ww
            denominator[pid, idx] += ww