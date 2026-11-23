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

# -----------------------------------------------------------------------------
# Per-shell stages of PDFCalculator.evaluate (CPU path), parallel over shells.
# Same arithmetic as the NumPy code they replace; they exist because those
# element-wise passes over millions of shells ran on one core.
# -----------------------------------------------------------------------------


@njit(cache=True, fastmath=False, parallel=True)
def shell_cart_and_distance(frac, MT, cart, r_out):
    """cart = frac @ MT ; r = |cart| (frac (n,3), MT (3,3))."""
    n = frac.shape[0]
    for i in prange(n):
        x = frac[i, 0] * MT[0, 0] + frac[i, 1] * MT[1, 0] + frac[i, 2] * MT[2, 0]
        y = frac[i, 0] * MT[0, 1] + frac[i, 1] * MT[1, 1] + frac[i, 2] * MT[2, 1]
        z = frac[i, 0] * MT[0, 2] + frac[i, 1] * MT[1, 2] + frac[i, 2] * MT[2, 2]
        cart[i, 0] = x
        cart[i, 1] = y
        cart[i, 2] = z
        r_out[i] = math.sqrt(x * x + y * y + z * z)


@njit(cache=True, fastmath=False, parallel=True)
def row_norms(cart, r_out):
    n = cart.shape[0]
    for i in prange(n):
        r_out[i] = math.sqrt(cart[i, 0] * cart[i, 0] + cart[i, 1] * cart[i, 1] + cart[i, 2] * cart[i, 2])


@njit(cache=True, fastmath=False, parallel=True)
def shell_correlation(
    r, upair_id, k_in, k_rank,
    lam_mat,          # (n_upair, max_k) NaN = not given; (n_upair, 0) if none
    lam_glob,         # (N_lambda,) global lambda_k by shell rank
    pair_d1, pair_d2, # (n_upair,) NaN = use global
    d1g, d2g,
    out,
):
    """corr = max(1 − λ, 0) with λ from lambda_<pair>_<k>, lambda_<k> or δ1/r + δ2/r²."""
    n = r.shape[0]
    n_k = lam_mat.shape[1]
    n_glob = lam_glob.shape[0]
    for i in prange(n):
        u = upair_id[i]
        done = False
        if n_k > 0:
            k = k_in[i]
            if k < n_k:
                v = lam_mat[u, k]
                if v == v:
                    out[i] = 1.0 - v
                    done = True
        if not done and k_rank[i] < n_glob:
            out[i] = 1.0 - lam_glob[k_rank[i]]
            done = True
        if not done:
            d1 = pair_d1[u]
            if d1 != d1:
                d1 = d1g
            d2 = pair_d2[u]
            if d2 != d2:
                d2 = d2g
            ri = r[i]
            out[i] = 1.0 - d1 / max(ri, 1e-12) - d2 / max(ri * ri, 1e-24)
        if out[i] < 0.0:
            out[i] = 0.0


@njit(cache=True, fastmath=False)
def _fstar_scalar(L, Re):
    x = L / Re
    if x <= 1.0:
        xm = max(x, 1e-300)
        return -math.log(xm) + 1.75 - math.log(2.0) + xm * xm / 6.0 - 32.0 * xm * xm * xm / (225.0 * _PI)
    return 512.0 / (90.0 * _PI * x) - (11.0 / 24.0 + 0.25 * math.log(2.0 * x)) / (x * x)


@njit(cache=True, fastmath=False, parallel=True)
def wilkens_strain_L2(r, chkl_unique, inv_dirs, Re, pref, out):
    """<ΔL²> = r²·pref·C_hkl·f*(r/Re) (pref = ρb²/4π), as the NumPy code (no clipping)."""
    n = r.shape[0]
    for i in prange(n):
        ri = r[i]
        out[i] = ri * ri * pref * chkl_unique[inv_dirs[i]] * _fstar_scalar(ri, Re)


@njit(cache=True, fastmath=False, parallel=True)
def pah_strain_L2(r, chkl_unique, inv_dirs, pah_a, pah_b, out):
    n = r.shape[0]
    for i in prange(n):
        ri = r[i]
        v = chkl_unique[inv_dirs[i]] * (pah_a * ri + pah_b * ri * ri)
        out[i] = v if v > 0.0 else 0.0


@njit(cache=True, fastmath=False, parallel=True)
def shell_sigma(r, corr, biso_by_species, alpha_id, beta_id, strain_L2, c_dyn, delta_g, delta_broad, out):
    """σ = sqrt(max(c·½(B_a + B_b)·corr + (δg r)² + (δb r)² + strain, 1e-24))."""
    n = r.shape[0]
    has_strain = strain_L2.shape[0] == n
    for i in prange(n):
        ri = r[i]
        b = 0.5 * (biso_by_species[alpha_id[i]] + biso_by_species[beta_id[i]])
        s2 = c_dyn * b * corr[i] + (delta_g * ri) ** 2 + (delta_broad * ri) ** 2
        if has_strain:
            s2 += strain_L2[i]
        out[i] = math.sqrt(max(s2, 1e-24))


@njit(cache=True, fastmath=False, parallel=True)
def shell_sigma_adp(
    r, corr, biso_by_species, alpha_id, beta_id, strain_L2, c_dyn, delta_g, delta_broad,
    frac, Gm, adp_c, adp_n, rot, beta_entry, out,
):
    """
    σ with anisotropic atoms. Per atom of a shell:
        isotropic    B/8π²                 (= ½·c_dyn·B with c_dyn = 1/4π²)
        anisotropic  wᵀ β w / 2π²,  w = Rᵀ v,  v = G f / r
    (f fractional pair vector, G direct metric, R rotation mapping the ASU
    site onto the atom, β of the ASU site). σ²_dyn = corr · (atom_a + atom_b).
    """
    n = r.shape[0]
    has_strain = strain_L2.shape[0] == n
    inv_2pi2 = 1.0 / (2.0 * _PI * _PI)
    for i in prange(n):
        ri = r[i]
        f0 = frac[i, 0]
        f1 = frac[i, 1]
        f2 = frac[i, 2]
        inv_r = 1.0 / max(ri, 1e-12)
        v0 = (Gm[0, 0] * f0 + Gm[0, 1] * f1 + Gm[0, 2] * f2) * inv_r
        v1 = (Gm[1, 0] * f0 + Gm[1, 1] * f1 + Gm[1, 2] * f2) * inv_r
        v2 = (Gm[2, 0] * f0 + Gm[2, 1] * f1 + Gm[2, 2] * f2) * inv_r
        dyn = 0.0
        for side in range(2):
            e = adp_c[i] if side == 0 else adp_n[i]
            if e < 0:
                sp = alpha_id[i] if side == 0 else beta_id[i]
                dyn += 0.5 * c_dyn * biso_by_species[sp]
            else:
                # w = Rᵀ v
                w0 = rot[e, 0, 0] * v0 + rot[e, 1, 0] * v1 + rot[e, 2, 0] * v2
                w1 = rot[e, 0, 1] * v0 + rot[e, 1, 1] * v1 + rot[e, 2, 1] * v2
                w2 = rot[e, 0, 2] * v0 + rot[e, 1, 2] * v1 + rot[e, 2, 2] * v2
                b = beta_entry[e]
                q = (b[0] * w0 * w0 + b[1] * w1 * w1 + b[2] * w2 * w2
                     + 2.0 * (b[3] * w0 * w1 + b[4] * w0 * w2 + b[5] * w1 * w2))
                dyn += q * inv_2pi2
        s2 = dyn * corr[i] + (delta_g * ri) ** 2 + (delta_broad * ri) ** 2
        if has_strain:
            s2 += strain_L2[i]
        out[i] = math.sqrt(max(s2, 1e-24))


@njit(cache=True, fastmath=False, parallel=True)
def cylinder_gamma_lognormal(cart, L, c_hat, d_grid, w_grid, mu, sigma, t_mean, out):
    """
    Σ_k w_k γ_cyl(L, D_k) per shell: the cylinder/disk common-volume factor
    (lognormal thickness, Malagutti/Scardi A_TD) averaged over a diameter
    quadrature. The thickness term does not depend on D and is computed once.
    Same expressions, validity mask, clipping to [0, 1] and non-finite → 0
    as PDFCalculator._cylinder_common_volume_shell_factor.
    """
    n = L.shape[0]
    nd = d_grid.shape[0]
    poly = sigma > 1e-12
    s2 = sigma * sigma
    sqrt2s = math.sqrt(2.0) * sigma
    e_mu = math.exp(mu + 0.5 * s2)
    pref_t = math.exp(-mu - 0.5 * s2) / _PI
    for i in prange(n):
        Li = L[i]
        z = abs(cart[i, 0] * c_hat[0] + cart[i, 1] * c_hat[1] + cart[i, 2] * c_hat[2])
        rp = math.sqrt(max(Li * Li - z * z, 0.0))
        if not (math.isfinite(Li) and math.isfinite(z) and math.isfinite(rp) and Li > 0.0):
            out[i] = 0.0
            continue
        if poly:
            if z <= 1e-300:
                tb = 2.0 * e_mu
            else:
                lnz = math.log(max(z, 1e-300))
                tb = e_mu * (1.0 + math.erf((mu + s2 - lnz) / sqrt2s)) + z * (math.erfc((mu - lnz) / sqrt2s) - 2.0)
            axial = 0.0
        else:
            tb = 0.0
            axial = 1.0 - z / t_mean
            if z >= t_mean or not math.isfinite(axial):
                axial = 0.0
            axial = min(max(axial, 0.0), 1.0)
        acc = 0.0
        for k in range(nd):
            D = d_grid[k]
            if not (rp < D):
                continue
            x = min(max(rp / D, 0.0), 1.0)
            root = math.sqrt(max(1.0 - x * x, 0.0))
            if poly:
                radial_bracket = max(D * math.acos(x) - rp * root, 0.0)
                g = pref_t / D * tb * radial_bracket
            else:
                g = (2.0 / _PI) * (math.acos(x) - x * root) * axial
            if not math.isfinite(g):
                g = 0.0
            g = min(max(g, 0.0), 1.0)
            acc += w_grid[k] * g
        out[i] = acc
