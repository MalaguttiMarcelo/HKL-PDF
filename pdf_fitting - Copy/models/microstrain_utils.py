# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
from math import gcd, ceil, log, sqrt, pi
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re

# ---------------------------------------------------------------------------
# Legacy cubic contrast factor coefficients (kept for backward compatibility)
# ---------------------------------------------------------------------------
# These are used when no invariant coefficients (E1..En) are provided.
CEdgeA = 0.265280
CEdgeB = -0.355950
CScrewA = 0.307288
CScrewB = -0.819979

# ---------------------------------------------------------------------------
# Generic (non-cubic) Chkl via invariant expansion (TOPAS-like macro)
# The project can provide two sets of coefficients:
#   - screw:  E1..En
#   - edge:   E1..En
# with n determined by symmetry (space group + cell setting).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0


def _deg2rad(x: float) -> float:
    return float(np.deg2rad(x))


def _is_close(x: float, y: float, tol: float = 1e-6) -> bool:
    return abs(float(x) - float(y)) <= tol


@lru_cache(maxsize=128)
def _reciprocal_params_cached(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> Tuple[float, float, float, float, float, float]:
    """Cached reciprocal parameters.

    NOTE: caching is very effective during refinements because (a,b,c,alpha,beta,gamma)
    change slowly and many shells share the same cell in a given evaluation.
    """
    a, b, c = float(a), float(b), float(c)
    al, be, ga = map(_deg2rad, (alpha, beta, gamma))
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    sa, sb, sg = np.sin(al), np.sin(be), np.sin(ga)

    V = a * b * c * np.sqrt(1.0 + 2.0 * ca * cb * cg - ca**2 - cb**2 - cg**2)
    if V <= 0:
        raise ValueError("Non-positive unit-cell volume computed from cell parameters.")

    astar = (b * c * sa) / V
    bstar = (a * c * sb) / V
    cstar = (a * b * sg) / V

    cos_al_star = (cb * cg - ca) / (sb * sg)
    cos_be_star = (ca * cg - cb) / (sa * sg)
    cos_ga_star = (ca * cb - cg) / (sa * sb)

    cos_al_star = np.clip(cos_al_star, -1.0, 1.0)
    cos_be_star = np.clip(cos_be_star, -1.0, 1.0)
    cos_ga_star = np.clip(cos_ga_star, -1.0, 1.0)

    al_star = float(np.arccos(cos_al_star))
    be_star = float(np.arccos(cos_be_star))
    ga_star = float(np.arccos(cos_ga_star))
    return float(astar), float(bstar), float(cstar), al_star, be_star, ga_star


def reciprocal_params(cell: Cell) -> Tuple[float, float, float, float, float, float]:
    """Return (a*, b*, c*, alpha*, beta*, gamma*) in reciprocal space."""
    return _reciprocal_params_cached(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)


def HKL_reciprocal_unit_components(h: int, k: int, l: int, cell: Cell) -> Tuple[float, float, float]:
    """Return direction cosines (H,K,L) of reciprocal vector g(hkl) in the reciprocal metric.

    Kept for backward compatibility, but for HKL-only invariant evaluation you
    should use feature_vector_hkl(), which internally normalizes HKL via metric tensor.
    """
    astar, bstar, cstar, al_s, be_s, ga_s = reciprocal_params(cell)
    ha = h * astar
    kb = k * bstar
    lc = l * cstar

    g2 = (
        ha**2 + kb**2 + lc**2
        + 2.0 * ha * kb * np.cos(ga_s)
        + 2.0 * ha * lc * np.cos(be_s)
        + 2.0 * kb * lc * np.cos(al_s)
    )
    if g2 <= 0:
        raise ValueError(f"Non-positive |g|^2 for hkl=({h},{k},{l}). Check cell parameters.")
    g = float(np.sqrt(g2))
    return float(ha / g), float(kb / g), float(lc / g)


def d_spacing(cell: Cell, h: int, k: int, l: int) -> float:
    """Interplanar spacing d(hkl) using the reciprocal metric (2π omitted).

    We build |g| from the reciprocal parameters and return d = 1/|g|.
    This matches the convention already used by HKL_reciprocal_unit_components().
    """
    astar, bstar, cstar, al_s, be_s, ga_s = reciprocal_params(cell)
    ha = float(h) * astar
    kb = float(k) * bstar
    lc = float(l) * cstar
    g2 = (
        ha**2 + kb**2 + lc**2
        + 2.0 * ha * kb * np.cos(ga_s)
        + 2.0 * ha * lc * np.cos(be_s)
        + 2.0 * kb * lc * np.cos(al_s)
    )
    if g2 <= 0.0:
        raise ValueError(f"Non-positive |g|^2 for hkl=({h},{k},{l}). Check cell parameters.")
    g = float(np.sqrt(g2))
    return 1.0 / g


def _normalized_hkl_components(h: int, k: int, l: int, cell: Cell) -> Tuple[float, float, float]:
    """Return normalized reciprocal-axis components (H,K,L) from integer HKL.

    This is intentionally *HKL-only*:
      - Uses only (h,k,l) and the reciprocal metric (via the unit cell)
      - Does not require any external orientation / Cartesian basis

    Implementation detail:
      H = (h a*) / |g|, K = (k b*) / |g|, L = (l c*) / |g|
    with |g| computed from the reciprocal metric (same as d_spacing()).
    """
    astar, bstar, cstar, al_s, be_s, ga_s = reciprocal_params(cell)
    ha = float(h) * astar
    kb = float(k) * bstar
    lc = float(l) * cstar
    g2 = (
        ha**2 + kb**2 + lc**2
        + 2.0 * ha * kb * np.cos(ga_s)
        + 2.0 * ha * lc * np.cos(be_s)
        + 2.0 * kb * lc * np.cos(al_s)
    )
    if g2 <= 0.0:
        raise ValueError(f"Non-positive |g|^2 for hkl=({h},{k},{l}). Check cell parameters.")
    g = float(np.sqrt(g2))
    return float(ha / g), float(kb / g), float(lc / g)


def required_terms_for_sg(sg: int, cell: Cell) -> List[str]:
    """Return the invariant term names required by the symmetry rules."""
    if sg in (1, 2):
        return [f"E{i}" for i in range(1, 16)]
    if 3 <= sg <= 15:
        # Monoclinic: accept all three unique-axis conventions
        # unique b -> alpha=90, gamma=90
        if _is_close(cell.alpha, 90.0) and _is_close(cell.gamma, 90.0):
            return [f"E{i}" for i in range(1, 10)]
        # unique c -> alpha=90, beta=90
        if _is_close(cell.alpha, 90.0) and _is_close(cell.beta, 90.0):
            return [f"E{i}" for i in range(1, 10)]
        # unique a -> beta=90, gamma=90
        if _is_close(cell.beta, 90.0) and _is_close(cell.gamma, 90.0):
            return [f"E{i}" for i in range(1, 10)]
        raise ValueError("Invalid monoclinic cell setting for invariant rules.")
    if 16 <= sg <= 74:
        return [f"E{i}" for i in range(1, 7)]
    if 75 <= sg <= 88:
        return [f"E{i}" for i in range(1, 6)]
    if 89 <= sg <= 142:
        return [f"E{i}" for i in range(1, 5)]
    if 143 <= sg <= 148:
        return ["E1", "E2", "E3", "E4", "E5"]
    if 149 <= sg <= 167:
        return ["E1", "E2", "E3", "E4"]
    if 168 <= sg <= 194:
        return ["E1", "E2", "E3"]
    if 195 <= sg <= 230:
        return ["E1", "E2"]
    raise ValueError("Space group out of range (1..230).")


def feature_vector(sg: int, H: float, K: float, L: float, cell: Cell) -> Tuple[List[str], np.ndarray]:
    """Return (term_names, x) such that C = sum_i E_i * x_i."""
    H2, K2, L2 = H * H, K * K, L * L
    H3, K3, L3 = H2 * H, K2 * K, L2 * L
    H4, K4, L4 = H2 * H2, K2 * K2, L2 * L2

    if sg in (1, 2):
        names = [f"E{i}" for i in range(1, 16)]
        x = np.array([
            H4, K4, L4,
            2*(H2*K2), 2*(K2*L2), 2*(H2*L2),
            4*(H3*K), 4*(H3*L), 4*(H*K3), 4*(K3*L),
            4*(H*L3), 4*(K*L3),
            4*(H2*K*L), 4*(H*K2*L), 4*(H*K*L2),
        ], dtype=float)
        return names, x

    if 3 <= sg <= 15:
        # unique b -> alpha=90, gamma=90
        if _is_close(cell.alpha, 90.0) and _is_close(cell.gamma, 90.0):
            names = [f"E{i}" for i in range(1, 10)]
            x = np.array([
                H4, K4, L4,
                2*(H2*L2), 2*(K2*L2), 2*(H2*K2),
                4*(H3*L), 4*(H*L3), 4*(H*K2*L),
            ], dtype=float)
            return names, x

        # unique c -> alpha=90, beta=90
        if _is_close(cell.alpha, 90.0) and _is_close(cell.beta, 90.0):
            names = [f"E{i}" for i in range(1, 10)]
            x = np.array([
                H4, K4, L4,
                2*(H2*K2), 2*(K2*L2), 2*(H2*L2),
                4*(H3*K), 4*(H*K3), 4*(H*K*L2),
            ], dtype=float)
            return names, x

        # unique a -> beta=90, gamma=90
        if _is_close(cell.beta, 90.0) and _is_close(cell.gamma, 90.0):
            names = [f"E{i}" for i in range(1, 10)]
            x = np.array([
                H4, K4, L4,
                2*(H2*K2), 2*(H2*L2), 2*(K2*L2),
                4*(K3*H), 4*(K*H3), 4*(K*L2*H),
            ], dtype=float)
            return names, x

        raise ValueError("Invalid monoclinic cell setting for invariant rules.")

    if 16 <= sg <= 74:
        names = [f"E{i}" for i in range(1, 7)]
        x = np.array([H4, K4, L4, 2*(H2*K2), 2*(K2*L2), 2*(H2*L2)], dtype=float)
        return names, x

    if 75 <= sg <= 88:
        names = ["E1", "E2", "E3", "E4", "E5"]
        x = np.array([
            (H4 + K4),
            L4,
            2*(H2*K2),
            2*((H2 + K2)*L2),
            4*(H*K*(H2 - K2)),
        ], dtype=float)
        return names, x

    if 89 <= sg <= 142:
        names = ["E1", "E2", "E3", "E4"]
        x = np.array([
            (H4 + K4),
            L4,
            2*(H2*K2),
            2*((H2 + K2)*L2),
        ], dtype=float)
        return names, x

    if 143 <= sg <= 148:
        rhombo_support = {146, 148}
        ga = float(cell.gamma)
        names = ["E1", "E2", "E3", "E4", "E5"]

        if sg in rhombo_support:
            if _is_close(ga, 120.0):
                A = (H2 - H*K + K2)
                x = np.array([A*A, 2*A*L2, L4,
                              4*(H*(H-K)*K*L),
                              4*((H3 - 3*H*K2 + K3)*L)], dtype=float)
                return names, x
            if _is_close(ga, 60.0):
                A = (H2 + H*K + K2)
                x = np.array([A*A, 2*A*L2, L4,
                              (4/3)*L*(H3 + 3*H2*K - K3),
                              (4/3)*L*(-H3 + 3*H*K2 + K3)], dtype=float)
                return names, x

        if _is_close(ga, 120.0):
            A = (H2 - H*K + K2)
            x = np.array([A*A, 2*A*L2, L4,
                          4*(H*(H-K)*K*L),
                          4*((H3 - 3*H*K2 + K3)*L)], dtype=float)
            return names, x

        A = (H2 + H*K + K2)
        x = np.array([A*A, 2*A*L2, L4,
                      (4/3)*L*(H3 + 3*H2*K - K3),
                      (4/3)*L*(-H3 + 3*H*K2 + K3)], dtype=float)
        return names, x

    if 149 <= sg <= 167:
        trig_m1 = {150, 152, 154, 155, 156, 158, 160, 161, 164, 165, 166, 167}
        rhombo_support = {155, 160, 161, 166, 167}
        ga = float(cell.gamma)
        names = ["E1", "E2", "E3", "E4"]

        if sg in trig_m1:
            if sg in rhombo_support:
                if _is_close(ga, 120.0):
                    A = (H2 - H*K + K2)
                    x = np.array([A*A, 2*L2*A, L4, 4*(H*K*L*(H-K))], dtype=float)
                    return names, x
                if _is_close(ga, 60.0):
                    A = (H2 + H*K + K2)
                    x = np.array([A*A, 2*L2*A, L4, 4*(H*K*L*(H+K))], dtype=float)
                    return names, x

            if _is_close(ga, 120.0):
                A = (H2 - H*K + K2)
                x = np.array([A*A, 2*L2*A, L4, 4*(H*K*L*(H-K))], dtype=float)
                return names, x

            A = (H2 + H*K + K2)
            x = np.array([A*A, 2*L2*A, L4, 4*(H*K*L*(H+K))], dtype=float)
            return names, x

        if _is_close(ga, 120.0):
            A = (H2 - H*K + K2)
            x = np.array([A*A, 2*L2*A, L4, L*(4*H3 - 6*H2*K - 6*H*K2 + 4*K3)], dtype=float)
            return names, x

        A = (H2 + H*K + K2)
        x = np.array([A*A, 2*L2*A, L4, (4/3)*L*(2*H3 + 3*H2*K - 3*H*K2 - 2*K3)], dtype=float)
        return names, x

    if 168 <= sg <= 194:
        ga = float(cell.gamma)
        names = ["E1", "E2", "E3"]
        A = (H2 + H*K + K2) if _is_close(ga, 120.0) else (H2 - H*K + K2)
        x = np.array([A*A, 2*L2*A, L4], dtype=float)
        return names, x

    if 195 <= sg <= 230:
        names = ["E1", "E2"]
        x = np.array([(H4 + K4 + L4), 2*(H2*K2 + H2*L2 + K2*L2)], dtype=float)
        return names, x

    raise ValueError("Space group out of range (1..230).")


def feature_vector_hkl(sg: int, h: int, k: int, l: int, cell: Cell) -> Tuple[List[str], np.ndarray]:
    """
    TOPAS/Popa-style invariant feature vector.

    The invariant polynomial must be evaluated with raw Miller indices h,k,l,
    then multiplied by d_hkl^4 / a^4.

    This makes equivalent orders consistent:
        (1,1,0), (2,2,0), (3,3,0), ...
    should give the same directional strain factor.
    """
    h = int(h)
    k = int(k)
    l = int(l)

    names = required_terms_for_sg(int(sg), cell)

    if h == 0 and k == 0 and l == 0:
        return names, np.zeros(len(names), dtype=float)

    d = d_spacing(cell, h, k, l)
    a = float(cell.a)

    scale = (d ** 4) / (a ** 4) if a > 0 else 1.0

    # Important: use raw h,k,l, not normalized components.
    names, x = feature_vector(int(sg), float(h), float(k), float(l), cell)

    return names, x * float(scale)


def _get_E_dict(contrast_coeffs: Optional[dict], kind: str) -> Optional[Dict[str, float]]:
    """Extract edge/screw invariant coefficient dict from the parsed config structure.

    Supports:
      contrast_coeffs['edge_E'] / ['screw_E']  (recommended)
    and (fallback) a flat dict with keys like 'EdgeE1', 'ScrewE1', etc.
    """
    if not contrast_coeffs:
        return None

    # Structured format from io_handler
    if isinstance(contrast_coeffs, dict):
        key = 'edge_E' if kind == 'edge' else 'screw_E'
        if key in contrast_coeffs and isinstance(contrast_coeffs[key], dict) and contrast_coeffs[key]:
            return {str(k).upper(): float(v) for k, v in contrast_coeffs[key].items()}

        # Flat fallback format
        out = {}
        prefix_variants = ['EDGE', 'EDGEE', 'E_EDGE', 'EDGE_E', 'EDGE_']
        if kind == 'screw':
            prefix_variants = ['SCREW', 'SCREWE', 'E_SCREW', 'SCREW_E', 'SCREW_']

        for k, v in contrast_coeffs.items():
            ku = str(k).upper()
            for pref in prefix_variants:
                if ku.startswith(pref) and re.match(rf"^{pref}E\d+$", ku):
                    ekey = ku.replace(pref, '')
                    out[ekey] = float(v)
        if out:
            return out

    return None


def contrast_factor(
    h: int,
    k: int,
    l: int,
    fE: float,
    *,
    sg: Optional[int] = None,
    cell: Optional[Cell] = None,
    contrast_coeffs: Optional[dict] = None,
) -> float:
    """Compute Chkl.

    Backward compatible behavior:
      - if (sg, cell, invariants) are not provided -> use legacy cubic form with (A,B)
      - if invariants are provided -> use invariant expansion for screw and edge, then mix by fE
    """
    E_edge = _get_E_dict(contrast_coeffs, 'edge')
    E_screw = _get_E_dict(contrast_coeffs, 'screw')

    # Invariant expansion (non-cubic capable)
    if (sg is not None) and (cell is not None) and (E_edge or E_screw):
        # HKL-only evaluation with metric-tensor normalization
        names, x = feature_vector_hkl(int(sg), int(h), int(k), int(l), cell)

        def eval_E(E: Optional[Dict[str, float]]) -> float:
            if not E:
                return 0.0
            return float(sum(float(E.get(n.upper(), 0.0)) * xi for n, xi in zip(names, x)))

        C_screw = eval_E(E_screw)
        C_edge = eval_E(E_edge)

        # If only one set exists, use it for both to keep the model usable
        if (E_screw is None or not E_screw) and (E_edge is not None and E_edge):
            C_screw = C_edge
        if (E_edge is None or not E_edge) and (E_screw is not None and E_screw):
            C_edge = C_screw

        return float(fE) * C_edge + (1.0 - float(fE)) * C_screw

    # Legacy cubic approximation
    h2, k2, l2 = h**2, k**2, l**2
    denom = (h2 + k2 + l2)**2
    if denom == 0:
        return 0.0
    num = h2*k2 + k2*l2 + l2*h2
    return (float(fE) * CEdgeA + (1.0 - float(fE)) * CScrewA) + \
           (float(fE) * CEdgeB + (1.0 - float(fE)) * CScrewB) * num / denom


def fstar(L: float, Re: float) -> float:
    x = float(L) / float(Re)
    if x <= 1:
        return -log(x) + 7/4 - log(2) + x**2/6 - 32*x**3/(225*pi)
    else:
        return 512 / (90 * pi * x) - (11/24 + 0.25 * log(2 * x)) / x**2


def fstar_vec(L: np.ndarray, Re: float) -> np.ndarray:
    """Vectorized version of fstar() for NumPy arrays."""
    L = np.asarray(L, dtype=float)
    Re = float(Re)
    x = L / Re
    out = np.empty_like(x, dtype=float)

    m = x <= 1.0
    # x<=1 branch
    xm = np.maximum(x[m], 1e-300)
    out[m] = -np.log(xm) + 7.0/4.0 - np.log(2.0) + (xm*xm)/6.0 - 32.0*(xm*xm*xm)/(225.0*np.pi)

    # x>1 branch
    xp = x[~m]
    out[~m] = 512.0 / (90.0*np.pi*xp) - (11.0/24.0 + 0.25*np.log(2.0*xp)) / (xp*xp)
    return out


def strain_variance(
    r: float,
    hkl: Tuple[int, int, int],
    a: float,
    rho: float,
    Re: float,
    fE: float,
    *,
    sg: Optional[int] = None,
    cell: Optional[Cell] = None,
    contrast_coeffs: Optional[dict] = None,
    burgers_mag: Optional[float] = None,
) -> float:
    """Wilkens-type mean square microstrain <eps^2>(L) along direction hkl.

    Notes
    -----
    - The original code assumed bcc with b = sqrt(3)/2 * a.
    - For non-bcc/non-cubic, you should supply burgers_mag (in Å) via input.
      If burgers_mag is omitted, we keep the legacy bcc formula to preserve old behavior.
    """
    if rho is None or Re is None:
        return 0.0
    if float(rho) <= 0.0 or float(Re) <= 0.0:
        return 0.0

    if burgers_mag is not None and float(burgers_mag) > 0:
        b = float(burgers_mag)
    else:
        # Backward compatible default (bcc)
        b = sqrt(3)/2 * float(a)

    Chkl = contrast_factor(
        int(hkl[0]), int(hkl[1]), int(hkl[2]), float(fE),
        sg=sg, cell=cell, contrast_coeffs=contrast_coeffs
    )
    return (float(rho) * b**2 / (4 * pi)) * float(Chkl) * float(fstar(float(r), float(Re)))


# ---------------------------------------------------------------------------
# Existing utilities (kept unchanged)
# ---------------------------------------------------------------------------

def reduce_hkl(h, k, l):
    def gcd3(a, b, c): return gcd(gcd(abs(a), abs(b)), abs(c))
    d = gcd3(h, k, l)
    if d != 0:
        h, k, l = h // d, k // d, l // d
    if h < 0 or (h == 0 and k < 0) or (h == 0 and k == 0 and l < 0):
        h, k, l = -h, -k, -l
    return (h, k, l)


def generate_bcc_shells_with_hkl(max_radius=5.0, a=2.86, bin_width=1e-5):
    shifts = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])]
    max_n = ceil(max_radius / a) + 1
    num_cores = min(multiprocessing.cpu_count(), 20)

    def compute_pair(i_shift, j_shift, nx, ny, nz):
        T = np.array([nx, ny, nz], dtype=float)
        delta = T + j_shift - i_shift
        if np.allclose(delta, 0.0): return None
        r_cart = a * delta
        r_mag = np.linalg.norm(r_cart)
        if r_mag > max_radius: return None
        dist_key = round(r_mag / bin_width) * bin_width
        hkl_raw = (int(round(delta[0]*2)), int(round(delta[1]*2)), int(round(delta[2]*2)))
        hkl = reduce_hkl(*hkl_raw)
        if hkl == (0, 0, 0): return None
        return (dist_key, hkl)

    tasks = [
        (i_shift, j_shift, nx, ny, nz)
        for i_shift in shifts
        for j_shift in shifts
        for nx in range(-max_n, max_n+1)
        for ny in range(-max_n, max_n+1)
        for nz in range(-max_n, max_n+1)
    ]

    results = Parallel(n_jobs=num_cores)(
        delayed(compute_pair)(*args) for args in tasks
    )

    shell_dict = defaultdict(lambda: {'distance': 0, 'multiplicity': 0, 'hkl_list': []})
    for r in results:
        if r is None: continue
        dist_key, hkl = r
        shell_dict[dist_key]['distance'] = dist_key
        shell_dict[dist_key]['multiplicity'] += 1
        shell_dict[dist_key]['hkl_list'].append(hkl)

    rows = []
    for i, (d, info) in enumerate(sorted(shell_dict.items())):
        rows.append({
            'shell': i+1,
            'distance': round(info['distance'], 5),
            'multiplicity': info['multiplicity'] // 4,
            'HKL': ', '.join([f"({h},{k},{l})" for (h,k,l) in sorted(set(info['hkl_list']))])
        })

    return pd.DataFrame(rows)

