"""
Crystallographic direction of a pair vector, used for the dislocation
contrast factor C_hkl of each shell.

The reciprocal-lattice vector parallel to a real-space pair vector
r = u·A (fractional u, lattice rows A) is h = G·u = A·r (G = A·Aᵀ, the
direct metric tensor; no 2π). h is exact but, in general, not rational
(non-cubic metrics, general atomic positions), so it is stored as the
smallest integer triple within ``tol_deg`` of the exact direction — exact
for rational directions such as [3 1 3̄] in a cubic cell.

The earlier convention rounded h/2π itself, which is coarse for short
vectors (often (0 0 0): no strain broadening) and wrong for long ones
(e.g. [3 1 3̄] stored as (2 1 2̄)).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange
except Exception:  # noqa: BLE001
    njit = None

TOL_DEG = 0.1
MAX_INDEX = 512


def _best_direction_py(x, y, z, max_index, tol_cos):
    m = max(abs(x), abs(y), abs(z))
    if m <= 1e-12:
        return 0, 0, 0
    x, y, z = x / m, y / m, z / m
    nrm = math.sqrt(x * x + y * y + z * z)
    best = (0, 0, 0)
    best_cos = -2.0
    for k in range(1, max_index + 1):
        a, b, c = round(k * x), round(k * y), round(k * z)
        dn = math.sqrt(a * a + b * b + c * c)
        if dn == 0:
            continue
        cs = (a * x + b * y + c * z) / (dn * nrm)
        if cs > best_cos:
            best_cos, best = cs, (a, b, c)
        if cs >= tol_cos:
            break
    a, b, c = best
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c)) or 1
    a, b, c = a // g, b // g, c // g
    s = 1
    if a != 0:
        s = 1 if a > 0 else -1
    elif b != 0:
        s = 1 if b > 0 else -1
    elif c != 0:
        s = 1 if c > 0 else -1
    return s * a, s * b, s * c


if njit is not None:

    @njit(cache=True)
    def _igcd(a, b):
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a

    @njit(cache=True, parallel=True)
    def _best_directions_nb(h, max_index, tol_cos):
        n = h.shape[0]
        out = np.zeros((n, 3), dtype=np.int16)
        for i in prange(n):
            x = h[i, 0]
            y = h[i, 1]
            z = h[i, 2]
            m = max(abs(x), max(abs(y), abs(z)))
            if m <= 1e-12:
                continue
            x /= m
            y /= m
            z /= m
            nrm = math.sqrt(x * x + y * y + z * z)
            ba = 0
            bb = 0
            bc = 0
            best_cos = -2.0
            for k in range(1, max_index + 1):
                a = int(np.rint(k * x))
                b = int(np.rint(k * y))
                c = int(np.rint(k * z))
                dn = math.sqrt(a * a + b * b + c * c)
                if dn == 0.0:
                    continue
                cs = (a * x + b * y + c * z) / (dn * nrm)
                if cs > best_cos:
                    best_cos = cs
                    ba = a
                    bb = b
                    bc = c
                if cs >= tol_cos:
                    break
            g = _igcd(_igcd(ba, bb), bc)
            if g == 0:
                g = 1
            ba //= g
            bb //= g
            bc //= g
            s = 1
            if ba != 0:
                s = 1 if ba > 0 else -1
            elif bb != 0:
                s = 1 if bb > 0 else -1
            elif bc != 0:
                s = 1 if bc > 0 else -1
            out[i, 0] = s * ba
            out[i, 1] = s * bb
            out[i, 2] = s * bc
        return out


def reciprocal_directions(cartesian, lattice_matrix, *, tol_deg: float = TOL_DEG, max_index: int = MAX_INDEX) -> np.ndarray:
    """
    Reduced integer reciprocal direction (int16, first non-zero index > 0)
    parallel to each Cartesian pair vector.
    """
    cart = np.ascontiguousarray(np.asarray(cartesian, dtype=np.float64).reshape(-1, 3))
    A = np.asarray(lattice_matrix, dtype=np.float64)
    if cart.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int16)
    h = np.ascontiguousarray(cart @ A.T)  # h_i = a_i · r
    tol_cos = math.cos(math.radians(float(tol_deg)))
    if njit is not None:
        return _best_directions_nb(h, int(max_index), float(tol_cos))
    out = np.zeros((cart.shape[0], 3), dtype=np.int16)
    for i, (x, y, z) in enumerate(h):
        out[i] = _best_direction_py(x, y, z, int(max_index), tol_cos)
    return out
