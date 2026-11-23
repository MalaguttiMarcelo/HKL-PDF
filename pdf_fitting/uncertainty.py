"""
Estimated standard deviations (esd) and the correlation matrix of refined
parameters.

At the solution x̂ (free parameters of the FitEngine), the Jacobian of the
weighted residual r = √w·(G_obs − G_calc) on the *full* r grid is computed by
central differences, with constraints, lattice constraints and tied
coordinates applied exactly as during the fit. Then

    S   = Σ r²,     s² = S / (N − p)          (reduced χ² for unit weights)
    C   = s² · (JᵀJ)⁻¹                         (via SVD; near-singular
                                                directions are reported and
                                                left out of the inverse)
    esd = √diag(C),  ρ_ij = C_ij / (esd_i esd_j)

Optional correction for correlated PDF points (off by default): the points of
G(r) are not independent — a PDF to Qmax carries about one independent value
per π/Qmax. With N_eff = Σ (r_max − r_min)·Qmax/π over the datasets, the
covariance is multiplied by N / N_eff (esd by √(N/N_eff)) when N_eff < N.

Parameters of the dependent kind (constraint expressions, lattice
constraints such as b = a, tied coordinates) get esds by linear propagation
through the parameter pipeline (no model evaluations needed).

The fit itself may use a robust loss; the esds are those of ordinary least
squares at the final point.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class UncertaintyResult:
    names: List[str]
    values: np.ndarray
    esd: np.ndarray
    correlation: np.ndarray
    chi2_red: float
    n_points: int
    n_params: int
    n_eff: Optional[float] = None
    correction: float = 1.0                    # covariance factor N/N_eff (1 = none)
    rank: int = 0
    singular: List[Dict[str, Any]] = field(default_factory=list)   # [{"sv", "params": [(name, coef)]}]
    at_bound: List[str] = field(default_factory=list)
    dependent: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    elapsed_s: float = 0.0
    message: str = ""

    def esd_dict(self) -> Dict[str, float]:
        out = {n: float(e) for n, e in zip(self.names, self.esd) if np.isfinite(e)}
        for n, (_v, e) in self.dependent.items():
            if np.isfinite(e):
                out[n] = float(e)
        return out

    def strong_correlations(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        out = []
        for i in range(len(self.names)):
            for j in range(i):
                c = self.correlation[i, j]
                if np.isfinite(c) and abs(c) >= threshold:
                    out.append((self.names[i], self.names[j], float(c)))
        return sorted(out, key=lambda t: -abs(t[2]))

    def to_json(self) -> Dict[str, Any]:
        def num(v):
            v = float(v)
            return v if np.isfinite(v) else None

        return {
            "names": list(self.names),
            "values": [num(v) for v in self.values],
            "esd": [num(v) for v in self.esd],
            "correlation": [[num(v) for v in row] for row in self.correlation],
            "chi2_red": num(self.chi2_red),
            "n_points": int(self.n_points),
            "n_params": int(self.n_params),
            "n_eff": num(self.n_eff) if self.n_eff is not None else None,
            "correction": num(self.correction),
            "rank": int(self.rank),
            "singular": [{"sv": num(s["sv"]), "params": [[n, num(c)] for n, c in s["params"]]} for s in self.singular],
            "at_bound": list(self.at_bound),
            "dependent": {k: [num(v), num(e)] for k, (v, e) in self.dependent.items()},
            "elapsed_s": num(self.elapsed_s),
            "message": self.message,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "UncertaintyResult":
        def arr(v):
            return np.asarray([np.nan if x is None else x for x in v], dtype=float)

        corr = np.asarray([[np.nan if x is None else x for x in row] for row in d.get("correlation", [])], dtype=float)
        return cls(
            names=list(d.get("names", [])),
            values=arr(d.get("values", [])),
            esd=arr(d.get("esd", [])),
            correlation=corr.reshape(len(d.get("names", [])), -1) if corr.size else np.zeros((0, 0)),
            chi2_red=float(d.get("chi2_red") or np.nan),
            n_points=int(d.get("n_points", 0)),
            n_params=int(d.get("n_params", 0)),
            n_eff=d.get("n_eff"),
            correction=float(d.get("correction") or 1.0),
            rank=int(d.get("rank", 0)),
            singular=[{"sv": s.get("sv"), "params": [(n, c) for n, c in s.get("params", [])]} for s in d.get("singular", [])],
            at_bound=list(d.get("at_bound", [])),
            dependent={k: (v[0] if v[0] is not None else np.nan, v[1] if v[1] is not None else np.nan) for k, v in (d.get("dependent") or {}).items()},
            elapsed_s=float(d.get("elapsed_s") or 0.0),
            message=str(d.get("message", "")),
        )


class EsdStopped(Exception):
    pass


def _param_pipeline(engine) -> Callable[[np.ndarray], Dict[str, float]]:
    """Free-parameter vector → full parameter dict, as the fit evaluates it."""
    from pdf_fitting.fit_engine import apply_constraints, apply_model_lattice_constraints_to_params

    base = np.asarray(engine.initial_values, dtype=float).copy()
    idx = np.asarray(engine.fit_indices, dtype=int)
    names = list(engine.param_names)

    def params_of(x):
        full = base.copy()
        full[idx] = x
        p = dict(zip(names, full))
        if getattr(engine, "constraints", None):
            p = apply_constraints(p, engine.constraints)
        p = apply_model_lattice_constraints_to_params(p, engine.model)
        try:
            p = engine._symmetrize_coordinates(p)
        except Exception:  # noqa: BLE001
            pass
        return p

    return params_of


def _steps(x, lb, ub) -> np.ndarray:
    """Central-difference steps: ~1e-3 relative, floored by the bound range / absolute scale."""
    x = np.asarray(x, dtype=float)
    h = 1e-3 * np.abs(x)
    rng = np.where(np.isfinite(lb) & np.isfinite(ub), ub - lb, np.inf)
    floor = np.where(np.isfinite(rng), 1e-4 * rng, 1e-6)
    h = np.maximum(h, floor)
    h = np.where(np.isfinite(rng), np.minimum(h, 0.25 * rng), h)
    return np.maximum(h, 1e-10)


def nyquist_points(spans_qmax: Sequence[Tuple[float, float]]) -> Optional[float]:
    """N_eff = Σ (r_max − r_min)·Qmax/π over datasets [(span, qmax)]."""
    tot = 0.0
    for span, qmax in spans_qmax:
        if span > 0 and qmax and qmax > 0:
            tot += span * qmax / math.pi
    return tot if tot > 0 else None


def compute_uncertainties(
    engine,
    *,
    n_eff: Optional[float] = None,
    status: Optional[Callable[[str], None]] = None,
    stop_event=None,
    rcond: float = 1e-8,
) -> UncertaintyResult:
    """
    esds and correlations of the engine's free parameters at
    engine.initial_values (the refined values after run_refinement).
    """
    t0 = time.perf_counter()
    status = status or (lambda _m: None)
    idx = np.asarray(engine.fit_indices, dtype=int)
    names = [engine.param_names[i] for i in idx]
    p = len(names)
    if p == 0:
        raise ValueError("No refined parameters.")
    x0 = np.asarray(engine.initial_values, dtype=float)[idx].copy()
    lb, ub = engine._build_bounds_for(idx)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    params_of = _param_pipeline(engine)

    def residual(x):
        if stop_event is not None and stop_event.is_set():
            raise EsdStopped("Stopped by the user.")
        _r, G_exp, G_fit, w = engine._evaluate_model(params_of(x), use_subgrid=False)
        res = np.asarray(G_exp, dtype=float) - np.asarray(G_fit, dtype=float)
        return np.sqrt(w) * res if w is not None else res

    status(f"esd: model at the solution ({p} parameters)…")
    r0 = residual(x0)
    n = r0.size
    h = _steps(x0, lb, ub)
    J = np.empty((n, p))
    at_bound = []
    for i in range(p):
        status(f"esd: derivative {i + 1}/{p} ({names[i]})…")
        up_ok = x0[i] + h[i] <= ub[i]
        dn_ok = x0[i] - h[i] >= lb[i]
        if not (up_ok and dn_ok) and (abs(x0[i] - ub[i]) < 2 * h[i] or abs(x0[i] - lb[i]) < 2 * h[i]):
            at_bound.append(names[i])
        if up_ok and dn_ok:
            xp = x0.copy(); xp[i] += h[i]
            xm = x0.copy(); xm[i] -= h[i]
            J[:, i] = (residual(xp) - residual(xm)) / (2 * h[i])
        elif up_ok:
            xp = x0.copy(); xp[i] += h[i]
            J[:, i] = (residual(xp) - r0) / h[i]
        else:
            xm = x0.copy(); xm[i] -= h[i]
            J[:, i] = (r0 - residual(xm)) / h[i]
    # residual = obs − calc, so J = −∂calc/∂x; the sign does not matter for JᵀJ.

    S = float(r0 @ r0)
    dof = max(n - p, 1)
    s2 = S / dof

    # Scale columns for a well-conditioned SVD.
    col = np.linalg.norm(J, axis=0)
    col_safe = np.where(col > 0, col, 1.0)
    Js = J / col_safe
    U, sv, Vt = np.linalg.svd(Js, full_matrices=False)
    keep = sv > rcond * (sv[0] if sv.size else 1.0)
    rank = int(np.count_nonzero(keep))
    inv_s2 = np.where(keep, 1.0 / np.maximum(sv, 1e-300) ** 2, 0.0)
    cov_s = (Vt.T * inv_s2) @ Vt
    cov = cov_s / np.outer(col_safe, col_safe) * s2

    correction = 1.0
    if n_eff is not None and 0 < n_eff < n:
        correction = n / float(n_eff)
        cov = cov * correction

    esd = np.sqrt(np.clip(np.diag(cov), 0, None))
    zero_col = col <= 0
    esd[zero_col] = np.nan
    d = np.where(esd > 0, esd, np.nan)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, np.where(np.isfinite(esd), 1.0, np.nan))

    singular = []
    for j in np.flatnonzero(~keep):
        v = Vt[j]
        order = np.argsort(-np.abs(v))[:4]
        singular.append({"sv": float(sv[j] / sv[0]), "params": [(names[k], float(v[k])) for k in order if abs(v[k]) > 0.1]})
    for k in np.flatnonzero(zero_col):
        singular.append({"sv": 0.0, "params": [(names[k], 1.0)]})

    # Dependent parameters: linear propagation through the parameter pipeline.
    status("esd: dependent parameters…")
    p0 = params_of(x0)
    free = set(names)
    D: Dict[str, np.ndarray] = {}
    for i in range(p):
        xp = x0.copy(); xp[i] += h[i]
        pp = params_of(xp)
        for k, v in pp.items():
            if k in free:
                continue
            try:
                dv = (float(v) - float(p0.get(k, v))) / h[i]
            except (TypeError, ValueError):
                continue
            if dv != 0.0 and np.isfinite(dv):
                D.setdefault(k, np.zeros(p))[i] = dv
    dependent = {}
    cov_f = np.where(np.isfinite(cov), cov, 0.0)
    for k, g in D.items():
        dependent[k] = (float(p0[k]), float(math.sqrt(max(g @ cov_f @ g, 0.0))))

    msg = f"{n} points, {p} parameters, rank {rank}"
    if correction != 1.0:
        msg += f", correlated-points correction ×{math.sqrt(correction):.2f} on esds (N_eff = {n_eff:.0f})"
    return UncertaintyResult(
        names=names, values=x0, esd=esd, correlation=corr, chi2_red=s2, n_points=n, n_params=p,
        n_eff=n_eff, correction=correction, rank=rank, singular=singular, at_bound=at_bound,
        dependent=dependent, elapsed_s=time.perf_counter() - t0, message=msg,
    )
