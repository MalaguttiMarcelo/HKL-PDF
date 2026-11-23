"""
Direction-resolved peak-width (Warren / Williamson–Hall-like) fit in real space.

For each free-peak cluster the observed width is compared with

    σ²_c = Σ_m w_m [ (B_ab / 4π²)·(1 − λ_m) + δ_broad²·r_m² + σ²_static,m ]

(members m of the cluster, weights w_m; B_ab = (B_a + B_b)/2).

δ_broad (instrument) is fixed and removed from what is plotted: the sample
widths σ²_sample = σ²_obs − δ_broad²r² are shown and fitted.

Thermal (dynamic) part — Scardi & Malagutti, Cryst. Growth Des. 24 (2024)
4380, eqs (2a), (5): MSRD = (B/4π²)(1 − λ). λ is free per element pair for
the first ``n_lambda`` shells (lambda_<pair>_<k>), δ1/r + δ2/r² beyond.
B, δ1 and δ2 can be shared by all directions, refined per direction
(anisotropic) or fixed.

Static (microstrain) part along the direction d of the pair vector:

    shape(r) = f*(r/Re)·r²   (Wilkens)   or   r²   (Gaussian strain)

    "isotropic"   δ_g²·r²
    "direction"   S_d·shape(r)       one free S_d per direction (independent)
    "invariant"   S_d = Σ_i Ẽ_i x_i(hkl_d),  Ẽ_i = (ρb²/4π)·E_i
                  Popa / DISCO invariant expansion of the powder-average
                  contrast factor for the Laue class of the phase
                  (Malagutti, D'Incau & Scardi, J. Appl. Cryst. 59 (2026)
                  1720, eq. 6 and appendix A)
    "model_cf"    S_d = (ρb²/4π)·[fE·C_edge(d) + (1 − fE)·C_screw(d)]
                  with the contrast factors of the phase: ρ and fE free

With the Wilkens shape, S_d = ρb²C_d/4π exactly as in the PDF model, so the
per-direction values are "measured" contrast factors up to the common scale
ρb²/4π; they are also fitted afterwards with the invariant expansion and
with the model contrast factors (DISCO-like least squares).

Weighted by the free-peak esds; parameter esds and correlations from the
covariance; reduced χ² reported (the problem can be ill-conditioned:
Scardi, Leoni & Delhez, J. Appl. Cryst. 37 (2004) 381).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FOUR_PI2 = 4.0 * math.pi ** 2

STATIC_MODELS = ("direction", "invariant", "model_cf", "isotropic")
PARAM_MODES = ("shared", "direction", "fixed")


@dataclass
class WarrenOptions:
    static_model: str = "direction"    # direction | invariant | model_cf | isotropic
    strain_form: str = "wilkens"       # wilkens | gaussian  (direction / invariant)
    B_mode: str = "shared"             # shared | direction | fixed
    d1_mode: str = "fixed"             # shared | direction | fixed
    d2_mode: str = "fixed"             # shared | direction | fixed
    n_lambda: int = 3                  # free λ for shells k < n_lambda (per pair)
    fit_Re: bool = False
    min_peaks: int = 2                 # peaks needed (at least) for a direction's own parameters
    use_merged: bool = True
    max_rel_esd: float = 0.25          # skip clusters with σ_esd / σ above this
    exclude: List[int] = field(default_factory=list)


@dataclass
class FamilyInfo:
    label: str                 # real-space direction family, e.g. "<111>"
    hkl: Tuple[float, float, float]  # reciprocal direction parallel to it
    own: bool                  # has its own per-direction parameters
    clusters: List[int]        # positions (in result.used) of the peaks containing it
    n_pure: int
    x_cf: float = float("nan")         # abscissa of the contrast-factor plot
    S: float = float("nan")            # static coefficient (ρb²C/4π or ε²)
    S_esd: float = float("nan")
    S_inv: float = float("nan")        # invariant least-squares value
    S_cf: float = float("nan")         # model contrast-factor value
    C_edge: float = float("nan")
    C_screw: float = float("nan")
    rho: float = float("nan")          # ρ from this direction alone (model C, current fE)
    rho_esd: float = float("nan")


@dataclass
class WarrenResult:
    names: List[str]
    values: np.ndarray
    esd: np.ndarray
    correlation: np.ndarray
    chi2_red: float
    used: List[int]                    # cluster indices
    observed: np.ndarray               # σ² (total) of used clusters
    observed_esd: np.ndarray
    r: np.ndarray
    predicted: np.ndarray
    thermal: np.ndarray
    static: np.ndarray
    instrumental: np.ndarray
    families: List[str]                # dominant family of each used cluster ("merged" if mixed)
    cluster_families: List[List[str]]  # all member families of each used cluster
    merged: List[bool]
    message: str
    options: WarrenOptions
    start: Dict[str, Any]
    family_info: Dict[str, FamilyInfo] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)      # fixed + fitted
    cf_axis: str = "index"             # "H" | "delta" | "index"
    inv_names: List[str] = field(default_factory=list)
    inv_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inv_esd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inv_source: str = ""
    cf_fit: Dict[str, float] = field(default_factory=dict)      # rho, fE (+ esd) from model C_hkl
    warnings: List[str] = field(default_factory=list)
    members: List[List[Dict[str, Any]]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def observed_sample(self) -> np.ndarray:
        """σ² with the instrumental contribution removed."""
        return self.observed - self.instrumental

    @property
    def predicted_sample(self) -> np.ndarray:
        return self.thermal + self.static

    def value(self, name, default=None):
        return float(self.values[self.names.index(name)]) if name in self.names else default

    def table(self) -> List[Tuple[str, float, float]]:
        return [(n, float(v), float(e)) for n, v, e in zip(self.names, self.values, self.esd)]

    def family_order(self) -> List[str]:
        return sorted(self.family_info, key=lambda f: (-len(self.family_info[f].clusters), f))

    def family_params(self, fam: str) -> List[Tuple[str, float, float]]:
        """Fitted parameters that belong to one direction (own or shared)."""
        tag = f" {fam}"
        own = [(n, v, e) for n, v, e in self.table() if n.endswith(tag)]
        shared = [(n, v, e) for n, v, e in self.table() if "<" not in n and "(" not in n and not n.endswith(" other")]
        return own + shared

    # ------------------------------------------------------------------
    def dynamic_curve(self, fam: str, pair: str, r: np.ndarray) -> np.ndarray:
        """Smooth dynamic σ² along one direction with λ = δ1/r + δ2/r²."""
        p = self.params
        a, b = pair.split("-")
        key = self._fam_key(fam)
        Bab = 0.5 * (_pget(p, f"B_{a}", key) + _pget(p, f"B_{b}", key))
        d1, d2 = _pget(p, "delta1", key), _pget(p, "delta2", key)
        lam = d1 / r + d2 / r ** 2
        return Bab / FOUR_PI2 * (1.0 - lam)

    def static_curve(self, fam: str, r: np.ndarray) -> np.ndarray:
        info = self.family_info.get(fam)
        p = self.params
        if self.options.static_model == "isotropic":
            return (p.get("delta_g", 0.0) * r) ** 2
        S = info.S if info is not None and np.isfinite(info.S) else 0.0
        return S * _shape(r, p.get("Re", 100.0), self._form())

    def _form(self) -> str:
        return "wilkens" if self.options.static_model == "model_cf" else self.options.strain_form

    def _fam_key(self, fam: str) -> str:
        info = self.family_info.get(fam)
        return fam if (info is not None and info.own) else "other"


def _pget(p: Dict[str, float], name: str, fam_key: str) -> float:
    """Per-direction value of a parameter, falling back to the shared one."""
    for key in (f"{name} {fam_key}", name):
        if key in p:
            return float(p[key])
    return 0.0


def _fstar(r, Re):
    from pdf_fitting.models.microstrain_utils import fstar_vec

    return fstar_vec(np.asarray(r, dtype=float), float(Re))


def _shape(r, Re, form: str):
    r = np.asarray(r, dtype=float)
    if form == "gaussian":
        return r * r
    return _fstar(r, Re) * r * r


# -----------------------------------------------------------------------------
# Crystallography helpers
# -----------------------------------------------------------------------------


def _direct_matrix(cell) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    al, be, ga = (math.radians(x) for x in (cell.alpha, cell.beta, cell.gamma))
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx_ = c * math.cos(be)
    cy_ = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz_ = math.sqrt(max(c * c - cx_ * cx_ - cy_ * cy_, 0.0))
    return np.array([ax, bx, [cx_, cy_, cz_]])


def member_hkl(m, cell) -> Tuple[float, float, float]:
    """
    Reciprocal-space direction (h, k, l) parallel to the pair vector u of a
    member: h = G·u (G = direct metric tensor). Exact (float) — in a
    non-cubic cell it is generally not a rational direction. Scaled so that
    rational directions come out as small integers.
    """
    A = _direct_matrix(cell)
    v = (A @ A.T) @ np.asarray(m.vector, dtype=float)
    nz = np.abs(v[np.abs(v) > 1e-9 * max(1.0, float(np.max(np.abs(v))))])
    if nz.size == 0:
        return (0.0, 0.0, 0.0)
    v = v / nz.min()
    for mult in range(1, 13):
        w = v * mult
        if np.allclose(w, np.round(w), atol=1e-3):
            ints = np.round(w).astype(int)
            g = int(np.gcd.reduce(np.abs(ints[ints != 0]))) or 1
            return tuple(float(x // g) for x in ints)
    return tuple(float(x) for x in v)


def hkl_label(hkl) -> str:
    if all(abs(x - round(x)) < 1e-6 for x in hkl):
        ints = [int(round(x)) for x in hkl]
        sep = " " if any(abs(x) >= 10 for x in ints) else ""
        return "(" + sep.join(str(x) for x in ints) + ")"
    return "(" + " ".join(f"{x:.2f}" for x in hkl) + ")"


def contrast_from_params(calc, params, hkl, fE: float) -> float:
    """Contrast factor of the phase model for a (float) reciprocal direction."""
    from pdf_fitting.project.contrast_factors import contrast

    return contrast(params, hkl, fE, int(getattr(calc, "spacegroup_number", 0) or 0), getattr(calc, "cell", None))


def crystal_family(sg: int) -> str:
    if sg >= 195:
        return "cubic"
    if sg >= 143:
        return "hexagonal"
    if sg >= 75:
        return "tetragonal"
    return "low"


def cf_abscissa(hkl, sg: int, cell) -> float:
    """Cubic: H = (h²k²+h²l²+k²l²)/(h²+k²+l²)²; uniaxial: angle δ (deg) between g and c."""
    h, k, l = (float(x) for x in hkl)
    fam = crystal_family(sg)
    if fam == "cubic":
        s = h * h + k * k + l * l
        return (h * h * k * k + h * h * l * l + k * k * l * l) / (s * s) if s else float("nan")
    if fam in ("hexagonal", "tetragonal"):
        A = _direct_matrix(cell)
        B = np.linalg.inv(A).T
        g = np.array([h, k, l]) @ B
        cvec = A[2] / np.linalg.norm(A[2])
        n = np.linalg.norm(g)
        return math.degrees(math.acos(min(1.0, abs(float(g @ cvec)) / n))) if n else float("nan")
    return float("nan")


def invariant_features(hkl, sg: int, cell) -> Tuple[List[str], np.ndarray]:
    """Popa / DISCO invariant basis x_i(hkl)·d⁴/a⁴ (scale-invariant, float hkl)."""
    from pdf_fitting.models.microstrain_utils import d_spacing, feature_vector

    h, k, l = (float(x) for x in hkl)
    names, x = feature_vector(int(sg), h, k, l, cell)
    return names, x * (d_spacing(cell, h, k, l) ** 4) / (float(cell.a) ** 4)


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def fit_warren(
    clusters,
    *,
    elements: List[str],
    start: Dict[str, Any],
    options: Optional[WarrenOptions] = None,
) -> WarrenResult:
    """
    start: {"B": {El: B}, "lambda": {(pair, k): λ}, "delta1", "delta2",
            "delta_broad", "delta_g", "rho", "fE", "Re", "burgers",
            "sg", "cell", "cf_edge": {family: C}, "cf_screw": {family: C}}
    """
    from scipy.optimize import least_squares

    opt = options or WarrenOptions()
    if opt.static_model not in STATIC_MODELS:
        raise ValueError(f"Unknown static model {opt.static_model!r}")
    sg = int(start.get("sg") or 0)
    cell = start.get("cell")
    if opt.static_model == "invariant" and (not sg or cell is None):
        raise ValueError("The invariant expansion needs the space group and cell of the phase.")

    used = []
    for c in clusters:
        if c.index in opt.exclude:
            continue
        if not np.isfinite(c.sigma) or not np.isfinite(c.sigma_esd):
            continue
        if c.sigma_esd <= 0 or c.sigma_esd / max(c.sigma, 1e-9) > opt.max_rel_esd:
            continue
        if c.merged and not opt.use_merged:
            continue
        used.append(c)
    if len(used) < 3:
        raise ValueError("Fewer than three reliable peaks: widen the r range or relax the esd limit.")

    warnings: List[str] = []

    # ---- directions ---------------------------------------------------
    fam_hkl: Dict[str, Tuple[int, int, int]] = {}
    fam_clusters: Dict[str, List[int]] = {}
    fam_pure: Dict[str, int] = {}
    for i, c in enumerate(used):
        seen = set()
        for m in c.members:
            if cell is not None:
                fam_hkl.setdefault(m.family, member_hkl(m, cell))
            else:
                fam_hkl.setdefault(m.family, tuple(float(x) for x in m.vector))
            if m.family not in seen:
                fam_clusters.setdefault(m.family, []).append(i)
                seen.add(m.family)
        if not c.merged:
            fam_pure[c.members[0].family] = fam_pure.get(c.members[0].family, 0) + 1
    n_el = len({e for c in used for m in c.members for e in m.pair.split("-")})
    n_own = (
        (1 if opt.static_model == "direction" else 0)
        + (n_el if opt.B_mode == "direction" else 0)
        + (1 if opt.d1_mode == "direction" else 0)
        + (1 if opt.d2_mode == "direction" else 0)
    )
    need = max(1, int(opt.min_peaks), n_own)
    own_fams = sorted(f for f, cl in fam_clusters.items() if len(cl) >= need)
    if n_own:
        thin = sorted(set(fam_clusters) - set(own_fams))
        if thin:
            warnings.append(
                f"{', '.join(thin)}: fewer than {need} peak(s) for their own parameters, "
                "they share the 'other' ones"
            )

    def fkey(fam):
        return fam if fam in own_fams else "other"

    members = [m for c in used for m in c.members]
    pairs_k = sorted({(m.lambda_pair, m.k) for m in members if m.k < opt.n_lambda})
    elements_used = sorted({e for m in members for e in m.pair.split("-")})
    fam_keys_used = sorted({fkey(m.family) for m in members})

    names: List[str] = []
    x0: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    fixed: Dict[str, float] = {}

    def add(name, value, lo, hi):
        names.append(name)
        x0.append(float(np.clip(value, lo + 1e-12 * max(1.0, abs(hi - lo)), hi - 1e-12 * max(1.0, abs(hi - lo)))))
        lb.append(lo)
        ub.append(hi)

    def add_mode(base, mode, value, lo, hi):
        if mode == "fixed":
            fixed[base] = float(value)
        elif mode == "direction":
            for fk in fam_keys_used:
                add(f"{base} {fk}", value, lo, hi)
        else:
            add(base, value, lo, hi)

    B_start = {e.lower(): float(v) for e, v in (start.get("B") or {}).items()}
    for el in elements_used:
        add_mode(f"B_{el}", opt.B_mode, B_start.get(el, 0.5), 0.0, 10.0)
    lam_start = start.get("lambda") or {}
    for lp, k in pairs_k:
        add(f"lambda_{lp}_{k}", lam_start.get((lp, k), 0.0), -1.0, 1.0)
    add_mode("delta1", opt.d1_mode, float(start.get("delta1", 0.0)), -5.0, 5.0)
    add_mode("delta2", opt.d2_mode, float(start.get("delta2", 0.0)), -20.0, 20.0)

    form = "wilkens" if opt.static_model == "model_cf" else opt.strain_form
    rho0 = float(start.get("rho", 0.0) or 0.0)
    bmag = float(start.get("burgers", 0.0) or 0.0)
    fE0 = float(start.get("fE", 0.5))
    cf_e = start.get("cf_edge") or {}
    cf_s = start.get("cf_screw") or {}
    has_cf = any(abs(v) > 0 for v in cf_e.values()) or any(abs(v) > 0 for v in cf_s.values())

    def C_model(fam, fE):
        return fE * cf_e.get(fam, 0.0) + (1.0 - fE) * cf_s.get(fam, 0.0)

    pref0 = rho0 * bmag * bmag / (4.0 * math.pi) if rho0 > 0 and bmag > 0 else 0.0
    dg0 = float(start.get("delta_g", 0.0))

    def S_guess(fam):
        if pref0 > 0 and has_cf:
            v = pref0 * C_model(fam, fE0)
            if v > 0:
                return v
        return max(dg0 ** 2, 1e-5) if form == "gaussian" else 1e-5

    inv_names: List[str] = []
    feat: Dict[str, np.ndarray] = {}
    if cell is not None and sg:
        for fam, hkl in fam_hkl.items():
            try:
                inv_names, feat[fam] = invariant_features(hkl, sg, cell)
            except Exception:  # noqa: BLE001
                pass

    if opt.static_model == "isotropic":
        add("delta_g", max(dg0, 1e-4), 0.0, 0.2)
    elif opt.static_model == "direction":
        for fk in fam_keys_used:
            if fk == "other":
                others = [f for f in fam_clusters if f not in own_fams]
                guess = float(np.mean([S_guess(f) for f in others])) if others else 1e-5
            else:
                guess = S_guess(fk)
            add(f"S {fk}", guess, 0.0, 1.0)
    elif opt.static_model == "invariant":
        if not inv_names or len(feat) < len(fam_hkl):
            raise ValueError("Could not evaluate the invariant expansion for these directions.")
        nfam = len(fam_hkl)
        if nfam < len(inv_names):
            warnings.append(
                f"{nfam} direction(s) for {len(inv_names)} invariant coefficients: "
                "the expansion is underdetermined (extend the r range)"
            )
        # Start: least squares of the guessed per-direction values.
        F = np.array([feat[f] for f in fam_hkl])
        s = np.array([S_guess(f) for f in fam_hkl])
        e0, *_ = np.linalg.lstsq(F, s, rcond=None)
        for n, v in zip(inv_names, e0):
            add(f"Ẽ{n[1:]}", float(v), -1.0, 1.0)
    elif opt.static_model == "model_cf":
        if not has_cf or bmag <= 0:
            raise ValueError(
                "The phase has no contrast-factor coefficients (EdgeE*/ScrewE* or CEdgeA/B, CScrewA/B): "
                "use the per-direction or invariant model."
            )
        add("rho", rho0 if rho0 > 0 else 1e-4, 0.0, 1.0)
        add("fE", fE0, 0.0, 1.0)
    if form == "wilkens" and opt.static_model != "isotropic":
        Re0 = float(start.get("Re", 100.0))
        if opt.fit_Re:
            add("Re", Re0, 1.0, 1e5)
        else:
            fixed["Re"] = Re0
    db = float(start.get("delta_broad", 0.0))

    # ---- data -----------------------------------------------------------
    obs = np.array([c.sigma ** 2 for c in used])
    obs_esd = np.array([2.0 * c.sigma * c.sigma_esd for c in used])
    obs_esd = np.where(obs_esd > 0, obs_esd, np.median(obs_esd[obs_esd > 0]) if np.any(obs_esd > 0) else 1e-4)

    def components(x):
        p = dict(fixed)
        p.update(dict(zip(names, x)))
        Re = p.get("Re", 100.0)
        th = np.zeros(len(used))
        st = np.zeros(len(used))
        ins = np.zeros(len(used))
        for i, c in enumerate(used):
            for m in c.members:
                fk = fkey(m.family)
                a, b = m.pair.split("-")
                Bab = 0.5 * (_pget(p, f"B_{a}", fk) + _pget(p, f"B_{b}", fk))
                key = f"lambda_{m.lambda_pair}_{m.k}"
                if key in p:
                    lam = p[key]
                else:
                    lam = _pget(p, "delta1", fk) / m.r + _pget(p, "delta2", fk) / m.r ** 2
                th[i] += m.weight * Bab / FOUR_PI2 * (1.0 - lam)
                ins[i] += m.weight * (db * m.r) ** 2
                if opt.static_model == "isotropic":
                    s = (p["delta_g"] * m.r) ** 2
                else:
                    if opt.static_model == "direction":
                        S = p[f"S {fk}"]
                    elif opt.static_model == "invariant":
                        S = float(sum(p[f"Ẽ{n[1:]}"] * xi for n, xi in zip(inv_names, feat[m.family])))
                    else:
                        S = p["rho"] * bmag * bmag / (4.0 * math.pi) * C_model(m.family, p["fE"])
                    s = S * float(_shape([m.r], Re, form)[0])
                st[i] += m.weight * s
        return th, st, ins, p

    def residual(x):
        th, st, ins, _p = components(x)
        return (obs - (th + st + ins)) / obs_esd

    res = least_squares(residual, np.array(x0), bounds=(np.array(lb), np.array(ub)), method="trf", x_scale="jac", max_nfev=5000)
    n, pn = obs.size, len(names)
    if n <= pn:
        warnings.append(f"{n} peaks for {pn} parameters: the fit is underdetermined")
    chi2_red = float(2.0 * res.cost / max(n - pn, 1))
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac) * max(chi2_red, 1e-12)
    except np.linalg.LinAlgError:
        cov = np.full((pn, pn), np.nan)
    if n <= pn:
        cov = np.full((pn, pn), np.nan)
    esd = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.where(esd > 0, esd, 1.0)
    corr = cov / np.outer(d, d)
    th, st, ins, pfin = components(res.x)

    # ---- per-direction static coefficients ------------------------------
    infos: Dict[str, FamilyInfo] = {}
    grads: Dict[str, np.ndarray] = {}
    for fam, cl in fam_clusters.items():
        info = FamilyInfo(
            label=fam, hkl=fam_hkl[fam], own=fam in own_fams, clusters=cl, n_pure=fam_pure.get(fam, 0),
        )
        if cell is not None and sg:
            info.x_cf = cf_abscissa(info.hkl, sg, cell)
        if has_cf:
            info.C_edge = float(cf_e.get(fam, float("nan")))
            info.C_screw = float(cf_s.get(fam, float("nan")))
        grad = np.zeros(pn)
        if opt.static_model == "direction":
            nm = f"S {fkey(fam)}"
            j = names.index(nm)
            info.S, grad[j] = float(res.x[j]), 1.0
        elif opt.static_model == "invariant":
            for nmi, xi in zip(inv_names, feat[fam]):
                j = names.index(f"Ẽ{nmi[1:]}")
                grad[j] = xi
            info.S = float(grad @ res.x)
        elif opt.static_model == "model_cf":
            jr, jf = names.index("rho"), names.index("fE")
            pref = bmag * bmag / (4.0 * math.pi)
            ce, cs = cf_e.get(fam, 0.0), cf_s.get(fam, 0.0)
            rho, fE = res.x[jr], res.x[jf]
            info.S = float(pref * rho * (fE * ce + (1 - fE) * cs))
            grad[jr] = pref * (fE * ce + (1 - fE) * cs)
            grad[jf] = pref * rho * (ce - cs)
        if opt.static_model != "isotropic":
            info.S_esd = float(math.sqrt(max(grad @ cov @ grad, 0.0))) if np.all(np.isfinite(cov)) else float("nan")
        infos[fam] = info
        grads[fam] = grad

    result = WarrenResult(
        names=names,
        values=res.x,
        esd=esd,
        correlation=corr,
        chi2_red=chi2_red,
        used=[c.index for c in used],
        observed=obs,
        observed_esd=obs_esd,
        r=np.array([c.r for c in used]),
        predicted=th + st + ins,
        thermal=th,
        static=st,
        instrumental=ins,
        families=[c.members[0].family if len({m.family for m in c.members}) == 1 else "merged" for c in used],
        cluster_families=[sorted({m.family for m in c.members}) for c in used],
        merged=[c.merged for c in used],
        message=str(res.message),
        options=opt,
        start=dict(start),
        family_info=infos,
        params={k: float(v) for k, v in pfin.items()},
        cf_axis={"cubic": "H", "hexagonal": "delta", "tetragonal": "delta"}.get(crystal_family(sg), "index") if sg else "index",
        warnings=warnings,
        members=[
            [{"pair": m.pair, "family": m.family, "k": m.k, "r": m.r, "weight": m.weight} for m in c.members]
            for c in used
        ],
    )
    if opt.static_model != "isotropic":
        _contrast_analysis(result, inv_names, feat, bmag, cov, grads)
    at_bound = [
        nm for nm, v, lo, hi in zip(names, res.x, lb, ub)
        if (v - lo) < 1e-6 * max(1.0, abs(hi - lo)) and lo != 0.0 or (hi - v) < 1e-6 * max(1.0, abs(hi - lo))
    ]
    if at_bound:
        result.warnings.append("at a limit (not determined): " + ", ".join(at_bound))
    return result


def _gls(A: np.ndarray, s: np.ndarray, W: np.ndarray, rank_W: int):
    """Generalized least squares s ≈ A·c with weight matrix W (pseudo-inverse covariance)."""
    M = A.T @ W @ A
    Minv = np.linalg.pinv(M)
    coef = Minv @ (A.T @ W @ s)
    dof = rank_W - np.linalg.matrix_rank(M)
    r = s - A @ coef
    scale = max(float(r @ W @ r) / dof, 1.0) if dof > 0 else 1.0
    return coef, Minv * scale


def _contrast_analysis(result: WarrenResult, inv_names, feat, bmag, cov, grads) -> None:
    """
    DISCO-like least squares of the per-direction static coefficients:
    invariant expansion Ẽ_i and model contrast factors (ρ, fE). Generalized
    least squares with the full covariance of the per-direction values (they
    are correlated through the shared dynamic parameters).
    """
    opt = result.options
    form = "wilkens" if opt.static_model == "model_cf" else opt.strain_form
    if not np.all(np.isfinite(cov)):
        return
    infos = [i for i in result.family_info.values() if i.own and np.isfinite(i.S)]
    if not infos:
        return
    Gm = np.array([grads[i.label] for i in infos])
    covS = Gm @ cov @ Gm.T
    W = np.linalg.pinv(covS, rcond=1e-10, hermitian=True)
    rank_W = int(np.linalg.matrix_rank(covS, tol=1e-10 * max(float(np.max(np.abs(covS))), 1e-300)))
    s = np.array([i.S for i in infos])

    # Invariant expansion
    if opt.static_model == "invariant":
        result.inv_names = list(inv_names)
        idx = [result.names.index(f"Ẽ{n[1:]}") for n in inv_names]
        result.inv_values = np.array([result.values[j] for j in idx])
        result.inv_esd = np.array([result.esd[j] for j in idx])
        result.inv_source = "joint fit (constrained)"
        for info in result.family_info.values():
            info.S_inv = info.S
    elif inv_names and all(i.label in feat for i in infos):
        F = np.array([feat[i.label] for i in infos])
        coef, cc = _gls(F, s, W, rank_W)
        e = np.sqrt(np.clip(np.diag(cc), 0, None))
        if np.linalg.matrix_rank(F) < len(inv_names):
            e = np.full(len(coef), np.nan)
            result.warnings.append(
                f"{len(infos)} direction(s) with their own value for {len(inv_names)} invariant "
                "coefficients: the expansion is not unique"
            )
        result.inv_names = list(inv_names)
        result.inv_values = coef
        result.inv_esd = e
        result.inv_source = "least squares of the per-direction values"
        for info in result.family_info.values():
            if info.label in feat:
                info.S_inv = float(feat[info.label] @ coef)

    # Model contrast factors: S_d = a·C_edge + c·C_screw, a = ρb²fE/4π, c = ρb²(1−fE)/4π
    if form != "wilkens" or bmag <= 0:
        return
    pref = bmag * bmag / (4.0 * math.pi)
    fE_cur = float(result.start.get("fE", 0.5))
    for info in result.family_info.values():
        C = fE_cur * info.C_edge + (1 - fE_cur) * info.C_screw
        if np.isfinite(C) and C > 0 and np.isfinite(info.S):
            info.rho = info.S / (pref * C)
            info.rho_esd = info.S_esd / (pref * C) if np.isfinite(info.S_esd) else float("nan")
    if opt.static_model == "model_cf":
        j, k = result.names.index("rho"), result.names.index("fE")
        result.cf_fit = {"rho": float(result.values[j]), "rho_esd": float(result.esd[j]),
                         "fE": float(result.values[k]), "fE_esd": float(result.esd[k]), "source": "joint fit"}
        for info in result.family_info.values():
            info.S_cf = info.S
        return
    ok = np.array([np.isfinite(i.C_edge) and np.isfinite(i.C_screw) and (i.C_edge > 0 or i.C_screw > 0) for i in infos])
    if not ok.any():
        return
    Cm = np.array([[i.C_edge, i.C_screw] for i in infos])[ok]
    Wk = W[np.ix_(ok, ok)]
    sk = s[ok]
    out: Dict[str, float] = {"source": "least squares of the per-direction values"}
    if np.linalg.matrix_rank(Cm.T @ Wk @ Cm) == 2:
        coef, cv = _gls(Cm, sk, Wk, int(np.linalg.matrix_rank(Wk)))
        a, c = (max(float(x), 0.0) for x in coef)
        tot = a + c
        if tot > 0:
            g_rho = np.array([1.0, 1.0]) / pref
            g_fE = np.array([c, -a]) / tot ** 2
            out.update(rho=tot / pref, rho_esd=float(math.sqrt(max(g_rho @ cv @ g_rho, 0))),
                       fE=a / tot, fE_esd=float(math.sqrt(max(g_fE @ cv @ g_fE, 0))))
            for info in result.family_info.values():
                if np.isfinite(info.C_edge) and np.isfinite(info.C_screw):
                    info.S_cf = a * info.C_edge + c * info.C_screw
    if "rho" not in out:
        # Only the scale: ρ with the current fE.
        C = (fE_cur * Cm[:, 0] + (1 - fE_cur) * Cm[:, 1])[:, None]
        coef, cv = _gls(C, sk, Wk, int(np.linalg.matrix_rank(Wk)))
        k = float(coef[0])
        out.update(rho=k / pref, rho_esd=float(math.sqrt(max(cv[0, 0], 0))) / pref, fE=fE_cur, fE_esd=float("nan"))
        for info in result.family_info.values():
            info.S_cf = k * (fE_cur * info.C_edge + (1 - fE_cur) * info.C_screw)
    result.cf_fit = out


# -----------------------------------------------------------------------------
# Starting values from / back to the phase parameters
# -----------------------------------------------------------------------------


def start_from_phase(calc, params: Dict[str, Any], clusters) -> Dict[str, Any]:
    """Starting values for fit_warren from the phase model and parameters."""
    lc = {str(k).lower(): v for k, v in params.items()}

    def f(name, default=0.0):
        try:
            return float(lc.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    B = {}
    try:
        vec = calc._biso_values_vector_from_params(params)
        for el, v in zip(calc._unique_species, vec):
            B[str(el)] = float(v)
    except Exception:  # noqa: BLE001
        pass
    d1, d2 = f("delta1"), f("delta2")
    lam = {}
    for c in clusters:
        for m in c.members:
            key = f"lambda_{m.lambda_pair}_{m.k}"
            if key in lc:
                lam[(m.lambda_pair, m.k)] = f(key)
            else:
                pd1 = f(f"delta1_{m.pair}", d1)
                pd2 = f(f"delta2_{m.pair}", d2)
                lam[(m.lambda_pair, m.k)] = pd1 / m.r + pd2 / m.r ** 2
    a = float(calc.structure_handler.structure.lattice.a)
    b = f("burgers_mag", 0.0) or f("burgers", 0.0) or (math.sqrt(3.0) / 2.0) * a
    fE = f("fe", 0.5)
    sg = int(getattr(calc, "spacegroup_number", 0) or 0)
    cell = getattr(calc, "cell", None)
    cf_edge: Dict[str, float] = {}
    cf_screw: Dict[str, float] = {}
    for c in clusters:
        for m in c.members:
            if m.family in cf_edge or cell is None:
                continue
            hkl = member_hkl(m, cell)
            try:
                cf_edge[m.family] = float(contrast_from_params(calc, params, hkl, 1.0))
                cf_screw[m.family] = float(contrast_from_params(calc, params, hkl, 0.0))
            except Exception:  # noqa: BLE001
                pass
    return {
        "B": B,
        "lambda": lam,
        "delta1": d1,
        "delta2": d2,
        "delta_broad": f("delta_broad"),
        "delta_g": f("delta_g"),
        "rho": f("rho", 0.0),
        "fE": fE,
        "Re": f("re", 100.0) if f("re", 0.0) > 0 else 100.0,
        "burgers": b,
        "sg": sg,
        "cell": cell,
        "cf_edge": cf_edge,
        "cf_screw": cf_screw,
    }


def _wmean(pairs: Sequence[Tuple[float, float]]) -> float:
    vals = [(v, e) for v, e in pairs if np.isfinite(v)]
    if not vals:
        return float("nan")
    w = np.array([1.0 / e ** 2 if np.isfinite(e) and e > 0 else 0.0 for _v, e in vals])
    v = np.array([x for x, _e in vals])
    return float(np.sum(w * v) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(v))


def starting_values(result: WarrenResult) -> List[Tuple[str, float, str, bool]]:
    """
    Suggested phase parameters (local names) as (name, value, note, default_on):
    biso_<El>, lambda_<pair>_<k>, delta1/2, delta_g, rho / fe / re and,
    optionally, invariant contrast-factor coefficients.
    """
    out: List[Tuple[str, float, str, bool]] = []
    table = result.table()
    opt = result.options

    def collect(base):
        return [(v, e) for n, v, e in table if n == base or n.startswith(base + " ")]

    for el in sorted({n.split(" ")[0][2:] for n, _v, _e in table if n.startswith("B_")}):
        vals = collect(f"B_{el}")
        v = _wmean(vals)
        note = "per-direction weighted mean" if len(vals) > 1 else "fit"
        out.append((f"biso_{el[0].upper()}{el[1:]}", v, note, True))
    for n, v, _e in table:
        if n.startswith("lambda_"):
            out.append((n, float(v), "fit", True))
    for base in ("delta1", "delta2"):
        vals = collect(base)
        if vals:
            out.append((base, _wmean(vals), "per-direction weighted mean" if len(vals) > 1 else "fit", True))
    if "delta_g" in result.names:
        out.append(("delta_g", result.value("delta_g"), "fit", True))
    cf = result.cf_fit
    if cf.get("rho") is not None and np.isfinite(cf.get("rho", np.nan)) and cf["rho"] > 0:
        out.append(("rho", float(cf["rho"]), f"from the contrast factors of the phase ({cf.get('source', '')})", True))
        fE_esd = cf.get("fE_esd", np.nan)
        if np.isfinite(fE_esd) and fE_esd < 0.25:
            out.append(("fe", float(cf["fE"]), f"edge fraction ± {fE_esd:.2f}", True))
    if "Re" in result.names:
        out.append(("re", result.value("Re"), "fit", True))
    # Invariant coefficients: E_i = Ẽ_i·4π/(ρ b²), same for edge and screw (fE then irrelevant).
    b = float(result.start.get("burgers", 0.0) or 0.0)
    rho = float(cf.get("rho", 0.0) or result.start.get("rho", 0.0) or 0.0)
    if result.inv_names and rho > 0 and b > 0 and np.all(np.isfinite(result.inv_values)):
        scale = 4.0 * math.pi / (rho * b * b)
        for n, v in zip(result.inv_names, result.inv_values):
            for pre in ("EdgeE", "ScrewE"):
                out.append((f"{pre}{n[1:]}", float(v * scale), f"measured invariant, ρ = {rho:.3g} Å⁻²", False))
    return out


# -----------------------------------------------------------------------------
# Saving the starting values in the project JSON
# -----------------------------------------------------------------------------


def start_to_json(start: Dict[str, Any]) -> Dict[str, Any]:
    d = {k: v for k, v in start.items() if k not in ("cell", "lambda")}
    cell = start.get("cell")
    d["cell"] = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma] if cell is not None else None
    d["lambda"] = [[p, int(k), float(v)] for (p, k), v in (start.get("lambda") or {}).items()]
    return d


def start_from_json(d: Dict[str, Any]) -> Dict[str, Any]:
    from pdf_fitting.models.microstrain_utils import Cell

    st = dict(d)
    st["cell"] = Cell(*[float(x) for x in d["cell"]]) if d.get("cell") else None
    st["lambda"] = {(str(p), int(k)): float(v) for p, k, v in (d.get("lambda") or [])}
    st["sg"] = int(d.get("sg") or 0)
    return st


def refresh_start_from_params(start: Dict[str, Any], params: Dict[str, Any], clusters) -> Dict[str, Any]:
    """
    Starting values updated from the current phase parameters (the
    Microstructure / Dynamics pages) without rebuilding the model:
    Biso, λ, δ1/δ2, δ_g, ρ, fE, Re, Burgers vector and contrast factors.
    """
    from pdf_fitting.project.contrast_factors import contrast

    lc = {str(k).lower(): v for k, v in params.items()}

    def f(name, default):
        try:
            return float(lc[name]) if name in lc else float(default)
        except (TypeError, ValueError):
            return float(default)

    st = dict(start)
    B = dict(st.get("B") or {})
    for el in list(B) or sorted({e for c in clusters for m in c.members for e in m.pair.split("-")}):
        key = f"biso_{str(el).lower()}"
        if key in lc:
            B[el] = f(key, B.get(el, 0.5))
        else:
            vals = [float(v) for k, v in lc.items() if re.match(rf"^biso_{re.escape(str(el).lower())}\d+$", k)]
            if vals:
                B[el] = float(np.mean(vals))
    st["B"] = B
    lam = dict(st.get("lambda") or {})
    for (lp, k) in list(lam):
        lam[(lp, k)] = f(f"lambda_{lp}_{k}", lam[(lp, k)])
    st["lambda"] = lam
    for name, key in (("delta1", "delta1"), ("delta2", "delta2"), ("delta_g", "delta_g"), ("delta_broad", "delta_broad"),
                      ("rho", "rho"), ("fE", "fe")):
        st[name] = f(key, st.get(name, 0.0))
    if f("re", 0.0) > 0:
        st["Re"] = f("re", st.get("Re", 100.0))
    b = f("burgers_mag", 0.0) or f("burgers", 0.0)
    if b > 0:
        st["burgers"] = b
    cell, sg = st.get("cell"), int(st.get("sg") or 0)
    cf_e: Dict[str, float] = {}
    cf_s: Dict[str, float] = {}
    if cell is not None:
        for c in clusters:
            for m in c.members:
                if m.family in cf_e:
                    continue
                hkl = member_hkl(m, cell)
                cf_e[m.family] = float(contrast(params, hkl, 1.0, sg, cell))
                cf_s[m.family] = float(contrast(params, hkl, 0.0, sg, cell))
    st["cf_edge"], st["cf_screw"] = cf_e, cf_s
    return st
