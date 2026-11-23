"""
Free-peak (Pawley-like) PDF fit and direction-resolved peak-width analysis.

Peak positions come from the structure (lattice + atomic positions of the
phase); every *peak group* in a chosen r range gets a free width σ and a free
amplitude multiplier. Everything else — shape function, Qdamp, Qmax
termination, the baseline, the other phases of the dataset and the peaks
outside the range — is the full model, which the fit evaluates with the
group overrides (PDFCalculator._peak_override).

Peak groups
-----------
Shells are first gathered into *orbits*: same unordered element pair, same
distance and symmetry-equivalent vector (crystal point-group rotations,
±). Orbits closer in distance than ``merge_tol`` cannot be told apart in a
one-dimensional PDF and form one *cluster*, which is the fitted unit (one σ,
one amplitude multiplier). A cluster with several orbits is flagged as
merged; its σ is an amplitude-weighted average over its members.

Jacobian
--------
G_peak is linear in the shell Gaussians: G = Interp(Termination(mult·Σ w A N)),
recorded by the model (``_peak_captured['lin_*']``). The Jacobian columns
∂G/∂σ_c and ∂G/∂a_c are therefore computed analytically, cluster by cluster
in parallel (numba), as a sparse banded matrix: one Jacobian costs about one
model evaluation instead of 2 × (number of clusters) evaluations, which is
what makes long r ranges (hundreds of clusters) feasible.

Results give σ ± esd per cluster with its members (pair, λ-shell index k,
direction family, multiplicity, weight) for the Warren-type analysis in
``warren_fit``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import Dataset, Phase, Project

try:
    from numba import njit, prange
except Exception:  # noqa: BLE001
    njit = None


class FreePeakStopped(Exception):
    """Raised inside the fit when the user presses Stop."""


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class PeakMember:
    """One symmetry orbit of shells inside a cluster."""

    pair: str                 # unordered element pair, e.g. "fe-fe"
    lambda_pair: str          # pair label used by lambda_<pair>_<k>
    k: int                    # shell rank of this distance within the pair
    r: float                  # distance (Å)
    family: str               # direction family label, e.g. "<111>"
    vector: Tuple[float, float, float]   # canonical fractional vector
    hkl_H: float              # cubic invariant H of the direction (Wilkens)
    multiplicity: float
    weight: float             # share of the cluster amplitude
    sigma_model: float        # σ of the starting model
    shells: List[int] = field(default_factory=list)
    hkl: Optional[Tuple[int, int, int]] = None   # reciprocal direction ∥ vector (model convention)


@dataclass
class PeakCluster:
    index: int
    r: float                  # amplitude-weighted mean distance
    members: List[PeakMember]
    shells: np.ndarray        # model shell indices
    sigma0: float             # starting σ (weighted rms of the model σ)
    sigma: float = float("nan")
    sigma_esd: float = float("nan")
    amp: float = 1.0
    amp_esd: float = float("nan")

    @property
    def merged(self) -> bool:
        return len(self.members) > 1

    @property
    def reliable(self) -> bool:
        return bool(
            np.isfinite(self.sigma_esd)
            and self.sigma_esd > 0
            and self.sigma_esd / max(self.sigma, 1e-9) <= 0.25
        )

    @property
    def label(self) -> str:
        return " + ".join(f"{m.pair} {m.family} (k={m.k})" for m in self.members)

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "r": self.r,
            "sigma": self.sigma,
            "sigma_esd": self.sigma_esd,
            "amp": self.amp,
            "amp_esd": self.amp_esd,
            "sigma0": self.sigma0,
            "merged": self.merged,
            "members": [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.__dict__.items() if k != "shells"}
                for m in self.members
            ],
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "PeakCluster":
        members = []
        for m in d.get("members", []):
            m = dict(m)
            m["vector"] = tuple(float(x) for x in m.get("vector", (0, 0, 0)))
            if m.get("hkl") is not None:
                m["hkl"] = tuple(int(x) for x in m["hkl"])
            members.append(PeakMember(**{k: v for k, v in m.items() if k in PeakMember.__dataclass_fields__}))
        return cls(
            index=int(d["index"]), r=float(d["r"]), members=members, shells=np.zeros(0, dtype=np.int64),
            sigma0=_f(d.get("sigma0")), sigma=_f(d.get("sigma")), sigma_esd=_f(d.get("sigma_esd")),
            amp=_f(d.get("amp"), 1.0), amp_esd=_f(d.get("amp_esd")),
        )


def _f(v, default=float("nan")) -> float:
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class FreePeakResult:
    clusters: List[PeakCluster]
    r: np.ndarray
    G_exp: np.ndarray
    G_start: np.ndarray
    G_fit: np.ndarray
    rwp_start: float
    rwp: float
    chi2_red: float
    correlation: Optional[np.ndarray]
    param_names: List[str]
    elapsed_s: float
    message: str = ""
    lattice: Dict[str, float] = field(default_factory=dict)
    elements: List[str] = field(default_factory=list)
    merge_tol: float = float("nan")
    nfev: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "clusters": [c.to_json() for c in self.clusters],
            "r": self.r.tolist(), "G_exp": self.G_exp.tolist(), "G_start": self.G_start.tolist(), "G_fit": self.G_fit.tolist(),
            "rwp_start": self.rwp_start, "rwp": self.rwp, "chi2_red": self.chi2_red, "elapsed_s": self.elapsed_s,
            "message": self.message, "lattice": dict(self.lattice), "elements": list(self.elements),
            "merge_tol": self.merge_tol, "nfev": self.nfev,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "FreePeakResult":
        arr = lambda k: np.asarray(d.get(k) or [], dtype=float)  # noqa: E731
        return cls(
            clusters=[PeakCluster.from_json(c) for c in d.get("clusters", [])],
            r=arr("r"), G_exp=arr("G_exp"), G_start=arr("G_start"), G_fit=arr("G_fit"),
            rwp_start=_f(d.get("rwp_start")), rwp=_f(d.get("rwp")), chi2_red=_f(d.get("chi2_red")),
            correlation=None, param_names=[], elapsed_s=_f(d.get("elapsed_s"), 0.0), message=str(d.get("message", "")),
            lattice=dict(d.get("lattice") or {}), elements=list(d.get("elements") or []),
            merge_tol=_f(d.get("merge_tol")), nfev=int(d.get("nfev", 0) or 0),
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _rwp(y, f) -> float:
    den = float(np.sum(y * y))
    return 100.0 * math.sqrt(float(np.sum((y - f) ** 2)) / den) if den > 0 else float("nan")


def _family_label(vec: np.ndarray) -> str:
    """'<111>' when the vector is a rational multiple of an integer direction."""
    v = np.asarray(vec, dtype=float)
    for m in range(1, 13):
        w = v * m
        if np.allclose(w, np.round(w), atol=2e-3):
            ints = np.round(w).astype(int)
            nz = np.abs(ints[ints != 0])
            g = int(np.gcd.reduce(nz)) if nz.size else 1
            ints = ints // max(g, 1)
            sep = " " if np.any(np.abs(ints) >= 10) else ""
            return "<" + sep.join(str(int(x)) for x in ints) + ">"
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _cubic_H(vec: np.ndarray) -> float:
    h, k, l = (float(x) for x in vec)
    s = h * h + k * k + l * l
    if s <= 0:
        return 0.0
    return (h * h * k * k + h * h * l * l + k * k * l * l) / (s * s)


def _unique_rotations(rotations) -> np.ndarray:
    out: List[np.ndarray] = []
    for R in rotations:
        R = np.asarray(R, dtype=float)
        if not any(np.allclose(R, Q) for Q in out):
            out.append(R)
    return np.asarray(out or [np.eye(3)], dtype=float)


def canonical_vectors(vecs: np.ndarray, rotations: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """
    Lexicographically largest image of each fractional vector under the
    point-group rotations and ± (vectorized; rounded to 1e-4).
    """
    vecs = np.asarray(vecs, dtype=float)
    R = np.concatenate([rotations, -rotations], axis=0)          # (2n_rot, 3, 3)
    out = np.empty_like(vecs)
    scale = 1e4 if float(np.max(np.abs(vecs), initial=0.0)) < 100.0 else 1e3
    off = 1 << 20
    for s in range(0, vecs.shape[0], chunk):
        v = vecs[s:s + chunk]
        img = np.einsum("rij,mj->mri", R, v)                     # (m, 2n_rot, 3)
        q = np.rint(img * scale).astype(np.int64) + off
        key = (q[..., 0] << 42) | (q[..., 1] << 21) | q[..., 2]
        best = np.argmax(key, axis=1)
        out[s:s + chunk] = np.round(img[np.arange(v.shape[0]), best], 4) + 0.0
    return out


def _phase_calc(model):
    return getattr(model, "pdf_calculator", None) or model


# -----------------------------------------------------------------------------
# Analytic Jacobian kernel
# -----------------------------------------------------------------------------

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

if njit is not None:

    @njit(cache=True, parallel=True)
    def _jac_columns(
        r_int, mult, kern, half,
        sh_r, sh_coef, sh_off,            # shells sorted by cluster; coef = w_pair·A
        sigma_c, amp_c,
        win_lo, win_hi,                   # per-cluster internal window [lo, hi)
        r_rows, row_lo, row_hi,           # fitted output rows and per-cluster row range
        identity_rows, row_int_idx,       # identity: rows are internal indices
        out_off, out_rows, out_sig, out_amp,
    ):
        nc = sigma_c.shape[0]
        n_int = r_int.shape[0]
        dr = r_int[1] - r_int[0]
        for c in prange(nc):
            lo = win_lo[c]
            hi = win_hi[c]
            L = hi - lo
            fa = np.zeros(L)
            fs = np.zeros(L)
            sg = sigma_c[c]
            inv = 1.0 / sg
            for s in range(sh_off[c], sh_off[c + 1]):
                rij = sh_r[s]
                a0 = int(math.ceil((rij - 5.0 * sg - r_int[0]) / dr - 1e-9))
                a1 = int(math.ceil((rij + 5.0 * sg - r_int[0]) / dr - 1e-9))
                if a0 < lo:
                    a0 = lo
                if a1 > hi:
                    a1 = hi
                pref = sh_coef[s] * _INV_SQRT_2PI * inv
                for i in range(a0, a1):
                    x = (r_int[i] - rij) * inv
                    g = pref * math.exp(-0.5 * x * x)
                    fa[i - lo] += g
                    fs[i - lo] += g * (x * x - 1.0) * inv
            for i in range(L):
                m = mult[lo + i]
                fa[i] *= m
                fs[i] *= m * amp_c[c]
            # Termination (symmetric kernel, zero outside the window).
            t0 = max(lo - half, 0)
            t1 = min(hi + half, n_int)
            T = t1 - t0
            ta = np.zeros(T)
            ts = np.zeros(T)
            for j in range(T):
                jj = t0 + j
                sa = 0.0
                ss = 0.0
                k0 = max(lo, jj - half)
                k1 = min(hi, jj + half + 1)
                for k in range(k0, k1):
                    w = kern[k - jj + half]
                    sa += w * fa[k - lo]
                    ss += w * fs[k - lo]
                ta[j] = sa
                ts[j] = ss
            # Output rows.
            o = out_off[c]
            for q in range(row_lo[c], row_hi[c]):
                if identity_rows:
                    j = row_int_idx[q] - t0
                    va = ta[j] if 0 <= j < T else 0.0
                    vs = ts[j] if 0 <= j < T else 0.0
                else:
                    x = (r_rows[q] - r_int[0]) / dr
                    i0 = int(math.floor(x))
                    f = x - i0
                    j = i0 - t0
                    va = 0.0
                    vs = 0.0
                    if 0 <= j < T:
                        va += (1.0 - f) * ta[j]
                        vs += (1.0 - f) * ts[j]
                    if 0 <= j + 1 < T:
                        va += f * ta[j + 1]
                        vs += f * ts[j + 1]
                out_rows[o] = q
                out_sig[o] = vs
                out_amp[o] = va
                o += 1


def _termination_kernel(qmax: float, n_zeros: int, dr: float, max_kernel_points: int = 2001):
    half = int(math.ceil(float(n_zeros) * math.pi / (qmax * dr)))
    half = max(1, min(half, max(1, int(max_kernel_points) // 2)))
    x = np.arange(-half, half + 1, dtype=float) * dr
    k = np.empty_like(x)
    m0 = np.abs(x) < 1e-14
    k[m0] = qmax / math.pi
    k[~m0] = np.sin(qmax * x[~m0]) / (math.pi * x[~m0])
    return k * dr, half


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


class FreePeakFitter:
    """
    Builds the dataset models (target phase on the CPU path), the peak
    clusters of the target phase in [r_lo, r_hi], and fits them.
    """

    def __init__(
        self,
        project: Project,
        dataset: Dataset,
        phase: Phase,
        *,
        r_lo: float,
        r_hi: float,
        merge_tol: Optional[float] = None,
        min_weight: float = 0.02,
        status: Optional[Callable[[str], None]] = None,
        stop_event=None,
    ):
        from pdf_fitting.models.model_factory import build_pdf_model
        from pdf_fitting.project.multiphase import load_gr_data
        from pdf_fitting.structure_handler import symmetry_operations

        self.project = project
        self.dataset = dataset
        self.phase = phase
        self.r_lo = float(r_lo)
        self.r_hi = float(r_hi)
        self.status = status or (lambda _m: None)
        self.stop_event = stop_event

        margin = 3.0
        r_all, G_all = load_gr_data(project.resolve_path(dataset.data_file), dataset.r_min, dataset.r_max)
        mask = (r_all >= self.r_lo) & (r_all <= self.r_hi + margin)
        self.r = r_all[mask]
        self.G = G_all[mask]
        self.fit_mask = self.r <= self.r_hi
        if self.r.size < 20:
            raise ValueError("Too few data points in the selected r range.")

        r_max_model = min(float(dataset.r_max), self.r_hi + margin)
        cutoff = r_max_model * float(dataset.r_extension) + 2.0

        def build(ph: Phase):
            cfg = project.phase_legacy_config(dataset, ph)
            cfg["r_max"] = r_max_model
            cfg["refinement"] = dict(cfg.get("refinement", {}) or {})
            cfg["refinement"]["pdf_backend"] = "cpu"
            return build_pdf_model(config=cfg, r_exp=self.r, pair_cutoff=cutoff)

        self.status(f"Building model of {phase.name}…")
        self.model = build(phase)
        self.calc = _phase_calc(self.model)
        if getattr(self.calc, "_uses_fixed_stacked_geometry", lambda: False)():
            raise ValueError("Free-peak analysis supports conventional (periodic) phases only.")
        self._check_stop()

        self.params = {k: p.value for k, p in dataset.parameters.items()}
        self.params.update({k: p.value for k, p in phase.parameters.items()})

        # Other phases of the dataset: fixed contribution.
        self.G_other = np.zeros_like(self.r)
        for other in dataset.active_phases():
            if other is phase:
                continue
            self.status(f"Building model of {other.name}…")
            m = build(other)
            local = {k: p.value for k, p in dataset.parameters.items()}
            local.update({k: p.value for k, p in other.parameters.items()})
            self.G_other += np.asarray(m.evaluate(local, r_override=self.r)["G_r"], dtype=float)
            self._check_stop()

        # Starting model with capture of every shell's σ / amplitude.
        self.calc._peak_captured = {}
        self.calc._peak_capture = True
        try:
            self.G_start = self._evaluate(None)
        finally:
            self.calc._peak_capture = False
        cap = getattr(self.calc, "_peak_captured", None)
        if not cap or "r_ij" not in cap:
            raise RuntimeError("The model did not expose its peak table (CPU path required).")
        self.cap = cap

        handler = self.calc.structure_handler
        self.rotations = _unique_rotations([R for R, _t in symmetry_operations(handler.structure)])
        self.lattice = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), handler.structure.lattice.parameters))
        self.status("Grouping peaks by pair, distance and direction…")
        self.clusters = self._build_clusters(merge_tol, min_weight)

    # ------------------------------------------------------------------
    def _check_stop(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise FreePeakStopped("Stopped by the user.")

    def _evaluate(self, override) -> np.ndarray:
        self.calc._peak_override = override
        try:
            out = self.model.evaluate(dict(self.params), r_override=self.r)
        finally:
            self.calc._peak_override = None
        return np.asarray(out["G_r"], dtype=float) + self.G_other

    def _build_clusters(self, merge_tol, min_weight) -> List[PeakCluster]:
        calc = self.calc
        r_ij = np.asarray(self.cap["r_ij"], dtype=float)
        sigma = np.asarray(self.cap["sigma"], dtype=float)
        A = np.asarray(self.cap["A_ij"], dtype=float)
        w_pair = np.asarray(calc._w_ordered, dtype=float)[np.asarray(calc._shell_pair_id, dtype=int)]
        weight_all = np.abs(A * w_pair) * r_ij  # peak area in G(r) (∝ scale·4πρ0·r·w·A)
        species = list(calc._unique_species_for_shells)
        ns = len(species)
        a_id = np.asarray(calc._shell_alpha_id, dtype=np.int64)
        b_id = np.asarray(calc._shell_beta_id, dtype=np.int64)
        upair = np.asarray(calc._shell_upair_id, dtype=int)
        kk = np.asarray(calc._shell_k_within_upair, dtype=int)
        frac = np.asarray(calc._shell_frac, dtype=float)
        mult = np.asarray(calc._shell_mult, dtype=float)
        lam_pairs = list(calc._upair_list)
        shell_dir = getattr(calc, "_shell_direction", None)

        sel = np.flatnonzero((r_ij >= self.r_lo - 1e-6) & (r_ij <= self.r_hi))
        if sel.size == 0:
            raise ValueError("No peaks of this phase in the selected r range.")
        canon = canonical_vectors(frac[sel], self.rotations)
        pair_code = np.minimum(a_id[sel], b_id[sel]) * ns + np.maximum(a_id[sel], b_id[sel])
        r_code = np.rint(r_ij[sel] * 1000.0).astype(np.int64)
        cq = np.rint(canon * 1e4).astype(np.int64)
        keys = np.stack([pair_code, r_code, cq[:, 0], cq[:, 1], cq[:, 2]], axis=1)
        _u, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        n_orb = first.size
        w_sel = weight_all[sel]
        o_mult = np.bincount(inv, weights=mult[sel], minlength=n_orb)
        o_w = np.bincount(inv, weights=w_sel, minlength=n_orb)
        o_s2w = np.bincount(inv, weights=w_sel * sigma[sel] ** 2, minlength=n_orb)
        o_r = r_ij[sel][first]

        sig_o = np.sqrt(o_s2w / np.maximum(o_w, 1e-30))
        tol = float(merge_tol) if merge_tol else max(0.5 * float(np.median(sig_o)), 0.01)
        self.merge_tol = tol

        # Drop orbits too weak to matter (they keep the model σ).
        keep = o_w >= min_weight * float(o_w.max())
        order = [o for o in np.argsort(o_r, kind="stable") if keep[o]]

        groups: List[List[int]] = []
        for o in order:
            # A group spans at most ``tol`` (no chaining of a dense sequence
            # of peaks into one wide group at long r).
            if groups and o_r[o] - o_r[groups[-1][0]] < tol:
                groups[-1].append(int(o))
            else:
                groups.append([int(o)])

        cluster_of_orbit = np.full(n_orb, -1, dtype=np.int64)
        for ci, g in enumerate(groups):
            cluster_of_orbit[g] = ci
        cl_of_shell = cluster_of_orbit[inv]
        ok = cl_of_shell >= 0
        shells_sorted = sel[ok][np.argsort(cl_of_shell[ok], kind="stable")]
        counts = np.bincount(cl_of_shell[ok], minlength=len(groups))
        offs = np.concatenate([[0], np.cumsum(counts)])

        clusters = []
        for ci, g in enumerate(groups):
            wsum = float(sum(o_w[o] for o in g)) or 1.0
            members = []
            for o in g:
                s0 = sel[first[o]]
                lp = lam_pairs[upair[s0]] if upair[s0] < len(lam_pairs) else (species[a_id[s0]].lower(), species[b_id[s0]].lower())
                vec = canon[first[o]]
                members.append(
                    PeakMember(
                        pair="-".join(sorted((species[a_id[s0]].lower(), species[b_id[s0]].lower()))),
                        lambda_pair="-".join(str(x) for x in lp),
                        k=int(kk[s0]),
                        r=float(o_r[o]),
                        family=_family_label(vec),
                        vector=tuple(float(x) for x in vec),
                        hkl_H=_cubic_H(vec),
                        multiplicity=float(o_mult[o]),
                        weight=float(o_w[o]) / wsum,
                        sigma_model=float(sig_o[o]),
                        hkl=tuple(int(x) for x in shell_dir[s0]) if shell_dir is not None else None,
                    )
                )
            shells = np.sort(shells_sorted[offs[ci]:offs[ci + 1]])
            s0 = math.sqrt(sum(o_s2w[o] for o in g) / wsum)
            r_mean = sum(o_w[o] * o_r[o] for o in g) / wsum
            clusters.append(PeakCluster(index=ci, r=float(r_mean), members=members, shells=shells.astype(np.int64), sigma0=float(s0)))
        return clusters

    # ------------------------------------------------------------------
    def _prepare_jacobian(self):
        """Static data of the analytic Jacobian (shells sorted by cluster)."""
        cap = self.cap
        if njit is None or "lin_mult" not in cap:
            return None
        calc = self.calc
        idx = np.concatenate([c.shells for c in self.clusters])
        owner = np.concatenate([np.full(c.shells.size, j) for j, c in enumerate(self.clusters)])
        w_pair = np.asarray(cap["lin_w_pair"], dtype=float)[np.asarray(calc._shell_pair_id, dtype=int)[idx]]
        coef = w_pair * np.asarray(cap["A_ij"], dtype=float)[idx]
        r_int = np.asarray(cap["lin_r"], dtype=float)
        if r_int.size < 2:
            return None
        dr = float(np.median(np.diff(r_int)))
        if dr <= 0 or np.max(np.abs(np.diff(r_int) - dr)) > 1e-2 * dr:
            return None
        # The model grid is float32; rebuild it exactly (uniform) in float64.
        r_int = float(r_int[0]) + dr * np.arange(r_int.size, dtype=float)
        if cap["lin_qmax"] > 0:
            kern, half = _termination_kernel(cap["lin_qmax"], cap["lin_qmax_zeros"], dr)
        else:
            kern, half = np.ones(1), 0
        r_rows = np.asarray(self.r[self.fit_mask], dtype=float)
        identity = not cap["lin_internal"]
        row_int_idx = np.zeros(r_rows.size, dtype=np.int64)
        if identity:
            r_out = np.asarray(cap["lin_r_out"], dtype=float)
            if r_out.size != r_int.size:
                return None
            row_int_idx = np.flatnonzero(self.fit_mask).astype(np.int64)
        offs = np.concatenate([[0], np.cumsum([c.shells.size for c in self.clusters])]).astype(np.int64)
        return {
            "r_int": r_int, "mult": np.asarray(cap["lin_mult"], dtype=float), "kern": np.asarray(kern, dtype=float), "half": int(half),
            "sh_r": np.asarray(cap["r_ij"], dtype=float)[idx], "sh_coef": coef, "sh_off": offs, "owner": owner,
            "r_rows": r_rows, "identity": identity, "row_int_idx": row_int_idx, "dr": dr,
        }

    def _jacobian(self, x, J):
        from scipy import sparse

        nc = len(self.clusters)
        sig = np.asarray(x[:nc], dtype=float)
        amp = np.asarray(x[nc:], dtype=float)
        r_int, dr, half = J["r_int"], J["dr"], J["half"]
        n_int = r_int.size
        sh_r, owner = J["sh_r"], J["owner"]
        s_sh = sig[owner]
        lo_sh = np.ceil((sh_r - 5.0 * s_sh - r_int[0]) / dr - 1e-9).astype(np.int64)
        hi_sh = np.ceil((sh_r + 5.0 * s_sh - r_int[0]) / dr - 1e-9).astype(np.int64)
        offs = J["sh_off"]
        win_lo = np.clip(np.minimum.reduceat(lo_sh, offs[:-1]), 0, n_int - 1)
        win_hi = np.clip(np.maximum.reduceat(hi_sh, offs[:-1]), 1, n_int)
        win_hi = np.maximum(win_hi, win_lo + 1)
        t_lo = np.maximum(win_lo - half, 0)
        t_hi = np.minimum(win_hi + half, n_int)
        r_rows = J["r_rows"]
        if J["identity"]:
            rid = J["row_int_idx"]
            row_lo = np.searchsorted(rid, t_lo, side="left")
            row_hi = np.searchsorted(rid, t_hi, side="left")
        else:
            row_lo = np.searchsorted(r_rows, r_int[0] + (t_lo - 1) * dr, side="right")
            row_hi = np.searchsorted(r_rows, r_int[0] + t_hi * dr, side="left")
        row_hi = np.maximum(row_hi, row_lo)
        out_off = np.concatenate([[0], np.cumsum(row_hi - row_lo)]).astype(np.int64)
        nnz = int(out_off[-1])
        out_rows = np.zeros(nnz, dtype=np.int64)
        out_sig = np.zeros(nnz)
        out_amp = np.zeros(nnz)
        # Coefficients include the current amplitude multiplier for ∂/∂σ and ∂/∂a.
        _jac_columns(
            r_int, J["mult"], J["kern"], half,
            sh_r, J["sh_coef"], offs, sig, amp,
            win_lo.astype(np.int64), win_hi.astype(np.int64),
            r_rows, row_lo.astype(np.int64), row_hi.astype(np.int64),
            bool(J["identity"]), J["row_int_idx"],
            out_off, out_rows, out_sig, out_amp,
        )
        cols_sig = np.repeat(np.arange(nc), np.diff(out_off))
        rows = np.concatenate([out_rows, out_rows])
        cols = np.concatenate([cols_sig, cols_sig + nc])
        vals = -np.concatenate([out_sig, out_amp])     # residual = y − G
        return sparse.csr_matrix((vals, (rows, cols)), shape=(r_rows.size, 2 * nc))

    def fit(self, max_nfev: int = 200, *, analytic_jacobian: bool = True) -> FreePeakResult:
        from scipy.optimize import least_squares

        t0 = time.perf_counter()
        nc = len(self.clusters)
        index = np.concatenate([c.shells for c in self.clusters])
        owner = np.concatenate([np.full(c.shells.size, j) for j, c in enumerate(self.clusters)])

        def override(x):
            sig = x[:nc][owner]
            amp = x[nc:][owner]
            return {"index": index, "sigma": sig, "amp_scale": amp}

        y = self.G[self.fit_mask]
        s0 = np.array([c.sigma0 for c in self.clusters])
        x0 = np.r_[s0, np.ones(nc)]
        # Widths within 0.3–3× the model, amplitudes within 0.3–3: positions
        # and multiplicities are known, so larger excursions only happen for
        # unresolved / weak peaks (flagged unreliable instead).
        lb = np.r_[np.maximum(0.3 * s0, 1e-3), np.full(nc, 0.3)]
        ub = np.r_[3.0 * s0, np.full(nc, 3.0)]
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        calls = {"n": 0, "j": 0}
        t_last = [time.perf_counter()]

        def residual(x):
            self._check_stop()
            calls["n"] += 1
            if time.perf_counter() - t_last[0] > 1.0:
                t_last[0] = time.perf_counter()
                self.status(f"Free-peak fit: {nc} peak groups · {calls['n']} evaluations · {calls['j']} Jacobians…")
            return (y - self._evaluate(override(x))[self.fit_mask])

        Jdata = self._prepare_jacobian() if analytic_jacobian else None
        p = x0.size
        kw: Dict[str, Any] = dict(method="trf", x_scale="jac", max_nfev=int(max_nfev), xtol=1e-8, ftol=1e-9)
        if Jdata is not None:
            dense = p <= 600

            def jac(x):
                self._check_stop()
                calls["j"] += 1
                J = self._jacobian(x, Jdata)
                return J.toarray() if dense else J

            kw["jac"] = jac
            if not dense:
                kw["tr_solver"] = "lsmr"
        else:
            kw["diff_step"] = 1e-4
        res = least_squares(residual, x0, bounds=(lb, ub), **kw)
        self._check_stop()

        n = y.size
        chi2_red = float(2.0 * res.cost / max(n - p, 1))
        Jf = res.jac
        if Jdata is not None:
            Jf = self._jacobian(res.x, Jdata)
        esd, corr = _esd_from_jacobian(Jf, chi2_red, keep_corr=p <= 400)

        for j, c in enumerate(self.clusters):
            c.sigma, c.sigma_esd = float(res.x[j]), float(esd[j])
            c.amp, c.amp_esd = float(res.x[nc + j]), float(esd[nc + j])

        G_fit = self._evaluate(override(res.x))
        names = [f"sigma_{j}" for j in range(nc)] + [f"amp_{j}" for j in range(nc)]
        return FreePeakResult(
            clusters=self.clusters,
            r=self.r[self.fit_mask],
            G_exp=y,
            G_start=self.G_start[self.fit_mask],
            G_fit=G_fit[self.fit_mask],
            rwp_start=_rwp(y, self.G_start[self.fit_mask]),
            rwp=_rwp(y, G_fit[self.fit_mask]),
            chi2_red=chi2_red,
            correlation=corr,
            param_names=names,
            elapsed_s=time.perf_counter() - t0,
            message=str(res.message),
            lattice=self.lattice,
            elements=list(self.calc._unique_species_for_shells),
            merge_tol=self.merge_tol,
            nfev=int(res.nfev),
        )


def _esd_from_jacobian(J, chi2_red: float, *, keep_corr: bool):
    """Parameter esds (and the correlation matrix when small) from J."""
    from scipy import sparse

    p = J.shape[1]
    if sparse.issparse(J):
        JTJ = (J.T @ J).tocsc()
        if p <= 3000:
            JTJ = JTJ.toarray()
    else:
        JTJ = J.T @ J
    if not sparse.issparse(JTJ):
        try:
            cov = np.linalg.pinv(JTJ) * chi2_red
        except np.linalg.LinAlgError:
            return np.full(p, np.nan), None
        esd = np.sqrt(np.clip(np.diag(cov), 0, None))
        if not keep_corr:
            return esd, None
        d = np.where(esd > 0, esd, 1.0)
        return esd, cov / np.outer(d, d)
    # Large problems: diagonal of the inverse via sparse LU, in column blocks.
    from scipy.sparse.linalg import splu

    try:
        lu = splu(JTJ + sparse.identity(p, format="csc") * (1e-12 * float(JTJ.diagonal().max())))
    except Exception:  # noqa: BLE001
        return np.full(p, np.nan), None
    diag = np.empty(p)
    for s in range(0, p, 256):
        e = min(p, s + 256)
        B = np.zeros((p, e - s))
        B[np.arange(s, e), np.arange(e - s)] = 1.0
        X = lu.solve(B)
        diag[s:e] = X[np.arange(s, e), np.arange(e - s)]
    return np.sqrt(np.clip(diag * chi2_red, 0, None)), None
