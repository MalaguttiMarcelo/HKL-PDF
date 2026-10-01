# -*- coding: utf-8 -*-

import numpy as np

try:
    from numba import get_num_threads
except Exception:  # pragma: no cover
    get_num_threads = None
from scipy.special import erfc, erf
import re
import warnings
from collections import defaultdict

import time
import os

_QMAX_KERNEL_CACHE = {}

from pdf_fitting.models.numba_kernels import (
    accumulate_gaussians_by_pair,
    accumulate_gaussians_by_shell_blocks,
    accumulate_pair_gamma_histogram_by_pair,
)

from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pdf_fitting.structure_handler import StructureHandler
from pdf_fitting.models.microstrain_utils import contrast_factor, fstar_vec, feature_vector_hkl

# If you added Cell in microstrain_utils, import it; otherwise define a tiny fallback.
try:
    from pdf_fitting.models.microstrain_utils import Cell  # type: ignore
except Exception:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Cell:
        a: float
        b: float
        c: float
        alpha: float = 90.0
        beta: float = 90.0
        gamma: float = 90.0


def clean_element_name(name: str) -> str:
    m = re.match(r"([A-Za-z]{1,2})", name)
    if not m:
        raise ValueError(f"Could not parse chemical symbol from '{name}'")
    sym = m.group(1)
    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()

def lambda_safe_species_label(name: str) -> str:
    """
    Convert species labels to safe lambda parameter labels.

    Examples
    --------
    Fe0+ -> fe0plus
    Fe2+ -> fe2plus
    O2-  -> o2minus
    """
    s = str(name).strip()
    s = s.replace("+", "plus")
    s = s.replace("-", "minus")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.lower()

def fractional_to_cartesian(frac_vector, a, b, c, alpha, beta, gamma):
    """
    General triclinic fractional->cartesian. Angles in degrees.
    Matches the original July implementation.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    alpha_r = np.deg2rad(float(alpha))
    beta_r = np.deg2rad(float(beta))
    gamma_r = np.deg2rad(float(gamma))

    cosa = np.cos(alpha_r)
    cosb = np.cos(beta_r)
    cosg = np.cos(gamma_r)
    sing = np.sin(gamma_r)

    volume = a * b * c * np.sqrt(1 - cosa**2 - cosb**2 - cosg**2 + 2 * cosa * cosb * cosg)

    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * cosg, b * sing, 0.0
    cx = c * cosb
    cy = c * (cosa - cosb * cosg) / sing
    cz = volume / (a * b * sing)

    matrix = np.array([[ax, bx, cx], [ay, by, cy], [az, bz, cz]], dtype=float)
    return matrix @ frac_vector


def lattice_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the 3x3 matrix M such that cart = frac @ M.T for triclinic cells."""
    a = float(a)
    b = float(b)
    c = float(c)
    alpha_r = np.deg2rad(float(alpha))
    beta_r = np.deg2rad(float(beta))
    gamma_r = np.deg2rad(float(gamma))

    cosa = np.cos(alpha_r)
    cosb = np.cos(beta_r)
    cosg = np.cos(gamma_r)
    sing = np.sin(gamma_r)

    volume = a * b * c * np.sqrt(1 - cosa**2 - cosb**2 - cosg**2 + 2 * cosa * cosb * cosg)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * cosg, b * sing, 0.0
    cx = c * cosb
    cy = c * (cosa - cosb * cosg) / sing
    cz = volume / (a * b * sing)
    return np.array([[ax, bx, cx], [ay, by, cy], [az, bz, cz]], dtype=float)

def apply_qmax_termination(
    G,
    r,
    qmax,
    n_zeros=8,
    max_kernel_points=2001,
    pad_mode="zero",
    use_sine_kernel=True,
    normalize=False,
):
    """
    Apply finite-Qmax termination to G(r).

    The finite sine-transform kernel is

        K(r, r') = 1/pi * [
            sin(Qmax * (r-r')) / (r-r')
            -
            sin(Qmax * (r+r')) / (r+r')
        ]

    The input grid should preferably begin at r=0. PDFCalculator.evaluate()
    constructs such an internal grid before calling this function.
    """
    qmax = float(qmax)

    if qmax <= 0.0:
        return np.asarray(G, dtype=float)

    G = np.asarray(G, dtype=float)
    r = np.asarray(r, dtype=float)

    if G.size < 2 or r.size < 2:
        return G.copy()

    if G.shape != r.shape:
        raise ValueError("G and r must have the same shape.")

    dr_values = np.diff(r)
    dr = float(np.median(dr_values))

    if not np.isfinite(dr) or dr <= 0.0:
        return G.copy()

    half_width = int(
        np.ceil(
            float(n_zeros)
            * np.pi
            / (qmax * dr)
        )
    )

    half_width = max(1, half_width)

    max_half = max(
        1,
        int(max_kernel_points) // 2,
    )

    half_width = min(
        half_width,
        max_half,
    )

    cache_key = (
        "qmax_kernel_v2",
        round(dr, 12),
        round(qmax, 8),
        int(n_zeros),
        int(max_kernel_points),
        int(half_width),
        str(pad_mode),
        bool(normalize),
        bool(use_sine_kernel),
    )

    cached = _QMAX_KERNEL_CACHE.get(
        cache_key,
        None,
    )

    if cached is None:
        x = (
            np.arange(
                -half_width,
                half_width + 1,
                dtype=float,
            )
            * dr
        )

        kernel = np.empty_like(
            x,
            dtype=float,
        )

        mask0 = np.abs(x) < 1.0e-14

        kernel[mask0] = qmax / np.pi

        kernel[~mask0] = (
            np.sin(qmax * x[~mask0])
            / (np.pi * x[~mask0])
        )

        kernel *= dr

        correction_scale = 1.0

        if normalize:
            kernel_sum = float(
                np.sum(kernel)
            )

            if abs(kernel_sum) > 1.0e-14:
                correction_scale = 1.0 / kernel_sum
                kernel *= correction_scale

        cached = (
            kernel,
            float(correction_scale),
        )

        _QMAX_KERNEL_CACHE[cache_key] = cached

    kernel, correction_scale = cached

    if pad_mode == "edge":
        padded = np.pad(
            G,
            (half_width, half_width),
            mode="edge",
        )

    elif pad_mode == "reflect":
        padded = np.pad(
            G,
            (half_width, half_width),
            mode="reflect",
        )

    else:
        padded = np.pad(
            G,
            (half_width, half_width),
            mode="constant",
            constant_values=0.0,
        )

    out = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    if not use_sine_kernel:
        return np.asarray(
            out,
            dtype=float,
        )

    width = float(half_width) * dr
    n = G.size

    for i in range(n):
        ri = float(r[i])
        max_rj = width - ri

        if max_rj < float(r[0]):
            continue

        jmax = int(
            np.searchsorted(
                r,
                max_rj,
                side="right",
            )
        )

        if jmax <= 0:
            continue

        x2 = ri + r[:jmax]

        k2 = np.empty_like(
            x2,
            dtype=float,
        )

        mask0 = np.abs(x2) < 1.0e-14

        k2[mask0] = qmax / np.pi

        k2[~mask0] = (
            np.sin(qmax * x2[~mask0])
            / (np.pi * x2[~mask0])
        )

        k2 *= float(correction_scale)

        out[i] -= float(
            np.sum(
                G[:jmax] * k2
            )
            * dr
        )

    return np.asarray(
        out,
        dtype=float,
    )

def apply_lattice_constraints(lattice: dict, constraints) -> dict:
    """
    Backward compatible:
    - constraints is a dict like {"b":"a","c":"a","alpha":90,...}
    - or an object with apply_constraints()
    """
    if constraints is None:
        return lattice

    if hasattr(constraints, "apply_constraints") and callable(getattr(constraints, "apply_constraints")):
        return constraints.apply_constraints(lattice)

    if isinstance(constraints, dict):
        out = dict(lattice)

        # Equality constraints first
        for _ in range(5):
            changed = False
            for k, v in constraints.items():
                if isinstance(v, str) and v in out:
                    if out.get(k) != out[v]:
                        out[k] = out[v]
                        changed = True
            if not changed:
                break

        # Fixed values
        for k, v in constraints.items():
            if not isinstance(v, str):
                out[k] = float(v)

        return out

    return lattice


class PDFCalculator:
    """
    Restored July PDF math:
      - build partial g_ab(r) with proper A_ij = m_ij/(4π r_ij^2 ρ_beta)
      - combine using Faber–Ziman weights
      - convert to PDF: G(r) = 4π ρ0 r (g(r) - 1)
      - apply size shape factor gamma(r) and qdamp
    """

    def __init__(self, config, structure_file=None, r_exp=None, pair_cutoff=None, **kwargs):
        self.config = config

        # ------------------------------------------------------------------
        # Performance settings
        # ------------------------------------------------------------------
        # Use float32 in the hot path to reduce memory bandwidth.
        # (Outputs are cast back to float64 for compatibility with SciPy fitting.)
        self._dtype = np.float32 if bool(config.get("use_float32", True)) else float

        # Numba threading controls (must be set BEFORE first parallel kernel compile/run)
        try:
            import numba  # type: ignore

            layer = str(config.get("numba_threading_layer", "")).strip().lower()
            if layer:
                # Accepted values: "tbb", "omp", "workqueue"
                numba.config.THREADING_LAYER = layer

            n_threads = config.get("numba_threads", None)

            # If user did not specify numba_threads, use all logical CPUs.
            if n_threads is None:
                n_threads = os.cpu_count() or 1

            try:
                numba.set_num_threads(int(n_threads))
            except Exception:
                pass

            try:
                print(f"[NUMBA] PDF kernels using {numba.get_num_threads()} thread(s)")
            except Exception:
                pass
        except Exception:
            # Numba not installed or not importable; kernels will raise on use.
            pass

        if structure_file is None:
            structure_file = config["structure_file"]

        self.structure_file = structure_file
        self.gr_data_file = config.get("gr_data_file")

        self.r_min = float(config["r_min"])
        self.r_max = float(config["r_max"])

        # experimental r-grid (important: must match FitEngine residuals)
        self.r = None if r_exp is None else np.asarray(r_exp, dtype=float)

        # pair cutoff
        r_extension = config.get("r_extension", config.get("pair_generation", {}).get("r_extension", 1.0))
        self.r_extension = float(r_extension)
        self.pair_cutoff = float(pair_cutoff) if pair_cutoff is not None else self.r_max * self.r_extension

        # Structure + shells
        self.structure_handler = StructureHandler(structure_file, self.pair_cutoff)
        self.constraints = self.structure_handler.get_constraints()
        # Prefer packed shell table arrays (fast load, avoids Python dict overhead)
        shell_table = None
        if hasattr(self.structure_handler, "get_shell_table_arrays"):
            try:
                shell_table = self.structure_handler.get_shell_table_arrays()
            except Exception:
                shell_table = None

        if shell_table is not None:
            self._load_shell_table_arrays(shell_table)
            self.cached_shells = None
        else:
            self.cached_shells = self.structure_handler.get_symmetry_shells()
            # --- Prepack shell information into NumPy arrays (critical for speed) ---
            self._build_shell_table(self.cached_shells)
        self.cached_lattice_tuple = None


        # [contrast_factors] is deprecated/removed.
        # Contrast-factor coefficients are now normal parameters in:
        #   [initial_values]
        #   [refinable_parameters]
        # Examples:
        #   Cubic:    CEdgeA, CEdgeB, CScrewA, CScrewB
        #   Non-cubic EdgeE1..., ScrewE1...
        self.contrast_coeffs = {}

        # Symmetry + cell (for non-cubic invariant Chkl)
        sga = SpacegroupAnalyzer(self.structure_handler.structure, symprec=1e-3)
        self.spacegroup_number = int(sga.get_space_group_number())
        a, b, c, alpha, beta, gamma = self.structure_handler.structure.lattice.parameters
        self.cell = Cell(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))

        # [contrast_factors] is deprecated/removed.
        # Contrast-factor coefficients are now normal parameters in:
        #   [initial_values]
        #   [refinable_parameters]
        # Examples:
        #   Cubic:    CEdgeA, CEdgeB, CScrewA, CScrewB
        #   Non-cubic EdgeE1..., ScrewE1...
        self.contrast_coeffs = {}

        # Optional Burgers magnitude is also now a normal parameter.
        # Recommended input:
        #   [initial_values]
        #   burgers_mag = ...
        self.burgers_mag = None
        try:
            init_lc = {
                str(k).strip().lower(): v
                for k, v in (config.get("initial", {}) or {}).items()
            }
            for k in ("burgers_mag", "burgers", "b_mag", "b"):
                if k in init_lc:
                    self.burgers_mag = float(init_lc[k])
                    break
        except Exception:
            self.burgers_mag = None

        # Do not build the Chkl feature matrix during normal initialization.
        # It can be very expensive for large shell tables, and cylinder/disk
        # fits usually do not need it unless Wilkens/PAH microstrain is active.
        #
        # The matrix is now built lazily inside _compute_chkl_unique_dirs_fast().
        self._cf_feature_names = []
        self._cf_feature_matrix = None

        # rho0 will be computed in evaluate
        self.rho0 = None

        # --- Precompute composition constants (do not depend on refined lattice volume) ---
        self._init_composition_constants()
        self._printed_qmax_debug = False
        self._printed_qmax_delta = False

        # Optional profiling of PDF evaluation
        refcfg = (config.get("refinement") or {}) if isinstance(config, dict) else {}
        self.profile_pdf = bool(refcfg.get("profile_pdf", config.get("profile_pdf", False)))
        self.profile_pdf_every = int(refcfg.get("profile_pdf_every", config.get("profile_pdf_every", 10)))
        self._profile_eval_n = 0

        # Cache Chkl values for Wilkens if lattice/fE/contrast coefficients are unchanged
        self._chkl_cache_key = None
        self._chkl_cache_values = None        
        self._printed_wilkens_diag = False

        self._warren_cache_key = None
        self._warren_cache_value = None

        self._local_trends_cache_key = None
        self._local_trends_cache_value = None


    def _init_composition_constants(self) -> None:
        species = [clean_element_name(str(site.specie)) for site in self.structure_handler.structure.sites]
        self._species_all = np.array(species, dtype=object)
        self._unique_species = sorted(set(species))
        self._species_index = {s: i for i, s in enumerate(self._unique_species)}
        self._counts = np.array([species.count(s) for s in self._unique_species], dtype=np.int64)
        self._n_total = int(self._counts.sum())

        conc = self._counts.astype(float) / max(self._n_total, 1)
        self._concentrations = conc
        Z = np.array([Element(s).Z for s in self._unique_species], dtype=float)
        self._Z = Z
        self._Z_avg = float(np.sum(conc * Z))

        # Unordered Faber–Ziman weights (pair key is sorted species tuple).
        # IMPORTANT:
        #   Faber–Ziman total g(r) uses a double sum over ALL (i,j) pairs:
        #       g(r) = sum_{i,j} w_ij * g_ij(r)
        #   with w_ij = c_i c_j b_i b_j / <b>^2.
        #   If we store only unordered (i<=j) keys, we must include a factor 2
        #   for i != j so that the unordered sum matches the full double sum.
        self._w_unordered = {}
        for i, si in enumerate(self._unique_species):
            for j, sj in enumerate(self._unique_species):
                if j < i:
                    continue
                w = (conc[i] * conc[j] * Z[i] * Z[j]) / (self._Z_avg ** 2)
                if i != j:
                    w *= 2.0
                self._w_unordered[(si, sj)] = float(w)

        # Ordered weights per pair_id (so we can do a single dot-product)
        w_ordered = np.zeros(self._n_pairs, dtype=float)
        for pid in range(self._n_pairs):
            ai = int(self._pair_alpha_id[pid])
            bi = int(self._pair_beta_id[pid])
            sa = self._unique_species[ai]
            sb = self._unique_species[bi]
            key = (sa, sb) if sa <= sb else (sb, sa)
            # We split the unordered weight across the two ordered channels (A-B and B-A)
            # so that summing over ordered pairs reproduces the full double sum.
            w = self._w_unordered[key]
            if ai != bi:
                w *= 0.5
            w_ordered[pid] = float(w)
        self._w_ordered = w_ordered


    def _build_shell_table(self, shells: list[dict]) -> None:
        """Convert cached shell dicts into packed NumPy arrays.

        Everything needed in the hot G(r) loop must be numeric arrays.
        """
        # Species list used to build stable indices
        species = [clean_element_name(str(site.specie)) for site in self.structure_handler.structure.sites]
        unique_species = sorted(set(species))
        sp_index = {s: i for i, s in enumerate(unique_species)}

        # Build a stable ordered-pair list from the shells
        pair_set = set()
        for sh in shells:
            a_sp, b_sp = [lambda_safe_species_label(x) for x in str(sh["pair_type"]).split("-")]
            pair_set.add((a_sp, b_sp))
        pair_list = sorted(pair_set)
        pair_index = {p: i for i, p in enumerate(pair_list)}

        upair_set = set()
        for (a_sp, b_sp) in pair_list:
            aa, bb = sorted((a_sp, b_sp))
            upair_set.add((aa, bb))
        upair_list = sorted(upair_set)
        upair_index = {p: i for i, p in enumerate(upair_list)}

        n = len(shells)
        frac = np.zeros((n, 3), dtype=self._dtype)
        mult = np.zeros(n, dtype=np.int32)
        pair_id = np.zeros(n, dtype=np.int32)
        alpha_id = np.zeros(n, dtype=np.int16)
        beta_id = np.zeros(n, dtype=np.int16)
        upair_id = np.zeros(n, dtype=np.int32)
        k_within_upair = np.zeros(n, dtype=np.int32)
        hkl = np.zeros((n, 3), dtype=np.int16)
        direction = np.zeros((n, 3), dtype=np.int16)
        k_rank = np.arange(n, dtype=np.int32)  # distance rank from cached ordering

        for i, sh in enumerate(shells):
            frac[i, :] = np.asarray(sh["fractional_vector"], dtype=self._dtype)
            mult[i] = int(sh.get("coordination", sh.get("multiplicity", 1)))

            a_sp, b_sp = [clean_element_name(x) for x in str(sh["pair_type"]).split("-")]
            pid = pair_index[(a_sp, b_sp)]
            pair_id[i] = int(pid)
            alpha_id[i] = int(sp_index[a_sp])
            beta_id[i] = int(sp_index[b_sp])

            aa, bb = sorted((a_sp, b_sp))
            upair_id[i] = int(upair_index[(aa, bb)])

            hkli = sh.get("hkl", (0, 0, 0))
            diri = sh.get("direction", hkli)
            hkl[i, :] = np.asarray(hkli, dtype=np.int16)
            direction[i, :] = np.asarray(diri, dtype=np.int16)

        # Rank shells by UNIQUE DISTANCE within each unordered pair.
        #
        # Several internal rows can share the same distance because the shell
        # grouping may include direction.  They must share the same lambda index
        # so lambda_pair_0 applies to the whole first PDF peak.
        dists = np.asarray([float(sh.get("distance", 0.0)) for sh in shells], dtype=np.float64)
        dist_keys = np.rint(dists * 1e5).astype(np.int64)

        dist_order = np.argsort(dists, kind="mergesort")

        last_dist_key = np.full(len(upair_list), np.iinfo(np.int64).min, dtype=np.int64)
        current_rank = np.full(len(upair_list), -1, dtype=np.int32)

        for ii in dist_order:
            uid = int(upair_id[ii])
            dk = int(dist_keys[ii])

            if dk != int(last_dist_key[uid]):
                current_rank[uid] += 1
                last_dist_key[uid] = dk

            k_within_upair[ii] = int(current_rank[uid])

        # Sort all shell arrays by pair_id so the Numba kernel can parallelize by pair safely
        order = np.argsort(pair_id, kind="mergesort")
        self._shell_frac = frac[order]
        self._shell_mult = mult[order]
        self._shell_mult_eff = self._shell_mult.astype(self._dtype, copy=False)
        self._shell_pair_id = pair_id[order]
        self._shell_alpha_id = alpha_id[order]
        self._shell_beta_id = beta_id[order]
        self._shell_hkl = hkl[order]
        self._shell_direction = direction[order]
        self._shell_k_rank = k_rank[order]
        self._shell_upair_id = upair_id[order]
        self._shell_k_within_upair = k_within_upair[order]

        # Pair metadata
        self._pair_list = pair_list
        self._pair_index = pair_index
        self._upair_list = upair_list
        self._upair_index = upair_index
        self._unique_species_for_shells = unique_species
        self._species_index_for_shells = sp_index

        # Build pair offsets into the sorted shell arrays
        n_pairs = len(pair_list)
        offsets = np.zeros(n_pairs + 1, dtype=np.int32)
        # count shells per pair
        counts = np.bincount(self._shell_pair_id, minlength=n_pairs).astype(np.int32)
        offsets[1:] = np.cumsum(counts)
        self._pair_offsets = offsets

        # Pair alpha/beta ids for ordered weights
        self._pair_alpha_id = np.array([sp_index[p[0]] for p in pair_list], dtype=np.int16)
        self._pair_beta_id = np.array([sp_index[p[1]] for p in pair_list], dtype=np.int16)
        self._n_pairs = int(n_pairs)

        # This path builds from legacy shells and has no cached strain direction arrays.
        self._loaded_strain_unique_dirs = None
        self._loaded_strain_inv_dirs = None

        self._prepare_shell_runtime_arrays()

    def _load_shell_table_arrays(self, shell_table: dict) -> None:
        """Load a pre-packed shell table (arrays) from StructureHandler.

        This bypasses Python dict shells entirely, which is important when
        r_max is large (e.g. 100 Å) and the shell list is huge.
        """
        # Validate expected keys
        required = (
            "shell_frac",
            "shell_mult",
            "shell_pair_id",
            "shell_alpha_id",
            "shell_beta_id",
            "shell_hkl",
            "shell_direction",
            "shell_k_rank",
            "shell_upair_id",
            "shell_k_within_upair",
            "pair_offsets",
            "pair_alpha_id",
            "pair_beta_id",
            "unique_species",
            "pair_list",
            "upair_list",
        )
        for k in required:
            if k not in shell_table:
                raise KeyError(f"shell_table missing key: {k}")

        # Enforce dtype choices
        self._shell_frac = np.asarray(shell_table["shell_frac"], dtype=self._dtype)
        self._shell_mult = np.asarray(shell_table["shell_mult"], dtype=np.int32)

        self._shell_bulk_mult = np.asarray(
            shell_table.get("shell_bulk_mult", shell_table["shell_mult"]),
            dtype=np.int32,
        )

        self._shell_mult_eff = np.asarray(
            shell_table.get("shell_mult_eff", shell_table["shell_mult"]),
            dtype=self._dtype,
        )

        self._shell_finite_mult_eff = np.asarray(
            shell_table.get("shell_finite_mult_eff", self._shell_mult_eff),
            dtype=self._dtype,
        )

        self._shell_bulk_mult_eff = np.asarray(
            shell_table.get("shell_bulk_mult_eff", self._shell_mult_eff),
            dtype=self._dtype,
        )

        self._shell_gamma = np.asarray(
            shell_table.get("shell_gamma", np.ones_like(self._shell_mult_eff)),
            dtype=self._dtype,
        )
        self._shell_pair_id = np.asarray(shell_table["shell_pair_id"], dtype=np.int32)
        self._shell_alpha_id = np.asarray(shell_table["shell_alpha_id"], dtype=np.int16)
        self._shell_beta_id = np.asarray(shell_table["shell_beta_id"], dtype=np.int16)
        self._shell_hkl = np.asarray(shell_table["shell_hkl"], dtype=np.int16)
        self._shell_direction = np.asarray(shell_table["shell_direction"], dtype=np.int16)
        self._shell_k_rank = np.asarray(shell_table["shell_k_rank"], dtype=np.int32)
        self._shell_upair_id = np.asarray(shell_table["shell_upair_id"], dtype=np.int32)
        self._shell_k_within_upair = np.asarray(shell_table["shell_k_within_upair"], dtype=np.int32)
        self._pair_offsets = np.asarray(shell_table["pair_offsets"], dtype=np.int32)

        # Pair metadata
        pair_list = [tuple(p) for p in shell_table["pair_list"]]
        self._pair_list = pair_list
        self._pair_index = {p: i for i, p in enumerate(pair_list)}

        upair_list = [tuple(p) for p in shell_table.get("upair_list", [])]
        self._upair_list = upair_list
        self._upair_index = {p: i for i, p in enumerate(upair_list)}

        self._unique_species_for_shells = list(shell_table["unique_species"])
        self._species_index_for_shells = {s: i for i, s in enumerate(self._unique_species_for_shells)}

        self._pair_alpha_id = np.asarray(shell_table["pair_alpha_id"], dtype=np.int16)
        self._pair_beta_id = np.asarray(shell_table["pair_beta_id"], dtype=np.int16)
        self._n_pairs = int(len(pair_list))
        
        # Cached strain direction arrays, if present in the shell cache.
        self._loaded_strain_unique_dirs = shell_table.get("strain_unique_dirs", None)
        self._loaded_strain_inv_dirs = shell_table.get("strain_inv_dirs", None)
        
        self._prepare_shell_runtime_arrays()


    def _finite_gamma_by_pair_on_grid(
        self,
        r_grid: np.ndarray,
        r_ij: np.ndarray,
    ) -> np.ndarray:
        """
        Build discrete gamma_finite(r) for each ordered pair channel.

        Option A:
            - no smoothing
            - no interpolation
            - each shell contributes to the nearest r-grid point
            - empty r-grid points remain gamma = 0

        For each shell:

            shell_gamma = finite_mult_eff / bulk_mult_eff

        To average multiple shells falling into the same r-bin, use the bulk
        effective coordination as the weight:

            gamma_bin =
                sum(shell_gamma * bulk_mult_eff) / sum(bulk_mult_eff)

        Since:
            shell_gamma * bulk_mult_eff = finite_mult_eff

        this is equivalent to:
            gamma_bin = sum(finite_mult_eff) / sum(bulk_mult_eff)
        """
        r_grid = np.asarray(r_grid, dtype=float)
        r_ij = np.asarray(r_ij, dtype=float)

        n_pair = int(self._n_pairs)
        n_r = int(r_grid.size)

        gamma_by_pair = np.zeros((n_pair, n_r), dtype=self._dtype)

        if n_pair <= 0 or n_r <= 0 or r_ij.size == 0:
            return gamma_by_pair

        shell_gamma = getattr(self, "_shell_gamma", None)
        shell_bulk_mult_eff = getattr(self, "_shell_bulk_mult_eff", None)

        if shell_gamma is None or shell_bulk_mult_eff is None:
            return gamma_by_pair

        shell_gamma = np.asarray(shell_gamma, dtype=float)
        shell_bulk_mult_eff = np.asarray(shell_bulk_mult_eff, dtype=float)
        shell_pair_id = np.asarray(self._shell_pair_id, dtype=np.int64)

        valid = (
            np.isfinite(r_ij)
            & np.isfinite(shell_gamma)
            & np.isfinite(shell_bulk_mult_eff)
            & (shell_bulk_mult_eff > 0.0)
            & (r_ij >= float(r_grid[0]))
            & (r_ij <= float(r_grid[-1]))
            & (shell_pair_id >= 0)
            & (shell_pair_id < n_pair)
        )

        if not np.any(valid):
            return gamma_by_pair

        rv = r_ij[valid]
        gv = shell_gamma[valid]
        bv = shell_bulk_mult_eff[valid]
        pv = shell_pair_id[valid]

        idx_right = np.searchsorted(r_grid, rv, side="left")
        idx_right = np.clip(idx_right, 0, n_r - 1)

        idx_left = np.clip(idx_right - 1, 0, n_r - 1)

        dist_left = np.abs(rv - r_grid[idx_left])
        dist_right = np.abs(rv - r_grid[idx_right])

        idx = np.where(dist_left <= dist_right, idx_left, idx_right).astype(np.int64)

        flat = pv * n_r + idx

        numerator = np.bincount(
            flat,
            weights=gv * bv,
            minlength=n_pair * n_r,
        )

        denominator = np.bincount(
            flat,
            weights=bv,
            minlength=n_pair * n_r,
        )

        numerator = numerator.reshape(n_pair, n_r)
        denominator = denominator.reshape(n_pair, n_r)

        mask = denominator > 0.0

        gamma = np.zeros_like(numerator, dtype=float)
        gamma[mask] = numerator[mask] / denominator[mask]

        gamma_by_pair[:, :] = gamma.astype(self._dtype, copy=False)

        return gamma_by_pair

  
    def _cylinder_common_volume_shell_factor(
        self,
        params: dict,
        lattice_tuple,
        r_ij: np.ndarray,
        cart_vectors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        HKL/shell-direction-dependent cylinder/disk common-volume factor using
        the Malagutti/Scardi lognormal thickness-distribution expression.

        Implements:

            A_TD(L,D) =
                exp(-mu - sigma^2/2)/(D*pi)
                *
                [
                    exp(mu + sigma^2/2)
                    *
                    (1 + erf((mu + sigma^2 - ln(L cos(phi)))/(sqrt(2)*sigma)))
                    +
                    L cos(phi)
                    *
                    (erfc((mu - ln(L cos(phi)))/(sqrt(2)*sigma)) - 2)
                ]
                *
                [
                    D acos(L sin(phi)/D)
                    -
                    L sin(phi) sqrt(1 - (L sin(phi)/D)^2)
                ]

        Parameters:
            cyl_diameter       = D in Angstrom
            cyl_thickness      = real-space mean thickness in Angstrom
            cyl_thickness_std  = real-space std thickness in Angstrom

        Optional direct lognormal parameters:
            cyl_mu
            cyl_sigma

        Optional cylinder axis:
            cyl_axis_h
            cyl_axis_k
            cyl_axis_l

        Default axis is [0 0 1].

        Notes
        -----
        This function returns a normalized shell-dependent gamma factor.
        It tends to 1 at L -> 0.
        """
        params = params or {}

        D = float(
            params.get(
                "cyl_diameter",
                params.get("cylinder_diameter", params.get("disk_diameter", 0.0)),
            )
        )

        if D <= 0.0:
            return np.ones_like(r_ij, dtype=self._dtype)

        # ------------------------------------------------------------------
        # Thickness lognormal parameters.
        #
        # User-friendly GUI mode:
        #   cyl_thickness     = real-space mean thickness
        #   cyl_thickness_std = real-space standard deviation
        #
        # Direct TOPAS-like mode:
        #   cyl_mu
        #   cyl_sigma
        # ------------------------------------------------------------------
        has_direct_mu_sigma = (
            "cyl_mu" in params
            or "cylinder_mu" in params
            or "disk_mu" in params
        ) and (
            "cyl_sigma" in params
            or "cylinder_sigma" in params
            or "disk_sigma" in params
        )

        if has_direct_mu_sigma:
            mu = float(
                params.get(
                    "cyl_mu",
                    params.get("cylinder_mu", params.get("disk_mu", 0.0)),
                )
            )
            sigma = float(
                params.get(
                    "cyl_sigma",
                    params.get("cylinder_sigma", params.get("disk_sigma", 0.0)),
                )
            )

            # Monodisperse fallback approximate mean.
            t_mean = float(np.exp(mu + 0.5 * sigma * sigma)) if sigma > 0.0 else float(np.exp(mu))

        else:
            t_mean = float(
                params.get(
                    "cyl_thickness",
                    params.get("cylinder_thickness", params.get("disk_thickness", 0.0)),
                )
            )

            t_std = float(
                params.get(
                    "cyl_thickness_std",
                    params.get(
                        "cylinder_thickness_std",
                        params.get("disk_thickness_std", 0.0),
                    ),
                )
            )

            if t_mean <= 0.0:
                return np.ones_like(r_ij, dtype=self._dtype)

            if t_std > 1e-12:
                sigma2_tmp = np.log(1.0 + (t_std * t_std) / (t_mean * t_mean))
                sigma = float(np.sqrt(sigma2_tmp))
                mu = float(np.log(t_mean) - 0.5 * sigma2_tmp)
            else:
                sigma = 0.0
                mu = float(np.log(t_mean))

        # ------------------------------------------------------------------
        # Cylinder axis.
        # ------------------------------------------------------------------
        ah = int(round(float(params.get("cyl_axis_h", 0))))
        ak = int(round(float(params.get("cyl_axis_k", 0))))
        al = int(round(float(params.get("cyl_axis_l", 1))))

        if ah == 0 and ak == 0 and al == 0:
            ah, ak, al = 0, 0, 1

        M = lattice_matrix(*lattice_tuple).astype(float, copy=False)

        # Reuse Cartesian shell vectors already computed in evaluate().
        # This avoids one large matrix multiplication per PDF evaluation.
        if cart_vectors is None:
            cart = np.asarray(self._shell_frac, dtype=float) @ M.T
        else:
            cart = np.asarray(cart_vectors, dtype=float)

        L = np.asarray(r_ij, dtype=float)

        axis_frac = np.asarray([ah, ak, al], dtype=float)
        axis_cart = axis_frac @ M.T

        n_axis = float(np.linalg.norm(axis_cart))
        if n_axis <= 1e-14:
            axis_cart = np.asarray([0.0, 0.0, 1.0], dtype=float)
            n_axis = 1.0

        c_hat = axis_cart / n_axis

        # L cos(phi)
        L_cos_phi = np.abs(cart @ c_hat)

        # L sin(phi)
        L_sin_phi2 = np.maximum(L * L - L_cos_phi * L_cos_phi, 0.0)
        L_sin_phi = np.sqrt(L_sin_phi2)

        gamma = np.zeros_like(L, dtype=float)

        valid = (
            np.isfinite(L)
            & np.isfinite(L_cos_phi)
            & np.isfinite(L_sin_phi)
            & (L > 0.0)
            & (L_sin_phi < D)
        )

        if not np.any(valid):
            return np.asarray(gamma, dtype=self._dtype)

        z = L_cos_phi[valid]
        rp = L_sin_phi[valid]

        # ------------------------------------------------------------------
        # Radial bracket:
        #
        #   D acos(rp/D) - rp sqrt(1 - (rp/D)^2)
        # ------------------------------------------------------------------
        x = np.clip(rp / D, 0.0, 1.0)

        radial_bracket = (
            D * np.arccos(x)
            - rp * np.sqrt(np.maximum(1.0 - x * x, 0.0))
        )

        radial_bracket = np.maximum(radial_bracket, 0.0)

        # ------------------------------------------------------------------
        # Monodisperse limit.
        #
        # Use normalized cylinder common volume:
        #
        #   gamma =
        #       [2/pi * (acos(x) - x sqrt(1-x^2))]
        #       *
        #       max(1 - z/t, 0)
        #
        # This tends to 1 at L = 0.
        # ------------------------------------------------------------------
        if sigma <= 1e-12:
            if t_mean <= 0.0:
                return np.ones_like(r_ij, dtype=self._dtype)

            axial = 1.0 - z / t_mean
            axial[z >= t_mean] = 0.0
            axial[~np.isfinite(axial)] = 0.0
            axial = np.clip(axial, 0.0, 1.0)

            radial_full = (2.0 / np.pi) * (
                np.arccos(x)
                - x * np.sqrt(np.maximum(1.0 - x * x, 0.0))
            )

            gamma_valid = radial_full * axial

            gamma[valid] = gamma_valid
            gamma[~np.isfinite(gamma)] = 0.0

            return np.asarray(np.clip(gamma, 0.0, 1.0), dtype=self._dtype)

        # ------------------------------------------------------------------
        # Lognormal thickness-distribution expression from Malagutti/Scardi.
        # ------------------------------------------------------------------
        sigma2 = sigma * sigma
        sqrt2sigma = np.sqrt(2.0) * sigma

        thickness_bracket = np.zeros_like(z, dtype=float)

        # z = L cos(phi)
        z_zero = z <= 1e-300
        z_pos = ~z_zero

        # Limit z -> 0:
        #   bracket = 2 exp(mu + sigma^2/2)
        thickness_bracket[z_zero] = 2.0 * np.exp(mu + 0.5 * sigma2)

        if np.any(z_pos):
            zp = z[z_pos]
            lnz = np.log(np.maximum(zp, 1e-300))

            arg_erf = (mu + sigma2 - lnz) / sqrt2sigma
            arg_erfc = (mu - lnz) / sqrt2sigma

            thickness_bracket[z_pos] = (
                np.exp(mu + 0.5 * sigma2)
                * (1.0 + erf(arg_erf))
                +
                zp
                * (erfc(arg_erfc) - 2.0)
            )

        prefactor = np.exp(-mu - 0.5 * sigma2) / (D * np.pi)

        gamma_valid = prefactor * thickness_bracket * radial_bracket

        gamma[valid] = gamma_valid
        gamma[~np.isfinite(gamma)] = 0.0

        return np.asarray(np.clip(gamma, 0.0, 1.0), dtype=self._dtype)

    def _shell_gamma_by_pair_on_grid(
        self,
        r_grid: np.ndarray,
        r_ij: np.ndarray,
        shell_gamma: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fast cylinder/disk gamma baseline.

        Old version:
            Python loop over pair -> direction -> unique distance.
            This is very slow for large shell tables.

        New version:
            1. parallel Numba histogram per ordered pair;
            2. average gamma values in occupied r bins;
            3. interpolate missing r bins per pair.

        This keeps the baseline smooth and avoids leaving most grid points at 0.
        """
        r_grid = np.asarray(r_grid, dtype=float)
        r_ij = np.asarray(r_ij, dtype=float)
        shell_gamma = np.asarray(shell_gamma, dtype=float)

        n_pair = int(self._n_pairs)
        n_r = int(r_grid.size)

        out = np.zeros((n_pair, n_r), dtype=self._dtype)

        if n_pair <= 0 or n_r <= 0 or r_ij.size == 0:
            return out

        if weights is None:
            weights_arr = np.ones_like(shell_gamma, dtype=float)
        else:
            weights_arr = np.asarray(weights, dtype=float)

        numerator = np.zeros((n_pair, n_r), dtype=np.float64)
        denominator = np.zeros((n_pair, n_r), dtype=np.float64)

        accumulate_pair_gamma_histogram_by_pair(
            np.asarray(r_grid, dtype=np.float64),
            np.asarray(r_ij, dtype=np.float64),
            np.asarray(shell_gamma, dtype=np.float64),
            np.asarray(weights_arr, dtype=np.float64),
            np.asarray(self._pair_offsets, dtype=np.int32),
            numerator,
            denominator,
        )

        for pid in range(n_pair):
            mask = denominator[pid] > 0.0

            if not np.any(mask):
                continue

            r_known = r_grid[mask]
            g_known = numerator[pid, mask] / denominator[pid, mask]

            finite = np.isfinite(r_known) & np.isfinite(g_known)

            if not np.any(finite):
                continue

            r_known = r_known[finite]
            g_known = g_known[finite]

            order = np.argsort(r_known)
            r_known = r_known[order]
            g_known = g_known[order]

            if r_known.size == 0:
                continue

            # Normalized common volume must be 1 at r = 0.
            if r_known[0] > 0.0:
                r_known = np.concatenate(([0.0], r_known))
                g_known = np.concatenate(([1.0], g_known))

            right_value = float(g_known[-1])

            if right_value < 1.0e-8:
                right_value = 0.0

            out[pid, :] = np.interp(
                r_grid,
                r_known,
                g_known,
                left=1.0,
                right=right_value,
            ).astype(self._dtype, copy=False)

        return out   

    def _cfg_bool_local(self, val, default=False) -> bool:
        """
        Local robust bool parser for model settings.
        """
        if val is None:
            return bool(default)

        if isinstance(val, bool):
            return val

        if isinstance(val, (int, float)):
            return bool(int(val))

        if isinstance(val, str):
            s = val.strip().lower()

            if s in ("1", "true", "yes", "y", "on"):
                return True

            if s in ("0", "false", "no", "n", "off"):
                return False

        return bool(default)


    def _saxs_refcfg(self) -> dict:
        """
        Return the typed [refinement] configuration dictionary.

        SAXS enable/model switches are stored in [refinement], rather than
        [initial_values], because some of them are strings or booleans and
        are not numeric refinement parameters.
        """
        if not isinstance(self.config, dict):
            return {}

        refcfg = self.config.get(
            "refinement",
            {},
        )

        if not isinstance(refcfg, dict):
            return {}

        return refcfg


    def _saxs_enabled(self, params: dict) -> bool:
        """
        Return True only when the independent SAXS/missing-low-Q correction
        is explicitly enabled.

        The normal negative PDF baseline remains active regardless of this
        switch.
        """
        params = params or {}
        refcfg = self._saxs_refcfg()

        return self._cfg_bool_local(
            params.get(
                "saxs_enabled",
                refcfg.get(
                    "saxs_enabled",
                    False,
                ),
            ),
            False,
        )
    
    def _saxs_model_name(self, params: dict) -> str:
        """
        SAXS belly model name.

        This is intentionally read from [refinement], not [initial_values],
        because saxs_model is a string and FitEngine expects initial values
        to be numeric.
        """
        params = params or {}
        refcfg = self._saxs_refcfg()

        model = str(
            params.get(
                "saxs_model",
                refcfg.get("saxs_model", "sphere"),
            )
        ).strip().lower()

        model = model.replace(" ", "_")

        if model in ("spherical",):
            return "sphere"

        if model in ("cyl",):
            return "cylinder"

        if model in ("disc",):
            return "disk"

        if model in ("same", "same_as_size", "same_as_size_model", "auto"):
            return "same"

        return model


    def _saxs_scale(self, params: dict) -> float:
        """
        Independent amplitude of the SAXS / missing-low-Q belly term.
        """
        params = params or {}

        try:
            return float(
                params.get(
                    "saxs_scale",
                    params.get("shape_belly_scale", 1.0),
                )
            )
        except Exception:
            return 1.0


    def _saxs_apply_qdamp(self, params: dict) -> bool:
        """
        Whether qdamp should be applied to the plotted/model SAXS belly.

        Recommended default: False.
        """
        params = params or {}
        refcfg = self._saxs_refcfg()

        return self._cfg_bool_local(
            params.get(
                "saxs_apply_qdamp",
                refcfg.get("saxs_apply_qdamp", False),
            ),
            False,
        )


    def _saxs_apply_qmax(self, params: dict) -> bool:
        """
        Whether qmax termination should be applied to the SAXS belly.

        Recommended default: True.
        """
        params = params or {}
        refcfg = self._saxs_refcfg()

        return self._cfg_bool_local(
            params.get(
                "saxs_apply_qmax",
                refcfg.get("saxs_apply_qmax", True),
            ),
            True,
        )


    def _has_explicit_saxs_size(self, params: dict) -> bool:
        """
        True when the user provided independent SAXS size information.

        If all SAXS size parameters are 0, the model falls back to the main
        size/shape model for backward compatibility.
        """
        params = params or {}

        for key in (
            "saxs_diameter",
            "saxs_diameter_std",
            "saxs_height",
            "saxs_height_std",
        ):
            try:
                if abs(float(params.get(key, 0.0))) > 1e-12:
                    return True
            except Exception:
                pass

        return False


    def _lognormal_average_cylinder_shell_gamma(
        self,
        params_for_cylinder: dict,
        lattice_tuple,
        r_ij: np.ndarray,
        cart_vectors: np.ndarray,
        *,
        diameter_mean: float,
        diameter_std: float,
    ) -> np.ndarray:
        """
        Average cylinder/disk shell gamma over a lognormal diameter distribution.

        Existing _cylinder_common_volume_shell_factor already supports a
        lognormal height/thickness distribution through cyl_thickness_std.
        This helper adds an approximate diameter distribution by numerical
        quadrature over diameter.
        """
        diameter_mean = float(diameter_mean)
        diameter_std = float(diameter_std)

        if diameter_mean <= 0.0:
            return np.ones_like(r_ij, dtype=self._dtype)

        if diameter_std <= 1e-12:
            p = dict(params_for_cylinder)
            p["cyl_diameter"] = diameter_mean

            return self._cylinder_common_volume_shell_factor(
                p,
                lattice_tuple,
                r_ij,
                cart_vectors=cart_vectors,
            )

        sigma2 = np.log(1.0 + (diameter_std * diameter_std) / (diameter_mean * diameter_mean))
        sigma = float(np.sqrt(sigma2))
        mu = float(np.log(diameter_mean) - 0.5 * sigma2)

        d_min = max(1e-6, diameter_mean * 0.05)
        d_max = diameter_mean + 4.0 * diameter_std

        if d_max <= d_min:
            d_max = diameter_mean * 3.0

        d_grid = np.linspace(d_min, d_max, 15)

        pdf = (
            1.0 / (d_grid * sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((np.log(d_grid) - mu) / sigma) ** 2)
        )

        weights = pdf / max(float(np.sum(pdf)), 1e-300)

        out = np.zeros_like(r_ij, dtype=float)

        for D, w in zip(d_grid, weights):
            p = dict(params_for_cylinder)
            p["cyl_diameter"] = float(D)

            g = self._cylinder_common_volume_shell_factor(
                p,
                lattice_tuple,
                r_ij,
                cart_vectors=cart_vectors,
            )

            out += float(w) * np.asarray(g, dtype=float)

        return np.asarray(out, dtype=self._dtype)


    def _compute_saxs_gamma_weighted(
        self,
        params: dict,
        r_grid: np.ndarray,
        r_ij: np.ndarray,
        lattice_tuple,
        cart_vectors: np.ndarray,
        *,
        fallback_gamma: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the single powder-averaged gamma(r) used for the smooth
        SAXS / missing-low-Q belly term.

        If the user did not provide independent SAXS size parameters, this
        returns fallback_gamma, preserving old behavior.

        Supported models:
            sphere
            cylinder
            disk
            same
        """
        params = params or {}

        r_grid = np.asarray(r_grid, dtype=float)
        fallback_gamma = np.asarray(fallback_gamma, dtype=float)

        model = self._saxs_model_name(params)

        if model == "same":
            return fallback_gamma.astype(self._dtype, copy=False)

        if not self._has_explicit_saxs_size(params):
            return fallback_gamma.astype(self._dtype, copy=False)

        if model == "sphere":
            D = float(
                params.get(
                    "saxs_diameter",
                    params.get("d", params.get("D", 0.0)),
                )
            )

            D_std = float(
                params.get(
                    "saxs_diameter_std",
                    params.get("d_std", 0.0),
                )
            )

            if D <= 0.0:
                return fallback_gamma.astype(self._dtype, copy=False)

            return self.shape_function(
                r_grid,
                D,
                D_std,
            ).astype(self._dtype, copy=False)

        if model in ("cylinder", "disk"):
            D = float(
                params.get(
                    "saxs_diameter",
                    params.get("cyl_diameter", 0.0),
                )
            )

            D_std = float(params.get("saxs_diameter_std", 0.0))

            H = float(
                params.get(
                    "saxs_height",
                    params.get("cyl_thickness", 0.0),
                )
            )

            H_std = float(
                params.get(
                    "saxs_height_std",
                    params.get("cyl_thickness_std", 0.0),
                )
            )

            if D <= 0.0 or H <= 0.0:
                return fallback_gamma.astype(self._dtype, copy=False)

            p_cyl = dict(params)
            p_cyl["cyl_diameter"] = D
            p_cyl["cyl_thickness"] = H
            p_cyl["cyl_thickness_std"] = H_std

            shell_gamma = self._lognormal_average_cylinder_shell_gamma(
                p_cyl,
                lattice_tuple,
                r_ij,
                cart_vectors,
                diameter_mean=D,
                diameter_std=D_std,
            )

            gamma_weights = getattr(self, "_shell_bulk_mult_eff", None)

            if gamma_weights is None:
                gamma_weights = getattr(self, "_shell_mult_eff", None)

            if gamma_weights is None:
                gamma_weights = self._shell_mult.astype(self._dtype, copy=False)

            gamma_by_pair = self._shell_gamma_by_pair_on_grid(
                r_grid.astype(self._dtype),
                r_ij,
                shell_gamma,
                weights=np.asarray(gamma_weights, dtype=float),
            )

            gamma_weighted = (
                self._w_ordered.astype(self._dtype)[:, None]
                * gamma_by_pair
            ).sum(axis=0).astype(self._dtype, copy=False)

            return gamma_weighted

        return fallback_gamma.astype(self._dtype, copy=False)


    def _postprocess_shape_term_for_plot(self, y, params, r_grid):
        """
        Apply optional qdamp and qmax processing to the plotted SAXS belly term.

        The input y is already:

            -scale * saxs_scale * 4*pi*rho0*r*gamma_saxs(r)

        qdamp is optional and defaults to false for the SAXS belly.
        qmax termination defaults to true.
        """
        y = np.asarray(y, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)

        params = params or {}

        qdamp = float(params.get("qdamp", 0.0))
        qmax = float(params.get("qmax", self.config.get("qmax", 0.0)))
        qmax_zeros = int(params.get("qmax_zeros", self.config.get("qmax_zeros", 5)))
        qmax_pad = str(
            params.get("qmax_pad", self.config.get("qmax_pad", "zero"))
        ).strip().lower()

        if self._saxs_apply_qdamp(params) and qdamp > 0.0:
            y = y * np.exp(-0.5 * (qdamp ** 2) * (r_grid ** 2))

        if self._saxs_apply_qmax(params) and qmax > 0.0 and r_grid.size > 1:
            y = apply_qmax_termination(
                y,
                r_grid,
                qmax,
                n_zeros=qmax_zeros,
                pad_mode=qmax_pad,
                use_sine_kernel=True,
                normalize=False,
            )

        return np.asarray(y, dtype=float)

    
    def compute_isotropic_shape_factor(self, params, r_grid):
        """
        Compute the actual plotted SAXS / missing-low-Q belly term.

        This returns a G(r)-scale curve:

            -scale * saxs_scale * 4*pi*rho0*r*gamma_saxs(r)

        It is not dimensionless gamma(r).

        The SAXS gamma may have its own independent size/distribution through:

            saxs_diameter
            saxs_diameter_std
            saxs_height
            saxs_height_std
            saxs_model

        If no independent SAXS size is provided, this falls back to the main
        shape model for backward compatibility.
        """
        params = params or {}
        r_grid = np.asarray(r_grid, dtype=float)

        if r_grid.size == 0:
            return np.asarray([], dtype=float)

        # ---------------------------------------------------------
        # Current constrained lattice
        # ---------------------------------------------------------
        a = float(params.get("a", self.structure_handler.structure.lattice.a))
        b = float(params.get("b", a))
        c = float(params.get("c", a))
        alpha = float(params.get("alpha", 90.0))
        beta = float(params.get("beta", 90.0))
        gamma = float(params.get("gamma", 90.0))

        lattice_dict = {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }

        lattice_dict = apply_lattice_constraints(
            lattice_dict,
            self.constraints,
        )

        lattice_tuple = (
            float(lattice_dict["a"]),
            float(lattice_dict["b"]),
            float(lattice_dict["c"]),
            float(lattice_dict["alpha"]),
            float(lattice_dict["beta"]),
            float(lattice_dict["gamma"]),
        )

        # ---------------------------------------------------------
        # rho0 and norm = 4*pi*rho0
        # ---------------------------------------------------------
        from pymatgen.core import Lattice

        num_atoms = len(self.structure_handler.structure.sites)
        vol = float(Lattice.from_parameters(*lattice_tuple).volume)

        if not np.isfinite(vol) or vol <= 1.0e-12:
            return np.zeros_like(r_grid, dtype=float)

        rho0 = float(num_atoms) / vol

        norm = 4.0 * np.pi * rho0

        scale = float(params.get("scale", 1.0))
        saxs_scale = float(self._saxs_scale(params))

        # ---------------------------------------------------------
        # Shell distances
        # ---------------------------------------------------------
        M = lattice_matrix(*lattice_tuple).astype(self._dtype, copy=False)
        cart = self._shell_frac @ M.T
        r_ij = np.sqrt(np.sum(cart * cart, axis=1)).astype(self._dtype, copy=False)

        r32 = np.asarray(r_grid, dtype=self._dtype)

        # ---------------------------------------------------------
        # Determine fallback gamma from current active size model
        # ---------------------------------------------------------
        shape_cfg = self.config.get("crystallite_shape", {}) or {}
        shape_mode = str(shape_cfg.get("mode", "conventional")).strip().lower()

        cyl_diameter = float(
            params.get(
                "cyl_diameter",
                params.get("cylinder_diameter", params.get("disk_diameter", 0.0)),
            )
        )

        cyl_thickness = float(
            params.get(
                "cyl_thickness",
                params.get("cylinder_thickness", params.get("disk_thickness", 0.0)),
            )
        )

        cylinder_common_volume_active = (
            cyl_diameter > 0.0
            and cyl_thickness > 0.0
            and shape_mode != "finite_shape"
        )

        if shape_mode == "finite_shape":
            gamma_by_pair = self._finite_gamma_by_pair_on_grid(
                r32,
                r_ij,
            )

            fallback_gamma = (
                self._w_ordered.astype(self._dtype)[:, None]
                * gamma_by_pair
            ).sum(axis=0).astype(self._dtype, copy=False)

        elif cylinder_common_volume_active:
            cylinder_shell_gamma = self._cylinder_common_volume_shell_factor(
                params,
                lattice_tuple,
                r_ij,
                cart_vectors=cart,
            )

            gamma_weights = getattr(self, "_shell_bulk_mult_eff", None)

            if gamma_weights is None:
                gamma_weights = getattr(self, "_shell_mult_eff", None)

            if gamma_weights is None:
                gamma_weights = self._shell_mult.astype(self._dtype, copy=False)

            gamma_by_pair = self._shell_gamma_by_pair_on_grid(
                r32,
                r_ij,
                cylinder_shell_gamma,
                weights=np.asarray(gamma_weights, dtype=float),
            )

            fallback_gamma = (
                self._w_ordered.astype(self._dtype)[:, None]
                * gamma_by_pair
            ).sum(axis=0).astype(self._dtype, copy=False)

        else:
            d = float(params.get("d", params.get("D", 0.0)))
            d_std = float(params.get("d_std", 0.0))

            fallback_gamma = self.shape_function(
                r_grid,
                d,
                d_std,
            ).astype(self._dtype, copy=False)

        # ---------------------------------------------------------
        # Select the plotted shape term
        #
        # SAXS enabled:
        #     plot the original independent SAXS belly.
        #
        # SAXS disabled:
        #     plot the mandatory conventional PDF baseline.
        # ---------------------------------------------------------
        fallback_gamma = np.asarray(
            fallback_gamma,
            dtype=float,
        )

        saxs_is_enabled = self._saxs_enabled(
            params
        )

        if saxs_is_enabled:
            gamma_saxs = self._compute_saxs_gamma_weighted(
                params,
                r32,
                r_ij,
                lattice_tuple,
                cart,
                fallback_gamma=fallback_gamma,
            )

            gamma_saxs = np.asarray(
                gamma_saxs,
                dtype=float,
            )

            G_shape = (
                -scale
                * saxs_scale
                * norm
                * r_grid
                * gamma_saxs
            )

            # In original SAXS mode, qdamp is optional for the belly.
            qdamp = float(
                params.get(
                    "qdamp",
                    0.0,
                )
            )

            if (
                self._saxs_apply_qdamp(params)
                and qdamp > 0.0
            ):
                G_shape *= np.exp(
                    -0.5
                    * qdamp ** 2
                    * r_grid ** 2
                )

        else:
            # Corrected mandatory conventional PDF baseline.
            G_shape = (
                -scale
                * norm
                * r_grid
                * fallback_gamma
            )

            # In conventional mode, this baseline is part of the
            # structural PDF and therefore receives qdamp.
            qdamp = float(
                params.get(
                    "qdamp",
                    0.0,
                )
            )

            if qdamp > 0.0:
                G_shape *= np.exp(
                    -0.5
                    * qdamp ** 2
                    * r_grid ** 2
                )

        qmax = float(
            params.get(
                "qmax",
                self.config.get(
                    "qmax",
                    0.0,
                ),
            )
        )

        qmax_zeros = int(
            params.get(
                "qmax_zeros",
                self.config.get(
                    "qmax_zeros",
                    5,
                ),
            )
        )

        qmax_pad = str(
            params.get(
                "qmax_pad",
                self.config.get(
                    "qmax_pad",
                    "zero",
                ),
            )
        ).strip().lower()

        apply_qmax_to_shape = True

        if saxs_is_enabled:
            apply_qmax_to_shape = self._saxs_apply_qmax(
                params
            )

        if (
            apply_qmax_to_shape
            and qmax > 0.0
            and r_grid.size > 1
        ):
            G_shape = apply_qmax_termination(
                G_shape,
                r_grid,
                qmax,
                n_zeros=qmax_zeros,
                pad_mode=qmax_pad,
                use_sine_kernel=True,
                normalize=False,
            )

        return np.asarray(
            G_shape,
            dtype=float,
        )

    def replace_shell_table(self, shell_table: dict) -> None:
        """
        Replace the current shell table with an externally generated shell table.

        Used by finite crystallite-shape grid search.

        Important:
        - shell_table must be compatible with _load_shell_table_arrays()
        - composition constants and direction caches are rebuilt
        """
        self._load_shell_table_arrays(shell_table)

        # Recompute Faber-Ziman weights and pair metadata because pair table may differ.
        self._init_composition_constants()

        # Recompute contrast-factor matrix because shell directions may differ.
        self._prepare_contrast_factor_matrix()

        # Clear plot/helper caches that depend on shell arrays.
        self._warren_cache_key = None
        self._warren_cache_value = None

        self._local_trends_cache_key = None
        self._local_trends_cache_value = None


    def shape_function(self, r, d, dstd):
        """
        Same July implementation.
        gamma(r)=1 - 1.5*(r/d)+0.5*(r/d)^3 for monodisperse spheres,
        and lognormal-smoothed version if dstd>0.
        """
        r = np.asarray(r, dtype=float)
        d = float(d)
        dstd = float(dstd)

        if d <= 0.0:
            return np.ones_like(r)

        if dstd < 1e-8:
            gamma = np.zeros_like(r)
            mask = r < d
            ratio = r[mask] / d
            gamma[mask] = 1 - 1.5 * ratio + 0.5 * ratio**3
            return gamma

        ss2 = np.log((dstd**2) / (d**2) + 1.0)
        ss = np.sqrt(ss2)
        mu = np.log(d) - 0.5 * ss2
        lnr = np.log(np.maximum(r, 1e-8))

        A = erfc((-mu - 3 * ss2 + lnr) / (np.sqrt(2) * ss))
        B = erfc((-mu + lnr) / (np.sqrt(2) * ss))
        C = erfc((-mu - 2 * ss2 + lnr) / (np.sqrt(2) * ss))

        expA = np.exp(-3 * mu - 4.5 * ss2)
        expB = np.exp(-mu - 2.5 * ss2)

        return 0.5 * A + 0.25 * r**3 * B * expA - 0.75 * r * C * expB

    def _shell_gr(self, shell, r, rho0, delta1, delta2, delta_g, delta_broad,
                  biso_avg, lambda_vals, k, N_lambda, rho, Re, fE, lattice_tuple,
                  densities, wilkens_active, params=None):
        """
        Compute g_ab(r) contribution from one shell (Gaussian broadening), properly normalized.
        This is the July normalization that prevents RDF-like amplitude blow-up.
        """
        frac_vector = np.array(shell["fractional_vector"], dtype=float)
        a, b, c, alpha, beta, gamma = lattice_tuple

        cart_vector = fractional_to_cartesian(frac_vector, a, b, c, alpha, beta, gamma)
        r_ij = float(np.linalg.norm(cart_vector))

        # weight/multiplicity
        m_ij = int(shell.get("coordination", shell.get("multiplicity", 1)))

        # correlation term (July behavior: lambda_k for first shells else delta1/delta2)
        if k < N_lambda:
            base_corr = 1.0 - float(lambda_vals[k])
        else:
            base_corr = 1.0 - float(delta1) / r_ij - float(delta2) / (r_ij**2)


        # HKL for Wilkens only
        hkl = tuple(shell.get("hkl", (0, 0, 0))) if wilkens_active else (0, 0, 0)

        # Wilkens microstrain (legacy single-shell path; kept only for compatibility/debug)
        strain_L2 = 0.0
        if wilkens_active:
            # burgers magnitude
            if self.burgers_mag is not None and float(self.burgers_mag) > 0:
                bmag = float(self.burgers_mag)
            else:
                bmag = (np.sqrt(3.0) / 2.0) * float(a)
        
            # directional contrast factor (one hkl at a time)
            h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
            Chkl = float(
                self._contrast_factor_one_from_params(
                    h,
                    k,
                    l,
                    params or {},
                    float(fE),
                )
            )
        
            # Wilkens separable form: <eps^2>(L)= (rho*b^2/(4*pi)) * Chkl * f*(L/Re)
            eps2 = (float(rho) * (bmag ** 2) / (4.0 * np.pi)) * Chkl * float(fstar_vec(np.array([r_ij], dtype=float), float(Re))[0])
            strain_L2 = (r_ij ** 2) * eps2

        corr = max(base_corr, 0.0)

        sigma2 = (
            (1.0 / (4.0 * np.pi**2)) * float(biso_avg) * corr
            + (float(delta_g) * r_ij) ** 2
            + (float(delta_broad) * r_ij) ** 2
            + strain_L2
        )

        sigma = np.sqrt(max(sigma2, 1e-24))


        x = r - r_ij
        mask = np.abs(x) < 5.0 * sigma
        if not np.any(mask):
            return np.zeros_like(r), r_ij

        # Density of the *neighbor* species beta
        beta_name = clean_element_name(shell["pair_type"].split("-")[1])
        rho_beta = float(densities[beta_name])

        # July normalization factor
        A_ij = m_ij / (4.0 * np.pi * (r_ij**2) * rho_beta)

        gaussian = np.exp(-0.5 * (x[mask] / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)

        out = np.zeros_like(r)
        out[mask] = A_ij * gaussian
        return out, r_ij

    def _prepare_shell_runtime_arrays(self) -> None:
        """
        Precompute helper arrays reused in every PDF evaluation.

        Uses cached strain_unique_dirs / strain_inv_dirs when available.
        """
        try:
            self._shell_upair_id_i64 = self._shell_upair_id.astype(np.int64, copy=False)
            self._shell_k_within_upair_i64 = self._shell_k_within_upair.astype(np.int64, copy=False)
            self._shell_k_rank_i64 = self._shell_k_rank.astype(np.int64, copy=False)

            # Prefer cached unique direction arrays from StructureHandler cache.
            if hasattr(self, "_loaded_strain_unique_dirs") and self._loaded_strain_unique_dirs is not None:
                self._strain_unique_dirs = self._loaded_strain_unique_dirs.astype(np.int32, copy=False)
                self._strain_inv_dirs = self._loaded_strain_inv_dirs.astype(np.int32, copy=False)

                print(
                    f"[PDF] Loaded cached {len(self._strain_unique_dirs)} unique "
                    f"strain direction(s) for {len(self._shell_direction)} shell rows."
                )
                return

            # Fallback: compute once.
            dirs_i32 = self._shell_direction.astype(np.int32, copy=False)

            self._strain_unique_dirs, self._strain_inv_dirs = np.unique(
                dirs_i32,
                axis=0,
                return_inverse=True,
            )

            self._strain_unique_dirs = self._strain_unique_dirs.astype(np.int32, copy=False)
            self._strain_inv_dirs = self._strain_inv_dirs.astype(np.int32, copy=False)

            print(
                f"[PDF] Precomputed {len(self._strain_unique_dirs)} unique "
                f"strain direction(s) for {len(self._shell_direction)} shell rows."
            )

        except Exception as e:
            print(f"[WARN] Could not precompute shell runtime arrays: {e}")
            self._shell_upair_id_i64 = self._shell_upair_id.astype(np.int64)
            self._shell_k_within_upair_i64 = self._shell_k_within_upair.astype(np.int64)
            self._shell_k_rank_i64 = self._shell_k_rank.astype(np.int64)
            self._strain_unique_dirs = None
            self._strain_inv_dirs = None


    def _prepare_contrast_factor_matrix(self) -> None:
        """
        Precompute the invariant feature matrix X(hkl) for all unique strain
        directions.

        Then during refinement:
            C_edge  = X @ E_edge
            C_screw = X @ E_screw
            Chkl    = fE*C_edge + (1-fE)*C_screw

        This replaces a very slow Python loop over ~10^6 directions.
        The matrix is built once using the initial lattice/cell.
        """
        self._cf_feature_names = []
        self._cf_feature_matrix = None

        try:
            dirs = getattr(self, "_strain_unique_dirs", None)
            if dirs is None or len(dirs) == 0:
                return

            dirs = np.asarray(dirs, dtype=np.int32)

            # Use first direction to get term names / matrix width.
            h0, k0, l0 = int(dirs[0, 0]), int(dirs[0, 1]), int(dirs[0, 2])
            names0, x0 = feature_vector_hkl(
                int(self.spacegroup_number),
                h0,
                k0,
                l0,
                self.cell,
            )

            n_dir = int(dirs.shape[0])
            n_terms = int(len(names0))

            X = np.zeros((n_dir, n_terms), dtype=np.float32)
            X[0, :] = np.asarray(x0, dtype=np.float32)

            for i in range(1, n_dir):
                h, k, l = int(dirs[i, 0]), int(dirs[i, 1]), int(dirs[i, 2])

                try:
                    names_i, x_i = feature_vector_hkl(
                        int(self.spacegroup_number),
                        h,
                        k,
                        l,
                        self.cell,
                    )
                    X[i, :] = np.asarray(x_i, dtype=np.float32)
                except Exception:
                    # Leave row as zero if something is invalid.
                    pass

            self._cf_feature_names = list(names0)
            self._cf_feature_matrix = X

            print(
                f"[PDF] Precomputed Chkl feature matrix: "
                f"{n_dir} direction(s) x {n_terms} term(s)."
            )

        except Exception as e:
            print(f"[WARN] Could not precompute Chkl feature matrix: {e}")
            self._cf_feature_names = []
            self._cf_feature_matrix = None

    def _params_lc(self, params: dict) -> dict:
        """Case-insensitive parameter dictionary."""
        return {str(k).strip().lower(): v for k, v in (params or {}).items()}

    def _pfloat(self, params_lc: dict, names, default=0.0) -> float:
        """Read a float parameter case-insensitively from params_lc."""
        for name in names:
            key = str(name).strip().lower()
            if key in params_lc:
                try:
                    return float(params_lc[key])
                except Exception:
                    pass
        return float(default)

    def _has_cubic_cf_params(self, params: dict) -> bool:
        """
        Cubic A/B contrast-factor coefficients as normal parameters:
            CEdgeA, CEdgeB, CScrewA, CScrewB
        """
        p = self._params_lc(params)
        return any(k in p for k in ("cedgea", "cedgeb", "cscrewa", "cscrewb"))

    def _cubic_cf_coeffs(self, params: dict):
        """
        Return cubic A/B coefficients from normal fit params.

        Defaults are your previous fixed cubic values.
        """
        p = self._params_lc(params)

        CEdgeA = self._pfloat(p, ("CEdgeA", "cedgea"), 0.265280)
        CEdgeB = self._pfloat(p, ("CEdgeB", "cedgeb"), -0.355950)
        CScrewA = self._pfloat(p, ("CScrewA", "cscrewa"), 0.307288)
        CScrewB = self._pfloat(p, ("CScrewB", "cscrewb"), -0.819979)

        return CEdgeA, CEdgeB, CScrewA, CScrewB

    def _contrast_factor_one_from_params(self, h: int, k: int, l: int, params: dict, fE: float) -> float:
        """
        Compute one C_hkl from normal fit parameters only.

        Cubic:
            CEdgeA, CEdgeB, CScrewA, CScrewB

        Non-cubic/invariant:
            EdgeE1..., ScrewE1...
        """
        h = int(h)
        k = int(k)
        l = int(l)
        fE = float(fE)

        if h == 0 and k == 0 and l == 0:
            return 0.0

        p = self._params_lc(params)

        # -------------------------
        # Cubic A/B form
        # -------------------------
        if self._has_cubic_cf_params(params):
            CEdgeA, CEdgeB, CScrewA, CScrewB = self._cubic_cf_coeffs(params)

            h2 = h * h
            k2 = k * k
            l2 = l * l

            denom = float((h2 + k2 + l2) ** 2)
            if denom <= 0.0:
                return 0.0

            q = float(h2 * k2 + k2 * l2 + l2 * h2) / denom

            C_edge = CEdgeA + CEdgeB * q
            C_screw = CScrewA + CScrewB * q

            return fE * C_edge + (1.0 - fE) * C_screw

        # -------------------------
        # Invariant EdgeE*/ScrewE* form
        # -------------------------
        has_edge = self._has_cf_params_or_config(params, "edge")
        has_screw = self._has_cf_params_or_config(params, "screw")

        if not has_edge and not has_screw:
            return 0.0

        try:
            names, x = feature_vector_hkl(
                int(self.spacegroup_number),
                h,
                k,
                l,
                self.cell,
            )
        except Exception:
            return 0.0

        def eval_family(prefix_pretty: str, prefix_lc: str) -> float:
            val = 0.0
            for name, xi in zip(names, x):
                term = str(name).upper()  # E1, E2, ...
                candidates = (
                    f"{prefix_pretty}{term}",
                    f"{prefix_pretty}_{term}",
                    f"{prefix_lc}{term}",
                    f"{prefix_lc}_{term}",
                )
                coeff = self._pfloat(p, candidates, 0.0)
                val += coeff * float(xi)
            return float(val)

        C_edge = eval_family("Edge", "edge")
        C_screw = eval_family("Screw", "screw")

        # If only one family exists, use it for both.
        if has_edge and not has_screw:
            C_screw = C_edge
        elif has_screw and not has_edge:
            C_edge = C_screw

        return fE * C_edge + (1.0 - fE) * C_screw
    

    def _has_cf_params_or_config(self, params: dict, kind: str) -> bool:
        """
        Check whether EdgeE*/ScrewE* coefficients exist as normal parameters.

        kind: 'edge' or 'screw'
        """
        try:
            prefix = "edge" if kind == "edge" else "screw"

            for k in (params or {}).keys():
                kl = str(k).strip().lower()

                # Accept:
                #   EdgeE1, Edge_E1, edgee1, edge_e1
                #   ScrewE1, Screw_E1, screwe1, screw_e1
                if re.match(rf"^{prefix}[_ ]*e\d+$", kl):
                    return True

        except Exception:
            pass

        return False


    def _contrast_coeff_vector(self, params: dict, kind: str) -> np.ndarray:
        """
        Build invariant coefficient vector from normal fit parameters only.

        Supports:
            EdgeE1, EdgeE2, ...
            ScrewE1, ScrewE2, ...
            Edge_E1, Screw_E1, ...
        """
        names = list(getattr(self, "_cf_feature_names", []) or [])
        vec = np.zeros(len(names), dtype=np.float32)

        if not names:
            return vec

        params_lc = self._params_lc(params)

        if kind == "edge":
            prefix = "edge"
            pretty_prefix = "Edge"
        else:
            prefix = "screw"
            pretty_prefix = "Screw"

        for i, term in enumerate(names):
            term_u = str(term).upper()  # E1, E2, ...

            candidates = [
                f"{pretty_prefix}{term_u}",     # EdgeE1
                f"{pretty_prefix}_{term_u}",    # Edge_E1
                f"{prefix}{term_u}",            # edgeE1
                f"{prefix}_{term_u}",           # edge_E1
            ]

            vec[i] = np.float32(self._pfloat(params_lc, candidates, 0.0))

        return vec
    

    def _compute_chkl_unique_dirs_fast(self, params: dict, fE: float) -> np.ndarray:
        """
        Fast Chkl computation for all unique directions.

        Supports both:
          1. Invariant coefficients:
                EdgeE1, EdgeE2, ...
                ScrewE1, ScrewE2, ...

          2. Legacy cubic coefficients:
                CEdgeA, CEdgeB, CScrewA, CScrewB
             parsed from input keys:
                cedgea, cedgeb, cscrewa, cscrewb
        """
        dirs = getattr(self, "_strain_unique_dirs", None)
        if dirs is None:
            return np.zeros((0,), dtype=self._dtype)

        dirs = np.asarray(dirs, dtype=np.int32)

        has_edge = self._has_cf_params_or_config(params, "edge")
        has_screw = self._has_cf_params_or_config(params, "screw")
        has_cubic_ab = self._has_cubic_cf_params(params)
        if not has_edge and not has_screw and not has_cubic_ab:
            return np.zeros(dirs.shape[0], dtype=self._dtype)

        # ---------------------------------------------------------
        # Case 1: legacy cubic contrast-factor formula
        #
        # Old formula:
        #   C_hkl = A + B * (h²k² + k²l² + l²h²)/(h²+k²+l²)²
        #
        # where:
        #   A = fE*CEdgeA + (1-fE)*CScrewA
        #   B = fE*CEdgeB + (1-fE)*CScrewB
        # ---------------------------------------------------------
        if (not has_edge) and (not has_screw) and has_cubic_ab:
            try:
                CEdgeA, CEdgeB, CScrewA, CScrewB = self._cubic_cf_coeffs(params)

                h = dirs[:, 0].astype(np.float64)
                k = dirs[:, 1].astype(np.float64)
                l = dirs[:, 2].astype(np.float64)

                h2 = h * h
                k2 = k * k
                l2 = l * l

                denom = (h2 + k2 + l2) ** 2
                num = h2 * k2 + k2 * l2 + l2 * h2

                A = float(fE) * CEdgeA + (1.0 - float(fE)) * CScrewA
                B = float(fE) * CEdgeB + (1.0 - float(fE)) * CScrewB

                out = np.zeros_like(denom, dtype=np.float64)
                mask = denom > 0.0
                out[mask] = A + B * num[mask] / denom[mask]

                out[~np.isfinite(out)] = 0.0
                out = np.maximum(out, 0.0)

                return np.asarray(out, dtype=self._dtype)

            except Exception:
                return np.zeros(dirs.shape[0], dtype=self._dtype)

        # ---------------------------------------------------------
        # Case 2: invariant Popa/TOPAS-like coefficients
        #
        #   Chkl = fE*(X @ E_edge) + (1-fE)*(X @ E_screw)
        # ---------------------------------------------------------
        if getattr(self, "_cf_feature_matrix", None) is None:
            self._prepare_contrast_factor_matrix()

        X = getattr(self, "_cf_feature_matrix", None)

        if X is None:
            # Fallback: slow path, should rarely happen.
            out = np.zeros(len(dirs), dtype=self._dtype)

            for i, d in enumerate(dirs):
                h, k, l = int(d[0]), int(d[1]), int(d[2])
                if h == 0 and k == 0 and l == 0:
                    continue

                try:
                    val = self._contrast_factor_one_from_params(
                        h,
                        k,
                        l,
                        params,
                        float(fE),
                    )
                    if not np.isfinite(val):
                        val = 0.0
                    out[i] = self._dtype(max(float(val), 0.0))
                except Exception:
                    out[i] = self._dtype(0.0)

            return out

        E_edge = self._contrast_coeff_vector(params, "edge")
        E_screw = self._contrast_coeff_vector(params, "screw")

        C_edge = X @ E_edge
        C_screw = X @ E_screw

        # Match previous behavior:
        # if only one family exists, use it for both.
        if has_edge and not has_screw:
            C_screw = C_edge
        elif has_screw and not has_edge:
            C_edge = C_screw

        Chkl = float(fE) * C_edge + (1.0 - float(fE)) * C_screw

        Chkl = np.asarray(Chkl, dtype=self._dtype)
        Chkl[~np.isfinite(Chkl)] = self._dtype(0.0)
        Chkl = np.maximum(Chkl, self._dtype(0.0))

        return Chkl

    def _plot_distance_limit_from_params(self, params: dict, fallback: float = 5.0) -> float:
        """
        Limit diagnostic plots to physical particle size.

        If d/D is present and > 0, use it.
        Otherwise use fallback, default 5 Å.
        """
        try:
            D = float(params.get("d", params.get("D", 0.0)))
            if np.isfinite(D) and D > 0.0:
                return float(D)
        except Exception:
            pass

        return float(fallback)

    def build_local_trends(self, params: dict, *, max_r_plot: float = 10.0) -> dict:
        """
        Build local lambda/delta trends up to max_r_plot.

        This is intentionally lightweight:
        - no G(r) calculation
        - no Gaussian accumulation
        - no qmax termination
        - only shell distances and local-dynamics parameters

        Output:
            {
                "ca-o": {
                    "r": [...],
                    "lambda": [...],
                    "delta_eff": [...],
                    "delta1": value,
                    "delta2": value,
                    "max_r_plot": value,
                },
                ...
            }
        """
        out = {}

        try:
            params = params or {}

            shell_frac = getattr(self, "_shell_frac", None)
            shell_upair_id = getattr(self, "_shell_upair_id", None)
            shell_k_within_upair = getattr(self, "_shell_k_within_upair", None)
            upair_list = getattr(self, "_upair_list", None)

            if (
                shell_frac is None
                or shell_upair_id is None
                or shell_k_within_upair is None
                or not upair_list
            ):
                return out

            shell_frac = np.asarray(shell_frac, dtype=float)
            shell_upair_id = np.asarray(shell_upair_id, dtype=np.int32)
            shell_k_within_upair = np.asarray(shell_k_within_upair, dtype=np.int32)

            max_r_plot = float(max_r_plot)
            if not np.isfinite(max_r_plot) or max_r_plot <= 0.0:
                max_r_plot = 10.0

            # ---------------------------------------------------------
            # Current lattice parameters
            # ---------------------------------------------------------
            lat0 = self.structure_handler.structure.lattice

            a = float(params.get("a", lat0.a))
            b = float(params.get("b", lat0.b))
            c = float(params.get("c", lat0.c))
            alpha = float(params.get("alpha", lat0.alpha))
            beta = float(params.get("beta", lat0.beta))
            gamma = float(params.get("gamma", lat0.gamma))

            lattice_dict = {
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }

            lattice_dict = apply_lattice_constraints(
                lattice_dict,
                getattr(self, "constraints", None),
            )

            lattice_tuple = (
                float(lattice_dict["a"]),
                float(lattice_dict["b"]),
                float(lattice_dict["c"]),
                float(lattice_dict["alpha"]),
                float(lattice_dict["beta"]),
                float(lattice_dict["gamma"]),
            )

            # ---------------------------------------------------------
            # Pair distances
            # ---------------------------------------------------------
            M = lattice_matrix(*lattice_tuple)
            cart = shell_frac @ M.T
            r_ij = np.sqrt(np.sum(cart * cart, axis=1))

            valid = (
                np.isfinite(r_ij)
                & (r_ij > 0.0)
                & (r_ij <= max_r_plot)
            )

            if not np.any(valid):
                return out

            uid_v = shell_upair_id[valid]
            k_v = shell_k_within_upair[valid]
            r_v = r_ij[valid]

            # ---------------------------------------------------------
            # Fast grouping:
            # for each (unordered pair, shell k), keep minimum distance.
            # This avoids repeated full-array masks for every k.
            # ---------------------------------------------------------
            order = np.lexsort((r_v, k_v, uid_v))

            uid_s = uid_v[order]
            k_s = k_v[order]
            r_s = r_v[order]

            first = np.ones(uid_s.size, dtype=bool)
            if uid_s.size > 1:
                first[1:] = (
                    (uid_s[1:] != uid_s[:-1])
                    | (k_s[1:] != k_s[:-1])
                )

            uid_g = uid_s[first]
            k_g = k_s[first]
            r_g = r_s[first]

            delta1_global = float(params.get("delta1", 0.0))
            delta2_global = float(params.get("delta2", 0.0))

            # ---------------------------------------------------------
            # Build curves per unordered pair
            # ---------------------------------------------------------
            for uid in np.unique(uid_g):
                uid = int(uid)

                if uid < 0 or uid >= len(upair_list):
                    continue

                try:
                    aa, bb = upair_list[uid]
                    pair = f"{aa}-{bb}"
                except Exception:
                    continue

                rows = uid_g == uid

                if not np.any(rows):
                    continue

                r_vals = r_g[rows].astype(float)
                k_vals = k_g[rows].astype(int)

                # Pair-specific delta keys
                d1_pair_key_dash = f"delta1_{pair}"
                d2_pair_key_dash = f"delta2_{pair}"

                d1_pair_key_us = f"delta1_{aa}_{bb}"
                d2_pair_key_us = f"delta2_{aa}_{bb}"

                delta1_use = float(
                    params.get(
                        d1_pair_key_dash,
                        params.get(d1_pair_key_us, delta1_global),
                    )
                )

                delta2_use = float(
                    params.get(
                        d2_pair_key_dash,
                        params.get(d2_pair_key_us, delta2_global),
                    )
                )

                lambda_vals = []
                delta_eff_vals = []

                for rr, kk in zip(r_vals, k_vals):
                    lam_key_dash = f"lambda_{pair}_{int(kk)}"
                    lam_key_us = f"lambda_{aa}_{bb}_{int(kk)}"

                    if lam_key_dash in params:
                        lam_val = float(params[lam_key_dash])
                    elif lam_key_us in params:
                        lam_val = float(params[lam_key_us])
                    else:
                        lam_val = float("nan")

                    lambda_vals.append(lam_val)
                    delta_eff_vals.append(float(delta1_use / rr + delta2_use / (rr * rr)))

                sort_order = np.argsort(r_vals)

                out[pair] = {
                    "r": [float(r_vals[i]) for i in sort_order],
                    "lambda": [float(np.asarray(lambda_vals)[i]) for i in sort_order],
                    "delta_eff": [float(np.asarray(delta_eff_vals)[i]) for i in sort_order],
                    "delta1": float(delta1_use),
                    "delta2": float(delta2_use),
                    "max_r_plot": float(max_r_plot),
                }

        except Exception:
            return {}

        return out

    def build_pdf_tick_data(
        self,
        params: dict,
        *,
        max_r: float | None = None,
        max_ticks_per_pair: int | None = None,
    ) -> dict:
        """
        Build lightweight tick-marker metadata from cached shell arrays.

        No PDF calculation is done here.

        Notes
        -----
        - By default, this does NOT downsample.
        - If max_ticks_per_pair is given, downsampling happens after sorting.
        - Ordered pair channels that map to the same displayed unordered pair
          label are merged, not overwritten.

        Returns
        -------
        {
            "pairs": {
                "ca-o": {
                    "r": [...],
                    "shell": [...],
                    "direction": ["[1 0 4]", ...],
                    "multiplicity": [...],
                },
                ...
            }
        }
        """
        out = {"pairs": {}}

        try:
            params = params or {}

            shell_frac = getattr(self, "_shell_frac", None)
            shell_pair_id = getattr(self, "_shell_pair_id", None)
            shell_dir = getattr(self, "_shell_direction", None)
            shell_mult = getattr(self, "_shell_mult", None)
            shell_k = getattr(self, "_shell_k_within_upair", None)
            pair_list = getattr(self, "_pair_list", None)

            if (
                shell_frac is None
                or shell_pair_id is None
                or shell_dir is None
                or shell_mult is None
                or shell_k is None
                or not pair_list
            ):
                return out

            shell_frac = np.asarray(shell_frac, dtype=float)
            shell_pair_id = np.asarray(shell_pair_id, dtype=np.int32)
            shell_dir = np.asarray(shell_dir, dtype=np.int32)
            shell_mult = np.asarray(shell_mult, dtype=np.int32)
            shell_k = np.asarray(shell_k, dtype=np.int32)

            # ---------------------------------------------------------
            # Current lattice
            # ---------------------------------------------------------
            lat0 = self.structure_handler.structure.lattice

            a = float(params.get("a", lat0.a))
            b = float(params.get("b", lat0.b))
            c = float(params.get("c", lat0.c))
            alpha = float(params.get("alpha", lat0.alpha))
            beta = float(params.get("beta", lat0.beta))
            gamma = float(params.get("gamma", lat0.gamma))

            lattice_dict = {
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }

            lattice_dict = apply_lattice_constraints(
                lattice_dict,
                getattr(self, "constraints", None),
            )

            lattice_tuple = (
                float(lattice_dict["a"]),
                float(lattice_dict["b"]),
                float(lattice_dict["c"]),
                float(lattice_dict["alpha"]),
                float(lattice_dict["beta"]),
                float(lattice_dict["gamma"]),
            )

            M = lattice_matrix(*lattice_tuple)
            cart = shell_frac @ M.T
            r_ij = np.sqrt(np.sum(cart * cart, axis=1))

            if max_r is None:
                max_r = float(np.nanmax(r_ij)) if r_ij.size else 0.0

            max_r = float(max_r)

            valid = (
                np.isfinite(r_ij)
                & (r_ij > 0.0)
                & (r_ij <= max_r)
            )

            if not np.any(valid):
                return out

            # ---------------------------------------------------------
            # Pair labels.
            #
            # The shell table can contain ordered pair channels such as:
            #   ca-o and o-ca
            #
            # For display, merge them into one unordered label:
            #   ca-o
            # ---------------------------------------------------------
            pair_labels = []

            for p in pair_list:
                try:
                    aa, bb = p
                    aa = str(aa)
                    bb = str(bb)

                    if aa <= bb:
                        pair_labels.append(f"{aa}-{bb}")
                    else:
                        pair_labels.append(f"{bb}-{aa}")

                except Exception:
                    pair_labels.append(str(p))

            # ---------------------------------------------------------
            # Temporary merged storage.
            # Each unordered pair label accumulates all ticks from all
            # ordered channels that map to that label.
            # ---------------------------------------------------------
            merged = {}

            for pid in np.unique(shell_pair_id[valid]):
                pid = int(pid)

                if pid < 0 or pid >= len(pair_labels):
                    continue

                pair = pair_labels[pid]
                mask = valid & (shell_pair_id == pid)

                if not np.any(mask):
                    continue

                rr = r_ij[mask]
                kk = shell_k[mask]
                dd = shell_dir[mask]
                mm = shell_mult[mask]

                if pair not in merged:
                    merged[pair] = {
                        "r": [],
                        "shell": [],
                        "direction": [],
                        "multiplicity": [],
                    }

                for rval, kval, dval, mval in zip(rr, kk, dd, mm):
                    h, k, l = [int(x) for x in dval]

                    merged[pair]["r"].append(float(rval))
                    merged[pair]["shell"].append(int(kval) + 1)
                    merged[pair]["direction"].append(f"[{h} {k} {l}]")
                    merged[pair]["multiplicity"].append(int(mval))

            # ---------------------------------------------------------
            # Sort each pair by distance and optionally downsample.
            #
            # THIS IS WHERE THE BLOCK GOES.
            # It must happen after rr/shells/directions/multiplicities exist,
            # and before writing out["pairs"][pair].
            # ---------------------------------------------------------
            for pair, data in merged.items():
                rr = np.asarray(data["r"], dtype=float)

                if rr.size == 0:
                    continue

                order = np.argsort(rr)

                rr = rr[order]
                shells = [data["shell"][i] for i in order]
                directions = [data["direction"][i] for i in order]
                multiplicities = [data["multiplicity"][i] for i in order]

                # Optional downsampling.
                # By default max_ticks_per_pair is None, so no downsampling happens.
                if (
                    max_ticks_per_pair is not None
                    and int(max_ticks_per_pair) > 0
                    and rr.size > int(max_ticks_per_pair)
                ):
                    idx = np.linspace(
                        0,
                        rr.size - 1,
                        int(max_ticks_per_pair),
                    ).astype(int)

                    rr = rr[idx]
                    shells = [shells[i] for i in idx]
                    directions = [directions[i] for i in idx]
                    multiplicities = [multiplicities[i] for i in idx]

                out["pairs"][pair] = {
                    "r": [float(x) for x in rr],
                    "shell": [int(x) for x in shells],
                    "direction": directions,
                    "multiplicity": [int(x) for x in multiplicities],
                }

        except Exception:
            return {"pairs": {}}

        return out

    def build_warren_plot(self, params: dict, *, max_L=None) -> dict:
        """
        Fast Warren-plot builder.

        This avoids computing full G(r), pair contributions, Gaussian accumulation, etc.
        Only shell distances and directional strain are computed.
        """
        params = params or {}

        refcfg = (self.config.get("refinement") or {}) if isinstance(self.config, dict) else {}

        microstrain_model = str(
            params.get(
                "microstrain_model",
                refcfg.get("microstrain_model", "auto"),
            )
        ).strip().lower()

        rho = float(params.get("rho", 0.0))
        Re = float(params.get("Re", params.get("re", 100.0)))
        fE = float(params.get("fE", params.get("fe", 0.5)))

        pah_a = float(params.get("pah_a", params.get("PAH_a", 0.0)))
        pah_b = float(params.get("pah_b", params.get("PAH_b", 0.0)))

        if microstrain_model in ("wilkens", "wilkins", "dislocation"):
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = False

        elif microstrain_model in ("pah", "adler-houska", "adler_houska"):
            wilkens_active = False
            pah_active = (pah_a != 0.0 or pah_b != 0.0)

        else:
            # Backward-compatible behavior
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = (not wilkens_active) and (pah_a != 0.0 or pah_b != 0.0)

        if not (wilkens_active or pah_active):
            return {}
        

        cache_key = (
            str(microstrain_model),
            round(float(params.get("a", self.structure_handler.structure.lattice.a)), 10),
            round(float(params.get("b", params.get("a", self.structure_handler.structure.lattice.b))), 10),
            round(float(params.get("c", params.get("a", self.structure_handler.structure.lattice.c))), 10),
            round(float(rho), 12),
            round(float(Re), 8),
            round(float(fE), 8),
            round(float(pah_a), 12),
            round(float(pah_b), 12),
            round(float(max_L if max_L is not None else -1.0), 8),
            tuple(
                sorted(
                    (str(k), float(v))
                    for k, v in params.items()
                    if re.match(r"(?i)^(edge|screw)[_ ]*e\d+$", str(k))
                    or str(k).lower() in ("cedgea", "cedgeb", "cscrewa", "cscrewb")
                )
            ),
        )

        if self._warren_cache_key == cache_key and self._warren_cache_value is not None:
            return self._warren_cache_value

        # --- lattice parameters with constraints ---
        a = float(params.get("a", self.structure_handler.structure.lattice.a))
        b = float(params.get("b", a))
        c = float(params.get("c", a))
        alpha = float(params.get("alpha", 90.0))
        beta = float(params.get("beta", 90.0))
        gamma = float(params.get("gamma", 90.0))

        lattice_dict = {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
        lattice_dict = apply_lattice_constraints(lattice_dict, self.constraints)

        lattice_tuple = (
            float(lattice_dict["a"]),
            float(lattice_dict["b"]),
            float(lattice_dict["c"]),
            float(lattice_dict["alpha"]),
            float(lattice_dict["beta"]),
            float(lattice_dict["gamma"]),
        )

        self.cell = Cell(*lattice_tuple)

        refcfg = (self.config.get("refinement") or {}) if isinstance(self.config, dict) else {}

        microstrain_model = str(
            params.get(
                "microstrain_model",
                refcfg.get("microstrain_model", "auto"),
            )
        ).strip().lower()

        rho = float(params.get("rho", 0.0))
        Re = float(params.get("Re", params.get("re", 100.0)))
        fE = float(params.get("fE", params.get("fe", 0.5)))

        pah_a = float(params.get("pah_a", params.get("PAH_a", 0.0)))
        pah_b = float(params.get("pah_b", params.get("PAH_b", 0.0)))

        if microstrain_model in ("wilkens", "wilkins", "dislocation"):
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = False
        elif microstrain_model in ("pah", "adler-houska", "adler_houska"):
            wilkens_active = False
            pah_active = (pah_a != 0.0 or pah_b != 0.0)
        else:
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = (not wilkens_active) and (pah_a != 0.0 or pah_b != 0.0)

        if not (wilkens_active or pah_active):
            return {}

        if max_L is None:
            max_L = self._plot_distance_limit_from_params(params, fallback=5.0)

        max_L = float(max_L)

        # --- shell distances ---
        M = lattice_matrix(*lattice_tuple).astype(self._dtype, copy=False)
        cart = self._shell_frac @ M.T
        r_ij = np.sqrt(np.sum(cart * cart, axis=1)).astype(float)

        # Limit Warren data by particle size or 5 Å fallback
        valid_L = np.isfinite(r_ij) & (r_ij > 0.5) & (r_ij <= max_L)
        if not np.any(valid_L):
            return {}

        # --- Burgers magnitude ---
        params_lc = self._params_lc(params)
        bmag = self._pfloat(
            params_lc,
            ("burgers_mag", "burgers", "b_mag", "b"),
            0.0,
        )

        if bmag <= 0.0:
            bmag = np.sqrt(3.0) / 2.0 * float(lattice_tuple[0])

        pref = float(rho) * (float(bmag) ** 2) / (4.0 * np.pi)

        try:
            crystal_system = SpacegroupAnalyzer(
                self.structure_handler.structure,
                symprec=1e-3,
            ).get_crystal_system().lower()
        except Exception:
            crystal_system = ""

        from math import gcd

        def _reduce_dir(h: int, k: int, l: int) -> tuple[int, int, int]:
            h, k, l = int(h), int(k), int(l)

            if h == 0 and k == 0 and l == 0:
                return (0, 0, 0)

            g = gcd(gcd(abs(h), abs(k)), abs(l))
            if g > 0:
                h //= g
                k //= g
                l //= g

            # Make opposite directions equivalent
            for v in (h, k, l):
                if v != 0:
                    if v < 0:
                        h, k, l = -h, -k, -l
                    break

            return (h, k, l)

        def _canonical_family(h: int, k: int, l: int) -> tuple[int, int, int]:
            h, k, l = _reduce_dir(h, k, l)

            if (h, k, l) == (0, 0, 0):
                return (0, 0, 0)

            # Cubic: full permutation family
            if crystal_system == "cubic":
                vals = sorted((abs(h), abs(k), abs(l)), reverse=True)
                return (int(vals[0]), int(vals[1]), int(vals[2]))

            # Tetragonal / hexagonal / trigonal:
            # a,b equivalent in-plane, c special
            if crystal_system in ("tetragonal", "hexagonal", "trigonal"):
                ab = sorted((abs(h), abs(k)), reverse=True)
                return (int(ab[0]), int(ab[1]), int(abs(l)))

            # Lower symmetry: only opposite directions equivalent
            return (h, k, l)

        dirs = self._shell_direction.astype(np.int32, copy=False)

        family_to_L = defaultdict(set)

        for dvec, L, ok in zip(dirs, r_ij, valid_L):
            if not ok:
                continue

            h, k, l = int(dvec[0]), int(dvec[1]), int(dvec[2])
            fam = _canonical_family(h, k, l)

            if fam == (0, 0, 0):
                continue

            family_to_L[fam].add(round(float(L), 5))

        strain_by_direction = {}

        for fam, L_set in family_to_L.items():
            if not L_set:
                continue

            h, k, l = fam
            L_sorted = np.asarray(sorted(L_set), dtype=float)

            # One contrast factor per direction family
            cval = self._contrast_factor_one_from_params(
                int(h),
                int(k),
                int(l),
                params,
                float(fE),
            )

            if not np.isfinite(cval):
                cval = 0.0

            cval = max(float(cval), 0.0)

            # Warren relation:
            # <Delta L^2>(L) = L^2 * rho*b^2/(4*pi) * C_hkl * f*(L/Re)
            if wilkens_active:
                dL2 = (
                    L_sorted * L_sorted
                    * pref
                    * cval
                    * fstar_vec(L_sorted, float(Re))
                )

            elif pah_active:
                # PAH Warren variance:
                #
                # <Delta L^2> =
                #     (d_hkl^4 / a^4) Gamma_hkl * (pah_a * L + pah_b * L^2)
                #
                # cval already contains:
                #     (d_hkl^4 / a^4) Gamma_hkl
                dL2 = cval * (
                    float(pah_a) * L_sorted
                    + float(pah_b) * L_sorted * L_sorted
                )

            else:
                dL2 = np.zeros_like(L_sorted)

            dL2 = np.maximum(np.asarray(dL2, dtype=float), 0.0)

            strain_by_direction[fam] = [
                (float(L), float(v)) for L, v in zip(L_sorted, dL2)
            ]


        self._warren_cache_key = cache_key
        self._warren_cache_value = strain_by_direction


        return strain_by_direction


    def _biso_values_vector_from_params(self, params: dict) -> np.ndarray:
        """
        Build Biso values aligned to self._unique_species.

        Supports:
          - species Biso:
                biso_Li, biso_Ge, biso_S
          - site Biso:
                biso_li1, biso_li2, biso_ge1, biso_s1, ...

        Current implementation:
          if site-specific Biso values exist for an element, use their average
          as that element's Biso.

        This makes site Biso parameters influence the model.

        A fully site-specific implementation would require shell rows to retain
        alpha/beta site IDs and not just species IDs.
        """
        params = params or {}
        params_lc = self._params_lc(params)

        # Collect site-specific Biso values by element.
        # Example:
        #   biso_li1 -> Li
        #   biso_li2 -> Li
        #   biso_ge1 -> Ge
        site_biso_by_species = {}

        for k, v in params_lc.items():
            key = str(k).strip().lower()

            if not key.startswith("biso_"):
                continue

            raw = key[len("biso_"):].strip()

            if not raw:
                continue

            try:
                sp = clean_element_name(raw)
            except Exception:
                continue

            # Species-level key, e.g. biso_li.
            # Do not treat it as site-specific.
            if raw == sp.lower():
                continue

            try:
                val = float(v)
            except Exception:
                continue

            site_biso_by_species.setdefault(sp, []).append(val)

        biso_by_species_cfg = params.get("biso_by_species", {}) or {}

        out = np.zeros(len(self._unique_species), dtype=float)

        for i, sp in enumerate(self._unique_species):
            sp_clean = clean_element_name(sp)

            # Priority 1: site-specific Biso average.
            if sp_clean in site_biso_by_species and site_biso_by_species[sp_clean]:
                out[i] = float(np.mean(site_biso_by_species[sp_clean]))
                continue

            # Priority 2: explicit species parameter.
            candidates = (
                f"biso_{sp_clean}",
                f"biso_{sp_clean.lower()}",
                f"biso_{sp_clean.upper()}",
            )

            found = False

            for cand in candidates:
                cand_lc = cand.lower()
                if cand_lc in params_lc:
                    try:
                        out[i] = float(params_lc[cand_lc])
                        found = True
                        break
                    except Exception:
                        pass

            if found:
                continue

            # Priority 3: old biso_by_species dict.
            for spk in (sp_clean, sp_clean.lower(), sp_clean.upper(), sp_clean.capitalize()):
                if spk in biso_by_species_cfg:
                    try:
                        out[i] = float(biso_by_species_cfg[spk])
                        found = True
                        break
                    except Exception:
                        pass

            if found:
                continue

            # Default.
            out[i] = 0.3

        return out

    def evaluate(
        self,
        params,
        return_gr: bool = False,
        return_jac: bool = False,
        return_strain_plot: bool = False,
        return_contributions: bool = False,
        r_override: np.ndarray | None = None,
        compute_gamma_avg: bool = False,
        **kwargs,
    ):
        # NOTE: the optimized path is single-process and uses Numba parallelism.
        _prof = bool(getattr(self, "profile_pdf", False))
        if _prof:
            self._profile_eval_n += 1
            _prof_t0 = time.perf_counter()
            _prof_last = _prof_t0
            _prof_parts = []

            def _mark(name):
                nonlocal _prof_last
                now = time.perf_counter()
                _prof_parts.append((name, now - _prof_last))
                _prof_last = now
        else:
            def _mark(name):
                return

        # --- lattice parameters with constraints ---
        a = float(params.get("a", self.structure_handler.structure.lattice.a))
        b = float(params.get("b", a))
        c = float(params.get("c", a))
        alpha = float(params.get("alpha", 90.0))
        beta = float(params.get("beta", 90.0))
        gamma = float(params.get("gamma", 90.0))

        lattice_dict = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
        lattice_dict = apply_lattice_constraints(lattice_dict, self.constraints)
        lattice_tuple = (
            float(lattice_dict["a"]), float(lattice_dict["b"]), float(lattice_dict["c"]),
            float(lattice_dict["alpha"]), float(lattice_dict["beta"]), float(lattice_dict["gamma"])
        )

        # Keep cell consistent with refined lattice for direction cosines etc.
        self.cell = Cell(*lattice_tuple)
        _mark("lattice+cell")
        # shells are prepacked numeric arrays; no dicts in the hot path

        # rho0 from current structure volume (volume changes with lattice refinement only if structure updated;
        # but we are not rebuilding structure each iteration). Use lattice_tuple to compute volume consistently:
        # For correctness, we compute rho0 from original structure N and refined volume via pymatgen lattice formula.
        # easiest: approximate using current structure but scale by volume ratio if needed; for now compute using current structure lattice:
        num_atoms = len(self.structure_handler.structure.sites)
        # recompute volume from lattice_tuple by constructing pymatgen lattice:
        from pymatgen.core import Lattice
        vol = float(Lattice.from_parameters(*lattice_tuple).volume)

        if not np.isfinite(vol) or vol <= 1.0e-12:
            raise ValueError(
                f"Invalid unit-cell volume during PDF evaluation: {vol}. "
                f"lattice_tuple={lattice_tuple}"
            )

        self.rho0 = num_atoms / vol

        # --- densities (number density per species) ---
        densities_vec = self._counts.astype(float) / vol  # aligned to self._unique_species

        # Biso values aligned to self._unique_species.
        #
        # Supports both:
        #   species-level Biso:  biso_Li, biso_Ge, ...
        #   site-level Biso:     biso_li1, biso_li2, biso_ge1, ...
        #
        # If site-level Biso parameters are present, they are averaged per element
        # and then used by the current species-level shell broadening model.
        biso_vals_vec = self._biso_values_vector_from_params(params)

        if not getattr(self, "_printed_biso_param_debug", False):
            self._printed_biso_param_debug = True
            print("[BISO DEBUG] Biso params in model:")
            for k, v in sorted((params or {}).items(), key=lambda kv: str(kv[0]).lower()):
                if str(k).lower().startswith("biso_"):
                    print(f"  {k} = {v}")
            print("[BISO DEBUG] species Biso vector:")
            for sp, val in zip(self._unique_species, biso_vals_vec):
                print(f"  {sp}: {val}")

        _mark("densities+biso")
        # ---------------------------------------------------------
        # Requested output grid
        # ---------------------------------------------------------
        if r_override is not None:
            r_requested = np.asarray(
                r_override,
                dtype=float,
            )

        elif self.r is None:
            r_requested = np.linspace(
                self.r_min,
                self.r_max,
                2000,
            )

        else:
            r_requested = np.asarray(
                self.r,
                dtype=float,
            )

        scale = float(
            params.get(
                "scale",
                1.0,
            )
        )

        delta1 = float(
            params.get(
                "delta1",
                0.0,
            )
        )

        delta2 = float(
            params.get(
                "delta2",
                0.0,
            )
        )

        delta_g = float(
            params.get(
                "delta_g",
                0.0,
            )
        )

        delta_broad = float(
            params.get(
                "delta_broad",
                0.0,
            )
        )

        qdamp = float(
            params.get(
                "qdamp",
                0.0,
            )
        )

        qmax = float(
            params.get(
                "qmax",
                self.config.get(
                    "qmax",
                    0.0,
                ),
            )
        )

        qmax_zeros = int(
            params.get(
                "qmax_zeros",
                self.config.get(
                    "qmax_zeros",
                    5,
                ),
            )
        )

        qmax_pad = str(
            params.get(
                "qmax_pad",
                self.config.get(
                    "qmax_pad",
                    "zero",
                ),
            )
        ).strip().lower()

        # ---------------------------------------------------------
        # Internal grid for Qmax termination
        #
        # The sine-transform kernel requires the model below the
        # fitted r_min. Starting the convolution at r_min creates an
        # artificial zero boundary and suppresses the first peaks.
        # ---------------------------------------------------------
        use_internal_qmax_grid = False

        r = r_requested

        if (
            qmax > 0.0
            and r_requested.size > 1
        ):
            dr_requested = float(
                np.median(
                    np.diff(r_requested)
                )
            )

            if (
                np.isfinite(dr_requested)
                and dr_requested > 0.0
            ):
                qmax_width = (
                    float(qmax_zeros)
                    * np.pi
                    / qmax
                )

                internal_r_max = (
                    float(r_requested[-1])
                    + qmax_width
                )

                r = np.arange(
                    0.0,
                    internal_r_max
                    + 0.5 * dr_requested,
                    dr_requested,
                    dtype=float,
                )

                use_internal_qmax_grid = True

        r32 = np.asarray(
            r,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Size / common-volume shape factors
        # ------------------------------------------------------------------
        shape_cfg = self.config.get("crystallite_shape", {}) or {}
        shape_mode = str(shape_cfg.get("mode", "conventional")).strip().lower()

        cyl_diameter = float(
            params.get(
                "cyl_diameter",
                params.get("cylinder_diameter", params.get("disk_diameter", 0.0)),
            )
        )

        cyl_thickness = float(
            params.get(
                "cyl_thickness",
                params.get("cylinder_thickness", params.get("disk_thickness", 0.0)),
            )
        )

        cylinder_common_volume_active = (
            cyl_diameter > 0.0
            and cyl_thickness > 0.0
            and shape_mode != "finite_shape"
        )

        # Conventional spherical size model:
        #     G(r) = gamma_sphere(r) * G_bulk(r)
        #
        # Cylinder/disk common-volume model:
        #     shell amplitudes are multiplied by shell-dependent gamma_hkl(L)
        #     later, so no scalar gamma_r is used here.
        #
        # Finite-shape mode:
        #     finite coordination already contains size effects.
        if shape_mode == "finite_shape" or cylinder_common_volume_active:
            D = 0.0
            D_std = 0.0
            gamma_r = np.ones_like(r32, dtype=self._dtype)
        else:
            D = float(params.get("d", params.get("D", 0.0)))
            D_std = float(params.get("d_std", 0.0))
            gamma_r = self.shape_function(r32, D, D_std).astype(self._dtype, copy=False)

        # Microstrain model selection
        refcfg = (self.config.get("refinement") or {}) if isinstance(self.config, dict) else {}

        microstrain_model = str(
            params.get(
                "microstrain_model",
                refcfg.get("microstrain_model", "auto"),
            )
        ).strip().lower()

        # Wilkens parameters
        rho = float(params.get("rho", 0.0))
        Re = float(params.get("Re", params.get("re", 100.0)))
        fE = float(params.get("fE", params.get("fe", 0.5)))

        # PAH parameters
        pah_a = float(params.get("pah_a", params.get("PAH_a", 0.0)))
        pah_b = float(params.get("pah_b", params.get("PAH_b", 0.0)))

        if microstrain_model in ("wilkens", "wilkins", "dislocation"):
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = False

        elif microstrain_model in ("pah", "adler-houska", "adler_houska"):
            wilkens_active = False
            pah_active = (pah_a != 0.0 or pah_b != 0.0)

        else:
            # Backward-compatible auto behavior:
            # old inputs with rho/re still use Wilkens.
            wilkens_active = rho > 0.0 and Re > 0.0
            pah_active = (not wilkens_active) and (pah_a != 0.0 or pah_b != 0.0)

        # Fast Warren-only path.
        # Do this before norm, correlation, sigma, Gaussian accumulation,
        # qmax, pair contributions, etc.
        if return_strain_plot:
            if not (wilkens_active or pah_active):
                return {}

            max_L = kwargs.get("max_L", None)
            return self.build_warren_plot(params, max_L=max_L)


        # norm for PDF
        norm = 4.0 * np.pi * self.rho0

        # --- vectorized distances r_ij from fractional vectors ---
        M = lattice_matrix(*lattice_tuple).astype(self._dtype, copy=False)
        cart = self._shell_frac @ M.T
        r_ij = np.sqrt(np.sum(cart * cart, axis=1)).astype(self._dtype, copy=False)
        _mark("distances")

        cylinder_shell_gamma = None

        if cylinder_common_volume_active:
            cylinder_shell_gamma = self._cylinder_common_volume_shell_factor(
                params,
                lattice_tuple,
                r_ij,
                cart_vectors=cart,
            )
        # --- correlation term (vectorized) ---
        # Supported schemes:
        #  (A) Per-unordered-pair lambdas: lambda_{A}-{B}_{k}, e.g. lambda_fe-fe_0
        #  (B) Legacy global lambdas:      lambda_0, lambda_1, ...
        #  (C) Pair-specific delta1/delta2:
        #          delta1_A-B, delta2_A-B
        #      where A-B is an unordered pair label
        #  (D) Global delta1/delta2 fallback
        base_corr = np.empty_like(r_ij, dtype=self._dtype)
        
        # Global defaults
        delta1_global = float(params.get("delta1", 0.0))
        delta2_global = float(params.get("delta2", 0.0))
        
        # Legacy global lambdas: lambda_0, lambda_1, ...
        N_lambda = 0
        while f"lambda_{N_lambda}" in params:
            N_lambda += 1
        lambda_vals = [float(params.get(f"lambda_{k}", 0.0)) for k in range(N_lambda)]
        
        # Per-pair lambdas: lambda_fe-fe_0, etc.
        lam_mat = None
        lam_items = []
        
        try:
            for key, val in (params or {}).items():
                s = str(key).strip()
        
                # Preferred format:
                #   lambda_fe-fe_0
                m = re.match(r"(?i)^lambda_([A-Za-z0-9]+)-([A-Za-z0-9]+)_(\d+)$", s)
        
                # Tolerant format:
                #   lambda_fe_fe_0
                if not m:
                    m = re.match(r"(?i)^lambda_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)$", s)
        
                if not m:
                    continue
                
                a0 = lambda_safe_species_label(m.group(1))
                b0 = lambda_safe_species_label(m.group(2))
                aa, bb = sorted((a0, b0))
        
                k = int(m.group(3))
                if k < 0:
                    continue
                
                uid = self._upair_index.get((aa, bb), None)
                if uid is None:
                    continue
                
                lam_items.append((uid, k, float(val)))
        
            if lam_items:
                max_k = max(k for (_, k, _) in lam_items) + 1
                lam_mat = np.full(
                    (len(self._upair_list), max_k),
                    np.nan,
                    dtype=self._dtype,
                )
        
                for uid, k, v in lam_items:
                    lam_mat[int(uid), int(k)] = self._dtype(v)
        
        except Exception:
            lam_mat = None
        
        # Pair-specific delta1/delta2
        pair_delta1 = np.full(len(self._upair_list), np.nan, dtype=self._dtype)
        pair_delta2 = np.full(len(self._upair_list), np.nan, dtype=self._dtype)
        
        try:
            for uid, (aa, bb) in enumerate(self._upair_list):
                # preferred key
                k1 = f"delta1_{aa}-{bb}"
                k2 = f"delta2_{aa}-{bb}"
        
                # tolerant underscore key
                k1b = f"delta1_{aa}_{bb}"
                k2b = f"delta2_{aa}_{bb}"
        
                if k1 in params:
                    pair_delta1[uid] = self._dtype(float(params[k1]))
                elif k1b in params:
                    pair_delta1[uid] = self._dtype(float(params[k1b]))
        
                if k2 in params:
                    pair_delta2[uid] = self._dtype(float(params[k2]))
                elif k2b in params:
                    pair_delta2[uid] = self._dtype(float(params[k2b]))
        except Exception:
            pass
        
        # Apply per-pair lambdas if present
        m_rem = np.ones(r_ij.shape[0], dtype=bool)
        
        if lam_mat is not None:
            uid = self._shell_upair_id_i64
            k_in = self._shell_k_within_upair_i64
        
            lam_val = np.full(r_ij.shape[0], np.nan, dtype=self._dtype)
        
            valid = k_in < lam_mat.shape[1]
            if np.any(valid):
                lam_val[valid] = lam_mat[uid[valid], k_in[valid]]
        
            m_pair = ~np.isnan(lam_val)
        
            if np.any(m_pair):
                base_corr[m_pair] = self._dtype(1.0) - lam_val[m_pair]
        
            m_rem = ~m_pair
        
        # Remaining shells:
        #   first try legacy global lambda_0/lambda_1 if available,
        #   otherwise use pair-specific delta1/delta2 if present,
        #   otherwise global delta1/delta2.
        if np.any(m_rem):
            k_rank = self._shell_k_rank_i64
        
            # First global lambda_k if present
            if N_lambda > 0:
                lam = np.asarray(lambda_vals, dtype=self._dtype)
        
                m_glob = m_rem & (k_rank < N_lambda)
        
                if np.any(m_glob):
                    base_corr[m_glob] = self._dtype(1.0) - lam[k_rank[m_glob]]
        
                m_delta = m_rem & ~m_glob
            else:
                m_delta = m_rem
        
            if np.any(m_delta):
                uid = self._shell_upair_id_i64[m_delta]
                rm = r_ij[m_delta]
        
                d1_use = np.full(rm.shape, self._dtype(delta1_global), dtype=self._dtype)
                d2_use = np.full(rm.shape, self._dtype(delta2_global), dtype=self._dtype)
        
                # Override with pair-specific values where present
                d1_pair = pair_delta1[uid]
                d2_pair = pair_delta2[uid]
        
                m_d1 = ~np.isnan(d1_pair)
                m_d2 = ~np.isnan(d2_pair)
        
                if np.any(m_d1):
                    d1_use[m_d1] = d1_pair[m_d1]
                if np.any(m_d2):
                    d2_use[m_d2] = d2_pair[m_d2]
        
                base_corr[m_delta] = (
                    self._dtype(1.0)
                    - d1_use / np.maximum(rm, self._dtype(1e-12))
                    - d2_use / np.maximum(rm * rm, self._dtype(1e-24))
                )

        corr = np.maximum(base_corr, self._dtype(0.0))
        _mark("lambda+delta_corr")                
        # --- Biso average per shell (vectorized) ---
        biso_avg = (self._dtype(0.5) * (biso_vals_vec[self._shell_alpha_id] + biso_vals_vec[self._shell_beta_id])).astype(self._dtype, copy=False)

            # --- Wilkens microstrain term (vectorized by unique hkl) ---
            # --- Wilkens microstrain term (vectorized by unique crystallographic direction) ---
        #
        # Important:
        # Use the reduced crystallographic direction self._shell_direction, not self._shell_hkl.
        #
        # Reason:
        # For a Warren/microstrain direction [hkl], all shells along that direction should use
        # the same C_hkl, independent of:
        #   - chemical pair type, e.g. Ca-O, O-O, Ca-Ca
        #   - order of the vector, e.g. [110], [220], [330]
        #
        # The L-dependence comes only from fstar(L/Re) and L^2.
        strain_L2 = np.zeros_like(r_ij, dtype=self._dtype)

        if wilkens_active or pah_active:
            uniq_dirs = getattr(self, "_strain_unique_dirs", None)
            inv_dirs = getattr(self, "_strain_inv_dirs", None)

            if uniq_dirs is None or inv_dirs is None:
                dirs_i32 = self._shell_direction.astype(np.int32, copy=False)
                uniq_dirs, inv_dirs = np.unique(
                    dirs_i32,
                    axis=0,
                    return_inverse=True,
                )
                uniq_dirs = uniq_dirs.astype(np.int32, copy=False)
                inv_dirs = inv_dirs.astype(np.int32, copy=False)

            # This returns the invariant directional factor.
            # For invariant coefficients, this already includes:
            #     (d_hkl^4 / a^4) * Gamma_hkl
            Chkl_u = self._compute_chkl_unique_dirs_fast(params, float(fE))
            strainINV = Chkl_u[inv_dirs].astype(self._dtype, copy=False)

            if wilkens_active:
                # Burgers magnitude
                params_lc = self._params_lc(params)
                bmag = self._pfloat(
                    params_lc,
                    ("burgers_mag", "burgers", "b_mag", "b"),
                    0.0,
                )

                if bmag <= 0.0:
                    # Backward-compatible default.
                    bmag = np.sqrt(3.0) / 2.0 * float(lattice_tuple[0])

                fstar_vals = fstar_vec(
                    r_ij.astype(float),
                    Re,
                ).astype(self._dtype, copy=False)

                eps2 = (
                    self._dtype(rho)
                    * (self._dtype(bmag) ** 2)
                    / self._dtype(4.0 * np.pi)
                ) * strainINV * fstar_vals

                strain_L2 = (r_ij * r_ij) * eps2

                if not getattr(self, "_printed_wilkens_diag", False):
                    self._printed_wilkens_diag = True

                    def _safe_stats(name, arr):
                        arr = np.asarray(arr, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size == 0:
                            print(f"[WILKENS DIAG] {name}: no finite values")
                            return
                        print(
                            f"[WILKENS DIAG] {name}: "
                            f"min={np.min(arr):.6g}, "
                            f"max={np.max(arr):.6g}, "
                            f"mean={np.mean(arr):.6g}, "
                            f"median={np.median(arr):.6g}"
                        )

                    print("[WILKENS DIAG] ----")
                    print(f"[WILKENS DIAG] rho={rho}, Re={Re}, fE={fE}, bmag={bmag}")
                    print(f"[WILKENS DIAG] unique Chkl dirs={len(Chkl_u)}, shell rows={len(r_ij)}")
                    _safe_stats("strainINV / Chkl", strainINV)
                    _safe_stats("fstar", fstar_vals)
                    _safe_stats("eps2", eps2)
                    _safe_stats("strain_L2", strain_L2)
                    print("[WILKENS DIAG] ----")

            elif pah_active:
                # PAH equation 10:
                #
                #   <eps^2_hkl(L)> =
                #       (d_hkl^4 / a^4) Gamma_hkl * (pah_a / L + pah_b)
                #
                # The PDF Gaussian peak variance uses:
                #
                #   sigma_strain^2 = L^2 * <eps^2_hkl(L)>
                #
                # Therefore:
                #
                #   sigma_PAH^2 =
                #       (d_hkl^4 / a^4) Gamma_hkl * (pah_a * L + pah_b * L^2)
                #
                strain_L2 = strainINV * (
                    self._dtype(pah_a) * r_ij
                    + self._dtype(pah_b) * r_ij * r_ij
                )

                # Avoid negative variance if refinement temporarily makes PAH term negative.
                strain_L2 = np.maximum(strain_L2, self._dtype(0.0))

                if not getattr(self, "_printed_pah_diag", False):
                    self._printed_pah_diag = True

                    def _safe_stats(name, arr):
                        arr = np.asarray(arr, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size == 0:
                            print(f"[PAH DIAG] {name}: no finite values")
                            return
                        print(
                            f"[PAH DIAG] {name}: "
                            f"min={np.min(arr):.6g}, "
                            f"max={np.max(arr):.6g}, "
                            f"mean={np.mean(arr):.6g}, "
                            f"median={np.median(arr):.6g}"
                        )

                    print("[PAH DIAG] ----")
                    print(f"[PAH DIAG] pah_a={pah_a}, pah_b={pah_b}, fE={fE}")
                    print(f"[PAH DIAG] unique invariant dirs={len(Chkl_u)}, shell rows={len(r_ij)}")
                    _safe_stats("strainINV", strainINV)
                    _safe_stats("strain_L2", strain_L2)
                    print("[PAH DIAG] ----")

        _mark("wilkens_strain")
        sigma2 = (
            self._dtype(1.0 / (4.0 * np.pi ** 2)) * biso_avg * corr
            + (self._dtype(delta_g) * r_ij) ** 2
            + (self._dtype(delta_broad) * r_ij) ** 2
            + strain_L2
        )
        sigma = np.sqrt(np.maximum(sigma2, self._dtype(1e-24))).astype(self._dtype, copy=False)
        _mark("sigma")

        # --- normalization A_ij (vectorized) ---
        rho_beta = densities_vec[self._shell_beta_id]
        # Use alpha-site-averaged multiplicity for correct normalization.
        # For simple structures this is equal to shell_mult.
        # For multi-site species, e.g. Li4GeS4 with Li1/Li2/Li3 and S1/S2/S3,
        # this prevents overcounting inequivalent alpha sites.
        shell_mult_norm = getattr(self, "_shell_mult_eff", None)

        if shell_mult_norm is None:
            shell_mult_norm = self._shell_mult.astype(self._dtype, copy=False)
        else:
            shell_mult_norm = np.asarray(shell_mult_norm, dtype=self._dtype)

        A_ij = shell_mult_norm / (
            self._dtype(4.0 * np.pi)
            * np.maximum(r_ij * r_ij, self._dtype(1e-24))
            * np.maximum(rho_beta.astype(self._dtype, copy=False), self._dtype(1e-30))
        )

        if cylinder_shell_gamma is not None:
            A_ij = A_ij * np.asarray(
                cylinder_shell_gamma,
                dtype=self._dtype,
            )

        # --- accumulate partial g_ab(r) with Numba (parallel) ---
        # --- accumulate partial g_ab(r) with Numba (parallel; shell-block kernel) ---
        g_by_pair = np.zeros((self._n_pairs, r32.size), dtype=self._dtype)
        if get_num_threads is None:
            # Fallback to pair-parallel kernel if Numba threading API unavailable
            accumulate_gaussians_by_pair(r32, r_ij, sigma, A_ij, self._pair_offsets, g_by_pair)
        else:
            nblk = int(get_num_threads())
            if nblk < 1:
                nblk = 1
            partial = np.zeros((nblk, self._n_pairs, r32.size), dtype=self._dtype)
            accumulate_gaussians_by_shell_blocks(r32, r_ij, sigma, A_ij, self._shell_pair_id, partial)
            g_by_pair[:] = partial.sum(axis=0)

        _mark("gaussian_accumulation")
        # --- Weighted total g(r) (single dot-product over ordered pairs) ---
        g_weighted = (
            self._w_ordered.astype(self._dtype)[:, None]
            * g_by_pair
        ).sum(axis=0).astype(self._dtype, copy=False)

        gamma_by_pair_finite = None
        gamma_weighted_finite = None

        gamma_by_pair_cylinder = None
        gamma_weighted_cylinder = None

        gamma_base = None

        # ---------------------------------------------------------
        # Construct the positive peak term and the mandatory
        # negative PDF baseline separately.
        #
        # The mandatory baseline implements:
        #
        #     G(r) = 4*pi*rho0*r*[g(r) - gamma(r)]
        #
        # and must not depend on whether the optional SAXS model
        # is enabled.
        # ---------------------------------------------------------
        if shape_mode == "finite_shape":
            gamma_by_pair_finite = self._finite_gamma_by_pair_on_grid(
                r32,
                r_ij,
            )

            gamma_weighted_finite = (
                self._w_ordered.astype(self._dtype)[:, None]
                * gamma_by_pair_finite
            ).sum(axis=0).astype(
                self._dtype,
                copy=False,
            )

            gamma_base = gamma_weighted_finite

            G_peak32 = (
                self._dtype(scale)
                * self._dtype(norm)
                * r32
                * g_weighted
            )

        elif cylinder_shell_gamma is not None:
            gamma_weights = getattr(
                self,
                "_shell_bulk_mult_eff",
                None,
            )

            if gamma_weights is None:
                gamma_weights = getattr(
                    self,
                    "_shell_mult_eff",
                    None,
                )

            if gamma_weights is None:
                gamma_weights = self._shell_mult.astype(
                    self._dtype,
                    copy=False,
                )

            gamma_by_pair_cylinder = self._shell_gamma_by_pair_on_grid(
                r32,
                r_ij,
                cylinder_shell_gamma,
                weights=np.asarray(
                    gamma_weights,
                    dtype=float,
                ),
            )

            gamma_weighted_cylinder = (
                self._w_ordered.astype(self._dtype)[:, None]
                * gamma_by_pair_cylinder
            ).sum(axis=0).astype(
                self._dtype,
                copy=False,
            )

            gamma_base = gamma_weighted_cylinder

            G_peak32 = (
                self._dtype(scale)
                * self._dtype(norm)
                * r32
                * g_weighted
            )

        else:
            gamma_base = gamma_r.astype(
                self._dtype,
                copy=False,
            )

            G_peak32 = (
                self._dtype(scale)
                * self._dtype(norm)
                * r32
                * gamma_r
                * g_weighted
            )

        gamma_base = np.asarray(
            gamma_base,
            dtype=self._dtype,
        )

        saxs_is_enabled = self._saxs_enabled(
            params
        )

        # ---------------------------------------------------------
        # Two selectable baseline modes
        #
        # SAXS disabled:
        #
        #     Corrected conventional PDF:
        #
        #     G = scale * 4*pi*rho0*r*gamma_base*(g - 1)
        #
        #     The complete structural PDF receives qdamp.
        #
        # SAXS enabled:
        #
        #     Original SAXS/missing-low-Q belly model:
        #
        #     G = G_peak + G_belly
        #
        #     G_belly =
        #         -scale*saxs_scale*4*pi*rho0*r*gamma_saxs
        #
        #     The peak contribution always receives qdamp.
        #     The belly receives qdamp only when
        #     saxs_apply_qdamp = true.
        # ---------------------------------------------------------
        if saxs_is_enabled:
            gamma_saxs = self._compute_saxs_gamma_weighted(
                params,
                r32,
                r_ij,
                lattice_tuple,
                cart,
                fallback_gamma=gamma_base,
            )

            gamma_saxs = np.asarray(
                gamma_saxs,
                dtype=self._dtype,
            )

            saxs_scale = self._dtype(
                self._saxs_scale(params)
            )

            # Original SAXS belly formulation.
            G_belly32 = (
                -self._dtype(scale)
                * self._dtype(norm)
                * r32
                * saxs_scale
                * gamma_saxs
            )

            # Original damping behavior:
            # qdamp always acts on the structural peaks.
            if qdamp > 0.0:
                damp = np.exp(
                    -0.5
                    * (self._dtype(qdamp) ** 2)
                    * (r32 ** 2)
                ).astype(
                    self._dtype,
                    copy=False,
                )

                G_peak32 *= damp

                if self._saxs_apply_qdamp(params):
                    G_belly32 *= damp

            G_r32 = (
                G_peak32
                + G_belly32
            )

        else:
            # Corrected conventional PDF baseline.
            G_baseline32 = (
                -self._dtype(scale)
                * self._dtype(norm)
                * r32
                * gamma_base
            )

            G_structural32 = (
                G_peak32
                + G_baseline32
            )

            # In conventional mode qdamp acts on the complete
            # structural PDF, including the mandatory baseline.
            if qdamp > 0.0:
                damp = np.exp(
                    -0.5
                    * (self._dtype(qdamp) ** 2)
                    * (r32 ** 2)
                ).astype(
                    self._dtype,
                    copy=False,
                )

                G_structural32 *= damp

            G_r32 = G_structural32

        # Partial outputs only if requested.
        #
        # G_ab:
        #   Raw/unweighted pair PDFs for inspection.
        #
        # G_contrib:
        #   Weighted additive pair contributions.
        #   These are the curves you want for plotting selected pairs because:
        #
        #       G_total(r) ≈ sum_pair G_contrib_pair(r)
        #
        G_by_pair = {}
        G_contrib = {}

        if return_gr or return_contributions:
            tmp_raw = {}
            tmp_contrib = {}

            for pid, (a_sp, b_sp) in enumerate(self._pair_list):
                key = (a_sp, b_sp) if a_sp <= b_sp else (b_sp, a_sp)
                key_str = f"{key[0]}-{key[1]}"

                g_pid = g_by_pair[pid].astype(self._dtype, copy=False)
                w_pid = self._dtype(self._w_ordered[pid])

                # Raw partial PDF, not weighted.
                if shape_mode == "finite_shape":
                    if gamma_by_pair_finite is None:
                        gamma_pid = np.zeros_like(g_pid, dtype=self._dtype)
                    else:
                        gamma_pid = gamma_by_pair_finite[pid].astype(self._dtype, copy=False)

                    raw_curve = (
                        self._dtype(scale)
                        * gamma_r
                        * self._dtype(norm)
                        * r32
                        * (g_pid - gamma_pid)
                    )

                    contrib_curve = (
                        self._dtype(scale)
                        * gamma_r
                        * self._dtype(norm)
                        * r32
                        * w_pid
                        * (g_pid - gamma_pid)
                    )

                elif cylinder_shell_gamma is not None:
                    if gamma_by_pair_cylinder is None:
                        gamma_pid = np.zeros_like(g_pid, dtype=self._dtype)
                    else:
                        gamma_pid = gamma_by_pair_cylinder[pid].astype(self._dtype, copy=False)

                    raw_curve = (
                        self._dtype(scale)
                        * self._dtype(norm)
                        * r32
                        * (g_pid - gamma_pid)
                    )

                    contrib_curve = (
                        self._dtype(scale)
                        * self._dtype(norm)
                        * r32
                        * w_pid
                        * (g_pid - gamma_pid)
                    )

                else:
                    raw_curve = (
                        self._dtype(scale)
                        * gamma_r
                        * self._dtype(norm)
                        * r32
                        * (g_pid - self._dtype(1.0))
                    )

                    contrib_curve = (
                        self._dtype(scale)
                        * gamma_r
                        * self._dtype(norm)
                        * r32
                        * w_pid
                        * (g_pid - self._dtype(1.0))
                    )

                tmp_raw.setdefault(key_str, []).append(raw_curve)
                tmp_contrib.setdefault(key_str, []).append(contrib_curve)

            for key_str, arrs in tmp_raw.items():
                if len(arrs) == 2:
                    G_by_pair[key_str] = 0.5 * (arrs[0] + arrs[1])
                else:
                    G_by_pair[key_str] = arrs[0]

            for key_str, arrs in tmp_contrib.items():
                # Contributions should be summed, not averaged.
                if len(arrs) == 2:
                    G_contrib[key_str] = arrs[0] + arrs[1]
                else:
                    G_contrib[key_str] = arrs[0]

        # qdamp
        if qdamp > 0.0:
            damp = np.exp(-0.5 * (self._dtype(qdamp) ** 2) * (r32 ** 2)).astype(self._dtype, copy=False)


            for key in list(G_by_pair.keys()):
                G_by_pair[key] *= damp

            for key in list(G_contrib.keys()):
                G_contrib[key] *= damp

        if qmax > 0.0 and r32.size > 1:
            r_float = np.asarray(r, dtype=float)
            dr = float(np.median(np.diff(r_float)))

            if not getattr(self, "_printed_qmax_debug", False):
                half_width = int(np.ceil(float(qmax_zeros) * np.pi / (qmax * dr)))
                self._printed_qmax_debug = True

            G_before_qmax = np.asarray(G_r32, dtype=float).copy()

            G_r32 = apply_qmax_termination(
                G_before_qmax,
                r_float,
                qmax,
                n_zeros=qmax_zeros,
                pad_mode=qmax_pad,
                use_sine_kernel=True,
                normalize=False,
            ).astype(self._dtype)

            # Debug: tells us whether qmax is actually changing the curve.
            if not getattr(self, "_printed_qmax_delta", False):
                diff = np.asarray(G_r32, dtype=float) - G_before_qmax
                print(
                    f"[QMAX] max|ΔG|={np.nanmax(np.abs(diff)):.6g}, "
                    f"rmsΔG={np.sqrt(np.nanmean(diff * diff)):.6g}"
                )
                self._printed_qmax_delta = True

            for key in list(G_by_pair.keys()):
                G_by_pair[key] = apply_qmax_termination(
                    np.asarray(G_by_pair[key], dtype=float),
                    r_float,
                    qmax,
                    n_zeros=qmax_zeros,
                    pad_mode=qmax_pad,
                    use_sine_kernel=True,
                    normalize=False,
                ).astype(self._dtype)

            for key in list(G_contrib.keys()):
                G_contrib[key] = apply_qmax_termination(
                    np.asarray(G_contrib[key], dtype=float),
                    r_float,
                    qmax,
                    n_zeros=qmax_zeros,
                    pad_mode=qmax_pad,
                    use_sine_kernel=True,
                    normalize=False,
                ).astype(self._dtype)

        # ---------------------------------------------------------
        # Return from the internal Qmax grid to the requested
        # experimental/refinement grid.
        # ---------------------------------------------------------
        if use_internal_qmax_grid:
            r_internal = np.asarray(
                r,
                dtype=float,
            )

            r_output = np.asarray(
                r_requested,
                dtype=float,
            )

            G_r32 = np.interp(
                r_output,
                r_internal,
                np.asarray(
                    G_r32,
                    dtype=float,
                ),
            ).astype(
                self._dtype,
                copy=False,
            )

            g_by_pair_output = np.zeros(
                (
                    self._n_pairs,
                    r_output.size,
                ),
                dtype=self._dtype,
            )

            for pid in range(self._n_pairs):
                g_by_pair_output[pid] = np.interp(
                    r_output,
                    r_internal,
                    np.asarray(
                        g_by_pair[pid],
                        dtype=float,
                    ),
                ).astype(
                    self._dtype,
                    copy=False,
                )

            g_by_pair = g_by_pair_output

            g_weighted = np.interp(
                r_output,
                r_internal,
                np.asarray(
                    g_weighted,
                    dtype=float,
                ),
            ).astype(
                self._dtype,
                copy=False,
            )

            for key in list(
                G_by_pair.keys()
            ):
                G_by_pair[key] = np.interp(
                    r_output,
                    r_internal,
                    np.asarray(
                        G_by_pair[key],
                        dtype=float,
                    ),
                ).astype(
                    self._dtype,
                    copy=False,
                )

            for key in list(
                G_contrib.keys()
            ):
                G_contrib[key] = np.interp(
                    r_output,
                    r_internal,
                    np.asarray(
                        G_contrib[key],
                        dtype=float,
                    ),
                ).astype(
                    self._dtype,
                    copy=False,
                )

            r = r_output
            r32 = np.asarray(
                r_output,
                dtype=self._dtype,
            )

        _mark("qmax+damping")
        # Warren plot output
        if return_strain_plot and wilkens_active:
            strain_by_direction = {}

            # -------------------------------------------------------------
            # Canonical Warren direction families.
            #
            # Important:
            # All atom pairs belonging to the same direction family must lie
            # on ONE Warren/microstrain curve.  The chemical pair type must
            # not create separate strain values.
            # -------------------------------------------------------------
            from math import gcd

            def _reduce_dir(h: int, k: int, l: int) -> tuple[int, int, int]:
                h, k, l = int(h), int(k), int(l)
                if h == 0 and k == 0 and l == 0:
                    return (0, 0, 0)

                g = gcd(gcd(abs(h), abs(k)), abs(l))
                if g > 0:
                    h //= g
                    k //= g
                    l //= g

                # make opposite directions equivalent
                for v in (h, k, l):
                    if v != 0:
                        if v < 0:
                            h, k, l = -h, -k, -l
                        break

                return (h, k, l)

            try:
                crystal_system = SpacegroupAnalyzer(
                    self.structure_handler.structure,
                    symprec=1e-3,
                ).get_crystal_system().lower()
            except Exception:
                crystal_system = ""

            def _canonical_family(h: int, k: int, l: int) -> tuple[int, int, int]:
                h, k, l = _reduce_dir(h, k, l)

                if (h, k, l) == (0, 0, 0):
                    return (0, 0, 0)

                # Cubic: full permutation/sign family, e.g. [110], [101], [011]
                if crystal_system == "cubic":
                    vals = sorted((abs(h), abs(k), abs(l)), reverse=True)
                    return (int(vals[0]), int(vals[1]), int(vals[2]))

                # Tetragonal / hexagonal / trigonal:
                # a,b are equivalent in-plane; c is special
                if crystal_system in ("tetragonal", "hexagonal", "trigonal"):
                    ab = sorted((abs(h), abs(k)), reverse=True)
                    return (int(ab[0]), int(ab[1]), int(abs(l)))

                # Lower symmetry: only opposite directions are equivalent
                return (h, k, l)

            # Burgers magnitude
            params_lc = self._params_lc(params)
            bmag = self._pfloat(
                params_lc,
                ("burgers_mag", "burgers", "b_mag", "b"),
                0.0,
            )

            if bmag <= 0.0:
                # Backward-compatible default. For non-cubic/non-bcc systems,
                # provide burgers_mag as a normal parameter in [initial_values].
                bmag = np.sqrt(3.0) / 2.0 * float(lattice_tuple[0])

            pref = float(rho) * (float(bmag) ** 2) / (4.0 * np.pi)

            dirs = self._shell_direction.astype(np.int32)
            L_all = np.asarray(r_ij, dtype=float)

            # Collect unique distances for each canonical direction family.
            # This removes duplicate chemical-pair contributions at the same L.
            family_to_L = defaultdict(set)

            for dvec, L in zip(dirs, L_all):
                if not np.isfinite(L) or L <= 0.5:
                    continue

                h, k, l = int(dvec[0]), int(dvec[1]), int(dvec[2])
                fam = _canonical_family(h, k, l)

                if fam == (0, 0, 0):
                    continue

                family_to_L[fam].add(round(float(L), 5))

            for fam, L_set in family_to_L.items():
                if not L_set:
                    continue

                h, k, l = fam
                L_sorted = np.asarray(sorted(L_set), dtype=float)

                # ONE contrast factor per direction family.
                cval = self._contrast_factor_one_from_params(
                    int(h),
                    int(k),
                    int(l),
                    params,
                    float(fE),
                )

                if not np.isfinite(cval):
                    cval = 0.0
                cval = max(float(cval), 0.0)

                # Warren relation:
                # <Delta L^2>(L) = L^2 * rho*b^2/(4*pi) * C_hkl * f*(L/Re)
                dL2 = (
                    L_sorted * L_sorted
                    * pref
                    * cval
                    * fstar_vec(L_sorted, float(Re))
                )

                dL2 = np.maximum(np.asarray(dL2, dtype=float), 0.0)

                strain_by_direction[fam] = [
                    (float(L), float(v)) for L, v in zip(L_sorted, dL2)
                ]

            return strain_by_direction

        # --- Normal return (dict expected by main.py / fit_engine.py) ---
        # Ordered partials (alpha-beta) for debugging / output
        g_species = {f"{a}-{b}": g_by_pair[pid].copy() for pid, (a, b) in enumerate(self._pair_list)}

        # Unordered weights dict (alpha<=beta) for reporting
        weights = {f"{a}-{b}": w for (a, b), w in self._w_unordered.items()}

        _mark("prepare_output")

        if _prof and (self._profile_eval_n % max(1, int(self.profile_pdf_every)) == 0):
            total = time.perf_counter() - _prof_t0
            msg = " | ".join([f"{name}={dt:.3f}s" for name, dt in _prof_parts])
            print(f"[PDF PROFILE] eval={self._profile_eval_n} total={total:.3f}s | {msg}")

        # ---------------------------------------------------------
        # Optional plotted shape/SAXS term
        #
        # This calculation can be expensive for cylinder/disk models
        # because it builds another pair-resolved gamma histogram.
        # Do it only when explicitly requested by the caller.
        # ---------------------------------------------------------
        if compute_gamma_avg:
            gamma_avg = self.compute_isotropic_shape_factor(
                params,
                r,
            )
        else:
            gamma_avg = None

        out = {
            "r": np.asarray(r, dtype=float),
            "G_r": np.asarray(G_r32, dtype=float),
            "g_r": np.asarray(g_weighted, dtype=float),
            "g_ab": g_species,
            "G_ab": {
                k: np.asarray(v, dtype=float)
                for k, v in G_by_pair.items()
            },
            "G_contrib": {
                k: np.asarray(v, dtype=float)
                for k, v in G_contrib.items()
            },
            "weights": weights,
            "gamma_avg": (
                None
                if gamma_avg is None
                else np.asarray(gamma_avg, dtype=float)
            ),
        }

        if not return_jac:
            # New behavior: if return_contributions=True, return the full dict.
            if return_contributions:
                return out

            # Old behavior preserved for code that expects the tuple.
            if return_gr:
                return out["r"], out["G_r"], out["g_r"], g_species, G_by_pair, weights

            return out

        # --- Jacobian path (fit_engine expects (G_r, jac_dict)) ---
        # Provide zero arrays unless you implement analytic derivatives.
        jac_dict = {}
        refinable = self.config.get("refinable", {})
        for p, is_ref in (refinable or {}).items():
            if is_ref:
                jac_dict[p] = np.zeros_like(out["G_r"], dtype=float)

        return out["G_r"], jac_dict

