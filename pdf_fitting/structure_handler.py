# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict
import os, hashlib, json, time
import re
from concurrent.futures import ThreadPoolExecutor


try:
    import numba
    from numba import njit, prange, get_num_threads, set_num_threads, threading_layer
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    numba = None
    njit = None
    prange = range
    get_num_threads = None
    set_num_threads = None
    threading_layer = None

from pdf_fitting.user_paths import user_cache_dir

CACHE_DIR = user_cache_dir(
    "shell_cache"
)

CACHE_VERSION = "shell_cache_hybrid_v16_adp"

# Occupancy arrays stored in the shell table and its NPZ cache.
#   site_occupancy         (n_sites,) CIF total occupancy per site
#   site_species_id        (n_sites,) element id (index into unique_species)
#   site_occ_group         (n_sites,) occupancy group (index into occ_group_keys)
#   occ_term_shell         (n_terms,) shell index after final shell ordering
#   occ_term_center_group  (n_terms,) occupancy group of the center site
#   occ_term_neighbor_group(n_terms,) occupancy group of the neighbor site
#   occ_term_weight        (n_terms,) summed center-orbit sizes
_OCCUPANCY_TABLE_ARRAYS = (
    "site_occupancy",
    "site_species_id",
    "site_occ_group",
    "occ_term_shell",
    "occ_term_center_group",
    "occ_term_neighbor_group",
    "occ_term_weight",
    # Ungrouped rows of movable sites (coordinate refinement):
    #   dyn_shell          (n_dyn,)   shell index (one row per shell)
    #   dyn_center_site    (n_dyn,)   center site index
    #   dyn_neighbor_site  (n_dyn,)   neighbor site index
    #   dyn_base_fvec      (n_dyn, 3) unshifted fractional pair vector
    "dyn_shell",
    "dyn_center_site",
    "dyn_neighbor_site",
    "dyn_base_fvec",
    # Anisotropic displacement (beta) sites: per shell, the (ASU site group,
    # rotation) entry of its center / neighbor atom (-1 = isotropic), and
    # the entry table (rotation R maps the ASU site onto that atom).
    "shell_adp_center",
    "shell_adp_neighbor",
    "adp_rotations",
    "adp_entry_group",
)

ADP_COMPONENTS = ("11", "22", "33", "12", "13", "23")


def adp_site_rotations(p0, ops):
    """Rotations of the site-symmetry group of a site at fractional p0."""
    out = []
    for R, t in ops:
        if same_mod1(np.asarray(R) @ np.asarray(p0) + np.asarray(t), p0):
            if not any(np.allclose(R, Q) for Q in out):
                out.append(np.asarray(R, dtype=np.float64))
    return out or [np.eye(3)]


def beta_matrix(b6) -> np.ndarray:
    b11, b22, b33, b12, b13, b23 = (float(x) for x in b6)
    return np.array([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]], dtype=np.float64)


def beta_vector(B) -> np.ndarray:
    return np.array([B[0, 0], B[1, 1], B[2, 2], B[0, 1], B[0, 2], B[1, 2]], dtype=np.float64)


def symmetrize_beta(b6, site_rotations) -> np.ndarray:
    """β → mean over the site-symmetry group of R β Rᵀ (the allowed part)."""
    B = beta_matrix(b6)
    acc = np.zeros((3, 3))
    for R in site_rotations:
        acc += R @ B @ R.T
    return beta_vector(acc / len(site_rotations))


def beta_constraints(site_rotations):
    """
    For a site: ("free" | "tied" | "fixed", expression) per β component.
    Free components are chosen by pivoting the symmetrisation projector;
    tied ones are linear combinations of them; fixed ones are 0.
    """
    P = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        P[:, j] = symmetrize_beta(e, site_rotations)
    # Column-pivoted basis of range(P).
    free: list = []
    basis = []
    for j in range(6):
        v = P[:, j]
        if np.linalg.norm(v) < 1e-9:
            continue
        M = np.column_stack(basis + [v]) if basis else v[:, None]
        if np.linalg.matrix_rank(M, tol=1e-8) > len(basis):
            basis.append(v)
            free.append(j)
    out = {}
    if not free:
        return {c: ("fixed", 0.0) for c in ADP_COMPONENTS}
    Bf = np.column_stack([np.eye(6)[:, j] for j in free])
    # β_sym = P β; with only free components set: β = P[:, free] c → solve for every component.
    Pf = P[:, free]
    # Express every component as a combination of the free components' values:
    # β_free = Pf[free] c  →  c = inv(Pf[free]) β_free ;  β_i = Pf[i] c.
    T = Pf @ np.linalg.inv(Pf[free, :])
    for i, comp in enumerate(ADP_COMPONENTS):
        if i in free:
            out[comp] = ("free", None)
        elif np.allclose(T[i], 0.0, atol=1e-9):
            out[comp] = ("fixed", 0.0)
        else:
            terms = [(ADP_COMPONENTS[free[k]], float(T[i, k])) for k in range(len(free)) if abs(T[i, k]) > 1e-9]
            out[comp] = ("tied", terms)
    del Bf
    return out

def clean_element_name(name: str) -> str:
    """Return clean chemical element symbol, e.g. Fe0+ -> Fe."""
    m = re.match(r"([A-Za-z]{1,2})", str(name))
    if not m:
        raise ValueError(f"Could not parse chemical symbol from '{name}'")
    sym = m.group(1)
    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()

def site_element(site) -> str:
    """
    Return the element symbol for a pymatgen site.

    Works for both:
      ordered site:          Zr
      partial site:          N0.575
      mixed site:            C0.5 O0.5

    For mixed sites, this returns the dominant species. Your PCN-224 CIF
    has partial single-element sites, so this is suitable there.
    """
    if site.is_ordered:
        return clean_element_name(site.specie.symbol)

    specie, _occ = max(
        site.species.items(),
        key=lambda item: float(item[1]),
    )

    return clean_element_name(specie.symbol)


def site_cif_occupancy(site) -> float:
    """
    Return total crystallographic occupancy of a pymatgen site.

    Normal atom: 1.0
    N1 in your CIF: 0.575
    N2 in your CIF: 0.425
    """
    return float(sum(float(occ) for occ in site.species.values()))

def site_element_symbol(site) -> str:
    """
    Return the chemical element symbol for an ordered or partially occupied
    pymatgen PeriodicSite.

    Works for:
        Zr
        N0.575
        C0.425

    For a mixed-species site, the species with largest occupancy is used only
    as the element label. The actual occupancy is handled separately.
    """
    species_items = list(site.species.items())

    if not species_items:
        raise ValueError("Site has no species.")

    specie, _occupancy = max(
        species_items,
        key=lambda item: float(item[1]),
    )

    return clean_element_name(specie.symbol)


def site_total_occupancy(site) -> float:
    """
    Total crystallographic occupancy of a site.

    Examples:
        Zr site: 1.0
        N1 site: 0.575
        N2 site: 0.425
    """
    return float(
        sum(float(occupancy) for occupancy in site.species.values())
    )


def site_display_label(site) -> str:
    """
    Safe display string for ordered and disordered sites.
    """
    return str(site.species)


def site_label(site, index: int) -> str:
    """
    Obtain a stable site label for occupancy refinement.

    Order of preference:
        1. CIF label stored as a site property (fast P1 CIF reader stores
           it as "cif_label");
        2. label explicitly set on the pymatgen site (CifParser sets the
           CIF label on every symmetry-equivalent copy);
        3. site_<index>.

    pymatgen's public ``site.label`` falls back to the species string
    (e.g. "N:0.575") when no label was set, so the private ``_label`` is
    checked instead to distinguish a real label from that fallback.
    """
    properties = getattr(site, "properties", {}) or {}

    for key in (
        "cif_label",
        "label",
        "site_label",
        "_atom_site_label",
        "atom_site_label",
    ):
        value = properties.get(key, None)

        if value is not None and str(value).strip():
            return str(value).strip()

    label = getattr(site, "_label", None)

    if label is not None and str(label).strip():
        return str(label).strip()

    return f"site_{index + 1}"


def safe_param_key(label: str) -> str:
    """
    Convert a site label to the parameter-key form used by the GUI.

    Must match SiteParamTableWidget._safe_key:
        "N1"    -> "n1"
        "O1'"   -> "o1"
        "site_3" -> "site3"
    """
    return re.sub(
        r"[^A-Za-z0-9]+",
        "",
        str(label).strip(),
    ).lower()


def _build_occupancy_terms(
    row_shell: np.ndarray,
    row_center_group: np.ndarray,
    row_neighbor_group: np.ndarray,
    row_center_orbit: np.ndarray,
    n_groups: int,
):
    """
    Compress neighbor rows into occupancy terms.

    Every neighbor row contributes to its shell with weight

        orbit_size(center) * occ(center group) * occ(neighbor group)

    Rows sharing (shell, center group, neighbor group) are summed, so the
    shell multiplicity for any set of occupancies is

        sum_terms weight * occ[cg] * occ[ng]

    Returns (term_shell, term_center_group, term_neighbor_group,
    term_weight) with term_weight equal to the summed orbit sizes.
    """
    row_shell = np.asarray(row_shell, dtype=np.int64)
    row_center_group = np.asarray(row_center_group, dtype=np.int64)
    row_neighbor_group = np.asarray(row_neighbor_group, dtype=np.int64)

    n_groups = max(int(n_groups), 1)

    if row_shell.size == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )

    n_shell = int(row_shell.max()) + 1

    if n_shell * n_groups * n_groups < np.iinfo(np.int64).max // 2:
        code = (
            row_shell * n_groups
            + row_center_group
        ) * n_groups + row_neighbor_group

        unique_code, inverse = np.unique(
            code,
            return_inverse=True,
        )

        term_shell = unique_code // (n_groups * n_groups)
        rest = unique_code % (n_groups * n_groups)
        term_center = rest // n_groups
        term_neighbor = rest % n_groups

    else:
        stacked = np.stack(
            [
                row_shell,
                row_center_group,
                row_neighbor_group,
            ],
            axis=1,
        )

        unique_rows, inverse = np.unique(
            stacked,
            axis=0,
            return_inverse=True,
        )

        term_shell = unique_rows[:, 0]
        term_center = unique_rows[:, 1]
        term_neighbor = unique_rows[:, 2]

    term_weight = np.bincount(
        np.asarray(inverse, dtype=np.int64).reshape(-1),
        weights=np.asarray(row_center_orbit, dtype=np.float64),
        minlength=int(term_shell.size),
    )

    return (
        term_shell.astype(np.int32, copy=False),
        term_center.astype(np.int32, copy=False),
        term_neighbor.astype(np.int32, copy=False),
        term_weight.astype(np.float64, copy=False),
    )


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

def _ensure_numba_uses_available_threads():
    """Try to let Numba use all logical CPUs.

    Returns
    -------
    (n_threads, layer)
    """
    if not NUMBA_AVAILABLE:
        return 1, "numba-unavailable"

    target = os.cpu_count() or 1

    try:
        # If NUMBA_NUM_THREADS was set externally to a smaller maximum,
        # set_num_threads(target) may fail. In that case, keep current setting.
        set_num_threads(int(target))
    except Exception:
        pass

    try:
        n_threads = int(get_num_threads())
    except Exception:
        n_threads = 1

    try:
        layer = str(threading_layer())
    except Exception:
        layer = "unknown"

    return n_threads, layer

def structure_hash(
    structure,
    pair_cutoff: float,
    *,
    force_p1_shells: bool = False,
) -> str:
    """
    Return a stable shell-cache key.

    The hash contains only data that affect generated shell rows:

        - cache format version;
        - pair cutoff;
        - whether P1 shell generation is forced;
        - lattice matrix;
        - fractional coordinates;
        - site species.

    This function intentionally performs no symmetry analysis and no
    neighbour search.
    """
    if structure is None:
        raise ValueError(
            "Cannot build a shell-cache hash without a structure."
        )

    hash_start = time.perf_counter()

    lattice_matrix = np.ascontiguousarray(
        np.asarray(
            structure.lattice.matrix,
            dtype=np.float64,
        )
    )

    fractional_coordinates = np.ascontiguousarray(
        np.asarray(
            structure.frac_coords,
            dtype=np.float64,
        )
    )

    # Coordinates read from a CIF already have finite precision.
    #
    # Quantize directly to integer values instead of creating a second
    # float64 array with np.round(...). This reduces temporary allocation
    # and makes the hash stable against insignificant floating-point noise.
    coordinate_integer = np.rint(
        fractional_coordinates
        * 1.0e10
    ).astype(
        np.int64,
        copy=False,
    )

    lattice_integer = np.rint(
        lattice_matrix
        * 1.0e10
    ).astype(
        np.int64,
        copy=False,
    )

    species_text = "\x1f".join(
        str(
            site.species
        )
        for site in structure.sites
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            CACHE_VERSION
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        b"|"
    )

    digest.update(
        np.asarray(
            [
                float(
                    pair_cutoff
                )
            ],
            dtype=np.float64,
        ).tobytes()
    )

    digest.update(
        np.asarray(
            [
                int(
                    bool(
                        force_p1_shells
                    )
                )
            ],
            dtype=np.uint8,
        ).tobytes()
    )

    digest.update(
        lattice_integer.tobytes()
    )

    digest.update(
        coordinate_integer.tobytes()
    )

    digest.update(
        species_text
    )

    key = digest.hexdigest()

    print(
        "[SHELL CACHE] Hash complete: "
        f"sites={len(structure.sites):,}, "
        f"cutoff={float(pair_cutoff):.6g} Å, "
        f"force_p1={bool(force_p1_shells)}, "
        f"time={time.perf_counter() - hash_start:.3f} s, "
        f"key={key[:16]}",
        flush=True,
    )

    return key

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _igcd(a: int, b: int) -> int:
        a = abs(a)
        b = abs(b)
        while b != 0:
            a, b = b, a % b
        return a

    @njit(cache=True)
    def _igcd3(a: int, b: int, c: int) -> int:
        return _igcd(_igcd(a, b), c)

    @njit(cache=True, fastmath=True, parallel=True)
    def _postprocess_neighbors_numba(
        frac,
        images,
        c_idx,
        n_idx,
        site_species_id,
        lattice_matrix_in,
        inv_recip_T,
    ):
        """
        Parallel post-processing of neighbor-list rows.

        Computes:
        - fractional pair vectors
        - cartesian vectors internally
        - nearest HKL
        - reduced direction
        - species ids for alpha/beta
        """
        n = c_idx.size

        fvec = np.empty((n, 3), dtype=np.float32)
        hkl = np.empty((n, 3), dtype=np.int32)
        direction = np.empty((n, 3), dtype=np.int16)
        a_id = np.empty(n, dtype=np.int16)
        b_id = np.empty(n, dtype=np.int16)

        for i in prange(n):
            ci = c_idx[i]
            ni = n_idx[i]

            f0 = frac[ni, 0] + images[i, 0] - frac[ci, 0]
            f1 = frac[ni, 1] + images[i, 1] - frac[ci, 1]
            f2 = frac[ni, 2] + images[i, 2] - frac[ci, 2]

            fvec[i, 0] = f0
            fvec[i, 1] = f1
            fvec[i, 2] = f2

            # cart = fvec @ lattice.matrix
            x = f0 * lattice_matrix_in[0, 0] + f1 * lattice_matrix_in[1, 0] + f2 * lattice_matrix_in[2, 0]
            y = f0 * lattice_matrix_in[0, 1] + f1 * lattice_matrix_in[1, 1] + f2 * lattice_matrix_in[2, 1]
            z = f0 * lattice_matrix_in[0, 2] + f1 * lattice_matrix_in[1, 2] + f2 * lattice_matrix_in[2, 2]

            # hkl_float = cart @ inv_recip_T.T
            hf0 = x * inv_recip_T[0, 0] + y * inv_recip_T[0, 1] + z * inv_recip_T[0, 2]
            hf1 = x * inv_recip_T[1, 0] + y * inv_recip_T[1, 1] + z * inv_recip_T[1, 2]
            hf2 = x * inv_recip_T[2, 0] + y * inv_recip_T[2, 1] + z * inv_recip_T[2, 2]

            hi = int(np.rint(hf0))
            ki = int(np.rint(hf1))
            li = int(np.rint(hf2))

            hkl[i, 0] = hi
            hkl[i, 1] = ki
            hkl[i, 2] = li

            g = _igcd3(hi, ki, li)
            if g == 0:
                g = 1

            dh = hi // g
            dk = ki // g
            dl = li // g

            sign = 1
            if dh != 0:
                sign = 1 if dh > 0 else -1
            elif dk != 0:
                sign = 1 if dk > 0 else -1
            elif dl != 0:
                sign = 1 if dl > 0 else -1

            direction[i, 0] = np.int16(sign * dh)
            direction[i, 1] = np.int16(sign * dk)
            direction[i, 2] = np.int16(sign * dl)

            a_id[i] = site_species_id[ci]
            b_id[i] = site_species_id[ni]

        return fvec, hkl, direction, a_id, b_id

    @njit(cache=True, fastmath=True, parallel=True)
    def _count_image_pairs_numba(
        translations,
        unique_center_indices,
        frac,
        lattice_matrix_in,
        rmax,
        tol,
    ):
        """
        Count valid neighbor rows per periodic image translation.

        Parallelized over translation vectors. This is efficient for high-symmetry
        structures with few symmetry-unique center sites.
        """
        n_trans = translations.shape[0]
        n_centers = unique_center_indices.size
        n_sites = frac.shape[0]

        counts = np.zeros(n_trans, dtype=np.int64)
        r2max = float(rmax) * float(rmax)
        tol2 = float(tol) * float(tol)

        for it in prange(n_trans):
            tx = translations[it, 0]
            ty = translations[it, 1]
            tz = translations[it, 2]

            cnt = 0

            for ic in range(n_centers):
                ci = unique_center_indices[ic]

                c0 = frac[ci, 0]
                c1 = frac[ci, 1]
                c2 = frac[ci, 2]

                for ni in range(n_sites):
                    f0 = frac[ni, 0] + tx - c0
                    f1 = frac[ni, 1] + ty - c1
                    f2 = frac[ni, 2] + tz - c2

                    # cart = fvec @ lattice.matrix
                    x = f0 * lattice_matrix_in[0, 0] + f1 * lattice_matrix_in[1, 0] + f2 * lattice_matrix_in[2, 0]
                    y = f0 * lattice_matrix_in[0, 1] + f1 * lattice_matrix_in[1, 1] + f2 * lattice_matrix_in[2, 1]
                    z = f0 * lattice_matrix_in[0, 2] + f1 * lattice_matrix_in[1, 2] + f2 * lattice_matrix_in[2, 2]

                    d2 = x * x + y * y + z * z

                    if d2 <= r2max and d2 > tol2:
                        cnt += 1

            counts[it] = cnt

        return counts


    @njit(cache=True, fastmath=True, parallel=True)
    def _fill_image_pairs_numba(
        translations,
        unique_center_indices,
        offsets,
        frac,
        site_species_id,
        lattice_matrix_in,
        inv_recip_T,
        rmax,
        tol,
    ):
        """
        Fill neighbor rows from periodic image translations.

        Outputs:
        - c_idx, n_idx, images, dists
        - fvec
        - hkl
        - direction
        - a_id, b_id

        This combines neighbor generation and post-processing in one Numba-parallel path.
        """
        n_trans = translations.shape[0]
        n_centers = unique_center_indices.size
        n_sites = frac.shape[0]

        total = int(offsets[-1])

        c_idx = np.empty(total, dtype=np.int32)
        n_idx = np.empty(total, dtype=np.int32)
        images = np.empty((total, 3), dtype=np.int32)
        dists = np.empty(total, dtype=np.float64)

        fvec = np.empty((total, 3), dtype=np.float32)
        hkl = np.empty((total, 3), dtype=np.int32)
        direction = np.empty((total, 3), dtype=np.int16)

        a_id = np.empty(total, dtype=np.int16)
        b_id = np.empty(total, dtype=np.int16)

        r2max = float(rmax) * float(rmax)
        tol2 = float(tol) * float(tol)

        for it in prange(n_trans):
            tx = translations[it, 0]
            ty = translations[it, 1]
            tz = translations[it, 2]

            pos = offsets[it]

            for ic in range(n_centers):
                ci = unique_center_indices[ic]

                c0 = frac[ci, 0]
                c1 = frac[ci, 1]
                c2 = frac[ci, 2]

                for ni in range(n_sites):
                    f0 = frac[ni, 0] + tx - c0
                    f1 = frac[ni, 1] + ty - c1
                    f2 = frac[ni, 2] + tz - c2

                    # cart = fvec @ lattice.matrix
                    x = f0 * lattice_matrix_in[0, 0] + f1 * lattice_matrix_in[1, 0] + f2 * lattice_matrix_in[2, 0]
                    y = f0 * lattice_matrix_in[0, 1] + f1 * lattice_matrix_in[1, 1] + f2 * lattice_matrix_in[2, 1]
                    z = f0 * lattice_matrix_in[0, 2] + f1 * lattice_matrix_in[1, 2] + f2 * lattice_matrix_in[2, 2]

                    d2 = x * x + y * y + z * z

                    if d2 <= r2max and d2 > tol2:
                        dist = np.sqrt(d2)

                        c_idx[pos] = ci
                        n_idx[pos] = ni

                        images[pos, 0] = tx
                        images[pos, 1] = ty
                        images[pos, 2] = tz

                        dists[pos] = dist

                        fvec[pos, 0] = f0
                        fvec[pos, 1] = f1
                        fvec[pos, 2] = f2

                        # hkl_float = cart @ inv_recip_T.T
                        hf0 = x * inv_recip_T[0, 0] + y * inv_recip_T[0, 1] + z * inv_recip_T[0, 2]
                        hf1 = x * inv_recip_T[1, 0] + y * inv_recip_T[1, 1] + z * inv_recip_T[1, 2]
                        hf2 = x * inv_recip_T[2, 0] + y * inv_recip_T[2, 1] + z * inv_recip_T[2, 2]

                        hi = int(np.rint(hf0))
                        ki = int(np.rint(hf1))
                        li = int(np.rint(hf2))

                        hkl[pos, 0] = hi
                        hkl[pos, 1] = ki
                        hkl[pos, 2] = li

                        g = _igcd3(hi, ki, li)
                        if g == 0:
                            g = 1

                        dh = hi // g
                        dk = ki // g
                        dl = li // g

                        sign = 1
                        if dh != 0:
                            sign = 1 if dh > 0 else -1
                        elif dk != 0:
                            sign = 1 if dk > 0 else -1
                        elif dl != 0:
                            sign = 1 if dl > 0 else -1

                        direction[pos, 0] = np.int16(sign * dh)
                        direction[pos, 1] = np.int16(sign * dk)
                        direction[pos, 2] = np.int16(sign * dl)

                        a_id[pos] = site_species_id[ci]
                        b_id[pos] = site_species_id[ni]

                        pos += 1

        return c_idx, n_idx, images, dists, fvec, hkl, direction, a_id, b_id
    

else:
    _postprocess_neighbors_numba = None
    _count_image_pairs_numba = None
    _fill_image_pairs_numba = None



def same_mod1(a, b, tol: float = 2.0e-3) -> bool:
    """True if two fractional positions coincide modulo lattice translations."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d -= np.rint(d)
    return bool(np.max(np.abs(d)) < tol)


def symmetry_operations(structure, symprec: float = 1.0e-3):
    """[(R, t)] fractional symmetry operations of the structure as given."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        return [
            (
                np.asarray(op.rotation_matrix, dtype=np.float64),
                np.asarray(op.translation_vector, dtype=np.float64),
            )
            for op in sga.get_symmetry_operations(cartesian=False)
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[XYZ] Symmetry analysis failed; using identity only: {exc}")
        return [(np.eye(3), np.zeros(3))]


def site_coordinate_basis(p0, ops):
    """
    Special-position constraint of a site at p0.

    Returns (B, free_axes): delta = B @ shift(free coordinates), from the
    mean of the stabilizer rotations (projector onto allowed displacements).
    """
    stabilizer = [R for R, t in ops if same_mod1(R @ p0 + t, p0)]

    if not stabilizer:
        stabilizer = [np.eye(3)]

    P = np.mean(np.asarray(stabilizer), axis=0)
    P[np.abs(P) < 1.0e-9] = 0.0

    free_axes = []

    for axis in range(3):
        cols = free_axes + [axis]

        if np.linalg.matrix_rank(P[:, cols], tol=1.0e-6) == len(cols):
            free_axes.append(axis)

    if free_axes:
        sub = P[np.ix_(free_axes, free_axes)]

        try:
            B = P[:, free_axes] @ np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            B = P[:, free_axes] @ np.linalg.pinv(sub)

        B[np.abs(B) < 1.0e-9] = 0.0
    else:
        B = np.zeros((3, 0))

    return B, free_axes


def build_coordinate_maps(structure, site_group, group_keys, reference, groups, ops=None):
    """
    Symmetry maps for coordinate changes of site groups (CIF labels).

    For every group g in ``groups``: its sites, the rotation R_s mapping the
    reference position p0 = reference[g] onto each site s (so site s moves by
    R_s @ delta), and the special-position basis B / free axes.
    """
    site_group = np.asarray(site_group, dtype=np.int64)
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    groups = list(groups)

    if ops is None:
        needs_symmetry = any(int(np.count_nonzero(site_group == g)) > 1 for g in groups)
        ops = symmetry_operations(structure) if needs_symmetry else [(np.eye(3), np.zeros(3))]

    out = {}

    for g in groups:
        key = group_keys[g]
        p0 = np.asarray(reference[g], dtype=np.float64)
        sites = np.flatnonzero(site_group == g)

        B, free_axes = site_coordinate_basis(p0, ops)

        rotations = np.zeros((len(sites), 3, 3), dtype=np.float64)
        unmatched = 0

        for k, s in enumerate(sites):
            match = None

            for R, t in ops:
                if same_mod1(R @ p0 + t, frac[s]):
                    match = R
                    break

            if match is None:
                unmatched += 1
                match = np.eye(3)

            rotations[k] = match

        if unmatched:
            print(
                f"[XYZ] {unmatched} site(s) of {key} could not be mapped "
                "to the CIF position by a symmetry operation; they move "
                "with the reference shift."
            )

        out[g] = {
            "key": key,
            "p0": p0.copy(),
            "B": B,
            "free_axes": list(free_axes),
            "sites": sites.astype(np.int64),
            "rotations": rotations,
        }

    return out


def get_lattice_constraints(structure):
    system = SpacegroupAnalyzer(structure).get_crystal_system().lower()
    if system == "cubic":
        return {"b": "a", "c": "a", "alpha": 90, "beta": 90, "gamma": 90}
    elif system == "tetragonal":
        return {"b": "a", "alpha": 90, "beta": 90, "gamma": 90}
    elif system == "hexagonal":
        return {"b": "a", "alpha": 90, "beta": 90, "gamma": 120}
    elif system == "trigonal":
        # Hexagonal axes (gamma = 120) or rhombohedral axes (a = b = c,
        # alpha = beta = gamma).
        if abs(float(structure.lattice.gamma) - 120.0) < 0.5:
            return {"b": "a", "alpha": 90, "beta": 90, "gamma": 120}
        return {"b": "a", "c": "a", "beta": "alpha", "gamma": "alpha"}
    elif system == "orthorhombic":
        return {"alpha": 90, "beta": 90, "gamma": 90}
    elif system == "monoclinic":
        return {"alpha": 90, "gamma": 90}
    return {}

def nearest_hkl_from_cartesian(vector_cart, lattice, tol=1e-3):
    """Return the nearest Miller indices for the given cartesian vector."""
    recip_matrix = lattice.reciprocal_lattice.matrix.T
    hkl_float = np.linalg.solve(recip_matrix, vector_cart)
    hkl_int = np.round(hkl_float).astype(int)
    if np.linalg.norm(vector_cart) < tol:
        return (0, 0, 0)
    return tuple(hkl_int)

def reduce_direction(h, k, l):
    """Reduce (hkl) to smallest integer triplet for crystallographic direction."""
    from math import gcd
    def gcd3(a, b, c): return gcd(gcd(abs(a), abs(b)), abs(c))
    d = gcd3(h, k, l)
    if d == 0:
        return (0, 0, 0)
    # Always use positive first nonzero index for direction
    h, k, l = int(h//d), int(k//d), int(l//d)
    # Choose a consistent sign (first nonzero positive)
    for val in (h, k, l):
        if val != 0:
            sign = 1 if val > 0 else -1
            break
    else:
        sign = 1
    return (sign*h, sign*k, sign*l)

def group_shells(entries):
    buckets = defaultdict(list)
    for e in entries:
        dist_key = round(e["distance"], 5)
        cart = np.array(e["cartesian_vector"])
        norm = np.linalg.norm(cart)
        vkey = tuple(np.round(np.abs(cart / norm if norm else cart), 6))
        # Optionally, you could also group by direction if you want to aggregate even more
        buckets[(e["pair_type"], dist_key, vkey)].append(e)

    shells = []
    for (ptype, dist, _), group in buckets.items():
        base = group[0]
        hkls = [tuple(e["hkl"]) for e in group]
        hkl = max(set(hkls), key=hkls.count)
        directions = [tuple(e["direction"]) for e in group]
        direction = max(set(directions), key=directions.count)
        shells.append({
            "fractional_vector": [float(x) for x in base["fractional_vector"]],
            "cartesian_vector": [float(x) for x in base["cartesian_vector"]],
            "distance": float(dist),
            "pair_type": str(ptype),
            "coordination": int(len(group)),
            "hkl": [int(i) for i in hkl],
            "direction": [int(i) for i in direction]
        })
    return sorted(shells, key=lambda x: x["distance"])



class StructureHandler:
    def __init__(
        self,
        cif_path,
        r_max: float,
        *,
        backend_config=None,
        coordinate_config=None,
    ):
        """
        coordinate_config (optional): {"initial": {...}, "refinable": {...}}
        from the input file. Sites whose x_/y_/z_<label> parameters are
        refinable, differ from the CIF, or whose element is listed in
        backend_config["refine_xyz_species"] become movable: their neighbor
        rows are kept ungrouped so their vectors can follow the coordinates.
        """
        self.backend_config = dict(
            backend_config
            or {}
        )

        self.force_p1_shells = bool(
            self.backend_config.get(
                "force_p1_shells",
                False,
            )
        )
        
        self.force_p1_shells = bool(
            self.backend_config.get(
                "force_p1_shells",
                False,
            )
        )

        self.shell_backend_requested = str(
            self.backend_config.get(
                "shell_backend",
                "cpu",
            )
        ).strip().lower()

        self.gpu_device = int(
            self.backend_config.get(
                "gpu_device",
                0,
            )
        )

        self.gpu_memory_fraction = float(
            self.backend_config.get(
                "gpu_memory_fraction",
                0.70,
            )
        )

        self.shell_backend_active = "cpu"

        if self.shell_backend_requested in (
            "cuda",
            "gpu",
            "cupy",
            "auto",
        ):
            try:
                from pdf_fitting.models.gpu.backend import (
                    gpu_summary,
                    resolve_backend,
                )

                self.shell_backend_active = resolve_backend(
                    self.shell_backend_requested,
                    default="cpu",
                )

                if self.shell_backend_active == "cuda":
                    print(
                        "[CUDA SHELLS] "
                        + gpu_summary(
                            self.gpu_device
                        )
                    )

            except Exception as exc:
                if self.shell_backend_requested in (
                    "cuda",
                    "gpu",
                    "cupy",
                ):
                    raise

                self.shell_backend_active = "cpu"

                print(
                    "[CUDA SHELLS] Falling back to CPU: "
                    f"{exc}"
                )

        # ---------------------------------------------------------
        # Structure source
        #
        # Normal calculations pass a path. Streaming stochastic-bulk
        # calculations pass an in-memory pymatgen Structure.
        # ---------------------------------------------------------
        if isinstance(
            cif_path,
            Structure,
        ):
            self.structure = cif_path.copy()

            self.structure_source_path = ""

            self.structure_source_label = (
                "in_memory_structure"
            )

        else:
            structure_path = os.path.abspath(
                str(
                    cif_path
                )
            )

            self.structure_source_path = (
                structure_path
            )

            self.structure_source_label = (
                structure_path
            )

            try:
                if structure_path.lower().endswith(
                    ".cif"
                ):
                    from pdf_fitting.gui.cif_utils import (
                        read_cif_asu_sites,
                    )

                    structure_fast, _asu = (
                        read_cif_asu_sites(
                            structure_path
                        )
                    )

                    if structure_fast is not None:
                        self.structure = (
                            structure_fast
                        )

                    else:
                        self.structure = (
                            Structure.from_file(
                                structure_path
                            )
                        )

                else:
                    self.structure = (
                        Structure.from_file(
                            structure_path
                        )
                    )

            except Exception:
                self.structure = (
                    Structure.from_file(
                        structure_path
                    )
                )

        self.r_max = float(
            r_max
        )

        # -----------------------------------------------------------------
        # Occupancy refinement metadata.
        #
        # site_labels:          CIF atom labels, e.g. N1, C7 (all
        #                       symmetry-equivalent copies share a label).
        # base_site_occupancy:  total CIF occupancy per site.
        # occupancy groups:     one group per safe label key; every site in
        #                       a group shares one occ_<key> parameter.
        #
        # The shell table built below stores these arrays (and the cache
        # restores them), so they are re-read from the table afterwards.
        # -----------------------------------------------------------------
        self._init_occupancy_groups()
        self._init_coordinate_model(coordinate_config)

        self.r_max_cache = float(r_max) * 1.10
        if self.force_p1_shells:
            self.constraints = {}

        else:
            self.constraints = get_lattice_constraints(
                self.structure
            )
        self._shell_table = None
        self.shells = self._generate_shells_fast_cached()
        self._sync_occupancy_groups_from_table()

    def _init_occupancy_groups(self) -> None:
        """
        Build per-site occupancy groups from CIF labels and occupancies.
        """
        sites = self.structure.sites

        self.site_labels = [
            site_label(site, i)
            for i, site in enumerate(sites)
        ]

        self.base_site_occupancy = np.asarray(
            [
                site_cif_occupancy(site)
                for site in sites
            ],
            dtype=np.float64,
        )

        group_index = {}
        site_group = np.zeros(len(sites), dtype=np.int32)

        for i, label in enumerate(self.site_labels):
            key = safe_param_key(label) or f"site{i + 1}"

            if key not in group_index:
                group_index[key] = len(group_index)

            site_group[i] = group_index[key]

        self.occupancy_group_keys = list(group_index.keys())
        self.site_occupancy_group = site_group

        n_groups = len(self.occupancy_group_keys)

        group_sum = np.bincount(
            site_group,
            weights=self.base_site_occupancy,
            minlength=n_groups,
        )

        group_count = np.bincount(
            site_group,
            minlength=n_groups,
        ).astype(np.float64)

        self.occupancy_group_base = group_sum / np.maximum(group_count, 1.0)

        # Majority element of the first site in every group.
        group_element = [""] * n_groups

        for i, g in enumerate(site_group):
            if not group_element[int(g)]:
                group_element[int(g)] = site_element(sites[i])

        self.occupancy_group_elements = group_element

        self.occupancy_param_names = [
            f"occ_{key}"
            for key in self.occupancy_group_keys
        ]

    # -----------------------------------------------------------------
    # Fractional-coordinate refinement
    #
    # Every occupancy group (CIF label) has a reference position p0 (its
    # CIF asymmetric-unit coordinates). The x_/y_/z_<key> parameters set a
    # shift delta = p - p0, projected onto the coordinates allowed by the
    # site symmetry (special positions). Every symmetry-equivalent site s
    # of the group moves by R_s @ delta, where (R_s, t_s) maps p0 onto s.
    # -----------------------------------------------------------------

    def _asu_reference_coordinates(self) -> dict:
        """{safe label key: CIF asymmetric-unit fractional coordinates}."""
        out = {}
        path = str(getattr(self, "structure_source_path", "") or "")

        if not path.lower().endswith(".cif"):
            return out

        try:
            from pdf_fitting.gui.cif_utils import read_cif_asu_sites

            _structure, asu = read_cif_asu_sites(path)

            for rec in asu or []:
                key = safe_param_key(rec.get("label", ""))
                frac = rec.get("frac", None)

                if key and frac is not None and key not in out:
                    out[key] = np.asarray(frac, dtype=np.float64).reshape(3)

        except Exception as exc:
            print(f"[XYZ] Could not read CIF asymmetric unit: {exc}")

        return out

    def _init_coordinate_model(self, coordinate_config) -> None:
        cfg = coordinate_config or {}

        initial = {
            str(k).strip().lower(): v
            for k, v in (cfg.get("initial", {}) or {}).items()
        }
        refinable = {
            str(k).strip().lower(): bool(v)
            for k, v in (cfg.get("refinable", {}) or {}).items()
        }

        species_value = self.backend_config.get("refine_xyz_species", "")

        if isinstance(species_value, (list, tuple, set)):
            species_items = [str(v) for v in species_value]
        elif isinstance(species_value, str):
            species_items = re.split(r"[,;\s]+", species_value)
        else:
            species_items = []

        wanted = {
            clean_element_name(item).lower()
            for item in species_items
            if item.strip()
        }

        keys = self.occupancy_group_keys
        site_group = np.asarray(self.site_occupancy_group, dtype=np.int64)
        frac = np.asarray(self.structure.frac_coords, dtype=np.float64)

        first_site = {}

        for i, g in enumerate(site_group):
            first_site.setdefault(int(g), i)

        reference = self._asu_reference_coordinates()

        p0 = np.zeros((len(keys), 3), dtype=np.float64)

        for g, key in enumerate(keys):
            if key in reference:
                p0[g] = reference[key]
            else:
                p0[g] = frac[first_site[g]]

        self.coordinate_reference = p0

        moving = set()

        for g, key in enumerate(keys):
            if str(self.occupancy_group_elements[g]).lower() in wanted:
                moving.add(g)

            for axis, letter in enumerate("xyz"):
                name = f"{letter}_{key}"

                if refinable.get(name, False):
                    moving.add(g)

                value = initial.get(name, None)

                if value is None:
                    continue

                try:
                    if abs(float(value) - p0[g, axis]) > 1.0e-7:
                        moving.add(g)
                except (TypeError, ValueError):
                    pass

        self.moving_groups = sorted(moving)
        self.coordinate_groups = {}

        if self.moving_groups:
            self._build_coordinate_maps()

        # Anisotropic displacement: sites with any beta<ij>_<key> parameter.
        adp = set()
        for g, key in enumerate(keys):
            if any(f"beta{c}_{key}" in initial for c in ADP_COMPONENTS):
                adp.add(g)
        self.adp_groups = sorted(adp)
        self._build_adp_maps()

    def adp_parameter_info(self) -> dict:
        """{key: {component: ("free" | "tied" | "fixed", terms)}} of anisotropic sites."""
        cache = getattr(self, "_adp_info_cache", None)
        if cache is None:
            cache = {}
            for g in getattr(self, "adp_groups", []) or []:
                cache[self.occupancy_group_keys[g]] = beta_constraints(self.adp_site_symmetry.get(g, [np.eye(3)]))
            self._adp_info_cache = cache
        return cache

    def adp_beta_from_params(self, params: dict) -> dict:
        """{group: β (6,)} built from the free components (tied follow, fixed = 0)."""
        p = {str(k).lower(): v for k, v in (params or {}).items()}
        info = self.adp_parameter_info()
        out = {}
        for g in getattr(self, "adp_groups", []) or []:
            key = self.occupancy_group_keys[g]
            cons = info.get(key, {})
            val = {}
            for c in ADP_COMPONENTS:
                kind, _terms = cons.get(c, ("free", None))
                if kind == "free":
                    try:
                        val[c] = float(p.get(f"beta{c}_{key.lower()}", 0.0))
                    except (TypeError, ValueError):
                        val[c] = 0.0
            b = []
            for c in ADP_COMPONENTS:
                kind, terms = cons.get(c, ("free", None))
                if kind == "free":
                    b.append(val[c])
                elif kind == "fixed":
                    b.append(0.0)
                else:
                    b.append(sum(w * val[t] for t, w in terms))
            out[g] = np.asarray(b, dtype=np.float64)
        return out

    def symmetrized_adp_params(self, params: dict) -> dict:
        """Tied / fixed beta<ij>_<key> set from the free components (same keys as given)."""
        out = dict(params)
        lc = {str(k).lower(): k for k in out}
        for g, b in self.adp_beta_from_params(params).items():
            key = self.occupancy_group_keys[g].lower()
            for c, v in zip(ADP_COMPONENTS, b):
                name = lc.get(f"beta{c}_{key}")
                if name is not None:
                    out[name] = float(v)
        return out

    def _build_adp_maps(self) -> None:
        """
        site_adp_entry[site] -> entry e (-1 = isotropic site); entry e is a
        (group, rotation R) with β_site = R β_ASU Rᵀ. adp_site_symmetry[g]
        lists the site-symmetry rotations of the ASU position of group g.
        """
        n_sites = len(self.structure)
        self.site_adp_entry = np.full(n_sites, -1, dtype=np.int32)
        self.adp_rotations = np.zeros((0, 3, 3), dtype=np.float64)
        self.adp_entry_group = np.zeros((0,), dtype=np.int32)
        self.adp_site_symmetry = {}
        if not getattr(self, "adp_groups", None):
            return
        ops = symmetry_operations(self.structure)
        maps = build_coordinate_maps(
            self.structure,
            np.asarray(self.site_occupancy_group, dtype=np.int64),
            self.occupancy_group_keys,
            self.coordinate_reference,
            self.adp_groups,
            ops=ops,
        )
        rots = []
        groups = []
        for g in self.adp_groups:
            info = maps[g]
            self.adp_site_symmetry[g] = adp_site_rotations(info["p0"], ops)
            local = []
            for s_idx, R in zip(info["sites"], info["rotations"]):
                e = next((k for k, Q in local if np.allclose(Q, R)), None)
                if e is None:
                    e = len(rots)
                    rots.append(np.asarray(R, dtype=np.float64))
                    groups.append(int(g))
                    local.append((e, R))
                self.site_adp_entry[int(s_idx)] = e
        self.adp_rotations = np.asarray(rots, dtype=np.float64).reshape(-1, 3, 3)
        self.adp_entry_group = np.asarray(groups, dtype=np.int32)

    def _build_coordinate_maps(self) -> None:
        """
        For every movable group: site list, per-site rotation R_s, and the
        special-position basis B (delta = B @ shift of free coordinates).
        """
        self.coordinate_groups = build_coordinate_maps(
            self.structure,
            np.asarray(self.site_occupancy_group, dtype=np.int64),
            self.occupancy_group_keys,
            self.coordinate_reference,
            self.moving_groups,
        )

    @staticmethod
    def _same_mod1(a, b, tol: float = 2.0e-3) -> bool:
        return same_mod1(a, b, tol)

    def coordinate_parameter_info(self) -> dict:
        """
        {"x_o1": {"group": "o1", "axis": 0, "reference": p0_x,
                  "status": "free" | "tied" | "fixed",
                  "depends_on": ["x_o1", ...]}, ...}

        Only movable groups are listed. "tied" coordinates follow free
        ones (e.g. x,x,x sites); "fixed" ones cannot move (e.g. x=1/2).
        """
        info = {}

        for g, grp in (getattr(self, "coordinate_groups", {}) or {}).items():
            key = grp["key"]
            B = grp["B"]
            free = grp["free_axes"]

            for axis, letter in enumerate("xyz"):
                name = f"{letter}_{key}"

                depends = [
                    f"{'xyz'[free[j]]}_{key}"
                    for j in range(len(free))
                    if abs(B[axis, j]) > 1.0e-9
                ]

                if axis in free:
                    status = "free"
                elif depends:
                    status = "tied"
                else:
                    status = "fixed"

                info[name] = {
                    "group": key,
                    "axis": axis,
                    "reference": float(grp["p0"][axis]),
                    "status": status,
                    "depends_on": depends,
                }

        return info

    def coordinate_shifts_from_params(self, params: dict | None) -> dict:
        """
        {group index: delta (3,)} for every movable group, from the free
        x_/y_/z_ parameters (missing ones keep the CIF value).
        """
        lookup = {
            str(k).strip().lower(): v
            for k, v in (params or {}).items()
        }

        out = {}

        for g, grp in (getattr(self, "coordinate_groups", {}) or {}).items():
            free = grp["free_axes"]

            if not free:
                out[g] = np.zeros(3)
                continue

            shift = np.zeros(len(free))

            for j, axis in enumerate(free):
                name = f"{'xyz'[axis]}_{grp['key']}"
                value = lookup.get(name, None)

                if value is None:
                    continue

                try:
                    shift[j] = float(value) - grp["p0"][axis]
                except (TypeError, ValueError):
                    pass

            out[g] = grp["B"] @ shift

        return out

    def symmetrized_coordinate_params(self, params: dict) -> dict:
        """
        Return params with tied/fixed coordinates of movable groups set to
        the values implied by the free ones.
        """
        result = dict(params)
        lookup = {str(k).strip().lower(): k for k in result}

        for g, delta in self.coordinate_shifts_from_params(params).items():
            grp = self.coordinate_groups[g]

            for axis, letter in enumerate("xyz"):
                if axis in grp["free_axes"]:
                    continue

                name = f"{letter}_{grp['key']}"
                existing = lookup.get(name)

                if existing is not None:
                    result[existing] = float(grp["p0"][axis] + delta[axis])

        return result

    def coordinate_parameters_for_elements(self, elements) -> dict:
        """
        {x_/y_/z_<key>: CIF value} for every movable group whose element
        is in ``elements``.
        """
        wanted = {
            clean_element_name(str(e)).lower()
            for e in (elements or [])
            if str(e).strip()
        }

        result = {}

        for g, grp in (getattr(self, "coordinate_groups", {}) or {}).items():
            if str(self.occupancy_group_elements[g]).lower() not in wanted:
                continue

            for axis, letter in enumerate("xyz"):
                result[f"{letter}_{grp['key']}"] = float(grp["p0"][axis])

        return result

    def _sync_occupancy_groups_from_table(self) -> None:
        """
        Make the cached shell table the source of truth for occupancy data.
        """
        st = self._shell_table

        if not st or "site_occ_group" not in st:
            return

        keys = list(st.get("occ_group_keys", []) or [])

        if not keys:
            return

        self.occupancy_group_keys = keys
        self.site_occupancy_group = np.asarray(
            st["site_occ_group"],
            dtype=np.int32,
        )
        self.base_site_occupancy = np.asarray(
            st["site_occupancy"],
            dtype=np.float64,
        )

        n_groups = len(keys)

        group_sum = np.bincount(
            self.site_occupancy_group,
            weights=self.base_site_occupancy,
            minlength=n_groups,
        )

        group_count = np.bincount(
            self.site_occupancy_group,
            minlength=n_groups,
        ).astype(np.float64)

        self.occupancy_group_base = group_sum / np.maximum(group_count, 1.0)

        species = list(st.get("unique_species", []) or [])
        site_species = np.asarray(st["site_species_id"], dtype=np.int64)

        group_element = [""] * n_groups

        for i, g in enumerate(self.site_occupancy_group):
            if not group_element[int(g)] and int(site_species[i]) < len(species):
                group_element[int(g)] = str(species[int(site_species[i])])

        self.occupancy_group_elements = group_element

        self.occupancy_param_names = [
            f"occ_{key}"
            for key in keys
        ]


    def update_lattice(self, lattice_params: dict):
        """
        Update the structure lattice and regenerate shells.

        lattice_params keys: a,b,c,alpha,beta,gamma (angles in degrees)
        This is needed for refinement when lattice parameters change.
        """
        # Current lattice
        old = self.structure.lattice
        a0, b0, c0 = old.abc
        al0, be0, ga0 = old.angles

        # New lattice (fallback to current values if missing)
        a = float(lattice_params.get("a", a0))
        b = float(lattice_params.get("b", b0))
        c = float(lattice_params.get("c", c0))
        alpha = float(lattice_params.get("alpha", al0))
        beta = float(lattice_params.get("beta", be0))
        gamma = float(lattice_params.get("gamma", ga0))

        # Avoid unnecessary recompute if nothing changed
        if (
            abs(a - a0) < 1e-9 and abs(b - b0) < 1e-9 and abs(c - c0) < 1e-9 and
            abs(alpha - al0) < 1e-9 and abs(beta - be0) < 1e-9 and abs(gamma - ga0) < 1e-9
        ):
            return

        # Build new lattice and apply to structure
        from pymatgen.core import Lattice
        new_lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.structure = Structure(new_lat, self.structure.species, self.structure.frac_coords)

        # Update constraints based on (possibly) new symmetry
        self.constraints = get_lattice_constraints(self.structure)

        # IMPORTANT: Do NOT regenerate shells here.
        # Shells are stored in fractional space (with a safety margin) and
        # real-space distances are re-evaluated during PDF calculation.

    def _generate_shells_fast_cached(self):
        """Generate shells once (if cache miss) and store as NPZ."""
        key = structure_hash(
            self.structure,
            self.r_max_cache,
            force_p1_shells=self.force_p1_shells,
        )

        # Movable sites change which rows stay ungrouped.
        moving_keys = [
            self.occupancy_group_keys[g]
            for g in getattr(self, "moving_groups", []) or []
        ]

        if moving_keys:
            key += "_xyz" + hashlib.sha256(
                "\x1f".join(moving_keys).encode("utf-8")
            ).hexdigest()[:16]

        # Anisotropic sites change the grouping of rows.
        adp_keys = [
            self.occupancy_group_keys[g]
            for g in getattr(self, "adp_groups", []) or []
        ]

        if adp_keys:
            key += "_adp" + hashlib.sha256(
                "\x1f".join(adp_keys).encode("utf-8")
            ).hexdigest()[:16]

        cache_npz = os.path.join(CACHE_DIR, f"shells_{key}.npz")
        cache_meta = os.path.join(CACHE_DIR, f"shells_{key}.meta.json")

        print(
            "[SHELL CACHE] Checking cache:\n"
            f"  NPZ:  {cache_npz}\n"
            f"  Meta: {cache_meta}\n"
            f"  NPZ exists: {os.path.exists(cache_npz)}\n"
            f"  Meta exists: {os.path.exists(cache_meta)}",
            flush=True,
        )

        if os.path.exists(cache_npz) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                npz = np.load(cache_npz, allow_pickle=False)
                self._shell_table = {
                    "shell_frac": npz["shell_frac"],
                    "shell_mult": npz["shell_mult"],
                    "shell_mult_eff": npz["shell_mult_eff"] if "shell_mult_eff" in npz else npz["shell_mult"].astype(np.float32),
                    "shell_pair_id": npz["shell_pair_id"],
                    "shell_alpha_id": npz["shell_alpha_id"],
                    "shell_beta_id": npz["shell_beta_id"],
                    "shell_hkl": npz["shell_hkl"],
                    "shell_direction": npz["shell_direction"],
                    "shell_k_rank": npz["shell_k_rank"],
                    "shell_upair_id": npz["shell_upair_id"],
                    "shell_k_within_upair": npz["shell_k_within_upair"],
                    "pair_offsets": npz["pair_offsets"],
                    "pair_alpha_id": npz["pair_alpha_id"],
                    "pair_beta_id": npz["pair_beta_id"],
                    "unique_species": meta["unique_species"],
                    "pair_list": [tuple(x) for x in meta["pair_list"]],
                    "upair_list": [tuple(x) for x in meta.get("upair_list", [])],
                    "strain_unique_dirs": npz["strain_unique_dirs"] if "strain_unique_dirs" in npz else None,
                    "strain_inv_dirs": npz["strain_inv_dirs"] if "strain_inv_dirs" in npz else None,
                }

                for occ_key in _OCCUPANCY_TABLE_ARRAYS:
                    if occ_key not in npz:
                        raise KeyError(
                            f"cache has no occupancy array {occ_key}"
                        )

                    self._shell_table[occ_key] = npz[occ_key]

                self._shell_table["occ_group_keys"] = list(
                    meta["occ_group_keys"]
                )

                print(
                    "[SHELL CACHE] Disk hit: "
                    f"loaded shell table from {cache_npz}",
                    flush=True,
                )

                # IMPORTANT:
                # Do NOT rebuild the legacy Python list of shell dictionaries here.
                # That is extremely slow for large/multi-element structures and defeats
                # the purpose of the packed NumPy cache.
                #
                # PDFCalculator uses self._shell_table directly.
                self.shells = []
                return self.shells
            except Exception:
                pass

        print(
            "[SHELL CACHE] Disk miss: computing shells "
            "(fast neighbour search, one-time)...",
            flush=True,
        )
        shells_legacy, shell_table = self._generate_shells_fast()
        self._shell_table = shell_table

        # Do not keep the large Python shell dictionary list in memory.
        # The fast PDF path uses self._shell_table arrays.
        self.shells = []

        # Use uncompressed NPZ for much faster loading.
        # Compressed NPZ can be much slower for large shell tables.
        np.savez(
            cache_npz,
            shell_frac=shell_table["shell_frac"],
            shell_mult=shell_table["shell_mult"],
            shell_mult_eff=shell_table["shell_mult_eff"],
            shell_pair_id=shell_table["shell_pair_id"],
            shell_alpha_id=shell_table["shell_alpha_id"],
            shell_beta_id=shell_table["shell_beta_id"],
            shell_hkl=shell_table["shell_hkl"],
            shell_direction=shell_table["shell_direction"],
            shell_k_rank=shell_table["shell_k_rank"],
            shell_upair_id=shell_table["shell_upair_id"],
            shell_k_within_upair=shell_table["shell_k_within_upair"],
            pair_offsets=shell_table["pair_offsets"],
            pair_alpha_id=shell_table["pair_alpha_id"],
            pair_beta_id=shell_table["pair_beta_id"],
            strain_unique_dirs=shell_table["strain_unique_dirs"],
            strain_inv_dirs=shell_table["strain_inv_dirs"],
            **{
                occ_key: shell_table[occ_key]
                for occ_key in _OCCUPANCY_TABLE_ARRAYS
            },
        )
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "unique_species": shell_table["unique_species"],
                    "pair_list": [list(x) for x in shell_table["pair_list"]],
                    "upair_list": [list(x) for x in shell_table.get("upair_list", [])],
                    "occ_group_keys": list(shell_table["occ_group_keys"]),
                    "r_max": float(self.r_max),
                    "r_max_cache": float(self.r_max_cache),
                },
                f,
                indent=2,
            )
        print(f"[CACHE] Saved shell table to {cache_npz}")
        return self.shells


    def get_refinable_parameters(self) -> set:
        """Get the set of refinable parameters based on symmetry."""
        try:
            structure = self.structure
            sga = SpacegroupAnalyzer(structure, symprec=1e-3)
            lattice_system = sga.get_lattice_type()
            
            if lattice_system == "cubic":
                return {"a"}
            elif lattice_system == "tetragonal":
                return {"a", "c"}
            elif lattice_system == "orthorhombic":
                return {"a", "b", "c"}
            elif lattice_system in ("hexagonal", "trigonal"):
                return {"a", "c"}
            elif lattice_system == "monoclinic":
                return {"a", "b", "c", "beta"}
            elif lattice_system == "triclinic":
                return {"a", "b", "c", "alpha", "beta", "gamma"}
            else:
                return {"a", "b", "c", "alpha", "beta", "gamma"}
        except Exception:
            return {"a", "b", "c", "alpha", "beta", "gamma"}
    

    def _get_unique_site_indices(self, symm):
        """Return representative site indices for symmetry-equivalent groups."""
        try:
            return [int(group[0]) for group in symm.equivalent_indices]
        except Exception:
            pass
        
        # Fallback if pymatgen version does not expose equivalent_indices
        unique_indices = []
        for group in symm.equivalent_sites:
            site0 = group[0]
            found = None
            for i, site in enumerate(self.structure.sites):
                try:
                    same_species = str(site.species) == str(site0.species)
                    same_frac = np.allclose(
                        np.asarray(site.frac_coords, dtype=float) % 1.0,
                        np.asarray(site0.frac_coords, dtype=float) % 1.0,
                        atol=1e-8,
                    )
                    if same_species and same_frac:
                        found = i
                        break
                except Exception:
                    pass
            if found is not None:
                unique_indices.append(int(found))

        if not unique_indices:
            unique_indices = list(range(len(self.structure.sites)))

        return unique_indices

    def _get_neighbor_list_for_unique_sites_parallel(
        self,
        unique_site_indices,
    ):
        """
        Build the neighbor list for all symmetry-unique centers in one
        vectorized pymatgen call.

        Calling Structure.get_neighbor_list() separately in many Python
        threads repeats expensive neighbor-search setup and can cause severe
        memory-bandwidth contention for large cutoffs.
        """
        unique_site_indices = np.asarray(
            unique_site_indices,
            dtype=np.int32,
        )

        if unique_site_indices.size == 0:
            return (
                np.zeros(
                    0,
                    dtype=np.int32,
                ),
                np.zeros(
                    0,
                    dtype=np.int32,
                ),
                np.zeros(
                    (
                        0,
                        3,
                    ),
                    dtype=np.int32,
                ),
                np.zeros(
                    0,
                    dtype=np.float64,
                ),
            )

        sites = [
            self.structure[
                int(site_index)
            ]
            for site_index in unique_site_indices
        ]

        start_time = time.perf_counter()

        center_local, neighbor_index, images, distances = (
            self.structure.get_neighbor_list(
                r=float(
                    self.r_max_cache
                ),
                sites=sites,
                numerical_tol=1.0e-8,
                exclude_self=True,
            )
        )

        center_local = np.asarray(
            center_local,
            dtype=np.int64,
        )

        center_index = unique_site_indices[
            center_local
        ].astype(
            np.int32,
            copy=False,
        )

        neighbor_index = np.asarray(
            neighbor_index,
            dtype=np.int32,
        )

        images = np.asarray(
            images,
            dtype=np.int32,
        )

        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        print(
            "[SHELLS] One-pass pymatgen neighbor list: "
            f"centers={unique_site_indices.size:,}, "
            f"rows={distances.size:,}, "
            f"time={time.perf_counter() - start_time:.3f} s",
            flush=True,
        )

        return (
            center_index,
            neighbor_index,
            images,
            distances,
        )


    def _translation_vectors_for_cutoff(self) -> np.ndarray:
        """
        Generate periodic image translations large enough for r_max_cache.

        Uses reciprocal-lattice heights, which is safer for non-orthogonal cells
        than simply using a,b,c lengths.
        """
        lattice = self.structure.lattice

        try:
            rec = lattice.reciprocal_lattice_crystallographic
            heights = 1.0 / np.maximum(np.asarray(rec.abc, dtype=float), 1e-12)
        except Exception:
            # Fallback: use direct lattice vector lengths
            heights = np.maximum(np.asarray(lattice.abc, dtype=float), 1e-12)

        nmax = np.ceil(float(self.r_max_cache) / heights).astype(int) + 1
        na, nb, nc = [int(x) for x in nmax]

        translations = np.array(
            [(i, j, k)
             for i in range(-na, na + 1)
             for j in range(-nb, nb + 1)
             for k in range(-nc, nc + 1)],
            dtype=np.int32,
        )

        return translations

    def _get_neighbor_rows_by_images_numba(
        self,
        unique_site_indices,
        site_species_id,
        inv_recip_T,
    ):
        """
        High-symmetry path:
        generate neighbor rows by explicitly scanning periodic image translations.

        This uses a two-pass Numba parallel algorithm:
        1. count valid neighbor rows per image
        2. allocate exact arrays and fill them in parallel
        """
        if _count_image_pairs_numba is None or _fill_image_pairs_numba is None:
            raise RuntimeError("Numba image-parallel shell generation is unavailable.")

        n_threads, layer = _ensure_numba_uses_available_threads()

        unique_site_indices = np.asarray(unique_site_indices, dtype=np.int32)
        frac = np.asarray(self.structure.frac_coords, dtype=np.float64)
        lattice_matrix_in = np.asarray(self.structure.lattice.matrix, dtype=np.float64)
        inv_recip_T = np.asarray(inv_recip_T, dtype=np.float64)

        translations = self._translation_vectors_for_cutoff()

        print(
            f"[SHELLS] Image-parallel generation: "
            f"{len(unique_site_indices)} unique center(s), "
            f"{len(self.structure.sites)} site(s), "
            f"{len(translations)} image translation(s), "
            f"Numba threads={n_threads}, layer={layer}"
        )

        t_count0 = time.perf_counter()
        counts = _count_image_pairs_numba(
            translations,
            unique_site_indices,
            frac,
            lattice_matrix_in,
            float(self.r_max_cache),
            1e-8,
        )
        t_count1 = time.perf_counter()

        offsets = np.zeros(counts.size + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)

        total = int(offsets[-1])
        print(
            f"[SHELLS] Image-parallel candidate neighbor rows: {total} "
            f"(count pass {t_count1 - t_count0:.2f} s)"
        )

        if total == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int16),
                np.zeros((0,), dtype=np.int16),
                np.zeros((0,), dtype=np.int16),
            )

        t_fill0 = time.perf_counter()
        out = _fill_image_pairs_numba(
            translations,
            unique_site_indices,
            offsets,
            frac,
            site_species_id,
            lattice_matrix_in,
            inv_recip_T,
            float(self.r_max_cache),
            1e-8,
        )
        t_fill1 = time.perf_counter()

        print(f"[SHELLS] Image-parallel fill pass completed in {t_fill1 - t_fill0:.2f} s")

        return out

    def _generate_shells_fast(self):
        """Fast shell generation using pymatgen neighbor list (vectorized)."""
        if self.force_p1_shells:
            symm = None

            unique_site_indices = list(
                range(
                    len(
                        self.structure.sites
                    )
                )
            )

        else:
            sga = SpacegroupAnalyzer(
                self.structure,
                symprec=1e-3,
            )

            symm = sga.get_symmetrized_structure()

            unique_site_indices = (
                self._get_unique_site_indices(
                    symm
                )
            )

        lattice = self.structure.lattice

        # Element labels for PDF physics/scattering/Biso/densities
        species_element_all = [
            site_element(site)
            for site in self.structure.sites
        ]
        unique_species = sorted(set(species_element_all))
        sp_index = {s: i for i, s in enumerate(unique_species)}

        # Lambda/species labels for local pair lambdas
        # Examples: Fe0+ -> fe0plus, O2- -> o2minus
        species_lambda_all = [
            lambda_safe_species_label(site_element(site))
            for site in self.structure.sites
        ]
        unique_lambda_species = sorted(set(species_lambda_all))
        lambda_sp_index = {s: i for i, s in enumerate(unique_lambda_species)}

        # Element id per structure site, used for physics/scattering arrays
        site_species_id = np.asarray([sp_index[s] for s in species_element_all], dtype=np.int16)

        # Lambda/species id per structure site, used for lambda pair labels
        site_lambda_id = np.asarray([lambda_sp_index[s] for s in species_lambda_all], dtype=np.int16)


        # Reciprocal helper for (hkl) estimation from cart vectors.
        # IMPORTANT: this must be defined before both image-parallel and fallback paths.
        recip_T = np.asarray(lattice.reciprocal_lattice.matrix.T, dtype=np.float64)
        inv_recip_T = np.linalg.inv(recip_T)


        # ------------------------------------------------------------------
        # Alpha-site weighting for correct partial-PDF normalization.
        #
        # For g_ab(r), shells must be averaged over all atoms of alpha species.
        # If one element has multiple inequivalent sites, each representative
        # center must be weighted by:
        #
        #     multiplicity_of_that_site / total_number_of_alpha_atoms
        #
        # Example Li4GeS4:
        #     Li1 4c -> 4/16
        #     Li2 8d -> 8/16
        #     Li3 4b -> 4/16
        #
        # Without this, multi-site elements are overcounted and G(r) develops
        # a large linear baseline error.
        # ------------------------------------------------------------------
        species_counts_by_id = np.bincount(
            site_species_id.astype(np.int64),
            minlength=len(unique_species),
        ).astype(np.float64)

        site_alpha_weight = np.zeros(len(self.structure.sites), dtype=np.float64)

        if self.force_p1_shells:
            for site_index in range(
                len(
                    self.structure.sites
                )
            ):
                species_id = int(
                    site_species_id[
                        site_index
                    ]
                )

                species_count = max(
                    float(
                        species_counts_by_id[
                            species_id
                        ]
                    ),
                    1.0,
                )

                site_alpha_weight[
                    site_index
                ] = (
                    1.0
                    / species_count
                )

        else:
            try:
                for group in symm.equivalent_indices:
                    group = [
                        int(
                            site_index
                        )
                        for site_index in group
                    ]

                    if not group:
                        continue

                    representative = int(
                        group[
                            0
                        ]
                    )

                    species_id = int(
                        site_species_id[
                            representative
                        ]
                    )

                    species_count = max(
                        float(
                            species_counts_by_id[
                                species_id
                            ]
                        ),
                        1.0,
                    )

                    alpha_weight = (
                        float(
                            len(
                                group
                            )
                        )
                        / species_count
                    )

                    for site_index in group:
                        site_alpha_weight[
                            site_index
                        ] = alpha_weight

            except Exception:
                for site_index in range(
                    len(
                        self.structure.sites
                    )
                ):
                    species_id = int(
                        site_species_id[
                            site_index
                        ]
                    )

                    species_count = max(
                        float(
                            species_counts_by_id[
                                species_id
                            ]
                        ),
                        1.0,
                    )

                    site_alpha_weight[
                        site_index
                    ] = (
                        1.0
                        / species_count
                    )
        # ------------------------------------------------------------------
        # Hybrid neighbor-row generation
        #
        # Many unique centers:
        #   use pymatgen neighbor-list split over center-site chunks.
        #
        # Few unique centers / high symmetry:
        #   use Numba image-translation parallel generation.
        # ------------------------------------------------------------------
        n_cpu = os.cpu_count() or 1
        n_unique = len(unique_site_indices)
        n_sites = len(self.structure.sites)
        
        # Estimate cost of the explicit image-loop path:
        #   n_translations * n_unique_centers * n_sites
        #
        # This is often faster than pymatgen neighbor_list for low-symmetry P1
        # molecular structures, especially when the search radius is moderate.
        try:
            n_trans_est = int(self._translation_vectors_for_cutoff().shape[0])
        except Exception:
            n_trans_est = 0
        
        n_checks_est = int(max(1, n_trans_est) * max(1, n_unique) * max(1, n_sites))
        
        # Allow user override from environment variable if needed.
        # Example:
        #   set PDF_FITTING_IMAGE_MAX_CHECKS=500000000
        try:
            image_max_checks = int(os.environ.get("PDF_FITTING_IMAGE_MAX_CHECKS", "200000000"))
        except Exception:
            image_max_checks = 200_000_000
        
        use_image_parallel = (
            NUMBA_AVAILABLE
            and _count_image_pairs_numba is not None
            and _fill_image_pairs_numba is not None
            and n_trans_est > 0
            and n_checks_est <= image_max_checks
        )
        
        print(
            f"[SHELLS] Strategy estimate: "
            f"n_sites={n_sites}, n_unique={n_unique}, "
            f"n_trans={n_trans_est}, checks≈{n_checks_est:,}, "
            f"use_image_parallel={use_image_parallel}"
        )

        image_precomputed = False

        if use_image_parallel:
            try:
                (
                    c_idx,
                    n_idx,
                    images,
                    dists,
                    fvec,
                    hkl,
                    direction_arr,
                    a_id,
                    b_id,
                ) = self._get_neighbor_rows_by_images_numba(
                    unique_site_indices,
                    site_species_id,
                    inv_recip_T,
                )

                c_idx = np.asarray(c_idx, dtype=np.int32)
                n_idx = np.asarray(n_idx, dtype=np.int32)
                images = np.asarray(images, dtype=np.int32)
                dists = np.asarray(dists, dtype=np.float64)

                fvec = np.asarray(fvec, dtype=np.float32)
                hkl = np.asarray(hkl, dtype=np.int32)
                direction_arr = np.asarray(direction_arr, dtype=np.int16)
                a_id = np.asarray(a_id, dtype=np.int16)
                b_id = np.asarray(b_id, dtype=np.int16)

                dh = direction_arr[:, 0].astype(np.int32)
                dk = direction_arr[:, 1].astype(np.int32)
                dl = direction_arr[:, 2].astype(np.int32)

                image_precomputed = True

            except Exception as e:
                print(f"[WARN] Image-parallel shell generation failed; falling back to pymatgen neighbor list: {e}")
                image_precomputed = False

        if not image_precomputed:
            print(
                f"[SHELLS] Center-site parallel generation: "
                f"{n_unique} unique center(s), using up to {min(int(n_cpu), max(1, n_unique))} worker(s)"
            )

            c_idx, n_idx, images, dists = self._get_neighbor_list_for_unique_sites_parallel(
                unique_site_indices
            )

            c_idx = np.asarray(c_idx, dtype=np.int32)
            n_idx = np.asarray(n_idx, dtype=np.int32)
            images = np.asarray(images, dtype=np.int32)
            dists = np.asarray(dists, dtype=np.float64)

            frac = np.asarray(self.structure.frac_coords, dtype=np.float64)

            # ---- Parallel post-processing of neighbor-list rows ----
            if _postprocess_neighbors_numba is not None and len(dists) > 0:
                # Numba uses all available threads by default unless NUMBA_NUM_THREADS is set.
                fvec, hkl, direction_arr, a_id, b_id = _postprocess_neighbors_numba(
                    frac,
                    images,
                    c_idx,
                    n_idx,
                    site_species_id,
                    np.asarray(lattice.matrix, dtype=np.float64),
                    np.asarray(inv_recip_T, dtype=np.float64),
                )
                dh = direction_arr[:, 0].astype(np.int32)
                dk = direction_arr[:, 1].astype(np.int32)
                dl = direction_arr[:, 2].astype(np.int32)

            else:
                # Fallback pure NumPy path
                fvec = ((frac[n_idx] + images) - frac[c_idx]).astype(np.float32)

                cart = fvec.astype(np.float64) @ lattice.matrix
                hkl_float = cart @ inv_recip_T.T
                hkl = np.rint(hkl_float).astype(np.int32)

                h = hkl[:, 0].copy()
                k = hkl[:, 1].copy()
                l = hkl[:, 2].copy()

                g = np.gcd(np.gcd(np.abs(h), np.abs(k)), np.abs(l))
                g[g == 0] = 1
                dh = h // g
                dk = k // g
                dl = l // g

                sign = np.ones_like(dh)
                mask_h = dh != 0
                sign[mask_h] = np.where(dh[mask_h] > 0, 1, -1)
                mask_k = (~mask_h) & (dk != 0)
                sign[mask_k] = np.where(dk[mask_k] > 0, 1, -1)
                mask_l = (~mask_h) & (~mask_k) & (dl != 0)
                sign[mask_l] = np.where(dl[mask_l] > 0, 1, -1)

                dh *= sign
                dk *= sign
                dl *= sign

                a_id = site_species_id[c_idx].astype(np.int16)
                b_id = site_species_id[n_idx].astype(np.int16)

        # Exact reciprocal direction ∥ pair vector (h = G·u), replacing the
        # rounded hkl/2π of the generators above (see direction_utils).
        if len(fvec) > 0:
            from pdf_fitting.models.direction_utils import reciprocal_directions

            _lat = np.asarray(self.structure.lattice.matrix, dtype=np.float64)
            _dirs = reciprocal_directions(np.asarray(fvec, dtype=np.float64) @ _lat, _lat)
            dh = _dirs[:, 0].astype(np.int32)
            dk = _dirs[:, 1].astype(np.int32)
            dl = _dirs[:, 2].astype(np.int32)

        # Lambda/species IDs for the same neighbor rows.
        # These are used only for lambda pair labels/grouping.
        lam_a_id = site_lambda_id[c_idx].astype(np.int16)
        lam_b_id = site_lambda_id[n_idx].astype(np.int16)

        # Weight of each neighbor row according to the alpha/center site.
        # Used to build shell_mult_eff for correct PDF normalization.
        alpha_weight_row = site_alpha_weight[c_idx].astype(np.float64)

        # Orbit size of every center site (1 in forced-P1 mode). Occupancy
        # terms keep this unnormalized weight; the per-element atom count is
        # applied at evaluation time because it depends on occupancy.
        site_orbit_size = np.rint(
            site_alpha_weight
            * np.maximum(
                species_counts_by_id[site_species_id.astype(np.int64)],
                1.0,
            )
        )

        site_occ_group = np.asarray(
            self.site_occupancy_group,
            dtype=np.int32,
        )

        occupancy_site_arrays = {
            "site_occupancy": np.asarray(
                self.base_site_occupancy,
                dtype=np.float64,
            ),
            "site_species_id": site_species_id.astype(np.int16, copy=False),
            "site_occ_group": site_occ_group,
            "occ_group_keys": list(self.occupancy_group_keys),
        }

        # Rows touching a movable site stay ungrouped (one shell per row) so
        # their vectors can follow refined coordinates.
        moving_group_mask = np.zeros(
            max(len(self.occupancy_group_keys), 1),
            dtype=bool,
        )

        for g in getattr(self, "moving_groups", []) or []:
            moving_group_mask[int(g)] = True

        if len(dists) > 0 and moving_group_mask.any():
            dyn_row = (
                moving_group_mask[site_occ_group[c_idx]]
                | moving_group_mask[site_occ_group[n_idx]]
            )
        else:
            dyn_row = np.zeros(len(dists), dtype=bool)

        has_dyn = bool(dyn_row.any())

        # Anisotropic atoms: rows group only with rows of the same
        # (ASU site, rotation) on both ends, so every shell has one β
        # projection per atom.
        site_adp_entry = np.asarray(
            getattr(self, "site_adp_entry", np.full(len(self.structure), -1)),
            dtype=np.int32,
        )
        has_adp = bool(len(dists) > 0 and np.any(site_adp_entry >= 0))
        if has_adp:
            row_adp_c = site_adp_entry[np.asarray(c_idx, dtype=np.int64)]
            row_adp_n = site_adp_entry[np.asarray(n_idx, dtype=np.int64)]

        if has_dyn:
            print(
                f"[SHELLS] Movable sites: {int(dyn_row.sum()):,} of "
                f"{len(dists):,} neighbor rows kept ungrouped for "
                "coordinate refinement."
            )

        empty_dyn_arrays = {
            "dyn_shell": np.zeros((0,), dtype=np.int32),
            "dyn_center_site": np.zeros((0,), dtype=np.int32),
            "dyn_neighbor_site": np.zeros((0,), dtype=np.int32),
            "dyn_base_fvec": np.zeros((0, 3), dtype=np.float64),
        }


        # If nothing returned
        if len(dists) == 0:
            return [], {
                **occupancy_site_arrays,
                **empty_dyn_arrays,
                "shell_adp_center": np.zeros((0,), dtype=np.int32),
                "shell_adp_neighbor": np.zeros((0,), dtype=np.int32),
                "adp_rotations": np.zeros((0, 3, 3), dtype=np.float64),
                "adp_entry_group": np.zeros((0,), dtype=np.int32),
                "occ_term_shell": np.zeros((0,), dtype=np.int32),
                "occ_term_center_group": np.zeros((0,), dtype=np.int32),
                "occ_term_neighbor_group": np.zeros((0,), dtype=np.int32),
                "occ_term_weight": np.zeros((0,), dtype=np.float64),
                "shell_frac": np.zeros((0, 3), dtype=np.float32),
                "shell_mult": np.zeros((0,), dtype=np.int32),
                "shell_mult_eff": np.zeros((0,), dtype=np.float32),
                "shell_pair_id": np.zeros((0,), dtype=np.int32),
                "shell_alpha_id": np.zeros((0,), dtype=np.int16),
                "shell_beta_id": np.zeros((0,), dtype=np.int16),
                "shell_hkl": np.zeros((0, 3), dtype=np.int16),
                "shell_direction": np.zeros((0, 3), dtype=np.int16),
                "shell_k_rank": np.zeros((0,), dtype=np.int32),
                "shell_upair_id": np.zeros((0,), dtype=np.int32),
                "shell_k_within_upair": np.zeros((0,), dtype=np.int32),
                "pair_offsets": np.zeros((1,), dtype=np.int32),
                "pair_alpha_id": np.zeros((0,), dtype=np.int16),
                "pair_beta_id": np.zeros((0,), dtype=np.int16),
                "unique_species": unique_species,
                "pair_list": [],
                "upair_list": [],
            }

        # ------------------------------------------------------------------
        # Bucket key:
        #   lambda alpha id
        #   lambda beta id
        #   distance bin
        #   reduced direction h, k, l
        #
        # CUDA grouping is optional. The existing NumPy implementation remains
        # the fallback and reference path.
        # ------------------------------------------------------------------
        t_group0 = time.perf_counter()

        dist_i = np.rint(
            dists
            * 1.0e5
        ).astype(
            np.int32
        )

        gpu_group_result = None

        if (
            getattr(
                self,
                "shell_backend_active",
                "cpu",
            )
            == "cuda"
            and not has_dyn
            and not has_adp
        ):
            try:
                from pdf_fitting.models.gpu.shell_grouping import (
                    group_shell_rows_cuda,
                )

                t_unique0 = time.perf_counter()

                gpu_group_result = group_shell_rows_cuda(
                    lambda_alpha_id=lam_a_id,
                    lambda_beta_id=lam_b_id,
                    distance_integer=dist_i,
                    direction_h=dh,
                    direction_k=dk,
                    direction_l=dl,
                    alpha_weight_row=alpha_weight_row,
                    device_id=int(
                        self.gpu_device
                    ),
                    memory_fraction=float(
                        self.gpu_memory_fraction
                    ),
                )

                t_unique1 = time.perf_counter()

                first_idx = np.asarray(
                    gpu_group_result[
                        "first_idx"
                    ],
                    dtype=np.int64,
                )

                counts = np.asarray(
                    gpu_group_result[
                        "counts"
                    ],
                    dtype=np.int32,
                )

                shell_mult_eff_grouped = np.asarray(
                    gpu_group_result[
                        "shell_mult_eff"
                    ],
                    dtype=np.float32,
                )

                n_group = int(
                    first_idx.size
                )

                uniq = np.zeros(
                    n_group,
                    dtype=[
                        ("a", np.int16),
                        ("b", np.int16),
                        ("di", np.int32),
                        ("dh", np.int16),
                        ("dk", np.int16),
                        ("dl", np.int16),
                    ],
                )

                uniq["a"] = np.asarray(
                    gpu_group_result[
                        "lambda_alpha"
                    ],
                    dtype=np.int16,
                )

                uniq["b"] = np.asarray(
                    gpu_group_result[
                        "lambda_beta"
                    ],
                    dtype=np.int16,
                )

                uniq["di"] = np.asarray(
                    gpu_group_result[
                        "distance_integer"
                    ],
                    dtype=np.int32,
                )

                uniq["dh"] = np.asarray(
                    gpu_group_result[
                        "direction_h"
                    ],
                    dtype=np.int16,
                )

                uniq["dk"] = np.asarray(
                    gpu_group_result[
                        "direction_k"
                    ],
                    dtype=np.int16,
                )

                uniq["dl"] = np.asarray(
                    gpu_group_result[
                        "direction_l"
                    ],
                    dtype=np.int16,
                )

                inv = None

                print(
                    f"[CUDA SHELLS] Grouping: "
                    f"{len(dists):,} rows -> {len(uniq):,} shells "
                    f"in {t_unique1 - t_unique0:.2f} s"
                )

            except Exception as exc:
                print(
                    "[CUDA SHELLS] GPU grouping failed; "
                    f"using NumPy grouping: {exc}"
                )

                if self.shell_backend_requested in (
                    "cuda",
                    "gpu",
                    "cupy",
                ):
                    raise

                gpu_group_result = None

        if gpu_group_result is None:
            key_fields = [
                ("a", np.int16),
                ("b", np.int16),
                ("di", np.int32),
                ("dh", np.int16),
                ("dk", np.int16),
                ("dl", np.int16),
            ]

            if has_dyn:
                key_fields.append(("tag", np.int64))

            if has_adp:
                key_fields.append(("ac", np.int32))
                key_fields.append(("an", np.int32))

            key = np.zeros(
                len(dists),
                dtype=key_fields,
            )

            if has_adp:
                key["ac"] = row_adp_c
                key["an"] = row_adp_n

            if has_dyn:
                # 0 for groupable rows; a unique id for every movable row.
                key["tag"] = np.where(
                    dyn_row,
                    np.arange(1, len(dists) + 1, dtype=np.int64),
                    0,
                )

            key["a"] = lam_a_id
            key["b"] = lam_b_id
            key["di"] = dist_i
            key["dh"] = dh.astype(
                np.int16
            )
            key["dk"] = dk.astype(
                np.int16
            )
            key["dl"] = dl.astype(
                np.int16
            )

            t_unique0 = time.perf_counter()

            key_contiguous = np.ascontiguousarray(
                key
            )

            packed_dtype = np.dtype(
                (
                    np.void,
                    key_contiguous.dtype.itemsize,
                )
            )

            packed_key = key_contiguous.view(
                packed_dtype
            ).reshape(
                -1
            )

            (
                unique_packed,
                first_idx,
                inv,
                counts,
            ) = np.unique(
                packed_key,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )

            uniq = unique_packed.view(
                key_contiguous.dtype
            ).reshape(
                -1
            )

            t_unique1 = time.perf_counter()

            shell_mult_eff_grouped = None

            print(
                f"[SHELLS] NumPy grouping: "
                f"{len(dists):,} rows -> {len(uniq):,} shells "
                f"in {t_unique1 - t_unique0:.2f} s"
            )        

        # Representative vectors from first occurrence in each group
                # ------------------------------------------------------------------
        # Build packed shell_table arrays DIRECTLY from np.unique output.
        #
        # IMPORTANT:
        # Do NOT build the large legacy Python list of shell dictionaries.
        # That is a major bottleneck for large/multi-element structures.
        #
        # Coordination / multiplicity is exactly the np.unique counts.
        # ------------------------------------------------------------------
        t_table0 = time.perf_counter()

        rep = first_idx.astype(np.int64)
        n_shell = int(len(uniq))

        # Representative vectors / hkl / directions from first occurrence
        shell_frac = fvec[rep].astype(np.float32, copy=False)

        # Raw integer multiplicity, useful for display/ticks.
        shell_mult = counts.astype(np.int32, copy=False)

        # Effective alpha-site-averaged multiplicity for PDF normalization.
        if shell_mult_eff_grouped is not None:
            shell_mult_eff = np.asarray(
                shell_mult_eff_grouped,
                dtype=np.float32,
            )

        else:
            shell_mult_eff = np.bincount(
                inv.astype(
                    np.int64
                ),
                weights=alpha_weight_row,
                minlength=n_shell,
            ).astype(
                np.float32,
                copy=False,
            )

        shell_hkl = hkl[rep].astype(np.int16, copy=False)
        shell_dir = np.stack(
            [dh[rep], dk[rep], dl[rep]],
            axis=1,
        ).astype(np.int16, copy=False)

        # Element IDs for physics/scattering/Biso/density
        rep_alpha_el_id = a_id[rep].astype(np.int16, copy=False)
        rep_beta_el_id = b_id[rep].astype(np.int16, copy=False)

        shell_alpha_id = rep_alpha_el_id.copy()
        shell_beta_id = rep_beta_el_id.copy()

        # Distance bin for each grouped shell
        dist_keys_group = dist_i[rep].astype(np.int64, copy=False)

        # Lambda/species pair IDs from uniq key
        lam_a_group = uniq["a"].astype(np.int64, copy=False)
        lam_b_group = uniq["b"].astype(np.int64, copy=False)

        n_lam_species = int(len(unique_lambda_species))

        # Encoded ordered pair code:
        #   code = a_lam_id * n_lam_species + b_lam_id
        pair_code = lam_a_group * n_lam_species + lam_b_group

        # Build ordered pair_list in stable lexical order, like before
        unique_pair_codes = np.unique(pair_code)

        pair_list_unsorted = [
            (
                unique_lambda_species[int(code // n_lam_species)],
                unique_lambda_species[int(code % n_lam_species)],
            )
            for code in unique_pair_codes
        ]

        pair_list = sorted(pair_list_unsorted)
        pair_code_order = np.asarray(
            [
                lambda_sp_index[p[0]] * n_lam_species + lambda_sp_index[p[1]]
                for p in pair_list
            ],
            dtype=np.int64,
        )

        # Fast vectorized code -> pair_id mapping
        code_to_pair_id = np.full(n_lam_species * n_lam_species, -1, dtype=np.int32)
        for pid, code in enumerate(pair_code_order):
            code_to_pair_id[int(code)] = int(pid)

        shell_pair_id = code_to_pair_id[pair_code].astype(np.int32, copy=False)

        # Build unordered pair IDs for lambda coefficients / local trends
        lam_min = np.minimum(lam_a_group, lam_b_group)
        lam_max = np.maximum(lam_a_group, lam_b_group)
        upair_code = lam_min * n_lam_species + lam_max

        unique_upair_codes = np.unique(upair_code)

        upair_list_unsorted = [
            (
                unique_lambda_species[int(code // n_lam_species)],
                unique_lambda_species[int(code % n_lam_species)],
            )
            for code in unique_upair_codes
        ]

        upair_list = sorted(upair_list_unsorted)
        upair_code_order = np.asarray(
            [
                lambda_sp_index[p[0]] * n_lam_species + lambda_sp_index[p[1]]
                for p in upair_list
            ],
            dtype=np.int64,
        )

        code_to_upair_id = np.full(n_lam_species * n_lam_species, -1, dtype=np.int32)
        for uid, code in enumerate(upair_code_order):
            code_to_upair_id[int(code)] = int(uid)

        shell_upair_id = code_to_upair_id[upair_code].astype(np.int32, copy=False)

        # ------------------------------------------------------------------
        # Rank shells by UNIQUE DISTANCE within each unordered pair.
        #
        # All rows at the same distance for the same unordered pair get
        # the same k index. This is used by lambda_{pair}_{k}.
        # ------------------------------------------------------------------
        shell_k_within_upair = np.zeros(n_shell, dtype=np.int32)

        if n_shell > 0:
            order_ud = np.lexsort((dist_keys_group, shell_upair_id))

            uid_sorted = shell_upair_id[order_ud]
            dist_sorted = dist_keys_group[order_ud]

            new_upair = np.ones(n_shell, dtype=bool)
            new_dist = np.ones(n_shell, dtype=bool)

            if n_shell > 1:
                new_upair[1:] = uid_sorted[1:] != uid_sorted[:-1]
                new_dist[1:] = new_upair[1:] | (dist_sorted[1:] != dist_sorted[:-1])

            cumulative_unique = np.cumsum(new_dist).astype(np.int32) - 1

            # Reset rank to zero at each new unordered pair
            reset_offsets = np.maximum.accumulate(
                np.where(new_upair, cumulative_unique, 0)
            ).astype(np.int32)

            rank_sorted = cumulative_unique - reset_offsets
            shell_k_within_upair[order_ud] = rank_sorted.astype(np.int32)

        # Legacy global shell rank, kept for compatibility
        shell_k_rank = np.arange(n_shell, dtype=np.int32)

        # ------------------------------------------------------------------
        # Occupancy terms: row -> (shell, center group, neighbor group).
        #
        # NumPy grouping provides the row->shell inverse directly. CUDA
        # grouping does not, so it is recovered by looking every row key up
        # in the grouped keys.
        # ------------------------------------------------------------------
        t_occ0 = time.perf_counter()

        if inv is not None:
            row_shell = np.asarray(inv, dtype=np.int64).reshape(-1)

        else:
            row_key = np.zeros(len(dists), dtype=uniq.dtype)
            row_key["a"] = lam_a_id
            row_key["b"] = lam_b_id
            row_key["di"] = dist_i
            row_key["dh"] = dh.astype(np.int16)
            row_key["dk"] = dk.astype(np.int16)
            row_key["dl"] = dl.astype(np.int16)

            void_dtype = np.dtype((np.void, uniq.dtype.itemsize))

            uniq_packed = np.ascontiguousarray(uniq).view(void_dtype).reshape(-1)
            row_packed = np.ascontiguousarray(row_key).view(void_dtype).reshape(-1)

            uniq_order = np.argsort(uniq_packed, kind="stable")
            position = np.searchsorted(uniq_packed[uniq_order], row_packed)
            row_shell = uniq_order[position].astype(np.int64)

            del row_key, row_packed

        (
            occ_term_shell,
            occ_term_center_group,
            occ_term_neighbor_group,
            occ_term_weight,
        ) = _build_occupancy_terms(
            row_shell,
            site_occ_group[c_idx],
            site_occ_group[n_idx],
            site_orbit_size[c_idx],
            len(self.occupancy_group_keys),
        )

        del row_shell

        print(
            f"[SHELLS] Occupancy terms: {occ_term_shell.size:,} term(s) "
            f"for {n_shell:,} shells, "
            f"{len(self.occupancy_group_keys):,} occupancy group(s) "
            f"in {time.perf_counter() - t_occ0:.2f} s"
        )

        # ------------------------------------------------------------------
        # Sort all shell arrays by ordered pair ID.
        # This is required by pair_offsets and the Numba accumulation kernels.
        # ------------------------------------------------------------------
        order = np.argsort(shell_pair_id, kind="mergesort")

        shell_frac = shell_frac[order]
        shell_mult = shell_mult[order]
        shell_mult_eff = shell_mult_eff[order]
        shell_pair_id = shell_pair_id[order]
        shell_alpha_id = shell_alpha_id[order]
        shell_beta_id = shell_beta_id[order]
        shell_hkl = shell_hkl[order]
        shell_dir = shell_dir[order]
        shell_k_rank = shell_k_rank[order]
        shell_upair_id = shell_upair_id[order]
        shell_k_within_upair = shell_k_within_upair[order]

        # Remap occupancy terms to the sorted shell order, then sort the
        # terms by shell for memory locality.
        shell_rank = np.empty(n_shell, dtype=np.int64)
        shell_rank[order] = np.arange(n_shell, dtype=np.int64)

        occ_term_shell = shell_rank[occ_term_shell].astype(np.int32)

        term_order = np.argsort(occ_term_shell, kind="stable")

        occ_term_shell = occ_term_shell[term_order]
        occ_term_center_group = occ_term_center_group[term_order]
        occ_term_neighbor_group = occ_term_neighbor_group[term_order]
        occ_term_weight = occ_term_weight[term_order]

        # Ungrouped movable rows: every such shell has exactly one row, its
        # representative.
        if has_dyn:
            dyn_shell_mask = dyn_row[rep]
            dyn_rows = rep[dyn_shell_mask]

            frac_all = np.asarray(
                self.structure.frac_coords,
                dtype=np.float64,
            )

            dyn_center = np.asarray(c_idx, dtype=np.int64)[dyn_rows]
            dyn_neighbor = np.asarray(n_idx, dtype=np.int64)[dyn_rows]

            dyn_arrays = {
                "dyn_shell": shell_rank[
                    np.flatnonzero(dyn_shell_mask)
                ].astype(np.int32),
                "dyn_center_site": dyn_center.astype(np.int32),
                "dyn_neighbor_site": dyn_neighbor.astype(np.int32),
                "dyn_base_fvec": (
                    frac_all[dyn_neighbor]
                    + np.asarray(images, dtype=np.float64)[dyn_rows]
                    - frac_all[dyn_center]
                ),
            }
        else:
            dyn_arrays = empty_dyn_arrays

        # Pair offsets
        n_pairs = int(len(pair_list))
        counts_pair = np.bincount(shell_pair_id, minlength=n_pairs).astype(np.int32)

        pair_offsets = np.zeros(n_pairs + 1, dtype=np.int32)
        pair_offsets[1:] = np.cumsum(counts_pair)

        # Pair alpha/beta element ids for ordered Faber-Ziman weights.
        # Use first shell row belonging to each pair.
        pair_alpha_id = np.zeros(n_pairs, dtype=np.int16)
        pair_beta_id = np.zeros(n_pairs, dtype=np.int16)

        for pid in range(n_pairs):
            rows_pid = np.where(shell_pair_id == pid)[0]
            if rows_pid.size > 0:
                first_row = int(rows_pid[0])
                pair_alpha_id[pid] = shell_alpha_id[first_row]
                pair_beta_id[pid] = shell_beta_id[first_row]

        if has_adp:
            adp_arrays = {
                "shell_adp_center": row_adp_c[rep][order].astype(np.int32),
                "shell_adp_neighbor": row_adp_n[rep][order].astype(np.int32),
                "adp_rotations": np.asarray(self.adp_rotations, dtype=np.float64).reshape(-1, 3, 3),
                "adp_entry_group": np.asarray(self.adp_entry_group, dtype=np.int32),
            }
        else:
            adp_arrays = {
                "shell_adp_center": np.zeros((0,), dtype=np.int32),
                "shell_adp_neighbor": np.zeros((0,), dtype=np.int32),
                "adp_rotations": np.zeros((0, 3, 3), dtype=np.float64),
                "adp_entry_group": np.zeros((0,), dtype=np.int32),
            }

        shell_table = {
            **adp_arrays,
            "shell_frac": shell_frac,
            "shell_mult": shell_mult,
            "shell_mult_eff": shell_mult_eff,
            "shell_pair_id": shell_pair_id,
            "shell_alpha_id": shell_alpha_id,
            "shell_beta_id": shell_beta_id,
            "shell_hkl": shell_hkl,
            "shell_direction": shell_dir,
            "shell_k_rank": shell_k_rank,
            "shell_upair_id": shell_upair_id,
            "shell_k_within_upair": shell_k_within_upair,
            "pair_offsets": pair_offsets,
            "pair_alpha_id": pair_alpha_id,
            "pair_beta_id": pair_beta_id,
            "unique_species": unique_species,
            "pair_list": pair_list,
            "upair_list": upair_list,
            **occupancy_site_arrays,
            **dyn_arrays,
            "occ_term_shell": occ_term_shell,
            "occ_term_center_group": occ_term_center_group,
            "occ_term_neighbor_group": occ_term_neighbor_group,
            "occ_term_weight": occ_term_weight,
        }

        # Cache unique strain directions and inverse map.
        # This avoids expensive np.unique(axis=0) after loading cache.
        try:
            dirs_i32 = shell_table["shell_direction"].astype(np.int32, copy=False)
            if (
                getattr(
                    self,
                    "shell_backend_active",
                    "cpu",
                )
                == "cuda"
            ):
                try:
                    from pdf_fitting.models.gpu.shell_grouping import (
                        unique_int3_cuda,
                    )

                    t_direction_gpu = time.perf_counter()

                    (
                        strain_unique_dirs,
                        strain_inv_dirs,
                    ) = unique_int3_cuda(
                        dirs_i32,
                        device_id=int(
                            self.gpu_device
                        ),
                        memory_fraction=float(
                            self.gpu_memory_fraction
                        ),
                    )

                    print(
                        "[CUDA SHELLS] Unique strain directions: "
                        f"{dirs_i32.shape[0]:,} rows -> "
                        f"{strain_unique_dirs.shape[0]:,} directions "
                        f"in {time.perf_counter() - t_direction_gpu:.2f} s"
                    )

                except Exception as exc:
                    print(
                        "[CUDA SHELLS] GPU unique-direction generation "
                        f"failed; using NumPy: {exc}"
                    )

                    if self.shell_backend_requested in (
                        "cuda",
                        "gpu",
                        "cupy",
                    ):
                        raise

                    (
                        strain_unique_dirs,
                        strain_inv_dirs,
                    ) = np.unique(
                        dirs_i32,
                        axis=0,
                        return_inverse=True,
                    )

            else:
                (
                    strain_unique_dirs,
                    strain_inv_dirs,
                ) = np.unique(
                    dirs_i32,
                    axis=0,
                    return_inverse=True,
                )

            #diagnostics
            #----------------------

            print(
                "[DIRECTION DIAG] "
                f"shell_rows={dirs_i32.shape[0]:,}, "
                f"unique_directions={strain_unique_dirs.shape[0]:,}"
            )
            
            if strain_unique_dirs.shape[0] > 0:
                max_abs_direction = int(
                    np.max(
                        np.abs(
                            strain_unique_dirs.astype(np.int64)
                        )
                    )
                )
            
                print(
                    "[DIRECTION DIAG] "
                    f"maximum absolute direction index={max_abs_direction:,}"
                )
            
                direction_norm = np.sqrt(
                    np.sum(
                        strain_unique_dirs.astype(np.float64) ** 2,
                        axis=1,
                    )
                )
            
                print(
                    "[DIRECTION DIAG] "
                    f"direction norm median={np.median(direction_norm):.3f}, "
                    f"p95={np.percentile(direction_norm, 95):.3f}, "
                    f"maximum={np.max(direction_norm):.3f}"
                )

            #----------------------------

            
            shell_table["strain_unique_dirs"] = strain_unique_dirs.astype(np.int32, copy=False)
            shell_table["strain_inv_dirs"] = strain_inv_dirs.astype(np.int32, copy=False)

            print(
                f"[SHELLS] Cached {len(strain_unique_dirs)} unique strain direction(s)."
            )
        except Exception as e:
            print(f"[WARN] Could not cache strain directions: {e}")
            shell_table["strain_unique_dirs"] = np.zeros((0, 3), dtype=np.int32)
            shell_table["strain_inv_dirs"] = np.zeros((0,), dtype=np.int32)

        t_table1 = time.perf_counter()
        print(
            f"[SHELLS] Built packed shell table directly "
            f"({n_shell} shells) in {t_table1 - t_table0:.2f} s"
        )

        # Return no legacy shell list.
        return [], shell_table

    def _shell_table_to_legacy_dicts(self):
        if self._shell_table is None:
            return []
        st = self._shell_table
        inv_order = np.argsort(st["shell_k_rank"], kind="mergesort")
        unique_species = st["unique_species"]
        pair_list = st["pair_list"]
        shells = []
        for idx in inv_order:
            pid = int(st["shell_pair_id"][idx])
            a_sp, b_sp = pair_list[pid]
            shells.append(
                {
                    "fractional_vector": [float(x) for x in st["shell_frac"][idx]],
                    "distance": float(idx),
                    "pair_type": f"{a_sp}-{b_sp}",
                    "coordination": int(st["shell_mult"][idx]),
                    "hkl": [int(x) for x in st["shell_hkl"][idx]],
                    "direction": [int(x) for x in st["shell_direction"][idx]],
                }
            )
        return shells

    def get_symmetry_shells(self):
        return self.shells

    def get_shell_table_arrays(self):
        return self._shell_table

    def get_constraints(self):
        return self.constraints

    def get_initial_occupancy_parameters(self) -> dict:
        """
        Return CIF-derived occupancy parameters, one per occupancy group.

        Example:
            {"occ_n1": 0.575, "occ_c7": 0.575, "occ_n2": 0.425, ...}
        """
        return {
            name: float(occupancy)
            for name, occupancy in zip(
                self.occupancy_param_names,
                self.occupancy_group_base,
            )
        }

    def get_group_occupancies(self, params: dict | None = None) -> np.ndarray:
        """
        Return the active occupancy of every occupancy group.

        A group without an occ_<key> value in params keeps its CIF
        occupancy. Values are clipped to [0, 1].
        """
        occupancies = np.asarray(
            self.occupancy_group_base,
            dtype=np.float64,
        ).copy()

        if not params:
            return occupancies

        lookup = {
            str(k).strip().lower(): v
            for k, v in params.items()
            if str(k).strip().lower().startswith("occ_")
        }

        if not lookup:
            return occupancies

        for i, name in enumerate(self.occupancy_param_names):
            value = lookup.get(name)

            if value is None:
                continue

            try:
                occupancies[i] = float(np.clip(float(value), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

        return occupancies

    def get_site_occupancies(self, params: dict | None = None) -> np.ndarray:
        """
        Return the active occupancy for every structure site.
        """
        return self.get_group_occupancies(params)[
            np.asarray(self.site_occupancy_group, dtype=np.int64)
        ]

    def occupancy_parameters_for_elements(self, elements) -> dict:
        """
        Return {occ_<key>: CIF occupancy} for every occupancy group whose
        element is in ``elements`` (case-insensitive element symbols).
        """
        wanted = {
            clean_element_name(str(e)).lower()
            for e in (elements or [])
            if str(e).strip()
        }

        result = {}

        for name, element, base in zip(
            self.occupancy_param_names,
            self.occupancy_group_elements,
            self.occupancy_group_base,
        ):
            if str(element).lower() in wanted:
                result[name] = float(base)

        return result
