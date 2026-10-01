# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import Dict, Any, Tuple

import numpy as np

from pymatgen.core import Structure

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
    shape_mask,
)


try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None
    prange = range

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _igcd_numba(a: int, b: int) -> int:
        a = abs(a)
        b = abs(b)

        while b != 0:
            a, b = b, a % b

        return a


    @njit(cache=True)
    def _igcd3_numba(a: int, b: int, c: int) -> int:
        return _igcd_numba(_igcd_numba(a, b), c)


    @njit(cache=True, fastmath=True, parallel=True)
    def _count_catalog_rows_numba(
        translations,
        frac_sites,
        lattice_matrix,
        pair_cutoff,
        tol,
    ):
        """
        Count all valid all-site pair rows per translation.

        Parallel over periodic image translations.
        """
        n_trans = translations.shape[0]
        n_sites = frac_sites.shape[0]

        counts = np.zeros(n_trans, dtype=np.int64)

        r2max = float(pair_cutoff) * float(pair_cutoff)
        tol2 = float(tol) * float(tol)

        for it in prange(n_trans):
            tx = float(translations[it, 0])
            ty = float(translations[it, 1])
            tz = float(translations[it, 2])

            cnt = 0

            for i in range(n_sites):
                fi0 = frac_sites[i, 0]
                fi1 = frac_sites[i, 1]
                fi2 = frac_sites[i, 2]

                for j in range(n_sites):
                    f0 = frac_sites[j, 0] + tx - fi0
                    f1 = frac_sites[j, 1] + ty - fi1
                    f2 = frac_sites[j, 2] + tz - fi2

                    x = (
                        f0 * lattice_matrix[0, 0]
                        + f1 * lattice_matrix[1, 0]
                        + f2 * lattice_matrix[2, 0]
                    )
                    y = (
                        f0 * lattice_matrix[0, 1]
                        + f1 * lattice_matrix[1, 1]
                        + f2 * lattice_matrix[2, 1]
                    )
                    z = (
                        f0 * lattice_matrix[0, 2]
                        + f1 * lattice_matrix[1, 2]
                        + f2 * lattice_matrix[2, 2]
                    )

                    d2 = x * x + y * y + z * z

                    if d2 <= r2max and d2 > tol2:
                        cnt += 1

            counts[it] = cnt

        return counts


    @njit(cache=True, fastmath=True, parallel=True)
    def _fill_catalog_rows_numba(
        translations,
        offsets,
        frac_sites,
        lattice_matrix,
        inv_recip_T,
        site_species_id,
        site_lambda_id,
        pair_cutoff,
        tol,
    ):
        """
        Fill all valid all-site pair rows.

        Parallel over periodic image translations. Each translation writes into
        its own precomputed slice offsets[it]:offsets[it+1], so there are no races.
        """
        n_trans = translations.shape[0]
        n_sites = frac_sites.shape[0]

        total = int(offsets[-1])

        center_site = np.empty(total, dtype=np.int32)
        neighbor_site = np.empty(total, dtype=np.int32)
        image = np.empty((total, 3), dtype=np.int32)
        frac = np.empty((total, 3), dtype=np.float32)
        dist_i = np.empty(total, dtype=np.int32)
        hkl = np.empty((total, 3), dtype=np.int16)
        direction = np.empty((total, 3), dtype=np.int16)

        alpha_id = np.empty(total, dtype=np.int16)
        beta_id = np.empty(total, dtype=np.int16)

        lam_a = np.empty(total, dtype=np.int32)
        lam_b = np.empty(total, dtype=np.int32)

        r2max = float(pair_cutoff) * float(pair_cutoff)
        tol2 = float(tol) * float(tol)

        for it in prange(n_trans):
            tx_i = translations[it, 0]
            ty_i = translations[it, 1]
            tz_i = translations[it, 2]

            tx = float(tx_i)
            ty = float(ty_i)
            tz = float(tz_i)

            pos = offsets[it]

            for i in range(n_sites):
                fi0 = frac_sites[i, 0]
                fi1 = frac_sites[i, 1]
                fi2 = frac_sites[i, 2]

                for j in range(n_sites):
                    f0 = frac_sites[j, 0] + tx - fi0
                    f1 = frac_sites[j, 1] + ty - fi1
                    f2 = frac_sites[j, 2] + tz - fi2

                    x = (
                        f0 * lattice_matrix[0, 0]
                        + f1 * lattice_matrix[1, 0]
                        + f2 * lattice_matrix[2, 0]
                    )
                    y = (
                        f0 * lattice_matrix[0, 1]
                        + f1 * lattice_matrix[1, 1]
                        + f2 * lattice_matrix[2, 1]
                    )
                    z = (
                        f0 * lattice_matrix[0, 2]
                        + f1 * lattice_matrix[1, 2]
                        + f2 * lattice_matrix[2, 2]
                    )

                    d2 = x * x + y * y + z * z

                    if d2 <= r2max and d2 > tol2:
                        dist = np.sqrt(d2)

                        center_site[pos] = i
                        neighbor_site[pos] = j

                        image[pos, 0] = tx_i
                        image[pos, 1] = ty_i
                        image[pos, 2] = tz_i

                        frac[pos, 0] = np.float32(f0)
                        frac[pos, 1] = np.float32(f1)
                        frac[pos, 2] = np.float32(f2)

                        dist_i[pos] = np.int32(np.rint(dist * 1.0e5))

                        # hkl_float = cart @ inv_recip_T.T
                        hf0 = (
                            x * inv_recip_T[0, 0]
                            + y * inv_recip_T[0, 1]
                            + z * inv_recip_T[0, 2]
                        )
                        hf1 = (
                            x * inv_recip_T[1, 0]
                            + y * inv_recip_T[1, 1]
                            + z * inv_recip_T[1, 2]
                        )
                        hf2 = (
                            x * inv_recip_T[2, 0]
                            + y * inv_recip_T[2, 1]
                            + z * inv_recip_T[2, 2]
                        )

                        hi = int(np.rint(hf0))
                        ki = int(np.rint(hf1))
                        li = int(np.rint(hf2))

                        hkl[pos, 0] = np.int16(hi)
                        hkl[pos, 1] = np.int16(ki)
                        hkl[pos, 2] = np.int16(li)

                        g = _igcd3_numba(hi, ki, li)

                        if g == 0:
                            dh = 0
                            dk = 0
                            dl = 0
                        else:
                            dh = hi // g
                            dk = ki // g
                            dl = li // g

                            # first nonzero positive
                            sign = 1

                            if dh != 0:
                                sign = 1 if dh > 0 else -1
                            elif dk != 0:
                                sign = 1 if dk > 0 else -1
                            elif dl != 0:
                                sign = 1 if dl > 0 else -1

                            dh *= sign
                            dk *= sign
                            dl *= sign

                        direction[pos, 0] = np.int16(dh)
                        direction[pos, 1] = np.int16(dk)
                        direction[pos, 2] = np.int16(dl)

                        alpha_id[pos] = site_species_id[i]
                        beta_id[pos] = site_species_id[j]

                        lam_a[pos] = site_lambda_id[i]
                        lam_b[pos] = site_lambda_id[j]

                        pos += 1

        return (
            center_site,
            neighbor_site,
            image,
            frac,
            dist_i,
            hkl,
            direction,
            alpha_id,
            beta_id,
            lam_a,
            lam_b,
        )

else:
    _count_catalog_rows_numba = None
    _fill_catalog_rows_numba = None

if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _count_overlap_by_unique_image_numba(
        cell_inside,
        unique_images,
        grid_n,
    ):
        """
        Count overlap for each unique image translation.

        Complete-cell mode:
            count(image) = number of cells C such that
                           C is inside and C + image is inside.

        Cell indexing must match build_cell_mask_for_shape():
            cell_trans = [(i,j,k) for i in vals for j in vals for k in vals]

        Therefore flat index is:
            idx = (ix * nx + iy) * nx + iz
        """
        n_img = unique_images.shape[0]

        nx = 2 * grid_n + 1

        out = np.zeros(n_img, dtype=np.int64)

        for u in prange(n_img):
            dx = unique_images[u, 0]
            dy = unique_images[u, 1]
            dz = unique_images[u, 2]

            cnt = 0

            for ix in range(nx):
                ix2 = ix + dx

                if ix2 < 0 or ix2 >= nx:
                    continue

                for iy in range(nx):
                    iy2 = iy + dy

                    if iy2 < 0 or iy2 >= nx:
                        continue

                    for iz in range(nx):
                        iz2 = iz + dz

                        if iz2 < 0 or iz2 >= nx:
                            continue

                        idx1 = (ix * nx + iy) * nx + iz
                        idx2 = (ix2 * nx + iy2) * nx + iz2

                        if cell_inside[idx1] != 0 and cell_inside[idx2] != 0:
                            cnt += 1

            out[u] = cnt

        return out
else:
    _count_overlap_by_unique_image_numba = None


from pdf_fitting.user_paths import user_cache_dir

FINITE_CACHE_DIR = user_cache_dir(
    "finite_shape_cache"
)

FINITE_CATALOG_CACHE_VERSION = "finite_pair_catalog_v4_image_map"
FINITE_SHELL_CACHE_VERSION = "finite_shape_shell_table_v3_direct_atom_coordination_with_gamma"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def clean_element_name(name: str) -> str:
    m = re.match(r"([A-Za-z]{1,2})", str(name))
    if not m:
        raise ValueError(f"Could not parse chemical symbol from '{name}'")
    sym = m.group(1)
    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()


def lambda_safe_species_label(name: str) -> str:
    s = str(name).strip()
    s = s.replace("+", "plus")
    s = s.replace("-", "minus")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.lower()


def _hash_structure_for_catalog(structure: Structure, pair_cutoff: float) -> str:
    """
    Hash only topology + fractional coordinates + cutoff.

    Lattice values are intentionally not included because pair vectors are stored
    in fractional coordinates. Distances are recomputed later from refined lattice.
    """
    atoms_str = ";".join(
        f"{site.species_string},{tuple(np.round(site.frac_coords, 12))}"
        for site in structure.sites
    )

    raw = (
        f"{FINITE_CATALOG_CACHE_VERSION}|"
        f"atoms={atoms_str}|"
        f"pair_cutoff={float(pair_cutoff):.8f}"
    )

    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _hash_shape(shape_spec: CrystalliteShapeSpec, extra: Dict[str, Any] | None = None) -> str:
    d = shape_spec.to_dict()
    if extra:
        d.update(extra)

    raw = json.dumps(
        d,
        sort_keys=True,
        default=lambda x: str(x),
    )

    raw = f"{FINITE_SHELL_CACHE_VERSION}|{raw}"

    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _igcd3_py(a: int, b: int, c: int) -> int:
    import math
    return math.gcd(math.gcd(abs(int(a)), abs(int(b))), abs(int(c)))


def _reduce_direction_py(h: int, k: int, l: int) -> Tuple[int, int, int]:
    h = int(h)
    k = int(k)
    l = int(l)

    g = _igcd3_py(h, k, l)

    if g == 0:
        return (0, 0, 0)

    h //= g
    k //= g
    l //= g

    # first nonzero positive
    for v in (h, k, l):
        if v != 0:
            if v < 0:
                h, k, l = -h, -k, -l
            break

    return (h, k, l)


# -----------------------------------------------------------------------------
# Pair-vector catalogue
# -----------------------------------------------------------------------------

def _build_catalog_rows_parallel_numba(
    *,
    translations: np.ndarray,
    frac_sites: np.ndarray,
    lattice_matrix: np.ndarray,
    inv_recip_T: np.ndarray,
    site_species_id: np.ndarray,
    site_lambda_id: np.ndarray,
    pair_cutoff: float,
):
    """
    Build all-site pair catalogue rows using Numba parallel kernels.
    """
    if _count_catalog_rows_numba is None or _fill_catalog_rows_numba is None:
        raise RuntimeError("Numba catalogue kernels are unavailable.")

    translations = np.asarray(translations, dtype=np.int32)
    frac_sites = np.asarray(frac_sites, dtype=np.float64)
    lattice_matrix = np.asarray(lattice_matrix, dtype=np.float64)
    inv_recip_T = np.asarray(inv_recip_T, dtype=np.float64)
    site_species_id = np.asarray(site_species_id, dtype=np.int16)
    site_lambda_id = np.asarray(site_lambda_id, dtype=np.int32)

    t0 = time.perf_counter()

    counts = _count_catalog_rows_numba(
        translations,
        frac_sites,
        lattice_matrix,
        float(pair_cutoff),
        1e-8,
    )

    t1 = time.perf_counter()

    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    total = int(offsets[-1])

    print(
        f"[FINITE] Catalogue count pass: {total} rows "
        f"in {t1 - t0:.2f} s"
    )

    if total <= 0:
        raise RuntimeError("Finite pair catalogue has no rows. Check pair_cutoff/structure.")

    out = _fill_catalog_rows_numba(
        translations,
        offsets,
        frac_sites,
        lattice_matrix,
        inv_recip_T,
        site_species_id,
        site_lambda_id,
        float(pair_cutoff),
        1e-8,
    )

    t2 = time.perf_counter()

    print(
        f"[FINITE] Catalogue fill pass: {total} rows "
        f"in {t2 - t1:.2f} s"
    )

    return out



def get_or_build_all_site_pair_catalog(
    structure: Structure,
    pair_cutoff: float,
    *,
    cache_dir: str = FINITE_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Generate/load an all-site ungrouped infinite pair-vector catalogue.

    This is deeper than the current grouped shell cache.

    It contains one row per all-site pair vector:
        center_site i
        neighbor_site j
        image translation ΔT
        fractional pair vector
        direction/hkl/pair metadata

    This catalogue is independent of crystallite shape.
    """
    os.makedirs(cache_dir, exist_ok=True)

    key = _hash_structure_for_catalog(structure, float(pair_cutoff))

    npz_path = os.path.join(cache_dir, f"finite_pair_catalog_{key}.npz")
    meta_path = os.path.join(cache_dir, f"finite_pair_catalog_{key}.json")

    if os.path.exists(npz_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            z = np.load(npz_path, allow_pickle=False)

            catalog = {
                "center_site": z["center_site"],
                "neighbor_site": z["neighbor_site"],
                "image": z["image"],
                "frac": z["frac"],
                "dist_i": z["dist_i"],
                "hkl": z["hkl"],
                "direction": z["direction"],
                "pair_id": z["pair_id"],
                "upair_id": z["upair_id"],
                "alpha_id": z["alpha_id"],
                "beta_id": z["beta_id"],
                "pair_alpha_id": z["pair_alpha_id"],
                "pair_beta_id": z["pair_beta_id"],
                "unique_species": list(meta["unique_species"]),
                "pair_list": [tuple(x) for x in meta["pair_list"]],
                "upair_list": [tuple(x) for x in meta["upair_list"]],
                "species_counts": np.asarray(meta["species_counts"], dtype=np.int64),
                "site_species_id": z["site_species_id"],
                "site_lambda_id": z["site_lambda_id"],
                "catalog_hash": key,
                "pair_cutoff": float(meta["pair_cutoff"]),
            }

            if "unique_images" in z.files and "image_inv" in z.files:
                catalog["unique_images"] = z["unique_images"]
                catalog["image_inv"] = z["image_inv"]
            else:
                unique_images, image_inv = np.unique(
                    catalog["image"],
                    axis=0,
                    return_inverse=True,
                )
                catalog["unique_images"] = unique_images.astype(np.int32, copy=False)
                catalog["image_inv"] = image_inv.astype(np.int32, copy=False)

            print(f"[FINITE CACHE] Loaded all-site pair catalogue: {npz_path}")
            return catalog

        except Exception:
            pass

    print("[FINITE] Building all-site pair-vector catalogue...")
    t0 = time.perf_counter()

    lattice = structure.lattice
    M = np.asarray(lattice.matrix, dtype=float)

    frac_sites = np.asarray(structure.frac_coords, dtype=float)
    n_sites = int(frac_sites.shape[0])

    species_element_all = [clean_element_name(str(s.specie)) for s in structure.sites]
    unique_species = sorted(set(species_element_all))
    sp_index = {s: i for i, s in enumerate(unique_species)}
    site_species_id = np.asarray([sp_index[s] for s in species_element_all], dtype=np.int16)

    species_counts = np.bincount(
        site_species_id.astype(np.int64),
        minlength=len(unique_species),
    ).astype(np.int64)

    species_lambda_all = [lambda_safe_species_label(str(s.specie)) for s in structure.sites]
    unique_lambda_species = sorted(set(species_lambda_all))
    lambda_sp_index = {s: i for i, s in enumerate(unique_lambda_species)}
    site_lambda_id = np.asarray([lambda_sp_index[s] for s in species_lambda_all], dtype=np.int16)

    # Image translations large enough for pair_cutoff.
    try:
        rec = lattice.reciprocal_lattice_crystallographic
        heights = 1.0 / np.maximum(np.asarray(rec.abc, dtype=float), 1e-12)
    except Exception:
        heights = np.maximum(np.asarray(lattice.abc, dtype=float), 1e-12)

    nmax = np.ceil(float(pair_cutoff) / heights).astype(int) + 1
    na, nb, nc = [int(x) for x in nmax]

    translations = np.array(
        [
            (i, j, k)
            for i in range(-na, na + 1)
            for j in range(-nb, nb + 1)
            for k in range(-nc, nc + 1)
        ],
        dtype=np.int32,
    )

    # hkl/direction helper
    recip_T = np.asarray(lattice.reciprocal_lattice.matrix.T, dtype=np.float64)
    inv_recip_T = np.linalg.inv(recip_T)

    if _count_catalog_rows_numba is not None and _fill_catalog_rows_numba is not None:
        (
            center_site,
            neighbor_site,
            image,
            frac,
            dist_i,
            hkl,
            direction,
            alpha_id,
            beta_id,
            lam_a,
            lam_b,
        ) = _build_catalog_rows_parallel_numba(
            translations=translations,
            frac_sites=frac_sites,
            lattice_matrix=M,
            inv_recip_T=inv_recip_T,
            site_species_id=site_species_id,
            site_lambda_id=site_lambda_id,
            pair_cutoff=float(pair_cutoff),
        )

    else:
        # Slow pure-Python fallback
        center_rows = []
        neighbor_rows = []
        image_rows = []
        frac_rows = []
        dist_rows = []

        r2max = float(pair_cutoff) * float(pair_cutoff)
        tol2 = 1e-16

        for t in translations:
            t_float = t.astype(float)

            for i in range(n_sites):
                fi = frac_sites[i]

                fvec_all_j = frac_sites + t_float[None, :] - fi[None, :]
                cart_all_j = fvec_all_j @ M
                d2 = np.einsum("ij,ij->i", cart_all_j, cart_all_j)

                valid = (d2 <= r2max) & (d2 > tol2)
                js = np.where(valid)[0]

                for j in js:
                    center_rows.append(i)
                    neighbor_rows.append(int(j))
                    image_rows.append(t.copy())
                    frac_rows.append(fvec_all_j[j].astype(np.float32))
                    dist_rows.append(float(np.sqrt(d2[j])))

        if not center_rows:
            raise RuntimeError("Finite pair catalogue has no rows. Check pair_cutoff/structure.")

        center_site = np.asarray(center_rows, dtype=np.int32)
        neighbor_site = np.asarray(neighbor_rows, dtype=np.int32)
        image = np.asarray(image_rows, dtype=np.int32)
        frac = np.asarray(frac_rows, dtype=np.float32)
        dists = np.asarray(dist_rows, dtype=np.float64)
        dist_i = np.rint(dists * 1e5).astype(np.int32)

        cart = frac.astype(np.float64) @ M
        hkl_float = cart @ inv_recip_T.T
        hkl = np.rint(hkl_float).astype(np.int32)

        direction = np.zeros_like(hkl, dtype=np.int16)

        for r in range(hkl.shape[0]):
            dh, dk, dl = _reduce_direction_py(hkl[r, 0], hkl[r, 1], hkl[r, 2])
            direction[r, 0] = np.int16(dh)
            direction[r, 1] = np.int16(dk)
            direction[r, 2] = np.int16(dl)

        alpha_id = site_species_id[center_site].astype(np.int16)
        beta_id = site_species_id[neighbor_site].astype(np.int16)

        lam_a = site_lambda_id[center_site].astype(np.int32)
        lam_b = site_lambda_id[neighbor_site].astype(np.int32)


    t_img0 = time.perf_counter()

    unique_images, image_inv = np.unique(
        image,
        axis=0,
        return_inverse=True,
    )

    unique_images = unique_images.astype(np.int32, copy=False)
    image_inv = image_inv.astype(np.int32, copy=False)

    print(
        f"[FINITE] Precomputed image mapping: "
        f"{image.shape[0]:,} rows -> {unique_images.shape[0]:,} unique images "
        f"in {time.perf_counter() - t_img0:.2f} s"
    )
    n_lam_species = int(len(unique_lambda_species))

    pair_code = lam_a * n_lam_species + lam_b
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

    code_to_pair_id = np.full(n_lam_species * n_lam_species, -1, dtype=np.int32)
    for pid, code in enumerate(pair_code_order):
        code_to_pair_id[int(code)] = int(pid)

    pair_id = code_to_pair_id[pair_code].astype(np.int32)

    lam_min = np.minimum(lam_a, lam_b)
    lam_max = np.maximum(lam_a, lam_b)
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

    upair_id = code_to_upair_id[upair_code].astype(np.int32)

    n_pairs = len(pair_list)
    pair_alpha_id = np.zeros(n_pairs, dtype=np.int16)
    pair_beta_id = np.zeros(n_pairs, dtype=np.int16)

    for pid in range(n_pairs):
        rows = np.where(pair_id == pid)[0]
        if rows.size:
            r0 = int(rows[0])
            pair_alpha_id[pid] = alpha_id[r0]
            pair_beta_id[pid] = beta_id[r0]

    catalog = {
    "center_site": center_site,
    "neighbor_site": neighbor_site,
    "image": image,
    "frac": frac,
    "dist_i": dist_i,
    "hkl": hkl.astype(np.int16),
    "direction": direction,
    "pair_id": pair_id,
    "upair_id": upair_id,
    "alpha_id": alpha_id,
    "beta_id": beta_id,
    "pair_alpha_id": pair_alpha_id,
    "pair_beta_id": pair_beta_id,
    "unique_species": unique_species,
    "pair_list": pair_list,
    "upair_list": upair_list,
    "species_counts": species_counts,
    "site_species_id": site_species_id,
    "site_lambda_id": site_lambda_id,
    "catalog_hash": key,
    "pair_cutoff": float(pair_cutoff),
    "unique_images": unique_images,
    "image_inv": image_inv,
    }


    np.savez(
        npz_path,
        center_site=center_site,
        neighbor_site=neighbor_site,
        image=image,
        frac=frac,
        dist_i=dist_i,
        hkl=hkl.astype(np.int16),
        direction=direction,
        pair_id=pair_id,
        upair_id=upair_id,
        alpha_id=alpha_id,
        beta_id=beta_id,
        pair_alpha_id=pair_alpha_id,
        pair_beta_id=pair_beta_id,
        site_species_id=site_species_id,
        site_lambda_id=site_lambda_id,
        unique_images=unique_images,
        image_inv=image_inv,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unique_species": unique_species,
                "pair_list": [list(x) for x in pair_list],
                "upair_list": [list(x) for x in upair_list],
                "species_counts": [int(x) for x in species_counts],
                "pair_cutoff": float(pair_cutoff),
            },
            f,
            indent=2,
        )

    print(
        f"[FINITE CACHE] Saved all-site pair catalogue with {len(center_site)} rows "
        f"in {time.perf_counter() - t0:.2f} s"
    )

    return catalog


# -----------------------------------------------------------------------------
# Finite cell mask
# -----------------------------------------------------------------------------
def _copy_shape_spec(spec: CrystalliteShapeSpec) -> CrystalliteShapeSpec:
    return CrystalliteShapeSpec.from_dict(spec.to_dict())


def build_cell_mask_for_shape(
    structure: Structure,
    shape_spec: CrystalliteShapeSpec,
    *,
    boundary_mode: str = "atom",
) -> Dict[str, Any]:
    """
    Build finite-shape inclusion mask.

    boundary_mode = "atom"
        Each atom in each translated unit cell is tested independently against
        the finite shape. This is the recommended physically meaningful mode.

    boundary_mode = "cell"
        Complete unit-cell mode. A whole unit cell is included/excluded based
        on the translated unit-cell origin. Kept only for backward compatibility.

    The returned site_cell_mask has shape:

        (n_sites, n_cells)

    and tells whether a specific atom site in a specific translated cell is
    inside the finite crystallite.
    """
    boundary_mode = str(boundary_mode).strip().lower()

    if boundary_mode not in ("atom", "cell"):
        raise NotImplementedError(
            "boundary_mode must be either 'atom' or 'cell'."
        )

    if not np.isfinite(float(shape_spec.diameter_cells)):
        raise NotImplementedError(
            "Exact infinite-diameter slab mode is not implemented in the CPU backend yet. "
            "Please scan a finite diameter first."
        )

    M = np.asarray(structure.lattice.matrix, dtype=float)

    max_dim = max(
        float(shape_spec.diameter_cells),
        float(shape_spec.height_cells),
        1.0,
    )

    n = int(np.ceil(max_dim)) + 3

    vals = np.arange(-n, n + 1, dtype=np.int32)

    cell_trans = np.asarray(
        [(i, j, k) for i in vals for j in vals for k in vals],
        dtype=np.int32,
    )

    n_sites = len(structure.sites)
    n_cells = int(cell_trans.shape[0])

    site_cell_mask = np.zeros((n_sites, n_cells), dtype=np.uint8)

    frac_sites = np.asarray(structure.frac_coords, dtype=float)

    if boundary_mode == "cell":
        cell_cart = cell_trans.astype(float) @ M
        cell_inside = shape_mask(cell_cart, shape_spec, M).astype(np.bool_)

        if np.any(cell_inside):
            site_cell_mask[:, cell_inside] = 1

    else:
        cell_inside = np.zeros(n_cells, dtype=np.bool_)

        for isite in range(n_sites):
            atom_frac = cell_trans.astype(float) + frac_sites[isite][None, :]
            atom_cart = atom_frac @ M

            inside = shape_mask(atom_cart, shape_spec, M).astype(np.bool_)

            site_cell_mask[isite, inside] = 1
            cell_inside |= inside

    site_species = np.asarray(
        [clean_element_name(str(site.specie)) for site in structure.sites],
        dtype=object,
    )

    unique_species = sorted(set(site_species.tolist()))
    sp_index = {s: i for i, s in enumerate(unique_species)}
    site_species_id = np.asarray([sp_index[s] for s in site_species], dtype=np.int16)

    n_alpha_inside = np.zeros(len(unique_species), dtype=np.int64)

    for isite in range(n_sites):
        sid = int(site_species_id[isite])
        n_alpha_inside[sid] += int(np.sum(site_cell_mask[isite]))

    return {
        "boundary_mode": boundary_mode,
        "grid_n": int(n),
        "cell_trans": cell_trans,
        "cell_inside": cell_inside,
        "site_cell_mask": site_cell_mask,
        "n_cells_inside": int(np.sum(cell_inside)),
        "n_alpha_inside": n_alpha_inside,
        "unique_species_mask": unique_species,
    }

# -----------------------------------------------------------------------------
# Numba finite coordination counter
# -----------------------------------------------------------------------------
if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _count_finite_coordination_numba(
        site_cell_mask,
        center_site,
        neighbor_site,
        image,
        grid_n,
    ):
        """
        Count finite-shape coordination for every catalogue row using atom-level
        inclusion.

        For each catalogue row:
            center atom type = center_site[row]
            neighbor atom type = neighbor_site[row]
            image translation = image[row]

        Count how many center atoms are inside the finite shape and have the
        corresponding neighbor atom also inside the finite shape.

        The flat cell index follows the order used by build_cell_mask_for_shape():

            cell_trans = [(i, j, k) for i in vals for j in vals for k in vals]

        Therefore:

            idx = (ix * nx + iy) * nx + iz
        """
        n_rows = center_site.size
        nx = 2 * grid_n + 1

        counts = np.zeros(n_rows, dtype=np.int64)

        for row in prange(n_rows):
            si = center_site[row]
            sj = neighbor_site[row]

            dx = image[row, 0]
            dy = image[row, 1]
            dz = image[row, 2]

            cnt = 0

            for ix in range(nx):
                ix2 = ix + dx

                if ix2 < 0 or ix2 >= nx:
                    continue

                for iy in range(nx):
                    iy2 = iy + dy

                    if iy2 < 0 or iy2 >= nx:
                        continue

                    for iz in range(nx):
                        iz2 = iz + dz

                        if iz2 < 0 or iz2 >= nx:
                            continue

                        cell1 = (ix * nx + iy) * nx + iz
                        cell2 = (ix2 * nx + iy2) * nx + iz2

                        if site_cell_mask[si, cell1] != 0 and site_cell_mask[sj, cell2] != 0:
                            cnt += 1

            counts[row] = cnt

        return counts
    
else:
    _count_finite_coordination_numba = None


def count_finite_coordination_cpu(
    catalog: Dict[str, Any],
    mask_data: Dict[str, Any],
) -> np.ndarray:
    """
    Count finite-shape coordination for every catalogue row.

    For boundary_mode='atom':
        atom-level inclusion is used. Counts depend on:
            center_site
            neighbor_site
            image translation

    For boundary_mode='cell':
        old complete-cell overlap counting is used. Counts depend only on image
        translation.
    """
    boundary_mode = str(mask_data.get("boundary_mode", "atom")).strip().lower()

    if boundary_mode == "atom":
        if _count_finite_coordination_numba is None:
            raise RuntimeError("Numba is required for atom-level finite coordination counting.")

        t0 = time.perf_counter()

        counts = _count_finite_coordination_numba(
            np.asarray(mask_data["site_cell_mask"], dtype=np.uint8),
            np.asarray(catalog["center_site"], dtype=np.int32),
            np.asarray(catalog["neighbor_site"], dtype=np.int32),
            np.asarray(catalog["image"], dtype=np.int32),
            int(mask_data["grid_n"]),
        )

        print(
            "[FINITE] Atom-level finite coordination: "
            f"catalog_rows={counts.shape[0]:,}, "
            f"grid_cells={mask_data['site_cell_mask'].shape[1]:,}, "
            f"time={time.perf_counter() - t0:.3f} s"
        )

        return counts.astype(np.int64, copy=False)

    if boundary_mode != "cell":
        raise RuntimeError(f"Unknown finite boundary_mode: {boundary_mode}")

    if _count_overlap_by_unique_image_numba is None:
        raise RuntimeError("Numba is required for finite coordination counting.")

    cell_inside = np.asarray(mask_data["cell_inside"], dtype=np.uint8)
    grid_n = int(mask_data["grid_n"])

    if "unique_images" in catalog and "image_inv" in catalog:
        unique_images = np.asarray(catalog["unique_images"], dtype=np.int32)
        image_inv = np.asarray(catalog["image_inv"], dtype=np.int32)
    else:
        image = np.asarray(catalog["image"], dtype=np.int32)

        t_uni0 = time.perf_counter()

        unique_images, image_inv = np.unique(
            image,
            axis=0,
            return_inverse=True,
        )

        unique_images = unique_images.astype(np.int32, copy=False)
        image_inv = image_inv.astype(np.int32, copy=False)

        print(
            f"[FINITE] WARNING: image mapping was not cached; "
            f"computed in {time.perf_counter() - t_uni0:.3f} s"
        )

    t0 = time.perf_counter()

    overlap_by_image = _count_overlap_by_unique_image_numba(
        cell_inside,
        unique_images,
        grid_n,
    )

    t1 = time.perf_counter()

    counts = overlap_by_image[image_inv].astype(np.int64, copy=False)

    t2 = time.perf_counter()

    print(
        "[FINITE] Cell-level finite coordination by image overlap: "
        f"catalog_rows={image_inv.shape[0]}, "
        f"unique_images={unique_images.shape[0]}, "
        f"grid_cells={cell_inside.size}, "
        f"overlap={t1 - t0:.3f} s, "
        f"scatter={t2 - t1:.3f} s"
    )

    return counts

# -----------------------------------------------------------------------------
# Build PDFCalculator-compatible shell table
# -----------------------------------------------------------------------------

def build_shell_table_from_finite_counts(
    catalog: Dict[str, Any],
    counts: np.ndarray,
    mask_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert finite atom-level pair counts into a packed shell table compatible
    with PDFCalculator.

    The peak multiplicity used in the PDF is the direct finite coordination:

        shell_mult_eff = finite_pair_count / n_alpha_inside

    No artificial clipping is applied to the peak multiplicity.

    Additionally, this function computes a discrete finite-shape background
    factor for each shell:

        shell_gamma = finite_mult_eff / bulk_mult_eff

    where:

        finite_mult_eff = finite_pair_count / n_alpha_inside
        bulk_mult_eff   = bulk_pair_count / n_alpha_bulk

    This shell_gamma is NOT used to scale the peak intensity. It is used later
    to build gamma_finite(r), replacing the infinite-crystal baseline "1" in:

        G(r) = 4*pi*rho0*r*(g(r) - baseline)

    For finite shapes:

        baseline = gamma_finite(r)

    Grouping preserves direction:

        pair_id + distance_bin + reduced direction

    This is important because finite shapes break crystallographic equivalence.
    """
    t0 = time.perf_counter()

    counts = np.asarray(counts, dtype=np.int64)

    valid = counts > 0

    if not np.any(valid):
        raise RuntimeError("Finite shape produced zero finite pair counts.")

    # ---------------------------------------------------------
    # Full catalogue arrays
    # ---------------------------------------------------------
    frac_all = np.asarray(catalog["frac"], dtype=np.float32)
    pair_id_all = np.asarray(catalog["pair_id"], dtype=np.int32)
    upair_id_all = np.asarray(catalog["upair_id"], dtype=np.int32)
    alpha_id_all = np.asarray(catalog["alpha_id"], dtype=np.int16)
    beta_id_all = np.asarray(catalog["beta_id"], dtype=np.int16)
    hkl_all = np.asarray(catalog["hkl"], dtype=np.int16)
    direction_all = np.asarray(catalog["direction"], dtype=np.int16)
    dist_i_all = np.asarray(catalog["dist_i"], dtype=np.int32)

    # ---------------------------------------------------------
    # Key for bulk and finite grouping:
    #   ordered pair + distance bin + reduced direction
    # ---------------------------------------------------------
    key_all = np.zeros(
        pair_id_all.size,
        dtype=[
            ("pair", np.int32),
            ("di", np.int32),
            ("dh", np.int16),
            ("dk", np.int16),
            ("dl", np.int16),
        ],
    )

    key_all["pair"] = pair_id_all
    key_all["di"] = dist_i_all
    key_all["dh"] = direction_all[:, 0]
    key_all["dk"] = direction_all[:, 1]
    key_all["dl"] = direction_all[:, 2]

    # ---------------------------------------------------------
    # Finite subset
    # ---------------------------------------------------------
    frac_v = frac_all[valid]
    pair_id_v = pair_id_all[valid]
    upair_id_v = upair_id_all[valid]
    alpha_id_v = alpha_id_all[valid]
    beta_id_v = beta_id_all[valid]
    hkl_v = hkl_all[valid]
    direction_v = direction_all[valid]
    dist_i_v = dist_i_all[valid]
    counts_v = counts[valid].astype(np.int64)
    key_v = key_all[valid]

    # ---------------------------------------------------------
    # Group finite rows
    # ---------------------------------------------------------
    uniq, first_idx, inv = np.unique(
        key_v,
        return_index=True,
        return_inverse=True,
    )

    n_shell = int(len(uniq))

    shell_frac = frac_v[first_idx].astype(np.float32, copy=False)
    shell_pair_id = pair_id_v[first_idx].astype(np.int32, copy=False)
    shell_upair_id = upair_id_v[first_idx].astype(np.int32, copy=False)
    shell_alpha_id = alpha_id_v[first_idx].astype(np.int16, copy=False)
    shell_beta_id = beta_id_v[first_idx].astype(np.int16, copy=False)
    shell_hkl = hkl_v[first_idx].astype(np.int16, copy=False)
    shell_direction = direction_v[first_idx].astype(np.int16, copy=False)
    dist_keys_group = dist_i_v[first_idx].astype(np.int64, copy=False)

    finite_mult_raw_i64 = np.bincount(
        inv.astype(np.int64),
        weights=counts_v.astype(np.float64),
        minlength=n_shell,
    ).astype(np.int64)

    # ---------------------------------------------------------
    # Bulk counts for exactly the same keys.
    #
    # bulk_count_raw = number of all-site catalogue rows in the infinite
    # unit cell belonging to the same:
    #     pair + distance + direction
    # ---------------------------------------------------------
    t_bulk0 = time.perf_counter()

    uniq_bulk, bulk_counts = np.unique(
        key_all,
        return_counts=True,
    )

    bulk_lookup = {
        tuple(k): int(v)
        for k, v in zip(uniq_bulk.tolist(), bulk_counts.tolist())
    }

    bulk_mult_raw_i64 = np.zeros(n_shell, dtype=np.int64)

    for i in range(n_shell):
        bulk_mult_raw_i64[i] = int(
            bulk_lookup.get(tuple(uniq[i].tolist()), 0)
        )

    print(
        "[FINITE] Bulk reference counts for gamma: "
        f"bulk_keys={len(uniq_bulk):,}, finite_keys={n_shell:,}, "
        f"time={time.perf_counter() - t_bulk0:.3f} s"
    )

    # ---------------------------------------------------------
    # Effective finite and bulk coordination per alpha atom
    # ---------------------------------------------------------
    n_alpha_inside = np.asarray(mask_data["n_alpha_inside"], dtype=np.float64)
    species_counts = np.asarray(catalog["species_counts"], dtype=np.float64)

    shell_mult_eff = np.zeros(n_shell, dtype=np.float32)
    shell_bulk_mult_eff = np.zeros(n_shell, dtype=np.float32)
    shell_gamma = np.zeros(n_shell, dtype=np.float32)

    for i in range(n_shell):
        aid = int(shell_alpha_id[i])

        finite_denom = float(n_alpha_inside[aid]) if aid < n_alpha_inside.size else 0.0
        bulk_denom = float(species_counts[aid]) if aid < species_counts.size else 0.0

        finite_eff = 0.0
        bulk_eff = 0.0

        if finite_denom > 0.0:
            finite_eff = float(finite_mult_raw_i64[i]) / finite_denom

        if bulk_denom > 0.0:
            bulk_eff = float(bulk_mult_raw_i64[i]) / bulk_denom

        gamma = 0.0

        if bulk_eff > 0.0:
            gamma = finite_eff / bulk_eff

        shell_mult_eff[i] = np.float32(finite_eff)
        shell_bulk_mult_eff[i] = np.float32(bulk_eff)
        shell_gamma[i] = np.float32(gamma)

    shell_mult = np.minimum(
        finite_mult_raw_i64,
        np.iinfo(np.int32).max,
    ).astype(np.int32)

    shell_bulk_mult_raw = np.minimum(
        bulk_mult_raw_i64,
        np.iinfo(np.int32).max,
    ).astype(np.int32)

    # ---------------------------------------------------------
    # Rank shells by unique distance within unordered pair.
    # ---------------------------------------------------------
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

        reset_offsets = np.maximum.accumulate(
            np.where(new_upair, cumulative_unique, 0)
        ).astype(np.int32)

        rank_sorted = cumulative_unique - reset_offsets

        shell_k_within_upair[order_ud] = rank_sorted.astype(np.int32)

    shell_k_rank = np.arange(n_shell, dtype=np.int32)

    # ---------------------------------------------------------
    # Sort by pair id for PDF kernels.
    # ---------------------------------------------------------
    order = np.argsort(shell_pair_id, kind="mergesort")

    shell_frac = shell_frac[order]
    shell_mult = shell_mult[order]
    shell_bulk_mult_raw = shell_bulk_mult_raw[order]
    shell_mult_eff = shell_mult_eff[order]
    shell_bulk_mult_eff = shell_bulk_mult_eff[order]
    shell_gamma = shell_gamma[order]
    shell_pair_id = shell_pair_id[order]
    shell_alpha_id = shell_alpha_id[order]
    shell_beta_id = shell_beta_id[order]
    shell_hkl = shell_hkl[order]
    shell_direction = shell_direction[order]
    shell_k_rank = shell_k_rank[order]
    shell_upair_id = shell_upair_id[order]
    shell_k_within_upair = shell_k_within_upair[order]

    pair_list = list(catalog["pair_list"])
    upair_list = list(catalog["upair_list"])
    unique_species = list(catalog["unique_species"])

    n_pairs = int(len(pair_list))
    counts_pair = np.bincount(shell_pair_id, minlength=n_pairs).astype(np.int32)

    pair_offsets = np.zeros(n_pairs + 1, dtype=np.int32)
    pair_offsets[1:] = np.cumsum(counts_pair)

    pair_alpha_id = np.asarray(catalog["pair_alpha_id"], dtype=np.int16)
    pair_beta_id = np.asarray(catalog["pair_beta_id"], dtype=np.int16)

    try:
        dirs_i32 = shell_direction.astype(np.int32, copy=False)

        strain_unique_dirs, strain_inv_dirs = np.unique(
            dirs_i32,
            axis=0,
            return_inverse=True,
        )

        strain_unique_dirs = strain_unique_dirs.astype(np.int32, copy=False)
        strain_inv_dirs = strain_inv_dirs.astype(np.int32, copy=False)

    except Exception:
        strain_unique_dirs = np.zeros((0, 3), dtype=np.int32)
        strain_inv_dirs = np.zeros((0,), dtype=np.int32)

    print(
        "[FINITE] Built finite shell table with discrete gamma: "
        f"valid_rows={counts_v.size:,}, shells={len(shell_frac):,}, "
        f"time={time.perf_counter() - t0:.3f} s"
    )

    return {
        "shell_frac": shell_frac,
        "shell_mult": shell_mult,
        "shell_bulk_mult": shell_bulk_mult_raw,
        "shell_mult_eff": shell_mult_eff,
        "shell_finite_mult_eff": shell_mult_eff.copy(),
        "shell_bulk_mult_eff": shell_bulk_mult_eff,
        "shell_gamma": shell_gamma,
        "shell_pair_id": shell_pair_id,
        "shell_alpha_id": shell_alpha_id,
        "shell_beta_id": shell_beta_id,
        "shell_hkl": shell_hkl,
        "shell_direction": shell_direction,
        "shell_k_rank": shell_k_rank,
        "shell_upair_id": shell_upair_id,
        "shell_k_within_upair": shell_k_within_upair,
        "pair_offsets": pair_offsets,
        "pair_alpha_id": pair_alpha_id,
        "pair_beta_id": pair_beta_id,
        "unique_species": unique_species,
        "pair_list": pair_list,
        "upair_list": upair_list,
        "strain_unique_dirs": strain_unique_dirs,
        "strain_inv_dirs": strain_inv_dirs,
    }


# -----------------------------------------------------------------------------
# Main finite shell-table cache entry point
# -----------------------------------------------------------------------------

def export_gamma_functions_by_direction(
    shell_table: Dict[str, Any],
    structure: Structure,
    output_dir: str,
    *,
    write_details: bool = True,
    r_round_decimals: int = 5,
) -> str:
    """
    Export finite-shape gamma functions into one folder, with one CSV file per
    crystallographic direction.

    This is a direction-resolved version of export_coordination_per_direction_csv().

    For each reduced crystallographic direction [dir_h dir_k dir_l], this writes:

        gamma_dir_h*_k*_l*.csv

    containing the discrete direction gamma function:

        gamma_direction(r)
            = sum(finite_mult_eff) / sum(bulk_mult_eff)

    over all shell rows belonging to that direction at the same distance bin.

    If write_details=True, it also writes:

        details_dir_h*_k*_l*.csv

    containing the pair-resolved shell rows for that direction.

    Parameters
    ----------
    shell_table
        Packed finite shell table produced by build_shell_table_from_finite_counts().

    structure
        Pymatgen Structure. Used only to convert shell_frac to real-space distances.

    output_dir
        Folder where the direction-resolved CSV files will be written.

    write_details
        If True, also write pair-resolved detailed files per direction.

    r_round_decimals
        Number of decimal places used to group distances into discrete gamma(r)
        points. The shell generation code groups distances at approximately 1e-5 Å,
        so the default 5 is consistent.

    Returns
    -------
    output_dir : str
        The folder containing the exported gamma functions.
    """
    import csv
    import os
    from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)

    M = np.asarray(structure.lattice.matrix, dtype=float)

    shell_frac = np.asarray(shell_table["shell_frac"], dtype=float)

    shell_mult = np.asarray(
        shell_table["shell_mult"],
        dtype=np.int64,
    )

    shell_bulk_mult = np.asarray(
        shell_table.get("shell_bulk_mult", shell_mult),
        dtype=np.int64,
    )

    # In finite-shape shell tables:
    #   shell_mult_eff is the finite effective multiplicity.
    shell_mult_eff = np.asarray(
        shell_table["shell_mult_eff"],
        dtype=float,
    )

    shell_bulk_mult_eff = np.asarray(
        shell_table.get("shell_bulk_mult_eff", shell_mult_eff),
        dtype=float,
    )

    # Per-shell finite gamma:
    #   gamma_s = finite_mult_eff / bulk_mult_eff
    shell_gamma = np.asarray(
        shell_table.get("shell_gamma", np.ones_like(shell_mult_eff)),
        dtype=float,
    )

    shell_pair_id = np.asarray(
        shell_table["shell_pair_id"],
        dtype=np.int32,
    )

    shell_hkl = np.asarray(
        shell_table["shell_hkl"],
        dtype=np.int32,
    )

    shell_direction = np.asarray(
        shell_table["shell_direction"],
        dtype=np.int32,
    )

    pair_list = list(shell_table["pair_list"])

    # Real-space shell distances.
    cart = shell_frac @ M
    r_ij = np.sqrt(np.sum(cart * cart, axis=1))

    def _pair_label(pid: int) -> str:
        if 0 <= pid < len(pair_list):
            return f"{pair_list[pid][0]}-{pair_list[pid][1]}"
        return f"pair_{pid}"

    def _dir_component(n: int) -> str:
        n = int(n)
        if n < 0:
            return f"m{abs(n)}"
        return str(n)

    def _direction_stem(dh: int, dk: int, dl: int) -> str:
        return (
            f"dir_h{_dir_component(dh)}"
            f"_k{_dir_component(dk)}"
            f"_l{_dir_component(dl)}"
        )

    def _direction_label(dh: int, dk: int, dl: int) -> str:
        return f"[{int(dh)} {int(dk)} {int(dl)}]"

    # -------------------------------------------------------------------------
    # Group shell rows by reduced crystallographic direction.
    # -------------------------------------------------------------------------
    direction_to_indices = defaultdict(list)

    for i in range(shell_direction.shape[0]):
        dh, dk, dl = [int(x) for x in shell_direction[i]]
        direction_to_indices[(dh, dk, dl)].append(i)

    # -------------------------------------------------------------------------
    # Write one gamma function file per direction.
    # Also write an index file so the output folder is easy to navigate.
    # -------------------------------------------------------------------------
    index_path = os.path.join(output_dir, "index.csv")

    with open(index_path, "w", newline="", encoding="utf-8") as f_index:
        index_writer = csv.writer(f_index)

        index_writer.writerow(
            [
                "direction",
                "dir_h",
                "dir_k",
                "dir_l",
                "n_shell_rows",
                "r_min_A",
                "r_max_A",
                "gamma_file",
                "details_file",
            ]
        )

        for direction in sorted(direction_to_indices.keys()):
            dh, dk, dl = direction
            indices = direction_to_indices[direction]

            if not indices:
                continue

            stem = _direction_stem(dh, dk, dl)
            direction_txt = _direction_label(dh, dk, dl)

            gamma_filename = f"gamma_{stem}.csv"
            gamma_path = os.path.join(output_dir, gamma_filename)

            details_filename = f"details_{stem}.csv"
            details_path = os.path.join(output_dir, details_filename)

            # -----------------------------------------------------------------
            # Build direction gamma(r):
            #
            #   gamma_direction(r)
            #       = sum finite_mult_eff / sum bulk_mult_eff
            #
            # grouped by rounded shell distance.
            # -----------------------------------------------------------------
            by_r = defaultdict(lambda: [0.0, 0.0, 0, 0])
            # value = [finite_eff_sum, bulk_eff_sum, finite_raw_sum, bulk_raw_sum]

            for i in indices:
                if not np.isfinite(r_ij[i]):
                    continue

                r_key = round(float(r_ij[i]), int(r_round_decimals))

                finite_eff = float(shell_mult_eff[i])
                bulk_eff = float(shell_bulk_mult_eff[i])

                finite_raw = int(shell_mult[i])
                bulk_raw = int(shell_bulk_mult[i])

                by_r[r_key][0] += finite_eff
                by_r[r_key][1] += bulk_eff
                by_r[r_key][2] += finite_raw
                by_r[r_key][3] += bulk_raw

            with open(gamma_path, "w", newline="", encoding="utf-8") as f_gamma:
                writer = csv.writer(f_gamma)

                writer.writerow(
                    [
                        "direction",
                        "dir_h",
                        "dir_k",
                        "dir_l",
                        "r_A",
                        "finite_mult_raw_sum",
                        "bulk_mult_raw_sum",
                        "finite_mult_eff_sum",
                        "bulk_mult_eff_sum",
                        "gamma_direction",
                    ]
                )

                for r_key in sorted(by_r.keys()):
                    finite_eff_sum, bulk_eff_sum, finite_raw_sum, bulk_raw_sum = by_r[r_key]

                    if bulk_eff_sum > 0.0:
                        gamma_dir = finite_eff_sum / bulk_eff_sum
                    else:
                        gamma_dir = 0.0

                    writer.writerow(
                        [
                            direction_txt,
                            int(dh),
                            int(dk),
                            int(dl),
                            f"{float(r_key):.10g}",
                            int(finite_raw_sum),
                            int(bulk_raw_sum),
                            f"{float(finite_eff_sum):.10g}",
                            f"{float(bulk_eff_sum):.10g}",
                            f"{float(gamma_dir):.10g}",
                        ]
                    )

            # -----------------------------------------------------------------
            # Optional detailed pair-resolved output for this direction.
            # -----------------------------------------------------------------
            if write_details:
                sorted_indices = sorted(
                    indices,
                    key=lambda i: (
                        _pair_label(int(shell_pair_id[i])),
                        float(r_ij[i]),
                        int(shell_hkl[i, 0]),
                        int(shell_hkl[i, 1]),
                        int(shell_hkl[i, 2]),
                    ),
                )

                with open(details_path, "w", newline="", encoding="utf-8") as f_details:
                    writer = csv.writer(f_details)

                    writer.writerow(
                        [
                            "pair",
                            "direction",
                            "r_A",
                            "h",
                            "k",
                            "l",
                            "dir_h",
                            "dir_k",
                            "dir_l",
                            "finite_mult_raw",
                            "bulk_mult_raw",
                            "finite_mult_eff",
                            "bulk_mult_eff",
                            "gamma_finite",
                        ]
                    )

                    for i in sorted_indices:
                        pid = int(shell_pair_id[i])
                        pair = _pair_label(pid)

                        h, k, l = [int(x) for x in shell_hkl[i]]
                        ddh, ddk, ddl = [int(x) for x in shell_direction[i]]

                        writer.writerow(
                            [
                                pair,
                                direction_txt,
                                f"{float(r_ij[i]):.10g}",
                                h,
                                k,
                                l,
                                ddh,
                                ddk,
                                ddl,
                                int(shell_mult[i]),
                                int(shell_bulk_mult[i]),
                                f"{float(shell_mult_eff[i]):.10g}",
                                f"{float(shell_bulk_mult_eff[i]):.10g}",
                                f"{float(shell_gamma[i]):.10g}",
                            ]
                        )

            else:
                details_filename = ""

            r_vals = np.asarray([r_ij[i] for i in indices], dtype=float)
            r_vals = r_vals[np.isfinite(r_vals)]

            r_min = float(np.min(r_vals)) if r_vals.size else float("nan")
            r_max = float(np.max(r_vals)) if r_vals.size else float("nan")

            index_writer.writerow(
                [
                    direction_txt,
                    int(dh),
                    int(dk),
                    int(dl),
                    int(len(indices)),
                    f"{r_min:.10g}" if np.isfinite(r_min) else "",
                    f"{r_max:.10g}" if np.isfinite(r_max) else "",
                    gamma_filename,
                    details_filename,
                ]
            )

    return output_dir


def export_coordination_per_direction_csv(
    shell_table: Dict[str, Any],
    structure: Structure,
    path: str,
) -> str:
    """
    Export finite-shape coordination and discrete gamma per pair distance
    and direction.

    This diagnostic table shows:

        finite_mult_eff = finite pairs per finite alpha atom
        bulk_mult_eff   = bulk pairs per bulk alpha atom
        gamma           = finite_mult_eff / bulk_mult_eff

    gamma is the discrete common-volume/background factor used to replace
    the infinite-crystal baseline 1.
    """
    import csv

    M = np.asarray(structure.lattice.matrix, dtype=float)

    shell_frac = np.asarray(shell_table["shell_frac"], dtype=float)
    shell_mult = np.asarray(shell_table["shell_mult"], dtype=np.int64)
    shell_bulk_mult = np.asarray(
        shell_table.get("shell_bulk_mult", shell_mult),
        dtype=np.int64,
    )

    shell_mult_eff = np.asarray(shell_table["shell_mult_eff"], dtype=float)
    shell_bulk_mult_eff = np.asarray(
        shell_table.get("shell_bulk_mult_eff", shell_mult_eff),
        dtype=float,
    )
    shell_gamma = np.asarray(
        shell_table.get("shell_gamma", np.ones_like(shell_mult_eff)),
        dtype=float,
    )

    shell_pair_id = np.asarray(shell_table["shell_pair_id"], dtype=np.int32)
    shell_hkl = np.asarray(shell_table["shell_hkl"], dtype=np.int32)
    shell_direction = np.asarray(shell_table["shell_direction"], dtype=np.int32)

    pair_list = list(shell_table["pair_list"])

    cart = shell_frac @ M
    r_ij = np.sqrt(np.sum(cart * cart, axis=1))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "pair",
                "r_A",
                "h",
                "k",
                "l",
                "dir_h",
                "dir_k",
                "dir_l",
                "finite_mult_raw",
                "bulk_mult_raw",
                "finite_mult_eff",
                "bulk_mult_eff",
                "gamma_finite",
            ]
        )

        for i in range(shell_frac.shape[0]):
            pid = int(shell_pair_id[i])

            if 0 <= pid < len(pair_list):
                pair = f"{pair_list[pid][0]}-{pair_list[pid][1]}"
            else:
                pair = f"pair_{pid}"

            h, k, l = [int(x) for x in shell_hkl[i]]
            dh, dk, dl = [int(x) for x in shell_direction[i]]

            writer.writerow(
                [
                    pair,
                    f"{float(r_ij[i]):.10g}",
                    h,
                    k,
                    l,
                    dh,
                    dk,
                    dl,
                    int(shell_mult[i]),
                    int(shell_bulk_mult[i]),
                    f"{float(shell_mult_eff[i]):.10g}",
                    f"{float(shell_bulk_mult_eff[i]):.10g}",
                    f"{float(shell_gamma[i]):.10g}",
                ]
            )

    return path


def get_or_build_finite_shell_table(
    structure: Structure,
    pair_cutoff: float,
    shape_spec: CrystalliteShapeSpec,
    *,
    cache_dir: str = FINITE_CACHE_DIR,
    boundary_mode: str = "cell",
) -> Dict[str, Any]:
    """
    Load or build a finite-shape shell table for one shape candidate.

    This is the main backend function the CS grid-search worker will call.
    """
    os.makedirs(cache_dir, exist_ok=True)

    catalog = get_or_build_all_site_pair_catalog(
        structure,
        pair_cutoff,
        cache_dir=cache_dir,
    )

    shape_key = _hash_shape(
        shape_spec,
        extra={
            "catalog_hash": catalog["catalog_hash"],
            "pair_cutoff": float(pair_cutoff),
            "boundary_mode": boundary_mode,
        },
    )

    npz_path = os.path.join(cache_dir, f"finite_shape_shells_{shape_key}.npz")
    meta_path = os.path.join(cache_dir, f"finite_shape_shells_{shape_key}.json")

    if os.path.exists(npz_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            z = np.load(npz_path, allow_pickle=False)

            shell_table = {
                "shell_frac": z["shell_frac"],
                "shell_mult": z["shell_mult"],
                "shell_bulk_mult": z["shell_bulk_mult"] if "shell_bulk_mult" in z.files else z["shell_mult"],
                "shell_mult_eff": z["shell_mult_eff"],
                "shell_finite_mult_eff": z["shell_finite_mult_eff"] if "shell_finite_mult_eff" in z.files else z["shell_mult_eff"],
                "shell_bulk_mult_eff": z["shell_bulk_mult_eff"] if "shell_bulk_mult_eff" in z.files else z["shell_mult_eff"],
                "shell_gamma": z["shell_gamma"] if "shell_gamma" in z.files else np.ones_like(z["shell_mult_eff"], dtype=np.float32),
                "shell_pair_id": z["shell_pair_id"],
                "shell_alpha_id": z["shell_alpha_id"],
                "shell_beta_id": z["shell_beta_id"],
                "shell_hkl": z["shell_hkl"],
                "shell_direction": z["shell_direction"],
                "shell_k_rank": z["shell_k_rank"],
                "shell_upair_id": z["shell_upair_id"],
                "shell_k_within_upair": z["shell_k_within_upair"],
                "pair_offsets": z["pair_offsets"],
                "pair_alpha_id": z["pair_alpha_id"],
                "pair_beta_id": z["pair_beta_id"],
                "strain_unique_dirs": z["strain_unique_dirs"],
                "strain_inv_dirs": z["strain_inv_dirs"],
                "unique_species": list(meta["unique_species"]),
                "pair_list": [tuple(x) for x in meta["pair_list"]],
                "upair_list": [tuple(x) for x in meta["upair_list"]],
            }

            print(f"[FINITE CACHE] Loaded finite shell table: {npz_path}")
            return shell_table

        except Exception:
            pass

    print(
        "[FINITE] Building finite shell table "
        f"shape={shape_spec.shape_type}, "
        f"D={shape_spec.diameter_cells}, "
        f"H={shape_spec.height_cells}"
    )

    t0 = time.perf_counter()

    mask_data = build_cell_mask_for_shape(
        structure,
        shape_spec,
        boundary_mode=boundary_mode,
    )

    counts = count_finite_coordination_cpu(
        catalog,
        mask_data,
    )

    shell_table = build_shell_table_from_finite_counts(
        catalog,
        counts,
        mask_data,
    )

    np.savez(
        npz_path,
        shell_frac=shell_table["shell_frac"],
        shell_mult=shell_table["shell_mult"],
        shell_bulk_mult=shell_table["shell_bulk_mult"],
        shell_mult_eff=shell_table["shell_mult_eff"],
        shell_finite_mult_eff=shell_table["shell_finite_mult_eff"],
        shell_bulk_mult_eff=shell_table["shell_bulk_mult_eff"],
        shell_gamma=shell_table["shell_gamma"],
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
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unique_species": shell_table["unique_species"],
                "pair_list": [list(x) for x in shell_table["pair_list"]],
                "upair_list": [list(x) for x in shell_table["upair_list"]],
                "shape_spec": shape_spec.to_dict(),
                "pair_cutoff": float(pair_cutoff),
                "boundary_mode": boundary_mode,
            },
            f,
            indent=2,
        )

    print(
        f"[FINITE CACHE] Saved finite shell table with "
        f"{len(shell_table['shell_frac'])} shell rows "
        f"in {time.perf_counter() - t0:.2f} s"
    )

    return shell_table