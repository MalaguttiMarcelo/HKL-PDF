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

CACHE_VERSION = "shell_cache_hybrid_v10_image_strategy"

def clean_element_name(name: str) -> str:
    """Return clean chemical element symbol, e.g. Fe0+ -> Fe."""
    m = re.match(r"([A-Za-z]{1,2})", str(name))
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

def structure_hash(structure: Structure, r_max_cache: float) -> str:
    """Cache key for shell generation.

    We store shells in fractional space and re-evaluate real-space distances
    under refined lattice parameters during PDF evaluation.

    Therefore we do *not* include lattice parameters in the cache key.
    The only lattice-dependent part is selecting which neighbors are within
    r_max; we address this by generating shells with a safety margin
    (r_max_cache).
    """
    atoms_str = ";".join(
        f"{site.species_string},{tuple(np.round(site.frac_coords, 12))}" for site in structure
    )
    spg = SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol()
    raw = f"{CACHE_VERSION}|{spg}|{atoms_str}|rmax_cache={float(r_max_cache):.6f}"
    return hashlib.md5(raw.encode()).hexdigest()
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


def get_lattice_constraints(structure):
    system = SpacegroupAnalyzer(structure).get_crystal_system().lower()
    if system == "cubic":
        return {"b": "a", "c": "a", "alpha": 90, "beta": 90, "gamma": 90}
    elif system == "tetragonal":
        return {"b": "a", "alpha": 90, "beta": 90, "gamma": 90}
    elif system == "hexagonal":
        return {"b": "a", "alpha": 90, "beta": 90, "gamma": 120}
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
    def __init__(self, cif_path: str, r_max: float):
        # Use fast CIF parser for simple/P1 CIFs when possible.
        # This avoids expensive parsing of huge bond loops from Materials Studio CIFs.
        try:
            if str(cif_path).lower().endswith(".cif"):
                from pdf_fitting.gui.cif_utils import read_cif_asu_sites

                s_fast, _asu = read_cif_asu_sites(cif_path)

                if s_fast is not None:
                    self.structure = s_fast
                else:
                    self.structure = Structure.from_file(cif_path)
            else:
                self.structure = Structure.from_file(cif_path)

        except Exception:
            self.structure = Structure.from_file(cif_path)

        self.r_max = r_max
        # Generate shells slightly beyond r_max so we don't need to rebuild shells
        # during refinement of lattice parameters.
        self.r_max_cache = float(r_max) * 1.10
        self.constraints = get_lattice_constraints(self.structure)
        self._shell_table = None
        self.shells = self._generate_shells_fast_cached()

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
        key = structure_hash(self.structure, self.r_max_cache)
        cache_npz = os.path.join(CACHE_DIR, f"shells_{key}.npz")
        cache_meta = os.path.join(CACHE_DIR, f"shells_{key}.meta.json")

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
                print(f"[CACHE] Loaded shell table from {cache_npz}")

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

        print("[INFO] Computing shells (fast neighbor search, one-time)...")
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
        )
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "unique_species": shell_table["unique_species"],
                    "pair_list": [list(x) for x in shell_table["pair_list"]],
                    "upair_list": [list(x) for x in shell_table.get("upair_list", [])],
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
                    same_species = str(site.specie) == str(site0.specie)
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

    def _get_neighbor_list_for_unique_sites_parallel(self, unique_site_indices):
        """
        Parallel neighbor-list construction over symmetry-unique center sites.

        Uses all available logical CPU cores, limited by the number of unique sites.
        If there is only one unique site, this falls back to one task.
        """
        unique_site_indices = np.asarray(unique_site_indices, dtype=np.int32)
        n_centers = int(unique_site_indices.size)

        if n_centers == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        n_cpu = os.cpu_count() or 1
        n_workers = max(1, min(int(n_cpu), n_centers))

        def _one_chunk(chunk_indices):
            chunk_indices = np.asarray(chunk_indices, dtype=np.int32)
            sites = [self.structure[int(i)] for i in chunk_indices]

            c_local, n_idx, images, dists = self.structure.get_neighbor_list(
                r=self.r_max_cache,
                sites=sites,
                numerical_tol=1e-8,
                exclude_self=True,
            )

            c_local = np.asarray(c_local, dtype=np.int64)
            c_orig = chunk_indices[c_local].astype(np.int32)

            return (
                c_orig,
                np.asarray(n_idx, dtype=np.int32),
                np.asarray(images, dtype=np.int32),
                np.asarray(dists, dtype=np.float64),
            )

        # For very small numbers of centers, serial is simpler and avoids thread overhead.
        if n_workers <= 1:
            return _one_chunk(unique_site_indices)

        chunks = [
            c for c in np.array_split(unique_site_indices, n_workers)
            if len(c) > 0
        ]

        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for out in ex.map(_one_chunk, chunks):
                results.append(out)

        if not results:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        c_idx = np.concatenate([r[0] for r in results]).astype(np.int32)
        n_idx = np.concatenate([r[1] for r in results]).astype(np.int32)
        images = np.concatenate([r[2] for r in results]).astype(np.int32)
        dists = np.concatenate([r[3] for r in results]).astype(np.float64)

        return c_idx, n_idx, images, dists
    
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
        sga = SpacegroupAnalyzer(self.structure, symprec=1e-3)
        symm = sga.get_symmetrized_structure()

        unique_site_indices = self._get_unique_site_indices(symm)

        lattice = self.structure.lattice

        # Element labels for PDF physics/scattering/Biso/densities
        species_element_all = [clean_element_name(str(s.specie)) for s in self.structure.sites]
        unique_species = sorted(set(species_element_all))
        sp_index = {s: i for i, s in enumerate(unique_species)}

        # Lambda/species labels for local pair lambdas
        # Examples: Fe0+ -> fe0plus, O2- -> o2minus
        species_lambda_all = [lambda_safe_species_label(str(s.specie)) for s in self.structure.sites]
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

        try:
            for group in symm.equivalent_indices:
                group = [int(i) for i in group]
                if not group:
                    continue
                
                rep = group[0]
                sid = int(site_species_id[rep])
                total_for_species = max(float(species_counts_by_id[sid]), 1.0)
                w_alpha = float(len(group)) / total_for_species

                for idx in group:
                    site_alpha_weight[int(idx)] = w_alpha

        except Exception:
            for idx in range(len(self.structure.sites)):
                sid = int(site_species_id[idx])
                total_for_species = max(float(species_counts_by_id[sid]), 1.0)
                site_alpha_weight[idx] = 1.0 / total_for_species

                # Reciprocal helper for (hkl) estimation from cart vectors
                recip_T = lattice.reciprocal_lattice.matrix.T
                inv_recip_T = np.linalg.inv(recip_T)

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

        # Lambda/species IDs for the same neighbor rows.
        # These are used only for lambda pair labels/grouping.
        lam_a_id = site_lambda_id[c_idx].astype(np.int16)
        lam_b_id = site_lambda_id[n_idx].astype(np.int16)

        # Weight of each neighbor row according to the alpha/center site.
        # Used to build shell_mult_eff for correct PDF normalization.
        alpha_weight_row = site_alpha_weight[c_idx].astype(np.float64)


        # If nothing returned
        if len(dists) == 0:
            return [], {
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

        # ---- Bucket key: (a_id, b_id, dist_1e5, dh, dk, dl) ----
        t_group0 = time.perf_counter()

        dist_i = np.rint(dists * 1e5).astype(np.int32)

        key = np.zeros(len(dists), dtype=[
            ("a", np.int16), ("b", np.int16), ("di", np.int32),
            ("dh", np.int16), ("dk", np.int16), ("dl", np.int16)
        ])
        # Group pair type by lambda/species labels, not just element.
        # This allows Fe0+ and Fe2+ to have different lambda coefficients.
        key["a"] = lam_a_id
        key["b"] = lam_b_id
        key["di"] = dist_i
        key["dh"] = dh.astype(np.int16)
        key["dk"] = dk.astype(np.int16)
        key["dl"] = dl.astype(np.int16)

        t_unique0 = time.perf_counter()
        uniq, first_idx, inv, counts = np.unique(
            key, return_index=True, return_inverse=True, return_counts=True
        )
        t_unique1 = time.perf_counter()

        print(
            f"[SHELLS] Grouping with np.unique: "
            f"{len(dists)} neighbor rows -> {len(uniq)} shells "
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
        shell_mult_eff = np.bincount(
            inv.astype(np.int64),
            weights=alpha_weight_row,
            minlength=n_shell,
        ).astype(np.float32, copy=False)

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

        shell_table = {
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
        }

        # Cache unique strain directions and inverse map.
        # This avoids expensive np.unique(axis=0) after loading cache.
        try:
            dirs_i32 = shell_table["shell_direction"].astype(np.int32, copy=False)
            strain_unique_dirs, strain_inv_dirs = np.unique(
                dirs_i32,
                axis=0,
                return_inverse=True,
            )

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
