"""
Edited structures of a phase (stage 3 of the structure editor).

The project never modifies the user's CIF. When sites are added or deleted,
the phase stores its asymmetric-unit site list in
``phase.sections["structure_edit"]``::

    {"base_cif": "<original cif>", "sites": [
        {"label": "O1", "element": "O", "x": 0.25992, "y": 0.0, "z": 0.25,
         "occ": 1.0, "biso": 2.73}, ...]}

The model reads a derived CIF: the original file with only its _atom_site
loop replaced (symmetry operations, setting and everything else kept), so
labels and the x_/y_/z_/occ_ parameters keep matching. Derived files are
written to the user cache, named by content hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

ATOM_SITE_PREFIXES = ("_atom_site_",)


def _cache_dir() -> str:
    from pdf_fitting.user_paths import user_cache_dir

    path = user_cache_dir("edited_structures")
    os.makedirs(path, exist_ok=True)
    return path


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") if v != 0 else "0"
    return str(v)


def base_sites(cif_path: str) -> List[Dict[str, Any]]:
    """Asymmetric-unit sites of a CIF in the editor's format."""
    from pdf_fitting.gui.cif_utils import read_cif_asu_sites

    _structure, asu = read_cif_asu_sites(cif_path)
    out = []
    for rec in asu or []:
        frac = rec.get("frac", (0.0, 0.0, 0.0))
        biso = rec.get("biso", None)
        out.append(
            {
                "label": str(rec.get("label", "")),
                "element": str(rec.get("el", "")),
                "x": float(frac[0]),
                "y": float(frac[1]),
                "z": float(frac[2]),
                "occ": float(rec.get("occ", 1.0)),
                "biso": None if biso in (None, "") else float(biso),
            }
        )
    return out


def cif_text_with_sites(
    base_cif_path: str,
    sites: List[Dict[str, Any]],
    *,
    lattice: Optional[Dict[str, float]] = None,
) -> str:
    """
    The base CIF with its _atom_site loop(s) replaced by ``sites`` (and,
    optionally, new cell lengths / angles). Everything else is kept.
    """
    from pymatgen.io.cif import CifFile

    with open(base_cif_path, "r", encoding="utf-8", errors="replace") as fh:
        cif = CifFile.from_str(fh.read())
    block = next(iter(cif.data.values()))

    # Drop every atom-site loop (positions, anisotropic ADPs...).
    keep_loops = []
    for loop in block.loops:
        if any(str(k).startswith(ATOM_SITE_PREFIXES) for k in loop):
            for k in loop:
                block.data.pop(k, None)
        else:
            keep_loops.append(loop)
    block.loops = keep_loops
    for k in [k for k in block.data if str(k).startswith(ATOM_SITE_PREFIXES)]:
        block.data.pop(k, None)

    keys = [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_B_iso_or_equiv",
    ]
    columns: Dict[str, List[str]] = {k: [] for k in keys}
    for s in sites:
        columns["_atom_site_label"].append(str(s["label"]))
        columns["_atom_site_type_symbol"].append(str(s["element"]))
        columns["_atom_site_fract_x"].append(_fmt(float(s["x"])))
        columns["_atom_site_fract_y"].append(_fmt(float(s["y"])))
        columns["_atom_site_fract_z"].append(_fmt(float(s["z"])))
        columns["_atom_site_occupancy"].append(_fmt(float(s.get("occ", 1.0))))
        b = s.get("biso", None)
        columns["_atom_site_B_iso_or_equiv"].append("." if b is None else _fmt(float(b)))
    for k in keys:
        block.data[k] = columns[k]
    block.loops.append(keys)

    # Anisotropic displacement (β) of the sites that have it.
    aniso = [s for s in sites if s.get("beta") is not None]
    if aniso:
        akeys = ["_atom_site_aniso_label"] + [f"_atom_site_aniso_beta_{c}" for c in ("11", "22", "33", "12", "13", "23")]
        acol: Dict[str, List[str]] = {k: [] for k in akeys}
        for s in aniso:
            acol["_atom_site_aniso_label"].append(str(s["label"]))
            for c, v in zip(("11", "22", "33", "12", "13", "23"), s["beta"]):
                acol[f"_atom_site_aniso_beta_{c}"].append(_fmt(float(v)))
        for k in akeys:
            block.data[k] = acol[k]
        block.loops.append(akeys)

    if lattice:
        names = {
            "a": "_cell_length_a", "b": "_cell_length_b", "c": "_cell_length_c",
            "alpha": "_cell_angle_alpha", "beta": "_cell_angle_beta", "gamma": "_cell_angle_gamma",
        }
        for k, tag in names.items():
            if k in lattice:
                block.data[tag] = _fmt(float(lattice[k]))
        block.data.pop("_cell_volume", None)

    return str(cif)


def cif_aniso_betas(cif_path: str) -> Dict[str, List[float]]:
    """
    {CIF label: [β11, β22, β33, β12, β13, β23]} from the CIF's aniso loop
    (_atom_site_aniso_beta_ij, _B_ij or _U_ij; U and B in the CIF convention
    w.r.t. a*_i a*_j): β_ij = 2π² a*_i a*_j U_ij = ¼ a*_i a*_j B_ij.
    """
    import math

    from pymatgen.io.cif import CifFile

    def num(v):
        try:
            return float(str(v).split("(")[0])
        except (TypeError, ValueError):
            return None

    with open(cif_path, "r", encoding="utf-8", errors="replace") as fh:
        block = next(iter(CifFile.from_str(fh.read()).data.values()))
    data = {str(k).lower(): v for k, v in block.data.items()}
    labels = data.get("_atom_site_aniso_label")
    if not labels:
        return {}
    if isinstance(labels, str):
        labels = [labels]
    a, b, c = (num(data.get(f"_cell_length_{x}")) for x in "abc")
    al, be, ga = (math.radians(num(data.get(f"_cell_angle_{x}")) or 90.0) for x in ("alpha", "beta", "gamma"))
    if not (a and b and c):
        return {}
    G = np.array([
        [a * a, a * b * math.cos(ga), a * c * math.cos(be)],
        [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
        [a * c * math.cos(be), b * c * math.cos(al), c * c],
    ])
    rl = np.sqrt(np.diag(np.linalg.inv(G)))   # a*, b*, c*
    comps = ("11", "22", "33", "12", "13", "23")
    idx = {"11": (0, 0), "22": (1, 1), "33": (2, 2), "12": (0, 1), "13": (0, 2), "23": (1, 2)}
    out: Dict[str, List[float]] = {}
    for kind, factor in (("beta", None), ("u", 2.0 * math.pi ** 2), ("b", 0.25)):
        cols = [data.get(f"_atom_site_aniso_{kind}_{c}") for c in comps]
        if not all(cols):
            continue
        cols = [[x] if isinstance(x, str) else list(x) for x in cols]
        for n, lab in enumerate(labels):
            vals = []
            for c, col in zip(comps, cols):
                v = num(col[n]) if n < len(col) else None
                if v is None:
                    v = 0.0
                if factor is not None:
                    i, j = idx[c]
                    v = factor * rl[i] * rl[j] * v
                vals.append(v)
            out.setdefault(str(lab), vals)
        break
    return out


_MATERIALIZED: Dict[str, str] = {}


def materialize(text: str) -> str:
    """Write a derived CIF into the cache (by content hash); returns its path."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    if digest in _MATERIALIZED and os.path.exists(_MATERIALIZED[digest]):
        return _MATERIALIZED[digest]
    path = os.path.join(_cache_dir(), f"edited_{digest}.cif")
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    _MATERIALIZED[digest] = path
    return path


_PATH_CACHE: Dict[str, str] = {}


def edited_structure_path(base_cif_path: str, sites: List[Dict[str, Any]]) -> str:
    """Path of the derived CIF for a site list (cached)."""
    try:
        mtime = os.path.getmtime(base_cif_path)
    except OSError:
        mtime = 0.0
    key = hashlib.sha256(
        json.dumps([os.path.abspath(base_cif_path), mtime, sites], sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    path = _PATH_CACHE.get(key)
    if path is None or not os.path.exists(path):
        path = materialize(cif_text_with_sites(base_cif_path, sites))
        _PATH_CACHE[key] = path
    return path


def unique_label(sites: List[Dict[str, Any]], element: str) -> str:
    from pdf_fitting.structure_handler import safe_param_key

    used = {safe_param_key(s["label"]) for s in sites}
    k = 1
    while safe_param_key(f"{element}{k}") in used:
        k += 1
    return f"{element}{k}"


def copy_sites(sites):
    return copy.deepcopy(list(sites or []))
