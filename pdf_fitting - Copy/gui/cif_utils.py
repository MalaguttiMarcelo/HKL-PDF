from __future__ import annotations

import os
import re
import shlex
from functools import lru_cache
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core import Lattice
from pymatgen.io.cif import CifParser


def parse_cif_number(x: Any) -> Optional[float]:
    """Parse CIF numeric strings like '1.45(8)' -> 1.45."""
    if x is None:
        return None

    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return None

    s = str(x).strip()

    if not s or s in (".", "?"):
        return None

    if "(" in s:
        s = s.split("(", 1)[0].strip()

    try:
        return float(s)
    except Exception:
        return None


def clean_el_symbol(name: Any) -> str:
    """Return a clean chemical symbol, e.g. 'Fe0+' -> 'Fe'."""
    s = str(name).strip()

    m = re.match(r"([A-Za-z]{1,2})", s)

    if not m:
        return s

    sym = m.group(1)

    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()


def _tokenize_cif_line(line: str) -> List[str]:
    """
    Tokenize one CIF data line.

    Handles simple quoted tokens using shlex.
    """
    line = str(line).strip()

    if not line:
        return []

    try:
        return shlex.split(line, posix=True)
    except Exception:
        return line.split()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _scalar_float(text: str, tag: str) -> Optional[float]:
    """
    Read a scalar CIF value.

    Example:
        _cell_length_a 25.3381
    """
    pat = re.compile(
        rf"(?im)^\s*{re.escape(tag)}\s+(.+?)\s*$"
    )

    m = pat.search(text)

    if not m:
        return None

    toks = _tokenize_cif_line(m.group(1))

    if not toks:
        return None

    return parse_cif_number(toks[0])


def _scalar_int(text: str, tag: str) -> Optional[int]:
    v = _scalar_float(text, tag)

    if v is None:
        return None

    try:
        return int(round(float(v)))
    except Exception:
        return None


def _symmetry_ops_are_identity_only(text: str) -> bool:
    """
    Return True if the CIF symmetry operators are effectively only x,y,z.

    This is a fast path for P1 CIFs.
    """
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if not line.strip().lower().startswith("loop_"):
            continue

        j = i + 1
        headers = []

        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1

        has_symop = any(
            h.lower() in (
                "_symmetry_equiv_pos_as_xyz",
                "_space_group_symop_operation_xyz",
            )
            for h in headers
        )

        if not has_symop:
            continue

        ops = []

        while j < len(lines):
            s = lines[j].strip()

            if not s:
                j += 1
                continue

            sl = s.lower()

            if sl.startswith("loop_") or sl.startswith("data_") or sl.startswith("_"):
                break

            toks = _tokenize_cif_line(s)

            if toks:
                ops.append(toks[0].replace(" ", "").lower())

            j += 1

        ops = [op for op in ops if op not in ("", ".", "?")]

        if not ops:
            return True

        return len(ops) == 1 and ops[0] in ("x,y,z", "'x,y,z'", '"x,y,z"')

    # If no symop loop exists, treat SG=1 as identity.
    return True


def _parse_atom_site_loop_fast(text: str) -> List[Dict[str, Any]]:
    """
    Parse only the atom_site loop. Ignore bond loops and everything else.
    """
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if not line.strip().lower().startswith("loop_"):
            continue

        j = i + 1
        headers = []

        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1

        h_lc = [h.lower() for h in headers]

        if "_atom_site_label" not in h_lc:
            continue

        if "_atom_site_fract_x" not in h_lc:
            continue

        if "_atom_site_fract_y" not in h_lc:
            continue

        if "_atom_site_fract_z" not in h_lc:
            continue

        def idx(name: str) -> Optional[int]:
            name = name.lower()
            try:
                return h_lc.index(name)
            except Exception:
                return None

        i_label = idx("_atom_site_label")
        i_el = idx("_atom_site_type_symbol")
        i_x = idx("_atom_site_fract_x")
        i_y = idx("_atom_site_fract_y")
        i_z = idx("_atom_site_fract_z")
        i_occ = idx("_atom_site_occupancy")
        i_biso = idx("_atom_site_b_iso_or_equiv")
        i_uiso = idx("_atom_site_u_iso_or_equiv")

        out: List[Dict[str, Any]] = []

        while j < len(lines):
            s = lines[j].strip()

            if not s:
                j += 1
                continue

            sl = s.lower()

            if sl.startswith("loop_") or sl.startswith("data_") or sl.startswith("save_"):
                break

            if sl.startswith("_"):
                break

            toks = _tokenize_cif_line(s)

            if len(toks) < len(headers):
                j += 1
                continue

            try:
                label = str(toks[i_label]).strip() if i_label is not None else f"site{len(out) + 1}"

                if not label:
                    label = f"site{len(out) + 1}"

                if i_el is not None:
                    el = clean_el_symbol(toks[i_el])
                else:
                    el = clean_el_symbol(label)

                fx = parse_cif_number(toks[i_x]) if i_x is not None else None
                fy = parse_cif_number(toks[i_y]) if i_y is not None else None
                fz = parse_cif_number(toks[i_z]) if i_z is not None else None

                if fx is None or fy is None or fz is None:
                    j += 1
                    continue

                occ = 1.0

                if i_occ is not None and i_occ < len(toks):
                    occ_v = parse_cif_number(toks[i_occ])
                    if occ_v is not None:
                        occ = float(occ_v)

                biso = None

                if i_biso is not None and i_biso < len(toks):
                    biso = parse_cif_number(toks[i_biso])

                elif i_uiso is not None and i_uiso < len(toks):
                    uiso = parse_cif_number(toks[i_uiso])

                    # Convert Uiso to Biso:
                    #   B = 8*pi^2*U
                    if uiso is not None:
                        biso = float(8.0 * np.pi * np.pi * float(uiso))

                out.append(
                    {
                        "label": label,
                        "el": el,
                        "frac": (float(fx), float(fy), float(fz)),
                        "occ": float(occ),
                        "biso": biso,
                    }
                )

            except Exception:
                pass

            j += 1

        return out

    return []


def _make_structure_from_fast_cif(text: str, sites: List[Dict[str, Any]]) -> Optional[Structure]:
    """
    Build a pymatgen Structure directly from parsed cell + atom loop.
    This avoids CifParser for P1/simple CIFs.
    """
    a = _scalar_float(text, "_cell_length_a")
    b = _scalar_float(text, "_cell_length_b")
    c = _scalar_float(text, "_cell_length_c")

    alpha = _scalar_float(text, "_cell_angle_alpha")
    beta = _scalar_float(text, "_cell_angle_beta")
    gamma = _scalar_float(text, "_cell_angle_gamma")

    if None in (a, b, c, alpha, beta, gamma):
        return None

    species = []
    frac = []

    for rec in sites:
        el = clean_el_symbol(rec.get("el", ""))

        if not el:
            continue

        species.append(el)
        frac.append(tuple(rec.get("frac", (0.0, 0.0, 0.0))))

    if not species:
        return None

    lat = Lattice.from_parameters(
        float(a),
        float(b),
        float(c),
        float(alpha),
        float(beta),
        float(gamma),
    )

    try:
        return Structure(
            lat,
            species,
            frac,
            coords_are_cartesian=False,
            to_unit_cell=False,
        )
    except TypeError:
        return Structure(
            lat,
            species,
            frac,
            coords_are_cartesian=False,
        )


@lru_cache(maxsize=16)
def _read_cif_asu_sites_cached(path: str, mtime: float):
    """
    Cached CIF parser.

    Fast path:
        P1/simple identity CIFs are parsed manually and ignore bond loops.

    Fallback:
        non-P1/more complex CIFs use pymatgen CifParser.
    """
    path = (path or "").strip()
    text = _read_text(path)

    sg_num = (
        _scalar_int(text, "_symmetry_Int_Tables_number")
        or _scalar_int(text, "_space_group_IT_number")
        or _scalar_int(text, "_space_group_IT_number")
    )

    fast_p1 = (
        sg_num == 1
        and _symmetry_ops_are_identity_only(text)
    )

    # ---------------------------------------------------------
    # Fast path for P1 / identity CIFs.
    # This is the important path for large Materials Studio P1 CIFs.
    # ---------------------------------------------------------
    if fast_p1:
        asu_sites = _parse_atom_site_loop_fast(text)
        full = _make_structure_from_fast_cif(text, asu_sites)

        if full is not None and asu_sites:
            frozen = tuple(tuple(sorted(d.items())) for d in asu_sites)
            return full, frozen

    # ---------------------------------------------------------
    # Fallback for non-P1 / complex CIFs.
    # ---------------------------------------------------------
    parser = CifParser(path)

    try:
        full = parser.parse_structures(primitive=False)[0]
    except AttributeError:
        full = parser.get_structures(primitive=False)[0]

    cif = parser.as_dict()
    block = next(iter(cif.values()))

    def _get_list(key: str) -> List[Any]:
        v = block.get(key, [])

        if isinstance(v, (list, tuple)):
            return list(v)

        if v is None:
            return []

        return [v]

    labels = _get_list("_atom_site_label")
    els = _get_list("_atom_site_type_symbol") or labels

    xs = _get_list("_atom_site_fract_x")
    ys = _get_list("_atom_site_fract_y")
    zs = _get_list("_atom_site_fract_z")
    occs = _get_list("_atom_site_occupancy")
    bisos = _get_list("_atom_site_B_iso_or_equiv")
    uisos = _get_list("_atom_site_U_iso_or_equiv")

    n = min(len(els), len(xs), len(ys), len(zs)) if els and xs and ys and zs else 0

    out: List[Dict[str, Any]] = []

    for i in range(n):
        raw_label = labels[i] if i < len(labels) else els[i]
        label = str(raw_label).strip()

        if not label:
            label = f"site{i + 1}"

        el = clean_el_symbol(els[i])

        fx = parse_cif_number(xs[i])
        fy = parse_cif_number(ys[i])
        fz = parse_cif_number(zs[i])

        if fx is None or fy is None or fz is None:
            continue

        occ = parse_cif_number(occs[i]) if i < len(occs) else 1.0

        biso = None

        if i < len(bisos):
            biso = parse_cif_number(bisos[i])

        elif i < len(uisos):
            uiso = parse_cif_number(uisos[i])
            if uiso is not None:
                biso = float(8.0 * np.pi * np.pi * float(uiso))

        out.append(
            {
                "label": label,
                "el": el,
                "frac": (fx, fy, fz),
                "occ": occ,
                "biso": biso,
            }
        )

    frozen = tuple(tuple(sorted(d.items())) for d in out)
    return full, frozen


def read_cif_asu_sites(path: str) -> Tuple[Optional[Structure], List[Dict[str, Any]]]:
    """
    Return (full_structure, asymmetric_unit_sites), cached by path+mtime.
    """
    path = (path or "").strip()

    if not path or not os.path.exists(path):
        return None, []

    try:
        mtime = os.path.getmtime(path)

        full, frozen_sites = _read_cif_asu_sites_cached(path, mtime)

        asu_sites = [dict(items) for items in frozen_sites]

        try:
            full = full.copy()
        except Exception:
            pass

        return full, asu_sites

    except Exception:
        try:
            return Structure.from_file(path), []
        except Exception:
            return None, []