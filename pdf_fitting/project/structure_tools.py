"""
Structure of a phase as currently parameterised (Qt-free).

``phase_structure`` builds the full conventional cell of a phase from its CIF
with the project's current lattice, fractional-coordinate (x_/y_/z_) and
occupancy (occ_) parameters applied, using the same symmetry maps as the
coordinate refinement. Also: site groups (CIF labels), neighbor tables with
the λ shell index k, and per-site symmetry constraints.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pdf_fitting.structure_handler import (
    build_coordinate_maps,
    get_lattice_constraints,
    safe_param_key,
    site_coordinate_basis,
    site_element,
    site_label,
    symmetry_operations,
)

LATTICE = ("a", "b", "c", "alpha", "beta", "gamma")


class PhaseStructure:
    """Full cell of a phase with group bookkeeping."""

    def __init__(self, structure, site_group, group_keys, group_labels, reference, cif_path: str):
        self.structure = structure
        self.site_group = np.asarray(site_group, dtype=np.int64)
        self.group_keys = list(group_keys)        # safe keys: "o1"
        self.group_labels = list(group_labels)    # CIF labels: "O1"
        self.reference = np.asarray(reference, dtype=float)  # CIF ASU positions
        self.cif_path = cif_path
        self._ops = None

    @property
    def ops(self):
        if self._ops is None:
            self._ops = symmetry_operations(self.structure)
        return self._ops

    def group_of_site(self, site_index: int) -> int:
        return int(self.site_group[int(site_index)])

    def label_of_site(self, site_index: int) -> str:
        return self.group_labels[self.group_of_site(site_index)]

    def key_of_site(self, site_index: int) -> str:
        return self.group_keys[self.group_of_site(site_index)]

    def multiplicity(self, group: int) -> int:
        return int(np.count_nonzero(self.site_group == int(group)))

    def constraints(self, group: int) -> Dict[str, str]:
        """{'x': 'free'|'tied'|'fixed', ...} for the coordinates of a site group."""
        B, free = site_coordinate_basis(self.reference[int(group)], self.ops)
        out = {}
        for axis, letter in enumerate("xyz"):
            if axis in free:
                out[letter] = "free"
            elif B.size and np.any(np.abs(B[axis]) > 1e-9):
                out[letter] = "tied"
            else:
                out[letter] = "fixed"
        return out


def _load_cif(path: str):
    from pdf_fitting.gui.cif_utils import read_cif_asu_sites

    structure, asu = read_cif_asu_sites(path)
    if structure is None:
        raise ValueError(f"Could not read the structure: {path}")
    return structure, list(asu or [])


def phase_structure(cif_path: str, params: Dict[str, Any]) -> PhaseStructure:
    """
    Full conventional cell with the current lattice, coordinates and
    occupancies of the phase (``params``: local parameter values).
    """
    from pymatgen.core import Lattice, Structure

    base, asu = _load_cif(cif_path)

    # Groups by CIF label (same rule as StructureHandler).
    labels = [site_label(site, i) for i, site in enumerate(base.sites)]
    group_index: Dict[str, int] = {}
    group_labels: List[str] = []
    site_group = np.zeros(len(base), dtype=np.int64)
    for i, label in enumerate(labels):
        key = safe_param_key(label) or f"site{i + 1}"
        if key not in group_index:
            group_index[key] = len(group_index)
            group_labels.append(label)
        site_group[i] = group_index[key]
    group_keys = list(group_index)

    reference_by_key = {}
    for rec in asu:
        k = safe_param_key(rec.get("label", ""))
        if k and k not in reference_by_key and rec.get("frac") is not None:
            reference_by_key[k] = np.asarray(rec["frac"], dtype=float)
    frac = np.asarray(base.frac_coords, dtype=float)
    reference = np.zeros((len(group_keys), 3))
    for g, k in enumerate(group_keys):
        first = int(np.flatnonzero(site_group == g)[0])
        reference[g] = reference_by_key.get(k, frac[first])

    # Lattice with symmetry constraints (b = a, fixed angles, ...).
    lat = dict(zip(LATTICE, base.lattice.parameters))
    for k in LATTICE:
        try:
            if k in params:
                lat[k] = float(params[k])
        except (TypeError, ValueError):
            pass
    for k, v in get_lattice_constraints(base).items():
        lat[k] = lat[v] if isinstance(v, str) else float(v)
    lattice = Lattice.from_parameters(*(lat[k] for k in LATTICE))

    # Coordinates: groups whose x_/y_/z_ differ from the CIF.
    lookup = {str(k).lower(): v for k, v in (params or {}).items()}
    moving = []
    for g, k in enumerate(group_keys):
        for axis, letter in enumerate("xyz"):
            v = lookup.get(f"{letter}_{k}")
            try:
                if v is not None and abs(float(v) - reference[g, axis]) > 1e-9:
                    moving.append(g)
                    break
            except (TypeError, ValueError):
                pass

    new_frac = frac.copy()
    if moving:
        maps = build_coordinate_maps(base, site_group, group_keys, reference, moving)
        for g, grp in maps.items():
            free = grp["free_axes"]
            shift = np.zeros(len(free))
            for j, axis in enumerate(free):
                v = lookup.get(f"{'xyz'[axis]}_{grp['key']}")
                if v is not None:
                    shift[j] = float(v) - grp["p0"][axis]
            delta = grp["B"] @ shift if free else np.zeros(3)
            for s, R in zip(grp["sites"], grp["rotations"]):
                new_frac[s] = frac[s] + R @ delta

    # Occupancies (total occupancy per site; species scaled proportionally).
    species = []
    for i, site in enumerate(base.sites):
        comp = {str(sp): float(o) for sp, o in site.species.items()}
        total = sum(comp.values()) or 1.0
        v = lookup.get(f"occ_{group_keys[site_group[i]]}")
        try:
            if v is not None:
                scale = max(min(float(v), 1.0), 0.0) / total
                comp = {sp: o * scale for sp, o in comp.items()}
        except (TypeError, ValueError):
            pass
        species.append(comp)

    props = {k: list(v) for k, v in base.site_properties.items()}
    structure = Structure(lattice, species, new_frac, site_properties=props or None,
                          coords_are_cartesian=False, to_unit_cell=False)
    for i, label in enumerate(labels):
        try:
            structure[i].label = label
        except Exception:  # noqa: BLE001
            pass
    return PhaseStructure(structure, site_group, group_keys, group_labels, reference, cif_path)


# -----------------------------------------------------------------------------
# Neighbors, shells, geometry
# -----------------------------------------------------------------------------


def pair_name(el_a: str, el_b: str) -> str:
    a, b = sorted((el_a.lower(), el_b.lower()))
    return f"{a}-{b}"


def pair_shell_distances(ps: PhaseStructure, el_a: str, el_b: str, rmax: float, decimals: int = 4) -> List[float]:
    """Sorted unique distances of an element pair up to rmax (shell k = index)."""
    s = ps.structure
    out = set()
    want = {el_a.lower(), el_b.lower()}
    centers = [i for i, site in enumerate(s) if site_element(site).lower() in want]
    for i in centers:
        ei = site_element(s[i]).lower()
        other = (want - {ei}) or {ei}
        for nb in s.get_neighbors(s[i], rmax):
            if site_element(nb).lower() in other:
                out.add(round(float(nb.nn_distance), decimals))
    return sorted(out)


def neighbor_table(ps: PhaseStructure, site_index: int, rmax: float = 4.0) -> List[Dict[str, Any]]:
    """
    Neighbors of one site, grouped by (neighbor label, distance):
    [{"label", "element", "distance", "count", "pair", "k"}].
    """
    s = ps.structure
    center = s[int(site_index)]
    el_c = site_element(center)
    groups: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for nb in s.get_neighbors(center, rmax):
        idx = int(nb.index)
        label = ps.label_of_site(idx)
        d = round(float(nb.nn_distance), 4)
        key = (label, d)
        if key not in groups:
            groups[key] = {
                "label": label,
                "element": site_element(nb),
                "distance": float(nb.nn_distance),
                "count": 0,
                "pair": pair_name(el_c, site_element(nb)),
            }
        groups[key]["count"] += 1
    rows = sorted(groups.values(), key=lambda r: (r["distance"], r["label"]))
    shells: Dict[str, List[float]] = {}
    for r in rows:
        if r["pair"] not in shells:
            a, b = r["pair"].split("-")
            shells[r["pair"]] = pair_shell_distances(ps, a, b, rmax + 0.01)
        dist_list = shells[r["pair"]]
        r["k"] = int(np.argmin(np.abs(np.asarray(dist_list) - r["distance"]))) if dist_list else None
    return rows


def distance(p1, p2) -> float:
    return float(np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)))


def angle(p1, vertex, p3) -> float:
    a = np.asarray(p1, float) - np.asarray(vertex, float)
    b = np.asarray(p3, float) - np.asarray(vertex, float)
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
