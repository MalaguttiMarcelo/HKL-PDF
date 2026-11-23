"""
Which panel shows which parameter, and structure information per phase.

Parameter names here are the local (un-namespaced) names used inside one
phase, exactly as in the single-phase code.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pdf_fitting.gui.defaults import DEFAULT_PARAMS, PARAM_META

LATTICE = ("a", "b", "c", "alpha", "beta", "gamma")
SIZE = ("d", "d_std")
CYLINDER = ("cyl_diameter", "cyl_thickness", "cyl_thickness_std", "cyl_axis_h", "cyl_axis_k", "cyl_axis_l")
SAXS = ("saxs_scale", "saxs_diameter", "saxs_diameter_std", "saxs_height", "saxs_height_std")
STRAIN_BY_MODEL = {
    "isotropic": ("delta_g",),
    "wilkens": ("rho", "re", "fe"),
    "pah": ("pah_a", "pah_b", "fe"),
}
CONTRAST_FIXED = ("cedgea", "cedgeb", "cscrewa", "cscrewb", "burgers_mag", "burgers", "b_mag")
INSTRUMENT = ("qdamp", "delta_broad", "qmax", "qmax_zeros", "qmax_pad")

LABELS = {
    "scale": "Scale factor",
    "a": "a (Å)", "b": "b (Å)", "c": "c (Å)",
    "alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)",
    "qdamp": "Qdamp (Å⁻¹)", "delta_broad": "δ broad", "qmax": "Qmax (Å⁻¹)",
    "qmax_zeros": "Qmax zero padding", "qmax_pad": "Qmax pad mode",
    "d": "Crystallite size d (Å)", "d_std": "Size spread d_std (Å)",
    "delta_g": "Isotropic microstrain",
    "rho": "Dislocation density ρ", "re": "Outer cut-off Re", "fe": "Edge fraction fE",
    "pah_a": "PAH a", "pah_b": "PAH b",
    "delta1": "δ1 (all pairs)", "delta2": "δ2 (all pairs)",
}


def is_contrast(name: str) -> bool:
    n = name.lower()
    return n in CONTRAST_FIXED or bool(re.match(r"^(edge|screw)[_ ]*e\d+$", n))


def is_site_param(name: str) -> bool:
    return bool(re.match(r"^(x|y|z|occ)_", name.lower()))


def is_dynamics(name: str) -> bool:
    n = name.lower()
    return n in ("delta1", "delta2") or n.startswith(("delta1_", "delta2_"))


def is_lambda(name: str) -> bool:
    return name.lower().startswith(("lambda", "lam"))


def is_stacking(name: str) -> bool:
    return name.lower().startswith(("stack_", "stacking_"))


def is_beta(name: str) -> bool:
    """Anisotropic displacement component beta<ij>_<site> (not the lattice angle beta)."""
    return bool(re.match(r"^beta(11|22|33|12|13|23)_.+", str(name).lower()))


def section_of(name: str, element_symbols=()) -> str:
    """Panel id for a phase parameter."""
    n = name.lower()
    if n == "scale":
        return "overview"
    if n in LATTICE:
        return "lat"
    if is_site_param(n) or n.startswith("biso_") or is_beta(n):
        return "atoms"
    if n in SIZE or n in CYLINDER or is_contrast(n) or n in {m for v in STRAIN_BY_MODEL.values() for m in v}:
        return "micro"
    if n in SAXS:
        return "saxs"
    if is_dynamics(n) or is_lambda(n):
        return "dyn"
    if is_stacking(n):
        return "shape"
    return "other"


def label_for(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    n = name.lower()
    if n.startswith("biso_"):
        return f"Biso {name[5:]} (Å²)"
    if is_beta(n):
        return f"β{name[4:6]} {name[7:]}"
    if n.startswith("delta1_"):
        return f"δ1 {name[7:]}"
    if n.startswith("delta2_"):
        return f"δ2 {name[7:]}"
    return name


def tooltip_for(name: str) -> str:
    meta = PARAM_META.get(name) or PARAM_META.get(name.lower())
    if not meta:
        return name
    unit = meta.get("unit", "")
    return f"{name} [{unit}]\n{meta.get('desc', '')}" if unit and unit != "-" else f"{name}\n{meta.get('desc', '')}"


def default_value(name: str) -> float:
    try:
        return float(DEFAULT_PARAMS.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


# -----------------------------------------------------------------------------
# Structure information of a phase (CIF), cached by path + mtime
# -----------------------------------------------------------------------------


class PhaseStructureInfo:
    def __init__(self, path: str):
        self.path = path
        self.ok = False
        self.error = ""
        self.sites: List[Dict[str, Any]] = []
        self.elements: List[str] = []
        self.pairs: List[str] = []
        self.lattice: Dict[str, float] = {}
        self.constraints: Dict[str, Any] = {}
        self.spacegroup = ""
        self.formula = ""
        self.z_avg = 1.0  # mean atomic number per atom (occupancy weighted)
        self.m_avg = 1.0  # mean atomic mass per atom

        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            from pdf_fitting.gui.cif_utils import clean_el_symbol, read_cif_asu_sites
            from pdf_fitting.structure_handler import get_lattice_constraints

            structure, asu = read_cif_asu_sites(path)

            if structure is None:
                raise ValueError("could not read the structure")

            self.sites = list(asu or [])
            els = []
            for site in structure:
                for sp in site.species:
                    el = clean_el_symbol(str(getattr(sp, "symbol", sp)))
                    if el and el not in els:
                        els.append(el)
            self.elements = els
            self.pairs = [
                f"{a.lower()}-{b.lower()}"
                for i, a in enumerate(sorted(els))
                for b in sorted(els)[i:]
            ]
            a, b, c, al, be, ga = structure.lattice.parameters
            self.lattice = {"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga}
            self.constraints = get_lattice_constraints(structure)
            try:
                sga = SpacegroupAnalyzer(structure, symprec=1e-3)
                self.spacegroup = f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()})"
            except Exception:
                self.spacegroup = "?"
            self.formula = structure.composition.reduced_formula
            comp = structure.composition
            total = float(comp.num_atoms) or 1.0
            self.z_avg = sum(float(n) * el.Z for el, n in comp.element_composition.items()) / total
            self.m_avg = float(comp.weight) / total
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)


@lru_cache(maxsize=32)
def _structure_info_cached(path: str, mtime: float) -> PhaseStructureInfo:
    return PhaseStructureInfo(path)


def structure_info(path: str) -> Optional[PhaseStructureInfo]:
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return _structure_info_cached(os.path.abspath(path), mtime)
