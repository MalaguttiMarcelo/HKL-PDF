"""
Dislocation contrast factors of a phase: invariant (Popa / TOPAS / DISCO)
terms per Laue class, evaluation from the phase parameters, and import of
coefficients from a file.

C_hkl = fE·C_edge + (1 − fE)·C_screw with, for each of edge and screw,

    cubic A/B form   C = A + B·H,  H = (h²k² + h²l² + k²l²)/(h² + k² + l²)²
                     (CEdgeA, CEdgeB, CScrewA, CScrewB)
    invariants       C = Σ_i E_i x_i(hkl) · d⁴/a⁴
                     (EdgeE1…EdgeEn, ScrewE1…ScrewEn; n set by the Laue class)

as in PDFCalculator._contrast_factor_one_from_params.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CUBIC_KEYS = ("CEdgeA", "CEdgeB", "CScrewA", "CScrewB")

# Term labels per Laue-class rule (h, k, l raw indices; × d⁴/a⁴).
_TERMS = {
    "triclinic": ["h⁴", "k⁴", "l⁴", "2h²k²", "2k²l²", "2h²l²", "4h³k", "4h³l", "4hk³", "4k³l", "4hl³", "4kl³", "4h²kl", "4hk²l", "4hkl²"],
    "monoclinic": ["h⁴", "k⁴", "l⁴", "2h²l²", "2k²l²", "2h²k²", "4h³l", "4hl³", "4hk²l"],
    "orthorhombic": ["h⁴", "k⁴", "l⁴", "2h²k²", "2k²l²", "2h²l²"],
    "tetragonal_low": ["h⁴ + k⁴", "l⁴", "2h²k²", "2(h² + k²)l²", "4hk(h² − k²)"],
    "tetragonal": ["h⁴ + k⁴", "l⁴", "2h²k²", "2(h² + k²)l²"],
    "trigonal_low": ["A²", "2A·l²", "l⁴", "4hkl(h − k)", "4l(h³ − 3hk² + k³)"],
    "trigonal": ["A²", "2A·l²", "l⁴", "4hkl(h − k)"],
    "hexagonal": ["A²", "2A·l²", "l⁴"],
    "cubic": ["h⁴ + k⁴ + l⁴", "2(h²k² + h²l² + k²l²)"],
}


def laue_rule(sg: int) -> str:
    sg = int(sg)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 88:
        return "tetragonal_low"
    if sg <= 142:
        return "tetragonal"
    if sg <= 148:
        return "trigonal_low"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def invariant_terms(sg: int) -> List[Tuple[str, str]]:
    """[(name 'E1', term label)] for the Laue class of space group ``sg``."""
    return [(f"E{i + 1}", t) for i, t in enumerate(_TERMS[laue_rule(sg)])]


def invariant_note(sg: int) -> str:
    rule = laue_rule(sg)
    text = ", ".join(f"E{i + 1}: {t}" for i, t in enumerate(_TERMS[rule]))
    if rule in ("hexagonal", "trigonal", "trigonal_low"):
        text += "  (A = h² + hk + k²)"
    return f"{rule.replace('_', ' ')} Laue class (space group {sg}) · C = Σ E_i x_i · d⁴/a⁴ · {text}"


def parameter_names(sg: Optional[int]) -> List[str]:
    """Contrast-factor parameter names offered for a phase."""
    n = len(_TERMS[laue_rule(sg)]) if sg else 3
    names = list(CUBIC_KEYS) if (sg or 0) >= 195 else []
    names += ["burgers_mag"]
    names += [f"EdgeE{i}" for i in range(1, n + 1)] + [f"ScrewE{i}" for i in range(1, n + 1)]
    if (sg or 0) < 195:
        names += list(CUBIC_KEYS)
    return names


# -----------------------------------------------------------------------------
# Evaluation from a parameter dict
# -----------------------------------------------------------------------------


def _lc(params: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in (params or {}).items():
        try:
            out[str(k).strip().lower().replace("_", "").replace(" ", "")] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def has_cubic(params) -> bool:
    p = _lc(params)
    return any(k.lower() in p for k in CUBIC_KEYS)


def has_family(params, kind: str) -> bool:
    return any(re.match(rf"^{kind}e\d+$", k) for k in _lc(params))


def contrast(params: Dict[str, Any], hkl, fE: float, sg: int, cell) -> float:
    """C_hkl for a (float) reciprocal direction, as the PDF model computes it."""
    from pdf_fitting.models.microstrain_utils import d_spacing, feature_vector

    h, k, l = (float(x) for x in hkl)
    p = _lc(params)
    if has_cubic(params):
        Ae, Be = p.get("cedgea", 0.265280), p.get("cedgeb", -0.355950)
        As, Bs = p.get("cscrewa", 0.307288), p.get("cscrewb", -0.819979)
        s = h * h + k * k + l * l
        H = (h * h * k * k + h * h * l * l + k * k * l * l) / (s * s) if s else 0.0
        return fE * (Ae + Be * H) + (1.0 - fE) * (As + Bs * H)
    he, hs = has_family(params, "edge"), has_family(params, "screw")
    if not (he or hs) or cell is None or not sg:
        return 0.0
    names, x = feature_vector(int(sg), h, k, l, cell)
    x = x * (d_spacing(cell, h, k, l) ** 4) / (float(cell.a) ** 4)
    ce = float(sum(p.get(f"edge{n.lower()}", 0.0) * xi for n, xi in zip(names, x)))
    cs = float(sum(p.get(f"screw{n.lower()}", 0.0) * xi for n, xi in zip(names, x)))
    if he and not hs:
        cs = ce
    elif hs and not he:
        ce = cs
    return fE * ce + (1.0 - fE) * cs


# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------

_CANON = {k.lower(): k for k in CUBIC_KEYS + ("burgers_mag", "rho", "fe", "re")}


def _canonical_name(raw: str) -> Optional[Tuple[str, ...]]:
    k = raw.strip().strip('"').replace(" ", "").replace("_", "")
    kl = k.lower()
    m = re.match(r"^(edge|screw)e(\d+)$", kl)
    if m:
        return (("Edge" if m.group(1) == "edge" else "Screw") + f"E{int(m.group(2))}",)
    m = re.match(r"^e(\d+)$", kl)
    if m:  # a single set (e.g. an average or a DISCO fit): edge and screw alike
        return (f"EdgeE{int(m.group(1))}", f"ScrewE{int(m.group(1))}")
    if kl in ("burgers", "burgersmag", "bmag", "b"):
        return ("burgers_mag",)
    if kl in _CANON:
        return (_CANON[kl],)
    return None


def read_contrast_file(path: str) -> Dict[str, float]:
    """
    Contrast-factor coefficients from a text or JSON file.

    Accepted: JSON object {"EdgeE1": 0.1, ...}; or lines "name = value",
    "name: value", "name value" (# comments). Names: EdgeE<i>, ScrewE<i>,
    E<i> (sets edge and screw), CEdgeA/B, CScrewA/B, burgers_mag, rho, fE, Re.
    """
    text = open(path, encoding="utf-8", errors="replace").read()
    pairs: List[Tuple[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            pairs = list(data.items())
    except ValueError:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z][\w ]*?)\s*(?:=|:|\s)\s*([-+0-9.eE]+)", line)
            if m:
                pairs.append((m.group(1), m.group(2)))
    out: Dict[str, float] = {}
    for k, v in pairs:
        names = _canonical_name(str(k))
        if not names:
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        for n in names:
            out[n] = val
    if not out:
        raise ValueError("No contrast-factor coefficients found (expected e.g. 'EdgeE1 = 0.12').")
    return out


def phase_space_group(cif_path: str, params: Dict[str, Any]) -> Tuple[int, Any]:
    """(space-group number, Cell) of a phase with its current lattice."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from pdf_fitting.models.microstrain_utils import Cell
    from pdf_fitting.project.structure_tools import phase_structure

    st = phase_structure(cif_path, params).structure
    sg = int(SpacegroupAnalyzer(st, symprec=1e-3).get_space_group_number())
    a, b, c = st.lattice.abc
    al, be, ga = st.lattice.angles
    return sg, Cell(float(a), float(b), float(c), float(al), float(be), float(ga))


def as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)
