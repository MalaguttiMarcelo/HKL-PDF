# pdf_fitting/models/crystallite_shapes.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Iterable

import numpy as np


@dataclass
class CrystalliteShapeSpec:
    shape_type: str = "cylinder"

    # NEW:
    # "predefined" = one fixed shape
    # "grid_search" = scan diameter/height ranges
    search_mode: str = "grid_search"

    diameter_cells: float = 10.0
    height_cells: float = 5.0

    diameter_min_cells: float = 5.0
    diameter_max_cells: float = 20.0
    height_min_cells: float = 2.0
    height_max_cells: float = 10.0
    step_cells: float = 1.0

    scan_diameter: bool = True
    scan_height: bool = True
    diameter_infinite: bool = False

    axis_h: int = 0
    axis_k: int = 0
    axis_l: int = 1

    base1_h: int = 1
    base1_k: int = 0
    base1_l: int = 0

    base2_h: int = 0
    base2_k: int = 1
    base2_l: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrystalliteShapeSpec":
        out = cls()

        def _to_bool(x) -> bool:
            if isinstance(x, bool):
                return x

            if isinstance(x, (int, float)):
                return bool(int(x))

            s = str(x).strip().lower()

            if s in ("1", "true", "yes", "y", "on"):
                return True

            if s in ("0", "false", "no", "n", "off"):
                return False

            return bool(x)

        for k, v in (d or {}).items():
            if not hasattr(out, k):
                continue

            old = getattr(out, k)

            try:
                if isinstance(old, bool):
                    setattr(out, k, _to_bool(v))
                elif isinstance(old, int):
                    setattr(out, k, int(float(v)))
                elif isinstance(old, float):
                    setattr(out, k, float(v))
                else:
                    setattr(out, k, str(v))
            except Exception:
                setattr(out, k, v)

        return out


def lattice_matrix_from_params(
    a: float,
    b: float,
    c: float,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
) -> np.ndarray:
    """
    Return matrix M with cart = frac @ M.

    This matches pymatgen-style row-vector lattice convention.
    """
    alpha_r = np.deg2rad(float(alpha))
    beta_r = np.deg2rad(float(beta))
    gamma_r = np.deg2rad(float(gamma))

    ca = np.cos(alpha_r)
    cb = np.cos(beta_r)
    cg = np.cos(gamma_r)
    sg = np.sin(gamma_r)

    volume = (
        float(a) * float(b) * float(c)
        * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    )

    ax, ay, az = float(a), 0.0, 0.0
    bx, by, bz = float(b) * cg, float(b) * sg, 0.0
    cx = float(c) * cb
    cy = float(c) * (ca - cb * cg) / sg
    cz = volume / (float(a) * float(b) * sg)

    return np.array(
        [
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz],
        ],
        dtype=float,
    )


def _safe_unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n > 1e-14:
        return v / n

    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)

    fallback = np.asarray(fallback, dtype=float)
    nf = float(np.linalg.norm(fallback))
    if nf <= 1e-14:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return fallback / nf


def direct_hkl_to_cart(h: int, k: int, l: int, lattice_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a direct crystallographic direction [h k l] to Cartesian.

    For shape orientation, this uses direct-space directions:
        v_cart = h*a_vec + k*b_vec + l*c_vec
    """
    coeff = np.array([float(h), float(k), float(l)], dtype=float)
    return coeff @ np.asarray(lattice_matrix, dtype=float)


def build_orientation_basis(
    lattice_matrix: np.ndarray,
    axis_hkl: Tuple[int, int, int] = (0, 0, 1),
    base1_hkl: Tuple[int, int, int] = (1, 0, 0),
    base2_hkl: Tuple[int, int, int] = (0, 1, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal Cartesian basis for the shape.

    Returns:
        e1, e2, e3

    where:
        e3 = cylinder/disk axis
        e1/e2 = base-plane axes
    """
    M = np.asarray(lattice_matrix, dtype=float)

    e3 = direct_hkl_to_cart(*axis_hkl, M)
    e3 = _safe_unit(e3, np.array([0.0, 0.0, 1.0]))

    b1 = direct_hkl_to_cart(*base1_hkl, M)

    # Remove projection of b1 along axis.
    b1_perp = b1 - np.dot(b1, e3) * e3

    if np.linalg.norm(b1_perp) < 1e-10:
        # Pick an arbitrary vector not parallel to e3.
        test = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(test, e3)) > 0.9:
            test = np.array([0.0, 1.0, 0.0])
        b1_perp = test - np.dot(test, e3) * e3

    e1 = _safe_unit(b1_perp)

    b2 = direct_hkl_to_cart(*base2_hkl, M)
    b2_perp = b2 - np.dot(b2, e3) * e3
    b2_perp = b2_perp - np.dot(b2_perp, e1) * e1

    if np.linalg.norm(b2_perp) < 1e-10:
        e2 = np.cross(e3, e1)
    else:
        e2 = b2_perp

    e2 = _safe_unit(e2)

    # Re-orthogonalize for numerical safety.
    e2 = _safe_unit(np.cross(e3, e1))
    e1 = _safe_unit(np.cross(e2, e3))

    return e1, e2, e3

def shape_dimension_lengths(
    lattice_matrix: np.ndarray,
    axis_hkl: Tuple[int, int, int] = (0, 0, 1),
    base1_hkl: Tuple[int, int, int] = (1, 0, 0),
    base2_hkl: Tuple[int, int, int] = (0, 1, 0),
) -> Tuple[float, float, float]:
    """
    Return real-space length scales for the shape basis.

    Returns
    -------
    base1_len : float
        Length in Å corresponding to one unit along the first base direction,
        after projecting perpendicular to the shape axis.

    base2_len : float
        Length in Å corresponding to one unit along the second base direction,
        after projecting perpendicular to the shape axis.

    axis_len : float
        Length in Å corresponding to one unit along the shape axis.

    Notes
    -----
    This replaces the old approximation:

        diameter_cells * mean(a, b)
        height_cells   * c

    which is only reasonable for an axis parallel to c in an approximately
    orthogonal cell.

    Now:
        height_cells follows the chosen axis [h k l]
        diameter_cells follows the chosen base vectors.
    """
    M = np.asarray(lattice_matrix, dtype=float)

    axis_vec = direct_hkl_to_cart(*axis_hkl, M)
    axis_len = float(np.linalg.norm(axis_vec))

    e3 = _safe_unit(axis_vec, np.array([0.0, 0.0, 1.0]))

    base1_vec = direct_hkl_to_cart(*base1_hkl, M)
    base2_vec = direct_hkl_to_cart(*base2_hkl, M)

    base1_perp = base1_vec - np.dot(base1_vec, e3) * e3
    base2_perp = base2_vec - np.dot(base2_vec, e3) * e3

    base1_len = float(np.linalg.norm(base1_perp))
    base2_len = float(np.linalg.norm(base2_perp))

    if axis_len <= 1e-12:
        # Safe fallback.
        axis_len = float(np.linalg.norm(M[2]))

    if base1_len <= 1e-12:
        base1_len = float(np.linalg.norm(M[0]))

    if base2_len <= 1e-12:
        base2_len = float(np.linalg.norm(M[1]))

    return base1_len, base2_len, axis_len


def generate_translation_grid(max_cells: int) -> np.ndarray:
    """
    Generate integer translation vectors around origin.

    Output:
        shape (N, 3), integer fractional-cell translations.
    """
    n = int(max(1, max_cells))
    vals = range(-n, n + 1)
    grid = np.array(
        [(i, j, k) for i in vals for j in vals for k in vals],
        dtype=float,
    )
    return grid


def shape_mask(
    cart: np.ndarray,
    spec: CrystalliteShapeSpec,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """
    Return boolean mask for Cartesian points inside the selected crystallite shape.

    The shape dimensions are now tied to the chosen crystallographic axis/base
    directions:

        height_cells   -> length along axis_hkl
        diameter_cells -> length along base1/base2 directions

    For cylinder/disk:
        a circular cylinder uses the average of the two base-direction length
        scales.

    For ellipsoid:
        base1 and base2 may have different real-space scales, so the cross-section
        can become elliptical if the basis directions have different lengths.
    """
    cart = np.asarray(cart, dtype=float)

    e1, e2, e3 = build_orientation_basis(
        lattice_matrix,
        axis_hkl=(spec.axis_h, spec.axis_k, spec.axis_l),
        base1_hkl=(spec.base1_h, spec.base1_k, spec.base1_l),
        base2_hkl=(spec.base2_h, spec.base2_k, spec.base2_l),
    )

    x = cart @ e1
    y = cart @ e2
    z = cart @ e3

    base1_len, base2_len, axis_len = shape_dimension_lengths(
        lattice_matrix,
        axis_hkl=(spec.axis_h, spec.axis_k, spec.axis_l),
        base1_hkl=(spec.base1_h, spec.base1_k, spec.base1_l),
        base2_hkl=(spec.base2_h, spec.base2_k, spec.base2_l),
    )

    D1_ang = float(spec.diameter_cells) * float(base1_len)
    D2_ang = float(spec.diameter_cells) * float(base2_len)
    H_ang = float(spec.height_cells) * float(axis_len)

    shape_type = str(spec.shape_type).strip().lower()

    if shape_type in ("sphere", "spherical"):
        R = 0.25 * (D1_ang + D2_ang)
        return (x * x + y * y + z * z) <= R * R

    if shape_type in ("ellipsoid", "ellipsoidal"):
        Rx = max(0.5 * D1_ang, 1e-12)
        Ry = max(0.5 * D2_ang, 1e-12)
        Rz = max(0.5 * H_ang, 1e-12)

        return (
            (x / Rx) ** 2
            + (y / Ry) ** 2
            + (z / Rz) ** 2
        ) <= 1.0

    if shape_type in ("cylinder", "disk", "disk-like", "disk_like"):
        # Circular cylinder in the chosen base plane.
        # If base1/base2 have different real-space lengths, use their average
        # so diameter_cells remains a single scalar.
        R = 0.25 * (D1_ang + D2_ang)
        half_h = 0.5 * H_ang

        return (x * x + y * y <= R * R) & (np.abs(z) <= half_h)

    if shape_type in ("prism", "rectangular_prism"):
        half_x = 0.5 * D1_ang
        half_y = 0.5 * D2_ang
        half_z = 0.5 * H_ang

        return (
            (np.abs(x) <= half_x)
            & (np.abs(y) <= half_y)
            & (np.abs(z) <= half_z)
        )

    # Fallback: cylinder.
    R = 0.25 * (D1_ang + D2_ang)
    half_h = 0.5 * H_ang

    return (x * x + y * y <= R * R) & (np.abs(z) <= half_h)

def build_shape_points_fractional(
    spec: CrystalliteShapeSpec,
    lattice_params: Dict[str, float],
    *,
    include_cell_basis_points: bool = False,
) -> Dict[str, Any]:
    """
    Build preview points in fractional-cell translation coordinates.

    This is for GUI preview, not final PDF calculation.

    Returns:
        {
            "frac": (N, 3) fractional/cell translation coordinates,
            "cart": (N, 3) Cartesian coordinates,
            "orientation": {
                "e1": ...,
                "e2": ...,
                "e3": ...
            }
        }
    """
    M = lattice_matrix_from_params(
        lattice_params.get("a", 1.0),
        lattice_params.get("b", 1.0),
        lattice_params.get("c", 1.0),
        lattice_params.get("alpha", 90.0),
        lattice_params.get("beta", 90.0),
        lattice_params.get("gamma", 90.0),
    )

    max_cells = int(
        np.ceil(
            max(
                float(spec.diameter_cells),
                float(spec.height_cells),
                1.0,
            )
        )
    ) + 2

    frac = generate_translation_grid(max_cells)

    # Center grid around origin. Translation coordinates are already centered.
    cart = frac @ M

    mask = shape_mask(cart, spec, M)

    frac_in = frac[mask]
    cart_in = cart[mask]

    e1, e2, e3 = build_orientation_basis(
        M,
        axis_hkl=(spec.axis_h, spec.axis_k, spec.axis_l),
        base1_hkl=(spec.base1_h, spec.base1_k, spec.base1_l),
        base2_hkl=(spec.base2_h, spec.base2_k, spec.base2_l),
    )

    return {
        "frac": frac_in,
        "cart": cart_in,
        "orientation": {
            "base1": e1,
            "base2": e2,
            "axis": e3,
        },
        "n_points": int(frac_in.shape[0]),
    }

def iter_shape_scan(spec: CrystalliteShapeSpec) -> Iterable[CrystalliteShapeSpec]:
    """
    Generate shape specs.

    search_mode = "predefined":
        generate exactly one shape:
            diameter_cells, height_cells

    search_mode = "grid_search":
        generate scan grid from min/max/step.
    """
    search_mode = str(getattr(spec, "search_mode", "grid_search")).strip().lower()

    if search_mode in ("predefined", "single", "fixed", "fixed_dimensions"):
        s = CrystalliteShapeSpec.from_dict(spec.to_dict())
        s.scan_diameter = False
        s.scan_height = False
        s.diameter_infinite = False
        yield s
        return

    step = max(float(spec.step_cells), 1e-12)

    if bool(getattr(spec, "scan_diameter", True)):
        dmin = float(spec.diameter_min_cells)
        dmax = float(spec.diameter_max_cells)
        d_vals = np.arange(dmin, dmax + 0.5 * step, step)
    else:
        if bool(getattr(spec, "diameter_infinite", False)):
            d_vals = np.asarray([np.inf], dtype=float)
        else:
            d_vals = np.asarray([float(spec.diameter_cells)], dtype=float)

    if bool(getattr(spec, "scan_height", True)):
        hmin = float(spec.height_min_cells)
        hmax = float(spec.height_max_cells)
        h_vals = np.arange(hmin, hmax + 0.5 * step, step)
    else:
        h_vals = np.asarray([float(spec.height_cells)], dtype=float)

    for d in d_vals:
        for h in h_vals:
            s = CrystalliteShapeSpec.from_dict(spec.to_dict())
            s.diameter_cells = float(d)
            s.height_cells = float(h)
            yield s