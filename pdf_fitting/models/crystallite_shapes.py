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

    # Backward-compatible generic dimensions.
    #
    # diameter_cells and height_cells are retained because old input files
    # and cylinder/disk calculations use them.
    diameter_cells: float = 10.0
    height_cells: float = 5.0

    # Shape-axis dimensions in unit-cell lengths.
    #
    # Meaning depends on shape_type:
    #
    # sphere:
    #     axis1_cells = sphere diameter
    #
    # cylinder / disk:
    #     axis1_cells = diameter
    #     axis2_cells = same as axis1_cells
    #     axis3_cells = height / thickness
    #
    # ellipsoid:
    #     axis1_cells = diameter along base direction 1
    #     axis2_cells = diameter along base direction 2
    #     axis3_cells = diameter along shape axis
    #
    # prism:
    #     axis1_cells = length along base direction 1
    #     axis2_cells = length along base direction 2
    #     axis3_cells = length along shape axis
    axis1_cells: float = 10.0
    axis2_cells: float = 10.0
    axis3_cells: float = 5.0

    # Axis-specific grid-search settings used by the new finite
    # crystallite Builder.
    scan_axis1: bool = True
    axis1_min_cells: float = 5.0
    axis1_max_cells: float = 20.0
    axis1_step_cells: float = 1.0

    scan_axis2: bool = False
    axis2_min_cells: float = 5.0
    axis2_max_cells: float = 20.0
    axis2_step_cells: float = 1.0

    scan_axis3: bool = True
    axis3_min_cells: float = 1.0
    axis3_max_cells: float = 15.0
    axis3_step_cells: float = 1.0


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
    def from_dict(
        cls,
        d: Dict[str, Any],
    ) -> "CrystalliteShapeSpec":
        """
        Read a shape specification.

        Old input files containing only:

            diameter_cells
            height_cells

        remain valid. Their values are copied into:

            axis1_cells
            axis2_cells
            axis3_cells
        """
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

        values = dict(d or {})

        for key, value in values.items():
            if not hasattr(out, key):
                continue

            old = getattr(out, key)

            try:
                if isinstance(old, bool):
                    setattr(
                        out,
                        key,
                        _to_bool(value),
                    )

                elif isinstance(old, int):
                    setattr(
                        out,
                        key,
                        int(float(value)),
                    )

                elif isinstance(old, float):
                    setattr(
                        out,
                        key,
                        float(value),
                    )

                else:
                    setattr(
                        out,
                        key,
                        str(value),
                    )

            except Exception:
                setattr(
                    out,
                    key,
                    value,
                )

        # ---------------------------------------------------------
        # Backward compatibility:
        #
        # If an old input has no axis1/axis2/axis3 dimensions, derive
        # them from the old diameter_cells / height_cells fields.
        # ---------------------------------------------------------
        if "axis1_cells" not in values:
            out.axis1_cells = float(
                out.diameter_cells
            )

        if "axis2_cells" not in values:
            out.axis2_cells = float(
                out.diameter_cells
            )

        if "axis3_cells" not in values:
            out.axis3_cells = float(
                out.height_cells
            )

        # ---------------------------------------------------------
        # Convert legacy diameter/height scan settings into the
        # axis-specific scan representation when the new keys are absent.
        # ---------------------------------------------------------
        if "scan_axis1" not in values:
            out.scan_axis1 = bool(
                out.scan_diameter
            )

        if "axis1_min_cells" not in values:
            out.axis1_min_cells = float(
                out.diameter_min_cells
            )

        if "axis1_max_cells" not in values:
            out.axis1_max_cells = float(
                out.diameter_max_cells
            )

        if "axis1_step_cells" not in values:
            out.axis1_step_cells = float(
                out.step_cells
            )

        if "scan_axis2" not in values:
            out.scan_axis2 = False

        if "axis2_min_cells" not in values:
            out.axis2_min_cells = float(
                out.axis1_min_cells
            )

        if "axis2_max_cells" not in values:
            out.axis2_max_cells = float(
                out.axis1_max_cells
            )

        if "axis2_step_cells" not in values:
            out.axis2_step_cells = float(
                out.axis1_step_cells
            )

        if "scan_axis3" not in values:
            out.scan_axis3 = bool(
                out.scan_height
            )

        if "axis3_min_cells" not in values:
            out.axis3_min_cells = float(
                out.height_min_cells
            )

        if "axis3_max_cells" not in values:
            out.axis3_max_cells = float(
                out.height_max_cells
            )

        if "axis3_step_cells" not in values:
            out.axis3_step_cells = float(
                out.step_cells
            )

        # Keep old aliases synchronized with the new values.
        out.diameter_cells = float(
            out.axis1_cells
        )

        out.height_cells = float(
            out.axis3_cells
        )

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

    shape_type = str(
        spec.shape_type
    ).strip().lower()

    axis1_cells = float(
        getattr(
            spec,
            "axis1_cells",
            spec.diameter_cells,
        )
    )

    axis2_cells = float(
        getattr(
            spec,
            "axis2_cells",
            spec.diameter_cells,
        )
    )

    axis3_cells = float(
        getattr(
            spec,
            "axis3_cells",
            spec.height_cells,
        )
    )

    # Sphere: only one physical dimension.
    if shape_type in (
        "sphere",
        "spherical",
    ):
        axis2_cells = axis1_cells
        axis3_cells = axis1_cells

    # Cylinder/disk: one circular diameter in the e1/e2 plane.
    elif shape_type in (
        "cylinder",
        "disk",
        "disk-like",
        "disk_like",
    ):
        axis2_cells = axis1_cells

    D1_ang = axis1_cells * float(base1_len)
    D2_ang = axis2_cells * float(base2_len)
    H_ang = axis3_cells * float(axis_len)

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


def iter_shape_scan(
    spec: CrystalliteShapeSpec,
) -> Iterable[CrystalliteShapeSpec]:
    """
    Generate finite crystallite-shape candidates.

    Predefined mode returns one candidate.

    Grid-search mode supports independent Axis 1, Axis 2, and Axis 3
    ranges from the finite-crystallite Builder.
    """
    search_mode = str(
        getattr(
            spec,
            "search_mode",
            "grid_search",
        )
        or "grid_search"
    ).strip().lower()

    if search_mode in {
        "predefined",
        "single",
        "fixed",
        "fixed_dimensions",
    }:
        candidate = CrystalliteShapeSpec.from_dict(
            spec.to_dict()
        )

        candidate.scan_axis1 = False
        candidate.scan_axis2 = False
        candidate.scan_axis3 = False

        candidate.scan_diameter = False
        candidate.scan_height = False
        candidate.diameter_infinite = False

        candidate.diameter_cells = float(
            candidate.axis1_cells
        )

        candidate.height_cells = float(
            candidate.axis3_cells
        )

        yield candidate
        return

    def scan_values(
        current_value: float,
        enabled: bool,
        minimum: float,
        maximum: float,
        step: float,
    ) -> np.ndarray:
        if not enabled:
            return np.asarray(
                [float(current_value)],
                dtype=float,
            )

        minimum = float(minimum)
        maximum = float(maximum)

        if maximum < minimum:
            minimum, maximum = (
                maximum,
                minimum,
            )

        step = max(
            float(step),
            1.0e-12,
        )

        return np.arange(
            minimum,
            maximum + 0.5 * step,
            step,
            dtype=float,
        )

    shape_type = str(
        spec.shape_type
        or "cylinder"
    ).strip().lower()

    axis1_values = scan_values(
        spec.axis1_cells,
        bool(spec.scan_axis1),
        spec.axis1_min_cells,
        spec.axis1_max_cells,
        spec.axis1_step_cells,
    )

    if shape_type in {
        "sphere",
        "spherical",
        "cylinder",
        "disk",
        "disk-like",
        "disk_like",
    }:
        axis2_values = np.asarray(
            [float(spec.axis1_cells)],
            dtype=float,
        )
    else:
        axis2_values = scan_values(
            spec.axis2_cells,
            bool(spec.scan_axis2),
            spec.axis2_min_cells,
            spec.axis2_max_cells,
            spec.axis2_step_cells,
        )

    axis3_values = scan_values(
        spec.axis3_cells,
        bool(spec.scan_axis3),
        spec.axis3_min_cells,
        spec.axis3_max_cells,
        spec.axis3_step_cells,
    )

    for axis1_value in axis1_values:
        for axis2_value in axis2_values:
            for axis3_value in axis3_values:
                candidate = CrystalliteShapeSpec.from_dict(
                    spec.to_dict()
                )

                candidate.axis1_cells = float(
                    axis1_value
                )

                if shape_type in {
                    "sphere",
                    "spherical",
                    "cylinder",
                    "disk",
                    "disk-like",
                    "disk_like",
                }:
                    candidate.axis2_cells = float(
                        axis1_value
                    )
                else:
                    candidate.axis2_cells = float(
                        axis2_value
                    )

                candidate.axis3_cells = float(
                    axis3_value
                )

                candidate.diameter_cells = float(
                    candidate.axis1_cells
                )

                candidate.height_cells = float(
                    candidate.axis3_cells
                )

                yield candidate

                
def shape_spec_from_config(
    config: Dict[str, Any] | None,
) -> CrystalliteShapeSpec:
    """
    Build a shape specification from the application configuration.
    """
    config = dict(
        config
        or {}
    )

    values = config.get(
        "crystallite_shape",
        config,
    )

    if not isinstance(
        values,
        dict,
    ):
        values = {}

    return CrystalliteShapeSpec.from_dict(
        values
    )

