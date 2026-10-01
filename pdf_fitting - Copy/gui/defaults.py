from __future__ import annotations

from typing import Dict, Any

# -----------------------------------------------------------------------------
# Parameter metadata for tooltips (units + description)
# Extend this dictionary as needed.
# -----------------------------------------------------------------------------
PARAM_META: Dict[str, Dict[str, str]] = {
    "scale": {"unit": "-", "desc": "Overall scale factor applied to G(r)."},
    "qdamp": {
        "unit": "1/Å",
        "desc": "Instrument resolution damping (PDFgui-style Qdamp). Larger -> stronger damping at high r.",
    },
    "qmax": {
        "unit": "1/Å",
        "desc": "Maximum Q cutoff used to apply Fourier-termination ripples through a sinc-equivalent FFT filter.",
    },
    "delta1": {"unit": "Å", "desc": "Correlated motion parameter (delta1)."},
    "delta2": {"unit": "Å", "desc": "Correlated motion parameter (delta2)."},
    "delta_broad": {
        "unit": "-",
        "desc": "Additional distance-dependent Gaussian broadening. Adds variance (delta_broad * r_ij)^2 to each PDF peak.",
    },
    "biso_a": {
        "unit": "Å²",
        "desc": "Isotropic ADP for element A (or group A) depending on structure parsing.",
    },
    "biso_b": {
        "unit": "Å²",
        "desc": "Isotropic ADP for element B (or group B) depending on structure parsing.",
    },
    "a": {"unit": "Å", "desc": "Lattice parameter a."},
    "b": {"unit": "Å", "desc": "Lattice parameter b."},
    "c": {"unit": "Å", "desc": "Lattice parameter c."},
    "alpha": {"unit": "deg", "desc": "Lattice angle alpha."},
    "beta": {"unit": "deg", "desc": "Lattice angle beta."},
    "gamma": {"unit": "deg", "desc": "Lattice angle gamma."},
    "d": {"unit": "Å", "desc": "Crystallite size / characteristic length scale (model-specific)."},

    "cyl_diameter": {
        "unit": "Å",
        "desc": "Cylinder/disk diameter D for the HKL-dependent common-volume function. Inactive if <= 0.",
    },
    "cyl_thickness": {
        "unit": "Å",
        "desc": "Mean cylinder/disk thickness t for the HKL-dependent common-volume function.",
    },
    "cyl_thickness_std": {
        "unit": "Å",
        "desc": "Real-space standard deviation of lognormal thickness distribution. Use 0 for monodisperse thickness.",
    },
    "cyl_axis_h": {
        "unit": "-",
        "desc": "Cylinder axis h index. Default axis is [0 0 1].",
    },
    "cyl_axis_k": {
        "unit": "-",
        "desc": "Cylinder axis k index. Default axis is [0 0 1].",
    },
    "cyl_axis_l": {
        "unit": "-",
        "desc": "Cylinder axis l index. Default axis is [0 0 1].",
    },

    "saxs_scale": {
        "unit": "-",
        "desc": "Independent scale factor for the SAXS/missing-low-Q belly term. 1.0 gives the physically tied shape correction.",
    },
    "saxs_diameter": {
        "unit": "Å",
        "desc": "Independent effective diameter for the SAXS/shape-belly term. If 0, the main crystallite size model is used.",
    },
    "saxs_diameter_std": {
        "unit": "Å",
        "desc": "Real-space standard deviation for the SAXS/shape-belly diameter distribution. Use 0 for monodisperse.",
    },
    "saxs_height": {
        "unit": "Å",
        "desc": "Independent cylinder/disk height or thickness for the SAXS/shape-belly term. If 0, the main cylinder thickness is used.",
    },
    "saxs_height_std": {
        "unit": "Å",
        "desc": "Real-space standard deviation for the SAXS/shape-belly height/thickness distribution. Use 0 for monodisperse.",
    },
    
}

# NOTE: Keep this as a small, "good defaults" table. GUI layers can add/override.
DEFAULT_PARAMS: Dict[str, Any] = {
    # Instrumental
    "qdamp": 0.04,
    "delta_broad": 0.0,
    "qmax": 0.0,
    "qmax_zeros": 5,

    # Structural
    "scale": 1.0,
    "a": None, "b": None, "c": None,
    "alpha": None, "beta": None, "gamma": None,

    # Microstructure
    "d": 50.0,
    "d_std": 0.2,
    "delta_g": 0.0,

    # Local dynamics
    "delta1": 0.0,
    "delta2": 0.0,

    # Wilkens core
    "rho": 2.2e-4,
    "re": 153.65,
    "fe": 0.5,

    # PAH microstrain
    "pah_a": 0.0,
    "pah_b": 0.0,

        # Cylinder/disk common-volume model.
    # Inactive when diameter or thickness is <= 0.
    "cyl_diameter": 0.0,
    "cyl_thickness": 0.0,
    "cyl_thickness_std": 0.0,

    # Cylinder axis as a direct crystallographic direction.
    # Default [0 0 1] = c-axis.
    "cyl_axis_h": 0.0,
    "cyl_axis_k": 0.0,
    "cyl_axis_l": 1.0,

    # Independent SAXS / missing-low-Q belly term
    "saxs_scale": 1.0,
    "saxs_diameter": 0.0,
    "saxs_diameter_std": 0.0,
    "saxs_height": 0.0,
    "saxs_height_std": 0.0,
}
