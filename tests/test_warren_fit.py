"""
Direction-resolved width analysis (warren_fit) on synthetic peak widths with
known dynamic and contrast-factor (invariant) parameters — no PDF model needed.
"""

import math

import numpy as np

from pdf_fitting.models.microstrain_utils import Cell
from pdf_fitting.project.free_peaks import PeakCluster, PeakMember
from pdf_fitting.project.warren_fit import (
    WarrenOptions,
    _shape,
    cf_abscissa,
    fit_warren,
    invariant_features,
    member_hkl,
    starting_values,
)

A = 2.87
CELL = Cell(A, A, A, 90.0, 90.0, 90.0)
SG = 229
TRUE = {"B": 0.40, "E1": 4.0e-5, "E2": 1.0e-5, "Re": 100.0}
DIRS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (3, 1, 0)]


def _clusters(noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    clusters = []
    k = 0
    for d in DIRS:
        u = np.array(d, dtype=float)
        for n in (1, 2, 3):
            r = n * A * float(np.linalg.norm(u))
            m = PeakMember(
                pair="fe-fe", lambda_pair="fe-fe", k=k, r=r, family="<" + "".join(map(str, d)) + ">",
                vector=tuple(n * u), hkl_H=0.0, multiplicity=1.0, weight=1.0, sigma_model=0.1,
            )
            names, x = invariant_features(member_hkl(m, CELL), SG, CELL)
            S = TRUE["E1"] * x[0] + TRUE["E2"] * x[1]
            s2 = TRUE["B"] / (4 * math.pi ** 2) + S * float(_shape([r], TRUE["Re"], "wilkens")[0])
            sig = math.sqrt(s2) * (1.0 + noise * rng.standard_normal())
            clusters.append(PeakCluster(index=len(clusters), r=r, members=[m], shells=np.array([0]), sigma0=sig,
                                        sigma=sig, sigma_esd=max(1e-4, 0.01 * sig)))
            k += 1
    return clusters


START = {"B": {"fe": 0.3}, "lambda": {}, "delta1": 0.0, "delta2": 0.0, "delta_broad": 0.0, "delta_g": 0.0,
         "rho": 0.0, "fE": 0.5, "Re": 100.0, "burgers": 2.48, "sg": SG, "cell": CELL, "cf_edge": {}, "cf_screw": {}}


def test_cubic_invariant_matches_A_plus_BH():
    # Γ/(a d*)⁴ = E1 + 2(E2 − E1)·H for cubic (DISCO eq. 7, appendix A9)
    for d in DIRS:
        _n, x = invariant_features(d, SG, CELL)
        H = cf_abscissa(d, SG, CELL)
        assert abs((x[0] * 1.0 + x[1] * 0.5) - (1.0 + 2 * (0.5 - 1.0) * H)) < 1e-12


def test_invariant_fit_recovers_contrast_coefficients():
    w = fit_warren(_clusters(), elements=["Fe"], start=START, options=WarrenOptions(static_model="invariant", n_lambda=0))
    assert abs(w.value("B_fe") - TRUE["B"]) < 1e-4
    assert abs(w.inv_values[0] - TRUE["E1"]) / TRUE["E1"] < 1e-3
    assert abs(w.inv_values[1] - TRUE["E2"]) / TRUE["E1"] < 1e-3


def test_independent_directions_then_invariant_least_squares():
    w = fit_warren(_clusters(noise=0.002), elements=["Fe"], start=START, options=WarrenOptions(static_model="direction", n_lambda=0))
    assert all(i.own for i in w.family_info.values())
    assert w.inv_source.startswith("least squares")
    assert abs(w.inv_values[0] - TRUE["E1"]) / TRUE["E1"] < 0.15
    assert abs(w.inv_values[1] - TRUE["E2"]) / TRUE["E1"] < 0.15
    names = [n for n, *_ in starting_values(w)]
    assert "biso_Fe" in names


def test_per_direction_B_needs_enough_peaks():
    w = fit_warren(_clusters(), elements=["Fe"], start=START,
                   options=WarrenOptions(static_model="direction", B_mode="direction", min_peaks=4, n_lambda=0))
    assert not any(i.own for i in w.family_info.values())
    assert any("other" in n for n in w.names)
