"""Anisotropic displacement parameters (β): symmetry constraints, CIF round trip, model consistency."""

import io
import contextlib

import numpy as np

from pdf_fitting.project.edited_cif import base_sites, cif_aniso_betas, cif_text_with_sites
from pdf_fitting.structure_handler import ADP_COMPONENTS, beta_constraints

CALCITE = "examples/2-Calcite-ball-milled/Calcite.cif"


def _rot_z(n):
    """n-fold rotation about c in a hexagonal (γ = 120°) fractional basis."""
    R3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=float)
    out, R = [np.eye(3)], np.eye(3)
    for _ in range(n - 1):
        R = R3 @ R
        out.append(R)
    return out


def test_constraints_threefold_axis():
    cons = beta_constraints(_rot_z(3))
    assert cons["11"][0] == "free" and cons["33"][0] == "free"
    assert cons["22"] == ("tied", [("11", 1.0)])
    assert cons["12"][0] == "tied" and abs(cons["12"][1][0][1] - 0.5) < 1e-12
    assert cons["13"][0] == "fixed" and cons["23"][0] == "fixed"


def test_constraints_cubic_site():
    # m-3m point group: β11 = β22 = β33, off-diagonal 0
    import itertools

    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = signs[i]
            rots.append(R)
    cons = beta_constraints(rots)
    assert [cons[c][0] for c in ADP_COMPONENTS] == ["free", "tied", "tied", "fixed", "fixed", "fixed"]


def test_cif_beta_roundtrip(tmp_path):
    sites = base_sites(CALCITE)
    sites[0]["beta"] = [0.01, 0.01, 0.0005, 0.005, 0.0, 0.0]
    path = tmp_path / "aniso.cif"
    path.write_text(cif_text_with_sites(CALCITE, sites), encoding="utf-8")
    got = cif_aniso_betas(str(path))
    assert np.allclose(got[sites[0]["label"]], sites[0]["beta"])


def test_isotropic_equivalent_beta_matches_biso():
    from pdf_fitting.models.model_factory import build_pdf_model
    from pdf_fitting.project import Project
    from pdf_fitting.project.multiphase import load_gr_data

    project = Project.from_inp("examples/1-FeCr-ball-milled/input_FeCr_paper_2.inp")
    ds, ph = project.datasets[0], project.datasets[0].phases[0]
    rmax = 10.0
    r, _G = load_gr_data(project.resolve_path(ds.data_file), ds.r_min, rmax)

    def build(extra):
        cfg = project.phase_legacy_config(ds, ph)
        cfg["r_max"] = rmax
        cfg.setdefault("refinement", {})["pdf_backend"] = "cpu"
        cfg["initial"].update(extra)
        with contextlib.redirect_stdout(io.StringIO()):
            return build_pdf_model(config=cfg, r_exp=r, pair_cutoff=14.0), cfg

    m0, cfg0 = build({})
    params = dict(cfg0["initial"])
    a = float(params["a"])
    B = float(params.get("biso_Fe", params.get("biso_fe")))
    beta = {f"beta{c}_fe1": v for c, v in zip(ADP_COMPONENTS, (B / 4 / a**2,) * 3 + (0.0,) * 3)}
    m1, _ = build(beta)
    p1 = dict(params)
    p1.update(beta)
    with contextlib.redirect_stdout(io.StringIO()):
        g0 = np.asarray(m0.evaluate(dict(params), r_override=r)["G_r"], float)
        g1 = np.asarray(m1.evaluate(p1, r_override=r)["G_r"], float)
    assert np.max(np.abs(g1 - g0)) / np.max(np.abs(g0)) < 1e-5
