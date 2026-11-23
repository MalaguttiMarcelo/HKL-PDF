"""
Project model / multi-phase tests.

They use the example structures in examples/ and synthetic G(r) data, so no
experimental file content matters. Each test keeps r_max small for speed.
"""

import copy
import os
import io
import json
import contextlib
from pathlib import Path

import numpy as np
import pytest

from pdf_fitting.io_handler import read_input_file
from pdf_fitting.project import Project
from pdf_fitting.project.multiphase import ProjectModel, VIRTUAL_OFFSET

ROOT = Path(__file__).resolve().parents[1]
CALCITE_INP = ROOT / "examples/2-Calcite-ball-milled/Calcite_150min.inp"
CALCITE_CIF = ROOT / "examples/2-Calcite-ball-milled/Calcite.cif"
FE_INP = ROOT / "examples/1-FeCr-ball-milled/input_FeCr_paper_2.inp"
FE_CIF = ROOT / "examples/1-FeCr-ball-milled/Fe_bcc_300K.cif"

pytestmark = pytest.mark.skipif(
    not (CALCITE_INP.exists() and FE_INP.exists()),
    reason="example inputs not available",
)

R_MAX = 12.0


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def write_xy(path: Path, r, G):
    np.savetxt(path, np.column_stack([r, G]))
    return str(path)


def calcite_project(tmp_path: Path) -> Project:
    project = Project.from_inp(str(CALCITE_INP))
    ds = project.datasets[0]
    ph = ds.phases[0]
    ph.cif = str(CALCITE_CIF)
    ds.r_max = R_MAX
    # Keep the tests on CPU and with a plain strain model.
    project.refinement.update(shell_backend="cpu", pdf_backend="cpu")
    ph.settings["microstrain_model"] = "isotropic"
    for p in ph.parameters.values():
        p.refine = False
    for p in ds.parameters.values():
        p.refine = False
    r = np.arange(ds.r_min, R_MAX + 1e-9, 0.02)
    ds.data_file = write_xy(tmp_path / "zeros.xy", r, np.zeros_like(r))
    return project


def fe_phase(project: Project, dataset):
    fe = Project.from_inp(str(FE_INP)).datasets[0].phases[0]
    fe.cif = str(FE_CIF)
    fe.id = project.new_phase_id(dataset, "fe")
    fe.name = "Fe"
    fe.settings["microstrain_model"] = "isotropic"
    for p in fe.parameters.values():
        p.refine = False
    return fe


# ---------------------------------------------------------------------------


def test_inp_import_round_trip(tmp_path):
    cfg = read_input_file(str(CALCITE_INP))
    project = Project.from_inp(str(CALCITE_INP))

    ds = project.datasets[0]
    ph = ds.phases[0]

    # Instrument parameters live on the dataset, the rest on the phase.
    assert "qdamp" in ds.parameters
    assert "qdamp" not in ph.parameters
    assert "a" in ph.parameters and "scale" in ph.parameters

    # Phase settings vs project settings.
    assert ph.settings.get("microstrain_model") == cfg["refinement"]["microstrain_model"]
    assert "microstrain_model" not in project.refinement

    # The per-phase legacy config reproduces the original parameters.
    legacy = project.phase_legacy_config(ds, ph)
    assert legacy["initial"] == cfg["initial"]
    assert {k: v for k, v in legacy["refinable"].items()} == {
        k: v for k, v in cfg["refinable"].items() if k in cfg["initial"]
    }
    for k, v in cfg["bounds"].items():
        if k in cfg["initial"]:
            assert tuple(legacy["bounds"][k]) == tuple(v)
    assert legacy["refinement"] == cfg["refinement"]

    # JSON round trip is lossless.
    path = tmp_path / "calcite.hklpdf.json"
    project.save(str(path))
    again = Project.load(str(path))
    assert again.to_json() == json.loads(path.read_text(encoding="utf-8"))
    assert again.flat_tables() == project.flat_tables()


def test_single_phase_matches_legacy_model(tmp_path):
    from pdf_fitting.models.gr_model import PDFCalculator

    project = calcite_project(tmp_path)
    ds = project.datasets[0]
    ph = ds.phases[0]

    pm = quiet(ProjectModel, project)
    initial, _ref, _b = project.flat_tables()
    G_project = quiet(pm.evaluate, initial)["G_r"]

    legacy = project.phase_legacy_config(ds, ph)
    r = pm.datasets[0].r_exp
    calc = quiet(PDFCalculator, legacy, legacy["structure_file"], r,
                 pair_cutoff=ds.r_max * ds.r_extension)
    G_legacy = np.asarray(quiet(calc.evaluate, dict(legacy["initial"]))["G_r"], float)

    assert np.max(np.abs(G_project - G_legacy)) == 0.0


def test_two_identical_phases_split_scale(tmp_path):
    project = calcite_project(tmp_path)
    ds = project.datasets[0]
    base = ds.phases[0]

    twin = copy.deepcopy(base)
    twin.id = project.new_phase_id(ds, base.name)
    twin.name = base.name + " (twin)"
    ds.phases.append(twin)

    s = float(base.parameters["scale"].value)
    base.parameters["scale"].value = 0.7 * s
    twin.parameters["scale"].value = 0.3 * s

    pm = quiet(ProjectModel, project)
    initial, _r, _b = project.flat_tables()
    out = quiet(pm.evaluate, initial)

    # Reference: one phase with the full scale.
    single = calcite_project(tmp_path)
    single.datasets[0].phases[0].parameters["scale"].value = s
    pm1 = quiet(ProjectModel, single)
    G1 = quiet(pm1.evaluate, single.flat_tables()[0])["G_r"]

    assert np.allclose(out["G_r"], G1, atol=1e-5 * np.max(np.abs(G1)))

    fractions = out["datasets"][0]["fractions"]
    assert fractions[base.id]["weight_fraction"] == pytest.approx(70.0)
    assert fractions[twin.id]["weight_fraction"] == pytest.approx(30.0)


def test_two_phase_fit_recovers_scales(tmp_path):
    from pdf_fitting.fit_engine import FitEngine

    project = calcite_project(tmp_path)
    ds = project.datasets[0]
    cal = ds.phases[0]
    fe = fe_phase(project, ds)
    ds.phases.append(fe)

    truth = {
        f"ds1.{cal.id}.scale": float(cal.parameters["scale"].value),
        f"ds1.{fe.id}.scale": 0.4 * float(cal.parameters["scale"].value),
        f"ds1.{cal.id}.a": float(cal.parameters["a"].value) * 1.002,
    }

    # Synthetic data from the true parameters.
    pm_truth = quiet(ProjectModel, project)
    initial, _r, _b = project.flat_tables()
    G_true = quiet(pm_truth.evaluate, {**initial, **truth})["G_r"]
    ds.data_file = write_xy(tmp_path / "two_phase.xy", pm_truth.datasets[0].r_exp, G_true)

    # Start away from the truth and refine scales + calcite a.
    cal.parameters["scale"].refine = True
    fe.parameters["scale"].refine = True
    cal.parameters["a"].refine = True
    fe.parameters["scale"].value = 0.1 * float(cal.parameters["scale"].value)

    pm = quiet(ProjectModel, project)
    initial, refinable, bounds = project.flat_tables()
    config = {
        "initial": dict(initial),
        "refinable": dict(refinable),
        "bounds": dict(bounds),
        "refinement": {**project.refinement, "staged_refinement": False, "max_nfev_final": 80},
        "constraints": dict(project.constraints),
        "r_max": ds.r_max,
    }
    engine = quiet(FitEngine, pm.r_exp, pm.G_exp, pm, config)
    result = quiet(engine.run_refinement)

    for name, value in truth.items():
        assert result[name] == pytest.approx(value, rel=2e-3), name

    project.apply_values(result)
    assert float(fe.parameters["scale"].value) == pytest.approx(truth[f"ds1.{fe.id}.scale"], rel=2e-3)


def test_two_datasets_virtual_axis(tmp_path):
    project = calcite_project(tmp_path)
    ds1 = project.datasets[0]
    ds2 = copy.deepcopy(ds1)
    ds2.id = project.new_dataset_id()
    ds2.name = "second"
    ds2.r_max = 8.0
    project.datasets.append(ds2)

    pm = quiet(ProjectModel, project)
    initial = project.flat_tables()[0]
    assert f"{ds2.id}.qdamp" in initial

    out = quiet(pm.evaluate, initial)
    n1 = pm.datasets[0].r_exp.size
    n2 = pm.datasets[1].r_exp.size
    assert out["G_r"].size == n1 + n2
    assert np.all(pm.r_exp[n1:] > VIRTUAL_OFFSET - 1)

    # Same phase, shorter range: dataset 2 equals dataset 1 on its r grid,
    # away from its upper end (its shorter pair cutoff and the Qmax
    # termination kernel change the last few Angstrom).
    G1 = out["datasets"][0]["G_r"]
    G2 = out["datasets"][1]["G_r"]
    inner = pm.datasets[1].r_exp <= 6.0
    assert np.allclose(G1[:n2][inner], G2[inner], atol=1e-4 * np.max(np.abs(G1)))

    # A subset of the virtual axis (as FitEngine's coarse grid) is split back
    # into each dataset's real r values. (The single-phase model's G on a
    # coarser grid differs slightly from the fine-grid G because Qmax
    # termination runs on the requested grid, so compare against the phase
    # model evaluated on the same real sub-grid.)
    sub = pm.r_exp[::3]
    out_sub = quiet(pm.evaluate, initial, r_override=sub)
    start = 0
    for e in pm.datasets:
        r_part = e.r_exp[::3] if e.index == 0 else (sub[sub > VIRTUAL_OFFSET / 2] - e.offset)
        pe = e.phases[0]
        G_ref = np.asarray(quiet(pe.model.evaluate, pe.local_params(initial), r_override=r_part)["G_r"], float)
        assert np.array_equal(out_sub["G_r"][start:start + r_part.size], G_ref)
        start += r_part.size
    assert start == sub.size


def test_edited_structure_delete_and_add_site(tmp_path):
    """Structure editor stage 3: sites stored in the project, derived CIF."""
    from pdf_fitting.project.edited_cif import base_sites
    from pdf_fitting.project.structure_tools import phase_structure

    project = calcite_project(tmp_path)
    ds = project.datasets[0]
    ph = ds.phases[0]
    original = project.structure_path(ph)
    assert original == str(CALCITE_CIF)

    sites = base_sites(str(CALCITE_CIF))
    assert [s["label"] for s in sites] == ["Ca", "C", "O"]

    # Delete the O orbit.
    ph.sections["structure_edit"] = {"base_cif": str(CALCITE_CIF), "sites": [s for s in sites if s["label"] != "O"]}
    derived = project.structure_path(ph)
    assert derived != original and os.path.exists(derived)
    ps = phase_structure(derived, {})
    assert len(ps.structure) == 12 and ps.group_labels == ["Ca", "C"]

    # Add it back (new label) at the CIF position: same model as the original CIF.
    o = next(s for s in sites if s["label"] == "O")
    ph.sections["structure_edit"]["sites"].append({**o, "label": "O9"})
    ps = phase_structure(project.structure_path(ph), {})
    assert len(ps.structure) == 30
    assert ps.multiplicity(ps.group_labels.index("O9")) == 18

    for name in [n for n in ph.parameters if n.endswith("_o")]:
        ph.parameters.pop(name)
    G_edit = quiet(ProjectModel(project).evaluate, project.flat_tables()[0])["G_r"]
    ph.sections.pop("structure_edit")
    G_orig = quiet(ProjectModel(project).evaluate, project.flat_tables()[0])["G_r"]
    assert np.allclose(G_edit, G_orig, atol=1e-6 * np.max(np.abs(G_orig)))

    # The original CIF is untouched and the edit survives a JSON round trip.
    ph.sections["structure_edit"] = {"base_cif": str(CALCITE_CIF), "sites": [s for s in sites if s["label"] != "O"]}
    path = tmp_path / "edited.hklpdf.json"
    project.save(str(path))
    again = Project.load(str(path))
    assert again.structure_path(again.datasets[0].phases[0]) == project.structure_path(ph)
