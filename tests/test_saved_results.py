"""The last fit saved in the project file round-trips (float32 precision)."""

import json

import numpy as np

from pdf_fitting.project.model import Project
from pdf_fitting.project.saved_results import decode_dataset_result, encode_dataset_result


def test_saved_fit_roundtrip(tmp_path):
    r = np.linspace(1, 30, 2901)
    d = {
        "id": "ds1", "r": r, "G_exp": np.sin(r), "G_fit": np.sin(r) * 0.9, "rwp": 12.5, "fractions": {"a": 0.4, "b": float("nan")},
        "phases": {"a": {"G_r": np.cos(r), "G_contrib": {"fe-fe": np.cos(r)}, "gamma_avg": None, "weights": {"fe-fe": 1.0}}},
    }
    p = Project(name="t")
    p.last_results = {"ds1": encode_dataset_result(d)}
    path = p.save(str(tmp_path / "t.hklpdf.json"))
    p2 = Project.load(path)
    out = decode_dataset_result(p2.last_results["ds1"])
    assert np.allclose(out["G_fit"], d["G_fit"], atol=1e-6)
    assert np.allclose(out["phases"]["a"]["G_contrib"]["fe-fe"], np.cos(r), atol=1e-6)
    assert out["rwp"] == 12.5 and np.isnan(out["fractions"]["b"])
    assert len(json.dumps(p2.last_results)) < 200_000
