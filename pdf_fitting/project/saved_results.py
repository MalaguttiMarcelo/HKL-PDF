"""
Compact storage of the last calculated fit in the project file.

Curves are stored as float32, base64-encoded ({"f32": "..."}); only what the
fit / phase plots need is kept (no raw pair PDFs, Warren or local-trend data,
which are recomputed by Calculate).
"""

from __future__ import annotations

import base64
from typing import Any, Dict

import numpy as np


def _enc(a) -> Dict[str, str]:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"f32": base64.b64encode(a.tobytes()).decode("ascii"), "n": int(a.size)}


def _dec(d) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["f32"]), dtype=np.float32).astype(float)


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def encode_dataset_result(d: Dict[str, Any]) -> Dict[str, Any]:
    phases = {}
    for pid, pr in (d.get("phases") or {}).items():
        phases[pid] = {
            "G_r": _enc(pr.get("G_r", [])),
            "G_contrib": {k: _enc(v) for k, v in (pr.get("G_contrib") or {}).items()},
            "gamma_avg": _enc(pr["gamma_avg"]) if pr.get("gamma_avg") is not None else None,
            "weights": {k: _finite(v) for k, v in (pr.get("weights") or {}).items()},
        }
    return {
        "id": d["id"],
        "r": _enc(d["r"]),
        "G_exp": _enc(d["G_exp"]),
        "G_fit": _enc(d["G_fit"]),
        "rwp": _finite(d.get("rwp")),
        "fractions": {k: _finite(v) for k, v in (d.get("fractions") or {}).items()},
        "phases": phases,
    }


def decode_dataset_result(e: Dict[str, Any]) -> Dict[str, Any]:
    phases = {}
    for pid, pr in (e.get("phases") or {}).items():
        phases[pid] = {
            "G_r": _dec(pr["G_r"]),
            "G_ab": {},
            "G_contrib": {k: _dec(v) for k, v in (pr.get("G_contrib") or {}).items()},
            "gamma_avg": _dec(pr["gamma_avg"]) if pr.get("gamma_avg") else None,
            "weights": {k: (v if v is not None else float("nan")) for k, v in (pr.get("weights") or {}).items()},
            "local_params": {},
        }
    rwp = e.get("rwp")
    return {
        "id": e["id"],
        "r": _dec(e["r"]),
        "G_exp": _dec(e["G_exp"]),
        "G_fit": _dec(e["G_fit"]),
        "rwp": float(rwp) if rwp is not None else float("nan"),
        "fractions": {k: (v if v is not None else float("nan")) for k, v in (e.get("fractions") or {}).items()},
        "phases": phases,
        "saved": True,
    }
