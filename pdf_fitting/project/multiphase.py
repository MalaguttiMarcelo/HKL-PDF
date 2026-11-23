"""
Multi-dataset, multi-phase PDF model built from a Project.

Every active phase of every dataset is an ordinary single-phase model
(PDFCalculator, finite-shape or stacking model) created by
``build_pdf_model`` from ``Project.phase_legacy_config``. The calculated
G(r) of a dataset is the sum of its phase G(r)s, each carrying its own
``scale`` parameter.

For the fit engine all datasets are concatenated on a *virtual* r axis:

    t = r + k * VIRTUAL_OFFSET       (k = dataset index)

so ``evaluate(params, r_override=t_subset)`` can be split back per dataset.

Weight fractions are derived from the phase scales. Each phase G(r) is
normalised with its own <f>^2 (Faber-Ziman), so in a mixture

    s_i  ∝  c_i <f>_i^2        (c_i: atom fraction of phase i)
    w_i  ∝  c_i <M>_i          (<M>_i: mean atomic mass of phase i)

with f approximated by Z, as in the single-phase weights.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .model import Dataset, Phase, Project

VIRTUAL_OFFSET = 1.0e4


def load_gr_data(path: str, r_min: float, r_max: float):
    """First two columns of a G(r) file, restricted to [r_min, r_max]."""
    data = np.loadtxt(path, unpack=True)
    r = np.asarray(data[0], dtype=float)
    G = np.asarray(data[1], dtype=float)

    mask = (r >= float(r_min)) & (r <= float(r_max))
    return r[mask], G[mask]


class PhaseEntry:
    """One active phase of one dataset, with its single-phase model."""

    def __init__(self, dataset: Dataset, phase: Phase, config: Dict[str, Any], model):
        self.dataset = dataset
        self.phase = phase
        self.config = config
        self.model = model
        self.prefix = f"{dataset.id}.{phase.id}."
        self.dataset_prefix = f"{dataset.id}."
        self.instrument_names = set(dataset.parameters)

    # Local <-> namespaced parameter names.

    def local_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for name in self.instrument_names:
            full = self.dataset_prefix + name
            if full in params:
                out[name] = params[full]

        n = len(self.prefix)

        for full, value in params.items():
            if full.startswith(self.prefix):
                out[full[n:]] = value

        return out

    def full_name(self, local: str) -> str:
        if local in self.instrument_names:
            return self.dataset_prefix + local
        return self.prefix + local

    @property
    def structure_handler(self):
        for candidate in (
            self.model,
            getattr(self.model, "pdf_calculator", None),
            getattr(self.model, "pdf_model", None),
        ):
            handler = getattr(candidate, "structure_handler", None)
            if handler is not None:
                return handler
        return None


class DatasetEntry:
    def __init__(self, index: int, dataset: Dataset, r_exp, G_exp):
        self.index = index
        self.dataset = dataset
        self.r_exp = np.asarray(r_exp, dtype=float)
        self.G_exp = np.asarray(G_exp, dtype=float)
        self.phases: List[PhaseEntry] = []

    @property
    def offset(self) -> float:
        return self.index * VIRTUAL_OFFSET


class ProjectModel:
    """
    Evaluates a whole Project; drop-in model for FitEngine.

    FitEngine-facing interface:
        evaluate(params, r_override=None)  -> {"r", "G_r", "datasets": [...]}
        phase_handlers()                   -> per-phase structure handlers
        apply_lattice_constraints(params)  -> params with derived lattice values
        r_exp, G_exp                       -> concatenated virtual data
    """

    def __init__(
        self,
        project: Project,
        *,
        status_callback: Optional[Callable[[str], None]] = None,
        stop_event=None,
    ):
        from pdf_fitting.models.model_factory import build_pdf_model

        self.project = project
        self.datasets: List[DatasetEntry] = []

        for k, ds in enumerate(project.datasets):
            r_exp, G_exp = load_gr_data(
                project.resolve_path(ds.data_file),
                ds.r_min,
                ds.r_max,
            )

            entry = DatasetEntry(k, ds, r_exp, G_exp)

            for ph in ds.active_phases():
                if status_callback is not None:
                    status_callback(f"Building model: {ds.name} › {ph.name}")

                cfg = project.phase_legacy_config(ds, ph)

                model = build_pdf_model(
                    config=cfg,
                    r_exp=r_exp,
                    pair_cutoff=float(ds.r_max) * float(ds.r_extension),
                    stop_event=stop_event,
                    status_callback=status_callback,
                )

                entry.phases.append(PhaseEntry(ds, ph, cfg, model))

            self.datasets.append(entry)

        self.r_exp = np.concatenate(
            [e.r_exp + e.offset for e in self.datasets]
        ) if self.datasets else np.zeros(0)

        self.G_exp = np.concatenate(
            [e.G_exp for e in self.datasets]
        ) if self.datasets else np.zeros(0)

        self.r_max = max((float(e.dataset.r_max) for e in self.datasets), default=0.0)

    # ------------------------------------------------------------------
    # Virtual r axis
    # ------------------------------------------------------------------

    def split_virtual(self, t: np.ndarray):
        """[(DatasetEntry, mask over t, real r)] for a virtual r array."""
        t = np.asarray(t, dtype=float)
        k = np.floor((t + 0.5 * VIRTUAL_OFFSET) / VIRTUAL_OFFSET).astype(int)

        out = []

        for e in self.datasets:
            mask = k == e.index

            if np.any(mask):
                out.append((e, mask, t[mask] - e.offset))

        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, params: Dict[str, Any], r_override=None, **kwargs) -> Dict[str, Any]:
        t = self.r_exp if r_override is None else np.asarray(r_override, dtype=float)
        G_total = np.zeros_like(t, dtype=float)
        per_dataset = []

        for e, mask, r_ds in self.split_virtual(t):
            G_ds = np.zeros_like(r_ds)
            phases = {}

            for pe in e.phases:
                local = pe.local_params(params)
                # Plain evaluation during fitting; callers that want partial
                # pair contributions pass return_contributions=True.
                res = pe.model.evaluate(
                    local,
                    r_override=r_ds,
                    **kwargs,
                )
                G_ph = np.asarray(res["G_r"], dtype=float)
                G_ds = G_ds + G_ph
                phases[pe.phase.id] = res

            G_total[mask] = G_ds
            per_dataset.append(
                {
                    "id": e.dataset.id,
                    "r": r_ds,
                    "G_r": G_ds,
                    "phases": phases,
                    "fractions": self.phase_fractions(params, e),
                }
            )

        return {
            "r": t,
            "G_r": G_total,
            "datasets": per_dataset,
        }

    # ------------------------------------------------------------------
    # Phase fractions
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_composition(pe: PhaseEntry):
        """(<Z>, <M>) per atom of a phase from its model composition."""
        from pymatgen.core import Element

        model = pe.model
        calc = getattr(model, "pdf_calculator", None) or model

        conc = np.asarray(getattr(calc, "_concentrations", []), dtype=float)
        species = list(getattr(calc, "_unique_species", []))

        if conc.size == 0 or not species:
            return 1.0, 1.0

        Z = np.array([Element(s).Z for s in species], dtype=float)
        M = np.array([float(Element(s).atomic_mass) for s in species], dtype=float)

        return float(np.sum(conc * Z)), float(np.sum(conc * M))

    def phase_fractions(self, params: Dict[str, Any], entry: DatasetEntry) -> Dict[str, Dict[str, float]]:
        """{phase id: {"scale", "atom_fraction", "weight_fraction"}} (fractions in %)."""
        raw = {}

        for pe in entry.phases:
            scale = float(pe.local_params(params).get("scale", 1.0))
            z_avg, m_avg = self._phase_composition(pe)
            atoms = max(scale, 0.0) / max(z_avg * z_avg, 1e-12)
            raw[pe.phase.id] = (scale, atoms, atoms * m_avg)

        n_sum = sum(v[1] for v in raw.values()) or 1.0
        w_sum = sum(v[2] for v in raw.values()) or 1.0

        return {
            pid: {
                "scale": v[0],
                "atom_fraction": 100.0 * v[1] / n_sum,
                "weight_fraction": 100.0 * v[2] / w_sum,
            }
            for pid, v in raw.items()
        }

    # ------------------------------------------------------------------
    # FitEngine hooks
    # ------------------------------------------------------------------

    def phase_entries(self) -> List[PhaseEntry]:
        return [pe for e in self.datasets for pe in e.phases]

    def phase_handlers(self) -> List[Dict[str, Any]]:
        """
        [{"prefix", "handler", "ref_cfg"}] for element-wise occupancy /
        coordinate refinement and coordinate limits.
        """
        return [
            {
                "prefix": pe.prefix,
                "handler": pe.structure_handler,
                "ref_cfg": dict(pe.config.get("refinement", {}) or {}),
            }
            for pe in self.phase_entries()
        ]

    def apply_lattice_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from pdf_fitting.fit_engine import apply_model_lattice_constraints_to_params

        out = dict(params)

        for pe in self.phase_entries():
            local = {
                k[len(pe.prefix):]: v
                for k, v in out.items()
                if k.startswith(pe.prefix)
            }

            if not local:
                continue

            fixed = apply_model_lattice_constraints_to_params(local, pe.model)

            for k, v in fixed.items():
                out[pe.prefix + k] = v

        return out
