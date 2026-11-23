"""
Project model for multi-dataset, multi-phase PDF refinement.

Qt-free. The GUI, the JSON project file, the legacy .inp importer and the
fit engine all work on this structure:

    Project
      refinement settings (solver, backend, outputs)
      constraints         {namespaced target: expression}
      datasets[]
        data file, r range, instrument parameters
        phases[]
          CIF, scale, structural / microstructural / dynamics parameters,
          phase settings (microstrain model, SAXS, element refinement, ...),
          crystallite-shape and stacking sections

Parameter names seen by the fit engine are namespaced:

    dataset level   "<dataset id>.<name>"              e.g. "ds1.qdamp"
    phase level     "<dataset id>.<phase id>.<name>"   e.g. "ds1.calcite.a"

Inside a phase every name is the same as in the single-phase code
("a", "occ_o1", "lambda_o-o_0", ...), so each phase still runs through the
existing PDFCalculator via ``phase_legacy_config``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_FORMAT = "hkl-pdf-project"
PROJECT_VERSION = 1

# Parameters that belong to the measurement, shared by all phases.
INSTRUMENT_PARAMS = (
    "qdamp",
    "delta_broad",
    "qbroad",
    "qmax",
    "qmax_zeros",
    "qmax_pad",
)

# [refinement] settings that belong to one phase. Every other setting is a
# project-wide solver / backend / output option.
PHASE_SETTING_KEYS = (
    "microstrain_model",
    "saxs_enabled",
    "saxs_model",
    "saxs_apply_qdamp",
    "saxs_apply_qmax",
    "refine_occ_species",
    "refine_xyz_species",
    "occ_min",
    "occ_max",
    "xyz_max_shift",
    "force_p1_shells",
)

# Free-form sections of the legacy config that belong to a phase.
PHASE_SECTIONS = (
    "contrast_factors",
    "crystallite_shape",
    "layer_stacking",
    "stacking_search",
    "stacking_vector_search",
    "stacking_refinement",
    "stacking_faults",
)

# Legacy config keys handled explicitly (everything else is kept in
# Phase.extra so nothing is lost on import).
_LEGACY_HANDLED = {
    "structure_file",
    "gr_data_file",
    "r_min",
    "r_max",
    "r_extension",
    "refinement",
    "refinable",
    "initial",
    "bounds",
    "constraints",
    "biso_by_species",
    *PHASE_SECTIONS,
}


def safe_id(text: str, fallback: str = "item") -> str:
    """Identifier usable inside namespaced parameter names (no dots)."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "")).strip("_").lower()

    if not s:
        s = fallback

    if s[0].isdigit():
        s = f"{fallback}_{s}"

    return s


def _clean_number(value):
    """JSON-friendly number (no NaN/inf); strings pass through."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        v = float(value)

        if not math.isfinite(v):
            return None

        return int(v) if float(v).is_integer() and isinstance(value, int) else v

    return value


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------


@dataclass
class Parameter:
    value: Any = 0.0
    refine: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    esd: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": _clean_number(self.value)}

        if self.refine:
            out["refine"] = True
        if self.min is not None:
            out["min"] = _clean_number(self.min)
        if self.max is not None:
            out["max"] = _clean_number(self.max)
        if self.esd is not None:
            out["esd"] = _clean_number(self.esd)

        return out

    @classmethod
    def from_json(cls, data) -> "Parameter":
        if not isinstance(data, dict):
            return cls(value=data)

        return cls(
            value=data.get("value", 0.0),
            refine=bool(data.get("refine", False)),
            min=data.get("min", None),
            max=data.get("max", None),
            esd=data.get("esd", None),
        )


ParameterTable = Dict[str, Parameter]


def _params_from_legacy(
    names,
    initial: Dict[str, Any],
    refinable: Dict[str, Any],
    bounds: Dict[str, Any],
) -> ParameterTable:
    table: ParameterTable = {}

    for name in names:
        lo, hi = None, None
        b = bounds.get(name)

        if b is not None:
            try:
                lo, hi = b
            except (TypeError, ValueError):
                lo, hi = None, None

        table[name] = Parameter(
            value=initial.get(name, 0.0),
            refine=bool(refinable.get(name, False)),
            min=lo,
            max=hi,
        )

    return table


def _params_to_json(table: ParameterTable) -> Dict[str, Any]:
    return {name: p.to_json() for name, p in table.items()}


def _params_from_json(data) -> ParameterTable:
    return {
        str(name): Parameter.from_json(value)
        for name, value in (data or {}).items()
    }


# -----------------------------------------------------------------------------
# Phase / Dataset / Project
# -----------------------------------------------------------------------------


@dataclass
class Phase:
    id: str
    name: str
    cif: str
    active: bool = True
    parameters: ParameterTable = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cif": self.cif,
            "active": bool(self.active),
            "settings": copy.deepcopy(self.settings),
            "parameters": _params_to_json(self.parameters),
            "sections": copy.deepcopy(self.sections),
            "extra": copy.deepcopy(self.extra),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Phase":
        return cls(
            id=safe_id(data.get("id", "phase"), "phase"),
            name=str(data.get("name", data.get("id", "Phase"))),
            cif=str(data.get("cif", "")),
            active=bool(data.get("active", True)),
            parameters=_params_from_json(data.get("parameters")),
            settings=dict(data.get("settings", {}) or {}),
            sections=dict(data.get("sections", {}) or {}),
            extra=dict(data.get("extra", {}) or {}),
        )


@dataclass
class Dataset:
    id: str
    name: str
    data_file: str
    r_min: float = 1.0
    r_max: float = 10.0
    r_extension: float = 1.2
    parameters: ParameterTable = field(default_factory=dict)
    phases: List[Phase] = field(default_factory=list)

    def phase(self, phase_id: str) -> Phase:
        for ph in self.phases:
            if ph.id == phase_id:
                return ph
        raise KeyError(f"Dataset {self.id} has no phase {phase_id!r}")

    def active_phases(self) -> List[Phase]:
        return [ph for ph in self.phases if ph.active]

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "data_file": self.data_file,
            "r_min": _clean_number(self.r_min),
            "r_max": _clean_number(self.r_max),
            "r_extension": _clean_number(self.r_extension),
            "parameters": _params_to_json(self.parameters),
            "phases": [ph.to_json() for ph in self.phases],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Dataset":
        return cls(
            id=safe_id(data.get("id", "ds1"), "ds"),
            name=str(data.get("name", data.get("id", "Dataset"))),
            data_file=str(data.get("data_file", "")),
            r_min=float(data.get("r_min", 1.0)),
            r_max=float(data.get("r_max", 10.0)),
            r_extension=float(data.get("r_extension", 1.2)),
            parameters=_params_from_json(data.get("parameters")),
            phases=[Phase.from_json(p) for p in data.get("phases", []) or []],
        )


@dataclass
class Project:
    name: str = "Untitled project"
    refinement: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, str] = field(default_factory=dict)
    datasets: List[Dataset] = field(default_factory=list)
    path: Optional[str] = None  # where it was loaded from / saved to
    # Last calculated fit per dataset (saved_results.encode_dataset_result),
    # shown when the project is opened.
    last_results: Dict[str, Any] = field(default_factory=dict)
    # Last esd / correlation analysis (uncertainty.UncertaintyResult.to_json()).
    last_uncertainty: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def dataset(self, dataset_id: str) -> Dataset:
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        raise KeyError(f"Project has no dataset {dataset_id!r}")

    def iter_phases(self) -> Iterator[Tuple[Dataset, Phase]]:
        for ds in self.datasets:
            for ph in ds.phases:
                yield ds, ph

    def new_dataset_id(self) -> str:
        used = {ds.id for ds in self.datasets}
        k = 1
        while f"ds{k}" in used:
            k += 1
        return f"ds{k}"

    @staticmethod
    def new_phase_id(dataset: Dataset, name: str) -> str:
        base = safe_id(name, "phase")
        used = {ph.id for ph in dataset.phases}
        candidate = base
        k = 2
        while candidate in used:
            candidate = f"{base}_{k}"
            k += 1
        return candidate

    # ------------------------------------------------------------------
    # Namespaced flat parameter view (what the fit engine works on)
    # ------------------------------------------------------------------

    def iter_parameters(self, *, active_only: bool = True):
        """
        Yield (full name, Parameter, dataset, phase or None).

        Inactive phases are skipped unless active_only is False.
        """
        for ds in self.datasets:
            for name, p in ds.parameters.items():
                yield f"{ds.id}.{name}", p, ds, None

            for ph in ds.phases:
                if active_only and not ph.active:
                    continue

                for name, p in ph.parameters.items():
                    yield f"{ds.id}.{ph.id}.{name}", p, ds, ph

    def flat_tables(self):
        """
        (initial, refinable, bounds) with namespaced names, in a stable order.
        """
        initial: Dict[str, Any] = {}
        refinable: Dict[str, bool] = {}
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for full, p, _ds, _ph in self.iter_parameters():
            initial[full] = p.value
            refinable[full] = bool(p.refine)

            if p.min is not None or p.max is not None:
                bounds[full] = (p.min, p.max)

        return initial, refinable, bounds

    def find_parameter(self, full_name: str) -> Optional[Parameter]:
        parts = str(full_name).split(".")

        try:
            ds = self.dataset(parts[0])
        except KeyError:
            return None

        if len(parts) == 2:
            return ds.parameters.get(parts[1])

        if len(parts) == 3:
            try:
                return ds.phase(parts[1]).parameters.get(parts[2])
            except KeyError:
                return None

        return None

    def apply_values(
        self,
        values: Dict[str, Any],
        *,
        refinable: Optional[Dict[str, bool]] = None,
        esd: Optional[Dict[str, float]] = None,
        add_missing: bool = True,
    ) -> None:
        """
        Write namespaced values (e.g. refined results) back into the model.

        Parameters that do not exist yet are created when add_missing is
        True (e.g. occ_/x_ parameters added by element-wise refinement).
        """
        for full, value in (values or {}).items():
            p = self.find_parameter(full)

            if p is None:
                if not add_missing:
                    continue

                parts = str(full).split(".")

                try:
                    ds = self.dataset(parts[0])

                    if len(parts) == 2:
                        p = ds.parameters.setdefault(parts[1], Parameter())
                    elif len(parts) == 3:
                        p = ds.phase(parts[1]).parameters.setdefault(parts[2], Parameter())
                    else:
                        continue
                except KeyError:
                    continue

            p.value = _clean_number(value) if not isinstance(value, str) else value

            if refinable is not None and full in refinable:
                p.refine = bool(refinable[full])

            if esd is not None and full in esd:
                p.esd = esd[full]

    # ------------------------------------------------------------------
    # Bridge to the single-phase engine
    # ------------------------------------------------------------------

    def phase_legacy_config(self, dataset: Dataset, phase: Phase) -> Dict[str, Any]:
        """
        Config dict in the shape returned by io_handler.read_input_file, for
        one phase of one dataset. Parameter names are local (un-namespaced);
        dataset-level instrument parameters are merged in.
        """
        initial: Dict[str, Any] = {}
        refinable: Dict[str, bool] = {}
        bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for table in (dataset.parameters, phase.parameters):
            for name, p in table.items():
                initial[name] = p.value
                refinable[name] = bool(p.refine)

                if p.min is not None or p.max is not None:
                    bounds[name] = (p.min, p.max)

        for angle in ("alpha", "beta", "gamma"):
            initial.setdefault(angle, 90.0)

        refinement = dict(self.refinement)
        refinement.update(phase.settings)

        cfg: Dict[str, Any] = {
            "structure_file": self.structure_path(phase),
            "gr_data_file": self.resolve_path(dataset.data_file),
            "r_min": float(dataset.r_min),
            "r_max": float(dataset.r_max),
            "r_extension": float(dataset.r_extension),
            "refinement": refinement,
            "refinable": refinable,
            "initial": initial,
            "bounds": bounds,
            "constraints": {},
            "biso_by_species": biso_by_species_from_initial(initial),
        }

        for section in PHASE_SECTIONS:
            cfg[section] = copy.deepcopy(phase.sections.get(section, {}) or {})

        for key, value in phase.extra.items():
            cfg.setdefault(key, copy.deepcopy(value))

        return cfg

    def structure_path(self, phase: Phase) -> str:
        """
        Structure file the model and viewers use for a phase: the original
        CIF, or a derived CIF when sites were added / deleted in the
        structure editor (phase.sections["structure_edit"]).
        """
        edit = (phase.sections or {}).get("structure_edit") or {}
        sites = edit.get("sites")
        base = self.resolve_path(edit.get("base_cif") or phase.cif)
        if sites is None:
            return base
        from .edited_cif import edited_structure_path

        return edited_structure_path(base, sites)

    def resolve_path(self, path: str) -> str:
        """Paths in the project file may be relative to the project file."""
        if not path:
            return path

        if os.path.isabs(path) or not self.path:
            return path

        return os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(self.path)), path)
        )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "name": self.name,
            "refinement": copy.deepcopy(self.refinement),
            "constraints": dict(self.constraints),
            "datasets": [ds.to_json() for ds in self.datasets],
            **({"last_results": copy.deepcopy(self.last_results)} if self.last_results else {}),
            **({"last_uncertainty": copy.deepcopy(self.last_uncertainty)} if self.last_uncertainty else {}),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], *, path: Optional[str] = None) -> "Project":
        if data.get("format") != PROJECT_FORMAT:
            raise ValueError("Not an HKL-PDF project file (missing format tag).")

        version = int(data.get("version", 0))

        if version > PROJECT_VERSION:
            raise ValueError(
                f"Project file version {version} is newer than this program "
                f"supports ({PROJECT_VERSION})."
            )

        return cls(
            name=str(data.get("name", "Untitled project")),
            refinement=dict(data.get("refinement", {}) or {}),
            constraints={str(k): str(v) for k, v in (data.get("constraints", {}) or {}).items()},
            datasets=[Dataset.from_json(d) for d in data.get("datasets", []) or []],
            path=path,
            last_results=dict(data.get("last_results", {}) or {}),
            last_uncertainty=dict(data.get("last_uncertainty", {}) or {}),
        )

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path

        if not path:
            raise ValueError("No path given for saving the project.")

        path = os.path.abspath(path)
        tmp = path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2, allow_nan=False)

        os.replace(tmp, path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str) -> "Project":
        path = os.path.abspath(path)

        if path.lower().endswith(".inp"):
            return cls.from_inp(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        return cls.from_json(data, path=path)

    # ------------------------------------------------------------------
    # Legacy .inp import
    # ------------------------------------------------------------------

    @classmethod
    def from_legacy_config(
        cls,
        cfg: Dict[str, Any],
        *,
        name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> "Project":
        """
        One dataset with one phase from a single-phase config dict.
        """
        initial = dict(cfg.get("initial", {}) or {})
        refinable = dict(cfg.get("refinable", {}) or {})
        bounds = dict(cfg.get("bounds", {}) or {})
        refinement = dict(cfg.get("refinement", {}) or {})

        # Flags/bounds without a value cannot be refined; only names with an
        # initial value become parameters.
        all_names = list(initial.keys())

        instrument_lc = {n.lower() for n in INSTRUMENT_PARAMS}
        ds_names = [n for n in all_names if n.lower() in instrument_lc]
        ph_names = [n for n in all_names if n.lower() not in instrument_lc]

        structure_file = str(cfg.get("structure_file", ""))
        data_file = str(cfg.get("gr_data_file", ""))

        phase_name = os.path.splitext(os.path.basename(structure_file))[0] or "Phase"
        ds_name = os.path.splitext(os.path.basename(data_file))[0] or "Dataset"

        phase_settings = {
            k: v for k, v in refinement.items() if k in PHASE_SETTING_KEYS
        }
        project_settings = {
            k: v for k, v in refinement.items() if k not in PHASE_SETTING_KEYS
        }

        phase = Phase(
            id=safe_id(phase_name, "phase"),
            name=phase_name,
            cif=structure_file,
            parameters=_params_from_legacy(ph_names, initial, refinable, bounds),
            settings=phase_settings,
            sections={
                s: copy.deepcopy(cfg.get(s, {}) or {})
                for s in PHASE_SECTIONS
                if cfg.get(s)
            },
            extra={
                k: copy.deepcopy(v)
                for k, v in cfg.items()
                if k not in _LEGACY_HANDLED
            },
        )

        dataset = Dataset(
            id="ds1",
            name=ds_name,
            data_file=data_file,
            r_min=float(cfg.get("r_min", 1.0)),
            r_max=float(cfg.get("r_max", 10.0)),
            r_extension=float(cfg.get("r_extension", 1.2)),
            parameters=_params_from_legacy(ds_names, initial, refinable, bounds),
            phases=[phase],
        )

        # Legacy constraints use local names; namespace them into this phase.
        constraints = {}

        for target, expr in (cfg.get("constraints", {}) or {}).items():
            constraints[_namespace_legacy(target, dataset, phase)] = _namespace_expr(
                str(expr), dataset, phase,
            )

        return cls(
            name=name or phase_name,
            refinement=project_settings,
            constraints=constraints,
            datasets=[dataset],
            path=path,
        )

    @classmethod
    def from_inp(cls, inp_path: str) -> "Project":
        from pdf_fitting.io_handler import read_input_file

        cfg = read_input_file(inp_path)
        project = cls.from_legacy_config(
            cfg,
            name=os.path.splitext(os.path.basename(inp_path))[0],
        )
        # Not saved yet: keep path None so Save asks for a .json name.
        project.path = None
        return project


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _local_names(dataset: Dataset, phase: Phase):
    instrument = set(dataset.parameters)
    phase_names = set(phase.parameters)
    return instrument, phase_names


def _namespace_legacy(name: str, dataset: Dataset, phase: Phase) -> str:
    name = str(name).strip()
    instrument, _ = _local_names(dataset, phase)

    if name in instrument:
        return f"{dataset.id}.{name}"

    return f"{dataset.id}.{phase.id}.{name}"


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def _namespace_expr(expr: str, dataset: Dataset, phase: Phase) -> str:
    """Prefix every known local parameter name in a legacy expression."""
    instrument, phase_names = _local_names(dataset, phase)
    known = instrument | phase_names

    def repl(match):
        token = match.group(0)

        if token in known:
            return _namespace_legacy(token, dataset, phase)

        return token

    return _NAME_RE.sub(repl, expr)


def biso_by_species_from_initial(initial: Dict[str, Any]) -> Dict[str, float]:
    """
    Species-level Biso values (biso_C, biso_Co, ...), as io_handler builds
    them. Site keys such as biso_c3 are not species keys.
    """
    from pdf_fitting.io_handler import clean_element_name

    out: Dict[str, float] = {}

    for k, v in initial.items():
        if not str(k).lower().startswith("biso_"):
            continue

        raw = str(k)[len("biso_"):].strip()

        try:
            species = clean_element_name(raw)
        except Exception:
            continue

        if not raw or raw.lower() != species.lower():
            continue

        try:
            out[species] = float(v)
        except (TypeError, ValueError):
            pass

    return out


def split_name(full_name: str) -> Tuple[str, Optional[str], str]:
    """'ds1.cal.a' -> ('ds1', 'cal', 'a'); 'ds1.qdamp' -> ('ds1', None, 'qdamp')."""
    parts = str(full_name).split(".")

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    if len(parts) == 2:
        return parts[0], None, parts[1]

    raise ValueError(f"Not a namespaced parameter name: {full_name!r}")
