import numpy as np
import time
import math
import threading
import re

class FitStopped(RuntimeError):
    """Raised to abort an ongoing refinement (e.g., user pressed Stop)."""
    pass

from scipy.optimize import least_squares
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def _as_bool(val, default=False) -> bool:
    """Robust bool parsing (handles bool, int, and common string forms)."""
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(int(val))
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)


def _as_int(val, default=0) -> int:
    if val is None:
        return int(default)
    if isinstance(val, bool):
        return int(val)
    try:
        return int(val)
    except Exception:
        return int(default)


def _as_float(val, default=0.0) -> float:
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)




# -----------------------------------------------------------------------------
# Constraints: evaluate [constraints] expressions safely (no raw eval)
# -----------------------------------------------------------------------------
import ast
import operator as _op
import math as _math

_ALLOWED_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.Mod: _op.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: _op.neg,
}
_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": _math.sqrt,
    "log": _math.log,
    "log10": _math.log10,
    "exp": _math.exp,
}

def _safe_eval_expr(expr: str, vars_dict: dict) -> float:
    """Evaluate a numeric expression using only basic ops and whitelisted funcs.

    Compatible with newer Python versions where ast.Num may not exist.
    """
    def _eval(node):
        # Python 3.8+: numbers are ast.Constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Non-numeric constant in constraint")

        # Older Python compatibility: ast.Num may not exist in newer versions.
        ast_num = getattr(ast, "Num", None)
        if ast_num is not None and isinstance(node, ast_num):  # pragma: no cover
            return float(node.n)

        if isinstance(node, ast.Name):
            name = node.id
            if name in vars_dict:
                return float(vars_dict[name])
            raise KeyError(name)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError("Operator not allowed in constraint")
            return _ALLOWED_BINOPS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError("Unary operator not allowed in constraint")
            return _ALLOWED_UNARYOPS[op_type](_eval(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed in constraints")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(f"Function '{fname}' not allowed in constraints")
            args = [_eval(a) for a in node.args]
            return float(_ALLOWED_FUNCS[fname](*args))

        raise ValueError("Unsupported expression in constraint")

    # Names such as "ds1.cal.a" or "lambda_o-o_0" are not Python
    # identifiers; replace every known one with a placeholder first.
    special = sorted(
        (k for k in vars_dict if re.search(r"[.\-]", str(k))),
        key=len,
        reverse=True,
    )

    if special:
        mapping = {}
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.])("
            + "|".join(re.escape(k) for k in special)
            + r")(?![A-Za-z0-9_.])"
        )

        def _sub(match):
            alias = f"__v{len(mapping)}__"
            mapping[alias] = vars_dict[match.group(1)]
            return alias

        expr = pattern.sub(_sub, expr)

        if mapping:
            vars_dict = {**vars_dict, **mapping}

    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree.body))

def apply_constraints(params: dict, constraints: dict) -> dict:
    """Apply constraints in a dependency-tolerant way (iterative resolution)."""
    if not constraints:
        return params
    pending = dict(constraints)
    # iterate a few times to resolve simple dependency chains
    for _ in range(max(5, len(pending) + 1)):
        progressed = False
        for name in list(pending.keys()):
            expr = pending[name]
            try:
                params[name] = _safe_eval_expr(expr, params)
                pending.pop(name, None)
                progressed = True
            except KeyError:
                # depends on a variable not yet available; try later
                continue
        if not pending or not progressed:
            break
    if pending:
        # leave unresolved constraints untouched but raise a clear error
        missing = {k: str(v) for k, v in pending.items()}
        raise ValueError(f"Unresolved constraints (missing deps or circular): {missing}")
    return params

def apply_model_lattice_constraints_to_params(params: dict, model) -> dict:
    """
    Apply lattice/symmetry constraints from the PDF model to the parameter dict.

    This is used for reporting/saving/display so that derived lattice parameters
    (e.g. b=a in hexagonal, c=a in cubic, fixed angles, etc.) are visible in the
    output parameter dict.
    """
    if hasattr(model, "apply_lattice_constraints"):
        return model.apply_lattice_constraints(params or {})

    out = dict(params or {})

    constraints = getattr(model, "constraints", None)

    if constraints is None and hasattr(model, "structure_handler"):
        try:
            constraints = model.structure_handler.get_constraints()
        except Exception:
            constraints = None

    if not isinstance(constraints, dict) or not constraints:
        return out

    lattice_keys = ("a", "b", "c", "alpha", "beta", "gamma")

    # Start with lattice values already present in params.
    lat = {k: out[k] for k in lattice_keys if k in out}

    if not lat:
        return out

    # Resolve equality constraints, e.g. b = a, c = a.
    # Iterate a few times to allow simple chains without risking an infinite loop.
    for _ in range(8):
        changed = False
        for k, v in constraints.items():
            if isinstance(v, str) and v in lat:
                new_val = lat[v]
                old_val = lat.get(k, None)
                if old_val is None or abs(float(old_val) - float(new_val)) > 1e-12:
                    lat[k] = float(new_val)
                    changed = True
        if not changed:
            break

    # Apply fixed numeric constraints, e.g. alpha = 90.
    for k, v in constraints.items():
        if not isinstance(v, str):
            try:
                new_val = float(v)
                old_val = lat.get(k, None)
                if old_val is None or abs(float(old_val) - new_val) > 1e-12:
                    lat[k] = new_val
            except Exception:
                pass

    out.update(lat)
    return out

class FitResult:
    def __init__(self, params, cost):
        self.params = params
        self.cost = cost


class FitEngine:
    """
    Least-squares refinement engine (SciPy) for PDF fitting.

    Key improvements:
    - automatic coarse r-grid during refinement for speed
    - automatic finite-difference step sizing (diff_step) based on bounds/value
    - staged refinement (automatic grouping) to reduce parameter correlation
    - optional robust loss for stability
    """

    def __init__(self, r_exp, G_exp, model, config, *, progress_callback=None, stop_event=None, pause_event=None):
        self.r_exp = np.asarray(r_exp, dtype=float)
        self.G_exp = np.asarray(G_exp, dtype=float)
        self.model = model
        self.config = config

        self.constraints = (config.get('constraints') or {}) if isinstance(config, dict) else {}


        # Normalized refinement config (from [refinement] section if provided)
        self.ref_cfg = (config.get("refinement") or {}) if isinstance(config, dict) else {}

        # One entry per phase: {"prefix", "handler", "ref_cfg"}. A
        # multi-phase ProjectModel gives namespaced prefixes ("ds1.cal.").
        self._phase_entry_list = self._phase_entries(model)

        # [refinement] refine_occ_species = N, C
        # adds/refines occ_<site> for every site of those elements.
        self._expand_species_occupancy_refinement(model, config)

        # [refinement] refine_xyz_species = O
        # adds/refines the symmetry-free x_/y_/z_<site> of those elements;
        # also resolves tied/fixed coordinates of per-site flags.
        self._coord_info = {}
        self._expand_coordinate_refinement(model, config)
        self._expand_adp_refinement(config)

        # Parameter order is fixed by [initial_values] in input
        self.param_names = list(config["initial"].keys())
        self.initial_values = np.array([float(config["initial"][p]) for p in self.param_names], dtype=float)

        # Determine auto lattice refinables by symmetry (only used if not explicitly in config['refinable'])
        auto_refinable = []
        if hasattr(model, "structure") and hasattr(model.structure, "lattice"):
            try:
                sga = SpacegroupAnalyzer(model.structure, symprec=1e-3)
                lattice_system = sga.get_lattice_type()
                auto_refinable = self._default_lattice_refinables(lattice_system)
            except Exception:
                auto_refinable = []

        self.refine_flags = np.array(
            [bool(config["refinable"].get(p, p in auto_refinable)) for p in self.param_names], dtype=bool
        )
        
        # Apply [constraints]: constrained params are not refined (they are derived)
        if isinstance(self.constraints, dict) and self.constraints:
            cons_keys = {str(k).strip() for k in self.constraints.keys()}
            for i, p in enumerate(self.param_names):
                if str(p).strip() in cons_keys:
                    self.refine_flags[i] = False

        self.fit_indices = np.where(self.refine_flags)[0]
        self.fixed_indices = np.where(~self.refine_flags)[0]

        # Optional GUI/progress hooks
        self._progress_callback = progress_callback
        self._stop_event = stop_event
        self._pause_event = pause_event

        # Runtime state
        self._iter_print = 0
        self._last_x_print = None

        # ------------------------------------------------------------------
        # Coarse r-grid during refinement (speed)
        # ------------------------------------------------------------------
        # refinement_r_step can be provided. If missing, choose automatically for large r_max.
        self._refine_idx = None
        refine_step = self._get_refinement_r_step()
        self._set_refine_subgrid(refine_step)

        # Weights (optional): should be same length as experimental G(r)
        self._weights = None
        w = config.get("weights", None)
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape[0] != self.G_exp.shape[0]:
                raise ValueError("weights length must match experimental G(r) length")
            self._weights = np.maximum(w, 0.0)
        
        self.final_model_result = None
        self.final_r = None
        self.final_G_exp = None
        self.final_G_fit = None
        self.final_residual = None
        self.final_chi2 = None

    # -------------------------
    # Configuration helpers
    # -------------------------

    @staticmethod
    def _find_structure_handler(model):
        """Return the StructureHandler behind a (possibly wrapped) model."""
        for candidate in (
            model,
            getattr(model, "pdf_calculator", None),
            getattr(model, "pdf_model", None),
        ):
            handler = getattr(candidate, "structure_handler", None)

            if handler is not None:
                return handler

        return None

    @staticmethod
    def parse_species_list(value) -> list:
        """'N, C' / 'N C' / ['N', 'C'] -> ['N', 'C']."""
        if value is None or isinstance(value, bool):
            return []

        if isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value]
        else:
            items = re.split(r"[,;\s]+", str(value))

        return [item.strip() for item in items if item.strip()]

    def _phase_entries(self, model) -> list:
        """
        [{"prefix", "handler", "ref_cfg"}] for every phase of the model.

        A multi-phase ProjectModel provides phase_handlers() with namespaced
        prefixes ("ds1.calcite."); a single-phase model is one entry with an
        empty prefix and the engine's own [refinement] settings.
        """
        if hasattr(model, "phase_handlers"):
            try:
                return [
                    e for e in model.phase_handlers()
                    if e.get("handler") is not None
                ]
            except Exception as exc:
                print(f"[FIT] Could not list phase handlers: {exc}")
                return []

        handler = self._find_structure_handler(model)

        if handler is None:
            return []

        return [{"prefix": "", "handler": handler, "ref_cfg": self.ref_cfg}]

    def _entry_for_name(self, name: str):
        """Phase entry whose prefix owns this parameter name (longest match)."""
        best = None
        name_lc = str(name).strip().lower()

        for entry in getattr(self, "_phase_entry_list", []) or []:
            prefix = entry["prefix"].lower()

            if name_lc.startswith(prefix):
                if best is None or len(prefix) > len(best["prefix"]):
                    best = entry

        return best

    @staticmethod
    def _local_name(name: str) -> str:
        """'ds1.calcite.occ_o' -> 'occ_o' (names have no other dots)."""
        return str(name).strip().rsplit(".", 1)[-1]

    def _expand_coordinate_refinement(self, model, config) -> None:
        """
        Prepare fractional-coordinate parameters of movable sites, per phase.

        * refine_xyz_species: add x_/y_/z_<site> for every site of those
          elements (CIF values when missing) and refine the free ones.
        * Per-site flags on a symmetry-tied coordinate (e.g. y of an x,x,x
          site) are moved to the free coordinate(s) it follows; flags on a
          symmetry-fixed coordinate are dropped.
        """
        initial = config.setdefault("initial", {})
        refinable = config.setdefault("refinable", {})
        initial_lc = {str(k).strip().lower(): k for k in initial}
        refinable_lc = {str(k).strip().lower(): k for k in refinable}

        def _ensure(full: str, reference: float) -> str:
            existing = initial_lc.get(full)

            if existing is None:
                initial[full] = float(reference)
                initial_lc[full] = full
                existing = full

            return existing

        coord_info = {}

        for entry in self._phase_entry_list:
            handler = entry["handler"]
            prefix = entry["prefix"]

            if not hasattr(handler, "coordinate_parameter_info"):
                continue

            info = handler.coordinate_parameter_info()

            try:
                lattice_abc = tuple(float(v) for v in handler.structure.lattice.abc)
            except Exception:
                lattice_abc = None

            for name, item in info.items():
                coord_info[(prefix + name).lower()] = {
                    **item,
                    "depends_on": [prefix + d for d in item["depends_on"]],
                    "lattice_abc": lattice_abc,
                    "ref_cfg": entry["ref_cfg"],
                }

            species = self.parse_species_list(
                (entry["ref_cfg"] or {}).get("refine_xyz_species", None)
            )

            if species:
                selected = handler.coordinate_parameters_for_elements(species)

                if not selected:
                    print(
                        f"[XYZ] {prefix}refine_xyz_species="
                        f"{', '.join(species)} matched no movable site."
                    )

                for name in selected:
                    full = (prefix + name).lower()
                    existing = _ensure(full, info[name]["reference"])

                    if info[name]["status"] == "free":
                        refinable[existing] = True
                        refinable_lc[full] = existing

        self._coord_info = coord_info

        # Resolve tied / fixed flags.
        for full, item in coord_info.items():
            key = refinable_lc.get(full)

            if key is None or not bool(refinable.get(key, False)):
                continue

            if item["status"] == "free":
                _ensure(full, item["reference"])
                continue

            refinable[key] = False

            if item["status"] == "fixed":
                print(f"[XYZ] {key} is fixed by site symmetry; not refined.")
                continue

            for free_name in item["depends_on"]:
                free_lc = free_name.lower()
                existing = _ensure(free_lc, coord_info[free_lc]["reference"])
                refinable[existing] = True
                refinable_lc[free_lc] = existing

            print(
                f"[XYZ] {key} is tied by site symmetry; refining "
                f"{', '.join(item['depends_on'])} instead."
            )

        # Every coordinate of a movable site is a parameter, so tied ones
        # can be written back after the fit.
        for full, item in coord_info.items():
            _ensure(full, item["reference"])

        n_free = sum(
            1 for full, item in coord_info.items()
            if item["status"] == "free"
            and bool(refinable.get(initial_lc.get(full, full), False))
        )

        if n_free:
            print(f"[XYZ] Refining {n_free} fractional coordinate(s).")

        # Flags on sites the model cannot move are dropped.
        for name_lc, key in list(refinable_lc.items()):
            if (
                re.match(r"^[xyz]_", self._local_name(name_lc))
                and bool(refinable.get(key, False))
                and name_lc not in coord_info
            ):
                refinable[key] = False
                print(
                    f"[XYZ] {key} is not a movable-site coordinate of this "
                    "model; not refined."
                )

    def _coordinate_limits(self, name_lc: str):
        """
        Default x_/y_/z_ limits: CIF value +/- xyz_max_shift (Angstrom,
        default 0.3) converted to fractional units of that axis.
        """
        item = (getattr(self, "_coord_info", {}) or {}).get(name_lc)

        if item is None:
            return -np.inf, np.inf

        ref_cfg = item.get("ref_cfg") or self.ref_cfg
        shift = _as_float(ref_cfg.get("xyz_max_shift", 0.3), 0.3)

        if not shift > 0.0:
            return -np.inf, np.inf

        abc = item.get("lattice_abc")

        if not abc:
            return -np.inf, np.inf

        length = float(abc[item["axis"]])

        if length <= 0.0:
            return -np.inf, np.inf

        p0 = float(item["reference"])
        return p0 - shift / length, p0 + shift / length

    def _symmetrize_coordinates(self, params: dict) -> dict:
        """Tied coordinates follow their free coordinate(s), per phase."""
        out = dict(params)

        for entry in getattr(self, "_phase_entry_list", []) or []:
            handler = entry["handler"]

            if not getattr(handler, "coordinate_groups", None):
                continue

            prefix = entry["prefix"]
            n = len(prefix)

            local = {
                k[n:]: v for k, v in out.items()
                if k.lower().startswith(prefix.lower())
                and "." not in k[n:]
            }

            try:
                fixed = handler.symmetrized_coordinate_params(local)
            except Exception:
                continue

            for k, v in fixed.items():
                out[prefix + k] = v

        # Anisotropic displacement: tied / fixed β components follow the free ones.
        for entry in getattr(self, "_phase_entry_list", []) or []:
            handler = entry["handler"]
            if not getattr(handler, "adp_groups", None):
                continue
            prefix = entry["prefix"]
            n = len(prefix)
            local = {
                k[n:]: v for k, v in out.items()
                if k.lower().startswith(prefix.lower()) and "." not in k[n:]
            }
            try:
                fixed = handler.symmetrized_adp_params(local)
            except Exception:
                continue
            for k, v in fixed.items():
                if k.lower().startswith("beta"):
                    out[prefix + k] = v

        return out

    def _expand_adp_refinement(self, config) -> None:
        """Refine flags on symmetry-tied / fixed β components are dropped (the free ones carry them)."""
        refinable = config.setdefault("refinable", {})
        for entry in self._phase_entry_list:
            handler = entry["handler"]
            if not getattr(handler, "adp_groups", None):
                continue
            prefix = entry["prefix"]
            info = handler.adp_parameter_info()
            for name in list(refinable):
                local = name[len(prefix):] if name.lower().startswith(prefix.lower()) else None
                if local is None or "." in local:
                    continue
                ll = local.lower()
                if not (ll.startswith("beta") and "_" in ll):
                    continue
                comp, key = ll[4:].split("_", 1)
                kind = (info.get(key, {}).get(comp) or ("free", None))[0]
                if kind != "free" and refinable.get(name):
                    refinable[name] = False

    def _expand_species_occupancy_refinement(self, model, config) -> None:
        """
        Expand refine_occ_species (per phase) into per-site occupancy
        parameters.

        Every occupancy group (CIF site label) whose element is listed is
        added to the initial values with its CIF occupancy if missing, and
        flagged refinable. Per-site flags already set stay as they are.
        """
        initial = config.setdefault("initial", {})
        refinable = config.setdefault("refinable", {})
        initial_lc = {str(k).strip().lower(): k for k in initial}

        for entry in self._phase_entry_list:
            handler = entry["handler"]
            prefix = entry["prefix"]

            species = self.parse_species_list(
                (entry["ref_cfg"] or {}).get("refine_occ_species", None)
            )

            if not species:
                continue

            if not hasattr(handler, "occupancy_parameters_for_elements"):
                print(
                    f"[OCCUPANCY] {prefix}refine_occ_species is set, but the "
                    "model has no structure handler; ignoring it."
                )
                continue

            selected = handler.occupancy_parameters_for_elements(species)

            if not selected:
                print(
                    f"[OCCUPANCY] {prefix}refine_occ_species="
                    f"{', '.join(species)} matched no site in the structure."
                )
                continue

            for name, base_value in selected.items():
                full = (prefix + name).lower()
                existing = initial_lc.get(full)

                if existing is None:
                    initial[prefix + name] = float(base_value)
                    existing = prefix + name
                    initial_lc[full] = existing

                refinable[existing] = True

            print(
                f"[OCCUPANCY] Refining {len(selected)} site occupancy "
                f"parameter(s) of {prefix or 'the structure'} for element(s) "
                f"{', '.join(species)}: "
                + ", ".join(prefix + n for n in sorted(selected))
            )

    def _get_refinement_r_step(self) -> float:
        """Auto choose a refinement r-step if not provided."""
        try:
            refine_step = _as_float(self.ref_cfg.get("refinement_r_step", self.config.get("refinement_r_step", 0.0)), 0.0)
        except Exception:
            refine_step = 0.0

        if refine_step > 0.0:
            return refine_step

        # Auto default: coarse grid for big r_max (fast FD Jacobians)
        try:
            rmax = float(self.config.get("r_max", self.config.get("range", {}).get("r_max", 0.0)))
        except Exception:
            rmax = 0.0

        # If rmax not available in config, try model
        if rmax <= 0.0 and hasattr(self.model, "r_max"):
            try:
                rmax = float(self.model.r_max)
            except Exception:
                rmax = 0.0

        # Default heuristic
        if rmax >= 120:
            return 0.10
        if rmax >= 80:
            return 0.05
        return 0.0


    def _set_refine_subgrid(self, refine_step: float) -> None:
        """(Re)build the coarse r-grid index for refinement given a target step.
        If refine_step <= 0, disable subgridding (use full r-grid).
        """
        self._refine_idx = None
        refine_step = float(refine_step or 0.0)
        if refine_step <= 0.0 or len(self.r_exp) <= 10:
            return

        dr = np.diff(self.r_exp)
        dr0 = float(np.median(dr)) if dr.size else 0.0
        if dr0 <= 0.0:
            return

        stride = int(round(refine_step / dr0))
        if stride > 1:
            self._refine_idx = np.arange(0, len(self.r_exp), stride, dtype=int)

    def _default_lattice_refinables(self, lattice_type):
        if lattice_type == "cubic":
            return ["a"]
        elif lattice_type == "tetragonal":
            return ["a", "c"]
        elif lattice_type == "orthorhombic":
            return ["a", "b", "c"]
        elif lattice_type in ("hexagonal", "trigonal"):
            return ["a", "c"]
        elif lattice_type == "monoclinic":
            return ["a", "b", "c", "beta"]
        elif lattice_type == "triclinic":
            return ["a", "b", "c", "alpha", "beta", "gamma"]
        else:
            return []

    # -------------------------
    # Parameter packing/unpacking
    # -------------------------

    def unpack_params(self, x_fit, fit_indices=None):
        """Build full parameter dict from current fit vector and fixed values."""
        if fit_indices is None:
            fit_indices = self.fit_indices
        full_params = self.initial_values.copy()
        full_params[fit_indices] = x_fit
        params = dict(zip(self.param_names, full_params))
        # Apply constraints (if any)
        if getattr(self, 'constraints', None):
            params = apply_constraints(params, self.constraints)
        return params

    # -------------------------
    # Residuals (weighted)
    # -------------------------

    def _evaluate_model(self, params_dict, use_subgrid=True):
        """Evaluate model on either full r-grid or a coarse subgrid."""
        if use_subgrid and self._refine_idx is not None:
            r_sub = self.r_exp[self._refine_idx]
            G_exp_use = self.G_exp[self._refine_idx]
            model_result = self.model.evaluate(params_dict, r_override=r_sub)
            G_fit = np.asarray(model_result["G_r"], dtype=float)
            w = self._weights[self._refine_idx] if self._weights is not None else None
            return r_sub, G_exp_use, G_fit, w

        model_result = self.model.evaluate(params_dict)
        G_fit = np.asarray(model_result["G_r"], dtype=float)
        w = self._weights if self._weights is not None else None
        return self.r_exp, self.G_exp, G_fit, w

    def residuals(self, x_fit, fit_indices=None):
        """Residual vector used by least_squares."""
        if fit_indices is None:
            fit_indices = self.fit_indices

        params_dict = self.unpack_params(x_fit, fit_indices=fit_indices)
        _, G_exp_use, G_fit, w = self._evaluate_model(params_dict, use_subgrid=True)

        res = (G_exp_use - G_fit)
        if w is None:
            return res
        return np.sqrt(w) * res

    # -------------------------
    # Automatic diff_step (finite difference step sizes)
    # -------------------------

    def _build_bounds_for(self, fit_indices):
        """Return bounds arrays aligned with fit_indices.

        Safety rule:
        - Any parameter named biso_* gets default bounds 0..10 Å²
          unless explicit bounds are provided in the input.
        """
        x0 = self.initial_values[fit_indices]
        lb = np.full_like(x0, -np.inf, dtype=float)
        ub = np.full_like(x0, np.inf, dtype=float)

        bounds_cfg = self.config.get("bounds", {}) or {}

        # Case-insensitive bounds lookup.
        bounds_lc = {
            str(k).strip().lower(): v
            for k, v in bounds_cfg.items()
        }

        names = list(np.array(self.param_names)[fit_indices])

        for i, name in enumerate(names):
            name_str = str(name).strip()
            # Defaults depend on the local name ("ds1.cal.a" -> "a");
            # explicit bounds are looked up with the full name.
            full_lc = name_str.lower()
            name_lc = self._local_name(full_lc)

            # Default lattice safety bounds.
            # Prevent zero or degenerate unit-cell volume.
            if name_lc in ("a", "b", "c"):
                lb[i] = 0.1
                ub[i] = np.inf

            elif name_lc in ("alpha", "beta", "gamma"):
                lb[i] = 10.0
                ub[i] = 170.0

            elif name_lc == "stacking_distance":
                lb[i] = 1.0e-4
                ub[i] = np.inf

            elif name_lc == "stacking_fraction":
                lb[i] = 1.0e-4
                ub[i] = 100.0
                
            # Default Biso safety bounds.
            elif name_lc.startswith("biso_"):
                lb[i] = 0.0
                ub[i] = 10.0

            elif name_lc.startswith("occ_"):
                # Crystallographic occupancy bounds: global
                # [refinement] occ_min / occ_max (default 0..1). Per-site
                # [bounds] occ_<site> = lo, hi override these below.
                lb[i], ub[i] = self._global_occupancy_limits(full_lc)

            elif re.match(r"^[xyz]_", name_lc):
                # Fractional coordinates: CIF value +/- xyz_max_shift.
                lb[i], ub[i] = self._coordinate_limits(full_lc)

            # Default SAXS belly safety bounds.
            elif name_lc == "saxs_scale":
                lb[i] = 0.0
                ub[i] = 5.0

            elif name_lc in ("saxs_diameter", "saxs_height"):
                lb[i] = 1.0
                ub[i] = 5000.0

            elif name_lc in ("saxs_diameter_std", "saxs_height_std"):
                lb[i] = 0.0
                ub[i] = 5000.0

            elif (
                name_lc.startswith(
                    (
                        "stack_du_",
                        "stack_dv_",
                    )
                )
                or re.match(
                    r"^stack_state_.+_(du|dv)$",
                    name_lc,
                )
            ):
                lb[i] = -0.5
                ub[i] = 0.5

            elif (
                name_lc.startswith(
                    "stack_h_"
                )
                or re.match(
                    r"^stack_state_.+_h$",
                    name_lc,
                )
            ):
                lb[i] = 1.0e-4
                ub[i] = np.inf

            # Backward-compatible D02 parameter bounds.
            elif name_lc in {
                "stack_u_ab",
                "stack_v_ab",
                "stack_u_repeat",
                "stack_v_repeat",
            }:
                lb[i] = -0.5
                ub[i] = 0.5

            elif name_lc == "stack_delta_h":
                lb[i] = -2.0
                ub[i] = 2.0

            # Explicit user bounds override defaults.
            if name_str in bounds_cfg:
                bval = bounds_cfg[name_str]
            elif full_lc in bounds_lc:
                bval = bounds_lc[full_lc]
            else:
                bval = None

            if bval is not None:
                try:
                    lo, hi = bval
                    if lo not in (None, ""):
                        lb[i] = float(lo)
                    if hi not in (None, ""):
                        ub[i] = float(hi)
                except Exception:
                    pass

            if name_lc.startswith("occ_"):
                # The model clips occupancy to [0, 1]; keep the limits there.
                lb[i] = min(max(lb[i], 0.0), 1.0)
                ub[i] = min(max(ub[i], 0.0), 1.0)

                if not lb[i] < ub[i]:
                    print(
                        f"[OCCUPANCY] Invalid limits for {name_str} "
                        f"({lb[i]:g}, {ub[i]:g}); using 0..1."
                    )
                    lb[i], ub[i] = 0.0, 1.0

        return lb, ub

    def _global_occupancy_limits(self, full_name: str = ""):
        """occ_min / occ_max of the phase owning the parameter, default (0, 1)."""
        entry = self._entry_for_name(full_name) if full_name else None
        ref_cfg = (entry or {}).get("ref_cfg") or self.ref_cfg
        lo = _as_float(ref_cfg.get("occ_min", 0.0), 0.0)
        hi = _as_float(ref_cfg.get("occ_max", 1.0), 1.0)

        lo = min(max(lo, 0.0), 1.0)
        hi = min(max(hi, 0.0), 1.0)

        if not lo < hi:
            print(
                f"[OCCUPANCY] Invalid occ_min/occ_max ({lo:g}, {hi:g}); "
                "using 0..1."
            )
            return 0.0, 1.0

        return lo, hi


    def _auto_diff_step(self, x0, lb, ub):
        """
        Absolute finite-difference steps (used by the engine's own Jacobian,
        see run_refinement; they are NOT SciPy's relative diff_step).

        - ~1e-3 of the value (the model is float32: smaller steps are noise),
        - at least 1e-4 of the bound range (parameters near 0, e.g. ρ ~ 1e-4),
        - at least 1e-4 when the value is 0 and there are no bounds,
        - at most 1/10 of the bound range.
        """
        x0 = np.asarray(x0, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        step = np.empty_like(x0, dtype=float)
        for i in range(x0.size):
            lo, hi = lb[i], ub[i]
            s = 1e-3 * abs(x0[i])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                rng = hi - lo
                s = max(s, 1e-4 * rng)
                s = min(s, 0.1 * rng)
            elif s == 0.0:
                s = 1e-4
            step[i] = max(s, 1e-12)

        return step

    # -------------------------
    # Staged refinement (automatic grouping)
    # -------------------------

    def _auto_stages(self, refinable_names):
        """
        Build stages automatically based on parameter names.
        Only uses names that are actually refinable in the input.
        """
        refinable = set(refinable_names)

        # Group definitions (name patterns)
        geometry_names = {
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "stacking_distance",
            "stacking_fraction",
            "stack_u_ab",
            "stack_v_ab",
            "stack_u_repeat",
            "stack_v_repeat",
            "stack_delta_h",
        }

        # Classify by local name ("ds1.cal.a" -> "a").
        loc = {p: self._local_name(p) for p in refinable}

        geometry = [
            parameter
            for parameter in refinable
            if (
                loc[parameter].lower()
                in geometry_names
                or loc[parameter].lower().startswith(
                    (
                        "stack_du_",
                        "stack_dv_",
                        "stack_h_",
                        "stack_state_",
                    )
                )
            )
        ]
        scale_biso = [p for p in refinable if loc[p].lower().startswith("biso") or loc[p].lower() == "scale"]
        shape = [
            p for p in refinable
            if loc[p] in {
                "d", "D", "d_std", "D_std",
                "cyl_diameter",
                "cyl_thickness",
                "cyl_thickness_std",
                "cyl_axis_h",
                "cyl_axis_k",
                "cyl_axis_l",
                "saxs_scale",
                "saxs_diameter",
                "saxs_diameter_std",
                "saxs_height",
                "saxs_height_std",
                
            }
        ]
        micro = [
            p for p in refinable
            if loc[p].lower() in {"rho", "re", "fe", "pah_a", "pah_b"}
        ]
        others = [
            parameter
            for parameter in refinable
            if parameter
            not in set(
                geometry
                + scale_biso
                + shape
                + micro
            )
            and not loc[parameter].startswith(
                "lambda_"
            )
        ]

        # Order stages to reduce correlation:
        stages = []
        if geometry:
            stages.append(
                (
                    "geometry",
                    sorted(
                        geometry
                    ),
                )
            )
        if scale_biso:
            stages.append(("scale_biso", sorted(scale_biso)))
        if shape:
            stages.append(("shape", sorted(shape)))
        if micro:
            stages.append(("microstrain", sorted(micro)))
        if others:
            stages.append(("other", sorted(others)))

        # Final combined stage (all refinables)
        stages.append(
            (
                "all",
                sorted(refinable_names),
            )
        )

        # Remove empty / duplicates while preserving order
        cleaned = []
        seen = set()
        for name, params in stages:
            t = tuple(params)
            if len(t) == 0:
                continue
            if t in seen:
                continue
            seen.add(t)
            cleaned.append((name, list(t)))
        return cleaned

    def _indices_for_param_list(self, params):
        """Convert a list of param names into indices in param_names."""
        idx = []
        for p in params:
            if p in self.param_names:
                idx.append(self.param_names.index(p))
        return np.array(idx, dtype=int)

    # -------------------------
    # Run refinement
    # -------------------------

    def run_refinement(self):
        if len(self.fit_indices) == 0:
            print("No parameters marked for refinement. Skipping fit.")
            return None

        all_refinable_names = list(np.array(self.param_names)[self.fit_indices])

        # Refinement options (prefer [refinement] section if present)
        ref_cfg = self.ref_cfg or {}
        use_staged = _as_bool(ref_cfg.get("staged_refinement", self.config.get("staged_refinement", True)), default=True)

        # Robust loss options (default on for stability)
        loss = str(ref_cfg.get("loss", self.config.get("loss", "soft_l1")))
        f_scale = _as_float(ref_cfg.get("f_scale", self.config.get("f_scale", 0.05)), 0.05)

        # Iteration limits per stage (keep small; stages add robustness)
        max_nfev_stage = _as_int(ref_cfg.get("max_nfev_stage", self.config.get("max_nfev_stage", 25)), 25)
        max_nfev_final = _as_int(ref_cfg.get("max_nfev_final", self.config.get("max_nfev_final", 60)), 60)

        # Termination tolerances: SciPy defaults are often overkill for PDF fitting.
        # Suggested fast/robust defaults (can be tightened later):
        ftol = _as_float(ref_cfg.get("ftol", 1e-3), 1e-3)
        xtol = _as_float(ref_cfg.get("xtol", 1e-3), 1e-3)
        gtol = _as_float(ref_cfg.get("gtol", 1e-3), 1e-3)

        # Verbosity (0 = silent, 1+ = verbose); keep low for speed
        verbose = _as_int(ref_cfg.get("verbose", 0), 0)

        # Optional multi-resolution r-grid:
        # - during stages: refinement_r_step (coarse)
        # - during final: refinement_r_step_final (default 0 = full grid)
        refine_step_stage = _as_float(ref_cfg.get("refinement_r_step", self._get_refinement_r_step()), 0.0)
        refine_step_final = _as_float(ref_cfg.get("refinement_r_step_final", 0.0), 0.0)

        # Start from initial values (but we will update self.initial_values after each stage)
        current_full = self.initial_values.copy()

        # Define stages
        if use_staged:
            stages = self._auto_stages(all_refinable_names)
        else:
            stages = [("all", all_refinable_names)]

        last_result = None

        for stage_name, stage_params in stages:
            # Stage indices are subset of global fit_indices
            stage_indices_global = self._indices_for_param_list(stage_params)
            stage_indices_global = np.array([i for i in stage_indices_global if self.refine_flags[i]], dtype=int)
            if stage_indices_global.size == 0:
                continue

            # Build x0 for this stage
            x0 = current_full[stage_indices_global].astype(float, copy=True)
            
            lb, ub = self._build_bounds_for(stage_indices_global)
            
            # If old/corrupt values are outside bounds, clip them before SciPy.
            finite_lb = np.isfinite(lb)
            finite_ub = np.isfinite(ub)
            
            if np.any(finite_lb):
                x0[finite_lb] = np.maximum(x0[finite_lb], lb[finite_lb])
            
            if np.any(finite_ub):
                x0[finite_ub] = np.minimum(x0[finite_ub], ub[finite_ub])
            
            # Keep current_full consistent with clipped starting values.
            current_full[stage_indices_global] = x0
            
            # Automatic FD steps
            diff_step = self._auto_diff_step(x0, lb, ub)

            # Choose stage max_nfev
            if stage_name == "all":
                max_nfev = max_nfev_final
            else:
                max_nfev = max_nfev_stage

            # Apply multi-resolution r-subgrid
            if stage_name == "all":
                self._set_refine_subgrid(refine_step_final)
            else:
                self._set_refine_subgrid(refine_step_stage)

            if verbose:
                print(f"\n[STAGE] {stage_name}: refining {stage_params} (n={len(stage_params)})")
            if verbose and self._refine_idx is not None:
                print(f"[STAGE] using coarse r-grid: {len(self._refine_idx)} / {len(self.r_exp)} points")

            
            # Progress output controls (prints while SciPy is running)
            # - progress_every_sec: print at most once every N seconds (default 1.0)
            # - progress_every_nfev: additionally print every N residual calls (0 disables)
            progress_every_sec = _as_float(ref_cfg.get("progress_every_sec", 1.0), 1.0)
            progress_every_nfev = _as_int(ref_cfg.get("progress_every_nfev", 0), 0)

            # Stage-local progress state
            nfev_local = 0
            t_start = time.perf_counter()
            t_last = t_start

            best_cost = float(
                "inf"
            )

            best_rwp = float(
                "inf"
            )

            best_x_stage = None
            
            # Local residual function for the stage
            def _res_stage(x_stage):
                # Allow GUI to pause/stop
                if self._stop_event is not None and self._stop_event.is_set():
                    raise FitStopped("Refinement stopped by user")
                if self._pause_event is not None:
                    while self._pause_event.is_set():
                        if self._stop_event is not None and self._stop_event.is_set():
                            raise FitStopped("Refinement stopped by user")
                        time.sleep(0.05)
            
                # Build full params for this stage
                tmp_full = current_full.copy()
                tmp_full[stage_indices_global] = x_stage
                params_dict = dict(zip(self.param_names, tmp_full))

                # Apply [constraints] during every model evaluation.
                # Example: [constraints] re = d
                if getattr(self, "constraints", None):
                    params_dict = apply_constraints(params_dict, self.constraints)
            
                # Evaluate on refinement grid (subgrid if enabled).
                # Invalid trial points should return a large penalty, not crash.
                t_model0 = time.perf_counter()

                try:
                    r_use, G_exp_use, G_fit, w = self._evaluate_model(
                        params_dict,
                        use_subgrid=True,
                    )

                    G_fit = np.asarray(G_fit, dtype=float)

                    if (
                        G_fit.size == 0
                        or not np.all(np.isfinite(G_fit))
                    ):
                        raise ValueError("Model returned invalid G_fit.")

                except FitStopped:
                    raise

                except Exception:
                    if self._refine_idx is not None:
                        G_exp_penalty = self.G_exp[self._refine_idx]
                    else:
                        G_exp_penalty = self.G_exp

                    return np.full_like(
                        np.asarray(G_exp_penalty, dtype=float),
                        1.0e6,
                        dtype=float,
                    )

                if self._stop_event is not None and self._stop_event.is_set():
                    raise FitStopped("Refinement stopped by user")

                model_eval_s = time.perf_counter() - t_model0
            
                res = (G_exp_use - G_fit)
                if w is not None:
                    res_w = (np.sqrt(w) * res)
                else:
                    res_w = res
            
                # ---- progress printing ----
                nonlocal nfev_local, t_start, t_last
                nonlocal best_cost, best_rwp, best_x_stage
            
                nfev_local += 1
                cost = 0.5 * float(np.dot(res_w, res_w))  # SciPy convention
            
                # Compute Rwp (on the SAME grid used for optimization)
                if w is not None:
                    num = float(np.sum(w * res * res))
                    den = float(np.sum(w * G_exp_use * G_exp_use))
                else:
                    num = float(np.sum(res * res))
                    den = float(np.sum(G_exp_use * G_exp_use))
                rwp = 100.0 * math.sqrt(num / den) if den > 0 else float("nan")
            
                # Update best-so-far envelope.
                #
                # Important:
                # SciPy may optimize a robust loss such as soft_l1, while the
                # displayed quantity is conventional Rwp. Therefore preserve
                # the actual parameter vector corresponding to the lowest Rwp.
                if cost < best_cost:
                    best_cost = cost

                if (
                    np.isfinite(
                        rwp
                    )
                    and rwp
                    < best_rwp
                ):
                    best_rwp = float(
                        rwp
                    )

                    best_x_stage = np.asarray(
                        x_stage,
                        dtype=float,
                    ).copy()
            
                now = time.perf_counter()
                do_print = False
                if progress_every_nfev > 0 and (nfev_local % progress_every_nfev == 0):
                    do_print = True
                if progress_every_sec > 0 and (now - t_last) >= progress_every_sec:
                    do_print = True
            
                if do_print:
                    elapsed = now - t_start
                    payload_params = apply_model_lattice_constraints_to_params(params_dict, self.model)

                    payload = {
                        "stage": stage_name,
                        "nfev": int(nfev_local),
                        "model_eval_s": float(model_eval_s),
                        "cost": float(cost),
                        "best_cost": float(best_cost),
                        "rwp": float(rwp),
                        "best_rwp": float(best_rwp),
                        "grid_n": int(len(r_use)),
                        "grid_total": int(len(self.r_exp)),
                        "elapsed_s": float(elapsed),
                        "params": payload_params,
                    }
            
                    # Optionally provide fit/residual snapshots
                    want_pdf_update = _as_bool(ref_cfg.get("update_pdf_during_refinement", False), default=False)
                    if want_pdf_update:
                        payload["r"] = r_use
                        payload["G_exp"] = G_exp_use
                        payload["G_fit"] = G_fit
                        payload["residual"] = res
            
                    if self._progress_callback is not None:
                        try:
                            self._progress_callback(payload)
                        except Exception:
                            pass
                    else:
                        print(
                            f"[RUN] stage={stage_name:>10s}  "
                            f"nfev={nfev_local:6d}  "
                            f"cost={cost:.6e}  best={best_cost:.6e}  "
                            f"Rwp={rwp:7.3f}%  bestRwp={best_rwp:7.3f}%  "
                            f"grid={len(r_use)}/{len(self.r_exp)}  "
                            f"eval={model_eval_s:6.3f}s  "
                            f"t={elapsed:7.2f}s"
                        )
                    t_last = now
            
                return res_w

            # Forward-difference Jacobian with the absolute steps above.
            # (SciPy's diff_step is relative — h = diff_step·|x| — which gave
            # wrong steps for large values and a ~1e-8 step at x = 0.) The
            # residual SciPy has just computed at x is reused.
            fd_cache = {"x": None, "f": None}

            def _fun(x_stage, _res=_res_stage, _c=fd_cache):
                f = _res(x_stage)
                _c["x"] = np.array(x_stage, dtype=float, copy=True)
                _c["f"] = f
                return f

            def _jac(x_stage, _res=_res_stage, _c=fd_cache, _h=diff_step, _lb=lb, _ub=ub):
                x = np.asarray(x_stage, dtype=float)
                if _c["x"] is not None and np.array_equal(_c["x"], x):
                    f0 = _c["f"]
                else:
                    f0 = _res(x)
                J = np.empty((f0.size, x.size), dtype=float)
                for i in range(x.size):
                    h = float(_h[i])
                    if x[i] + h > _ub[i]:
                        h = -h  # backward difference at an upper bound
                        if x[i] + h < _lb[i]:
                            h = 0.5 * (_lb[i] - x[i]) if _lb[i] < x[i] else 0.5 * (_ub[i] - x[i])
                    if h == 0.0:
                        J[:, i] = 0.0
                        continue
                    xi = x.copy()
                    xi[i] = x[i] + h
                    J[:, i] = (_res(xi) - f0) / h
                return J

            # Actually run least squares for the stage
            try:
                result = least_squares(
                fun=_fun,
                x0=x0,
                jac=_jac,
                method="trf",
                bounds=(lb, ub),
                x_scale="jac",
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                    verbose=verbose,
                )
            except FitStopped:
                # Propagate to caller (GUI) with partial progress
                raise


            last_result = result

            # Commit the best Rwp point seen in this stage, not necessarily
            # the final optimizer point.
            #
            # This is essential when robust loss is enabled because the final
            # soft_l1 minimum can have a slightly worse conventional Rwp.
            if (
                best_x_stage is not None
                and best_x_stage.size == x0.size
            ):
                current_full[
                    stage_indices_global
                ] = best_x_stage

                print(
                    "[STAGE BEST RWP] "
                    f"{stage_name}: "
                    f"Rwp={best_rwp:.6f}%"
                )

            elif (
                result.x is not None
                and result.x.size == x0.size
            ):
                current_full[
                    stage_indices_global
                ] = result.x

            # Print stage summary on full grid
            stage_params_dict = dict(zip(self.param_names, current_full))
            if getattr(self, "constraints", None):
                stage_params_dict = apply_constraints(stage_params_dict, self.constraints)

            stage_params_dict = apply_model_lattice_constraints_to_params(stage_params_dict, self.model)

            _, G_exp_full, G_fit_full, w_full = self._evaluate_model(stage_params_dict, use_subgrid=False)
            res_full = (G_exp_full - G_fit_full)
            if w_full is not None:
                chi2 = float(np.mean((np.sqrt(w_full) * res_full) ** 2))
            else:
                chi2 = float(np.mean(res_full ** 2))

            print(f"[STAGE DONE] {stage_name} χ²(full grid) = {chi2:.6f}")
            for p in stage_params:
                if p in stage_params_dict:
                    print(f"  {p} = {stage_params_dict[p]:.6f}")

            # Optional early stop: if stage did not improve, you could break
            # (not enabled by default)

        # Commit final params
        self.initial_values = current_full.copy()
        final_params = dict(zip(self.param_names, self.initial_values))
        if getattr(self, "constraints", None):
            final_params = apply_constraints(final_params, self.constraints)
        final_params = apply_model_lattice_constraints_to_params(final_params, self.model)
        # Tied coordinates follow their free coordinate(s) in the output.
        final_params = self._symmetrize_coordinates(final_params)

        # Evaluate final G(r) on full grid (only once)
        model_result = self.model.evaluate(final_params)

        G_fit_full = np.asarray(model_result["G_r"], dtype=float)
        res_full = self.G_exp - G_fit_full
        chi2 = float(np.mean(res_full ** 2))

        # Store final full-grid result so worker.py does not recompute it.
        self.final_model_result = model_result
        self.final_r = np.asarray(model_result.get("r", self.r_exp), dtype=float)
        self.final_G_exp = np.asarray(self.G_exp, dtype=float)
        self.final_G_fit = G_fit_full
        self.final_residual = res_full
        self.final_chi2 = chi2

        # Optional: skip verbose output
        verbose = _as_int(ref_cfg.get("verbose", 0), 0)
        if verbose > 0:
            print(f"\n[Refinement completed] χ² = {chi2:.6f}")
            refinable_params = {
                k: final_params[k] for k in self.param_names
                if self.config["refinable"].get(k, False)
            }
            for k, v in refinable_params.items():
                print(f"  {k} = {v:.6f}")

        return final_params