from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

import dataclasses
import hashlib
import json
import threading
from collections import OrderedDict

# Frozen topologies of recent runs (same process): Calculate / Refine reuse
# them instead of recompiling (paths, ensemble geometry, pair and shell
# topology), which costs ~20 s for a COF ensemble. Keyed on every input of
# the compile steps; at most two kept.
_TOPOLOGY_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_TOPOLOGY_CACHE_SIZE = 2
_TOPOLOGY_LOCK = threading.Lock()
_GEOMETRY_PREFIXES = ("stack", "state", "layer", "nominal", "repeat", "p_")

# Disk copy of slimmed topologies (raw pair rows dropped): survives restarts.
_TOPOLOGY_DISK_VERSION = "stochastic_topology_v1"
_TOPOLOGY_DISK_KEEP = 4


def _topology_disk_dir():
    import os

    from pdf_fitting.user_paths import user_cache_dir

    path = os.path.join(user_cache_dir(), "stochastic_topology")
    os.makedirs(path, exist_ok=True)
    return path


def _slim_rows(rows, keep_weight: bool):
    """Same dataclass with every per-row array cut to 0 rows (or 1 summed row)."""
    import numpy as _np

    fields = {}
    for f in dataclasses.fields(rows):
        v = getattr(rows, f.name)
        if isinstance(v, _np.ndarray) and v.ndim >= 1:
            fields[f.name] = v[:0]
        else:
            fields[f.name] = v
    if keep_weight:
        c = _np.asarray(rows.transition_coefficients, dtype=_np.float64)
        fields["transition_coefficients"] = _np.sum(_np.abs(c), axis=0)[None, :]
    return dataclasses.replace(rows, **fields)


def _slim_entry(entry: dict) -> dict:
    """Topology entry without the raw pair rows (only needed while compiling)."""
    pairs = entry["compiled_pairs"]
    slim_pairs = dataclasses.replace(
        pairs,
        finite_rows=_slim_rows(pairs.finite_rows, keep_weight=False),
        bulk_rows=_slim_rows(pairs.bulk_rows, keep_weight=True),
    )
    out = dict(entry)
    out["compiled_pairs"] = slim_pairs
    out["compiled_shells"] = dataclasses.replace(entry["compiled_shells"], pair_topology=slim_pairs)
    return out


def _disk_load(key: str):
    import os
    import pickle

    path = os.path.join(_topology_disk_dir(), f"{key}.pkl")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if data.get("version") != _TOPOLOGY_DISK_VERSION:
            return None
        os.utime(path, None)
        return data["entry"]
    except Exception as exc:  # noqa: BLE001
        print(f"[STOCHASTIC PDF] Topology cache file unreadable ({exc}); recompiling.")
        return None


def _disk_save(key: str, entry: dict) -> None:
    import os
    import pickle

    try:
        folder = _topology_disk_dir()
        path = os.path.join(folder, f"{key}.pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"version": _TOPOLOGY_DISK_VERSION, "entry": _slim_entry(entry)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        files = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")),
            key=os.path.getmtime,
            reverse=True,
        )
        for old in files[_TOPOLOGY_DISK_KEEP:]:
            try:
                os.remove(old)
            except OSError:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"[STOCHASTIC PDF] Topology not saved to disk: {exc}")
_LATTICE_KEYS = {"a", "b", "c", "alpha", "beta", "gamma"}


def _freeze(obj):
    """Deterministic JSON-able form of specs / structures / arrays."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _freeze(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _freeze(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_freeze(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_freeze(v) for v in obj.tolist()]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 10)
    if isinstance(obj, (int, np.integer, bool, np.bool_)) or obj is None or isinstance(obj, str):
        return obj if not isinstance(obj, (np.integer, np.bool_)) else obj.item()
    if hasattr(obj, "to_dict"):
        try:
            return _freeze(obj.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        return {k: _freeze(v) for k, v in sorted(vars(obj).items()) if not k.startswith("_")}
    return repr(obj)

from pdf_fitting.models.compiled_stochastic_geometry import (
    compile_stochastic_ensemble_geometry,
)

from pdf_fitting.models.compiled_stochastic_pairs import (
    compile_stochastic_pair_topology,
)

from pdf_fitting.models.compiled_stochastic_shells import (
    compile_stochastic_shell_topology,
    evaluate_stochastic_shell_geometry,
    initial_stochastic_shell_table,
)

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
)

from pdf_fitting.models.stacking_disorder import (
    generate_correlated_ab_paths,
    generate_generic_markov_paths,
    state_transition_statistics,
)


print(
    "[STOCHASTIC PDF MODEL] FAST VERSION:",
    __file__,
    flush=True,
)


class StochasticStackedPdfModel:
    """
    TOPAS-like correlated stochastic A/B finite-stacking PDF model.

    This model wraps the existing PDFCalculator.

    During one P(alternate) candidate:

        - random uniforms are frozen;
        - Markov paths are frozen;
        - finite/bulk pair topology is compiled once;
        - state-A/state-B vectors and lattice parameters remain continuously
          refinable;
        - a new stochastic ensemble shell table is generated only when the
          geometry parameters actually change.

    P(alternate) must be treated as an outer grid-search variable. It is not
    a normal continuous scipy least-squares parameter.
    """

    def __init__(
        self,
        *,
        pdf_calculator,
        source_structure,
        shape_spec: CrystalliteShapeSpec,
        stack_spec: LayerStackSpec,
        pair_cutoff: float,
        reference_parameters: Optional[
            Mapping[str, Any]
        ] = None,
        topology_margin: float = 1.0,
        bulk_layers: Optional[int] = None,
        bulk_max_center_layers: Optional[int] = None,
    ):
        self.pdf_calculator = pdf_calculator
        self.source_structure = source_structure

        self.shape_spec = CrystalliteShapeSpec.from_dict(
            shape_spec.to_dict()
        )

        self.stack_spec = stack_spec.normalized()

        if (
            self.stack_spec.model
            not in {
                "correlated_ab_disorder",
                "generic_markov",
            }
        ):
            raise ValueError(
                "StochasticStackedPdfModel requires "
                "layer_stacking model = correlated_ab_disorder "
                "or generic_markov."
            )

        if not self.stack_spec.enabled:
            raise ValueError(
                "The stochastic stacking model is disabled."
            )

        self.pair_cutoff = float(
            pair_cutoff
        )

        self.topology_margin = max(
            float(
                topology_margin
            ),
            0.0,
        )

        self.reference_parameters = dict(
            reference_parameters
            or {}
        )

        refinement_config = dict(
            self.pdf_calculator.config.get(
                "refinement",
                {},
            )
            or {}
        )

        self.stochastic_topology_workers = int(
            refinement_config.get(
                "stochastic_topology_workers",
                0,
            )
        )

        self.stochastic_collapse_realizations = bool(
            refinement_config.get(
                "stochastic_collapse_realizations",
                True,
            )
        )

        self.stochastic_deduplicate_paths = bool(
            refinement_config.get(
                "stochastic_deduplicate_paths",
                True,
            )
        )

        if bulk_layers is None:
            bulk_layers = int(
                getattr(
                    self.stack_spec,
                    "stochastic_bulk_layers",
                    refinement_config.get(
                        "stochastic_bulk_layers",
                        100,
                    ),
                )
            )

        if bulk_max_center_layers is None:
            bulk_max_center_layers = int(
                getattr(
                    self.stack_spec,
                    "stochastic_bulk_max_center_layers",
                    refinement_config.get(
                        "stochastic_bulk_max_center_layers",
                        8,
                    ),
                )
            )

        self.bulk_layers = max(
            4,
            int(
                bulk_layers
            ),
        )

        self.bulk_max_center_layers = max(
            1,
            int(
                bulk_max_center_layers
            ),
        )

        self._last_geometry_key = None
        self._last_shell_table = None
        self._geometry_update_count = 0

        self._build_frozen_topology()

        self._last_geometry = None
        self._last_parameters = dict(
            self.reference_parameters
        )

        initial_table = initial_stochastic_shell_table(
            self.compiled_shells,
            self.reference_parameters,
        )

        self.pdf_calculator.replace_shell_table(
            initial_table
        )

        self._apply_geometry(
            self.reference_parameters,
            force=True,
        )

        self._cuda_backend = None
        self._cuda_reason = "CPU backend selected"
        self._init_cuda_backend()

    # ------------------------------------------------------------------
    # GPU (CUDA) evaluation: geometry, σ, amplitudes, Gaussians, finite-size
    # and SAXS γ on the device; Qmax termination on the CPU as elsewhere.
    # ------------------------------------------------------------------
    def _init_cuda_backend(self) -> None:
        refcfg = (self.pdf_calculator.config.get("refinement") or {}) if isinstance(self.pdf_calculator.config, dict) else {}
        requested = str(refcfg.get("pdf_backend", "cpu") or "cpu").strip().lower()
        if requested not in ("cuda", "gpu", "cupy", "auto"):
            return
        try:
            from pdf_fitting.models.gpu.generic_stacked_pdf import GenericStackedCudaPdfBackend

            t = self.compiled_shells
            calc = self.pdf_calculator
            self._cuda_backend = GenericStackedCudaPdfBackend(
                delta_fractional=t.delta_fractional,
                delta_a=t.delta_a,
                delta_b=t.delta_b,
                transition_coefficients=t.transition_coefficients,
                shell_pair_id=np.asarray(calc._shell_pair_id, dtype=np.int32),
                shell_alpha_id=np.asarray(calc._shell_alpha_id, dtype=np.int32),
                shell_beta_id=np.asarray(calc._shell_beta_id, dtype=np.int32),
                shell_upair_id=np.asarray(calc._shell_upair_id, dtype=np.int32),
                shell_k_rank=np.asarray(calc._shell_k_rank, dtype=np.int32),
                shell_k_within_upair=np.asarray(calc._shell_k_within_upair, dtype=np.int32),
                shell_mult_eff=t.shell_mult_eff,
                shell_bulk_mult_eff=t.shell_bulk_mult_eff,
                species_count=len(calc._unique_species_for_shells),
                pair_weights=np.asarray(calc._w_ordered, dtype=np.float32),
                pair_offsets=np.asarray(calc._pair_offsets, dtype=np.int32),
                pair_cutoff=float(self.pair_cutoff),
                device_id=int(refcfg.get("gpu_device", 0) or 0),
            )
            self._cuda_reason = ""
            print(
                f"[STOCHASTIC GPU] {t.number_of_shells:,} shells resident on the GPU "
                f"({self._cuda_backend.memory_bytes() / 2**20:.0f} MiB)."
            )
        except Exception as exc:  # noqa: BLE001
            self._cuda_backend = None
            self._cuda_reason = f"GPU backend unavailable: {exc}"
            print(f"[STOCHASTIC GPU] {self._cuda_reason}; using the CPU.")

    def _strain_active(self, parameters) -> bool:
        p = {str(k).lower(): v for k, v in parameters.items()}
        refcfg = (self.pdf_calculator.config.get("refinement") or {}) if isinstance(self.pdf_calculator.config, dict) else {}
        model = str(p.get("microstrain_model", refcfg.get("microstrain_model", "auto"))).strip().lower()
        rho = float(p.get("rho", 0.0) or 0.0)
        pah = float(p.get("pah_a", 0.0) or 0.0) != 0.0 or float(p.get("pah_b", 0.0) or 0.0) != 0.0
        if model in ("isotropic", "iso", "none", "off"):
            return False
        if model in ("wilkens", "wilkins", "dislocation"):
            return rho > 0.0
        if model in ("pah", "adler-houska", "adler_houska"):
            return pah
        return rho > 0.0 or pah

    def _can_use_cuda(self, parameters, kwargs) -> bool:
        if self._cuda_backend is None:
            return False
        for key, value in (kwargs or {}).items():
            if key != "r_override" and value:
                return False  # contributions / partial outputs: CPU path
        calc = self.pdf_calculator
        if getattr(calc, "_adp", None) is not None:
            return False
        if float(parameters.get("cyl_diameter", 0.0) or 0.0) > 0.0:
            return False  # cylinder common-volume amplitude factor: CPU path
        return True

    def _evaluate_cuda(self, parameters, kwargs):
        from pdf_fitting.models.compiled_stochastic_shells import _effective_stacking_lattice, _state_vectors
        from pdf_fitting.models.dynamic_stacked_model import DynamicStackedPdfModel
        from pdf_fitting.models.gr_model import apply_qmax_termination
        from pdf_fitting.models.layer_stacking import layer_basis

        calc = self.pdf_calculator
        pair_topology = self.compiled_shells.pair_topology
        parent_lattice, _stack_lat, _stack_inv, stacking_volume, _h, _prob = _effective_stacking_lattice(pair_topology, parameters)
        state_du, state_dv, state_h = _state_vectors(pair_topology, parameters)
        vector_a, vector_b, stack_normal = layer_basis(np.asarray(parent_lattice, dtype=float))
        n_source = int(len(pair_topology.geometry.source_structure.sites))
        calc._stacking_volume = float(stacking_volume)
        calc._stacking_bulk_number_density = float(n_source) / max(float(stacking_volume), 1e-30)
        rho0, density_by_species = calc._current_number_density_and_species_densities(float(stacking_volume))
        biso_by_species = calc._biso_values_vector_from_params(parameters)
        (glob_lam, pair_lam, d1, d2, pd1, pd2) = DynamicStackedPdfModel._gpu_lambda_delta_arrays(self, parameters)
        saxs = DynamicStackedPdfModel._gpu_saxs_parameters(self, parameters, lattice_matrix=parent_lattice)

        r_override = kwargs.get("r_override")
        r_requested = np.asarray(calc.r if r_override is None else r_override, dtype=float)
        cfg = calc.config
        qmax = float(parameters.get("qmax", cfg.get("qmax", 0.0)))
        qmax_zeros = int(parameters.get("qmax_zeros", cfg.get("qmax_zeros", 5)))
        qmax_pad = str(parameters.get("qmax_pad", cfg.get("qmax_pad", "zero"))).strip().lower()
        r_gpu = r_requested
        internal = False
        if qmax > 0.0 and r_requested.size > 1:
            step = float(np.median(np.diff(r_requested)))
            if np.isfinite(step) and step > 0.0:
                width = float(qmax_zeros) * np.pi / qmax
                r_gpu = np.arange(0.0, float(r_requested[-1]) + width + 0.5 * step, step, dtype=np.float32)
                internal = True

        backend = self._cuda_backend
        from pdf_fitting.models.gpu.generic_stacked_pdf import configure_strain_from_calculator

        configure_strain_from_calculator(backend, calc, parameters, float(np.linalg.norm(np.asarray(parent_lattice)[0])))
        backend.set_state_vectors(state_du, state_dv, state_h)
        backend.update_pair_weights(np.asarray(calc._w_ordered, dtype=np.float32))
        res = backend.evaluate(
            lattice_matrix=np.asarray(parent_lattice, dtype=float),
            vector_a=vector_a, vector_b=vector_b, stack_normal=stack_normal, spacing=1.0,
            global_lambda_values=glob_lam, pair_lambda_values=pair_lam,
            global_delta1=d1, global_delta2=d2, pair_delta1=pd1, pair_delta2=pd2,
            biso_by_species=biso_by_species, density_by_species=density_by_species,
            delta_g=float(parameters.get("delta_g", 0.0)), delta_broad=float(parameters.get("delta_broad", 0.0)),
            r_grid=r_gpu, scale=float(parameters.get("scale", 1.0)), rho0=float(rho0), **saxs,
        )
        G = np.asarray(res["G_r"], dtype=float)
        if qmax > 0.0:
            G = apply_qmax_termination(G, np.asarray(r_gpu, dtype=float), qmax, n_zeros=qmax_zeros,
                                       pad_mode=qmax_pad, use_sine_kernel=True, normalize=False)
        g = np.asarray(res["g_r"], dtype=float)
        if internal:
            G = np.interp(r_requested, np.asarray(r_gpu, dtype=float), G)
            g = np.interp(r_requested, np.asarray(r_gpu, dtype=float), g)
        # The CPU shell table no longer matches these parameters.
        self._last_geometry_key = None
        return {"r": r_requested, "G_r": G, "g_r": g, "g_ab": {}, "G_ab": {}, "G_contrib": {}, "weights": {}, "gamma_avg": None}

    def __getattr__(
        self,
        name: str,
    ):
        """
        Forward unsupported methods/attributes to PDFCalculator.

        This keeps FitEngine, diagnostics, Warren plots, pair contributions,
        local trends, and shape-factor tools compatible.
        """
        calculator = object.__getattribute__(
            self,
            "pdf_calculator",
        )

        return getattr(
            calculator,
            name,
        )

    def _number_of_layers(
        self,
    ) -> int:
        return max(
            1,
            int(
                round(
                    float(
                        getattr(
                            self.shape_spec,
                            "axis3_cells",
                            self.shape_spec.height_cells,
                        )
                    )
                )
            ),
        )

    def _topology_cache_key(self, number_of_layers: int) -> str:
        st = self.source_structure
        geom_params = {
            k: v for k, v in (self.reference_parameters or {}).items()
            if str(k).lower() in _LATTICE_KEYS or str(k).lower().startswith(_GEOMETRY_PREFIXES)
        }
        payload = {
            "lattice": _freeze(np.asarray(st.lattice.matrix)),
            "sites": [[str(site.species), _freeze(np.round(site.frac_coords, 8))] for site in st.sites],
            "shape": _freeze(self.shape_spec),
            "stack": _freeze(self.stack_spec),
            "layers": int(number_of_layers),
            "cutoff": round(float(self.pair_cutoff), 8),
            "margin": round(float(self.topology_margin), 8),
            "bulk_layers": int(self.bulk_layers),
            "bulk_center": int(self.bulk_max_center_layers),
            "collapse": bool(self.stochastic_collapse_realizations),
            "dedupe": bool(self.stochastic_deduplicate_paths),
            "params": _freeze(geom_params),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")).hexdigest()

    def _build_frozen_topology(
        self,
    ) -> None:
        """
        Build frozen random paths plus finite and long-bulk pair topology.

        This is intentionally called once per P(alternate) grid candidate.
        """
        number_of_layers = self._number_of_layers()

        cache_key = self._topology_cache_key(number_of_layers)
        try:
            from pdf_fitting.models import fast_unique as _fu

            _refcfg = (self.pdf_calculator.config.get("refinement") or {}) if isinstance(self.pdf_calculator.config, dict) else {}
            # GPU sorting of the pair descriptors is exact but, with the
            # compile already running in parallel CPU threads, it did not
            # shorten the build (COF test: 24.1 s vs 25.2 s). Opt-in only.
            _fu.set_gpu(str(_refcfg.get("gpu_topology_grouping", "false")).strip().lower() in ("1", "true", "yes", "on"))
        except Exception:  # noqa: BLE001
            pass
        with _TOPOLOGY_LOCK:
            cached = _TOPOLOGY_CACHE.get(cache_key)
            if cached is not None:
                _TOPOLOGY_CACHE.move_to_end(cache_key)
        if cached is None:
            cached = _disk_load(cache_key)
            if cached is not None:
                with _TOPOLOGY_LOCK:
                    _TOPOLOGY_CACHE[cache_key] = cached
                    while len(_TOPOLOGY_CACHE) > _TOPOLOGY_CACHE_SIZE:
                        _TOPOLOGY_CACHE.popitem(last=False)
                print("[STOCHASTIC PDF] Frozen topology loaded from the disk cache (no recompilation).", flush=True)
        if cached is not None:
            for name, value in cached.items():
                setattr(self, name, value)
            print("[STOCHASTIC PDF] Frozen topology reused (no recompilation).", flush=True)
            return

        if self.stack_spec.model == "generic_markov":
            self.path_set = generate_generic_markov_paths(
                number_of_layers=number_of_layers,
                ensemble_samples=int(
                    self.stack_spec.ensemble_samples
                ),
                transition_matrix=np.asarray(
                    self.stack_spec.markov_transition_matrix,
                    dtype=float,
                ),
                seed=int(
                    self.stack_spec.ensemble_seed
                ),
                start_state=int(
                    self.stack_spec.markov_start_state
                ),
            )

        else:
            self.path_set = generate_correlated_ab_paths(
                number_of_layers=number_of_layers,
                ensemble_samples=int(
                    self.stack_spec.ensemble_samples
                ),
                p_alternate=float(
                    self.stack_spec.p_alternate
                ),
                seed=int(
                    self.stack_spec.ensemble_seed
                ),
                start_state=str(
                    self.stack_spec.ensemble_start_state
                ),
            )

        if self.stack_spec.model == "generic_markov":
            self.path_statistics = self._generic_transition_statistics(
                self.path_set.states
            )
        else:
            self.path_statistics = state_transition_statistics(
                self.path_set.states
            )

        self.compiled_geometry = (
            compile_stochastic_ensemble_geometry(
                structure=self.source_structure,
                shape_spec=self.shape_spec,
                stack_spec=self.stack_spec,
                state_paths=self.path_set.states,
                reference_parameters=self.reference_parameters,
            )
        )

        self.compiled_pairs = (
            compile_stochastic_pair_topology(
                geometry=self.compiled_geometry,
                pair_cutoff=self.pair_cutoff,
                topology_margin=self.topology_margin,
                reference_parameters=self.reference_parameters,
                bulk_layers=self.bulk_layers,
                bulk_max_center_layers=(
                    self.bulk_max_center_layers
                ),
                cpu_workers=(
                    self.stochastic_topology_workers
                ),
                collapse_realizations=(
                    self.stochastic_collapse_realizations
                ),
                deduplicate_paths=(
                    self.stochastic_deduplicate_paths
                ),
            )
        )

        self.compiled_shells = (
            compile_stochastic_shell_topology(
                pair_topology=self.compiled_pairs,
                reference_parameters=self.reference_parameters,
            )
        )

        print(
            "[STOCHASTIC PDF MODEL] Static shell descriptors compiled once.",
            flush=True,
        )

        with _TOPOLOGY_LOCK:
            _TOPOLOGY_CACHE[cache_key] = {
                "path_set": self.path_set,
                "path_statistics": self.path_statistics,
                "compiled_geometry": self.compiled_geometry,
                "compiled_pairs": self.compiled_pairs,
                "compiled_shells": self.compiled_shells,
            }
            _TOPOLOGY_CACHE.move_to_end(cache_key)
            while len(_TOPOLOGY_CACHE) > _TOPOLOGY_CACHE_SIZE:
                _TOPOLOGY_CACHE.popitem(last=False)
        _disk_save(cache_key, _TOPOLOGY_CACHE.get(cache_key) or {
            "path_set": self.path_set, "path_statistics": self.path_statistics,
            "compiled_geometry": self.compiled_geometry, "compiled_pairs": self.compiled_pairs,
            "compiled_shells": self.compiled_shells,
        })

        print(
            "[STOCHASTIC PDF] Frozen topology ready: "
            f"P_alt={self.stack_spec.p_alternate:.6f}, "
            f"finite_layers={number_of_layers}, "
            f"ensemble={self.compiled_geometry.ensemble_samples}, "
            f"bulk_layers={self.compiled_pairs.bulk_layers}, "
            f"finite_rows={self.compiled_pairs.finite_pair_rows:,}, "
            f"bulk_rows={self.compiled_pairs.bulk_pair_rows:,}"
        )


    def _generic_transition_statistics(
        self,
        states: np.ndarray,
    ) -> Dict[
        str,
        int,
    ]:
        """
        Return transition counts for an arbitrary N-state Markov path set.
        """
        states = np.asarray(
            states,
            dtype=np.int32,
        )

        output: Dict[
            str,
            int,
        ] = {}

        if (
            states.ndim != 2
            or states.shape[1] <= 1
        ):
            return output

        labels = [
            str(
                state.label
            )
            for state in self.stack_spec.markov_states
        ]

        current_states = states[
            :,
            :-1
        ].reshape(
            -1
        )

        next_states = states[
            :,
            1:
        ].reshape(
            -1
        )

        for current_state, next_state in zip(
            current_states,
            next_states,
        ):
            current_index = int(
                current_state
            )

            next_index = int(
                next_state
            )

            current_label = (
                labels[
                    current_index
                ]
                if (
                    0
                    <= current_index
                    < len(
                        labels
                    )
                )
                else f"S{current_index}"
            )

            next_label = (
                labels[
                    next_index
                ]
                if (
                    0
                    <= next_index
                    < len(
                        labels
                    )
                )
                else f"S{next_index}"
            )

            key = (
                f"{current_label}"
                " -> "
                f"{next_label}"
            )

            output[
                key
            ] = int(
                output.get(
                    key,
                    0,
                )
                + 1
            )

        return output

    
    def _geometry_key(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
    ) -> Tuple[
        float,
        ...,
    ]:
        """
        Parameters that modify stochastic shell distances/topology grouping.
        """
        parameters = dict(
            parameters
            or {}
        )

        lattice = self.source_structure.lattice

        values = [
            float(
                parameters.get(
                    "a",
                    lattice.a,
                )
            ),
            float(
                parameters.get(
                    "b",
                    lattice.b,
                )
            ),
            float(
                parameters.get(
                    "c",
                    lattice.c,
                )
            ),
            float(
                parameters.get(
                    "alpha",
                    lattice.alpha,
                )
            ),
            float(
                parameters.get(
                    "beta",
                    lattice.beta,
                )
            ),
            float(
                parameters.get(
                    "gamma",
                    lattice.gamma,
                )
            ),
        ]

        if self.stack_spec.model == "generic_markov":
            for state_index, state in enumerate(
                self.stack_spec.markov_states
            ):
                state_uid = str(
                    state.uid
                )

                values.extend(
                    [
                        float(
                            parameters.get(
                                f"stack_state_{state_uid}_du",
                                state.du,
                            )
                        ),
                        float(
                            parameters.get(
                                f"stack_state_{state_uid}_dv",
                                state.dv,
                            )
                        ),
                        float(
                            parameters.get(
                                f"stack_state_{state_uid}_h",
                                state.h,
                            )
                        ),
                    ]
                )

        else:
            values.extend(
                [
                    float(
                        parameters.get(
                            "stack_state_a_du",
                            self.stack_spec.state_a_du,
                        )
                    ),
                    float(
                        parameters.get(
                            "stack_state_a_dv",
                            self.stack_spec.state_a_dv,
                        )
                    ),
                    float(
                        parameters.get(
                            "stack_state_a_h",
                            self.stack_spec.state_a_h,
                        )
                    ),
                    float(
                        parameters.get(
                            "stack_state_b_du",
                            self.stack_spec.state_b_du,
                        )
                    ),
                    float(
                        parameters.get(
                            "stack_state_b_dv",
                            self.stack_spec.state_b_dv,
                        )
                    ),
                    float(
                        parameters.get(
                            "stack_state_b_h",
                            self.stack_spec.state_b_h,
                        )
                    ),
                ]
            )        

        return tuple(
            round(
                value,
                10,
            )
            for value in values
        )

    def _apply_geometry(
        self,
        parameters: Mapping[
            str,
            Any,
        ],
        *,
        force: bool = False,
    ) -> None:
        """
        Rebuild the PDF-compatible stochastic ensemble shell table only when
        geometry parameters have changed.
        """
        parameters = dict(
            parameters
            or {}
        )

        geometry_key = self._geometry_key(
            parameters
        )

        if (
            not force
            and geometry_key
            == self._last_geometry_key
        ):
            return

        evaluated = evaluate_stochastic_shell_geometry(
            self.compiled_shells,
            parameters,
        )

        self.pdf_calculator.update_dynamic_stacked_geometry(
            shell_cart=evaluated[
                "shell_cart"
            ],
            shell_frac=evaluated[
                "shell_frac"
            ],
            shell_mult_eff=evaluated[
                "shell_mult_eff"
            ],
            shell_bulk_mult_eff=evaluated[
                "shell_bulk_mult_eff"
            ],
            shell_gamma=evaluated[
                "shell_gamma"
            ],
            shell_active=evaluated[
                "shell_active"
            ],
            stacking_lattice_matrix=evaluated[
                "stacking_lattice_matrix"
            ],
            stacking_inverse_lattice_matrix=evaluated[
                "stacking_inverse_lattice_matrix"
            ],
            stacking_volume=float(
                evaluated[
                    "stacking_volume"
                ]
            ),
            stacking_bulk_number_density=float(
                evaluated[
                    "stacking_bulk_number_density"
                ]
            ),
            stacking_spacing=float(
                evaluated[
                    "stacking_spacing"
                ]
            ),
            stacking_period_height=float(
                evaluated[
                    "stacking_period_height"
                ]
            ),
        )

        self._last_geometry = evaluated
        self._last_parameters = dict(
            parameters
        )
        self._last_geometry_key = geometry_key

        self._geometry_update_count += 1

    @property
    def geometry_update_count(
        self,
    ) -> int:
        return int(
            self._geometry_update_count
        )

    @property
    def current_stacking_distance(
        self,
    ) -> float:
        """
        Effective mean layer-to-layer normal step.

        For generic Markov models this is the empirical/statistical mean
        state-vector height used by the effective stacking lattice.
        """
        if isinstance(
            self._last_geometry,
            dict,
        ):
            return float(
                self._last_geometry.get(
                    "stacking_spacing",
                    0.0,
                )
            )

        if self.stack_spec.model == "correlated_ab_disorder":
            return 0.5 * (
                float(
                    self.stack_spec.state_a_h
                )
                + float(
                    self.stack_spec.state_b_h
                )
            )

        if self.stack_spec.markov_states:
            return float(
                np.mean(
                    [
                        float(
                            state.h
                        )
                        for state in self.stack_spec.markov_states
                    ]
                )
            )

        return 0.0

    @property
    def current_stack_thickness(
        self,
    ) -> float:
        """
        Mean finite-stack thickness over the frozen ensemble.
        """
        try:
            geometries = self.compiled_geometry.evaluate_all(
                self._last_parameters
            )

            values = [
                float(
                    geometry.boundary_thickness
                )
                for geometry in geometries
            ]

            if values:
                return float(
                    np.mean(
                        values
                    )
                )

        except Exception:
            pass

        return 0.0

    
    @property
    def stochastic_metadata(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        metadata = {
            "model": str(
                self.stack_spec.model
            ),
            "ensemble_samples": int(
                self.compiled_geometry.ensemble_samples
            ),
            "ensemble_seed": int(
                self.stack_spec.ensemble_seed
            ),
            "bulk_layers": int(
                self.compiled_pairs.bulk_layers
            ),
            "bulk_realizations": int(
                self.compiled_pairs.bulk_realizations
            ),
            "transition_statistics": dict(
                self.path_statistics
            ),
        }

        if self.stack_spec.model == "correlated_ab_disorder":
            metadata.update(
                {
                    "p_alternate": float(
                        self.stack_spec.p_alternate
                    ),
                    "p_same_state": float(
                        1.0
                        - self.stack_spec.p_alternate
                    ),
                    "ensemble_start_state": str(
                        self.stack_spec.ensemble_start_state
                    ),
                }
            )

        elif self.stack_spec.model == "generic_markov":
            metadata.update(
                {
                    "markov_start_state": int(
                        self.stack_spec.markov_start_state
                    ),
                    "markov_states": [
                        state.to_dict()
                        for state in self.stack_spec.markov_states
                    ],
                    "markov_transition_matrix": [
                        list(
                            row
                        )
                        for row in self.stack_spec.markov_transition_matrix
                    ],
                }
            )

        return metadata

    
    def evaluate(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )
        if self._can_use_cuda(parameters, kwargs):
            try:
                return self._evaluate_cuda(parameters, kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[STOCHASTIC GPU] evaluation failed, using the CPU: {exc}")
                self._cuda_backend = None
        self._apply_geometry(
            parameters
        )
        return self.pdf_calculator.evaluate(
            parameters,
            **kwargs,
        )

    def build_warren_plot(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_warren_plot(
            parameters,
            **kwargs,
        )

    def build_local_trends(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_local_trends(
            parameters,
            **kwargs,
        )

    def compute_isotropic_shape_factor(
        self,
        parameters,
        r_grid,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.compute_isotropic_shape_factor(
            parameters,
            r_grid,
        )

    def build_shape_factor_directions(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_shape_factor_directions(
            parameters,
            **kwargs,
        )

    def build_pdf_tick_data(
        self,
        parameters,
        **kwargs,
    ):
        parameters = dict(
            parameters
            or {}
        )

        self._apply_geometry(
            parameters
        )

        return self.pdf_calculator.build_pdf_tick_data(
            parameters,
            **kwargs,
        )