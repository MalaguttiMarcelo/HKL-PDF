from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pdf_fitting.models.crystallite_shapes import (
    CrystalliteShapeSpec,
)

from pdf_fitting.models.finite_shape_shells import (
    FINITE_CACHE_DIR,
    _hash_shape,
    build_cell_mask_for_shape,
    build_shell_table_from_finite_counts,
    count_finite_coordination_cpu,
    get_or_build_all_site_pair_catalog,
)

from pdf_fitting.models.layer_stacking import (
    LayerStackSpec,
)


@dataclass
class FiniteShapeBuildResult:
    """
    Result of finite crystallite construction.

    shell_table:
        Host NumPy table compatible with PDFCalculator.

    device_shell_table:
        Reserved for the next stage, where grouped finite shell arrays remain
        on GPU. The current stage keeps it None because final grouping is
        intentionally performed by the validated CPU code.

    backend:
        Actual finite-shape construction backend:
            "cpu"
            "cuda"
            "cache"

    cache_hit:
        True when a saved finite shell table was loaded.

    build_seconds:
        Wall-clock construction/load time.
    """

    shell_table: Dict[str, Any]
    device_shell_table: Optional[Any]
    backend: str
    cache_hit: bool
    build_seconds: float


def _load_finite_shell_cache(
    *,
    npz_path: str,
    meta_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Load a finite shell table in the existing cache format.
    """
    if (
        not os.path.exists(
            npz_path
        )
        or not os.path.exists(
            meta_path
        )
    ):
        return None

    try:
        import json

        with open(
            meta_path,
            "r",
            encoding="utf-8",
        ) as file:
            meta = json.load(
                file
            )

        z = np.load(
            npz_path,
            allow_pickle=False,
        )

        return {
            "shell_frac": z["shell_frac"],
            "shell_mult": z["shell_mult"],
            "shell_bulk_mult": (
                z["shell_bulk_mult"]
                if "shell_bulk_mult" in z.files
                else z["shell_mult"]
            ),
            "shell_mult_eff": z["shell_mult_eff"],
            "shell_finite_mult_eff": (
                z["shell_finite_mult_eff"]
                if "shell_finite_mult_eff" in z.files
                else z["shell_mult_eff"]
            ),
            "shell_bulk_mult_eff": (
                z["shell_bulk_mult_eff"]
                if "shell_bulk_mult_eff" in z.files
                else z["shell_mult_eff"]
            ),
            "shell_gamma": (
                z["shell_gamma"]
                if "shell_gamma" in z.files
                else np.ones_like(
                    z["shell_mult_eff"],
                    dtype=np.float32,
                )
            ),
            "shell_pair_id": z["shell_pair_id"],
            "shell_alpha_id": z["shell_alpha_id"],
            "shell_beta_id": z["shell_beta_id"],
            "shell_hkl": z["shell_hkl"],
            "shell_direction": z["shell_direction"],
            "shell_k_rank": z["shell_k_rank"],
            "shell_upair_id": z["shell_upair_id"],
            "shell_k_within_upair": z["shell_k_within_upair"],
            "pair_offsets": z["pair_offsets"],
            "pair_alpha_id": z["pair_alpha_id"],
            "pair_beta_id": z["pair_beta_id"],
            "strain_unique_dirs": z["strain_unique_dirs"],
            "strain_inv_dirs": z["strain_inv_dirs"],
            "unique_species": list(
                meta["unique_species"]
            ),
            "pair_list": [
                tuple(item)
                for item in meta["pair_list"]
            ],
            "upair_list": [
                tuple(item)
                for item in meta["upair_list"]
            ],
        }

    except Exception:
        return None


def _save_finite_shell_cache(
    *,
    shell_table: Dict[str, Any],
    shape_spec: CrystalliteShapeSpec,
    pair_cutoff: float,
    boundary_mode: str,
    npz_path: str,
    meta_path: str,
) -> None:
    """
    Save the same portable NumPy finite-shell cache format used by the CPU path.
    """
    import json

    np.savez(
        npz_path,
        shell_frac=shell_table["shell_frac"],
        shell_mult=shell_table["shell_mult"],
        shell_bulk_mult=shell_table["shell_bulk_mult"],
        shell_mult_eff=shell_table["shell_mult_eff"],
        shell_finite_mult_eff=shell_table[
            "shell_finite_mult_eff"
        ],
        shell_bulk_mult_eff=shell_table[
            "shell_bulk_mult_eff"
        ],
        shell_gamma=shell_table["shell_gamma"],
        shell_pair_id=shell_table["shell_pair_id"],
        shell_alpha_id=shell_table["shell_alpha_id"],
        shell_beta_id=shell_table["shell_beta_id"],
        shell_hkl=shell_table["shell_hkl"],
        shell_direction=shell_table["shell_direction"],
        shell_k_rank=shell_table["shell_k_rank"],
        shell_upair_id=shell_table["shell_upair_id"],
        shell_k_within_upair=shell_table[
            "shell_k_within_upair"
        ],
        pair_offsets=shell_table["pair_offsets"],
        pair_alpha_id=shell_table["pair_alpha_id"],
        pair_beta_id=shell_table["pair_beta_id"],
        strain_unique_dirs=shell_table[
            "strain_unique_dirs"
        ],
        strain_inv_dirs=shell_table[
            "strain_inv_dirs"
        ],
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "unique_species": shell_table[
                    "unique_species"
                ],
                "pair_list": [
                    list(item)
                    for item in shell_table[
                        "pair_list"
                    ]
                ],
                "upair_list": [
                    list(item)
                    for item in shell_table[
                        "upair_list"
                    ]
                ],
                "shape_spec": shape_spec.to_dict(),
                "pair_cutoff": float(
                    pair_cutoff
                ),
                "boundary_mode": str(
                    boundary_mode
                ),
            },
            file,
            indent=2,
        )


def get_or_build_finite_shape_result(
    *,
    structure,
    pair_cutoff: float,
    shape_spec: CrystalliteShapeSpec,
    stack_spec: Optional[LayerStackSpec] = None,
    refinement_config: Optional[Dict[str, Any]] = None,
    boundary_mode: str = "atom",
) -> FiniteShapeBuildResult:
    """
    Build/load one finite atomistic shell table.

    Backend choice follows [refinement]:

        shell_backend = cpu
        shell_backend = cuda
        shell_backend = auto

    Current CUDA stage:
        GPU atom mask
        GPU direct finite coordination counting
        CPU final shell grouping/caching

    The final host shell-table format remains exactly compatible with
    PDFCalculator.replace_shell_table().
    """
    from pdf_fitting.models.gpu.backend import (
        resolve_backend,
    )

    refinement_config = dict(
        refinement_config
        or {}
    )

    requested_backend = str(
        refinement_config.get(
            "shell_backend",
            "cpu",
        )
    ).strip().lower()

    actual_backend = resolve_backend(
        requested_backend,
        default="cpu",
    )

    gpu_device = int(
        refinement_config.get(
            "gpu_device",
            0,
        )
    )

    gpu_memory_fraction = float(
        refinement_config.get(
            "gpu_memory_fraction",
            0.70,
        )
    )

    boundary_mode = str(
        boundary_mode
        or "atom"
    ).strip().lower()

    if boundary_mode != "atom":
        raise NotImplementedError(
            "Finite-shape backend dispatcher currently supports "
            "boundary_mode='atom'."
        )

    t0 = time.perf_counter()

    if (
        stack_spec is not None
        and bool(
            stack_spec.enabled
        )
    ):
        from pdf_fitting.models.stacked_finite_shells import (
            build_stacked_finite_shell_table,
        )

        print(
            "[FINITE BACKEND] Periodic stacking is enabled. "
            "Using the fixed-geometry CPU stacked-shell backend."
        )

        stacked_result = build_stacked_finite_shell_table(
            structure=structure,
            pair_cutoff=float(
                pair_cutoff
            ),
            shape_spec=shape_spec,
            stack_spec=stack_spec,
            analyze_supercell_symmetry=True,
        )

        return FiniteShapeBuildResult(
            shell_table=stacked_result.shell_table,
            device_shell_table=None,
            backend="stacked_cpu",
            cache_hit=False,
            build_seconds=float(
                stacked_result.build_seconds
            ),
        )
    
    os.makedirs(
        FINITE_CACHE_DIR,
        exist_ok=True,
    )

    catalog = get_or_build_all_site_pair_catalog(
        structure,
        float(
            pair_cutoff
        ),
        cache_dir=FINITE_CACHE_DIR,
    )

    shape_key = _hash_shape(
        shape_spec,
        extra={
            "catalog_hash": catalog[
                "catalog_hash"
            ],
            "pair_cutoff": float(
                pair_cutoff
            ),
            "boundary_mode": boundary_mode,
        },
    )

    npz_path = os.path.join(
        FINITE_CACHE_DIR,
        f"finite_shape_shells_{shape_key}.npz",
    )

    meta_path = os.path.join(
        FINITE_CACHE_DIR,
        f"finite_shape_shells_{shape_key}.json",
    )

    cached = _load_finite_shell_cache(
        npz_path=npz_path,
        meta_path=meta_path,
    )

    if cached is not None:
        print(
            "[FINITE CACHE] Loaded finite shell table: "
            f"{npz_path}"
        )

        return FiniteShapeBuildResult(
            shell_table=cached,
            device_shell_table=None,
            backend="cache",
            cache_hit=True,
            build_seconds=float(
                time.perf_counter()
                - t0
            ),
        )

    print(
        "[FINITE BACKEND] Building finite shell table with "
        f"backend={actual_backend}, "
        f"shape={shape_spec.shape_type}, "
        f"axis1={shape_spec.axis1_cells}, "
        f"axis2={shape_spec.axis2_cells}, "
        f"axis3={shape_spec.axis3_cells}"
    )

    if actual_backend == "cuda":
        from pdf_fitting.models.gpu.finite_shape import (
            CudaFiniteShapeBackend,
        )

        cuda_backend = CudaFiniteShapeBackend(
            device_id=gpu_device,
            memory_fraction=gpu_memory_fraction,
        )

        mask_data, counts = cuda_backend.build_mask_and_counts(
            structure=structure,
            shape_spec=shape_spec,
            catalog=catalog,
        )

        backend_label = "cuda"

    else:
        mask_data = build_cell_mask_for_shape(
            structure,
            shape_spec,
            boundary_mode=boundary_mode,
        )

        counts = count_finite_coordination_cpu(
            catalog,
            mask_data,
        )

        backend_label = "cpu"

    shell_table = build_shell_table_from_finite_counts(
        catalog,
        counts,
        mask_data,
    )

    _save_finite_shell_cache(
        shell_table=shell_table,
        shape_spec=shape_spec,
        pair_cutoff=float(
            pair_cutoff
        ),
        boundary_mode=boundary_mode,
        npz_path=npz_path,
        meta_path=meta_path,
    )

    elapsed = float(
        time.perf_counter()
        - t0
    )

    print(
        "[FINITE BACKEND] Built finite shell table: "
        f"backend={backend_label}, "
        f"shells={len(shell_table['shell_frac']):,}, "
        f"time={elapsed:.3f} s"
    )

    return FiniteShapeBuildResult(
        shell_table=shell_table,
        device_shell_table=None,
        backend=backend_label,
        cache_hit=False,
        build_seconds=elapsed,
    )