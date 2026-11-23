"""
GPU PDF for stacked models with arbitrary transition coefficients
(stochastic stacking ensembles and generic dynamic transitions).

Pair vectors are linear in the stacking vectors of the N states/transitions:

    r = Δfrac·L + (Δa + C·du)·â + (Δb + C·dv)·b̂ + (C·h)·n̂

(Δfrac, Δa, Δb fixed per shell; C (n_shell × N) transition counts; du, dv, h
the state vectors; â, b̂, n̂ the layer basis of the parent lattice L). Only
the geometry step differs from DynamicStackedCudaPdfBackend; correlation,
σ, amplitudes, Gaussian accumulation, finite-size γ and the SAXS terms are
inherited. Multiplicities are the topology's already-normalised values.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .dynamic_stacked_pdf import DynamicStackedCudaPdfBackend

# Σ_k w_k γ_cyl(r, D_k) per shell (cylinder/disk common volume, lognormal
# thickness; same expressions as numba_kernels.cylinder_gamma_lognormal).
_CYLINDER_GAMMA_SOURCE = r"""
extern "C" __global__
void cylinder_gamma_lognormal(
    const float* cart, const float* dist, const unsigned char* active, const int n,
    const float cx, const float cy, const float cz,
    const float* d_grid, const float* w_grid, const int nd,
    const float mu, const float sigma, const float t_mean, float* out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float L = dist[i];
    if (!active[i] || !(L > 0.0f) || !isfinite(L)) { out[i] = 0.0f; return; }
    float z = fabsf(cart[3*i] * cx + cart[3*i+1] * cy + cart[3*i+2] * cz);
    float rp = sqrtf(fmaxf(L * L - z * z, 0.0f));
    const float PI = 3.14159265358979f;
    bool poly = sigma > 1e-12f;
    float s2 = sigma * sigma;
    float tb = 0.0f, axial = 0.0f, pref_t = 0.0f;
    if (poly) {
        float e_mu = expf(mu + 0.5f * s2);
        pref_t = expf(-mu - 0.5f * s2) / PI;
        float sq = 1.41421356f * sigma;
        if (z <= 1e-30f) {
            tb = 2.0f * e_mu;
        } else {
            float lnz = logf(z);
            tb = e_mu * (1.0f + erff((mu + s2 - lnz) / sq)) + z * (erfcf((mu - lnz) / sq) - 2.0f);
        }
    } else {
        axial = 1.0f - z / t_mean;
        if (z >= t_mean || !isfinite(axial)) axial = 0.0f;
        axial = fminf(fmaxf(axial, 0.0f), 1.0f);
    }
    float acc = 0.0f;
    for (int k = 0; k < nd; ++k) {
        float D = d_grid[k];
        if (!(rp < D)) continue;
        float x = fminf(fmaxf(rp / D, 0.0f), 1.0f);
        float root = sqrtf(fmaxf(1.0f - x * x, 0.0f));
        float g;
        if (poly) {
            float rb = fmaxf(D * acosf(x) - rp * root, 0.0f);
            g = pref_t / D * tb * rb;
        } else {
            g = (2.0f / PI) * (acosf(x) - x * root) * axial;
        }
        if (!isfinite(g)) g = 0.0f;
        acc += w_grid[k] * fminf(fmaxf(g, 0.0f), 1.0f);
    }
    out[i] = acc;
}
"""


class GenericStackedCudaPdfBackend(DynamicStackedCudaPdfBackend):
    def __init__(
        self,
        *,
        delta_fractional: np.ndarray,
        delta_a: np.ndarray,
        delta_b: np.ndarray,
        transition_coefficients: np.ndarray,
        shell_pair_id: np.ndarray,
        shell_alpha_id: np.ndarray,
        shell_beta_id: np.ndarray,
        shell_upair_id: np.ndarray,
        shell_k_rank: np.ndarray,
        shell_k_within_upair: np.ndarray,
        shell_mult_eff: np.ndarray,
        shell_bulk_mult_eff: np.ndarray,
        species_count: int,
        pair_weights: np.ndarray,
        pair_offsets: np.ndarray,
        pair_cutoff: float,
        device_id: int = 0,
    ) -> None:
        n = int(np.asarray(delta_a).size)
        ones = np.ones(int(species_count), dtype=np.float32)
        super().__init__(
            shell_delta_fractional=np.asarray(delta_fractional, dtype=np.float32).reshape(n, 3),
            shell_delta_a=np.asarray(delta_a, dtype=np.float32),
            shell_delta_b=np.asarray(delta_b, dtype=np.float32),
            shell_delta_s=np.zeros(n, dtype=np.float32),
            shell_delta_normal=np.zeros(n, dtype=np.float32),
            shell_pair_id=shell_pair_id,
            shell_alpha_id=shell_alpha_id,
            shell_beta_id=shell_beta_id,
            shell_upair_id=shell_upair_id,
            shell_k_rank=shell_k_rank,
            shell_k_within_upair=shell_k_within_upair,
            # Normalised multiplicities: divide by 1 in the parent formulas.
            shell_mult=np.asarray(shell_mult_eff, dtype=np.float32),
            shell_bulk_mult=np.asarray(shell_bulk_mult_eff, dtype=np.float32),
            finite_species_counts=ones,
            bulk_species_counts=ones,
            pair_weights=pair_weights,
            pair_offsets=pair_offsets,
            pair_cutoff=pair_cutoff,
            device_id=device_id,
        )
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self.d_transition_coefficients = cp.asarray(
                np.ascontiguousarray(transition_coefficients, dtype=np.float32).reshape(n, -1)
            )
        self.state_count = int(self.d_transition_coefficients.shape[1])
        self._state = None
        self._strain = None
        self.d_strain_inv = None

    # ------------------------------------------------------------------
    # Microstrain (Wilkens / PAH) on the device
    # ------------------------------------------------------------------
    def set_strain_directions(self, inv_dirs) -> None:
        """Per-shell index into the unique-direction contrast-factor table."""
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self.d_strain_inv = cp.asarray(np.ascontiguousarray(inv_dirs, dtype=np.int32))

    def set_strain(self, mode=None, chkl_unique=None, *, Re=100.0, pref=0.0, pah_a=0.0, pah_b=0.0) -> None:
        """mode None | "wilkens" | "pah"; chkl_unique: C_hkl per unique direction."""
        if mode is None or chkl_unique is None or self.d_strain_inv is None:
            self._strain = None
            return
        self._strain = (str(mode), np.asarray(chkl_unique, dtype=np.float32), float(Re), float(pref), float(pah_a), float(pah_b))

    def _strain_L2(self):
        cp = self.cp
        mode, chk, Re, pref, pa, pb = self._strain
        r = self.d_shell_distance
        c = cp.asarray(chk)[self.d_strain_inv]
        if mode == "wilkens":
            if not hasattr(self, "_fstar_kernel"):
                self._fstar_kernel = cp.ElementwiseKernel(
                    "float32 L, float32 Re", "float32 f",
                    """
                    float x = L / Re;
                    if (x <= 1.0f) {
                        float xm = fmaxf(x, 1e-30f);
                        f = -logf(xm) + 1.75f - 0.69314718f + xm * xm / 6.0f - 32.0f * xm * xm * xm / (225.0f * 3.14159265f);
                    } else {
                        f = 512.0f / (90.0f * 3.14159265f * x) - (11.0f / 24.0f + 0.25f * logf(2.0f * x)) / (x * x);
                    }
                    """,
                    "wilkens_fstar",
                )
            return r * r * cp.float32(pref) * c * self._fstar_kernel(r, cp.float32(Re))
        return cp.maximum(c * (cp.float32(pa) * r + cp.float32(pb) * r * r), cp.float32(0.0))

    def _prepare_sigma_and_amplitude(self, **kwargs) -> None:
        super()._prepare_sigma_and_amplitude(**kwargs)
        if self._strain is not None:
            cp = self.cp
            s2 = self.d_sigma * self.d_sigma + self._strain_L2().astype(cp.float32)
            self.d_sigma = cp.sqrt(cp.maximum(s2, cp.float32(1.0e-24))).astype(cp.float32, copy=False)

    def set_state_vectors(self, state_du, state_dv, state_h) -> None:
        du = np.asarray(state_du, dtype=np.float32).reshape(-1)
        dv = np.asarray(state_dv, dtype=np.float32).reshape(-1)
        h = np.asarray(state_h, dtype=np.float32).reshape(-1)
        if not (du.size == dv.size == h.size == self.state_count):
            raise ValueError(f"expected {self.state_count} state vectors, got {du.size}/{dv.size}/{h.size}")
        self._state = (du, dv, h)

    def update_pair_weights(self, pair_weights) -> None:
        """Faber–Ziman weights change with refined occupancies."""
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self.d_pair_weights = cp.asarray(np.ascontiguousarray(pair_weights, dtype=np.float32))

    def _evaluate_geometry(self, *, lattice_matrix, vector_a, vector_b, stack_normal, spacing) -> None:
        if self._state is None:
            raise RuntimeError("set_state_vectors() must be called before evaluate().")
        cp = self.cp
        f32 = cp.float32
        du, dv, h = (cp.asarray(x) for x in self._state)
        C = self.d_transition_coefficients
        eff_a = self.d_shell_delta_a + C @ du
        eff_b = self.d_shell_delta_b + C @ dv
        eff_n = C @ h
        L = cp.asarray(np.ascontiguousarray(lattice_matrix, dtype=np.float32))
        va = cp.asarray(np.ascontiguousarray(vector_a, dtype=np.float32))
        vb = cp.asarray(np.ascontiguousarray(vector_b, dtype=np.float32))
        vn = cp.asarray(np.ascontiguousarray(stack_normal, dtype=np.float32))
        self.d_shell_cart = (
            self.d_shell_delta_fractional @ L
            + eff_a[:, None] * va[None, :]
            + eff_b[:, None] * vb[None, :]
            + eff_n[:, None] * vn[None, :]
        ).astype(f32, copy=False)
        self.d_shell_distance = cp.sqrt(cp.sum(self.d_shell_cart * self.d_shell_cart, axis=1)).astype(f32, copy=False)
        self.d_shell_active = (
            cp.isfinite(self.d_shell_distance)
            & (self.d_shell_distance > f32(1.0e-8))
            & (self.d_shell_distance <= f32(self.pair_cutoff + 1.0e-10))
        )
        zero = f32(0.0)
        self.d_shell_mult_eff = cp.where(self.d_shell_active, self.d_shell_mult, zero).astype(f32, copy=False)
        self.d_shell_bulk_mult_eff = cp.where(self.d_shell_active, self.d_shell_bulk_mult, zero).astype(f32, copy=False)
        self.d_shell_gamma = cp.where(
            self.d_shell_bulk_mult_eff > zero,
            self.d_shell_mult_eff / cp.maximum(self.d_shell_bulk_mult_eff, f32(1.0e-30)),
            zero,
        ).astype(f32, copy=False)

    def _saxs_cylinder_gamma(self, *, diameter, diameter_std, height, height_std, axis_cart):
        """
        One fused kernel for the diameter-averaged shell γ, then a single
        r-grid histogram (the histogram is linear in γ).
        """
        cp = self.cp
        diameter, diameter_std = float(diameter), float(diameter_std)
        height, height_std = float(height), float(height_std)
        if diameter <= 0.0 or height <= 0.0:
            return None
        if height_std > 1e-12:
            s2 = np.log(1.0 + (height_std * height_std) / (height * height))
            sigma, mu = float(np.sqrt(s2)), float(np.log(height) - 0.5 * s2)
        else:
            sigma, mu = 0.0, float(np.log(height))
        if diameter_std <= 1e-12:
            d_grid = np.asarray([diameter], dtype=np.float64)
            w_grid = np.ones(1)
        else:
            s2d = np.log(1.0 + (diameter_std * diameter_std) / (diameter * diameter))
            sd, md = float(np.sqrt(s2d)), float(np.log(diameter) - 0.5 * s2d)
            d_min = max(1.0e-6, diameter * 0.05)
            d_max = diameter + 4.0 * diameter_std
            if d_max <= d_min:
                d_max = diameter * 3.0
            d_grid = np.linspace(d_min, d_max, 15)
            pdf = 1.0 / (d_grid * sd * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((np.log(d_grid) - md) / sd) ** 2)
            w_grid = pdf / max(float(np.sum(pdf)), 1.0e-30)
        axis = np.asarray(axis_cart, dtype=np.float64)
        nrm = float(np.linalg.norm(axis))
        axis = axis / nrm if nrm > 1e-12 else np.asarray([0.0, 0.0, 1.0])
        if not hasattr(self, "_cyl_kernel"):
            self._cyl_kernel = cp.RawKernel(_CYLINDER_GAMMA_SOURCE, "cylinder_gamma_lognormal")
        n = int(self.shell_count)
        shell_gamma = cp.empty(n, dtype=cp.float32)
        threads = 256
        self._cyl_kernel(
            ((n + threads - 1) // threads,), (threads,),
            (
                cp.ascontiguousarray(self.d_shell_cart, dtype=cp.float32),
                cp.ascontiguousarray(self.d_shell_distance, dtype=cp.float32),
                self.d_shell_active.astype(cp.uint8),
                np.int32(n),
                np.float32(axis[0]), np.float32(axis[1]), np.float32(axis[2]),
                cp.asarray(d_grid, dtype=cp.float32), cp.asarray(w_grid, dtype=cp.float32), np.int32(d_grid.size),
                np.float32(mu), np.float32(sigma), np.float32(height),
                shell_gamma,
            ),
        )
        gamma_by_pair = self._shell_gamma_to_pair_grid(shell_gamma, self.d_shell_bulk_mult_eff)
        return self._weighted_pair_gamma(gamma_by_pair)

    def memory_bytes(self) -> int:
        total = 0
        for v in vars(self).values():
            if isinstance(v, self.cp.ndarray):
                total += int(v.nbytes)
        return total


def configure_strain_from_calculator(backend, calc, params, lattice_a: float) -> None:
    """
    Same activation rules and values as PDFCalculator.evaluate (Wilkens /
    PAH, 'auto', no contrast coefficients → off), applied to the backend.
    """
    refcfg = (calc.config.get("refinement") or {}) if isinstance(calc.config, dict) else {}
    model = str(params.get("microstrain_model", refcfg.get("microstrain_model", "auto"))).strip().lower()
    rho = float(params.get("rho", 0.0))
    Re = float(params.get("Re", params.get("re", 100.0)))
    fE = float(params.get("fE", params.get("fe", 0.5)))
    pah_a = float(params.get("pah_a", params.get("PAH_a", 0.0)))
    pah_b = float(params.get("pah_b", params.get("PAH_b", 0.0)))
    if model in ("wilkens", "wilkins", "dislocation"):
        wilkens, pah = rho > 0.0 and Re > 0.0, False
    elif model in ("pah", "adler-houska", "adler_houska"):
        wilkens, pah = False, (pah_a != 0.0 or pah_b != 0.0)
    elif model in ("isotropic", "iso", "delta_g", "none", "off"):
        wilkens, pah = False, False
    else:
        wilkens = rho > 0.0 and Re > 0.0
        pah = (not wilkens) and (pah_a != 0.0 or pah_b != 0.0)
    has_contrast = (
        calc._has_cubic_cf_params(params)
        or calc._has_cf_params_or_config(params, "edge")
        or calc._has_cf_params_or_config(params, "screw")
    )
    inv = getattr(calc, "_strain_inv_dirs", None)
    if not (wilkens or pah) or not has_contrast or inv is None:
        backend.set_strain(None)
        return
    if backend.d_strain_inv is None:
        backend.set_strain_directions(inv)
    chk = calc._compute_chkl_unique_dirs_fast(params, fE)
    if wilkens:
        lc = calc._params_lc(params)
        bmag = calc._pfloat(lc, ("burgers_mag", "burgers", "b_mag", "b"), 0.0)
        if bmag <= 0.0:
            bmag = np.sqrt(3.0) / 2.0 * float(lattice_a)
        backend.set_strain("wilkens", chk, Re=Re, pref=float(rho) * float(bmag) ** 2 / (4.0 * np.pi))
    else:
        backend.set_strain("pah", chk, pah_a=pah_a, pah_b=pah_b)
