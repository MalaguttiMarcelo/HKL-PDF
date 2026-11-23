"""
Plots of the direction-resolved width analysis (warren_fit.WarrenResult):

* overview      σ²_sample(r) and static σ² vs r² of every direction
* direction(s)  one row per direction: σ²_sample(r) with dynamic / static
                parts, and the static part vs r² (Williamson–Hall-like)
* contrast      per-direction static coefficient S_d = ρb²C_d/4π against the
                cubic invariant H (or the angle to c), with the invariant
                expansion and the model contrast factors (DISCO-like)

σ²_sample = σ²_obs − δ_broad²r²: the instrumental part is removed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

FAMILY_COLORS = ["#1d4ed8", "#c2410c", "#15803d", "#7c3aed", "#b91c1c", "#0e7490", "#a16207", "#be185d", "#4b5563", "#65a30d"]
OTHER_COLOR = "#9ca3af"
FG = "#111827"


def family_colors(wr) -> Dict[str, str]:
    own = [f for f in wr.family_order() if wr.family_info[f].own]
    cmap = {f: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, f in enumerate(own)}
    for f in wr.family_info:
        cmap.setdefault(f, OTHER_COLOR)
    return cmap


def family_title(wr, fam: str) -> str:
    from pdf_fitting.project.warren_fit import hkl_label

    info = wr.family_info[fam]
    return f"{fam} ∥ {hkl_label(info.hkl)}"


def _empty(fig, text):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#6b7280")
    ax.set_axis_off()


def _instr_note(wr) -> str:
    db = float(wr.start.get("delta_broad", 0.0) or 0.0)
    return f"instrument removed (δ_broad = {db:.4g} Å⁻¹)" if db else "no instrumental broadening (δ_broad = 0)"


def _sorted(idx: Sequence[int], wr) -> List[int]:
    return sorted(idx, key=lambda i: wr.r[i])


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------


def plot_overview(fig, wr) -> None:
    if wr is None:
        _empty(fig, "Fit the widths after the free-peak fit")
        return
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)
    cmap = family_colors(wr)
    obs = wr.observed_sample
    stat_obs = obs - wr.thermal
    groups: Dict[str, List[int]] = {}
    for i, fam in enumerate(wr.families):
        if fam == "merged":
            groups.setdefault("merged", []).append(i)
        elif wr.family_info[fam].own:
            groups.setdefault(fam, []).append(i)
        else:
            groups.setdefault("other", []).append(i)
    order = [f for f in wr.family_order() if f in groups] + [g for g in ("other", "merged") if g in groups]
    for g in order:
        idx = _sorted(groups[g], wr)
        if g == "merged":
            color, label, mfc = OTHER_COLOR, "merged peaks", "none"
        elif g == "other":
            color, label, mfc = OTHER_COLOR, "other directions", OTHER_COLOR
        else:
            color, label, mfc = cmap[g], family_title(wr, g), cmap[g]
        connect = g not in ("merged", "other")
        ls = "-" if connect else "none"
        ax1.errorbar(wr.r[idx], obs[idx], yerr=wr.observed_esd[idx], fmt="o", ms=4, color=color, mfc=mfc, ls=ls, lw=0.8, capsize=2, label=label)
        ax2.errorbar(wr.r[idx] ** 2, stat_obs[idx], yerr=wr.observed_esd[idx], fmt="o", ms=4, color=color, mfc=mfc, ls=ls, lw=0.8, capsize=2)
        if connect:
            ax1.plot(wr.r[idx], wr.predicted_sample[idx], "--", color=color, lw=0.9, alpha=0.8)
            ax2.plot(wr.r[idx] ** 2, wr.static[idx], "--", color=color, lw=0.9, alpha=0.8)
    o = np.argsort(wr.r)
    ax1.plot(wr.r[o], wr.thermal[o], ":", color=FG, lw=1.0, label="dynamic part (model)")
    ax1.set_xlabel("r (Å)")
    ax1.set_ylabel("σ²_sample (Å²)")
    ax1.set_title(_instr_note(wr), fontsize=9)
    ax1.legend(fontsize=7, ncols=2, loc="best", framealpha=0.6)
    ax2.axhline(0, color="#9ca3af", lw=0.6)
    ax2.set_xlabel("r² (Å²)")
    ax2.set_ylabel("σ²_sample − dynamic = static (Å²)")
    ax2.set_title("solid: observed · dashed: model", fontsize=9)


# -----------------------------------------------------------------------------
# One or more directions
# -----------------------------------------------------------------------------


def plot_directions(fig, wr, fams: Sequence[str]) -> None:
    fams = [f for f in fams if wr is not None and f in wr.family_info]
    if wr is None or not fams:
        _empty(fig, "Select a direction")
        return
    fig.clear()
    axes = fig.subplots(len(fams), 2, squeeze=False)
    cmap = family_colors(wr)
    obs = wr.observed_sample
    for row, fam in enumerate(fams):
        info = wr.family_info[fam]
        ax1, ax2 = axes[row]
        color = cmap[fam]
        idx = _sorted(info.clusters, wr)
        pure = [i for i in idx if not wr.merged[i] or wr.cluster_families[i] == [fam]]
        mixed = [i for i in idx if i not in pure]
        r = wr.r[idx]
        # Smooth curves along the direction.
        rr = np.linspace(max(0.5, float(r.min()) * 0.8), float(r.max()) * 1.1, 200)
        pairs = sorted({m["pair"] for i in idx for m in wr.members[i] if m["family"] == fam}) or ["-".join(["x", "x"])]
        try:
            dyn = wr.dynamic_curve(fam, pairs[0], rr)
            st = wr.static_curve(fam, rr)
            ax1.plot(rr, dyn, ":", color="#6b7280", lw=1.0, label="dynamic (δ1/r + δ2/r² law)")
            ax1.plot(rr, st, "-.", color=color, lw=1.0, alpha=0.7, label="static")
            ax2.plot(rr ** 2, st, "-", color=color, lw=1.2, alpha=0.8, label="static model")
        except Exception:  # noqa: BLE001
            pass
        ax1.errorbar(wr.r[idx], obs[idx], yerr=wr.observed_esd[idx], fmt="none", ecolor=color, capsize=2, lw=0.8)
        ax1.plot(wr.r[idx], obs[idx], "-", color=color, lw=0.8, alpha=0.6)
        ax1.plot(wr.r[pure], obs[pure], "o", color=color, ms=5, label="observed")
        if mixed:
            ax1.plot(wr.r[mixed], obs[mixed], "o", color=color, mfc="none", ms=6, label="merged with other directions")
        ax1.plot(wr.r[idx], wr.predicted_sample[idx], "x-", color=FG, ms=6, lw=1.0, label="model (dynamic + static)")
        for i in idx:
            ks = sorted({m["k"] for m in wr.members[i] if m["family"] == fam})
            ax1.annotate(f"k={','.join(map(str, ks))}", (wr.r[i], obs[i]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#6b7280")
        ax1.set_xlabel("r (Å)")
        ax1.set_ylabel("σ²_sample (Å²)")
        ttl = f"{family_title(wr, fam)} · {len(idx)} peak(s)"
        if not info.own:
            ttl += " · shares the 'other' parameters"
        ax1.set_title(ttl, fontsize=9)
        if row == 0:
            ax1.legend(fontsize=7, loc="best", framealpha=0.6)

        stat_obs = obs[idx] - wr.thermal[idx]
        ax2.errorbar(r ** 2, stat_obs, yerr=wr.observed_esd[idx], fmt="o", ms=5, color=color, ls="-", lw=0.8, capsize=2, label="observed − dynamic")
        if mixed:
            ax2.plot(wr.r[mixed] ** 2, (obs - wr.thermal)[mixed], "o", mfc="none", ms=8, color=color)
        ax2.axhline(0, color="#9ca3af", lw=0.6)
        ax2.set_xlabel("r² (Å²)")
        ax2.set_ylabel("static σ² (Å²)")
        lines = []
        if np.isfinite(info.S):
            lines.append(f"S = {info.S:.3g} ± {info.S_esd:.2g}")
        if np.isfinite(info.rho):
            lines.append(f"ρ (model C, fE) = {info.rho:.3g} ± {info.rho_esd:.2g} Å⁻²")
        key = fam if info.own else "other"
        for n, v, e in wr.table():
            if n.endswith(f" {key}") and not n.startswith("S "):
                lines.append(f"{n.rsplit(' ', 1)[0]} = {v:.4g} ± {e:.2g}")
        if lines:
            ax2.text(0.02, 0.97, "\n".join(lines), transform=ax2.transAxes, va="top", fontsize=8,
                     bbox=dict(boxstyle="round", fc="white", ec="#d1d5db", alpha=0.85))
        if row == 0:
            ax2.legend(fontsize=7, loc="lower right")


# -----------------------------------------------------------------------------
# Contrast factors
# -----------------------------------------------------------------------------


def plot_contrast(fig, wr) -> None:
    if wr is None or wr.options.static_model == "isotropic":
        _empty(fig, "Contrast factors need a direction-dependent static model")
        return
    from pdf_fitting.project.warren_fit import hkl_label

    fig.clear()
    ax = fig.add_subplot(111)
    cmap = family_colors(wr)
    fams = [f for f in wr.family_order()]
    axis = wr.cf_axis
    xs: Dict[str, float] = {}
    if axis == "index" or not all(np.isfinite(wr.family_info[f].x_cf) for f in fams):
        axis = "index"
        for k, f in enumerate(fams):
            xs[f] = float(k)
    else:
        for f in fams:
            xs[f] = wr.family_info[f].x_cf
    independent = wr.options.static_model == "direction"
    own = [f for f in fams if wr.family_info[f].own]
    if independent:
        for f in own:
            i = wr.family_info[f]
            ax.errorbar(xs[f], i.S, yerr=i.S_esd, fmt="o", ms=7, color=cmap[f], capsize=3)
            ax.annotate(f"{f}", (xs[f], i.S), textcoords="offset points", xytext=(6, 4), fontsize=8, color=cmap[f])
    else:
        for f in fams:
            i = wr.family_info[f]
            ax.errorbar(xs[f], i.S, yerr=i.S_esd, fmt="s", ms=4, color=FG, mfc="white", capsize=2)
            ax.annotate(f"{f}", (xs[f], i.S), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#6b7280")
    ordered = sorted(fams, key=lambda f: xs[f])
    xv = np.array([xs[f] for f in ordered])
    lab_inv = "invariant expansion (" + ("joint fit" if wr.inv_source.startswith("joint") else "least squares") + ")"
    S_inv = np.array([wr.family_info[f].S_inv for f in ordered])
    if np.any(np.isfinite(S_inv)):
        if axis == "H" and len(wr.inv_values) == 2:
            E1, E2 = (float(v) for v in wr.inv_values)
            H = np.linspace(0, 1.0 / 3.0, 50)
            ax.plot(H, E1 + 2.0 * (E2 - E1) * H, "-", color="#1d4ed8", lw=1.4, label=lab_inv)
        else:
            ax.plot(xv, S_inv, "-", color="#1d4ed8", lw=1.2, marker=".", label=lab_inv)
    S_cf = np.array([wr.family_info[f].S_cf for f in ordered])
    if np.any(np.isfinite(S_cf)):
        cf = wr.cf_fit
        lbl = "model C_hkl"
        if "rho" in cf:
            lbl += f": ρ = {cf['rho']:.3g} ± {cf.get('rho_esd', float('nan')):.2g} Å⁻²"
            if np.isfinite(cf.get("fE_esd", float("nan"))):
                lbl += f", fE = {cf['fE']:.2f} ± {cf['fE_esd']:.2f}"
            else:
                lbl += f", fE = {cf.get('fE', float('nan')):.2f} (kept)"
        ax.plot(xv, S_cf, "--", color="#c2410c", lw=1.2, marker="+", ms=8, label=lbl)
    if axis == "H":
        ax.set_xlabel("H = (h²k² + h²l² + k²l²) / (h² + k² + l²)²")
    elif axis == "delta":
        ax.set_xlabel("angle between g ∥ r and the c axis (°)")
    else:
        ax.set_xticks([xs[f] for f in fams])
        ax.set_xticklabels([f"{f}\n{hkl_label(wr.family_info[f].hkl)}" for f in fams], fontsize=7)
    form = "ρb²⟨C⟩/4π" if (wr.options.strain_form == "wilkens" or wr.options.static_model == "model_cf") else "ε²"
    ax.set_ylabel(f"S_d = {form}_d  (dimensionless)")
    note = "per-direction values (independent)" if independent else f"directions constrained by the {wr.options.static_model} model"
    ax.set_title(note, fontsize=9)
    if wr.inv_names and len(wr.inv_values):
        txt = "\n".join(
            f"Ẽ{n[1:]} = {v:.3g} ± {e:.2g}" for n, v, e in zip(wr.inv_names, wr.inv_values, wr.inv_esd)
        )
        if len(wr.inv_values) >= 2 and abs(wr.inv_values[0]) > 0:
            ratios = ", ".join(f"{v / wr.inv_values[0]:.3f}" for v in wr.inv_values[1:])
            txt += f"\nE_i/E1 = {ratios}"
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#d1d5db", alpha=0.85))
    ax.legend(fontsize=8, loc="lower left")
    ax.axhline(0, color="#d1d5db", lw=0.6)
