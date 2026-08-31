# -*- coding: utf-8 -*-
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import re  # NEW
import os
from pdf_fitting.io_handler import update_input_file_with_refined_params, write_fit_log

from pdf_fitting.io_handler import read_input_file
from pdf_fitting.fit_engine import FitEngine
from pdf_fitting.models.gr_model import PDFCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def clean_element_name(name):
    """
    Remove oxidation state or charge from element symbol and fix capitalization.
    Example: fe0+ -> Fe, cu2+ -> Cu, LA -> La
    """
    m = re.match(r"([A-Za-z]{1,2})", name)
    if m:
        sym = m.group(1)
        if len(sym) == 1:
            return sym.upper()
        else:
            return sym[0].upper() + sym[1:].lower()
    else:
        raise ValueError(f"Could not parse chemical symbol from '{name}'")

def calculate_rwp(G_exp, G_fit, weights=None):
    """
    Compute the weighted profile R-factor (Rwp) for PDF fit.
    If weights are not given, uniform weights are used.
    """
    G_exp = np.asarray(G_exp)
    G_fit = np.asarray(G_fit)
    if weights is None:
        weights = np.ones_like(G_exp)
    num = np.sum(weights * (G_exp - G_fit)**2)
    denom = np.sum(weights * G_exp**2)
    if denom == 0:
        return np.nan
    return np.sqrt(num / denom)



def main():
    t0 = time.time()
    config = read_input_file("config/input.inp")
    print(f"[TIMING] Configuration loaded in {time.time() - t0:.2f} s")

    print("Initial values for fitting:")
    for name in config['initial']:
        if name in ("ln_mu", "ln_s"):
            continue  # Skip internal lognormal keys
        print(f"  {name}: {config['initial'][name]}")

    t1 = time.time()
    r_exp, G_exp = np.loadtxt(config['gr_data_file'], unpack=True)
    print(f"[TIMING] Experimental data loaded in {time.time() - t1:.2f} s")

    r_min = config['r_min']
    r_max = config['r_max']
    r_extension = config.get('r_extension', 1.0)
    pair_cutoff = r_max * r_extension
    mask = (r_exp >= r_min) & (r_exp <= r_max)
    r_exp = r_exp[mask]
    G_exp = G_exp[mask]

    t3 = time.time()
    pdf_model = PDFCalculator(config, config['structure_file'], r_exp, pair_cutoff=pair_cutoff)
    structure = pdf_model.structure_handler.structure
    crystal_system = SpacegroupAnalyzer(structure).get_crystal_system().lower()
    if crystal_system != "cubic":
        print("WARNING: Chkl (dislocation contrast factor) is only valid for cubic crystals. Do not use the Wilkens microstrain model for non-cubic structures!")

    print(f"[TIMING] PDF model initialized in {time.time() - t3:.2f} s")

    print("Initial values for fitting:")
    for name in config['initial']:
        if name in ("ln_mu", "ln_s"):
            continue  # Skip internal lognormal keys
        print(f"  {name}: {config['initial'][name]}")

    t4 = time.time()
    engine = FitEngine(r_exp, G_exp, pdf_model, config)
    result = engine.run_refinement()
    print(f"[TIMING] Refinement completed in {time.time() - t4:.2f} s")

    if result is None:
        print("No fit performed. Using initial parameters to compute G(r).")
        result_params = config['initial']
    else:
        result_params = result if isinstance(result, dict) else result.params

    print("[INFO] Recalculating real-space pair distances from symbolic vectors...")

    t5 = time.time()
    # IMPORTANT: to debug multi-element systems we need partial g_ab(r) and G_ab(r).
    # These are only constructed when return_gr=True.
    results = pdf_model.evaluate(result_params, return_gr=True)
    print(f"[TIMING] Final G(r) evaluation completed in {time.time() - t5:.2f} s")

    G_fit = results["G_r"]
    g_r  = results.get("g_r", np.zeros_like(G_fit))
    # Ordered partials (alpha-beta) from the kernel
    g_ab = results.get("g_ab", {})
    # Unordered (A-B) averaged partial PDFs (see PDFCalculator.evaluate)
    G_ab = results.get("G_ab", {})
    weights = results.get("weights", {})
    r = results["r"]

    chi2 = np.sum((G_exp - G_fit) ** 2)
    print(f"Chi² = {chi2:.6f}")

    # ---- Properly normalized total g(r) ----
    species = [clean_element_name(str(site.specie)) for site in pdf_model.structure_handler.structure.sites]
    unique_species = sorted(set(species))
    counts = {s: species.count(s) for s in unique_species}
    N_total = len(species)
    concentrations = {s: counts[s] / N_total for s in unique_species}

    g_total = np.zeros_like(r)
    for pair, g_curve in g_ab.items():
        alpha, beta = [clean_element_name(x) for x in pair.split("-")]
        c_a = concentrations[alpha]
        c_b = concentrations[beta]
        g_total += c_a * c_b * g_curve

    # -------------------- CLEAN COMPARISON PLOT --------------------
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: G(r) curves
    axs[0].plot(r_exp, G_exp, label="Experimental G(r)", linestyle='-', color='black', linewidth=2)
    axs[0].plot(r, G_fit, label="Fitted G(r)", color='royalblue', linewidth=2)
    axs[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[0].set_ylabel("G(r)")
    axs[0].set_title("Experimental and Fitted G(r)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':')
    
    # Subplot 2: Residuals
    # Interpolate G_fit onto r_exp for accurate residuals if grids differ
    if not np.allclose(r, r_exp):
        from scipy.interpolate import interp1d
        interp_fit = interp1d(r, G_fit, kind='linear', fill_value="extrapolate")
        G_fit_interp = interp_fit(r_exp)
    else:
        G_fit_interp = G_fit
    
    residuals = G_exp - G_fit_interp
    axs[1].plot(r_exp, residuals, color='firebrick', label="Residuals: G(exp) - G(fit)")
    axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[1].set_xlabel("r (Å)")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Fit Residuals")
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------------------



    # Save output files
    np.savetxt("output/fit_result.xy", np.column_stack((r_exp, G_exp, G_fit, G_exp - G_fit)))
    with open("output/parameters.json", "w", encoding="utf-8") as f:
        json.dump(result_params, f, indent=2)

    # Save the pair weights used to build the total g(r)
    try:
        with open("output/pair_weights.json", "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
    except Exception:
        pass

    # --- Output partial g_ab(r) and G_ab(r) for each pair type ---
    os.makedirs("output", exist_ok=True)

    # 1) Save the *small* g_ab(r) (ordered pairs as computed internally)
    for pair, g_curve in g_ab.items():
        filename = f"output/g_{pair}.xy"
        data = np.column_stack((r, g_curve))
        np.savetxt(filename, data, header="r [Å]\tg_ab(r)", comments='')
        print(f"[INFO] Saved partial g_ab(r) for pair {pair} to {filename}")

    # 2) Save the *big* G_ab(r) (unordered A-B, averaged)
    for pair, G_curve in G_ab.items():
        filename = f"output/G_{pair}.xy"
        # Each file will have columns: r, G_ab(r)
        data = np.column_stack((r, G_curve))
        np.savetxt(filename, data, header="r [Å]\tG(r)", comments='')
        print(f"[INFO] Saved partial G(r) for pair {pair} to {filename}")

    # 3) Optional compact debug table with *both* g and G for every pair in one file
    #    (handy to inspect tails: whether g_ab(r)->1 and thus G_ab(r)->0)
    try:
        dbg_path = os.path.join("output", "partials_g_G_allpairs.tsv")
        # Build a stable column order
        g_keys = sorted(g_ab.keys())
        G_keys = sorted(G_ab.keys())
        cols = [r]
        header = ["r_A"]
        for k in g_keys:
            cols.append(np.asarray(g_ab[k]))
            header.append(f"g_{k}")
        for k in G_keys:
            cols.append(np.asarray(G_ab[k]))
            header.append(f"G_{k}")
        mat = np.column_stack(cols)
        np.savetxt(dbg_path, mat, delimiter="\t", header="\t".join(header), comments='')
        print(f"[INFO] Saved combined partials table to {dbg_path}")
    except Exception as e:
        print(f"[WARN] Could not write combined partials table: {e}")

    # At the end of main(), after fitting and after the G(r) plot:
    if result_params.get("rho", 0.0) != 0:
        print("[INFO] Plotting Wilkens microstrain (directional Warren plot)...")
        strain_by_dir = pdf_model.evaluate(result_params, return_strain_plot=True)
        # Only plot for direction keys (tuples of 3 ints, not (0,0,0))
        dir_counts = [
            (d, len(v)) for d, v in strain_by_dir.items()
            if isinstance(d, tuple) and len(d) == 3 and all(isinstance(x, int) for x in d) and d != (0,0,0)
        ]
        dir_counts.sort(key=lambda x: -x[1])
        n_plot = 5
        plt.figure(figsize=(10, 6))
        for direction, _ in dir_counts[:n_plot]:
            arr = np.array(strain_by_dir[direction])
            if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 2:
                print(f"[WARREN] Skipping {direction}: arr.shape={arr.shape}")
                continue
            arr = arr[arr[:,0].argsort()]
            Ls, DeltaL2 = arr[:,0], arr[:,1]
            plt.plot(Ls, np.sqrt(np.maximum(DeltaL2, 0)), marker='o', label=f"dir {direction}")
        plt.xlabel("Pair Distance $L$ (Å)")
        plt.ylabel(r"$\sqrt{\langle \Delta L^2 \rangle}$ (Å)")
        plt.title(r"Dislocation-Induced Broadening: $\sqrt{\langle \Delta L^2 \rangle}$ vs $L$ (Wilkens Model)")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

    # After fit and plotting:
    chi2 = np.sum((G_exp - G_fit) ** 2)
    rwp = calculate_rwp(G_exp, G_fit)
    print(f"Chi² = {chi2:.6f}")
    print(f"Rwp = {rwp:.6f}")
    
    # Logging
    input_path = "config/input.inp"  # Make sure this matches your config path!
    write_fit_log("output", result_params, chi2, rwp, input_path)
    
    # Ask the user if they want to update the input.inp with the refined parameters
    response = input("\n[USER INPUT] Is the fit satisfactory? Do you want to update input.inp with the refined values? [y/n]: ").strip().lower()
    if response == "y":
        update_input_file_with_refined_params(input_path, result_params, config['refinable'])
        print("[INFO] input.inp has been updated with the refined parameters.")


    print(f"[TIMING] Total runtime: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
