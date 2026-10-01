# -*- coding: utf-8 -*-

import configparser
import re
import os
import datetime
from typing import Dict, Any, Tuple, Optional


# -----------------------------------------------------------------------------
# Contrast factors parsing
# -----------------------------------------------------------------------------
def _parse_contrast_factors_section(section_items):
    """Parse [contrast_factors] into a structured dict.

    Supported input styles (case-insensitive keys):
      - Legacy cubic: CEdgeA, CEdgeB, CScrewA, CScrewB
      - Invariants (recommended):
            EdgeE1..EdgeE15 and ScrewE1..ScrewE15
        Also accepts Edge_E1 / Screw_E1 / EDGE1 / SCREW1 etc.

      - Optional: burgers_mag (or b) in Å for Wilkens strain model.

    Returns a dict:
      {
        "legacy": {"CEdgeA":..., "CEdgeB":..., "CScrewA":..., "CScrewB":...}   (optional)
        "edge_E": {"E1":..., ... }  (optional)
        "screw_E": {"E1":..., ... } (optional)
        "burgers_mag": float        (optional)
      }
    """
    legacy = {}
    edge_E = {}
    screw_E = {}
    burgers_mag = None

    for k, v in section_items:
        key = str(k).strip()
        ku = key.upper()

        # skip empty values
        if v is None:
            continue

        try:
            val = float(str(v).strip())
        except Exception:
            # Ignore non-numeric values silently
            continue

        # Legacy cubic
        if ku in ("CEDGEA", "CEDGEB", "CSCREWA", "CSCREWB"):
            if ku == "CEDGEA":
                legacy["CEdgeA"] = val
            elif ku == "CEDGEB":
                legacy["CEdgeB"] = val
            elif ku == "CSCREWA":
                legacy["CScrewA"] = val
            elif ku == "CSCREWB":
                legacy["CScrewB"] = val
            continue

        # Burgers magnitude
        if ku in ("BURGERS_MAG", "BURGERS", "B_MAG", "B", "BURGERSMAG"):
            burgers_mag = val
            continue

        # EdgeE1 / ScrewE1 (allow underscore/space)
        m = re.match(r"^(EDGE|SCREW)[_ ]*E(\d{1,2})$", ku)
        if m:
            kind, idx = m.group(1), int(m.group(2))
            if 1 <= idx <= 15:
                if kind == "EDGE":
                    edge_E[f"E{idx}"] = val
                else:
                    screw_E[f"E{idx}"] = val
            continue

        # Edge1 / Screw1 shorthand
        m2 = re.match(r"^(EDGE|SCREW)[_ ]*(\d{1,2})$", ku)
        if m2:
            kind, idx = m2.group(1), int(m2.group(2))
            if 1 <= idx <= 15:
                if kind == "EDGE":
                    edge_E[f"E{idx}"] = val
                else:
                    screw_E[f"E{idx}"] = val
            continue

        # Plain E1..E15 -> apply to both if not split
        m3 = re.match(r"^E(\d{1,2})$", ku)
        if m3:
            idx = int(m3.group(1))
            if 1 <= idx <= 15:
                edge_E.setdefault(f"E{idx}", val)
                screw_E.setdefault(f"E{idx}", val)
            continue

    out = {}
    if legacy:
        out["legacy"] = legacy
    if edge_E:
        out["edge_E"] = edge_E
    if screw_E:
        out["screw_E"] = screw_E
    if burgers_mag is not None:
        out["burgers_mag"] = float(burgers_mag)
    return out


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def clean_element_name(name: str) -> str:
    m = re.match(r"([A-Za-z]{1,2})", str(name))
    if not m:
        raise ValueError(f"Could not parse chemical symbol from '{name}'")
    sym = m.group(1)
    return sym.upper() if len(sym) == 1 else sym[0].upper() + sym[1:].lower()


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Write-back helpers
# -----------------------------------------------------------------------------
def update_input_file_with_refined_params(input_path, refined_params, refinable_flags):
    """
    Update refined parameter values in the input file.

    Important:
    - Preserve original key spelling when possible.
    - Add missing GUI-generated parameters, e.g. biso_li1, delta1_ca-o.
    - Preserve refine flags.
    """
    config = configparser.RawConfigParser()
    config.optionxform = str
    config.read(input_path)

    def _existing_key_map(section_name: str) -> dict:
        if not config.has_section(section_name):
            return {}
        return {k.strip().lower(): k for k in config.options(section_name)}

    def _apply_updates(section_name: str, updates: dict, allow_add: bool = True) -> None:
        if not config.has_section(section_name):
            if allow_add:
                config.add_section(section_name)
            else:
                return

        key_map = _existing_key_map(section_name)

        for k, v in (updates or {}).items():
            key = str(k).strip()
            if not key:
                continue

            lk = key.lower()

            if lk in key_map:
                orig = key_map[lk]
                config.set(section_name, orig, str(v))
            elif allow_add:
                config.set(section_name, key, str(v))

    # Add/update refined values.
    _apply_updates(
        "initial_values",
        refined_params or {},
        allow_add=True,
    )

    # Add/update refine flags.
    updates = {
        str(k): ("True" if bool(v) else "False")
        for k, v in (refinable_flags or {}).items()
    }

    _apply_updates(
        "refinable_parameters",
        updates,
        allow_add=True,
    )

    with open(input_path, "w", encoding="utf-8") as configfile:
        config.write(configfile)

        
def write_fit_log(output_dir: str, params: dict, chi2: float, rwp: float, input_path: str):
    """
    Append a fit summary to a log file inside output_dir.
    Creates output_dir if missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "fit_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Chi2: {chi2:.6f}\n")
        f.write(f"Rwp: {rwp:.6f}\n")
        f.write("Refined parameters:\n")
        for k in sorted(params.keys()):
            try:
                f.write(f"  {k}: {float(params[k]):.12g}\n")
            except Exception:
                f.write(f"  {k}: {params[k]}\n")
        f.write("\n")


# -----------------------------------------------------------------------------
# Main reader
# -----------------------------------------------------------------------------
def read_input_file(filename: str) -> Dict[str, Any]:
    # IMPORTANT: use RawConfigParser (no %-interpolation).
    # This allows constraint expressions like "a%b" or strings containing '%'
    # without breaking parsing.
    config_raw = configparser.RawConfigParser()
    config_raw.read(filename)

    # --- Required files section ---
    if "files" not in config_raw:
        raise KeyError("Missing [files] section.")
    if "structure_file" not in config_raw["files"] or "gr_data_file" not in config_raw["files"]:
        raise KeyError("Missing required keys in [files]: structure_file, gr_data_file")

    structure_file = config_raw["files"]["structure_file"]
    gr_data_file = config_raw["files"]["gr_data_file"]

    # --- Range ---
    if "range" not in config_raw:
        raise KeyError("Missing [range] section.")
    r_min = float(config_raw["range"]["r_min"])
    r_max = float(config_raw["range"]["r_max"])

    # --- Parameters ---
    initial: Dict[str, Any] = {}
    refinable: Dict[str, bool] = {}
    bounds: Dict[str, Tuple[float, float]] = {}

    # initial_values
    if "initial_values" in config_raw:
        for k, v in config_raw["initial_values"].items():
            vv = str(v).strip()
            # allow floats; keep strings if not parseable
            valf = _to_float(vv, None)
            initial[k] = valf if valf is not None else vv

    # refinable section name (support both)
    ref_section = None
    if "refinable_flags" in config_raw:
        ref_section = "refinable_flags"
    elif "refinable_parameters" in config_raw:
        ref_section = "refinable_parameters"

    if ref_section is not None:
        for k, v in config_raw[ref_section].items():
            refinable[k] = str(v).strip().lower() in ("true", "1", "yes", "y")

    # bounds
    if "bounds" in config_raw:
        for k, v in config_raw["bounds"].items():
            parts = [p.strip() for p in str(v).split(",", 1)]
            if len(parts) == 2:
                lo = _to_float(parts[0], None) if parts[0] != "" else None
                hi = _to_float(parts[1], None) if parts[1] != "" else None
                if lo is not None or hi is not None:
                    bounds[k] = (
                        float(lo) if lo is not None else None,
                        float(hi) if hi is not None else None,
                    )

    # Default angles if missing
    for angle in ("alpha", "beta", "gamma"):
        if angle not in initial:
            initial[angle] = 90.0

    # pair_generation
    r_extension = float(config_raw.get("pair_generation", "r_extension", fallback="1.2"))

    # contrast_factors
    # [contrast_factors] is deprecated.
    # If an old file contains it, migrate values into [initial_values]-style params.
    contrast_factors = {}

    if "contrast_factors" in config_raw:
        old_cf = _parse_contrast_factors_section(config_raw["contrast_factors"].items())

        try:
            legacy = old_cf.get("legacy", {}) if isinstance(old_cf, dict) else {}

            if "CEdgeA" in legacy:
                initial.setdefault("CEdgeA", float(legacy["CEdgeA"]))
                refinable.setdefault("CEdgeA", False)

            if "CEdgeB" in legacy:
                initial.setdefault("CEdgeB", float(legacy["CEdgeB"]))
                refinable.setdefault("CEdgeB", False)

            if "CScrewA" in legacy:
                initial.setdefault("CScrewA", float(legacy["CScrewA"]))
                refinable.setdefault("CScrewA", False)

            if "CScrewB" in legacy:
                initial.setdefault("CScrewB", float(legacy["CScrewB"]))
                refinable.setdefault("CScrewB", False)

            edge_E = old_cf.get("edge_E", {}) if isinstance(old_cf, dict) else {}
            screw_E = old_cf.get("screw_E", {}) if isinstance(old_cf, dict) else {}

            for ekey, val in edge_E.items():
                pname = f"Edge{str(ekey).upper()}"   # EdgeE1, EdgeE2, ...
                initial.setdefault(pname, float(val))
                refinable.setdefault(pname, False)

            for ekey, val in screw_E.items():
                pname = f"Screw{str(ekey).upper()}"  # ScrewE1, ScrewE2, ...
                initial.setdefault(pname, float(val))
                refinable.setdefault(pname, False)

            if "burgers_mag" in old_cf:
                initial.setdefault("burgers_mag", float(old_cf["burgers_mag"]))
                refinable.setdefault("burgers_mag", False)

        except Exception:
            pass

    # constraints
    constraints: Dict[str, str] = {}
    if "constraints" in config_raw:
        for k, v in config_raw["constraints"].items():
            key = str(k).strip()
            expr = "" if v is None else str(v).strip()
            if key and expr:
                constraints[key] = expr

    # --- Microstrain validation (accept both cases: Re/re, fE/fe) ---
    # We don't force them to exist; just warn if user seems to want Wilkens.
    for name in ("rho", "re", "fe"):
        if name not in initial and name.upper() not in initial and name.capitalize() not in initial:
            # only warn if any microstrain key exists
            pass
        if name not in refinable and name.upper() not in refinable and name.capitalize() not in refinable:
            # don't spam warnings; your pipeline can handle fixed defaults
            pass

    # --- Extract species-specific Biso values ---
    #
    # Only treat pure element keys as species-level Biso:
    #   biso_Li -> Li
    #   biso_Ge -> Ge
    #
    # Do NOT treat site keys as species-level Biso:
    #   biso_li1
    #   biso_ge1
    #   biso_s2
    #
    # Site Biso parameters remain in initial/refinable dictionaries and are
    # handled directly by PDFCalculator.
    biso_by_species: Dict[str, float] = {}

    for k, v in initial.items():
        kl = str(k).strip().lower()

        if not kl.startswith("biso_"):
            continue

        raw_species = str(k)[len("biso_"):].strip()

        try:
            species = clean_element_name(raw_species)
        except Exception:
            continue

        # Only accept exact element-symbol keys.
        # Examples accepted:
        #   biso_Li
        #   biso_li
        # Rejected:
        #   biso_li1
        #   biso_ge1
        if raw_species.lower() != species.lower():
            continue

        try:
            biso_by_species[species] = float(v)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Size distribution handling (THIS FIXES YOUR WARNING)
    # Accept either:
    #   - d + d_std     (real-space mean + std)  [your input]
    #   - d_mu + d_std  (log-space mu + sigma)   [optional]
    # Only warn if neither mode has a valid size.
    # -------------------------------------------------------------------------
    d_val = _to_float(initial.get("d", None), None)
    dstd_val = _to_float(initial.get("d_std", None), 0.0)
    dmu_val = _to_float(initial.get("d_mu", None), None)

    if d_val is not None and d_val > 0.0:
        # Real-space mean/std mode (no warning)
        initial["d"] = float(d_val)
        initial["d_std"] = max(0.0, float(dstd_val))

        # If d_mu exists too, ignore it to avoid ambiguity
        if "d_mu" in initial:
            # keep but do not use; you can delete if you prefer
            pass

    elif dmu_val is not None:
        # Lognormal mu/sigma mode (requires positive sigma)
        sig = _to_float(initial.get("d_std", None), None)
        if sig is None or sig <= 0.0:
            print("[WARNING] 'd_mu' provided but 'd_std' (lognormal sigma) missing/invalid. Defaulting to monodisperse.")
            initial["d"] = 0.0
            initial["d_std"] = 0.0
        else:
            initial["d_mu"] = float(dmu_val)
            initial["d_std"] = float(sig)

    else:
        # Nothing valid -> monodisperse
        # Only warn if user tried to use distribution keys but failed
        if ("d" in initial) or ("d_std" in initial) or ("d_mu" in initial):
            print("[WARNING] Grain size parameters invalid. Defaulting to monodisperse.")
        initial["d"] = 0.0
        initial["d_std"] = 0.0

    # --- Refinement options ---
    # Parse [refinement] section into a typed dict (bool/int/float/str).
    refinement_cfg: Dict[str, Any] = {}
    if "refinement" in config_raw:
        for k, v in config_raw["refinement"].items():
            key = str(k).strip()
            raw = "" if v is None else str(v).strip()

            if raw == "":
                continue

            low = raw.lower()
            if low in ("true", "false", "yes", "no", "y", "n", "on", "off", "1", "0"):
                refinement_cfg[key] = low in ("true", "yes", "y", "on", "1")
                continue

            # int?
            if re.fullmatch(r"[+-]?\d+", raw):
                try:
                    refinement_cfg[key] = int(raw)
                    continue
                except Exception:
                    pass

            # float?
            try:
                refinement_cfg[key] = float(raw)
                continue
            except Exception:
                refinement_cfg[key] = raw

    # --- Handle lambda parameters ---
    # Extract lambda parameters from the input file
    lambda_params = {}
    for k, v in initial.items():
        if str(k).lower().startswith("lambda_"):
            # Handle both formats: lambda_Fe-Fe_0 and lambda_0
            if "-" in k:
                # Format: lambda_A-B_k
                try:
                    # Extract A, B, k from the key
                    parts = k.split("_")
                    if len(parts) >= 3:
                        a = parts[1]
                        b = parts[2]
                        k_val = int(parts[3])
                        pair_key = f"{a}-{b}"
                        lambda_params[pair_key] = lambda_params.get(pair_key, {})
                        lambda_params[pair_key][k_val] = float(v)
                except Exception:
                    pass
            else:
                # Format: lambda_k
                try:
                    k_val = int(k.split("_")[1])
                    lambda_params[f"global_{k_val}"] = float(v)
                except Exception:
                    pass

    

    crystallite_shape: Dict[str, Any] = {}

    if "crystallite_shape" in config_raw:
        for k, v in config_raw["crystallite_shape"].items():
            key = str(k).strip()
            raw = "" if v is None else str(v).strip()
    
            if not key or raw == "":
                continue
            
            low = raw.lower()
    
            if low in ("true", "false", "yes", "no", "y", "n", "on", "off", "1", "0"):
                crystallite_shape[key] = low in ("true", "yes", "y", "on", "1")
                continue
            
            try:
                if re.fullmatch(r"[+-]?\d+", raw):
                    crystallite_shape[key] = int(raw)
                    continue
            except Exception:
                pass
            
            try:
                crystallite_shape[key] = float(raw)
                continue
            except Exception:
                crystallite_shape[key] = raw

    # --- Add lambda parameters to the config ---
    # This ensures they're properly handled by the GUI
    config = {
        "structure_file": structure_file,
        "gr_data_file": gr_data_file,
        "r_min": r_min,
        "r_max": r_max,
        "refinement": refinement_cfg,
        "refinable": refinable,
        "initial": initial,
        "bounds": bounds,
        "r_extension": r_extension,
        "contrast_factors": contrast_factors,
        "constraints": constraints,
        "biso_by_species": biso_by_species,
        "lambda_params": lambda_params,  # Add lambda parameters to config
        "crystallite_shape": crystallite_shape,
    }


    return config
