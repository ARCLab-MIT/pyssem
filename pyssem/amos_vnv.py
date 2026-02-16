#!/usr/bin/env python3
import os
# --- Prevent BLAS/OMP oversubscription when using multi-process ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import json
import time
import csv
import itertools
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
import numpy as np
import math

# ---- Flexible imports; tweak if your project layout differs ----
try:
    from model import Model
except ImportError:
    try:
        from model import Model
    except ImportError:
        raise

try:
    from utils.plotting.SEPDataExport import *
except ImportError:
    try:
        from utils.plotting.SEPDataExport import *
    except ImportError:
        SEPDataExport = None  # optional

try:
    from utils.plotting.plotting import Plots, results_to_json
except ImportError:
    try:
        from utils.plotting.plotting import Plots, results_to_json
    except ImportError:
        Plots = None  # optional
# ----------------------------
# Grid & helpers
# ----------------------------
MODEL_TYPE_SETTINGS = {
    "frag_spread": {"fragment_spreading": True,  "elliptical": False},
    "circular":    {"fragment_spreading": False, "elliptical": False},
    "elliptical":  {"fragment_spreading": False, "elliptical": True},
}

# N_SHELLS_OPTIONS = [15, 20, 25, 30, 35, 40]
N_SHELLS_OPTIONS = [15, 20, 25, 30, 35]

SPECIES_K_OPTIONS = {
    "S":  [1, 3, 5],
    "N":  [5],
    "B":  [1, 2, 3],
}

# path to saved cluster centers from your notebook run
CLUSTERS_CSV_PATH = os.environ.get(
    "CLUSTERS_CSV_PATH",
    "/Users/indigobrownhall/Code/pyssem/species_kmeans_results/cluster_centres_by_species.csv"
)

DEFAULT_BASE_JSON = os.path.join("pyssem", "simulation_configurations", "mocat-mc.json")
MAIN_OUTPUT_DIR = "grid_search"

# Optional: set via env var if needed
MC_POP_TIME_PATH = os.environ.get("MC_POP_TIME_PATH", "/Users/indigobrownhall/Code/pyssem/figures/mocat-mc-elliptical-test/pop_time_mc.csv")


def build_simulation_name(model_type: str, n_shells: int) -> str:
    return f"{model_type}_nshell_{n_shells}"

def _standardize_cluster_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make the cluster-centers CSV tolerant to column-name variants."""
    colmap = {}
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    # canonical names we want: species, k, cluster, radius, mass, area, bstar
    need = {
        "species": pick("species", "species_class", "sym_name"),
        "k":       pick("k", "n_clusters", "clusters"),
        "cluster": pick("cluster", "cluster_id", "idx"),
        "radius":  pick("radius", "rad"),
        "mass":    pick("mass",),
        "area":    pick("area", "A"),
        "bstar":   pick("bstar"),
    }
    for want, have in need.items():
        if have is not None:
            colmap[have] = want
    df = df.rename(columns=colmap)

    # sanity checks
    for key in ("species", "k", "radius", "mass"):
        if key not in df.columns:
            raise ValueError(f"cluster-centers CSV missing required column '{key}'")

    # add cluster index if missing
    if "cluster" not in df.columns:
        # assign 0..k-1 per (species,k) by radius ascending
        df = df.sort_values(["species", "k", "radius"]).copy()
        df["cluster"] = (
            df.groupby(["species", "k"])
              .cumcount()
              .astype(int)
        )

    # Always recalculate area and bstar from radius and mass (ignore precomputed values)
    df["area"] = math.pi * (df["radius"].astype(float) ** 2)
    df["bstar"] = 2.2 * (df["area"].astype(float) / df["mass"].astype(float))
    return df


def load_cluster_centers(path: str | Path) -> dict:
    """
    Returns nested dict:
      centers[species][k] -> list of dicts with keys: radius, mass, area, bstar (sorted by 'cluster')
    Note: bstar is always recalculated from radius and mass, ignoring any precomputed values.
    """
    df = pd.read_csv(path)
    df = _standardize_cluster_cols(df)

    centers: dict[str, dict[int, list[dict]]] = {}
    for (sp, k), sub in df.groupby(["species", "k"]):
        sub = sub.sort_values("cluster")
        # Always recalculate bstar from radius and mass (area already recalculated in _standardize_cluster_cols)
        recs = sub[["radius", "mass", "area", "bstar"]].to_dict("records")
        # Recalculate bstar for each record to ensure consistency
        for rec in recs:
            area = rec["area"]
            mass = rec["mass"]
            rec["bstar"] = 2.2 * (area / mass) if mass > 0 else 0.0
        centers.setdefault(str(sp), {})[int(k)] = recs
    return centers


def species_k_grid() -> list[dict]:
    """Cartesian product of requested Ks per species."""
    combos = []
    for ks in SPECIES_K_OPTIONS["S"]:
        for kn in SPECIES_K_OPTIONS["N"]:
            for kb in SPECIES_K_OPTIONS["B"]:
                combos.append({"S": ks, "N": kn, "B": kb})
    return combos


def build_simulation_name(model_type: str, n_shells: int, species_k: dict) -> str:
    total = int(species_k.get("S", 1) +
                species_k.get("N", 1) + species_k.get("B", 1))
    # Keep your original format, with sp_{total} appended
    return f"{model_type}_nshell_{n_shells}_sp_{total}"


def prepare_config(base_cfg: dict, model_type: str, n_shells: int,
                   species_k: dict, cluster_map: dict) -> dict:
    """
    Build a config for a specific model type, shell count, and species-K choice.
    Expands the 'species' section with scalar or list values depending on K.
    Ensures mass/radius/A/bstar/deltat are consistent (all scalars or lists of the same length).
    """
    cfg = copy.deepcopy(base_cfg)
    sim_name = build_simulation_name(model_type, n_shells, species_k)
    cfg["simulation_name"] = sim_name

    sp = cfg.setdefault("scenario_properties", {})
    sp["n_shells"] = n_shells
    sp["fragment_spreading"] = MODEL_TYPE_SETTINGS[model_type]["fragment_spreading"]
    sp["elliptical"] = MODEL_TYPE_SETTINGS[model_type]["elliptical"]
    # Set integrator based on elliptical mode
    if sp["elliptical"]:
        sp["integrator"] = "Euler"
    else:
        sp["integrator"] = "BDF"

    base_species = cfg.get("species", [])
    if not isinstance(base_species, list) or not base_species:
        raise ValueError("Base configuration must include a non-empty 'species' list.")

    def _to_scalar_or_list(vals, k):
        """Return scalar if k==1 else a plain Python list of length k."""
        if k == 1:
            return vals[0]
        return [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]

    def _recalculate_bstar(area_val, mass_val):
        """Recalculate bstar from area and mass: 2.2 * (area / mass)"""
        if isinstance(area_val, list) and isinstance(mass_val, list):
            if len(area_val) != len(mass_val):
                raise ValueError(f"Mismatch: area has {len(area_val)} elements, mass has {len(mass_val)}")
            return [2.2 * (a / m) if m > 0 else 0.0 for a, m in zip(area_val, mass_val)]
        elif isinstance(area_val, list) or isinstance(mass_val, list):
            raise ValueError("Area and mass must both be scalars or both be lists")
        else:
            # Both scalars
            return 2.2 * (float(area_val) / float(mass_val)) if float(mass_val) > 0 else 0.0

    new_species = []
    for entry in base_species:
        entry2 = copy.deepcopy(entry)
        sym = str(entry2.get("sym_name", "")).strip()

        # Only expand for S, N, B. Leave others (e.g., D) as-is.
        if sym in ("S", "N", "B"):
            k = int(species_k.get(sym, 1))

            # Special case: Use hardcoded parameters for species B
            if sym == "B":
                # Hardcoded B parameters (k=3)
                masses = [66.01351, 1872.35294, 617.89313]
                radii = [0.6349590076464132, 1.8157556500788736, 1.792676077595715]
                areas = [1.26708, 7.80024, 7.93641]
                bstars = [0.042227356188149975, 0.009165220740914371, 0.028257478764976723]
                
                # Use the requested k (should be 3, but handle other cases)
                if k != 3:
                    print(f"[WARN] Species B hardcoded for k=3, but requested k={k}. Using first {k} values.")
                    masses = masses[:k]
                    radii = radii[:k]
                    areas = areas[:k]
                    bstars = bstars[:k]
            else:
                # For S and N, fetch centers from cluster map
                centers = cluster_map.get(sym, {}).get(k)
                if centers is None or len(centers) != k:
                    raise ValueError(
                        f"No cluster centers for {sym} with k={k} "
                        f"in {CLUSTERS_CSV_PATH}. Found: {0 if centers is None else len(centers)}"
                    )

                masses  = [float(c["mass"])   for c in centers]
                radii   = [float(c["radius"]) for c in centers]

                # compute areas from radii (m^2) to guarantee length matches k
                areas   = [math.pi * (r**2) for r in radii]

                # compute bstar from areas & masses
                # bstar = 2.2 * (area / mass)
                bstars  = [2.2 * (a / m) if m > 0 else 0.0 for a, m in zip(areas, masses)]

            # replicate base deltat across k (keep scalar if k==1)
            base_dt = entry2.get("deltat", 8)
            deltas  = [base_dt] * k

            # assign as scalar if k==1, otherwise as lists of length k
            entry2["mass"]   = _to_scalar_or_list(masses, k)
            entry2["radius"] = _to_scalar_or_list(radii,  k)
            entry2["A"]      = _to_scalar_or_list(areas,  k)
            entry2["bstar"]  = _to_scalar_or_list(bstars, k)
            entry2["deltat"] = _to_scalar_or_list(deltas, k)
        else:
            # For other species, recalculate bstar if both A and mass are present
            if "A" in entry2 and "mass" in entry2:
                entry2["bstar"] = _recalculate_bstar(entry2["A"], entry2["mass"])

        new_species.append(entry2)

    cfg["species"] = new_species
    cfg["species_count"] = (
        species_k.get("S", 1) +
        species_k.get("N", 1) + species_k.get("B", 1)
    )
    return cfg


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_cpu_time_csv(folder: str | Path, scenario_name: str, cpu_seconds: float):
    path = Path(folder) / "cpu_time.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_name", "cpu_seconds"])
        writer.writerow([scenario_name, f"{cpu_seconds:.6f}"])


def run_single_sim(simulation_data: dict, main_dir: str = MAIN_OUTPUT_DIR) -> tuple[str, float, float, str]:
    """
    Worker-safe: configure & run a single simulation.
    Returns: (scenario_name, cpu_seconds, wall_seconds, status)
    """
    sim_name = simulation_data["simulation_name"]
    scenario_props = simulation_data["scenario_properties"]

    sim_dir = Path(main_dir) / sim_name
    ensure_dir(sim_dir)

    # Save exact config used
    write_json(sim_dir / "config_used.json", simulation_data)

    # Instantiate model
    model = Model(
        start_date=scenario_props["start_date"].split("T")[0],
        simulation_duration=scenario_props["simulation_duration"],
        steps=scenario_props["steps"],
        min_altitude=scenario_props["min_altitude"],
        max_altitude=scenario_props["max_altitude"],
        n_shells=scenario_props["n_shells"],
        launch_function=scenario_props["launch_function"],
        integrator=scenario_props["integrator"],
        density_model=scenario_props["density_model"],
        LC=scenario_props["LC"],
        v_imp=scenario_props.get("v_imp", None),
        fragment_spreading=scenario_props.get("fragment_spreading", False),
        parallel_processing=scenario_props.get("parallel_processing", True),
        baseline=scenario_props.get("baseline", False),
        indicator_variables=scenario_props.get("indicator_variables", None),
        launch_scenario=scenario_props["launch_scenario"],
        SEP_mapping=simulation_data.get("SEP_mapping"),
        elliptical=scenario_props.get("elliptical", None),
        eccentricity_bins=scenario_props.get("eccentricity_bins", None),
    )

    # Configure species
    species = simulation_data.get("species", [])
    _ = model.configure_species(species)

    # Measure CPU time just for run_model()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    status = "OK"
    try:
        _results = model.run_model()
    except Exception as e:
        status = f"FAIL: {type(e).__name__}: {e}"
        with open(sim_dir / "error.txt", "w") as f:
            f.write(status + "\n")
    finally:
        cpu_end = time.process_time()
        wall_end = time.perf_counter()

    cpu_seconds = cpu_end - cpu_start
    wall_seconds = wall_end - wall_start

    # Save runtimes
    with open(sim_dir / "model_runtime.txt", "w") as f:
        f.write(f"CPU seconds: {cpu_seconds:.6f}\n")
        f.write(f"Wall seconds: {wall_seconds:.6f}\n")
        f.write(f"Status: {status}\n")

    # CPU time CSV as requested
    write_cpu_time_csv(sim_dir, sim_name, cpu_seconds)

    # Results JSON (best effort)
    try:
        data = model.results_to_json()
        write_json(sim_dir / "results.json", data)
    except Exception as e:
        with open(sim_dir / "results_serialize_error.txt", "w") as f:
            f.write(f"{type(e).__name__}: {e}\n")

    # Optional: export plots / comparisons
    try:
        plot_names = "all_plots"
        if SEPDataExport is not None:
            SEPDataExport(
                model.scenario_properties,
                sim_name,
                elliptical=model.scenario_properties.elliptical,
                MOCAT_MC_Path=MC_POP_TIME_PATH,
                output_dir=str(sim_dir),
            )
        if Plots is not None and plot_names:
            # Plots writes into main_dir/sim_name internally
            Plots(model.scenario_properties, plot_names, sim_name, main_dir)
    except Exception as e:
        with open(sim_dir / "plotting_warning.txt", "w") as f:
            f.write(f"{type(e).__name__}: {e}\n")
    finally:
        # Ensure all matplotlib figures are closed
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass

    return sim_name, cpu_seconds, wall_seconds, status


def main(base_json_path: str = DEFAULT_BASE_JSON, max_workers: int | None = None):
    # Load base configuration
    with open(base_json_path, "r") as f:
        base_cfg = json.load(f)

    # Load cluster centers once
    try:
        cluster_map = load_cluster_centers(CLUSTERS_CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load cluster centers from {CLUSTERS_CSV_PATH}: {e}")

    ensure_dir(MAIN_OUTPUT_DIR)

    # Build grid: (model_type × n_shells × species-K combinations)
    configs = []
    for model_type in MODEL_TYPE_SETTINGS.keys():
        for n_shells in N_SHELLS_OPTIONS:
            for k_choice in species_k_grid():
                cfg = prepare_config(base_cfg, model_type, n_shells, k_choice, cluster_map)
                configs.append(cfg)

    # Use all available cores by default
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    print(f"Launching {len(configs)} runs with {max_workers} workers...")

    summary_rows = [("scenario_name", "cpu_seconds", "wall_seconds", "status")]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(run_single_sim, cfg, MAIN_OUTPUT_DIR): cfg["simulation_name"] for cfg in configs}
        for fut in as_completed(future_map):
            sim_name = future_map[fut]
            try:
                name, cpu_s, wall_s, status = fut.result()
                print(f"[{status}] {name}: CPU {cpu_s:.2f}s | Wall {wall_s:.2f}s")
            except Exception as e:
                name = sim_name
                cpu_s = wall_s = float("nan")
                status = f"FAIL: {type(e).__name__}: {e}"
                print(f"[{status}] {name}")
            summary_rows.append((name, f"{cpu_s:.6f}", f"{wall_s:.6f}", status))

    with open(Path(MAIN_OUTPUT_DIR) / "grid_search_new_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)

    print(f"\nSummary written to {Path(MAIN_OUTPUT_DIR) / 'grid_search_new_summary.csv'}")


if __name__ == "__main__":
    # macOS uses 'spawn' by default; forcing here for clarity/consistency.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # already set
        pass
    # Optionally allow override via env var GRID_WORKERS
    workers_env = os.environ.get("GRID_WORKERS")
    workers = int(workers_env) if workers_env else None
    main(DEFAULT_BASE_JSON, workers)