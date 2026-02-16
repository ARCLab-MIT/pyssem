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
import copy
from pathlib import Path

# Import run_single_sim from amos_vnv
try:
    from amos_vnv import run_single_sim
except ImportError:
    try:
        from pyssem.amos_vnv import run_single_sim
    except ImportError:
        raise ImportError("Could not import run_single_sim from amos_vnv")

# Configuration
BASE_CONFIG_PATH = Path("grid_search/elliptical_nshell_20_sp_13/config_used.json")
OUTPUT_DIR = Path("monte_carlo_results")
N_RUNS = 10


def load_base_config(config_path: Path) -> dict:
    """Load the base configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_monte_carlo_simulation(config: dict, run_number: int) -> tuple[str, float, float, str]:
    """
    Run a single Monte Carlo simulation using run_single_sim from amos_vnv.
    Returns: (scenario_name, cpu_seconds, wall_seconds, status)
    """
    sim_name = f"monte-carlo_run_{run_number:02d}"
    config_copy = copy.deepcopy(config)
    config_copy["simulation_name"] = sim_name
    
    # Use run_single_sim to run and save the simulation
    return run_single_sim(config_copy, str(OUTPUT_DIR))


def main():
    """Main function to run Monte Carlo simulations."""
    # Load base config
    if not BASE_CONFIG_PATH.exists():
        print(f"ERROR: Base config not found at {BASE_CONFIG_PATH}")
        return
    
    base_config = load_base_config(BASE_CONFIG_PATH)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the base config
    with open(OUTPUT_DIR / "base_config.json", "w") as f:
        json.dump(base_config, f, indent=2)
    
    # Run Monte Carlo simulations
    print(f"\nRunning {N_RUNS} Monte Carlo simulations...")
    results = []
    
    for run_num in range(1, N_RUNS + 1):
        sim_name, cpu_s, wall_s, status = run_monte_carlo_simulation(base_config, run_num)
        results.append((sim_name, cpu_s, wall_s, status))
        print(f"[{status}] {sim_name}: CPU {cpu_s:.2f}s | Wall {wall_s:.2f}s")
    
    successful = sum(1 for _, _, _, status in results if status == "OK")
    print(f"\nCompleted {successful}/{N_RUNS} successful runs")
    print(f"Results saved in {OUTPUT_DIR}")
    print("\nRun amos_vnv_summary.py to generate plots from the saved results.")


if __name__ == "__main__":
    main()
