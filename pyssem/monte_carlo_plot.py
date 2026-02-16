#!/usr/bin/env python3
"""
Plot Monte Carlo simulation results.
Reads pop_time.csv from each monte-carlo_run_* directory and creates comparison plots.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Import helper functions from amos_vnv_summary
try:
    from amos_vnv_summary import read_pop_time_csv
except ImportError:
    try:
        from pyssem.amos_vnv_summary import read_pop_time_csv
    except ImportError:
        # Fallback: define locally if import fails
        def read_pop_time_csv(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            assert {"Species","Year","Population"}.issubset(df.columns), f"Missing cols in {path}"
            df["Species"] = df["Species"].astype(str)
            df["Year"] = df["Year"].astype(int)
            df["Population"] = df["Population"].astype(float)
            return df

# Configuration
MC_RESULTS_DIR = Path(os.environ.get("MC_RESULTS_DIR", "monte_carlo_results"))
OUTPUT_DIR = Path(os.environ.get("MC_PLOT_OUTPUT", MC_RESULTS_DIR))


def plot_monte_carlo_runs(mc_results_dir: Path, out_dir: Path):
    """
    Plot all Monte Carlo runs from monte_carlo_results directory.
    Reads pop_time.csv from each run folder and plots S, N, D, B on one graph.
    """
    if not mc_results_dir.exists():
        print(f"[WARN] Monte Carlo results directory not found: {mc_results_dir}")
        return
    
    # Find all monte-carlo run directories
    run_dirs = sorted([d for d in mc_results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("monte-carlo_run_")])
    
    if not run_dirs:
        print(f"[WARN] No monte-carlo run directories found in {mc_results_dir}")
        return
    
    print(f"Found {len(run_dirs)} Monte Carlo runs")
    
    # Load data from each run
    all_runs_data = []
    target_species = ['S', 'N', 'D', 'B']
    
    for run_dir in run_dirs:
        pop_time_path = run_dir / "pop_time.csv"
        if not pop_time_path.exists():
            print(f"[WARN] pop_time.csv not found in {run_dir.name}")
            continue
        
        try:
            df = read_pop_time_csv(pop_time_path)
            
            # Extract run number from directory name
            run_num = int(run_dir.name.split("_")[-1])
            
            # Get years
            years = sorted(df["Year"].unique())
            
            # Extract data for target species
            run_data = {
                'run': run_num,
                'years': np.array(years),
                'species_data': {}
            }
            
            for species in target_species:
                sp_df = df[df["Species"] == species]
                if not sp_df.empty:
                    # Sum across all sub-species if there are multiple
                    pop_by_year = sp_df.groupby("Year")["Population"].sum()
                    run_data['species_data'][species] = pop_by_year.reindex(years, fill_value=0.0).values
            
            all_runs_data.append(run_data)
        except Exception as e:
            print(f"[WARN] Error loading {run_dir.name}: {e}")
            continue
    
    if not all_runs_data:
        print("[WARN] No valid Monte Carlo run data found")
        return
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for ax, species in zip(axes, target_species):
        # Plot each run
        for run_data in all_runs_data:
            if species in run_data['species_data']:
                ax.plot(run_data['years'], run_data['species_data'][species], 
                       alpha=0.6, linewidth=1.5, label=f"Run {run_data['run']}")
        
        ax.set_title(f"Species {species}")
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    
    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    
    plt.suptitle(f"Monte Carlo Runs: Population Over Time ({len(all_runs_data)} runs)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_png = out_dir / "monte_carlo_runs_comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"📊 Wrote {out_png}")


def main():
    """Main function."""
    print("\n=== Monte Carlo Plotting ===")
    print(f"MC_RESULTS_DIR: {MC_RESULTS_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print("============================\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_monte_carlo_runs(MC_RESULTS_DIR, OUTPUT_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
