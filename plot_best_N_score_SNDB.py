"""
Standalone script: 2x2 S,N,D,B comparison using the best scenario per model_type
selected by N species score only (lowest Score_group for Group=='N').

Output: best_N_score_grouped_population_comparison_SNDB.png

Run after amos_vnv_summary.py so all_scenario_scores*.csv and all_group_metrics.csv exist.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyssem.amos_vnv_summary import (
    add_factor_columns,
    load_mc_species_time_series,
    read_pop_time_csv,
)


def get_best_by_N_score(scores_df: pd.DataFrame, all_group_df: pd.DataFrame) -> dict[str, str]:
    """
    For each model_type (elliptical, frag_spread, circular), return the simulation_name
    with the lowest N species score (Score_group where Group == 'N').
    """
    if all_group_df.empty or "Group" not in all_group_df.columns or "Score_group" not in all_group_df.columns:
        return {}
    n_scores = all_group_df[all_group_df["Group"] == "N"][["simulation_name", "Score_group"]].copy()
    if n_scores.empty:
        return {}
    # Merge with model_type
    df = n_scores.merge(
        scores_df[["simulation_name", "model_type"]].drop_duplicates(),
        on="simulation_name",
        how="inner",
    )
    best = {}
    for model_type in ["elliptical", "frag_spread", "circular"]:
        sub = df[df["model_type"] == model_type]
        if sub.empty:
            continue
        idx = sub["Score_group"].idxmin()
        best[model_type] = sub.loc[idx, "simulation_name"]
        print(f"Best by N score {model_type}: {best[model_type]} (N Score_group={sub.loc[idx, 'Score_group']:.4f})")
    return best


def plot_best_N_score_SNDB(
    scores_df: pd.DataFrame,
    all_group_df: pd.DataFrame,
    root_dir: Path,
    mc_dir: Path,
    out_dir: Path,
) -> None:
    """
    Same layout as best_all_models_grouped_population_comparison_SNDB.png (2x2 S, N, D, B),
    but the three SSEM lines per panel are the scenarios with best N score per model_type,
    not best overall score.
    """
    if scores_df.empty or "model_type" not in scores_df.columns:
        print("[WARN] Cannot create best-N-score SNDB plot: missing model_type")
        return
    best_scenarios = get_best_by_N_score(scores_df, all_group_df)
    if not best_scenarios:
        print("[WARN] No best-by-N scenarios found (need all_group_metrics.csv with Group N)")
        return

    # Load MC data for S, N, D, B
    mc_data = {}
    for species in ["S", "N", "D", "B"]:
        result = load_mc_species_time_series(mc_dir, species)
        if result is not None:
            mc_data[species] = result
        else:
            print(f"[WARN] MC data not found for species {species}")
    if not mc_data:
        print("[WARN] No MC data found")
        return

    # Load SSEM data for the best-by-N scenarios
    ssem_data = {}
    for model_type, sim_name in best_scenarios.items():
        sim_dir = root_dir / sim_name
        ssem_pop_path = sim_dir / "pop_time.csv"
        if not ssem_pop_path.exists():
            print(f"[WARN] pop_time.csv not found for {sim_name}")
            continue
        ssem_df = read_pop_time_csv(ssem_pop_path)
        pivot_ssem = (
            ssem_df.pivot_table(
                index="Year",
                columns="Species",
                values="Population",
                aggfunc="sum",
            ).fillna(0.0)
        )
        ssem_data[model_type] = {
            "pivot": pivot_ssem,
            "years": np.array(sorted(pivot_ssem.index.values), dtype=int),
            "sim_name": sim_name,
        }
    if not ssem_data:
        print("[WARN] No SSEM data loaded")
        return

    model_colors = {
        "elliptical": "#1f77b4",
        "frag_spread": "#2ca02c",
        "circular": "#d62728",
    }
    species_groups = ["S", "N", "D", "B"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for idx, (ax, group) in enumerate(zip(axes, species_groups)):
        if group in mc_data:
            years_mc, mean_mc, std_mc = mc_data[group]
            ax.plot(
                years_mc, mean_mc, "--",
                label="MC (mean)" if idx == 0 else "",
                linewidth=2, color="black", alpha=0.7,
            )
            ax.fill_between(
                years_mc, mean_mc - std_mc, mean_mc + std_mc,
                alpha=0.2, color="black", label="MC ±1σ" if idx == 0 else "",
            )
        for model_type, data in ssem_data.items():
            years = data["years"]
            pivot_ssem = data["pivot"]
            if group in pivot_ssem.columns:
                ax.plot(
                    years,
                    pivot_ssem[group].reindex(years, fill_value=0.0),
                    label=f"SSEM {model_type}",
                    linewidth=2,
                    color=model_colors.get(model_type, "#9467bd"),
                )
        ax.set_title(group)
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")

    best_names = [f"{mt}: {data['sim_name']}" for mt, data in ssem_data.items()]
    title = (
        "SSEM vs MOCAT-MC: Grouped Population Over Time (S, N, D, B)\n"
        "Best scenarios by N score only: " + ", ".join(best_names)
    )
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = out_dir / "best_N_score_grouped_population_comparison_SNDB.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"📊 Wrote {out_png}")


def main() -> None:
    ROOT_DIR_DEFAULT = "./grid_search"
    MC_DIR_DEFAULT = "/Users/indigobrownhall/Code/pyssem/pyssem/VnV"
    OUT_DIR_DEFAULT = None

    ROOT_DIR = Path(os.environ.get("SSEM_ROOT", ROOT_DIR_DEFAULT)).expanduser().resolve()
    MC_DIR = Path(os.environ.get("SSEM_MC", MC_DIR_DEFAULT)).expanduser().resolve()
    OUT_DIR = Path(
        os.environ.get("SSEM_OUT", str(ROOT_DIR if OUT_DIR_DEFAULT is None else OUT_DIR_DEFAULT))
    ).expanduser().resolve()

    print("\n=== Best-by-N-score SNDB plot ===")
    print(f"ROOT_DIR : {ROOT_DIR}")
    print(f"MC_DIR   : {MC_DIR}")
    print(f"OUT_DIR  : {OUT_DIR}\n")

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Root folder not found: {ROOT_DIR}")
    if not MC_DIR.exists():
        raise FileNotFoundError(f"MC folder not found: {MC_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores_path = OUT_DIR / "all_scenario_scores_with_cpu.csv"
    if not scores_path.exists():
        scores_path = OUT_DIR / "all_scenario_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"No all_scenario_scores*.csv in {OUT_DIR}. Run amos_vnv_summary.py first."
        )

    groups_path = OUT_DIR / "all_group_metrics.csv"
    if not groups_path.exists():
        raise FileNotFoundError(
            f"No all_group_metrics.csv in {OUT_DIR}. Run amos_vnv_summary.py first."
        )

    scores_df = pd.read_csv(scores_path)
    scores_df = add_factor_columns(scores_df)
    all_group_df = pd.read_csv(groups_path)

    plot_best_N_score_SNDB(scores_df, all_group_df, ROOT_DIR, MC_DIR, OUT_DIR)


if __name__ == "__main__":
    main()
