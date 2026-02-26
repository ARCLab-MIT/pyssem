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


MODEL_COLORS: dict[str, str] = {
    "elliptical": "#1f77b4",
    "frag_spread": "#2ca02c",
    "circular": "#d62728",
}


def _plot_species_on_axis(
    ax: plt.Axes,
    species: str,
    years_mc: np.ndarray,
    mean_mc: np.ndarray,
    std_mc: np.ndarray,
    scores_df: pd.DataFrame,
    root_dir: Path,
    ylim: tuple[float, float] | None = (0, 100_000),
    show_legend: bool = True,
) -> None:
    """Draw MC envelope, faint SSEM runs, and model-type means for one species on ax."""
    # MC mean and ±1σ envelope
    ax.plot(
        years_mc,
        mean_mc,
        "--",
        label="MC (mean)",
        linewidth=2,
        color="black",
        alpha=0.8,
    )
    ax.fill_between(
        years_mc,
        mean_mc - std_mc,
        mean_mc + std_mc,
        alpha=0.2,
        color="black",
        label="MC ±1σ",
    )

    first_label_for_type: dict[str, bool] = {}
    avg_data: dict[str, dict[str, object]] = {}

    for _, row in scores_df.iterrows():
        sim_name = row["simulation_name"]
        model_type = row.get("model_type", "unknown")
        sim_dir = root_dir / sim_name
        ssem_pop_path = sim_dir / "pop_time.csv"

        if not ssem_pop_path.exists():
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

        if species not in pivot_ssem.columns:
            continue

        years = np.array(sorted(pivot_ssem.index.values), dtype=int)
        series = pivot_ssem[species].reindex(years, fill_value=0.0).values.astype(float)

        if model_type not in avg_data:
            avg_data[model_type] = {
                "years": years,
                "sum": series.copy(),
                "count": 1,
            }
        else:
            avg_data[model_type]["sum"] = avg_data[model_type]["sum"] + series
            avg_data[model_type]["count"] = avg_data[model_type]["count"] + 1

        label = f"SSEM {model_type}" if not first_label_for_type.get(model_type) else ""
        first_label_for_type[model_type] = True

        ax.plot(
            years,
            series,
            label=label,
            linewidth=0.7,
            alpha=0.2,
            color=MODEL_COLORS.get(model_type, "#9467bd"),
        )

    for model_type, data in avg_data.items():
        years = data["years"]
        sum_vals = data["sum"]
        count = data["count"]
        mean_vals = sum_vals / float(count)
        ax.plot(
            years,
            mean_vals,
            label=f"SSEM {model_type} mean",
            linewidth=2.5,
            alpha=0.9,
            color=MODEL_COLORS.get(model_type, "#9467bd"),
        )

    ax.set_title(f"{species} species: SSEM vs MOCAT-MC (all scenarios)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=9)


def plot_N_species_all_models(
    scores_df: pd.DataFrame,
    root_dir: Path,
    mc_dir: Path,
    out_dir: Path,
) -> None:
    """
    Plot grouped population over time for the N species only, comparing:
      - MOCAT-MC envelope (mean ±1σ across runs)
      - All available SSEM scenarios, colored by model type

    This is similar to plot_best_scenarios_comparison within amos_vnv_summary.py
    but focuses on a single species ('N') and includes all runs, not just the best.
    """
    if scores_df.empty or "model_type" not in scores_df.columns:
        print("[WARN] Cannot create N-species plot: missing model_type column")
        return

    species = "N"
    mc_result = load_mc_species_time_series(mc_dir, species)
    if mc_result is None:
        print(f"[WARN] MC data not found for species {species} in {mc_dir}")
        return

    years_mc, mean_mc, std_mc = mc_result
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_species_on_axis(
        ax, species, years_mc, mean_mc, std_mc, scores_df, root_dir,
        ylim=(0, 100_000), show_legend=True,
    )
    plt.tight_layout()
    out_png = out_dir / "all_models_N_population_comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"📊 Wrote {out_png}")


def plot_N_D_species_all_models(
    scores_df: pd.DataFrame,
    root_dir: Path,
    mc_dir: Path,
    out_dir: Path,
) -> None:
    """
    Side-by-side plot: N species (left) and D species (right), each with
    MC envelope, faint SSEM runs, and model-type means.
    """
    if scores_df.empty or "model_type" not in scores_df.columns:
        print("[WARN] Cannot create N+D plot: missing model_type column")
        return

    mc_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sp in ("N", "D"):
        result = load_mc_species_time_series(mc_dir, sp)
        if result is None:
            print(f"[WARN] MC data not found for species {sp} in {mc_dir}")
            return
        mc_data[sp] = result

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, species in zip(axes, ("N", "D"), strict=True):
        years_mc, mean_mc, std_mc = mc_data[species]
        ylim = (0, 100_000) if species == "N" else None
        _plot_species_on_axis(
            ax,
            species,
            years_mc,
            mean_mc,
            std_mc,
            scores_df,
            root_dir,
            ylim=ylim,
            show_legend=True,
        )
    plt.suptitle("SSEM vs MOCAT-MC (all scenarios)", fontsize=12, y=1.02)
    plt.tight_layout()
    out_png = out_dir / "all_models_N_D_population_comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"📊 Wrote {out_png}")


def main() -> None:
    """
    Standalone entry point for generating the N-species comparison plot.

    Uses the same environment variables and defaults as amos_vnv_summary.main:
      - SSEM_ROOT: root folder containing scenario subfolders (default: ./grid_search)
      - SSEM_MC:   folder containing MC truth files (default: pyssem/VnV)
      - SSEM_OUT:  output folder for plots/CSVs (default: SSEM_ROOT)

    Assumes that amos_vnv_summary.main has already been run so that
    all_scenario_scores_with_cpu.csv (or all_scenario_scores.csv) exists.
    """
    ROOT_DIR_DEFAULT = "./grid_search"
    MC_DIR_DEFAULT = "/Users/indigobrownhall/Code/pyssem/pyssem/VnV"
    OUT_DIR_DEFAULT = None  # None -> defaults to ROOT_DIR

    ROOT_DIR = Path(os.environ.get("SSEM_ROOT", ROOT_DIR_DEFAULT)).expanduser().resolve()
    MC_DIR = Path(os.environ.get("SSEM_MC", MC_DIR_DEFAULT)).expanduser().resolve()
    OUT_DIR = Path(
        os.environ.get(
            "SSEM_OUT",
            str(ROOT_DIR if OUT_DIR_DEFAULT is None else OUT_DIR_DEFAULT),
        )
    ).expanduser().resolve()

    print("\n=== N-species comparison configuration ===")
    print(f"ROOT_DIR : {ROOT_DIR}")
    print(f"MC_DIR   : {MC_DIR}")
    print(f"OUT_DIR  : {OUT_DIR}")
    print("=========================================\n")

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Root folder not found: {ROOT_DIR}")
    if not MC_DIR.exists():
        raise FileNotFoundError(f"MC folder not found: {MC_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer the scores file with CPU attached; fall back to the simpler one.
    scores_path_with_cpu = OUT_DIR / "all_scenario_scores_with_cpu.csv"
    scores_path_simple = OUT_DIR / "all_scenario_scores.csv"

    if scores_path_with_cpu.exists():
        scores_path = scores_path_with_cpu
    elif scores_path_simple.exists():
        scores_path = scores_path_simple
    else:
        raise FileNotFoundError(
            f"Could not find all_scenario_scores*.csv in {OUT_DIR}. "
            "Please run amos_vnv_summary.py first to generate scenario scores."
        )

    scores_df = pd.read_csv(scores_path)
    scores_df = add_factor_columns(scores_df)

    plot_N_species_all_models(scores_df, ROOT_DIR, MC_DIR, OUT_DIR)
    plot_N_D_species_all_models(scores_df, ROOT_DIR, MC_DIR, OUT_DIR)


if __name__ == "__main__":
    main()

