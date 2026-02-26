#!/usr/bin/env python3
"""
ANOVA model-type breakdown with composite sub-metrics as extra columns.

Produces the same 2x2 heatmap layout as importance_anova_model_type_breakdown.png
but with columns: N, D, B, WAPE, Spatial, Final, Shape, Score, CPU-s
(variance explained by model type / shells / species for each).

Usage:
  python -m pyssem.plot_anova_breakdown_with_components [OUT_DIR]
  or set SSEM_OUT (default: SSEM_ROOT).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from pyssem.amos_vnv_summary import (
    add_factor_columns,
    build_scenario_component_table,
    _anova_share,
    _anova_absolute_two_factors,
    mc_species_weights,
    GROUPS,
)


def _load_data(out_dir: Path, mc_dir: Path | None):
    """Load scores, group metrics; return (scores_df with factors, all_group_df, species_w)."""
    scores_path = out_dir / "all_scenario_scores_with_cpu.csv"
    groups_path = out_dir / "all_group_metrics.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path} (run amos_vnv_summary first)")
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path} (run amos_vnv_summary first)")

    scores_df = pd.read_csv(scores_path)
    all_group_df = pd.read_csv(groups_path)
    scores_df = add_factor_columns(scores_df)

    if mc_dir is not None and mc_dir.exists():
        try:
            species_w = mc_species_weights(mc_dir)
        except Exception:
            species_w = {g: 1.0 / len(GROUPS) for g in GROUPS}
    else:
        species_w = {g: 1.0 / len(GROUPS) for g in GROUPS}

    return scores_df, all_group_df, species_w


def plot_anova_breakdown_with_components(
    scores_df: pd.DataFrame,
    all_group_df: pd.DataFrame,
    species_w: dict,
    out_dir: Path,
    out_filename: str = "importance_anova_model_type_breakdown_with_components.png",
) -> None:
    """
    Same 2x2 layout as importance_anova_model_type_breakdown:
    columns = N, D, B, WAPE, Spatial, Final, Shape, Score, CPU-s.
    """
    col_order = ["N", "D", "B", "WAPE", "Spatial", "Final", "Shape", "Score", "CPU-s"]
    col_labels = ["N", "D", "B", "WAPE", "Spatial", "Final", "Shape", "Final Score", "CPU-s"]

    factors = scores_df[
        ["simulation_name", "model_type", "n_shells", "n_species", "Score_scenario", "cpu_seconds"]
    ].copy()
    factors = factors.dropna(subset=["model_type", "n_shells", "n_species"]).drop_duplicates()
    factors_idx = factors.set_index("simulation_name")

    scen_components = build_scenario_component_table(all_group_df, species_w)
    merged = factors.merge(
        scen_components[["simulation_name", "comp_WAPE", "comp_SPAT", "comp_FINAL", "comp_SHAPE"]],
        on="simulation_name",
        how="inner",
    )
    merged = merged.dropna(subset=["model_type", "n_shells", "n_species"])
    merged_idx = merged.set_index("simulation_name")

    def get_y_and_factors(outcome: str):
        if outcome == "Score":
            return factors_idx["Score_scenario"], factors_idx[["model_type", "n_shells", "n_species"]]
        if outcome == "CPU-s":
            return factors_idx["cpu_seconds"], factors_idx[["model_type", "n_shells", "n_species"]]
        if outcome == "WAPE":
            return merged_idx["comp_WAPE"], merged_idx[["model_type", "n_shells", "n_species"]]
        if outcome == "Spatial":
            return merged_idx["comp_SPAT"], merged_idx[["model_type", "n_shells", "n_species"]]
        if outcome == "Final":
            return merged_idx["comp_FINAL"], merged_idx[["model_type", "n_shells", "n_species"]]
        if outcome == "Shape":
            return merged_idx["comp_SHAPE"], merged_idx[["model_type", "n_shells", "n_species"]]
        # N, D, B
        sub = all_group_df[all_group_df["Group"] == outcome][["simulation_name", "Score_group"]].dropna()
        if sub.empty:
            return None, None
        m = sub.merge(factors[["simulation_name", "model_type", "n_shells", "n_species"]], on="simulation_name", how="inner")
        if m.empty:
            return None, None
        y = m.set_index("simulation_name")["Score_group"]
        X = m.set_index("simulation_name")[["model_type", "n_shells", "n_species"]]
        return y, X

    # Model type row (partial eta² for model_type)
    model_type_row = {}
    for outcome in col_order:
        y, X = get_y_and_factors(outcome)
        if y is None or X is None:
            model_type_row[outcome] = np.nan
            continue
        sh = _anova_share(y=y, X=X)
        model_type_row[outcome] = sh["model_type"]

    # Per model type: shells and species (absolute partial eta²)
    model_types = ["circular", "frag_spread", "elliptical"]
    shells_by_mt = {mt: {} for mt in model_types}
    species_by_mt = {mt: {} for mt in model_types}

    for outcome in col_order:
        y_data, factors_subset = get_y_and_factors(outcome)
        if y_data is None or factors_subset is None:
            for mt in model_types:
                shells_by_mt[mt][outcome] = np.nan
                species_by_mt[mt][outcome] = np.nan
            continue

        for mt in model_types:
            mt_mask = factors_subset["model_type"] == mt
            if mt_mask.sum() < 3:
                shells_by_mt[mt][outcome] = np.nan
                species_by_mt[mt][outcome] = np.nan
                continue
            y_subset = y_data[mt_mask]
            X_subset = factors_subset.loc[mt_mask, ["n_shells", "n_species"]]
            if len(y_subset) < 3:
                shells_by_mt[mt][outcome] = np.nan
                species_by_mt[mt][outcome] = np.nan
                continue
            abs_shares = _anova_absolute_two_factors(y_subset, X_subset)
            shells_by_mt[mt][outcome] = abs_shares["n_shells"]
            species_by_mt[mt][outcome] = abs_shares["n_species"]

    model_type_display = {"circular": "Circular", "frag_spread": "Fragment Spreading", "elliptical": "Elliptical"}

    def _draw_heatmap(ax, grid: pd.DataFrame, title: str, font_title=28, font_axis=24, font_cell=20):
        V = np.asarray(grid.values, dtype=float)
        im = ax.imshow(V, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels(grid.columns, fontsize=font_axis, rotation=45, ha="right")
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels(grid.index, fontsize=font_axis)
        ax.set_title(title, fontsize=font_title, fontweight="bold")

        def _txt_color(val):
            if not np.isfinite(val):
                return "black"
            r, g, b, _ = im.cmap(im.norm(np.clip(val, 0, 1)))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if lum >= 0.58 else "white"

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                v = V[i, j]
                if np.isfinite(v):
                    pct_int = int(round(100 * v))
                    text_str = "<1%" if pct_int < 1 else f"{pct_int}%"
                    ax.text(j, i, text_str, ha="center", va="center", fontsize=font_cell, color=_txt_color(v), fontweight="bold")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=font_cell, color="black", fontweight="bold")
        return im

    # Vertically stacked: Model type, then Circular, Fragment Spreading, Elliptical
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), gridspec_kw={"hspace": 0.12})
    # Same width and uniform gap for all panels
    fig.subplots_adjust(left=0.12, right=0.88, top=0.96, bottom=0.06)

    row1 = pd.DataFrame(
        [[model_type_row.get(c, np.nan) for c in col_order]],
        index=["Model type"],
        columns=col_labels,
    )
    im1 = _draw_heatmap(axes[0], row1, "Model type")

    for i, mt in enumerate(["circular", "frag_spread", "elliptical"], start=1):
        grid_mt = pd.DataFrame(
            [
                [shells_by_mt[mt].get(c, np.nan) for c in col_order],
                [species_by_mt[mt].get(c, np.nan) for c in col_order],
            ],
            index=["Shells", "Species"],
            columns=col_labels,
        )
        _draw_heatmap(axes[i], grid_mt, model_type_display[mt])

    # Remove x-axis labels on top plot and middle two panels (only Elliptical shows x labels)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticklabels([])

    cbar = fig.colorbar(im1, ax=axes, pad=0.02, shrink=0.4, location="right")
    cbar.set_label("Variance explained (partial $\\eta^2$)", fontsize=28)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=24)

    out_path = out_dir / out_filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Wrote {out_path}")


def main() -> None:
    out_dir = Path(os.environ.get("SSEM_OUT", os.environ.get("SSEM_ROOT", "./grid_search"))).expanduser().resolve()
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1]).expanduser().resolve()

    mc_dir = None
    if os.environ.get("SSEM_MC"):
        mc_dir = Path(os.environ.get("SSEM_MC")).expanduser().resolve()

    if not out_dir.exists():
        raise FileNotFoundError(f"OUT_DIR does not exist: {out_dir}")

    scores_df, all_group_df, species_w = _load_data(out_dir, mc_dir)
    plot_anova_breakdown_with_components(scores_df, all_group_df, species_w, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
