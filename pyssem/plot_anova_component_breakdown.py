#!/usr/bin/env python3
"""
ANOVA on composite score components (WAPE, Spatial, Final, Shape).

Reads outputs from amos_vnv_summary (all_scenario_scores_with_cpu.csv and
all_group_metrics.csv), builds scenario-level component scores, and produces
ANOVA share heatmaps for model_type / n_shells / n_species effects on:
  - Overall Score (composite)
  - WAPE (magnitude)
  - Spatial
  - Final (final state)
  - Shape

Usage:
  python -m pyssem.plot_anova_component_breakdown [OUT_DIR]
  or set SSEM_OUT to output directory (default: SSEM_ROOT).
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

# Import from amos_vnv_summary so we reuse same weights and ANOVA logic
from pyssem.amos_vnv_summary import (
    add_factor_columns,
    build_scenario_component_table,
    _anova_share,
    _anova_absolute_two_factors,
    mc_species_weights,
    GROUPS,
)


def _load_data(out_dir: Path, root_dir: Path | None, mc_dir: Path | None):
    """Load scores, group metrics, and add factors + scenario components."""
    scores_path = out_dir / "all_scenario_scores_with_cpu.csv"
    groups_path = out_dir / "all_group_metrics.csv"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path} (run amos_vnv_summary first)")
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path} (run amos_vnv_summary first)")

    scores_df = pd.read_csv(scores_path)
    all_group_df = pd.read_csv(groups_path)

    scores_df = add_factor_columns(scores_df)
    factors = scores_df[
        ["simulation_name", "model_type", "n_shells", "n_species", "Score_scenario", "cpu_seconds"]
    ].copy()
    factors = factors.dropna(subset=["model_type", "n_shells", "n_species"]).drop_duplicates()

    if factors.empty:
        raise ValueError("No rows with model_type, n_shells, n_species after dropna")

    # Species weights for scenario-level component aggregation
    if mc_dir is not None and mc_dir.exists():
        try:
            species_w = mc_species_weights(mc_dir)
        except Exception:
            species_w = {g: 1.0 / len(GROUPS) for g in GROUPS}
    else:
        species_w = {g: 1.0 / len(GROUPS) for g in GROUPS}

    scen_components = build_scenario_component_table(all_group_df, species_w)
    # Merge so we have one row per scenario with factors and all outcome columns
    merged = factors.merge(
        scen_components[["simulation_name", "comp_WAPE", "comp_SPAT", "comp_FINAL", "comp_SHAPE"]],
        on="simulation_name",
        how="inner",
    )
    return merged, factors


def plot_anova_component_shares_heatmap(merged: pd.DataFrame, out_dir: Path) -> None:
    """
    Compact heatmap: rows = model type, # shells, # species;
    columns = Overall Score, WAPE, Spatial, Final, Shape.
    Values = share of variance (partial eta² normalized to sum=1).
    """
    factors_idx = merged.set_index("simulation_name")[["model_type", "n_shells", "n_species"]]
    grid_cols = []

    for col_label, y_name in [
        ("Overall Score", "Score_scenario"),
        ("WAPE", "comp_WAPE"),
        ("Spatial", "comp_SPAT"),
        ("Final", "comp_FINAL"),
        ("Shape", "comp_SHAPE"),
    ]:
        y = merged.set_index("simulation_name")[y_name]
        sh = _anova_share(y=y, X=factors_idx)
        grid_cols.append((col_label, sh))

    mat = {}
    for col_name, d in grid_cols:
        mat[col_name] = [d["model_type"], d["n_shells"], d["n_species"]]
    grid = pd.DataFrame(mat, index=["model type", "# shells", "# species"])

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(grid.values.astype(float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, fontsize=11)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=11)

    def _txt_color(val):
        if not np.isfinite(val):
            return "black"
        r, g, b, _ = im.cmap(im.norm(np.clip(val, 0, 1)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if lum >= 0.58 else "white"

    V = grid.values.astype(float)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=10, color=_txt_color(v))
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=10, color="black")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of variance explained (partial $\\eta^2$)")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "25%", "50%", "75%", "100%"])

    plt.tight_layout()
    fig.savefig(out_dir / "importance_anova_component_shares.png", dpi=220)
    plt.close(fig)
    print(f"📊 Wrote {out_dir / 'importance_anova_component_shares.png'}")


def plot_anova_component_model_type_breakdown(merged: pd.DataFrame, out_dir: Path) -> None:
    """
    Four heatmaps (like importance_anova_model_type_breakdown):
    1) Model type row: partial eta² for each of Score, WAPE, Spatial, Final, Shape.
    2)–4) Circular / Fragment Spreading / Elliptical: shells & species (absolute) for each outcome.
    """
    factors = merged.set_index("simulation_name")
    col_order = ["Score", "WAPE", "Spatial", "Final", "Shape"]
    outcome_to_col = {
        "Score": "Score_scenario",
        "WAPE": "comp_WAPE",
        "Spatial": "comp_SPAT",
        "Final": "comp_FINAL",
        "Shape": "comp_SHAPE",
    }

    # Model type row: share of variance due to model_type for each outcome
    model_type_row = {}
    for outcome in col_order:
        y = merged.set_index("simulation_name")[outcome_to_col[outcome]]
        sh = _anova_share(y=y, X=factors[["model_type", "n_shells", "n_species"]])
        model_type_row[outcome] = sh["model_type"]

    # Per model type: shells and species (absolute partial eta²) for each outcome
    model_types = ["circular", "frag_spread", "elliptical"]
    shells_by_mt = {mt: {} for mt in model_types}
    species_by_mt = {mt: {} for mt in model_types}

    for outcome in col_order:
        y_data = merged.set_index("simulation_name")[outcome_to_col[outcome]]
        factors_subset = factors[["model_type", "n_shells", "n_species"]]

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
            absolute_shares = _anova_absolute_two_factors(y_subset, X_subset)
            shells_by_mt[mt][outcome] = absolute_shares["n_shells"]
            species_by_mt[mt][outcome] = absolute_shares["n_species"]

    model_type_display = {
        "circular": "Circular",
        "frag_spread": "Fragment Spreading",
        "elliptical": "Elliptical",
    }

    def _draw_heatmap(ax, grid: pd.DataFrame, title: str, font_title=18, font_axis=16, font_cell=14):
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
                    ax.text(
                        j, i, text_str, ha="center", va="center", fontsize=font_cell,
                        color=_txt_color(v), fontweight="bold",
                    )
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=font_cell, color="black", fontweight="bold")
        return im

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"wspace": 0.5, "hspace": -0.4})

    row1 = pd.DataFrame(
        [[model_type_row.get(c, np.nan) for c in col_order]],
        index=["Model type"],
        columns=col_order,
    )
    im1 = _draw_heatmap(axes[0, 0], row1, "Model type")

    for mt, ax in [("circular", axes[0, 1]), ("frag_spread", axes[1, 0]), ("elliptical", axes[1, 1])]:
        grid_mt = pd.DataFrame(
            [
                [shells_by_mt[mt].get(c, np.nan) for c in col_order],
                [species_by_mt[mt].get(c, np.nan) for c in col_order],
            ],
            index=["Shells", "Species"],
            columns=col_order,
        )
        _draw_heatmap(ax, grid_mt, model_type_display[mt])

    plt.tight_layout(rect=[0, 0, 0.95, 1], h_pad=-2.0, w_pad=0.5)
    cbar = fig.colorbar(im1, ax=axes, pad=0.02, shrink=0.33)
    cbar.set_label("Variance explained (partial $\\eta^2$)", fontsize=16)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(out_dir / "importance_anova_component_model_type_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Wrote {out_dir / 'importance_anova_component_model_type_breakdown.png'}")


def main() -> None:
    out_dir = Path(os.environ.get("SSEM_OUT", os.environ.get("SSEM_ROOT", "./grid_search"))).expanduser().resolve()
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1]).expanduser().resolve()

    root_dir = Path(os.environ.get("SSEM_ROOT", out_dir)).expanduser().resolve()
    mc_dir = root_dir.parent / "mocat" if (root_dir.parent / "mocat").exists() else None
    if os.environ.get("SSEM_MC"):
        mc_dir = Path(os.environ.get("SSEM_MC")).expanduser().resolve()

    if not out_dir.exists():
        raise FileNotFoundError(f"OUT_DIR does not exist: {out_dir}")

    merged, _ = _load_data(out_dir, root_dir, mc_dir)
    plot_anova_component_shares_heatmap(merged, out_dir)
    plot_anova_component_model_type_breakdown(merged, out_dir)
    print("Done. Component ANOVA plots written to", out_dir)


if __name__ == "__main__":
    main()
