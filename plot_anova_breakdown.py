#!/usr/bin/env python3
"""
Standalone script to generate ANOVA model type breakdown plot.
Can be run independently without re-running the full amos_vnv_summary.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _anova_share(y, X):
    """Return dict of partial-eta-squared for the 3 factors, normalized to sum=1.
    y: pandas Series with name (e.g., Score_scenario); X: DataFrame with columns
       model_type, n_shells, n_species (model_type treated categorically).
    """
    import pandas as _pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = _pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return {"model_type": np.nan, "n_shells": np.nan, "n_species": np.nan}

    model = ols("y ~ C(model_type) + n_shells + n_species",
                data=df.rename(columns={y.name: "y"})).fit()
    a = anova_lm(model, typ=2)
    if "sum_sq" not in a.columns or "Residual" not in a.index:
        return {"model_type": np.nan, "n_shells": np.nan, "n_species": np.nan}

    ss_res = float(a.loc["Residual", "sum_sq"])
    a = a.drop(index="Residual", errors="ignore")
    a["partial_eta_sq"] = a["sum_sq"] / (a["sum_sq"] + ss_res)

    out = {
        "model_type": float(a.loc["C(model_type)", "partial_eta_sq"]) if "C(model_type)" in a.index else np.nan,
        "n_shells":   float(a.loc["n_shells",       "partial_eta_sq"]) if "n_shells"       in a.index else np.nan,
        "n_species":  float(a.loc["n_species",      "partial_eta_sq"]) if "n_species"      in a.index else np.nan,
    }
    vals = np.array([out["model_type"], out["n_shells"], out["n_species"]], float)
    s = np.nansum(vals)
    if s > 0:
        vals = vals / s
    out["model_type"], out["n_shells"], out["n_species"] = vals
    return out


def _anova_absolute_two_factors(y, X):
    """Return dict of absolute partial-eta-squared for n_shells and n_species (NOT normalized).
    Also returns total explained variance and residual.
    Used when model_type is already filtered out (subset of data).
    y: pandas Series; X: DataFrame with columns n_shells, n_species.
    Returns: {"n_shells": float, "n_species": float, "total": float, "residual": float}
    """
    import pandas as _pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = _pd.concat([y, X], axis=1).dropna()
    if df.empty or len(df) < 3:
        return {"n_shells": np.nan, "n_species": np.nan, "total": np.nan, "residual": np.nan}

    model = ols("y ~ n_shells + n_species",
                data=df.rename(columns={y.name: "y"})).fit()
    try:
        a = anova_lm(model, typ=2)
        if "sum_sq" not in a.columns or "Residual" not in a.index:
            return {"n_shells": np.nan, "n_species": np.nan, "total": np.nan, "residual": np.nan}

        ss_res = float(a.loc["Residual", "sum_sq"])
        ss_total = float(a["sum_sq"].sum())
        
        # Partial eta-squared: SS_factor / (SS_factor + SS_residual)
        ss_shells = float(a.loc["n_shells", "sum_sq"]) if "n_shells" in a.index else 0.0
        ss_species = float(a.loc["n_species", "sum_sq"]) if "n_species" in a.index else 0.0
        
        # Absolute partial eta-squared (not normalized)
        eta_shells = ss_shells / (ss_shells + ss_res) if (ss_shells + ss_res) > 0 else 0.0
        eta_species = ss_species / (ss_species + ss_res) if (ss_species + ss_res) > 0 else 0.0
        
        # Total variance explained by both factors together
        total_explained = float(model.rsquared) if hasattr(model, 'rsquared') else np.nan
        
        # Residual (unexplained)
        residual = 1.0 - total_explained if np.isfinite(total_explained) else np.nan
        
        return {
            "n_shells": eta_shells,
            "n_species": eta_species,
            "total": total_explained,
            "residual": residual
        }
    except Exception:
        return {"n_shells": np.nan, "n_species": np.nan, "total": np.nan, "residual": np.nan}


def plot_anova_model_type_breakdown(scores_df: pd.DataFrame, all_group_df: pd.DataFrame, out_dir: Path):
    """
    Create 4 heatmaps showing ABSOLUTE variance explained in a 2x2 grid with square cells:
    - X-axis always: N, D, B, Score, CPU-s
    - All values on same 0-100% scale (absolute partial eta-squared)
    1. Top-left: Model type (1 row)
    2. Top-right: Circular (2 rows: Shells, Species)
    3. Bottom-left: Fragment Spreading (2 rows: Shells, Species)
    4. Bottom-right: Elliptical (2 rows: Shells, Species)
    Saves PNG in out_dir as importance_anova_model_type_breakdown.png
    """
    import pandas as _pd
    import matplotlib.pyplot as plt

    # Fixed column order for all heatmaps
    col_order = ["N", "D", "B", "Score", "CPU-s"]

    # Factor table
    factors = scores_df[["simulation_name","model_type","n_shells","n_species","Score_scenario","cpu_seconds"]].copy()
    factors = factors.dropna(subset=["model_type","n_shells","n_species"]).drop_duplicates()

    # Build model_type row in col_order
    model_type_row = {}
    for outcome in col_order:
        if outcome == "Score":
            sh = _anova_share(
                y=factors.set_index("simulation_name")["Score_scenario"],
                X=factors.set_index("simulation_name")[["model_type","n_shells","n_species"]],
            )
        elif outcome == "CPU-s":
            sh = _anova_share(
                y=factors.set_index("simulation_name")["cpu_seconds"],
                X=factors.set_index("simulation_name")[["model_type","n_shells","n_species"]],
            )
        else:
            sub = all_group_df[all_group_df["Group"]==outcome][["simulation_name","Score_group"]].dropna()
            if sub.empty:
                model_type_row[outcome] = np.nan
                continue
            merged = sub.merge(factors[["simulation_name","model_type","n_shells","n_species"]],
                               on="simulation_name", how="inner")
            if merged.empty:
                model_type_row[outcome] = np.nan
                continue
            sh = _anova_share(
                y=merged.set_index("simulation_name")["Score_group"].rename("y"),
                X=merged.set_index("simulation_name")[["model_type","n_shells","n_species"]],
            )
        model_type_row[outcome] = sh["model_type"]

    # For each model type: compute # shells and # species per outcome (ABSOLUTE values)
    model_types = ["circular", "frag_spread", "elliptical"]
    shells_by_mt = {mt: {} for mt in model_types}
    species_by_mt = {mt: {} for mt in model_types}

    for outcome in col_order:
        if outcome == "Score":
            y_data = factors.set_index("simulation_name")["Score_scenario"]
            factors_subset = factors.set_index("simulation_name")[["model_type","n_shells","n_species"]]
        elif outcome == "CPU-s":
            y_data = factors.set_index("simulation_name")["cpu_seconds"]
            factors_subset = factors.set_index("simulation_name")[["model_type","n_shells","n_species"]]
        else:
            sub = all_group_df[all_group_df["Group"]==outcome][["simulation_name","Score_group"]].dropna()
            if sub.empty:
                for mt in model_types:
                    shells_by_mt[mt][outcome] = np.nan
                    species_by_mt[mt][outcome] = np.nan
                continue
            merged = sub.merge(factors[["simulation_name","model_type","n_shells","n_species"]],
                               on="simulation_name", how="inner")
            if merged.empty:
                for mt in model_types:
                    shells_by_mt[mt][outcome] = np.nan
                    species_by_mt[mt][outcome] = np.nan
                continue
            y_data = merged.set_index("simulation_name")["Score_group"]
            factors_subset = merged.set_index("simulation_name")[["model_type","n_shells","n_species"]]

        for mt in model_types:
            mt_mask = factors_subset["model_type"] == mt
            if mt_mask.sum() < 3:
                shells_by_mt[mt][outcome] = np.nan
                species_by_mt[mt][outcome] = np.nan
                continue
            y_subset = y_data[mt_mask]
            X_subset = factors_subset.loc[mt_mask, ["n_shells","n_species"]]
            if len(y_subset) < 3:
                shells_by_mt[mt][outcome] = np.nan
                species_by_mt[mt][outcome] = np.nan
                continue
            # Use absolute values, not normalized
            absolute_shares = _anova_absolute_two_factors(y_subset, X_subset)
            shells_by_mt[mt][outcome] = absolute_shares["n_shells"]
            species_by_mt[mt][outcome] = absolute_shares["n_species"]

    # Display labels for publication (cleaned, capitals, Fragment Spreading)
    col_labels = ["N", "D", "B", "Score", "CPU-s"]
    model_type_display = {"circular": "Circular", "frag_spread": "Fragment Spreading", "elliptical": "Elliptical"}

    def _draw_heatmap(ax, grid: _pd.DataFrame, title: str, font_title=18, font_axis=16, font_cell=14):
        """Draw one heatmap - let matplotlib handle sizing."""
        V = np.asarray(grid.values, dtype=float)
        
        # Let matplotlib handle aspect ratio automatically
        im = ax.imshow(V, vmin=0.0, vmax=1.0, cmap="viridis", interpolation='nearest')
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels(grid.columns, fontsize=font_axis, rotation=45, ha='right')
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
                    # Convert to integer percentage
                    pct_int = int(round(100 * v))
                    # If less than 1%, show "<1%"
                    if pct_int < 1:
                        text_str = "<1%"
                    else:
                        text_str = f"{pct_int}%"
                    ax.text(j, i, text_str, ha="center", va="center", fontsize=font_cell, 
                           color=_txt_color(v), fontweight='bold')
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=font_cell, 
                           color="black", fontweight='bold')
        return im

    # Figure: 2x2 grid layout - let matplotlib handle sizing automatically
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'wspace': 0.5, 'hspace': -0.4})
    fig.suptitle("", fontsize=0)  # No main title, just spacing

    # 1) Top-left: Model type row (1 row x 5 cols)
    row1 = _pd.DataFrame(
        [[model_type_row.get(c, np.nan) for c in col_order]],
        index=["Model type"],
        columns=col_labels,
    )
    im1 = _draw_heatmap(axes[0, 0], row1, "Model type")

    # 2) Top-right: Circular
    grid_circular = _pd.DataFrame(
        [
            [shells_by_mt["circular"].get(c, np.nan) for c in col_order],
            [species_by_mt["circular"].get(c, np.nan) for c in col_order],
        ],
        index=["Shells", "Species"],
        columns=col_labels,
    )
    _draw_heatmap(axes[0, 1], grid_circular, model_type_display["circular"])

    # 3) Bottom-left: Fragment Spreading
    grid_frag = _pd.DataFrame(
        [
            [shells_by_mt["frag_spread"].get(c, np.nan) for c in col_order],
            [species_by_mt["frag_spread"].get(c, np.nan) for c in col_order],
        ],
        index=["Shells", "Species"],
        columns=col_labels,
    )
    _draw_heatmap(axes[1, 0], grid_frag, model_type_display["frag_spread"])

    # 4) Bottom-right: Elliptical
    grid_elliptical = _pd.DataFrame(
        [
            [shells_by_mt["elliptical"].get(c, np.nan) for c in col_order],
            [species_by_mt["elliptical"].get(c, np.nan) for c in col_order],
        ],
        index=["Shells", "Species"],
        columns=col_labels,
    )
    _draw_heatmap(axes[1, 1], grid_elliptical, model_type_display["elliptical"])

    # Single shared colorbar - adjust height to be shorter
    # Use tight_layout with minimal vertical padding to reduce row gaps
    plt.tight_layout(rect=[0, 0, 0.95, 1], h_pad=-2.0, w_pad=0.5)
    # Create colorbar with reduced height (shrink parameter) - increased by 10% from 0.3
    cbar = fig.colorbar(im1, ax=axes, pad=0.02, shrink=0.33)
    cbar.set_label("Variance explained (partial $\\eta^2$)", fontsize=16)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(out_dir / "importance_anova_model_type_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Wrote {out_dir / 'importance_anova_model_type_breakdown.png'}")


def main():
    """Main entry point - reads CSV files and generates plot."""
    import sys
    
    # Default paths
    if len(sys.argv) > 1:
        grid_search_dir = Path(sys.argv[1])
    else:
        grid_search_dir = Path("./grid_search")
    
    if not grid_search_dir.exists():
        print(f"Error: Directory not found: {grid_search_dir}")
        sys.exit(1)
    
    # Read required CSV files
    scores_path = grid_search_dir / "all_scenario_scores_with_cpu.csv"
    all_groups_path = grid_search_dir / "all_group_metrics.csv"
    
    if not scores_path.exists():
        print(f"Error: File not found: {scores_path}")
        print("Please run amos_vnv_summary.py first to generate the required CSV files.")
        sys.exit(1)
    
    if not all_groups_path.exists():
        print(f"Error: File not found: {all_groups_path}")
        print("Please run amos_vnv_summary.py first to generate the required CSV files.")
        sys.exit(1)
    
    # Load data
    print(f"Loading {scores_path}...")
    scores_df = pd.read_csv(scores_path)
    
    print(f"Loading {all_groups_path}...")
    all_group_df = pd.read_csv(all_groups_path)
    
    # Parse model_type, n_shells, n_species from simulation_name if not present
    if "model_type" not in scores_df.columns:
        import re
        def _parse_sim_name(name: str):
            model_type = name.split("_nshell_")[0]
            nshells = None
            nspecies = None
            m = re.search(r"_nshell_(\d+)", name)
            if m:
                try: nshells = int(m.group(1))
                except Exception: nshells = None
            m2 = re.search(r"_sp_(\d+)", name)
            if m2:
                try: nspecies = int(m2.group(1))
                except Exception: nspecies = None
            return model_type, nshells, nspecies
        
        parsed = scores_df["simulation_name"].astype(str).apply(_parse_sim_name)
        scores_df["model_type"] = parsed.apply(lambda t: t[0])
        scores_df["n_shells"] = parsed.apply(lambda t: t[1])
        scores_df["n_species"] = parsed.apply(lambda t: t[2])
    
    # Generate plot
    print("Generating ANOVA model type breakdown plot...")
    plot_anova_model_type_breakdown(scores_df, all_group_df, grid_search_dir)
    print("Done!")


if __name__ == "__main__":
    main()
