#!/usr/bin/env python3
"""
Standalone script to create the best all models plot with component breakdowns.
This script can be run independently to iterate on the plot formatting.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path to import from amos_vnv_summary if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import necessary functions from amos_vnv_summary
from pyssem.amos_vnv_summary import (
    read_pop_time_csv,
    load_mc_species_time_series,
    add_factor_columns
)

# Composite score weights (from amos_vnv_summary)
W_WAPE  = 0.40
W_SPAT  = 0.40
W_FINAL = 0.10
W_SHAPE = 0.10

# Spatial internal weights
SPAT_CENT_W = 0.40
SPAT_TAIL_W = 0.20
SPAT_JS_W   = 0.40

# Shape mix
SHAPE_R_W   = 0.60
SHAPE_DTW_W = 0.40

# Normalization scales
CENTROID_SCALE_KM = 50.0
TAIL_SCALE        = 0.10
JS_SCALE          = 0.06
DTW_SCALE_DEFAULT = 7.0

def _decompose_components_from_row(row: pd.Series) -> dict:
    """
    Given a per-group metrics row, return component contributions BEFORE species weighting.
    Keys: comp_WAPE, comp_SPAT, comp_FINAL, comp_SHAPE
    """
    # Magnitude
    wape = float(row.get("WAPE_all", np.nan))
    comp_WAPE = W_WAPE * (wape if np.isfinite(wape) else 0.0)

    # Final state (absolute relative error)
    final = float(row.get("Final_rel_err", np.nan))
    comp_FINAL = W_FINAL * (abs(final) if np.isfinite(final) else 0.0)

    # Shape: correlation + DTW
    r = float(row.get("r_clamped", 0.0))
    dtw_n = float(row.get("DTW_norm", np.nan))
    shape_base = SHAPE_R_W*(1.0 - r) + SHAPE_DTW_W*min(dtw_n/DTW_SCALE_DEFAULT, 2.0 if np.isfinite(dtw_n) else 0.0)
    comp_SHAPE = W_SHAPE * shape_base

    # Spatial (volume-normalised)
    cent = float(row.get("AltCentroidDiff_km_ref_vol", np.nan))
    tail = float(row.get("TailFracDelta_gt1000km_vol", np.nan))
    js   = float(row.get("JSdiv_alt_smoothed_ref_vol", np.nan))
    parts, wsum = [], 0.0
    if np.isfinite(cent):
        parts.append(SPAT_CENT_W * min(abs(cent)/CENTROID_SCALE_KM, 2.0)); wsum += SPAT_CENT_W
    if np.isfinite(tail):
        parts.append(SPAT_TAIL_W * min(abs(tail)/TAIL_SCALE, 2.0));       wsum += SPAT_TAIL_W
    if np.isfinite(js):
        parts.append(SPAT_JS_W   * min(js/JS_SCALE, 2.0));                 wsum += SPAT_JS_W
    if wsum > 0:
        spatial_norm = sum(parts)/wsum
    else:
        vw = row.get("VWAPE_alt", np.nan)
        spatial_norm = float(vw) if isinstance(vw, (int,float)) and np.isfinite(vw) else 0.0
    comp_SPAT = W_SPAT * spatial_norm

    return dict(comp_WAPE=comp_WAPE, comp_SPAT=comp_SPAT, comp_FINAL=comp_FINAL, comp_SHAPE=comp_SHAPE)

def plot_best_all_models_with_components(scores_df: pd.DataFrame, all_group_df: pd.DataFrame,
                                         root_dir: Path, mc_dir: Path, out_dir: Path):
    """
    Create a plot showing the best scenario from each model type (elliptical, frag_spread, circular)
    with component scores displayed below each subplot, colored to match the line colors.
    """
    if scores_df.empty or "model_type" not in scores_df.columns:
        print("[WARN] Cannot create best all models plot: missing model_type column")
        return
    
    if all_group_df.empty:
        print("[WARN] Cannot create best all models plot: missing all_group_df")
        return
    
    # Find best scenario for each model type
    best_scenarios = {}
    for model_type in ["elliptical", "frag_spread", "circular"]:
        model_df = scores_df[scores_df["model_type"] == model_type].copy()
        if model_df.empty:
            print(f"[WARN] No scenarios found for model_type={model_type}")
            continue
        
        best = model_df.loc[model_df["Score_scenario"].idxmin()]
        best_scenarios[model_type] = {
            'name': best["simulation_name"],
            'score': best["Score_scenario"]
        }
        print(f"Best {model_type}: {best_scenarios[model_type]['name']} (score={best_scenarios[model_type]['score']:.4f})")
    
    if not best_scenarios:
        print("[WARN] No best scenarios found to plot")
        return
    
    # Load MC data
    mc_data = {}  # species -> (years, mean, std)
    for species in ['S', 'N', 'D', 'B']:
        result = load_mc_species_time_series(mc_dir, species)
        if result is not None:
            mc_data[species] = result
    
    if not mc_data:
        print(f"[WARN] No MC data found in {mc_dir}")
        return
    
    # Load SSEM data for all best scenarios
    ssem_data = {}
    for model_type, info in best_scenarios.items():
        sim_dir = root_dir / info['name']
        ssem_pop_path = sim_dir / "pop_time.csv"
        
        if not ssem_pop_path.exists():
            print(f"[WARN] pop_time.csv not found for {info['name']}")
            continue
        
        ssem_df = read_pop_time_csv(ssem_pop_path)
        pivot_ssem = ssem_df.pivot_table(
            index="Year", 
            columns="Species", 
            values="Population", 
            aggfunc="sum"
        ).fillna(0.0)
        
        ssem_data[model_type] = {
            'pivot': pivot_ssem,
            'years': np.array(sorted(pivot_ssem.index.values), dtype=int),
            'sim_name': info['name'],
            'score': info['score']
        }
    
    if not ssem_data:
        print("[WARN] No SSEM data loaded for any best scenarios")
        return
    
    # Get component breakdowns for all scenarios
    scenario_component_data = {}
    for model_type, info in best_scenarios.items():
        scenario_groups = all_group_df[all_group_df["simulation_name"] == info['name']].copy()
        if not scenario_groups.empty:
            scenario_component_data[model_type] = scenario_groups
    
    # Color palette for model types (matching original)
    model_colors = {
        'elliptical': '#1f77b4',    # Blue
        'frag_spread': '#2ca02c',    # Green
        'circular': '#d62728'        # Red
    }
    
    # Create figure with subplots and space for legend at bottom
    # Increased figure size and plot sizes, more gap between rows
    # Bottom row split into two rows: legend and summary
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.2, 0.02, 0.15], hspace=0.42, wspace=0.2)  # Slightly increased gap between rows of plots
    
    species_groups = ['S', 'N', 'D', 'B']
    species_labels = {
        'S': 'Satellites (S)',
        'N': 'Debris (N)',
        'D': 'Derelicts (D)',
        'B': 'Rocket Bodies (B)'
    }
    
    # Plot each species
    for idx, group in enumerate(species_groups):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Plot MC mean and ±1 std
        if group in mc_data:
            years_mc, mean_mc, std_mc = mc_data[group]
            ax.plot(years_mc, mean_mc, '--', label='MC (mean)', linewidth=2, color='black', alpha=0.7)
            ax.fill_between(years_mc, mean_mc - std_mc, mean_mc + std_mc,
                          alpha=0.2, color='black', label='MC ±1σ')
        
        # Plot SSEM lines for each model type
        for model_type, data in ssem_data.items():
            if group in data['pivot'].columns:
                ax.plot(data['years'], data['pivot'][group].reindex(data['years'], fill_value=0.0), 
                       label=f'SSEM {model_type}', linewidth=2, 
                       color=model_colors.get(model_type, '#9467bd'))
        
        # Set consistent font size for all text
        uniform_fontsize = 15
        
        ax.set_title(species_labels[group], fontsize=uniform_fontsize, fontweight='bold', pad=15)
        ax.set_ylabel("Population", fontsize=uniform_fontsize)
        ax.grid(True, alpha=0.3)
        # Don't add legend here - will add at bottom of figure
        
        # Set xlabel only for bottom row plots (D and B)
        if row == 1:  # Bottom row
            ax.set_xlabel("Year", fontsize=uniform_fontsize, labelpad=-1)
        
        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=uniform_fontsize)
        
        # Set specific y-axis limits for D and B plots
        if group == 'D':
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], 3500)
        elif group == 'B':
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], 1550)
        
        # Add component scores as boxes ON the graph (in upper right corner)
        component_parts = []
        for model_type in ["elliptical", "frag_spread", "circular"]:
            if model_type not in scenario_component_data:
                continue
            
            scenario_groups = scenario_component_data[model_type]
            group_row = scenario_groups[scenario_groups["Group"] == group]
            
            if not group_row.empty:
                row_data = group_row.iloc[0]
                comps = _decompose_components_from_row(row_data)
                
                # Format component text with shortened names and no model type prefix
                color = model_colors.get(model_type, '#000000')
                comp_text = (
                    f"W={comps['comp_WAPE']:.4f}, "
                    f"S={comps['comp_SPAT']:.4f}, "
                    f"F={comps['comp_FINAL']:.4f}, "
                    f"Sh={comps['comp_SHAPE']:.4f}"
                )
                component_parts.append((comp_text, color))
        
        # Display component texts directly on the graph, anchored to corners
        # No box, just text - all using axes coordinates to stay within subplot
        if component_parts:
            # Position: bottom right for S, top left for N, D, B
            if group == 'S':
                # Bottom right - anchor to bottom right corner using axes coordinates
                text_x = 0.98  # Near right edge
                text_y_start = 0.04  # Near bottom
                ha_anchor = 'right'
                va_anchor = 'bottom'
                line_spacing = 0.065  # Positive to go upward
            else:
                # Top left - anchor to top left corner
                text_x = 0.02  # Near left edge
                text_y_start = 0.98  # Near top
                ha_anchor = 'left'
                va_anchor = 'top'
                line_spacing = -0.065  # Negative to go downward
            
            # Add text lines with colors, anchored to corner
            for i, (comp_text, color) in enumerate(component_parts):
                text_y = text_y_start + i * line_spacing
                ax.text(
                    text_x, 
                    text_y, 
                    comp_text, 
                    transform=ax.transAxes,  # Use axes coordinates for all
                    ha=ha_anchor, 
                    va=va_anchor, 
                    fontsize=15,  # Uniform font size 
                    color=color, 
                    fontweight='bold',
                    zorder=10  # High zorder to be on top
                )
    
    # Add legend and summary statistics at the bottom
    # Stack them vertically to avoid overlap
    legend_ax = fig.add_subplot(gs[2, :])  # Full width for legend
    legend_ax.axis('off')
    
    summary_ax = fig.add_subplot(gs[3, :])  # Full width for summary, below legend
    summary_ax.axis('off')
    
    # Create legend handles for the plot elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='black', linewidth=2, alpha=0.7, label='MC (mean)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='black', alpha=0.2, label='MC ±1σ'),
        Line2D([0], [0], color=model_colors['elliptical'], linewidth=2, label='SSEM Elliptical'),
        Line2D([0], [0], color=model_colors['frag_spread'], linewidth=2, label='SSEM Fragment Spread'),
        Line2D([0], [0], color=model_colors['circular'], linewidth=2, label='SSEM Circular'),
    ]
    
    # Place legend at bottom, centered with more spacing
    uniform_fontsize = 15  # Same size as all other text
    legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=5, 
                              fontsize=uniform_fontsize, frameon=True, fancybox=True, shadow=True,
                              columnspacing=1.5, handletextpad=0.5)  # More spacing between legend items
    
    # Extract scores and colors for each model type
    scores_data = []
    for model_type in ["elliptical", "frag_spread", "circular"]:
        if model_type not in scenario_component_data:
            continue
        overall_score = ssem_data[model_type]['score']
        color = model_colors.get(model_type, '#000000')
        scores_data.append((overall_score, color))
    
    # Display summary below legend (no box, just text)
    if scores_data:
        uniform_fontsize = 15  # Same size as all other text
        fontsize = uniform_fontsize
        
        # Position text in center (no box)
        text_x = 0.5
        text_y = 0.5
        vertical_offset = 0.50  # Amount to move both text and scores higher
        label_score_gap = 0.15  # Gap between label and scores
        
        # Draw "Weighted Overall Score" label in black (much higher above scores)
        summary_ax.text(text_x, text_y + 0.65 + vertical_offset, "Weighted Overall Score", 
                       transform=summary_ax.transAxes,
                       ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')
        
        # Draw colored scores separated by | (below label with increased gap)
        # Calculate spacing for scores with minimal gaps
        num_scores = len(scores_data)
        total_width = 0.4  # Reduced total width to bring items closer
        pipe_gap = 0.001  # Extremely small gap between values and pipes
        
        # Distribute scores evenly with minimal gaps
        score_spacing = total_width / num_scores
        
        score_x_start = text_x - total_width/2 + score_spacing/2
        
        for i, (score_val, color) in enumerate(scores_data):
            x_pos = score_x_start + i * score_spacing
            summary_ax.text(x_pos, text_y - 0.05 + vertical_offset - label_score_gap, f"{score_val:.4f}", 
                           transform=summary_ax.transAxes,
                           ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color)
            # Add pipe separator (except after last) - positioned very close to the value
            if i < len(scores_data) - 1:
                pipe_x = x_pos + score_spacing/2 - pipe_gap
                summary_ax.text(pipe_x, text_y - 0.05 + vertical_offset - label_score_gap, "|", 
                               transform=summary_ax.transAxes,
                               ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')
    
    # No main title - removed as requested
    
    # Tight layout to minimize white space
    plt.tight_layout(rect=[0, 0, 1, 1])  # Full layout, no title space needed
    
    out_png = out_dir / "best_all_models_with_components.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"📊 Wrote {out_png}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create best all models plot with component breakdowns")
    parser.add_argument("--scores-file", type=str, 
                       default="./grid_search/all_scenario_scores_with_cpu.csv",
                       help="Path to all_scenario_scores_with_cpu.csv")
    parser.add_argument("--groups-file", type=str,
                       default="./grid_search/all_group_metrics.csv",
                       help="Path to all_group_metrics.csv")
    parser.add_argument("--root-dir", type=str, default="./grid_search",
                       help="Root directory containing scenario subfolders")
    parser.add_argument("--mc-dir", type=str,
                       default="/Users/indigobrownhall/Code/pyssem/pyssem/VnV",
                       help="Directory containing MC data")
    parser.add_argument("--out-dir", type=str, default="./grid_search",
                       help="Output directory for the plot")
    
    args = parser.parse_args()
    
    # Load data
    scores_df = pd.read_csv(args.scores_file)
    all_group_df = pd.read_csv(args.groups_file)
    
    # Add model_type column if missing
    if "model_type" not in scores_df.columns:
        scores_df = add_factor_columns(scores_df)
    
    # Convert to Path objects
    root_dir = Path(args.root_dir)
    mc_dir = Path(args.mc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    plot_best_all_models_with_components(scores_df, all_group_df, root_dir, mc_dir, out_dir)
    print("✅ Done!")

if __name__ == "__main__":
    main()
