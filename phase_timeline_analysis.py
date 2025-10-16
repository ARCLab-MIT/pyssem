#!/usr/bin/env python3
"""
Script to analyze phase changes for satellites and create timeline plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def create_phase_timeline_plots():
    """Create phase timeline plots for satellites."""
    
    # Load the data
    print("Loading satellite data...")
    data_file = '/Users/indigobrownhall/Code/pyssem/pyssem/utils/launch/data/ref_scen_SEP2.csv'
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    # Load the full dataset
    print("Loading full dataset...")
    df = pd.read_csv(data_file)
    print(f"Total rows: {len(df)}")
    
    # Apply the same filtering as in the notebook
    print("Applying filters...")
    
    # Filter active satellites (object type 2)
    active_satellites = df[df['obj_type'] == 2].copy()
    print(f"After obj_type filter: {len(active_satellites)}")
    
    # only if date is greater than 2025
    active_satellites = active_satellites[active_satellites['year_start'] < 2025]
    print(f"After year_start filter: {len(active_satellites)}")
    
    # Replace NaN constellation names with a placeholder
    active_satellites['const_name'] = active_satellites['const_name'].fillna('Unassigned')
    active_satellites = active_satellites[active_satellites['const_name'] == 'Unassigned']
    print(f"After const_name filter: {len(active_satellites)}")
    
    # Filter for LEO (semi-major axis < 8371 km)
    active_satellites = active_satellites[active_satellites['sma'] < 8371]
    print(f"After LEO filter: {len(active_satellites)}")
    
    print(f"Final filtered satellites: {len(active_satellites)}")
    
    # Create datetime columns for start and end dates
    # Create start dates
    active_satellites['start_date'] = pd.to_datetime(
        active_satellites['year_start'].astype(str) + '-' + 
        active_satellites['month_start'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    
    # Create end dates
    active_satellites['end_date'] = pd.to_datetime(
        active_satellites['year_final'].astype(str) + '-' + 
        active_satellites['month_final'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    
    # Remove rows with invalid dates
    active_satellites = active_satellites.dropna(subset=['start_date', 'end_date'])
    print(f"After date filtering: {len(active_satellites)}")
    
    # Get unique satellite IDs
    unique_ids = active_satellites['obj_id'].unique()
    print(f"Unique satellite IDs: {len(unique_ids)}")
    
    # Create plots for every 100 satellites
    satellites_per_plot = 100
    num_plots = (len(unique_ids) + satellites_per_plot - 1) // satellites_per_plot
    
    print(f"Creating {num_plots} plots with {satellites_per_plot} satellites each...")
    
    for plot_num in range(num_plots):
        start_idx = plot_num * satellites_per_plot
        end_idx = min((plot_num + 1) * satellites_per_plot, len(unique_ids))
        plot_ids = unique_ids[start_idx:end_idx]
        
        # Filter data for this plot
        plot_data = active_satellites[active_satellites['obj_id'].isin(plot_ids)].copy()
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Create a mapping from obj_id to y position
        id_to_y = {obj_id: i for i, obj_id in enumerate(plot_ids)}
        
        # Plot each satellite's timeline
        for idx, (_, row) in enumerate(plot_data.iterrows()):
            obj_id = row['obj_id']
            y_pos = id_to_y[obj_id]
            
            # Plot the phase timeline
            start_date = row['start_date']
            end_date = row['end_date']
            phase = row['phase']
            
            # Color based on phase
            phase_colors = {
                1: 'red',
                2: 'blue', 
                3: 'green',
                4: 'orange',
                5: 'purple'
            }
            color = phase_colors.get(phase, 'gray')
            
            # Plot horizontal line for this phase
            plt.plot([start_date, end_date], [y_pos, y_pos], 
                    color=color, linewidth=2, alpha=0.7)
            
            # Add phase label at the start
            plt.text(start_date, y_pos, f'P{phase}', 
                    fontsize=8, va='center', ha='right')
        
        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel('Satellite ID')
        plt.title(f'Phase Timeline - Satellites {start_idx+1}-{end_idx} (Plot {plot_num+1}/{num_plots})')
        
        # Set y-axis to show satellite IDs
        plt.yticks(range(len(plot_ids)), [f'ID {id}' for id in plot_ids])
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add legend for phases
        legend_elements = []
        for phase, color in phase_colors.items():
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Phase {phase}'))
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        output_file = f'figures/phase_timeline_plot_{plot_num+1}.png'
        os.makedirs('figures', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot {plot_num+1}: {output_file}")
        
        # Show the plot
        plt.show()
    
    print(f"\nAnalysis complete! Created {num_plots} plots.")
    
    # Print summary statistics
    print("\n=== Phase Summary ===")
    phase_counts = active_satellites['phase'].value_counts().sort_index()
    for phase, count in phase_counts.items():
        print(f"Phase {phase}: {count} satellites")
    
    print(f"\nDate range:")
    print(f"  Start: {active_satellites['start_date'].min()}")
    print(f"  End: {active_satellites['end_date'].max()}")

if __name__ == "__main__":
    create_phase_timeline_plots()
