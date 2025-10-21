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
    
    # Read only the first few rows to understand the structure
    df_sample = pd.read_csv(data_file, nrows=1000)
    print("Column names:", df_sample.columns.tolist())
    
    # Load the full dataset
    print("Loading full dataset...")
    df = pd.read_csv(data_file)
    print(f"Total rows: {len(df)}")
    
    # Apply the filtering for active satellites in phases 2 and 3
    print("Applying filters...")
    
    # Filter active satellites (object type 2)
    active_satellites = df[df['obj_type'] == 2].copy()
    print(f"After obj_type filter: {len(active_satellites)}")
    
    # Filter for date range 2020-2030 (more specific with months)
    # Start from January 2020
    active_satellites = active_satellites[
        (active_satellites['year_start'] > 2020) | 
        ((active_satellites['year_start'] == 2020) & (active_satellites['month_start'] >= 1))
    ]
    # End before January 2031
    active_satellites = active_satellites[
        (active_satellites['year_start'] < 2030) | 
        ((active_satellites['year_start'] == 2030) & (active_satellites['month_start'] <= 12))
    ]
    print(f"After date range filter (Jan 2020 - Dec 2030): {len(active_satellites)}")
    
    # Replace NaN constellation names with a placeholder
    active_satellites['const_name'] = active_satellites['const_name'].fillna('Unassigned')
    active_satellites = active_satellites[active_satellites['const_name'] == 'Unassigned']
    print(f"After const_name filter: {len(active_satellites)}")
    
    # Filter for LEO (semi-major axis < 8371 km)
    active_satellites = active_satellites[active_satellites['sma'] < 8371]
    print(f"After LEO filter: {len(active_satellites)}")
    
    # Filter for phases 2 and 3 only
    active_satellites = active_satellites[active_satellites['phase'].isin([2, 3])]
    print(f"After phase filter (2,3): {len(active_satellites)}")
    
    print(f"Final filtered satellites: {len(active_satellites)}")
    
    # Debug: Check the data before date conversion
    print(f"\nDebugging date columns:")
    print(f"year_start range: {active_satellites['year_start'].min()} to {active_satellites['year_start'].max()}")
    print(f"month_start range: {active_satellites['month_start'].min()} to {active_satellites['month_start'].max()}")
    print(f"year_final range: {active_satellites['year_final'].min()} to {active_satellites['year_final'].max()}")
    print(f"month_final range: {active_satellites['month_final'].min()} to {active_satellites['month_final'].max()}")
    
    # Check for NaN values
    print(f"NaN values in year_start: {active_satellites['year_start'].isna().sum()}")
    print(f"NaN values in month_start: {active_satellites['month_start'].isna().sum()}")
    print(f"NaN values in year_final: {active_satellites['year_final'].isna().sum()}")
    print(f"NaN values in month_final: {active_satellites['month_final'].isna().sum()}")
    
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
    
    # Debug: Check date conversion results
    print(f"\nAfter date conversion:")
    print(f"Valid start_date: {active_satellites['start_date'].notna().sum()}")
    print(f"Valid end_date: {active_satellites['end_date'].notna().sum()}")
    print(f"Invalid start_date: {active_satellites['start_date'].isna().sum()}")
    print(f"Invalid end_date: {active_satellites['end_date'].isna().sum()}")
    
    # Show some examples of invalid dates
    invalid_start = active_satellites[active_satellites['start_date'].isna()].head(5)
    if len(invalid_start) > 0:
        print(f"\nExamples of invalid start dates:")
        print(invalid_start[['year_start', 'month_start', 'start_date']].head())
    
    # Remove rows with invalid dates
    active_satellites = active_satellites.dropna(subset=['start_date', 'end_date'])
    print(f"After date filtering: {len(active_satellites)}")
    
    # If no valid dates, try a different approach
    if len(active_satellites) == 0:
        print("\nNo valid dates found. Trying alternative approach...")
        # Reset to original data with all filters including phase
        active_satellites = df[df['obj_type'] == 2].copy()
        # Start from January 2020
        active_satellites = active_satellites[
            (active_satellites['year_start'] > 2020) | 
            ((active_satellites['year_start'] == 2020) & (active_satellites['month_start'] >= 1))
        ]
        # End before January 2031
        active_satellites = active_satellites[
            (active_satellites['year_start'] < 2030) | 
            ((active_satellites['year_start'] == 2030) & (active_satellites['month_start'] <= 12))
        ]
        active_satellites['const_name'] = active_satellites['const_name'].fillna('Unassigned')
        active_satellites = active_satellites[active_satellites['const_name'] == 'Unassigned']
        active_satellites = active_satellites[active_satellites['sma'] < 8371]
        active_satellites = active_satellites[active_satellites['phase'].isin([2, 3])]
        
        # Try using just year and month as integers for plotting
        print(f"Using year/month as integers. Found {len(active_satellites)} satellites")
        print(f"Year range: {active_satellites['year_start'].min()} to {active_satellites['year_start'].max()}")
        print(f"Month range: {active_satellites['month_start'].min()} to {active_satellites['month_start'].max()}")
    
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
        
        # Group by satellite to plot both phases
        for obj_id in plot_ids:
            y_pos = id_to_y[obj_id]
            satellite_data = plot_data[plot_data['obj_id'] == obj_id]
            
            # Color based on phase
            phase_colors = {
                1: 'red',
                2: 'blue', 
                3: 'green',
                4: 'orange',
                5: 'purple'
            }
            
            # Plot each phase for this satellite
            for _, row in satellite_data.iterrows():
                phase = row['phase']
                color = phase_colors.get(phase, 'gray')
                
                # Check if we have datetime objects or need to use year/month
                if 'start_date' in row and pd.notna(row['start_date']):
                    # Use datetime objects
                    start_date = row['start_date']
                    end_date = row['end_date']
                    
                    # Plot horizontal line for this phase
                    plt.plot([start_date, end_date], [y_pos, y_pos], 
                            color=color, linewidth=3, alpha=0.8)
                else:
                    # Use year and month as integers
                    start_year = row['year_start']
                    start_month = row['month_start']
                    end_year = row['year_final']
                    end_month = row['month_final']
                    
                    # Convert to decimal years for plotting
                    start_decimal = start_year + (start_month - 1) / 12
                    end_decimal = end_year + (end_month - 1) / 12
                    
                    # Plot horizontal line for this phase
                    plt.plot([start_decimal, end_decimal], [y_pos, y_pos], 
                            color=color, linewidth=3, alpha=0.8)
        
        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel('Satellite ID')
        plt.title(f'Phase Timeline - Satellites {start_idx+1}-{end_idx} (Plot {plot_num+1}/{num_plots})')
        
        # Set y-axis to show satellite IDs
        plt.yticks(range(len(plot_ids)), [f'ID {id}' for id in plot_ids])
        
        # Limit x-axis to 2020-2030
        plt.xlim(2020, 2030)
        
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
