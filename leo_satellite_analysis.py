#!/usr/bin/env python3
"""
Script to analyze active satellites in LEO that launched before January 1, 2025.
"""

import pandas as pd
from datetime import datetime
import os

def analyze_leo_satellites():
    """Analyze active satellites in LEO."""
    
    # Find the data file
    data_file = 'pyssem/utils/launch/data/satcat (2).csv'
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return None
    
    # Load the satellite data
    print("Loading satellite data...")
    df = pd.read_csv(data_file)
    
    print(f"Total satellites in dataset: {len(df)}")
    
    # Convert launch_date to datetime
    df['LAUNCH_DATE'] = pd.to_datetime(df['LAUNCH_DATE'], errors='coerce')
    
    # Convert decay_date to datetime (handle empty values)
    df['DECAY_DATE'] = pd.to_datetime(df['DECAY_DATE'], errors='coerce')
    
    # Filter for satellites launched before January 1, 2025
    cutoff_date = datetime(2025, 1, 1)
    df_launched_before_2025 = df[df['LAUNCH_DATE'] < cutoff_date]
    print(f"Satellites launched before Jan 1, 2025: {len(df_launched_before_2025)}")
    
    # Filter for active satellites (OPS_STATUS_CODE: +, P, B, X)
    active_status_codes = ['+', 'P', 'B', 'X']
    df_active = df_launched_before_2025[df_launched_before_2025['OPS_STATUS_CODE'].isin(active_status_codes)]
    print(f"Active satellites launched before Jan 1, 2025: {len(df_active)}")
    
    # Filter for satellites still on orbit (no decay date)
    df_still_on_orbit = df_active[df_active['DECAY_DATE'].isna()]
    print(f"Active satellites still on orbit: {len(df_still_on_orbit)}")
    
    # Filter for LEO (perigee altitude below 2000 km)
    # Convert perigee to numeric, handling any non-numeric values
    df_still_on_orbit['PERIGEE'] = pd.to_numeric(df_still_on_orbit['PERIGEE'], errors='coerce')
    df_leo = df_still_on_orbit[df_still_on_orbit['PERIGEE'] < 2000]
    print(f"Active satellites in LEO (perigee < 2000 km): {len(df_leo)}")
    
    # Additional analysis
    print("\n=== Additional Analysis ===")
    
    # Count by object type
    print("\nObject types in LEO:")
    object_type_counts = df_leo['OBJECT_TYPE'].value_counts()
    for obj_type, count in object_type_counts.items():
        print(f"  {obj_type}: {count}")
    
    # Count by operations status
    print("\nOperations status in LEO:")
    ops_status_counts = df_leo['OPS_STATUS_CODE'].value_counts()
    for status, count in ops_status_counts.items():
        print(f"  {status}: {count}")
    
    # Launch year distribution
    print("\nLaunch year distribution (top 10):")
    df_leo['LAUNCH_YEAR'] = df_leo['LAUNCH_DATE'].dt.year
    launch_year_counts = df_leo['LAUNCH_YEAR'].value_counts().head(10)
    for year, count in launch_year_counts.items():
        print(f"  {year}: {count}")
    
    # Altitude statistics
    print(f"\nAltitude statistics:")
    print(f"  Mean perigee: {df_leo['PERIGEE'].mean():.1f} km")
    print(f"  Median perigee: {df_leo['PERIGEE'].median():.1f} km")
    print(f"  Min perigee: {df_leo['PERIGEE'].min():.1f} km")
    print(f"  Max perigee: {df_leo['PERIGEE'].max():.1f} km")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total active satellites in LEO launched before Jan 1, 2025: {len(df_leo)}")
    
    return df_leo

if __name__ == "__main__":
    leo_satellites = analyze_leo_satellites()
