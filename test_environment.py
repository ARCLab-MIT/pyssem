#!/usr/bin/env python3
"""
Test script to verify environment and create basic plots
"""

import sys
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

print("Testing environment...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test imports
try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"Matplotlib import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

# Test density data loading
try:
    density_path = Path("pyssem/utils/drag/dens_highvar_2000_dens_highvar_2000_lookup.json")
    if density_path.exists():
        print(f"Density data file exists: {density_path}")
        with density_path.open("r") as f:
            density_data = json.load(f)
        print(f"Loaded density data with {len(density_data)} time points")
        print(f"Sample dates: {list(density_data.keys())[:5]}")
    else:
        print(f"Density data file not found: {density_path}")
except Exception as e:
    print(f"Error loading density data: {e}")

# Test basic plot creation
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Plot')
    ax.grid(True, alpha=0.3)
    
    # Create figures directory
    Path("figures").mkdir(exist_ok=True)
    
    plt.savefig("figures/test_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully created test plot: figures/test_plot.png")
    
except Exception as e:
    print(f"Error creating test plot: {e}")

print("Environment test complete!")
