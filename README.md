# pySSEM - Source Sink Evolutionary Model

**This is still at pre-alpha stage, the model is still actively being developed and tested. Please do not rely on results.**

## Description

pySSEM is a tool that investigates the evolution of the space objects population in Low Earth Orbit (LEO) by exploiting a new probabilistic source-sink model. The objective is to estimate the LEO orbital capacity. This is carried out through the long-term propagation of the proposed source-sink model, which globally takes into account different object species, such as active satellites, derelict satellites, debris, and additional subgroups. Since the Space Objects (SOs) are propagated as species, the information about single objects is missing, but it allows the model to be computationally fast and provide essential information about the projected future distribution of SOs in the space environment for long prediction horizons.

## Installation

Ensure that you have a Python version above 3.8 before running the package. 

Download the python package using pip (currently Test Environment) and install the required packages:

```bash
pip install -i https://test.pypi.org/simple/ pyssem==1.0
pip install -r requirements.txt
```

To create a Model you need the following properties:
```json
"scenario_properties": {
    "start_date": "01/03/2022",   
    "simulation_duration": 100,              
    "steps": 200,                            
    "min_altitude": 200,                   
    "max_altitude": 1400,                   
    "n_shells": 40,                         
    "launch_function": "Constant", 
    "integrator": "BDF",                
    "density_model": "static_exp_dens_func", 
    "LC": 0.1,                             
    "v_imp": 10.0                          
  }
```

Species are defined as a separate "species" list within your json. Each item is a new species type, each species can have multiple lengths (see documentation for more information). 
```json
"species": {
    "S": {
      "sym_name": "S",
      "Cd": 2.2,
      "mass": [1250, 750, 148],
      "radius": [4, 2, 0.5],
      "A": "Calculated based on radius",
      "active": true,
      "maneuverable": true,
      "trackable": true,
      "deltat": [8],
      "Pm": 0.90,
      "alpha": 1e-5,
      "alpha_active": 1e-5,
      "slotted": true, 
      "slotting_effectiveness": 1.0,
      "drag_effected": false,
      "launch_func": "launch_func_constant",
      "pmd_func": "pmd_func_sat",
      "drag_func": "drag_func_exp"
  },
  "Su": {
      "sym_name": "Su",
      "Cd": 2.2,
      "mass": [260, 473],
      "A": [1.6652, 13.5615],
      "radius": [0.728045069, 2.077681285],
      "active": true,
      "maneuverable": true,
      "trackable": true,
      "deltat": [8, 8],
      "Pm": 0.65,
      "alpha": 1e-5,
      "alpha_active": 1e-5,
      "RBflag": 0,
      "slotting_effectiveness": 1.0,
      "drag_effected": false,
      "launch_func": "launch_func_constant",
      "pmd_func": "pmd_func_sat",
      "drag_func": "drag_func_exp"
  }
```

An example of running the simulation:
```python
from pyssem.model import Model
import json
import os

# Load simulation configuration
with open('/path/to/example-sim-simple.json') as f:
  simulation_data = json.load(f)

scenario_props = simulation_data['scenario_properties']

# Create an instance of the Model with the simulation parameters
model = Model(
    start_date=scenario_props["start_date"].split("T")[0],  # Assuming date is in ISO format
    simulation_duration=scenario_props["simulation_duration"],
    steps=scenario_props["steps"],
    min_altitude=scenario_props["min_altitude"],
    max_altitude=scenario_props["max_altitude"],
    n_shells=scenario_props["n_shells"],
    launch_function=scenario_props["launch_function"],
    integrator=scenario_props["integrator"],
    density_model=scenario_props["density_model"],
    LC=scenario_props["LC"],
    v_imp=scenario_props["v_imp"],
    launchfile='path/to/launchfile.csv'
)

species = simulation_data["species"]
species_list = model.configure_species(species)

# Run the model
results = model.run_model()
```




