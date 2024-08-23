# pySSEM - Source Sink Evolutionary Model

## Description

pySSEM is a tool that investigates the evolution of the space objects population in Low Earth Orbit (LEO) by exploiting a new probabilistic source-sink model. The objective is to estimate the LEO orbital capacity. This is carried out through the long-term propagation of the proposed source-sink model, which globally takes into account different object species, such as active satellites, derelict satellites, debris, and additional subgroups. Since the Space Objects (SOs) are propagated as species, the information about single objects is missing, but it allows the model to be computationally fast and provide essential information about the projected future distribution of SOs in the space environment for long prediction horizons.

## Creating a simulation
If you are running a simulation for the first time, we would recomment starting with the `pyssem/example-sim.json` file, which is a simulation presented by [Miles Lifson at AMOS 2023](https://amostech.com/TechnicalPapers/2023/Poster/Lifson.pdf). Use this as a starting template and then updating the parameters as desired.

Unfortunately, in version 1, adding your own launch file will not be possible and you will only be able to add the launch file created in the above paper. 

## Simulation Parameters

When creating a model, the following properties are required within the `scenario_properties` section of your JSON configuration file. These parameters define the core aspects of your simulation environment:

- **start_date**: The start date of the simulation in `DD/MM/YYYY` format.
  - Example: `"start_date": "01/03/2022"`

- **simulation_duration**: The total duration of the simulation in days.
  - Example: `"simulation_duration": 100`

- **steps**: The number of simulation steps to be executed.
  - Example: `"steps": 200`

- **min_altitude**: The minimum altitude for the simulation in kilometers.
  - Example: `"min_altitude": 200`

- **max_altitude**: The maximum altitude for the simulation in kilometers.
  - Example: `"max_altitude": 1400`

- **n_shells**: The number of altitude shells to be used in the simulation.
  - Example: `"n_shells": 40`

- **launch_function**: The function that defines the launch rate of objects.
  - Example: `"launch_function": "Constant"`

- **integrator**: The numerical integration method to be used.
  - Example: `"integrator": "BDF"`

- **density_model**: The model used to define atmospheric density.
  - Example: `"density_model": "static_exp_dens_func"`

- **LC**: Characteristic length scale (in kilometers).
  - Example: `"LC": 0.1`

- **v_imp**: Impact velocity (in km/s).
  - Example: `"v_imp": 10.0`

- **fragment_spreading**: Debris fragments are spread across orbital shells after a collision. This will drastically increase run time, but will lead to more accurate results
  - Example: `true`

- **parallel_processing**: Will use all available cores to speed up the computations.
  - Example: `true`

- **baseline**: No futher launches. 
  - Example: `false`

## Species Definition

Each species in the simulation is defined in the `species` section of the JSON configuration file. Each species has its own unique properties and can contain multiple length variations.

### Species Parameters:

- **sym_name**: The symbolic name of the species. This can be any combination of letters, but unique for each species
  - Example: `"sym_name": "S"`

- **Cd**: Drag coefficient. 
  - Example: `2.2`

- **mass**: List of masses for different lengths (in kg). If you want to make multiple copies of the same species, ensure that mass and radius are the same length. 
  - Example: `"mass": [1250, 750, 148]"`

- **radius**: List of radii for different lengths (in metres).
  - Example: `"radius": [4, 2, 0.5]"`

- **A**: Cross-sectional area. If not known, it can be derived from your radius. 
  - Example: `"A": "Calculated based on radius"`

- **active**: Indicates if the species is active. If active, it will station keep and not be affected by pertubations. 
  - Example: `"active": true"`

- **maneuverable**: Indicates if the species is maneuverable. If true, it will not be involved in collisions. 
  - Example: `"maneuverable": true"`

- **trackable**: Indicates if the species is trackable. 
  - Example: `"trackable": true"`

- **Pm**: Post-Mission Disposal Effectiveness
  - Example: `"Pm": 0.90"`

- **deltat**: If PMD fails, this will be the number of years of active operations. 
  - Example: `"deltat": [8]`

- **alpha**: failure rate of collision avoidance vs inactive trackable objects [0 = perfect, 1 = none]
  - Example: `"alpha": 1e-5"`

- **alpha_active**: failure rate of collision avoidance vs active maneuverable objects [0 = perfect, 1 = none]. If unknown leave at 1e-5 as default. 
  - Example: `"alpha_active": 1e-5"`

- **slotted**: Indicates if the species is slotted in a constellation. 
  - Example: `"slotted": true"`

- **slotting_effectiveness**: Effectiveness of slotting (%), if a lower effectiveness, collision likelihood will increase. 
  - Example: `"slotting_effectiveness": 1.0"`

- **drag_effected**: Indicates if drag affects the species. If false, it will remain at the desired orbit. 
  - Example: `"drag_effected": false"`

- **launch_func**: Launch function for the species. Either constant or none. Add your own launch function in launch.py if another is required. 
  - Example: `"launch_func": "launch_func_constant"`

- **pmd_func**: Post-mission disposal function for the species. Either sat, derelict or none. 
  - Example: `"pmd_func": "pmd_func_sat"`

- **drag_func**: Drag function for the species, either exp or none. 
  - Example: `"drag_func": "drag_func_exp"`


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
    start_date=scenario_props["start_date"].split("T")[0],
    simulation_duration=scenario_props["simulation_duration"], 
    steps=scenario_props["steps"], 
    min_altitude=scenario_props["min_altitude"], # 
    max_altitude=scenario_props["max_altitude"],
    n_shells=scenario_props["n_shells"],
    launch_function=scenario_props["launch_function"],
    integrator=scenario_props["integrator"],
    density_model=scenario_props["density_model"],
    LC=scenario_props["LC"],
    v_imp=scenario_props["v_imp"],
    fragment_spreading=false,
    parallel_processing=false, 
    baseline=false
)

species = simulation_data["species"]
species_list = model.configure_species(species)

# Run the model
results = model.run_model()
```




