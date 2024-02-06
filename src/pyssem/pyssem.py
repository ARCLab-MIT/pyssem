from utils.simulation.scene_properties import SceneProperties
from utils.simulation.species import SpeciesProperties
from utils.launch.launch import launch_func_constant
from datetime import datetime

# Create a scene properties object
scene_properties = SceneProperties(datetime.strptime('01/06/1998', "%m/%d/%Y"), 1000, 100, 500, 1500, 10, delta=10, integrator = "rk4", 
                 density_model = "static_exp_dens_func", LC=0.1, v_imp=10)

# Create a list of species for the scene
species_properties_json = '{"mass": 10, "Cd": 2.2, "active": true, "Pm": 0.95}'
species_properties = SpeciesProperties(species_properties_json)

print(species_properties.mass)
