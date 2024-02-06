from utils.simulation.scene_properties import SceneProperties
from utils.simulation.species import Species
from utils.launch.launch import launch_func_constant
from datetime import datetime

# Create a scene properties object
scene_properties = SceneProperties(datetime.strptime('01/06/2023', "%m/%d/%Y"), 
                simulation_duration=1000, steps=100, min_altitude=500, 
                max_altitude=1500, shells=10, delta=10, integrator = "rk4", 
                density_model = "static_exp_dens_func", LC=0.1, v_imp=10)

# Create a Launch Model
launch = scene_properties.launch_func_constant()

# Post Mission Disposal


# Create a list of species for the scene
species_properties = Species()
species_properties.add_template_species(["Su", "S", "sns", "N"])

