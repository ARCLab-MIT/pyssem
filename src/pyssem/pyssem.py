from utils.simulation.scene_properties import SceneProperties
import datetime

# Create a scene properties object
scene_properties = SceneProperties("01/06/1998", 1000, 100, 500, 1500, 10, delta=10, integrator = "rk4", 
                 density_model = "static_exp_dens_func", LC=0.1, v_imp=10)

print(scene_properties.max_altitude)