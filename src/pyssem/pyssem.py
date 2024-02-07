from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from datetime import datetime

# Create a scenaaio properties object, this is the high level simulation parameters
scenario_properties = ScenarioProperties(datetime.strptime('01/06/2023', "%m/%d/%Y"), 
                simulation_duration=1000, steps=100, min_altitude=500, 
                max_altitude=1500, n_shells=10, delta=10, integrator = "rk4", 
                density_model = "static_exp_dens_func", LC=0.1, v_imp=10)

# Create a list of species for the scene
species_properties = Species()
species = species_properties.add_species_from_template(["Su", "S", "sns", "N"])

# Then add these generated species to the scenario properties
scenario_properties.add_species_set(species)

print(scenario_properties.species[0].pmd_func(1, 1, species_properties, scenario_properties))


