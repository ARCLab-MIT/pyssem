from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from utils.collisions.collisions import create_collision_pairs
from datetime import datetime
import json

def main(species_json):
    # Create a scenaaio properties object, this is the high level simulation parameters
    scenario_properties = ScenarioProperties(
                    start_date=datetime.strptime('01/06/2023', "%m/%d/%Y"), 
                    simulation_duration=100, steps=100, min_altitude=500, 
                    max_altitude=1500, n_shells=10, launch_function="Constant", 
                    delta=10, integrator = "rk4", density_model = "static_exp_dens_func", 
                    LC=0.1, v_imp=10)

    # Create a list of species for the scene
    species_list = Species()

    # Import the species from a JSON file - this will be defined by the user
    species_list.add_species_from_json(species_json)

    # Apply Launch Rates and create symbolic variables
    # launch function should be constant over time, rather than a function of altitude=
    species_list.apply_launch_rates(scenario_properties.n_shells)
    species_list.create_symbolic_variables(scenario_properties.n_shells)

    # Pair the active species to the debris species for PMD modeling
    species_list.pair_actives_to_debris(species_list.species['active'], species_list.species['debris'])

    # Add the final species to the scenario properties to be used in the simulation
    scenario_properties.add_species_set(species_list.species)

    # Create collision pairs
    #scenario_properties.add_collision_pairs(create_collision_pairs(scenario_properties))
    scenario_properties.initial_pop_and_launch()

    return
    
    # # Then add these generated species to the scenario properties
    # scenario_properties.add_species_set(species)

if __name__ == "__main__":
    # import the template species.json file
    with open('src\pyssem\species.json') as f:
        species_template = json.load(f)
    main(species_template)