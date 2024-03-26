from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from utils.collisions.collisions import create_collision_pairs
from datetime import datetime
import json
import pickle

def main(species_json):
    # Create a scenaaio properties object, this is the high level simulation parameters
    scenario_properties = ScenarioProperties(
                    start_date=datetime.strptime('01/03/2022', "%m/%d/%Y"), 
                    simulation_duration=100, steps=200, min_altitude=200, 
                    max_altitude=1400, n_shells=40, launch_function="Constant", 
                    delta=10, integrator = "rk4", density_model = "static_exp_dens_func", 
                    LC=0.1, v_imp=10)

    # Create a list of species for the scene
    species_list = Species()

    # Import the species from a JSON file - this will be defined by the user
    species_list.add_species_from_json(species_json)

    # Pass functions for drag and PMD
    species_list.convert_params_to_functions()

    # Apply Launch Rates and create symbolic variables
    # launch function should be constant over time, rather than a function of altitude=
    species_list.apply_launch_rates(scenario_properties.n_shells)
    species_list.create_symbolic_variables(scenario_properties.n_shells)

    # Pair the active species to the debris species for PMD modeling
    species_list.pair_actives_to_debris(species_list.species['active'], species_list.species['debris'])

    # Add the final species to the scenario properties to be used in the simulation
    scenario_properties.add_species_set(species_list.species)

    # Create collision pairs
    scenario_properties.add_collision_pairs(create_collision_pairs(scenario_properties))

    # Initial Population and ADEPT Launch Model 
    scenario_properties.initial_pop_and_launch()

    # If scenario properties are saved, then load them
    # with open('scenario_properties_1.pkl', 'rb') as f:
    #     scenario_properties = pickle.load(f)

    scenario_properties.build_model()

    # Save scenario properties is a pickle file
    with open('scenario_properties_1.pkl', 'wb') as f:
        pickle.dump(scenario_properties, f)


    scenario_properties.run_model()

    return
    
    # # Then add these generated species to the scenario properties
    # scenario_properties.add_species_set(species)

if __name__ == "__main__":
    # import the template species.json file
    with open('src\pyssem\species.json') as f:
        species_template = json.load(f)
    main(species_template)