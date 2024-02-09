from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from utils.collisions.collisions import create_collision_pairs
from datetime import datetime
import json

def main(species_json):
    # Create a scenaaio properties object, this is the high level simulation parameters
    scenario_properties = ScenarioProperties(
                    start_date=datetime.strptime('01/06/2023', "%m/%d/%Y"), 
                    simulation_duration=1000, steps=100, min_altitude=500, 
                    max_altitude=1500, n_shells=10, launch_function="Constant", 
                    delta=10, integrator = "rk4", density_model = "static_exp_dens_func", 
                    LC=0.1, v_imp=10)

    # Create a list of species for the scene
    species_list = Species()

    # Import the species from a JSON file - this will be defined by the user
    species_list.add_species_from_json(species_json)

    # Apply Launch Rates and create symbolic variables
    species_list.apply_launch_rates(scenario_properties.n_shells)
    species_list.create_symbolic_variables(scenario_properties.n_shells)

    # Split the species into either debris or active satellites, based from the 'active' property
    active_species = [s for s in species_list.species if s.active]
    debris_species = [s for s in species_list.species if not s.active and s.RBflag != 1] # B is the rocket body species

    # Pair the active species to the debris species for PMD modeling
    species_list.pair_actives_to_debris(active_species, debris_species)

    # Add the final species to the scenario properties to be used in the simulation
    scenario_properties.add_species_set(species_list.species)

    for s in species_list.species:
        if s.sym_name == 'S_148kg':
            print(s)

    # Create collision pairs
    #species_pairs = create_collision_pairs(scenario_properties.species)
    
    # # Then add these generated species to the scenario properties
    # scenario_properties.add_species_set(species)

if __name__ == "__main__":
    # import the template species.json file
    with open('src\pyssem\species.json') as f:
        species_template = json.load(f)
    main(species_template)