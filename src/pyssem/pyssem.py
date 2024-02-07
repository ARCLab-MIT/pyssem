from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from datetime import datetime
import json

def main(species_json):
    # Create a scenaaio properties object, this is the high level simulation parameters
    scenario_properties = ScenarioProperties(
                    start_date=datetime.strptime('01/06/2023', "%m/%d/%Y"), 
                    simulation_duration=1000, steps=100, min_altitude=500, 
                    max_altitude=1500, n_shells=10, delta=10, integrator = "rk4", 
                    density_model = "static_exp_dens_func", LC=0.1, v_imp=10)

    # Create a list of species for the scene
    species_properties = Species()

    # Import the species from a JSON file - this will be defined by the user
    species = species_properties.add_species_from_json(species_json)

    # Split the species into either debris or active satellites, based from the 'active' property
    active_species = [s for s in species if s.active]
    debris_species = [s for s in species if not s.active]

    # Pair the active species to the debris species for PMD modeling
    scenario_properties.pair_actives_to_debris(active_species, debris_species)

    # Then add these generated species to the scenario properties
    scenario_properties.add_species_set(species)

if __name__ == "__main__":
    # import the template species.json file
    with open('src\pyssem\species.json') as f:
        species_template = json.load(f)
    main(species_template)