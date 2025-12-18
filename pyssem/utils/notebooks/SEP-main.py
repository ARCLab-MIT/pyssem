import os
from model import Model
from .utils.plotting.plotting import Plots, results_to_json
import json
import pickle
import time


def run_one_sep(sep_name):
    
    sep_json = f"{sep_name}.json"
    with open(os.path.join('pyssem', 'simulation_configurations', sep_json)) as f:
        simulation_data = json.load(f)

    scenario_props = simulation_data["scenario_properties"]

    start_time = time.time()

    # Create an instance of the pySSEM_model with the simulation parameters
    model = Model(
        start_date=scenario_props["start_date"].split("T")[0],  # Assuming the date is in ISO format
        simulation_duration=scenario_props["simulation_duration"],
        steps=scenario_props["steps"],
        min_altitude=scenario_props["min_altitude"],
        max_altitude=scenario_props["max_altitude"],
        n_shells=scenario_props["n_shells"],
        launch_function=scenario_props["launch_function"],
        integrator=scenario_props["integrator"],
        density_model=scenario_props["density_model"],
        LC=scenario_props["LC"],
        v_imp = scenario_props.get("v_imp", None), 
        fragment_spreading=scenario_props.get("fragment_spreading", False),
        parallel_processing=scenario_props.get("parallel_processing", True),
        baseline=scenario_props.get("baseline", False),
        indicator_variables=scenario_props.get("indicator_variables", None),
        launch_scenario=scenario_props["launch_scenario"],
        SEP_mapping=simulation_data["SEP_mapping"] if "SEP_mapping" in simulation_data else None,
    )

    species = simulation_data["species"]

    species_list = model.configure_species(species)

    start_time_2 = time.time()

    results = model.run_model()

    elapsed = time.time() - start_time
    elapsed_just_model = time.time() - start_time_2

    data = model.results_to_json()

    # dump the pickle file in the same directory 

    # Create the figures directory if it doesn't exist
    os.makedirs(f'figures/{simulation_data["simulation_name"]}', exist_ok=True)
    # Save the results to a JSON file
    with open(f'figures/{simulation_data["simulation_name"]}/results.json', 'w') as f:
        json.dump(data, f, indent=4)

    with open(f'scenario-properties-{sep_name}.pkl', 'wb') as f:
        pickle.dump(model.scenario_properties, f)

    try:
        plot_names = simulation_data["plots"]
        Plots(model.scenario_properties, plot_names, simulation_data["simulation_name"])
    except Exception as e:
        print(e)
        print("No plots specified in the simulation configuration file.")

    return elapsed, elapsed_just_model

if __name__ == "__main__":

    # seps = ['SEP1', 'SEP2', 'SEP3H', 'SEP3M', 'SEP4', 'SEP5H', 'SEP5M', 'SEP6H', 'SEP6M']
    seps = ['SEP2']
    results_path = os.path.join(os.getcwd(), "results.txt")

    for sep in seps:
        
        elapsed, elapsed_just_model = run_one_sep(sep)  # your simulation function
        
        result_line = f"{sep}: Just model {elapsed_just_model:.2f} seconds | Inlcuding Collisions: {elapsed:.2f}\n"
        
        with open(results_path, "a") as f:
            f.write(result_line)

        print(f"✅ Finished {sep} in {elapsed:.2f} seconds (logged to results.txt)")