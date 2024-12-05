from .utils.simulation.scen_properties import ScenarioProperties
from .utils.simulation.species import Species
from .utils.collisions.collisions import create_collision_pairs
from .utils.plotting.plotting import create_plots, results_to_json
# if testing locally, use the following import statements
# from utils.simulation.scen_properties import ScenarioProperties
# from utils.simulation.species import Species
# from utils.collisions.collisions import create_collision_pairs
# from utils.plotting.plotting import create_plots, results_to_json
from datetime import datetime
import json
import os
import pickle

class Model:
    def __init__(self, start_date, simulation_duration, steps, min_altitude, max_altitude, 
                        n_shells, launch_function, integrator, density_model, LC, v_imp=None,
                        fragment_spreading=True, parallel_processing=False, baseline=False):
        """
        Initialize the scenario properties for the simulation model.

        Parameters:
        - start_date (str): Start date of the simulation in MM/DD/YYYY format.
        - simulation_duration (int): Duration of the simulation in days.
        - steps (int): Number of steps in the simulation.
        - min_altitude (int): Minimum altitude in meters.
        - max_altitude (int): Maximum altitude in meters.
        - n_shells (int): Number of shells in the simulation.
        - launch_function (str): Type of launch function (e.g., "Constant").
        - integrator (str): Type of integrator to use (e.g., "BDF").
        - density_model (str): Density model to use ("static_exp_dens_func").
        - LC (float): Coefficient related to launch (unitless).
        - v_imp (float): Impact velocity of objects in m/s.
        - _return (bool): Whether to return the scenario properties object.

        Returns:
        ScenarioProperties: An initialized scenario properties object.
        
        Raises:
        ValueError: If any parameters are of incorrect type or invalid value.
        """
        try:
            # Validate and convert start_date
            parsed_date = datetime.strptime(start_date, "%m/%d/%Y")

            # Validate numeric parameters
            if not isinstance(simulation_duration, int) or simulation_duration <= 0:
                raise ValueError("simulation_duration must be a positive integer.")
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError("steps must be a positive integer.")
            if not isinstance(min_altitude, int) or min_altitude < 0:
                raise ValueError("min_altitude must be a non-negative integer.")
            if not isinstance(max_altitude, int) or max_altitude <= min_altitude:
                raise ValueError("max_altitude must be greater than min_altitude.")
            if not isinstance(n_shells, int) or n_shells <= 0:
                raise ValueError("n_shells must be a positive integer.")
            if not isinstance(LC, (int, float)):
                raise ValueError("LC must be a numeric type.")

            # Create the ScenarioProperties object
            self.scenario_properties = ScenarioProperties(
                start_date=parsed_date,
                simulation_duration=simulation_duration,
                steps=steps,
                min_altitude=min_altitude,
                max_altitude=max_altitude,
                n_shells=n_shells,
                launch_function=launch_function,
                integrator=integrator,
                density_model=density_model,
                LC=LC,
                v_imp=v_imp,
                fragment_spreading=fragment_spreading,
                parallel_processing=parallel_processing,
                baseline=baseline
            )

            # Define parameters needed at Model level
            self.baseline = baseline
            
        except Exception as e:
            raise ValueError(f"An error occurred initializing the model: {str(e)}")
        
    def configure_species(self, species_json):
        """
        Configure species into Species objects from JSON. This will pass the multiple species and split them, creating the symbolic variables.
        Then pairs the debris and active species for PMD modeling.Finally, it will create the collision pairs between the species to enable the simulation.
        
        Parameters:
        - species_json (dict): JSON object containing species data.

        Returns:
        - Species: A configured species object.
        """
        try:
            species_list = Species()
            
            species_list.add_species_from_json(species_json)

            # Pass functions for drag and PMD
            species_list.convert_params_to_functions()

            # Create symbolic variables for the species
            self.all_symbolic_vars = species_list.create_symbolic_variables(self.scenario_properties.n_shells)

            # Pair the active species to the debris species for PMD modeling
            species_list.pair_actives_to_debris(species_list.species['active'], species_list.species['debris'])

            # Add the final species to the scenario properties to be used in the simulation
            self.scenario_properties.add_species_set(species_list.species, self.all_symbolic_vars)

            # Create Collision Pairs
            self.scenario_properties.add_collision_pairs(create_collision_pairs(self.scenario_properties))

            return species_list
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for species.")
        except Exception as e:
            raise ValueError(f"An error occurred configuring species: {str(e)}")

    def run_model(self):
        """
        Execute the simulation model using the provided scenario properties.
        
        Parameters:
        - scenario_properties (ScenarioProperties): Scenario properties object.item

        Returns:
        - Result of the simulation.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:

            self.scenario_properties.initial_pop_and_launch(baseline=self.baseline) # Initial population is considered but not launch
            self.scenario_properties.build_model()
            self.scenario_properties.run_model()

            # save the scenario properties to a pickle file
            with open('scenario-properties-baseline.pkl', 'wb') as f:
                pickle.dump(self.scenario_properties, f)
            
            return self.scenario_properties
        
        except Exception as e:
            raise RuntimeError(f"Failed to run model: {str(e)}")
        
    def create_plots(self):
        """
        Create plots for the simulation results.
        
        Parameters:
        - scenario_properties (ScenarioProperties): Scenario properties object.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            create_plots(self)
        except Exception as e:
            raise RuntimeError(f"Failed to create plots: {str(e)}")
    
    def results_to_json(self):
        """
        Convert the simulation results to JSON format.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            return results_to_json(self)
        except Exception as e:
            raise RuntimeError(f"Failed to convert results to JSON: {str(e)}")


if __name__ == "__main__":

    with open(os.path.join('pyssem', 'three_species.json')) as f:
        simulation_data = json.load(f)

    scenario_props = simulation_data["scenario_properties"]

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
        fragment_spreading=scenario_props.get("fragment_spreading", True),
        parallel_processing=scenario_props.get("parallel_processing", False),
        baseline=scenario_props.get("baseline", False)
    )

    species = simulation_data["species"]

    species_list = model.configure_species(species)

    results = model.run_model()
    model.create_plots()