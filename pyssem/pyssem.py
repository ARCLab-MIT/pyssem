from utils.simulation.scen_properties import ScenarioProperties
from utils.simulation.species import Species
from utils.collisions.collisions import create_collision_pairs
from datetime import datetime
import json
from dill import pickle

class pySSEM_model:
    def __init__(self, start_date, simulation_duration, steps, min_altitude, max_altitude, 
                        n_shells, launch_function, integrator, density_model, LC, v_imp):
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
            if not isinstance(v_imp, (int, float)):
                raise ValueError("v_imp must be a numeric type.")

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
                v_imp=v_imp
            )


        except Exception as e:
            raise ValueError(f"An error occurred initializing the model: {str(e)}")
        
    def configure_species(self, species_json):
        """
        Configure species into Species objects from JSON.
        
        Parameters:
        - species_json (dict): JSON object containing species data.

        Returns:
        - Species: A configured species object.
        """
        try:
            species_list = Species()
            
            # Import the species from a JSON file 
            species_list.add_species_from_json(species_json)
            
            # Pass functions for drag and PMD
            species_list.convert_params_to_functions()

            # Apply Launch Rates and create symbolic variables
            species_list.apply_launch_rates(self.scenario_properties.n_shells)
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

    def run_model(self, print_time = False):
        """
        Execute the simulation model using the provided scenario properties.
        
        Parameters:
        - scenario_properties (ScenarioProperties): Scenario properties object.

        Returns:
        - Result of the simulation.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:

            self.scenario_properties.initial_pop_and_launch()
            self.scenario_properties.build_model()
            self.scenario_properties.run_model(print_time)
            return self.scenario_properties
        
        except Exception as e:
            raise RuntimeError(f"Failed to run model: {str(e)}")

if __name__ == "__main__":
    # import the template species.json file
    with open('pyssem\species-long.json') as f:
        species_data = json.load(f)

    # Create an instance of the pySSEM_model with the simulation parameters
    model = pySSEM_model(
        start_date="01/03/2022",
        simulation_duration=100,
        steps=200,
        min_altitude=200,
        max_altitude=1400,
        n_shells=40,
        launch_function="Constant",
        integrator="BDF",
        density_model="static_exp_dens_func",
        LC=0.1,
        v_imp=10
    )

    # Configure species
    species_list = model.configure_species(species_data)
    
    # Run the model
    results = model.run_model()
    print("Simulation results:", results)