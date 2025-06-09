from .utils.simulation.scen_properties import ScenarioProperties
from .utils.simulation.species import Species
from .utils.collisions.collisions import create_collision_pairs
from .utils.drag.drag import calculate_orbital_lifetimes
from .utils.plotting.plotting import results_to_json, Plots
# if testing locally, use the following import statements
# from utils.simulation.scen_properties import ScenarioProperties
# from utils.simulation.species import Species
# from utils.collisions.collisions import create_collision_pairs
# from utils.drag.drag import calculate_orbital_lifetimes
# from utils.plotting.plotting import results_to_json, Plots
import numpy as np
from datetime import datetime
import json
import os
import pickle
import numpy as np
import pandas as pd


class Model:
    """
    A class to represent a simulation model for pySSEM.

    Attributes:
        start_date (str): Start date of the simulation.
        simulation_duration (int): Duration of the simulation in days.
        steps (int): Number of simulation steps.
        min_altitude (int): Minimum altitude for the simulation in meters.
        max_altitude (int): Maximum altitude for the simulation in meters.
        n_shells (int): Number of altitude shells in the simulation.
        launch_function (str): Type of launch function used (e.g., "Constant").
        integrator (str): Numerical integration method (e.g., "BDF").
        density_model (str): Atmospheric density model (e.g., "static_exp_dens_func").
        LC (float): Launch coefficient.
        v_imp (float, optional): Impact velocity of objects in m/s.
        fragment_spreading (bool, optional): Enable/disable fragment spreading.
        parallel_processing (bool, optional): Use parallel processing if True.
        baseline (bool, optional): If True, assumes no further launches.
        indicator_variables (dict, optional): Additional indicator variables for the model.
    """
    def __init__(self, start_date, simulation_duration, steps, min_altitude, max_altitude, 
                        n_shells, launch_function, integrator, density_model, LC, 
                        v_imp=None,
                        fragment_spreading=True, parallel_processing=False, baseline=False, 
                        indicator_variables=None, launch_scenario=None, SEP_mapping=None):
        """
        Initialize the scenario properties for the simulation model.

        Args:
            start_date (str): Start date of the simulation in MM/DD/YYYY format.
            simulation_duration (int): Duration of the simulation in days.
            steps (int): Number of steps in the simulation.
            min_altitude (int): Minimum altitude in meters.
            max_altitude (int): Maximum altitude in meters.
            n_shells (int): Number of shells in the simulation.
            launch_function (str): Type of launch function (e.g., "Constant").
            integrator (str): Type of integrator to use (e.g., "BDF").
            density_model (str): Density model to use ("static_exp_dens_func").
            LC (float): Coefficient related to launch (unitless).
            v_imp (float, optional): Impact velocity of objects in m/s.
            fragment_spreading (bool, optional): Whether to enable fragment spreading.
            parallel_processing (bool, optional): Enable parallel processing.
            baseline (bool, optional): Use baseline assumptions (no further launches).
            indicator_variables (dict, optional): Additional indicator variables for the model.

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
                v_imp=v_imp,
                fragment_spreading=fragment_spreading,
                parallel_processing=parallel_processing,
                baseline=baseline,
                indicator_variables=indicator_variables,
                launch_scenario=launch_scenario,
                SEP_mapping=SEP_mapping,
            )
            
        except Exception as e:
            raise ValueError(f"An error occurred initializing the model: {str(e)}")
        
    def configure_species(self, species_json):
        """
        Configure species into `Species` objects from a JSON file.

        This method processes the multiple species, splits them, creates symbolic variables,
        pairs debris and active species for Post-Mission Disposal (PMD) modeling, and
        creates collision pairs between the species for simulation.

        Args:
            species_json (dict): JSON object containing species data.

        Returns:
            Species: A configured `Species` object.

        Raises:
            ValueError: If the JSON is invalid or an error occurs during configuration.
        """
        try:
            species_list = Species()
            
            species_list.add_species_from_json(species_json)

            # Set up elliptical orbits for species
            # species_list.set_elliptical_orbits(self.scenario_properties.n_shells, self.scenario_properties.R0_km, self.scenario_properties.HMid, self.scenario_properties.mu, self.scenario_properties.parallel_processing)
            
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

            # Create Indicator Variables if provided
            if self.scenario_properties.indicator_variables is not None:
                # Calculate Orbital Lifetimes if "umpy" is in the indicator variables
                if "umpy" in self.scenario_properties.indicator_variables:
                    self.scenario_properties.species = calculate_orbital_lifetimes(self.scenario_properties)

                self.scenario_properties.build_indicator_variables()

            return species_list
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for species.")
        except Exception as e:
            raise ValueError(f"An error occurred configuring species: {str(e)}")
        
    def calculate_collisions(self):
        """
        Calculate the collisions between species in the simulation.
        
        Parameters:
        - scenario_properties (ScenarioProperties): Scenario properties object.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            self.scenario_properties.add_collision_pairs(create_collision_pairs(self.scenario_properties))
        except Exception as e:
            raise ValueError(f"An error occurred calculating collisions: {str(e)}")
        
    def opus_active_loss_setup(self, fringe_species):
        """
        The OPUS economic model requires an indicator variable to be correctly configured: "actactive_loss_per_species" to be a proxy for probability of collision. 

        This function find the correct economic indicator, lambdify the equations for numpy, then add it to its own variable for easy access.

        Parameters:
        - fringe_species (str): The fringe satellite name that is going to be used for the active satellites
        - scenario_properties (ScenarioProperties): Scenario properties object.

        Returns:
        - None

        Raises:
        - ValueError: If the fringe species is not found in the species list.
        - TypeError: If scenario_properties is not passed
        """

        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise TypeError("Invalid scenario properties provided.")
        if fringe_species not in self.scenario_properties.species_names:
            raise ValueError(f"Invalid fringe species provided: {fringe_species}. Please ensure that a fringe species name is provided in the configuration JSON.")
        if self.scenario_properties.indicator_variables is None:
            raise NameError("Indicator variables not found. Please ensure that the indicator variables are provided in the configuration JSON. If you are an OPUS user please use 'active_loss_per_species'")
        
        try:
            self.scenario_properties.configure_active_satellite_loss(fringe_species)
        except Exception as e:
            raise ValueError(f"An error occurred setting up OPUS active loss: {str(e)}")
    
    def opus_umpy_calculation(self, state_matrix):
        """
        This will calculate the UMPY (Undisposed Mass Per Year metric). 
        """
        # First check that current self has 'umpy' within indicator_variables
        if 'umpy' not in self.scenario_properties.indicator_variables:
            raise ValueError("Indicator variable 'umpy' not found in the scenario properties. Please configure it in the MOCAT json.")

        try:
            umpy = self.scenario_properties.calculate_umpy_for_opus(state_matrix)
            return umpy
        except Exception as e:
            raise ValueError(f"An error occurred calculating UMPY: {str(e)}")  
        

        

    def run_model(self):
        """
        Execute the simulation model using the provided scenario properties.

        This method initializes the population, builds the model, and runs the simulation.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs while running the simulation.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            self.scenario_properties.initial_pop_and_launch(baseline=self.scenario_properties.baseline, launch_file=self.scenario_properties.launch_scenario) # Initial population is considered but not launch
            self.scenario_properties.build_model()
            self.scenario_properties.run_model()
            
            # CSI Index
            # self.scenario_properties.cum_CSI()

            # save self as a pickle file
            with open('scenario-properties-baseline.pkl', 'wb') as f:

                # first remove the lambdified equations as pickle cannot serialize them
                self.scenario_properties.equations = None
                self.scenario_properties.coll_eqs_lambd = None
                self.scenario_properties.full_lambda_flattened = None
            
                pickle.dump(self.scenario_properties, f)
        
        except Exception as e:
            raise RuntimeError(f"Failed to run model: {str(e)}")
        
    def initial_population(self):
        """
            Initialize the population of the species in the simulation. This is mainly used by the OPUS model, so baseline is forced True. 
        """

        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            # If this function is called, only create x0. 
            self.scenario_properties.initial_pop_and_launch(baseline=True, launch_file=self.scenario_properties.launch_scenario)
        
        except Exception as e:
            raise RuntimeError(f"Failed to initialize population: {str(e)}")


    def build_model(self):
        """
            Build the model for the simulation.
        """

        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            self.scenario_properties.build_model()
        
        except Exception as e:
            raise RuntimeError(f"Failed to build model: {str(e)}")
        

    def propagate(self, times, population, launch=False, time_step=0):
        """
            This is when you would like to integrate forward a specific population set. This can be any amount aslong as it follows the same structure of x0 to fit the equations.

            Parameters:
            - times (list): List of times to integrate over.
            - population: One dimensional array, n_species x n_shells.
        """
        
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            results = self.scenario_properties.propagate(population, times, launch, time_step)
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to integrate: {str(e)}")
                                                        
    def results_to_json(self):
        """
        Convert the simulation results to JSON format.

        Returns:
            dict: JSON representation of the simulation results.

        Raises:
            RuntimeError: If an error occurs during the conversion process.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:
            return results_to_json(self)
        except Exception as e:
            raise RuntimeError(f"Failed to convert results to JSON: {str(e)}")

if __name__ == "__main__":

    with open(os.path.join('pyssem', 'simulation_configurations', 'example-sim.json')) as f:
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
        fragment_spreading=scenario_props.get("fragment_spreading", False),
        parallel_processing=scenario_props.get("parallel_processing", True),
        baseline=scenario_props.get("baseline", False),
        indicator_variables=scenario_props.get("indicator_variables", None),
        launch_scenario=scenario_props["launch_scenario"],
        SEP_mapping=simulation_data["SEP_mapping"] if "SEP_mapping" in simulation_data else None,
    )

    species = simulation_data["species"]

    species_list = model.configure_species(species)

    results = model.run_model()

    data = model.results_to_json()
    # Create the figures directory if it doesn't exist
    os.makedirs(f'figures/{simulation_data["simulation_name"]}', exist_ok=True)
    # Save the results to a JSON file
    with open(f'figures/{simulation_data["simulation_name"]}/results.json', 'w') as f:
        json.dump(data, f, indent=4)

    try:
        plot_names = simulation_data["plots"]
        Plots(model.scenario_properties, plot_names, simulation_data["simulation_name"])
    except Exception as e:
        print(e)
        print("No plots specified in the simulation configuration file.")