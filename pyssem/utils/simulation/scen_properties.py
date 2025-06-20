import numpy as np
from math import pi
from datetime import datetime
from scipy.integrate import solve_ivp
from tqdm import tqdm
import sympy as sp
from ..drag.drag import *
from ..launch.launch import ADEPT_traffic_model, SEP_traffic_model
from ..handlers.handlers import download_file_from_google_drive
from ..indicators.indicators import *
import pandas as pd
import os
import multiprocessing
from collections import defaultdict

def lambdify_equation(all_symbolic_vars, eq):
    return sp.lambdify(all_symbolic_vars, eq, 'numpy')

# Function to parallelize lambdification using loky
def parallel_lambdify(equations_flattened, all_symbolic_vars):
    from loky import get_reusable_executor

    # Prepare arguments for parallel processing
    args = [(all_symbolic_vars, eq) for eq in equations_flattened]
    
    # Determine the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    print('Number of cores:', num_cores)
    # Use loky's reusable executor for parallel processing with all available cores
    with get_reusable_executor(max_workers=num_cores) as executor:
        futures = [executor.submit(lambdify_equation, all_symbolic_vars, eq) for all_symbolic_vars, eq in args]
        equations = [future.result() for future in futures]
    
    return equations


class ScenarioProperties:
    def __init__(self, start_date: datetime, simulation_duration: int, steps: int, min_altitude: float, 
                 max_altitude: float, n_shells: int, launch_function: str,
                 integrator: str, density_model: str, LC: float = 0.1, v_imp: float = None, 
                 fragment_spreading: bool = True, parallel_processing: bool = False, baseline: bool = False,
                 indicator_variables: list = None, launch_scenario: str = None, SEP_mapping: str = None,
                 ):
        """
        Constructor for ScenarioProperties. This is the main focal point for the simulation, nearly all other methods are run from this parent class. 
        
        There is no validation here as this should have been completed within the Model class. 
        Args:
            start_date (datetime): Start date of the simulation
            simulation_duration (int): Years of the simulation to run 
            steps (int): Number of steps to run in a simulation 
            min_altitude (float): Minimum Altitude shell in km
            max_altitude (float): Maximum Altitude shell in km
            n_shells (int): Number of Altitude Shells 
            integrator (str, optional): Integrator type. Defaults to "rk4".
            density_model (str, optional): Density Model of Choice. Defaults to "static_exp_dens_func".
            LC (float, optional): Minimum size of fragments [m]. Defaults to 0.1.
            v_imp (float, optional): Impact velocity [km/s]. Defaults to 10.
        """
        self.start_date = start_date
        self.simulation_duration = simulation_duration
        self.end_date = start_date + pd.DateOffset(years=simulation_duration)
        self.steps = steps
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.n_shells = n_shells
        self.launch_function = launch_function
        self.density_model = density_model
        self.integrator = integrator
        self.LC = LC
        self.v_imp = v_imp
        self.SEP_mapping = SEP_mapping
        
        # Set the density model to be time dependent or not, JB2008 is time dependent
        self.time_dep_density = False
        if self.density_model == "static_exp_dens_func":
            self.density_model = static_exp_dens_func
        elif self.density_model == "JB2008_dens_func":
            self.density_model = JB2008_dens_func
            self.time_dep_density = True
        else:
            print("Warning: Unable to parse density model, setting to static exponential density model")
            self.density_model = static_exp_dens_func

        # Indicator Variables
        self.indicator_variables = indicator_variables
        self.indicator_variables_list = []

        # Parameters
        self.scen_times = np.linspace(0, self.simulation_duration, self.steps) 
        self.scen_times_dates = self.calculate_scen_times_dates()
        self.mu = 3.986004418e14  # earth's gravitational constant meters^3/s^2
        self.re = 6378.1366  # radius of the earth [km]

        # MOCAT specific parameters
        R0 = np.linspace(self.min_altitude, self.max_altitude, self.n_shells + 1) # Altitude of the shells [km]
        self.R0_km = R0  # Lower bound of altitude of the shells [km]
        self.HMid = R0[:-1] + np.diff(R0) / 2 # Midpoint of the shells [km]
        self.deltaH = np.diff(R0)[0]  # thickness of the shell [km]
        R0 = (self.re + R0) * 1000  # Convert to meters and the radius of the earth
        self.V = 4 / 3 * pi * np.diff(R0**3)  # volume of the shells [m^3]
        if self.v_imp is not None:
            self.v_imp_all = self.v_imp * np.ones_like(self.V)  # impact velocity [km/s] Shell-wise
        else: 
            # Calculate v_imp for each orbital shell using the vis viva equation
            self.v_imp_all = np.sqrt(2 * self.mu / (self.HMid * 1000)) / 1000  # impact velocity [km/s] Shell-wise
        self.v_imp_all * 1000 * (24 * 3600 * 365.25)  # impact velocity [m/year]
        self.Dhl = self.deltaH * 1000 # thickness of the shell [m]
        self.Dhu = -self.deltaH * 1000 # thickness of the shell [m]
        self.options = {'reltol': 1.e-4, 'abstol': 1.e-4}  # Integration options # these are likely to change
        self.R0 = R0 # gives you the shells <- gives you the top or bottom of shells -> is this needed in python?

        # An empty list for the species
        self.species = []
        self.species_cells = {} #dict with S, D, N, Su, B arrays or whatever species types exist}
        self.species_names = []
        self.species_length = 0
        self.all_symbolic_vars = []
        
        self.collision_pairs = [] 

        # Parameters for simulation
        self.full_Cdot_PMD = sp.Matrix([])
        self.full_lambda = sp.Matrix([])
        self.full_coll = sp.Matrix([])
        self.full_drag = sp.Matrix([])
        self.equations = sp.Matrix([])
        self.drag_term_upper = None
        self.drag_term_cur = None
        self.sym_drag = False
        self.coll_eqs_lambd = None # Used for OPUS when only collision equations are required
        
        # Outputs
        self.output = None

        # Restults
        self.results = None

        # Varying collision shells 
        self.fragment_spreading = fragment_spreading

        # Parallel Processing
        self.parallel_processing = parallel_processing

        # Baseline Scenario
        self.baseline = baseline  

        # Integator Results
        self.indicator_results = {}  

        # Progress bar for the final integration
        self.progress_bar = None

        # Launch Scenario
        self.launch_scenario = launch_scenario

    def calculate_scen_times_dates(self):
        # Calculate the number of months for each step
        months_per_step = self.simulation_duration / self.steps
        
        # Initialize an empty array for storing the dates
        scen_times_dates = np.empty_like(self.scen_times, dtype='datetime64[M]')
        
        # Calculate the dates for each step
        for i, time in enumerate(self.scen_times):
            # Calculate the number of months to add based on the time step
            months_to_add = int(round(time / months_per_step))
            
            # Add the months to the start date
            date = self.start_date + pd.DateOffset(months=months_to_add)
            
            # Store the date in the array
            scen_times_dates[i] = date
        
        return scen_times_dates
    
    def add_species_set(self, species_list: list, all_symbolic_vars: None):
        """
        Adds a list of species to the overall scenario properties. 
        It will update the species_cell dictionary with the species types as the keys and the species as the values.

        :param species_list: List of species to add to the scenario
        :type species_list: list
        :param all_symbolic_vars: List of symbolic variables for the species: List of Symbolic variables, optional.
        """
        for species_group in species_list.values():
            for species in species_group:
                # If _ does not exist in the species name, match it straight to the key 
                if "_" not in species.sym_name:
                    #self.species_cells[species.name] = species
                    name = species.sym_name
                else: 
                    # If _ does exist, the key is the before _
                    name = species.sym_name.split("_")[0]

                # If the key does not exist, create a new list with the species
                if name not in self.species_cells:
                    self.species_cells[name] = [species]
                else:
                    # If the key does exist, append the species to the list
                    self.species_cells[name].append(species)
                
                self.species_length += 1
                self.species_names.append(species.sym_name)
    
        self.species = species_list

        if all_symbolic_vars:
            self.all_symbolic_vars = all_symbolic_vars

    def add_collision_pairs(self, collision_pairs: list):
        """
        Adds a list of collision pairs to the overall scenario properties. 

        :param collision_pairs: List of collision pairs to add to the scenario
        :type collision_pairs: list
        """
        self.collision_pairs = collision_pairs

    
    def future_launch_model(self, FLM_steps):
        """
        This will take the FLM steps and convert them into lambda functions for each species. 
        The code uses the np.arrays() to create the number of objects launched into each shell, for each species. These are then interpolated at simulation time. 

        It does not return anything, but updates the species objects with the lambda functions.

        :param FLM_steps: The FLM steps from the launch file
        :type FLM_steps: pd.DataFrame
        """
        # Check for consistent time step
        scen_times = np.array(self.scen_times)
        if len(np.unique(np.round(np.diff(scen_times), 5))) == 1:
            time_step = np.unique(np.round(np.diff(scen_times), 5))[0]
        else:
            raise ValueError("FLM to Launch Function is not set up for variable time step runs.")

        for species_group in self.species.values():
            for species in species_group:

                # Extract the species columns, with altitude and time
                if species.sym_name in FLM_steps.columns:
                    temp_df = FLM_steps.loc[:, ['alt_bin', 'epoch_start_date', species.sym_name]]

                else:
                    continue

                species_FLM = temp_df.pivot(index='alt_bin', columns='epoch_start_date', values=species.sym_name)

                # Convert spec_FLM to interpolating functions (lambdadot) for each shell
                # Remember indexing starts at 0 (40th shell is index 39)
                species.lambda_funs = []
                
                if species.launch_altitude is not None:
                    closest_shell = np.argmin(np.abs(self.HMid - species.launch_altitude))

                for shell in range(self.n_shells):
                    y = species_FLM.loc[shell, :].values / time_step  

                    if species.launch_altitude is not None and shell == closest_shell:
                        # Add the lambda_constant to each value in the array y
                        y += species.lambda_constant

                    if np.all(y == 0):
                        species.lambda_funs.append(None)  
                    else:
                        species.lambda_funs.append(np.array(y))

    def build_indicator_variables(self):
        """
            This will create the indicator variables for the simulation. The different indicators will be provided by the user. 
            As new indicators are added, they will need to be included first here to pass from the input JSON. 
        """
        if self.indicator_variables is None:
            return
        try:
            for indicator in self.indicator_variables:
                if indicator == "orbital_volume":
                    self.indicator_variables_list.append(make_intrinsic_cap_indicator(self, sep_dist_method="distance", 
                                                                                        sep_dist=60, 
                                                                                        inc = 40, 
                                                                                        shell_sep=5,
                                                                                        graph=True))
                elif indicator == "ca_man_struct":
                    self.indicator_variables_list.append(make_ca_counter(self, "maneuverable", "trackable", 
                                                                            per_species=True, per_spacecraft=True))
                elif indicator == "ca_man_struct_agg":
                    self.indicator_variables_list.append(make_ca_counter(self, "maneuverable", "trackable", 
                                                                            per_species=False, per_spacecraft=False))
                elif indicator == "active_loss_per_shell":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = False, 
                                                                                    per_species = False))
                elif indicator == "active_loss_per_shell_percentage":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = True, 
                                                                                    per_species = False))
                elif indicator == "active_loss_per_species":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = False, 
                                                                                    per_species = True))
                elif indicator == "active_loss_per_species_percentage":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = True, 
                                                                                    per_species = True))
                elif indicator == "umpy":
                    self.indicator_variables_list.append(make_umpy_indicator(self,
                                                                             X=4
                                                                             ))
                elif indicator == "all_col_indicators":
                    self.indicator_variables_list.append(make_all_col_indicators(self))
        
        except Exception as e:
            print(f"An error occurred creating the indicator variables: {str(e)}")
            print("Continuing without indicator variables.")
            self.indicator_variables = None
            self.indicator_variables_list = []
            return
        
    def configure_active_satellite_loss(self, fringe_satellites):
        """
            This will find the equations that have been created by the active_loss_per_species, then lambdify the equations and save them separately. 

            This function is normally required for the OPUS model.

            Parameters:
                fringe_satellites (str): Fringe Satellite Name
        """

        fringe_satellite_items = [
            item for sublist in self.indicator_variables_list for item in sublist 
            if item.name.startswith("Su")
        ]

        # there should only be one item
        if len(fringe_satellite_items) != 1:
            raise ValueError("There should only be one fringe satellite. Multiple found.")
        
        fringe_satellite_items = fringe_satellite_items[0].eqs

        # Lambdify the equations
        simplified_eqs = sp.simplify(fringe_satellite_items)
        self.fringe_active_loss = sp.lambdify(self.all_symbolic_vars, simplified_eqs, 'numpy')
        
        return
    
    def calculate_umpy_for_opus(self, state_matrix):
        """
            Calculate the undispossed mass per year (UMPY) from the current state_matrix using indicator variables.
        """

        # if self.umpy_lambdified exists
        if not hasattr(self, 'umpy_lambdified'):
            # Get the index of umpy in list
            umpy_index = self.indicator_variables.index("umpy")

            # Use this index to get the indicator vars
            umpy_eqs = self.indicator_variables_list[umpy_index][0].eqs
            
            # Simplify and Lambdify the equations
            simplified_eqs = sp.simplify(umpy_eqs)
            self.umpy_lambdified = sp.lambdify(self.all_symbolic_vars, simplified_eqs, 'numpy')

        # Calculate the UMPY
        umpy = self.umpy_lambdified(*state_matrix)
        
        return umpy


    def initial_pop_and_launch(self, baseline=False, launch_file=None):
        """
           This function will determine which launch file to use. 
           Users must select on of the Space Environment Pathways (SEPs), see: https://www.researchgate.net/publication/385299836_Development_of_Reference_Scenarios_and_Supporting_Inputs_for_Space_Environment_Modeling

           There are seven possible launch scenarios:
                SEP1: No Future Launch 

                SEP 2: Continuing Current Behaviours 

                SEP 3 M: Space Winter (Medium Sustainability Effort) 

                SEP 3 H: Space Winter (High Sustainability Effort) 

                SEP 4: Strategic Rivalry 

                SEP 5 M: Commercial-driven Development (Medium Sustainability Effort) 

                SEP 5 H: Commercial-driven Development (High Sustainability Effort) 

                SEP 6 M: Intensive Space Demand (Medium Sustainability Effort) 

                SEP 6 H: Intensive Space Demand (High Sustainability Effort) 
        """

        launch_file_path = os.path.join('pyssem', 'utils', 'launch', 'data',f'ref_scen_{launch_file}.csv')
        
        # Check to see if the data folder exists, if not, create it
        if not os.path.exists(os.path.join('pyssem', 'utils', 'launch', 'data')):
            os.makedirs(os.path.join('pyssem', 'utils', 'launch', 'data'))

        # Check to see if launch_file_path exists
        if not os.path.exists(launch_file_path):
            raise FileNotFoundError(f"Launch file {launch_file_path} does not exist. Please provide a valid launch file.")
        
        print('Using launch file:', launch_file_path)

        [x0, FLM_steps] = SEP_traffic_model(self, launch_file_path)

        # Store as part of the class, as it is needed for the run_model()
        self.x0 = x0
        self.FLM_steps = FLM_steps

        # Export x0 to csv
        x0.to_csv(os.path.join('pyssem', 'utils', 'launch', 'data', 'x0.csv'))

        if not baseline:
            self.future_launch_model(FLM_steps)

    def initial_pop_and_launch2(self, baseline=False):
        """
        Generate the initial population and the launch rates. 
        The Launch File path should be within the launch/data folder, however, it is not, then download it from Google Drive.
        
        Returns: None
        """
        launch_file_path = os.path.join('pyssem', 'utils', 'launch', 'data', 'x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv')

        # Check to see if the data folder exists, if not, create it
        if not os.path.exists(os.path.join('pyssem', 'utils', 'launch', 'data')):
            os.makedirs(os.path.join('pyssem', 'utils', 'launch', 'data'))

        if os.path.exists(launch_file_path):
            filepath = launch_file_path
        else:
            print('As no file is provided. Downloading a launch file...:')
            file_id = '1O8EAyGhydH0Qj2alZEeEoj0dJLy7c5KE' # This is a google docs link - eventually should be added as a .env
            
            download_file_from_google_drive(file_id, launch_file_path)

            # Check to see if the file has been downloaded
            if os.path.exists(launch_file_path):
                filepath = launch_file_path
                print('File downloaded successfully.')
            else:
                print('Failed to download the file.')

        # Example usage: print the filepath to verify
        print("File used for launch model:", filepath)
              
        [x0, FLM_steps] = ADEPT_traffic_model(self, filepath, baseline)

        # Store as part of the class, as it is needed for the run_model()
        self.x0 = x0
        self.FLM_steps = FLM_steps

        # Export x0 to csv
        x0.to_csv(os.path.join('pyssem', 'utils', 'launch', 'data', 'x0.csv'))

        if not baseline:
            self.future_launch_model(FLM_steps)
    
    def build_model(self):
        """
        Build the model for the simulation. This will convert the equations to lambda functions and run the simulation.

        This does not take any arguments, as the ScenarioProperties should now be fully configured. It will go through each species, launch, pmd, drag and collisions equations
        and add them shape them into a matrix of symbolic expressions. 

        :return: None
        """
        t = sp.symbols('t')

        species_list = [species for group in self.species.values() for species in group]
        self.full_Cdot_PMD = sp.zeros(self.n_shells, self.species_length)
        self.full_lambda = []
        self.full_coll = sp.zeros(self.n_shells, self.species_length)
        self.drag_term_upper = sp.zeros(self.n_shells, self.species_length)
        self.drag_term_cur = sp.zeros(self.n_shells, self.species_length)

        # Equations are going to be a matrix of symbolic expressions
        # Each row corresponds to a shell, and each column corresponds to a species
        for i, species in enumerate(species_list):

            lambda_expr = species.launch_func(self.scen_times, self.HMid, species, self)
            self.full_lambda.append(lambda_expr)

            # Post mission Disposal
            Cdot_PMD = species.pmd_func(t, self.HMid, species, self)
            self.full_Cdot_PMD[:, i] = Cdot_PMD

            # Drag
            [upper_term, current_term] = species.drag_func(t, self.HMid, species, self)
            try:
                self.drag_term_upper[:, i] = upper_term
                self.drag_term_cur[:, i] = current_term
            except:
                continue
        
        # Collisions
        for i in self.collision_pairs:
            self.full_coll += i.eqs

        self.equations = sp.zeros(self.n_shells, self.species_length)      
        self.equations = self.full_Cdot_PMD + self.full_coll

        # Recalculate objects based on density, as this is time varying 
        if not self.time_dep_density: 
            # Take the shell altitudes, this will be n_shells + 1
            rho = self.density_model(0, self.R0_km, self.species, self)
            rho_reshape = rho.reshape(-1, 1) # Convert to column vector
            rho_mat = np.tile(rho_reshape, (1, self.species_length)) 
            rho_mat = sp.Matrix(rho_mat)
            
            # Second to last row
            upper_rho = rho_mat[1:, :]
            
            # First to penultimate row (mimics rho_mat(1:end-1, :))
            current_rho = rho_mat[:-1, :]

            drag_upper_with_density = self.drag_term_upper.multiply_elementwise(upper_rho)
            drag_cur_with_density = self.drag_term_cur.multiply_elementwise(current_rho)
            self.full_drag = drag_upper_with_density + drag_cur_with_density
            self.equations += self.full_drag
            self.sym_drag = True 

        # Make Integrated Indicator Variables if passed
        if hasattr(self, 'integrated_indicator_var_list'):
            integrated_indicator_var_list = self.integrated_indicator_var_list
            for ind_var in integrated_indicator_var_list:
                if not ind_var.eqs:
                    ind_var = self.make_indicator_eqs(ind_var)

            self.num_integrated_indicator_vars = 0
            end_indicator_idxs = len(self.xdot_eqs)

            for ind_var in integrated_indicator_var_list:
                num_add_indicator_vars = len(ind_var.eqs)
                self.num_integrated_indicator_vars += num_add_indicator_vars

                start_indicator_idxs = end_indicator_idxs + 1
                end_indicator_idxs = start_indicator_idxs + num_add_indicator_vars - 1
                ind_var.indicator_idxs = list(range(start_indicator_idxs, end_indicator_idxs + 1))

                self.xdot_eqs = sp.Matrix.vstack(self.xdot_eqs, sp.Matrix(ind_var.eqs))

            if not self.sym_lambda:
                indicator_pad = [lambda x, t: 0] * self.num_integrated_indicator_vars
                self.full_lambda.extend(indicator_pad)

        # Non Integrated Indicator Variables should already be compiled - so just used in run_model()
                
        # Dont add drag if time dependent density, this will be added during integration due to time dependent density
        if self.time_dep_density:
            self.full_drag = self.drag_term_upper + self.drag_term_cur

        # Lambdify the equations to be used for Scipy integration
        collisions_flattened = [self.full_coll[i, j] for j in range(self.full_coll.cols) for i in range(self.full_coll.rows)]
        self.coll_eqs_lambd = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in collisions_flattened]

        self.equations, self.full_lambda_flattened = self.lambdify_equations(), self.lambdify_launch()       
  
        return
    
    def lambdify_equations(self):
        """
            Convert the Sympy symbolic equations to lambda functions, this allows for a quicker integration for SciPy.

            Returns: equations, full_lambda_flattened
        """

        equations_flattened = [self.equations[i, j] for j in range(self.equations.cols) for i in range(self.equations.rows)]

        # Convert the equations to lambda functions
        if self.parallel_processing:
            equations = parallel_lambdify(equations_flattened, self.all_symbolic_vars)
        else:
            equations = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in equations_flattened]

        return equations

    def lambdify_launch(self, full_lambda=None):
        """ 
            Convert the Numpy launch rates to Scipy lambdified functions for integration.
        
        """
        # Launch rates
        full_lambda_flattened = []

        if full_lambda is None:
            for i in range(len(self.full_lambda)):
                if self.full_lambda[i] is not None:
                    full_lambda_flattened.extend(self.full_lambda[i])
                else:
                    # Append None to the list, length of scenario_properties.n_shells
                    full_lambda_flattened.extend([None]*self.n_shells)
        else:
            for i in range(len(full_lambda)):
                if full_lambda[i] is not None:
                    full_lambda_flattened.extend(full_lambda[i])
                else:
                    # Append None to the list, length of scenario_properties.n_shells
                    full_lambda_flattened.extend([None]*self.n_shells)

        return full_lambda_flattened
    
    def run_model(self):
        """
        For each species, integrate the equations of population change for each shell and species.

        The starting point will be, x0, the initial population.

        The launch rate will be first calculated at time t, then the change of population in that species will be calculated using the ODEs. 

        :return: None
        """
        print("Preparing equations for integration (Lambdafying) ...")
        
        # Initial Population
        x0 = self.x0.T.values.flatten()

        if self.time_dep_density:
            # Drag equations will have to be lamdified separately as they will not be part of equations_flattened
            drag_upper_flattened = [self.drag_term_upper[i, j] for j in range(self.drag_term_upper.cols) for i in range(self.drag_term_upper.rows)]
            drag_current_flattened = [self.drag_term_cur[i, j] for j in range(self.drag_term_cur.cols) for i in range(self.drag_term_cur.rows)]

            self.drag_upper_lamd = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in drag_upper_flattened]
            self.drag_cur_lamd = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in drag_current_flattened]

            # Set up time varying density 
            self.density_data = preload_density_data(os.path.join('pyssem', 'utils', 'drag', 'dens_highvar_2000_dens_highvar_2000_lookup.json'))
            self.date_mapping = precompute_date_mapping(pd.to_datetime(self.start_date), pd.to_datetime(self.end_date) + pd.DateOffset(years=self.simulation_duration
                                                                                                                                       ))
            
            # This will change when jb2008 is updated
            available_altitudes = list(map(int, list(self.density_data['2020-03'].keys())))
            available_altitudes.sort()

            self.nearest_altitude_mapping = precompute_nearest_altitudes(available_altitudes)

            self.prev_t = -1  # Initialize to an invalid time
            self.prev_rho = None


            print("Integrating equations...")
            output = solve_ivp(self.population_shell_time_varying_density, [self.scen_times[0], self.scen_times[-1]], x0,
                            args=(self.full_lambda_flattened, self.equations, self.scen_times),
                            t_eval=self.scen_times, method=self.integrator)
            
            self.drag_upper_lamd = None
            self.drag_cur_lamd = None

        else:
            self.progress_bar = tqdm(total=self.scen_times[-1] - self.scen_times[0], desc="Integrating Equations", unit="year")

            output = solve_ivp(self.population_shell, [self.scen_times[0], self.scen_times[-1]], x0,
                            args=(self.full_lambda_flattened, self.equations, self.scen_times),
                            t_eval=self.scen_times, method=self.integrator)
            
            self.progress_bar.close()
            self.progress_bar = None # Set back to None becuase a tqdm object cannot be pickled

        if output.success:
            print(f"Model run completed successfully.")
        else:
            print(f"Model run failed: {output.message}")

        self.output = output # Save

        # Indicator Variables
        # Evaluate non-indicator variables using states
        if hasattr(self, 'indicator_variables_list'):
            print("Evaluating post-processed indicator variables...")
            self.indicator_results['indicators'] = {}

            for i in self.indicator_variables_list:
                # Convert the symbolic equations into a callable function
                for indicator_var in i:
                    simplified_eqs = sp.simplify(indicator_var.eqs)
                    indicator_fun = sp.lambdify(self.all_symbolic_vars, simplified_eqs, 'numpy')
                    evaluated_indicator_dict = {}

                    # Iterate over states (rows in y) and corresponding time steps (t)
                    for state, t in zip(self.output.y.T, self.output.t):
                        # Evaluate the indicator function for the current state
                        evaluated_value = indicator_fun(*state)
                        # Store the result in the dictionary with the corresponding time step
                        evaluated_indicator_dict[t] = evaluated_value

                    # Store the results for this indicator in the results dictionary
                    self.indicator_results['indicators'][indicator_var.name] = evaluated_indicator_dict


        return 
    
    def propagate(self, population, times, launch=None):
        """
            This will use the equations that have been built already by the model, and then integrate the differential equations
            over a chosen timestep. The population and launch (if provided) must be the same length as the species and shells.

            :param population: Initial population
            :param times: Times to integrate over
            :param launch: Launch rates

            :return: results_matrix
        """
        # check to see if the equations have already been lamdified
        if self.equations is None:
            self.equations = self.lambdify_equations()

        # if launch is not None:
        #     full_lambda_flattened = self.lambdify_launch(launch)

        output = solve_ivp(self.population_shell_for_OPUS, [times[0], times[-1]], population,
                            args=(self.equations, times, launch), 
                            t_eval=times, method=self.integrator)
        
        if output.success:
            # Extract the results at the specified time points
            results_matrix = output.y.T  # Transpose to make it [time, variables]
            return results_matrix
        else:
            print(f"Model run failed: {output.message}")
            return None

        
    def population_shell_for_OPUS(self, t, N, equations, times, launch):
        dN_dt = np.zeros_like(N)

        # Iterate over each component in N
        for i in range(len(N)):
        
            # Compute and add the external modification rate, if applicable
            # Now using np.interp to calculate the increase
            if launch[i] is not None:
                # increase = np.interp(t, times, launch[i])
                increase = launch[i]
                # If increase is nan set to 0
                if np.isnan(increase) or np.isinf(increase) or increase is None:
                    increase = 0
                else:
                    dN_dt[i] += increase

            # Compute the intrinsic rate of change from the differential equation
            change = equations[i](*N)
        
            dN_dt[i] += change

        return dN_dt

    def population_shell_time_varying_density(self, t, N, full_lambda, equations, times):
        """
        Seperate function to ScenarioProperties, this will be used in the solve_ivp function.

        :param t: Timestep
        :param N: Population Count
        :param full_lambda: Launch rates
        :param equations: Equations
        :param times: Times
        :param density_model: Density Model
        :param R0_km: Altitude of the shells

        :return: Rate of change of population
        """
        print(f"Time: {t}")
        dN_dt = np.zeros_like(N)

        if self.time_dep_density:
            # Cache management logic for rho
            current_t_step = int(t)
            if current_t_step > self.prev_t:
                rho = JB2008_dens_func(t, self.R0_km, self.density_data, self.date_mapping, self.nearest_altitude_mapping)
                self.prev_rho = rho
                self.prev_t = current_t_step
            else:
                rho = self.prev_rho  # Use cached rho

            rho_full = np.repeat(rho, self.species_length)

            species_per_shell = self.species_length

            # Apply drag computations
            for i in range(len(N)):
                shell_index = i // species_per_shell

                # Ensure drag_cur_lamd and drag_upper_lamd functions are correctly accessed and used
                if i < len(N) - 1:
                    current_drag = self.drag_cur_lamd[i](*N) * rho_full[shell_index]
                    upper_drag = self.drag_upper_lamd[i](*N) * rho_full[shell_index + 1]
                    dN_dt[i] += current_drag + upper_drag
                else:
                    current_drag = self.drag_cur_lamd[i](*N) * rho_full[shell_index]
                    dN_dt[i] += current_drag

                # Handle incoming new species
                if full_lambda[i] is not None:
                    increase = np.interp(t, times, full_lambda[i])
                    dN_dt[i] += 0 if np.isnan(increase) else increase

                # Apply general equation dynamics
                dN_dt[i] += equations[i](*N)

        return dN_dt
    
    def population_shell(self, t, N, full_lambda, equations, times, progress_bar=True):
        """
        Seperate function to ScenarioProperties, this will be used in the solve_ivp function.

        :param t: Timestep (int)
        :param N: Population Count (Flattened array of species and shells)
        :param full_lambda: Launch rates (Flattened np.array of species and shells)
        :param equations: Equations (Lambdified sympy functions for each species and shell)
        :param times: Times (Times for the simulation, usually years)

        :return: Rate of change of population at the given timestep, t. 
        """
        # Update the progress bar
        if self.progress_bar is not None and progress_bar:
            self.progress_bar.update(t - self.progress_bar.n)

        # Initialize the rate of change array
        dN_dt = np.zeros_like(N)

        # Iterate over each component in N
        for i in range(len(N)):
        
            # Compute and add the external modification rate, if applicable
            # Now using np.interp to calculate the increase
            if full_lambda[i] is not None:
                increase = np.interp(t, times, full_lambda[i])
                # If increase is nan set to 0
                if np.isnan(increase) or np.isinf(increase):
                    increase = 0
                else:
                    dN_dt[i] += increase

            # Compute the intrinsic rate of change from the differential equation
            dN_dt[i] += equations[i](*N)

        return dN_dt

    def cum_CSI(self):
        k = 0.6
        def life(h):
            return np.exp(14.18 * h ** 0.1831 - 42.94)

        M_ref = 10000 # kg
        h_ref = 1000 # km
        life_h_ref = 1468 # years, it corresponds to life0 = life(1000)

        if isinstance(self.results, str):
            self.results = json.loads(self.results)

        initial_populations = [data['populations'][0] for data in self.results['population_data']]
        V = np.array(self.V)
        D_ref = np.max(np.sum(initial_populations, axis=0) / V)
        
        den = M_ref * D_ref * life_h_ref * (1+k) / 10
        #den = 2.4477e-09

        cos_i_av = 2/pi #average value of cosine of inclination in the range -pi/2 pi/2 calculated using integral average
        Gamma_av = (1-cos_i_av)/2

        rgb_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        def life(h):
            return np.exp(14.18 * h**0.1831 - 42.94)

        if hasattr(self, 'results'):
            print("Producing two visuals of CSI.")
            plt.figure()
            plt.grid(True)
            CSI_S_sum_array = np.zeros((len(self.results['times']), 0))
            CSI_D_sum_array = np.zeros((len(self.results['times']), 0))
            
            unique_species = set([data['species'] for data in self.results['population_data']])
            
            for i2, species in enumerate(unique_species):
                if i2 >= len(rgb_c):
                    colorset = np.random.rand(3)
                else:
                    colorset = rgb_c[i2]
                
                CSI_X_mat = np.zeros((len(self.results['times']), self.n_shells))
                species_list = [sp for species_group in self.species.values() for sp in species_group]

                if 'S' in species or 'D' in species:
                    for i in range(self.n_shells):
                        shell_data = [data for data in self.results['population_data'] if data['species'] == species and data['shell'] == (i + 1)]
                        if shell_data:
                            life_i = life((self.R0_km[i] + self.R0_km[i + 1]) / 2)
                            num = life_i * (1 + k * Gamma_av)
                            try:
                                mass = next((item.mass for item in species_list if item.sym_name == species), 0)
                            except TypeError as e:
                                print(f"Error accessing species_properties for species '{species}': {e}")
                                print(f"species_list: {species_list}")
                                raise
                            dum_X = mass * num
                            D_X = np.array(shell_data[0]['populations']) / self.V[i]
                            CSI_X_mat[:, i] = D_X * dum_X
                    
                    CSI_X_mat /= den
                    CSI_X = np.sum(CSI_X_mat, axis=1)
                    plt.plot(self.results['times'], CSI_X, label=f'CSI for {species.replace("p", ".")}', linewidth=2, color=colorset)
                    
                    if 'S' in species and 'D' not in species:
                        CSI_S_sum_array = np.column_stack((CSI_S_sum_array, CSI_X))
                    elif 'D' in species:
                        CSI_D_sum_array = np.column_stack((CSI_D_sum_array, CSI_X))

            if CSI_S_sum_array.shape[1] > 0:
                CSI_S_sum = np.sum(CSI_S_sum_array, axis=1)
            else:
                CSI_S_sum = np.zeros(len(self.results['times']))

            if CSI_D_sum_array.shape[1] > 0:
                CSI_D_sum = np.sum(CSI_D_sum_array, axis=1)
            else:
                CSI_D_sum = np.zeros(len(self.results['times']))

            plt.plot(self.results['times'], CSI_S_sum + CSI_D_sum, label='Total CSI', linewidth=2, color='black', linestyle='--')
            plt.xlabel('Time (years)')
            plt.ylabel('CSI')
            plt.title('Cumulative Space Index (CSI) per Species')
            plt.xlim([0, np.max(self.results['times'])])
            plt.legend(loc='best', frameon=False)
            plt.savefig('figures/CSI_per_species.png')

            plt.figure()
            plt.grid(True)
            plt.plot(self.results['times'], CSI_S_sum, label='Total CSI for Active Satellites', linewidth=2, color='#1f77b4')
            plt.plot(self.results['times'], CSI_D_sum, label='Total CSI for Derelict Satellites', linewidth=2, color='#ff7f0e')
            plt.plot(self.results['times'], CSI_S_sum + CSI_D_sum, label='Total CSI', linewidth=2, color='black', linestyle='--')
            plt.xlabel('Time (years)')
            plt.ylabel('Cumulative CSI')
            plt.xlim([0, np.max(self.results['times'])])
            plt.title('Cumulative Space Index (CSI) for Active and Derelict Species')
            plt.legend(loc='best', frameon=False)
            plt.savefig('figures/CSI_active_derelict.png')
        else:
            raise ValueError("Simulation does not contain results. Please run the function run_model(x0) to produce simulation results required for CSI computation.")
        
        return
    