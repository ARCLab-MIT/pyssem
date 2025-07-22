import numpy as np
from math import pi
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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

class SymbolicCollisionTerm:
    def __init__(self, s1_idx, s2_idx, eqs_sources, eqs_sinks, fragment_spread_totals):
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.eqs_sources = eqs_sources     # list of sympy expressions
        self.eqs_sinks = eqs_sinks         # list of sympy expressions

        # Optionally lambdify now or later
        self.lambdified_sources = None
        self.lambdified_sinks = None

        # This is for the distribution of the fragments across a, e
        self.fragment_spread_totals = fragment_spread_totals

class StepFunction:
    """
    A callable object that acts as a fast, piecewise constant step function
    for evenly spaced time series data.
    """
    def __init__(self, start_time, time_step_duration, rate_values):
        self.start_time = start_time
        self.time_step_duration = time_step_duration
        self.rate_values = np.array(rate_values)
        self.num_steps = len(rate_values)

    def __call__(self, t):
        """
        This makes the object callable, e.g., func(t).
        It finds the correct index for time 't' and returns the corresponding rate.
        """
        # If t is outside the defined time range, return 0
        if t < self.start_time or t >= self.start_time + self.num_steps * self.time_step_duration:
            return 0.0

        # Calculate the index for the time step
        # This is extremely fast because the steps are uniform.
        index = int((t - self.start_time) / self.time_step_duration)
        
        # Clamp the index to be within the valid range of the array
        index = min(index, self.num_steps - 1)
        
        return self.rate_values[index]

def lambdify_equation(all_symbolic_vars, eq):
    return sp.lambdify(all_symbolic_vars, eq, 'numpy')

# Function to parallelize lambdification using loky
def parallel_lambdify(equations_flattened, all_symbolic_vars):
    from loky import get_reusable_executor

    # Prepare arguments for parallel processing
    from loky import get_reusable_executor
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
                 elliptical: bool = False, eccentricity_bins: list = None
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
        # semi-major-axis midpoints in meters:
        R0_alt_km = np.linspace(self.min_altitude, self.max_altitude, self.n_shells + 1)
        self.R0_rad_km = self.re + R0_alt_km          # length = n_shells+1
        self.sma_HMid_km = 0.5 * (self.R0_rad_km[:-1] + self.R0_rad_km[1:]) 
        self.R0_km = R0  # Lower bound of altitude of the shells [km]
        self.HMid = R0[:-1] + np.diff(R0) / 2 # Midpoint of the shells [km]
        self.deltaH = np.diff(R0)[0]  # thickness of the shell [km]
        R0 = (self.re + R0) * 1000  # Convert to meters and the radius of the earth
        self.V = 4 / 3 * pi * np.diff(R0**3)  # volume of the shells [m^3]
        if self.v_imp is not None:
            self.v_imp2 = self.v_imp * np.ones_like(self.V)  # impact velocity [km/s] Shell-wise
        else: 
            # Calculate v_imp for each orbital shell using the vis viva equation
            self.v_imp2 = np.sqrt(2 * self.mu / (self.HMid * 1000)) / 1000  # impact velocity [km/s] Shell-wise
        self.v_imp2 * 1000 * (24 * 3600 * 365.25)  # impact velocity [m/year]
        self.Dhl = self.deltaH * 1000 # thickness of the shell [m]
        self.Dhu = -self.deltaH * 1000 # thickness of the shell [m]
        self.options = {'reltol': 1.e-4, 'abstol': 1.e-4}  # Integration options # these are likely to change
        self.R0 = R0 # gives you the shells <- gives you the top or bottom of shells -> is this needed in python?

        # An empty list for the species
        self.species = []
        self.species_types = []
        self.species_cells = {} #dict with S, D, N, Su, B arrays or whatever species types exist}
        self.species_names = []
        self.debris_names = []
        self.debris_length = 0
        self.species_length = 0
        self.all_symbolic_vars = []
        
        self.collision_pairs = [] 

        # Elliptical orbits
        self.elliptical = elliptical
        self.eccentricity_bins = eccentricity_bins
        self.time_in_shell = None

        # Parameters for simulation
        self.full_Cdot_PMD = sp.Matrix([])
        self.full_lambda = sp.Matrix([])
        self.full_coll = sp.Matrix([])
        self.full_drag = sp.Matrix([])
        self.equations = sp.Matrix([])
        self.drag_term_upper = None
        self.drag_term_cur = None
        self.sym_drag = False
        
        # Outputs
        self.output = None
        self.prev_t = -1
        self.launch = np.zeros(self.species_length * self.n_shells)

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

    def add_collision_pairs(self, collision_pairs):
        """
        Adds a list of collision pairs to the overall scenario properties. 

        :param collision_pairs: List of collision pairs to add to the scenario
        :type collision_pairs: list
        """
        self.collision_pairs = collision_pairs

    def future_launch_model(self, FLM_steps):
        """
        Processes FLM_steps to assign raw launch values per shell and year to each species.
        No interpolation is performed. Each shell's launch values are stored as a list
        where each entry corresponds to a specific year.

        Updates `species.lambda_funs` to be a list of arrays: one per shell.

        :param FLM_steps: DataFrame containing 'alt_bin', 'epoch_start_date', and launch values.
        """

        for species_group in self.species.values():
            for species in species_group:
                # Skip if the species is not in the FLM
                if species.sym_name not in FLM_steps.columns:
                    species.lambda_funs = [None for _ in range(self.n_shells)]
                    continue

                # Extract relevant FLM data for this species
                temp_df = FLM_steps.loc[:, ['alt_bin', 'epoch_start_date', species.sym_name]]
                species_FLM = temp_df.pivot(index='alt_bin', columns='epoch_start_date', values=species.sym_name)

                species.lambda_funs = []

                if species.launch_altitude is not None:
                    closest_shell = np.argmin(np.abs(self.HMid - species.launch_altitude))
                else:
                    closest_shell = None

                for shell in range(self.n_shells):
                    # Get raw launch counts (no division)
                    y = species_FLM.loc[shell, :].values

                    if closest_shell is not None and shell == closest_shell:
                        y += species.lambda_constant

                    if np.all(y == 0):
                        species.lambda_funs.append(0)
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
                                                                                    per_species = False,
                                                                                    per_pair = False))
                elif indicator == "active_loss_per_shell_percentage":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = True, 
                                                                                    per_species = False, 
                                                                                    per_pair = False))
                elif indicator == "active_loss_per_species":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = False, 
                                                                                    per_species = True, 
                                                                                    per_pair = False))
                elif indicator == "active_loss_per_species_per_pair":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = False, 
                                                                                    per_species = True, 
                                                                                    per_pair = True
                                                                                    ))
                elif indicator == "active_loss_per_species_percentage":
                    self.indicator_variables_list.append(make_active_loss_per_shell(self, 
                                                                                    percentage = True, 
                                                                                    per_species = True, 
                                                                                    per_pair=False))
                elif indicator == "all_col_indicators":
                    self.indicator_variables_list.append(make_all_col_indicators(self))
        
        except Exception as e:
            print(f"An error occurred creating the indicator variables: {str(e)}")
            print("Continuing without indicator variables.")
            self.indicator_variables = None
            self.indicator_variables_list = []
            return

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
        # x0.to_csv(os.path.join('pyssem', 'utils', 'launch', 'data', 'x0.csv'))

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
        if self.elliptical:
            self.collision_terms = []   # flat list of SymbolicCollisionTerm objects
            self.full_coll_sink = []    # optionally initialize here
            self.full_coll_source = []

            for i in self.collision_pairs:
                # Accumulate global source/sink expressions
                self.full_coll_sink += i.eqs_sinks
                self.full_coll_source += i.eqs_sources

                # Get indices of the two species from sym names
                s1_idx = self.species_names.index(i.species1.sym_name)
                s2_idx = self.species_names.index(i.species2.sym_name)

                # Create and store the symbolic collision term
                term = SymbolicCollisionTerm(
                    s1_idx=s1_idx,
                    s2_idx=s2_idx,
                    eqs_sources=i.eqs_sources,
                    eqs_sinks=i.eqs_sinks, 
                    fragment_spread_totals=i.fragment_spread_totals
                )

                self.collision_terms.append(term)

            self.equations = self.full_Cdot_PMD
        else:
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
        # collisions_flattened = [self.full_coll[i, j] for j in range(self.full_coll.cols) for i in range(self.full_coll.rows)]
        # self.coll_eqs_lambd = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in collisions_flattened]

        self.equations, self.full_lambda_flattened = self.lambdify_equations(), self.lambdify_launch()
            
        return

    def build_sym_model(self):
        """
        Build the model for the simulation. This will convert the equations to lambda functions and run the simulation.

        This does not take any arguments, as the ScenarioProperties should now be fully configured. It will go through each species, launch, pmd, drag and collisions equations
        and add them shape them into a matrix of symbolic expressions. 

        :return: None
        """
        t = sp.symbols('t')

        species_list = [species for group in self.species.values() for species in group]
        self.full_Cdot_PMD = sp.zeros(self.n_shells, self.species_length)
        # self.full_lambda = []
        self.full_lambda = sp.zeros(self.n_shells, self.species_length)
        self.full_coll = sp.zeros(self.n_shells, self.species_length)
        self.drag_term_upper = sp.zeros(self.n_shells, self.species_length)
        self.drag_term_cur = sp.zeros(self.n_shells, self.species_length)
        self.full_control = sp.zeros(self.n_shells, self.species_length)

        # Equations are going to be a matrix of symbolic expressions
        # Each row corresponds to a shell, and each column corresponds to a species
        for i, species in enumerate(species_list):

            # lambda_expr = species.launch_func(self.scen_times, self.HMid, species, self)
            # self.full_lambda.append(lambda_expr)
            lambda_expr = species.launch_func(t, self.HMid, species, self)
            self.full_lambda[:, i] = lambda_expr

            # Post mission Disposal
            Cdot_PMD = species.pmd_func(t, self.HMid, species, self)
            self.full_Cdot_PMD[:, i] = Cdot_PMD

            # Control
            U = species.control_func(t, self.HMid, species, self)
            self.full_control[:, i] = U

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
        # self.equations = self.full_Cdot_PMD + self.full_coll
        self.equations = self.full_Cdot_PMD + self.full_coll + self.full_lambda + self.full_control

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
        # collisions_flattened = [self.full_coll[i, j] for j in range(self.full_coll.cols) for i in range(self.full_coll.rows)]
        # self.coll_eqs_lambd = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in collisions_flattened]

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
    
    # === 4. Integration ===
    # def get_dadt(self, a_current, e_current, p, dt):
    #         re   = p['req']
    #         mu   = p['mu']
    #         n0   = np.sqrt(mu) * a_current ** -1.5
    #         a_minus_re = a_current - re
    #         rho0 = densityexp(a_minus_re) * 1e9  # kg/km^3
    #         C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)
    #         # dt   = p['t'] - p['t_0']
    #         ang  = np.arctan((np.sqrt(3)/2)*e_current) - (np.sqrt(3)/2)*e_current * n0 * a_current * C0 * dt
    #         sec2 = 1.0 / np.cos(ang) ** 2
    #         return -(4 / np.sqrt(3)) * (a_current**2 * n0 * C0 / e_current) * np.tan(ang) * sec2

    # def get_dedt(self, a_current, e_current, p, dt):
    #     re   = p['req']
    #     mu   = p['mu']
    #     n0   = np.sqrt(mu) * a_current ** -1.5
    #     beta = (np.sqrt(3)/2) * e_current
    #     a_minus_re = a_current - re
    #     rho0 = densityexp(a_minus_re) * 1e9
    #     C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)
    #     # dt   = p['t'] - p['t_0']
    #     arg  = np.arctan(beta) - beta * n0 * a_current * C0 * dt
    #     sec2 = 1.0 / np.cos(arg) ** 2
    #     return -e_current * n0 * a_current * C0 * sec2

    def sma_ecc_mat_to_altitude_mat(self, population_matrix_sma_ecc):
        """
        Converts a population matrix in semi-major axis and eccentricity space to an altitude matrix.

        Uses the time in shell matrix to distribute the population across altitude shells.
        Args:
            population_matrix_sma_ecc (np.ndarray): Population matrix in semi-major axis and eccentricity space.
                Shape: (n_sma_bins, n_species, n_ecc_bins)  
        Returns:
            np.ndarray: Population matrix in altitude space.
                Shape: (n_alt_shells, n_species)
        """

        effective_altitude_matrix = np.zeros((self.n_alt_shells, self.species_length))

        # Reshape the population matrix to match the time in shell matrix
        for species in range(self.species_length):
            for alt_shell in range(self.n_alt_shells):
                n_effective = 0
                for sma in range(self.n_sma_bins):
                    for ecc in range(self.n_ecc_bins):
                        tis = self.time_in_shell[alt_shell, ecc, sma]
                        n_pop = population_matrix_sma_ecc[sma, species, ecc]
                        n_effective_a_e = n_pop * tis
                        n_effective += n_effective_a_e
                        effective_altitude_matrix[alt_shell, species] += n_effective_a_e

                self.effective_altitude_matrix[alt_shell, species] = n_effective
        
        return effective_altitude_matrix

    def population_rhs(self, t, x_flat, launch_funcs, n_sma_bins, n_species, n_ecc_bins, n_alt_shells,
                      species_to_mass_bin, years, adot, edot, Δa, Δe):
        #############################
        # Reshape the population (3d) into sma, species, ecc
        #############################
        x_matrix = x_flat.reshape((n_sma_bins, n_species, n_ecc_bins))  # shape: (sma_shells, species, ecc)
        time_in_shell = self.time_in_shell  # shape: (alt_shells, sma_shells, ecc)

        #############################
        # We need to loop over each species, then for each sma and ecc pairing, calculate the number of objects in each altitude bin. 
        #  This is the effective_altitude_matrix, as the population is essentially split across the shells based on their time in shell.
        # Secondly, keep track of which a e bins, for each species, are contributing to each shell. Used in the sink equations. (normalised_species_distribution_in_sma_e_space)
        #############################
        self.effective_altitude_matrix = np.zeros((n_alt_shells, n_species))
        normalised_species_distribution_in_sma_e_space = np.zeros((n_alt_shells, n_species, n_sma_bins, n_ecc_bins))
        # for each species, in each shell, trying to find the ae that contribute to those bins. 
        try:
            for species in range(n_species):
                for alt_shell in range(n_alt_shells):
                    n_effective = 0
                    for sma in range(n_sma_bins):
                        for ecc in range(n_ecc_bins):
                            tis = time_in_shell[alt_shell, ecc, sma]
                            n_pop = x_matrix[sma, species, ecc]
                            n_effective_a_e = n_pop * tis
                            n_effective = n_effective + n_effective_a_e
                            normalised_species_distribution_in_sma_e_space[alt_shell, species, sma, ecc] = n_effective_a_e

                    normalised_species_distribution_in_sma_e_space[alt_shell, species, :, :] = ( normalised_species_distribution_in_sma_e_space[alt_shell, species, :, :] / n_effective )
                    # convert any nans to 0
                    normalised_species_distribution_in_sma_e_space[alt_shell, species, :, :] = np.nan_to_num(normalised_species_distribution_in_sma_e_space[alt_shell, species, :, :])
                    self.effective_altitude_matrix[alt_shell, species] = n_effective
        except Exception as e:
            print(f"Error in calculating effective altitude matrix: {e}")
            raise ValueError("The population matrix is not defined correctly. Please check your population matrix.")
        
        total_dNdt_alt = np.zeros((n_alt_shells, n_species))
        total_dNdt_sma_ecc_sources = np.zeros((n_sma_bins, n_species, n_ecc_bins))
        # #############################
        # # Now it is in altitude space, we can remove objects due to post mission disosal
        # #############################
        # try:
        #     flattened = self.effective_altitude_matrix.flatten()
        #     val = np.array([eq(*flattened) for eq in self.full_Cdot_PMD])
        #     self.effective_altitude_matrix += val.reshape((n_alt_shells, n_species))
        # except Exception as e:
        #         print(f"Error in post mission disposal: {e}")
        #         raise ValueError("Post mission disposal equations are not defined correctly. Please check your equations.")
        # === PMD Sink: Apply only to ecc=0 ===


        #############################
        # Our population (x_matrix) is now in the form of altitude and species, which is now for the collision equations.
        #############################    
        x_flat_ordered = self.effective_altitude_matrix.flatten()
        # collision pair in altitude space 
        for term in self.collision_terms:
            dNdt_term = term.lambdified_sources(*x_flat_ordered)
            total_dNdt_alt = np.array(dNdt_term, dtype=float) # n_alt_shells x n_species

            # multiply the growth rate for each species by the distribution of that species in a,e space
            for shell in range(n_alt_shells):
                for species in range(n_species):
                    # Get the mass bin index (skip if not a debris species)
                    mass_bin = species_to_mass_bin.get(species, None)
                    if mass_bin is None:
                        # Add this slice to total_dNdt_sma_ecc as zeros - as no growth fragments
                        continue

                    sma_ecc_distribution = term.spread_distribution[shell, mass_bin, :, :] # this should be to equal to on
                    species_frag = total_dNdt_alt[shell, species] # get the column of the debris species
                    if np.sum(sma_ecc_distribution) == 0 and np.sum(species_frag) != 0:
                        print("fragments made but no distribution in sma_ecc space")
                    frag_spread_sma_ecc = species_frag * sma_ecc_distribution
                    total_dNdt_sma_ecc_sources[:, species, :] = total_dNdt_sma_ecc_sources[:, species, :] + frag_spread_sma_ecc

        #############################
        # Now we need to calculate the sink equations, which are the same as the source equations
        # but multiplied by the time in shell.
        #############################
        dNdt_sink_sma_ecc = np.zeros((n_sma_bins, n_species, n_ecc_bins)) 
        for term in self.collision_terms: # for each species pair
            dNdt_term = term.lambdified_sinks(*x_flat_ordered) # n_shells x n_species
            
            for species in range(n_species): # for each species essentially find where the fragments came from (using effective pop)
                for shell in range(n_alt_shells):
                    frag = dNdt_term[shell, species]
                    norm_a_e = normalised_species_distribution_in_sma_e_space[shell, species, :, :]
                    frag_sink_sma_ecc = frag * norm_a_e
                    dNdt_sink_sma_ecc[:, species, :] = dNdt_sink_sma_ecc[:, species, :] + frag_sink_sma_ecc
                    # if frag_sink_sma_ecc has any nans stop
                    if np.isnan(dNdt_sink_sma_ecc).any():
                        raise ValueError(f"NaN found in dNdt_sink_sma_ecc for species {species} at shell {shell}. Check your collision equations.")
            
        output = total_dNdt_sma_ecc_sources + dNdt_sink_sma_ecc


        # so we no have the change of the points, we need to multiply each species sma and ecc by this matrix of change
        # Loop over species and compute finite-difference transport using rates (no dt)
        dN_all_species = np.zeros_like(x_matrix) 
        for species in range(n_species):
            N_sma_ecc = x_matrix[:, species, :]
            dN = np.zeros_like(N_sma_ecc)

            for sma in range(n_sma_bins - 1, -1, -1):
                for ecc in range(n_ecc_bins - 1, -1, -1):
                    Nrc = N_sma_ecc[sma, ecc]
                    out_a = Nrc * adot[sma, ecc] / Δa
                    out_e = Nrc * edot[sma, ecc] / Δe

                    total_out = out_a + out_e
                    if abs(total_out) > Nrc and Nrc > 0:
                        factor = Nrc / abs(total_out)
                        out_a *= factor
                        out_e *= factor
                    elif Nrc == 0:
                        out_a = 0
                        out_e = 0

                    dN[sma, ecc] += out_a + out_e
                    if sma > 0: dN[sma - 1, ecc] -= out_a
                    if ecc > 0: dN[sma, ecc - 1] -= out_e

            dN_all_species[:, species, :] = dN
            
        # self.t_0 = t # update global variable 
        dN_all_species = dN_all_species + output

        #############################
        # Post Mission Disposal of Existing Population
        #############################    
        flattened = self.effective_altitude_matrix.flatten()
        val = np.array([eq(*flattened) for eq in self.full_Cdot_PMD])
        val_reshaped = val.reshape((n_alt_shells, n_species))  # shape: (alt_shell, species)

        # Apply sink only to ecc=0
        for species in range(n_species):
            for shell in range(n_alt_shells):
                total_pmd = val_reshaped[shell, species]

                if total_pmd == 0:
                    continue
                
                # remove from the same sma shell and ecc=0
                dN_all_species[shell, species, 0] += total_pmd


        ############################
        # Add the change in population due to launches
        ############################    
        if launch_funcs is not None and self.baseline is False:
            t_launch = t / years  # convert to years for launch functions
            launch_rates = np.array([func(t_launch) for func in launch_funcs])

            # launch rates will be a 1d array of length n_sma_bins * n_species
            launch_rates = launch_rates.reshape((n_sma_bins, n_species)) 

            # assumption that all launches will be in the first eccentricity bin
            for sma in range(n_sma_bins):
                for species in range(n_species):
                    launch = launch_rates[sma, species] #* dt / years # launch rate is in years
                    dN_all_species[sma, species, 0] += launch

        
        print(f"Amount removed due to PMD: {np.sum(val)} Amount added due to launches: {np.sum(launch_rates)}")
        print(t)
        return dN_all_species.flatten()
    
    def run_model(self):
        """
        For each species, integrate the equations of population change for each shell and species.

        The starting point will be, x0, the initial population.

        The launch rate will be first calculated at time t, then the change of population in that species will be calculated using the ODEs. 

        :return: None
        """
        print("Preparing equations for integration (Lambdafying) ...")
        
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

            # This should change location, but first make the fragments spread distribution
            for term in self.collision_terms:
                # === 1. Sum over SMA × ECC for each (shell, mass) bin ===
                totals = term.fragment_spread_totals.sum(axis=(2, 3), keepdims=True)  # shape: (shell, mass, 1, 1)

                # === 2. Normalize safely using np.where (broadcasting-friendly) ===
                with np.errstate(invalid='ignore', divide='ignore'):
                    spread_distribution = np.where(
                        totals > 0,
                        term.fragment_spread_totals / totals,
                        0.0
                    )

                # === 3. Store for later use ===
                term.spread_distribution = spread_distribution

                if np.sum(spread_distribution) == 0 and np.sum(totals) != 0:
                    print(f"Warning: No fragments produced for term {term.name}. Check your collision parameters.")

                # # # === 4. Sanity check: each (shell, mass) should sum to ≈ 1.0 or 0.0 ===
                # per_bin_sums = spread_distribution.sum(axis=(2, 3))
                # print("Sanity check (each value should be ~1.0 or 0.0):")
                # print(per_bin_sums)

            # === 1. Setup ===
            x0_sum = np.sum(self.x0, axis=2)  # shape (n_shells, n_species)
            flat_vars = self.all_symbolic_vars
            self.n_sma_bins, n_species, self.n_ecc_bins = self.x0.shape
            self.n_alt_shells = self.n_shells # remember the shells are in altitude

            # === 2. Lambdify each collision term’s eqs_sources ===
            for term in self.collision_terms:
                term.lambdified_sources = sp.lambdify(flat_vars, term.eqs_sources, modules="numpy")
                term.lambdified_sinks = sp.lambdify(flat_vars, term.eqs_sinks, modules="numpy")

            # Map species index to mass bin index, only for debris
            species_to_mass_bin = {
                i: j for j, (i, name) in enumerate(
                    [(i, name) for i, name in enumerate(self.species_names) if name.startswith("N")]
                )
            }
            
            ## NEW IMPLEMENTATION THAT SEEMS WORKING WITH INTERP
            # Let's assume full_lambda_flattened is your list of launch rate arrays
            launch_rate_functions = []
            start_time = self.scen_times[0]
            time_step_duration = self.scen_times[1] - self.scen_times[0]

            if not self.baseline:
                for rate_array in self.full_lambda_flattened:
                    try: 
                        if rate_array is not None:
                            clean_rate_array = np.array(rate_array)
                            clean_rate_array[np.isnan(clean_rate_array)] = 0 # Replace any NaN values with 0.
                            clean_rate_array[np.isinf(clean_rate_array)] = 0 # Replace any infinity values (positive or negative) with 0.

                            ## USE INTERPOLATION
                            interp_func = interp1d(self.scen_times, clean_rate_array, 
                                                kind='cubic', # 'linear', 'cubic'
                                                bounds_error=False, 
                                                fill_value=0)
                            launch_rate_functions.append(interp_func)

                            # USE STEP FUNCTION
                            # step_func = StepFunction(start_time, time_step_duration, clean_rate_array)
                            # launch_rate_functions.append(step_func)
                            
                        else:
                            # If there are no launches, create a simple lambda that always returns 0
                            launch_rate_functions.append(lambda t: 0.0)
                    except:
                        launch_rate_functions.append(lambda t: 0.0)

            # Finally lambdify the equations for integration, this will just be pmd
            # equations_flattened = [self.equations[i, j] for j in r÷
            self.full_Cdot_PMD = [sp.lambdify(flat_vars, eq, 'numpy') for eq in self.full_Cdot_PMD]

            # now we need to propagate using the dynamical equations
            param = {
                'req': 6378.136, 
                'mu': 398600.0, # should already be defined
                'Bstar': 2.2 * (1e-6 / 100.0), # this will change for each species, km^2
                'j2': 1082.63e-6
            }

            # Constants
            self.t_0 = 0
            hours = 3600.0
            days = 24.0 * hours
            years = 365.25 * days

            def get_dadt(a_current, e_current, p):
                    re   = p['req']
                    mu   = p['mu']
                    n0   = np.sqrt(mu) * a_current ** -1.5
                    a_minus_re = a_current - re
                    rho0 = densityexp(a_minus_re) * 1e9  # kg/km^3
                    C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)
                    
                    beta = (np.sqrt(3)/2)*e_current
                    ang  = np.arctan(beta)
                    sec2 = 1.0 / np.cos(ang) ** 2
                    return -(4 / np.sqrt(3)) * (a_current**2 * n0 * C0 / e_current) * np.tan(ang) * sec2

            def get_dedt(a_current, e_current, p):
                re   = p['req']
                mu   = p['mu']
                n0   = np.sqrt(mu) * a_current ** -1.5
                beta = (np.sqrt(3)/2) * e_current
                a_minus_re = a_current - re
                rho0 = densityexp(a_minus_re) * 1e9
                C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)

                sec2 = 1.0 / np.cos(np.arctan(beta)) ** 2
                return -e_current * n0 * a_current * C0 * sec2

            binE_ecc = self.eccentricity_bins
            binE_ecc = np.sort(binE_ecc)
            self.binE_ecc_mid_point = (binE_ecc[:-1] + binE_ecc[1:]) / 2
            Δa      = self.sma_HMid_km[1] - self.sma_HMid_km[0]
            Δe      = self.eccentricity_bins[1] - self.eccentricity_bins[0]

            # Calculate da/dt and de/dt at each point
            adot = np.zeros((self.n_sma_bins, self.n_ecc_bins))
            edot = np.zeros((self.n_sma_bins, self.n_ecc_bins))

            for sma in range(self.n_sma_bins):
                a_val = self.sma_HMid_km[sma]
                for ecc in range(self.n_ecc_bins):
                    e_val = self.binE_ecc_mid_point[ecc]
                    adot[sma, ecc] = get_dadt(a_val, e_val, param)
                    edot[sma, ecc] = get_dedt(a_val, e_val, param)


            output = solve_ivp(
                fun=self.population_rhs,
                t_span=(self.scen_times[0], self.scen_times[-1]),
                y0=self.x0.flatten(),
                t_eval=self.scen_times,
                args=(launch_rate_functions, self.n_sma_bins, n_species, self.n_ecc_bins, self.n_alt_shells,
                      species_to_mass_bin, years, adot, edot, Δa, Δe),
                method="RK45"
            )
            # output = 1
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
                    try:
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
                    except Exception:
                        print(f"Cannot make indicator for {indicator_var}")
                        print(Exception)

            print("Indicator variables succesfully ran")
            print(self.indicator_results['indicators'].keys())


        return 
    # def population_shell(self, t, N, full_lambda, equations, times, progress_bar=True):
    def population_shell(self, t, N, launch_funcs, eq_funcs, progress_bar=True):
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

        # # Initialize the rate of change array
        # dN_dt = np.zeros_like(N)

        # # Iterate over each component in N
        # for i in range(len(N)):
        
        #     # Compute and add the external modification rate, if applicable
        #     # Now using np.interp to calculate the increase
        #     if full_lambda[i] is not None:
        #         increase = np.interp(t, times, full_lambda[i])
        #         # If increase is nan set to 0
        #         if np.isnan(increase) or np.isinf(increase):
        #             increase = 0
        #         else:
        #             dN_dt[i] += increase

        #     # Compute the intrinsic rate of change from the differential equation
        #     dN_dt[i] += equations[i](*N)

        # NEW IMPLEMENTATION THAT SEEMS WORKING WITH INTERP
        dN_dt = np.zeros_like(N)
        # --- This is now much more efficient ---
        # Calculate the intrinsic rate of change from the differential equations
        # This part can be vectorized if your `equations` list is lambdified correctly
        intrinsic_rates = np.array([eq(*N) for eq in eq_funcs])

        # Calculate the launch rates at the current time 't' by calling the functions
        launch_rates = np.array([func(t) for func in launch_funcs])

        # The total rate of change is the sum
        dN_dt = intrinsic_rates + launch_rates

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
        # full_lambda_flattened = list(self.full_lambda)  
        full_lambda_flattened = []
        # # Iterate through columns first, then rows
        # for c in range(self.full_lambda.cols):      # Iterate over column indices (0, 1, 2)
        #     for r in range(self.full_lambda.rows):  # Iterate over row indices (0 to 23)
        #         full_lambda_flattened.append(self.full_lambda[r, c])

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