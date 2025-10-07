import numpy as np
from math import pi
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, make_interp_spline
from tqdm import tqdm
import sympy as sp
from ..drag.drag import *
from ..launch.launch import ADEPT_traffic_model, SEP_traffic_model
from ..handlers.handlers import download_file_from_google_drive
from ..simulation.build_run_model_helpers import *
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
            self.v_imp_all = self.v_imp * np.ones_like(self.V)  # impact velocity [km/s] Shell-wise
        else: 
            # Calculate v_imp for each orbital shell using the vis viva equation
            self.v_imp_all = np.sqrt(2 * self.mu / (self.HMid * 1000)) / 1000  # impact velocity [km/s] Shell-wise
        self.v_imp_all * 1000 * (24 * 3600 * 365.25)  # impact velocity [m/year]
        self.Dhl = self.deltaH * 1000 # thickness of the shell [m]
        self.Dhu = -self.deltaH * 1000 # thickness of the shell [m]
        self.options = {'reltol': 1.e-4, 'abstol': 1.e-4}  # Integration options # these are likely to change
        self.R0 = R0 # gives you the shells <- gives you the top or bottom of shells -> is this needed in python?
        self.prev_t = -1
        
        # An empty list for the species
        self.species = []
        self.species_cells = {} #dict with S, D, N, Su, B arrays or whatever species types exist}
        self.species_names = []
        self.debris_names = []
        self.debris_length = 0
        self.species_length = 0
        self.all_symbolic_vars = []
        self.pmd_debris_names = []
        
        self.collision_pairs = [] 

        # Varying collision shells 
        self.fragment_spreading = fragment_spreading

        # Elliptical orbits
        self.elliptical = elliptical
        if self.elliptical:
            # make sure fragment_spreading is False
            self.fragment_spreading = False
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
        self.coll_eqs_lambd = None # Used for OPUS when only collision equations are required
        
        # Outputs
        self.output = None
        self.prev_t = -1
        self.launch = np.zeros(self.species_length * self.n_shells)

        # Restults
        self.results = None

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
        Processes FLM_steps to assign raw launch values per shell, eccentricity bin, and year to each species.
        No interpolation is performed. Each [alt_bin][ecc_bin] entry contains a launch array or 0.
        
        Updates `species.lambda_funs` to be:
            - If elliptical: list of list (n_shells × n_ecc_bins) of arrays or 0
            - If circular:   list (n_shells) of arrays or 0

        :param FLM_steps: DataFrame containing ['alt_bin', 'epoch_start_date', 'species', ...] 
                        and optionally 'ecc_bin' if elliptical
        """
        elliptical = self.elliptical
        n_shells = self.n_shells

        if elliptical:
            self.n_sma_bins, n_species, self.n_ecc_bins = self.x0.shape

            for species_group in self.species.values():
                for species in species_group:
                    if species.sym_name not in FLM_steps.columns:
                        if elliptical:
                            # Flat list of n_shells * n_ecc_bins
                            species.lambda_funs = [0 for _ in range(n_shells * self.n_ecc_bins)]
                        else:
                            species.lambda_funs = [0 for _ in range(n_shells)]
                        continue

                    if elliptical:
                        # Ensure the species column exists and is numeric (NaNs -> 0)
                        if species.sym_name not in FLM_steps.columns:
                            # Flat list of n_shells * n_ecc_bins with zeros
                            species.lambda_funs = [0 for _ in range(n_shells * self.n_ecc_bins)]
                            continue

                        # Work on a copy; coerce to numeric and zero-fill NaNs/infs
                        temp_df = FLM_steps.loc[:, ['alt_bin', 'ecc_bin', 'epoch_start_date', species.sym_name]].copy()
                        temp_df[species.sym_name] = pd.to_numeric(temp_df[species.sym_name], errors='coerce')
                        temp_df[species.sym_name] = temp_df[species.sym_name].replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        # Drop rows with NaN bins and cast bins to int
                        temp_df = temp_df.dropna(subset=['alt_bin', 'ecc_bin'])
                        temp_df['alt_bin'] = temp_df['alt_bin'].astype(int)
                        temp_df['ecc_bin'] = temp_df['ecc_bin'].astype(int)

                        # Sortable epoch: parse to datetime if needed
                        if not np.issubdtype(temp_df['epoch_start_date'].dtype, np.datetime64):
                            temp_df['epoch_start_date'] = pd.to_datetime(temp_df['epoch_start_date'], errors='coerce', utc=True)

                        # Flat list of length n_shells * n_ecc_bins
                        lambda_funs = [0 for _ in range(n_shells * self.n_ecc_bins)]

                        grouped = temp_df.groupby(['alt_bin', 'ecc_bin'], sort=True)
                        for (shell, ecc_bin), group in grouped:
                            group = group.sort_values('epoch_start_date')
                            y = group[species.sym_name].to_numpy(dtype=float)
                            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                            flat_index = int(shell) * self.n_ecc_bins + int(ecc_bin)
                            lambda_funs[flat_index] = y if (y.size > 0 and np.any(y != 0.0)) else 0

                        species.lambda_funs = lambda_funs

                        # NaN-safe total
                        total_count = float(np.nansum([np.nansum(entry) if isinstance(entry, np.ndarray) else 0.0
                                                       for entry in lambda_funs]))
                        if species.sym_name == 'B':
                            nan_cells = sum(int(np.isnan(entry).any()) for entry in lambda_funs if isinstance(entry, np.ndarray))
                            print(f"Species: {species.sym_name} Total Count: {total_count} (cells with NaNs: {nan_cells})")
                        else:
                            print(f"Species: {species.sym_name} Total Count: {total_count}")
                    else:
                        temp_df = FLM_steps.loc[:, ['alt_bin', 'epoch_start_date', species.sym_name]]
                        species_FLM = temp_df.pivot(index='alt_bin', columns='epoch_start_date', values=species.sym_name)

                        lambda_funs = []
                        if species.launch_altitude is not None:
                            closest_shell = np.argmin(np.abs(self.HMid - species.launch_altitude))
                        else:
                            closest_shell = None

                        for shell in range(n_shells):
                            if shell in species_FLM.index:
                                y = species_FLM.loc[shell, :].values
                            else:
                                y = np.zeros(len(species_FLM.columns))

                            if closest_shell is not None and shell == closest_shell:
                                y += species.lambda_constant

                            lambda_funs.append(y if not np.all(y == 0) else 0)

                        species.lambda_funs = lambda_funs

        else: # circular orbits
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
                elif indicator == "collisions_per_species_altitude":
                    self.indicator_variables_list.append(make_collisions_per_species_altitude(self, 
                                                                                            percentage = False, 
                                                                                            per_species = True, 
                                                                                            per_pair = False))
                elif indicator == "collisions_per_species_altitude_per_pair":
                    self.indicator_variables_list.append(make_collisions_per_species_altitude_per_pair(self, 
                                                                                                    percentage = False, 
                                                                                                    per_species = False, 
                                                                                                    per_pair = True))
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

            For the multi-species model, it will now store them in a dictionary with the species name as the key.

            Parameters:
                fringe_satellites (str): Fringe Satellite Name
        """

        fringe_satellite_items = [
            item for sublist in self.indicator_variables_list for item in sublist 
            if item.name == fringe_satellites or item.name.split('_')[0] == fringe_satellites
        ]

        # there should only be one item
        if len(fringe_satellite_items) != 1:
            raise ValueError("There should only be one fringe satellite. Multiple found.")
        
        fringe_satellite_items = fringe_satellite_items[0].eqs

        # Lambdify the equations
        simplified_eqs = sp.simplify(fringe_satellite_items)

        # Save as part of a dictionary
        if hasattr(self, 'fringe_active_loss'):
            # Add to the dictionary
            self.fringe_active_loss[fringe_satellites] = sp.lambdify(self.all_symbolic_vars, simplified_eqs, 'numpy')
        else:
            # Create the dictionary
            self.fringe_active_loss = {}
            self.fringe_active_loss[fringe_satellites] = sp.lambdify(self.all_symbolic_vars, simplified_eqs, 'numpy')
                    
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

        # Calculate the UMPY, if state_matix is a matrix, flatten 
        if len(state_matrix.shape) > 1:
            state_matrix = state_matrix.flatten()
            
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
            # Initialize as Sympy matrices to support + operator
            self.full_coll_sink = sp.zeros(self.n_shells, self.species_length)
            self.full_coll_source = sp.zeros(self.n_shells, self.species_length)

            # Determine debris insertion range
            debris_species = [spc for spc in self.species['debris']]
            if len(debris_species) > 0:
                first_deb_name = debris_species[0].sym_name
                deb_start_idx = next((j for j, spc in enumerate([spc for grp in self.species.values() for spc in grp])
                                      if spc.sym_name == first_deb_name), None)
                deb_len = len(debris_species)
            else:
                deb_start_idx, deb_len = None, 0

            for cp in self.collision_pairs:
                # indices of the two active species
                s1_idx = self.species_names.index(cp.species1.sym_name)
                s2_idx = self.species_names.index(cp.species2.sym_name)

                # cp.eqs is an (n_shells x species_length) matrix
                eqs = cp.eqs

                # Build sinks matrix with contributions only in active species columns
                sinks = sp.zeros(self.n_shells, self.species_length)
                sinks[:, s1_idx] = eqs[:, s1_idx]
                sinks[:, s2_idx] = eqs[:, s2_idx]

                # Build sources matrix in debris columns
                sources = sp.zeros(self.n_shells, self.species_length)
                if deb_len > 0 and deb_start_idx is not None:
                    sources[:, deb_start_idx:deb_start_idx + deb_len] = eqs[:, deb_start_idx:deb_start_idx + deb_len]

                # Accumulate
                self.full_coll_sink = self.full_coll_sink + sinks
                self.full_coll_source = self.full_coll_source + sources

                # Store term for RHS use
                term = SymbolicCollisionTerm(
                    s1_idx=s1_idx,
                    s2_idx=s2_idx,
                    eqs_sources=sources,
                    eqs_sinks=sinks,
                    fragment_spread_totals=getattr(cp, 'fragments_sd', None)
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

        if self.baseline:
            self.store_equations = self.equations
            self.equations = self.lambdify_equations()
        else:
            self.equations, self.full_lambda_flattened = self.lambdify_equations(), self.lambdify_launch()
            
        return


    def build_model_elliptical(self):
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
            # Initialize as SymPy Matrix objects for efficient matrix operations
            self.full_coll_sink = sp.zeros(self.n_shells, self.species_length)
            self.full_coll_source = sp.zeros(self.n_shells, self.species_length)

            for i in self.collision_pairs:
                # Accumulate global source/sink expressions using matrix addition
                self.full_coll_sink = self.full_coll_sink + i.eqs_sinks
                self.full_coll_source = self.full_coll_source + i.eqs_sources

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

        ## from old run model 

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

        # === 1. Setup ===
        flat_vars = self.all_symbolic_vars
        self.n_sma_bins, self.species_length, self.n_ecc_bins = self.x0.shape
        self.n_alt_shells = self.n_shells # remember the shells are in altitude

        # === 2. Lambdify each collision term’s eqs_sources ===
        for term in self.collision_terms:
            term.lambdified_sources = sp.lambdify(flat_vars, term.eqs_sources, modules="numpy")
            term.lambdified_sinks = sp.lambdify(flat_vars, term.eqs_sinks, modules="numpy")

        # === 3. lambdify the pmd equations
        # Finally lambdify the equations for integration, this will just be pmd
        self.full_Cdot_PMD = [sp.lambdify(flat_vars, eq, 'numpy') for eq in self.full_Cdot_PMD]

        #  Map species index to mass bin index, only for debris
        self.species_to_mass_bin = {
            i: j for j, (i, name) in enumerate(
                [(i, name) for i, name in enumerate(self.species_names) if name.startswith("N")]
            )
        }

        # create a boolean list that is the same length as species, depending on whether they are active or not
        self.active_species_bool = []
        for species_group in self.species.values():
            for species in species_group:
                self.active_species_bool.append(species.drag_effected)

        # get a list of all species for pmd 
        self.all_species_list = [species for category in self.species.values() for species in category]

        binE_ecc = self.eccentricity_bins
        binE_ecc = np.sort(binE_ecc)
        self.binE_ecc_mid_point = (binE_ecc[:-1] + binE_ecc[1:]) / 2
        self.Δa      = self.sma_HMid_km[1] - self.sma_HMid_km[0]
        self.Δe      = self.eccentricity_bins[1] - self.eccentricity_bins[0]

        self.adot_all_species = []
        self.edot_all_species = []

        bstar_vals = []
        for species_group in self.species.values():
            for species in species_group:
                bstar_vals.append(species.bstar)
        
        for bstar in bstar_vals:
            # now we need to propagate using the dynamical equations
            param = {
                'req': 6378.136, 
                'mu': 398600.0, # should already be defined
                'Bstar': bstar, # 2.2000e-08, # 2.2 * ((2.687936011/1e3)**2/ 1783),  # bstar = cd * ((radius/1e3)**2/ mass) 0.5, 148
                'j2': 1082.63e-6
            }

            # Calculate da/dt and de/dt at each point
            adot = np.zeros((self.n_sma_bins, self.n_ecc_bins))
            edot = np.zeros((self.n_sma_bins, self.n_ecc_bins))

            for sma in range(self.n_sma_bins):
                a_val = self.sma_HMid_km[sma]
                for ecc in range(self.n_ecc_bins):
                    e_val = self.binE_ecc_mid_point[ecc]
                    adot[sma, ecc] = get_dadt(a_val, e_val, param) * years 
                    edot[sma, ecc] = get_dedt(a_val, e_val, param) * years

            self.adot_all_species.append(adot)
            self.edot_all_species.append(edot)

        self.full_lambda_flattened = self.lambdify_launch_elliptical()
            
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

        # Get the equations in the correct format for lambdification
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
    
    def sma_ecc_mat_to_altitude_mat(self, population_matrix_sma_ecc):
        """
        Convert (sma, species, ecc) -> (alt, species), zeroing cells whose perigee altitude < 150 km.
        population_matrix_sma_ecc shape: (n_sma_bins, n_species, n_ecc_bins)
        time_in_shell shape:              (n_alt_shells, n_ecc_bins, n_sma_bins)
        """
        import numpy as np

        n_sma, n_species, n_ecc = population_matrix_sma_ecc.shape
        assert n_sma == self.n_sma_bins and n_species == self.species_length and n_ecc == self.n_ecc_bins

        # --- Midpoints (fallback to scenario_properties if attributes live there) ---
        try:
            ecc_mid = np.asarray(self.binE_ecc_mid_point, dtype=float)   # (n_ecc,)
            sma_mid = np.asarray(self.sma_HMid_km, dtype=float)          # (n_sma,)
        except AttributeError:
            ecc_mid = np.asarray(self.scenario_properties.binE_ecc_mid_point, dtype=float)
            sma_mid = np.asarray(self.scenario_properties.sma_HMid_km, dtype=float)

        # --- Perigee filter: keep only a(1-e) > R_earth + 150 km ---
        R_earth_km = getattr(
            self, "R_earth_km",
            getattr(getattr(self, "scenario_properties", self), "R_earth_km", 6378.137)
        )
        perigee_altitude_threshold_km = 150.0

        # Grid of (a,e) to compute r_p = a(1-e)
        A_km, E = np.meshgrid(sma_mid, ecc_mid, indexing="ij")      # both (n_sma, n_ecc)
        rp_km = A_km * (1.0 - E)
        keep_mask = rp_km > (R_earth_km + perigee_altitude_threshold_km)  # True => keep

        # Zero out decaying cells across all species
        pop_filtered = population_matrix_sma_ecc * keep_mask[:, None, :]  # (n_sma, n_species, n_ecc)

        # --- Map to altitude via time-in-shell weights ---
        # time_in_shell: (alt, ecc, sma); pop_filtered: (sma, species, ecc)
        # Result: (alt, species)
        effective_altitude = np.einsum("aes, spe -> ap", self.time_in_shell, pop_filtered, optimize=True)

        self.effective_altitude_matrix = effective_altitude
        return effective_altitude
    
    def population_rhs(self, t, x_flat, launch_funcs, n_sma_bins, n_species, n_ecc_bins, n_alt_shells,
                      species_to_mass_bin, years, adot_all_species, edot_all_species, Δa, Δe,
                      drag_affected_bool, all_species_list, progress_bar=True, opus=False):

        # dt = years * (t - self.t_0)
        # self.t_0 = t
        if self.progress_bar is not None and progress_bar:
            self.progress_bar.update(t - self.progress_bar.n)
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
        # IMPORTANT: Use self.n_shells (altitude shells) for collision calculations, not n_alt_shells
        n_collision_shells = self.n_shells  # This is the number of altitude shells for collisions
        self.effective_altitude_matrix = np.zeros((n_collision_shells, n_species))
        normalised_species_distribution_in_sma_e_space = np.zeros((n_collision_shells, n_species, n_sma_bins, n_ecc_bins))
        # for each species, in each shell, trying to find the ae that contribute to those bins. 
        try:
            for species in range(n_species):
                for alt_shell in range(n_collision_shells):
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
        
        total_dNdt_alt = np.zeros((n_collision_shells, n_species))
        total_dNdt_sma_ecc_sources = np.zeros((n_sma_bins, n_species, n_ecc_bins))


        # # #############################
        # # # Our population (x_matrix) is now in the form of altitude and species, which is now for the collision equations.
        # # # #############################    
        x_flat_ordered = self.effective_altitude_matrix.flatten()
        # collision pair in altitude space 
        for term in self.collision_terms:
            dNdt_term = term.lambdified_sources(*x_flat_ordered)
            total_dNdt_alt = np.array(dNdt_term, dtype=float) # n_collision_shells x n_species

            # multiply the growth rate for each species by the distribution of that species in a,e space
            for shell in range(n_collision_shells):
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
                for shell in range(n_collision_shells):
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
            # Only apply to species that are not drag affected
            if drag_affected_bool[species] is False:
                continue 

            adot = adot_all_species[species]
            edot = edot_all_species[species]

            N_sma_ecc = x_matrix[:, species, :]
            dN = np.zeros_like(N_sma_ecc)

            for sma in range(n_sma_bins - 1, -1, -1):
                for ecc in range(n_ecc_bins - 1, -1, -1):
                    Nrc = N_sma_ecc[sma, ecc]
                    out_a = Nrc * adot[sma, ecc] / Δa #* dt
                    out_e = Nrc * edot[sma, ecc] / Δe #* dt

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

        # #############################
        # # Post Mission Disposal of Existing Population, this should be from the circular population of x_flat
        # #############################

        # First loop through the species and do their sinks
        if not opus:
            pmd = np.zeros_like(dN_all_species)
            for species in range(n_species):
                # remove the total number of satellites based on deltat
                if all_species_list[species].active == True:
                    for sma in range(n_sma_bins):
                        for ecc in range(n_ecc_bins):
                            pmd[sma, species, ecc] -= (1/ all_species_list[species].deltat) * x_matrix[sma, species, ecc]

                # Then gain the number of derelicts based on deltat
                if not all_species_list[species].active:
                    if all_species_list[species].pmd_linked_species:
                        linked_sym = all_species_list[species].pmd_linked_species[0].sym_name
                        linked_idx = next(i for i, sp in enumerate(all_species_list)
                                        if sp.sym_name == linked_sym)

                        Pm      = all_species_list[linked_idx].Pm
                        dt_link = all_species_list[linked_idx].deltat
                        fail_rate = (1.0 - Pm) / dt_link

                        for sma in range(n_sma_bins):
                            for ecc in range(n_ecc_bins):
                                pop_linked = x_matrix[sma, linked_idx, ecc]
                                pmd[sma, species, ecc] += fail_rate * pop_linked

            dN_all_species += pmd
             
        ############################
        # Add the change in population due to launches
        ############################    
        if launch_funcs is not None and self.baseline is False:
            for sma in range(n_sma_bins):
                for species in range(n_species):
                    for ecc in range(n_ecc_bins):
                        func = launch_funcs[sma, species, ecc]
                        if func is not None:
                            try:
                                launch = func(t)
                                dN_all_species[sma, species, ecc] += launch
                            except Exception as e:
                                raise RuntimeError(
                                    f"Failed evaluating launch_func at [sma={sma}, species={species}, ecc={ecc}]: {e}"
                                )

        
        # print(f"Amount removed due to PMD: {np.sum(val)} Amount added due to launches: {np.sum(launch_rates)}")
        # print(t)
        return dN_all_species.flatten()

    def run_model_elliptical(self):
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
            launch_rate_functions = np.full(
                (self.n_sma_bins,  self.species_length, self.n_ecc_bins), 
                None, 
                dtype=object
            )

            if not self.baseline:
                for sma in range(self.n_sma_bins):
                    for species in range( self.species_length):
                        for ecc in range(self.n_ecc_bins):
                            rate_array = self.full_lambda_flattened[sma, species, ecc]

                            try:
                                launch_rate_functions[sma, species, ecc] = None  # default

                                if rate_array is None:
                                    continue

                                # Case 1: ndarray directly
                                if isinstance(rate_array, np.ndarray):
                                    flattened_array = rate_array.astype(float)

                                # Case 2: mixed list of array + zeros
                                elif isinstance(rate_array, list):
                                    array_found = next(
                                        (np.asarray(r).astype(float) for r in rate_array if isinstance(r, np.ndarray)),
                                        None
                                    )
                                    if array_found is None:
                                        continue
                                    flattened_array = array_found

                                # Case 3: scalar/unexpected → skip
                                else:
                                    continue

                                # Clean
                                flattened_array[np.isnan(flattened_array)] = 0.0
                                flattened_array[np.isinf(flattened_array)] = 0.0

                                # Validate
                                if flattened_array.shape[0] != len(self.scen_times):
                                    continue
                                if np.all(flattened_array == 0.0):
                                    continue

                                # === Interpolate with make_interp_spline ===
                                spline_func, _, _ = self._zero_padded_spline(self.scen_times, flattened_array, bc_type="natural")
                                launch_rate_functions[sma, species, ecc] = spline_func

                            except Exception as e:
                                raise ValueError(
                                    f"Failed processing rate_array at [sma={sma}, species={species}, ecc={ecc}]:\n"
                                     f"{rate_array}\n\n{e}"
                                )
            self.t_0 = 0
            
            self.progress_bar = tqdm(total=self.scen_times[-1] - self.scen_times[0], desc="Integrating Equations", unit="year")

            output = solve_ivp(
                fun=self.population_rhs,
                t_span=(self.scen_times[0], self.scen_times[-1]),
                y0=self.x0.flatten(),
                t_eval=self.scen_times,
                args=(launch_rate_functions, self.n_sma_bins, self.species_length, self.n_ecc_bins, self.n_alt_shells,
                      self.species_to_mass_bin, years, self.adot_all_species, self.edot_all_species, self.Δa, self.Δe, self.active_species_bool, self.all_species_list),
                method = self.integrator # or any other method you prefer
            )

            # output = 1
            self.progress_bar.close()
            self.progress_bar = None # Set back to None becuase a tqdm object cannot be pickled

        if output.success:
            print(f"Model run completed successfully.")
        else:
            print(f"Model run failed: {output.message}")

        self.output = output # Save

        # --- outputs ---
        n_species = self.species_length
        n_time    = self.output.y.shape[1]

        # --- Unpack and reshape population data: (sma, species, ecc, time) ---
        x_matrix = self.output.y.reshape(self.n_sma_bins, n_species, self.n_ecc_bins, n_time)

        # --- Project to altitude shells over time ---
        n_eff = np.zeros((self.n_shells, n_species, n_time))  # (alt/shell, species, time)
        for s in range(n_species):
            for t_idx, _ in enumerate(self.output.t):
                snap = x_matrix[:, s, :, t_idx]  # (sma, ecc)
                cube = np.zeros((self.n_sma_bins, n_species, self.n_ecc_bins))
                cube[:, s, :] = snap
                alt_proj = self.sma_ecc_mat_to_altitude_mat(cube)  # (alt, species)
                n_eff[:, s, t_idx] = alt_proj[:, s]

        self.output.y_alt = n_eff  # shape: (n_alt_shells, n_species, n_time)

        ###############
        # Indicator variables rely on altitude shells to calculate the number of collisions,
        # Therefore, project (sma,ecc) -> altitude shells (y_alt), then evaluate indicators.
        ###############
        if hasattr(self, 'indicator_variables_list'):
            print("Evaluating post-processed indicator variables...")

            # --- Prepare indicator results dict ONCE (not inside loops) ---
            if not hasattr(self, 'indicator_results') or self.indicator_results is None:
                self.indicator_results = {}
            self.indicator_results['indicators'] = {}

            # --- Sanity checks ---
            y_alt = self.output.y_alt
            n_shells, n_species_chk, n_time_chk = y_alt.shape
            if n_shells != self.n_shells or n_species_chk != self.species_length:
                raise ValueError("Shape mismatch: y_alt dims do not match self.n_shells/self.species_length.")
            if n_time_chk != len(self.output.t):
                raise ValueError("Time axis mismatch: len(self.output.t) != y_alt.shape[2].")

            # --- Build symbol→(species_idx, shell_idx) order mapping ---
            order_map = self._build_symbol_order_map()

            # --- Evaluate each indicator with states ordered as self.all_symbolic_vars ---
            for group in self.indicator_variables_list:
                for indicator_var in group:
                    try:
                        eq  = sp.simplify(indicator_var.eqs)
                        fun = sp.lambdify(self.all_symbolic_vars, eq, 'numpy')

                        evaluated_indicator_dict = {}

                        for t_idx, t in enumerate(self.output.t):
                            # Build state vector matching self.all_symbolic_vars
                            state = np.empty(len(order_map), dtype=float)
                            for j, (sp_idx, sh_idx) in enumerate(order_map):
                                state[j] = y_alt[sh_idx, sp_idx, t_idx]

                            evaluated_indicator_dict[t] = fun(*state)

                        self.indicator_results['indicators'][indicator_var.name] = evaluated_indicator_dict

                    except Exception as e:
                        print(f"Cannot make indicator for {getattr(indicator_var, 'name', '<unnamed>')}")
                        print(e)

            # --- Make output.y match the non-elliptical plotting layout ---
            # optional: preserve original solver array
            self.output.y_sma_ecc = self.output.y

            # species-major stacking of shells → (n_species*n_shells, n_time)
            self.output.y_alt = np.transpose(n_eff, (1, 0, 2)).reshape(self.species_length * self.n_shells,
                                                                n_time)
            
            # print("Indicator variables succesfully ran")
            # print(self.indicator_results['indicators'].keys())

        return
    
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
                        # interp_func = interp1d(self.scen_times, clean_rate_array, 
                        #                     kind='cubic', # 'linear', 'cubic'
                        #                     bounds_error=False, 
                        #                     fill_value=0)
                        # launch_rate_functions.append(interp_func)

                        # USE STEP FUNCTION
                        step_func = StepFunction(start_time, time_step_duration, clean_rate_array)
                        launch_rate_functions.append(step_func)
                        
                    else:
                        # If there are no launches, create a simple lambda that always returns 0
                        launch_rate_functions.append(lambda t: 0.0)
                except:
                    launch_rate_functions.append(lambda t: 0.0)


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

            # print("Integrating equations...")
            output = solve_ivp(self.population_shell_time_varying_density, [self.scen_times[0], self.scen_times[-1]], x0,
                            args=(launch_rate_functions, self.equations, self.scen_times),
                            t_eval=self.scen_times, method=self.integrator)
            
            self.drag_upper_lamd = None
            self.drag_cur_lamd = None

        else:
            self.progress_bar = tqdm(total=self.scen_times[-1] - self.scen_times[0], desc="Integrating Equations", unit="year")
            
            output = solve_ivp(self.population_shell, [self.scen_times[0], self.scen_times[-1]], x0,
                                        args=(launch_rate_functions, self.equations),
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

            # print("Indicator variables succesfully ran")
            # print(self.indicator_results['indicators'].keys())


        return 
        
    def population_shell_for_OPUS(self, t, N, equations, times, launch, time_idx):
        dN_dt = np.zeros_like(N)

        if self.time_dep_density:
            # No need to cache rho, as propagation is one timestep 
            current_t_step = int(t) + time_idx
            if current_t_step > self.prev_t:
                rho = JB2008_dens_func(time_idx, self.R0_km, self.density_data, self.date_mapping, self.nearest_altitude_mapping)
                self.prev_rho = rho
                self.prev_t = current_t_step
            else:
                rho = self.prev_rho  # Use cached rho

            rho = JB2008_dens_func(t, self.R0_km, self.density_data, self.date_mapping, self.nearest_altitude_mapping)

            species_per_shell = self.species_length

        # Iterate over each component in N
        for i in range(len(N)):
            if self.time_dep_density:
                shell_index = i // species_per_shell

                shell_rho = rho[shell_index]  # use directly

                # Apply drag
                current_drag = self.drag_cur_lamd[i](*N) * shell_rho
                dN_dt[i] += current_drag

                if shell_index < (self.n_shells - 1):
                    upper_rho = rho[shell_index + 1]
                    upper_drag = self.drag_upper_lamd[i](*N) * upper_rho
                    dN_dt[i] += upper_drag

            # Incoming new species
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
        # Update the progress bar
        if self.progress_bar is not None:
            self.progress_bar.update(t - self.progress_bar.n)

        # Clean the derivative array
        dN_dt = np.zeros_like(N)

        # fetch time-varying density with cache management logic for rho
        current_t_step = int(t)
        if current_t_step > self.prev_t:
            rho = JB2008_dens_func(t, self.R0_km, self.density_data, self.date_mapping, self.nearest_altitude_mapping)
            self.prev_rho = rho
            self.prev_t = current_t_step
        else:
            rho = self.prev_rho  # Use cached rho

        # Apply drag computations
        for i in range(len(N)):
            # get appropriate shell index, as the flattened functions iterate over every shell
            # within a species first (rather than each species in a shell)
            shell_index = i % self.n_shells

            # Ensure drag_cur_lamd and drag_upper_lamd functions are correctly accessed and used
            if i < len(N) - 1:
                current_drag = self.drag_cur_lamd[i](*N) * rho[shell_index]
                upper_drag = self.drag_upper_lamd[i](*N) * rho[shell_index + 1]
                dN_dt[i] += current_drag + upper_drag
            else:
                current_drag = self.drag_cur_lamd[i](*N) * rho[shell_index]
                dN_dt[i] += current_drag

            # Apply general equation dynamics
            dN_dt[i] += equations[i](*N)

        if not self.baseline:
            # Calculate the launch rates at the current time 't' by calling the functions
            launch_rates = np.array([func(t) for func in launch_funcs])

            # The total rate of change is the sum
            dN_dt += launch_rates
            
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
    
    def propagate(self, population, times, launch=None, elliptical=False):
        """
            This will use the equations that have been built already by the model, and then integrate the differential equations
            over a chosen timestep. The population and launch (if provided) must be the same length as the species and shells.

            :param population: Initial population
            :param times: Times to integrate over
            :param launch: Launch rates

            :return: results_matrix
        """
        if not elliptical:
            # check to see if the equations have already been lamdified
            if self.equations is None:
                self.equations = self.lambdify_equations()

            output = solve_ivp(self.population_shell_for_OPUS, [times[0], times[-1]], population,
                                args=(self.equations, times, launch), 
                                t_eval=times, method=self.integrator)
            
            if output.success:
                # Extract the results at the specified time points
                results_matrix = output.y.T  # Transpose to make it [time, variables]
                return results_matrix, None 
            else:
                print(f"Model run failed: {output.message}")
                return None

        if elliptical:
            # OPUS provides a one-year launch file. Wrap numeric per-cell rates
            # into constant callables expected by population_rhs.
            if launch is not None:
                launch_rate_functions = np.full((self.n_sma_bins, self.species_length, self.n_ecc_bins), None, dtype=object)
                for sma in range(self.n_sma_bins):
                    for species in range(self.species_length):
                        for ecc in range(self.n_ecc_bins):
                            try:
                                rate = float(launch[sma, species, ecc])
                            except Exception:
                                rate = 0.0
                            if np.isfinite(rate) and rate != 0.0:
                                # constant rate over this one-year propagate window
                                launch_rate_functions[sma, species, ecc] = (lambda r=rate: (lambda t: r))()
                            else:
                                launch_rate_functions[sma, species, ecc] = None
            else:
                # Fallback to any precomputed functions
                launch_rate_functions = getattr(self, 'launch_rate_functions', None)

            output = solve_ivp(
                fun=self.population_rhs,
                t_span=(times[0], times[-1]),
                y0=population.flatten(),
                t_eval=times,
                args=(launch_rate_functions, self.n_sma_bins, self.species_length, self.n_ecc_bins, self.n_alt_shells,
                      self.species_to_mass_bin, years, self.adot_all_species, self.edot_all_species, self.Δa, self.Δe, 
                      self.active_species_bool, self.all_species_list, False, True),
                method = self.integrator # or any other method you prefer
            )

            # convert back to the original shape, only the final state is needed
            results_matrix = output.y[:, -1].reshape(self.n_sma_bins, self.species_length, self.n_ecc_bins)
            results_matrix_alt = self.sma_ecc_mat_to_altitude_mat(results_matrix)
            
            return results_matrix, results_matrix_alt


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

    # def lambdify_launch(self, full_lambda=None):
    #     """ 
    #         Convert the Numpy launch rates to Scipy lambdified functions for integration.
        
    #     """
    #     # Launch rates
    #     # full_lambda_flattened = list(self.full_lambda)  
    #     full_lambda_flattened = []
    #     # # Iterate through columns first, then rows
    #     # for c in range(self.full_lambda.cols):      # Iterate over column indices (0, 1, 2)
    #     #     for r in range(self.full_lambda.rows):  # Iterate over row indices (0 to 23)
    #     #         full_lambda_flattened.append(self.full_lambda[r, c])

    #     if full_lambda is None:
    #         for i in range(len(self.full_lambda)):
    #             if self.full_lambda[i] is not None:
    #                 full_lambda_flattened.extend(self.full_lambda[i])
    #             else:
    #                 # Append None to the list, length of scenario_properties.n_shells
    #                 full_lambda_flattened.extend([None]*self.n_shells)
    #     else:
    #         for i in range(len(full_lambda)):
    #             if full_lambda[i] is not None:
    #                 full_lambda_flattened.extend(full_lambda[i])
    #             else:
    #                 # Append None to the list, length of scenario_properties.n_shells
    #                 full_lambda_flattened.extend([None]*self.n_shells)

    #     return full_lambda_flattened

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
    

    def lambdify_launch_elliptical(self, full_lambda=None):
        full_lambda_flattened = []

        if self.baseline:
            return full_lambda_flattened

        if full_lambda is None:
            full_lambda = self.full_lambda

        for species_lambda in full_lambda:
            if species_lambda is None:
                if self.elliptical:
                    full_lambda_flattened.extend([None] * (self.n_shells * self.n_ecc_bins))
                else:
                    full_lambda_flattened.extend([None] * self.n_shells)
                continue
            full_lambda_flattened.extend(species_lambda)

        # Print total launches per species
        species_names = self.species_names
        idx = 0
        for i, species_lambda in enumerate(full_lambda):
            if species_lambda is None:
                idx += self.n_shells * (self.n_ecc_bins if self.elliptical else 1)
                continue

            total_launches = sum(
                np.sum(entry) if isinstance(entry, np.ndarray) else 0
                for entry in species_lambda
            )
            print(f"Species: {species_names[i]} — Total Launches: {int(total_launches)}")

        # ============================
        # Reshape for use with [sma, species, ecc]
        # ============================
        if self.elliptical:
            reshaped = np.array(full_lambda_flattened, dtype=object).reshape(
                (len(species_names), self.n_shells, self.n_ecc_bins)
            )
            full_lambda_reshaped = np.transpose(reshaped, (1, 0, 2))  # → (sma, species, ecc)
        else:
            reshaped = np.array(full_lambda_flattened, dtype=object).reshape(
                (len(species_names), self.n_shells)
            )
            full_lambda_reshaped = np.transpose(reshaped, (1, 0))  # → (sma, species)

        self.full_lambda_flattened = full_lambda_reshaped
        return self.full_lambda_flattened

    def _build_symbol_order_map(self):
                """
                Returns a list of (species_idx, shell_idx) in the exact order of self.all_symbolic_vars.
                shell_idx is 0-based; symbol names are assumed to end with '_<shell>' (1-based in the name).
                """
                species_to_idx = {name: i for i, name in enumerate(self.species_names)}
                order_map = []

                for sym in self.all_symbolic_vars:
                    name = str(sym)  # e.g., 'N_0.00141372kg_7' or 'S_3'
                    base, sep, shell_str = name.rpartition('_')
                    if sep == '' or not shell_str.isdigit():
                        raise ValueError(f"Symbol '{name}' must end with '_<shell_index>' (1-based).")

                    shell_idx = int(shell_str) - 1  # convert to 0-based
                    if base not in species_to_idx:
                        # strict match against species_names to avoid accidental mis-ordering
                        raise KeyError(f"Symbol base '{base}' not found in species_names {self.species_names}.")

                    species_idx = species_to_idx[base]
                    if not (0 <= shell_idx < self.n_shells):
                        raise IndexError(f"Shell index {shell_idx} out of range for symbol '{name}' with n_shells={self.n_shells}.")

                    order_map.append((species_idx, shell_idx))

                # Sanity: number of symbols must match n_shells * n_species
                expected = self.n_shells * self.species_length
                if len(order_map) != expected:
                    raise ValueError(f"Symbol count {len(order_map)} != n_shells*n_species ({expected}).")
                return order_map
    def _zero_padded_spline(self, x, y, bc_type="natural"):
                """
                Build a spline f(t) that returns 0 outside [x[0], x[-1]].
                Handles short series by reducing k automatically.
                Dedups x by averaging y at duplicate times.
                """
                x = np.asarray(x, float)
                y = np.asarray(y, float)

                # sort & dedup x, average y on duplicates
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                xu, inv = np.unique(x, return_inverse=True)
                if xu.size != x.size:
                    y_agg = np.zeros_like(xu, dtype=float)
                    counts = np.zeros_like(xu, dtype=float)
                    np.add.at(y_agg, inv, y)
                    np.add.at(counts, inv, 1.0)
                    y = y_agg / counts
                    x = xu

                # choose spline degree
                k = int(min(3, max(1, len(x) - 1)))
                if len(x) == 1:
                    # constant inside the single support point
                    v = float(y[0])
                    t0 = t1 = float(x[0])
                    def f(tt):
                        tt = np.asarray(tt, float)
                        out = np.zeros_like(tt, float)
                        mask = (tt == t0)  # only defined at that point
                        out[mask] = v
                        return out
                    return f, t0, t1

                spl = make_interp_spline(x, y, k=k, bc_type=bc_type)
                t0, t1 = float(x[0]), float(x[-1])

                def f(tt):
                    tt = np.asarray(tt, float)
                    out = np.zeros_like(tt, float)
                    mask = (tt >= t0) & (tt <= t1)
                    if np.any(mask):
                        out[mask] = spl(tt[mask])
                    return out

                return f, t0, t1