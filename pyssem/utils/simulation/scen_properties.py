import numpy as np
from math import pi
from datetime import datetime
from scipy.integrate import solve_ivp
import sympy as sp
from ..drag.drag import static_exp_dens_func, JB2008_dens_func
from ..launch.launch import ADEPT_traffic_model
from pkg_resources import resource_filename
import pandas as pd
import os
import pickle

class ScenarioProperties:
    def __init__(self, start_date: datetime, simulation_duration: int, steps: int, min_altitude: float, 
                 max_altitude: float, n_shells: int, launch_function: str, launchfile: str, 
                 integrator: str, density_model: str, LC: float = 0.1, v_imp: float = 10.0,
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
        self.launch_file = launchfile
        
        # Set the density model to be time dependent or not, JB2008 is time dependent
        self.time_dep_density = False
        if self.density_model == "static_exp_dens_func":
            self.density_model = static_exp_dens_func
        elif self.density_model == "JB2008_dens_func":
            self.density_model = JB2008_dens_func
            self.time_dep_density = True
            # if not self.density_filepath:
            #     self.density_filepath = "./Atmosphere Model/JB2008/Precomputed/dens_highvar_2000.mat"
        else:
            print("Warning: Unable to parse density model, setting to static exponential density model")
            self.density_model = static_exp_dens_func

        # FILL OUT THE INTEGRATOR FIXED STEPS WHEN REQUIRED
            
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
        self.v_imp2 = self.v_imp * np.ones_like(self.V)  # impact velocity [km/s] Shell-wise
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
        
        # Outputs
        self.output = None

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

                for shell in range(self.n_shells):
                    y = species_FLM.loc[shell, :].values / time_step  
        
                    if np.all(y == 0):
                        species.lambda_funs.append(None)  
                    else:
                        species.lambda_funs.append(np.array(y))
                         
    def initial_pop_and_launch(self):
        """
        Generate the initial population and the launch rates. The Launch File path should be included in the scenario properties instantiation.
        
        Returns: None
        """

        # Load the launch file
        if self.launch_file is not None:
            print('Using provided launch file:')
            filepath = self.launch_file
        else:
            print('Using default launch file:')            
            resource_path = 'x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv'
            filespath = os.path.join(os.path.dirname(__file__), resource_path)
            if not os.path.exists(filespath):
                raise ValueError("Default launch file not found. Please provide a launch file")
                   
        [x0, FLM_steps] = ADEPT_traffic_model(self, filepath)


        # Store as part of the class, as it is needed for the run_model()
        self.x0 = x0
        self.FLM_steps = FLM_steps

        #self.future_launch_model(FLM_steps)
    
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
        if not self.time_dep_density: # static density
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
        
        if self.time_dep_density:
            # Don't apply rho as this will occur at integration
            # Dont add to full equations either
            self.full_drag = self.drag_term_upper + self.drag_term_cur
            
        return

    def run_model(self):
        """
        For each species, integrate the equations of population change for each shell and species.

        The starting point will be, x0, the initial population.

        The launch rate will be first calculated at time t, then the change of population in that species will be calculated using the ODEs. 

        :return: None
        """
        print("Conversion of equations to lambda functions...")
        
        # Initial Population
        x0 = self.x0.T.values.flatten()

        equations_flattened = [self.equations[i, j] for j in range(self.equations.cols) for i in range(self.equations.rows)]

        # Convert the equations to lambda functions
        equations = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in equations_flattened]
        #equations = [self.equations[i, j] for j in range(self.equations.cols) for i in range(self.equations.rows)]
        # Launch rates
        full_lambda_flattened = []

        for i in range(len(self.full_lambda)):
            if self.full_lambda[i] is not None:
                full_lambda_flattened.extend(self.full_lambda[i])
            else:
                # Append None to the list, length of scenario_properties.n_shells
                full_lambda_flattened.extend([None]*self.n_shells)

        print("Integrating equations...")

        output = solve_ivp(population_shell, [self.scen_times[0], self.scen_times[-1]], x0,
                        args=(full_lambda_flattened, equations, self.scen_times),
                        t_eval=self.scen_times, method='BDF')
            
        if output.success:
            print(f"Model run completed successfully.")
        else:
            print(f"Model run failed: {output.message}")

        # Process results
        self.output = output

        return 


def population_shell_time_varying_density(t, N, full_lambda, equations, times, density_model, R0_km, scen_times_dates, start_date, end_date, steps):
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

    # Initialize the rate of change array
    dN_dt = np.zeros_like(N)       

    rho = density_model(t, R0_km, scen_times_dates, start_date, end_date, steps) 

    
    return dN_dt

    # # Iterate over each component in N
    # for i in range(len(N)):
    
    #     # # Compute and add the external modification rate, if applicable
    #     # # Now using np.interp to calculate the increase
    #     # if full_lambda[i] is not None:
    #     #     increase = np.interp(t, times, full_lambda[i])
    #     #     # If increase is nan set to 0
    #     #     if np.isnan(increase) or np.isinf(increase):
    #     #         increase = 0
    #     #     else:
    #     #         dN_dt[i] += increase

    #     # Compute the intrinsic rate of change from the differential equation
    #     dN_dt[i] += equations[i](*N)

    

def population_shell(t, N, full_lambda, equations, times):
    """
    Seperate function to ScenarioProperties, this will be used in the solve_ivp function.

    :param t: Timestep (int)
    :param N: Population Count (Flattened array of species and shells)
    :param full_lambda: Launch rates (Flattened np.array of species and shells)
    :param equations: Equations (Lambdified sympy functions for each species and shell)
    :param times: Times (Times for the simulation, usually years)

    :return: Rate of change of population at the given timestep, t. 
    """

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


