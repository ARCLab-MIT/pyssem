import numpy as np
from math import pi
from datetime import datetime
from scipy.integrate import solve_ivp
import sympy as sp
from utils.drag.drag import static_exp_dens_func, JB2008_dens_func
from utils.launch.launch import ADEPT_traffic_model
from pkg_resources import resource_filename


class ScenarioProperties:
    def __init__(self, start_date: datetime, simulation_duration: int, steps: int, min_altitude: float, 
                 max_altitude: float, n_shells: int, launch_function: str, integrator: str = "rk4", 
                 density_model: str = "static_exp_dens_func", LC: float = 0.1, v_imp: float = 10.0):
        """
        Constructor for ScenarioProperties
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
        if not isinstance(start_date, datetime):
            raise TypeError("start_date must be a datetime object")
        if not isinstance(simulation_duration, int):
            raise TypeError("simulation_duration must be an integer")
        if not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if not isinstance(min_altitude, (int, float)):
            raise TypeError("min_altitude must be a number (int or float)")
        if not isinstance(max_altitude, (int, float)):
            raise TypeError("max_altitude must be a number (int or float)")
        if not isinstance(n_shells, int):
            raise TypeError("shells must be an integer")
        if not isinstance(launch_function, str):
            raise TypeError("launch_function must be a string")
        if not isinstance(integrator, str):
            raise TypeError("integrator must be a string")
        if not isinstance(density_model, str):
            raise TypeError("density_model must be a string")
        if not isinstance(LC, (int, float)):
            raise TypeError("LC must be a number (int or float)")
        if not isinstance(v_imp, (int, float)):
            raise TypeError("v_imp must be a number (int or float)")

        self.start_date = start_date
        self.simulation_duration = simulation_duration
        self.steps = steps
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.n_shells = n_shells
        self.launch_function = launch_function
        self.density_model = density_model
        self.integrator = integrator
        self.LC = LC
        self.v_imp = v_imp
        
        # Set the density model to be time dependent or not, JB2008 is time dependent
        self.time_dep_density = False
        if self.density_model == "static_exp_dens_func":
            self.density_model = static_exp_dens_func
            self.time_dep_density = False
        elif self.density_model == "JB2008_dens_func":
            self.density_model = JB2008_dens_func
            self.time_dep_density = True
            if not self.density_filepath:
                self.density_filepath = "./Atmosphere Model/JB2008/Precomputed/dens_highvar_2000.mat"
        else:
            print("Warning: Unable to parse density model, setting to static exponential density model")
            self.density_model = static_exp_dens_func

        # FILL OUT THE INTEGRATOR FIXED STEPS WHEN REQUIRED
            
        # Parameters
        self.scen_times = np.linspace(0, self.simulation_duration, self.steps) 
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

    def get_species(self):
        return self.species
    
    def future_launch_model(self, FLM_steps):
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
        Generate the initial population and the launch rates. 
        """
        #filepath = os.path.join(os.path.dirname(__file__), '../data', 'x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv')
        #filepath = r"D:\ucl\pyssem\src\pyssem\utils\launch\data\x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv"
        # filepath = r"C:\Users\IT\Documents\UCL\pyssem\pyssem\utils\launch\data\x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv"
        #filepath = resource_filename('utils.launch', 'data/x0_launch_repeatlaunch_2018to2022_megaconstellationLaunches_Constellations.csv')


        [x0, FLM_steps] = ADEPT_traffic_model(self, filepath)

        # save as csv
        # x0.to_csv('src/pyssem/utils/launch/data/x0.csv', sep=',', index=False, header=True)
        # FLM_steps.to_csv('src/pyssem/utils/launch/data/FLM_steps.csv', sep=',', index=False, header=True)

        # Store as part of the class, as it is needed for the run_model()
        self.x0 = x0
        self.FLM_steps = FLM_steps

        self.future_launch_model(FLM_steps)
        return
    
    ## Simulation Part

    def build_model(self):

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
            return
        
        # Need to re-shape the equations to be a 1D array
        

        return

    def run_model(self):
        print("Conversion of equations to lambda functions...")
        
        # Initial Population
        x0 = self.x0.T.values.flatten()

        equations_flattened = [self.equations[i, j] for j in range(self.equations.cols) for i in range(self.equations.rows)]

        # Convert the equations to lambda functions
        equations = [sp.lambdify(self.all_symbolic_vars, eq, 'numpy') for eq in equations_flattened]

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

def population_shell(t, N, full_lambda, equations, times):
    # Initialize the rate of change array
    dN_dt = np.zeros_like(N)
    # print(t)
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