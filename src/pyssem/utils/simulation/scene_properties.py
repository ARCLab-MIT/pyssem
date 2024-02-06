import numpy as np
from math import pi
from datetime import datetime
import random
from sympy import symbols, Matrix

class SceneProperties:
    def __init__(self, start_date: datetime, simulation_duration: int, steps: int, min_altitude: float, 
                 max_altitude: float, shells: int, delta: float = 10.0, integrator: str = "rk4", 
                 density_model: str = "static_exp_dens_func", LC: float = 0.1, v_imp: float = 10.0):
        """
        Constructor for SceneProperties
        Args:
            start_date (datetime): Start date of the simulation
            simulation_duration (int): Years of the simulation to run 
            steps (int): Number of steps to run in a simulation 
            min_altitude (float): Minimum Altitude shell in km
            max_altitude (float): Maximum Altitude shell in km
            shells (int): Number of Altitude Shells 
            delta (float): Ratio of the density of disabling due to lethal debris (collisions)
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
        if not isinstance(shells, int):
            raise TypeError("shells must be an integer")
        if not isinstance(delta, (int, float)):
            raise TypeError("delta must be a number (int or float)")
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
        self.shells = shells
        self.delta = delta
        self.integrator = integrator
        self.density_model = density_model
        self.LC = LC
        self.v_imp = v_imp
        
        # Set the density model to be time dependent or not, JB2008 is time dependent
        self.time_dep_density = False
        if self.density_model == 'static_exp_dens_func':
            self.time_dep_density = False
        elif self.density_model == 'JB2008_dens_func':
            self.time_dep_density = True
            if not self.density_filepath:
                self.density_filepath = "./Atmosphere Model/JB2008/Precomputed/dens_highvar_2000.mat"
        else:
            print("Warning: Unable to parse density model, setting to static exponential density model")
            self.density_model = self.static_exp_dens_func

        # FILL OUT THE INTEGRATOR FIXED STEPS WHEN REQUIRED
            
        # Parameters
        self.scen_times = np.linspace(0, self.simulation_duration, self.steps) 
        self.mu = 3.986004418e14  # earth's gravitational constant meters^3/s^2
        self.re = 6378.1366  # radius of the earth [km]

        # MOCAT specific parameters
        R0 = np.linspace(self.min_altitude, self.max_altitude, self.shells + 1)
        self.HMid = R0[:-1] + np.diff(R0) / 2
        self.deltaH = np.diff(R0)[0]  # thickness of the shell [km]
        R0 = (self.re + R0) * 1000  # Convert to meters
        self.V = 4 / 3 * pi * np.diff(R0**3)  # volume of the shells [m^3]
        self.v_imp2 = self.v_imp * np.ones_like(self.V)  # impact velocity [km/s] Shell-wise
        self.v = self.v_imp2 * 1000 * (24 * 3600 * 365.25)  # impact velocity [m/year]
        self.Dhl = self.deltaH * 1000
        self.Dhu = -self.deltaH * 1000
        self.options = {'reltol': 1.e-4, 'abstol': 1.e-4}  # Integration options # these are likely to change
        self.R0 = R0
        self.R02 = R0  # Assuming R02 is meant to be the same as R0 based on MATLAB code

    def lunch_func_null(self):
        """
        Null launch function for species without a launch function.
        Takes discrete launch function from file. 

        Args:
            None

        Returns: 
            numpy.ndarray: Lambdadot, the rate of change in the species in each shell at the specified time due to launch.
        """

        # Create an array filled with zeros
        Lambdadot = np.zeros(self.shells, 1)


    def launch_func_constant(self):
        """
        Adds constant launch rate from species_properties.lambda_constant

        Args:
            t (float): Time from scenario start in years
            h (array_like): The set of altitudes of the scenario above ellipsoid in km of shell lower edges.
            species_properties (dict): A dictionary with properties for the species
            scen_properties (dict): A dictionary with properties for the scenario

        Returns:
            numpy.ndarray: The rate of change in the species in each shell at the specified time due to launch.
                        If only one value is applied, it is assumed to be true for all shells.
        """

        lambda_constant = [500 * random.random() for i in range(self.shells)]

        # Generate symbolic variables and multiply each by the corresponding lambda_constant value
        Lambdadot_symbols = symbols('Lambdadot_1:%d' % (self.shells + 1))  # Create n shells symbolic variables
        Lambdadot = Matrix(self.shells, 1, lambda i, j: Lambdadot_symbols[i] * lambda_constant[i])
        
        return Lambdadot