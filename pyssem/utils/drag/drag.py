from sympy import zeros, symbols, sqrt, exp
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import json
import os

def densityexp(h):
    """
    Calculates atmospheric density based on altitude using a exponential model.

    Args:
        h (np.array): Height of orbital shells in km.

    Returns:
        np.ndarray: Atmospheric density in kg/km^3.
    """
    
    # Convert h to a numpy array for vectorized operations
    h = np.array(h)

    # Initialize the pressure array
    p = np.zeros_like(h, dtype=float)

    # Define altitude layers and corresponding parameters (h0, p0, H) based on Vallado (2013)
    layers = [
        (0, 1.225, 7.249),
        (25, 3.899e-2, 6.349),
        (30, 1.774e-2, 6.682),
        (40, 3.972e-3, 7.554),
        (50, 1.057e-3, 8.382),
        (60, 3.206e-4, 7.714),
        (70, 8.770e-5, 6.549),
        (80, 1.905e-5, 5.799),
        (90, 3.396e-6, 5.382),
        (100, 5.297e-7, 5.877),
        (110, 9.661e-8, 7.263),
        (120, 2.438e-8, 9.473),
        (130, 8.484e-9, 12.636),
        (140, 3.845e-9, 16.149),
        (150, 2.070e-9, 22.523),
        (180, 5.464e-10, 29.740),
        (200, 2.789e-10, 37.105),
        (250, 7.248e-11, 45.546),
        (300, 2.418e-11, 53.628),
        (350, 9.518e-12, 53.298),
        (400, 3.725e-12, 58.515),
        (450, 1.585e-12, 60.828),
        (500, 6.967e-13, 63.822),
        (600, 1.454e-13, 71.835),
        (700, 3.614e-14, 88.667),
        (800, 1.170e-14, 124.64),
        (900, 5.245e-15, 181.05),
        (1000, 3.019e-15, 268.00),
    ]

    # Calculate density for each altitude value
    for h0, p0, H in layers:
        mask = (h >= h0) & (h < h0 + 100)
        p[mask] = p0 * np.exp((h0 - h[mask]) / H)

    # Handle altitudes >= 1000 km using the last layer's parameters
    h0, p0, H = layers[-1]
    mask = h >= 1000
    p[mask] = p0 * np.exp((h0 - h[mask]) / H)

    return p

def drag_func_none(t, species, scen_properties):
    """
    Drag function for species with no drag. Returns a zero matrix.

    :param t: _description_
    :type t: _type_
    :param species: _description_
    :type species: _type_
    :param scen_properties: _description_
    :type scen_properties: _type_
    """

    return zeros(scen_properties.n_shells, 1)

def drag_func_exp(t, h, species, scen_properties):
    """
    Creating the symbolic variables for the drag function. 

    Drag function for the species, without density. This allows for time varying rhos to be used at much better speed. 

    Args:
        t (float): Time from scenario start in years
        species (Species): A Species Object with properties for the species
        scen_properties (ScenProperties): A ScenarioProperties Object with properties for the scenario

    Returns:
        numpy.ndarray: The rate of change in the species in each shell at the specified time due to drag.
                       If only one value is applied, it is assumed to be true for all shells.
    """
    rvel_upper = zeros(scen_properties.n_shells, 1)
    rvel_current = zeros(scen_properties.n_shells, 1)
    upper_term = zeros(scen_properties.n_shells, 1)
    current_term = zeros(scen_properties.n_shells, 1)

    seconds_per_year = 365.25 * 24 * 3600

    if species.drag_effected:
        # Set up equations for the rate of change of the semi major axis, density not included
        for k in range(scen_properties.n_shells):            

            # Check the shell is not the top shell
            if k < scen_properties.n_shells - 1:
                n0 = species.sym[k+1]
                # Calculate Drag Flux (Relative Velocity)
                rvel_upper[k] = -species.beta * sqrt(scen_properties.mu * scen_properties.R0[k+1]) * seconds_per_year
            
            # Otherwise assume that no flux is coming down from the highest shell
            else:
                n0 = 0
                rvel_upper[k] = -species.beta * sqrt(scen_properties.mu * scen_properties.R0[k+1]) * seconds_per_year
        
            rvel_current[k] = -species.beta * np.sqrt(scen_properties.mu * scen_properties.R0[k]) * seconds_per_year
            upper_term[k] = n0 * rvel_upper[k] / scen_properties.Dhu
            current_term[k] = rvel_current[k] / scen_properties.Dhl * species.sym[k]
    
    return upper_term, current_term

def static_exp_dens_func(t, h, species, scen_properties):
    """
    This is a wrapper for densityexp to be used in the simulation. 

    :param t: time is the time in years (unused in this function)
    :type t: int
    :param h: height above the ellipsoid in km
    :type h: int
    :param species: _description_
    :type species: _type_
    :param scen_properties: _description_
    :type scen_properties: _type_
    """
    return densityexp(h)

def JB2008_dens_func(t, h, scen_times_dates, start_date, end_date, num_steps):
    """
    This will take in an array of species at different altitudes and then 
    will calulate the density at each altitude and return the new number of species in each shell.

    :param t: Time from t0
    :type t: int
    :param h: np.array of altitudes in km
    :type h: np.array
    :param scen_times_dates: A list of the scen_times in year-month format
    :type scen_times_dates: List of str. 
    """

    # Calculate the total time range in days
    try:
        total_days = (end_date - start_date).days

        # Calculate the target date based on the total time range and the timestep
        target_date_dt = start_date + pd.DateOffset(days=t * total_days / num_steps)

        # Find the closest date in year-month format from the available dates
        closest_date = min(scen_times_dates, key=lambda x: abs(pd.to_datetime(x) - target_date_dt))

        # Restate target_date in year-month format
        target_date_str = target_date_dt.strftime('%Y-%m')

        # Load the density data
        path = os.path.join(os.path.dirname(__file__), 'dens_highvar_2000_dens_highvar_2000_lookup.json')

        with open(path, 'r') as file:
            density_data = json.load(file)

        # Extract the altitudes and densities for the closest date
        altitude_values = np.array([int(alt) for alt in density_data[target_date_str].keys()])
        altitude_tree = KDTree(h.reshape(-1, 1))

        # Output array for densities
        density_values = np.empty_like(h, dtype=float)
        density_values = [] 

    except KeyError:
        print(f"Error: Altitude {closest_alt} not found in density data for date {target_date_str}.")

    try:
        for i, alt in enumerate(h):
            closest_alt_idx = altitude_tree.query([alt])[1]  
            closest_alt = altitude_values[closest_alt_idx]

            # Check if closest_alt_idx is a single index or an array
            if isinstance(closest_alt_idx, int):
                closest_alt_idx = [closest_alt_idx]

            density_values.append(density_data[target_date_str][str(closest_alt)])
   
    except:
        print(f"Error: Altitude {closest_alt} not found in density data for date {target_date_str}.")

    return density_values

    
def population_shell(t, x, obj):
    """
    For time varying atmosphere, density needs to be computed within the integrated function, 
    not as an argument outside it. 

    :param t: is a time in years from start date
    :type t: _type_
    :param x: is the equation state
    :type x: _type_
    :param obj: is the simulation object
    :type obj: _type_
    Returns: the rate of change in the species in each shell at the specified time due to drag
    """

    # need to continue closer to the time
    obj.scen_properties.X = x
    obj.scen_properties.t = t



