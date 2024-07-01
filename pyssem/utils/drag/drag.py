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
j
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

def preload_density_data(file_path):
    with open(file_path, 'r') as file:
        density_data = json.load(file)
    return density_data

# Function to precompute date mapping for given time range
def precompute_date_mapping(start_date, end_date, num_points=101):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    total_days = (end_date - start_date).days
    dates = [start_date + pd.to_timedelta(i / (num_points - 1) * total_days, unit='d') for i in range(num_points)]
    date_mapping = [date.strftime('%Y-%m') for date in dates]
    return date_mapping

# Function to precompute nearest altitude mapping using KDTree for efficient lookup
def precompute_nearest_altitudes(available_altitudes, max_query=2000, resolution=1):
    altitude_tree = KDTree(np.array(available_altitudes).reshape(-1, 1))
    altitude_mapping = {}
    for alt in range(0, max_query + resolution, resolution):
        _, idx = altitude_tree.query([[alt]])
        nearest_alt = available_altitudes[idx[0]]
        altitude_mapping[alt] = nearest_alt
    return altitude_mapping

def JB2008_dens_func(t, h, density_data, date_mapping, nearest_altitude_mapping):
    """
    Calculate density at various altitudes based on a percentage through a time range
    using precomputed data for efficiency.

    :param t: Percentage of the way through the simulation (0-100).
    :param h: List of altitudes for which densities are required.
    :param density_data: Preloaded density data.
    :param date_mapping: Precomputed date mapping.
    :param nearest_altitude_mapping: Precomputed nearest altitude mapping.
    :return: List of densities corresponding to each altitude in h.
    """
    num_dates = len(date_mapping)
    t_normalized = min(max(t / 100 * (num_dates - 1), 0), num_dates - 1)
    
    # Find the two nearest indices and their corresponding dates
    t_index_floor = int(np.floor(t_normalized))
    t_index_ceil = int(np.ceil(t_normalized))

    if t_index_ceil >= num_dates:
        t_index_ceil = num_dates - 1

    date_floor = date_mapping[t_index_floor]
    date_ceil = date_mapping[t_index_ceil]
    
    # Interpolation weight
    if t_index_floor == t_index_ceil:
        weight = 1
    else:
        weight = (t_normalized - t_index_floor) / (t_index_ceil - t_index_floor)

    # Get density values for the floor and ceil dates
    density_values_floor = []
    density_values_ceil = []

    for alt in h:
        query_alt = round(min(alt, max(nearest_altitude_mapping.keys())), 0) # wont index if a decimal
        nearest_alt = nearest_altitude_mapping[query_alt]

        try:
            density_floor = density_data[date_floor][str(nearest_alt)]
            density_ceil = density_data[date_ceil][str(nearest_alt)]
        except KeyError as e:
            print(f"KeyError: {e} for date_floor: {date_floor}, date_ceil: {date_ceil}, nearest_alt: {nearest_alt}")
            return None

        density_values_floor.append(density_floor)
        density_values_ceil.append(density_ceil)

    # Ensure that the interpolated values correctly capture the cyclical variations
    density_values_floor = np.array(density_values_floor)
    density_values_ceil = np.array(density_values_ceil)
    
    density_values = density_values_floor * (1 - weight) + density_values_ceil * weight

    return density_values