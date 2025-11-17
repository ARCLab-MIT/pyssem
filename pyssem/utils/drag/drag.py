from sympy import zeros, symbols, sqrt, exp
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import json
import os

<<<<<<< HEAD
def densityexp(h_km):
    """
    Vallado Table 8-4 exponential density model (vectorized), matching the MATLAB densityexp_vec.
    
    Parameters
    ----------
    h_km : array_like
        Altitude above the ellipsoid in **kilometers** (must be >= 0).
    
    Returns
    -------
    p : ndarray
        Density in kg/m^3 (matches the MATLAB code as given, where the km^3 conversion is commented out).
        To convert to kg/km^3, multiply by 1e9.
    """
    h = np.asarray(h_km, dtype=float)

    # Lower edges h0 and corresponding (p0, H) for each interval
    h0 = np.array([   0,    25,    30,    40,    50,    60,    70,    80,    90,   100,
                     110,   120,   130,   140,   150,   180,   200,   250,   300,   350,
                     400,   450,   500,   600,   700,   800,   900,  1000], dtype=float)
    p0 = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5,
                   3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10,
                   2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                   1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15], dtype=float)
    H  = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877,
                   7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
                   58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00], dtype=float)

    # Build edges exactly like MATLAB: [0, 25, 30, ..., 1000, Inf]
    edges = np.concatenate([h0, [np.inf]])

    # Bin index k so that edges[k] <= h < edges[k+1]
    k = np.searchsorted(edges, h, side='right') - 1

=======
# def densityexp(h):
#     """
#     Calculates atmospheric density based on altitude using a exponential model.

#     Args:
#         h (np.array): Height of orbital shells in m.

#     Returns:
#         np.ndarray: Atmospheric density in kg/m^3
#     """
    
#     # Convert h to a numpy array for vectorized operations
#     h = np.array(h)

#     # Initialize the pressure array
#     p = np.zeros_like(h, dtype=float)

#     # Define altitude layers and corresponding parameters (h0, p0, H) based on Vallado (2013)
#     layers = [
#         (0, 1.225, 7.249),
#         (25, 3.899e-2, 6.349),
#         (30, 1.774e-2, 6.682),
#         (40, 3.972e-3, 7.554),
#         (50, 1.057e-3, 8.382),
#         (60, 3.206e-4, 7.714),
#         (70, 8.770e-5, 6.549),
#         (80, 1.905e-5, 5.799),
#         (90, 3.396e-6, 5.382),
#         (100, 5.297e-7, 5.877),
#         (110, 9.661e-8, 7.263),
#         (120, 2.438e-8, 9.473),
#         (130, 8.484e-9, 12.636),
#         (140, 3.845e-9, 16.149),
#         (150, 2.070e-9, 22.523),
#         (180, 5.464e-10, 29.740),
#         (200, 2.789e-10, 37.105),
#         (250, 7.248e-11, 45.546),
#         (300, 2.418e-11, 53.628),
#         (350, 9.518e-12, 53.298),
#         (400, 3.725e-12, 58.515),
#         (450, 1.585e-12, 60.828),
#         (500, 6.967e-13, 63.822),
#         (600, 1.454e-13, 71.835),
#         (700, 3.614e-14, 88.667),
#         (800, 1.170e-14, 124.64),
#         (900, 5.245e-15, 181.05),
#         (1000, 3.019e-15, 268.00),
#     ]

#     # Calculate density for each altitude value
#     for h0, p0, H in layers:
#         mask = (h >= h0) & (h < h0 + 100)
#         p[mask] = p0 * np.exp((h0 - h[mask]) / H)

#     # Handle altitudes >= 1000 km using the last layer's parameters
#     h0, p0, H = layers[-1]
#     mask = h >= 1000
#     p[mask] = p0 * np.exp((h0 - h[mask]) / H)

#     return p

def densityexp(h_km):
    """
    Vallado Table 8-4 exponential density model (vectorized), matching the MATLAB densityexp_vec.
    
    Parameters
    ----------
    h_km : array_like
        Altitude above the ellipsoid in **kilometers** (must be >= 0).
    
    Returns
    -------
    p : ndarray
        Density in kg/m^3 (matches the MATLAB code as given, where the km^3 conversion is commented out).
        To convert to kg/km^3, multiply by 1e9.
    """
    h = np.asarray(h_km, dtype=float)

    # Lower edges h0 and corresponding (p0, H) for each interval
    h0 = np.array([   0,    25,    30,    40,    50,    60,    70,    80,    90,   100,
                     110,   120,   130,   140,   150,   180,   200,   250,   300,   350,
                     400,   450,   500,   600,   700,   800,   900,  1000], dtype=float)
    p0 = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5,
                   3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10,
                   2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                   1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15], dtype=float)
    H  = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877,
                   7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
                   58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00], dtype=float)

    # Build edges exactly like MATLAB: [0, 25, 30, ..., 1000, Inf]
    edges = np.concatenate([h0, [np.inf]])

    # Bin index k so that edges[k] <= h < edges[k+1]
    k = np.searchsorted(edges, h, side='right') - 1

>>>>>>> amos-vnv
    # MATLAB errors if any value is below 0
    if np.any(k < 0):
        raise ValueError("Input altitude h has element(s) below 0 km.")

    # Select parameters per element and compute density
    p = p0[k] * np.exp((h0[k] - h) / H[k])

    # To match MATLAB's current output units (kg/m^3), do NOT convert.
    # If you want kg/km^3 instead, uncomment the next line:
    # p = p * (1000.0**3)

    

    return p

<<<<<<< HEAD
=======
# def densityexp(h):
#     """
#     Calculates atmospheric density based on altitude using a 
#     simple exponential model (Vallado, Table 8-4).

#     Args:
#         h (array_like): Height above ellipsoid in **kilometres**.

#     Returns:
#         np.ndarray: Atmospheric density in kg/m³
#     """
#     # — make array and sanity check —
#     h = np.asarray(h, dtype=float)
#     if np.any(h < 0):
#         raise ValueError("Altitude must be ≥ 0 km")

#     # — Table of (h0 [km], p0 [kg/m³], H [km]) —
#     layers = np.array([
#         (   0, 1.225e+0,   7.249),
#         (  25, 3.899e-2,   6.349),
#         (  30, 1.774e-2,   6.682),
#         (  40, 3.972e-3,   7.554),
#         (  50, 1.057e-3,   8.382),
#         (  60, 3.206e-4,   7.714),
#         (  70, 8.770e-5,   6.549),
#         (  80, 1.905e-5,   5.799),
#         (  90, 3.396e-6,   5.382),
#         ( 100, 5.297e-7,   5.877),
#         ( 110, 9.661e-8,   7.263),
#         ( 120, 2.438e-8,   9.473),
#         ( 130, 8.484e-9,  12.636),
#         ( 140, 3.845e-9,  16.149),
#         ( 150, 2.070e-9,  22.523),
#         ( 180, 5.464e-10, 29.740),
#         ( 200, 2.789e-10, 37.105),
#         ( 250, 7.248e-11, 45.546),
#         ( 300, 2.418e-11, 53.628),
#         ( 350, 9.518e-12, 53.298),
#         ( 400, 3.725e-12, 58.515),
#         ( 450, 1.585e-12, 60.828),
#         ( 500, 6.967e-13, 63.822),
#         ( 600, 1.454e-13, 71.835),
#         ( 700, 3.614e-14, 88.667),
#         ( 800, 1.170e-14,124.640),
#         ( 900, 5.245e-15,181.050),
#         (1000, 3.019e-15,268.000),
#     ])
#     h0 = layers[:,0]
#     p0 = layers[:,1]
#     H  = layers[:,2]

#     # — find for each h which layer to use —
#     # np.digitize with bins = h0[1:] gives:
#     #   idx = 0 for h <  25 km  → layer 0 (h0=0)
#     #   idx = 1 for 25 ≤ h < 30 km → layer 1 (h0=25)
#     #   … 
#     #   idx = 29 for h ≥ 1000 km    → layer 29 (h0=1000)
#     idx = np.digitize(h, bins=h0[1:], right=False)

#     # — compute density exactly as MATLAB does —
#     ρ = p0[idx] * np.exp((h0[idx] - h) / H[idx])

#     return ρ

>>>>>>> amos-vnv
def densityexp_jbvalues(h):
    """
    Returns interpolated atmospheric density values based on reference altitudes and densities.

    Args:
        h (np.array): Altitude(s) in km.

    Returns:
        np.ndarray: Atmospheric density in kg/km^3.
    """
    h = np.array(h)

    # Reference altitudes (km) and corresponding densities (kg/km^3)
    ref_altitudes = np.arange(200, 2001, 50)
    ref_densities = np.array([
        2.583811e-10, 5.520264e-11, 1.534066e-11, 4.964659e-12,
        1.760085e-12, 6.636416e-13, 2.637648e-13, 1.110813e-13,
        5.034028e-14, 2.496753e-14, 1.375360e-14, 8.467861e-15,
        5.746924e-15, 4.183609e-15, 3.185043e-15, 2.492537e-15,
        1.983579e-15, 1.608618e-15, 1.333113e-15, 1.123039e-15,
        9.574195e-16, 8.229503e-16, 7.110234e-16, 6.276950e-16,
        5.557886e-16, 4.923830e-16, 4.358694e-16, 3.862553e-16,
        3.435880e-16, 3.068270e-16, 2.750936e-16, 2.476455e-16,
        2.238544e-16, 2.031884e-16, 1.851967e-16, 1.694964e-16,
        1.557627e-16])

    # Linear interpolation of log-density for smoothness
    log_densities = np.log(ref_densities)
    interp_logs = np.interp(h, ref_altitudes, log_densities, left=np.nan, right=np.nan)
    densities = np.exp(interp_logs)

    return densities


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

def drag_func_exp_time_dep(t, h, species, scen_properties):
    """
    Creating the symbolic variables for the drag function for time-dependent density.
    
    This version does NOT include the ballistic coefficient (beta), allowing it to be
    multiplied by density during integration without double-counting.

    Args:
        t (float): Time from scenario start in years
        species (Species): A Species Object with properties for the species
        scen_properties (ScenProperties): A ScenarioProperties Object with properties for the scenario
        
    Returns:
        tuple: (upper_term, current_term) - drag terms without ballistic coefficient
    """
    rvel_upper = zeros(scen_properties.n_shells, 1)
    rvel_current = zeros(scen_properties.n_shells, 1)
    upper_term = zeros(scen_properties.n_shells, 1)
    current_term = zeros(scen_properties.n_shells, 1)

    seconds_per_year = 365.25 * 24 * 3600

    if species.drag_effected:
        # Set up equations for the rate of change of the semi major axis, WITHOUT beta
        for k in range(scen_properties.n_shells):            

            # Check the shell is not the top shell
            if k < scen_properties.n_shells - 1:
                n0 = species.sym[k+1]
                # Calculate Drag Flux (Relative Velocity) WITHOUT beta
                rvel_upper[k] = -sqrt(scen_properties.mu * scen_properties.R0[k+1]) * seconds_per_year
            
            # Otherwise assume that no flux is coming down from the highest shell
            else:
                n0 = 0
                rvel_upper[k] = -sqrt(scen_properties.mu * scen_properties.R0[k+1]) * seconds_per_year
        
            rvel_current[k] = -np.sqrt(scen_properties.mu * scen_properties.R0[k]) * seconds_per_year
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
    # val = densityexp_jbvalues(h)

    old = densityexp(h) 

    return old

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
    using precomputed data for efficiency. Corrected implementation to match MATLAB logic.

    :param t: Percentage of the way through the simulation (0-100).
    :param h: List of altitudes for which densities are required.
    :param density_data: Preloaded density data.
    :param date_mapping: Precomputed date mapping.
    :param nearest_altitude_mapping: Precomputed nearest altitude mapping.
    :return: List of densities corresponding to each altitude in h.
    """
    # Ensure t is in valid range
    t = max(0, min(100, t))
    
    # Convert percentage to index in date_mapping
    num_dates = len(date_mapping)
    t_index = t / 100.0 * (num_dates - 1)
    
    # Get the two nearest dates for interpolation
    t_index_floor = int(np.floor(t_index))
    t_index_ceil = int(np.ceil(t_index))
    
    # Ensure indices are within bounds
    t_index_floor = max(0, min(t_index_floor, num_dates - 1))
    t_index_ceil = max(0, min(t_index_ceil, num_dates - 1))
    
    date_floor = date_mapping[t_index_floor]
    date_ceil = date_mapping[t_index_ceil]
    
    # Calculate interpolation weight
    if t_index_floor == t_index_ceil:
        weight = 0.0
    else:
        weight = (t_index - t_index_floor) / (t_index_ceil - t_index_floor)
    
    # Get density values for each altitude
    density_values = []
    
    for alt in h:
        # Find nearest altitude
        query_alt = round(alt, 0)
        if query_alt in nearest_altitude_mapping:
            nearest_alt = nearest_altitude_mapping[query_alt]
        else:
            # Find closest altitude
            available_alts = list(nearest_altitude_mapping.keys())
            nearest_alt = min(available_alts, key=lambda x: abs(x - query_alt))
            nearest_alt = nearest_altitude_mapping[nearest_alt]
        
        try:
            # Get density values for both dates
            density_floor = density_data[date_floor][str(nearest_alt)]
            density_ceil = density_data[date_ceil][str(nearest_alt)]
            
            # Interpolate between the two dates
            density_value = density_floor * (1 - weight) + density_ceil * weight
            density_values.append(density_value)
            
        except KeyError as e:
            print(f"KeyError: {e} for date_floor: {date_floor}, date_ceil: {date_ceil}, nearest_alt: {nearest_alt}")
            return None
    
    return density_values

def calculate_orbital_lifetimes(scenario_properties):
    """
        This function is mainly used for UMPY calculations. For each species, it will calculate the orbital lifetimes based off the static density model. 
    """

    shell_marginal_decay_rates = np.zeros(scenario_properties.n_shells)
    shell_marginal_residence_times = np.zeros(scenario_properties.n_shells)

    # loop through each of the species
    for species_group in scenario_properties.species.values():
        for species in species_group:

            species.orbital_lifetimes = [None] * scenario_properties.n_shells

            if not species.drag_effected:
                # create an array that is the length of n_shells with each a value of deltat
                species.orbital_lifetimes = np.full(scenario_properties.n_shells, species.deltat)
            else:
                for k in range(scenario_properties.n_shells):
                    rhok = densityexp(scenario_properties.R0_km[k])

                    # satellite 
                    # beta = 0.0172 # ballastic coefficient, area * mass * drag coefficient. This should be done for each species!
                    if species.beta is None:
                        raise ValueError("Beta is not defined for species")
                    
                    rvel_current_D = -rhok * species.beta * np.sqrt(scenario_properties.mu * scenario_properties.R0[k]) * (24 * 3600 * 365.25)
                    shell_marginal_decay_rates[k] = -rvel_current_D/scenario_properties.Dhl
                    shell_marginal_residence_times[k] = 1/shell_marginal_decay_rates[k]
    
                species.orbital_lifetimes = np.cumsum(shell_marginal_residence_times)
                
                # Maximum orbital lifetime is the simulation duration
                species.orbital_lifetimes = np.minimum(
                    species.orbital_lifetimes,
                    scenario_properties.simulation_duration
                )
    
    return scenario_properties.species

