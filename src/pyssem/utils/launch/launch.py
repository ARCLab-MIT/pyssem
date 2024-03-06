from sympy import zeros, Matrix, symbols
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def find_alt_bin(altitude, scen_properties):
    # Convert altitude ranges to numpy arrays for vectorized operations
    lower = np.array(scen_properties['R02'][:-1])
    upper = np.array(scen_properties['R02'][1:])
    
    # Create a boolean array where True indicates altitude is within the shell bounds
    shell_logic = (lower < altitude) & (altitude <= upper)
    
    # Find the index (or indices) where shell_logic is True
    shell_indices = np.where(shell_logic)[0]
    
    # Return the first index if found, otherwise return NaN
    shell_index = shell_indices[0] + 1 if shell_indices.size > 0 else np.nan
    return shell_index


def launch_func_null(t, h, species_properties, scen_properties):
    """
    No launch function for species without a launch function.

    Args:
        t (float): Time from scenario start in years
        h (array_like): The set of altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): A dictionary with properties for the species
        scen_properties (dict): A dictionary with properties for the scenario

    Returns:
        numpy.ndarray: The rate of change in the species in each shell at the specified time due to launch.
                       If only one value is applied, it is assumed to be true for all shells.
    """

    Lambdadot = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shell):
        Lambdadot[k, 0] = 0 * species_properties.sym[k]

    Lambdadot_list = [Lambdadot[k, 0] for k in range(scen_properties.n_shell)]

    return Lambdadot_list

def launch_func_constant(t, h, species_properties, scen_properties):
    """
    Adds a constant launch rate from species_properties.lambda_constant.

    Args:
        t (float): Time from scenario start in years.
        h (list or numpy.ndarray): Altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): Properties for the species, including 'lambda_constant'.
        scen_properties (dict): Properties for the scenario, including 'N_shell'.

    Returns:
        list: Lambdadot, a list of symbolic expressions representing the rate of change in the species in each shell due to launch.
    """
    if len(h) != scen_properties.n_shells:
        raise ValueError("Constant launch rate must be specified per altitude shell.")

    # Create a symbolic variable for the launch rate
    lambda_constant = symbols('lambda_constant')

    # Assign the constant launch rate to each shell
    Lambdadot = Matrix(scen_properties.n_shells, 1, lambda i, j: lambda_constant)

    # Convert the Matrix of symbolic expressions to a list
    Lambdadot_list = [Lambdadot[i] for i in range(scen_properties.n_shells)]

    return Lambdadot_list

# def ADEPT_Traffic_model(scen_properties, filepath):
#     """_summary_

#     :param scen_properties: _description_
#     :type scen_properties: _type_
#     :param filepath: _description_
#     :type filepath: _type_
#     """
#     T = pd.read_csv(filepath)
#     T['epoch_start_datime'] = T['epoch_start']

#     # Define object classes if 'object_class' is not already defined
#     if 'object_class' not in T.columns:
#         T = define_object_class(T)
#         T['object_class'] = 'UNKNOWN'

#     # Calculate Altitude
#     T['apogee'] = T['sma'] * (1 + T['ecc'])
#     T['perigee'] = T['sma'] * (1 - T['ecc'])
#     T['alt'] = (T['apogee'] + T['perigee']) / 2 - scen_properties.re
    
#     # Map species types using a dictionary
#     species_dict = {
#         "Non-station-keeping Satellite": "sns",
#         "Rocket Body": "B",
#         "Station-keeping Satellite": "Su",
#         "Coordinated Satellite": "S",
#         "Debris": "N",
#         "Candidate Satellite": "C"
#     }

#     T['species_class'] = T['obj_class'].map(species_dict)

#     # Define T_new
#     T_new = pd.DataFrame(columns=T.columns)

#     for obj_class in T['obj_class'].unique():
#         species_class = species_dict[obj_class]       
#         print(f"{obj_class}, {species_class}")

#         # Check that the species cell exists
#         try:
#             species_cell = scen_properties.species_cells[species_class]
#         except KeyError:
#             print(f"Species cell not found for {species_class}.")
#             continue

#         # Create a boolean mask for rows belonging to the current obj_class
#         mask = T['obj_class'] == obj_class
#         # Directly modify the 'species' column of the original DataFrame 'T' for rows that match the mask
#         T.loc[mask, 'species'] = T.loc[mask, 'mass'].apply(find_mass_bin, args=(scen_properties, species_cell))

#         T_new = pd.concat([T_new, T[mask]])

#     return

def julian_to_datetime(julian_date):
    # Julian Date for Unix epoch (1970-01-01)
    JULIAN_EPOCH = 2440587.5
    try:
        # Calculate the number of days from the Unix epoch
        days_from_epoch = julian_date - JULIAN_EPOCH
        # Create a datetime object for the Unix epoch and add the calculated days
        unix_epoch = datetime(1970, 1, 1)
        result_date = unix_epoch + timedelta(days=days_from_epoch)
        return result_date
    except OverflowError as e:
        # Handle dates that are out of range
        print(f"Date conversion error: {e}")
        return None

def ADEPT_traffic_model(scen_properties, file_path):
    # Read the CSV file into a DataFrame
    T = pd.read_csv(file_path)
    
    # Convert 'epoch_start' from Julian to datetime format
    T['epoch_start_datime'] = T['epoch_start'].apply(lambda x: julian_to_datetime(x))
    
    # Define object classes if 'obj_class' is not defined
    if 'obj_class' not in T.columns:
        T = define_object_class(T)  # Make sure this function is defined and imported
    
    # Calculate Apogee, Perigee, and Altitude
    T['apogee'] = T['sma'] * (1 + T['ecc'])
    T['perigee'] = T['sma'] * (1 - T['ecc'])
    T['alt'] = (T['apogee'] + T['perigee']) / 2 - scen_properties.re
    
    # Map species type based on object class
    species_dict = {"Non-station-keeping Satellite": "Sns",
                    "Rocket Body": "B",
                    "Station-keeping Satellite": "Su",
                    "Coordinated Satellite": "S",
                    "Debris": "N",
                    "Candidate Satellite": "C"}
    
    T['species_class'] = T['obj_class'].map(species_dict)
    
    # Initialize an empty DataFrame for new data
    T_new = pd.DataFrame()
    
    # Loop through object classes and assign species based on mass
    for obj_class in T['obj_class'].unique():
        species_class = species_dict.get(obj_class)
        if species_class in scen_properties.species_cells:
            species_cell = scen_properties.species_cells[species_class]
            
            T_obj_class = T[T['obj_class'] == obj_class].copy()
            T_obj_class['species'] = T_obj_class['mass'].apply(find_mass_bin, args=(scen_properties, species_cell))
            
            T_new = pd.concat([T_new, T_obj_class])
    
    # Assign objects to corresponding altitude bins
    T_new['alt_bin'] = T_new['alt'].apply(find_alt_bin, args=(scen_properties,))
    
    # Filter T_new to include only species present in scen_properties
    T_new = T_new[T_new['species_class'].isin(scen_properties.species_cells.keys())]
    
    # Initial population
    x0 = T_new[T_new['epoch_start_datime'] < scen_properties.start_date]
    x0_pivot = x0.pivot_table(values='weight', index='alt_bin', columns='species', aggfunc='sum', fill_value=0)
    
    # Future Launch Model
    flm_steps = []
    time_steps = [scen_properties.start_date + timedelta(days=365.25) * i for i in range(scen_properties.simulation_duration + 1)]
    
    for start, end in zip(time_steps[:-1], time_steps[1:]):
        flm_step = T_new[(T_new['epoch_start_datime'] >= start) & (T_new['epoch_start_datime'] < end)]
        print(f"Step: {start} - {end}, Objects: {flm_step.shape[0]}")
        flm_step_pivot = flm_step.pivot_table(values='weight', index='alt_bin', columns='species', aggfunc='sum', fill_value=0)
        flm_steps.append(flm_step_pivot)
    
    return x0_pivot, flm_steps

def find_mass_bin(mass, scen_properties, species_cell):
    """
    Find the mass bin for a given mass.

    :param mass: Mass of the object in kg
    :type mass: float
    :param scen_properties: The scenario properties object
    :type scen_properties: ScenarioProperties
    :param species_cell: The species cell to find the mass bin for
    :type species_cell: Species
    :return: The mass bin for the given mass
    :rtype: int
    """
    for species in species_cell:
        if species.mass_lb <= mass < species.mass_ub:
            return species.sym_name
        return None

def find_alt_bin(altitude, scen_properties):
    # Convert altitude ranges to numpy arrays for vectorized operations
    lower = np.array(scen_properties.R02[:-1])
    upper = np.array(scen_properties.R02[1:])
    
    # Create a boolean array where True indicates altitude is within the shell bounds
    shell_logic = (lower < altitude) & (altitude <= upper)
    
    # Find the index (or indices) where shell_logic is True
    shell_indices = np.where(shell_logic)[0]
    
    # Return the first index if found, otherwise return NaN
    shell_index = shell_indices[0] + 1 if shell_indices.size > 0 else np.nan
    return shell_index

def define_object_class(T):
    """
    Define the object class of each object in the traffic model.
    Adds them to a new column named "obj_type" or overwrites the existing column.

    :param T: list of launches
    :type T: pandas.DataFrame
    """

    T['obj_class'] = "Unknown"

    # Classify Rocket Bodies
    T.loc[T['obj_type'] == 1, 'obj_class'] = "Rocket Body"

    # Classify Satellites
    T.loc[(T['obj_type'] == 2) & (T['stationkeeping'] != 0) & (T['stationkeeping'] < 5), 'obj_class'] = "Station-keeping Satellite"
    T.loc[(T['obj_type'] == 2) & (T['stationkeeping'] == 0), 'obj_class'] = "Non-station-keeping Satellite"
    T.loc[(T['obj_type'] == 2) & (T['stationkeeping'] == 5), 'obj_class'] = "Coordinated Satellite"
    T.loc[(T['obj_type'] == 2) & (T['stationkeeping'] == 6), 'obj_class'] = "Candidate Satellite"

    # Classify Debris
    T.loc[T['obj_type'].isin([3, 4]), 'obj_class'] = "Debris"

    # Count unclassified rows
    unclassed_rows = (T['obj_class'] == "Unknown").sum()
    if unclassed_rows > 0:
        print(f'\t{unclassed_rows} Unclassified rows remain.')

    return T



