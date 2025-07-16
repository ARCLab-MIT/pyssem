from sympy import zeros, Matrix, symbols
import pandas as pd
from datetime import datetime, timedelta
from ..handlers.datetime_helper import jd_to_datetime, mjd_to_jd
import os
import numpy as np
from tqdm import tqdm 

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

    for k in range(scen_properties.n_shells):
        Lambdadot[k, 0] = 0 * species_properties.sym[k]

    Lambdadot_list = [Lambdadot[k, 0] for k in range(scen_properties.n_shells)]

    return Lambdadot_list

def launch_lambda_sym(t, h, species_properties, scen_properties):

    Lambdadot = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        Lambdadot[k, 0] = symbols(f'lambda_{species_properties.sym_name}{k+1}')

    return Lambdadot

def launch_lambda_sym_null(t, h, species_properties, scen_properties):
    return launch_func_null(t, h, species_properties, scen_properties)

def launch_func_constant(t, h, species_properties, scen_properties):
    """
    Adds a constant launch rate from species_properties.lambda_constant.
    Given a certain altitude, this will return the rate of change in the species in each shell at the specified time due to launch.

    Args:
        t (float): Time from scenario start in years.
        h (list or numpy.ndarray): Altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): Properties for the species, including 'lambda_constant'.
        scen_properties (dict): Properties for the scenario, including 'N_shell'.

    Returns:
        list: np., a list of symbolic expressions representing the rate of change in the species in each shell due to launch.
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

def launch_func_random(t, h, species_properties, scen_properties):
    """
    Adds a random launch rate from species_properties.lambda_random.
    Given a certain altitude, this will return the rate of change in the species in each shell at the specified time due to launch.

    Args:
        t (float): Time from scenario start in years.
        h (list or numpy.ndarray): Altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): Properties for the species, including 'lambda_random'.
        scen_properties (dict): Properties for the scenario, including 'N_shell'.

    Returns:
        list: np., a list of symbolic expressions representing the rate of change in the species in each shell due to launch.
    """

    if len(h) != scen_properties.n_shells:
        raise ValueError("Random launch rate must be specified per altitude shell.")

    # Create a symbolic variable for the launch rate
    lambda_random = symbols('lambda_random')

    # Assign the random launch rate to each shell
    Lambdadot = Matrix(scen_properties.n_shells, 1, lambda i, j: lambda_random)

    # Convert the Matrix of symbolic expressions to a list
    Lambdadot_list = [Lambdadot[i] for i in range(scen_properties.n_shells)]

    return Lambdadot_list

def launch_func_lambda_fun(t, h, species_properties, scen_properties):
    """
    This function will return the lambda function for a required species. 

    :param t: The time from the scenario start in years
    :type t: int
    :param h: The altitude above the ellipsoid in km of shell lower edge
    :type h: int
    :param species_properties: Species properties
    :type species_properties: Species
    :param scen_properties: Scenario properties
    :type scen_properties: ScenarioProperties
    :return: Lambdadot is the rate of change in the species in each sheel at the specified time due to launch
    :rtype: SciPy interp1d function
    """
    # # Find the index for the given altitude
    # h_inds = np.where(scen_properties.HMid == h)
    # print(species_properties.sym_name)

    # Retrieve the appropriate lambda function for the altitude and evaluate it at time t
    Lambdadot = species_properties.lambda_funs
    return Lambdadot


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

def find_bin_index(bin_edges, value):
    """Find the index of the bin a value belongs to."""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value < bin_edges[i + 1]:
            return i
    return -1  # Return an invalid index if not found

def find_closest_species(obj, species_list, category):
    """
    Match an object to its best-fit species using:
    1. Stationkeeping (drag-affected logic) + maneuverability
    2. Maneuverability only
    3. Mass bin fallback
    """
    obj_mass = obj['mass']
    if pd.isna(obj_mass):
        return None

    stkp_flg = obj['stkp_flg'] if not pd.isna(obj['stkp_flg']) else 0
    maneuverable = obj['maneuverable'] if not pd.isna(obj['maneuverable']) else 0

    matched_species = []

    if maneuverable == 1:
        # Match: maneuverable AND drag-affected
        if stkp_flg == 3:
            stkp_maneuverable_species = [
                s for s in species_list
                if getattr(s, 'maneuverable', True) and getattr(s, 'drag_effected', True)
            ]
            matched_species.extend(filter_species_by_mass(obj_mass, stkp_maneuverable_species))

        # Match: maneuverable AND NOT drag-affected
        elif stkp_flg == 0:
            non_drag_maneuverable_species = [
                s for s in species_list
                if getattr(s, 'maneuverable', True) and not getattr(s, 'drag_effected', False)
            ]
            matched_species.extend(filter_species_by_mass(obj_mass, non_drag_maneuverable_species))

        # If stationkeeping wasn't matched, fallback to any maneuverable species
        if not matched_species:
            fallback_maneuverable_species = [
                s for s in species_list if getattr(s, 'maneuverable', False)
            ]
            matched_species.extend(filter_species_by_mass(obj_mass, fallback_maneuverable_species))

    elif maneuverable == 0:
        # Match only non-maneuverable species
        non_maneuverable_species = [
            s for s in species_list if not getattr(s, 'maneuverable', False)
        ]
        matched_species.extend(filter_species_by_mass(obj_mass, non_maneuverable_species))

    # Step 3: If nothing matched yet, fallback to all by mass bin
    if not matched_species:
        matched_species.extend(filter_species_by_mass(obj_mass, species_list))

    # Step 4: Warn if multiple species matched
    unique_matches = {s.sym_name: s for s in matched_species}.values()
    if len(unique_matches) > 1:
        print(f"Warning: obj_id {obj['obj_id']} matches multiple species: {[s.sym_name for s in unique_matches]}")

    # Step 5: Return closest by mass center
    return min(unique_matches, key=lambda s: abs(obj_mass - get_species_mass_center(s)), default=None)

def filter_species_by_mass(obj_mass, species_list):
    """
    Return list of species where object mass falls within their mass bounds.
    """
    matches = []
    for s in species_list:
        if s.mass_lb <= obj_mass < s.mass_ub:
            matches.append(s)
    return matches

def get_species_mass_center(species):
    """
    Estimate the central mass value for a species.
    """
    if isinstance(species.mass, (list, tuple)):
        return (species.mass[0] + species.mass[1]) / 2
    elif species.mass is not None:
        return species.mass
    else:
        return float('inf')  # fallback if undefined
    
def find_mass_bin_species(obj_mass, species_list):
    """
    Find the best match based on mass bin or nearest mass center.
    """
    for s in species_list:
        if s.mass_lb <= obj_mass < s.mass_ub:
            return s

    # Fallback to closest by mean mass
    closest = None
    min_diff = float('inf')
    for s in species_list:
        if isinstance(s.mass, (list, tuple)):
            mean_mass = (s.mass[0] + s.mass[1]) / 2
        else:
            mean_mass = s.mass
        diff = abs(obj_mass - mean_mass)
        if diff < min_diff:
            min_diff = diff
            closest = s
    return closest

def get_species_category(obj, species_dict):
    """
    Determine species category ('Active', 'Debris', 'RocketBodies') using obj_type and flags.
    """
    obj_type = int(obj['obj_type']) if not pd.isna(obj['obj_type']) else None

    if obj_type == 1:
        return "rocket_body"
    elif obj_type == 2:
        return "active"
    elif obj_type == 3:
        return "debris"
    elif obj_type == 4:
        return "debris"
    elif obj_type == 5:
        return "active"
    elif obj_type == 6:
        return "debris"
    else:
        # Fallback: use active flag if available
        if not pd.isna(obj['active']):
            return "Active" if bool(obj['active']) else "debris"
        return "debris"

def assign_species_to_population(T, species_mapping):
    """
    Applies a list of pandas query strings to assign species classes to the population.
    
    :param T: pandas.DataFrame representing the population
    :param species_mapping: list of assignment strings (e.g., T.loc[...] = ...)
    :return: updated DataFrame with 'species_class' assigned
    """
    # Initialize the column
    T['species_class'] = "Unknown"

    # Apply each mapping rule via exec
    for rule in species_mapping:
        try:
            exec(rule)
        except Exception as e:
            print(f"Error applying rule: {rule}\n\t{e}")

    # Print summary of resulting species_class assignments
    print("\nSpecies class distribution:")
    print(T['species_class'].value_counts())

    try:
        T = T[T['species_class'] != "Unknown"]
        print(f"\n{T['species_class'].value_counts()['Unknown']} objects/rows are being removed.")
    except KeyError:
        print("No unknown species classes found.")

    return T

def SEP_traffic_model(scen_properties, file_path):
    """
    This will take one of the SEP files, users must select on of the Space Environment Pathways (SEPs), see: https://www.researchgate.net/publication/385299836_Development_of_Reference_Scenarios_and_Supporting_Inputs_for_Space_Environment_Modeling
    
    This function will create an initial population (x0) and a future launch model (FLM_steps)
    for the given scenario and the species properties that have been configured. 
    """

    # Calculate Apogee, Perigee, and altitude
    T = pd.read_csv(file_path)

    T['apogee'] = T['sma'] * (1 + T['ecc'])
    T['perigee'] = T['sma'] * (1 - T['ecc'])
    T['alt'] = (T['apogee'] + T['perigee']) / 2 - scen_properties.re

    # Filter Rows Based on Min and Max_Altitude
    T = T[(T['alt'] >= scen_properties.min_altitude) & (T['alt'] <= scen_properties.max_altitude)] 

    T_new = assign_species_to_population(T, scen_properties.SEP_mapping)

    for species_class in T['species_class'].unique():
            if species_class in scen_properties.species_cells:
                if len(scen_properties.species_cells[species_class]) == 1:
                    T_obj_class = T[T['species_class'] == species_class].copy()
                    T_obj_class['species'] = scen_properties.species_cells[species_class][0].sym_name
                    T_new = pd.concat([T_new, T_obj_class])
                else:
                    species_cells = scen_properties.species_cells[species_class]
                    T_obj_class = T[T['species_class'] == species_class].copy()
                    T_obj_class['species'] = T_obj_class['mass'].apply(find_mass_bin, args=(scen_properties, species_cells)) 
                    T_new = pd.concat([T_new, T_obj_class])

    print(f"Number of objects for each species in T_new: {T_new['species'].value_counts()}")

    # T_new['epoch_start_datetime'] = T_new['year_start'].apply(
    #         lambda y: datetime(int(y), 1, 1)
    #     )
    # T_new['epoch_end_datetime'] = T_new['year_final'].apply(
    #     lambda y: datetime(int(y), 1, 1)
    # )
    T_new['epoch_start_datetime'] = pd.to_datetime(dict(
        year=T_new['year_start'].astype(int),
        month=T_new['month_start'].astype(int),
        day=T_new['day_start'].astype(int)
    ), errors='coerce')

    T_new['epoch_end_datetime'] = pd.to_datetime(dict(
        year=T_new['year_final'].astype(int),
        month=T_new['month_final'].astype(int),
        day=T_new['day_final'].astype(int)
    ), errors='coerce')

    T_new['alt_bin'] = T_new['alt'].apply(find_alt_bin, args=(scen_properties,))

    # Filter T_new to include only species present in scen_properties
    T_new = T_new[T_new['species'].isin(scen_properties.species_names)]

    # Initial population
    x0 = T_new[T_new['epoch_start_datetime'] < scen_properties.start_date]

    # x0['species'].value_counts().plot(kind='bar', figsize=(12, 6))

    x0.to_csv(os.path.join('pyssem', 'utils', 'launch', 'data', 'x0.csv'))

    if scen_properties.elliptical:
        # === 3D case: [alt_bin, species_idx, ecc_bin] ===

        # Bin eccentricity
        ecc_edges = np.array(scen_properties.eccentricity_bins)
        x0['ecc_bin'] = pd.cut(x0['ecc'], bins=ecc_edges, labels=False, include_lowest=True)

        # Create empty 3D summary array
        n_shells = scen_properties.n_shells
        n_species = len(scen_properties.species_names)
        n_ecc_bins = len(ecc_edges) - 1

        x0_summary = np.zeros((n_shells, n_species, n_ecc_bins), dtype=int)

        # Map species name → index
        species_name_to_index = {name: idx for idx, name in enumerate(scen_properties.species_names)}

        # Fill in the summary
        for _, row in x0.iterrows():
            alt_bin = row['alt_bin']
            species_idx = species_name_to_index.get(row['species'], None)
            ecc_bin = row['ecc_bin']

            if pd.notna(ecc_bin) and species_idx is not None:
                x0_summary[alt_bin, species_idx, int(ecc_bin)] += 1

    else:
        # === Standard 2D case: DataFrame [alt_bin, species] ===
        df = x0.pivot_table(index='alt_bin', columns='species', aggfunc='size', fill_value=0)
        x0_summary = pd.DataFrame(index=range(scen_properties.n_shells), columns=scen_properties.species_names).fillna(0)
        x0_summary.update(df.reindex(columns=x0_summary.columns, fill_value=0))

    if scen_properties.baseline:
        return x0_summary, None

    # Future Launch Model (updated)
    flm_steps = pd.DataFrame()

    time_increment_per_step = scen_properties.simulation_duration / scen_properties.steps

    time_steps = [
        scen_properties.start_date + timedelta(days=365.25 * time_increment_per_step * i) 
        for i in range(scen_properties.steps + 1)
    ]    

    # Distribute the Yearly Launches, USED FOR STEP FUNCTION, but also works w/ current and old interp
    start_year = scen_properties.start_date.year
    end_year = start_year + scen_properties.simulation_duration

    for year in tqdm(range(start_year, end_year), desc="Processing Launch Years"):
        
        launches_this_year = T_new[T_new['year_start'] == year]
        
        if launches_this_year.empty:
            continue

        # Group by shell and species to get total counts for the entire year
        yearly_counts = launches_this_year.groupby(['alt_bin', 'species']).size().unstack(fill_value=0)

        # --- START OF THE FIX ---
        # Ensure the yearly_counts DataFrame has a row for every possible shell.
        # This is the step that was missing from my previous version.
        yearly_counts = yearly_counts.reindex(range(scen_properties.n_shells), fill_value=0)
        # --- END OF THE FIX ---

        # Find which simulation time steps fall within this calendar year
        year_start_date = datetime(year, 1, 1)
        year_end_date = datetime(year + 1, 1, 1)
        
        relevant_steps_mask = (np.array(time_steps[:-1]) >= year_start_date) & (np.array(time_steps[:-1]) < year_end_date)
        relevant_start_times = np.array(time_steps[:-1])[relevant_steps_mask]

        if len(relevant_start_times) == 0:
            continue

        # Use only the first time step in the year (mimicking MC behavior)
        step_counts = yearly_counts.copy()
        step_counts = step_counts.reset_index()
        step_counts['epoch_start_date'] = relevant_start_times[0]

        flm_steps = pd.concat([flm_steps, step_counts], ignore_index=True)
        # num_sub_steps = len(relevant_start_times)

        # if num_sub_steps == 0:
        #     continue

        # # Create records for these time steps
        # for start_time in relevant_start_times:
        #     # Pro-rate the yearly counts to get the count for this smaller time step
        #     step_counts = yearly_counts / num_sub_steps
            
        #     step_counts = step_counts.reset_index()
        #     step_counts['epoch_start_date'] = start_time
            
        #     flm_steps = pd.concat([flm_steps, step_counts], ignore_index=True)

    # Final re-ordering and cleanup
    # Ensure all species columns from the scenario are present, even if they had no launches
    all_species_columns = scen_properties.species_names
    for col in all_species_columns:
        if col not in flm_steps.columns:
            flm_steps[col] = 0

    # Ensure consistent column order
    final_cols = ['epoch_start_date', 'alt_bin'] + all_species_columns
    flm_steps = flm_steps[final_cols]

    return x0_summary, flm_steps

def find_species_bin(row, scen_properties, species_cells):
    """
    Determine the species bin for a given row, based on mass and/or eccentricity.

    :param row: Row from DataFrame (must have 'mass' and 'ecc')
    :param scen_properties: ScenarioProperties object
    :param species_cells: List of Species objects
    :return: species.sym_name that matches the object's properties
    """
    for species in species_cells:
        # Check for eccentricity-based binning
        if getattr(species, "elliptical", False):
            if species.mass_lb <= row.mass < species.mass_ub and \
               species.ecc_lb <= row.ecc < species.ecc_ub:
                return species.sym_name
        else:
            if species.mass_lb <= row.mass < species.mass_ub:
                return species.sym_name

    return None  # No match found


def ADEPT_traffic_model(scen_properties, file_path):
    """
    From an initial population and future model csv, this function will create for the starting population, 
    then one for each time step in the future model.

    The output matrices will be in the form of a matrix, with the species as columns and the number of orbital shells as rows based on alt_bin.
    e.g. if you have 5 species and 10 shells, the matrix will be 10x5.

    :param scen_properties: Scenario properties
    :type scen_properties: ScenarioProperties
    :param file_path: Local File Path of CSV
    :type file_path: str
    :return: The initial population and future launch model
    :rtype:  pandas.DataFrame, pandas.DataFrame
    """
    # Load the traffic model data
    T = pd.read_csv(file_path)
    
    T['epoch_start_datime'] = T['epoch_start'].apply(lambda x: julian_to_datetime(x))

    if 'obj_class' not in T.columns:
        T = define_object_class(T)  # Make sure this function is defined and imported

    # Calculate Apogee, Perigee, and Altitude
    T['apogee'] = T['sma'] * (1 + T['ecc'])
    T['perigee'] = T['sma'] * (1 - T['ecc'])
    T['alt'] = (T['apogee'] + T['perigee']) / 2 - scen_properties.re

    # Map species type based on object class
    species_dict = {
        "Non-station-keeping Satellite": "Su",
        "Rocket Body": "B",
        "Station-keeping Satellite": "S",
        "Coordinated Satellite": "S",
        "Debris": "N",
        "Candidate Satellite": "S"
    }

    T['species_class'] = T['obj_class'].map(species_dict)

    # Initialize an empty DataFrame for new data
    T_new = pd.DataFrame()

    # Loop through object classes and assign species based on mass
    for obj_class in T['obj_class'].unique():
        species_class = species_dict.get(obj_class)
        if species_class in scen_properties.species_cells:
            if len(scen_properties.species_cells[species_class]) == 1:
                T_obj_class = T[T['obj_class'] == obj_class].copy()
                T_obj_class['species'] = scen_properties.species_cells[species_class][0].sym_name
                T_new = pd.concat([T_new, T_obj_class])
            else:
                species_cells = scen_properties.species_cells[species_class]
                T_obj_class = T[T['obj_class'] == obj_class].copy()
                T_obj_class['species'] = T_obj_class['mass'].apply(find_mass_bin, args=(scen_properties, species_cells)) 
                T_new = pd.concat([T_new, T_obj_class])

    # Assign objects to corresponding altitude bins
    T_new['alt_bin'] = T_new['alt'].apply(find_alt_bin, args=(scen_properties,))

    # Filter T_new to include only species present in scen_properties
    T_new = T_new[T_new['species_class'].isin(scen_properties.species_cells.keys())]

    # Initial population
    x0 = T_new[T_new['epoch_start_datime'] < scen_properties.start_date]

    x0.to_csv(os.path.join('pyssem', 'utils', 'launch', 'data', 'x0.csv'))

    # Create a pivot table, keep alt_bin
    df = x0.pivot_table(index='alt_bin', columns='species', aggfunc='size', fill_value=0)

    # Create a new data frame with column names like scenario_properties.species_sym_names and rows of length n_shells
    x0_summary = pd.DataFrame(index=range(scen_properties.n_shells), columns=scen_properties.species_names).fillna(0)
    x0_summary.index.name = 'alt_bin'

    # Merge the two dataframes
    for column in df.columns:
        if column in x0_summary.columns:
            x0_summary[column] = df[column]

    # fill NaN with 0
    x0_summary.fillna(0, inplace=True)

    if baseline:
        # No need to calculate the launch model
        return x0_summary, None

    # Future Launch Model
    flm_steps = pd.DataFrame()

    time_increment_per_step = scen_properties.simulation_duration / scen_properties.steps

    time_steps = [scen_properties.start_date + timedelta(days=365.25 * time_increment_per_step * i) 
                  for i in range(scen_properties.steps + 1)]    

    for i, (start, end) in tqdm(enumerate(zip(time_steps[:-1], time_steps[1:])), total=len(time_steps)-1, desc="Processing Time Steps"):
        flm_step = T_new[(T_new['epoch_start_datime'] >= start) & (T_new['epoch_start_datime'] < end)]
        flm_summary = flm_step.groupby(['alt_bin', 'species']).size().unstack(fill_value=0)

        # All objects aren't always in shells, so you need to these back in. 
        flm_summary = flm_summary.reindex(range(0, scen_properties.n_shells), fill_value=0)

        flm_summary.reset_index(inplace=True)
        flm_summary.rename(columns={'index': 'alt_bin'}, inplace=True)

        flm_summary['epoch_start_date'] = start  # Add the start date to the table for reference
        flm_steps = pd.concat([flm_steps, flm_summary])
    
    return x0_summary, flm_steps

def find_eccentricity_bin(eccentricity, scen_properties, species_cell):
    """
    Find the eccentricity bin for a given eccentricity.

    :param eccentricity: Eccentricity of the object's orbit
    :type eccentricity: float
    :param scen_properties: The scenario properties object
    :type scen_properties: ScenarioProperties
    :param species_cell: The species cell to find the eccentricity bin for
    :type species_cell: Species
    :return: The species name corresponding to the given eccentricity
    :rtype: str
    """

    for species in species_cell:
        if species.ecc_lb <= eccentricity < species.ecc_ub:
            return species.sym_name


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
    """
    Given an altidude and the generic pySSEM properties, it will calculate the index from the R02 array

    :param altitude: Altitude of an object
    :type altitude: int
    :param scen_properties: The scenario properties object
    :type scen_properties: ScenarioProperties
    :return: Orbital Shell Array Index or None if out of range
    :rtype: int
    """
    shell_altitudes = scen_properties.R0_km

    # The case for an object where it is below the lowest altitude
    if altitude < shell_altitudes[0]:
        return
    
    # The case for an object where it is above the highest altitude
    if altitude >= shell_altitudes[-1]:
        return 

    for i in range(len(shell_altitudes)):  # -1 to prevent index out of range
        try:
            if shell_altitudes[i] <= altitude < shell_altitudes[i + 1]:
                return i  
        except IndexError: # This is the top most shell and will be the last one
            return len(shell_altitudes) 
    

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