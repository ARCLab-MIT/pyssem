from sympy import zeros, Matrix, symbols
import pandas as pd

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

def ADEPT_Traffic_model(scen_properties, filepath):
    """_summary_

    :param scen_properties: _description_
    :type scen_properties: _type_
    :param filepath: _description_
    :type filepath: _type_
    """
    T = pd.read_csv(filepath)
    T['epoch_start_datime'] = T['epoch_start']

    # Define object classes if 'object_class' is not already defined
    if 'object_class' not in T.columns:
        T = define_object_class(T)
        T['object_class'] = 'UNKNOWN'

    # Calculate Altitude
    T['apogee'] = T['sma'] * (1 + T['ecc'])
    T['perigee'] = T['sma'] * (1 - T['ecc'])
    T['alt'] = (T['apogee'] + T['perigee']) / 2 - scen_properties.re
    
    # Map species types using a dictionary
    species_dict = {
        "Non-station-keeping Satellite": "Sns",
        "Rocket Body": "B",
        "Station-keeping Satellite": "Su",
        "Coordinated Satellite": "S",
        "Debris": "N",
        "Candidate Satellite": "C"
    }

    T['species_class'] = T['obj_class'].map(species_dict)

    for obj_class in T['obj_class'].unique():
        species_class = species_dict[obj_class]
        species_cell = scen_properties.species[species_class]
        
        print(f"{obj_class}, {species_class}")
        T_obj_class = T[T['obj_class'] == obj_class]
        T_obj_class['species'] = T_obj_class['mass'].apply(find_mass_bin, args=(scen_properties, species_cell))

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
        if species.mass_min <= mass < species.mass_max:
            return species.sym_name
        return None


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



