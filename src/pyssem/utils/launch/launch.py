import numpy as np

def launch_func(t, h, species_properties, scen_properties):
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
    if len(h) != scen_properties['N_shell']:
        raise ValueError("Constant launch rate must be specified per altitude shell.")

    # Create an array filled with species_properties.lambda_constant
    Lambdadot = np.ones(scen_properties['N_shell']) * species_properties['lambda_constant']

    return Lambdadot
