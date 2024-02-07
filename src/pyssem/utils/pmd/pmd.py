import numpy as np

def pmd_func(t, h, species_properties, scen_properties):
    """
    Post Mission Disposal function
    Args:
        t (float): Time from scenario start in years
        h (array_like): The set of altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): A dictionary with properties for the species
        scen_properties (dict): A dictionary with properties for the scenario
    Returns:
        numpy.ndarray: The rate of change in the species in each shell at the specified time due to post mission disposal.
                        If only one value is applied, it is assumed to be true for all shells.
    """
    # Create an array filled with zeros

    ## this function is to be completed later
    Pmdot = np.zeros(scen_properties['N_shell'])

    return Pmdot
