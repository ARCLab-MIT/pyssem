
from sympy import zeros, symbols

def pmd_func_none(t, h, species_properties, scen_properties):
    """
    PMD function for species with no PMD. Returns a zero matrix.

    Args:
        t (float): Time from scenario start in years (unused).
        h (float or np.ndarray): Height above ellipsoid in km (unused).
        species_properties (dict): Properties for this species.
        scen_properties (dict): Properties for the scenario.

    Returns:
        sympy.Matrix: Cpmdot, the rate of change in the species due to post-mission
                      disposal, an N_shell x 1 matrix.
    """
    return zeros(scen_properties.n_shells, 1)

def pmd_func_sat(t, h, species_properties, scen_properties):
    """
    PMD function for active classes that decay to a derelict. It only decreases
    the class based on assumed successful PMD.

    Args:
        t (float): Time from scenario start in years (unused).
        h (float or np.ndarray): Height above ellipsoid in km (unused).
        species_properties (dict): Properties for this species.
        scen_properties (dict): Properties for the scenario.

    Returns:
        sympy.Matrix: Cpmdot, the rate of change in the species due to post-mission
                      disposal, an N_shell x 1 matrix.
    """
    # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scen_properties.n_shells, 1)
    
    # Iterate over each shell and calculate the PMD rate
    try:
        for k in range(scen_properties.n_shells):
            Cpmddot[k, 0] = (-1 / species_properties.deltat) * species_properties.sym[k]
    except Exception as e:
        print(f"Error in pmd_func_sat: {e}")
    return Cpmddot

def pmd_func_derelict(t, h, species_properties, scen_properties):
    """
    PMD function for derelict objects. Only increase the class based on assumed failed PMD 
    from species in species_properties.linked_species

    Args:
        t (float): Time from scenario start in years (unused).
        h (float or np.ndarray): Height above ellipsoid in km (unused).
        species_properties (dict): Properties for this species.
        scen_properties (dict): Properties for the scenario.

    Returns:
        sympy.Matrix: Cpmdot, the rate of change in the species due to post-mission
                      disposal, an N_shell x 1 matrix.
    """
    # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scen_properties.n_shells, 1)

    # Iterate over each shell and calculate the PMD rate
    for i, species in enumerate(species_properties.pmd_linked_species):
        Pm = species.Pm # 0 = no Pmd, 1 = full Pm
        
        # Failed PMD contribution for each linked species
        for k in range(scen_properties.n_shells):
            Cpmddot[k, i] = (1 - Pm) / species.deltat * species.sym[k]

    return Cpmddot

def pmd_func_opus(t, h, species_properties, scen_properties):
    """
    PMD function for objects that are part of the OPUS model. 
    Here, PMD is handled on the OPUS side of the Integrated model, so this function
    will return all zeros.

    Args:
        t (float): Time from scenario start in years (unused).
        h (float or np.ndarray): Height above ellipsoid in km (unused).
        species_properties (dict): Properties for this species.
        scen_properties (dict): Properties for the scenario.

    Returns:
        sympy.Matrix: Cpmdot, the rate of change in the species due to post-mission
                      disposal, an N_shell x 1 matrix.
    """
    # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scen_properties.n_shells, 1)
    
    return Cpmddot