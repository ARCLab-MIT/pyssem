
from sympy import zeros, symbols
import numpy as np
from ..drag.drag import densityexp

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

def pmd_func_iadc(t, h, species_properties, scenario_properties):
    """
    PMD function for objects that are part of the IADC model. 

    Here, the succesful derelicts are not removed from the population, 
    they are placed at the 25 year disposal altitude. 

    The failed derelicts remain at the same altitude.
    """
     # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scenario_properties.n_shells, 1)

    shell_marginal_decay_rates = np.zeros(scenario_properties.n_shells)
    shell_marginal_residence_times = np.zeros(scenario_properties.n_shells)
    shell_cumulative_residence_times = np.zeros(scenario_properties.n_shells)

    # Here using the ballastic coefficient of the species, we are trying to find the highest compliant altitude/shell
    for k in range(scenario_properties.n_shells):
        rhok = densityexp(scenario_properties.R0_km[k])
        rvel_current_D = -rhok * species_properties.beta * np.sqrt(scenario_properties.mu * scenario_properties.R0[k]) * (24 * 3600 * 365.25)
        shell_marginal_decay_rates[k] = -rvel_current_D/scenario_properties.Dhl
        shell_marginal_residence_times[k] = 1/shell_marginal_decay_rates[k]
    
    shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)
    
    # Find the index of shell_cumulative_residence_times, k_star, which is the largest index that  shell_cumulative_residence_times(k_star) <= self.disposalTime
    indices = np.where(shell_cumulative_residence_times <= 25)[0]
    k_star = max(indices) if len(indices) > 0 else 0

    # loop through the species and if it is linked, then all in the naturally compliant shells are left where they are
    # for succesful derelicts, they are moved to the highest compliant shell, k_star
    # for failed derelicts, they remain at the same altitude
    for i, species in enumerate(species_properties.pmd_linked_species):
        # create empty equation to add to Cpmdot
        successful_disposal = 0

        # first handle the failed derelicts
        for k in range(scenario_properties.n_shells):
            Cpmddot[k, i] = (1 - species.Pm) / species.deltat * species.sym[k]
        
        # handle the succesful pmd. 
        for k in range(scenario_properties.n_shells):
            if k <= k_star:
                Cpmddot[k, i] += (species.Pm) / species.deltat * species.sym[k]
            
            if k > k_star:
                # add the succesful derelicts to the kstar shell
                Cpmddot[k_star, i] += (species.Pm) / species.deltat * species.sym[k]
            

        
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