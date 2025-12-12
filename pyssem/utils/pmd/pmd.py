
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
        
        # Check if the linked active species uses IADC PMD function
        # If so, we need to handle both failed and successful PMD differently
        if hasattr(species, 'pmd_func') and species.pmd_func == pmd_func_iadc:
            # For IADC: handle via pmd_func_derelict_iadc
            # This will be called separately if needed
            # For now, fall through to standard failed PMD handling
            pass
        
        # Failed PMD contribution for each linked species
        for k in range(scen_properties.n_shells):
            Cpmddot[k, i] = (1 - Pm) / species.deltat * species.sym[k]

    return Cpmddot

def pmd_func_derelict_iadc(t, h, species_properties, scenario_properties):
    """
    PMD function for derelict objects in IADC model.
    
    PMD applies to ALL satellites. For each timestep:
    - 0.1 (10%) of satellites convert to derelicts at the same altitude (failed PMD)
    - 0.9 (90%) convert to derelicts and are spread evenly across shells that decay in under 25 years (successful PMD)
    """
    n_shells = scenario_properties.n_shells
    Cpmddot = zeros(n_shells, 1)
    
    # Compute shells with decay times under 25 years
    if species_properties.pmd_linked_species:
        linked_species = species_properties.pmd_linked_species[0]
        
        # Get beta from linked species (ballistic coefficient)
        beta = getattr(linked_species, 'beta', None)
        if beta is None:
            # Fallback: compute from A and mass if available
            if hasattr(linked_species, 'A') and hasattr(linked_species, 'mass'):
                Cd = getattr(linked_species, 'Cd', 2.2)
                beta = Cd * linked_species.A / linked_species.mass
            else:
                beta = getattr(species_properties, 'beta', 0.01)
        
        shell_marginal_decay_rates = np.zeros(n_shells)
        shell_marginal_residence_times = np.zeros(n_shells)
        
        for k in range(n_shells):
            rhok = densityexp(scenario_properties.R0_km[k])
            rvel_current_D = -rhok * beta * np.sqrt(
                scenario_properties.mu * scenario_properties.R0[k]
            ) * (24 * 3600 * 365.25)
            shell_marginal_decay_rates[k] = -rvel_current_D / scenario_properties.Dhl
            shell_marginal_residence_times[k] = 1.0 / shell_marginal_decay_rates[k]
        
        shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)
        # Find all shells where objects will decay in under 25 years
        valid_indices = np.where(shell_cumulative_residence_times <= 25.0)[0]
        
        # If no shells decay in under 25 years, use the lowest shell (index 0)
        if len(valid_indices) == 0:
            valid_indices = np.array([0])
        
        num_valid_shells = len(valid_indices)
        
        # Process each linked active species (apply to ALL shells)
        for i, linked_species in enumerate(species_properties.pmd_linked_species):
            tau = linked_species.deltat
            
            for k in range(n_shells):
                Xk = linked_species.sym[k]
                
                # Failed PMD (0.1): add to derelict at same shell k
                Cpmddot[k, i] += 0.1 / tau * Xk
                
                # Successful PMD (0.9): spread evenly across shells that decay in under 25 years
                successful_pmd_rate = 0.9 / tau * Xk
                rate_per_shell = successful_pmd_rate / num_valid_shells
                
                for valid_k in valid_indices:
                    Cpmddot[valid_k, i] += rate_per_shell
    
    return Cpmddot

def pmd_func_iadc(t, h, species_properties, scenario_properties):
    """
    PMD function for objects that are part of the IADC model.

    PMD applies to ALL satellites (both naturally and non-naturally compliant).
    For each timestep:
    - 0.1 (10%) of satellites convert to derelicts at the same altitude (failed PMD)
    - 0.9 (90%) convert to derelicts and are spread evenly across shells that decay in under 25 years (successful PMD)
    
    This function returns the change to the active species population.
    The transfer to derelict species is handled by pmd_func_derelict_iadc on the
    derelict species, which receives both failed and successful PMD from pmd_linked_species.
    """
    # Return a single column vector for the active species
    n_shells = scenario_properties.n_shells
    Cpmddot = zeros(n_shells, 1)

    # --- 1. Compute cumulative residence time per shell and find k_star (5-year disposal altitude) ---
    shell_marginal_decay_rates = np.zeros(n_shells)
    shell_marginal_residence_times = np.zeros(n_shells)

    for k in range(n_shells):
        rhok = densityexp(scenario_properties.R0_km[k])
        rvel_current_D = -rhok * species_properties.beta * np.sqrt(
            scenario_properties.mu * scenario_properties.R0[k]
        ) * (24 * 3600 * 365.25)
        shell_marginal_decay_rates[k] = -rvel_current_D / scenario_properties.Dhl
        shell_marginal_residence_times[k] = 1.0 / shell_marginal_decay_rates[k]

    shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)

    # Find k_star: shell where objects will decay in 5 years
    indices = np.where(shell_cumulative_residence_times <= 5.0)[0]
    k_star = max(indices) if len(indices) > 0 else 0

    # --- 2. Apply PMD to active species (ALL shells, not just non-compliant ones) ---
    tau = species_properties.deltat  # PMD decision timescale

    for k in range(n_shells):
        # Use the active species population
        Xk = species_properties.sym[k]

        # Everyone in this shell "attempts" PMD at rate 1/tau:
        attempt_rate = Xk / tau

        # Remove all attempts from shell k (active species decreases)
        # Both successful (0.9) and failed (0.1) PMD satellites leave the active species
        Cpmddot[k, 0] -= attempt_rate

        # Successful PMD (0.9): satellites move to k_star disposal altitude and become derelicts
        # Failed PMD (0.1): satellites become derelicts in the same shell k
        # Both are handled by pmd_func_derelict_iadc on the derelict species

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