
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
    for k in range(scen_properties.n_shells):
        Cpmddot[k, 0] = (-1 / species_properties.deltat) * species_properties.sym[k]
    
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
    num_linked_species = len(species_properties.pmd_linked_species)

    # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scen_properties.n_shells, 1)

    # Iterate over each shell and calculate the PMD rate
    for i, species in enumerate(species_properties.pmd_linked_species):
        Pm = species.Pm # 0 = no Pmd, 1 = full Pm
        deltat = species.deltat
        
        # Failed PMD contribution for each linked species
        for k in range(scen_properties.n_shells):
            Cpmddot[k, i] = (1 - Pm) / deltat * species.sym[k]

def pair_actives_to_debris(scen_properties, active_species, debris_species):
    """
    Pairs all active species to debris species for PMD modeling.

    Args:
        scen_properties (dict): Properties for the scenario.
        active_species (list): List of active species objects.
        debris_species (list): List of debris species objects.
    """
    # Collect active species and their names
    linked_spec_list = []
    linked_spec_names = []
    for cur_species in active_species:
        if cur_species.species_properties['active']:
            linked_spec_list.append(cur_species)
            linked_spec_names.append(cur_species.species_properties['sym_name'])

    print("Pairing the following active species to debris classes for PMD modeling...")
    print(linked_spec_names)

    # Assign matching debris increase for a species due to failed PMD
    for active_spec in linked_spec_list:
        found_mass_match_debris = False
        spec_mass = active_spec.species_properties['mass']

        for deb_spec in debris_species:
            if spec_mass == deb_spec.species_properties['mass']:
                # Assume pmd_func_derelict is a function or method to be set for the debris species
                deb_spec.pmd_func = pmd_func_derelict  # You need to define this function
                if 'pmd_linked_species' not in deb_spec.species_properties:
                    deb_spec.species_properties['pmd_linked_species'] = []
                deb_spec.species_properties['pmd_linked_species'].append(active_spec)
                print(f"Matched species {active_spec.species_properties['sym_name']} to debris species {deb_spec.species_properties['sym_name']}.")
                found_mass_match_debris = True

        if not found_mass_match_debris:
            print(f"No matching mass debris species found for species {active_spec.species_properties['sym_name']} with mass {spec_mass}.")

    # Display information about linked active species for each debris species
    for deb_spec in debris_species:
        linked_spec_names = [spec.species_properties['sym_name'] for spec in deb_spec.species_properties.get('pmd_linked_species', [])]
        print(f"    Name: {deb_spec.species_properties['sym_name']}")
        print(f"    pmd_linked_species: {linked_spec_names}")
        # Additional processing for disposal_altitude and pmd_linked_multiplier can be added here

# Example usage of find_alt_bin, assuming altitude bins are predefined in scen_properties
def find_alt_bin(disposal_altitude, scen_properties):
    # This is a placeholder function; you need to define the logic based on your altitude bins
    pass

