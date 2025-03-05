import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp

class Indicator:
    """
    This is a class to crate indicator variables. 

    These are quantities of interest that do not correspond directly to the quantity of a species in a shell. They can be differential equations
    that need to be integrated alongside species, functions of system states, or even derivaives of a system states. T
    """
    def __init__(self, name, ind_type, species, eqs=None):
            self.name = name
            self.ind_type = ind_type
            self.species = species
            self.eqs = eqs
            self.indicator_idxs = None
            self.indicators = []  

def make_indicator_struct(obj, name, ind_type, species, eqs=None):
    """
    Helper function that creates an indicator variable structure.
    This is intended to be called during the build_model method
    for each indicator variable intended to be added to the simulation.

    :param obj: The main simulation object containing scenario properties.
    :param name: The name for the indicator variable.
    :param ind_type: The type of the indicator variable.
    :param species: The species involved in the indicator variable.
    :param eqs: Symbolic value corresponding to the equations (only for manual type).
    :return: An Indicator object.
    """
    valid_types = ["collision", "successful PMD", "failed PMD", "mitigated conjunctions", "manual"]

    # Error checking for manual type
    if ind_type == "manual" and eqs is None:
        raise ValueError("Argument eqs must be passed for indicator variable with type 'manual'.")
    
    if ind_type != "manual" and eqs is not None:
        raise Warning("Argument eqs passed for indicator with non-manual type. Overriding equations to use passed value. To remove this error, use type 'manual'.")

    # Check if ind_type is a supported type
    if ind_type not in valid_types:
        raise ValueError(f"Passed type {ind_type} is not a supported type. Please use values from {valid_types}.")

    if ind_type == "collision":
        if len(species) != 2:
            raise ValueError("Exactly two species must be provided for collision indicator variables.")
        
        # Find the correct pair
        pair = None
        for test_pair in obj.collision_pairs:
            species1match = test_pair.species1.sym_name == species[0]
            species2match = test_pair.species2.sym_name == species[1]
            if species1match and species2match:
                pair = test_pair
                break
        
        if pair is None:
            raise ValueError(f"No matching species pair has been found for: '{species[0]}', '{species[1]}'.")

        # Non-Gamma
        intrinsic_collisions = sp.Matrix(pair.phi).multiply_elementwise(sp.Matrix(pair.species1.sym)).multiply_elementwise(sp.Matrix(pair.species2.sym))
        
        # Make Eqs
        eqs = -1 * pair.gammas[0] * intrinsic_collisions  # Negative 1 to counteract decrease to quantity to provide positive number of collisions

    elif ind_type == "mitigated conjunction":
        if len(species) != 2:
            raise ValueError("Exactly two species must be provided for mitigated conjunction indicator variables.")
        
        # Find the correct pair
        pair = None
        for test_pair in obj.collision_pairs:
            species1match = test_pair.species1.sym_name == species[0]
            species2match = test_pair.species2.sym_name == species[1]
            if species1match and species2match:
                pair = test_pair
                break
        
        if pair is None:
            raise ValueError(f"No matching species pair has been found for: '{species[0]}', '{species[1]}'.")

        # Non-Gamma
        intrinsic_collisions = sp.Matrix(pair.phi).multiply_elementwise(sp.Matrix(pair.species1.sym)).multiply_elementwise(sp.Matrix(pair.species2.sym))
        
        # Make Eqs
        eqs = (1 - pair.gammas[0]) * intrinsic_collisions

    elif ind_type == "manual":
        # Already handled above
        pass

    # Create the Indicator object
    return Indicator(name, ind_type, species, eqs)

def make_intrinsic_cap_indicator(scen_properties, sep_dist_method, sep_angle=0.2, sep_dist=25.0, shell_sep=5, inc=45.0, graph=False):
    """
    This looks at the estimated number of slotted spacraft that fit in LEO subject to certain assumptions.

    :param scen_properties: Scenario properties object
    :type scen_properties: object
    :param sep_dist_method: Separation distance method ('angle' or 'distance')
    :type sep_dist_method: str
    :param sep_angle: Separation angle in degrees, defaults to 0.2
    :type sep_angle: float, optional
    :param sep_dist: Separation distance in km, defaults to 25.0
    :type sep_dist: float, optional
    :param shell_sep: Shell separation in km, defaults to 5
    :type shell_sep: int, optional
    :param inc: Inclination in degrees, defaults to 45.0
    :type inc: float, optional
    :param graph: Whether to plot the graph, defaults to False
    :type graph: bool, optional
    :return: Indicator object
    :rtype: Indicator
    """
    def inc_validation(x):
        return isinstance(x, (int, float)) and 0 < x <= 90
    
    def valid_scalar_pos_num(x):
        return isinstance(x, (int, float)) and x > 0
    
    # Input validation
    if sep_dist_method not in ["angle", "distance"]:
        raise ValueError("sep_dist_method must be 'angle' or 'distance'")
    if not valid_scalar_pos_num(sep_angle):
        raise ValueError("sep_angle must be a positive number")
    if not valid_scalar_pos_num(sep_dist):
        raise ValueError("sep_dist must be a positive number")
    if not valid_scalar_pos_num(shell_sep):
        raise ValueError("shell_sep must be a positive number")
    if not inc_validation(inc):
        raise ValueError("inc must be a number between 0 and 90")

    # Provide warning for mismatched sep_dist_method and distance value.
    if sep_dist_method == "angle" and sep_dist != 25.0:
        warnings.warn("'sep_dist' argument was passed despite 'sep_dist_method' being set to 'angle'. 'sep_dist' parameter was ignored. 'sep_angle' may have been set to a default value")

    if sep_dist_method == "distance" and sep_angle != 0.2:
        warnings.warn("'sep_angle' argument was passed despite 'sep_dist_method' being set to 'distance'. 'sep_angle' parameter was ignored. 'sep_dist' may have been set to a default value")

    # Read data
    filename = 'LawDF_min_500_fit_range_max_10000_sols_1.csv'
    data = pd.read_csv(os.path.join('pyssem', 'utils', 'indicators', filename)).values[1:]  # skip header
    index_mat = data[:, 0]
    inc_mat = data[:, 1]
    c_mat = data[:, 2]
    b_mat = data[:, 3]
    R2_mat = data[:, 4]

    # Figure out which row to use based on the chosen inclination.
    closest_index = np.argmin(np.abs(inc_mat - inc))
    ind_intrinsic = closest_index

    rad = np.pi / 180
    
    if sep_dist_method == 'angle':
        N_sat_eq_ang = lambda c, b: (sep_angle / c) ** (1 / b)
        shells_per_bin = np.diff(scen_properties.R0) / shell_sep
        N_sat = N_sat_eq_ang(c_mat[ind_intrinsic], b_mat[ind_intrinsic]) * shells_per_bin
    else:
        N_sat_eq = lambda h, c, b: (sep_dist / ((scen_properties.re + h) * rad * c)) ** (1 / b)
        shells_per_bin = np.diff(scen_properties.R0) / shell_sep
        N_sat = N_sat_eq(scen_properties.R0[1:], c_mat[ind_intrinsic], b_mat[ind_intrinsic]) * shells_per_bin

    slotted_sats_eqs = sp.Matrix([sp.zeros(scen_properties.n_shells, 1)])
    for species_group in scen_properties.species.values():
        for species in species_group:
            if species.slotted:
                slotted_sats_eqs += species.sym

    N_sat = sp.Matrix(N_sat)
    unconsumed_intrinsic_capacity = N_sat - slotted_sats_eqs
    ind_struct = Indicator("unconsumed_intrinsic_capacity", "manual", None, unconsumed_intrinsic_capacity)

    if graph:
        # create a plot and save in the figures folder at root
        plt.figure()
        plt.scatter(scen_properties.R0[1:], N_sat)
        plt.title("Intrinsic Capacity per Altitude Bin")
        plt.xlabel("Altitude of Bin Ceiling [km]")
        plt.ylabel("Intrinsic Capacity [Satellites]")
        # Check to see if the figures directory exists, if not create it
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig("figures/intrinsic_capacity.png")

    return ind_struct


def make_ca_counter(scen_properties, primary_species_list, secondary_species_list, per_species=False, ind_name="", per_spacecraft=False):
    """
    This method makes an indicator variable structure corresponding to the number of collision avoidance maneuvers performed in a given year by the primary species. 
    It assumes that collision avoidance burden is divided evenly if two active species have a conjunction. Slotting effectiveness is not considered, but should be
    handled through inclusion of Gamma. 

    :param scen_properties: Simulation Object
    :type scen_properties: ScenarioProperties
    :param primary_species_list: List of primary species
    :type primary_species_list: list
    :param secondary_species_list: List of secondary species
    :type secondary_species_list: list
    :param per_species: Whether to return values for each species as independent indicators, defaults to False
    :type per_species: bool, optional
    :param ind_name: Name of the indicator, defaults to ""
    :type ind_name: str, optional
    :param per_spacecraft: Whether to normalize by the number of spacecraft, defaults to False
    :type per_spacecraft: bool, optional
    :return: Indicator variables
    :rtype: list or Indicator
    """
    primary_species_list_attribute = ""
    secondary_species_list_attribute = ""

    if isinstance(primary_species_list, str):
        primary_species_list_attribute = primary_species_list
        primary_species_list = [species for group in scen_properties.species.values() for species in group if primary_species_list_attribute == "all" or getattr(species, primary_species_list_attribute)]
        
    if isinstance(secondary_species_list, str):
        secondary_species_list_attribute = secondary_species_list
        secondary_species_list = [species for group in scen_properties.species.values() for species in group if secondary_species_list_attribute == "all" or getattr(species, secondary_species_list_attribute)]

    primary_species_names = [species.sym_name for species in primary_species_list]
    secondary_species_names = [species.sym_name for species in secondary_species_list]

    if ind_name == "":
        ind_name = "col_avoidance_maneuvers_pri_" + ('_'.join(primary_species_names) if primary_species_list_attribute == "" else primary_species_list_attribute)
        ind_name += "_sec_" + ('_'.join(secondary_species_names) if secondary_species_list_attribute == "" else secondary_species_list_attribute)
        if per_spacecraft:
            ind_name += "_per_spacecraft"

    ind_eqs = {species.sym_name: sp.zeros(scen_properties.n_shells, 1) for species in primary_species_list + secondary_species_list if species.maneuverable and species.active}

    for primary_species in primary_species_list:
        primary_species_name = primary_species.sym_name
        for pair in scen_properties.collision_pairs:
            pair_primary_name = pair.species1.sym_name
            pair_secondary_name = pair.species2.sym_name
            s1_prim = primary_species_name == pair_primary_name
            s2_prim = primary_species_name == pair_secondary_name

            if s1_prim or s2_prim:
                secondary_species_name = pair_secondary_name if s1_prim else pair_primary_name
                if secondary_species_name in secondary_species_names:
                    intrinsic_collisions = sp.Matrix(pair.phi).multiply_elementwise(sp.Matrix(pair.species1.sym)).multiply_elementwise(sp.Matrix(pair.species2.sym))
                    both_man = pair.species1.maneuverable and pair.species2.maneuverable
                    both_act = pair.species1.active and pair.species2.active
                    s1_man_act = pair.species1.maneuverable and pair.species1.active
                    s2_man_act = pair.species2.maneuverable and pair.species2.active
                    s1_trackable = pair.species1.trackable
                    s2_trackable = pair.species2.trackable
                    one_man_act = s1_man_act != s2_man_act

                    if both_man and both_act:
                        if pair.species1.slotted and pair.species2.slotted:
                            slotting_effectiveness = min(pair.species1.slotting_effectiveness, pair.species2.slotting_effectiveness)
                            maneuver_amount = 0.5 * (1 + pair.gammas[0]) * (1 / slotting_effectiveness) * intrinsic_collisions
                        else:
                            maneuver_amount = 0.5 * (1 + pair.gammas[0]) * intrinsic_collisions
                        ind_eqs[primary_species_name] += maneuver_amount
                        ind_eqs[secondary_species_name] += maneuver_amount
                    elif one_man_act:
                        if s1_man_act and s2_trackable:
                            ind_eqs[pair_primary_name] += (1 + pair.gammas[0]) * intrinsic_collisions
                        elif s2_man_act and s1_trackable:
                            ind_eqs[pair_secondary_name] += (1 + pair.gammas[0]) * intrinsic_collisions

    if not per_species:
        ag_man_counts = sp.zeros(scen_properties.n_shells, 1)
        for eq in ind_eqs.values():
            ag_man_counts += eq
        if per_spacecraft:
            ag_man_sat_totals = sp.zeros(scen_properties.n_shells, 1)
            for species_group in scen_properties.species.values():
                for species in species_group:
                    if species.maneuverable:
                        ag_man_sat_totals += sp.Matrix(species.sym)
            ag_man_counts = ag_man_counts.multiply_elementwise(1 / ag_man_sat_totals)
        spec_man_indc = [make_indicator_struct(scen_properties, ind_name, "manual", None, ag_man_counts)]
    else: # if per spcies
        spec_man_indc = []
        for species_name, eq in ind_eqs.items():
            if per_spacecraft:
                for species_group in scen_properties.species.values():
                    for species in species_group:
                        if species.sym_name == species_name:
                            inverse_sym = sp.Matrix(species.sym).applyfunc(lambda x: 1 / x)
                            spec_man_indc.append(make_indicator_struct(scen_properties, f"{species_name}_maneuvers_per_spacecraft", "manual", [species_name], eq.multiply_elementwise(inverse_sym)))
            else:
                spec_man_indc.append(make_indicator_struct(scen_properties, f"{species_name}_maneuvers", "manual", [species_name], eq))        

    return spec_man_indc

def make_active_loss_per_shell(scen_properties, percentage, per_species):
    """
    Calculates the indicator variable for number of active spacecraft lost in each orbit shell
    to collision events in a given year. 

    :param scen_properties: Simulation Object
    :type scen_properties: ScenarioProperties
    :param percentage: False uses absolute number, true gives the percentage
    :type percentage: Boolean
    :param per_species: True returns values for each species as independent, false sums by shell
    :type per_species: Boolean
    """

    dummy_obj = scen_properties
    species_pairs_classes = scen_properties.collision_pairs
    all_col_indicators = []

    for species_group in scen_properties.species.values():
        for species in species_group:
            species_1_name = species.sym_name
        
            spec_col_indicators = []

            for pair in species_pairs_classes:
                if (species_1_name == pair.species1.sym_name or
                    species_1_name == pair.species2.sym_name):
                    
                    species_2_name = pair.species2.sym_name
                    ind_name = f"collisions_{species_1_name}_{species_2_name}"
                    spec_pair = [pair.species1.sym_name, pair.species2.sym_name]
                    col_indicator = make_indicator_struct(dummy_obj, ind_name, "collision", spec_pair)
                    spec_col_indicators.append(col_indicator)
            
            ag_col_eqs = sp.zeros(scen_properties.n_shells, 1)
            for col_ind in spec_col_indicators:
                ag_col_eqs += col_ind.eqs
            
            spec_ag_col_indc = make_indicator_struct(dummy_obj, f"{species_1_name}_aggregate_collisions", "manual", [species], ag_col_eqs)
            all_col_indicators.append(spec_ag_col_indc)

    if per_species:
        if not percentage:
            indicator_var = all_col_indicators
        else:
            for col_ind in all_col_indicators:
                species_1_name = col_ind.species[0].sym_name
                species_1_totals = col_ind.species[0].sym
                col_ind.eqs = 100 * col_ind.eqs.multiply_elementwise(sp.Matrix(species_1_totals).applyfunc(lambda x: 1 / x))
            indicator_var = all_col_indicators
    else:
        ag_active_col_eqs = sp.zeros(scen_properties.n_shells, 1)
        for col_ind in all_col_indicators:
            if col_ind.species[0].active:
                ag_active_col_eqs += col_ind.eqs

        if not percentage:
            indicator_var = [make_indicator_struct(dummy_obj, "active_aggregate_collisions", "manual", None, ag_active_col_eqs)]
        else:
            ag_active_sat_totals = sp.zeros(scen_properties.n_shells, 1)
            for species_group in scen_properties.species.values():
                for species in species_group:
                    if species.active:
                        ag_active_sat_totals += species.sym
            
            perc_eqs = 100 * ag_active_col_eqs.multiply_elementwise(sp.Matrix(ag_active_sat_totals).applyfunc(lambda x: 1 / x))
            indicator_var = [make_indicator_struct(dummy_obj, "active_aggregate_collisions_percentage", "manual", None, perc_eqs)]
    
    return indicator_var


def make_all_col_indicators(scen_properties):
    """
    This function returns an array with a list of indicators
    corresponding to the collisions per year for each species.

    :param scen_properties: Simulation Object
    :type scen_properties: ScenarioProperties
    :return: List of indicators for collisions per species
    :rtype: list
    """
    
    dummy_obj = scen_properties
    all_col_indicators = []

    for species_group in scen_properties.species.values():
        for species in species_group:
            species_name = species.sym_name
            spec_col_indicators = []

            for pair in scen_properties.collision_pairs:
                species_1_name = pair.species1.sym_name
                species_2_name = pair.species2.sym_name
                if species_name == species_1_name or species_name == species_2_name:
                    ind_name = f"collisions_{species_1_name}_{species_2_name}"
                    spec_pair = [species_1_name, species_2_name]
                    col_indicator = make_indicator_struct(dummy_obj, ind_name, "collision", spec_pair)
                    spec_col_indicators.append(col_indicator)
            
            ag_col_eqs = sp.zeros(scen_properties.n_shells, 1)
            for col_ind in spec_col_indicators:
                ag_col_eqs += col_ind.eqs
            
            spec_ag_col_indc = make_indicator_struct(dummy_obj, f"{species_name}_aggregate_collisions", "manual", [species], ag_col_eqs)
            all_col_indicators.append(spec_ag_col_indc)
    
    return all_col_indicators

def make_umpy_indicator(scen_properties, X=4, indicator_name="umpy_indicator"):
    """
    Creates a UMPY indicator (vector of length n_shells) using a similar approach
    to 'make_active_loss_per_shell'. This sums contributions from all species
    in each shell, based on their masses, lifetimes, and symbolic population.

    :param scen_properties: The scenario properties object
    :param X: Exponent in the UMPY formula (default=4)
    :param indicator_name: Name for the resulting indicator
    :return: A list containing one IndicatorStruct, or multiple if you want per-species
    """


    # One aggregated vector eqs (n_shells x 1) summing across species
    umpy_eqs = sp.zeros(scen_properties.n_shells, 1)

    for species_group in scen_properties.species.values():
        for species in species_group:
            mass_i = species.mass

            for shell_idx in range(scen_properties.n_shells):
                if not species.active:
                    # Usual UMPY formula for inactive species
                    pop_ij  = species.sym[shell_idx]             # population in shell i
                    life_ij = species.orbital_lifetimes[shell_idx]
                    umpy_factor = ((sp.exp(X * (life_ij / scen_properties.simulation_duration)) - 1)
                                / (sp.exp(X) - 1))
                    umpy_eqs[shell_idx] += (mass_i * pop_ij * umpy_factor) / scen_properties.simulation_duration
                else:
                    # If active, just add zero
                    umpy_eqs[shell_idx] += 0

    umpy_indicator = make_indicator_struct(
        scen_properties,
        indicator_name,
        "manual",
        None,
        umpy_eqs
    )

    return [umpy_indicator]

def make_indicator_eqs(obj, ind_struct):
    """
    Helper method that creates the equations (eqs) for non-manual types and validates input.
    This is intended to be called during the simulation build_model method for each 
    indicator variable intended to be added to the simulation.

    :param obj: The main simulation object containing scenario properties.
    :param ind_struct: A dictionary-like object representing the indicator structure.
    :return: Updated ind_struct with the equations added.
    """
    name = ind_struct.name
    ind_type = ind_struct.ind_type
    species = ind_struct.species
    eqs = getattr(ind_struct, 'eqs', None)

    # Supported indicator types
    valid_types = ["collision", "successful PMD", "failed PMD", "mitigated conjunctions", "manual"]

    # Error checking for manual type
    if ind_type == "manual" and (not hasattr(ind_struct, 'eqs') or not ind_struct.eqs):
        raise ValueError("Argument eqs must be passed for indicator variable with type 'manual'.")

    if ind_type != "manual" and eqs:
        import warnings
        warnings.warn("Argument eqs passed for indicator with non-manual type. "
                      "Overriding equations to use passed value. To remove this warning, use type='manual'.")

    # Check if ind_type is supported
    if ind_type not in valid_types:
        raise ValueError(f"Passed type {ind_type} is not a supported type. Please use values from {valid_types}.")

    # Build equations
    if ind_type == "collision":
        if len(species) != 2:
            raise ValueError("Exactly two species must be provided for collision indicator variables.")

        # Find the correct pair
        pair = next(
            (test_pair for test_pair in obj.collision_pairs
             if test_pair.species1.sym_name == species[0] and test_pair.species2.sym_name == species[1]),
            None
        )
        if pair is None:
            raise ValueError(f"No matching species pair has been found for: '{species[0]}', '{species[1]}'.")

        # Non-Gamma intrinsic collisions
        intrinsic_collisions = sp.Matrix(pair.phi).multiply_elementwise(
            sp.Matrix(pair.species1.sym)).multiply_elementwise(sp.Matrix(pair.species2.sym)
        )
        # Create equations
        ind_struct.eqs = pair.gammas[0] * intrinsic_collisions

    elif ind_type == "mitigated conjunction":
        if len(species) != 2:
            raise ValueError("Exactly two species must be provided for mitigated conjunction indicator variables.")

        # Find the correct pair
        pair = next(
            (test_pair for test_pair in obj.collision_pairs
             if test_pair.species1.sym_name == species[0] and test_pair.species2.sym_name == species[1]),
            None
        )
        if pair is None:
            raise ValueError(f"No matching species pair has been found for: '{species[0]}', '{species[1]}'.")

        # Non-Gamma intrinsic collisions
        intrinsic_collisions = sp.Matrix(pair.phi).multiply_elementwise(
            sp.Matrix(pair.species1.sym)).multiply_elementwise(sp.Matrix(pair.species2.sym)
        )
        # Create equations
        ind_struct.eqs = (1 - pair.gammas[0]) * intrinsic_collisions

    elif ind_type == "successful PMD":
        if len(species) > 1:
            raise ValueError("Only single species arguments are currently supported for the 'successful PMD' type.")

        species_obj = obj.get_species_list_from_names(species[0])
        if not callable(species_obj.pmd_func):
            import warnings
            warnings.warn("Custom PMD function detected. Only constant (non-time-varying) PMD functions are currently supported.")

        ind_struct.eqs = species_obj.pmd_func(0, obj.scen_properties.HMid, species_obj.species_properties, obj.scen_properties)

    elif ind_type == "failed PMD":
        if len(species) > 1:
            raise ValueError("Only single species arguments are currently supported for the 'failed PMD' type.")

        species_obj = obj.get_species_list_from_names(species[0])
        if not callable(species_obj.pmd_func):
            import warnings
            warnings.warn("Custom PMD function detected. Only constant (non-time-varying) PMD functions are currently supported.")

        eqs = sp.zeros(obj.scen_properties.N_shell, 1)
        for species_test in obj.scen_properties.species:
            for linked_species in species_test.species_properties.pmd_linked_species:
                if linked_species.species_properties.sym_name == species[0]:
                    actual_pmd = species_test.species_properties.pmd_linked_species
                    species_test.species_properties.pmd_linked_species = linked_species
                    eqs += species_test.pmd_func(
                        0,
                        obj.scen_properties.HMid,
                        species_test.species_properties,
                        obj.scen_properties
                    )
                    species_test.species_properties.pmd_linked_species = actual_pmd
        ind_struct.eqs = eqs

    elif ind_type == "manual":
        ind_struct.eqs = eqs

    return ind_struct