import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp

class Indicators:
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

def make_intrinsic_cap_indicator(scen_properties, sep_dist_method, sep_angle=0.2, sep_dist=25.0, shell_sep=5, inc=45.0, graph=False):
    """
    This looks at the estimated number of slotted spacraft that fit in LEO subject to certain assumptions.

    :param scen_properties: _description_
    :type scen_properties: _type_
    :param sep_dist_method: _description_
    :type sep_dist_method: _type_
    :param sep_angle: _description_, defaults to 0.2
    :type sep_angle: float, optional
    :param sep_dist: _description_, defaults to 25.0
    :type sep_dist: float, optional
    :param shell_sep: _description_, defaults to 5
    :type shell_sep: int, optional
    :param inc: _description_, defaults to 45.0
    :type inc: float, optional
    :param graph: _description_, defaults to False
    :type graph: bool, optional
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
        N_sat_eq = lambda h, c, b: (sep_dist / (scen_properties.re + h) / rad / c) ** (1 / b)
        shells_per_bin = np.diff(scen_properties.R0) / shell_sep
        N_sat = N_sat_eq(scen_properties.R0[1:], c_mat[ind_intrinsic], b_mat[ind_intrinsic]) * shells_per_bin

    slotted_sats_eqs = sp.Matrix([sp.zeros(scen_properties.n_shells, 1)])
    for species_group in scen_properties.species.values():
            for species in species_group:
                if species.slotted:
                    slotted_sats_eqs += species.sym
        
    N_sat = sp.Matrix(N_sat)
    unconsumed_intrinsic_capacity = N_sat - slotted_sats_eqs
    ind_struct = Indicators("unconsumed_intrinsic_capacity", "manual", None, unconsumed_intrinsic_capacity)

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
    It assumes that collision avoidance burden is divided evenly if two active species have a conjunction. Slotting effectivenss is not considered, bnut should be
    handled through inclusion of Gamma. 

    :param scen_properties: _description_
    :type scen_properties: _type_
    :param primary_species_list: _description_
    :type primary_species_list: _type_
    :param secondary_species_list: _description_
    :type secondary_species_list: _type_
    :param per_species: _description_, defaults to False
    :type per_species: bool, optional
    :param ind_name: _description_, defaults to ""
    :type ind_name: str, optional
    :param per_spacecraft: _description_, defaults to False
    :type per_spacecraft: bool, optional
    :return: _description_
    :rtype: _type_
    """
    primary_species_list_attribute = ""
    secondary_species_list_attribute = ""
    
    if isinstance(primary_species_list, str):
        primary_species_list_attribute = primary_species_list
        primary_species_list = []
        for species_group in scen_properties.species.values():
            for species in species_group:
                if primary_species_list_attribute == "all":
                    primary_species_list = scen_properties.species
                    break
                if getattr(species, primary_species_list_attribute):
                    primary_species_list.append(species)
                
    if isinstance(secondary_species_list, str):
        secondary_species_list_attribute = secondary_species_list
        secondary_species_list = []
        for species_group in scen_properties.species.values():
            for species in species_group:
                if secondary_species_list_attribute == "all":
                    secondary_species_list = scen_properties.species
                    break
                if getattr(species, secondary_species_list_attribute):
                    secondary_species_list.append(species)
                
    primary_species_names = [species.sym_name for species in primary_species_list]
    secondary_species_names = [species.sym_name for species in secondary_species_list]
    
    if ind_name == "":
        ind_name = "col_avoidance_maneuvers_pri_"
        if primary_species_list_attribute:
            ind_name += primary_species_list_attribute
        else:
            ind_name += '_'.join(primary_species_names)
        if secondary_species_list_attribute:
            ind_name += "_sec_" + secondary_species_list_attribute
        else:
            ind_name += "_sec_" + '_'.join(secondary_species_names)
        if per_spacecraft:
            ind_name += "_per_spacecraft"
    
    full_spec_list = primary_species_list + secondary_species_list
    
    ind_eqs = {}
    for species in full_spec_list:
        species_name = species.sym_name
        if species.maneuverable and species.active:
            ind_eqs[species_name] = sp.zeros(scen_properties.n_shells, 1)
    
    for primary_species in primary_species_list:
        primary_species_name = primary_species.sym_name
        for pair in scen_properties.collision_pairs:
            pair_primary_name = pair.species1.sym_name
            pair_secondary_name = pair.species2.sym_name
            s1_prim = primary_species_name == pair_primary_name
            s2_prim = primary_species_name == pair_secondary_name
            
            if s1_prim:
                sec_in_sec_list = pair_secondary_name in secondary_species_names
                secondary_species_name = pair_secondary_name
            if s2_prim:
                sec_in_sec_list = pair_primary_name in secondary_species_names
                secondary_species_name = pair_primary_name
                
            if (s1_prim or s2_prim) and sec_in_sec_list:
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
                        ind_eqs[primary_species_name] += 0.5 * (1 + pair.gammas[0]) * (1 / slotting_effectiveness) * intrinsic_collisions
                        ind_eqs[secondary_species_name] += 0.5 * (1 + pair.gammas[0]) * (1 / slotting_effectiveness) * intrinsic_collisions
                    else:
                        ind_eqs[primary_species_name] += 0.5 * (1 + pair.gammas[0]) * intrinsic_collisions
                        ind_eqs[secondary_species_name] += 0.5 * (1 + pair.gammas[0]) * intrinsic_collisions
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
            for species in scen_properties.species:
                if species.maneuverable:
                    ag_man_sat_totals += species.sym
            ag_man_counts /= ag_man_sat_totals
        ind_struct = Indicators(ind_name, "manual", None, ag_man_counts)
    else:
        ind_struct = []
        for species_name, eq in ind_eqs.items():
            if per_spacecraft:
                spec_index = next(i for i, species in enumerate(scen_properties.species) if species.sym_name == species_name)
                spec_man_indc = Indicators(f"{species_name}_maneuvers_per_spacecraft", "manual", None, eq / scen_properties.species[spec_index].sym)
            else:
                spec_man_indc = Indicators(f"{species_name}_maneuvers", "manual", None, eq)
            ind_struct.append(spec_man_indc)
    
    return ind_struct