import numpy as np
from itertools import combinations
import multiprocessing as mp
from sympy import symbols, Matrix
from tqdm import tqdm
import traceback
from utils.collisions.NASA_SBM_Evolve import evolve_bins_circular, evolve_bins_elliptical
from utils.simulation.species_pair_class import SpeciesPairClass

def process_species_pair(args):    
    i, (s1, s2), scen_properties, debris_species, binE_mass, LBgiven = args
    m1, m2 = s1.mass, s2.mass
    r1, r2 = s1.radius, s2.radius

    # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
    gammas = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

    # Create a list of source sinks, first two are the active species
    source_sinks = [s1, s2]

    # Implementing logic for gammas calculations based on species properties
    if s1.maneuverable and s2.maneuverable:
        # Multiplying each element in the first column of gammas by the product of alpha_active values
        gammas[:, 0] = gammas[:, 0] * s1.alpha_active * s2.alpha_active
        if s1.slotted and s2.slotted:
            # Applying the minimum slotting effectiveness if both are slotted
            gammas[:, 0] = gammas[:, 0] * min(s1.slotting_effectiveness, s2.slotting_effectiveness)

    elif (s1.maneuverable and not s2.maneuverable) or (s2.maneuverable and not s1.maneuverable):
        if s1.trackable and s2.maneuverable:
            gammas[:, 0] = gammas[:, 0] * s2.alpha
        elif s2.trackable and s1.maneuverable:
            gammas[:, 0] = gammas[:, 0] * s1.alpha

    # Applying symmetric loss to both colliding species
    gammas[:, 1] = gammas[:, 0]

    # Rocket Body Flag - 1: RB; 0: not RB
    # Will be 0 if both are None type
    if s1.RBflag is None and s2.RBflag is None:
        RBflag = 0
    elif s1.RBflag is None:
        RBflag = s2.RBflag
    elif s2.RBflag is None:
        RBflag = s1.RBflag
    else:
        RBflag = max(s1.RBflag, s2.RBflag)

    # Calculate the number of fragments made for each debris species
    frags_made = np.zeros((len(scen_properties.v_imp2), len(debris_species)))
    alt_nums = np.zeros((scen_properties.n_shells * 2, len(debris_species)))


    if scen_properties.elliptical:
        #########
        # Elliptical Collisions. Uses full EVolve 4.0. 
        #########
        # Requires binning of eccentricity bins too
        binE_ecc = np.array(np.sort(scen_properties.eccentricity_bins))
        if not np.all(np.diff(binE_ecc) > 0):
            raise ValueError("binE_ecc must be strictly increasing and contain no duplicates.")   
        binE_ecc = np.unique(binE_ecc)
        n_mass_bins = len(binE_mass) - 1
        n_sma_bins = len(scen_properties.semi_major_bins_km) - 1
        n_ecc_bins = len(binE_ecc) - 1
        # Shape: [source_shell, mass_bin, sma_bin, ecc_bin]
        fragment_spread_totals = np.zeros(
            (scen_properties.n_shells, n_mass_bins, n_sma_bins, n_ecc_bins),
            dtype=np.float64
        )         
        for shell in range(scen_properties.n_shells):
            # First need a representative semi-major axis
            sma1 = scen_properties.sma_HMid_km[shell]
            sma2 = sma1
            e1 = 0.01
            e2 = 0.02

            try:
                # Result is summed over: bins=[binE_sma, binE_mass, binE_ecc]
                result_3d = evolve_bins_elliptical(scen_properties, m1, m2, r1, r2, sma1, sma2, e1, e2, 
                                            binE_mass, binE_ecc, shell, n_shells=scen_properties.n_shells, RBflag=RBflag)

                # To get just mass, sum everything on the second axis. 
                mass_distribution = np.sum(result_3d, axis=(0, 2))

                transpose = np.transpose(result_3d, (1, 0, 2))  # Transpose to [mass_bin, sma_bin, ecc_bin]

                fragment_spread_totals[shell, :, :, :] = transpose

                assert np.sum(mass_distribution) == np.sum(transpose), "Mass distribution should match the total fragments produced."
            except Exception:
                print(f"no fragments produced for {m1, m2}")
                print(Exception)
                continue

            frags_made[shell, :] = mass_distribution 

    elif scen_properties.fragment_spreading:
        #########
        # Fragment spreading
        #########
        for dv_index, dv in enumerate(scen_properties.v_imp2): # This is the case for circular orbits
            dv1, dv2 = 10, 10 # for now we are going to assume the same velocity. 
            try:
                results = evolve_bins_circular(m1, m2, r1, r2, dv1, dv2, [], binE_mass, [], LBgiven, RBflag, source_sinks, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km)
                frags_made[dv_index, :] = results[0] # nums is the number of fragments related to the shell of dv_index (same shell)
                alt_nums = results[3] # Is the additional term from the spreading of the collision (all other shells)
            except IndexError as ie:
                alt_nums = None
                continue
            except ValueError as e:
                    continue
    
    else:
        #########
        # Basic SSEM 
        #########
        for dv_index, dv in enumerate(scen_properties.v_imp2): # This is the case for circular orbits 
            dv1, dv2 = 10, 10 # for now we are going to assume the same velocity. 
            try:
                results = evolve_bins_circular(m1, m2, r1, r2, dv1, dv2, [], binE_mass, [], LBgiven, RBflag, source_sinks, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km)
                frags_made[dv_index, :] = results[0]
            except IndexError as ie:
                alt_nums = None
                continue
            except ValueError as e:
                    continue

    
    ## Create the symbolic matrix - this should be the same for each. 
    for i, species in enumerate(debris_species):
        frags_made_sym = Matrix(frags_made[:, i]) 

        # Multiply it by the likelihood of collision (gammas) to get the number of fragments made for each shell
        new_column = -gammas[:, 1].multiply_elementwise(frags_made_sym)
        new_column = new_column.reshape(gammas.rows, 1)  # Ensure it's a column vector

        # Use col_insert to add the new column. Insert at index 2+i
        gammas = gammas.col_insert(2 + i, new_column)

        if 2 + i < len(source_sinks):
            source_sinks[2 + i] = species
        else:
            source_sinks.append(species)

    if scen_properties.elliptical:
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, fragment_spread_totals=fragment_spread_totals, model_type='elliptical')
    if scen_properties.fragment_spreading:
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, fragsMadeDV=alt_nums, model_type='fragment_spreading')
    else:
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, model_type='baseline')

def create_collision_pairs(scen_properties):
    """
    Function takes a scen_properties object with a list of species and the same species organised in species_cells into 
    archetypical categories. It calculate and creates a set of species_pair objects which is stored at the scen_properties
    object and used to compile collision equations during the model building process.

    The model is aware of trackability, maneuverability and slotting. Object fragmentation counts are based on the NASA 
    Standard Breakup model. 

    :param scen_properties: ScenarioProperties object
    :type scen_properties: ScenarioProperties
    """
    
    # Get the binomial coefficient of the species
    species =  [species for species_group in scen_properties.species.values() for species in species_group]
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    species_pairs_classes = [] 

    # Debris species - remember, we don't want PMD linked species. Just raw debris.
    # debris_species = [species for species in scen_properties.species['debris'] if not species.pmd_linked_species]
    debris_species = [species for species in scen_properties.species['debris']]

    # Calculate the Mass bin edges
    binE_mass = np.zeros(2 * len(debris_species))
    LBgiven = scen_properties.LC
    for index, debris in enumerate(debris_species):
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
    binE_mass = np.unique(binE_mass)

    args = [(i, species_pair, scen_properties, debris_species, binE_mass, LBgiven) for i, species_pair in enumerate(species_pairs)]

    if scen_properties.elliptical:
        if scen_properties.parallel_processing:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(process_species_pair, args), total=len(species_pairs), desc="Creating collision pairs"))
        else:
            results = [process_species_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    else:
        if scen_properties.parallel_processing:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(process_species_pair, args), total=len(species_pairs), desc="Creating collision pairs"))
        else:
            results = [process_species_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    # Collect results
    species_pairs_classes.extend(results)

    return species_pairs_classes

