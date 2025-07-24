from itertools import combinations
from sympy import symbols, Matrix
import numpy as np
from ..simulation.species_pair_class_new import SpeciesPairClass
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import math
import random
from utils.collisions.NASA_SBN6 import *
from utils.collisions.collisions_elliptical import evolve_bins

def process_species_pair(args):
    
    i, (s1, s2), scen_properties, debris_species, binE, LBgiven = args
    m1, m2 = s1.mass, s2.mass
    r1, r2 = s1.radius, s2.radius

    # print(f"{s1.sym_name} vs. {s2.sym_name}")

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

    # # Mass Binning
    binC_mass = np.zeros(len(debris_species))
    binE_mass = np.zeros(2 * len(debris_species))
    binW_mass = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC_mass[index] = debris.mass
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW_mass[index] = debris.mass_ub - debris.mass_lb

    binE_mass = np.unique(binE_mass)

    # Eccentricity Binning, multiple debris species will have the same eccentricity bins
    binE_ecc = scen_properties.eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    # binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # # Create bin edges starting at 0 and finishing at 1
    # binE_ecc = np.concatenate(([0], binE_ecc, [1]))
    binE_ecc = np.array(binE_ecc)
    if not np.all(np.diff(binE_ecc) > 0):
        raise ValueError("binE_ecc must be strictly increasing and contain no duplicates.")

    # Optionally sort it just to be safe
    binE_ecc = np.unique(binE_ecc)

    n_shells = scen_properties.n_shells
    n_mass_bins = len(binE_mass) - 1
    n_sma_bins = len(scen_properties.semi_major_bins_km) - 1
    n_ecc_bins = len(binE_ecc) - 1

    # Shape: [source_shell, mass_bin, sma_bin, ecc_bin]
    fragment_spread_totals = np.zeros(
        (n_shells, n_mass_bins, n_sma_bins, n_ecc_bins),
        dtype=np.float64
    )
    # This will tell you the number of fragments in each debris bin
    for shell in range(n_shells):
        dv1, dv2 = 10, 10 # for now we are going to assume the same velocity. This can change later. 

        # First need a representative semi-major axis
        sma1 = scen_properties.sma_HMid_km[shell]
        sma2 = sma1
        e1 = 0.01
        e2 = 0.02

        try:
            # Result is summed over: bins=[binE_sma, binE_mass, binE_ecc]
            result_3d = evolve_bins(scen_properties, m1, m2, r1, r2, sma1, sma2, e1, e2, 
                                        binE_mass, binE_ecc, shell, n_shells=scen_properties.n_shells, RBflag=RBflag)

            # To get just mass, sum everything on the second axis. 
            mass_distribution = np.sum(result_3d, axis=(0, 2))

            transpose = np.transpose(result_3d, (1, 0, 2))  # Transpose to [mass_bin, sma_bin, ecc_bin]

            fragment_spread_totals[shell, :, :, :] = transpose

            assert np.sum(mass_distribution) == np.sum(transpose), "Mass distribution should match the total fragments produced."
        except:
            print(f"no fragments produced for {m1, m2}")
            continue

        frags_made[shell, :] = mass_distribution

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

    return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, fragment_spread_totals)
        

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
    # This returns all possible combinations of the species
    species =  [species for species_group in scen_properties.species.values() for species in species_group]
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    species_pairs_classes = [] 
    # n_f = symbols('n_f:{0}'.format(scen_properties.n_shells))

    # Debris species - remember, we don't want PMD linked species. Just raw debris.
    # debris_species = [species for species in scen_properties.species['debris'] if not species.pmd_linked_species]
    debris_species = [species for species in scen_properties.species['debris']]

    # Calculate the Mass bin centres, edges and widths
    binC = np.zeros(len(debris_species))
    binE = np.zeros(2 * len(debris_species))
    binW = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC[index] = debris.mass
        binE[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW[index] = debris.mass_ub - debris.mass_lb

    binE = np.unique(binE)

    args = [(i, species_pair, scen_properties, debris_species, binE, LBgiven) for i, species_pair in enumerate(species_pairs)]
    
    # Use multiprocessing Pool for parallel processing
    if scen_properties.parallel_processing:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_species_pair, args), total=len(species_pairs), desc="Creating collision pairs"))
    else:
        results = [process_species_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    # Collect results
    species_pairs_classes.extend(results)

    return species_pairs_classes


if __name__ == "__main__":
    # Testing evolve_bins
    m1 = 1000
    m2 = 250
    r1 = 2
    r2 = 0.7
    dv = 10
    binE = np.array([1.4137200e-03, 2.8420686e-01, 1.3028350e+02, 3.6650000e+02,
       6.1150000e+02, 1.0000000e+05])
    R02 = np.arange(200, 2050, 50)
    nums, is_catastrophic, bin_out, alt_nums = evolve_bins(m1, m2, r1, r2, dv, [], binE, [], 0.1, RBflag=0, fragment_spreading=True, n_shells=10, R02=R02)

    range_values = range(-(len(alt_nums)//2), len(alt_nums)//2)

    # Check lengths of range_values and alt_nums
    print("Length of range_values:", len(range_values))
    print("Shape of alt_nums:", alt_nums.shape)

    # Plot the stacked bar chart
    plt.figure()
    for i in range(alt_nums.shape[1]):
        if i == 0:
            plt.bar(range_values, alt_nums[:, i], label=f'{i}', alpha=0.6)
        else:
            plt.bar(range_values, alt_nums[:, i], bottom=np.sum(alt_nums[:, :i], axis=1), label=f'{i}', alpha=0.6)

    plt.legend(title='Bin Edges')
    plt.xlabel('Shell offset')
    plt.ylabel('Count')
    plt.show()