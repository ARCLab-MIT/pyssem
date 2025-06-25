from utils.collisions.collisions import func_Am, func_dv
from utils.collisions.NASA_SBM_frags import frag_col_SBM_vec_lc2
from utils.collisions.NASA_SBN6 import *
import numpy as np
from tqdm import tqdm
from utils.collisions.cartesian_to_kep import cart_2_kep, kep_2_cart
from concurrent.futures import ProcessPoolExecutor, as_completed
from sympy import symbols, Matrix, pi, S, Expr, zeros
import matplotlib.pyplot as plt
import re
from itertools import combinations
import math
import traceback
from line_profiler import profile

def create_elliptical_collision_pairs(scen_properties):

    all_species = [
        sp for grp in scen_properties.species.values() for sp in grp
    ]
    pairs = list(combinations(all_species, 2)) + [(sp, sp) for sp in all_species]

    collision_pairs = []
    LC = scen_properties.LC
    n_shells = scen_properties.n_shells

    for sp1, sp2 in pairs:
        # skip purely maneuverable–maneuverable
        if sp1.maneuverable and sp2.maneuverable:
            continue

        cp = SpeciesCollisionPair(sp1, sp2, scen_properties)

        # figure out how many ecc bins each has
        n_e1 = sp1.time_per_shells.shape[1]
        n_e2 = sp2.time_per_shells.shape[1]

        for i_sma in range(n_shells):
            for j_e1 in range(n_e1):
                for j_e2 in range(n_e2):
                    for k_shell in range(n_shells):
                        t1 = sp1.time_per_shells[i_sma, j_e1, k_shell]
                        t2 = sp2.time_per_shells[i_sma, j_e2, k_shell]

                        if t1 > 0 and t2 > 0:
                            # skip sub‐LC collisions
                            if sp1.mass < LC and sp2.mass < LC:
                                continue
                            cp.collision_pair_by_shell.append(
                                EllipticalCollisionPair(
                                    species1=sp1,
                                    species2=sp2,
                                    sma_index=i_sma,
                                    ecc1_index=j_e1,
                                    ecc2_index=j_e2,
                                    shell_index=k_shell
                                )
                            )

        collision_pairs.append(cp)
    
    print(f"Total number of unique species pairs: {len(collision_pairs)}")
    # loop through each species pair and then sum the collision pairs
    count = 0
    for species_pair in collision_pairs:
        count += len(species_pair.collision_pair_by_shell)

    print(f"Total number of collision pairs: {count}")

    debris_species = [species for species in scen_properties.species['debris']]

    # Mass Binning
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
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # Create bin edges starting at 0 and finishing at 1
    binE_ecc = np.concatenate(([0], binE_ecc, [1]))

    debris_species = scen_properties.species['debris']
    D = len(debris_species)
    S = scen_properties.n_shells
    E = len(binE_ecc) - 1

    for i, sp_pair in tqdm(enumerate(collision_pairs),
                        total=len(collision_pairs),
                        desc="Processing species pairs"):

        # 1) initialize per-pair accumulators
        sp_pair.debris_per_shell_species = np.zeros((S, D), dtype=float)
        sp_pair.eccumul_full            = np.zeros((S, E), dtype=float)
        sp_pair.collision_processed     = []

        # 2) loop over every shell_collision in this pair
        for shell_col in sp_pair.collision_pair_by_shell:
            gamma = process_elliptical_collision_pair(
                (i, shell_col, scen_properties,
                debris_species, binE_mass, binE_ecc, LBgiven)
            )

            # keep the raw fragments 3D array on gamma
            # gamma.fragments.shape == (S, D, E)
            if gamma.fragments is None:
                # no fragments generated, skip this collision
                continue

        #     # collapse over ecc → (S, D)
        #     sp_pair.debris_per_shell_species += gamma.fragments
        #     # collapse over species → (S, E)
        #     sp_pair.eccumul_full += gamma.ecc_matrix
        #     sp_pair.collision_processed.append(gamma)

        # # 3) build the normalized ecc distribution across all shells
        # ecc_totals = sp_pair.eccumul_full.sum(axis=0)  # shape (E,)
        # total     = ecc_totals.sum()
        # if total > 0:
        #     sp_pair.ecc_distribution = ecc_totals / total
        # else:
        #     sp_pair.ecc_distribution = np.zeros_like(ecc_totals)

    return 


def create_elliptical_collision_pairs2(scen_properties):
    """
    This function will take each species that is elliptical, then it will search across all other ellitpical objects and shells to assess 
    which other objects it could collide with. 

    This function relies on two main classes:
    - SpeciesCollisionPair: This is the parent houser for all objects that could collide
    - EllipticalCollisionPair: This is the individual shell and eccentricity pairing of all possible outcomes. 
    
    """

    ########################################
    # Create the species pairs and Elliptical Collision Pairs
    # Species Pairs will be the unique number of species in the simulation that could collide
    # Elliptical Collision Pairs will be the unique number of species pairs that could collide in each shell
    ########################################
    all_elliptical_collision_species = []

    species =  [species for species_group in scen_properties.species.values() for species in species_group]
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    species_pairs_classes = [] 

    # loop through each of the species_pairs and create a SpeciesCollisionPair
    collision_pairs_unique = []
    for species_pair in species_pairs:
        collision_pairs_unique.append(SpeciesCollisionPair(species_pair[0], species_pair[1], scen_properties))

    # for species_group in scen_properties.species.values():
    #     count = 0
    #     # For every pair of species in the same group
    #     for i, species1 in enumerate(species_group):
    #         for j, species2 in enumerate(species_group):
                
    #             collision_pair_unique = SpeciesCollisionPair(species1, species2, scen_properties)

    for collision_pair in collision_pairs_unique:
        # Make some initial checks to save computation time for now, we will have to back to this later
        species1 = collision_pair.species1
        species2 = collision_pair.species2

        if species1.maneuverable and species2.maneuverable:
            continue

        # Loop through the semi-major axis bins (assuming species1 and species2 have the same number of bins)
        for k in range(len(species1.semi_major_axis_bins)):
            # Extract the time spent in shells for both species at this semi-major axis bin
            time_in_shells_1 = species1.time_per_shells[k]  # Array for species1
            time_in_shells_2 = species2.time_per_shells[k]  # Array for species2

            # Loop through the shells and check if both species spend time in the same shell
            for shell_index in range(len(time_in_shells_1)):
                time_1 = time_in_shells_1[shell_index]
                time_2 = time_in_shells_2[shell_index]

                # Only print if both species spend time in this shell
                if time_1 > 0 and time_2 > 0:
                    
                    # There is also a check on the mass of the objects, if they are both too small then they will just create dust
                    # this is defined by the LC. LC is the diameter of the smallest object that can be tracked.
                    if species1.mass < scen_properties.LC and species2.mass < scen_properties.LC:
                        continue
                            
                    collision_pair.collision_pair_by_shell.append(EllipticalCollisionPair(species1, species2, shell_index))
        
        all_elliptical_collision_species.append(collision_pair)

    print(f"Total number of u`nique species pairs: {len(all_elliptical_collision_species)}")
    # loop through each species pair and then sum the collision pairs
    count = 0
    for species_pair in all_elliptical_collision_species:
        count += len(species_pair.collision_pair_by_shell)

    print(f"Total number of collision pairs: {count}")

    ########################################
    # Fragmentation Code
    # This code will calculate the number of fragments that are generaed if the two objects were to collide at each shell. 
    ########################################

    debris_species = [species for species in scen_properties.species['debris']]

    # Mass Binning
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
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # Create bin edges starting at 0 and finishing at 1
    binE_ecc = np.concatenate(([0], binE_ecc, [1]))

    # Modify the args creation to loop over each species pair and its nested shell collisions
    for i, species_pair in tqdm(enumerate(all_elliptical_collision_species), total=len(all_elliptical_collision_species), desc="Processing species pairs"):
        for shell_collision in species_pair.collision_pair_by_shell:
            # Process each shell-specific collision and append result to collision_processed list
            gamma = process_elliptical_collision_pair((i, shell_collision, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven))
            species_pair.collision_processed.append(gamma)

    # find the unique mass - this should be done in the scenario properties
    debris_species_names = [species.sym_name for species in scen_properties.species['debris']]
    pattern = re.compile(r'^N_[^_]+kg')
    unique_debris_names = set()

    for name in debris_species_names:
        match = pattern.match(name)
        if match:
            unique_debris_names.add(match.group())
    unique_debris_names = list(unique_debris_names)

    return generate_collision_equations(all_elliptical_collision_species, scen_properties, mass_bins=unique_debris_names, ecc_bins=scen_properties.species['debris'][0].eccentricity_bins)

    # all_elliptical_collision_species = parallel_collision_processing(
    #     all_elliptical_collision_species, scen_properties, unique_debris_names, ecc_bins, debris_species_names
    # )

    return all_elliptical_collision_species
def is_catastrophic(mass1, mass2, vels):
    """
    Determines if a collision is catastropic or non-catastrophic by calculating the 
    relative kinetic energy. If the energy is greater than 40 J/g, the collision is
    catastrophic from Johnson et al. 2001 (Collision Section)

    Args:
        mass1 (float): mass of species 1, kg
        mass2 (float): mass of species 2, kg
        vels (np.ndarray): array of the relative velocities (km/s) for each shell
    
    Returns:
        shell-wise list of bools (true if catastrophic, false if not catastrophic)
    """

    if mass1 <= mass2:
        smaller_mass = mass1
    else:
        smaller_mass = mass2
    
    smaller_mass_g = smaller_mass * (1000) # kg to g
    energy = [0.5 * smaller_mass * (v)**2 for v in vels] # Need to also convert km/s to m/s
    is_catastrophic = [True if e/smaller_mass_g > 40 else False for e in energy]

    return is_catastrophic

def generate_collision_equations(all_elliptical_collision_species, scen_properties, mass_bins, ecc_bins):
    n_shells = scen_properties.n_shells

    # Initialize debris species lists
    debris_species_names = []
    debris_symbolic_vars = []

    for s in range(n_shells):
        for mass_value in mass_bins:
            for ecc_value in ecc_bins:
                debris_species_name = f'{mass_value}_e{ecc_value}_{s + 1}'
                debris_species_names.append(debris_species_name)
                debris_var = symbols(debris_species_name)
                debris_symbolic_vars.append(debris_var)

    # Ensure that debris species variables are included in 'all_symbolic_vars'
    all_symbolic_vars = scen_properties.all_symbolic_vars

    # Rebuild the mappings to include the debris species
    species_name_to_idx = {str(var): idx for idx, var in enumerate(all_symbolic_vars)}
    species_sym_vars = {str(var): var for var in all_symbolic_vars}

    # Initialize the equations matrix
    n_species = len(all_symbolic_vars)
    equations = Matrix.zeros(n_shells, n_species)

    phi_matrix = None

    for collision_pair in tqdm(all_elliptical_collision_species, total=len(all_elliptical_collision_species), desc="Generating collision equations"):

        if len(collision_pair.collision_pair_by_shell) == 0:
            # If no fragments, exit with a zero matrix
            collision_pair.eqs = Matrix.zeros(n_shells, scen_properties.species_length)
            continue
        
        # Map out the major components
        species1 = collision_pair.species1
        species2 = collision_pair.species2
        species1_name = species1.sym_name
        species2_name = species2.sym_name

        # Conversion factor
        meter_to_km = 1 / 1000

        # Compute collision cross-section (sigma)
        collision_pair.sigma = (species1.radius * meter_to_km + species2.radius * meter_to_km) ** 2

        # Compute collision rate (phi) for each shell
        collision_pair.phi = (
            pi * scen_properties.v_imp2
            / (scen_properties.V * meter_to_km**3)
            * collision_pair.sigma
            * S(86400)
            * S(365.25)
        )
        phi_matrix = Matrix(collision_pair.phi)

        # If phi_matrix is a scalar, convert it into a 1x1 matrix
        if phi_matrix.shape == ():  # Single scalar case
            phi_matrix = Matrix([phi_matrix])  # Convert scalar to 1x1 matrix
        else:
            # If phi_matrix is a 1D row or flat matrix, reshape it into a column vector (n, 1)
            phi_matrix = phi_matrix.reshape(len(phi_matrix), 1) 

        for elliptical_pair in collision_pair.collision_pair_by_shell:

            elliptical_pair.gamma = collision_pair.gamma.copy() 

            s_source = elliptical_pair.shell_index  # Source shell index (0-based)

            # Get species variable names including shell number
            species1_var_name = f'{species1_name}_{s_source + 1}'
            species2_var_name = f'{species2_name}_{s_source + 1}'

            # Get symbolic variables
            N_species1_s = species_sym_vars.get(species1_var_name)
            N_species2_s = species_sym_vars.get(species2_var_name)

            if N_species1_s is None or N_species2_s is None:
                continue  # Skip if species variables are not found

            phi_s = collision_pair.phi[s_source]

            # Process fragments from the collision
            fragments = elliptical_pair.fragments  # Should be an array of shape [n_destination_shells, n_mass_bins, n_ecc_bins]

            if fragments is not None:
                n_destination_shells, n_mass_bins, n_ecc_bins = fragments.shape

                for s_destination in range(n_destination_shells):
                    fragments_sd = fragments[s_destination]  # Fragments ending up in shell s_destination
                    for mass_bin_index in range(n_mass_bins):
                        for ecc_bin_index in range(n_ecc_bins):
                            num_frags = fragments_sd[mass_bin_index, ecc_bin_index]
                            if num_frags != 0:
                                mass_value = mass_bins[mass_bin_index]
                                ecc_value = ecc_bins[ecc_bin_index]
                                # Generate debris species variable name including destination shell number
                                debris_species_name = f'{mass_value}_e{ecc_value}_{s_destination + 1}'
                                idx_debris = species_name_to_idx.get(debris_species_name)
                                if idx_debris is not None:
                                    debris_var = species_sym_vars[debris_species_name]
                                    # Compute delta gain for debris species (symbolic)
                                    delta_gain = phi_s * N_species1_s * N_species2_s * num_frags
                                    equations[s_destination, idx_debris] += delta_gain  
            else:
                continue
        
        # Now, extract the equations for the debris species and store them in a (10, 20) matrix
        # Initialize the debris equations matrix
        debris_length = len(mass_bins) * len(ecc_bins) 
        equations_debris = Matrix.zeros(n_shells, debris_length)

        # Map debris species variable names to indices within the shell
        debris_species_idx_within_shell = {}
        for idx_within_shell, debris_species_name in enumerate(debris_species_names[:debris_length]):
            base_name = '_'.join(debris_species_name.split('_')[:-1])
            debris_species_idx_within_shell[base_name] = idx_within_shell

        # Populate the debris equations matrix
        for s in range(n_shells):
            for mass_value in mass_bins:
                for ecc_value in ecc_bins:
                    debris_species_name = f'{mass_value}_e{ecc_value}_{s + 1}'
                    base_name = f'{mass_value}_e{ecc_value}'
                    idx_species = species_name_to_idx.get(debris_species_name)
                    idx_debris = debris_species_idx_within_shell.get(base_name)
                    if idx_species is not None and idx_debris is not None:
                        eq = equations[s, idx_species]
                        equations_debris[s, idx_debris] += eq

        species_names = scen_properties.species_names
        species1_idx = species_names.index(species1.sym_name)
        species2_idx = species_names.index(species2.sym_name)
        # print(species1_idx, species2_idx)
        # find the start of the debris species index, which will be the first item in species_names that starts with 'N_'
        debris_start_idx = next(i for i, name in enumerate(species_names) if name.startswith('N_'))
        # print(debris_start_idx)

        try:
            eq_s1 = collision_pair.gamma[:, 1].multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
            eq_s2 = collision_pair.gamma[:, 1].multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
        except Exception as e:
            print(f"Exception caught: {e}")
            print("Error in multiplying gammas, check that each component is a column vector and correct shape.")
            
        eqs = Matrix(zeros(n_shells, scen_properties.species_length))

        # # add in eq_1 at species1_idx and eq_2 at species2_idx
        eqs[:, species1_idx] = eq_s1
        eqs[:, species2_idx] += eq_s2

        # Loop through each debris species
        for i in range(len(scen_properties.species['debris'])):
            # Calculate the corresponding index in the overall species list
            deb_index = debris_start_idx + i
            # Assign the columns from equations_debris to the appropriate columns in eqs
            eqs[:, deb_index] = equations_debris[:, i]

        collision_pair.eqs = eqs
        
    return all_elliptical_collision_species

def process_elliptical_collision_pair_new(args):
    i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius

    t1 = collision_pair.species1.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    t2 = collision_pair.species2.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    min_TIS = min(t1, t2)
    prod_TIS = t1 * t2

    return

def perifocal_r_and_v(a, e, nu, mu):
    r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
    r = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

    h = np.sqrt(mu * a * (1 - e**2))
    v = (mu / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    return r, v

def rotate_vector_45_deg_in_plane(v):
    theta = np.pi / 4  # 45 degrees
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return rot_matrix @ v

@profile
def process_elliptical_collision_pair(args):
    i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args

    sp1 = collision_pair.species1
    sp2 = collision_pair.species2
    r1 = collision_pair.species1.radius
    r2 = collision_pair.species2.radius
    sma1 = collision_pair.species1_sma
    sma2 = collision_pair.species2_sma

    # pull out the sma, ecc1, ecc2 and shell indices
    i_sma    = collision_pair.sma_index
    j_e1     = collision_pair.ecc1_index
    j_e2     = collision_pair.ecc2_index
    k_shell  = collision_pair.shell_index

    if collision_pair.species1.eccentricity_bins is not None:
        e1 = collision_pair.species1.eccentricity_bins[j_e1]
    else: 
        e1 = 0
    
    if collision_pair.species2.eccentricity_bins is not None:
        e2 = collision_pair.species2.eccentricity_bins[j_e2]
    else:
        e2 = 0
    # instantaneous velocities in that exact (sma,ecc) cell
    v1 = sp1.velocity_per_shells[i_sma, j_e1, k_shell]
    v2 = sp2.velocity_per_shells[i_sma, j_e2, k_shell]

    # time‐in‐shell fractions in that same cell
    t1 = sp1.time_per_shells[i_sma, j_e1, k_shell]
    t2 = sp2.time_per_shells[i_sma, j_e2, k_shell]

    # simple “overlap” metrics
    min_TIS  = min(t1, t2)
    prod_TIS = t1 * t2

    # now you can call your SBM with m1,m2,v_rel
    m1, m2 = sp1.mass, sp2.mass

    # if m2 is bigger than m1, swap mass radius, sma and ecc
    if m2 > m1:
        m1, m2 = m2, m1
        r1, r2 = r2, r1
        sma1, sma2 = sma2, sma1
        e1, e2 = e2, e1
        v1, v2 = v2, v1

    # This will have to change 
    try:
        if m1 < 1 or m2 < 1:
            fragments = None
        else:
            fragments = evolve_bins(scen_properties, m1, m2, r1, r2, sma1, sma2, e1, e2, 
                                    binE_mass, binE_ecc, collision_pair.shell_index, n_shells=scen_properties.n_shells)    
    except Exception as e:
        print("Error in Evolve Bins")
        print(f"Exception caught: {e}")
        fragments = None

    if fragments is not None:
        collision_pair.fragments = fragments * prod_TIS
        collision_pair.ecc_matrix = np.array([])#ecc_mat
    else: 
        # print(f'No fragments generated between species {m1} and {m2} in shell {collision_pair.shell_index}')
        collision_pair.fragments = fragments

    return collision_pair

def func_de(R_list, V_list):
    mu = 398600.4418  # km^3/s^2
    eccentricities = []

    for R, V in zip(R_list, V_list):    
        # Calculate specific angular momentum vector h = R x V
        V = np.abs(V)
        h = np.cross(R, V)
        
        # Calculate eccentricity vector e
        e_vector = (np.cross(V, h) / mu) - (R / np.linalg.norm(R))
        
        # Calculate the magnitude of the eccentricity vector
        e = np.linalg.norm(e_vector)
        
        # Append the eccentricity to the list
        eccentricities.append(e)
    
    # Return the list of eccentricities
    return eccentricities

@profile
def evolve_bins(scen_properties, m1, m2, rad_1, rad_2, sma1, sma2, e1, e2, binE_mass, binE_ecc, collision_index, n_shells=0):
    param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }

    # # Lower bound (LB)
    LB = 0.1
    SS = 20
    earth_radius_km = 6371
    true_anomaly_deg = 90
    TA = np.radians(true_anomaly_deg)
    mu = param["mu"]

    # try:
    # Object 1 (larger mass): low eccentricity
    r1, v1 = perifocal_r_and_v(sma1, e2, TA, mu)

    # Object 2 (lighter mass): higher eccentricity, rotated velocity
    _, v2_mag_vec = perifocal_r_and_v(sma2, e1, TA, mu)
    v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
    v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
    r2 = r1  # collision point is same

    # Define input vectors
    p1 = np.array([m1, rad_1, *r1, *v1, 1.0])
    p2 = np.array([m2, rad_2, *r2, *v2, 1.0])

    try:
        debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1, p2, LB=LB)
        # debris will now come out in the format of [a, ecco, mass]
    except Exception as e:
        print(f"Error in frag_col_SBM_vec_lc2: {e} \n for m1={m1}, m2={m2}, r1={rad_1}, r2={rad_2}, sma1={sma1}, sma2={sma2}, e1={e1}, e2={e2}")
        traceback.print_exc()
        return None

    if debris1.size == 0:
        print(f"m1={m1}, m2={m2}, r1={rad_1}, r2={rad_2}, sma1={sma1}, sma2={sma2}, No debris generated")
        print(len(debris1), len(debris2))

    # Loop through 
    frag_a = []
    frag_e = []
    frag_mass = []

    for debris in debris1:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue # decayed

        frag_a.append((norm_earth_radius - 1) * 6371 + 6371) 
        frag_e.append(debris[1])
        frag_mass.append(debris[2])
    
    for debris in debris2:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue # decayed

        frag_a.append((norm_earth_radius - 1) * 6371 + 6371) 
        frag_e.append(debris[1])
        frag_mass.append(debris[2])

    frag_properties = np.array([frag_a, frag_mass, frag_e]).T

    binE_alt = scen_properties.R0_rad_km  # We add 1 for bin edges

    # hist, edges = np.histogramdd(frag_properties, bins=bins)

    # hist = hist / (SS * 3)

    # return hist
    hist3d, _ = np.histogramdd(
        frag_properties,
        bins=[binE_alt, binE_mass, binE_ecc]
    )

    # normalize per your SS factor
    hist3d /= (SS * 3)

    # collapse into the two 2D matrices:
    # debris_matrix = hist3d.sum(axis=2)  # shape (n_shells, n_mass_species)
    # ecc_matrix    = hist3d.sum(axis=1)  # shape (n_shells, n_ecc_bins)

    # return debris_matrix, ecc_matrix
    return hist3d




# def evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_index, n_shells=0):
        
#         # Need to now follow the NASA SBM route, first we need to create p1_in and p2_in
#         #  Parameters:
#         # - ep: Epoch
#         # - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#         # - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#         # - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
#         # - LB: Lower bound for fragment sizes (meters)
#         # Super sampling ratio
#         SS = 20
#         R_EARTH = 6371.0  # km
#         theta = np.radians(45)  # Rotation angle

#         # Get collision altitude and position magnitude
#         collision_altitude = scen_properties.HMid[collision_index]
#         r_mag = R_EARTH + collision_altitude

#         # Define position vectors with slight z offsets
#         r1_vec = r_mag * np.array([np.cos(theta), np.sin(theta), 0.01])
#         r2_vec = r_mag * np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2), -0.01])

#         # Define velocity vectors with orthogonal direction and small z component
#         v_half = v1 / 2
#         v1_vec = v_half * np.array([-np.sin(theta), np.cos(theta), 0.02])
#         v2_vec = v_half * np.array([np.sin(theta), -np.cos(theta), -0.02])

#         # Compose p1 and p2 inputs
#         p1_in = np.array([m1, r1, *r1_vec, *v1_vec, 1]) # 1 is the object class (dimensionless)
#         p2_in = np.array([m2, r2, *r2_vec, *v2_vec, 1])

#         param = {
#             'req': 6.3781e+03,
#             'mu': 3.9860e+05,
#             'j2': 0.0011,
#             'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
#             'maxID': 0,
#             'density_profile': 'static'
#         }
        
#         altitude = scen_properties.HMid[collision_index] 
#         earth_radius = 6371  # Earth's mean radius in km
#         latitude_deg = 45  # in degrees
#         longitude_deg = 60  # in degrees

#         # Convert degrees to radians
#         latitude_rad = math.radians(latitude_deg)
#         longitude_rad = math.radians(longitude_deg)

#         # Compute the radial distance from Earth's center
#         r = earth_radius + altitude

#         # Calculate the position vector in ECEF coordinates
#         x = r * math.cos(latitude_rad) * math.cos(longitude_rad)
#         y = r * math.cos(latitude_rad) * math.sin(longitude_rad)
#         z = r * math.sin(latitude_rad)

#         # Return the position vector
#         x, y, z

#         # up to correct mass too
#         if m1 < m2:
#             m1, m2 = m2, m1
#             r1, r2 = r2, r1

#         p1_in[0], p2_in[0] = m1, m2 
#         p1_in[1], p2_in[1] = r1, r2

#         # remove a from r_x from both p1_in and p2_in
#         # the initial norm is 1000, so we need to remove the difference
#         p1_in[2], p1_in[3], p1_in[4] = x, y, z 
#         p2_in[2], p2_in[3], p2_in[4] = x, y, z
            
#         LB = 0.1

#         try:
#             debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB)
#         except Exception as e:
#             print("Error in frag_col_SBM_vec_lc2")
#             print(f"Exception caught: {e}")
#             return None
#         # print(len(debris1), len(debris2))

#         # Loop through 
#         frag_a = []
#         frag_e = []
#         frag_mass = []

#         for debris in debris1:
#             norm_earth_radius = debris[0]
#             if norm_earth_radius < 1:
#                 continue # decayed

#             frag_a.append((norm_earth_radius - 1) * 6371) 
#             frag_e.append(debris[1])
#             frag_mass.append(debris[7])
        
#         for debris in debris2:
#             norm_earth_radius = debris[0]
#             if norm_earth_radius < 1:
#                 continue # decayed

#             frag_a.append((norm_earth_radius - 1) * 6371) 
#             frag_e.append(debris[1])
#             frag_mass.append(debris[7])

#         frag_properties = np.array([frag_a, frag_mass, frag_e]).T

#         binE_alt = np.linspace(scen_properties.min_altitude, scen_properties.max_altitude, n_shells + 1)  # We add 1 for bin edges

#         bins = [binE_alt, binE_mass, binE_ecc]

#         hist, edges = np.histogramdd(frag_properties, bins=bins)

#         hist = hist / (SS * 3)

#         return hist

def perifocal_r_and_v(a, e, nu, mu):
            r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
            r = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

            h = np.sqrt(mu * a * (1 - e**2))
            v = (mu / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
            return r, v

def rotate_vector_45_deg_in_plane(v):
    theta = np.pi / 4  # 45 degrees
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return rot_matrix @ v

# def evolve_bins(scen_properties, m1, m2, radius_1, radius_2, v1, v2, ecc_1, ecc_2, binE_mass, binE_ecc, collision_index, n_shells=0):  
#         # Need to now follow the NASA SBM route, first we need to create p1_in and p2_in
#         #  Parameters:
#         # - ep: Epoch
#         # - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#         # - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#         # - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
#         # - LB: Lower bound for fragment sizes (meters)
#         # Super sampling ratio
#         SS = 20
#         R_EARTH = 6371.0  # km
#         true_anomaly_deg = 90
#         TA = np.radians(true_anomaly_deg)
#         param = {
#             'req': 6.3781e+03,
#             'mu': 3.9860e+05,
#             'j2': 0.0011,
#             'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
#             'maxID': 0,
#             'density_profile': 'static'
#         }

#         results = []
#         collision_altitude = scen_properties.HMid[collision_index] + R_EARTH  # Convert altitude to radius in km
        
#         # Calculate position and velocity vectors in perifocal coordinates
#         # Object 1 (larger mass): low eccentricity
#         r1, v1 = perifocal_r_and_v(collision_altitude, ecc_2, TA, param["mu"])

#          # Object 2 (lighter mass): higher eccentricity, rotated velocity
#         _, v2_mag_vec = perifocal_r_and_v(collision_altitude, ecc_1, TA, param["mu"])
#         v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
#         v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
#         r2 = r1 # Same position as r1, but different velocity

#         # Define input vectors
#         p1 = np.array([m1, radius_1, *r1, *v1, 1.0])
#         p2 = np.array([m2, radius_2, *r2, *v2, 1.0])

#         debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1, p2, param, scen_properties.LC)

#         if debris1.size == 0:
#             print(f"[{collision_altitude} km] No debris generated")


#         # Loop through 
#         frag_a = []
#         frag_e = []
#         frag_mass = []

#         for debris in debris1:
#             norm_earth_radius = debris[0]
#             if norm_earth_radius < 1:
#                 continue # decayed

#             frag_a.append((norm_earth_radius - 1) * 6371) 
#             frag_e.append(debris[1])
#             frag_mass.append(debris[7])
        
#         for debris in debris2:
#             norm_earth_radius = debris[0]
#             if norm_earth_radius < 1:
#                 continue # decayed

#             frag_a.append((norm_earth_radius - 1) * 6371) 
#             frag_e.append(debris[1])
#             frag_mass.append(debris[7])

#         frag_properties = np.array([frag_a, frag_mass, frag_e]).T

#         binE_alt = np.linspace(scen_properties.min_altitude, scen_properties.max_altitude, n_shells + 1)  # We add 1 for bin edges

#         bins = [binE_alt, binE_mass, binE_ecc]

#         hist, edges = np.histogramdd(frag_properties, bins=bins)

#         hist = hist / (SS * 3)

#         return hist

class EllipticalCollisionPair:
    def __init__(self, species1, species2, sma_index, ecc1_index, ecc2_index, shell_index) -> None:
        self.species1 = species1
        self.species2 = species2
        self.sma_index = sma_index
        self.ecc1_index = ecc1_index
        self.ecc2_index = ecc2_index
        self.shell_index = shell_index

        self.species1_sma = self.species1.semi_major_axis_bins_HMid[shell_index]
        self.species2_sma = self.species2.semi_major_axis_bins_HMid[shell_index]

        # Time-in-shell for each species at this (sma, ecc) in this shell
        self.species1_TIS = species1.time_per_shells[sma_index, ecc1_index, shell_index]
        self.species2_TIS = species2.time_per_shells[sma_index, ecc2_index, shell_index]

        # Velocity-in-shell for each species at this (sma, ecc) in this shell
        self.species1_VIS = species1.velocity_per_shells[sma_index, ecc1_index, shell_index]
        self.species2_VIS = species2.velocity_per_shells[sma_index, ecc2_index, shell_index]

        # Full TIS and VIS profiles across all ecc‐bins at this sma
        self.species1_TIS_all = species1.time_per_shells[sma_index]
        self.species2_TIS_all = species2.time_per_shells[sma_index]
        self.species1_VIS_all = species1.velocity_per_shells[sma_index]
        self.species2_VIS_all = species2.velocity_per_shells[sma_index]

        # Combined metrics
        self.combined_mass_TIS = self.species1_TIS * self.species2_TIS
        self.min_TIS = min(self.species1_TIS, self.species2_TIS)
        self.max_TIS = max(self.species1_TIS, self.species2_TIS)

        # Symbolic column names include eccentricity indices
        self.s1_col_sym_name = f"{species1.sym_name}_sh{shell_index}"
        self.s2_col_sym_name = f"{species2.sym_name}_sh{shell_index}"

        # Placeholders for collision outcomes
        self.gamma                   = None  # collision probability
        self.fragments               = None
        self.catastrophic            = None
        self.binOut                  = None
        self.altA                    = None  # debris semi-major axis
        self.altE                    = None  # debris eccentricity
        self.debris_eccentricity_bins = None

class SpeciesCollisionPair:
    def __init__(self, species1, species2, scen_properties) -> None:
        self.species1 = species1
        self.species2 = species2
        self.collision_pair_by_shell = []
        self.collision_processed = []

        # # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
        self.gamma = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

        # Create a list of source sinks, first two are the active species then the active speices
        self.source_sinks = [species1, species2] + scen_properties.species['debris']

        # Implementing logic for gammas calculations based on species properties
        if species1.maneuverable and species2.maneuverable:
            # Multiplying each element in the first column of gammas by the product of alpha_active values
            self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha_active * species2.alpha_active
            if species1.slotted and species2.slotted:
                # Applying the minimum slotting effectiveness if both are slotted
                self.gamma[:, 0] = self.gamma[:, 0] * min(species1.slotting_effectiveness, species2.slotting_effectiveness)

        elif (species1.maneuverable and not species2.maneuverable) or (species2.maneuverable and not species1.maneuverable):
            if species1.trackable and species2.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species2.alpha
            elif species2.trackable and species1.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha

        # Applying symmetric loss to both colliding species
        self.gamma[:, 1] = self.gamma[:, 0]

        # Rocket Body Flag - 1: RB; 0: not RB
        # Will be 0 if both are None type
        if species1.RBflag is None and species2.RBflag is None:
            self.RBflag = 0
        elif species2.RBflag is None:
            self.RBflag = species2.RBflag
        elif species2.RBflag is None:
            self.RBflag = species1.RBflag
        else:
            self.RBflag = max(species1.RBflag, species2.RBflag)

        self.phi = None # Proabbility of collision, based on v_imp, shell volumen and object_radii
        self.catastrophic = None # Catastrophic flag for each shell
        self.eqs = None # All Symbolic equations 
        self.nf = None # Initial symbolic equations for the number of fragments
        self.sigma = None # Square of the impact parameter


if __name__ == "__main__":
    p1_in = np.array([
            m1,  # mass in kg
            r1,     # radius in meters
            2372.4,  # r_x in km, 1000 km
            2743.1,  # r_y in km
            6224.8,  # r_z in km
            -5.5,    # v_x in km/s
            -3.0,    # v_y in km/s
            3.8,     # v_z in km/s
            1      # object_class (dimensionless)
        ])

    p2_in = np.array([
            m1,     # mass in kg
            r1,     # radius in meters
            2372.4,  # r_x in km
            2743.1,  # r_y in km
            6224.8,  # r_z in km
            3.2,     # v_x in km/s
            5.4,     # v_y in km/s
            -3.9,    # v_z in km/s
            1      # object_class (dimensionless)
        ])
    import pickle 
    
    # open scnario properties
    with open(r'C:\Users\IT\Documents\UCL\pyssem\scenario-properties-elliptical.pkl', 'rb') as f:
        scen_properties = pickle.load(f)

    s1, s2 = scen_properties.species['rocket_body'][0], scen_properties.species['debris'][0]
    v1, v2 = s1.velocity_per_shells[6][6], s2.velocity_per_shells[6][6]
    debris_species = [species for species in scen_properties.species['debris']]

    # Mass Binning
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
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # Create bin edges starting at 0 and finishing at 1
    binE_ecc = np.concatenate(([0], binE_ecc, [1]))

    m1 = 1250  # kg
    m2 = 6  # kg
    r1 = 4  # radius in some units
    r2 = 0.11  # radius in some units
    collision_index = 6  # example collision index


    results = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, 5, n_shells=10)

    print(results.shape)


