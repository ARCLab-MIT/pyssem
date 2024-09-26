from utils.collisions.collisions import func_Am, func_dv
from utils.collisions.NASA_SBN6 import *
import numpy as np
from tqdm import tqdm
from utils.collisions.cartesian_to_kep import cart_2_kep, kep_2_cart
from sympy import symbols, Matrix, pi, S, Expr, zeros
import matplotlib.pyplot as plt
import re

def create_collision_pairs(scen_properties):
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

    for species_group in scen_properties.species.values():
        count = 0
        # For every pair of species in the same group
        for i, species1 in enumerate(species_group):
            for j, species2 in enumerate(species_group):
                
                collision_pair_unique = SpeciesCollisionPair(species1, species2, scen_properties)

                # Make some initial checks to save computation time for now, we will have to back to this later
                if species1.maneuverable or species2.maneuverable:
                    continue

                # Loop through the semi-major axis bins (assuming species1 and species2 have the same number of bins)
                for k in range(len(species1.semi_major_axis_bins)): # back to altitude
                    # Extract the time spent in shells for both species at this semi-major axis bin
                    time_in_shells_1 = species1.time_per_shells[k]  # Array for species1
                    time_in_shells_2 = species2.time_per_shells[k]  # Array for species2

                    # Loop through the shells and check if both species spend time in the same shell
                    for shell_index in range(len(time_in_shells_1)):
                        time_1 = time_in_shells_1[shell_index]
                        time_2 = time_in_shells_2[shell_index]

                        # Only print if both species spend time in this shell
                        if time_1 > 0 and time_2 > 0:
                            count += 1
                            
                            # There is also a check on the mass of the objects, if they are both too small then they will just create dust
                            # this is defined by the LC. LC is the diameter of the smallest object that can be tracked.
                            if species1.mass < scen_properties.LC and species2.mass < scen_properties.LC:
                                continue
                                    
                            collision_pair_unique.collision_pair_by_shell.append(EllipticalCollisionPair(species1, species2, shell_index))
                
                all_elliptical_collision_species.append(collision_pair_unique)

    print(f"Total number of unique species pairs: {len(all_elliptical_collision_species)}")
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
    for i, species_pair in enumerate(tqdm(all_elliptical_collision_species, desc="Processing species pairs")):
        for shell_collision in tqdm(species_pair.collision_pair_by_shell, desc="Processing shell collisions", leave=False):
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

    for collision_pair in tqdm(all_elliptical_collision_species):

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

        # Determine if the collision is catastrophic for each shell
        catastrophic = is_catastrophic(species1.mass, species2.mass, scen_properties.v_imp2)

        # Compute the number of fragments (nf) for each shell
        nf = np.zeros(len(scen_properties.v_imp2), dtype=object)
        for i, dv in enumerate(scen_properties.v_imp2):
            if catastrophic[i]:
                n_f_catastrophic = 0.1 * scen_properties.LC ** (-S(1.71)) * (
                    species1.mass + species2.mass
                ) ** (S(0.75))
                nf[i] = n_f_catastrophic
            else:
                n_f_damaging = 0.1 * scen_properties.LC ** (-S(1.71)) * (
                    min(species1.mass, species2.mass) * dv ** 2
                ) ** (S(0.75))
                nf[i] = n_f_damaging
        collision_pair.nf = nf  # 'nf' is now a NumPy array

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
            nf_s = collision_pair.nf[s_source]

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
                                    delta_gain = phi_s * N_species1_s * N_species2_s * nf_s * num_frags
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

def process_elliptical_collision_pair(args):
    """
    A similar function to the process_species_pair, apart from it as the shells are already defined and the velocity, 
    you are able to calculate evolve bins just once. 

    """
    i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius


    # there needs to be some changes here to account for the fact that the shells are already defined
    # set fragment_spreading to True
    v1 = collision_pair.species1.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    v2 = collision_pair.species2.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index]

    # This the time per shell for each species - the output still assumes that it is always there, so you need to divide
    t1 = collision_pair.species1.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    t2 = collision_pair.species2.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    min_TIS = min(t1, t2)
    print(m1, m2, min_TIS)

    fragments = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_pair.shell_index, n_shells=scen_properties.n_shells)    
    
    if fragments is not None:
        collision_pair.fragments = fragments # * min_TIS
    else: 
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

def evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_index, n_shells=0):

    SS = 20  # super sampling ratio
    MU = 398600.4418  # km^3/s^2
    RE = 6378.1  # km

    # Ensure that m1 > m2, if not swap
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

    dv = 10
    catastrophe_ratio = (m2 * ((dv * 1000) ** 2) / (2 * m1 * 1000))  # J/g = kg*(km/s)^2 / g

    # If the specific energy is < 40 J/g: non catastrophic collision
    if catastrophe_ratio < 40:
        M = m2 * dv ** 2  # Correction from ODQN [kg*km^2/s^2]
        isCatastrophic = 0
    else:  # Catastrophic collision
        M = m1 + m2
        isCatastrophic = 1


    isCatastrophic = 0 # TESTING 

    # find the number of fragments generated
    num = (0.1 * M ** 0.75 * scen_properties.LC ** (-1.71)) - (0.1 * M ** 0.75 * min(1, 2 * r1) ** (-1.71))
    numSS = SS * num

    # Create PDF of power law distribution, then sample 'num' selections to find each mass
    dd_edges = np.logspace(np.log10(scen_properties.LC), np.log10(min(1, 2 * r1)), 500)
    dd_means = 10 ** (np.log10(dd_edges[:-1]) + np.diff(np.log10(dd_edges)) / 2)

    # Cumulative distribution
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)
    ndd = np.maximum(0, -np.diff(nddcdf))

    # Make sure int, 0 to 1, random number for stochastic sampling of fragment diameters
    repeat_counts = np.floor(ndd).astype(int) + (np.random.rand(len(ndd)) > (1 - (ndd - np.floor(ndd)))).astype(int)
    d_pdf = np.repeat(dd_means, repeat_counts)

    try:
        dss = d_pdf[np.random.randint(0, len(d_pdf), size=int(np.ceil(numSS)))]
    except ValueError:
        dss = 0

    # Calculate the mass of objects
    A = 0.556945 * dss ** 2.0047077  # Equation 2.72
    Am = func_Am(dss, 5 if isCatastrophic == 0 else 0)  # use Am conversion of the larger object

    # order the output in descending order
    Am = np.sort(Am)[::-1]
    m = A / Am

    # Remove any fragments that are more than the original mass of m1
    m = m[m < m1]

    try:
        if max(m) == 0:
            # No fragments bigger the LNT are generated
            return None
    except ValueError:
        # Where no fragments are made
        return None
    
    # Find difference in orbital velocity for shells
    dDV = np.abs(np.median(np.diff(np.sqrt(MU / (RE + scen_properties.HMid)) * 1000)))

    # Compute delta-v for the fragments
    dv_values = func_dv(Am, 'col') / 1000  # km/s

    # Generate random directions for the velocity vectors
    u = np.random.rand(len(dv_values)) * 2 - 1
    theta = np.random.rand(len(dv_values)) * 2 * np.pi
    v = np.sqrt(1 - u ** 2)
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
    dv_vec = p * dv_values[:, np.newaxis]  # velocity vectors

    # # Calculate the new eccentricity of the fragments
    if len(m) < 100:
        a = scen_properties.HMid[collision_index] + scen_properties.re
        r_vec = np.tile([a, 0, 0], (len(m), 1))  # create an R
        v_vec = [np.linalg.norm(vec) for vec in dv_vec] + v1  # create a v vec
        v_vec_2d = np.array([[0, v, 0] for v in v_vec])
        new_ecc = func_de(r_vec, v_vec_2d)
        scen_properties = np.array([dv_vec.ravel()[:len(m)], m, new_ecc]).T  # Transpose to get the right shape (n_samples, 3)

        # Step 3: Define the bins for the 3D histogram
        binE_vel = np.arange(-n_shells, n_shells) * dDV / 1000

        zero_index = np.where(binE_vel == 0.0)[0][0]

        # Step 2: Calculate the shift to align 0 with the collision_index
        shift_amount = collision_index - zero_index

        # Step 3: Shift the array to make 0 appear at collision_index
        binE_vel_shifted = np.roll(binE_vel, shift_amount)

        # Step 4: Handle different cases
        if collision_index > n_shells // 2:
            binE_vel_selected = binE_vel_shifted[:n_shells]  # Take the first n_shells values (below 0)
        else:
            # If the collision index is less than or equal to 5, take values around 0
            start_index = max(0, collision_index - (n_shells // 2))
            end_index = start_index + n_shells  # Select the next `n_shells` values
            binE_vel_selected = binE_vel_shifted[start_index:end_index]

        bin_edges = np.zeros(len(binE_vel_selected) + 1)  # Bin edges will be one more than selected values

        # Calculate the midpoints between consecutive bin values to generate the edges
        bin_edges[1:-1] = (binE_vel_selected[:-1] + binE_vel_selected[1:]) / 2
        # For the first and last bin edge, extrapolate slightly beyond the min/max
        bin_edges[0] = binE_vel_selected[0] - (binE_vel_selected[1] - binE_vel_selected[0]) / 2
        bin_edges[-1] = binE_vel_selected[-1] + (binE_vel_selected[-1] - binE_vel_selected[-2]) / 2

        # Slice the array to get the values within the desired range
        bins = [bin_edges, binE_mass, binE_ecc]

        # Step 4: Use np.histogramdd to create the 3D histogram
        hist, edges = np.histogramdd(scen_properties, bins=bins)

        hist = hist/SS

        # Sum over the velocity bins to collapse the 3D histogram into a 2D histogram

        return hist

    else:
        # Bin the velocities
        def func_de_single(R, V):
            mu = 398600.4418  # km^3/s^2
            V = np.abs(V)
            h = np.cross(R, V)
            e_vector = (np.cross(V, h) / mu) - (R / np.linalg.norm(R))
            e = np.linalg.norm(e_vector)
            return e

        # Flatten dv_vec
        flattened_dv_vec = dv_vec.ravel()

        # Take only the values up to the length of m
        sliced_dv_vec = flattened_dv_vec[:len(m)]

        # Define the number of bins
        num_bins = 100  # Adjust this number based on your needs

        # Bin the velocities
        dv_bins = np.linspace(np.min(sliced_dv_vec), np.max(sliced_dv_vec), num_bins)
        # bin the masses
        m_bins = np.linspace(np.min(m), np.max(m), num_bins)

        # Create a single R vector
        a = scen_properties.HMid[collision_index] + scen_properties.re
        r_vec = np.array([a, 0, 0])  # create a single R

        # create new v vectors
        v_vec = [np.linalg.norm(vec) for vec in dv_bins] + v1  # create a v vec

        # Compute eccentricities for each bin
        new_ecc = [func_de_single(r_vec, np.array([0, v, 0])) for v in v_vec]

        # Combine the scen_properties into a single array and transpose to get the right shape (n_samples, 3)
        scen_properties = np.array([dv_bins, m_bins, new_ecc]).T

        # Step 3: Define the bins for the 3D histogram
        binE_vel = np.arange(-n_shells, n_shells) * dDV / 1000

        zero_index = np.where(binE_vel == 0.0)[0][0]

        # Step 2: Calculate the shift to align 0 with the collision_index
        shift_amount = collision_index - zero_index

        # Step 3: Shift the array to make 0 appear at collision_index
        binE_vel_shifted = np.roll(binE_vel, shift_amount)

        # Step 4: Handle different cases
        if collision_index > n_shells // 2:
            # If the collision index is greater than 5, take the bottom values (below 0)
            binE_vel_selected = binE_vel_shifted[:n_shells]  # Take the first n_shells values (below 0)
        else:
            # If the collision index is less than or equal to 5, take values around 0
            start_index = max(0, collision_index - (n_shells // 2))
            end_index = start_index + n_shells  # Select the next `n_shells` values
            binE_vel_selected = binE_vel_shifted[start_index:end_index]

        bin_edges = np.zeros(len(binE_vel_selected) + 1)  # Bin edges will be one more than selected values

        # Calculate the midpoints between consecutive bin values to generate the edges
        bin_edges[1:-1] = (binE_vel_selected[:-1] + binE_vel_selected[1:]) / 2
        # For the first and last bin edge, extrapolate slightly beyond the min/max
        bin_edges[0] = binE_vel_selected[0] - (binE_vel_selected[1] - binE_vel_selected[0]) / 2
        bin_edges[-1] = binE_vel_selected[-1] + (binE_vel_selected[-1] - binE_vel_selected[-2]) / 2

        # Slice the array to get the values within the desired range
        bins = [bin_edges, binE_mass, binE_ecc]

        # Use np.histogramdd to create the 3D histogram
        hist, edges = np.histogramdd(scen_properties, bins=bins)

        # Normalize the histogram
        hist = hist / SS

        return hist

class EllipticalCollisionPair:
    def __init__(self, species1, species2, index) -> None:
        self.species1 = species1
        self.species2 = species2
        self.species1_TIS = species1.time_per_shells[index][index]
        self.species2_TIS = species2.time_per_shells[index][index]
        self.species1_VIS = species1.velocity_per_shells[index][index]
        self.species2_VIS = species2.velocity_per_shells[index][index]
        self.species1_TIS_all = species1.time_per_shells[index]
        self.species2_TIS_all = species2.time_per_shells[index]
        self.species1_VIS_all = species1.velocity_per_shells[index]
        self.species2_VIS_all = species2.velocity_per_shells[index]
        self.shell_index = index
        self.s1_col_sym_name = f"{species1.sym_name}_sh_{index}"
        self.s2_col_sym_name = f"{species2.sym_name}_sh_{index}"
        self.col_sym_name = f"{species1.sym_name}__{species2.sym_name}_sh_{index}" 
        self.combinEd_mass_TIS = self.species1_TIS * self.species2_TIS
        self.min_TIS = min(self.species1_TIS, self.species2_TIS)
        self.max_TIS = max(self.species1_TIS, self.species2_TIS)

        self.gamma = None 
        self.fragments = None
        self.catastrophic = None
        self.binOut = None
        self.altA = None
        self.altE = None

        self.debris_eccentrcity_bins = None

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
    import pickle 
    
    # open scnario properties
    with open(r'C:\Users\IT\Documents\UCL\pyssem\scenario-properties-collision.pkl', 'rb') as f:
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


