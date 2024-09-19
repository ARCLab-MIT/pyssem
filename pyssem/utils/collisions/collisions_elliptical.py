from utils.collisions.collisions import func_Am, func_dv
from utils.collisions.NASA_SBN6 import *
import numpy as np
from tqdm import tqdm
from utils.collisions.cartesian_to_kep import cart_2_kep, kep_2_cart
from sympy import symbols, Matrix, pi, S, Expr, zeros
import matplotlib.pyplot as plt
from copy import deepcopy


def create_collision_pairs(scen_properties):
    """
    This function will take each species that is elliptical, then it will search across all other ellitpical objects and shells to assess 
    which other objects it could collide with. 
    
    """

    all_elliptical_collision_species = []

    for species_group in scen_properties.species.values():
        count = 0
        # For every pair of species in the same group
        for i, species1 in enumerate(species_group):
            for j, species2 in enumerate(species_group):
                if i >= j:  # Skip redundant pairs and self-comparisons
                    continue
                
                collision_pair_unique = SpeciesCollisionPair(species1, species2)

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
                            # print(f"Species {species1.sym_name} and {species2.sym_name} are in shell {shell_index}")
                            # print(f"Time spent: {species1.sym_name}: {time_1}, {species2.sym_name}: {time_2}\n")
                            
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

    # args = [(i, species_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven) for i, species_pair in enumerate(all_elliptical_collision_species)]
    
    # results = [process_elliptical_collision_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    # Modify the args creation to loop over each species pair and its nested shell collisions
    for i, species_pair in enumerate(tqdm(all_elliptical_collision_species, desc="Processing species pairs")):
    # Add progress bar for the inner loop over shell-specific collisions
        for shell_collision in tqdm(species_pair.collision_pair_by_shell, desc="Processing shell collisions", leave=False):
            # Process each shell-specific collision and append result to collision_processed list
            result = process_elliptical_collision_pair((i, shell_collision, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven))
            species_pair.collision_processed.append(result)

    # Return the list of all elliptical collision species with processed collision data
    return all_elliptical_collision_species

 

def process_elliptical_collision_pair(args):
    """
    A similar function to the process_species_pair, apart from it as the shells are already defined and the velocity, 
    you are able to calculate evolve bins just once. 

    """
    i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius

    # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
    gammas = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

    # Create a list of source sinks, first two are the active species
    source_sinks = [collision_pair.species1, collision_pair.species2]

    s1 = collision_pair.species1
    s2 = collision_pair.species2

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

    dv1, dv2 = 10, 10 # this will have to change when velocities are properly introduced

    debris_eccentricity_bins = debris_species[0].eccentricity_bins

    # there needs to be some changes here to account for the fact that the shells are already defined
    # set fragment_spreading to True
    v1, v2 = s1.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index], s2.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    

    fragments = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_pair.shell_index, n_shells=scen_properties.n_shells)    
    collision_pair.gammas = gammas
    collision_pair.fragments = fragments
    # collision_pair.catastrophic = catastrophic
    # collision_pair.binOut = binOut
    # collision_pair.altA = altA # This tells you which new shells the fragments end up in
    # collision_pair.altE = altE # This tells you which new elliptical shells the fragments end up in

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

    dv = 7.5
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

    if max(m) == 0:
        # No fragments bigger the LNT are generated
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
        data = np.array([dv_vec.ravel()[:len(m)], m, new_ecc]).T  # Transpose to get the right shape (n_samples, 3)

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
        hist, edges = np.histogramdd(data, bins=bins)

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

        # Combine the data into a single array and transpose to get the right shape (n_samples, 3)
        data = np.array([dv_bins, m_bins, new_ecc]).T

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
        hist, edges = np.histogramdd(data, bins=bins)

        # Normalize the histogram
        hist = hist / SS

        return hist, edges

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

        self.gammas = None 
        self.fragments = None
        self.catastrophic = None
        self.binOut = None
        self.altA = None
        self.altE = None

        self.debris_eccentrcity_bins = None

class SpeciesCollisionPair:
    def __init__(self, species1, species2) -> None:
        self.species1 = species1
        self.species2 = species2
        self.collision_pair_by_shell = []
        self.collision_processed = []