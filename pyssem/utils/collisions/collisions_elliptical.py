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
                            count += 1
                            print(f"Species {species1.sym_name} and {species2.sym_name} are in shell {shell_index}")
                            print(f"Time spent: {species1.sym_name}: {time_1}, {species2.sym_name}: {time_2}\n")
                            
                            # There is also a check on the mass of the objects, if they are both too small then they will just create dust
                            # this is defined by the LC. LC is the diameter of the smallest object that can be tracked.
                            if species1.mass < scen_properties.LC and species2.mass < scen_properties.LC:
                                continue
                                    
                            all_elliptical_collision_species.append(EllipticalCollisionPair(species1, species2, shell_index))

    print(f"Total number of elliptical collision pairs: {count}")

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
    ecc_values = np.insert(binE_ecc, 0, 0) # Insert 0 at the beginning
    ecc_values = np.append(ecc_values, 1) # Append 1 at end 


    args = [(i, species_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven) for i, species_pair in enumerate(all_elliptical_collision_species)]
    
    results = [process_elliptical_collision_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    return results 

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
    fragments = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, s1.eccentricity, s2.eccentricity, [], binE_mass, binE_ecc, [], 
                            LBgiven, source_sinks, collision_pair.shell_index, RBflag, True, scen_properties.n_shells, scen_properties.R0_km, debris_eccentricity_bins)

    collision_pair.gammas = gammas
    collision_pair.fragments = fragments
    # collision_pair.catastrophic = catastrophic
    # collision_pair.binOut = binOut
    # collision_pair.altA = altA # This tells you which new shells the fragments end up in
    # collision_pair.altE = altE # This tells you which new elliptical shells the fragments end up in

    return collision_pair

def func_de(R, V):
    # Calculate specific angular momentum vector h = R x V
    V = np.abs(V)
    h = np.cross(R, V)

    mu = 398600.4418  # km^3/s^2
    
    # Calculate eccentricity vector e
    e_vector = (np.cross(V, h) / mu) - (R / np.linalg.norm(R))

    e = np.linalg.norm(e_vector)
    
    # Return the magnitude of the eccentricity vector
    return e

def evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, e1, e2, binC, binE_mass, binE_ecc, binW, LBdiam, source_sinks, collision_index, RBflag = 0, fragment_spreading=False, n_shells=0, R02 = None, debris_eccentricity_bins = None): # eventually add stochastic ability
    
    SS = 20 # super sampling ratio
    MU = 398600.4418  # km^3/s^2
    RE = 6378.1  # km
    altNums = None

    # Bin Center is given
    if len(binC) > 0 and len(binE_mass) == 0 and len(binW) == 0: 
        LBm = binC[0] - (binC[1] - binC[0]) / 2  
        UBm = binC[-1] + (binC[-1] - binC[-2]) / 2 
        binEd_mass = [LBm] + list((np.array(binC[:-1]) + np.array(binC[1:])) / 2) + [UBm]
    
    # Bin Edges are given
    elif len(binC) > 0 and len(binW) > 0 and len(binE_mass) == 0: 
        binEd_mass1 = binC - binW / 2
        binEd_mass2 = binC + binW / 2
        binEd_mass = np.sort(np.concatenate((binEd_mass1, binEd_mass2)))
        
        # Check for overlapping bin edges
        if any(np.diff(binC) < binW):
            errinds = np.where(np.diff(binC) < binW)[0]
            raise ValueError(f"Overlapping bin edges between bin centered at {binC[errinds[0]]:.1f} and {binC[errinds[0] + 1]:.1f}")

    # Bin Widths are given     
    elif len(binE_mass) > 0 and len(binC) == 0 and len(binW) == 0:
        binEd_mass = np.sort(binE_mass)

    # Eccentric Bin Widths are given
    if len(binE_ecc) > 0:
        binE_ecc = np.sort(binE_ecc)

    LB = LBdiam
    objclass = 5 if RBflag == 0 else 0

    # Ensure that m1 > m2, if not swap
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

    dv = 10 # this will have to change when velocities are properly introduced
    catastrophe_ratio = (m2*((dv*1000)**2)/(2*m1*1000)) #J/g = kg*(km/s)^2 / g

    # If the specific energy is < 40 J/g: non catastrophic collision
    if catastrophe_ratio < 40:
        M = m2 * dv ** 2 # Correction from ODQN [kg*km^2/s^2]    
        isCatastrophic = 0
    else: # Catastrophic collision
        M = m1 + m2
        isCatastrophic = 1

    # find the number of fragments generated
    num = (0.1 * M ** 0.75 * LB ** (-1.71)) - (0.1 * M ** 0.75 * min(1, 2 * r1) ** (-1.71))
    numSS = SS * num

    # Create PDF of power law distribution, then sample 'num' selections to find each mass
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * r1)), 500)
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
    A = 0.556945 * dss ** 2.0047077 # Equation 2.72 
    Am = func_Am(dss, objclass) # use Am conversion of the larger object
    
    # order the output in descending order
    Am = np.sort(Am)[::-1]
    m = A/Am

    # Remove any fragments that are more than the original mass of m1
    m = m[m < m1]

    # Calculate which parent object the fragments came from
    mass_ratio1 = m1 / (m1 + m2)
    mass_ratio2 = m2 / (m1 + m2)

    num_fragments = len(m)
    num_fragments1 = int(num_fragments * mass_ratio1)
    num_fragments2 = int(num_fragments * mass_ratio2)  # Remaining fragments

    # Split the mass array
    fragments_obj1 = m[:num_fragments1]
    fragments_obj2 = m[num_fragments1:]

    m_by_root_species = np.concatenate((fragments_obj1, fragments_obj2))

    # Find difference in orbital velocity for shells
    dDV = np.abs(np.median(np.diff(np.sqrt(MU / (RE + scen_properties.HMid)) * 1000)))

    # Compute delta-v for the fragments
    dv_values = func_dv(Am, 'col') / 1000  # km/s

    # Generate random directions for the velocity vectors
    u = np.random.rand(len(dv_values)) * 2 - 1
    theta = np.random.rand(len(dv_values)) * 2 * np.pi
    v = np.sqrt(1 - u**2)
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
    dv_vec = p * dv_values[:, np.newaxis]  # velocity vectors

    # # Split the dv into the root species array
    dv_vec_by_root_species = [dv_vec[:num_fragments1], dv_vec[num_fragments1:]]

    # # Get the original velocity vectors for time in shells, and apply the change
    # #fragments_new_vel = np.array([v1 + dv_vec_by_root_species, v2 + dv_vec_by_root_species])

    # # Original eccentricity of the fragments
    # ecc_initial_fragments = [np.full(len(fragments_obj1), e1), np.full(len(fragments_obj2), e2)]

    # a = scen_properties.HMid[collision_index] + scen_properties.re

    # r_vec = np.array([a, 0, 0])  # Initial position vector
    # fragments_r_vec = [np.array([r_vec] * len(fragments_obj1)), np.array([r_vec] * len(fragments_obj2))]

    
    # dE_values_fragments1 = []
    # dE_values_fragments2 = []

    # for i in range(len(fragments_obj1)):
    #     e_new = func_de(fragments_r_vec[0][i], dv_vec_by_root_species[0][i])
    #     dE_values_fragments1.append(e_new - e1)

    # for i in range(len(fragments_obj2)):
    #     e_new = func_de(fragments_r_vec[1][i], dv_vec_by_root_species[1][i])
    #     dE_values_fragments2.append(e_new - e2)

    # # Combine eccentricity changes
    # dE_values = np.concatenate([dE_values_fragments1, dE_values_fragments2]) 
    # Take the initial velocity (10km/s) which will need to be a vector - 
    # all you need is r, v  => then cross for h, the mag, then go to eccentricity
    # take single value for r (alt) make a vecotr that is the dv
    # velocity vector that is the same as initial velocity
    # add vector of dvs to original velocity vectors
    # then you have a function of eccentricity since original 
    # this will give you a set of ecc, then bin
    # the original eccentricity will have the be taken into account to work out the difference - this will come from the SBM.

    # better explanation of the document - better explanation of the variables on the github. Include warnings - if someone picks a large LC for example. 

    # minimim value where things no longer collide. LC. # characteristic length, its diameter, not the radius. Has to match what the SBM does.  

    # Ensure the eccentricity bins start at 0 and end at 1
    # binE_ecc = np.insert(binE_ecc, 0, 0)
    # binE_ecc = np.append(binE_ecc, 1)

    # # Find the eccentricity bin for each fragment
    # binEccIndex = np.digitize(dE_values, binE_ecc) - 1


    # WHILE ECCENTRICITY IS INCORRECT - WE ARE JUST GOING TO USE RANDOM NUMBERS FOR NEW ECC
    random_list1 = np.random.uniform(low=0.0, high=0.1, size=num_fragments1)
    random_list2 = np.random.uniform(low=0.0, high=0.1, size=num_fragments2)

    dE_values = np.concatenate([random_list1, random_list2])

    # I need to bin the fragments based on the eccentricity change using binEd_ecc
    binEccIndex = np.digitize(dE_values, binE_ecc)

    # # # Generate velocity-mass distribution using 2D histogram
    # hc, _, _ = np.histogram2d(dv_vec.ravel(), np.tile(m, 3), bins=[np.arange(-n_shells, n_shells) * dDV / 1000, binEd_mass])
    # altNums = hc / (SS * 3)

    # Initialize the final fragment matrix with size equal to the number of shells
    fragment_matrix = np.zeros((n_shells, len(binE_ecc) - 1, len(binEd_mass) - 1))

    # Use the eccentricity change to spread fragments across eccentricity bins proportionally
    for fragment_idx in range(len(dv_vec)):
        # Get the velocity bin index and mass bin index for this fragment
        vel_bin_idx = np.digitize(dv_vec[fragment_idx], np.arange(-n_shells, n_shells) * dDV / 1000) - 1
        mass_bin_idx = np.digitize(m[fragment_idx], binEd_mass) - 1

        # Ensure indices are within valid bounds
        if 0 <= vel_bin_idx < n_shells and 0 <= mass_bin_idx < len(binEd_mass) - 1:
            # Get the eccentricity bin index from binEccIndex
            ecc_bin_idx = binEccIndex[fragment_idx]

            # Calculate the fragment position relative to the collision index
            position_idx = collision_index + vel_bin_idx - n_shells // 2

            # Calculate the new altitude for this fragment based on its velocity change
            new_altitude = calculate_new_altitude(scen_properties.HMid, collision_index, dv_vec[fragment_idx])

            # Exclude fragments that have decayed below a certain threshold (e.g., Earth's atmosphere)
            if new_altitude < 0:
                continue  # The fragment has decayed, skip it

            # Ensure fragments stay within bounds
            if position_idx < 0:
                continue  # Remove fragments below the bottom shell
            elif position_idx >= n_shells:
                position_idx = n_shells - 1  # Add excess fragments to the top shell

            # Assign fragments to the appropriate shell, mass, and eccentricity bin
            fragment_matrix[position_idx, ecc_bin_idx, mass_bin_idx] += 1

    # Ensure the total count of fragments is conserved
    total_fragments_generated = len(dE_values)
    total_fragments_in_matrix = np.sum(fragment_matrix)
    fragment_matrix = fragment_matrix * (total_fragments_generated / total_fragments_in_matrix)

    return fragment_matrix

def calculate_new_altitude(HMid, collision_index, dv, mu=398600.4418):
    """
    Calculate the new altitude of a fragment based on its initial altitude and change in velocity.
    HMid: array of midpoints (altitudes in km) of the shells.
    collision_index: index representing the original shell of the fragment.
    dv: change in velocity (km/s) of the fragment.
    mu: gravitational parameter (default is Earth's gravitational parameter in km^3/s^2).
    
    Returns the new altitude after the velocity change.
    """
    # Initial altitude corresponding to the collision_index
    r_initial = HMid[collision_index]

    # Orbital energy before and after the velocity change
    v_initial = np.sqrt(mu / r_initial)  # Circular orbit velocity at the initial altitude
    v_new = v_initial + dv  # New velocity after collision

    # Calculate the new orbital radius based on the energy change
    r_new = mu / (v_new**2)
    return np.linalg.norm(r_new) + 6378.1



def simulateCollision(mass1=1000, mass2=10, a=450+6371, e=0, i=0, omega_AP=0, omega_LAN=0, EA=0):
    
    fragments, catastrophic = createFragmentsLikeNASAdata(mass1, mass2, 100, printOut=0)
    #trackable = pullTrackableObjects(fragments)
    
    #print ("{} fragments to simulate".format(numpy.sum(trackable[:,1])))
    
    rInit, vInit = kep_2_cart(a,e,i,omega_AP,omega_LAN, EA) #Initial location and speed of collision
    
    cartesianStore = []
    
    #for fragmentsgroup in trackable: #Loop over trackable fragment groups (mutliples of objects inside):
    for fragmentsgroup in fragments: #Loop over trackable fragment groups (mutliples of objects inside):
        for fragment in range(int(fragmentsgroup[1])): #each individual fragment
            
            rInitcurr = deepcopy(rInit)
            
            dV = numpy.real(fragmentsgroup[6]) * random_three_vector() #dV vector with random direction
            
            new_velocity = vInit + dV
            #print (rInitcurr, new_velocity)
            
            newCart = cart_2_kep(rInitcurr,new_velocity,rInit, vInit, fragmentsgroup[6], dV)
            cartesianStore += [newCart]
            
    return numpy.array(cartesianStore)

def pullTrackableObjects(fragments):
    
    trackable = []
    
    for fragment in fragments:
        if fragment[2]>0.1:
            trackable += [fragment]

    return numpy.array(trackable)

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = numpy.random.uniform(0,numpy.pi*2)
    costheta = numpy.random.uniform(-1,1)

    theta = numpy.arccos( costheta )
    x = numpy.sin( theta) * numpy.cos( phi )
    y = numpy.sin( theta) * numpy.sin( phi )
    z = numpy.cos( theta )
    return numpy.array((x,y,z))

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
