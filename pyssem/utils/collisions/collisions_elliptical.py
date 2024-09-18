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
    binE_ecc = np.insert(binE_ecc, 0, 0) # Insert 0 at the beginning
    binE_ecc = np.append(binE_ecc, 1) # Append 1 at end 


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

    if max(m) == 0:
        # No fragments bigger the LNT are generated
        return None

    # # Calculate which parent object the fragments came from
    # mass_ratio1 = m1 / (m1 + m2)
    # mass_ratio2 = m2 / (m1 + m2)

    # num_fragments = len(m)
    # num_fragments1 = int(num_fragments * mass_ratio1)
    # num_fragments2 = int(num_fragments * mass_ratio2)  # Remaining fragments

    # # Split the mass array
    # fragments_obj1 = m[:num_fragments1]
    # fragments_obj2 = m[num_fragments1:]

    # m_by_root_species = np.concatenate((fragments_obj1, fragments_obj2))

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

    # Calculate the new eccentricity of the fragments
    a = scen_properties.HMid[collision_index] + scen_properties.re
    r_vec = np.tile([a, 0, 0], (len(m), 1)) # create an R
    v_vec = [np.linalg.norm(vec) for vec in dv_vec] + v1 # create a v vec
    v_vec_2d = np.array([[0, v, 0] for v in v_vec])
    new_ecc = func_de(r_vec, v_vec_2d)

    # I need to bin the fragments based on the eccentricity change using binEd_ecc
    binEccIndex = np.digitize(new_ecc, binE_ecc)

    # # # Generate velocity-mass distribution using 2D histogram - old way
    # hc, _, _ = np.histogram2d(dv_vec.ravel(), np.tile(m, 3), bins=[np.arange(-n_shells, n_shells) * dDV / 1000, binEd_mass])
    # altNums = hc / (SS * 3)

    # Step 1: Calculate the magnitude of the velocity change from the u, v, w components
    dv = np.sqrt(dv_vec[:, 0]**2 + dv_vec[:, 1]**2 + dv_vec[:, 2]**2)  # Magnitude of velocity

    # Step 2: Determine the direction (positive or negative) based on the primary velocity component
    # Assuming the 'u' component tells us the main direction of altitude change.
    direction_flags = np.sign(dv_vec[:, 0])  # Positive or negative movement along 'u' axis (altitude)

    # Step 3: Calculate the number of velocity changes required to reach each shell
    # Define shell bins based on the velocity differences
    shell_bins = np.arange(0, n_shells) * dDV / 1000  # Only positive shells, since negative velocities won't reach new altitudes

    # Initialize an array to hold the shell index for each fragment
    n_shell_indices = np.full(len(dv), None)  # Initially set all fragments to None

    # Step 4: For fragments moving in the positive direction (ascending):
    positive_movement = direction_flags > 0
    positive_dv = dv[positive_movement]

    # Bin the fragments with positive movement into shell bins based on their velocity magnitude
    if len(positive_dv) > 0:
        positive_shell_indices = np.digitize(positive_dv, shell_bins) - 1  # Get the shell index for each fragment
        n_shell_indices[positive_movement] = collision_index + positive_shell_indices

    # Step 5: For fragments moving in the negative direction (descending):
    # Here, we assume negative velocities won't reach any altitude above the collision point.
    negative_movement = direction_flags < 0
    n_shell_indices[negative_movement] = -99  # Fragments moving downward don't reach any higher altitude shell

    n_shell_indices = np.where((n_shell_indices >= 0) & (n_shell_indices < n_shells-1), n_shell_indices, None)

    # bin the mass of the fragments based on the binEd_mass
    binMassIndex = np.digitize(m, binEd_mass)

    # Create a final matrix, which is for each each fragment, the shell index, the eccentricity index and the mass index
    fragment_matrix = np.array([n_shell_indices, binEccIndex, binMassIndex]).T

    fragment_3D_matrix = np.zeros((n_shells, len(binEd_mass)-1, len(binE_ecc)-2)) # this is wrong, the eccentricity is 2 more than it should be

    # # Loop through the fragment_matrix and populate the 3D matrix
    # for row in fragment_matrix:
    #     shell_idx = row[0]      # n_shell_indices
    #     ecc_idx = row[1]         # binEccIndex
    #     mass_idx = row[2]    # binMassIndex
        
    #     # Increment the count at the corresponding shell, mass, and eccentricity bin
    #     if shell_idx != -99 and shell_idx is not None:
    #         fragment_3D_matrix[shell_idx-1, mass_idx-1, ecc_idx-2] += 1
    
    return fragment_3D_matrix

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
