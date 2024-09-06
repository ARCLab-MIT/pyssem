from utils.collisions.collisions import func_Am, func_dv
import numpy as np
from tqdm import tqdm
from sympy import symbols, Matrix, pi, S, Expr, zeros

def create_collision_pairs(scen_properties):
    """
    This function will take each species that is elliptical, then it will search across all other ellitpical objects and shells to assess 
    which other objects it could collide with. 
    
    """

    print(scen_properties)

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
                            all_elliptical_collision_species.append(EllipticalCollisionPair(species1, species2, shell_index))

    print(f"Total number of collisions: {count}")

    debris_species = [species for species in scen_properties.species['debris']]

    binC = np.zeros(len(debris_species))
    binE = np.zeros(2 * len(debris_species))
    binW = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC[index] = debris.mass
        binE[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW[index] = debris.mass_ub - debris.mass_lb

    binE = np.unique(binE)

    args = [(i, species_pair, scen_properties, debris_species, binE, LBgiven) for i, species_pair in enumerate(all_elliptical_collision_species)]
    
    results = [process_elliptical_collision_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    print(results)

def process_elliptical_collision_pair(args):
    """
    A similar function to the process_species_pair, apart from it as the shells are already defined and the velocity, 
    you are able to calculate evolve bins just once. 

    """
    i, collision_pair, scen_properties, debris_species, binE, LBgiven = args
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
    fragments, catastrophic, binOut, altA, altE = evolve_bins(m1, m2, r1, r2, dv1, dv2, [], binE, [], LBgiven, RBflag, source_sinks, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km, debris_eccentricity_bins)

    collision_pair.gammas = gammas
    collision_pair.fragments = fragments
    collision_pair.catastrophic = catastrophic
    collision_pair.binOut = binOut
    collision_pair.altA = altA # This tells you which new shells the fragments end up in
    collision_pair.altE = altE # This tells you which new elliptical shells the fragments end up in

    return collision_pair

def evolve_bins(m1, m2, r1, r2, dv1, dv2, binC, binE, binW, LBdiam, source_sinks, RBflag = 0, fragment_spreading=False, n_shells=0, R02 = None, debris_eccentricity_bins = None): # eventually add stochastic ability
    # for now just doing binE
    SS = 20 # super sampling ratio
    MU = 398600.4418  # km^3/s^2
    RE = 6378.1  # km
    altNums = None

    # Bin Center is given
    if len(binC) > 0 and len(binE) == 0 and len(binW) == 0: 
        LBm = binC[0] - (binC[1] - binC[0]) / 2  
        UBm = binC[-1] + (binC[-1] - binC[-2]) / 2 
        binEd = [LBm] + list((np.array(binC[:-1]) + np.array(binC[1:])) / 2) + [UBm]
    
    # Bin Edges are given
    elif len(binC) > 0 and len(binW) > 0 and len(binE) == 0: 
        binEd1 = binC - binW / 2
        binEd2 = binC + binW / 2
        binEd = np.sort(np.concatenate((binEd1, binEd2)))
        
        # Check for overlapping bin edges
        if any(np.diff(binC) < binW):
            errinds = np.where(np.diff(binC) < binW)[0]
            raise ValueError(f"Overlapping bin edges between bin centered at {binC[errinds[0]]:.1f} and {binC[errinds[0] + 1]:.1f}")

    # Bin Widths are given     
    elif len(binE) > 0 and len(binC) == 0 and len(binW) == 0:
        binEd = np.sort(binE)

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

    num = (0.1 * M ** 0.75 * LB ** (-1.71)) - (0.1 * M ** 0.75 * min(1, 2 * r1) ** (-1.71))
    numSS = SS * num

    # Below shouldn't be becessary for bin edges
    #if numSS == 0:
        #return np.zerios(len(binEd) -1)

    # Create PDF of power law distribution, then sample 'num' selections
    # Only up to 1m, then randomly sample larger objects as quoted above
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * r1)), 500) # logg space, up to either 1m or diameter of the larger object
    dd_means = 10 ** (np.log10(dd_edges[:-1]) + np.diff(np.log10(dd_edges)) / 2) # Log 10 of diameter edge bins, then mean values of each diameter edge bin (linear as bins are log-spaced)
    
    # Cumulative distribution
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)  #eq 2.68
    ndd = np.maximum(0, -np.diff(nddcdf)) # diff to get the PDF count for the bins (dd_edges), if negative, set to 0
    
    # Make sure int, 0 to 1, random number for stochastic sampling of fragment diameters
    repeat_counts = np.floor(ndd).astype(int) + (np.random.rand(len(ndd)) > (1 - (ndd - np.floor(ndd)))).astype(int)
    d_pdf = np.repeat(dd_means, repeat_counts) # PDF of debris objects between LB and 1m.

    try:
        dss = d_pdf[np.random.randint(0, len(d_pdf), size=int(np.ceil(numSS)))]
    except ValueError: # This is when the probability breaks as the objects are too small
        dss = 0
        #return np.zeros(len(binEd) - 1)

    # Calculate the mass of objects
    A = 0.556945 * dss ** 2.0047077 # Equation 2.72
    Am = func_Am(dss, objclass) # use Am conversion of the larger object
    m = A/Am

    # Binning via histcounts
    nums, _ = np.histogram(m, bins=binEd)
    nums = nums / SS # Correct for super sampling

    # Define binOut based on the option chosen for bin setup
    binOut = []
    if binC is not None and binE is None and binW is None:  # Option 1: bin center given; output = edges
        binOut = binEd
    elif binE is not None and binC is None and binW is None:  # Option 3: bin edges given; output = centers
        binOut = binE[:-1] + np.diff(binE) / 2

    # Always need to assume fragment spreading.
    nShell = len(np.diff(R02))

    # find difference in orbital velocity for shells
    # dDV = np.abs(np.median(np.diff(np.sqrt(MU / (RE + R02)) * 1000))) # use equal spacing in dv space for binning to altitude base 
    dDV = np.abs(np.median(np.diff(np.sqrt(MU / (RE + np.arange(200, 2000, 50))) * 1000)))
    dv_values = func_dv(Am, 'col') / 1000 # km/s
    u = np.random.rand(len(dv_values)) * 2 - 1
    theta = np.random.rand(len(dv_values)) * 2 * np.pi

    v = np.sqrt(1 - u**2)
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
    dv_vec = p * dv_values[:, np.newaxis]

    hc, _, _ = np.histogram2d(dv_vec.ravel(), np.tile(m, 3), bins=[np.arange(-nShell, nShell + 1) * dDV / 1000, binEd])
    altA = hc / (SS * 3)

    # Also need to calculate the change in eccentricity bins for the debris objects. 

    altE = debris_eccentricity_bins

    return nums, isCatastrophic, binOut, altA, altE













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
        self.combined_TIS = self.species1_TIS * self.species2_TIS
        self.min_TIS = min(self.species1_TIS, self.species2_TIS)
        self.max_TIS = max(self.species1_TIS, self.species2_TIS)

        self.gammas = None 
        self.fragments = None
        self.catastrophic = None
        self.binOut = None
        self.altA = None
        self.altE = None
