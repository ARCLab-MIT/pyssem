from itertools import combinations
from sympy import symbols, Matrix
import numpy as np
from ..simulation.species_pair_class import SpeciesPairClass
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


def func_Am(d, ObjClass):
    """
    Calculates the area-to-mass ratio for spacecraft fragments based on NASA's new breakup model of evolve 4.0.
    
    Parameters:
    d : ndarray
        Array of diameters in meters.
    ObjClass : int or float
        Object class indicating whether the object is a rocket body or not.
    
    Returns:
    out : ndarray
        Area-to-mass ratio for each fragment.
    """
    numObj = d.size
    logds = np.log10(d)
    amsms = np.nan * np.ones((numObj, 5))  # alpha, mu1, sigma1, mu2, sigma2

    if 4.5 < ObjClass < 8.5:  # Rocket-body related
        for ind, logd in enumerate(logds):
            alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_for_rocket_body(logd)
            amsms[ind, :] = [alpha, mu1, sigma1, mu2, sigma2]
    else:  # Not rocket body
        for ind, logd in enumerate(logds):
            alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_not_rocket_body(logd)
            amsms[ind, :] = [alpha, mu1, sigma1, mu2, sigma2]

    N1 = amsms[:, 1] + amsms[:, 2] * np.random.randn(numObj)
    N2 = amsms[:, 3] + amsms[:, 4] * np.random.randn(numObj)

    out = 10 ** (amsms[:, 0] * N1 + (1 - amsms[:, 0]) * N2)

    return out

def func_dv(Am, mode):
    """
    Calculate the change in velocity (delta-v) for debris fragments based on their area-to-mass ratio.

    This will tell you velocity based off H. Kilnkrad - "Space Debris: Models and Risk Analysis" (2006).
    Equation 3.42 (Coefficients) and 3.44 (Full Equation)

    Args:
        Am (np.ndarray): Area-to-mass ratio of fragments.
        mode (str): Mode of calculation, e.g., 'col' for collision-induced delta-v or 'exp' for explosions

    Returns:
        np.ndarray: Calculated delta-v values for each fragment.
    """
    if mode == 'col':
       mu = 0.2 * np.log10(Am) + 1.85 # Explosion
    elif mode == 'exp':
        mu = 0.9 * np.log10(Am) + 2.9

    sigma = 0.4
    N = mu + sigma * np.random.randn(*np.shape(mu))
    z = 10 ** N # m/s
    return z 

def calculate_amsms_for_rocket_body(logd):
    """
    Calculate alpha, mu1, sigma1, mu2, sigma2 for rocket body related objects.
    """
    # alpha calculation
    if logd <= -1.4:
        alpha = 1
    elif -1.4 < logd < 0:
        alpha = 1 - 0.3571 * (logd + 1.4)
    else:
        alpha = 0.5

    # mu1 calculation
    if logd <= -0.5:
        mu1 = -0.45
    elif -0.5 < logd < 0:
        mu1 = -0.45 - 0.9 * (logd + 0.5)
    else:
        mu1 = -0.9

    # sigma1 is constant
    sigma1 = 0.55

    # mu2 is constant
    mu2 = -0.9

    # sigma2 calculation
    if logd <= -1.0:
        sigma2 = 0.28
    elif -1 < logd < 0.1:
        sigma2 = 0.28 - 0.1636 * (logd + 1)
    else:
        sigma2 = 0.1

    return alpha, mu1, sigma1, mu2, sigma2

def calculate_amsms_not_rocket_body(logd):
    """
    Calculate alpha, mu1, sigma1, mu2, sigma2 for non-rocket body related objects.
    """
    # alpha calculation
    if logd <= -1.95:
        alpha = 0
    elif -1.95 < logd < 0.55:
        alpha = 0.3 + 0.4 * (logd + 1.2)
    else:
        alpha = 1

    # mu1 calculation
    if logd <= -1.1:
        mu1 = -0.6
    elif -1.1 < logd < 0:
        mu1 = -0.6 - 0.318 * (logd + 1.1)
    else:
        mu1 = -0.95

    # sigma1 calculation
    if logd <= -1.3:
        sigma1 = 0.1
    elif -1.3 < logd < -0.3:
        sigma1 = 0.1 + 0.2 * (logd + 1.3)
    else:
        sigma1 = 0.3

    # mu2 calculation
    if logd <= -0.7:
        mu2 = -1.2
    elif -0.7 < logd < -0.1:
        mu2 = -1.2 - 1.333 * (logd + 0.7)
    else:
        mu2 = -2.0

    # sigma2 calculation
    if logd <= -0.5:
        sigma2 = 0.5
    elif -0.5 < logd < -0.3:
        sigma2 = 0.5 - (logd + 0.5)
    else:
        sigma2 = 0.3

    return alpha, mu1, sigma1, mu2, sigma2

def evolve_bins(m1, m2, r1, r2, dv, binC, binE, binW, LBdiam, RBflag = 0, fragment_spreading=False, n_shells=0, R02 = None): # eventually add stochastic ability
    """
    Function to evolve the mass bins of a debris cloud after a collision. The function is based on the NASA Standard Breakup
    Model. The function returns the number of fragments in each bin, whether the collision was catastrophic or not, and the
    diameters of the fragments.

    :param m1: Mass of Object 1 [kg]
    :type m1: int or float
    :param m2: Mass of Object 2 [kg]
    :type m2:  int or float
    :param r1: Radius of Object 1 [m]
    :type r1: int or float
    :param r2: Radius of Object 2 [m]
    :type r2: int or float
    :param dv: Collision velocity [km/s]
    :type dv: int or float
    :param binC: bin center for mass binning
    :type binC: int or float
    :param binE: bin edges for mass binning
    :type binE: int or float
    :param binW: bin widths for mass binning
    :type binW: int or float
    :param LBdiam:  Lower bound of Characteristic Length
    :type LBdiam: int or float
    :param RBflag: for area to mass ratio (func_Am.m), 1: RB; 0: not RB (default) (optional, defaults to 0)
    :type RBflag: int, optional
    :param sto: _stochastic flag (default: 1) (optional) (0 for deterministic, not implemented yet) 
    :type sto: int, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    # Super sampling ratio
    SS = 20
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
        
    else:
        raise ValueError(f"Wrong setup for bins given (binC empty: {len(binC) == 0}; binE empty: {len(binE) == 0}; binW empty: {len(binW) == 0})")

    LB = LBdiam
    objclass = 5 if RBflag == 0 else 0

    # Ensure that m1 > m2, if not swap
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

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

    if numSS == 0:  # check if 0 (e.g., if LB = r1)
        #return np.zeros(len(binEd) - 1), isCatastrophic, []
        return np.zeros(len(binEd) - 1)
    
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
        return np.zeros(len(binEd) - 1)

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

    # Assing delta-V to spherically random directions
    if fragment_spreading:
        dAlt = np.median(np.diff(R02))
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
        altNums = hc / (SS * 3)

    return nums, isCatastrophic, binOut, altNums

def process_species_pair(args):
    
    i, (s1, s2), scen_properties, debris_species, binE, LBgiven = args
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
    is_catastrophic = np.zeros((1, scen_properties.species_length))
    alt_nums = np.zeros((scen_properties.n_shells * 2, len(debris_species)))

    # This will tell you the number of fragments in each debris bin
    for dv_index, dv in enumerate(scen_properties.v_imp2):
        # If using the collision spreading function             
        if scen_properties.fragment_spreading:
            try:
                results = evolve_bins(m1, m2, r1, r2, dv, [], binE, [], LBgiven, RBflag, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km)
                frags_made[dv_index, :] = results[0]
                alt_nums = results[3]
            except IndexError as ie:
                alt_nums  = None
                continue
            except ValueError as e:
                continue
        else:
            results = evolve_bins(m1, m2, r1, r2, dv, [], binE, [], LBgiven, RBflag)
            frags_made[dv_index, :] = results[0]

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

    if scen_properties.fragment_spreading:
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, alt_nums)
    else:
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties)
        

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
    n_f = symbols('n_f:{0}'.format(scen_properties.n_shells))

    # Debris species
    debris_species = scen_properties.species['debris']

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
