import numpy as np
from itertools import combinations
import multiprocessing as mp
from sympy import symbols, Matrix
from tqdm import tqdm
from .NASA_SBM_Evolve import evolve_bins_circular, evolve_bins_elliptical
from ..simulation.species_pair_class import SpeciesPairClass
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import math
import random


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
    # ensure Am is a list of floats
    if isinstance(Am, (int, float)):
        Am_list = [float(Am)]
    else:
        Am_list = [float(x) for x in Am]

    sigma = 0.4
    result = []
    for am_val in Am_list:
        if mode == 'col':
            mu_val = 0.9 * math.log10(am_val) + 2.9
        elif mode == 'exp':
            mu_val = 1.85 * math.log10(am_val) + 1.85
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # add Gaussian noise
        N_val = mu_val + sigma * random.gauss(0, 1)
        result.append(10 ** N_val)

    # return scalar if single input, else list
    return result[0] if len(result) == 1 else result

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

def evolve_bins(m1, m2, r1, r2, dv1, dv2, binC, binE, binW, LBdiam, source_sinks, RBflag = 0, fragment_spreading=False, n_shells=0, R02 = None): # eventually add stochastic ability
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
    :param dv: Collision velocity [km/s] of species 1
    :type dv1: int or float
    :param dv: Collision velocity [km/s] of species 2
    :type dv2: int or float
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
    SS = 100
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

    # Calculate the relative delta-V
    if dv1 != dv2:
        dv = (m1 * dv1 + m2 * dv2) / (m1 + m2)
    else:
        dv = dv1

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
        dv_values = np.array(func_dv(Am, 'col')) / 1000 # km/s
        u = np.random.rand(len(dv_values)) * 2 - 1
        theta = np.random.rand(len(dv_values)) * 2 * np.pi

        v = np.sqrt(1 - u**2)
        p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
        dv_vec = p * dv_values[:, np.newaxis]

        hc, _, _ = np.histogram2d(dv_vec.ravel(), np.tile(m, 3), bins=[np.arange(-nShell, nShell + 1) * dDV / 1000, binEd])
        altNums = hc / (SS * 3)

        if altNums is None:
            print(hc)  # Check if hc is correct
            print(SS)  # Check if SS is not zero
            print(hc.shape)  # Check if hc has the expected shape
            print(SS * 3)  # Ensure that the denominator is not zero

    return nums, isCatastrophic, binOut, altNums

def process_species_pair(args):    
    i, (s1, s2), scen_properties, debris_species, binE_mass, LBgiven = args
    m1, m2 = s1.mass, s2.mass
    r1, r2 = s1.radius, s2.radius
    
    # Determine if fragments should be included based on IADC collision rules
    include_fragments = False
    catastrophic = False
    s1_name = s1.sym_name
    s2_name = s2.sym_name
    
    # Collisions between satellites and between satellites and large debris were catastrophic.
    if s1_name == 'S' and s2_name in ['S', 'N_446kg']:
        include_fragments = False
        catastrophic = True
    # if a combination of large debris, then include fragments
    if s1_name in ['N_446kg'] and s2_name in ['N_446kg']:
        include_fragments = True
        catastrophic = False
    if s1_name in ['N_32kg'] and s2_name in ['N_32kg']:
        include_fragments = True
        catastrophic = False
    # if a combination of medium debris, then include fragments
    if s1_name == 'N_0.64kg' and s2_name == 'N_0.64kg':
        include_fragments = False
        catastrophic = False

    # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
    gammas = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

    # Create a list of source sinks, first two are the active species
    source_sinks = [s1, s2]

    # Implementing logic for gammas calculations based on species properties
    # if s1.maneuverable and s2.maneuverable:
    #     # Multiplying each element in the first column of gammas by the product of alpha_active values
    #     gammas[:, 0] = gammas[:, 0] * s1.alpha_active * s2.alpha_active
    #     if s1.slotted and s2.slotted:
    #         # Applying the minimum slotting effectiveness if both are slotted
    #         gammas[:, 0] = gammas[:, 0] * min(s1.slotting_effectiveness, s2.slotting_effectiveness)

    # elif (s1.maneuverable and not s2.maneuverable) or (s2.maneuverable and not s1.maneuverable):
    #     if s1.trackable and s2.maneuverable:
    #         gammas[:, 0] = gammas[:, 0] * s2.alpha
    #     elif s2.trackable and s1.maneuverable:
    #         gammas[:, 0] = gammas[:, 0] * s1.alpha

    # # Applying symmetric loss to both colliding species
    # gammas[:, 1] = gammas[:, 0]

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
    frags_made = np.zeros((len(scen_properties.v_imp_all), len(debris_species)))
    # alt_nums = np.zeros((scen_properties.n_shells * 2, len(debris_species)))
    # Alt nums is now a 3D array, the first dimension is the collision shells, the second dimension is the debris species, 
    # and the third dimension is where the fragments end up. 
    alt_nums = np.zeros((scen_properties.n_shells, len(debris_species), scen_properties.n_shells))

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
        # Initialize a 3D matrix: [source_shell, debris_species, destination_shell]
        alt_nums_3d = np.zeros((scen_properties.n_shells, len(debris_species), scen_properties.n_shells))
        
        for dv_index, dv in enumerate(scen_properties.v_imp_all): # This is the case for circular orbits
            dv1 = scen_properties.v_imp_all[dv_index]
            dv2 = dv1
            try:
                results = evolve_bins_circular(m1, m2, r1, r2, dv1, dv2, [], binE_mass, [], LBgiven, RBflag, source_sinks, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km)
                frags_made[dv_index, :] = results[0] # nums is the number of fragments related to the shell of dv_index (same shell)
                
                # results[3]
                velocity_fragment_data = results[3]  # Shape: (n_shells * 2 - 1, debris_species)
                
                # Map velocity bins to destination shells
                # The middle bin represents the collision shell
                # Bins below middle represent shells below collision shell
                # Bins above middle represent shells above collision shell
                
                n_velocity_bins = velocity_fragment_data.shape[0]
                middle_bin_idx = n_velocity_bins // 2  # Middle bin index
                
                for vel_bin_idx in range(n_velocity_bins):
                    # Calculate destination shell index
                    dest_shell_offset = vel_bin_idx - middle_bin_idx
                    dest_shell = dv_index + dest_shell_offset
                    
                    # Only include valid shell indices
                    if 0 <= dest_shell < scen_properties.n_shells:
                        alt_nums_3d[dv_index, :, dest_shell] += velocity_fragment_data[vel_bin_idx, :]
                        
            except IndexError as ie:
                continue
            except ValueError as e:
                continue
    
    else:
        #########
        # Basic SSEM 
        #########
        for dv_index, dv in enumerate(scen_properties.v_imp_all): # This is the case for circular orbits 
            # dv1, dv2 = 7.5, 7.5 # for now we are going to assume the same velocity. 
            dv1 = scen_properties.v_imp_all[dv_index]
            dv2 = dv1
            try:
                results = evolve_bins_circular(m1, m2, r1, r2, dv1, dv2, [], binE_mass, [], LBgiven, RBflag, source_sinks, scen_properties.fragment_spreading, scen_properties.n_shells, scen_properties.R0_km, catastrophic=catastrophic)
                # frags_made[dv_index, :] = results[0]
                if not include_fragments:
                    frags_made[dv_index, :] = 0
                else:
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
        return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, fragsMadeDV_3d=alt_nums_3d, model_type='fragment_spreading')
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


    # for the iadc paper, there are few rules:
    # Collisions between satellites and between satellites and large debris were catastrophic.
    # •
    # Collisions between medium debris and satellites always led to non-catastrophic collisions: 
    # in that case, it was assumed that no fragments were generated but that satellites were disabled 
    # and lost their ability to perform CAMs and PMD maneuvers.
    # •
    # Collisions between small debris and satellites or large debris exhibited extremely low specific energy.
    # •
    # All other collisions between debris led to catastrophic collisions. 
    # The number of fragments created in these collisions were calculated accordingly based on 
    # NASA SBM but since they are usually not modeled or ignored in other debris models, 
    # these catastrophic collisions were discounted for comparability.

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

    # Now we need to add a include_fragments flag to the SpeciesPairClass objects.
    # large debris is N_446kg and N_32kg
    # medium debris is N_0.64kg 
    # small debris is N_7.85e-06kg and N_0.000785kg
    
    for species_pair_class in species_pairs_classes:
        species_pair_class.include_fragments = False
        s1_name = species_pair_class.species1.sym_name
        s2_name = species_pair_class.species2.sym_name
        
        # Collisions between satellites and between satellites and large debris were catastrophic.
        if s1_name == 'S' and s2_name in ['S', 'N_446kg', 'N_32kg']:
            species_pair_class.include_fragments = True
        # if a combination of large debris, then include fragments
        if s1_name in ['N_446kg', 'N_32kg'] and s2_name in ['N_446kg', 'N_32kg']:
            species_pair_class.include_fragments = True
        # if a combination of medium debris, then include fragments
        if s1_name == 'N_0.64kg' and s2_name == 'N_0.64kg':
            species_pair_class.include_fragments = True

    return species_pairs_classes

