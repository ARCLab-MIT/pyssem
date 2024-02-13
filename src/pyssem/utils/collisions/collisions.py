from itertools import combinations
from sympy import symbols
import numpy as np
#from tqdm import tqdm

import numpy as np

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

def evolve_bins(m1, m2, r1, r2, dv, binC, binE, binW, LBdiam, RBflag = 0, sto=1):
    # Super sampling ratio
    SS = 10

    # Define and validate the bin edges
    if binC is not None and binE is None and binW is None:
        # Bin centers are given, but not edges or widths
        LBm = binC[0] - (binC[1] - binC[0]) / 2
        UBm = binC[-1] + (binC[-1] - binC[-2]) / 2
        binEd = np.concatenate(([LBm], (binC[:-1] + binC[1:]) / 2, [UBm]))
    
    # bin width is given, but gaps exist
    elif binC is not None and binE is not None and binE is None:
        binEd1 = binC - binW / 2
        binEd2 = binC + binW / 2
        binEd = np.concatenate(([binEd1[0]], binEd2))
    
    # bin edges are given
    elif binE is not None and binC is None and binW is None:
        binEd = np.sort(binE)

    else:
        raise ValueError('Wrong setup for bins given')

    LB = LBdiam
    objclass = 5 if RBflag == 0 else 0

    # Ensure that m1 > m2
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

    # what is this code doing?
    num = (0.1 * M ** 0.75 * LB ** (-1.71)) - (0.1 * M ** 0.75 * min(1, 2 * r1) ** (-1.71))
    numSS = SS * num

    if numSS == 0:  # check if 0 (e.g., if LB = r1)
        return np.zeros(len(binEd) - 1), isCatastrophic, []
    
    # Create PDF of power law distribution, then sample 'num' selections
    # Only up to 1m, then randomly sample larger objects as quoted above

    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * r1)), 500)
    dd_means = 10 ** (np.log10(dd_edges[:-1]) + np.diff(np.log10(dd_edges)) / 2)
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)  # Cumulative distribution
    ndd = np.maximum(0, -np.diff(nddcdf))
    d_pdf = np.repeat(dd_means, np.floor(ndd) + (np.random.rand(len(ndd)) > (1 - (ndd - np.floor(ndd)))))

    try:
        dss = d_pdf[np.random.randint(0, len(d_pdf), size=int(np.ceil(numSS)))]
    except ValueError:
        dss = 0

    # Calculate the mass of objects
    A = 0.556945 * dss ** 2.0047077
    Am = 1
    m = A/Am


def create_collision_pairs(scen_properties):
    
    # Get the binomial coefficient of the species
    # This returns all possible combinations of the species
    species = scen_properties.species
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    species_pairs_classes = [] 
    n_f = symbols('n_f:{0}'.format(scen_properties.n_shells))

    # Debris species
    debris_species = [s for s in scen_properties.species if not s.active and s.RBflag != 1]

    # not sure what this code does 
    # if nargin < 2
    #     N_species = scen_properties.species_cell.N;
    # end

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

    for i, (s1, s2) in enumerate(species_pairs):
        # Get names
        n1, n2 = s1.sym_name, s2.sym_name

        gammas = -np.ones((scen_properties.n_shells, 2), dtype='object')  # SymPy symbols can be used as dtype=object

        source_sinks = [s1, s2]

        if s1.maneuverable and s2.maneuverable:
            # Both species are maneuverable
            gammas[:, 1] = gammas[:, 1] * s1.alpha_active * s2.alpha_active
            if s1.slotted and s2.slotted:
                # Both species are slotted
                gammas[:, 1] = gammas[:, 1] * min(s1.slotting_effectiveness, s2.slotting_effectiveness)
        
        elif s1.maneuverable and not s2.maneuverable or s2.maneuverable and not s1.maneuverable:
            if s1.trackable and s2.maneuverable:
                gammas[:, 1] = gammas[:, 1] * s2.alpha
            elif s2.trackable and s1.maneuverable:
                gammas[:, 1] = gammas[:, 1] * s1.alpha

        # The gamma burden is symmetric lost to both colliding species
        gammas[:, 2] = gammas[:, 1]            
    
        # Find the debris generation for each debris class from S1-S2 collision
        RBflag = max(s1.RBflag, s2.RBflag)

        # Initialise empty array. Rows = altitudes with different dv values
        # Cols = debris species in order of debris species list
        frags_made = np.zeros(len(scen_properties.v_imp2, len(debris_species)))
        is_catastrophic = np.zeros(1, len(debris_species))

        for dv_index in range(len(scen_properties.v_imp2)):
            dv = scen_properties.v_imp2[dv_index]
            [frags_made[dv_index, :], is_catastrophic[dv_index]] = evolve_bins(m1, m2, r1, r2, dv, binC, binE, binW, LBdiam, RBflag, sto)




    
    



