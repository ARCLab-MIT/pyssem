import numpy as np
from ..collisions.rv2coe_vec import rv2coe_vec 
# when testing locally use
# from rv2coe_vec import rv2coe_vec 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


# def func_Am(d, ObjClass):
#     """
#     Calculates the area-to-mass ratio for spacecraft fragments based on NASA's new breakup model of evolve 4.0.
    
#     Parameters:
#     d : ndarray
#         Array of diameters in meters.
#     ObjClass : int or float
#         Object class indicating whether the object is a rocket body or not.
    
#     Returns:
#     out : ndarray
#         Area-to-mass ratio for each fragment.
#     """
#     numObj = d.size
#     logds = np.log10(d)
#     amsms = np.nan * np.ones((numObj, 5))  # alpha, mu1, sigma1, mu2, sigma2

#     if 4.5 < ObjClass < 8.5:  # Rocket-body related
#         for ind, logd in enumerate(logds):
#             alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_for_rocket_body(logd)
#             amsms[ind, :] = [alpha, mu1, sigma1, mu2, sigma2]
#     else:  # Not rocket body
#         for ind, logd in enumerate(logds):
#             alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_not_rocket_body(logd)
#             amsms[ind, :] = [alpha, mu1, sigma1, mu2, sigma2]

#     N1 = amsms[:, 1] + amsms[:, 2] * np.random.randn(numObj)
#     N2 = amsms[:, 3] + amsms[:, 4] * np.random.randn(numObj)

#     out = 10 ** (amsms[:, 0] * N1 + (1 - amsms[:, 0]) * N2)

#     return out

def func_Am(d, ObjClass):
    """
    Calculates the area-to-mass ratio for spacecraft fragments based on NASA's new breakup model of evolve 4.0.
    
    Parameters:
    d : ndarray or float
        Array of diameters in meters (or a single diameter).
    ObjClass : int or float
        Object class indicating whether the object is a rocket body or not.
    
    Returns:
    out : ndarray
        Area-to-mass ratio for each fragment.
    """
    # --- ensure d is always an array, even if scalar ---
    d = np.atleast_1d(d).astype(float)

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
       mu = 0.9 * np.log10(Am) + 2.9 # Collision
    elif mode == 'exp':
        mu = 1.85 * np.log10(Am) + 1.85 # Explosion

    sigma = 0.4
    N = mu + sigma * np.random.randn(*np.shape(mu))
    z = 10 ** N # m/s
    return z 

def frag_col_SBM_vec_lc2(ep, p1_in, p2_in, param, LB):
    """
    Collision model following NASA EVOLVE 4.0 standard breakup model (2001)
    with the revision in ODQN "Proper Implementation of the 1998 NASA Breakup Model" (2011)

    The Debris output looks lke this:
    1. `a`: Semi-major axis (in km) - defines the size of the orbit.
    2. `ecco`: Eccentricity - describes the shape of the orbit (0 for circular, closer to 1 for highly elliptical).
    3. `inclo`: Inclination (in degrees) - the tilt of the orbit relative to the equatorial plane.
    4. `nodeo`: Right ascension of ascending node (in degrees) - describes the orientation of the orbit in the plane.
    5. `argpo`: Argument of perigee (in degrees) - the angle between the ascending node and the orbit's closest point to Earth.
    6. `mo`: Mean anomaly (in degrees) - the position of the debris along the orbit at a specific time.
    7. `bstar`: Drag coefficient - relates to atmospheric drag affecting the debris.
    8. `mass`: Mass of the debris fragment (in kg).
    9. `radius`: Radius of the debris fragment (in meters).
    10. `errors`: Error values (various metrics used for error tracking or uncertainty).
    11. `controlled`: A flag indicating whether the fragment is controlled (e.g., through active debris removal or a controlled reentry).
    12. `a_desired`: Desired semi-major axis (in km) - often used in mission planning.
    13. `missionlife`: Expected mission lifetime (in years).
    14. `constel`: Identifier for the constellation to which the debris belongs (if any).
    15. `date_created`: Date the fragment data was created (format depends on implementation, e.g., `YYYY-MM-DD`).
    16. `launch_date`: Date of launch of the parent object (format depends on implementation).
    17. `r[idx_a]`: Position vector (in km) - describes the 3D position of the fragment in space.
    18. `v[idx_a]`: Velocity vector (in km/s) - describes the 3D velocity of the fragment.
    19. `frag_objectclass`: Object class - a dimensionless identifier describing the type of fragment or debris.
    20. `ID_frag`: Unique identifier for the fragment (could be a string or numeric ID).
    
    Parameters:
    - ep: Epoch
    - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
    - LB: Lower bound for fragment sizes (meters)
    
    Returns:
    - debris1: Array containing debris fragments from parent 1
    - debris2: Array containing debris fragments from parent 2
    - isCatastrophic: Boolean indicating if the collision is catastrophic
    """
    
    # Ensure p1_mass > p2_mass, or p1_radius > p2_radius if p1_mass == p2_mass
    if p1_in[0] < p2_in[0] or (p1_in[0] == p2_in[0] and p1_in[1] < p2_in[1]):
        p1_in, p2_in = p2_in, p1_in
    
    # Extract parameters for p1 and p2
    p1_mass, p1_radius = p1_in[0], p1_in[1]
    p1_r, p1_v = p1_in[2:5], p1_in[5:8]
    p1_objclass = p1_in[8]
    
    p2_mass, p2_radius = p2_in[0], p2_in[1]
    p2_r, p2_v = p2_in[2:5], p2_in[5:8]
    p2_objclass = p2_in[8]
    
    # Compute relative velocity (dv) and catastrophic ratio - 
    dv = np.linalg.norm(np.array(p1_v) - np.array(p2_v))  # km/s
    catastroph_ratio = (p2_mass * (dv * 1000) ** 2) / (2 * p1_mass * 1000)  # J/g
    
    # If the specific energy is < 40 J/g: non-catastrophic collision
    if catastroph_ratio < 40:
        M = p2_mass * dv ** 2 # power law unitl M, then rest is disposted into one larger fragment from larger parent
        isCatastrophic = False
    else: # Catastrophic collision
        M = p1_mass + p2_mass
        isCatastrophic = True
    
    # Create debris size distribution
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * p1_radius)), 200) # log space, up to either 1m or diameter of larger satellite
    log10_dd = np.log10(dd_edges) # log10 of diameter edge bins
    dd_means = 10 ** (log10_dd[:-1] + np.diff(log10_dd) / 2) # mean value of each diameter edge bin (in log scale as fragments are in log scale)
    
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71) # cumulative distribution fo collision (eq. 2.68)
    ndd = np.maximum(0, -np.diff(nddcdf)) # diff to get PDF count for the bins - if negative set to zero
    floor_ndd = np.floor(ndd).astype(int) # floor of PDF count for each bin
    rand_sampling = np.random.rand(len(ndd))  # 0-1, random number for stochastic sampling
    add_sampling = rand_sampling > (1 - (ndd - floor_ndd)) # 0 if rand number is lower than 1 (decimal part of ndd), 1 if randomnumber is greater
    d_pdf = np.repeat(dd_means, floor_ndd + add_sampling.astype(int)) # PDF of debris objects between LB and 1m 
    
    # Shuffle the diameters
    d = np.random.permutation(d_pdf) # do not limit number of fragments to be equal to 'num'
    
    # Calculate mass of fragments
    A = 0.556945 * d ** 2.0047077 # calculate area, eq 2.72
    Am = func_Am(d, p1_objclass) # us Am converstion of larger object
    m = A / Am
    
    # print(f"Number of fragments: {len(m)}")
    
    # Initialize remnant indices
    idx_rem1 = np.array([], dtype=int)
    idx_rem2 = np.array([], dtype=int)
    
    # Handle fragment mass allocation
    if np.sum(m) < M:
        if isCatastrophic:
            # Catastrophic fragment handling
            largeidx = (d > 2 * p2_radius) & (d < 2 * p1_radius)
            m_assigned_large = max(0, np.sum(m[largeidx]))
    
            if m_assigned_large > p1_mass:
                idx_large = np.where(largeidx)[0]
                dord1 = np.argsort(m[idx_large])
                cumsum_m1 = np.cumsum(m[idx_large[dord1]])
                indices = np.where(cumsum_m1 < p1_mass)[0]
                if indices.size > 0:
                    lastidx1 = indices[-1]
                    to_remove = idx_large[dord1[lastidx1 + 1:]]
                    m = np.delete(m, to_remove)
                    d = np.delete(d, to_remove)
                    A = np.delete(A, to_remove)
                    Am = np.delete(Am, to_remove)
                    largeidx = np.delete(largeidx, to_remove)
                    m_assigned_large = cumsum_m1[lastidx1]
                else:
                    m_assigned_large = 0
    
            mass_max_small = min(p2_mass, m_assigned_large)
    
            smallidx_temp = np.where(~largeidx)[0]
            dord = np.argsort(m[smallidx_temp])
            cumsum_m = np.cumsum(m[smallidx_temp[dord]])
            indices = np.where(cumsum_m <= mass_max_small)[0]
            if indices.size > 0:
                lastidx_small = indices[-1]
                small_indices = smallidx_temp[dord[:lastidx_small + 1]]
                smallidx = np.zeros(len(d), dtype=bool)
                smallidx[small_indices] = True
                m_assigned_small = max(0, np.sum(m[smallidx]))
            else:
                m_assigned_small = 0
                smallidx = np.zeros(len(d), dtype=bool)
    
            m_remaining_large = p1_mass - m_assigned_large
            m_remaining_small = p2_mass - m_assigned_small
            m_remaining = [m_remaining_large, m_remaining_small]
    
            # Handle remnant mass distribution
            m_remSum = M - np.sum(m)
            remDist = np.random.rand(np.random.randint(2, 9))
            m_rem_temp = m_remSum * remDist / np.sum(remDist)
            num_rem = len(m_rem_temp)
    
            m_rem_sort = np.sort(m_rem_temp)[::-1]
            rem_temp_ordered = 1 + np.round(np.random.rand(num_rem)).astype(int)
    
            for i_rem in range(num_rem):
                if rem_temp_ordered[i_rem] == 1:
                    idx = 0
                else:
                    idx = 1
                if m_rem_sort[i_rem] > m_remaining[idx]:
                    diff_mass = m_rem_sort[i_rem] - m_remaining[idx]
                    m_rem_sort[i_rem] = m_remaining[idx]
                    m_remaining[idx] = 0
                    if i_rem == num_rem - 1:
                        m_rem_sort = np.append(m_rem_sort, diff_mass)
                        rem_temp_ordered = np.append(rem_temp_ordered, 2 if idx == 0 else 1)
                        num_rem += 1
                    else:
                        m_rem_sort[i_rem + 1:] += diff_mass / (num_rem - i_rem - 1)
                else:
                    m_remaining[idx] -= m_rem_sort[i_rem]
    
            m_rem = m_rem_sort
            d_rem_approx = np.zeros_like(m_rem)
            rem1_temp = rem_temp_ordered == 1
            d_rem_approx[rem1_temp] = ((m_rem[rem1_temp] / p1_mass * p1_radius ** 3) ** (1 / 3)) * 2
            d_rem_approx[~rem1_temp] = ((m_rem[~rem1_temp] / p2_mass * p2_radius ** 3) ** (1 / 3)) * 2
            Am_rem = func_Am(d_rem_approx, p1_objclass)
            A_rem = m_rem * Am_rem
            d_rem = d_rem_approx
        else:
            # Non-catastrophic collision fragmnents are deposited from 1mm upward, until the total mass, M = Mp * v_i^2 is achieved, 
            # The final framgnet is deposited in a sinle massive fragnment reminiscen of the createred target mass. 
            # assign remnant mass > d > A > Am > dv

            # Identify “large” fragments: too big for the smaller satellite but smaller than the larger one
            large_mask = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)

            # Mass of fragments tentatively assigned to the larger satellite
            m_assigned_large = max(0.0, m[large_mask].sum())

            # If that exceeds the larger satellite’s mass, prune the smallest fragments until it fits
            if m_assigned_large > p1_mass:
                # get indices of those large fragments
                idx_large = np.where(large_mask)[0]
                # sort those indices by their fragment mass m[idx]
                order = idx_large[np.argsort(m[idx_large])]
                # cumulative mass in ascending order
                csum = np.cumsum(m[order])
                # find last position where cumulative sum is still below p1_mass
                valid = np.where(csum < p1_mass)[0]
                if valid.size > 0:
                    last = valid[-1]
                    # remove the “overflow” fragments
                    to_remove = order[last+1:]
                    m    = np.delete(m,    to_remove)
                    d    = np.delete(d,    to_remove)
                    A    = np.delete(A,    to_remove)
                    Am   = np.delete(Am,   to_remove)
                    large_mask = np.delete(large_mask, to_remove)
                    # update assigned mass
                    m_assigned_large = csum[last]
                else:
                    # nothing fits, zero it out
                    m_assigned_large = 0.0

            # remaining mass capacity in the larger satellite
            m_remaining_large = p1_mass - m_assigned_large

            # “small” fragments: any fragment bigger than the large‐sat diameter or unassigned
            small_mask = (d > 2 * p1_radius) & (~large_mask)

            # remaining mass to distribute
            m_rem_sum = M - m.sum()

            if m_rem_sum > m_remaining_large:
                # if remaining mass exceeds the larger satellite’s capacity,
                # split it into two remnant fragments, one for each parent
                m_rem1 = m_remaining_large   # remnant mass for the larger satellite
                # approximate diameter assuming constant density ∝ mass / radius³
                d_rem_approx1 = (m_rem1 / p1_mass * p1_radius**3)**(1/3) * 2

                m_rem2 = m_rem_sum - m_remaining_large  # remnant mass for the smaller satellite
                d_rem_approx2 = (m_rem2 / p2_mass * p2_radius**3)**(1/3) * 2

                # concatenate remnant diameters and masses
                d_rem = np.array([d_rem_approx1, d_rem_approx2])
                m_rem = np.array([m_rem1,         m_rem2])

                # compute A/m and area for each remnant fragment
                Am_rem = func_Am(d_rem, p1_objclass)   # A/m using same object class
                A_rem  = m_rem * Am_rem                # area = mass × (area/mass)

                # indices (in Python, 0-based) of which remnant goes to which parent
                idx_rem1 = 0  # first element → larger satellite
                idx_rem2 = 1  # second element → smaller satellite

            else:
                # all remaining mass fits under the larger satellite
                m_rem = m_rem_sum
                d_rem = np.array([(m_rem / p1_mass * p1_radius**3)**(1/3) * 2])

                Am_rem = func_Am(d_rem, p1_objclass)
                A_rem  = m_rem * Am_rem

                idx_rem1 = 0   # only one remnant goes to larger satellite
                idx_rem2 = None  # no remnant for smaller satellite

            # remove any remnant fragments that are below the characteristic length LB
            # *and* contribute less than 0.1% of M
            too_small_mask = (d_rem < LB) & (m_rem < M / 1000.0)
            if np.any(too_small_mask):
                # drop them from all arrays
                d_rem  = d_rem[~too_small_mask]
                m_rem  = m_rem[~too_small_mask]
                A_rem  = A_rem[~too_small_mask]
                Am_rem = Am_rem[~too_small_mask]

                removed = np.where(too_small_mask)[0]
                # if the first remnant was removed, clear idx_rem1
                if 0 in removed:
                    idx_rem1 = None
                # if the second remnant was removed, clear idx_rem2
                if 1 in removed:
                    idx_rem2 = None
    else:
        dord = np.argsort(m)  # indices that would sort m ascending
        cumsum_m = np.cumsum(m[dord])  # cumulative sum of sorted masses

        # find last index where cumulative mass < M
        below = np.where(cumsum_m < M)[0]
        if below.size > 0:
            lastidx = below[-1]  # last valid index
            cumsum_lower = max(0.0, cumsum_m[lastidx])  # cumulative mass below M
            valididx = dord[: lastidx + 1]  # indices of fragments to keep
        else:
            lastidx = None
            cumsum_lower = 0.0
            valididx = np.array([], dtype=int)

        # select only those fragments
        m  = m[valididx]
        d  = d[valididx]
        A  = A[valididx]
        Am = Am[valididx]

        # assign “large” vs “small” based on diameter and mass
        largeidx = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)
        smallidx = (d > 2 * p1_radius) & (~largeidx)

        # --- Check if there is mass remaining, and generate an additional fragment if needed ---
        m_rem = M - cumsum_lower  # remaining mass to accumulate to M

        if m_rem > M / 1000.0:  # if the remaining mass is larger than 0.1% of M
            # determine how to assign the extra fragment
            mass_small_assigned = m[smallidx].sum() if smallidx.any() else 0.0
            mass_large_assigned = m[largeidx].sum() if largeidx.any() else 0.0

            if m_rem > (p2_mass - mass_small_assigned):
                rand_assign_frag = 1  # assign to larger satellite
            elif m_rem > (p1_mass - mass_large_assigned):
                rand_assign_frag = 2  # assign to smaller satellite
            else:
                rand_assign_frag = 1 + int(round(np.random.rand()))  # 1 or 2 randomly

            # compute approximate diameter for the remnant fragment
            if rand_assign_frag == 1:
                # larger satellite
                d_rem_approx = (m_rem / p1_mass * p1_radius**3)**(1/3) * 2
                idx_rem1 = 0
                idx_rem2 = None
            else:
                # smaller satellite
                d_rem_approx = (m_rem / p2_mass * p2_radius**3)**(1/3) * 2
                idx_rem1 = None
                idx_rem2 = 0

            # compute its A/m and area
            Am_rem = func_Am(np.atleast_1d(d_rem_approx), p1_objclass)
            A_rem  = m_rem * Am_rem
            d_rem  = np.atleast_1d(d_rem_approx)

            # remove if below lower bound and negligible mass
            if (d_rem < LB).all() and (m_rem < M / 1000.0):
                d_rem   = np.array([])
                A_rem   = np.array([])
                Am_rem  = np.array([])
                m_rem   = np.array([])
                idx_rem1 = None
                idx_rem2 = None

        else:
            # no extra fragment needed
            d_rem   = np.array([])
            A_rem   = np.array([])
            Am_rem  = np.array([])
            m_rem   = np.array([])
            idx_rem1 = None
            idx_rem2 = None
    
    # Assign dv to random directions; create samples on unit sphere
    # concatenate Am and Am_rem into one array for func_dv
    dv = func_dv(np.concatenate((Am, Am_rem)), 'col') / 1000.0  # km/s

    # number of fragments
    N = dv.size

    # sample random points on the sphere
    u     = np.random.rand(N) * 2 - 1           # uniform in [-1, 1]
    theta = np.random.rand(N) * 2 * np.pi       # uniform in [0, 2π)

    # compute spherical coordinates
    v = np.sqrt(1 - u**2)
    p = np.column_stack((v * np.cos(theta),
                        v * np.sin(theta),
                        u))                       # shape (N, 3)

    # scale directions by dv to get dv_vec
    try:
        dv_vec = p * dv[:, None]                 # shape (N, 3)
    except Exception:
        return  # if inside a function, bail out on error
    
    # --- handle the non-catastrophic remnant append ---
    if not isCatastrophic:
        # find remnant mass of parent object; add to *_rem object
        p1remnantmass = p1_mass + p2_mass - (np.sum(m) + np.sum(np.atleast_1d(m_rem))) 
        m_rem = np.concatenate((np.atleast_1d(m_rem), [p1remnantmass]))
        d_rem = np.concatenate((np.atleast_1d(d_rem), [2 * p1_radius]))
        A_rem = np.concatenate((np.atleast_1d(A_rem), [np.pi * p1_radius**2]))
        Am_rem = np.concatenate((np.atleast_1d(Am_rem), [np.pi * p1_radius**2 / p1remnantmass]))
        dv = np.concatenate((dv, [0.0])) # nonchange from parent object
        dv_vec = np.vstack((dv_vec, [0.0, 0.0, 0.0]))

    # combine m and m_rem as 1D arrays to avoid dimension mismatch
    total_debris = np.sum(np.concatenate((np.atleast_1d(m), np.atleast_1d(m_rem))))
    if abs(total_debris - (p1_mass + p2_mass)) > M * 0.05:
        print(f"Warning: Total sum of debris mass ({total_debris:.1f} kg) "
            f"differs from mass of original objects ({p1_mass + p2_mass:.1f} kg)")

    # ensure all fragment arrays are 1D before stacking
    d_all   = np.concatenate((np.atleast_1d(d),    np.atleast_1d(d_rem)))
    A_all   = np.concatenate((np.atleast_1d(A),    np.atleast_1d(A_rem)))
    Am_all  = np.concatenate((np.atleast_1d(Am),   np.atleast_1d(Am_rem)))
    m_all   = np.concatenate((np.atleast_1d(m),    np.atleast_1d(m_rem)))
    # dv and dv_vec are already 1D or 2D with correct shape
    fragments = np.column_stack([
        d_all,
        A_all,
        Am_all,
        m_all,
        np.atleast_1d(dv),
        np.atleast_1d(dv_vec[:, 0]),
        np.atleast_1d(dv_vec[:, 1]),
        np.atleast_1d(dv_vec[:, 2]),
    ])

    # Distribute fragments amongst parent 1 and parent 2
    if isCatastrophic:
        # Initialize assignment arrays
        largeidx_all = (fragments[:, 0] > 2 * p2_radius) & (fragments[:, 0] < 2 * p1_radius)
        smallidx_all = (fragments[:, 0] > 2 * p1_radius) & (~largeidx_all)

        # Assign remnant fragments if any
        if 'idx_rem1' in locals() and isinstance(idx_rem1, (list, np.ndarray)) and len(idx_rem1) > 0:
            largeidx_all = largeidx_all | np.isin(np.arange(len(fragments)), idx_rem1)
        if 'idx_rem2' in locals() and isinstance(idx_rem2, (list, np.ndarray)) and len(idx_rem2) > 0:
            smallidx_all = smallidx_all | np.isin(np.arange(len(fragments)), idx_rem2)

        assignedidx = largeidx_all | smallidx_all
        idx_unassigned = np.where(~assignedidx)[0]

        # Assign fragments to p1 until p1_mass is filled
        fragments1 = fragments[largeidx_all, :]
        fragments2 = fragments[smallidx_all, :]

        if len(idx_unassigned) > 0:
            fragments_unassigned = fragments[idx_unassigned, :]
            cum_mass_p1 = np.sum(fragments1[:, 3])
            cum_mass_p1 += np.cumsum(fragments_unassigned[:, 3])
            p1_assign = cum_mass_p1 <= p1_mass
            p1indx = np.where(p1_assign)[0]
            fragments1 = np.vstack([fragments1, fragments_unassigned[:len(p1indx), :]])
            fragments2 = np.vstack([fragments2, fragments_unassigned[len(p1indx):, :]])
    else:
        # Non-catastrophic collision: assign largest fragment to p1, others to p2
        if fragments.shape[0] > 0:
            heaviestInd = fragments[:, 3] == np.max(fragments[:, 3])
            lighterInd = ~heaviestInd
            fragments1 = fragments[heaviestInd, :]
            fragments2 = fragments[lighterInd, :]
        else:
            fragments1 = np.array([])
            fragments2 = np.array([])
    
    # Remove fragments smaller than LB
    if fragments1.size > 0:
        fragments1 = fragments1[fragments1[:, 0] >= LB]
    if fragments2.size > 0:
        fragments2 = fragments2[fragments2[:, 0] >= LB]
    
    # Create debris objects
    debris1 = func_create_tlesv2_vec(ep, p1_r, p1_v, p1_objclass, fragments1, param)
    # param['maxID'] += debris1.shape[0]
    debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, fragments2, param)
    # param['maxID'] += debris2.shape[0]
    
    return debris1, debris2, isCatastrophic

def func_create_tlesv2_vec(ep, r_parent, v_parent, class_parent, fragments, param):
    """
    Create new satellite objects from fragmentation information.

    Parameters:
    - ep: Epoch
    - r_parent: Parent satellite position vector [x, y, z]
    - v_parent: Parent satellite velocity vector [vx, vy, vz]
    - class_parent: Parent satellite object class
    - fragments: Nx8 array containing fragment data:
        [diameter, Area, AMR, mass, total_dv, dv_X, dv_Y, dv_Z]
    - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID'

    Returns:
    - mat_frag: Array with fragment orbital elements and related data
    """

    max_frag = param.get('max_frag', float('inf'))
    mu = param.get('mu')
    req = param.get('req')
    maxID = param.get('maxID', 0)

    r0 = np.array(r_parent)
    v0 = np.array(v_parent)

    num_fragments = fragments.shape[0]

    if num_fragments > max_frag:
        print(f"Warning: number of fragments {num_fragments} exceeds max_frag {max_frag}")
    n_frag = min(num_fragments, max_frag)

    # Sort fragments by mass in descending order
    sort_idx = np.argsort(-fragments[:, 3])  # mass is in column 4 (index 3)
    fragments = fragments[sort_idx[:n_frag], :]

    # Compute new velocities by adding dv to parent velocity
    v = np.column_stack((
        fragments[:, 5] + v0[0],
        fragments[:, 6] + v0[1],
        fragments[:, 7] + v0[2]
    ))

    # Positions are the same as parent position
    r = np.tile(r0, (n_frag, 1))

    # Compute orbital elements from position and velocity vectors
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)

    # Initialize lists to store orbital elements
    a_list = []
    ecc_list = []
    incl_list = []
    nodeo_list = []
    argpo_list = []
    mo_list = []

    # Loop through each fragment and compute orbital elements
    for i in range(n_frag):
        p,a,ecc,incl,omega,argp,nu,m,arglat,truelon,lonper = rv2coe_vec(r[i], v[i], 398600.5) # mu in km^3/s^2
        a_list.append(a)
        ecc_list.append(ecc)
        incl_list.append(incl)
        nodeo_list.append(nu)
        argpo_list.append(argp)
        mo_list.append(m)

    # Convert lists to arrays
    a = np.array(a_list)
    ecc = np.array(ecc_list)
    incl = np.array(incl_list)
    nodeo = np.array(nodeo_list)
    argpo = np.array(argpo_list)
    mo = np.array(mo_list)

    # Process only valid orbits (a > 0)
    idx_a = np.where(a > 0)[0]
    num_a = len(idx_a)

    a = a[idx_a] / req
    ecco = ecc[idx_a]
    inclo = incl[idx_a]
    nodeo = nodeo[idx_a]
    argpo = argpo[idx_a]
    mo = mo[idx_a]

    # Bstar parameter
    rho_0 = 0.157  # kg/(m^2 * Re)
    A_M = fragments[idx_a, 2]  # AMR is in column 3 (index 2)
    bstar = (0.5 * 2.2 * rho_0) * A_M  # Bstar in units of 1/Re

    mass = fragments[idx_a, 3]  # mass is in column 4 (index 3)
    radius = fragments[idx_a, 0] / 2  # diameter is in column 1 (index 0)

    errors = np.zeros(num_a)
    controlled = np.zeros(num_a)
    a_desired = np.full(num_a, np.nan)
    missionlife = np.full(num_a, np.nan)
    constel = np.zeros(num_a)

    date_created = np.full(num_a, ep)
    launch_date = np.full(num_a, np.nan)

    frag_objectclass = np.full(num_a, filter_objclass_fragments_int(class_parent))

    ID_frag = np.arange(maxID + 1, maxID + num_a + 1)

    # Assemble mat_frag
    mat_frag = np.column_stack((
        a, ecco, inclo, nodeo, argpo, mo, bstar, mass, radius,
        errors, controlled, a_desired, missionlife, constel,
        date_created, launch_date, r[idx_a], v[idx_a], frag_objectclass, ID_frag
    ))

    return mat_frag



def filter_objclass_fragments_int(class_parent):
    """
    Assign object class to fragments according to the parent particle.

    For this implementation, we'll assume fragments inherit the parent's class.
    """
    return class_parent

# Example usage
if __name__ == "__main__":    
    # Define the param dictionary
    param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }

    # # Lower bound (LB)
    LB = 0.1  # Assuming this is the lower bound in meters

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # === Helper functions ===
    def perifocal_r_and_v(a, e, nu, mu):
        r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
        r = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

        h = np.sqrt(mu * a * (1 - e**2))
        v = (mu / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
        return r, v

    def rotate_vector_45_deg_in_plane(v):
        theta = np.pi / 4  # 45 degrees
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return R @ v

    def test_fragmentation_by_sma_new(frag_col_func, param, LB):
        """
        Run frag_col_SBM_vec_lc2 at different SMA altitudes using realistic r and v vectors
        derived from orbital mechanics. Object 1 uses a base orbit, object 2 impacts at 45 degrees
        using same position but rotated velocity vector.

        Generates a grid of 2D histograms with dynamic subplot layout (max 4 columns).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import math

        # SMA test values (Earth radius + altitude in km)
        sma_tests = [6671, 6871, 6971, 7171, 7371, 7571, 7771, 7971]
        earth_radius_km = 6371
        # m1=148, m2=4, r1=0.5, r2=0.5, sma1=7253.1366, sma2=7253.1366, e1=0.4, e2=0.4
        # m1=1783.94, m2=4, r1=2.687936011, r2=0.5, sma1=7403.1366, sma2=7403.1366, e1=0.2005, e2=0.2005
        m1=1783.94
        m2=4
        rad1=2.687936011
        rad2=0.4
        sma1=7403.1366
        sma2=7403.1366
        ecc_1=0.2005
        ecc_2=0.2005
        # ecc_1 = 0   # Eccentricity of object 1
        # ecc_2 = 0  # Eccentricity of object 2
        true_anomaly_deg = 90
        TA = np.radians(true_anomaly_deg)
        mu = param["mu"]

    
        results = []

        for sma in sma_tests:
            # Object 1 (larger mass): low eccentricity
            r1, v1 = perifocal_r_and_v(sma, ecc_2, TA, mu)

            # Object 2 (lighter mass): higher eccentricity, rotated velocity
            _, v2_mag_vec = perifocal_r_and_v(sma, ecc_1, TA, mu)
            v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
            v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
            r2 = r1  # collision point is same

            # Define input vectors
            p1 = np.array([m1, rad1, *r1, *v1, 1.0])
            p2 = np.array([m2, rad2, *r2, *v2, 1.0])

            debris1, debris2, isCatastrophic = frag_col_func(0, p1, p2, param, LB)

            if debris1.size == 0:
                print(f"[{sma} km] No debris generated")
                continue

            if debris2.size > 0:
                print(f"Debris generated at SMA {sma} km: {debris2.shape[0]} fragments from object 1, ")
                # concatenate rows of debris2 onto debris1
                # debris1 = np.concatenate((debris1, debris2), axis=0)
                # alternatively:
                debris1 = np.vstack([debris1, debris2])

            results.append((sma, debris1))

        # === Dynamic 2D Histogram Plotting ===
        n_cases = len(results)
        max_cols = 4
        n_cols = min(max_cols, n_cases)
        n_rows = math.ceil(n_cases / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
        axs = np.array(axs).reshape(-1)
        mappable = None

        for ax, (sma, debris) in zip(axs, results):
            sma_alt_km = (debris[:, 0] - 1) * earth_radius_km
            ecc = debris[:, 1]

            hist, xedges, yedges = np.histogram2d(
                sma_alt_km, ecc,
                bins=[np.arange(0, 5001, 100), np.arange(0, 1.01, 0.01)]
            )
            hist[hist == 0] = np.nan

            if np.isnan(hist).all():
                ax.set_title(f"SMA = {sma} km (No data)")
                continue

            mappable = ax.imshow(
                hist.T, origin='lower', norm=LogNorm(vmin=np.nanmin(hist), vmax=np.nanmax(hist)),
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto'
            )
            ax.set_title(f"SMA = {sma} km")
            ax.set_xlabel("SMA as altitude (km)")
            ax.set_ylabel("Eccentricity")
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 0.5)
            ax.grid(True)

        # Turn off any unused axes
        for ax in axs[len(results):]:
            ax.axis('off')

        if mappable:
            fig.colorbar(mappable, ax=axs.tolist(), label='Count')

        fig.suptitle("2D Histogram of SMA and Eccentricity for Varying Initial SMA", fontsize=16)
        plt.show()

    def test_one_sma(frag_col_func, param, LB):
        """
        Test fragmentation at a single SMA value.
        """
        earth_radius_km = 6371
        # m1=148, m2=4, r1=0.5, r2=0.5, sma1=7253.1366, sma2=7253.1366, e1=0.4, e2=0.4
        # m1=1783.94, m2=4, r1=2.687936011, r2=0.5, sma1=7403.1366, sma2=7403.1366, e1=0.2005, e2=0.2005
        # m1=148, m2=4, r1=0.5, r2=0.5, sma1=7703.1366, sma2=7703.1366s
        # for m1=148, m2=4, r1=0.5, r2=0.5, sma1=7643.1366, sma2=7643.1366, e1=0, e2=0.001
        m1=148
        m2=4
        rad1=0.5
        rad2=0.5
        sma1=7463.1366
        sma2=7463.1366
        ecc_1=0
        ecc_2=0.001
        # ecc_1 = 0   # Eccentricity of object 1
        # ecc_2 = 0  # Eccentricity of object 2
        true_anomaly_deg = 90
        TA = np.radians(true_anomaly_deg)
        mu = param["mu"]

        # Object 1 (larger mass): low eccentricity
        r1, v1 = perifocal_r_and_v(sma1, ecc_2, TA, mu)

        # Object 2 (lighter mass): higher eccentricity, rotated velocity
        _, v2_mag_vec = perifocal_r_and_v(sma2, ecc_1, TA, mu)
        v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
        v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
        r2 = r1  # collision point is same

        # Define input vectors
        p1 = np.array([m1, rad1, *r1, *v1, 1.0])
        p2 = np.array([m2, rad2, *r2, *v2, 1.0])

        debris1, debris2, isCatastrophic = frag_col_func(0, p1, p2, param, LB)

        if debris1.size == 0:
            print(f"[{sma1} km] No debris generated")

        print(f"Debris generated at SMA {sma1} km: {debris1.shape[0]} fragments")
        print(f"Debris generated at SMA {sma2} km: {debris2.shape[0]} fragments")

        # # sum together debris1 and debris2 plot sma and ecc
        # debris = np.vstack([debris1, debris2])
        # sma_alt_km = (debris[:, 0] - 1) * earth_radius_km
        # ecc = debris[:, 1]
        # plt.figure(figsize=(10, 6))
        # plt.scatter(sma_alt_km, ecc, alpha=0.5, s=10, c='blue')
        # plt.title(f"Debris Distribution at SMA {sma1} km")
        # plt.xlabel("SMA as altitude (km)")
        # plt.ylabel("Eccentricity") 
        # plt.show()


    # run 100 times
    for i in range(100):
        print(f"Test run {i+1}")
        test_one_sma(frag_col_SBM_vec_lc2, param, LB)
    # test_one_sma(frag_col_SBM_vec_lc2, param, LB)