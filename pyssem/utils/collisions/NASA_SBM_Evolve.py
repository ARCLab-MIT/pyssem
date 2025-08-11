import traceback
import numpy as np
import math
import random

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

def evolve_bins_elliptical(scen_properties, m1, m2, rad_1, rad_2, sma1, sma2, e1, e2, binE_mass, binE_ecc, collision_index, n_shells=0, RBflag=0):
    # param = {
    #     'req': 6.3781e+03,
    #     'mu': 3.9860e+05,
    #     'j2': 0.0011,
    #     'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
    #     'maxID': 0,
    #     'density_profile': 'static'
    # }

    # # Lower bound (LB)
    LB = 0.1
    # SS = 60
    true_anomaly_deg = 90
    TA = np.radians(true_anomaly_deg)
    mu = 3.9860e+05

    # Object 1 (larger mass): low eccentricity
    r1, v1 = perifocal_r_and_v(sma1, e2, TA, mu)

    # Object 2 (lighter mass): higher eccentricity, rotated velocity
    _, v2_mag_vec = perifocal_r_and_v(sma2, e1, TA, mu)
    v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
    v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
    r2 = r1  # collision point is same

    # Define input vectors
    # rocket bodies will always be the larger object, so if rbflag is set, object_type ==5
    if RBflag == 1:
        p1 = np.array([m1, rad_1, *r1, *v1, 5.0])
    else:
        p1 = np.array([m1, rad_1, *r1, *v1, 0.0])
    p2 = np.array([m2, rad_2, *r2, *v2, 1.0])

    try:
        debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1, p2, LB=LB)
        # debris will now come out in the format of [a, ecco, mass]
    except Exception as e:
        print(f"Error in frag_col_SBM_vec_lc2: {e} \n for m1={m1}, m2={m2}, r1={rad_1}, r2={rad_2}, sma1={sma1}, sma2={sma2}, e1={e1}, e2={e2}")
        traceback.print_exc()
        return None

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

    # print(frag_properties)

    binE_alt = scen_properties.semi_major_bins_km  # We add 1 for bin edges

    # return hist
    hist3d, _ = np.histogramdd(
        frag_properties,
        bins=[binE_alt, binE_mass, binE_ecc]
    )
    
    return hist3d

def frag_col_SBM_vec_lc2(ep, p1_in, p2_in, req=6371.0, max_frag=1000, LB=0.1):
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
                idx_rem1 = np.array([0], dtype=int)
                idx_rem2 = np.array([], dtype=int)

            else:
                # all remaining mass fits under the larger satellite
                m_rem = m_rem_sum
                d_rem = np.array([(m_rem / p1_mass * p1_radius**3)**(1/3) * 2])

                Am_rem = func_Am(d_rem, p1_objclass)
                A_rem  = m_rem * Am_rem

                idx_rem1 = np.array([0], dtype=int)
                idx_rem2 = np.array([], dtype=int)

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
                idx_rem1 = np.array([0], dtype=int)
                idx_rem2 = np.array([], dtype=int)
            else:
                # smaller satellite
                d_rem_approx = (m_rem / p2_mass * p2_radius**3)**(1/3) * 2
                idx_rem1 = np.array([], dtype=int)
                idx_rem2 = np.array([0], dtype=int)

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
                idx_rem1 = np.array([], dtype=int)
                idx_rem2 = np.array([], dtype=int)

        else:
            # no extra fragment needed
            d_rem   = np.array([])
            A_rem   = np.array([])
            Am_rem  = np.array([])
            m_rem   = np.array([])
            idx_rem1 = np.array([], dtype=int)
            idx_rem2 = np.array([], dtype=int)
        
    # Assign dv to random directions; create samples on unit sphere
    # concatenate Am and Am_rem into one array for func_dv
    # dv = func_dv(np.concatenate((Am, Am_rem)), 'col') / 1000.0  # km/s
    dv = np.array(func_dv(np.concatenate((Am, Am_rem)), 'col')) / 1000.0  # km/s

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
        print("Warning: Total sum of debris mass differs from mass of original objects")

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
        # Step 1: Initialize size-based assignment masks
        largeidx_all = (fragments[:, 0] > 2 * p2_radius) & (fragments[:, 0] < 2 * p1_radius)
        smallidx_all = (fragments[:, 0] > 2 * p1_radius) & (~largeidx_all)

        # Step 2: Add remnant fragment indices (must be arrays, even if empty)
        if idx_rem1.size > 0:
            largeidx_all |= np.isin(np.arange(len(fragments)), idx_rem1)

        if idx_rem2.size > 0:
            largeidx_all |= np.isin(np.arange(len(fragments)), idx_rem2)

        # Step 3: Determine assigned vs unassigned fragments
        assignedidx = largeidx_all | smallidx_all
        idx_unassigned = np.where(~assignedidx)[0]

        # Step 4: Split assigned fragments into two groups
        fragments1 = fragments[largeidx_all, :]
        fragments2 = fragments[smallidx_all, :]

        # Step 5: Distribute unassigned fragments by cumulative mass to p1
        if len(idx_unassigned) > 0:
            fragments_unassigned = fragments[idx_unassigned, :]
            cum_mass_p1 = np.sum(fragments1[:, 3]) + np.cumsum(fragments_unassigned[:, 3])
            p1_assign = cum_mass_p1 <= p1_mass

            # Assign based on cumulative threshold
            p1indx = np.where(p1_assign)[0]
            fragments1 = np.vstack([fragments1, fragments_unassigned[p1indx, :]])
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
    debris1 = func_create_tlesv2_vec(ep, p1_r, p1_v, p1_objclass, fragments1, max_frag, req)
    debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, fragments2, max_frag, req)
    
    return debris1, debris2, isCatastrophic

def func_create_tlesv2_vec(ep, r_parent, v_parent, class_parent, fragments, max_frag, req):
    """
    Pure-Python version, no NumPy.

    Parameters:
    - ep: Epoch (passed through, unused here)
    - r_parent: tuple/list of 3 floats (x, y, z) [km]
    - v_parent: tuple/list of 3 floats (vx, vy, vz) [km/s]
    - class_parent: unused
    - fragments: iterable of 8-tuples/lists:
        [diameter, area, AMR, mass, total_dv, dv_x, dv_y, dv_z]
    - max_frag: int, maximum number of fragments to keep
    - req: float, normalization factor for semi-major axis

    Returns:
    - List of (a_normalized, ecc, mass) for each valid fragment
    """
    # 1) ensure we have a list
    # frags = list(fragments)
    num_frag = len(fragments)

    # 2) warn if too many
    if num_frag > max_frag:
        print(f"Warning: number of fragments ({num_frag}) exceeds max_frag ({max_frag})")

    # 3) take top-mass fragments
    n_frag = min(num_frag, max_frag)
    frags = sorted(fragments, key=lambda f: f[3], reverse=True)[:n_frag]

    # unpack parent pos/vel
    rx, ry, rz = r_parent
    v0x, v0y, v0z = v_parent

    out = []
    for diam, area, amr, mass, total_dv, dvx, dvy, dvz in frags:
        # new position same as parent
        r_new = (rx, ry, rz)
        # new velocity = parent + delta-v
        v_new = (v0x + dvx, v0y + dvy, v0z + dvz)

        # compute scalar SMA and ecc
        a, ecc = compute_sma_ecc_scalar(r_new, v_new, 398600.5)  # using your existing scalar version
        if a > 0:
            a_norm = a / req
            out.append((a_norm, ecc, mass))

    return out

def evolve_bins_circular(m1, m2, r1, r2, dv1, dv2, binC, binE, binW, LBdiam, source_sinks, RBflag = 0, fragment_spreading=False, n_shells=0, R02 = None): # eventually add stochastic ability
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

def compute_sma_ecc_scalar(r, v, mu):
    rx, ry, rz = r
    vx, vy, vz = v

    # --- angular-momentum vector h = r × v ---
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h2 = hx*hx + hy*hy + hz*hz              # |h|²

    # --- specific orbital energy ε = v²/2 – μ/r ---
    v2      = vx*vx + vy*vy + vz*vz
    r_norm  = math.sqrt(rx*rx + ry*ry + rz*rz)
    eps     = 0.5 * v2 - mu / r_norm

    # semi-major axis a
    a = math.inf if eps == 0.0 else -mu / (2.0 * eps)

    # --- eccentricity vector ---
    rv_dot = rx*vx + ry*vy + rz*vz
    coef   = v2 - mu / r_norm
    ex = (coef * rx - rv_dot * vx) / mu
    ey = (coef * ry - rv_dot * vy) / mu
    ez = (coef * rz - rv_dot * vz) / mu
    ecc = math.sqrt(ex*ex + ey*ey + ez*ez)

    return a, ecc

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