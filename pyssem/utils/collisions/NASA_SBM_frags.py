import numpy as np
from utils.collisions.collisions import func_Am, func_dv
from poliastro.core.elements import rv2coe
from astropy import units as u
from poliastro.bodies import Earth
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import pickle


def frag_col_SBM_vec_lc2(ep, p1_in, p2_in, param, LB):
    """
    Collision model following NASA EVOLVE 4.0 standard breakup model (2001)
    with the revision in ODQN "Proper Implementation of the 1998 NASA Breakup Model" (2011)
    
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
    
    # Compute relative velocity (dv) and catastrophic ratio
    dv = np.linalg.norm(np.array(p1_v) - np.array(p2_v))  # km/s
    catastroph_ratio = (p2_mass * (dv * 1000) ** 2) / (2 * p1_mass * 1000)  # J/g
    
    # Determine if collision is catastrophic
    if catastroph_ratio < 40:
        M = p2_mass * dv ** 2
        isCatastrophic = False
    else:
        M = p1_mass + p2_mass
        isCatastrophic = True
    
    # Create debris size distribution
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * p1_radius)), 200)
    log10_dd = np.log10(dd_edges)
    dd_means = 10 ** (log10_dd[:-1] + np.diff(log10_dd) / 2)
    
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)
    ndd = np.maximum(0, -np.diff(nddcdf))
    floor_ndd = np.floor(ndd).astype(int)
    rand_sampling = np.random.rand(len(ndd))
    add_sampling = rand_sampling > (1 - (ndd - floor_ndd))
    d_pdf = np.repeat(dd_means, floor_ndd + add_sampling.astype(int))
    
    # Shuffle the diameters
    d = np.random.permutation(d_pdf)
    
    # Calculate mass of fragments
    A = 0.556945 * d ** 2.0047077
    Am = func_Am(d, p1_objclass)
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
            # Non-catastrophic collision fragment handling
            d_rem = np.array([])
            A_rem = np.array([])
            Am_rem = np.array([])
            m_rem = np.array([])
    else:
        # Adjust fragments to match mass budget
        sort_idx_mass = np.argsort(m)
        cumsum_m = np.cumsum(m[sort_idx_mass])
        indices = np.where(cumsum_m < M)[0]
        if indices.size > 0:
            lastidx = indices[-1]
            valididx = sort_idx_mass[:lastidx + 1]
            m = m[valididx]
            d = d[valididx]
            A = A[valididx]
            Am = Am[valididx]
        else:
            m = np.array([])
            d = np.array([])
            A = np.array([])
            Am = np.array([])
    
        largeidx = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)
        smallidx = (d > 2 * p1_radius) & (~largeidx)
    
        m_rem = M - np.sum(m)
    
        if m_rem > M / 1000:
            if m_rem > (p2_mass - np.sum(m[smallidx])):
                rand_assign_frag = 1
            elif m_rem > (p1_mass - np.sum(m[largeidx])):
                rand_assign_frag = 2
            else:
                rand_assign_frag = 1 + int(round(np.random.rand()))
    
            if rand_assign_frag == 1:
                d_rem_approx = ((m_rem / p1_mass * p1_radius ** 3) ** (1 / 3)) * 2
                idx_rem1 = np.array([1])  # Assign to parent 1
                idx_rem2 = np.array([])
            else:
                d_rem_approx = ((m_rem / p2_mass * p2_radius ** 3) ** (1 / 3)) * 2
                idx_rem1 = np.array([])
                idx_rem2 = np.array([1])  # Assign to parent 2
    
            Am_rem = func_Am(d_rem_approx, p1_objclass)
            A_rem = m_rem * Am_rem
            d_rem = d_rem_approx
    
            if (d_rem < LB).any() and (m_rem < M / 1000):
                d_rem = np.array([])
                A_rem = np.array([])
                Am_rem = np.array([])
                m_rem = np.array([])
                idx_rem1 = np.array([])
                idx_rem2 = np.array([])
        else:
            d_rem = np.array([])
            A_rem = np.array([])
            Am_rem = np.array([])
            m_rem = np.array([])
    

    dv = func_dv(Am, 'col') / 1000   # Convert to km/s
    u = np.random.rand(len(dv)) * 2 - 1
    theta = np.random.rand(len(dv)) * 2 * np.pi

    v = np.sqrt(1 - u**2)
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
    dv_vec = p * dv[:, np.newaxis]
    
    # Compute dv vectors
    if not isCatastrophic:
        # Find remnant mass of parent object; add to *_rem object
        try:                
            p1remnantmass = p1_mass + p2_mass - np.sum(np.concatenate((m, m_rem)))  # could be larger than p1_mass..!
        except ValueError:
            m_rem = np.array([m_rem])
            p1remnantmass = p1_mass + p2_mass - np.sum(np.concatenate((m, m_rem)))

        m_rem = np.concatenate((m_rem, [p1remnantmass]))
        d_rem = np.concatenate((d_rem, [p1_radius * 2]))
        A_rem = np.concatenate((A_rem, [np.pi * p1_radius**2]))  # area not used for func_create_tlesv2_vec
        Am_rem = np.concatenate((Am_rem, [np.pi * p1_radius**2 / p1remnantmass]))  # AMR used only for Bstar in func_create_tlesv2_vec
        dv = np.concatenate((dv))  # no change from parent object
        dv_vec = np.vstack((dv_vec, [0, 0, 0]))
        total_debris_mass = np.sum(np.concatenate((m, m_rem)))

    else:
        try:
            if len(m_rem) == 1:
                m_rem = np.array(m_rem)
        except TypeError:
            # Case where there are no fragments
            return np.array([]), np.array([]), isCatastrophic
        total_debris_mass = np.sum(np.concatenate((m, m_rem)))

    # Calculate the sum of p1_mass and p2_mass
    original_mass = p1_mass + p2_mass

    # Check if the absolute difference is greater than M * 0.05
    if abs(total_debris_mass - original_mass) > M * 0.05:
        print(f'Warning: Total sum of debris mass ({total_debris_mass:.1f} kg) differs from "mass" of original objects ({original_mass:.1f} kg)')
        # Assign dv vectors randomly
        dv_total = np.linalg.norm([fragments_all[:, 5], fragments_all[:, 6], fragments_all[:, 7]], axis=0)

    d_combined = np.concatenate((d, d_rem))
    A_combined = np.concatenate((A, A_rem))
    Am_combined = np.concatenate((Am, Am_rem))
    m_combined = np.concatenate((m, m_rem))

    # Combine these concatenated arrays with dv and the columns of dv_vec
    fragments_all = np.column_stack((d, A, Am, m, dv, dv_vec[:, 0], dv_vec[:, 1], dv_vec[:, 2]))
        
    # Distribute fragments amongst parent 1 and parent 2
    if isCatastrophic:
        # Initialize assignment arrays
        largeidx_all = (fragments_all[:, 0] > 2 * p2_radius) & (fragments_all[:, 0] < 2 * p1_radius)
        smallidx_all = (fragments_all[:, 0] > 2 * p1_radius) & (~largeidx_all)
    
        # Assign remnant fragments if any
        if 'idx_rem1' in locals() and idx_rem1.size > 0:
            largeidx_all = largeidx_all | np.isin(np.arange(len(fragments_all)), idx_rem1)
        if 'idx_rem2' in locals() and idx_rem2.size > 0:
            smallidx_all = smallidx_all | np.isin(np.arange(len(fragments_all)), idx_rem2)
    
        assignedidx = largeidx_all | smallidx_all
        idx_unassigned = np.where(~assignedidx)[0]
    
        # Assign fragments to p1 until p1_mass is filled
        fragments1 = fragments_all[largeidx_all, :]
        fragments2 = fragments_all[smallidx_all, :]
    
        if len(idx_unassigned) > 0:
            fragments_unassigned = fragments_all[idx_unassigned, :]
            cum_mass_p1 = np.sum(fragments1[:, 3])
            cum_mass_p1 += np.cumsum(fragments_unassigned[:, 3])
            p1_assign = cum_mass_p1 <= p1_mass
            p1indx = np.where(p1_assign)[0]
            fragments1 = np.vstack([fragments1, fragments_unassigned[:len(p1indx), :]])
            fragments2 = np.vstack([fragments2, fragments_unassigned[len(p1indx):, :]])
    else:
        # Non-catastrophic collision: assign largest fragment to p1, others to p2
        if fragments_all.shape[0] > 0:
            heaviestInd = fragments_all[:, 3] == np.max(fragments_all[:, 3])
            lighterInd = ~heaviestInd
            fragments1 = fragments_all[heaviestInd, :]
            fragments2 = fragments_all[lighterInd, :]
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
    param['maxID'] += debris1.shape[0]
    debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, fragments2, param)
    param['maxID'] += debris2.shape[0]
    
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

    # mu in poliastro terms
    k = Earth.k.to_value(u.km ** 3 / u.s ** 2)
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
        a, ecc, incl, nodeo, argpo, mo = rv2coe(k, r[i], v[i])
        a_list.append(a)
        ecc_list.append(ecc)
        incl_list.append(incl)
        nodeo_list.append(nodeo)
        argpo_list.append(argpo)
        mo_list.append(mo)

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
    # Define p1_in and p2_in
    # p1_in = 1.0e+03 * np.array([1.2500, 0.0040, 2.8016, 2.7285, 6.2154, -0.0055, -0.0030, 0.0038, 0.0010])

    # p2_in = 1.0e+03 * np.array([0.0060, 0.0001, 2.8724, 2.7431, 6.2248, 0.0032, 0.0054, -0.0039, 0.0010])

    p1_in = np.array([
        1250,  # mass in kg
        1,     # radius in meters
        2743.4,  # r_x in km
        2743.1,  # r_y in km
        6224.8,  # r_z in km
        -5.5,    # v_x in km/s
        -3.0,    # v_y in km/s
        3.8,     # v_z in km/s
        1.0      # object_class (dimensionless)
    ])

    p2_in = np.array([
        1250,     # mass in kg
        1,     # radius in meters
        2743.4,  # r_x in km
        2743.1,  # r_y in km
        6224.8,  # r_z in km
        3.2,     # v_x in km/s
        5.4,     # v_y in km/s
        -3.9,    # v_z in km/s
        1.0      # object_class (dimensionless)
    ])

    
    # Define the param dictionary
    param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }

    # Lower bound (LB)
    LB = 0.1  # Assuming this is the lower bound in meters

    debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB)

    # if debris 1 is empty then contine
    if debris1.size == 0:
        print("No debris fragments were generated")
        exit()

    # Assuming debris1 is already defined
    idx_a = 0
    idx_ecco = 1

    # 1. 1D Histogram for SMA (semi-major axis)
    plt.figure()
    plt.hist((debris1[:, idx_a] - 1) * 6371, bins=np.arange(0, 5001, 100))
    plt.title('SMA as altitude (km)')
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 2. 1D Histogram for Eccentricity
    plt.figure()
    plt.hist(debris1[:, idx_ecco], bins=50)
    plt.title('Eccentricity')
    plt.xlabel('Eccentricity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 3. 2D Histogram using histogram2d with LogNorm for color scale
    plt.figure()
    hist, xedges, yedges = np.histogram2d(
        (debris1[:, idx_a] - 1) * 6371, debris1[:, idx_ecco],
        bins=[np.arange(0, 5001, 100), np.arange(0, 1.01, 0.01)]
    )

    # Avoid any zero counts for logarithmic color scaling
    hist[hist == 0] = np.nan  # Replace zeros with NaNs to avoid LogNorm issues

    # Plotting the 2D histogram
    mappable = plt.imshow(
        hist.T, origin='lower', norm=LogNorm(), 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto'
    )
    plt.colorbar(mappable, label='Count')
    plt.xlim([0, 3000])
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Eccentricity')
    plt.title('2D Histogram of SMA and Eccentricity')
    plt.grid(True)
    plt.show()

    # with open(r'C:\Users\IT\Documents\UCL\pyssem\scenario-properties-elliptical.pkl', 'rb') as f:
    #     scen_properties = pickle.load(f)

    # all_elliptical_collision_species = scen_properties.collision_pairs

    # debris_species = [species for species in scen_properties.species['debris']]

    # # Mass Binning
    # binC_mass = np.zeros(len(debris_species))
    # binE_mass = np.zeros(2 * len(debris_species))
    # binW_mass = np.zeros(len(debris_species))
    # LBgiven = scen_properties.LC

    # for index, debris in enumerate(debris_species):
    #     binC_mass[index] = debris.mass
    #     binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
    #     binW_mass[index] = debris.mass_ub - debris.mass_lb

    # binE_mass = np.unique(binE_mass)

    # # Eccentricity Binning, multiple debris species will have the same eccentricity bins
    # binE_ecc = debris_species[0].eccentricity_bins
    # binE_ecc = np.sort(binE_ecc)
    # # Calculate the midpoints
    # binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # # Create bin edges starting at 0 and finishing at 1
    # binE_ecc = np.concatenate(([0], binE_ecc, [1]))

        
    # def evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_index, n_shells=0):
        
    #     # Need to now follow the NASA SBM route, first we need to create p1_in and p2_in
    #     #  Parameters:
    #     # - ep: Epoch
    #     # - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    #     # - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    #     # - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
    #     # - LB: Lower bound for fragment sizes (meters)

    #     p1_in = np.array([
    #         1250.0,  # mass in kg
    #         4.0,     # radius in meters
    #         2372.4,  # r_x in km, 1000 km
    #         2743.1,  # r_y in km
    #         6224.8,  # r_z in km
    #         -5.5,    # v_x in km/s
    #         -3.0,    # v_y in km/s
    #         3.8,     # v_z in km/s
    #         1      # object_class (dimensionless)
    #     ])

    #     p2_in = np.array([
    #         6.0,     # mass in kg
    #         0.1,     # radius in meters
    #         2372.4,  # r_x in km
    #         2743.1,  # r_y in km
    #         6224.8,  # r_z in km
    #         3.2,     # v_x in km/s
    #         5.4,     # v_y in km/s
    #         -3.9,    # v_z in km/s
    #         1      # object_class (dimensionless)
    #     ])

    #     param = {
    #         'req': 6.3781e+03,
    #         'mu': 3.9860e+05,
    #         'j2': 0.0011,
    #         'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
    #         'maxID': 0,
    #         'density_profile': 'static'
    #     }
        
    #     a = scen_properties.HMid[collision_index] 

    #     # up to correct mass too
    #     if m1 < m2:
    #         m1, m2 = m2, m1
    #         r1, r2 = r2, r1

    #     p1_in[0], p2_in[0] = m1, m2 
    #     p1_in[1], p2_in[1] = r1, r2

    #     # remove a from r_x from both p1_in and p2_in
    #     p1_in[2] = p1_in[2] - a
    #     p2_in[2] = p2_in[2] - a
            
    #     LB = 0.1

    #     debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB)

    #     print(len(debris1), len(debris2))


    # def process_elliptical_collision_pair(args):
    #     """
    #     A similar function to the process_species_pair, apart from it as the shells are already defined and the velocity, 
    #     you are able to calculate evolve bins just once. 

    #     """
    #     i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    #     m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    #     r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius


    #     # there needs to be some changes here to account for the fact that the shells are already defined
    #     # set fragment_spreading to True
    #     v1 = collision_pair.species1.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    #     v2 = collision_pair.species2.velocity_per_shells[collision_pair.shell_index][collision_pair.shell_index]

    #     # This the time per shell for each species - the output still assumes that it is always there, so you need to divide
    #     t1 = collision_pair.species1.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    #     t2 = collision_pair.species2.time_per_shells[collision_pair.shell_index][collision_pair.shell_index]
    #     min_TIS = min(t1, t2)
    #     print(m1, m2, min_TIS)

    #     if m1 < 1 or m2 < 1:
    #         fragments = None
    #     else:
    #         fragments = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_pair.shell_index, n_shells=scen_properties.n_shells)    
        
    #     if fragments is not None:
    #         collision_pair.fragments = fragments # * min_TIS
    #     else: 
    #         collision_pair.fragments = fragments

    #     return collision_pair


    
    # for i, species_pair in enumerate(tqdm(all_elliptical_collision_species, desc="Processing species pairs")):
    #     for shell_collision in tqdm(species_pair.collision_pair_by_shell, desc="Processing shell collisions", leave=False):
            
    #         # Process each shell-specific collision and append result to collision_processed list
    #         gamma = process_elliptical_collision_pair((i, shell_collision, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven))
    #         species_pair.collision_processed.append(gamma)