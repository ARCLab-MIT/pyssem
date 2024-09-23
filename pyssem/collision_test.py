import pickle
import numpy as np
from utils.collisions.collisions import func_Am
import matplotlib.pyplot as plt

def frag_col_SBM_vec_lc2(ep, p1_in, p2_in, param, LB):
    """
    Collision model following NASA EVOLVE 4.0 standard breakup model (2001)
    with the revision in ODQN "Proper Implementation of the 1998 NASA Breakup Model" (2011)
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
    
    # # Compute number of fragments based on the NASA breakup model
    # num = np.floor(0.1 * M ** 0.75 * LB ** (-1.71) - 0.1 * M ** 0.75 * min([1, 2 * p1_radius]) ** (-1.71))
    min_value =
    num = np.floor(0.1 * M ** 0.75 * LB ** (-1.71) - 0.1 * M ** 0.75 *  min(1, 2 * p1_radius) ** (-1.71))


    
    # Create debris size distribution
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * p1_radius)), 200)  # logspace in Python
    log10_dd = np.log10(dd_edges)
    dd_means = 10 ** (log10_dd[:-1] + np.diff(log10_dd) / 2)  # Calculate mean of diameter edges

    
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)
    ndd = np.maximum(0, -np.diff(nddcdf))  # Ensure no negative values
    floor_ndd = np.floor(ndd).astype(int)  # Convert to integer for np.repeat
    rand_sampling = np.random.rand(len(ndd))  # Random numbers
    add_sampling = rand_sampling > (1 - (ndd - floor_ndd))  # Stochastic sampling
    d_pdf = np.repeat(dd_means, floor_ndd + add_sampling.astype(int))  # Final PDF

    # Repeat dd_means based on floor_ndd and add_sampling_int
    d = np.random.permutation(d_pdf) 

    print(f"Number of fragments: {len(d_pdf)}")
    
    # Calculate mass of objects [LB, 1 m]
    A = 0.556945 * d ** 2.0047077
    Am = func_Am(d, p1_objclass)
    m = A / Am

    print(f"Number of fragments: {len(m)}")
    
    # Handle remnant mass and fragment assignment based on collision type
    if np.sum(m) < M:
        if isCatastrophic:
            # Catastrophic fragment handling
            largeidx = (d > 2 * p2_radius) & (d < 2 * p1_radius)
            m_assigned_large = max([0, np.sum(m[largeidx])])
            
            if m_assigned_large > p1_mass:
                idx_large = np.where(largeidx)[0]
                dord1 = np.argsort(m[idx_large])
                cumsum_m1 = np.cumsum(m[idx_large[dord1]])
                lastidx1 = np.where(cumsum_m1 < p1_mass)[0][-1]
                # Remove fragments exceeding mass constraint
                to_remove = idx_large[dord1[lastidx1 + 1:]]
                m = np.delete(m, to_remove)
                d = np.delete(d, to_remove)
                A = np.delete(A, to_remove)
                Am = np.delete(Am, to_remove)
                largeidx = np.delete(largeidx, to_remove)
                m_assigned_large = cumsum_m1[lastidx1]
            
            mass_max_small = min([p2_mass, m_assigned_large])
            
            smallidx_temp = np.where(~largeidx)[0]
            dord = np.argsort(m[smallidx_temp])
            cumsum_m = np.cumsum(m[smallidx_temp[dord]])
            lastidx_small = np.where(cumsum_m <= mass_max_small)[0][-1]
            
            smallidx = np.zeros(len(d), dtype=bool)
            smallidx[smallidx_temp[dord[:lastidx_small + 1]]] = True
            m_assigned_small = max([0, np.sum(m[smallidx])])
            
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
                if m_rem_sort[i_rem] > m_remaining[rem_temp_ordered[i_rem] - 1]:
                    diff_mass = m_rem_sort[i_rem] - m_remaining[rem_temp_ordered[i_rem] - 1]
                    m_rem_sort[i_rem] = m_remaining[rem_temp_ordered[i_rem] - 1]
                    m_remaining[rem_temp_ordered[i_rem] - 1] = 0
                    
                    if i_rem == num_rem - 1:
                        m_rem_sort = np.append(m_rem_sort, diff_mass)
                        rem_temp_ordered = np.append(rem_temp_ordered, 3 - rem_temp_ordered[i_rem])
                    else:
                        m_rem_sort[i_rem + 1:] += diff_mass / (num_rem - i_rem)
            
            # Final remnant calculations
            m_rem = m_rem_sort
            d_rem_approx = np.zeros_like(m_rem)
            rem1_temp = rem_temp_ordered == 1
            d_rem_approx[rem1_temp] = (m_rem[rem1_temp] / p1_mass * p1_radius ** 3) ** (1 / 3) * 2
            d_rem_approx[~rem1_temp] = (m_rem[~rem1_temp] / p2_mass * p2_radius ** 3) ** (1 / 3) * 2
            Am_rem = func_Am(d_rem_approx, p1_objclass)
            A_rem = m_rem * Am_rem
            d_rem = d_rem_approx
    else:
        # Handle non-catastrophic collision
        d_rem = np.array([])
        A_rem = np.array([])
        Am_rem = np.array([])
        m_rem = np.array([])
    
    # Return fragments and debris
    debris1 = func_create_tlesv2_vec(ep, p1_r, p1_v, p1_objclass, np.column_stack((d, A, Am, m)), param)
    debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, np.column_stack((d_rem, A_rem, Am_rem, m_rem)), param)
    
    return debris1, debris2, isCatastrophic


def func_create_tlesv2_vec(ep, r, v, objclass, fragments, param):
    # Placeholder for actual function to create debris
    return fragments

if __name__ == "__main__":
    # Define p1_in and p2_in
    p1_in = 1.0e+03 * np.array([1.2500, 0.0040, 2.8016, 2.7285, 6.2154, -0.0055, -0.0030, 0.0038, 0.0010])

    p2_in = 1.0e+03 * np.array([0.0060, 0.0001, 2.8724, 2.7431, 6.2248, 0.0032, 0.0054, -0.0039, 0.0010])

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

    print(len(debris1), len(debris2), isCatastrophic)

    # Extract indices (You'll need to map the correct indices for `idx_a` and `idx_ecco`)
    idx_a = 1  # Replace with actual index for SMA
    idx_ecco = 2  # Replace with actual index for Eccentricity

    # 1D Histogram for SMA (semi-major axis)
    plt.figure()
    plt.hist((debris1[:, idx_a] - 1) * 6371, bins=np.arange(0, 5000 + 100, 100))  # Bin size of 100
    plt.title('SMA as altitude (km)')
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Frequency')
    plt.show()

    # 1D Histogram for Eccentricity
    plt.figure()
    plt.hist(debris1[:, idx_ecco], bins=50)  # Default bin size or specify as needed
    plt.title('Eccentricity')
    plt.xlabel('Eccentricity')
    plt.ylabel('Frequency')
    plt.show()

    # 2D Histogram using histogram2d (equivalent to histogram2 in MATLAB)
    plt.figure()
    H, xedges, yedges = np.histogram2d((debris1[:, idx_a] - 1) * 6371, debris1[:, idx_ecco], bins=[np.arange(0, 5000 + 100, 100), np.arange(0, 1 + 0.01, 0.01)])

    plt.pcolormesh(xedges, yedges, H.T, norm=plt.Normalize(vmin=1), shading='auto')
    plt.yscale('log')
    plt.xlim([0, 3000])
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Eccentricity')
    plt.colorbar(label='Counts')
    plt.grid(True)
    plt.show()

    # 2D Histogram using surf (equivalent to surf in MATLAB)
    n, xedges, yedges = np.histogram2d((debris1[:, idx_a] - 1) * 6371, debris1[:, idx_ecco], bins=[np.arange(0, 5000 + 100, 100), np.arange(0, 0.5 + 0.01, 0.01)])

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, n.T, cmap='viridis')

    ax.set_zscale('log')
    ax.set_zlim([2, 100000])
    ax.set_xlabel('SMA as altitude (km)')
    ax.set_ylabel('Eccentricity')
    ax.set_zlabel('Counts')
    ax.view_init(54, 134)
    plt.show()
