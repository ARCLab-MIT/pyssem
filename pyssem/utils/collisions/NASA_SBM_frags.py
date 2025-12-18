import numpy as np
# from ..collisions.rv2coe_vec import rv2coe_vec 
# when testing locally use
# from rv2coe_vec import compute_sma_ecc  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .collisions import func_dv, func_Am
# from numba.typed import List
# from numba import int64, float64, boolean
# from numba import njit, prange


# @njit(fastmath=True) # @profile
import math
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


# @njit(fastmath=True)
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

# @njit(fastmath=True)
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

# @njit
# def func_Am(d, ObjClass):
#     if isinstance(d, float):  # only float allowed in njit
#         d_list = List()
#         d_list.append(d)
#     else:
#         d_list = List()
#         for x in d:
#             d_list.append(x)

#     numObj = len(d_list)
#     logds = List()
#     for x in d_list:
#         logds.append(math.log10(x))

#     out = List.empty_list(float64)
#     for ind in range(numObj):
#         logd = logds[ind]

#         if 4.5 < ObjClass < 8.5:
#             alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_for_rocket_body(logd)
#         else:
#             alpha, mu1, sigma1, mu2, sigma2 = calculate_amsms_not_rocket_body(logd)

#         # You must rewrite these random.gauss() into njit-compatible versions or move out of njit
#         z1 = random.gauss(0.0, 1.0)
#         z2 = random.gauss(0.0, 1.0)

#         N1 = mu1 + sigma1 * z1
#         N2 = mu2 + sigma2 * z2
#         am = 10 ** (alpha * N1 + (1 - alpha) * N2)
#         out.append(am)

#     return out

# # @njit(fastmath=True)
# def func_dv(Am, mode):
#     """
#     Calculate the change in velocity (delta-v) for debris fragments based on their
#     area-to-mass ratio, without using numpy.

#     Args:
#         Am (list of float): Area-to-mass ratios of fragments.
#         mode (str): 'col' for collision-induced Δv or 'exp' for explosion-induced Δv.

#     Returns:
#         list of float: Δv values (in m/s) for each fragment.
#     """
#     sigma = 0.4
#     result = []
#     for a_m in Am:
#         # pick the right coefficient
#         if mode == 'col':
#             mu_val = 0.9 * math.log10(a_m) + 2.9
#         elif mode == 'exp':
#             mu_val = 1.85 * math.log10(a_m) + 1.85
#         else:
#             raise ValueError(f"Unknown mode '{mode}'")

#         # add Gaussian noise
#         N_val = mu_val + sigma * random.gauss(0, 1)

#         # final Δv (in m/s)
#         result.append(10 ** N_val)

#     return result


# from math import sqrt
# import math
# import random

# @njit(fastmath=True)
# def sum_by_index(values, idx_list):
#     total = 0.0
#     for j in idx_list:
#         total += values[j]
#     return total

# @njit
# def mask_to_indices(mask):
#     out = []
#     for i in range(len(mask)):
#         if mask[i]:
#             out.append(i)
#     return out

# @njit
# def sort_indices_by_values(values, indices):
#     n = len(indices)
#     for i in range(n):
#         for j in range(i + 1, n):
#             if values[indices[i]] > values[indices[j]]:
#                 indices[i], indices[j] = indices[j], indices[i]
#     return indices

# @njit
# def safe_sum(arr):
#     total = 0.0
#     for x in arr:
#         total += x
#     return total

# @njit
# def build_largeidx(d, p2_radius, p1_radius):
#     result = List.empty_list(int64)
#     for i in range(len(d)):
#         di = d[i]
#         if di > 2 * p2_radius and di < 2 * p1_radius:
#             result.append(i)
#     return result


# @njit(fastmath=True)# @profile
# def frag_col_SBM_vec_lc(ep, p1_in, p2_in, req=6.3781e+03, max_frag=float('inf'), LB=0.1):
#     """

#     Collision model following NASA EVOLVE 4.0 standard breakup model (2001)
#     with the revision in ODQN "Proper Implementation of the 1998 NASA Breakup Model" (2011)

#     The Debris output looks lke this:
#     1. `a`: Semi-major axis (in km) - defines the size of the orbit.
#     2. `ecco`: Eccentricity - describes the shape of the orbit (0 for circular, closer to 1 for highly elliptical).
#     3. `inclo`: Inclination (in degrees) - the tilt of the orbit relative to the equatorial plane.
#     4. `nodeo`: Right ascension of ascending node (in degrees) - describes the orientation of the orbit in the plane.
#     5. `argpo`: Argument of perigee (in degrees) - the angle between the ascending node and the orbit's closest point to Earth.
#     6. `mo`: Mean anomaly (in degrees) - the position of the debris along the orbit at a specific time.
#     7. `bstar`: Drag coefficient - relates to atmospheric drag affecting the debris.
#     8. `mass`: Mass of the debris fragment (in kg).
#     9. `radius`: Radius of the debris fragment (in meters).
#     10. `errors`: Error values (various metrics used for error tracking or uncertainty).
#     11. `controlled`: A flag indicating whether the fragment is controlled (e.g., through active debris removal or a controlled reentry).
#     12. `a_desired`: Desired semi-major axis (in km) - often used in mission planning.
#     13. `missionlife`: Expected mission lifetime (in years).
#     14. `constel`: Identifier for the constellation to which the debris belongs (if any).
#     15. `date_created`: Date the fragment data was created (format depends on implementation, e.g., `YYYY-MM-DD`).
#     16. `launch_date`: Date of launch of the parent object (format depends on implementation).
#     17. `r[idx_a]`: Position vector (in km) - describes the 3D position of the fragment in space.
#     18. `v[idx_a]`: Velocity vector (in km/s) - describes the 3D velocity of the fragment.
#     19. `frag_objectclass`: Object class - a dimensionless identifier describing the type of fragment or debris.
#     20. `ID_frag`: Unique identifier for the fragment (could be a string or numeric ID).
    
#     Parameters:
#     - ep: Epoch
#     - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#     - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
#     - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
#     - LB: Lower bound for fragment sizes (meters)
    
#     Returns:
#     - debris1: Array containing debris fragments from parent 1
#     - debris2: Array containing debris fragments from parent 2
#     - isCatastrophic: Boolean indicating if the collision is catastrophic
#     """
#     m = List.empty_list(float64)
#     m_rem = List.empty_list(float64)

#     # -- swap so p1 is always the heavier/larger --
#     if p1_in[0] < p2_in[0] or (p1_in[0] == p2_in[0] and p1_in[1] < p2_in[1]):
#         temp = p1_in
#         p1_in = p2_in
#         p2_in = temp

#     p1_mass, p1_radius = p1_in[0], p1_in[1]
#     p1_r = (p1_in[2], p1_in[3], p1_in[4])
#     p1_v = (p1_in[5], p1_in[6], p1_in[7])
#     p1_objclass = p1_in[8]
#     p2_mass, p2_radius = p2_in[0], p2_in[1]
#     p2_r = (p2_in[2], p2_in[3], p2_in[4])
#     p2_v = (p2_in[5], p2_in[6], p2_in[7])
#     p2_objclass = p2_in[8]

#     # unpack velocities and compute Δv
#     dx = p1_v[0] - p2_v[0]
#     dy = p1_v[1] - p2_v[1]
#     dz = p1_v[2] - p2_v[2]
#     dv = math.sqrt(dx*dx + dy*dy + dz*dz)  # km/s
#     catastroph_ratio = (p2_mass * (dv * 1000)**2) / (2 * p1_mass * 1000)

#     if catastroph_ratio < 40:
#         M = p2_mass * dv**2
#         isCatastrophic = False
#     else:
#         M = p1_mass + p2_mass
#         isCatastrophic = True

#     # build dd_edges via log‐space manually
#     n_edges = 200
#     log_min = math.log10(LB)
#     log_max = math.log10(min(1.0, 2.0 * p1_radius))
#     step = (log_max - log_min) / (n_edges - 1)
#     dd_edges = List()
#     for i in range(n_edges):
#         dd_edges.append(10**(log_min + i * step))

#     # compute mid‐points
#     dd_means = List()
#     for i in range(n_edges - 1):
#         a, b = dd_edges[i], dd_edges[i+1]
#         dd_means.append(10**((math.log10(a) + math.log10(b)) / 2))

#     # compute nddcdf and diff → ndd
#     nddcdf = List()
#     for i in range(n_edges):
#         nddcdf.append(0.1 * M**0.75 * dd_edges[i]**(-1.71))

#     ndd = List()
#     for i in range(n_edges - 1):
#         diff = -(nddcdf[i+1] - nddcdf[i])
#         ndd.append(diff if diff > 0 else 0.0)

#     # floor and fractional part
#     floor_ndd = List()
#     frac = List()
#     for i in range(len(ndd)):
#         f = math.floor(ndd[i])
#         floor_ndd.append(int(f))
#         frac.append(ndd[i] - f)

#     d_pdf = List.empty_list(float64)
#     for i in range(len(floor_ndd)):
#         mean = dd_means[i]
#         for _ in range(floor_ndd[i]):
#             d_pdf.append(mean)

#     for i in range(len(frac)):
#         if np.random.random() < frac[i]:
#             d_pdf.append(dd_means[i])

#     # shuffle inplace
#     n = len(d_pdf)
#     for i in range(n - 1, 0, -1):
#         j = int(np.floor(np.random.random() * (i + 1)))
#         tmp = d_pdf[i]
#         d_pdf[i] = d_pdf[j]
#         d_pdf[j] = tmp

#     d = d_pdf

#     # compute A and Am
#     A = List()
#     for i in range(len(d)):
#         A.append(0.556945 * d[i]**2.0047077)

#     Am = func_Am(d, p1_objclass)

#     for i in range(len(A)):
#         m.append(A[i] / Am[i])

#     # initialize remnant indices
#     idx_rem1, idx_rem2 = List.empty_list(int64), List.empty_list(int64)
#     largeidx = build_largeidx(d, p2_radius, p1_radius)

#     # --- Numba-compatible fragment mass allocation ---
#     def is_not_in_largeidx(i, largeidx):
#         for j in range(len(largeidx)):
#             if i == largeidx[j]:
#                 return False
#         return True

#     def argsort_by_value(m, idxs):
#         # Selection sort indices by m[idxs]
#         idxs_out = List()
#         for i in range(len(idxs)):
#             idxs_out.append(idxs[i])
#         n = len(idxs_out)
#         for i in range(n):
#             for j in range(i+1, n):
#                 if m[idxs_out[i]] > m[idxs_out[j]]:
#                     temp = idxs_out[i]
#                     idxs_out[i] = idxs_out[j]
#                     idxs_out[j] = temp
#         return idxs_out

#     if safe_sum(m) < M:
#         if isCatastrophic:
#             # Catastrophic fragment handling (Numba-compatible)
#             # largeidx already built above
#             m_assigned_large = sum_by_index(m, largeidx)

#             if m_assigned_large > p1_mass:
#                 # Copy largeidx to idx_large
#                 idx_large = List()
#                 for i in range(len(largeidx)):
#                     idx_large.append(largeidx[i])
#                 # Sort idx_large by m value
#                 dord1 = argsort_by_value(m, idx_large)

#                 cumsum = List()
#                 total = 0.0
#                 for i in range(len(dord1)):
#                     total += m[dord1[i]]
#                     cumsum.append(total)

#                 last = -1
#                 for j in range(len(cumsum)):
#                     if cumsum[j] < p1_mass:
#                         last = j
#                 if last >= 0:
#                     # Build keep mask
#                     keep_mask = List()
#                     for i in range(len(m)):
#                         keep_mask.append(True)
#                     for k in range(last + 1, len(dord1)):
#                         keep_mask[dord1[k]] = False
#                     # Filter lists in-place
#                     m_new = List.empty_list(float64)
#                     d_new = List.empty_list(float64)
#                     A_new = List.empty_list(float64)
#                     Am_new = List.empty_list(float64)
#                     new_largeidx = List.empty_list(int64)
#                     for i in range(len(m)):
#                         if keep_mask[i]:
#                             m_new.append(m[i])
#                             d_new.append(d[i])
#                             A_new.append(A[i])
#                             Am_new.append(Am[i])
#                             # Check if i in largeidx
#                             for j in range(len(largeidx)):
#                                 if largeidx[j] == i:
#                                     new_largeidx.append(i)
#                                     break
#                     m = m_new
#                     d = d_new
#                     A = A_new
#                     Am = Am_new
#                     largeidx = new_largeidx
#                     m_assigned_large = cumsum[last]
#                 else:
#                     m_assigned_large = 0.0

#             # Now assign smallidx up to min(p2_mass, m_assigned_large)
#             mass_max_small = p2_mass
#             if m_assigned_large < p2_mass:
#                 mass_max_small = m_assigned_large
#             # Build smallidx_temp: indices not in largeidx
#             smallidx_temp = List.empty_list(int64)
#             for i in range(len(d)):
#                 if is_not_in_largeidx(i, largeidx):
#                     smallidx_temp.append(i)
#             dord1 = argsort_by_value(m, smallidx_temp)
#             cumsum = List()
#             total = 0.0
#             for i in range(len(dord1)):
#                 total += m[dord1[i]]
#                 cumsum.append(total)
#             last = -1
#             for j in range(len(cumsum)):
#                 if cumsum[j] <= mass_max_small:
#                     last = j
#             smallidx = List.empty_list(boolean)
#             m_assigned_small = 0.0
#             if last >= 0:
#                 # Build smallidx mask
#                 smallidx_mask = List()
#                 for i in range(len(d)):
#                     smallidx_mask.append(False)
#                 for i in range(last+1):
#                     smallidx_mask[dord1[i]] = True
#                 # Collect indices
#                 smallidx_indices = List.empty_list(int64)
#                 for i in range(len(d)):
#                     if smallidx_mask[i]:
#                         smallidx_indices.append(i)
#                 m_assigned_small = sum_by_index(m, smallidx_indices)
#             else:
#                 smallidx_mask = List()
#                 for i in range(len(d)):
#                     smallidx_mask.append(False)
#                 m_assigned_small = 0.0
#             m_remaining_large = p1_mass - m_assigned_large
#             m_remaining_small = p2_mass - m_assigned_small
#             m_remaining = List()
#             m_remaining.append(m_remaining_large)
#             m_remaining.append(m_remaining_small)
#             # Handle remnant mass distribution (Numba compatible)
#             m_remSum = M - safe_sum(m)
#             # Generate remDist: random floats, random length between 2 and 8
#             n_rem = 2 + int(np.floor(np.random.random() * 7))  # 2 to 8
#             remDist = List.empty_list(float64)
#             for i in range(n_rem):
#                 remDist.append(np.random.random())
#             totalDist = safe_sum(remDist)
#             m_rem_temp = List.empty_list(float64)
#             for i in range(n_rem):
#                 m_rem_temp.append(m_remSum * remDist[i] / totalDist)
#             # Sort m_rem_temp descending (simple selection sort)
#             m_rem_sort = List()
#             m_rem_temp_used = List()
#             for i in range(len(m_rem_temp)):
#                 m_rem_temp_used.append(m_rem_temp[i])
#             for i in range(len(m_rem_temp)):
#                 # Find max
#                 maxidx = 0
#                 for j in range(1, len(m_rem_temp_used)):
#                     if m_rem_temp_used[j] > m_rem_temp_used[maxidx]:
#                         maxidx = j
#                 m_rem_sort.append(m_rem_temp_used[maxidx])
#                 m_rem_temp_used[maxidx] = -1e99  # Mark as used
#             # rem_temp_ordered: 1 or 2 (randomly)
#             rem_temp_ordered = List()
#             for i in range(len(m_rem_sort)):
#                 r = np.random.random()
#                 if r < 0.5:
#                     rem_temp_ordered.append(1)
#                 else:
#                     rem_temp_ordered.append(2)
#             # Remnant mass assignment
#             for i_rem in range(len(m_rem_sort)):
#                 idx_choice = 0 if rem_temp_ordered[i_rem] == 1 else 1
#                 if m_rem_sort[i_rem] > m_remaining[idx_choice]:
#                     diff = m_rem_sort[i_rem] - m_remaining[idx_choice]
#                     m_rem_sort[i_rem] = m_remaining[idx_choice]
#                     m_remaining[idx_choice] = 0.0
#                     if i_rem == len(m_rem_sort) - 1:
#                         m_rem_sort.append(diff)
#                         if idx_choice == 0:
#                             rem_temp_ordered.append(2)
#                         else:
#                             rem_temp_ordered.append(1)
#                     else:
#                         share = diff / (len(m_rem_sort) - i_rem - 1)
#                         for k in range(i_rem + 1, len(m_rem_sort)):
#                             m_rem_sort[k] += share
#                 else:
#                     m_remaining[idx_choice] = m_remaining[idx_choice] - m_rem_sort[i_rem]
#             m_rem = m_rem_sort
#             # d_rem_approx: assign based on rem_temp_ordered
#             d_rem_approx = List.empty_list(float64)
#             for i in range(len(m_rem)):
#                 mass_val = m_rem[i]
#                 if rem_temp_ordered[i] == 1:
#                     d_rem_approx.append(((mass_val / p1_mass * p1_radius**3)**(1/3)) * 2)
#                 else:
#                     d_rem_approx.append(((mass_val / p2_mass * p2_radius**3)**(1/3)) * 2)
#             # Am_rem, A_rem, d_rem as lists
#             Am_rem = func_Am(d_rem_approx, p1_objclass)
#             A_rem = List.empty_list(float64)
#             for i in range(len(m_rem)):
#                 A_rem.append(m_rem[i] * Am_rem[i])
#             d_rem = d_rem_approx
#         else:
#             # Non-catastrophic collision fragments are deposited from LB upward, until total mass M is reached
#             # Build large_mask: indices i where (di > 2*p2_radius or m[i] > p2_mass) and di < 2*p1_radius
#             large_mask = List.empty_list(int64)
#             for i in range(len(d)):
#                 di = d[i]
#                 if (di > 2 * p2_radius or m[i] > p2_mass) and di < 2 * p1_radius:
#                     large_mask.append(i)
#             m_assigned_large = sum_by_index(m, large_mask)

#             if m_assigned_large > p1_mass:
#                 # Manual copy of large_mask
#                 idx_large = List.empty_list(int64)
#                 for i in range(len(large_mask)):
#                     idx_large.append(large_mask[i])
#                 # Selection sort on idx_large by m[idx]
#                 for i in range(len(idx_large)):
#                     for j in range(i + 1, len(idx_large)):
#                         if m[idx_large[i]] > m[idx_large[j]]:
#                             temp = idx_large[i]
#                             idx_large[i] = idx_large[j]
#                             idx_large[j] = temp
#                 # csum as explicit List
#                 csum = List.empty_list(float64)
#                 total = 0.0
#                 for i in range(len(idx_large)):
#                     total += m[idx_large[i]]
#                     csum.append(total)
#                 last = -1
#                 for j in range(len(csum)):
#                     if csum[j] < p1_mass:
#                         last = j
#                 if last >= 0:
#                     # keep_mask as List
#                     keep_mask = List()
#                     for i in range(len(m)):
#                         keep_mask.append(True)
#                     for i in range(last + 1, len(idx_large)):
#                         keep_mask[idx_large[i]] = False
#                     # Filter arrays
#                     m_new = List.empty_list(float64)
#                     d_new = List.empty_list(float64)
#                     A_new = List.empty_list(float64)
#                     Am_new = List.empty_list(float64)
#                     new_large_mask = List.empty_list(int64)
#                     for i in range(len(m)):
#                         if keep_mask[i]:
#                             m_new.append(m[i])
#                             d_new.append(d[i])
#                             A_new.append(A[i])
#                             Am_new.append(Am[i])
#                             # check if i in large_mask
#                             in_large = False
#                             for k in range(len(large_mask)):
#                                 if large_mask[k] == i:
#                                     in_large = True
#                                     break
#                             if in_large:
#                                 new_large_mask.append(i)
#                     m = m_new
#                     d = d_new
#                     A = A_new
#                     Am = Am_new
#                     large_mask = new_large_mask
#                     m_assigned_large = csum[last]
#                 else:
#                     m_assigned_large = 0.0

#             m_remaining_large = p1_mass - m_assigned_large
#             # Build small_mask: indices i not in large_mask and d[i] > 2*p1_radius
#             small_mask = List.empty_list(int64)
#             for i in range(len(d)):
#                 in_large = False
#                 for j in range(len(large_mask)):
#                     if large_mask[j] == i:
#                         in_large = True
#                         break
#                 if (not in_large) and d[i] > 2 * p1_radius:
#                     small_mask.append(i)
#             # m_rem_sum
#             m_sum = safe_sum(m)
#             m_rem_sum = M - m_sum

#             if m_rem_sum > m_remaining_large:
#                 m_rem1 = m_remaining_large
#                 d_rem1 = (m_rem1 / p1_mass * p1_radius**3)**(1/3) * 2
#                 m_rem2 = m_rem_sum - m_remaining_large
#                 d_rem2 = (m_rem2 / p2_mass * p2_radius**3)**(1/3) * 2
#                 d_rem = List.empty_list(float64)
#                 d_rem.append(d_rem1)
#                 d_rem.append(d_rem2)
#                 m_rem = List.empty_list(float64)
#                 m_rem.append(m_rem1)
#                 m_rem.append(m_rem2)
#                 Am_rem = func_Am(d_rem, p1_objclass)
#                 A_rem = List.empty_list(float64)
#                 for i in range(len(m_rem)):
#                     A_rem.append(m_rem[i] * Am_rem[i])
#                 idx_rem1, idx_rem2 = 0, 1
#             else:
#                 m_rem = List.empty_list(float64)
#                 m_rem.append(m_rem_sum)
#                 d_rem = List.empty_list(float64)
#                 d_rem.append((m_rem_sum / p1_mass * p1_radius**3)**(1/3) * 2)
#                 Am_rem = func_Am(d_rem, p1_objclass)
#                 A_rem = List.empty_list(float64)
#                 for i in range(len(m_rem)):
#                     A_rem.append(m_rem[i] * Am_rem[i])
#                 idx_rem1, idx_rem2 = 0, None

#             # drop any too-small remnants
#             mask = List.empty_list(int64)
#             for i in range(len(d_rem)):
#                 if d_rem[i] >= LB and m_rem[i] >= M/1000.0:
#                     mask.append(1)
#                 else:
#                     mask.append(0)
#             # Filter d_rem, m_rem, A_rem, Am_rem by mask
#             d_rem_filt = List.empty_list(float64)
#             m_rem_filt = List.empty_list(float64)
#             A_rem_filt = List.empty_list(float64)
#             Am_rem_filt = List.empty_list(float64)
#             for i in range(len(d_rem)):
#                 if mask[i]:
#                     d_rem_filt.append(d_rem[i])
#                     m_rem_filt.append(m_rem[i])
#                     A_rem_filt.append(A_rem[i])
#                     Am_rem_filt.append(Am_rem[i])
#             d_rem = d_rem_filt
#             m_rem = m_rem_filt
#             A_rem = A_rem_filt
#             Am_rem = Am_rem_filt
#             # explicit idx_rem1, idx_rem2 logic
#             idx_rem1_valid = 1 if len(mask) > 0 and mask[0] else 0
#             idx_rem2_valid = 1 if len(mask) > 1 and mask[1] else 0
#             if not idx_rem1_valid:
#                 idx_rem1 = None
#             if not idx_rem2_valid:
#                 idx_rem2 = None
#     else:
#         # Numba-compatible: sort m ascending and keep fragments until cumulative mass < M
#         # Build dord list (indices)
#         dord = List()
#         for i in range(len(m)):
#             dord.append(i)
#         # Bubble sort dord by m
#         for i in range(len(dord)):
#             for j in range(i + 1, len(dord)):
#                 if m[dord[i]] > m[dord[j]]:
#                     temp = dord[i]
#                     dord[i] = dord[j]
#                     dord[j] = temp
#         # Compute cumulative mass until total + m[i] < M
#         total = 0.0
#         valididx = List()
#         for i in range(len(dord)):
#             mi = m[dord[i]]
#             if total + mi < M:
#                 valididx.append(dord[i])
#                 total += mi
#         cumsum_lower = total
#         # Filter fragments by valididx
#         m_new = List()
#         d_new = List()
#         A_new = List()
#         Am_new = List()
#         for i in range(len(valididx)):
#             idx = valididx[i]
#             m_new.append(m[idx])
#             d_new.append(d[idx])
#             A_new.append(A[idx])
#             Am_new.append(Am[idx])
#         m = m_new
#         d = d_new
#         A = A_new
#         Am = Am_new
#         # Assign “large” and “small” fragments based on size and mass (Numba compatible)
#         largeidx = List()
#         for i in range(len(d)):
#             if (d[i] > 2.0 * p2_radius or m[i] > p2_mass) and d[i] < 2.0 * p1_radius:
#                 largeidx.append(i)
#         smallidx = List()
#         for i in range(len(d)):
#             is_large = False
#             for j in range(len(largeidx)):
#                 if i == largeidx[j]:
#                     is_large = True
#                     break
#             if (d[i] > 2.0 * p1_radius) and (not is_large):
#                 smallidx.append(i)
#         # --- Check if there is mass remaining, and generate an additional fragment if needed ---
#         m_rem_val = M - cumsum_lower  # remaining mass to accumulate to M
#         m_rem = List()
#         d_rem = List()
#         A_rem = List()
#         Am_rem = List()
#         idx_rem1 = -1
#         idx_rem2 = -1
#         if m_rem_val > M / 1000.0:  # if the remaining mass is larger than 0.1% of M
#             # determine how to assign the extra fragment
#             mass_small_assigned = sum_by_index(m, smallidx)
#             mass_large_assigned = sum_by_index(m, largeidx)
#             # Use integer flags for assignment
#             if m_rem_val > (p2_mass - mass_small_assigned):
#                 rand_assign_frag = 1  # assign to larger satellite
#             elif m_rem_val > (p1_mass - mass_large_assigned):
#                 rand_assign_frag = 2  # assign to smaller satellite
#             else:
#                 if random.random() < 0.5:
#                     rand_assign_frag = 1
#                 else:
#                     rand_assign_frag = 2
#             # compute approximate diameter for the remnant fragment
#             if rand_assign_frag == 1:
#                 d_rem_approx = (m_rem_val / p1_mass * p1_radius**3)**(1/3) * 2
#                 idx_rem1 = 0
#                 idx_rem2 = -1
#             else:
#                 d_rem_approx = (m_rem_val / p2_mass * p2_radius**3)**(1/3) * 2
#                 idx_rem1 = -1
#                 idx_rem2 = 0
#             # compute its A/m and area
#             Am_rem_list = func_Am([d_rem_approx], p1_objclass)
#             A_rem_val = m_rem_val * Am_rem_list[0]
#             # Remove if below lower bound and negligible mass
#             if d_rem_approx < LB and m_rem_val < M / 1000.0:
#                 # leave d_rem, A_rem, Am_rem, m_rem empty
#                 pass
#             else:
#                 d_rem.append(d_rem_approx)
#                 A_rem.append(A_rem_val)
#                 Am_rem.append(Am_rem_list[0])
#                 m_rem.append(m_rem_val)
#         else:
#             # no extra fragment needed
#             pass
    
#     # --- Final fragment processing (Numba compatible) ---
#     Am_all = List()
#     for i in range(len(Am)):
#         Am_all.append(Am[i])
#     for i in range(len(Am_rem)):
#         Am_all.append(Am_rem[i])

#     dv_raw = func_dv(Am_all, 'col')  # returns list
#     dv_values = List()
#     for i in range(len(dv_raw)):
#         dv_values.append(dv_raw[i] / 1000.0)

#     N = len(dv_values)
#     u = List(); theta = List(); v_vals = List()
#     for _ in range(N):
#         u_val = 2.0 * random.random() - 1.0
#         theta_val = 2.0 * math.pi * random.random()
#         u.append(u_val)
#         theta.append(theta_val)
#         v_vals.append(math.sqrt(1.0 - u_val * u_val))

#     dv_vec = List()
#     for i in range(N):
#         dv_vec.append((
#             v_vals[i] * math.cos(theta[i]) * dv_values[i],
#             v_vals[i] * math.sin(theta[i]) * dv_values[i],
#             u[i] * dv_values[i]
#         ))

#     if not isCatastrophic:
#         p1remnantmass = p1_mass + p2_mass - (safe_sum(m) + safe_sum(m_rem))
#         m_rem.append(p1remnantmass)
#         d_rem.append(2 * p1_radius)
#         area = math.pi * p1_radius * p1_radius
#         A_rem.append(area)
#         Am_rem.append(area / p1remnantmass)
#         dv_values.append(0.0)
#         dv_vec.append((0.0, 0.0, 0.0))

#     d_all = List(); A_all = List(); Am_all_final = List(); m_all = List()
#     for i in range(len(d)):
#         d_all.append(d[i])
#         A_all.append(A[i])
#         Am_all_final.append(Am[i])
#         m_all.append(m[i])
#     for i in range(len(d_rem)):
#         d_all.append(d_rem[i])
#         A_all.append(A_rem[i])
#         Am_all_final.append(Am_rem[i])
#         m_all.append(m_rem[i])

#     fragments = List()
#     for i in range(len(d_all)):
#         fragments.append((
#             d_all[i], A_all[i], Am_all_final[i], m_all[i],
#             dv_values[i], dv_vec[i][0], dv_vec[i][1], dv_vec[i][2]
#         ))

#     fragments1 = List(); fragments2 = List()
#     if isCatastrophic:
#         for i in range(len(fragments)):
#             d_val = fragments[i][0]
#             if d_val > 2.0 * p2_radius and d_val < 2.0 * p1_radius:
#                 fragments1.append(fragments[i])
#             elif d_val > 2.0 * p1_radius:
#                 fragments2.append(fragments[i])
#             else:
#                 mass_sum = 0.0
#                 if len(fragments1) > 0:
#                     mass_sum = safe_sum([f[3] for f in fragments1])

#                 if mass_sum + fragments[i][3] <= p1_mass:
#                     fragments1.append(fragments[i])
#                 else:
#                     fragments2.append(fragments[i])
#     else:
#         max_mass = -1.0
#         max_idx = -1
#         for i in range(len(fragments)):
#             if fragments[i][3] > max_mass:
#                 max_mass = fragments[i][3]
#                 max_idx = i
#         for i in range(len(fragments)):
#             if i == max_idx:
#                 fragments1.append(fragments[i])
#             else:
#                 fragments2.append(fragments[i])

#     def filter_by_diameter(frag_list, LB):
#         filtered = List()
#         for i in range(len(frag_list)):
#             if frag_list[i][0] >= LB:
#                 filtered.append(frag_list[i])
#         return filtered

#     fragments1 = filter_by_diameter(fragments1, LB)
#     fragments2 = filter_by_diameter(fragments2, LB)

#     debris1 = func_create_tlesv2_vec(ep, p1_r, p1_v, p1_objclass, fragments1, max_frag, req)
#     debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, fragments2, max_frag, req)

#     return debris1, debris2, isCatastrophic


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
    except Exception as e:
        print(f"Error in dv_vec calculation: {e}")
        return None, None, False  # return empty debris and non-catastrophic flag
    
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

# @njit(fastmath=True)# @profile
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

def filter_objclass_fragments_int(class_parent):
    """
    Assign object class to fragments according to the parent particle.

    For this implementation, we'll assume fragments inherit the parent's class.
    """
    return class_parent


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
    from matplotlib.colors import LogNorm

    # === Helper functions ===

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

            debris1, debris2, isCatastrophic = frag_col_func(0, p1, p2, LB=LB)

            debris = np.vstack([debris1, debris2])
            sma_alt_km = (debris[:, 0] - 1) * earth_radius_km
            ecc = debris[:, 1]

            results.append((sma, debris))

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
        ecc_2=0.01
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

        debris1, debris2, isCatastrophic = frag_col_func(0, p1, p2, LB=LB)

        print(f"Debris generated at SMA {sma1} km: {len(debris1)} fragments")
        print(f"Debris generated at SMA {sma2} km: {len(debris2)} fragments")

        # sum together debris1 and debris2 plot sma and ecc
        debris = np.vstack([debris1, debris2])
        sma_alt_km = (debris[:, 0] - 1) * earth_radius_km
        ecc = debris[:, 1]
        plt.figure(figsize=(10, 6))
        plt.scatter(sma_alt_km, ecc, alpha=0.5, s=10, c='blue')
        plt.title(f"Debris Distribution at SMA {sma1} km")
        plt.xlabel("SMA as altitude (km)")
        plt.ylabel("Eccentricity") 
        plt.show()

    # Run the test
    test_one_sma(frag_col_SBM_vec_lc2, param, LB)
