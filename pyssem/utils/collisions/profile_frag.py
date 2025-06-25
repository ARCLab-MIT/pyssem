import cProfile, pstats
from NASA_SBM_frags import frag_col_SBM_vec_lc2, perifocal_r_and_v, rotate_vector_45_deg_in_plane
import numpy as np

param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }

# --- set up a realistic test call ---
ep = 0
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
# p1_in, p2_in, param, LB: fill these in just as you do in test_one_sma
p1 = np.array([1783, 0.5, *r1, *v1, 1.0])
p2 = np.array([   4, 0.5, *r2, *v2, 1.0])
param = { 'req':6378.1, 'mu':3.986e5, 'max_frag':float('inf'), 'maxID':0, 'j2':0.0 }
LB = 0.1

# --- run under the profiler ---
pr = cProfile.Profile()
pr.enable()
debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(ep, p1, p2, param, LB)
pr.disable()

# --- dump the top 30 callers by cumulative time ---
stats = pstats.Stats(pr).sort_stats('cumtime')
stats.print_stats(30)