import numpy as np
import matplotlib.pyplot as plt
from NASA_SBM_frags import frag_col_SBM_vec_lc, perifocal_r_and_v, rotate_vector_45_deg_in_plane

# Needed for param dictionary
param = {
    "mu": 3.9860e+05,
}

LB = 0.1
earth_radius_km = 6371

@profile
def run_test_one_sma():
    m1 = 148
    m2 = 4
    rad1 = 0.5
    rad2 = 0.5
    sma1 = 7463.1366
    sma2 = 7463.1366
    ecc_1 = 0
    ecc_2 = 0.01
    TA = np.radians(90)
    mu = param["mu"]

    # Orbital states
    r1, v1 = perifocal_r_and_v(sma1, ecc_2, TA, mu)
    _, v2_mag_vec = perifocal_r_and_v(sma2, ecc_1, TA, mu)
    v2_hat_rotated = rotate_vector_45_deg_in_plane(v1 / np.linalg.norm(v1))
    v2 = np.linalg.norm(v2_mag_vec) * v2_hat_rotated
    r2 = r1

    # Input vectors
    p1 = np.array([m1, rad1, *r1, *v1, 1.0])
    p2 = np.array([m2, rad2, *r2, *v2, 1.0])

    debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc(0, p1, p2, req=6378.1, max_frag=float('inf'), LB=0.1)

    print(f"Debris generated: {len(debris1)} + {len(debris2)}")
    debris = np.vstack([debris1, debris2])
    sma_alt_km = (debris[:, 0] - 1) * earth_radius_km
    ecc = debris[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(sma_alt_km, ecc, alpha=0.5, s=10, c='blue')
    plt.title(f"Debris Distribution at SMA {sma1} km")
    plt.xlabel("SMA as altitude (km)")
    plt.ylabel("Eccentricity")
    plt.show()

if __name__ == "__main__":
    run_test_one_sma()