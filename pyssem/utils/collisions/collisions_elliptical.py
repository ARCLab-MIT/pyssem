from utils.collisions.collisions import func_Am, func_dv
from utils.collisions.NASA_SBM_frags import frag_col_SBM_vec_lc
from utils.collisions.NASA_SBN6 import *
import numpy as np
from tqdm import tqdm
from utils.collisions.cartesian_to_kep import cart_2_kep, kep_2_cart
from utils.collisions.collisions import SpeciesPairClass
from concurrent.futures import ProcessPoolExecutor, as_completed
from sympy import symbols, Matrix, pi, S, Expr, zeros
import matplotlib.pyplot as plt
import re
from itertools import combinations
import math
import traceback
from line_profiler import profile
import os
import pickle
from utils.collisions.collisions import evolve_bins as evolve_bins_old

# Helper function for parallel processing of species pairs
def _process_sp_pair(args):
    i, sp_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    S = scen_properties.n_shells
    D = len(debris_species)
    E = len(binE_ecc) - 1
    fragments_3d = np.zeros((S, D, E), dtype=float)
    for shell_col in sp_pair.collision_pair_by_shell:
        col_pair = process_elliptical_collision_pair(
            (i, shell_col, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven)
        )
        if col_pair.fragments is None:
            continue
        fragments_3d += col_pair.fragments
        col_pair.gamma = fragments_3d.sum(axis=2)
    sp_pair.fragments_sd   = fragments_3d.sum(axis=2)             # → (S, D)
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_sde = fragments_3d / sp_pair.fragments_sd[:, :, None] # → (S, D, E)
    
    # replace any NaNs (from 0/0) with 0
    sp_pair.ecc_distribution_sde = np.nan_to_num(dist_sde, nan=0.0, posinf=0.0, neginf=0.0)

    return sp_pair
    
def create_elliptical_collision_pairs(scen_properties):

    all_species = [
        sp for grp in scen_properties.species.values() for sp in grp
    ]
    pairs = list(combinations(all_species, 2)) + [(sp, sp) for sp in all_species]

    collision_pairs = []
    LC = scen_properties.LC
    n_shells = scen_properties.n_shells

    for sp1, sp2 in pairs:
        # skip purely maneuverable–maneuverable
        if sp1.maneuverable and sp2.maneuverable:
            continue

        cp = SpeciesCollisionPair(sp1, sp2, scen_properties)

        # figure out how many ecc bins each has
        n_e1 = sp1.time_per_shells.shape[1]
        n_e2 = sp2.time_per_shells.shape[1]

        for i_sma in range(n_shells):
            for j_e1 in range(n_e1):
                for j_e2 in range(n_e2):
                    for k_shell in range(n_shells):
                        t1 = sp1.time_per_shells[i_sma, j_e1, k_shell]
                        t2 = sp2.time_per_shells[i_sma, j_e2, k_shell]

                        if t1 > 0 and t2 > 0:
                            # skip sub‐LC collisions
                            if sp1.mass < LC and sp2.mass < LC:
                                continue
                            cp.collision_pair_by_shell.append(
                                EllipticalCollisionPair(
                                    species1=sp1,
                                    species2=sp2,
                                    sma_index=i_sma,
                                    ecc1_index=j_e1,
                                    ecc2_index=j_e2,
                                    shell_index=k_shell, 
                                    n_shells=scen_properties.n_shells,
                                    species_length=scen_properties.species_length
                                )
                            )                            

        collision_pairs.append(cp)
    
    print(f"Total number of unique species pairs: {len(collision_pairs)}")
    # loop through each species pair and then sum the collision pairs
    count = 0
    for species_pair in collision_pairs:
        count += len(species_pair.collision_pair_by_shell)

    print(f"Total number of collision pairs: {count}")

    debris_species = [species for species in scen_properties.species['debris']]

    # Mass Binning
    binC_mass = np.zeros(len(debris_species))
    binE_mass = np.zeros(2 * len(debris_species))
    binW_mass = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC_mass[index] = debris.mass
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW_mass[index] = debris.mass_ub - debris.mass_lb

    binE_mass = np.unique(binE_mass)

    # Eccentricity Binning, multiple debris species will have the same eccentricity bins
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # Create bin edges starting at 0 and finishing at 1
    binE_ecc = np.concatenate(([0], binE_ecc, [1]))

    debris_species = scen_properties.species['debris']
    D = len(debris_species)
    S = scen_properties.n_shells
    E = len(binE_ecc) - 1

    cache_path="collision_pairs.pkl"
    # 0) attempt to load from cache
    if os.path.exists(cache_path):
        print(f"Loading cached collision_pairs from {cache_path}")
        with open(cache_path, "rb") as f:
            collision_pairs = pickle.load(f)
    else:
        # Parallel processing of species pairs
        args = [
            (i, sp_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven)
            for i, sp_pair in enumerate(collision_pairs)
        ]
        
        import multiprocessing as mp
        with mp.Pool(processes=mp.cpu_count()) as pool:
            collision_pairs = list(tqdm(
                pool.imap(_process_sp_pair, args),
                total=len(args),
                desc="Processing species pairs in parallel"
            ))

        #  save to cache for next time
        print(f"Caching collision_pairs to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(collision_pairs, f)

    species_sym_vars = { str(var): var  for var in scen_properties.all_symbolic_vars }

    # outer loop over species‐pairs
    # for sp_pair in tqdm(collision_pairs, desc="Processing species pairs"):
    #     # inner loop over that pair’s elliptical collisions
    #     all_individual_eqs = []
    #     for ellip_pair in tqdm(
    #         sp_pair.collision_pair_by_shell,
    #         desc="  Elliptical collisions",
    #         leave=False
    #     ):
    #         if ellip_pair.fragments is not None:
    #             all_individual_eqs.append(ellip_pair.create_sympy_matrix_at_elliptical_level(
    #                 scen_properties.v_imp2,
    #                 scen_properties.V,
    #                 species_sym_vars
    #             ))
        
    #     sp_pair.create_symbolic_col_equation(all_species, debris_species, scen_properties.n_shells, all_individual_eqs)
    # Loop over species pairs (sequential)
    for sp_pair in collision_pairs:
        all_individual_eqs = []

        # Loop over each elliptical collision for this species pair
        for ellip_pair in sp_pair.collision_pair_by_shell:
            if ellip_pair.fragments is not None:
                eq_matrix = ellip_pair.create_sympy_matrix_at_elliptical_level(
                    scen_properties.v_imp2,
                    scen_properties.V,
                    species_sym_vars
                )
                all_individual_eqs.append(eq_matrix)

        # Build symbolic collision equation for this species pair
        sp_pair.create_symbolic_col_equation(
            all_species,
            debris_species,
            scen_properties.n_shells,
            all_individual_eqs
    )

    return collision_pairs

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

@profile
def process_elliptical_collision_pair(args):
    i, collision_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args

    sp1 = collision_pair.species1
    sp2 = collision_pair.species2
    r1 = collision_pair.species1.radius
    r2 = collision_pair.species2.radius
    sma1 = collision_pair.species1_sma
    sma2 = collision_pair.species2_sma

    # pull out the sma, ecc1, ecc2 and shell indices
    i_sma    = collision_pair.sma_index
    j_e1     = collision_pair.ecc1_index
    j_e2     = collision_pair.ecc2_index
    k_shell  = collision_pair.shell_index

    if collision_pair.species1.eccentricity_bins is not None:
        e1 = collision_pair.species1.eccentricity_bins[j_e1]
    else: 
        e1 = 0
    
    if collision_pair.species2.eccentricity_bins is not None:
        e2 = collision_pair.species2.eccentricity_bins[j_e2]
    else:
        e2 = 0
    # instantaneous velocities in that exact (sma,ecc) cell
    v1 = sp1.velocity_per_shells[i_sma, j_e1, k_shell]
    v2 = sp2.velocity_per_shells[i_sma, j_e2, k_shell]

    # time‐in‐shell fractions in that same cell
    t1 = sp1.time_per_shells[i_sma, j_e1, k_shell]
    t2 = sp2.time_per_shells[i_sma, j_e2, k_shell]

    # simple “overlap” metrics
    min_TIS  = min(t1, t2)
    prod_TIS = t1 * t2

    # now you can call your SBM with m1,m2,v_rel
    m1, m2 = sp1.mass, sp2.mass

    # if m2 is bigger than m1, swap mass radius, sma and ecc
    if m2 > m1:
        m1, m2 = m2, m1
        r1, r2 = r2, r1
        sma1, sma2 = sma2, sma1
        e1, e2 = e2, e1
        v1, v2 = v2, v1

    # This will have to change 
    try:
        if m1 < 1 or m2 < 1:
            fragments = None
        else:
            fragments = evolve_bins(scen_properties, m1, m2, r1, r2, sma1, sma2, e1, e2, 
                                    binE_mass, binE_ecc, collision_pair.shell_index, n_shells=scen_properties.n_shells)
            # fragments_old = evolve_bins_old(m1, m2, r1, r2, 10, 10, [], binE_mass, [], 0.1, [collision_pair.species1, collision_pair.species2], RBflag = 0, fragment_spreading=False, n_shells=0)
            # n_ecc_bins = len(binE_ecc) - 1
            # fragments = np.repeat(fragments_old[:, np.newaxis, :], n_ecc_bins, axis=1)
    except Exception as e:
        print("Error in Evolve Bins")
        print(f"Exception caught: {e}")
        fragments = None

    if fragments is not None:
        # print(prod_TIS)
        collision_pair.fragments = fragments * prod_TIS
    else: 
        # print(f'No fragments generated between species {m1} and {m2} in shell {collision_pair.shell_index}')
        collision_pair.fragments = fragments

    return collision_pair

@profile
def evolve_bins(scen_properties, m1, m2, rad_1, rad_2, sma1, sma2, e1, e2, binE_mass, binE_ecc, collision_index, n_shells=0):
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
    SS = 60
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
    p1 = np.array([m1, rad_1, *r1, *v1, 1.0])
    p2 = np.array([m2, rad_2, *r2, *v2, 1.0])

    try:
        debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc(0, p1, p2, LB=LB)
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

    print(frag_properties)

    binE_alt = scen_properties.R0_rad_km  # We add 1 for bin edges

    # return hist
    hist3d, _ = np.histogramdd(
        frag_properties,
        bins=[binE_alt, binE_mass, binE_ecc]
    )

    # normalize per your SS factor
    hist3d /= (SS * 3)
    
    return hist3d

class EllipticalCollisionPair:
    def __init__(self, species1, species2, sma_index, ecc1_index, ecc2_index, shell_index, n_shells, species_length) -> None:
        self.species1 = species1
        self.species2 = species2
        self.sma_index = sma_index
        self.ecc1_index = ecc1_index
        self.ecc2_index = ecc2_index
        self.shell_index = shell_index
        self.n_shells = n_shells
        self.species_length = species_length

        self.species1_sma = self.species1.semi_major_axis_bins_HMid[shell_index]
        self.species2_sma = self.species2.semi_major_axis_bins_HMid[shell_index]

        # Time-in-shell for each species at this (sma, ecc) in this shell
        self.species1_TIS = species1.time_per_shells[sma_index, ecc1_index, shell_index]
        self.species2_TIS = species2.time_per_shells[sma_index, ecc2_index, shell_index]

        # Velocity-in-shell for each species at this (sma, ecc) in this shell
        self.species1_VIS = species1.velocity_per_shells[sma_index, ecc1_index, shell_index]
        self.species2_VIS = species2.velocity_per_shells[sma_index, ecc2_index, shell_index]

        # Full TIS and VIS profiles across all ecc‐bins at this sma
        self.species1_TIS_all = species1.time_per_shells[sma_index]
        self.species2_TIS_all = species2.time_per_shells[sma_index]
        self.species1_VIS_all = species1.velocity_per_shells[sma_index]
        self.species2_VIS_all = species2.velocity_per_shells[sma_index]

        # Combined metrics
        self.combined_mass_TIS = self.species1_TIS * self.species2_TIS
        self.min_TIS = min(self.species1_TIS, self.species2_TIS)
        self.max_TIS = max(self.species1_TIS, self.species2_TIS)

        # Symbolic column names include eccentricity indices
        self.s1_col_sym_name = f"{species1.sym_name}_{shell_index + 1}"
        self.s2_col_sym_name = f"{species2.sym_name}_{shell_index + 1}"


        # Placeholders for collision outcomes
        self.gamma                   = None  # collision probability
        self.fragments               = None
        self.eqs = Matrix(n_shells, species_length, lambda i, j: 0)


        self.altA                    = None  # debris semi-major axis
        self.altE                    = None  # debris eccentricity
        self.debris_eccentricity_bins = None



    def create_sympy_matrix_at_elliptical_level(self, v_imp2, V, species_sym_vars):
        """
            To understand where the fragments are sourced from, we will need to generate symbolic expressions compined with the fragments to find 
            their original location. 
        """
        meter_to_km = 1 / 1000
        
        # Square of the impact parameter
        self.sigma = (self.species1.radius * meter_to_km + \
                    self.species2.radius * meter_to_km) ** 2
        
        self.phi = pi * v_imp2 / (V * meter_to_km**3) * self.sigma * S(86400) * S(365.25)    
        phi_matrix = Matrix([self.phi] * self.gamma.shape[1]).T

        self.gamma = Matrix(self.gamma)

        s1_sym = species_sym_vars[self.s1_col_sym_name]
        s2_sym = species_sym_vars[self.s2_col_sym_name]

        # compute phi‐weighted gamma first
        tmp = self.gamma.multiply_elementwise(phi_matrix)

        # now multiply by the symbols as *scalars*
        self.eqs = tmp * s1_sym * s2_sym

        return self.eqs


class SpeciesCollisionPair:
    def __init__(self, species1, species2, scen_properties) -> None:
        self.species1 = species1
        self.species2 = species2
        self.collision_pair_by_shell = []
        self.collision_processed = []

        # # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
        self.gamma = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

        # Create a list of source sinks, first two are the active species then the active speices
        self.source_sinks = [species1, species2] + scen_properties.species['debris']

        # Implementing logic for gammas calculations based on species properties
        if species1.maneuverable and species2.maneuverable:
            # Multiplying each element in the first column of gammas by the product of alpha_active values
            self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha_active * species2.alpha_active
            if species1.slotted and species2.slotted:
                # Applying the minimum slotting effectiveness if both are slotted
                self.gamma[:, 0] = self.gamma[:, 0] * min(species1.slotting_effectiveness, species2.slotting_effectiveness)

        elif (species1.maneuverable and not species2.maneuverable) or (species2.maneuverable and not species1.maneuverable):
            if species1.trackable and species2.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species2.alpha
            elif species2.trackable and species1.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha

        # Applying symmetric loss to both colliding species
        self.gamma[:, 1] = self.gamma[:, 0]

        # Rocket Body Flag - 1: RB; 0: not RB
        # Will be 0 if both are None type
        if species1.RBflag is None and species2.RBflag is None:
            self.RBflag = 0
        elif species2.RBflag is None:
            self.RBflag = species2.RBflag
        elif species2.RBflag is None:
            self.RBflag = species1.RBflag
        else:
            self.RBflag = max(species1.RBflag, species2.RBflag)

        # self.phi = None # Proabbility of collision, based on v_imp, shell volumen and object_radii
        # Square of impact parameter
        meter_to_km = 1 / 1000
        self.sigma = (species1.radius * meter_to_km + \
                      species2.radius * meter_to_km) ** 2

        # Scaling based on v_imp, shell volume, and object radii
        self.phi = pi * scen_properties.v_imp2 / (scen_properties.V * meter_to_km**3) * self.sigma * S(86400) * S(365.25)

        self.catastrophic = None # Catastrophic flag for each shell
        self.eqs = Matrix(scen_properties.n_shells, scen_properties.species_length, lambda i, j: 0) # A zero symbolic equation n_shells x n_species
        self.sigma = None # Square of the impact parameter
        self.fragments_sd = None # This will temporarily store the fragments as it is being built (prior to symbolic handling)
        self.ecc_distribution_sde = None # This will store the eccentric distribution of S,D,E
        self.source_sinks = [self.species1, self.species2]

    def create_symbolic_col_equation(self,
                                    all_species, # list of all species names 
                                    debris_species,    # list of debris Sympy symbols
                                    n_shells,          # integer
                                    all_individual_eqs  # list of (n_shells × D) Sympy matrices
                                    ):
        """
        Build a block of equations where:
        - first columns are the collision‐sink terms for each species pair
        - next columns are the debris‐source terms
        """

        # # Fragment generation equations
        # M1 = self.species1.mass
        # M2 = self.species2.mass
        # LC = scen_properties.LC
        
        # nf = zeros(len(scen_properties.v_imp2), 1)

        # for i, dv in enumerate(scen_properties.v_imp2):
        #     if self.catastrophic[i]:
        #         # number of fragments generated during a catastrophic collision (NASA standard break-up model). M is the sum of the mass of the objects colliding in kg
        #         n_f_catastrophic = 0.1 * LC**(-S(1.71)) * (M1 + M2)**(S(0.75))
        #         nf[i] = n_f_catastrophic
        #     else:
        #         # number of fragments generated during a non-catastrophic collision (improved NASA standard break-up model: takes into account the kinetic energy). M is the mass of the less massive object colliding in kg
        #         n_f_damaging = 0.1 * LC**(-S(1.71)) * (min(M1, M2) * dv**2)**(S(0.75))
        #         nf[i] = n_f_damaging

        # self.nf = nf.transpose() 
        
        # --- 3) Pre‐allocate the collision‐sink block (n_shells × num_pairs) ---
        num_pairs = self.gamma.shape[1]
        sink_block = Matrix.zeros(n_shells, num_pairs)

        # --- 4) Loop over each collision pair / gamma‐column ---
        for i in range(num_pairs):
            gamma_col = self.gamma[:, i]                          # (n_shells × 1) Sympy Matrix
            # find which species this column corresponds to:
            target_name = self.source_sinks[i].sym_name
            eq_idx = next(
                (j for j, sp in enumerate(all_species)
                if sp.sym_name == target_name),
                None
            )
            if eq_idx is None:
                raise ValueError(f"Could not find equation index for {target_name}")
            
            phi_matrix = Matrix(self.phi)
            
            eq = gamma_col.multiply_elementwise(phi_matrix).multiply_elementwise(self.species1.sym).multiply_elementwise(self.species2.sym)

            # Insert the sink column in the right part of the equations
            self.eqs[:, eq_idx] = self.eqs[:, eq_idx] + eq  

        # 1) get the sym_name of the very first debris species
        debris_block = Matrix.zeros(n_shells, len(debris_species))
        for m in all_individual_eqs:
            debris_block += m

        # --- 2) Find where to insert it among all_species ---
        first_deb_name = debris_species[0].sym_name
        deb_start_idx = next(
            (j for j, sp in enumerate(all_species)
            if sp.sym_name == first_deb_name),
            None
        )
        if deb_start_idx is None:
            raise ValueError(f"Could not find debris species '{first_deb_name}' in all_species")

        # --- 3) Split the existing sink_block at that column index ---
        left  = sink_block[:, :deb_start_idx]
        right = sink_block[:, deb_start_idx:]

        # --- 4) Splice them together: [ left | debris_block | right ] ---
        self.eqs = left.row_join(debris_block).row_join(right)



if __name__ == "__main__":
    p1_in = np.array([
            m1,  # mass in kg
            r1,     # radius in meters
            2372.4,  # r_x in km, 1000 km
            2743.1,  # r_y in km
            6224.8,  # r_z in km
            -5.5,    # v_x in km/s
            -3.0,    # v_y in km/s
            3.8,     # v_z in km/s
            1      # object_class (dimensionless)
        ])

    p2_in = np.array([
            m1,     # mass in kg
            r1,     # radius in meters
            2372.4,  # r_x in km
            2743.1,  # r_y in km
            6224.8,  # r_z in km
            3.2,     # v_x in km/s
            5.4,     # v_y in km/s
            -3.9,    # v_z in km/s
            1      # object_class (dimensionless)
        ])
    import pickle 
    
    # open scnario properties
    with open(r'C:\Users\IT\Documents\UCL\pyssem\scenario-properties-elliptical.pkl', 'rb') as f:
        scen_properties = pickle.load(f)

    s1, s2 = scen_properties.species['rocket_body'][0], scen_properties.species['debris'][0]
    v1, v2 = s1.velocity_per_shells[6][6], s2.velocity_per_shells[6][6]
    debris_species = [species for species in scen_properties.species['debris']]

    # Mass Binning
    binC_mass = np.zeros(len(debris_species))
    binE_mass = np.zeros(2 * len(debris_species))
    binW_mass = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC_mass[index] = debris.mass
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW_mass[index] = debris.mass_ub - debris.mass_lb

    binE_mass = np.unique(binE_mass)

    # Eccentricity Binning, multiple debris species will have the same eccentricity bins
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.sort(binE_ecc)
    # Calculate the midpoints
    binE_ecc = (binE_ecc[:-1] + binE_ecc[1:]) / 2
    # Create bin edges starting at 0 and finishing at 1
    binE_ecc = np.concatenate(([0], binE_ecc, [1]))

    m1 = 1250  # kg
    m2 = 6  # kg
    r1 = 4  # radius in some units
    r2 = 0.11  # radius in some units
    collision_index = 6  # example collision index


    results = evolve_bins(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, 5, n_shells=10)

    print(results.shape)


