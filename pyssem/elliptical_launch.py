# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from utils.drag.drag import densityexp

# import pickle 
# with open('/Users/indigobrownhall/Code/pyssem/scenario-properties-collision.pkl', 'rb') as f:
#     collisions_scen = pickle.load(f)

# import numpy as np

# def coupled_ae_ode(t, y, param):
#     """
#     Compute the time‐derivatives of semi‐major axis a and eccentricity e under
#     your coupled-ae model.

#     Parameters
#     ----------
#     t : float
#         Current time.
#     y : array_like, shape (2,)
#         Current state [a, e].
#     param : object or dict
#         Must provide:
#           - req   : float   (equilibrium radius)
#           - mu    : float   (gravitational parameter)
#           - t_0   : float   (reference time)
#           - Bstar : float   (aerodynamic constant)
#         And you must have a function `densityexp_vec(x)` in scope.

#     Returns
#     -------
#     dydt : ndarray, shape (2,)
#         [da/dt, de/dt]
#     """
#     a, e = y[0], y[1]

#     # unpack parameters
#     re    = param.req    if hasattr(param, 'req')    else param['req']
#     mu    = param.mu     if hasattr(param, 'mu')     else param['mu']
#     t0    = param.t_0    if hasattr(param, 't_0')    else param['t_0']
#     Bstar = param.Bstar  if hasattr(param, 'Bstar')  else param['Bstar']

#     # mean motion
#     n0 = np.sqrt(mu) * a**(-1.5)

#     # initial angle factors
#     beta0 = (np.sqrt(3)/2) * e

#     # ambient density at a - re
#     a_minus_re = a - re
#     rho0 = densityexp(a_minus_re) * 1e9

#     # drag coefficient (floored to avoid zeros)
#     C0 = np.maximum(0.5 * Bstar * rho0, 1e-20)

#     # time since t0
#     τ = t - t0

#     # common phase for ȧ
#     φ_a = np.arctan((np.sqrt(3)/2)*e) \
#           - (np.sqrt(3)/2)*e * n0 * a * C0 * τ

#     # dadt
#     dadt = -(4/np.sqrt(3)) * (a**2 * n0 * C0 / e) \
#            * np.tan(φ_a) * (1/np.cos(φ_a))**2

#     # common phase for ė
#     φ_e = np.arctan(beta0) - beta0 * n0 * a * C0 * τ

#     # dedt
#     dedt = -e * n0 * a * C0 * (1/np.cos(φ_e))**2

#     return np.array([dadt, dedt])

# for term in collisions_scen.collision_terms:
#     # === 1. Sum over SMA × ECC for each (shell, mass) bin ===
#     totals = term.fragment_spread_totals.sum(axis=(2, 3), keepdims=True)  # shape: (shell, mass, 1, 1)

#     # === 2. Normalize safely using np.where (broadcasting-friendly) ===
#     with np.errstate(invalid='ignore', divide='ignore'):
#         spread_distribution = np.where(
#             totals > 0,
#             term.fragment_spread_totals / totals,
#             0.0
#         )

#     # === 3. Store for later use ===
#     term.spread_distribution = spread_distribution

#     # # === 4. Sanity check: each (shell, mass) should sum to ≈ 1.0 or 0.0 ===
#     # per_bin_sums = spread_distribution.sum(axis=(2, 3))
#     # print("Sanity check (each value should be ~1.0 or 0.0):")
#     # print(per_bin_sums)

# # === 1. Setup ===
# x0_sum = np.sum(collisions_scen.x0, axis=2)  # shape (n_shells, n_species)
# flat_vars = collisions_scen.all_symbolic_vars
# n_shells, n_species, n_ecc = collisions_scen.x0.shape

# # === 2. Lambdify each collision term’s eqs_sources ===
# for term in collisions_scen.collision_terms:
#     term.lambdified_sources = sp.lambdify(flat_vars, term.eqs_sources, modules="numpy")
#     term.lambdified_sinks = sp.lambdify(flat_vars, term.eqs_sinks, modules="numpy")

# # === 4. Integration ===

# def get_dadt(a_current, e_current, p, dt):
#         re   = p['req']
#         mu   = p['mu']
#         n0   = np.sqrt(mu) * a_current ** -1.5
#         a_minus_re = a_current - re
#         rho0 = densityexp(a_minus_re) * 1e9  # kg/km^3
#         C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)
#         # dt   = p['t'] - p['t_0']
#         ang  = np.arctan((np.sqrt(3)/2)*e_current) - (np.sqrt(3)/2)*e_current * n0 * a_current * C0 * dt
#         sec2 = 1.0 / np.cos(ang) ** 2
#         return -(4 / np.sqrt(3)) * (a_current**2 * n0 * C0 / e_current) * np.tan(ang) * sec2

# def get_dedt(a_current, e_current, p, dt):
#     re   = p['req']
#     mu   = p['mu']
#     n0   = np.sqrt(mu) * a_current ** -1.5
#     beta = (np.sqrt(3)/2) * e_current
#     a_minus_re = a_current - re
#     rho0 = densityexp(a_minus_re) * 1e9
#     C0   = max(0.5 * p['Bstar'] * rho0, 1e-20)
#     # dt   = p['t'] - p['t_0']
#     arg  = np.arctan(beta) - beta * n0 * a_current * C0 * dt
#     sec2 = 1.0 / np.cos(arg) ** 2
#     return -e_current * n0 * a_current * C0 * sec2

# t_0 = 0
# hours = 3600.0
# days = 24.0 * hours
# years = 365.25 * days

# # def population_rhs(t, x_flat):
# #     global t_0

# #     t = t * years

# #     if t_0 is None:
# #         t_0 = t

# #     dt = t - t_0  # change in seconds between steps of integrator

# #     # Step 1: Reshape flat population input to (sma_shells, species, ecc)
# #     x_matrix = x_flat.reshape((5, 4, 10))  # shape: (sma_shells, species, ecc)

# #     # Step 2: Apply time_in_shell to map into altitude shells
# #     # time_in_shell has shape: (alt_shells, sma_shells, ecc)
# #     time_in_shell = collisions_scen.time_in_shell  # shape: (alt_shells, sma_shells, ecc)
# #     n_alt_shells = time_in_shell.shape[0]

# #     # Step 3: Multiply time_in_shell fraction with each (sma, species, ecc) population
# #     # We'll loop over species dimension for clarity
# #     effective_altitude_matrix = np.zeros((5, 4))

# #     # keep track of which a e bins, for each species, are contributing to each shell. Used in the sink equations.
# #     normalised_species_distribution_in_sma_e_space = np.zeros((n_shells, n_species, n_shells, n_ecc))

# #     # for each species, in each shell, trying to find the ae that contribute to those bins. 
# #     for species in range(n_species):
# #         for shell in range(n_shells):
# #             n_effective = 0
# #             for sma in range(n_alt_shells):
# #                 for ecc in range(10): # change later
# #                     tis = time_in_shell[shell, ecc, sma]
# #                     n_pop = x_matrix[sma, species, ecc]
# #                     n_effective_a_e = n_pop * tis
# #                     n_effective = n_effective + n_effective_a_e
# #                     normalised_species_distribution_in_sma_e_space[shell, species, sma, ecc] = n_effective_a_e

# #             normalised_species_distribution_in_sma_e_space[shell, species, :, :] = ( normalised_species_distribution_in_sma_e_space[shell, species, :, :] / n_effective )
# #             effective_altitude_matrix[shell, species] = n_effective

# #     total_dNdt_alt = np.zeros((n_shells, n_species))
# #     total_dNdt_sma_ecc_sources = np.zeros((n_shells, n_species, n_ecc))

# #     x_flat_ordered = effective_altitude_matrix.flatten()

# #     # collision pair in altitude space 
# #     for term in collisions_scen.collision_terms:
# #         dNdt_term = term.lambdified_sources(*x_flat_ordered)

# #         # n_shells x n_species
# #         total_dNdt_alt += np.array(dNdt_term, dtype=float)

# #         # should be n_species, n_sma, n_ecc 
# #         test = term.spread_distribution # (n_shells, n_deb_species, n_shells, n_ecc)
# #         _, n_deb_species, _, _ = term.spread_distribution.shape

# #         # multiply the growth rate for each species by the distribution of that species in a,e space
# #         for shell in range(n_shells):
# #             for species in range(n_species):
# #                 # Get the mass bin index (skip if not a debris species)
# #                 mass_bin = species_to_mass_bin.get(species, None)
# #                 if mass_bin is None:
# #                     # Add this slice to total_dNdt_sma_ecc as zeros - as no growth fragments
# #                     continue

# #                 sma_ecc_distribution = term.spread_distribution[shell, mass_bin, :, :] # this should be to equal to one
# #                 species_frag = total_dNdt_alt[shell, species] # get the column of the debris species
# #                 frag_spread_sma_ecc = species_frag * sma_ecc_distribution
# #                 total_dNdt_sma_ecc_sources[:, species, :] = frag_spread_sma_ecc


# #     dNdt_sink_sma_ecc = np.zeros((n_shells, n_species, n_ecc)) 
# #     # for the sink
# #     # the sink equations and multiply them by the time in shell, then take away that
# #     for term in collisions_scen.collision_terms: # species pair
# #         dNdt_term = term.lambdified_sinks(*x_flat_ordered) # n_shells x n_species
        
# #         for species in range(n_species):
# #             species_dNdt_sink_sma_ecc = np.zeros((n_shells, n_ecc))
# #             for shell in range(n_shells):
                
# #                 frag = dNdt_term[shell, species]
# #                 norm_a_e = normalised_species_distribution_in_sma_e_space[shell, species, :, :]
# #                 frag_sink_sma_ecc = frag * norm_a_e
# #                 dNdt_sink_sma_ecc[:, species, :] = dNdt_sink_sma_ecc[:, species, :] + frag_sink_sma_ecc
        

# #     output = total_dNdt_sma_ecc_sources + dNdt_sink_sma_ecc

# #     # # now we need to propagate using the dynamical equations
# #     # param = {
# #     #     'req': 6378.136, 
# #     #     'mu': 398600.0, # should already be defined
# #     #     'Bstar': 2.2 * (1e-6 / 100.0), # this will change for each species, km^2
# #     #     'j2': 1082.63e-6
# #     #     # 't_0' : t_0,
# #     #     # 't' : t
# #     # }

# #     # # for each species
# #     # binE_ecc = collisions_scen.eccentricity_bins
# #     # binE_ecc = np.sort(binE_ecc)
# #     # binE_ecc_mid_point = (binE_ecc[:-1] + binE_ecc[1:]) / 2
# #     # Δa      = collisions_scen.sma_HMid_km[1] - collisions_scen.sma_HMid_km[0]
# #     # Δe      = collisions_scen.eccentricity_bins[1] - collisions_scen.eccentricity_bins[0]

# #     # # Calculate the midpoints
# #     # N = np.zeros((len(collisions_scen.sma_HMid_km), len(binE_ecc_mid_point)))
# #     # adot = np.zeros_like(N)
# #     # edot = np.zeros_like(N) 
# #     # for r in range(n_shells):
# #     #     a_val = collisions_scen.sma_HMid_km[r]
# #     #     for c in range(len(binE_ecc_mid_point)):
# #     #         e_val      = binE_ecc_mid_point[c]
# #     #         adot[r, c] = get_dadt(a_val, e_val, param, dt)
# #     #         edot[r, c] = get_dedt(a_val, e_val, param, dt)


# #     # # so we no have the change of the points, we need to multiply each species sma and ecc by this matrix of change
# #     # dN_all_species = np.zeros_like(output)
# #     # for species in range(n_species):
# #     #     N_sma_ecc = x_matrix[:, species, :]
# #     #     dN = np.zeros_like(N_sma_ecc)
        
# #     #     for r in range(len(collisions_scen.sma_HMid_km) - 1, -1, -1):
# #     #         for c in range(len(binE_ecc_mid_point) - 1, -1, -1):
# #     #             Nrc = N_sma_ecc[r, c]
# #     #             out_a = Nrc * adot[r, c] * dt / Δa
# #     #             out_e = Nrc * edot[r, c] * dt / Δe

# #     #             total_out = out_a + out_e
# #     #             if abs(total_out) > Nrc and Nrc > 0:
# #     #                 factor = Nrc / abs(total_out)
# #     #                 out_a *= factor
# #     #                 out_e *= factor
# #     #             elif Nrc == 0:
# #     #                 out_a = 0
# #     #                 out_e = 0

# #     #             dN[r, c] += out_a + out_e
# #     #             if r > 0: dN[r-1, c]   -= out_a
# #     #             if c > 0: dN[r, c-1]   -= out_e

# #     #     dN_all_species[:, species, :] = dN
        
# #     # t_0 = t # update global variable 
# #     print(t)
# #     # dN_all_species = dN_all_species + output
# #     return output.flatten()


# def population_rhs(t, x_flat):
#     # t, time
#     # x_flat is population n_shells_sma x n_species, n_ecc, flattened 

#     # for each shell (altitude bin), sum across the sma x ecc components, then multiply the fractional time in shell for that x population of that ae bin 
#     # this should be the same number of shells, this will give you the effective number in that altitude shell. 

#     # if you sum this, this will be the same as the initial population

#     # Step 1: Reshape flat population input to (sma_shells, species, ecc)
#     x_matrix = x_flat.reshape((5, 4, 10))  # shape: (sma_shells, species, ecc)

#     # Step 2: Apply time_in_shell to map into altitude shells
#     # time_in_shell has shape: (alt_shells, sma_shells, ecc)
#     time_in_shell = collisions_scen.time_in_shell  # shape: (alt_shells, sma_shells, ecc)
#     n_alt_shells = time_in_shell.shape[0]

#     # Step 3: Multiply time_in_shell fraction with each (sma, species, ecc) population
#     # We'll loop over species dimension for clarity
#     effective_altitude_matrix = np.zeros((5, 4))

#     # keep track of which a e bins, for each species, are contributing to each shell. Used in the sink equations.
#     normalised_species_distribution_in_sma_e_space = np.zeros((n_shells, n_species, n_shells, n_ecc))

#     # for each species, in each shell, trying to find the ae that contribute to those bins. 
#     for species in range(n_species):
#         for shell in range(n_shells):
#             n_effective = 0
#             for sma in range(n_alt_shells):
#                 for ecc in range(10): # change later
#                     tis = time_in_shell[shell, ecc, sma]
#                     n_pop = x_matrix[sma, species, ecc]
#                     n_effective_a_e = n_pop * tis
#                     n_effective = n_effective + n_effective_a_e
#                     normalised_species_distribution_in_sma_e_space[shell, species, sma, ecc] = n_effective_a_e

#             normalised_species_distribution_in_sma_e_space[shell, species, :, :] = ( normalised_species_distribution_in_sma_e_space[shell, species, :, :] / n_effective )
#             effective_altitude_matrix[shell, species] = n_effective

#     total_dNdt_alt = np.zeros((n_shells, n_species))
#     total_dNdt_sma_ecc_sources = np.zeros((n_shells, n_species, n_ecc))

#     x_flat_ordered = effective_altitude_matrix.flatten()

#     # collision pair in altitude space 
#     for term in collisions_scen.collision_terms:
#         dNdt_term = term.lambdified_sources(*x_flat_ordered)

#         # n_shells x n_species
#         total_dNdt_alt += np.array(dNdt_term, dtype=float)

#         # should be n_species, n_sma, n_ecc 
#         test = term.spread_distribution # (n_shells, n_deb_species, n_shells, n_ecc)
#         _, n_deb_species, _, _ = term.spread_distribution.shape

#         # multiply the growth rate for each species by the distribution of that species in a,e space
#         for shell in range(n_shells):
#             for species in range(n_species):
#                 # Get the mass bin index (skip if not a debris species)
#                 mass_bin = species_to_mass_bin.get(species, None)
#                 if mass_bin is None:
#                     # Add this slice to total_dNdt_sma_ecc as zeros - as no growth fragments
#                     continue

#                 sma_ecc_distribution = term.spread_distribution[shell, mass_bin, :, :] # this should be to equal to one

#                 species_frag = total_dNdt_alt[shell, species] # get the column of the debris species
#                 frag_spread_sma_ecc = species_frag * sma_ecc_distribution

#                 total_dNdt_sma_ecc_sources[:, species, :] = frag_spread_sma_ecc


#     dNdt_sink_sma_ecc = np.zeros((n_shells, n_species, n_ecc)) 
#     # for the sink
#     # the sink equations and multiply them by the time in shell, then take away that
#     for term in collisions_scen.collision_terms: # species pair
#         dNdt_term = term.lambdified_sinks(*x_flat_ordered) # n_shells x n_species
        
#         for species in range(n_species):
#             species_dNdt_sink_sma_ecc = np.zeros((n_shells, n_ecc))
#             for shell in range(n_shells):
                
#                 frag = dNdt_term[shell, species]
#                 norm_a_e = normalised_species_distribution_in_sma_e_space[shell, species, :, :]
#                 frag_sink_sma_ecc = frag * norm_a_e
#                 dNdt_sink_sma_ecc[:, species, :] = dNdt_sink_sma_ecc[:, species, :] + frag_sink_sma_ecc
        

#     output = total_dNdt_sma_ecc_sources + dNdt_sink_sma_ecc
#     print(t)
#     return output.flatten()

# # Find where debris species start (assume all debris species are contiguous)
# debris_start_idx = next(i for i, name in enumerate(collisions_scen.species_names) if name.startswith("N"))
# n_mass_bins = sum(name.startswith("N") for name in collisions_scen.species_names)

# # Map species index to mass bin index, only for debris
# species_to_mass_bin = {
#     debris_start_idx + i: i
#     for i in range(n_mass_bins)
# }

# # output = solve_ivp(
# #     fun=population_rhs,
# #     t_span=(collisions_scen.scen_times[0], collisions_scen.scen_times[-1]),
# #     y0=collisions_scen.x0.flatten(),
# #     t_eval=collisions_scen.scen_times,
# #     method="RK45"
# # )
# t_eval = np.linspace(0, 100, 1000)
# tester_over_time = np.zeros((len(t_eval), n_shells, n_species, 13))

# output = solve_ivp(
#     fun=population_rhs,
#     t_span=(t_eval[0], t_eval[-1]),
#     y0=collisions_scen.x0.flatten(),
#     t_eval=t_eval,
#     method="RK45"
# )

# n_time = len(collisions_scen.scen_times)
# expected_size = n_time * n_shells * n_species * n_ecc

# if not output.success:
#     raise RuntimeError(f"ODE solver failed: {output.message}")

# # 4) Check that the data can be reshaped
# actual_size = output.y.size
# if actual_size != expected_size:
#     raise ValueError(
#         f"Unexpected output size: got {actual_size} elements, "
#         f"but expected {expected_size} (n_time×n_shells×n_species×n_ecc)."
#     )

# # === 5. Reshape and plot total population per species ===

# pop_over_time = output.y.T.reshape(len(collisions_scen.scen_times), n_shells, n_species, n_ecc)

# # Sum over sma and ecc to get total per species
# total_pop = pop_over_time.sum(axis=(1, 3))  # shape: (n_time, n_species)

# plt.figure(figsize=(10, 6))
# for j in range(n_species):
#     plt.plot(collisions_scen.scen_times, total_pop[:, j], label=f"Species {j}")
# plt.xlabel("Time")
# plt.ylabel("Total Population")
# plt.title("Total Population per Species Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# pop_by_shell = pop_over_time.sum(axis=(2, 3))

# times = collisions_scen.scen_times           # shape (n_time,)
# shell_centers = collisions_scen.sma_HMid_km  # shape (n_shells,)

# # 2) One figure, one axes
# fig, ax = plt.subplots(figsize=(10, 6))

# # 3) Plot each shell
# for i, center in enumerate(shell_centers):
#     ax.plot(times, pop_by_shell[:, i], label=f"{center:.0f} km")

# # 4) Labels & legend
# ax.set_xlabel("Time")
# ax.set_ylabel("Population in Shell")
# ax.set_title("Population in Each Altitude Shell Over Time")
# ax.grid(True)
# ax.legend(title="Shell midpoints", loc="upper right", ncol=2, fontsize="small")

# plt.tight_layout()
# plt.show()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import pickle 
with open('/Users/indigobrownhall/Code/pyssem/scenario-properties-collision.pkl', 'rb') as f:
    collisions_scen = pickle.load(f)

for term in collisions_scen.collision_terms:
    # === 1. Sum over SMA × ECC for each (shell, mass) bin ===
    totals = term.fragment_spread_totals.sum(axis=(2, 3), keepdims=True)  # shape: (shell, mass, 1, 1)

    # === 2. Normalize safely using np.where (broadcasting-friendly) ===
    with np.errstate(invalid='ignore', divide='ignore'):
        spread_distribution = np.where(
            totals > 0,
            term.fragment_spread_totals / totals,
            0.0
        )

    # === 3. Store for later use ===
    term.spread_distribution = spread_distribution

    # === 4. Sanity check: each (shell, mass) should sum to ≈ 1.0 or 0.0 ===
    per_bin_sums = spread_distribution.sum(axis=(2, 3))
    print("Sanity check (each value should be ~1.0 or 0.0):")
    print(per_bin_sums)

# === 1. Setup ===
x0_sum = np.sum(collisions_scen.x0, axis=2)  # shape (n_shells, n_species)
flat_vars = collisions_scen.all_symbolic_vars
n_shells, n_species, n_ecc = collisions_scen.x0.shape

# === 2. Lambdify each collision term’s eqs_sources ===
for term in collisions_scen.collision_terms:
    term.lambdified_sources = sp.lambdify(flat_vars, term.eqs_sources, modules="numpy")
    term.lambdified_sinks = sp.lambdify(flat_vars, term.eqs_sinks, modules="numpy")

# === 3. Define population_rhs using summed numerical output ===
# def population_rhs(t, x_flat):
#     x_matrix = x_flat.reshape((n_shells, n_species))
#     x_flat_ordered = x_matrix.flatten()

#     total_dNdt = np.zeros_like(x_matrix)

#     for term in collisions_scen.collision_terms:
#         dNdt_term = term.lambdified_sources(*x_flat_ordered)
#         total_dNdt += np.array(dNdt_term, dtype=float)  # ensure it's a NumPy array

#     return total_dNdt.flatten()

# tester = np.zeros_like(collisions_scen.x0)
# === 4. Integration ===
t_eval = np.linspace(0, 100, 1000)
tester_over_time = np.zeros((len(t_eval), n_shells, n_species, 13))


def population_rhs(t, x_flat):
    # t, time
    # x_flat is population n_shells_sma x n_species, n_ecc, flattened 

    # for each shell (altitude bin), sum across the sma x ecc components, then multiply the fractional time in shell for that x population of that ae bin 
    # this should be the same number of shells, this will give you the effective number in that altitude shell. 

    # if you sum this, this will be the same as the initial population

    # Step 1: Reshape flat population input to (sma_shells, species, ecc)
    x_matrix = x_flat.reshape((5, 4, 10))  # shape: (sma_shells, species, ecc)

    # Step 2: Apply time_in_shell to map into altitude shells
    # time_in_shell has shape: (alt_shells, sma_shells, ecc)
    time_in_shell = collisions_scen.time_in_shell  # shape: (alt_shells, sma_shells, ecc)
    n_alt_shells = time_in_shell.shape[0]

    # Step 3: Multiply time_in_shell fraction with each (sma, species, ecc) population
    # We'll loop over species dimension for clarity
    effective_altitude_matrix = np.zeros((5, 4))

    # keep track of which a e bins, for each species, are contributing to each shell. Used in the sink equations.
    normalised_species_distribution_in_sma_e_space = np.zeros((n_shells, n_species, n_shells, n_ecc))

    # for each species, in each shell, trying to find the ae that contribute to those bins. 
    for species in range(n_species):
        for shell in range(n_shells):
            n_effective = 0
            for sma in range(n_alt_shells):
                for ecc in range(10): # change later
                    tis = time_in_shell[shell, ecc, sma]
                    n_pop = x_matrix[sma, species, ecc]
                    n_effective_a_e = n_pop * tis
                    n_effective = n_effective + n_effective_a_e
                    normalised_species_distribution_in_sma_e_space[shell, species, sma, ecc] = n_effective_a_e

            normalised_species_distribution_in_sma_e_space[shell, species, :, :] = ( normalised_species_distribution_in_sma_e_space[shell, species, :, :] / n_effective )
            effective_altitude_matrix[shell, species] = n_effective

    total_dNdt_alt = np.zeros((n_shells, n_species))
    total_dNdt_sma_ecc_sources = np.zeros((n_shells, n_species, n_ecc))

    x_flat_ordered = effective_altitude_matrix.flatten()

    # collision pair in altitude space 
    for term in collisions_scen.collision_terms:
        dNdt_term = term.lambdified_sources(*x_flat_ordered)

        # n_shells x n_species
        total_dNdt_alt += np.array(dNdt_term, dtype=float)

        # should be n_species, n_sma, n_ecc 
        test = term.spread_distribution # (n_shells, n_deb_species, n_shells, n_ecc)
        _, n_deb_species, _, _ = term.spread_distribution.shape

        # multiply the growth rate for each species by the distribution of that species in a,e space
        for shell in range(n_shells):
            for species in range(n_species):
                # Get the mass bin index (skip if not a debris species)
                mass_bin = species_to_mass_bin.get(species, None)
                if mass_bin is None:
                    # Add this slice to total_dNdt_sma_ecc as zeros - as no growth fragments
                    continue

                sma_ecc_distribution = term.spread_distribution[shell, mass_bin, :, :] # this should be to equal to one

                species_frag = total_dNdt_alt[shell, species] # get the column of the debris species
                frag_spread_sma_ecc = species_frag * sma_ecc_distribution

                total_dNdt_sma_ecc_sources[:, species, :] = frag_spread_sma_ecc


    dNdt_sink_sma_ecc = np.zeros((n_shells, n_species, n_ecc)) 
    # for the sink
    # the sink equations and multiply them by the time in shell, then take away that
    for term in collisions_scen.collision_terms: # species pair
        dNdt_term = term.lambdified_sinks(*x_flat_ordered) # n_shells x n_species
        
        for species in range(n_species):
            species_dNdt_sink_sma_ecc = np.zeros((n_shells, n_ecc))
            for shell in range(n_shells):
                
                frag = dNdt_term[shell, species]
                norm_a_e = normalised_species_distribution_in_sma_e_space[shell, species, :, :]
                frag_sink_sma_ecc = frag * norm_a_e
                dNdt_sink_sma_ecc[:, species, :] = dNdt_sink_sma_ecc[:, species, :] + frag_sink_sma_ecc
        

    output = total_dNdt_sma_ecc_sources + dNdt_sink_sma_ecc
    print(t)
    return output.flatten()


# Find where debris species start (assume all debris species are contiguous)
debris_start_idx = next(i for i, name in enumerate(collisions_scen.species_names) if name.startswith("N"))
n_mass_bins = sum(name.startswith("N") for name in collisions_scen.species_names)

# Map species index to mass bin index, only for debris
species_to_mass_bin = {
    debris_start_idx + i: i
    for i in range(n_mass_bins)
}

output = solve_ivp(
    fun=population_rhs,
    t_span=(t_eval[0], t_eval[-1]),
    y0=collisions_scen.x0.flatten(),
    t_eval=t_eval,
    method="RK45"
)

# === 5. Reshape and plot total population per species ===
n_time = len(t_eval)
pop_over_time = output.y.T.reshape(n_time, n_shells, n_species, n_ecc)

# Sum over sma and ecc to get total per species
total_pop = pop_over_time.sum(axis=(1, 3))  # shape: (n_time, n_species)

plt.figure(figsize=(10, 6))
for j in range(n_species):
    plt.plot(t_eval, total_pop[:, j], label=f"Species {j}")
plt.xlabel("Time")
plt.ylabel("Total Population")
plt.title("Total Population per Species Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()