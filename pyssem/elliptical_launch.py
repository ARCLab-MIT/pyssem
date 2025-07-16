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

    # # === 4. Sanity check: each (shell, mass) should sum to ≈ 1.0 or 0.0 ===
    # per_bin_sums = spread_distribution.sum(axis=(2, 3))
    # print("Sanity check (each value should be ~1.0 or 0.0):")
    # print(per_bin_sums)

# === 1. Setup ===
x0_sum = np.sum(collisions_scen.x0, axis=2)  # shape (n_shells, n_species)
flat_vars = collisions_scen.all_symbolic_vars
n_shells, n_species, n_ecc = collisions_scen.x0.shape

# === 2. Lambdify each collision term’s eqs_sources ===
for term in collisions_scen.collision_terms:
    term.lambdified_sources = sp.lambdify(flat_vars, term.eqs_sources, modules="numpy")
    term.lambdified_sinks = sp.lambdify(flat_vars, term.eqs_sinks, modules="numpy")

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
