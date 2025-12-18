
from sympy import zeros, symbols
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import json

#==========================================================================
# Functions used to create symbolic variables for the controller
#==========================================================================

def control_none(t, h, species_properties, scen_properties):
    return zeros(scen_properties.n_shells, 1)

def control_launch_sym(t, h, species_properties, scen_properties):
    U = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        U[k, 0] = symbols(f"u_l_{species_properties.sym_name}{k+1}") 

    return U

def control_adr_sym(t, h, species_properties, scen_properties):
    U = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        U[k, 0] = symbols(f"u_r_{species_properties.sym_name}{k+1}") 

    return U

#==========================================================================
# Launch rate - options
#==========================================================================

# Null 
def lam_f1(t):
    return 0

# Time and shell constant based on constant multiplier of the initial population
def lam_f2(t, const, ref, active_species_indices, N_shell):
    out = []
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[active_species_indices[species_index]] * ref[start_idx:end_idx]
    return out

# Time and shell varying based on random multiplier of the initial population
def lam_f3(t, ref, active_species_per_shells):
    return np.random.rand(active_species_per_shells) * ref[:active_species_per_shells]

# Time and shell varying based on mean and std
def lam_f4(t, active_species_per_shells):
    lam_mean = 100
    lam_std = 15
    return np.round(lam_mean + lam_std * randn2(active_species_per_shells)).astype(int)

# Time and shell based on cosinusoidal function
def lam_f5(t, ref, active_species_per_shells):
    phase_shifts = 2 * np.pi / np.random.randint(1, active_species_per_shells + 1, size=active_species_per_shells)
    return 0.5 * ref[:active_species_per_shells] * (np.cos((2*np.pi/10)*t + phase_shifts) / 6 + 1)

#==========================================================================
# PMD - options
#==========================================================================

# 1. Time and shell constant 
def pmd_f1(t,const, active_species_indices, N_shell):
    out = []
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[active_species_indices[species_index]] * np.ones(N_shell)
    return out

# 2. Increasing over time, constant for each shell
def pmd_f2(t,lower_bound,upper_bound, active_species_indices, N_shell, tspan):
    out = []
    for species_index in range(active_species_indices):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell    
        out[start_idx:end_idx] = ( lower_bound[species_index] + ( t / tspan[-1] * (upper_bound[species_index] - lower_bound[species_index])) ) * np.ones(N_shell)
    return out

# 3. Increasing over time with randomness, constant for each shell
def pmd_f3(t,lower_bound,upper_bound, active_species_indices, N_shell):
    out = []
    random_variations = (np.random.rand(active_species_indices) - 0.5) * 0.5
    final_sequence = t / 100 + random_variations
    final_sequence = np.clip(final_sequence, 0, upper_bound) # Keep values within [0, upper_bound]
    for species_index in range(active_species_indices):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell    
        out[start_idx:end_idx] = ( lower_bound[species_index] + ( final_sequence[species_index] * (upper_bound[species_index] - lower_bound[species_index])) ) * np.ones(N_shell)
    return out

#==========================================================================
# Orbital lifetime [years] - options
#==========================================================================

# 1. Time and shell constant
def deltat_f1(t,const, active_species_indices, N_shell):
    out = []
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[active_species_indices[species_index]] * np.ones(N_shell)
    return out

#==========================================================================
# Criticality/Capacity Metrics
#==========================================================================

# def cum_CSI_orig(obj, baseline):
#     # Maya's code

#     N_shell = baseline.n_shells  # Number of shells
#     R02 = baseline.R0_km  # Altitude bins (shell boundaries) in km
#     num_species = baseline.species_length  # Number of species
#     re = baseline.re

#     # Reference values for normalization
#     M0 = 10000
#     D0 = 5e-8  # Arbitrarily chosen, may need revision
#     life0 = 1468
#     A0 = 1

#     # Cumulative CSI variables
#     num_species = len(baseline.species_names)
#     cum_CSI_total = 0
#     csi_per_shell = np.zeros(N_shell)
#     csi_per_species = np.zeros(num_species)
#     csi_per_species_per_shell = np.zeros((num_species, N_shell))

#     species_mass = []
#     species_area = []
#     total_objects_per_shell = np.zeros(N_shell)
#     total_mass_per_shell = np.zeros(N_shell)
#     total_area_per_shell = np.zeros(N_shell)
#     total_objects_per_species = np.zeros(len(baseline.species_names))
#     total_mass_per_species = np.zeros(len(baseline.species_names))
#     total_area_per_species = np.zeros(len(baseline.species_names))
    
#     # Store species mass and area in lists by index
#     for species_index, (species_name, species_list) in enumerate(baseline.species_cells.items()):
#         for species_properties in species_list:
#             species_mass.append(species_properties.mass)
#             species_area.append(species_properties.A)

#     total_objects_per_shell = [0] * N_shell
#     total_mass_per_shell = [0] * N_shell
#     total_area_per_shell = [0] * N_shell

#     for shell_index in range(N_shell):
#         for species_index in range(num_species):
#             idx = shell_index + species_index * N_shell  # Corrected indexing
#             total_objects_per_shell[shell_index] += obj[idx]
#             total_mass_per_shell[shell_index] += obj[idx] * species_mass[species_index]
#             total_area_per_shell[shell_index] += obj[idx] * species_area[species_index]

#     D = [0] * N_shell
#     lifetime = [0] * N_shell

#     # Calculate and sum CSI for each shell for total CSI
#     for shell_index in range(N_shell):
#         h = (R02[shell_index] + R02[shell_index + 1]) / 2
#         lifetime[shell_index] = 10 ** (14.18 * (h ** 0.1831) - 42.94)
#         M = total_mass_per_shell[shell_index]
#         r_inner = re + R02[shell_index]
#         r_outer = re + R02[shell_index + 1]
#         volume_outer = (4 / 3) * np.pi * (r_outer**3)
#         volume_inner = (4 / 3) * np.pi * (r_inner**3)
#         V = volume_outer - volume_inner
#         D[shell_index] = M / V
#         #weighted_area = total_area_per_shell[shell_index] / total_objects_per_shell[shell_index]
#         csi_per_shell[shell_index] = M / M0 * D[shell_index] / D0 * lifetime[shell_index] / life0
#         cum_CSI_total += csi_per_shell[shell_index]  # Accumulate CSI for this shell
#         #cum_CSI_total += M / M0 * D / D0 * lifetime / life0 * weighted_area / A0

#     # Initialize storage arrays
#     total_objects_per_species = [0] * num_species
#     total_mass_per_species = [0] * num_species
#     total_area_per_species = [0] * num_species

#     # Calculate CSI for each species
#     for species_index in range(num_species):
#         for shell_index in range(N_shell):
#             idx = shell_index + species_index * N_shell  # Correct species-shell indexing

#             # Species-specific mass, area, and object count in this shell
#             num_objects = obj[idx]
#             mass = num_objects * species_mass[species_index]
#             area = num_objects * species_area[species_index]

#             csi_per_species_per_shell[species_index, shell_index] = (mass / M0) * (D[shell_index]/D0) * (lifetime[shell_index] / life0)

#             # Accumulate total per species
#             total_objects_per_species[species_index] += num_objects
#             total_mass_per_species[species_index] += mass
#             total_area_per_species[species_index] += area

#             csi_per_species[species_index] += csi_per_species_per_shell[species_index, shell_index]  # Accumulate CSI for this species 
    
#     return csi_per_species, csi_per_shell, csi_per_species_per_shell, cum_CSI_total

# def cum_CSI_old(obj, baseline):
#     # Gio's edit from Maya's code
#     N_shell = baseline.n_shells  # Number of shells
#     R02 = baseline.R0_km  # Altitude bins (shell boundaries) in km
#     num_species = len(baseline.species_names)  # Number of species
#     re = baseline.re

#     # Reference values for normalization
#     M0 = 10000
#     D0 = 5e-8
#     life0 = 1468

#     # Initialize output array
#     csi_per_species_per_shell = np.zeros((num_species, N_shell))

#     # Extract species mass and area directly as NumPy arrays
#     species_mass = np.array([prop.mass for species_list in baseline.species_cells.values() for prop in species_list])

#     # Reshape the object counts for efficient vectorized calculation
#     obj_matrix = obj.reshape(num_species, N_shell)

#     # Calculate total mass per shell using vectorized operations
#     total_mass_per_shell = np.sum(obj_matrix * species_mass[:, None], axis=0)

#     D = np.zeros(N_shell)
#     lifetime = np.zeros(N_shell)

#     # Vectorized calculation of D and lifetime
#     h = (R02[:-1] + R02[1:]) / 2  # Use slicing for vectorized addition
#     # lifetime[:] = 10 ** (14.18 * (h ** 0.1831) - 42.94) # maya
#     lifetime[:] = np.exp(14.18 * (h ** 0.1831) - 42.94)
#     r_inner = re + R02[:-1]
#     r_outer = re + R02[1:]
#     volume_outer = (4 / 3) * np.pi * (r_outer**3)
#     volume_inner = (4 / 3) * np.pi * (r_inner**3)
#     V = volume_outer - volume_inner
#     V = np.array(baseline.V)
#     D[:] = total_mass_per_shell / V

#     # Calculate CSI for each species per shell using vectorized operations
#     mass_matrix = obj_matrix * species_mass[:, None]
#     csi_per_species_per_shell[:] = (mass_matrix / M0) * (D / D0) * (lifetime / life0) # * (1/(1+0.6))         

#     return csi_per_species_per_shell.flatten()

def cum_CSI(obj, baseline):
    # Validated by Gio on 06/27/25 based on the matlab mocat-3 notebook

    # --- Extract baseline parameters ---
    N_shell = baseline.n_shells
    R02 = baseline.R0_km
    num_species = len(baseline.species_names)
    V_per_shell = np.array(baseline.V)

    # --- Parameters from the new model in `cum_CSI_new` ---
    k = 0.6
    cos_i_av = 2 / np.pi  # Average value of cos(i)
    Gamma_av = (1 - cos_i_av) / 2
    
    # New inclination-based physics factor
    inclination_factor = (1 + k * Gamma_av)

    # --- Reference values from the new model ---
    M_ref = 10000  # kg
    life_h_ref = 1468  # years, corresponds to lifetime at 1000 km

    # initial_populations = baseline.x0.T.values.flatten()
    # D_ref = np.max(np.sum(initial_populations, axis=0) / V_per_shell)
    initial_populations_flat = np.array(baseline.x0)
    initial_populations_matrix = initial_populations_flat.reshape(num_species, N_shell)
    total_objects_per_shell = np.sum(initial_populations_matrix, axis=0) # Shape: (24,)
    initial_number_density_per_shell = total_objects_per_shell / V_per_shell # Shape: (24,)
    D_ref = np.max(initial_number_density_per_shell)

    # Calculate the single, combined denominator for normalization
    den = M_ref * D_ref * life_h_ref * (1 + k) #/ 10

    # --- Vectorized Calculations ---
    # 1. Reshape the input population counts into a (species, shell) matrix
    obj_matrix = obj.reshape(num_species,N_shell)

    # 2. Get species mass as a column vector for broadcasting
    species_mass = np.array([prop.mass for species_list in baseline.species_cells.values() for prop in species_list])
    species_mass_col = species_mass[:, np.newaxis] # Shape: (num_species, 1)

    # 3. Calculate shell-dependent properties (lifetime and volume)
    h = (R02[:-1] + R02[1:]) / 2
    lifetime_per_shell = np.exp(14.18 * (h ** 0.1831) - 42.94) # Shape: (N_shell,)

    # 4. Calculate Species Number Density (D_X) for each species and shell
    # This replaces the old total mass density calculation.
    # Broadcasting (num_species, N_shell) / (1, N_shell)
    D_X_matrix = obj_matrix / V_per_shell

    # 5. Calculate the numerator term (dum_X)
    # Combines mass, lifetime, and the new inclination factor.
    # Broadcasting (num_species, 1) * (1, N_shell)
    num_term = lifetime_per_shell * inclination_factor
    dum_X_matrix = species_mass_col * num_term
    
    # 6. Calculate the final CSI matrix using the formula from `cum_CSI_new`
    csi_per_species_per_shell = (D_X_matrix * dum_X_matrix) / den

    return csi_per_species_per_shell.flatten()

# def cum_CSI_new(self):
#     # From pyssem code; to use it in the jup)yter notebook, you need to call it with the baseline object: cum_CSI_new(baseline)
#     # I think the cum_CSI above is the correct one, this one has some bugs 

#     baseline = self
#     baseline.results = {}
#     baseline.results['times'] = baseline.output['t']
    
#     n_species = baseline.species_length
#     num_shells = baseline.n_shells
#     species_names = baseline.species_names
#     # Initialize the data dictionary
#     data = {"population_data": []}
#     # Initialize population data structure
#     population_data_dict = {species: [[0] * len( baseline.results['times']) for _ in range(num_shells)]
#                             for species in species_names}
#     # Populate population data
#     for i in range(n_species):
#         species = species_names[i]
#         for j in range(num_shells):
#             shell_index = i * num_shells + j
#             population_data_dict[species][j] = baseline.output['y'][shell_index, :].tolist()
#             shell_data = {
#                 "species": species,
#                 "shell": j + 1,
#                 "populations": baseline.output['y'][shell_index, :].tolist()
#             }
#             data["population_data"].append(shell_data)
#     baseline.results['population_data'] = data["population_data"]    

#     k = 0.6
#     def life(h):
#         return np.exp(14.18 * h ** 0.1831 - 42.94)

#     M_ref = 10000 # kg
#     h_ref = 1000 # km
#     life_h_ref = 1468 # years, it corresponds to life0 = life(1000)

#     initial_populations = [data['populations'][0] for data in self.results['population_data']]
#     V = np.array(self.V)
#     D_ref = np.max(np.sum(initial_populations, axis=0) / V)
    
#     den = M_ref * D_ref * life_h_ref * (1+k) / 10
#     #den = 2.4477e-09

#     cos_i_av = 2/np.pi #average value of cosine of inclination in the range -pi/2 pi/2 calculated using integral average
#     Gamma_av = (1-cos_i_av)/2

#     rgb_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
#     def life(h):
#         return np.exp(14.18 * h**0.1831 - 42.94)

#     if hasattr(self, 'results'):
#         print("Producing two visuals of CSI.")
#         plt.figure()
#         plt.grid(True)
#         CSI_S_sum_array = np.zeros((len(self.results['times']), 0))
#         CSI_D_sum_array = np.zeros((len(self.results['times']), 0))
        
#         unique_species = set([data['species'] for data in self.results['population_data']])
        
#         for i2, species in enumerate(unique_species):
#             if i2 >= len(rgb_c):
#                 colorset = np.random.rand(3)
#             else:
#                 colorset = rgb_c[i2]
            
#             CSI_X_mat = np.zeros((len(self.results['times']), self.n_shells))
#             species_list = [sp for species_group in self.species.values() for sp in species_group]

#             if 'S' in species or 'D' in species:
#                 for i in range(self.n_shells):
#                     shell_data = [data for data in self.results['population_data'] if data['species'] == species and data['shell'] == (i + 1)]
#                     if shell_data:
#                         life_i = life((self.R0_km[i] + self.R0_km[i + 1]) / 2)
#                         num = life_i * (1 + k * Gamma_av)
#                         try:
#                             mass = next((item.mass for item in species_list if item.sym_name == species), 0)
#                         except TypeError as e:
#                             print(f"Error accessing species_properties for species '{species}': {e}")
#                             print(f"species_list: {species_list}")
#                             raise
#                         dum_X = mass * num
#                         D_X = np.array(shell_data[0]['populations']) / self.V[i]
#                         CSI_X_mat[:, i] = D_X * dum_X
                
#                 CSI_X_mat /= den
#                 CSI_X = np.sum(CSI_X_mat, axis=1)
#                 plt.plot(self.results['times'], CSI_X, label=f'CSI for {species.replace("p", ".")}', linewidth=2, color=colorset)
                
#                 if 'S' in species and 'D' not in species:
#                     CSI_S_sum_array = np.column_stack((CSI_S_sum_array, CSI_X))
#                 elif 'D' in species:
#                     CSI_D_sum_array = np.column_stack((CSI_D_sum_array, CSI_X))

#         if CSI_S_sum_array.shape[1] > 0:
#             CSI_S_sum = np.sum(CSI_S_sum_array, axis=1)
#         else:
#             CSI_S_sum = np.zeros(len(self.results['times']))

#         if CSI_D_sum_array.shape[1] > 0:
#             CSI_D_sum = np.sum(CSI_D_sum_array, axis=1)
#         else:
#             CSI_D_sum = np.zeros(len(self.results['times']))

#         plt.plot(self.results['times'], CSI_S_sum + CSI_D_sum, label='Total CSI', linewidth=2, color='black', linestyle='--')
#         plt.xlabel('Time (years)')
#         plt.ylabel('CSI')
#         plt.title('Cumulative Space Index (CSI) per Species')
#         plt.xlim([0, np.max(self.results['times'])])
#         plt.legend(loc='best', frameon=False)
#         plt.show()

#         plt.figure()
#         plt.grid(True)
#         plt.plot(self.results['times'], CSI_S_sum, label='Total CSI for Active Satellites', linewidth=2, color='#1f77b4')
#         plt.plot(self.results['times'], CSI_D_sum, label='Total CSI for Derelict Satellites', linewidth=2, color='#ff7f0e')
#         plt.plot(self.results['times'], CSI_S_sum + CSI_D_sum, label='Total CSI', linewidth=2, color='black', linestyle='--')
#         plt.xlabel('Time (years)')
#         plt.ylabel('Cumulative CSI')
#         plt.xlim([0, np.max(self.results['times'])])
#         plt.title('Cumulative Space Index (CSI) for Active and Derelict Species')
#         plt.legend(loc='best', frameon=False)
#         plt.show()
#     else:
#         raise ValueError("Simulation does not contain results. Please run the function run_model(x0) to produce simulation results required for CSI computation.")
        

def cum_umpy(obj, baseline, PMD_no_noise):
    # From pyssem code

    base_species_names = baseline.species_names
    n_species = baseline.species_length
    num_shells = baseline.n_shells
    
    species_mass = []
    orbital_lifetimes = []
    for species_index, (species_name, species_list) in enumerate(baseline.species_cells.items()):
        for species_properties in species_list:
            species_mass.append(species_properties.mass)
            orbital_lifetimes.append(species_properties.orbital_lifetimes)
    obj = obj.reshape((n_species, num_shells))

    X = 4

    # One aggregated vector eqs (n_shells x 1) summing across species
    # umpy_eqs = np.zeros((num_shells, 1))
    umpy_eqs = np.zeros((n_species,num_shells))
    for species_index in range(n_species):
        for shell_idx in range(num_shells):
            if not base_species_names[species_index].startswith('S'):
                mass_i = species_mass[species_index]
                pop_ij  = obj[species_index][shell_idx]            # population in shell i
                life_ij = orbital_lifetimes[species_index][shell_idx]
                umpy_factor = ((np.exp(X * (life_ij / baseline.simulation_duration)) - 1) / (np.exp(X) - 1))
                # umpy_eqs[shell_idx] += (mass_i * pop_ij * umpy_factor) / baseline.simulation_duration
                umpy_eqs[species_index,shell_idx] = (mass_i * pop_ij * umpy_factor) / baseline.simulation_duration
            else:
                # If active, just add zero
                # umpy_eqs[shell_idx]+= 0
                umpy_eqs[species_index,shell_idx] = 0

    return umpy_eqs.flatten()

# def cum_umpy_old(obj, baseline, PMD_no_noise):
#     # Maya's code

#     N_shell = baseline.n_shells  # Number of shells
#     R02 = baseline.R0_km  # Altitude bins (shell boundaries) in km
#     num_species = baseline.species_length  # Number of species
#     re = baseline.re

#     # Cumulative UMPY variables
#     num_species = len(baseline.species_names)
#     umpy_total = 0
#     umpy_per_shell = np.zeros(N_shell)
#     umpy_per_species = np.zeros(num_species)
#     umpy_per_species_per_shell = np.zeros((num_species, N_shell))

#     # assuming same PMD for all shells and all satellites, will have to modify otherwise
#     umpy_pmd = np.mean(PMD_no_noise)

#     species_mass = []
#     species_area = []
#     total_objects_per_shell = np.zeros(N_shell)
#     total_mass_per_shell = np.zeros(N_shell)
#     total_area_per_shell = np.zeros(N_shell)
#     total_objects_per_species = np.zeros(len(baseline.species_names))
#     total_mass_per_species = np.zeros(len(baseline.species_names))
#     total_area_per_species = np.zeros(len(baseline.species_names))
    
#     # Store species mass and area in lists by index
#     for species_index, (species_name, species_list) in enumerate(baseline.species_cells.items()):
#         for species_properties in species_list:
#             species_mass.append(species_properties.mass)
#             species_area.append(species_properties.A)

#     total_objects_per_shell = [0] * N_shell
#     total_mass_per_shell = [0] * N_shell
#     total_area_per_shell = [0] * N_shell

#     for shell_index in range(N_shell):
#         for species_index in range(num_species):
#             idx = shell_index + species_index * N_shell  # Corrected indexing
#             total_objects_per_shell[shell_index] += obj[idx]
#             total_mass_per_shell[shell_index] += obj[idx] * species_mass[species_index]
#             total_area_per_shell[shell_index] += obj[idx] * species_area[species_index]

#     lifetime = [0] * N_shell

#     # Calculate and sum UMPY for each shell for total UMPY
#     for shell_index in range(N_shell):
#         h = (R02[shell_index] + R02[shell_index + 1]) / 2
#         # lifetime[shell_index] = 10 ** (14.18 * (h ** 0.1831) - 42.94) # maya
#         lifetime[shell_index] = np.exp(14.18 * (h ** 0.1831) - 42.94)
#         M = total_mass_per_shell[shell_index]
#         umpy_per_shell[shell_index] = (M * (1-umpy_pmd))/lifetime[shell_index]
#         umpy_total += umpy_per_shell[shell_index]  # Accumulate CSI for this shell

#     # Initialize storage arrays
#     total_objects_per_species = [0] * num_species
#     total_mass_per_species = [0] * num_species
#     total_area_per_species = [0] * num_species

#     # Calculate CSI for each species
#     for species_index in range(num_species):
#         for shell_index in range(N_shell):
#             idx = shell_index + species_index * N_shell  # Correct species-shell indexing

#             # Species-specific mass, area, and object count in this shell
#             num_objects = obj[idx]
#             mass = num_objects * species_mass[species_index]
#             area = num_objects * species_area[species_index]

#             umpy_per_species_per_shell[species_index, shell_index] = ((mass) * (1 - umpy_pmd)) / (lifetime[shell_index])

#             # Accumulate total per species
#             total_objects_per_species[species_index] += num_objects
#             total_mass_per_species[species_index] += mass
#             total_area_per_species[species_index] += area

#             umpy_per_species[species_index] += umpy_per_species_per_shell[species_index, shell_index]  # Accumulate UMPY for this species

#     return umpy_per_species_per_shell.flatten() # umpy_total, umpy_per_species

# def cum_OAR(obj, baseline):

#     N_shell = baseline.n_shells  # Number of shells

#     oar = 0
#     total_objects_per_shell = np.zeros(N_shell)

#     # Calculate the total objects and total mass in each shell
#     for species_index, (species_name, species_list) in enumerate(baseline.species_cells.items()):
#         start_idx = species_index * N_shell
#         end_idx = start_idx + N_shell

#         # Extract the population for the species in all shells
#         species_counts = obj[start_idx:end_idx]
#         total_objects_per_shell += species_counts

#     # Calculate and sum OAR for each shell for total OAR
#     for shell_index in range(N_shell):
#         oar = oar + 1

#     return oar

#==========================================================================
# NMPC - cost function options
#==========================================================================

def myCostFunction(x, u, r):
    """
    Custom cost function for nonlinear MPC.
    """
    J = np.sum((x - r)**2)

    return J 

def myCostFunction_CSI(m, x, baseline):
    """
    Calculates the CSI risk index symbolically using Gekko functions.

    This function separates one-time pre-computations from the symbolic
    part that depends on the Gekko state variable 'x'.
    """
    # ======================================================================
    #  Part 1: One-Time Pre-computation of Constants (using NumPy)
    #  This is efficient as it's done only once when the model is built.
    # ======================================================================

    N_shell = baseline.n_shells
    R02 = baseline.R0_km
    num_species = len(baseline.species_names)
    V_per_shell = np.array(baseline.V)

    k = 0.6
    cos_i_av = 2 / np.pi
    Gamma_av = (1 - cos_i_av) / 2
    inclination_factor = (1 + k * Gamma_av)

    M_ref = 10000.0
    life_h_ref = 1468.0

    initial_populations_flat = np.array(baseline.x0)
    initial_populations_matrix = initial_populations_flat.reshape(num_species, N_shell)
    total_objects_per_shell = np.sum(initial_populations_matrix, axis=0)
    initial_number_density_per_shell = total_objects_per_shell / V_per_shell
    D_ref = np.max(initial_number_density_per_shell)

    den = M_ref * D_ref * life_h_ref * (1 + k)

    species_mass = np.array([prop.mass for sl in baseline.species_cells.values() for prop in sl])
    species_mass_col = species_mass[:, np.newaxis]

    h = (R02[:-1] + R02[1:]) / 2
    lifetime_per_shell = np.exp(14.18 * (h ** 0.1831) - 42.94)

    num_term = lifetime_per_shell * inclination_factor
    dum_X_matrix = species_mass_col * num_term  # This is a constant matrix

    # ======================================================================
    #  Part 2: Symbolic Calculation (using Gekko)
    #  This part uses the pre-computed constants and the Gekko variable 'x'.
    # ======================================================================
    
    csi_list = []

    for i in range(num_species):
        for j in range(N_shell):
            # Get the symbolic population variable for this state
            pop_ij = x[i * N_shell + j]

            # Calculate symbolic Species Number Density (D_X_ij)
            # This is a Gekko expression because pop_ij is a Gekko variable.
            D_X_ij = pop_ij / V_per_shell[j]

            # Get the pre-computed constant value for this species/shell
            dum_X_ij = dum_X_matrix[i, j]

            # Calculate the final symbolic CSI expression for this element
            csi_calculation = (D_X_ij * dum_X_ij) / den

            # Create the Gekko Intermediate variable and add it to our list
            csi_var = m.Intermediate(csi_calculation)
            csi_list.append(csi_var)

    return csi_list
    
def myCostFunction_UMPY(m, x, baseline):
    """
    Calculates the risk index symbolically using Gekko functions.
    This version correctly creates Intermediate variables.
    """
    n_species = baseline.species_length
    num_shells = baseline.n_shells

    # --- Step 1: Initialize an empty Python list to store the variables ---
    risk_idx_list = []

    # Extract parameters (these are constants, so this is fine)
    species_mass = []
    orbital_lifetimes = []
    for species_list in baseline.species_cells.values():
        for props in species_list:
            species_mass.append(props.mass)
            orbital_lifetimes.append(props.orbital_lifetimes)
    
    X = 4.0

    # --- Step 2: Loop to build the list of Intermediate variables ---
    for i in range(n_species):
        for j in range(num_shells):
            if not baseline.species_names[i].startswith('S'):
                mass_i = species_mass[i]
                pop_ij = x[i * num_shells + j] # Assuming x is a flat Gekko array
                life_ij = orbital_lifetimes[i][j]
                
                umpy_factor = (m.exp(X * (life_ij / baseline.simulation_duration)) - 1) / (m.exp(X) - 1)
                
                # --- Step 3: Define the expression and create the Intermediate ---
                #    The entire calculation is the 'equation' argument.
                risk_calculation = (mass_i * pop_ij * umpy_factor) / baseline.simulation_duration
                
                # Create the intermediate and add it to our list
                risk_var = m.Intermediate(risk_calculation)
                risk_idx_list.append(risk_var)

            else:
                # For 'S' species, the risk is zero.
                # Create an Intermediate with a constant value of 0.
                risk_var = m.Intermediate(0)
                risk_idx_list.append(risk_var)

    # --- Step 4: Return the completed list of Gekko variables ---
    return risk_idx_list

#==========================================================================
# Launch rate - plot
#==========================================================================

def launch_rate_plot(baseline, active_species_indices, x0_lam, x0_lam_no_noise, sel_LineWidth, sel_LineWidthAxis, sel_FontSize):
    N_shell = baseline.n_shells
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell

        plot_x0_lam = x0_lam[start_idx:end_idx,:]
        plot_x0_lam_no_noise = x0_lam_no_noise[start_idx:end_idx,:]

        plt.figure(facecolor='white')
        plt.xlabel('Time (years)')
        plt.ylabel('$\lambda$')
        plt.grid(True)
        plt.plot(sum(plot_x0_lam), linewidth=sel_LineWidth, label='w/ noise')
        plt.plot(sum(plot_x0_lam_no_noise), '--', linewidth=sel_LineWidth, label='w/o noise')
        plt.legend(loc='best')
        plt.title(baseline.species_names[active_species_indices[species_index]])
        plt.show()

        plt.figure(facecolor='white', figsize=(10,6), layout="constrained")
        n_sub_plot = int(np.ceil(np.sqrt(N_shell)))
        for i in range(N_shell):
            ax = plt.subplot(n_sub_plot, n_sub_plot, i+1)
            ax.grid(True)
            ax.plot(plot_x0_lam[i], linewidth=sel_LineWidth, linestyle='-', label='w/ noise')
            ax.plot(plot_x0_lam_no_noise[i], linewidth=sel_LineWidth, linestyle='--', label='w/o noise')
            if (i+1) % int(np.ceil(np.sqrt(N_shell))) == 1 or i == 0:
                ax.set_ylabel('Count/year', fontsize=sel_FontSize)
            if N_shell - i <= np.ceil(np.sqrt(N_shell)):
                ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
            ax.set_title(baseline.species_names[active_species_indices[species_index]]+f", $\lambda$$_{{{i+1}}}$", fontsize=sel_FontSize)
            ax.tick_params(width=sel_LineWidthAxis)
            plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
            plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.show()
    return 1

#==========================================================================
# PMD - plot
#==========================================================================

def pmd_plot(baseline, active_species_indices, PMD, PMD_no_noise, sel_LineWidth, sel_LineWidthAxis, sel_FontSize):
    N_shell = baseline.n_shells
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell

        plot_PMD = PMD[start_idx:end_idx,:]
        plot_PMD_no_noise = PMD_no_noise[start_idx:end_idx,:]

        plt.figure(facecolor='white', figsize=(10,6), layout="constrained")
        n_sub_plot = int(np.ceil(np.sqrt(N_shell)))
        for i in range(N_shell):
            ax = plt.subplot(n_sub_plot, n_sub_plot, i+1)
            ax.grid(True)
            ax.plot(plot_PMD[i], linewidth=sel_LineWidth, linestyle='-', label='w/ noise')
            ax.plot(plot_PMD_no_noise[i], linewidth=sel_LineWidth, linestyle='--', label='w/o noise')
            if (i+1) % int(np.ceil(np.sqrt(N_shell))) == 1 or i == 0:
                ax.set_ylabel('Percentage', fontsize=sel_FontSize)
            if N_shell - i <= np.ceil(np.sqrt(N_shell)):
                ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
            ax.set_title(baseline.species_names[active_species_indices[species_index]]+f", PMD$_{{{i+1}}}$", fontsize=sel_FontSize)
            ax.tick_params(width=sel_LineWidthAxis)
            plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
            plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.show()
    return 1

#==========================================================================
# Orbital lifetime [years] - plots
#==========================================================================

def deltat_plot(baseline, active_species_indices, deltat, deltat_no_noise, sel_LineWidth, sel_LineWidthAxis, sel_FontSize):
    N_shell = baseline.n_shells
    for species_index in range(len(active_species_indices)):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell

        plot_deltat = deltat[start_idx:end_idx,:]
        plot_deltat_no_noise = deltat_no_noise[start_idx:end_idx,:]

        plt.figure(facecolor='white', figsize=(10,6), layout="constrained")
        n_sub_plot = int(np.ceil(np.sqrt(N_shell)))
        for i in range(N_shell):
            ax = plt.subplot(n_sub_plot, n_sub_plot, i+1)
            ax.grid(True)
            ax.plot(plot_deltat[i], linewidth=sel_LineWidth, linestyle='-', label='w/ noise')
            ax.plot(plot_deltat_no_noise[i], linewidth=sel_LineWidth, linestyle='--', label='w/o noise')
            if (i+1) % int(np.ceil(np.sqrt(N_shell))) == 1 or i == 0:
                ax.set_ylabel('Years', fontsize=sel_FontSize)
            if N_shell - i <= np.ceil(np.sqrt(N_shell)):
                ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
            ax.set_title(baseline.species_names[active_species_indices[species_index]]+f", $\delta$$_{{{i+1}}}$", fontsize=sel_FontSize)
            ax.tick_params(width=sel_LineWidthAxis)
            plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
            plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.show()
    return 1

#==========================================================================
# Cumulative plots
#==========================================================================

def cumulative_plot(baseline, output, active_species_indices, sel_pmd_control, sel_risk_index, sel_LineWidth, sel_LineWidthAxis, sel_FontSize):
    n_species = baseline.species_length
    num_shells = baseline.n_shells
    species_names = baseline.species_names
    orbital_shell_labels = baseline.R0_km[:num_shells]
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

    base_species_names = baseline.species_names
    unique_base_species = list(set(base_species_names))
    color_map = plt.cm.get_cmap('tab20', len(unique_base_species))

    # Total objects over time for each species and total
    plt.figure(facecolor='white',figsize=(12, 8))
    plt.grid(True)
    total_objects_all_species = np.zeros_like(output.t)
    handles = []
    labels = []
    for species_index in range(n_species):
        color = color_map(unique_base_species.index(base_species_names[species_index]))
        # marker = markers[species_index % len(markers)]
        start_idx = species_index * num_shells
        end_idx = start_idx + num_shells
        total_objects_per_species = np.sum(output.y[start_idx:end_idx, :], axis=0)
        plt.plot(output.t, total_objects_per_species, linewidth=sel_LineWidth, label=species_names[species_index]+"$^{C}$", color=color, linestyle='-') #, marker=marker, markersize=sel_MarkerWidth)
        plt.plot(output.t, np.sum(output.y_ref[start_idx:end_idx, :], axis=0), linewidth=sel_LineWidth, label=species_names[species_index]+"$^{REF}$", color=color, linestyle='--')
        plt.plot(output.t, np.sum(output.y_nc[start_idx:end_idx, :], axis=0), linewidth=sel_LineWidth, label=species_names[species_index]+"$^{NC}$", color=color, linestyle=':')
        total_objects_all_species += total_objects_per_species
        if species_names[species_index] not in labels:
            handles.append(Line2D([], [], color=color, linewidth=sel_LineWidth, label=species_names[species_index]))
            labels.append(species_names[species_index])
    plt.plot(output.t, total_objects_all_species, label='Total', color='k', linewidth=sel_LineWidth, linestyle='-')
    handles.append(Line2D([], [], color='k', linewidth=sel_LineWidth, label='Total'))
    labels.append('Total')
    handles.append(Line2D([], [], color='k', linestyle='-', linewidth=sel_LineWidth, label='Controlled'))
    handles.append(Line2D([], [], color='k', linestyle='--', linewidth=sel_LineWidth, label='Reference'))
    handles.append(Line2D([], [], color='k', linestyle=':', linewidth=sel_LineWidth, label='No Control'))
    labels.append("Controlled")
    labels.append("Reference")
    labels.append("No Control")
    plt.xlabel('Time (years)', fontsize=sel_FontSize)
    plt.ylabel('Count', fontsize=sel_FontSize)
    plt.title('SOs vs Time')
    plt.xlim(0, max(output.t))
    plt.legend(handles=handles, labels=labels, loc="best")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=8, fancybox=True, shadow=True)
    # plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fancybox=True, shadow=True)
    # plt.yscale('log') # for log plot
    plt.tight_layout() 
    plt.xticks(fontsize=sel_FontSize)
    plt.yticks(fontsize=sel_FontSize)
    plt.gca().tick_params(width=sel_LineWidthAxis)
    plt.show()

    if sel_pmd_control == 1:
        # Figure: PMD with and without noise
        for species_index in range(len(active_species_indices)):
            start_idx = species_index * num_shells
            end_idx = start_idx + num_shells

            plot_PMD = output.PMD[start_idx:end_idx,:]
            plot_PMD_no_noise = output.PMD_no_noise[start_idx:end_idx,:]
            plot_PMD_orig = output.PMD_orig[start_idx:end_idx,:]
            plot_PMD_no_noise_orig = output.PMD_no_noise_orig[start_idx:end_idx,:]

            plt.figure(facecolor='white', figsize=(10,6), layout="constrained")
            n_sub_plot = int(np.ceil(np.sqrt(num_shells)))
            for i in range(num_shells):
                ax = plt.subplot(n_sub_plot, n_sub_plot, i+1)
                ax.grid(True)
                ax.plot(plot_PMD[i], linewidth=sel_LineWidth, linestyle='-', label='w/ noise')
                ax.plot(plot_PMD_no_noise[i], linewidth=sel_LineWidth, linestyle='--', label='w/o noise')
                ax.plot(plot_PMD_orig[i], linewidth=sel_LineWidth, linestyle='-', label='w/ noise (orig)')
                ax.plot(plot_PMD_no_noise_orig[i], linewidth=sel_LineWidth, linestyle='--', label='w/o noise (orig)')
                if (i+1) % int(np.ceil(np.sqrt(num_shells))) == 1 or i == 0:
                    ax.set_ylabel('Percentage', fontsize=sel_FontSize)
                if num_shells - i <= np.ceil(np.sqrt(num_shells)):
                    ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
                ax.set_title(baseline.species_names[active_species_indices[species_index]]+f", PMD$_{{{i+1}}}$", fontsize=sel_FontSize)
                ax.tick_params(width=sel_LineWidthAxis)
                plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
                plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
            plt.legend(loc='best', bbox_to_anchor=(1, 1))
            plt.show()

    if sel_risk_index != 0:
        # Figure: Risk index all shells vs time
        plt.figure(facecolor='white',figsize=(12, 8))
        plt.grid(True)
        risk_idx_total = np.zeros_like(output.t)
        risk_idx_total_nc = np.zeros_like(output.t)
        handles = []
        labels = []
        for species_index in range(n_species):
            color = color_map(unique_base_species.index(base_species_names[species_index]))
            # marker = markers[species_index % len(markers)]
            start_idx = species_index * num_shells
            end_idx = start_idx + num_shells
            risk_idx_per_species = np.sum(output.risk_idx_per_species_per_shell[start_idx:end_idx, :], axis=0)
            risk_idx_per_species_nc = np.sum(output.risk_idx_per_species_per_shell_nc[start_idx:end_idx, :], axis=0)
            plt.plot(output.t, risk_idx_per_species, linewidth=sel_LineWidth, label=species_names[species_index]+"$^{C}$", color=color, linestyle='-') #, marker=marker, markersize=sel_MarkerWidth)
            plt.plot(output.t, risk_idx_per_species_nc, linewidth=sel_LineWidth, label=species_names[species_index]+"$^{NC}$", color=color, linestyle=':')
            risk_idx_total += risk_idx_per_species
            risk_idx_total_nc += risk_idx_per_species_nc
            if species_names[species_index] not in labels:
                handles.append(Line2D([], [], color=color, linewidth=sel_LineWidth, label=species_names[species_index]))
                labels.append(species_names[species_index])
        plt.plot(output.t, risk_idx_total, label='Total$^{C}$', color='k', linewidth=sel_LineWidth, linestyle='-')
        plt.plot(output.t, risk_idx_total_nc, label='Total$^{NC}$', color='k', linewidth=sel_LineWidth, linestyle=':')
        handles.append(Line2D([], [], color='k', linewidth=sel_LineWidth, label='Total'))
        labels.append('Total')
        handles.append(Line2D([], [], color='k', linestyle='-', linewidth=sel_LineWidth, label='Controlled'))
        handles.append(Line2D([], [], color='k', linestyle=':', linewidth=sel_LineWidth, label='No Control'))
        labels.append("Controlled")
        labels.append("No Control")
        plt.xlabel('Time (years)', fontsize=sel_FontSize)
        if sel_risk_index == 1:
            plt.ylabel('CSI (-)', fontsize=sel_FontSize)
            plt.title('Cumulative CSI vs Time')
        elif sel_risk_index == 2:
            plt.ylabel('UMPY (-)', fontsize=sel_FontSize)
            plt.title('Cumulative UMPY vs Time')
        plt.xlim(0, max(output.t))
        plt.legend(handles=handles, labels=labels, loc="best")
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=8, fancybox=True, shadow=True)
        # plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fancybox=True, shadow=True)
        # plt.yscale('log') # for log plot
        plt.tight_layout() 
        plt.xticks(fontsize=sel_FontSize)
        plt.yticks(fontsize=sel_FontSize)
        plt.gca().tick_params(width=sel_LineWidthAxis)
        plt.show()

        # Figure: Surface plot of risk index over time and orbital shells
        fig = plt.figure(facecolor='white', figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_xlabel('Time', fontsize=sel_FontSize)
        ax.set_ylabel('Orbital Shell', fontsize=sel_FontSize)
        ax.set_zlabel('Indicator Value', fontsize=sel_FontSize)
        if sel_risk_index == 1:
            ax.set_title('Surface Plot of CSI for All Species', fontsize=sel_FontSize)
        elif sel_risk_index == 2:
            ax.set_title('Surface Plot of UMPY for All Species', fontsize=sel_FontSize)
        risk_idx_per_species = output.risk_idx_per_species_per_shell.T
        risk_idx_per_species = risk_idx_per_species.reshape(len(output.t), n_species, num_shells).sum(axis=1)
        risk_idx_per_species = risk_idx_per_species.T  # Shape: [num_shells, num_times]
        X, Y = np.meshgrid(output.t, np.arange(num_shells))
        surf = ax.plot_surface(X, Y, risk_idx_per_species, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Indicator Value')
        plt.xticks(fontsize=sel_FontSize)
        plt.yticks(fontsize=sel_FontSize)
        ax.tick_params(axis='z', labelsize=sel_FontSize)
        # plt.tight_layout()
        plt.show()

    # ADR Total vs Time
    for species_index in range(n_species):
        if not base_species_names[species_index].startswith('S'):
            plt.figure(facecolor='white')
            plt.grid(True)
            color = color_map(unique_base_species.index(base_species_names[species_index]))
            start_idx = species_index * num_shells
            end_idx = start_idx + num_shells
            plt.plot(output.t, -output.satur[species_index](output.t), linewidth=sel_LineWidth, label='Max', color='k', linestyle='--') 
            plt.plot(output.t, -np.sum(output.y_u[start_idx:end_idx, :], axis=0), linewidth=sel_LineWidth, label=species_names[species_index], color=color, linestyle='-')
            plt.title('ADR Total vs Time')
            plt.xlabel('Time (years)', fontsize=sel_FontSize)
            plt.ylabel('Count', fontsize=sel_FontSize)
            plt.xticks(fontsize=sel_FontSize)
            plt.yticks(fontsize=sel_FontSize)
            plt.tight_layout()
            plt.xlim(0, max(output.t))
            plt.gca().tick_params(width=sel_LineWidthAxis)
            plt.legend(loc='best')
            plt.show()


    # Each species over time for each shell
    for species_index in range(n_species):
        color = color_map(unique_base_species.index(base_species_names[species_index]))
        start_idx = species_index * num_shells
        end_idx = start_idx + num_shells

        plot_y = output.y[start_idx:end_idx,:]
        plot_y_nc = output.y_nc[start_idx:end_idx,:]
        plot_y_ref = output.y_ref[start_idx:end_idx,:]

        plt.figure(facecolor='white', figsize=(10,6), layout="constrained")
        n_sub_plot = int(np.ceil(np.sqrt(num_shells)))
        for i in range(num_shells):
            ax = plt.subplot(n_sub_plot, n_sub_plot, i+1)
            ax.grid(True)
            ax.plot(output.t, plot_y[i,:], linewidth=sel_LineWidth, label=species_names[species_index]+"$^C$", color=color, linestyle='-')
            ax.plot(output.t, plot_y_ref[i,:], linewidth=sel_LineWidth, label=species_names[species_index]+"$^{REF}$", color=color, linestyle='--')
            ax.plot(output.t, plot_y_nc[i,:], linewidth=sel_LineWidth, label=species_names[species_index]+"$^{NC}$", color=color, linestyle=':')
            if (i+1) % int(np.ceil(np.sqrt(num_shells))) == 1 or i == 0:
                ax.set_ylabel('Count', fontsize=sel_FontSize)
            if num_shells - i <= np.ceil(np.sqrt(num_shells)):
                ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
            ax.set_title(species_names[species_index]+f"$_{{{i+1}}}$", fontsize=sel_FontSize)
            ax.tick_params(width=sel_LineWidthAxis)
            plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
            plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
            ymin = -5
            ymax = ax.get_ylim()[1]  # Get the current upper limit
            ax.set_ylim(ymin, ymax)  # Set the new y-axis limits
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.show()

    # Each species over time for all shells
    cols = 5  # Define the number of columns you want
    rows = (n_species + cols - 1) // cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D, even for a single row
    for species_index in range(n_species):
        ax = axes.flatten()[species_index]
        species_data = output.y[species_index * num_shells:(species_index + 1) * num_shells]
        for shell_index in range(num_shells):
            ax.plot(output.t, species_data[shell_index], label=f'Shell {shell_index + 1}')
        ax.set_title(f'{species_names[species_index]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    # Hide any unused axes
    for i in range(n_species, rows * cols):
        fig.delaxes(axes.flatten()[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # --- HEATMAPS FOR EACH SPECIES OVER TIME AND ORBITAL SHELLS ---
    cols = 3  # Define the number of columns for the subplot grid
    rows = (n_species + cols - 1) // cols  # Calculate rows needed

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 4 * rows), facecolor='white', constrained_layout=True)
    axs = np.atleast_2d(axs)  # Ensure axs is always a 2D array for consistent indexing

    for i, species_name in enumerate(species_names):
        row_idx = i // cols
        col_idx = i % cols
        ax = axs[row_idx, col_idx]

        # Extract the data for the current species across all shells
        start_idx = i * num_shells
        end_idx = start_idx + num_shells
        data_per_species = output.y[start_idx:end_idx, :]

        # Create the heatmap using imshow
        # extent=[left, right, bottom, top] defines the data coordinates of the image
        im = ax.imshow(data_per_species, 
                       aspect='auto', 
                       origin='lower',
                       extent=[output.t[0], output.t[-1], 0, num_shells],
                       interpolation='nearest',
                       cmap='viridis')
        
        # Add a colorbar for the heatmap
        fig.colorbar(im, ax=ax, label='Count')

        # Set labels and title
        ax.set_xlabel('Time (years)', fontsize=sel_FontSize)
        ax.set_ylabel('Altitude (km)', fontsize=sel_FontSize)
        ax.set_title(species_name, fontsize=sel_FontSize)

        # Configure ticks for better readability
        ax.set_xticks(np.linspace(output.t[0], output.t[-1], num=5, dtype=int))
        
        # Set y-axis ticks to show shell indices and label them with actual altitudes
        # This makes the plot much more informative
        tick_step = max(1, num_shells // 5) # Show about 5-6 ticks
        ax.set_yticks(np.arange(0.5, num_shells, tick_step)) # Center ticks in the cells
        ax.set_yticklabels([f'{alt:.0f}' for alt in orbital_shell_labels[::tick_step]])
        
        ax.tick_params(axis='both', which='major', labelsize=sel_FontSize)

    # Hide any unused subplots if the number of species doesn't perfectly fill the grid
    for i in range(n_species, rows * cols):
        fig.delaxes(axs.flatten()[i])
    plt.show()

    return 1

#==========================================================================
# Functions used in jupyter notebook for controller implementation
#==========================================================================

def MC2SSEM_population(popMC, VAR):
    """
    Creates the initial population used by SSEM, divided by species and altitude bin,
    starting from the MC population. 
    """
    # popMC = sats
    # VAR = baseline

    # Filter species based on VAR.species_types 
    active_species_indices = []
    for index, species in enumerate(VAR.species_types):
        if species:
            active_species_indices.append(index + 1)

    # Initialize lists to store classified objects
    popMC_S = []; popMC_D = []; popMC_N = []; popMC_Su = []; popMC_B = []; popMC_U = []

    # Object classification categories
    potential_payload_classes = ['Payload', 'Payload Mission Related Object', 'Other Mission Related Object', 'Rocket Mission Related Object']
    debris_classes = ['Other Debris', 'Payload Debris', 'Payload Fragmentation Debris', 'Rocket Debris', 'Rocket Fragmentation Debris']
    untracked_debris_classes = ['Untracked Debris']
    rocket_body_classes = ['Rocket Body']

    # Classify objects based on their properties
    for sat in popMC:
        alt = sat[0]['a'] * VAR.re - VAR.re  # Calculate altitude
    
        # Check if object is within altitude boundaries (% Outside boundaries TO DO: understand why this is not an || ?)
        if alt < VAR.min_altitude and alt > VAR.max_altitude:
            continue

        # Classify based on objectclass and control status 
        if any(obj_class in sat[0]['objectclass'] for obj_class in potential_payload_classes):
            if 1 in active_species_indices and 4 in active_species_indices:  # S and Su exist
                # MOCAT-4S
                if sat[0]['controlled'] == 1 and sat[0]['constel'] == 0:  # Active payload
                    if 4 in active_species_indices:
                        popMC_Su.append(sat)
                elif sat[0]['controlled'] == 1 and sat[0]['constel'] == 1: # Constellation
                    if 1 in active_species_indices:
                        popMC_S.append(sat)
                elif sat[0]['controlled'] == 0:  # Inactive payload (derelict)
                    if 2 in active_species_indices:
                        popMC_D.append(sat)
            else:
                # MOCAT-3
                if sat[0]['controlled'] == 1:
                    if 4 in active_species_indices:
                        popMC_Su.append(sat)
                else:
                    if 2 in active_species_indices:
                        popMC_D.append(sat)
        elif any(obj_class in sat[0]['objectclass'] for obj_class in debris_classes):
            if 3 in active_species_indices:
                popMC_N.append(sat)
        elif any(obj_class in sat[0]['objectclass'] for obj_class in rocket_body_classes): 
            if 5 in active_species_indices:
                popMC_B.append(sat)
        elif any(obj_class in sat[0]['objectclass'] for obj_class in untracked_debris_classes): 
            if 6 in active_species_indices:
                popMC_U.append(sat)

    # Create count lists for each species
    pop_list = [popMC_S, popMC_D, popMC_N, popMC_Su, popMC_B, popMC_U]
    count_list = np.zeros((len(pop_list), 0)).tolist()
    speciesName_list = ['Su', 'D', 'N', 'S', 'B', 'U']

    i1 = 0
    for speciesName, pop in zip(speciesName_list, pop_list):
        for species_properties in VAR.species_cells.get(speciesName, []):
            count = np.zeros((VAR.n_shells, 1))
            count_list[i1].append(count)
        i1 += 1

    # Bin objects by altitude and mass
    for species_idx, count in enumerate(count_list):
        if not count:
            continue
        for shell_idx in range(VAR.n_shells - 1):
            pop = pop_list[species_idx]
            speciesName = speciesName_list[species_idx]

            for sat in pop:
                alt = sat[0]['a'] * VAR.re - VAR.re

                if VAR.R0_km[shell_idx] <= alt < VAR.R0_km[shell_idx + 1]:
                    for species_properties in VAR.species_cells.get(speciesName, []):
                        mass_idx = 0
                        if (species_properties.mass_lb < sat[0]['mass'] <= species_properties.mass_ub):
                            count[mass_idx][shell_idx] += 1
                            break  # Object assigned to a mass bin, move to next object

    # Assemble popSSEM based on VAR.species order 
    popSSEM = np.zeros((VAR.n_shells, 0))

    for species_name, species_list in VAR.species_cells.items():
        species_idx = speciesName_list.index(species_name)
        if species_idx < len(count_list):
            popSSEM = np.hstack((popSSEM, count_list[species_idx][0]))

    return popSSEM.T.flatten().astype(int)

def extract_vars(vars_dict):
    """
    Extracts and organizes variables from a dictionary based on prefixes.

    Args:
        vars_dict: A dictionary where keys are variable names (strings)
                   and values are lists of symbolic variables.
        all_symbolic_vars:  A list (or other iterable) of all symbolic variables
                           that are relevant.  This is important for creating
                           'var_s'.

    Returns:
        A tuple containing:
            - Pm_s: List of Pm variables.
            - deltat_s: List of deltat variables.
            - lam_s: List of lambda variables.
            - u_var_s: Combined list of u_l, u_r_N, and u_r_N_192kg variables.
            - var_s:  A list of all the symbolic variables extracted in this
                      function (filtered from all_symbolic_vars).
    
    Manually defined variables for testing:
        Pm_s = vars_dict['Pm_S_148kg'] + vars_dict['Pm_S_750kg'] + vars_dict['Pm_S_1250kg'] + vars_dict['Pm_Sns'] + vars_dict['Pm_Su_260kg'] + vars_dict['Pm_Su_473kg'] 
        Pm_s = vars_dict['Pm_Su']
        deltat_s = vars_dict['deltat_Su']
        lam_s = vars_dict['lambda_Su']
        u_var_s = vars_dict['u_l_Su']+vars_dict['u_r_N']+vars_dict['u_r_N_192kg']
    """
    
    Pm_s = []
    deltat_s = []
    lam_s = []
    u_var_s = []
    extracted_vars = set()  # Use a set to avoid duplicates

    for key, value_list in vars_dict.items():
        if not isinstance(key, str):  # Important: Ensure key is a string
            continue  # Skip non-string keys

        if key.startswith("Pm_"):
            Pm_s.extend(value_list)
            extracted_vars.update(value_list)
        elif key.startswith("deltat_"):
            deltat_s.extend(value_list)
            extracted_vars.update(value_list)
        elif key.startswith("lambda_"):
            lam_s.extend(value_list)
            extracted_vars.update(value_list)
        elif key.startswith("u_l_") or key.startswith("u_r_"):
            u_var_s.extend(value_list)
            extracted_vars.update(value_list)

    return Pm_s, deltat_s, lam_s, u_var_s

def assign_saturations(species_names, saturation_values):
    """
    Assigns saturation values to species based on their names.

    Species starting with "S" (case-insensitive) are assigned a saturation of 0.
    Other species are assigned saturation values from the provided list,
    cycling through the list if necessary.

    Args:
        species_names: A list of strings representing the names of the species.
        saturation_values: A list of saturation values to assign to
                           non-"S" species.  Can be an empty list.

    Returns:
        A NumPy array of saturation values, with one value for each species
        in the same order as `species_names`.
    """

    saturations = []
    non_s_index = 0  # Index for cycling through saturation_values

    for species_name in species_names:
        if species_name.lower().startswith("s"):
            saturations.append(0)
        else:
            if saturation_values:  # Check if saturation_values is not empty
                saturations.append(saturation_values[non_s_index % len(saturation_values)])
                non_s_index += 1
            else:
                saturations.append(0)  # Default to 0 if no saturation values are given

    return saturations

def assign_stats_to_species(species_names, species_stat_1, species_stat_2):
    """
    Assigns pair of values to species based on their names.

    For example, mean and standard deviation values:
        Args:
            species_names: A list of strings representing the names of the species.
            species_stat_1: A NumPy array of mean values.
            species_stat_2: A NumPy array of standard deviation values.

        Returns:
            A tuple of two lists:
            - assigned_stat_1: list of means assigned to each specie
            - assigned_stat_2: list of stds assigned to each specie
            Returns None, None if there are input errors.

    """
    num_species = len(species_names)
    assigned_stat_1 = [0] * num_species  # Initialize with zeros
    assigned_stat_2 = [0] * num_species

    if len(species_stat_1) != 3 or len(species_stat_2) != 3:
      print("Error. species_stat_1 and species_stat_2 must have length equal to 3.")
      return None, None

    for i, name in enumerate(species_names):
        if name.startswith('S'):
            assigned_stat_1[i] = species_stat_1[0]
            assigned_stat_2[i] = species_stat_2[0]
        elif name == 'N':
            assigned_stat_1[i] = species_stat_1[1]
            assigned_stat_2[i] = species_stat_2[1]
        elif name.startswith('N'):
            # Extract mass using regular expressions
            match = re.search(r"N_(\d+(\.\d+)?)kg", name)
            if match:
                mass = float(match.group(1))  # Convert to float
                if mass < 10:
                    assigned_stat_1[i] = species_stat_1[1]
                    assigned_stat_2[i] = species_stat_2[1]
                else:
                    assigned_stat_1[i] = species_stat_1[2]
                    assigned_stat_2[i] = species_stat_2[2]
            else: #Handles cases like N_0.00141372kg (without integer part).
                match = re.search(r"N_(0\.\d+)kg", name)
                if match:
                    mass = float(match.group(1))
                    if mass < 10:
                        assigned_stat_1[i] = species_stat_1[1]
                        assigned_stat_2[i] = species_stat_2[1]
                    else:
                        assigned_stat_1[i] = species_stat_1[2]
                        assigned_stat_2[i] = species_stat_2[2]
        elif name.startswith('B'):
            assigned_stat_1[i] = species_stat_1[2]
            assigned_stat_2[i] = species_stat_2[2]

    return assigned_stat_1, assigned_stat_2

def randn2(*args,**kwargs):
    '''
    Calls rand and applies inverse transform sampling to the output.
    '''
    uniform = np.random.rand(*args, **kwargs)
    return np.sqrt(2) * erfinv(2 * uniform - 1)

# It was ordering the vars based on the kg in descending order.
# def sort_key(symbol):
#     match = re.match(r"([a-zA-Z_]+)(.*)kg(\d+)", symbol.name)  # Updated regex
#     if match:
#         name, middle_part, number = match.groups()
#         return (name, middle_part, int(number))  # Sort by name, middle, then number
#     match = re.match(r"([a-zA-Z_]+)(\d+)", symbol.name) # Original regex for other cases
#     if match:
#         name, number = match.groups()
#         return (name, int(number))
#     return (symbol.name,)  # For symbols without numbers
# ordered_difference_vars = sorted(list(difference_vars), key=sort_key) 

def sort_key(var_name, baseline_species_names):
    """
    Custom sorting key function to prioritize variable order based on species names and shell number.

    Args:
        var_name (str): The variable name (e.g., 'Pm_S_1250kg1').
        baseline_species_names (list): A list of baseline species names in desired order (e.g., ['S_148kg', 'S_1250kg']).

    Returns:
        tuple: A tuple for sorting. The first element is the index of the species name in
               baseline_species_names.  If the species name is not found, it defaults to
               a high value (len(baseline_species_names)) so that unknown species are
               sorted at the end. The second element is the integer value of the shell number
               extracted from the var_name.  If no shell number is found, a large value
               is assigned (1000) to push these variables to the end within their species.
               The third element is the variable name itself, to ensure consistent ordering
               within the same species and shell number.
    """
    for i, species_name in enumerate(baseline_species_names):
        if species_name in var_name:
            # Extract the shell number using regex
            match = re.search(r'(\d+)$', var_name)  # Find digits at the end of the string
            if match:
                try:
                    shell_number = int(match.group(1))
                except ValueError:
                    shell_number = 1000 # Assign a large number to avoid errors if the shell number cannot be converted to int
            else:
                shell_number = 1000 #If no shell number, put at end
            return (i, shell_number, var_name)  # Sort by species, shell number, then variable name

    # If no matching species is found, put it at the end
    return (len(baseline_species_names), 1000, var_name)  # Assign higher index if species not found


def create_vars_dict(ordered_difference_vars):
    vars_dict = {}
    for var in ordered_difference_vars:
        match = re.match(r"([a-zA-Z_]+.*kg)", str(var))  # Match up to "kg"
        if match:
            name = match.group(1) # Extract the name including "kg"
            if name not in vars_dict:
                vars_dict[name] = []
            vars_dict[name].append(var)
        else:
            match = re.match(r"([a-zA-Z_]+)(\d+)", str(var))
            if match:
                name, index = match.groups()
                if name not in vars_dict:
                    vars_dict[name] = []
                vars_dict[name].append(var)
            else: # Single variables
                vars_dict[str(var)] = [var] # Use variable name as key directly
                # if 'other_vars' not in vars_dict:
                #     vars_dict['other_vars'] = []
                # vars_dict['other_vars'].append(var)
    return vars_dict

class CreateOutput:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)
