
from sympy import zeros, symbols
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

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
    for species_index in range(active_species_indices):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[species_index] * ref[start_idx:end_idx]
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
    for species_index in range(active_species_indices):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[species_index] * np.ones(N_shell)
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
    for species_index in range(active_species_indices):
        start_idx = species_index * N_shell
        end_idx = start_idx + N_shell
        out[start_idx:end_idx] = const[species_index] * np.ones(N_shell)
    return out

#==========================================================================
# Criticality/Capacity Metrics
#==========================================================================

# def cum_CSI_orig(obj, baseline):
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

# print(csi_per_species_per_shell)
# print(csi_per_shell)
# print(csi_per_species)
# print(cum_CSI_total)

# print(csi_per_species_per_shell_new)
# print(np.sum(csi_per_species_per_shell_new, axis=0)) # csi_per_shell
# print(np.sum(csi_per_species_per_shell_new, axis=1)) # csi_per_species
# print(np.sum(csi_per_species_per_shell_new)) # cum_CSI_total 

def cum_CSI(obj, baseline):
    N_shell = baseline.n_shells  # Number of shells
    R02 = baseline.R0_km  # Altitude bins (shell boundaries) in km
    num_species = len(baseline.species_names)  # Number of species
    re = baseline.re

    # Reference values for normalization
    M0 = 10000
    D0 = 5e-8
    life0 = 1468

    # Initialize output array
    csi_per_species_per_shell = np.zeros((num_species, N_shell))

    # Extract species mass and area directly as NumPy arrays
    species_mass = np.array([prop.mass for species_list in baseline.species_cells.values() for prop in species_list])

    # Reshape the object counts for efficient vectorized calculation
    obj_matrix = obj.reshape(num_species, N_shell)

    # Calculate total mass per shell using vectorized operations
    total_mass_per_shell = np.sum(obj_matrix * species_mass[:, None], axis=0)

    D = np.zeros(N_shell)
    lifetime = np.zeros(N_shell)

    # Vectorized calculation of D and lifetime
    h = (R02[:-1] + R02[1:]) / 2  # Use slicing for vectorized addition
    # lifetime[:] = 10 ** (14.18 * (h ** 0.1831) - 42.94) # maya
    lifetime[:] = np.exp(14.18 * (h ** 0.1831) - 42.94)
    r_inner = re + R02[:-1]
    r_outer = re + R02[1:]
    volume_outer = (4 / 3) * np.pi * (r_outer**3)
    volume_inner = (4 / 3) * np.pi * (r_inner**3)
    V = volume_outer - volume_inner
    D[:] = total_mass_per_shell / V

    # Calculate CSI for each species per shell using vectorized operations
    mass_matrix = obj_matrix * species_mass[:, None]
    csi_per_species_per_shell[:] = (mass_matrix / M0) * (D / D0) * (lifetime / life0) # * (1/(1+0.6))         

    return csi_per_species_per_shell.flatten()

# def cum_UMPY(obj, baseline):

#     N_shell = baseline.n_shells  # Number of shells
#     R02 = baseline.R0_km  # Altitude bins (shell boundaries) in km

#     umpy = 0.0
#     species_mass = {}
#     total_objects_per_shell = np.zeros(N_shell)
#     total_mass_per_shell = np.zeros(N_shell)

#     # Create mappings of species names to mass and area
#     for species_name, species_list in baseline.species_cells.items():
#         for species_properties in species_list:
#             species_mass[species_name] = species_properties.mass

#     # Calculate the total objects and total mass in each shell
#     for species_index, (species_name, species_list) in enumerate(baseline.species_cells.items()):
#         start_idx = species_index * N_shell
#         end_idx = start_idx + N_shell

#         # Extract the population for the species in all shells
#         species_counts = obj[start_idx:end_idx]
#         total_objects_per_shell += species_counts
#         total_mass_per_shell += species_counts * species_mass[species_name]

#     # Calculate and sum CSI for each shell for total CSI
#     for shell_index in range(N_shell):
#         h = (R02[shell_index] + R02[shell_index + 1]) / 2
#         lifetime = 10 ** (14.18 * (h ** 0.1831) - 42.94)
#         M = total_mass_per_shell[shell_index]
#         umpy += M * lifetime

#     return umpy

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
# Launch rate - plot
#==========================================================================

def launch_rate_plot(baseline, active_species_indices, x0_lam, x0_lam_no_noise, sel_LineWidth, sel_LineWidthAxis, sel_FontSize):
    N_shell = baseline.n_shells
    for species_index in range(active_species_indices):
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
        plt.title(baseline.species_names[species_index])
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
            ax.set_title(baseline.species_names[species_index]+f", $\lambda$$_{{{i+1}}}$", fontsize=sel_FontSize)
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
    for species_index in range(active_species_indices):
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
            ax.set_title(baseline.species_names[species_index]+f", PMD$_{{{i+1}}}$", fontsize=sel_FontSize)
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
    for species_index in range(active_species_indices):
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
            ax.set_title(baseline.species_names[species_index]+f", $\delta$$_{{{i+1}}}$", fontsize=sel_FontSize)
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
        for species_index in range(active_species_indices):
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
                ax.set_title(baseline.species_names[species_index]+f", PMD$_{{{i+1}}}$", fontsize=sel_FontSize)
                ax.tick_params(width=sel_LineWidthAxis)
                plt.setp(ax.get_xticklabels(), fontsize=sel_FontSize)
                plt.setp(ax.get_yticklabels(), fontsize=sel_FontSize)
            plt.legend(loc='best', bbox_to_anchor=(1, 1))
            plt.show()

    if sel_risk_index == 1:
        # Figure: Risk index
        plt.figure(facecolor='white',figsize=(12, 8))
        plt.grid(True)
        csi_total = np.zeros_like(output.t)
        csi_total_nc = np.zeros_like(output.t)
        handles = []
        labels = []
        for species_index in range(n_species):
            color = color_map(unique_base_species.index(base_species_names[species_index]))
            # marker = markers[species_index % len(markers)]
            start_idx = species_index * num_shells
            end_idx = start_idx + num_shells
            csi_per_species = np.sum(output.csi_per_species_per_shell[start_idx:end_idx, :], axis=0)
            csi_per_species_nc = np.sum(output.csi_per_species_per_shell_nc[start_idx:end_idx, :], axis=0)
            plt.plot(output.t, csi_per_species, linewidth=sel_LineWidth, label=species_names[species_index]+"$^{C}$", color=color, linestyle='-') #, marker=marker, markersize=sel_MarkerWidth)
            plt.plot(output.t,csi_per_species_nc, linewidth=sel_LineWidth, label=species_names[species_index]+"$^{NC}$", color=color, linestyle=':')
            csi_total += csi_per_species
            csi_total_nc += csi_per_species_nc
            if species_names[species_index] not in labels:
                handles.append(Line2D([], [], color=color, linewidth=sel_LineWidth, label=species_names[species_index]))
                labels.append(species_names[species_index])
        plt.plot(output.t, csi_total, label='Total$^{C}$', color='k', linewidth=sel_LineWidth, linestyle='-')
        plt.plot(output.t, csi_total_nc, label='Total$^{NC}$', color='k', linewidth=sel_LineWidth, linestyle=':')
        handles.append(Line2D([], [], color='k', linewidth=sel_LineWidth, label='Total'))
        labels.append('Total')
        handles.append(Line2D([], [], color='k', linestyle='-', linewidth=sel_LineWidth, label='Controlled'))
        handles.append(Line2D([], [], color='k', linestyle=':', linewidth=sel_LineWidth, label='No Control'))
        labels.append("Controlled")
        labels.append("No Control")
        plt.xlabel('Time (years)', fontsize=sel_FontSize)
        plt.ylabel('CSI (-)', fontsize=sel_FontSize)
        plt.title('Cumulative CSI vs Time')
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
    elif sel_risk_index == 2: # TO DO
        1
    elif sel_risk_index == 3: # TO DO
        1


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
