import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import imageio

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

def create_plots(self):
    """
    Generates a number of plots and deposits them into the figures folder.
    """
    print('Making plots')
    scenario_properties = self.scenario_properties
    output = scenario_properties.output
    os.makedirs('figures', exist_ok=True)

    # Plot each species across all shells
    n_species = scenario_properties.species_length
    num_shells = scenario_properties.n_shells

    # Dynamically determine the layout for subplots
    cols = 5  # Define the number of columns you want
    rows = (n_species + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D, even for a single row

    for species_index in range(n_species):
        ax = axes.flatten()[species_index]
        species_data = output.y[species_index * num_shells:(species_index + 1) * num_shells]

        for shell_index in range(num_shells):
            ax.plot(output.t, species_data[shell_index], label=f'Shell {shell_index + 1}')

        total = np.sum(species_data, axis=0)
        ax.set_title(f'{scenario_properties.species_names[species_index]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

    # Hide any unused axes
    for i in range(n_species, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.suptitle('Species 1 All Shells')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/species_all_shells.png')
    plt.close(fig)

    # Plot total objects over time for each species and total
    species_names = scenario_properties.species_names
    plt.figure(figsize=(10, 6))

    total_objects_all_species = np.zeros_like(output.t)

    for i in range(n_species):
        start_idx = i * num_shells
        end_idx = start_idx + num_shells
        total_objects_per_species = np.sum(output.y[start_idx:end_idx, :], axis=0)
        plt.plot(output.t, total_objects_per_species, label=f'{species_names[i]}')
        total_objects_all_species += total_objects_per_species

    plt.plot(output.t, total_objects_all_species, label='Total All Species', color='k', linewidth=2, linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Total Number of Objects')
    plt.title('Objects Over Time for Each Species and Total')
    plt.xlim(0, max(output.t))
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/total_objects_over_time.png')
    plt.close()

    # Plot heatmap for each species
    n_time_points = len(output["t"])
    cols = 3
    rows = (n_species + cols - 1) // cols

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 6 * rows))
    axs = np.atleast_2d(axs)  # Ensure axs is always 2D

    for i, species_name in enumerate(species_names):
        row = i // cols
        col = i % cols
        ax = axs[row, col]

        start_idx = i * num_shells
        end_idx = start_idx + num_shells
        data_per_species = output["y"][start_idx:end_idx, :]

        cax = ax.imshow(data_per_species, aspect='auto', origin='lower',
                        extent=[output["t"][0], output["t"][-1], 0, num_shells],
                        interpolation='nearest')
        fig.colorbar(cax, ax=ax, label='Number of Objects')
        ax.set_xlabel('Time')
        ax.set_ylabel('Orbital Shell')
        ax.set_title(species_name)
        ax.set_xticks(np.linspace(output["t"][0], output["t"][-1], num=5))
        ax.set_yticks(np.arange(0, num_shells, max(1, num_shells // 5)))
        ax.set_yticklabels([f'{alt:.0f}' for alt in scenario_properties.HMid[::max(1, num_shells // 5)]])

    # Hide any unused axes
    for i in range(n_species, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.savefig('figures/heatmaps_species.png')
    plt.close(fig)

    time_points = output.t

    # Extract the unique base species names (part before the underscore)
    base_species_names = [name.split('_')[0] for name in species_names]
    unique_base_species = list(set(base_species_names))

    # Extract weights from species names
    def extract_weight(name):
        try:
            return float(name.split('_')[1].replace('kg', ''))
        except (IndexError, ValueError):
            return 0

    weights = [extract_weight(name) for name in species_names]

    # Normalize weights to range [0, 1] for color shading and invert to make lower weights darker
    max_weight = max(weights)
    min_weight = min(weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    inverted_weights = [1 - nw for nw in normalized_weights]

    # Create a color map for the base species
    color_map = plt.cm.get_cmap('tab20', len(unique_base_species))

    # Reshape the data to separate species and shells
    n_time_points = len(time_points)
    data_reshaped = output.y.reshape(n_species, num_shells, n_time_points)

    # Get the x-axis labels from scenario_properties.R0_km and slice to match shells_per_species
    orbital_shell_labels = scenario_properties.R0_km[:num_shells]

    # Define markers for each species (reuse if more species than markers)
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

    # Directory to save the frames
    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Check if there are species starting with 'S'
    species_with_s = [species_names[i] for i in range(n_species) if base_species_names[i].startswith('S')]

    if species_with_s:
        # Generate frames for each timestep
        for t_idx, t in enumerate(time_points):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

            # Plot species names that begin with 'S' on the left plot
            for species_index in range(n_species):
                if base_species_names[species_index].startswith('S'):
                    base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                    color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                    marker = markers[species_index % len(markers)]
                    ax1.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=species_names[species_index], color=color, marker=marker)

            ax1.set_title('Final Timestep: Species Starting with S')
            ax1.set_xlabel('Orbital Shell (R0_km)')
            ax1.set_ylabel('Count of Objects')
            ax1.legend(title='Species')

            # Plot the rest of the species on the right plot
            for species_index in range(n_species):
                if not base_species_names[species_index].startswith('S'):
                    base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                    color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                    marker = markers[species_index % len(markers)]
                    ax2.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=species_names[species_index], color=color, marker=marker)

            ax2.set_title('Final Timestep: Other Species')
            ax2.set_xlabel('Orbital Shell (R0_km)')
            ax2.set_ylabel('Count of Objects')
            ax2.legend(title='Species')

            plt.tight_layout()

            frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
            plt.savefig(frame_path)
            plt.close(fig)
    else:
        # Generate frames for each timestep with only one plot
        for t_idx, t in enumerate(time_points):
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot all species in one plot
            for species_index in range(n_species):
                base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                marker = markers[species_index % len(markers)]
                ax.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=species_names[species_index], color=color, marker=marker)

            ax.set_title('Final Timestep: All Species')
            ax.set_xlabel('Orbital Shell (R0_km)')
            ax.set_ylabel('Count of Objects')
            ax.legend(title='Species')

            plt.tight_layout()

            frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
            plt.savefig(frame_path)
            plt.close(fig)

    # Create the GIF
    images = [imageio.imread(os.path.join(frames_dir, f'frame_{t_idx:04d}.png')) for t_idx in range(len(time_points))]
    gif_path = 'figures/species_shells_evolution_side_by_side.gif'
    imageio.mimsave(gif_path, images, duration=0.5)

    # Cleanup frames
    import shutil
    shutil.rmtree(frames_dir)


    # Splitting by sub-group
    output = scenario_properties.output  # Output object from simulation
    n_species = scenario_properties.species_length  # Number of species
    num_shells = scenario_properties.n_shells  # Number of shells per species
    species_names = scenario_properties.species_names  # List of species names
    plt.figure(figsize=(10, 6))

    total_objects_all_species = np.zeros_like(output.t)

    # Initialize arrays for different species groups
    total_objects_S_group = np.zeros_like(output.t)
    total_objects_N_group = np.zeros_like(output.t)
    total_objects_B_group = np.zeros_like(output.t)

    for i in range(n_species):
        start_idx = i * num_shells
        end_idx = start_idx + num_shells
        total_objects_per_species = np.sum(output.y[start_idx:end_idx, :], axis=0)

        # Check species group by name and sum accordingly
        species_name = species_names[i]
        
        if species_name.startswith('S_'):
            total_objects_S_group += total_objects_per_species
        elif species_name.startswith('N_'):
            total_objects_N_group += total_objects_per_species
        elif species_name.startswith('B_'):
            total_objects_B_group += total_objects_per_species
        
        total_objects_all_species += total_objects_per_species

    # Plot each species group in different colors
    plt.plot(output.t, total_objects_S_group, label='S_ Group', color='blue', linewidth=2)
    plt.plot(output.t, total_objects_N_group, label='N_ Group', color='green', linewidth=2)
    plt.plot(output.t, total_objects_B_group, label='B_ Group', color='red', linewidth=2)

    # Plot the total number of objects for all species combined
    plt.plot(output.t, total_objects_all_species, label='Total All Species', color='k', linewidth=2, linestyle='--')

    # Add labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Total Number of Objects')
    plt.title('Objects Over Time for Each Species Group and Total')
    plt.xlim(0, max(output.t))
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')

    # Save the figure
    plt.savefig('figures/total_objects_by_species_group.png')
    plt.close()

def results_to_json(self):
        """
        Converts the output of solve_ivp (integrator) to a JSON format. 
        
        :return: JSON string representation of the solve_ivp output.
        """
        # Initialize the data dictionary
        data = {
            "times": self.scenario_properties.output.t.tolist(),
            "n_shells": self.scenario_properties.n_shells,
            "species": [species for species in self.scenario_properties.species_names],
            "Hmid": self.scenario_properties.HMid.tolist(),
            "max_altitude": self.scenario_properties.max_altitude,
            "min_altitude": self.scenario_properties.min_altitude,
            "population_data": [],
            "launch": []
        }

        # Extract relevant parts from the scenario properties for population data
        num_species = len(self.scenario_properties.species_names)
        num_time_steps = len(self.scenario_properties.output.t)

        # Create a DataFrame to mimic the structure of FLM_steps
        df_data = {
            'epoch_start_date': self.scenario_properties.output.t.tolist()
        }

        # Initialize population data structure
        population_data_dict = {species: [[0] * num_time_steps for _ in range(self.scenario_properties.n_shells)]
                                for species in self.scenario_properties.species_names}

        # Populate population data
        for i in range(num_species):
            species = self.scenario_properties.species_names[i]
            for j in range(self.scenario_properties.n_shells):
                shell_index = i * self.scenario_properties.n_shells + j
                population_data_dict[species][j] = self.scenario_properties.output.y[shell_index, :].tolist()
                shell_data = {
                    "species": species,
                    "shell": j + 1,
                    "populations": self.scenario_properties.output.y[shell_index, :].tolist()
                }
                data["population_data"].append(shell_data)

        # Add species columns data to DataFrame
        for species, population_by_shell in population_data_dict.items():
            df_data[species] = [sum(x) for x in zip(*population_by_shell)]

        # Create DataFrame
        df = pd.DataFrame(df_data)
        df['epoch_start_date'] = pd.to_datetime(df['epoch_start_date'], unit='s')

        # Filter species columns starting with 'S'
        species_columns = [col for col in df.columns if col.startswith('S')]

        # Group by 'epoch_start_date' and sum the values for each species across all alt_bins
        grouped_data = df.groupby('epoch_start_date')[species_columns].sum().reset_index()

        # Append launch counts for each species
        for species in species_columns:
            data["launch"].append({
                "species": species,
                "counts": grouped_data[species].tolist()
            })

        # Convert the dictionary to a JSON string
        json_output = json.dumps(data, indent=4, default=str)  # Use default=str to handle datetime serialization

        return json_output

