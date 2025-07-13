import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import imageio
from ..simulation.scen_properties import ScenarioProperties

class Plots:
    """
    This will take a scenario properties class as input and generate each of the specified plots.

    Possible Plots:
    - total_objects_over_time
    - heatmaps_species
    - evolution_of_species_gif
    - total_objects_by_species_group
    - indicator_variables
    - all_plots

    :param scenario_properties: ScenarioProperties object containing the simulation properties.
    :param plots: List of plots to generate. If 'all_plots' is included, all plots will be generated.
    """

    def __init__(self, scenario_properties: ScenarioProperties, plots: list, simulation_name: str = None):
        self.scenario_properties = scenario_properties
        self.output = scenario_properties.output
        self.n_species = scenario_properties.species_length
        self.num_shells = scenario_properties.n_shells
        self.plots = plots
        self.species_names = scenario_properties.species_names
        self.simulation_name = simulation_name

        # Create the figures directory if it doesn't exist
        os.makedirs(f'figures/{self.simulation_name}', exist_ok=True)

        if "all_plots" in self.plots:
            self.all_plots()
        else:
            # Dynamically generate plots
            for plot_name in self.plots:
                plot_method = getattr(self, plot_name, None)
                if callable(plot_method):
                    print(f"Creating plot: {plot_name}")
                    plot_method()
                else:
                    print(f"Warning: Plot '{plot_name}' not found. Skipping...")

        print("Plots generated successfully.")

    def total_objects_over_time(self):
        # Implementation for total_objects_over_time plot

        cols = 5  # Define the number of columns you want
        rows = (self.n_species + cols - 1) // cols  # Calculate the number of rows needed

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_2d(axes)  # Ensure axes is always 2D, even for a single row

        for species_index in range(self.n_species):
            ax = axes.flatten()[species_index]
            species_data = self.output.y[species_index * self.num_shells:(species_index + 1) * self.num_shells]

            for shell_index in range(self.num_shells):
                ax.plot(self.output.t, species_data[shell_index], label=f'Shell {shell_index + 1}')

            total = np.sum(species_data, axis=0)
            ax.set_title(f'{self.scenario_properties.species_names[species_index]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')


        # Hide any unused axes
        for i in range(self.n_species, rows * cols):
            fig.delaxes(axes.flatten()[i])

        plt.suptitle('Species 1 All Shells')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'figures/{self.simulation_name}/species_all_shells.png')
        plt.close(fig)

        # Plot total objects over time for each species and total
        species_names = self.scenario_properties.species_names
        plt.figure(figsize=(10, 6))

        total_objects_all_species = np.zeros_like(self.output.t)

        for i in range(self.n_species):
            start_idx = i * self.num_shells
            end_idx = start_idx + self.num_shells
            total_objects_per_species = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            plt.plot(self.output.t, total_objects_per_species, label=f'{species_names[i]}')
            total_objects_all_species += total_objects_per_species

        plt.plot(self.output.t, total_objects_all_species, label='Total All Species', color='k', linewidth=2, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Total Number of Objects')
        plt.title('Objects Over Time for Each Species and Total')
        plt.xlim(0, max(self.output.t))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/{self.simulation_name}/total_objects_over_time.png')
        plt.close()

    def heatmaps_species(self):
        # Implementation for heatmaps_species plot
        # Plot heatmap for each species
        n_time_points = len(self.output["t"])
        cols = 3
        rows = (self.n_species + cols - 1) // cols

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 6 * rows))
        axs = np.atleast_2d(axs)  # Ensure axs is always 2D

        for i, species_name in enumerate(self.species_names):
            row = i // cols
            col = i % cols
            ax = axs[row, col]

            start_idx = i * self.num_shells
            end_idx = start_idx + self.num_shells
            data_per_species = self.output["y"][start_idx:end_idx, :]

            cax = ax.imshow(data_per_species, aspect='auto', origin='lower',
                            extent=[self.output["t"][0], self.output["t"][-1], 0, self.num_shells],
                            interpolation='nearest')
            fig.colorbar(cax, ax=ax, label='Number of Objects')
            ax.set_xlabel('Time')
            ax.set_ylabel('Orbital Shell')
            ax.set_title(species_name)
            ax.set_xticks(np.linspace(self.output["t"][0], self.output["t"][-1], num=5))
            ax.set_yticks(np.arange(0, self.num_shells, max(1, self.num_shells // 5)))
            ax.set_yticklabels([f'{alt:.0f}' for alt in self.scenario_properties.HMid[::max(1, self.num_shells // 5)]])

        # Hide any unused axes
        for i in range(self.n_species, rows * cols):
            fig.delaxes(axs.flatten()[i])

        plt.tight_layout()
        plt.savefig(f'figures/{self.simulation_name}/heatmaps_species.png')
        plt.close(fig)

    def evolution_of_species_gif(self):
        # Implementation for evolution_of_species_gif plot
        time_points = self.output.t

        # Extract the unique base species names (part before the underscore)
        base_species_names = [name.split('_')[0] for name in self.species_names]
        unique_base_species = list(set(base_species_names))

        # Extract weights from species names
        def extract_weight(name):
            try:
                return float(name.split('_')[1].replace('kg', ''))
            except (IndexError, ValueError):
                return 0

        weights = [extract_weight(name) for name in self.species_names]

        # Normalize weights to range [0, 1] for color shading and invert to make lower weights darker
        max_weight = max(weights)
        min_weight = min(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        inverted_weights = [1 - nw for nw in normalized_weights]

        # Create a color map for the base species
        color_map = plt.cm.get_cmap('tab20', len(unique_base_species))

        # Reshape the data to separate species and shells
        n_time_points = len(time_points)
        data_reshaped = self.output.y.reshape(self.n_species, self.num_shells, n_time_points)

        # Get the x-axis labels from self.scenario_properties.R0_km and slice to match shells_per_species
        orbital_shell_labels = self.scenario_properties.R0_km[:self.num_shells]

        # Define markers for each species (reuse if more species than markers)
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

        # Directory to save the frames
        frames_dir = 'frames'
        os.makedirs(frames_dir, exist_ok=True)

        # Check if there are species starting with 'S'
        species_with_s = [self.species_names[i] for i in range(self.n_species) if base_species_names[i].startswith('S')]

        if species_with_s:
            # Generate frames for each timestep
            for t_idx, t in enumerate(time_points):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

                # Plot species names that begin with 'S' on the left plot
                for species_index in range(self.n_species):
                    if base_species_names[species_index].startswith('S'):
                        base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                        color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                        marker = markers[species_index % len(markers)]
                        ax1.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

                ax1.set_title('Final Timestep: Species Starting with S')
                ax1.set_xlabel('Orbital Shell (R0_km)')
                ax1.set_ylabel('Count of Objects')
                ax1.legend(title='Species')

                # Plot the rest of the species on the right plot
                for species_index in range(self.n_species):
                    if not base_species_names[species_index].startswith('S'):
                        base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                        color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                        marker = markers[species_index % len(markers)]
                        ax2.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

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
                for species_index in range(self.n_species):
                    base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                    color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
                    marker = markers[species_index % len(markers)]
                    ax.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

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
        gif_path = f'figures/{self.simulation_name}/species_shells_evolution_side_by_side.gif'
        imageio.mimsave(gif_path, images, duration=0.5)

        # Cleanup frames
        import shutil
        shutil.rmtree(frames_dir)


    def total_objects_by_species_group(self):
        # Implementation for total_objects_by_species_group plot
        # Splitting by sub-group
        self.output = self.scenario_properties.output  # self.output object from simulation
        self.n_species = self.scenario_properties.species_length  # Number of species
        self.num_shells = self.scenario_properties.n_shells  # Number of shells per species
        species_names = self.scenario_properties.species_names  # List of species names
        plt.figure(figsize=(10, 6))

        total_objects_all_species = np.zeros_like(self.output.t)

        # Initialize arrays for different species groups
        total_objects_S_group = np.zeros_like(self.output.t)
        total_objects_N_group = np.zeros_like(self.output.t)
        total_objects_B_group = np.zeros_like(self.output.t)

        for i in range(self.n_species):
            start_idx = i * self.num_shells
            end_idx = start_idx + self.num_shells
            total_objects_per_species = np.sum(self.output.y[start_idx:end_idx, :], axis=0)

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
        plt.plot(self.output.t, total_objects_S_group, label='S_ Group', color='blue', linewidth=2)
        plt.plot(self.output.t, total_objects_N_group, label='N_ Group', color='green', linewidth=2)
        plt.plot(self.output.t, total_objects_B_group, label='B_ Group', color='red', linewidth=2)

        # Plot the total number of objects for all species combined
        plt.plot(self.output.t, total_objects_all_species, label='Total All Species', color='k', linewidth=2, linestyle='--')

        # Add labels, title, and legend
        plt.xlabel('Time')
        plt.ylabel('Total Number of Objects')
        plt.title('Objects Over Time for Each Species Group and Total')
        plt.xlim(0, max(self.output.t))
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')

        # Save the figure
        plt.savefig(f'figures/{self.simulation_name}/total_objects_by_species_group.png')
        plt.close()

    def indicator_variables(self):
        # Implementation for indicator_variables plot
        print("Generating indicator_variables plot...")
        from mpl_toolkits.mplot3d import Axes3D

        # Define the directory for saving indicator plots
        indicator_dir = f'figures/{self.simulation_name}/indicator_vars'
        os.makedirs(indicator_dir, exist_ok=True)  # Create the directory if it does not exist

        # Loop through all indicators in the dataset
        for indicator_name, time_values in self.scenario_properties.indicator_results['indicators'].items():
            # Extract time and indicator values
            times = np.array(list(time_values.keys()))  # Time array
            indicator_matrix = np.array([np.squeeze(values) for values in time_values.values()])  # Shape: [num_times, num_shells]

            # Orbital shells (assume consistent number of shells from the matrix)
            num_shells = indicator_matrix.shape[1]
            orbital_shells = np.arange(num_shells)

            # Transpose the indicator matrix to align with the meshgrid
            Z = indicator_matrix.T  # Shape: [num_shells, num_times]

            # Create a meshgrid for the surface plot
            X, Y = np.meshgrid(times, orbital_shells)

            # Create the plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

            # Add labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Orbital Shell')
            ax.set_zlabel('Indicator Value')
            ax.set_title(f'Surface Plot of {indicator_name}')
            fig.colorbar(surf, shrink=0.5, aspect=10, label='Indicator Value')

            # Save the plot with the title as the filename
            filename = f"{indicator_name.replace(' ', '_')}.png"  # Replace spaces with underscores
            file_path = os.path.join(indicator_dir, filename)
            plt.savefig(file_path)

            # Close the plot to free up memory
            plt.close()

            print(f"Saved indicator plot: {file_path}")
        pass

    def all_plots(self):
        """
        Run all plot functions, irrespective of the plots list.
        """
        print("Generating all plots...")
        for attr_name in dir(self):
            if callable(getattr(self, attr_name)) and attr_name not in ("__init__", "all_plots"):
                if not attr_name.startswith("_"):
                    print(f"Creating plot: {attr_name}")
                    getattr(self, attr_name)()


def results_to_json(self):
        """
        Converts the self.output of solve_ivp (integrator) to a JSON format. 
        
        :return: JSON string representation of the solve_ivp self.output.
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
        self.output = data #json.dumps(data, indent=4, default=str)  # Use default=str to handle datetime serialization

        return self.output

