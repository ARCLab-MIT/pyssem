import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import imageio
try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
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

    def __init__(self, scenario_properties: ScenarioProperties, plots: list, 
                 simulation_name: str = None, main_path: str = 'figures'):
        self.scenario_properties = scenario_properties
        # Handle output attribute access
        if hasattr(scenario_properties, 'output'):
            self.output = scenario_properties.output
        else:
            self.output = scenario_properties.scenario_properties.output
        
        # Ensure output has the expected attributes
        if not hasattr(self.output, 'y') or not hasattr(self.output, 't'):
            print(f"⚠️  Output object missing 'y' or 't' attributes. Type: {type(self.output)}")
            if hasattr(self.output, 'keys'):
                print(f"Available keys: {list(self.output.keys())}")
            # Try to get the actual scenario_properties object
            if hasattr(scenario_properties, 'scenario_properties'):
                self.output = scenario_properties.scenario_properties.output
                print(f"Trying scenario_properties.output: {type(self.output)}")
        # Handle both Model and ScenarioProperties objects
        if hasattr(scenario_properties, 'species_length'):
            # Direct ScenarioProperties object
            self.n_species = scenario_properties.species_length
            self.num_shells = scenario_properties.n_shells
            self.species_names = scenario_properties.species_names
            self.scenario_properties = scenario_properties
        else:
            # For Model objects, get attributes from scenario_properties
            self.n_species = scenario_properties.scenario_properties.species_length
            self.num_shells = scenario_properties.scenario_properties.n_shells
            self.species_names = scenario_properties.scenario_properties.species_names
            self.scenario_properties = scenario_properties.scenario_properties
        self.plots = plots
        self.simulation_name = simulation_name
        self.main_path = main_path

        # Create the figures directory if it doesn't exist
        os.makedirs(f'{self.main_path}/{self.simulation_name}', exist_ok=True)

        # Handle elliptical attribute access
        elliptical = self.scenario_properties.elliptical

        if elliptical:
            # Use the altitude-resolved data directly
            self.output.y = self.scenario_properties.output.y_alt
            print("Using altitude-resolved data for plots.")

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

    def total_objects_over_time_by_species(self):
        import numpy as np
        # --- Setup and filtering ---
        # Handle both Model and ScenarioProperties objects for species_names access
        if hasattr(self.scenario_properties, 'species_names'):
            species_names = list(self.scenario_properties.species_names)
        else:
            species_names = list(self.scenario_properties.scenario_properties.species_names)
        selected_indices = [i for i, name in enumerate(species_names)
                            if isinstance(name, str) and name and name[0] in ('S', 'N', 'B')]

        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)

        if not selected_indices:
            fig = plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No species starting with S, N, or B.", ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/species_totals_SNB.png')
            plt.close(fig)
            return

        # --- One subplot per selected species: SINGLE line = sum over shells ---
        cols = 5
        rows = (len(selected_indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        flat_axes = axes.flatten()

        for plot_idx, species_index in enumerate(selected_indices):
            ax = flat_axes[plot_idx]
            start_idx = species_index * self.num_shells
            end_idx   = start_idx + self.num_shells
            # sum over shells → single line per species
            total_per_species = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            ax.plot(self.output.t, total_per_species, label='Sum over shells')
            ax.set_title(f'{species_names[species_index]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Total objects')
            ax.grid(True, ls='--', lw=0.5, alpha=0.6)

        # Hide unused axes
        for i in range(len(selected_indices), rows * cols):
            fig.delaxes(flat_axes[i])

        plt.suptitle('Selected species (S*/N*/B*): total over shells', y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f'{out_dir}/species_totals_SNB.png', dpi=150)
        plt.close(fig)

        # --- Totals plot across selected species + combined ---
        import numpy as np
        plt.figure(figsize=(10, 6))
        total_all_selected = np.zeros_like(self.output.t, dtype=float)

        for species_index in selected_indices:
            start_idx = species_index * self.num_shells
            end_idx   = start_idx + self.num_shells
            total_per_species = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            plt.plot(self.output.t, total_per_species, label=f'{species_names[species_index]}')
            total_all_selected += total_per_species

        plt.plot(self.output.t, total_all_selected, label='Total (S*/N*/B*)', color='k',
                linewidth=2, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Total Number of Objects')
        plt.title('Objects Over Time — Selected Species (S*/N*/B*) and Total')
        plt.xlim(0, float(np.max(self.output.t)))
        plt.grid(True, ls='--', lw=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_dir}/total_objects_over_time_SNB.png', dpi=150)
        plt.close()

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
            # Handle both Model and ScenarioProperties objects for species_names access
            if hasattr(self.scenario_properties, 'species_names'):
                species_names = self.scenario_properties.species_names
            else:
                species_names = self.scenario_properties.scenario_properties.species_names
            ax.set_title(f'{species_names[species_index]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

        # Hide any unused axes
        for i in range(self.n_species, rows * cols):
            fig.delaxes(axes.flatten()[i])

        plt.suptitle('Species 1 All Shells')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{self.main_path}/{self.simulation_name}/species_all_shells.png')
        plt.close(fig)

        # Plot total objects over time for each species and total
        # Handle both Model and ScenarioProperties objects for species_names access
        if hasattr(self.scenario_properties, 'species_names'):
            species_names = self.scenario_properties.species_names
        else:
            species_names = self.scenario_properties.scenario_properties.species_names
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
        plt.savefig(f'{self.main_path}/{self.simulation_name}/total_objects_over_time.png')
        plt.close()

    def heatmaps_species(self):
        """
        Legacy heatmap output replaced with a combined time-series view that keeps the
        gridded aesthetic and graduated colour palette from the previous version,
        while presenting the evolution of each species as a line plot.
        """
        if self.n_species == 0 or len(self.output.t) == 0:
            fig = plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, "No species data available.", ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.main_path}/{self.simulation_name}/heatmaps_species.png', dpi=150)
            plt.close(fig)
            return

        time_values = np.asarray(self.output.t, dtype=float)
        if hasattr(self.scenario_properties, 'start_date'):
            try:
                start_year = self.scenario_properties.start_date.year
                x_values = start_year + time_values
                x_label = 'Year'
            except AttributeError:
                x_values = time_values
                x_label = 'Time'
        else:
            x_values = time_values
            x_label = 'Time'

        colour_map = plt.cm.get_cmap('viridis', max(self.n_species, 3))

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title('Population Over Time by Species')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Population')

        # Light grid background reminiscent of the heatmap layout
        ax.set_facecolor('#f7f7f7')
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

        for idx, species_name in enumerate(self.species_names):
            start_idx = idx * self.num_shells
            end_idx = start_idx + self.num_shells
            totals = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            ax.plot(
                x_values,
                totals,
                color=colour_map(idx),
                linewidth=2,
                label=species_name
            )

        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax.set_xlim(x_values[0], x_values[-1])

        plt.tight_layout()
        plt.savefig(f'{self.main_path}/{self.simulation_name}/heatmaps_species.png', dpi=150)
        plt.close(fig)

    # def evolution_of_species_gif(self):
    #     # Implementation for evolution_of_species_gif plot
    #     time_points = self.output.t

    #     # Extract the unique base species names (part before the underscore)
    #     base_species_names = [name.split('_')[0] for name in self.species_names]
    #     unique_base_species = list(set(base_species_names))

    #     # Extract weights from species names
    #     def extract_weight(name):
    #         try:
    #             return float(name.split('_')[1].replace('kg', ''))
    #         except (IndexError, ValueError):
    #             return 0

    #     weights = [extract_weight(name) for name in self.species_names]

    #     # Normalize weights to range [0, 1] for color shading and invert to make lower weights darker
    #     max_weight = max(weights)
    #     min_weight = min(weights)
    #     normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    #     inverted_weights = [1 - nw for nw in normalized_weights]

    #     # Create a color map for the base species
    #     color_map = plt.cm.get_cmap('tab20', len(unique_base_species))

    #     # Reshape the data to separate species and shells
    #     n_time_points = len(time_points)
    #     data_reshaped = self.output.y.reshape(self.n_species, self.num_shells, n_time_points)

    #     # Get the x-axis labels from self.scenario_properties.R0_km and slice to match shells_per_species
    #     orbital_shell_labels = self.scenario_properties.R0_km[:self.num_shells]

    #     # Define markers for each species (reuse if more species than markers)
    #     markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

    #     # Directory to save the frames
    #     frames_dir = 'frames'
    #     os.makedirs(frames_dir, exist_ok=True)

    #     # Check if there are species starting with 'S'
    #     species_with_s = [self.species_names[i] for i in range(self.n_species) if base_species_names[i].startswith('S')]

    #     if species_with_s:
    #         # Generate frames for each timestep
    #         for t_idx, t in enumerate(time_points):
    #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    #             # Plot species names that begin with 'S' on the left plot
    #             for species_index in range(self.n_species):
    #                 if base_species_names[species_index].startswith('S'):
    #                     base_color = color_map(unique_base_species.index(base_species_names[species_index]))
    #                     color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
    #                     marker = markers[species_index % len(markers)]
    #                     ax1.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

    #             ax1.set_title('Final Timestep: Species Starting with S')
    #             ax1.set_xlabel('Orbital Shell (R0_km)')
    #             ax1.set_ylabel('Count of Objects')
    #             ax1.legend(title='Species')

    #             # Plot the rest of the species on the right plot
    #             for species_index in range(self.n_species):
    #                 if not base_species_names[species_index].startswith('S'):
    #                     base_color = color_map(unique_base_species.index(base_species_names[species_index]))
    #                     color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
    #                     marker = markers[species_index % len(markers)]
    #                     ax2.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

    #             ax2.set_title('Final Timestep: Other Species')
    #             ax2.set_xlabel('Orbital Shell (R0_km)')
    #             ax2.set_ylabel('Count of Objects')
    #             ax2.legend(title='Species')

    #             plt.tight_layout()

    #             frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
    #             plt.savefig(frame_path)
    #             plt.close(fig)
    #     else:
    #         # Generate frames for each timestep with only one plot
    #         for t_idx, t in enumerate(time_points):
    #             fig, ax = plt.subplots(figsize=(12, 8))

    #             # Plot all species in one plot
    #             for species_index in range(self.n_species):
    #                 base_color = color_map(unique_base_species.index(base_species_names[species_index]))
    #                 color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])
    #                 marker = markers[species_index % len(markers)]
    #                 ax.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=self.species_names[species_index], color=color, marker=marker)

    #             ax.set_title('Final Timestep: All Species')
    #             ax.set_xlabel('Orbital Shell (R0_km)')
    #             ax.set_ylabel('Count of Objects')
    #             ax.legend(title='Species')

    #             plt.tight_layout()

    #             frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
    #             plt.savefig(frame_path)
    #             plt.close(fig)

    #     # Create the GIF
    #     images = [imageio.imread(os.path.join(frames_dir, f'frame_{t_idx:04d}.png')) for t_idx in range(len(time_points))]
    #     gif_path = f'{self.main_path}/{self.simulation_name}/species_shells_evolution_side_by_side.gif'
    #     imageio.mimsave(gif_path, images, duration=0.5)

    #     # Cleanup frames
    #     import shutil
    #     shutil.rmtree(frames_dir)


    def total_objects_by_species_group(self):
        # Implementation for total_objects_by_species_group plot
        # Splitting by sub-group
        # Handle both Model and ScenarioProperties objects
        if hasattr(self.scenario_properties, 'species_length'):
            # Direct ScenarioProperties object
            self.n_species = self.scenario_properties.species_length
            self.num_shells = self.scenario_properties.n_shells
            species_names = self.scenario_properties.species_names
            self.output = self.scenario_properties.output
        else:
            # For Model objects, get attributes from scenario_properties
            self.n_species = self.scenario_properties.scenario_properties.species_length
            self.num_shells = self.scenario_properties.scenario_properties.n_shells
            species_names = self.scenario_properties.scenario_properties.species_names
            self.output = self.scenario_properties.scenario_properties.output
        plt.figure(figsize=(10, 6))

        # Use the solve_ivp object directly
        time_array = self.output.t

        total_objects_all_species = np.zeros_like(time_array)

        # Initialize arrays for different species groups
        total_objects_S_group = np.zeros_like(time_array)
        total_objects_N_group = np.zeros_like(time_array)
        total_objects_B_group = np.zeros_like(time_array)

        for i in range(self.n_species):
            start_idx = i * self.num_shells
            end_idx = start_idx + self.num_shells
            # Use the solve_ivp object directly
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
        plt.plot(time_array, total_objects_S_group, label='S_ Group', color='blue', linewidth=2)
        plt.plot(time_array, total_objects_N_group, label='N_ Group', color='green', linewidth=2)
        plt.plot(time_array, total_objects_B_group, label='B_ Group', color='red', linewidth=2)

        # Plot the total number of objects for all species combined
        plt.plot(time_array, total_objects_all_species, label='Total All Species', color='k', linewidth=2, linestyle='--')

        # Add labels, title, and legend
        plt.xlabel('Time')
        plt.ylabel('Total Number of Objects')
        plt.title('Objects Over Time for Each Species Group and Total')
        plt.xlim(0, max(time_array))
        plt.legend()
        plt.tight_layout()
        # plt.yscale('log')

        # Save the figure
        plt.savefig(f'{self.main_path}/{self.simulation_name}/total_objects_by_species_group.png')
        plt.close()

    def indicator_variables(self):
        # Implementation for indicator_variables plot
        print("Generating indicator_variables plot...")
        from mpl_toolkits.mplot3d import Axes3D

        # Define the directory for saving indicator plots
        indicator_dir = f'{self.main_path}/{self.simulation_name}/indicator_vars'
        os.makedirs(indicator_dir, exist_ok=True)  # Create the directory if it does not exist

        # Handle both Model and ScenarioProperties objects for indicator_results access
        if hasattr(self.scenario_properties, 'indicator_results'):
            indicator_results = self.scenario_properties.indicator_results
        else:
            indicator_results = self.scenario_properties.scenario_properties.indicator_results
        
        # Loop through all indicators in the dataset
        for indicator_name, time_values in indicator_results['indicators'].items():
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

    def collision_plots(self):
        """
        Create collision plots including 3D visualizations and save them in a collision folder.
        """
        print("Generating collision plots...")
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create collision directory
        collision_dir = f'{self.main_path}/{self.simulation_name}/collisions'
        os.makedirs(collision_dir, exist_ok=True)
        
        # Check if we have collision data from species pairs
        if hasattr(self.scenario_properties, 'collision_pairs') and self.scenario_properties.collision_pairs:
            self._create_3d_collision_plots(collision_dir)
        
        # Check if we have indicator variables for collisions
        if (hasattr(self.scenario_properties, 'indicator_results') and 
            self.scenario_properties.indicator_results is not None):
            self._create_collision_indicator_plots(collision_dir)
        
        print(f"Collision plots saved to: {collision_dir}")
    
    def total_objects_by_altitude(self):
        """
        Bar chart of total objects per altitude bin at the final timestep.
        X-axis: altitude bin midpoints (km) from `HMid`.
        Y-axis: total objects in each bin (summed across all species).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Resolve altitude bin midpoints
        if hasattr(self.scenario_properties, 'HMid'):
            altitude_midpoints_km = np.asarray(self.scenario_properties.HMid, dtype=float)
        else:
            # Fallback: derive from min/max and n_shells if needed
            altitude_midpoints_km = np.linspace(
                float(self.scenario_properties.min_altitude),
                float(self.scenario_properties.max_altitude),
                int(self.num_shells)
            )

        # Final timestep index
        final_idx = -1

        # Reshape y to (n_species, n_shells, n_time)
        n_time = self.output.y.shape[1]
        y_reshaped = self.output.y.reshape(self.n_species, self.num_shells, n_time)

        # Sum across species at final timestep → (n_shells,)
        totals_per_shell = np.sum(y_reshaped[:, :, final_idx], axis=0)

        # Plot
        plt.figure(figsize=(12, 6))
        bar_width = max(1.0, (altitude_midpoints_km[1] - altitude_midpoints_km[0]) * 0.8) if len(altitude_midpoints_km) > 1 else 10.0
        plt.bar(altitude_midpoints_km, totals_per_shell, width=bar_width, align='center', edgecolor='black')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Objects per bin')
        plt.title('Total Objects by Altitude Bin (final timestep)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/total_objects_by_altitude.png', dpi=150)
        plt.close()

    def total_objects_large_diameter(self):
        """
        Time series plot of total objects with diameter >10cm (radius >5cm).
        Filters species by radius property and sums populations over time.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Get species properties to filter by radius
        species_properties = []
        if hasattr(self.scenario_properties, 'species_cells'):
            # Access species from species_cells dictionary
            for species_group in self.scenario_properties.species_cells.values():
                if isinstance(species_group, list):
                    species_properties.extend(species_group)
                else:
                    species_properties.append(species_group)
        elif hasattr(self.scenario_properties, 'species'):
            # Access species from species list
            if isinstance(self.scenario_properties.species, dict):
                for species_group in self.scenario_properties.species.values():
                    species_properties.extend(species_group)
            else:
                species_properties = self.scenario_properties.species

        # Filter species with radius > 5cm (diameter > 10cm)
        large_species_indices = []
        for i, species in enumerate(species_properties):
            if hasattr(species, 'radius') and species.radius is not None:
                if species.radius > 0.05:  # 5cm in meters
                    large_species_indices.append(i)

        if not large_species_indices:
            # No large species found, create empty plot
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No species with diameter >10cm found.", ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            out_dir = f'{self.main_path}/{self.simulation_name}'
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f'{out_dir}/total_objects_large_diameter.png', dpi=150)
            plt.close(fig)
            return

        # Calculate total large objects over time
        total_large_objects = np.zeros_like(self.output.t)
        
        for species_idx in large_species_indices:
            start_idx = species_idx * self.num_shells
            end_idx = start_idx + self.num_shells
            # Sum across all shells for this species
            species_total = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            total_large_objects += species_total

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.output.t, total_large_objects, linewidth=2, color='blue')
        plt.xlabel('Time')
        plt.ylabel('Total Objects (diameter >10cm)')
        plt.title('Total Objects with Diameter >10cm Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/total_objects_large_diameter.png', dpi=150)
        plt.close()

    def catastrophic_collision_summary(self):
        """
        Create collision summary plots for catastrophic collisions only.
        Similar to collision_plots but filtered for catastrophic collisions.
        """
        print("Generating catastrophic collision summary plots...")
        
        # Create collision directory
        collision_dir = f'{self.main_path}/{self.simulation_name}/collisions'
        os.makedirs(collision_dir, exist_ok=True)
        
        # Check if we have indicator variables for catastrophic collisions
        if (hasattr(self.scenario_properties, 'indicator_results') and 
            self.scenario_properties.indicator_results is not None):
            self._create_catastrophic_collision_plots(collision_dir)
        
        print(f"Catastrophic collision plots saved to: {collision_dir}")

    def _create_catastrophic_collision_plots(self, collision_dir):
        """
        Create collision plots from indicator variables, filtered for catastrophic collisions.
        """
        try:
            indicators = self.scenario_properties.indicator_results.get('indicators', {})
            
            # Filter for catastrophic collision indicators
            catastrophic_indicators = {}
            for indicator_name, time_data in indicators.items():
                if 'catastrophic' in indicator_name.lower() or 'collision' in indicator_name.lower():
                    catastrophic_indicators[indicator_name] = time_data
            
            if not catastrophic_indicators:
                print("No catastrophic collision indicators found.")
                return
            
            # Create 2x2 subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Total Catastrophic Collisions Over Time
            total_collisions = np.zeros(len(self.output.t))
            for indicator_name, time_data in catastrophic_indicators.items():
                times = np.array(list(time_data.keys()))
                for time in times:
                    time_idx = np.argmin(np.abs(self.output.t - time))
                    data_matrix = time_data[time]
                    if hasattr(data_matrix, 'shape') and len(data_matrix.shape) > 1:
                        total_collisions[time_idx] += np.sum(data_matrix)
                    else:
                        total_collisions[time_idx] += np.sum(data_matrix)
            
            ax1.plot(self.output.t, total_collisions, 'r-', linewidth=2, label='Total Catastrophic Collisions')
            ax1.set_xlabel('Time (years)')
            ax1.set_ylabel('Cumulative Collisions')
            ax1.set_title('Total Catastrophic Collisions Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Species-Specific Catastrophic Collisions
            species_collisions = {}
            for indicator_name, time_data in catastrophic_indicators.items():
                if 'per_species' in indicator_name or any(sp in indicator_name for sp in self.species_names):
                    times = np.array(list(time_data.keys()))
                    species_name = indicator_name.split('_')[0] if '_' in indicator_name else indicator_name
                    species_collisions[species_name] = np.zeros(len(self.output.t))
                    
                    for time in times:
                        time_idx = np.argmin(np.abs(self.output.t - time))
                        data_matrix = time_data[time]
                        if hasattr(data_matrix, 'shape') and len(data_matrix.shape) > 1:
                            species_collisions[species_name][time_idx] = np.sum(data_matrix)
                        else:
                            species_collisions[species_name][time_idx] = np.sum(data_matrix)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(species_collisions)))
            for i, (species, collisions) in enumerate(species_collisions.items()):
                ax2.plot(self.output.t, collisions, color=colors[i], linewidth=2, label=species)
            
            ax2.set_xlabel('Time (years)')
            ax2.set_ylabel('Cumulative Collisions')
            ax2.set_title('Species-Specific Catastrophic Collisions')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. Final Collision Counts by Species
            final_counts = {}
            for species, collisions in species_collisions.items():
                final_counts[species] = collisions[-1]
            
            species_names = list(final_counts.keys())
            counts = list(final_counts.values())
            
            bars = ax3.bar(species_names, counts, color=colors[:len(species_names)])
            ax3.set_xlabel('Species')
            ax3.set_ylabel('Final Collision Count')
            ax3.set_title('Final Catastrophic Collision Counts by Species')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{count:.1f}', ha='center', va='bottom')
            
            # 4. Collision Rate Over Time
            if len(self.output.t) > 1:
                dt = np.diff(self.output.t)
                collision_rate = np.diff(total_collisions) / dt
                ax4.plot(self.output.t[1:], collision_rate, 'g-', linewidth=2, label='Collision Rate')
                ax4.set_xlabel('Time (years)')
                ax4.set_ylabel('Collision Rate (collisions/year)')
                ax4.set_title('Catastrophic Collision Rate Over Time')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            
            plt.tight_layout()
            plt.savefig(f'{collision_dir}/collision_summary_catastrophic.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating catastrophic collision plots: {e}")

    def total_catastrophic_collisions_sum(self):
        """
        Create a plot showing the total sum of all catastrophic collisions over time.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Check if we have indicator variables
        if not (hasattr(self.scenario_properties, 'indicator_results') and 
                self.scenario_properties.indicator_results is not None):
            print("No indicator results found for catastrophic collisions.")
            return
        
        indicators = self.scenario_properties.indicator_results.get('indicators', {})
        
        # Filter for catastrophic collision indicators
        catastrophic_indicators = {}
        for indicator_name, time_data in indicators.items():
            if 'catastrophic' in indicator_name.lower() or 'collision' in indicator_name.lower():
                catastrophic_indicators[indicator_name] = time_data
        
        if not catastrophic_indicators:
            print("No catastrophic collision indicators found.")
            return
        
        # Calculate total catastrophic collisions over time
        total_collisions = np.zeros(len(self.output.t))
        
        for indicator_name, time_data in catastrophic_indicators.items():
            times = np.array(list(time_data.keys()))
            for time in times:
                time_idx = np.argmin(np.abs(self.output.t - time))
                data_matrix = time_data[time]
                if hasattr(data_matrix, 'shape') and len(data_matrix.shape) > 1:
                    total_collisions[time_idx] += np.sum(data_matrix)
                else:
                    total_collisions[time_idx] += np.sum(data_matrix)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.output.t, total_collisions, 'r-', linewidth=3, label='Total Catastrophic Collisions')
        plt.xlabel('Time (years)')
        plt.ylabel('Cumulative Catastrophic Collisions')
        plt.title('Total Sum of Catastrophic Collisions Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/total_catastrophic_collisions_sum.png', dpi=150)
        plt.close()

    def catastrophic_collisions_vs_altitude(self):
        """
        Create catastrophic collisions vs altitude plot with 100km bins.
        Shows cumulative catastrophic collisions per 100km altitude bin.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Check if we have indicator variables
        if not (hasattr(self.scenario_properties, 'indicator_results') and 
                self.scenario_properties.indicator_results is not None):
            print("No indicator results found for catastrophic collisions.")
            return
        
        indicators = self.scenario_properties.indicator_results.get('indicators', {})
        
        # Filter for catastrophic collision indicators with altitude data
        altitude_indicators = {}
        for indicator_name, time_data in indicators.items():
            if ('catastrophic' in indicator_name.lower() and 
                ('altitude' in indicator_name.lower() or 'per_altitude' in indicator_name.lower())):
                altitude_indicators[indicator_name] = time_data
        
        if not altitude_indicators:
            print("No catastrophic collision altitude indicators found.")
            return
        
        # Get altitude bins (100km bins)
        min_alt = self.scenario_properties.min_altitude
        max_alt = self.scenario_properties.max_altitude
        altitude_bins = np.arange(min_alt, max_alt + 100, 100)  # 100km bins
        altitude_centers = (altitude_bins[:-1] + altitude_bins[1:]) / 2
        
        # Calculate cumulative collisions per altitude bin
        cumulative_collisions = np.zeros(len(altitude_centers))
        
        for indicator_name, time_data in altitude_indicators.items():
            # Get final time step data
            final_time = max(time_data.keys())
            data_matrix = time_data[final_time]
            
            if hasattr(data_matrix, 'shape') and len(data_matrix.shape) > 1:
                # Sum across time or other dimensions to get per-altitude values
                altitude_collisions = np.sum(data_matrix, axis=0) if data_matrix.shape[0] > 1 else data_matrix.flatten()
            else:
                altitude_collisions = data_matrix.flatten()
            
            # Map to 100km bins
            for i, alt_center in enumerate(altitude_centers):
                # Find corresponding altitude shell
                alt_shell_idx = int((alt_center - min_alt) / (max_alt - min_alt) * len(altitude_collisions))
                alt_shell_idx = min(alt_shell_idx, len(altitude_collisions) - 1)
                cumulative_collisions[i] += altitude_collisions[alt_shell_idx]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.plot(altitude_centers, cumulative_collisions, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Altitude (km)')
        plt.ylabel('Cumulative Catastrophic Collisions / 100km bin')
        plt.title('Catastrophic Collisions vs Altitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(min_alt, max_alt)
        
        # Add vertical lines for major altitude markers
        for alt in np.arange(200, max_alt + 1, 200):
            if min_alt <= alt <= max_alt:
                plt.axvline(x=alt, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/catastrophic_collisions_vs_altitude.png', dpi=150)
        plt.close()
    
    # ===== IADC study comparison utilities and plots =====
    def _load_iadc_study(self):
        """
        Load digitized IADC study datasets (population, cumulative collisions, altitude distribution).
        Returns a dict with keys 'dd', 'cc', 'aa'.
        """
        try:
            base_dir = os.path.dirname(__file__)
            iadc_path = os.path.join(base_dir, 'iadc_study.json')
            with open(iadc_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading IADC study data: {e}")
            return None

    @staticmethod
    def _split_dd_groups(dd_array: np.ndarray, threshold_year: float = 2010.5) -> list:
        """
        Split dd (population) into agency groups using year < threshold as group starts.
        Emulates MATLAB logic using indices where x < 2010.5 as cut points.
        """
        if dd_array.size == 0:
            return []
        years = dd_array[:, 0]
        cut_idxs = np.where(years < threshold_year)[0]
        if cut_idxs.size == 0:
            return [dd_array]
        groups = []
        for idx, start in enumerate(cut_idxs):
            if idx == cut_idxs.size - 1:
                g = dd_array[start:, :]
            else:
                g = dd_array[start:cut_idxs[idx + 1], :]
            # Fix first row to common start per MATLAB script
            first = np.array([[2008.5, 17076.0]])
            g = g.copy()
            g[0, :] = first
            groups.append(g)
        return groups

    @staticmethod
    def _split_decreasing_groups(arr: np.ndarray) -> list:
        """
        Split an array into groups when x decreases (for cc and aa datasets).
        Keeps the first common point as group prefix for cc; for aa, uses raw groups.
        """
        if arr.size == 0:
            return []
        years = arr[:, 0]
        # indices where sequence decreases
        dec_idxs = np.where(np.diff(years) < 0)[0]
        # group starts are dec_idx+1; include 0 as implicit start
        starts = np.concatenate(([0], dec_idxs + 1))
        groups = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(arr)
            g = arr[s:e, :]
            groups.append(g)
        return groups

    def _compute_total_large_over_time(self):
        """
        Compute our simulation's total objects over time for diameter >10 cm (radius >5 cm).
        Returns (t, total_series).
        """
        # Repurpose logic from total_objects_large_diameter but return data instead of plotting
        species_properties = []
        if hasattr(self.scenario_properties, 'species_cells'):
            for species_group in self.scenario_properties.species_cells.values():
                if isinstance(species_group, list):
                    species_properties.extend(species_group)
                else:
                    species_properties.append(species_group)
        elif hasattr(self.scenario_properties, 'species'):
            if isinstance(self.scenario_properties.species, dict):
                for species_group in self.scenario_properties.species.values():
                    species_properties.extend(species_group)
            else:
                species_properties = self.scenario_properties.species

        large_species_indices = []
        for i, species in enumerate(species_properties):
            if hasattr(species, 'radius') and species.radius is not None:
                if species.radius > 0.05:
                    large_species_indices.append(i)


        total_large_objects = np.zeros_like(self.output.t)
        for species_idx in large_species_indices:
            start_idx = species_idx * self.num_shells
            end_idx = start_idx + self.num_shells
            species_total = np.sum(self.output.y[start_idx:end_idx, :], axis=0)
            total_large_objects += species_total
        return np.asarray(self.output.t, dtype=float), np.asarray(total_large_objects, dtype=float)

    def _compute_initial_total_large(self):
        """
        Compute initial total objects >10cm (radius >5cm) from x0.
        Returns the total count.
        """
        # Check if x0 exists
        if not hasattr(self.scenario_properties, 'x0') or self.scenario_properties.x0 is None:
            print("Warning: x0 not found in scenario_properties")
            return 0.0
        
        x0 = self.scenario_properties.x0
        
        # Debug: print x0 structure
        if isinstance(x0, pd.DataFrame):
            print(f"Debug: x0 columns: {list(x0.columns)}")
            print(f"Debug: x0 shape: {x0.shape}")
        else:
            print(f"Warning: x0 is not a DataFrame, type: {type(x0)}")
            return 0.0
        
        # Get species names for objects >10cm (radius >5cm)
        species_properties = []
        if hasattr(self.scenario_properties, 'species_cells'):
            for species_group in self.scenario_properties.species_cells.values():
                if isinstance(species_group, list):
                    species_properties.extend(species_group)
                else:
                    species_properties.append(species_group)
        elif hasattr(self.scenario_properties, 'species'):
            if isinstance(self.scenario_properties.species, dict):
                for species_group in self.scenario_properties.species.values():
                    species_properties.extend(species_group)
            else:
                species_properties = self.scenario_properties.species

        large_species_names = []
        for species in species_properties:
            if hasattr(species, 'radius') and species.radius is not None:
                radius = species.radius
                # Handle both scalar and array radii
                if isinstance(radius, (list, np.ndarray)):
                    # If any radius in the array is >5cm, include this species
                    if np.any(np.array(radius) > 0.05):
                        large_species_names.append(species.sym_name)
                else:
                    if radius > 0.05:
                        large_species_names.append(species.sym_name)

        # Fallback: use hardcoded species names if no radius info
        if not large_species_names:
            # For IADC: S, N_446kg, N_32kg, B are >10cm
            # Also include N_0.64kg (large debris) which might be >10cm
            large_species_names = ['S', 'N_446kg', 'N_32kg', 'B']
            # Filter to only species that exist in x0 columns
            if isinstance(x0, pd.DataFrame):
                available_species = [s for s in large_species_names if s in x0.columns]
                if available_species:
                    large_species_names = available_species
                    print(f"Debug: Using fallback species names: {large_species_names}")
                else:
                    print(f"Warning: None of the fallback species names found in x0 columns")
                    print(f"Debug: Available x0 columns: {list(x0.columns)}")
                    return 0.0

        # Sum across large species
        total_initial = 0.0
        if isinstance(x0, pd.DataFrame):
            for species_name in large_species_names:
                if species_name in x0.columns:
                    species_total = x0[species_name].sum()
                    total_initial += species_total
                    print(f"Debug: {species_name}: {species_total} objects")
        
        print(f"Debug: Total initial population >10cm: {total_initial}")
        return float(total_initial)

    def _compute_cumulative_catastrophic_collisions(self):
        """
        Compute cumulative catastrophic collisions over time from indicator variables.
        Returns (t, cumulative_collisions).
        
        NOTE: This should match the same indicators used in _compute_collisions_by_altitude()
        to ensure consistency between cumulative and altitude plots.
        """
        if not (hasattr(self.scenario_properties, 'indicator_results') and 
                self.scenario_properties.indicator_results is not None):
            # Return zeros if no indicator data
            return np.asarray(self.output.t, dtype=float), np.zeros_like(self.output.t, dtype=float)
        
        indicators = self.scenario_properties.indicator_results.get('indicators', {})
        
        # Filter for catastrophic collision indicators (ORIGINAL FILTER - restored)
        catastrophic_indicators = {}
        for indicator_name, time_data in indicators.items():
            if 'catastrophic' in indicator_name.lower() or 'collision' in indicator_name.lower():
                catastrophic_indicators[indicator_name] = time_data
        
        if not catastrophic_indicators:
            return np.asarray(self.output.t, dtype=float), np.zeros_like(self.output.t, dtype=float)
        
        # Calculate total catastrophic collisions over time
        total_collisions = np.zeros(len(self.output.t))
        indicator_totals = {}  # Track totals per indicator for debugging
        
        for indicator_name, time_data in catastrophic_indicators.items():
            indicator_sum = 0.0
            times = np.array(list(time_data.keys()))
            for time in times:
                time_idx = np.argmin(np.abs(self.output.t - time))
                data_matrix = np.asarray(time_data[time])
                collision_count = np.sum(data_matrix)
                total_collisions[time_idx] += collision_count
                indicator_sum += collision_count
            indicator_totals[indicator_name] = indicator_sum
        
        # Print diagnostic information
        final_total = total_collisions[-1] if len(total_collisions) > 0 else 0.0
        print(f"\n[_compute_cumulative_catastrophic_collisions] Diagnostic:")
        print(f"  Number of indicators used: {len(catastrophic_indicators)}")
        print(f"  Final cumulative collisions: {final_total:.4f}")
        for name, total in indicator_totals.items():
            print(f"    {name}: {total:.4f}")
        
        return np.asarray(self.output.t, dtype=float), np.asarray(total_collisions, dtype=float)

    def _compute_collisions_by_altitude(self):
        """
        Compute catastrophic collisions by altitude from indicator_results.

        Returns:
            altitude_centers_km : (n_shells,) array of altitude bin midpoints (km)
            collisions_per_alt  : (n_shells,) array of cumulative catastrophic collisions per bin
        """
        # 1) Check indicator_results exists
        if not (hasattr(self.scenario_properties, 'indicator_results') and
                self.scenario_properties.indicator_results is not None):
            return np.array([]), np.array([])

        indicators = self.scenario_properties.indicator_results.get('indicators', {})

        # 2) Select catastrophic collision indicators - USE SAME FILTER AS CUMULATIVE PLOT
        # Use the same filter as cumulative plot: 'catastrophic' in name OR 'collision' in name
        all_catastrophic_indicators = {}
        for indicator_name, time_data in indicators.items():
            if 'catastrophic' in indicator_name.lower() or 'collision' in indicator_name.lower():
                all_catastrophic_indicators[indicator_name] = time_data
        
        # Then filter for those with altitude structure (needed for altitude plot)
        altitude_indicators = {}
        non_altitude_indicators = []
        for indicator_name, time_data in all_catastrophic_indicators.items():
            lname = indicator_name.lower()
            if ('altitude' in lname or 'per_altitude' in lname):
                altitude_indicators[indicator_name] = time_data
            else:
                non_altitude_indicators.append(indicator_name)

        if not altitude_indicators:
            print(f"\n[_compute_collisions_by_altitude] WARNING: No altitude-structured indicators found!")
            print(f"  Found {len(all_catastrophic_indicators)} catastrophic/collision indicators total")
            print(f"  Non-altitude indicators: {non_altitude_indicators}")
            return np.array([]), np.array([])
        
        # Print diagnostic information about which indicators are used
        print(f"\n[_compute_collisions_by_altitude] Diagnostic:")
        print(f"  Total catastrophic/collision indicators: {len(all_catastrophic_indicators)}")
        print(f"  Altitude-structured indicators: {len(altitude_indicators)}")
        print(f"  Non-altitude indicators (excluded from altitude plot): {len(non_altitude_indicators)}")
        for name in altitude_indicators.keys():
            print(f"    {name} (used)")
        if non_altitude_indicators:
            print(f"  Non-altitude indicators (not used in altitude plot):")
            for name in non_altitude_indicators[:5]:  # Show first 5
                print(f"    {name}")
            if len(non_altitude_indicators) > 5:
                print(f"    ... and {len(non_altitude_indicators) - 5} more")

        # 3) Altitude bin centres from scenario_properties (preferred)
        if hasattr(self.scenario_properties, 'HMid'):
            altitude_centers = np.asarray(self.scenario_properties.HMid, dtype=float)
        else:
            # Fallback: uniform shells between min/max
            n_shells = int(self.scenario_properties.n_shells)
            min_alt = float(self.scenario_properties.min_altitude)
            max_alt = float(self.scenario_properties.max_altitude)
            edges = np.linspace(min_alt, max_alt, n_shells + 1)
            altitude_centers = 0.5 * (edges[:-1] + edges[1:])

        n_shells = len(altitude_centers)
        cumulative_collisions = np.zeros(n_shells, dtype=float)

        # 4) For each indicator, grab the final time and sum to per-altitude values
        for indicator_name, time_data in altitude_indicators.items():
            # Find final time key
            final_time = max(time_data.keys())
            data_matrix = np.asarray(time_data[final_time])

            # Now reduce to shape (n_shells,) – sum over all non-altitude axes.
            if data_matrix.ndim == 1:
                # Already 1D – hope it matches n_shells
                if data_matrix.shape[0] != n_shells:
                    # Simple rescaling if lengths differ
                    x_old = np.linspace(0.0, 1.0, data_matrix.shape[0])
                    x_new = np.linspace(0.0, 1.0, n_shells)
                    altitude_collisions = np.interp(x_new, x_old, data_matrix)
                else:
                    altitude_collisions = data_matrix

            elif data_matrix.ndim == 2:
                # Heuristic: whichever axis matches n_shells is altitude
                if data_matrix.shape[0] == n_shells:
                    # shape (alt, something)
                    altitude_collisions = np.sum(data_matrix, axis=1)
                elif data_matrix.shape[1] == n_shells:
                    # shape (something, alt)
                    altitude_collisions = np.sum(data_matrix, axis=0)
                else:
                    # No obvious altitude axis – skip this indicator
                    continue

            else:
                # Higher-dimensional (e.g. species x alt x something)
                # Find an axis with length n_shells and treat it as altitude
                alt_axis = None
                for ax, dim in enumerate(data_matrix.shape):
                    if dim == n_shells:
                        alt_axis = ax
                        break
                if alt_axis is None:
                    continue

                # Sum over all other axes
                axes_to_sum = tuple(i for i in range(data_matrix.ndim) if i != alt_axis)
                altitude_collisions = np.sum(data_matrix, axis=axes_to_sum)

            # Accumulate from this indicator into the total
            cumulative_collisions += altitude_collisions
            print(f"    {indicator_name}: sum={np.sum(altitude_collisions):.4f}, integral={np.trapz(altitude_collisions, altitude_centers):.4f}")

        # Filter to keep only altitudes between 200-2000 km (keep pySSEM original data points)
        if len(altitude_centers) == 0:
            return np.array([]), np.array([])
        
        # Filter to 200-2000 km range
        mask_200_2000 = (altitude_centers >= 200.0) & (altitude_centers <= 2000.0)
        altitude_centers = altitude_centers[mask_200_2000]
        cumulative_collisions = cumulative_collisions[mask_200_2000]
        
        # Print diagnostic information
        total_filtered = np.trapz(cumulative_collisions, altitude_centers)
        sum_filtered = np.sum(cumulative_collisions)
        print(f"\n[IADC Altitude Collisions] pySSEM original data points (200-2000 km):")
        print(f"  Number of altitude points: {len(altitude_centers)}")
        print(f"  Altitude range: {np.min(altitude_centers):.1f} - {np.max(altitude_centers):.1f} km")
        print(f"  Total collisions (integral): {total_filtered:.4f}")
        print(f"  Sum of values: {sum_filtered:.4f}")

        return altitude_centers, cumulative_collisions

    def _compute_final_population_by_altitude(self, bin_width_km=100.0):
        """
        Compute final population by altitude for objects >10cm (radius >5cm).
        Rebins into specified bin width (default 100km).
        
        Returns (altitude_centers_100km, population_per_bin)
        """
        # Get species indices for objects >10cm (radius >5cm)
        species_properties = []
        if hasattr(self.scenario_properties, 'species_cells'):
            for species_group in self.scenario_properties.species_cells.values():
                if isinstance(species_group, list):
                    species_properties.extend(species_group)
                else:
                    species_properties.append(species_group)
        elif hasattr(self.scenario_properties, 'species'):
            if isinstance(self.scenario_properties.species, dict):
                for species_group in self.scenario_properties.species.values():
                    species_properties.extend(species_group)
            else:
                species_properties = self.scenario_properties.species

        large_species_indices = []
        for i, species in enumerate(species_properties):
            if hasattr(species, 'radius') and species.radius is not None:
                radius = species.radius
                # Handle both scalar and array radii
                if isinstance(radius, (list, np.ndarray)):
                    # If any radius in the array is >5cm, include this species
                    if np.any(np.array(radius) > 0.05):
                        large_species_indices.append(i)
                else:
                    if radius > 0.05:
                        large_species_indices.append(i)

        # Fallback: use hardcoded indices if no radius info (matching _compute_total_large_over_time)
        if not large_species_indices:
            large_species_indices = [0, 4, 5, 6]

        # Get final timestep
        final_idx = -1
        n_time = self.output.y.shape[1]
        
        # Reshape y to (n_species, n_shells, n_time)
        y_reshaped = self.output.y.reshape(self.n_species, self.num_shells, n_time)
        
        # Sum across large species at final timestep → (n_shells,)
        final_population_per_shell = np.zeros(self.num_shells)
        for species_idx in large_species_indices:
            if species_idx < self.n_species:
                final_population_per_shell += y_reshaped[species_idx, :, final_idx]

        # Get altitude bin midpoints
        if hasattr(self.scenario_properties, 'HMid'):
            altitude_midpoints_km = np.asarray(self.scenario_properties.HMid, dtype=float)
        else:
            # Fallback: derive from min/max and n_shells
            altitude_midpoints_km = np.linspace(
                float(self.scenario_properties.min_altitude),
                float(self.scenario_properties.max_altitude),
                int(self.num_shells)
            )

        # Get altitude bin edges for rebinning
        min_alt = float(self.scenario_properties.min_altitude)
        max_alt = float(self.scenario_properties.max_altitude)
        
        # Create 100km bins
        bin_edges_100km = np.arange(min_alt, max_alt + bin_width_km, bin_width_km)
        bin_centers_100km = 0.5 * (bin_edges_100km[:-1] + bin_edges_100km[1:])
        
        # Rebin: assign each original shell to a 100km bin
        rebinned_population = np.zeros(len(bin_centers_100km))
        
        for i, alt_mid in enumerate(altitude_midpoints_km):
            # Find which 100km bin this altitude belongs to
            bin_idx = np.digitize(alt_mid, bin_edges_100km) - 1
            # Clamp to valid range
            bin_idx = max(0, min(bin_idx, len(rebinned_population) - 1))
            rebinned_population[bin_idx] += final_population_per_shell[i]

        return bin_centers_100km, rebinned_population

    def _compute_initial_population_by_altitude(self, bin_width_km=100.0):
        """
        Compute initial population by altitude for objects >10cm (radius >5cm) from x0.
        Rebins into specified bin width (default 100km).
        
        Returns (altitude_centers_100km, population_per_bin)
        """
        # Check if x0 exists
        if not hasattr(self.scenario_properties, 'x0') or self.scenario_properties.x0 is None:
            raise ValueError("x0 (initial population) not found in scenario_properties")
        
        x0 = self.scenario_properties.x0
        
        # Get species indices for objects >10cm (radius >5cm)
        # Use same logic as _compute_final_population_by_altitude
        species_properties = []
        if hasattr(self.scenario_properties, 'species_cells'):
            for species_group in self.scenario_properties.species_cells.values():
                if isinstance(species_group, list):
                    species_properties.extend(species_group)
                else:
                    species_properties.append(species_group)
        elif hasattr(self.scenario_properties, 'species'):
            if isinstance(self.scenario_properties.species, dict):
                for species_group in self.scenario_properties.species.values():
                    species_properties.extend(species_group)
            else:
                species_properties = self.scenario_properties.species

        large_species_names = []
        for species in species_properties:
            if hasattr(species, 'radius') and species.radius is not None:
                radius = species.radius
                # Handle both scalar and array radii
                if isinstance(radius, (list, np.ndarray)):
                    # If any radius in the array is >5cm, include this species
                    if np.any(np.array(radius) > 0.05):
                        large_species_names.append(species.sym_name)
                else:
                    if radius > 0.05:
                        large_species_names.append(species.sym_name)

        # Fallback: use hardcoded species names if no radius info
        if not large_species_names:
            # For IADC: S, N_446kg, N_32kg, B are >10cm
            large_species_names = ['S', 'N_446kg', 'N_32kg', 'B']
            # Also check if these exist in x0 columns
            available_species = [s for s in large_species_names if s in x0.columns]
            if available_species:
                large_species_names = available_species

        # Sum across large species for each altitude shell
        if isinstance(x0, pd.DataFrame):
            # x0 is DataFrame with columns = species names, rows = altitude shells
            initial_population_per_shell = np.zeros(len(x0))
            for species_name in large_species_names:
                if species_name in x0.columns:
                    initial_population_per_shell += x0[species_name].values
        else:
            # x0 might be a numpy array - need to map species indices
            # This is less common, but handle it
            raise ValueError("x0 format not recognized. Expected DataFrame.")

        # Get altitude bin midpoints
        if hasattr(self.scenario_properties, 'HMid'):
            altitude_midpoints_km = np.asarray(self.scenario_properties.HMid, dtype=float)
        else:
            # Fallback: derive from min/max and n_shells
            altitude_midpoints_km = np.linspace(
                float(self.scenario_properties.min_altitude),
                float(self.scenario_properties.max_altitude),
                len(initial_population_per_shell)
            )

        # Get altitude bin edges for rebinning
        min_alt = float(self.scenario_properties.min_altitude)
        max_alt = float(self.scenario_properties.max_altitude)
        
        # Create 100km bins
        bin_edges_100km = np.arange(min_alt, max_alt + bin_width_km, bin_width_km)
        bin_centers_100km = 0.5 * (bin_edges_100km[:-1] + bin_edges_100km[1:])
        
        # Rebin: assign each original shell to a 100km bin
        rebinned_population = np.zeros(len(bin_centers_100km))
        
        for i, alt_mid in enumerate(altitude_midpoints_km):
            # Find which 100km bin this altitude belongs to
            bin_idx = np.digitize(alt_mid, bin_edges_100km) - 1
            # Clamp to valid range
            bin_idx = max(0, min(bin_idx, len(rebinned_population) - 1))
            rebinned_population[bin_idx] += initial_population_per_shell[i]

        return bin_centers_100km, rebinned_population

    def iadc_study(self):
        """
        Generate comparison plots between current simulation and digitized IADC study:
        - Population >10cm over time vs agencies
        - Cumulative catastrophic collisions over time vs agencies
        - Catastrophic collisions per altitude band (qualitative comparison)
        """
        data = self._load_iadc_study()
        if not data:
            return

        out_dir = f'{self.main_path}/{self.simulation_name}'
        os.makedirs(out_dir, exist_ok=True)

        # Get simulation start year for time conversion
        start_year = 2008
        if hasattr(self.scenario_properties, 'start_date'):
            try:
                start_year = self.scenario_properties.start_date.year
            except AttributeError:
                start_year = 2008

        # 1) Population comparison (dd)
        try:
            dd = np.asarray(data.get('dd', []), dtype=float)
            dd_groups = self._split_dd_groups(dd)
            yrs = np.arange(2008.0, 2208.0 + 0.0001, 0.1)
            dinterp = []
            for g in dd_groups:
                x = g[:, 0]
                y = g[:, 1]
                # pchip-like monotone cubic not in numpy; use linear as robust fallback
                y_interp = np.interp(yrs, x, y)
                dinterp.append(y_interp)

            # our series - convert time from 0-based to start from 2008
            t_sim, y_sim = self._compute_total_large_over_time()
            t_sim_years = start_year + t_sim
            
            # Get initial population from x0 and prepend to the time series
            try:
                initial_total = self._compute_initial_total_large()
                if initial_total > 0:
                    # Check if first point is already at start_year (might be duplicate)
                    if len(t_sim_years) > 0 and t_sim_years[0] == start_year:
                        # Replace first point with initial population
                        y_sim[0] = initial_total
                        print(f"Debug: Replaced first point with initial population: {initial_total}")
                    else:
                        # Prepend initial point at start_year (t=0)
                        t_sim_years = np.concatenate([[start_year], t_sim_years])
                        y_sim = np.concatenate([[initial_total], y_sim])
                        print(f"Debug: Prepended initial population point: {initial_total} at year {start_year}")
                else:
                    print(f"Warning: Initial population computed as {initial_total}, not adding to plot")
            except Exception as e:
                import traceback
                print(f"Warning: Could not add initial population to plot: {e}")
                traceback.print_exc()

            plt.figure(figsize=(12, 6))
            for arr in dinterp:
                plt.plot(yrs, arr, linewidth=1.2)
            # overlay our simulation
            plt.plot(t_sim_years, y_sim, color='k', linewidth=2.0, label='pySSEM (this run)')
            plt.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSEM'], loc='best')
            plt.ylabel('Total population >10 cm')
            plt.xlabel('Time (year)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_population_comparison.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating IADC population comparison: {e}")

        # 2) Cumulative catastrophic collisions comparison (cc)
        try:
            cc = np.asarray(data.get('cc', []), dtype=float)
            cc_groups = self._split_decreasing_groups(cc)
            yrs = np.arange(2008.0, 2208.0 + 0.0001, 0.1)
            cinterp = []
            for i, g in enumerate(cc_groups):
                # prepend the first common point cc(1,:) per MATLAB logic
                if g.shape[0] == 0:
                    continue
                g_with_first = np.vstack([cc[0, :], g]) if not np.allclose(g[0, :], cc[0, :]) else g
                x = g_with_first[:, 0]
                y = g_with_first[:, 1]
                y_interp = np.interp(yrs, x, y)
                cinterp.append(y_interp)

            # Get our simulation's cumulative catastrophic collisions
            t_sim, y_sim_collisions = self._compute_cumulative_catastrophic_collisions()
            t_sim_years = start_year + t_sim
            
            # Print final cumulative total for comparison
            final_cumulative = y_sim_collisions[-1] if len(y_sim_collisions) > 0 else 0.0
            print(f"\n[Cumulative Collisions Plot] Final cumulative collisions: {final_cumulative:.4f}")

            plt.figure(figsize=(12, 6))
            for arr in cinterp:
                plt.plot(yrs, arr, linewidth=1.2)
            # overlay our simulation
            plt.plot(t_sim_years, y_sim_collisions, color='k', linewidth=2.0, label='pySSEM (this run)')
            plt.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSEM'], loc='upper left')
            plt.ylabel('Cumulative catastrophic collisions')
            plt.xlabel('Time (year)')
            plt.ylim(0, 60)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_collisions_comparison.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating IADC collisions comparison: {e}")

        # 3) Collision altitude distributions (aa)
        try:
            aa = np.asarray(data.get('aa', []), dtype=float)
            aa_groups = self._split_decreasing_groups(aa)
            
            # Get our simulation's collision altitude data
            alt_sim, collisions_sim = self._compute_collisions_by_altitude()
            
            plt.figure(figsize=(12, 6))
            for g in aa_groups:
                if g.shape[0] == 0:
                    continue
                plt.plot(g[:, 0], g[:, 1], linewidth=1.2)
            # overlay our simulation
            plt.plot(alt_sim, collisions_sim, color='k', linewidth=2.0, marker='o', markersize=4, label='pySSEM (this run)')
            plt.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSSEM'], loc='upper left')
            plt.ylabel('Catastrophic collisions per altitude bin')
            plt.xlabel('Altitude (km)')
            plt.xlim(0, 2000)
            plt.ylim(0, 10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_collisions_altitude.png', dpi=150)
            plt.close()
            
            # Print total collisions for comparison
            if len(alt_sim) > 0 and len(collisions_sim) > 0:
                total_pyssem = np.trapz(collisions_sim, alt_sim)
                sum_pyssem = np.sum(collisions_sim)
                print(f"\n[IADC Altitude Plot Totals]")
                print(f"  pySSEM total collisions (integral): {total_pyssem:.4f}")
                print(f"  pySSEM sum of values: {sum_pyssem:.4f}")
                
                # Cross-check: Compare with cumulative collisions at final time
                # The altitude plot shows collisions at final time, so it should match
                # the final cumulative value (if using same indicators)
                try:
                    t_cum, y_cum = self._compute_cumulative_catastrophic_collisions()
                    if len(y_cum) > 0:
                        final_cumulative = y_cum[-1]
                        print(f"  Final cumulative collisions (from time plot): {final_cumulative:.4f}")
                        print(f"  Difference (altitude integral - cumulative final): {total_pyssem - final_cumulative:.4f}")
                        if abs(total_pyssem - final_cumulative) > 0.1:
                            print(f"  WARNING: Significant difference! Altitude and cumulative plots may use different indicators.")
                except Exception as e:
                    print(f"  Could not cross-check with cumulative: {e}")
                
                # Calculate totals for JAXA (Study 4)
                jaxa_group = None
                for g in aa_groups:
                    if g.shape[0] > 0:
                        # Check if this might be JAXA by checking altitude range
                        alt_range = np.max(g[:, 0]) - np.min(g[:, 0])
                        if 449 <= np.min(g[:, 0]) <= 450 and len(g) == 23:
                            jaxa_group = g
                            break
                
                if jaxa_group is None and len(aa_groups) >= 4:
                    # JAXA is Study 4 (index 3)
                    jaxa_group = aa_groups[3] if len(aa_groups) > 3 else None
                
                if jaxa_group is not None and jaxa_group.shape[0] > 0:
                    jaxa_alts = jaxa_group[:, 0]
                    jaxa_colls = jaxa_group[:, 1]
                    total_jaxa = np.trapz(jaxa_colls, jaxa_alts)
                    sum_jaxa = np.sum(jaxa_colls)
                    print(f"  JAXA total collisions (integral): {total_jaxa:.4f}")
                    print(f"  JAXA sum of values: {sum_jaxa:.4f}")
                    print(f"  Ratio (pySSEM/JAXA): {total_pyssem/total_jaxa:.4f}" if total_jaxa > 0 else "  Ratio: N/A")
        except Exception as e:
            print(f"Error creating IADC collision altitude comparison: {e}")

        # 4) Final population vs altitude (objects >10cm, 100km bins)
        try:
            # Get our simulation's final population by altitude
            alt_sim, pop_sim = self._compute_final_population_by_altitude(bin_width_km=100.0)
            
            plt.figure(figsize=(12, 6))
            # Plot our simulation
            plt.plot(alt_sim, pop_sim, color='k', linewidth=2.0, label='pySSEM (this run)')
            plt.legend(['MOCAT-SSEM (this run)'], loc='best')
            plt.ylabel('Objects / 100 km bin')
            plt.xlabel('Altitude (km)')
            plt.title('Final population vs altitude')
            plt.xlim(200, 2000)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_final_population_altitude.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating IADC final population vs altitude plot: {e}")

        # 5) Initial population vs altitude (objects >10cm, 100km bins)
        try:
            # Get our simulation's initial population by altitude from x0
            alt_sim, pop_sim = self._compute_initial_population_by_altitude(bin_width_km=100.0)
            
            plt.figure(figsize=(12, 6))
            # Plot our simulation
            plt.plot(alt_sim, pop_sim, color='k', linewidth=2.0, label='pySSEM (this run)')
            plt.legend(['pySSEM (this run)'], loc='best')
            plt.ylabel('Objects / 100 km bin')
            plt.xlabel('Altitude (km)')
            plt.title('Initial population vs altitude')
            plt.xlim(200, 2000)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_initial_population_altitude.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating IADC initial population vs altitude plot: {e}")

        # 6) Initial vs final population on one plot (thin line + markers)
        try:
            alt_init, pop_init = self._compute_initial_population_by_altitude(bin_width_km=100.0)
            alt_final, pop_final = self._compute_final_population_by_altitude(bin_width_km=100.0)

            # Ensure we have matching altitude bins; if not, skip overlay
            if not np.array_equal(alt_init, alt_final):
                print("Warning: initial and final altitude bins differ; skipping combined plot.")
            else:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    alt_init, pop_init,
                    color='tab:blue', linewidth=1.0,
                    marker='o', markersize=4,
                    label='2008 population'
                )
                plt.plot(
                    alt_final, pop_final,
                    color='tab:orange', linewidth=1.0,
                    marker='o', markersize=4,
                    label='2208 population'
                )
                plt.ylabel('Objects / 100 km bin')
                plt.xlabel('Altitude (km)')
                plt.title('Initial vs Final population by altitude')
                plt.xlim(200, 2000)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                # plt.savefig(f'{out_dir}/iadc_initial_final_population_altitude.png', dpi=150)
                plt.close()
        except Exception as e:
            print(f"Error creating combined initial/final population plot: {e}")

        # 7) Thesis plot: 2x2 subplot layout
        try:
            # Set up figure with larger size for A4 page
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Set larger font sizes for all text
            font_size = 14
            label_size = 16
            legend_size = 14
            tick_size = 12
            
            # Top left: Initial vs Final population by altitude
            alt_init, pop_init = self._compute_initial_population_by_altitude(bin_width_km=100.0)
            alt_final, pop_final = self._compute_final_population_by_altitude(bin_width_km=100.0)
            
            if np.array_equal(alt_init, alt_final):
                ax1.plot(alt_init, pop_init, color='tab:blue', linewidth=3.0, 
                        marker='o', markersize=6, label='Initial (2008)')
                ax1.plot(alt_final, pop_final, color='tab:orange', linewidth=3.0, 
                        marker='s', markersize=6, label='Final (2208)')
                ax1.set_xlabel('Altitude (km)', fontsize=label_size, fontweight='bold')
                ax1.set_ylabel('Objects / 100 km bin', fontsize=label_size, fontweight='bold')
                ax1.set_xlim(200, 2000)
                ax1.tick_params(axis='both', which='major', labelsize=tick_size, width=2, length=6)
                ax1.grid(True, alpha=0.3, linewidth=1.5)
                leg1 = ax1.legend(fontsize=legend_size, frameon=True, framealpha=0.9, edgecolor='black')
                if leg1 is not None:
                    leg1.get_frame().set_linewidth(1.5)
                for spine in ax1.spines.values():
                    spine.set_linewidth(2)
            
            # Top right: Total population comparison over time
            dd = np.asarray(data.get('dd', []), dtype=float)
            dd_groups = self._split_dd_groups(dd)
            yrs = np.arange(2008.0, 2208.0 + 0.0001, 0.1)
            dinterp = []
            for g in dd_groups:
                x = g[:, 0]
                y = g[:, 1]
                y_interp = np.interp(yrs, x, y)
                dinterp.append(y_interp)
            
            t_sim, y_sim = self._compute_total_large_over_time()
            t_sim_years = start_year + t_sim
            
            try:
                initial_total = self._compute_initial_total_large()
                if initial_total > 0:
                    if len(t_sim_years) > 0 and t_sim_years[0] == start_year:
                        y_sim[0] = initial_total
                    else:
                        t_sim_years = np.concatenate([[start_year], t_sim_years])
                        y_sim = np.concatenate([[initial_total], y_sim])
            except Exception:
                pass
            
            # Plot IADC agency data with thinner lines
            for arr in dinterp:
                ax2.plot(yrs, arr, linewidth=1.5, alpha=0.7)
            # Plot our simulation with thicker line
            ax2.plot(t_sim_years, y_sim, color='k', linewidth=3.5, label='pySSEM')
            ax2.set_xlabel('Time (year)', fontsize=label_size, fontweight='bold')
            ax2.set_ylabel('Total population >10 cm', fontsize=label_size, fontweight='bold')
            ax2.set_ylim(0, 30000)
            ax2.tick_params(axis='both', which='major', labelsize=tick_size, width=2, length=6)
            ax2.grid(True, alpha=0.3, linewidth=1.5)
            leg2 = ax2.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSEM', 'pySSEM'], 
                              fontsize=legend_size, frameon=True, framealpha=0.9, edgecolor='black')
            if leg2 is not None:
                leg2.get_frame().set_linewidth(1.5)
            for spine in ax2.spines.values():
                spine.set_linewidth(2)
            
            # Bottom left: Cumulative number of collisions over time
            cc = np.asarray(data.get('cc', []), dtype=float)
            cc_groups = self._split_decreasing_groups(cc)
            yrs = np.arange(2008.0, 2208.0 + 0.0001, 0.1)
            cinterp = []
            for i, g in enumerate(cc_groups):
                if g.shape[0] == 0:
                    continue
                g_with_first = np.vstack([cc[0, :], g]) if not np.allclose(g[0, :], cc[0, :]) else g
                x = g_with_first[:, 0]
                y = g_with_first[:, 1]
                y_interp = np.interp(yrs, x, y)
                cinterp.append(y_interp)
            
            t_sim, y_sim_collisions = self._compute_cumulative_catastrophic_collisions()
            t_sim_years = start_year + t_sim
            
            # Plot IADC agency data with thinner lines
            for arr in cinterp:
                ax3.plot(yrs, arr, linewidth=1.5, alpha=0.7)
            # Plot our simulation with thicker line
            ax3.plot(t_sim_years, y_sim_collisions, color='k', linewidth=3.5, label='pySSEM')
            ax3.set_xlabel('Time (year)', fontsize=label_size, fontweight='bold')
            ax3.set_ylabel('Cumulative catastrophic collisions', fontsize=label_size, fontweight='bold')
            ax3.set_ylim(0, 60)
            ax3.tick_params(axis='both', which='major', labelsize=tick_size, width=2, length=6)
            ax3.grid(True, alpha=0.3, linewidth=1.5)
            leg3 = ax3.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSEM', 'pySSEM'], 
                              fontsize=legend_size, frameon=True, framealpha=0.9, edgecolor='black')
            if leg3 is not None:
                leg3.get_frame().set_linewidth(1.5)
            for spine in ax3.spines.values():
                spine.set_linewidth(2)
            
            # Bottom right: Cumulative collisions by altitude
            aa = np.asarray(data.get('aa', []), dtype=float)
            aa_groups = self._split_decreasing_groups(aa)
            alt_sim, collisions_sim = self._compute_collisions_by_altitude()
            
            # Plot IADC agency data with thinner lines
            for g in aa_groups:
                if g.shape[0] == 0:
                    continue
                ax4.plot(g[:, 0], g[:, 1], linewidth=1.5, alpha=0.7)
            # Plot our simulation with thicker line
            ax4.plot(alt_sim, collisions_sim, color='k', linewidth=3.5, marker='o', 
                    markersize=8, label='pySSEM')
            ax4.set_xlabel('Altitude (km)', fontsize=label_size, fontweight='bold')
            ax4.set_ylabel('Total catastrophic collisions', fontsize=label_size, fontweight='bold')
            ax4.set_xlim(0, 2000)
            ax4.set_ylim(0, 10)
            ax4.tick_params(axis='both', which='major', labelsize=tick_size, width=2, length=6)
            ax4.grid(True, alpha=0.3, linewidth=1.5)
            leg4 = ax4.legend(['ASI', 'ESA', 'ISRO', 'JAXA', 'NASA', 'UKSA', 'MOCAT-SSEM', 'pySSEM'], 
                              fontsize=legend_size, frameon=True, framealpha=0.9, edgecolor='black')
            if leg4 is not None:
                leg4.get_frame().set_linewidth(1.5)
            for spine in ax4.spines.values():
                spine.set_linewidth(2)
            
            plt.tight_layout()
            plt.savefig(f'{out_dir}/iadc_study_thesis.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            import traceback
            print(f"Error creating IADC thesis plot: {e}")
            traceback.print_exc()
    
    def _create_3d_collision_plots(self, collision_dir):
        """
        Create 3D collision plots from collision pair data.
        """
        try:
            # Get collision data from species pairs
            for i, pair in enumerate(self.scenario_properties.collision_pairs):
                if hasattr(pair, 'fragsMadeDV_3d') and pair.fragsMadeDV_3d is not None:
                    # Create 3D plot of fragment distribution
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Get the 3D data: [source_shell, debris_species, destination_shell]
                    data_3d = pair.fragsMadeDV_3d
                    
                    # Create coordinates for 3D scatter plot
                    source_shells, debris_species, dest_shells = np.where(data_3d > 0)
                    fragment_counts = data_3d[source_shells, debris_species, dest_shells]
                    
                    # Create 3D scatter plot
                    scatter = ax.scatter(source_shells, debris_species, dest_shells, 
                                       c=fragment_counts, s=fragment_counts*10, 
                                       cmap='viridis', alpha=0.6)
                    
                    ax.set_xlabel('Source Shell')
                    ax.set_ylabel('Debris Species')
                    ax.set_zlabel('Destination Shell')
                    ax.set_title(f'3D Fragment Distribution - Collision Pair {i+1}\n'
                               f'{pair.s1.sym_name} vs {pair.s2.sym_name}')
                    
                    # Add colorbar
                    plt.colorbar(scatter, label='Fragment Count', shrink=0.8)
                    
                    # Save plot
                    filename = f'3d_collision_pair_{i+1}_{pair.s1.sym_name}_vs_{pair.s2.sym_name}.png'
                    filepath = os.path.join(collision_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved 3D collision plot: {filename}")
                    
        except Exception as e:
            print(f"Error creating 3D collision plots: {e}")
    
    def _create_collision_indicator_plots(self, collision_dir):
        """
        Create collision plots from indicator variables.
        """
        try:
            indicators = self.scenario_properties.indicator_results.get('indicators', {})
            
            for indicator_name, time_data in indicators.items():
                if 'collision' in indicator_name.lower():
                    # Extract time and data
                    times = np.array(list(time_data.keys()))
                    
                    # Create time series plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    for time in times[::max(1, len(times)//10)]:  # Sample times
                        data_matrix = time_data[time]
                        if hasattr(data_matrix, 'shape') and len(data_matrix.shape) > 1:
                            # Sum across one dimension for plotting
                            total_collisions = np.sum(data_matrix, axis=0)
                            ax.plot(range(len(total_collisions)), total_collisions, 
                                   alpha=0.7, label=f'Time {time:.1f}')
                    
                    ax.set_xlabel('Shell/Species Index')
                    ax.set_ylabel('Collision Count')
                    ax.set_title(f'Collision Evolution: {indicator_name}')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot
                    filename = f'collision_evolution_{indicator_name.replace(" ", "_")}.png'
                    filepath = os.path.join(collision_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved collision indicator plot: {filename}")
                    
        except Exception as e:
            print(f"Error creating collision indicator plots: {e}")

    def launch_analysis(self):
        """
        Create launch analysis plots showing yearly launch counts per species.
        """
        if not hasattr(self.scenario_properties, 'FLM_steps') or self.scenario_properties.FLM_steps is None:
            print("Warning: No FLM_steps data available for launch analysis")
            return
        
        # Create launch subfolder
        launch_dir = os.path.join(self.main_path, self.simulation_name, 'launch')
        os.makedirs(launch_dir, exist_ok=True)
        
        # Process FLM_steps data
        df = self.scenario_properties.FLM_steps.copy()
        df['epoch_start_date'] = pd.to_datetime(df['epoch_start_date'])
        
        # Sum across all alt_bin values (i.e., group by epoch_start_date)
        grouped = df.groupby('epoch_start_date').sum(numeric_only=True)
        
        # Identify species columns of interest based on actual species names
        # Look for constellation satellites (S), non-constellation satellites (Su), 
        # non-maneuverable satellites (Sns), and rocket bodies (B)
        s_cols = [col for col in grouped.columns if col == 'S']
        su_cols = [col for col in grouped.columns if col.startswith('Su')]
        sns_cols = [col for col in grouped.columns if col.startswith('Sns')]
        b_cols = [col for col in grouped.columns if col.startswith('B')]
        
        # If no grouped species found, use individual species
        if not s_cols and not su_cols and not sns_cols and not b_cols:
            # Use all species columns except epoch_start_date and alt_bin
            species_cols = [col for col in grouped.columns if col not in ['epoch_start_date', 'alt_bin']]
            grouped['Total_Launches'] = grouped[species_cols].sum(axis=1)
        else:
            # Sum within each category
            if s_cols:
                grouped['S_total'] = grouped[s_cols].sum(axis=1)
            if su_cols:
                grouped['Su_total'] = grouped[su_cols].sum(axis=1)
            if sns_cols:
                grouped['Sns_total'] = grouped[sns_cols].sum(axis=1)
            if b_cols:
                grouped['B_total'] = grouped[b_cols].sum(axis=1)
        
        # Create individual species plots
        self._create_individual_species_launch_plots(grouped, launch_dir)
        
        # Create combined species plot
        self._create_combined_species_launch_plot(grouped, launch_dir)
        
        # Create summary statistics
        self._create_launch_summary_stats(grouped, launch_dir)

    def _create_individual_species_launch_plots(self, grouped_data, launch_dir):
        """
        Create individual launch plots for each species category.
        """
        # Check if we have grouped species or individual species
        if 'Total_Launches' in grouped_data.columns:
            # Individual species mode - create plots for each species
            species_cols = [col for col in grouped_data.columns if col not in ['epoch_start_date', 'alt_bin']]
            
            for species in species_cols:
                if species in grouped_data.columns and grouped_data[species].sum() > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(grouped_data.index, grouped_data[species], linewidth=2, marker='o', markersize=4)
                    
                    plt.xlabel('Year')
                    plt.ylabel('Number of Launches')
                    plt.title(f'Yearly Launch Counts - {species}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save plot
                    filename = f'launch_counts_{species}.png'
                    filepath = os.path.join(launch_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Created launch plot: {filename}")
        else:
            # Grouped species mode
            species_categories = ['S_total', 'Su_total', 'Sns_total', 'B_total']
            species_labels = ['S (Constellation Satellites)', 'Su (Non-Constellation Satellites)', 
                             'Sns (Non-Maneuverable Satellites)', 'B (Rocket Bodies)']
            
            for category, label in zip(species_categories, species_labels):
                if category in grouped_data.columns and grouped_data[category].sum() > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(grouped_data.index, grouped_data[category], linewidth=2, marker='o', markersize=4)
                    
                    plt.xlabel('Year')
                    plt.ylabel('Number of Launches')
                    plt.title(f'Yearly Launch Counts - {label}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save plot
                    filename = f'launch_counts_{category.lower()}.png'
                    filepath = os.path.join(launch_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Created launch plot: {filename}")

    def _create_combined_species_launch_plot(self, grouped_data, launch_dir):
        """
        Create a combined plot showing all species categories together.
        """
        plt.figure(figsize=(14, 8))
        
        # Check if we have grouped species or individual species
        if 'Total_Launches' in grouped_data.columns:
            # Individual species mode - plot all species
            species_cols = [col for col in grouped_data.columns if col not in ['epoch_start_date', 'alt_bin']]
            colors = plt.cm.tab10(np.linspace(0, 1, len(species_cols)))
            
            for species, color in zip(species_cols, colors):
                if species in grouped_data.columns and grouped_data[species].sum() > 0:
                    plt.plot(grouped_data.index, grouped_data[species], 
                            label=species, linewidth=2, marker='o', markersize=4, color=color)
            
            plt.title('Yearly Launch Counts by Species')
        else:
            # Grouped species mode
            species_categories = ['S_total', 'Su_total', 'Sns_total', 'B_total']
            species_labels = ['S (Constellation)', 'Su (Non-Constellation)', 
                             'Sns (Non-Maneuverable)', 'B (Rocket Bodies)']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for category, label, color in zip(species_categories, species_labels, colors):
                if category in grouped_data.columns and grouped_data[category].sum() > 0:
                    plt.plot(grouped_data.index, grouped_data[category], 
                            label=label, linewidth=2, marker='o', markersize=4, color=color)
            
            plt.title('Yearly Launch Counts by Species Category')
        
        plt.xlabel('Year')
        plt.ylabel('Number of Launches')
        plt.legend(title='Species Type')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(launch_dir, 'launch_counts_combined.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created combined launch plot: launch_counts_combined.png")

    def _create_launch_summary_stats(self, grouped_data, launch_dir):
        """
        Create summary statistics and save to CSV.
        """
        # Check if we have grouped species or individual species
        if 'Total_Launches' in grouped_data.columns:
            # Individual species mode - create stats for each species
            species_cols = [col for col in grouped_data.columns if col not in ['epoch_start_date', 'alt_bin']]
            summary_stats = []
            
            for species in species_cols:
                if species in grouped_data.columns:
                    data = grouped_data[species]
                    stats = {
                        'Species': species,
                        'Total_Launches': data.sum(),
                        'Average_Per_Year': data.mean(),
                        'Max_Yearly_Launches': data.max(),
                        'Min_Yearly_Launches': data.min(),
                        'Years_With_Launches': (data > 0).sum(),
                        'First_Launch_Year': data[data > 0].index.min().year if (data > 0).any() else None,
                        'Last_Launch_Year': data[data > 0].index.max().year if (data > 0).any() else None
                    }
                    summary_stats.append(stats)
        else:
            # Grouped species mode
            species_categories = ['S_total', 'Su_total', 'Sns_total', 'B_total']
            species_labels = ['S (Constellation)', 'Su (Non-Constellation)', 
                             'Sns (Non-Maneuverable)', 'B (Rocket Bodies)']
            
            summary_stats = []
            for category, label in zip(species_categories, species_labels):
                if category in grouped_data.columns:
                    data = grouped_data[category]
                    stats = {
                        'Species': label,
                        'Total_Launches': data.sum(),
                        'Average_Per_Year': data.mean(),
                        'Max_Yearly_Launches': data.max(),
                        'Min_Yearly_Launches': data.min(),
                        'Years_With_Launches': (data > 0).sum(),
                        'First_Launch_Year': data[data > 0].index.min().year if (data > 0).any() else None,
                        'Last_Launch_Year': data[data > 0].index.max().year if (data > 0).any() else None
                    }
                    summary_stats.append(stats)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_stats)
        csv_path = os.path.join(launch_dir, 'launch_summary_statistics.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # Print summary
        print("\nLaunch Summary Statistics:")
        print("=" * 50)
        for stats in summary_stats:
            print(f"\n{stats['Species']}:")
            print(f"  Total Launches: {stats['Total_Launches']:,.0f}")
            print(f"  Average per Year: {stats['Average_Per_Year']:.1f}")
            print(f"  Max Yearly: {stats['Max_Yearly_Launches']:,.0f}")
            print(f"  Years with Launches: {stats['Years_With_Launches']}")
            if stats['First_Launch_Year']:
                print(f"  First Launch: {stats['First_Launch_Year']}")
                print(f"  Last Launch: {stats['Last_Launch_Year']}")
        
        print(f"\nSummary statistics saved to: {csv_path}")

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
