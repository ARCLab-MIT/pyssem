import os
import matplotlib.pyplot as plt
import numpy as np

def create_plots(scenario_properties):
        """
        Generates a number of plots and diposits them into the figures folder:
        """

        output = scenario_properties.output
        os.makedirs('figures', exist_ok=True)

        # Plot each species across all shells
        n_species = scenario_properties.species_length
        num_shells = scenario_properties.n_shells

        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        for species_index in range(n_species):
            ax = axes.flatten()[species_index]
            species_data = output.y[species_index * num_shells:(species_index + 1) * num_shells]

            for shell_index in range(num_shells):
                ax.plot(output.t, species_data[shell_index], label=f'Shell {shell_index + 1}')

            total = np.sum(species_data, axis=0)
            ax.set_title(f'{scenario_properties.species_names[species_index]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

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
        rows = np.ceil(n_species / cols).astype(int)

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(12, rows * 6))
        for i, species_name in enumerate(species_names):
            row = i // cols
            col = i % cols
            ax = axs[row, col] if rows > 1 else axs[col]

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
            ax.set_yticks(np.arange(0, num_shells, 5))
            ax.set_yticklabels([f'{alt:.0f}' for alt in scenario_properties.HMid[::5]])

        for i in range(n_species, rows * cols):
            if rows == 1:
                fig.delaxes(axs[i])
            else:
                axs.flatten()[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('figures/heatmaps_species.png')
        plt.close(fig)