from .utils.simulation.scen_properties import ScenarioProperties
from .utils.simulation.species import Species
from .utils.collisions.collisions import create_collision_pairs
# if testing locally, use the following import statements
# from utils.simulation.scen_properties import ScenarioProperties
# from utils.simulation.species import Species
# from utils.collisions.collisions import create_collision_pairs
from datetime import datetime
import json
import os
import pickle
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd


class Model:
    def __init__(self, start_date, simulation_duration, steps, min_altitude, max_altitude, 
                        n_shells, launch_function, integrator, density_model, LC, v_imp=None,
                        fragment_spreading=True, parallel_processing=False, baseline=False):
        """
        Initialize the scenario properties for the simulation model.

        Parameters:
        - start_date (str): Start date of the simulation in MM/DD/YYYY format.
        - simulation_duration (int): Duration of the simulation in days.
        - steps (int): Number of steps in the simulation.
        - min_altitude (int): Minimum altitude in meters.
        - max_altitude (int): Maximum altitude in meters.
        - n_shells (int): Number of shells in the simulation.
        - launch_function (str): Type of launch function (e.g., "Constant").
        - integrator (str): Type of integrator to use (e.g., "BDF").
        - density_model (str): Density model to use ("static_exp_dens_func").
        - LC (float): Coefficient related to launch (unitless).
        - v_imp (float): Impact velocity of objects in m/s.
        - _return (bool): Whether to return the scenario properties object.

        Returns:
        ScenarioProperties: An initialized scenario properties object.
        
        Raises:
        ValueError: If any parameters are of incorrect type or invalid value.
        """
        try:
            # Validate and convert start_date
            parsed_date = datetime.strptime(start_date, "%m/%d/%Y")

            # Validate numeric parameters
            if not isinstance(simulation_duration, int) or simulation_duration <= 0:
                raise ValueError("simulation_duration must be a positive integer.")
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError("steps must be a positive integer.")
            if not isinstance(min_altitude, int) or min_altitude < 0:
                raise ValueError("min_altitude must be a non-negative integer.")
            if not isinstance(max_altitude, int) or max_altitude <= min_altitude:
                raise ValueError("max_altitude must be greater than min_altitude.")
            if not isinstance(n_shells, int) or n_shells <= 0:
                raise ValueError("n_shells must be a positive integer.")
            if not isinstance(LC, (int, float)):
                raise ValueError("LC must be a numeric type.")

            # Create the ScenarioProperties object
            self.scenario_properties = ScenarioProperties(
                start_date=parsed_date,
                simulation_duration=simulation_duration,
                steps=steps,
                min_altitude=min_altitude,
                max_altitude=max_altitude,
                n_shells=n_shells,
                launch_function=launch_function,
                integrator=integrator,
                density_model=density_model,
                LC=LC,
                v_imp=v_imp,
                fragment_spreading=fragment_spreading,
                parallel_processing=parallel_processing,
                baseline=baseline
            )

            # Define parameters needed at Model level
            self.baseline = baseline
            
        except Exception as e:
            raise ValueError(f"An error occurred initializing the model: {str(e)}")
        
    def configure_species(self, species_json):
        """
        Configure species into Species objects from JSON.
        
        Parameters:
        - species_json (dict): JSON object containing species data.

        Returns:
        - Species: A configured species object.
        """
        try:
            species_list = Species()
            
            species_list.add_species_from_json(species_json)
            
            # Pass functions for drag and PMD
            species_list.convert_params_to_functions()

            # Create symbolic variables for the species
            self.all_symbolic_vars = species_list.create_symbolic_variables(self.scenario_properties.n_shells)

            # Pair the active species to the debris species for PMD modeling
            species_list.pair_actives_to_debris(species_list.species['active'], species_list.species['debris'])

            # Add the final species to the scenario properties to be used in the simulation
            self.scenario_properties.add_species_set(species_list.species, self.all_symbolic_vars)

            # Create Collision Pairs
            self.scenario_properties.add_collision_pairs(create_collision_pairs(self.scenario_properties))

            return species_list
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for species.")
        except Exception as e:
            raise ValueError(f"An error occurred configuring species: {str(e)}")

    def run_model(self):
        """
        Execute the simulation model using the provided scenario properties.
        
        Parameters:
        - scenario_properties (ScenarioProperties): Scenario properties object.item

        Returns:
        - Result of the simulation.
        """
        if not isinstance(self.scenario_properties, ScenarioProperties):
            raise ValueError("Invalid scenario properties provided.")
        try:

            self.scenario_properties.initial_pop_and_launch(baseline=self.baseline) # Initial population is considered but not launch
            self.scenario_properties.build_model()
            self.scenario_properties.run_model()

            # save the scenario properties to a pickle file
            with open('scenario-properties-baseline.pkl', 'wb') as f:
                pickle.dump(self.scenario_properties, f)
            
            return self.scenario_properties
        
        except Exception as e:
            raise RuntimeError(f"Failed to run model: {str(e)}")
    
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

        
    def create_plots(self):
        """
        Generates a number of plots and diposits them into the figures folder:
        """
        print('Making plots')
        scenario_properties = self.scenario_properties
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

        time_points = output.t

        n_species = scenario_properties.species_length
        shells_per_species = scenario_properties.n_shells

        # Get the species names
        species_names = scenario_properties.species_names

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
        data_reshaped = output.y.reshape(n_species, shells_per_species, n_time_points)

        # Get the x-axis labels from scenario_properties.R0_km and slice to match shells_per_species
        orbital_shell_labels = scenario_properties.R0_km[:shells_per_species]

        # Define markers for each species (reuse if more species than markers)
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

        # Directory to save the frames
        frames_dir = 'frames'
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        # Generate frames for each timestep
        for t_idx, t in enumerate(time_points):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))  # Adjust the size as needed
            
            # Plot species names that begin with 'S' on the left plot
            for species_index in range(n_species):
                if base_species_names[species_index].startswith('S'):
                    base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                    color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])  # Adjust alpha based on inverted weight
                    marker = markers[species_index % len(markers)]
                    ax1.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=species_names[species_index], color=color, marker=marker)
            
            # Setting titles and labels for the left plot
            ax1.set_title('Final Timestep: Species Starting with S')
            ax1.set_xlabel('Orbital Shell (R0_km)')
            ax1.set_ylabel('Count of Objects')
            ax1.legend(title='Species')
            
            # Plot the rest of the species on the right plot
            for species_index in range(n_species):
                if not base_species_names[species_index].startswith('S'):
                    base_color = color_map(unique_base_species.index(base_species_names[species_index]))
                    color = (base_color[0], base_color[1], base_color[2], inverted_weights[species_index])  # Adjust alpha based on inverted weight
                    marker = markers[species_index % len(markers)]
                    ax2.plot(orbital_shell_labels, data_reshaped[species_index, :, t_idx], label=species_names[species_index], color=color, marker=marker)
            
            # Setting titles and labels for the right plot
            ax2.set_title('Final Timestep: Other Species')
            ax2.set_xlabel('Orbital Shell (R0_km)')
            ax2.set_ylabel('Count of Objects')
            ax2.legend(title='Species')
            
            plt.tight_layout()
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
            plt.savefig(frame_path)
            plt.close(fig)

        # Create the GIF
        images = []
        for t_idx in range(len(time_points)):
            frame_path = os.path.join(frames_dir, f'frame_{t_idx:04d}.png')
            images.append(imageio.imread(frame_path))
        gif_path = 'figures/species_shells_evolution_side_by_side.gif'
        imageio.mimsave(gif_path, images, duration=0.5)  # Adjust the duration as needed

        # Cleanup frames
        import shutil
        shutil.rmtree(frames_dir)


if __name__ == "__main__":

    with open(os.path.join('pyssem', 'three_species.json')) as f:
        simulation_data = json.load(f)

    scenario_props = simulation_data["scenario_properties"]

    # Create an instance of the pySSEM_model with the simulation parameters
    model = Model(
        start_date=scenario_props["start_date"].split("T")[0],  # Assuming the date is in ISO format
        simulation_duration=scenario_props["simulation_duration"],
        steps=scenario_props["steps"],
        min_altitude=scenario_props["min_altitude"],
        max_altitude=scenario_props["max_altitude"],
        n_shells=scenario_props["n_shells"],
        launch_function=scenario_props["launch_function"],
        integrator=scenario_props["integrator"],
        density_model=scenario_props["density_model"],
        LC=scenario_props["LC"],
        v_imp = scenario_props.get("v_imp", None),
        fragment_spreading=scenario_props.get("fragment_spreading", True),
        parallel_processing=scenario_props.get("parallel_processing", False),
        baseline=scenario_props.get("baseline", False)
    )

    species = simulation_data["species"]

    species_list = model.configure_species(species)

    results = model.run_model()

    model.create_plots()

    ouput = model.results_to_json()
    # convert to json file
    with open('output.json', 'w') as f:
        f.write(ouput)
