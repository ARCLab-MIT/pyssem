from sympy import symbols, Matrix, pi, S, Expr, zeros
# import matplotlib.pyplot as plt
import numpy as np

class SpeciesPairClass:
    def __init__(self, species1, species2, gammas, source_sinks, scen_properties, fragment_spread_totals=None, fragsMadeDV_3d=None,
                 model_type='baseline'):
        """
        This makes the species pair class associated with a collision between species1
        and species2. It will then create equations for the collision probability modifiers
        in gamma and the species in source_sinks.

        If the symbolic argument "n_f" is passed, it will be replaced with the n_f value
        for a collision involved species1 and species2 at each dv in scen_properties.v_imp_all

        Other species will gain from a collision - e.g. debris. 

        Args:
            species1 (Species): The first species in the collision
            species2 (Species): The second species in the collision
            gammas (np.ndarray): The collision probability modifiers for each species in source_sinks.
            A scalar or a N x M matrix, where N is the number of altitude bins and M is the number of species
            with population addition/subtractions where this collision types occur. 
            source_sinks (list): A list of species that are either sources or sinks in the collision
            scen_properties (ScenarioProperties): The scenario properties object
        """
        self.fragment_spread_totals = fragment_spread_totals
        self.fragments = fragsMadeDV_3d
        if gammas.shape[1] != len(source_sinks):
            raise ValueError("Gammas and source_sinks must be the same length")
    
        # As species is a dictionary, it needs to be flatted first        
        all_species = [species for category in scen_properties.species.values() for species in category]

        self.name = f"species_pair({species1.sym_name}, {species2.sym_name})"
        self.species1 = species1
        self.species2 = species2

        meter_to_km = 1 / 1000

        # Square of impact parameter
        self.sigma = (species1.radius * meter_to_km + \
                      species2.radius * meter_to_km) ** 2

        # Scaling based on v_imp, shell volume, and object radii
        self.phi = pi * scen_properties.v_imp_all / (scen_properties.V * meter_to_km**3) * self.sigma * S(86400) * S(365.25)

        # Check if collision is catastrophic
        self.catastrophic = self.is_catastrophic(species1.mass, species2.mass, scen_properties.v_imp_all)

        # Fragment generation equations
        M1 = species1.mass
        M2 = species2.mass
        LC = scen_properties.LC
        
        self.gammas = gammas
        self.source_sinks = source_sinks
        
        if model_type == 'elliptical':
            self.eqs_sources = Matrix(scen_properties.n_shells, len(all_species), lambda i, j: 0)
            self.eqs_sinks = Matrix(scen_properties.n_shells, len(all_species), lambda i, j: 0)
        else:
            self.eqs = Matrix(scen_properties.n_shells, len(all_species), lambda i, j: 0)


        if isinstance(self.phi, (int, float, Expr)):
            phi_matrix = Matrix([self.phi] * len(gamma))
        else:
            phi_matrix = Matrix(self.phi)

        if scen_properties.fragment_spreading:
            product_sym = species1.sym.multiply_elementwise(species2.sym).T

        # Go through each gamma (which modifies collision for things like collision avoidance, or fragmentation into 
        # derelicsts, etc.) We increment the eqs matrix with the gamma * phi * species1 * species2.
        for i in range(gammas.shape[1]):
            gamma = gammas[:, i]
            eq_index = None
            for idx, spec in enumerate(all_species):
                if spec.sym_name == source_sinks[i].sym_name:
                    eq_index = idx
                    break

            if eq_index is None:
                raise ValueError(f"Equation index not found for {source_sinks[i].sym_name}")
            
            n_f = symbols(f'n_f:{scen_properties.n_shells}')

            if scen_properties.fragment_spreading:
                if i < 2:  # As first two columns are the reduction of the species in the collision (i.e -1)
                    eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                else:  # Debris generated from collision
                    try:                         
                        # Extract fragment data for this debris species from 3D matrix
                        fragsMadeDV_3d_current = fragsMadeDV_3d[:, i-2, :]  # Shape: (n_shells, n_shells)

                        # Example of fragsMadeDV_3d_current, this is for one debris species. Eeach item is from the collision source shell and where the fragments end up. 
                        # array([[6.11666667, 5.43333333, 0.65      , 0.21666667, 0.03333333],
                        #     [0.5       , 5.75      , 5.75      , 0.73333333, 0.16666667],
                        #     [0.15      , 0.63333333, 5.58333333, 5.7       , 0.71666667],
                        #     [0.05      , 0.23333333, 0.5       , 5.78333333, 5.95      ],
                        #     [0.01666667, 0.06666667, 0.23333333, 0.43333333, 5.86666667]])
                        
                        # Check if there are any non-zero fragments
                        if np.any(fragsMadeDV_3d_current > 0):
                            # Create symbolic equations for each destination shell
                            eq_list = []
                            
                            for dest_shell in range(scen_properties.n_shells):
                                # Sum contributions from all source shells to this destination shell
                                source_contributions = []
                                
                                for source_shell in range(scen_properties.n_shells):
                                    fragment_count = fragsMadeDV_3d_current[source_shell, dest_shell]
                                    
                                    if fragment_count > 0:  # Only include non-zero contributions
                                        # Create symbolic term: collision_rate * fragment_count
                                        # This represents: species1[source] * species2[source] * fragments_from_source_to_dest
                                        collision_term = species1.sym[source_shell] * species2.sym[source_shell] * fragment_count
                                        source_contributions.append(collision_term)
                                
                                # Sum all source contributions for this destination shell
                                if source_contributions:
                                    eq_list.append(sum(source_contributions))
                                else:
                                    eq_list.append(0)
                            
                            # Create the final equation matrix
                            eq = Matrix(eq_list)
                            
                            # Multiply by gammas and phi to get the final collision equations
                            eq = -gammas[:, 0].multiply_elementwise(phi_matrix).multiply_elementwise(eq)
                        else:
                            # No fragments, use simple collision equation
                            eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)

                    except Exception as e:
                        if np.any(fragsMadeDV_3d == 0):
                            eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                            continue
                        # print(f"Error in creating debris matrix: {e}")
                
                self.eqs[:, eq_index] = self.eqs[:, eq_index] + eq
            elif model_type == 'elliptical':
                eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                if i < 2:
                    # These are the sinks
                    self.eqs_sinks[:, eq_index] = self.eqs_sinks[:, eq_index] + eq
                else:
                    # Sources:
                    self.eqs_sources[:, eq_index] = self.eqs_sources[:, eq_index] + eq

            else:
                eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                
                self.eqs[:, eq_index] = self.eqs[:, eq_index] + eq  
            

    def is_catastrophic(self, mass1, mass2, vels):
        """
        Determines if a collision is catastropic or non-catastrophic by calculating the 
        relative kinetic energy. If the energy is greater than 40 J/g, the collision is
        catastrophic from Johnson et al. 2001 (Collision Section)

        Args:
            mass1 (float): mass of species 1, kg
            mass2 (float): mass of species 2, kg
            vels (np.ndarray): array of the relative velocities (km/s) for each shell
        
        Returns:
            shell-wise list of bools (true if catastrophic, false if not catastrophic)
        """

        if mass1 <= mass2:
            smaller_mass = mass1
        else:
            smaller_mass = mass2
        
        smaller_mass_g = smaller_mass * (1000) # kg to g
        energy = [0.5 * smaller_mass * (v)**2 for v in vels] # Need to also convert km/s to m/s
        is_catastrophic = [True if e/smaller_mass_g > 40 else False for e in energy]

        return is_catastrophic
    
    def _map_fragments_to_shells_corrected(self, fragment_distribution, n_shells, collision_shell):
        """
        Map fragment distribution from velocity bins to shell indices using corrected binning.
        The collision shell is the center, and fragments spread above and below it.
        
        Args:
            fragment_distribution: 1D array of fragment counts per velocity bin
            n_shells: Number of shells
            collision_shell: The shell where the collision occurs (center of distribution)
            
        Returns:
            2D array with shape (n_shells, n_shells) representing fragment distribution
        """
        n_velocity_bins = len(fragment_distribution)
        n_shells_hist = (n_velocity_bins - 1) // 2  # Middle bin index (collision shell)
        
        # Create shell mapping for this specific collision
        shell_totals = np.zeros(n_shells)
        
        for hist_bin in range(n_velocity_bins):
            # Map velocity bin to shell index relative to collision shell
            shell_idx = collision_shell + (hist_bin - n_shells_hist)
            
            # Only include shells that are within bounds
            if 0 <= shell_idx < n_shells:
                shell_totals[shell_idx] = fragment_distribution[hist_bin]
        
        # Create 2D matrix for the species pair system
        # Each row represents a source shell, each column represents a destination shell
        fragsMade2D = np.zeros((n_shells, n_shells))
        
        # For this collision at collision_shell, distribute fragments to all shells
        # based on the shell_totals calculated above
        fragsMade2D[collision_shell, :] = shell_totals
        
        return fragsMade2D
