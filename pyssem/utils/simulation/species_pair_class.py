from sympy import symbols, Matrix, pi, S, Expr, zeros, transpose
import matplotlib.pyplot as plt
import numpy as np

class SpeciesPairClass:
    def __init__(self, species1, species2, gammas, source_sinks, scen_properties, fragsMadeDV=None):
        """
        This makes the species pair class associated with a collision between species1
        and species2. It will then create equations for the collision probability modifiers
        in gamma and the species in source_sinks.

        If the symbolic argument "n_f" is passed, it will be replaced with the n_f value
        for a collision involved species1 and species2 at each dv in scen_properties.v_imp2

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
        self.phi = pi * scen_properties.v_imp2 / (scen_properties.V * meter_to_km**3) * self.sigma * S(86400) * S(365.25)

        # Check if collision is catastrophic
        self.catastrophic = self.is_catastrophic(species1.mass, species2.mass, scen_properties.v_imp2)

        # Fragment generation equations
        M1 = species1.mass
        M2 = species2.mass
        LC = scen_properties.LC
        
        nf = zeros(len(scen_properties.v_imp2), 1)

        for i, dv in enumerate(scen_properties.v_imp2):
            if self.catastrophic[i]:
                n_f_catastrophic = 0.1 * LC**(-S(1.71)) * (M1 + M2)**(S(0.75))
                nf[i] = n_f_catastrophic
            else:
                n_f_damaging = 0.1 * LC**(-S(1.71)) * (min(M1, M2) * dv**2)**(S(0.75))
                nf[i] = n_f_damaging

        self.nf = nf.transpose() 
            
        self.gammas = gammas
        self.source_sinks = source_sinks
        self.eqs = Matrix(scen_properties.n_shells, len(all_species), lambda i, j: 0)

        if isinstance(self.phi, (int, float, Expr)):
            phi_matrix = Matrix([self.phi] * len(gamma))
        else:
            phi_matrix = Matrix(self.phi)  # Assuming self.phi is already a list or a column vector

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
            
            if scen_properties.collision_spread:
                if i < 2:  # As first two columns are the reduction of the species in the collision (i.e -1)
                    eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                else:  # Debris generated from collision
                    try: 
                        fragsMadeDVcurrentDeb = np.array(fragsMadeDV[:, i-2]) # this is 1, 80

                        # Create the 2D fragment matrix with circular shifts
                        fragsMade2D_list = [np.roll(fragsMadeDVcurrentDeb, shift) for shift in range(scen_properties.n_shells)] # len 40
                        fragsMade2D = np.vstack(fragsMade2D_list) # 40, 1
                        fragsMade2D = fragsMade2D[:scen_properties.n_shells, :scen_properties.n_shells] # 40, 1
                        fragsMade2D_sym = Matrix(fragsMade2D) # 40, 1

                        # Create species product matrix
                        product_sym = species1.sym.multiply_elementwise(species2.sym).T
                        rep_mat_sym = Matrix.vstack(*[product_sym for _ in range(scen_properties.n_shells)])

                        # Perform element-wise multiplication
                        temp = fragsMade2D_sym.multiply_elementwise(rep_mat_sym)
                        
                        # Sum the columns of the multiplied_matrix
                        sum_matrix = Matrix([sum(temp[:, col]) for col in range(temp.shape[1])])

                        # Multiply gammas, phi, and the sum_matrix element-wise
                        eq = -gamma.multiply_elementwise(phi_matrix).multiply_elementwise(sum_matrix)
                    except Exception as e:
                        print(f"Error in creating debris matrix: {e}")
            else:
                eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                
            for j, val in enumerate(self.nf):
                eq = eq.subs(n_f[j], val)

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
        energy = [0.5 * smaller_mass * (v * 1000)**2 for v in vels] # Need to also convert km/s to m/s
        is_catastrophic = [True if e/smaller_mass_g > 40 else False for e in energy]

        return is_catastrophic
