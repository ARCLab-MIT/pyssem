from sympy import symbols, Matrix, pi, S, Expr

class SpeciesPairClass:
    def __init__(self, species1, species2, gammas, source_sinks, scen_properties):
        """
        This makes the species pair class associated with a collision between species1
        and species2. It will then create equations for the collision probability modifiers
        in gamma and the species in source_sinks.

        If the symbolic argument "n_f" is passed, it will be replaced with the n_f value
        for a collision involved species1 and species2 at each dv in scen_properties.v_imp2

        Args:
            species1 (Species): The first species in the collision
            species2 (Species): The second species in the collision
            gammas (np.ndarray): The collision probability modifiers for each species in source_sinks.
            A scalar or a N x M matrix, where N is the number of altitude bins and M is the number of species
            with population addition/subtractoins where this collision types occur. 
            source_sinks (list): A list of species that are either sources or sinks in the collision
            scen_properties (ScenarioProperties): The scenario properties object
        """
        if gammas.shape[1] != len(source_sinks):
            raise ValueError("Gammas and source_sinks must be the same length")
        
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
        n_f_catastrophic = S(0.1) * LC**(-S(1.71)) * (M1 + M2)**(S(0.75)) * Matrix.ones(scen_properties.v_imp2.shape[0], 1)
        n_f_damaging = S(0.1) * LC**(-S(1.71)) * (min(M1, M2) * scen_properties.v_imp2**2)**(S(0.75))

        if self.catastrophic:
            self.nf = n_f_catastrophic.transpose()
        else:
            self.nf = n_f_damaging.transpose()

        self.gammas = gammas
        self.source_sinks = source_sinks
        self.eqs = Matrix(scen_properties.n_shells, len(scen_properties.species), lambda i, j: 0)

        if isinstance(self.phi, (int, float, Expr)):
            phi_matrix = Matrix([self.phi] * len(gamma))
        else:
            phi_matrix = Matrix(self.phi)  # Assuming self.phi is already a list or a column vector

        # Go through each gamma (which modifies collision for things like collision avoidance, or fragmentation into 
        # derelicsts, etc.) We increment the eqs matrix with the gamma * phi * species1 * species2.
        for i in range(gammas.shape[1]):
            gamma = gammas[:, i]
            eq_index = None
            for idx, spec in enumerate(scen_properties.species):
                if spec.sym_name == source_sinks[i].sym_name:
                    eq_index = idx
                    break

            if eq_index is None:
                raise ValueError(f"Equation index not found for {source_sinks[i].sym_name}")

            n_f = symbols(f'n_f:{scen_properties.n_shells}')

            eq = gamma.multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)

            for j, val in enumerate(self.nf):
                eq = eq.subs(n_f[j], val)

            self.eqs[:, eq_index] = self.eqs[:, eq_index] + eq  

    def is_catastrophic(self, mass1, mass2, vels):
        """
        Determiins if a collision is catastropic or non-catastrophic.
        40 j/g threshold from Johnson et al. 2001

        Args:
            mass1 (float): mass of species 1, kg
            mass2 (float): mass of species 2, kg
            vels (np.ndarray): array of the relative velocities
        
        Returns:
            shell-wise list of bools (true if catastrophic, false if not catastrophic)
        """

        if mass1 <= mass2:
            smaller_mass = mass1
        else:
            smaller_mass = mass2
        
        smaller_mass_g = smaller_mass * (1000) # kg to g
        energy = [0.5 * smaller_mass_g * v**2 for v in vels]
        is_catastrophic = [True if e > 40 else False for e in energy]
        return is_catastrophic

    def calculate_equations(self, gammas, source_sinks, scen_properties):
        """
        Create the symbolic equations for the collision probability modifiers in gamma
        and the species in source_sinks.

        Args:
            gammas (sympy array): The collision probability modifiers for each species in source_sinks.
            A scalar or a N x M matrix, where N is the number of altitude bins and M is the number of species
            with population addition/subtractoins where this collision types occur. 
            source_sinks (list): A list of species that are either sources or sinks in the collision
            scen_properties (ScenarioProperties): The scenario properties object
        
        Returns:
            sympy array: An array of symbolic equations for the collision probability modifiers in gamma
            and the species in source_sinks.
        """
        # eqs = zeros(scen_properties.n_shells, len(scen_properties.species))
        eqs = Matrix(scen_properties.n_shells, len(scen_properties.species), lambda i, j: 0)

        # Define symbols for each shell
        n_f_symbols = symbols(f'n_f:{scen_properties.n_shells}')

        for i, gamma in enumerate(gammas.transpose()):
            eq_index = next((index for index, spec in enumerate(scen_properties.species) if spec.sym_name == source_sinks[i].sym_name), None)
            if eq_index is None:
                raise ValueError(f"Equation index not found for {source_sinks[i].sym_name}")

            # Calculate the equation assuming self.phi, self.species1.sym, and self.species2.sym are SymPy expressions
            eq = gamma * self.phi * self.species1.sym * self.species2.sym

            # Perform substitution for each n_f symbol with its corresponding value in self.nf
            for n_f, value in zip(n_f_symbols, self.nf):
                eq = eq.subs(n_f, value)

            # # Since SymPy matrices are immutable, use row_insert and col_insert for updating 'eqs'
            # # First, construct a column matrix for the updated equations
            # updated_col = eqs.col(eq_index) + eq

            # # Insert the updated column back into 'eqs'
            # if eq_index > 0:
            #     eqs = eqs[:, :eq_index].row_join(updated_col)
            # else:
            #     eqs = updated_col

            # if eq_index < eqs.cols - 1:
            #     eqs = eqs.row_join(eqs[:, eq_index + 1:])
                
            # Update 'eqs' directly since SymPy Matrices are mutable
            for j in range(eqs.rows):
                eqs[j, eq_index] += eq[j]

        return eqs
