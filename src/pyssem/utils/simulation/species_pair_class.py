import numpy as np
import sympy as sp

class SpeciesPairClass:
    def __init__(self, species1, species2, gammas, source_sinks, scen_properties):
        if gammas.shape[1] != len(source_sinks):
            raise ValueError("Gammas and source_sinks must be the same length")
        
        self.name = f"species_pair({species1.species_properties['sym_name']}, {species2.species_properties['sym_name']})"
        self.species1 = species1
        self.species2 = species2

        meter_to_km = 1 / 1000

        # Square of impact parameter
        self.sigma = (species1.species_properties['radius'] * meter_to_km + \
                      species2.species_properties['radius'] * meter_to_km) ** 2

        # Scaling based on v_imp, shell volume, and object radii
        self.phi = np.pi * scen_properties['v_imp2'] / (scen_properties['V'] * meter_to_km**3) * self.sigma * 86400 * 365.25

        # Check if collision is catastrophic
        self.catastrophic = self.is_catastrophic(species1['mass'], species2['mass'], scen_properties)

        M1 = species1.species_properties['mass']
        M2 = species2.species_properties['mass']
        LC = scen_properties['LC']
        n_f_catastrophic = lambda M1, M2: 0.1 * LC**(-1.71) * (M1 + M2)**0.75 * np.ones_like(scen_properties['v_imp2'])
        n_f_damaging = lambda M1, M2: 0.1 * LC**(-1.71) * (np.minimum(M1, M2) * scen_properties['v_imp2']**2)**0.75

        if self.catastrophic:
            self.nf = n_f_catastrophic(M1, M2)
        else:
            self.nf = n_f_damaging(M1, M2)

        self.gammas = gammas
        self.source_sinks = source_sinks
        self.eqs = self.calculate_equations(gammas, source_sinks, scen_properties.vels)

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
        energy = 0.5 * smaller_mass_g * vels**2
        if energy > 40:
            return True
        else:
            return False
        
    def is_catastrophic_species(self, species1, species2, scen_properties):
            mass1 = species1.species_properties['mass']
            mass2 = species2.species_properties['mass']
            vels = scen_properties['v_imp2']
            return is_catastrophic(mass1, mass2, vels)  # Assuming is_catastrophic is a separate function


    def calculate_equations(self, gammas, source_sinks, scen_properties):
        eqs = np.zeros((scen_properties['N_shell'], len(scen_properties['species'])), dtype=object)  # Assuming symbolic equations
        for i, gamma in enumerate(gammas.T):
            # Find index for species in source_sinks
            eq_index = next((index for index, spec in enumerate(scen_properties['species']) if spec.species_properties['sym_name'] == source_sinks[i].species_properties['sym_name']), None)
            if eq_index is None:
                raise ValueError(f"Equation index not found for {source_sinks[i].species_properties['sym_name']}")

            n_f = sp.symbols(f'n_f:{scen_properties["N_shell"]}')
            try:
                eq = gamma * self.phi * self.species1.species_properties['sym'] * self.species2.species_properties['sym']
            except Exception as e:
                print('Error in calculating equations:', e)

            eq = sp.subs(eq, n_f, self.nf)

            eqs[:, eq_index] += eq

        return eqs
