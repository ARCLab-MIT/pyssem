import json
from sympy import Matrix
from ..pmd.pmd import *
from ..drag.drag import *
from ..launch.launch import *

class SpeciesProperties:
    def __init__(self, properties_json=None):
        
        # Set default values
        self.sym_name = None # Name of Species Object
        self.sym = None # symbolic variables
        self.Cd = None # drag coefficient
        self.mass = None # in kg
        self.mass_lb = 0.00141372  # Lower bound of mass class for object binning (inclusive), 1 cm diameter Al sphere
        self.mass_ub = 100000  # Upper bound of mass class for object binning (exclusive, except for top mass bin, which is inclusive)
        self.radius = None 
        self.A = None  # m^2
        self.amr = None  # m^2/kg
        self.beta = None  # m^2/kg, ballistic coefficient (Cd * A / mass)
        self.B = None # 

        # Orbit Properties
        self.slotted = False  # bool
        self.slotting_effectiveness = 0  # double [0, 1] where 0 is no col reduction, 1 is perfect col reduction
        self.disposal_altitude = 0  # km

        # Capabilities
        self.drag_effected = False  # bool
        self.active = False  # bool
        self.maneuverable = False  # bool
        self.trackable = False  # bool
        self.deltat = None  # years
        self.Pm = None  # double [0, 1], 0 = no Pmd, 1 = full Pm
        self.alpha = None  # failure rate of collision avoidance vs inactive trackable objects [0 = perfect, 1 = none]
        self.alpha_active = None  # failure rate of collision avoidance vs active maneuverable objects [0 = perfect, 1 = none]
        self.RBflag = None  # bool 1 = rocket body, 0 = not rocket body

        # Orbit Raising (Not implemented)
        self.orbit_raising = False  # Bool
        self.insert_altitude = None  # altitude -> semimajor axis
        self.onstation_altitude = None
        self.epsilon = None  # lt_mag
        self.e_mean = None

        # For Launch Functions
        self.lambda_multiplier = None  # only for launch_func_fixed_multiplier
        self.lambda_funs = None  # only for launch_func_gauss
        self.lambda_constant = None  # only for launch_func_constant
        self.lambda_python_args = None

        # For Derelicts
        self.pmd_linked_species = []
        self.pmd_linked_multiplier = None  # Used when multiple shells dispose to one orbit.

        # References
        self.eq_idxs = None  # indexes of species states in X

        # For knowing when to recalculate custom launch functions.
        self.last_calc_x = float('nan')
        self.last_calc_t = float('nan')
        self.last_lambda_dot = float('nan')

        # For looped model
        self.saved_model_path = None
        self.t_plan_max = -1
        self.t_plan_period = 1
        self.prev_prop_results = None

        # Functions
        self.launch_func = None
        self.pmd_func = None
        self.drag_func = None

        self.trackable_radius_threshold = 0.05  # m

        # If a JSON string is provided, parse it and update the properties
        if properties_json:
            for key, value in properties_json.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Property {key} not found in SpeciesProperties class.")
                    # Post mission disposal functions
            
            # Handle derived properties
            if self.radius is not None and self.A is None:
                self.A = np.pi * self.radius ** 2
            if self.A == "Calculated based on radius":
                self.A = np.pi * self.radius ** 2
            if self.A is not None and self.amr is None:
                self.amr = self.A / self.mass
            if self.Cd is not None and self.amr is not None and self.beta is None:
                self.beta = self.Cd * self.amr

            if self.radius is not None and hasattr(self, 'trackable'):
                self.trackable = self.radius >= self.trackable_radius_threshold
                        
            # Ballistic Coefficient
            if hasattr(self, 'Cd') and hasattr(self, 'amr'):
                self.beta = self.Cd * self.amr
            else:
                self.beta = None
            if self.beta is None:
                print(f"Warning: No ballistic coefficient provided for species {self.sym_name}.")


class Species:
    """
    This is a collection of SpeciesProperties objects. It is used to store the properties of multiple species in a
    scene. 

    It will also be used to create
    """
    species = None

    def __init__(self) -> None:
        self.species = {'active': [], 'debris': [], 'rocket_body': []}
    
    def add_multi_property_species(self, species_properties):
        """
        This is a more flexible way to add species to the simulation, unlike the templated files.

        This class an input structure to create a multi_property_species class with the species_list propeorty
        with a set of species with different properties. 

        If multiple masses are provided, buy other values are provided as single values, then radiu, A, area to mass ratio, 
        and beta will be scaled based on spherical assumption. Trackabiolity will be set based on scaled radius relative
        to the trackable threshold.

        :return: _description_
        :rtype: _type_
        """

        if len(species_properties['mass']) == 1:
            raise ValueError("Multi-property species must have multiple masses.")
        
        if "_" in species_properties['sym_name']:
            raise ValueError("Species names cannot contain underscores.")

        multi_species_list = []
        num_species = len(species_properties['mass'])

        for i in range(num_species):
            species_props_copy = species_properties.copy()
            species_props_copy['mass'] = species_properties['mass'][i] if isinstance(species_properties['mass'], list) else species_properties['mass']
            species_props_copy['sym_name'] = f"{species_properties['sym_name']}_{species_properties['mass'][i]}kg"

            for field in species_properties:
                if field == "sym_name":
                    continue

                field_value = species_properties[field]
                if isinstance(field_value, list):
                    if len(field_value) == num_species:
                        species_props_copy[field] = field_value[i]
                    elif len(field_value) == 1:
                        species_props_copy[field] = field_value[0]
                    else:
                        raise ValueError(f"The field '{field}' list length does not match the number of species and is not a single-element list.")
                else:
                    species_props_copy[field] = field_value

                 
            # Create the species instance and append it to the species list
            species_instance = SpeciesProperties(species_props_copy)
            multi_species_list.append(species_instance)

        # # Sort the species list by mass and set upper and lower bounds for mass bins
        multi_species_list.sort(key=lambda x: x.mass) # sorts by mass

        # Update the mass_lb and mass_ub for each species
        # This will be based on the mass of the species and the mass of the species before and after it in the list
        for i in range(len(multi_species_list)):
            if i == 0:  # First element (lowest mass)
                multi_species_list[i].mass_ub = 0.5 * (multi_species_list[i].mass + multi_species_list[i + 1].mass)
            elif i == len(multi_species_list) - 1:  # Last element (highest mass)
                multi_species_list[i].mass_lb = 0.5 * (multi_species_list[i - 1].mass + multi_species_list[i].mass)
            else:  # Elements in between
                multi_species_list[i].mass_ub = 0.5 * (multi_species_list[i].mass + multi_species_list[i + 1].mass)
                multi_species_list[i].mass_lb = 0.5 * (multi_species_list[i - 1].mass + multi_species_list[i].mass)


        # Add to global species list
        print(f"Splitting species {species_properties['sym_name']} into {num_species} species with masses {species_properties['mass']}.")
        return multi_species_list
        
    def add_species_from_json(self, species_json: json) -> None:
        """
        Creates a dictionary of the species properties from a json file.
        It will create a dictionary of 'active' 'debris' and 'rocket_body' species.

        :param json_string: _description_
        :type json_string: json
        :return: _description_
        :rtype: None
        """

        # Loop through the json and pass create and instance of species properties for each species

        if isinstance(species_json, dict):
            # convert to list
            raise ValueError("Species JSON must be a list.")
            
        for properties in species_json:      
            species_object = None

            # Active
            if properties.get('active', False):
                if isinstance(properties['mass'], list) and len(properties['mass']) > 1:
                    multiple_species = self.add_multi_property_species(properties)
                    self.species['active'].extend(multiple_species)
                else:
                    species_object = SpeciesProperties(properties)
                    self.species['active'].append(species_object)

            # Debris and Rocket Body
            rb_flag = properties.get('RBflag', 0)  # Defaults to 0 if 'RBflag' is not found
            if rb_flag == 1:
                if isinstance(properties['mass'], list) and len(properties['mass']) > 1:
                    multiple_species = self.add_multi_property_species(properties)
                    self.species['rocket_body'].extend(multiple_species)
                else:
                    species_object = SpeciesProperties(properties)
                    self.species['rocket_body'].append(species_object)

        # The debris species (N) will need to be created for the active species. 
        # Therefore you have to re-loop after the active satellites have been created.
        active_masses = [i.mass for i in self.species['active']]
        active_radii = [i.radius for i in self.species['active']]

        for item in species_json:
            properties = item  # Now item itself is the properties dictionary

            if not properties.get('active', True):
                if properties.get('RBflag', 0) == 0:
                    debris_species = properties
                    if isinstance(debris_species.get('mass', []), list) and len(debris_species['mass']) > 1:
                        debris_species['mass'].extend(active_masses)
                        debris_species['radius'].extend(active_radii)
                    else:  # Just take the one value
                        if active_masses:
                            # Ensure debris_species['mass'] is a list
                            current_mass = debris_species.get('mass', [])
                            if not isinstance(current_mass, list):
                                current_mass = [current_mass]
                            debris_species['mass'] = current_mass + [active_masses[0]]
                        if active_radii:
                            # Ensure debris_species['radius'] is a list
                            current_radius = debris_species.get('radius', [])
                            if not isinstance(current_radius, list):
                                current_radius = [current_radius]
                            debris_species['radius'] = current_radius + [active_radii[0]]

                    # Create species objects for non-active species
                    if isinstance(properties.get('mass', []), list) and len(properties['mass']) > 1:
                        multiple_species = self.add_multi_property_species(properties)
                        self.species['debris'].extend(multiple_species)
                    else:
                        species_object = SpeciesProperties(properties)
                        self.species['debris'].append(species_object)


        print(f"Added {len(self.species['active'])} active species, {len(self.species['debris'])} debris species, and {len(self.species['rocket_body'])} rocket body species to the simulation.")
        
        # Pass any required functions
        return self.species
    
    def convert_params_to_functions(self):
        """
        Pass functions that are in string format to actual functions.
        """

        for species_group in self.species.values():
            for species in species_group:
                if species.pmd_func == "pmd_func_derelict":
                    species.pmd_func = pmd_func_derelict
                elif species.pmd_func == "pmd_func_sat":
                    species.pmd_func = pmd_func_sat
                else:
                    species.pmd_func = pmd_func_none

                if species.drag_func == "drag_func_none":
                    species.drag_func = drag_func_none
                else:                 
                    species.drag_func = drag_func_exp

                # if species.launch_func == "launch_func_constant":
                #     species.launch_func = launch_func_constant
                # else:
                #     species.launch_func = launch_func_lambda_fun   
                species.launch_func = launch_func_lambda_fun 

        return

    def create_symbolic_variables(self, n_shells: int):
        """
        This will create the symbolic variables for each of the species. 

        Args:
            n_shells (int): Number of shells in the simulation.
        
        Returns:
            list: List of symbolic variables for all species.
        """
        all_species_symbols = []
        for species_group in self.species.values():
            for species in species_group:
                # if a sym_name contains '.' then it will be replaced with 'p'
                species.sym_name.replace('.', 'p') # P means decimal point
                species.sym = Matrix(symbols([f'{species.sym_name}_{i+1}' for i in range(n_shells)]))
                all_species_symbols.extend(species.sym)
        
        return all_species_symbols


    
    def pair_actives_to_debris(self, active_species, debris_species):
        """
        Pairs all active species to debris species for PMD modeling.

        Args:
            active_species (list): List of active species objects.
            debris_species (list): List of debris species objects.
        """
        # Collect active species and their names
        linked_spec_names = [item.sym_name for item in active_species]
        print("Pairing the following active species to debris classes for PMD modeling...")
        print(linked_spec_names)

        # Assign matching debris increase for a species due to failed PMD
        for active_spec in active_species:
            found_mass_match_debris = False
            spec_mass = active_spec.mass

            for deb_spec in debris_species:
                if spec_mass == deb_spec.mass:
                    deb_spec.pmd_func = pmd_func_derelict
                    deb_spec.pmd_linked_species = []                
                    deb_spec.pmd_linked_species.append(active_spec)
                    print(f"Matched species {active_spec.sym_name} to debris species {deb_spec.sym_name}.")
                    found_mass_match_debris = True

            if not found_mass_match_debris:
                print(f"No matching mass debris species found for species {active_spec.sym_name} with mass {spec_mass}.")

        # Display information about linked active species for each debris species
        for deb_spec in debris_species:
            linked_spec_names = [spec.sym_name for spec in deb_spec.pmd_linked_species if not None]
            print(f"    Name: {deb_spec.sym_name}")
            print(f"    pmd_linked_species: {linked_spec_names}")

        # Find the species in self.species and update the pmd_linked_species property
        for deb_spec in debris_species:
            for spec in self.species['active']:
                if spec.sym_name == deb_spec.sym_name:
                    spec.pmd_linked_species = deb_spec.pmd_linked_species
    