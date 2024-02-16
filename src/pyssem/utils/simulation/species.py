import json
from math import pi
from utils.pmd.pmd import *
from utils.drag.drag import *
from utils.launch.launch import *

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
        self.beta = None  # m^2/kg
        self.B = None # 
        self.density_filepath = None  # For drag

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

        # If a JSON string is provided, parse it and update the properties
        if properties_json:
            for key, value in properties_json.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Property {key} not found in SpeciesProperties class.")

class Species:
    """
    This is a collection of SpeciesProperties objects. It is used to store the properties of multiple species in a
    scene. 
    """
    species = []

    def __init__(self) -> None:
        pass

    def add_species(self, species_properties: SpeciesProperties) -> None:
        self.species.append(species_properties)
    
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
        trackable_radius_threshold = 0.05  # m

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

            # Handle derived properties
            if 'radius' in species_props_copy and 'A' not in species_props_copy:
                species_props_copy['A'] = np.pi * species_props_copy['radius'] ** 2
            if species_props_copy['A'] == "Calculated based on radius":
                species_props_copy['A'] = np.pi * species_props_copy['radius'] ** 2
            if 'A' in species_props_copy and 'amr' not in species_props_copy:
                species_props_copy['amr'] = species_props_copy['A'] / species_props_copy['mass']
            if 'Cd' in species_props_copy and 'amr' in species_props_copy and 'beta' not in species_props_copy:
                species_props_copy['beta'] = species_props_copy['Cd'] * species_props_copy['amr']
            if 'radius' in species_props_copy and 'trackable' not in species_props_copy:
                species_props_copy['trackable'] = species_props_copy['radius'] >= trackable_radius_threshold

            # Create the species instance and append it to the species list
            species_instance = SpeciesProperties(species_props_copy)
            multi_species_list.append(species_instance)

        # # Sort the species list by mass and set upper and lower bounds for mass bins
        multi_species_list.sort(key=lambda x: x.mass) # sorts by mass

        # Each species has a mass, the code will need to understand what the lower and upper bounds are for each species.
        # you need to able to define the upper and lower bounds for the average of that sub-species 

        # Add to global species list
        print(f"Splitting species {species_properties['sym_name']} into {num_species} species with masses {species_properties['mass']}.")
        return multi_species_list
        
    def add_species_from_json(self, species_json: json) -> None:
        """
        Adds a set of species to the simulation based on a predefined list. 
        if a species name is not in the list, it will not be added. A warning should be thrown. 

        :param json_string: _description_
        :type json_string: json
        :return: _description_
        :rtype: None
        """
        # loop through the json and pass create and instance of species properties for each species
        for species_name, properties in species_json.items():         
            # This will add pass the Json Species to the SpeciesProperties class. 
            # If the mass is a list, then we need to create multiple species with the same properties
            if properties['active'] == True:
                if isinstance(properties['mass'], list) and len(properties['mass']) > 1:
                    multiple_species = self.add_multi_property_species(properties)
                    self.species.extend(multiple_species)
                else:
                    self.species.append(SpeciesProperties(properties))

        # Finally, all species that have been created will be added to the debris list for PMD pairing.
        for species_name, properties in species_json.items():
            # This will add all of the other species with multiple masses to the debris list for PMD Pairing. 
            if properties['active'] == False:
                
                # Loop through the current set of species and get the mass and radius and add it to the debris list
                active_masses = []
                active_radii = []

                for species in self.species:
                    if species.active:
                        active_masses.append(species.mass)
                        active_radii.append(species.radius)

                # Get the raw json of the debris species
                debris_species = properties
                if isinstance(properties['mass'], list):
                    debris_species['mass'].extend(active_masses)
                    debris_species['radius'].extend(active_radii)
                else:
                    debris_species['mass'] = [debris_species['mass']] # convert to list if not one
                    debris_species['mass'].extend(active_masses)

                    debris_species['radius'] = [debris_species['radius']] # convert to list if not one
                    debris_species['radius'].extend(active_radii)
             
                debris_species_object = self.add_multi_property_species(debris_species)
                self.species.extend(debris_species_object)

        return self.species
    
    def apply_launch_rates(self, n_shells: int):
        """
        This will loop through each of the species, if launch rate is constant, it will create a launch array. 
        This should be edited in the future to be more dynamic for the user. 

        """
        for species in self.species:
            if species.launch_func == "launch_func_constant":
                # probably should remove for the constant launch_rate 
                # they should provide a scalar or a vector, this would have to be called if you want a time varying function for the ODE
                # if you provide something that is non-constant, then a user should have to provide a vector 
                # time not alt
                species.lambda_constant = [20 for _ in range(n_shells)]
                # Direct copy from MATLAB
                #species.lambda_constant = (500 * np.random.rand(scen_properties.N_shell, 1)).tolist()

    def create_symbolic_variables(self, n_shells: int):
        """
        This will create the symbolic variables for each of the species. 
        """
        for species in self.species:
            species.sym = symbols([f'{species.sym_name}_{i+1}' for i in range(n_shells)])

    
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

        # Update the species list 
        self.species = []
        self.species.extend(active_species)
        self.species.extend(debris_species)
        
        