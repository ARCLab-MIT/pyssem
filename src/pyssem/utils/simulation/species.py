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
            species_props_copy['sym_name'] = f"{species_properties['sym_name']}_{i}"

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
        # # Remember now an object not a json, so we need to use the class properties
        multi_species_list.sort(key=lambda x: x.mass)

        # I am not entirely sure what this does... Removing for now
        # for i, species in enumerate(multi_species_list):
        #     if i == len(self.species) - 1:
        #         species.mass_lb = 0.5 * (multi_species_list[i - 1].mass + species.mass)
        #     else:
        #         species.mass_ub = 0.5 * (species.mass + multi_species_list[i + 1].mass)
        #         species.mass_lb = 0.5 * (multi_species_list[i - 1].mass + species.mass)

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
            # Multiple masses means it needs to be copied and passed correctly
            if isinstance(properties['mass'], list) and len(properties['mass']) > 1:
                multiple_species = self.add_multi_property_species(properties)
                self.species.extend(multiple_species)
            else:
                temp = SpeciesProperties(properties)
                self.species.append(temp)

        return self.species