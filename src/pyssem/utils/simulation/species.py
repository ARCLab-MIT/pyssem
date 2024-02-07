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
            properties_dict = json.loads(properties_json)
            for key, value in properties_dict.items():
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

    def add_single_species(self, json_string: json) -> None:
        """
        Adds a single species to the simulation based on a JSON string. 

        :param json_string: _description_
        :type json_string: json
        :return: _description_
        :rtype: None
        """
        # need to complete
        pass

    def add_species_from_template(self, species_names: ["Su", "S", "sns", "N", "B"] = ["Su", "S", "sns", "N", "B"]):
        """
        Adds a set of species to the simulation based on a predefined list. 
        if a species name is not in the list, it will not be added. A warning should be thrown. 
        Su = unslotted satellites, S = slotted satellites, sns = Sns 3U cubesat with no station_keeping, N = debris

        :param species_names: _description_, defaults to ["Su", "S", "sns", "N", "B"]
        :type species_names: Su&quot;, &quot;S&quot;, &quot;sns&quot;, &quot;N&quot;, &quot;B&quot;], optional
        :return: _description_
        :rtype: List of SpeciesProperties
        """
        
        # Su - Unslotted Satellites
        if "Su" in species_names:
            su_properties = SpeciesProperties()  # Assuming you have a SpeciesProperties class

            # Default values
            su_properties.sym_name = "Su"
            su_properties.Cd = 2.2
            su_properties.mass = [260, 473]
            su_properties.A = [1.6652, 13.5615]
            su_properties.amr = [a/m for a, m in zip(su_properties.A, su_properties.mass)]
            su_properties.beta = [su_properties.Cd * amr for amr in su_properties.amr]

            # Orbit Properties
            su_properties.slotting_effectiveness = 1.0

            # Capabilities
            su_properties.drag_effected = False
            su_properties.active = True
            su_properties.maneuverable = True
            su_properties.trackable = True
            su_properties.deltat = [8, 8]
            su_properties.Pm = 0.65
            su_properties.alpha = 1e-5
            su_properties.alpha_active = 1e-5
            su_properties.RBflag = 0

            # Functions
            su_properties.launch_func = launch_func_constant
            su_properties.pmd_func = pmd_func_sat
            su_properties.drag_func = drag_func_exp

            # Append Species
            self.species.append(su_properties)
       
        # S - Slotted Satellites
        if "S" in species_names:
            s_properties = SpeciesProperties()

            # Default values
            s_properties.sym_name = "S"
            s_properties.Cd = 2.2
            s_properties.mass = [1250] #[1250, 750, 148]
            s_properties.radius = [4, 2, 0.5]
            s_properties.A = sum([pi*_**2 for _ in s_properties.radius]) # m^2
            s_properties.amr = s_properties.A/s_properties.mass[0] # m^2/kg
            s_properties.beta = s_properties.Cd * s_properties.amr # ballistic coefficient

            # Orbit Properties
            s_properties.slotted = True
            s_properties.slotting_effectiveness = 1.0

            # Capabilities
            s_properties.drag_effected = False
            s_properties.active = True
            s_properties.maneuverable = True
            s_properties.trackable = True
            s_properties.deltat = [8] # lifetime in years
            s_properties.Pm = 0.90 # post mission disposal efficacy
            s_properties.alpha = 1e-5
            s_properties.alpha_active = 1e-5
            
            # Functions
            s_properties.launch_func = launch_func_constant
            s_properties.pmd_func = pmd_func_sat
            s_properties.drag_func = drag_func_exp

            # Append Species
            self.species.append(s_properties)
            
        # sns - Sns 3U cubesat with no station_keeping
        if "sns" in species_names:
            sns_properties = SpeciesProperties()

            # Default values
            sns_properties.sym_name = "sns"
            sns_properties.Cd = 2.2
            sns_properties.mass = 6
            sns_properties.radius = 0.105550206
            sns_properties.A = 0.035 # m^2
            sns_properties.amr = sns_properties.A/sns_properties.mass # m^2/kg
            sns_properties.beta = sns_properties.Cd * sns_properties.amr # ballistic coefficient

            # Orbit Properties
            sns_properties.slotted = False

            # Capabilities
            sns_properties.drag_effected = True
            sns_properties.active = True
            sns_properties.maneuverable = False
            sns_properties.deltat = 3
            sns_properties.Pm = 0

            # Functions
            sns_properties.launch_func = launch_func_constant
            sns_properties.pmd_func = pmd_func_sat
            sns_properties.drag_func = drag_func_exp

            # Append Species
            self.species.append(sns_properties)

        # N - Debris
        # Assuming all spheres of 1 cm, 10cm diameter, 15 kg      
        if "N" in species_names:
            n_properties = SpeciesProperties()

            # Default values
            n_properties.sym_name = "N"
            n_properties.Cd = 2.2
            n_properties.mass = 15
            n_properties.radius = 0.05
            n_properties.A = pi*n_properties.radius**2
            n_properties.amr = n_properties.A/n_properties.mass
            n_properties.beta = n_properties.Cd * n_properties.amr

            # Orbit Properties
            n_properties.slotted = False
            n_properties.slotting_effectiveness = 1 

            # Capabilities
            n_properties.drag_effected = True
            n_properties.active = False
            n_properties.maneuverable = False
            n_properties.deltat = None
            n_properties.Pm = 0
            n_properties.alpha = 0
            n_properties.alpha_active = 0
            n_properties.RBflag = 1

            # Functions
            n_properties.launch_func = launch_func_null
            n_properties.pmd_func = pmd_func_derelict
            n_properties.drag_func = drag_func_exp

            # Append Species
            self.species.append(n_properties)

        return self.species
    
    def multi_property_species(launch_func, pmd_func, drag_func, scen_properties, species):
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

        if len(species.mass) == 1:
            raise ValueError("Multi-property species must have multiple masses.")
        
        if species.sym_name.contains("_"):
            raise ValueError("Species names cannot contain underscores.")
        
        species_list = []
        num_species = len(species.mass)

        # If the species has been specified with muliple massess, this will go through making a 
        # of the class with a different mass for each species.
        for i in range(num_species):
            # Create a new species
            new_species = SpeciesProperties()

            # Set the properties
            new_species.sym_name = species.sym_name + f"_{i}"
            new_species.Cd = species.Cd
            new_species.mass = species.mass[i]
            new_species.radius = species.radius[i]
            new_species.A = species.A[i]
            new_species.amr = species.amr[i]
            new_species.beta = species.beta[i]
            new_species.density_filepath = species.density_filepath
            new_species.slotted = species.slotted
            new_species.slotting_effectiveness = species.slotting_effectiveness
            new_species.disposal_altitude = species.disposal_altitude
            new_species.drag_effected = species.drag_effected
            new_species.active = species.active
            new_species.maneuverable = species.maneuverable
            new_species.trackable = species.trackable
            new_species.deltat = species.deltat
            new_species.Pm = species.Pm
            new_species.alpha = species.alpha
            new_species.alpha_active = species.alpha_active
            new_species.RBflag = species.RBflag
            new_species.orbit_raising = species.orbit_raising
            new_species.insert_altitude = species.insert_altitude
            new_species.onstation_altitude = species.onstation_altitude
            new_species.epsilon = species.epsilon
            new_species.e_mean = species.e_mean
            new_species.lambda_multiplier = species.lambda_multiplier
            new_species.lambda_funs = species.lambda_funs
            new_species.lambda_constant = species.lambda_constant
            new_species.lambda_python_args = species.lambda_python_args
            new_species.pmd_linked_species = species.pmd_linked_species
            new_species.pmd_linked_multiplier = species.pmd_linked_multiplier
            new_species.eq_idxs = species.eq_idxs
            new_species.last_calc_x = species.last_calc_x
            new_species.last_calc_t = species.last_calc_t
            new_species.last_lambda_dot = species.last_lambda_dot
            new_species.saved_model_path = species.saved_model_path
            new_species.t_plan_max = species.t_plan_max
            new_species.t_plan_period = species.t_plan_period
            new_species.prev_prop_results = species.prev_prop_results
            new_species.launch_func = launch_func
            new_species.pmd_func = pmd_func
            new_species.drag_func = drag_func



