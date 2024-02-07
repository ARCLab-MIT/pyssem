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
        self.pmd_linked_species = None
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


            # NEED TO FORK THIS CODE TO PYTHON
            # Sns_species = species(@launch_func_null, @pmd_func_sat, my_drag_func, species_properties, scen_properties);

            # for i = 1:length(Sns_species) 
            #     Sns_masses(i) = Sns_species(i).species_properties.mass; 
            #     Sns_radii(i) = Sns_species(i).species_properties.radius;
            # end

        # N - Debris
        # Assuming all spheres of 1 cm, 10cm diameter, 15 kg      
        if "N" in species_names:
            n_properties = SpeciesProperties()

            # Default values
            n_properties.sym_name = "N"
            n_properties.Cd = 2.2
            n_properties.mass = 15
            n_properties.radius = 0.05
            # species_properties.mass =   [0.00141372    0.5670    S_masses Su_masses Sns_masses]; 
            # species_properties.radius = [0.01          0.1321    S_radii  Su_radii  Sns_radii]; % m
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