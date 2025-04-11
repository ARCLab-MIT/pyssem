
Species Properties
================

At the bottom of the PySSEM page, there is a button to add new species to the model. This button opens a flyout, which allows the user to enter a set of parameters that control the dynamics and propagation of the species.  
These options can also be used to model debris already present in the environment, turning the ‘Active’, ‘Maneuverable’, ‘Slotted’ toggles to [Off]. Additionally, for debris, the ‘PMD %’ can be adjusted accordingly.  

Behavioral Inputs
----------------

**Active** 

(Toggle On/Off) [Sink] This parameter controls whether the satellites of the selected species are modeled to perform station keeping to control against perturbations. This selection is controlled also by other factors.


**Maneuverable**  

(Toggle On/Off) [Sink] This parameter controls whether the satellites of the selected species are modeled to maneuver to avoid collisions. The success of the maneuverability is determined by either probabilistic models or user-defined inputs.


**Delta t (years)** 

(Numeric Entry) [Sink] This input controls the amount of time that an active satellite stays ‘active’, if that satellite’s “Post Mission Disposal” attempt were to fail (based upon the PMD percentage).


**PMD (%)**  

(Slider, numeric) PMD refers to “Post Mission Disposal”. The PMD percentage is the percentage of vehicles that are disposed of (by their simulated operators) after their mission ends. If a mission fails, this value determines the disposal rate.


**Alpha**  

(Numeric Entry) This value represents the failure rate of the maneuverable satellite to perform a collision avoidance maneuver with an inactive object.  

Values for Alpha range from [0] to [1], where [0 = perfect collision avoidance, 1 = none]. A value of 0.1 would mean that 1/10 collision avoidance maneuvers fail. If Alpha is unknown for the capability, a default value may be used.


**Alpha Active**

(Numeric Entry) This value represents the failure rate of the maneuverable satellite to perform a collision avoidance maneuver with another active, maneuverable object.  

Values for Alpha Active range from [0] to [1], where [0 = perfect collision avoidance, 1 = none]. A value of 0.1 would mean that 1/10 collision avoidance maneuvers fail. If Alpha Active is unknown for the capability, a default value may be used.


**Slotted**  

(Toggle On/Off) [Sink] This toggle controls whether the satellite orbits within a ‘slot’ as part of a larger constellation within the species. Slotted orbits are those which are deconflicted with other satellites in the same constellation.


**Slotting Effectiveness** 

(Numeric Entry) [Sink] This value determines the effectiveness of slotting for those satellites in a ‘Slotted’ Species.  

Values for Slotting Effectiveness range from [1] to [0], where [1 = perfect slotting, 0 = no slotting].


**Affected by Drag**  

(Toggle On/Off) This toggle controls whether the selected species of satellite is affected by drag. If toggled on, which is a more realistic option for a LEO scenario, the propagator will consider the drag force acting on the satellites.


**Launch Function** 

(Text Entry) [Source] This value controls how new satellites of this species are launched into the modeled space environment. In conjunction with the ‘Launch Coefficient’, this gives pySSEM the direction for satellite launches.  

The options available for Launch Function are either [launch_func_constant] for constant launches at the rate of the launch coefficient, or [launch_func_null] for no launches for this species (such as debris).


**PMD Function**  

(Text Entry) [Sink] This value controls the modeling of how satellites are disposed. In conjunction with the ‘PMD %’, this gives pySSEM the direction to dispose of satellites at each time step.  

The options available for PMD Function are either [pmd_func_sat] for PMD-enabled satellites, [pmd_func_derelict] for those satellites with PMD attempts that have been assumed to have already failed, or other custom-defined functions.


**Drag Function** 

(Text Entry) [Sink] This value is used to support pySSEM in drag force calculation.  

The options available for Drag Function are either [drag_func_exp] for satellites that experience drag, or [drag_func_none] for those that don’t.
