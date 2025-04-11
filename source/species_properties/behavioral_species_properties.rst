Species Properties
================

At the bottom of the page, there is a button to add new species to the model. This button opens a flyout, which allows the user to enter a set of parameters that control the dynamics and propagation of the species. These options will be applied to every satellite in the created species. 
These options can also be used to model debris already present in the environment, turning the ‘Active’, ‘Maneuverable’, ‘Slotted’ toggles to [Off]. Additionally, for debris, the ‘PMD %’ should be set to [0], the ‘Launch Function’ should be set to [launch_func_null], and the ‘PMD Function’ set to [pmd_func_derelict] or [pmd_func_none]. These changes for debris are already pre-set in the Debris Template when adding a new species.


Behavioral Inputs
----------------

Active
(Toggle On/Off) [Sink] This parameter controls whether the satellites of the selected species are modeled to perform station keeping to control against perturbations. This selection is controlled also by the lifespan of the vehicle after a failed ‘PMD’, as determined by ‘Delta t’.
Maneuverable
(Toggle On/Off) [Sink] This parameter controls whether the satellites of the selected species are modeled to maneuver to avoid collisions. The success of the maneuverability is determined by either ‘Alpha’ or ‘Alpha Active’.
Delta t (years)
(Numeric Entry) [Sink] This input controls the amount of time that an active satellite stays ‘active’, if that satellites “Post Mission Disposal” attempt were to fail (based upon the PMD percentage below). For example, if the ‘Delta t’ value was 10 years and the PMD attempt were to fail, and the ‘Active’ toggle for this species was enabled, the satellite would still retain the ‘Active’ designation until the propagator reached a time step 10 years past the initial PMD attempt. At this point, the satellite would become inactive, and no longer possess any station-keeping ability.
PMD (%) 
(Slider, numeric) PMD refers to “Post Mission Disposal”. The PMD percentage is the percentage of vehicles that are disposed of (by their simulated operators) after their mission ends. If a mission is disposed, that satellite is moved to a lower disposal-altitude orbit, where the satellite will decay in 5 years or less. If a time step for the model is greater than 5 years, this may occur over one time step.
Alpha
(Numeric Entry) This value represents the failure rate of the maneuverable satellite to perform a collision avoidance maneuver with an inactive object. 
Values for Alpha range from [0] to [1], where [0 = perfect collision avoidance, 1 = none]. A value of 0.1 would mean that 1/10 collision avoidance maneuvers fail. If Alpha is unknown for the capabilities of the satellite, the developer of pySSEM recommends to leave the value at [1e-5] as default.
Alpha Active
(Numeric Entry) This value represents the failure rate of the maneuverable satellite to perform a collision avoidance maneuver with another active, maneuverable object. 
Values for Alpha Active range from [0] to [1], where [0 = perfect collision avoidance, 1 = none]. A value of 0.1 would mean that 1/10 collision avoidance maneuvers fail. If Alpha Active is unknown for the capabilities of the satellite, the developer of pySSEM recommends to leave the value at [1e-5] as default.
Slotted
(Toggle On/Off) [Sink] This toggle controls whether the satellite orbits within a ‘slot’ as part of a larger constellation within the species. Slotted orbits are those which are deconflicted with each other to reduce the overlap of orbits and reduce collision occurrences.
Slotting Effectiveness
(Numeric Entry) [Sink] This value determines the effectiveness of slotting for those satellites in a ‘Slotted’ Species. Perfect slotting of a species of satellites means that their orbits will never overlap, and they will have no collisions. 
Values for Slotting Effectiveness range from [1] to [0], where [1 = perfect slotting, 0 = no slotting].
Affected by Drag
(Toggle On/Off) This toggle controls whether the selected species of satellite is affected by drag. If toggled on, which is a more realistic option for a LEO scenario, the propagator will consider the size (radius/area/mass) of the satellite against the density model for the atmosphere, and determine the effect of drag forces on that object over the selected time step. Atmospheric drag reduces the altitude of a satellite that is ‘Affected By Drag’ [On].
Launch Function
(Text Entry) [Source] This value controls how new satellites of this species are launched into the modeled space environment. In conjunction with the ‘Launch Coefficient’, this gives pySSEM the direction to add more satellites (or not) at each time step.
The options available for Launch Function are either [launch_func_constant] for constant launches at the rate of the launch coefficient, or [launch_func_null] for no launches for this species (such as in the case of space debris.
PMD Function
(Text Entry) [Sink] This value controls the modeling of how satellites are disposed. In conjunction with the ‘PMD %’, this gives pySSEM the direction to dispose of satellites at each time step.
The options available for PMD Function are either [pmd_func_sat] for PMD enabled satellites, [pmd_func_derelict] for those satellites with PMD attempts that have been assumed to have already failed, or [pmd_func_none] for satellites where a PMD option does not exist.
Drag Function
(Text Entry) [Sink] This value is used to support pySSEM in drag force calculation.
The options available for Drag Function are either [drag_func_exp] for satellites that experience drag, or [drag_func_none] for those that don’t.
