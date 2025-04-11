Overall Scenario Properties
===================

**Overview**

This page provides a description of all properties that have an effect on the simulation of the entire scenario. These properties are static throughout the entire timespan of the scenario and do not change.

Inputs
-----------

**Start Date (DD/MM/YYYY)**

(Text Entry) This is the start date of your simulation, and must be entered in the DD/MM/YYYY format.
This parameter, when paired with the ‘Simulation Duration’ parameter, determines how far into the
future the simulation will model.


**Simulation Duration (years)**

(Numeric Entry) This field is number of years, from your chosen ‘Start Date’ that your simulation will run.


**Time Steps**

(Numeric Entry) This is the number of times that the SSEM calculates changes to the environment.
Propagation from time step to time step is the method by which the model advances forward in time.
The higher the entered number of time steps, the higher fidelity the model will be, but the longer it may
take to run. This entry is paired with the ‘Simulation Duration’ parameter.

As an example, if the duration is set to 100 years, and the number of time steps is 50, then each
calculation of the environment will represent years of change

**Maximum Altitude (km)**

(Numeric Entry) This is the maximum altitude of the simulation, measured in kilometers. This altitude is
the top height of the highest shell in the simulation. Atmospheric drag is the generally lower at higher
altitudes.


**Minimum Altitude (km)**

(Numeric Entry) This is the minimum altitude of the simulation, measured in kilometers. This altitude is
the bottom of the lowest shell in the simulation. Atmospheric drag is the generally higher at lower
altitudes.


**Number of Shells**

(Numeric Entry) This is the number of shells, or bins, that satellites can move between in the simulation.
The higher the number of shells, the smaller each shell will be (and the higher fidelity the model will be),
but the longer it may take to run. This entry is paired the ‘Maximum Altitude’ and ‘Minimum Altitude’
parameters.

As an example, if the maximum altitude is 4000km, and the minimum is 500km, and the number of
shells is set to 10, then each shell will span in altitude.


Advanced Options
---------------
These parameters are able to be changed once the slider at the top right of the page is switched.


**Integrator Type**

(Text Entry) This value is the type of integrator that pySSEM uses to propagate the current state of the
model to the subsequent state of the model, over a time step (of size determined by the ‘Simulation
Duration’ and ‘Time Steps’ values.

The default value is [BDF].


**Density Model**

(Text Entry) [Sink] This is the atmospheric density model that controls how pySSEM models the drag
effect against satellites in a species.

The default value is [static_exp_dens_func], which uses pre-computed density values at specific layers
to return atmospheric density for each bin altitude. These densities are returned to the Drag Function
for each species to find drag forces.


**Launch Coefficient**

(Numeric Entry) [Source] In conjunction with the ‘Launch Function’ of each species, this gives pySSEM
the direction to add more satellites (or not) at each time step.

The default value is [0.1].


**Impact Velocity (km/s)**

(Numeric Entry) [Sink] This is the assumed relative velocity of the two objects paired together for
collision calculation, and is held constant across the simulation.

The default value is [10] km/s. For reference, the active Iridium 33 satellite and the derelict Cosmos
2251 had a relative speed of approximately 11.7km/s when they collided in 2009. [2]
