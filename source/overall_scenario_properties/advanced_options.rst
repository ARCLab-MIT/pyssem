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
