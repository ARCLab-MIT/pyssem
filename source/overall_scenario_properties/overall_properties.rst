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
