Species Options
============
 
At the bottom of the PySSEM page, there is a button to add new species to the model. This button opens a flyout, which allows the user to enter a set of parameters that control the dynamics and propagation of the species. These options will be applied to every satellite in the created species. 

Physical Properties Inputs
--------------------------
  
**Cd**
  
(Numeric Entry) [Sink] Coefficient of Drag. This parameter, along with the altitude shell of the satellite and size (radius/area/mass) will determine how much of a force drag contributes to the modeled dynamics of the satellite. A larger Cd value indicates the satellite is more prone to atmospheric drag effects.
  
This value has historically been assumed to be ~2.2 for LEO satellites, but may be smaller or larger for certain species.


**Mass (kg)**

(Numeric Entry) [Sink] This is the mass of the satellite. With the radius and area, this informs the drag calculation for satellites unable to station-keep against drag. The larger the mass, the slower the decay of the satellite’s orbit due to drag will occur. 

Satellites and debris can vary in mass. Smallsats like a 1U CubeSat can measure around 2kg. SpaceX Starlink satellites are larger, ranging between 200kg to 1250kg depending on their versions.  The International Space Station is the heaviest spacecraft currently in orbit, with a mass of approximately 450,000kg.


**Radius (m)**

(Numeric Entry) [Sink] This is the radius of the satellite. This is used to inform the area of the satellite if an area value was not provided by the user. An increased radius will increase the affect of drag on the satellite. Additionally, the radius of the satellite is used by the collision calculator in pySSEM to determine debris creation parameters in the case of a collision.


**Area (m2)**

(Numeric Entry) [Sink] This is the cross-sectional area of the satellite. An increased area will increase the effect of drag on the satellite. Additionally, the area of the satellite is used by the collision calculator in pySSEM to determine debris creation parameters in the case of a collision.

