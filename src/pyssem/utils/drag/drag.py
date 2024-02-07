from sympy import zeros, symbols, sqrt, exp
import numpy as np
#from math import sqrt

def densityexp(h):
    """
    Calculates atmospheric density based on altitude using a exponential model.

    Args:
        h (float or np.ndarray): Height above ellipsoid in km.

    Returns:
        np.ndarray: Atmospheric density in kg/km^3.
    """
    
    # Ensure h is a NumPy array to handle both scalar and vector inputs
    h = np.maximum(h, 0)  # Ensure altitude is non-negative

    # Initialize density array
    p = np.zeros_like(h)

    # Define altitude layers and corresponding parameters (h0, p0, H) based on Vallado (2013)
    layers = [
        (0, 1.225, 7.249),
        (25, 3.899e-2, 6.349),
        (30, 1.774e-2, 6.682),
        (40, 3.972e-3, 7.554),
        (50, 1.057e-3, 8.382),
        (60, 3.206e-4, 7.714),
        (70, 8.770e-5, 6.549),
        (80, 1.905e-5, 5.799),
        (90, 3.396e-6, 5.382),
        (100, 5.297e-7, 5.877),
        (110, 9.661e-8, 7.263),
        (120, 2.438e-8, 9.473),
        (130, 8.484e-9, 12.636),
        (140, 3.845e-9, 16.149),
        (150, 2.070e-9, 22.523),
        (180, 5.464e-10, 29.740),
        (200, 2.789e-10, 37.105),
        (250, 7.248e-11, 45.546),
        (300, 2.418e-11, 53.628),
        (350, 9.518e-12, 53.298),
        (400, 3.725e-12, 58.515),
        (450, 1.585e-12, 60.828),
        (500, 6.967e-13, 63.822),
        (600, 1.454e-13, 71.835),
        (700, 3.614e-14, 88.667),
        (800, 1.170e-14, 124.64),
        (900, 5.245e-15, 181.05),
        (1000, 3.019e-15, 268.00),
    ]

    # Calculate density for each altitude value
    for h0, p0, H in layers:
        mask = (h >= h0) & (h < h0 + 100)
        p[mask] = p0 * np.exp((h0 - h[mask]) / H)

    # Handle altitudes >= 1000 km using the last layer's parameters
    h0, p0, H = layers[-1]
    mask = h >= 1000
    p[mask] = p0 * np.exp((h0 - h[mask]) / H)

    return p


def drag_func(t, species, scene_properties):
    """
    Drag function for the species

    Args:
        t (float): Time from scenario start in years
        species (Species): A Species Object with properties for the species
        scene_properties (SceneProperties): A SceneProperties Object with properties for the scenario

    Returns:
        numpy.ndarray: The rate of change in the species in each shell at the specified time due to drag.
                       If only one value is applied, it is assumed to be true for all shells.
    """
    Fdot = zeros(scene_properties.n_shells, 1)

    if species.drag_effected:
        # Calculate the Shell's altitde and Atmopsheric Density
        h = species.R02
        rho = densityexp(h) # Currently only exponential

        # Calculate the drag force 
        for k in range(scene_properties.n_shell):
            
            # Check the shell is not the top shell
            if k < scene_properties.n_shell:
                n0 = species.sym(k+1)
                h = scene_properties.sym(k+1)
                rho_k1 = rho(k+1)

                # Calculate Drag Flux (Relative Velocity)
                rvel_upper = -rho_k1 * species.beta * sqrt(scene_properties.mu * scene_properties.RO(k+1)) * (24 * 3600* 365.25)
            
            # Otherwise assume that no flux is coming down from the highest shell
            else:
                n0 = 0
                h = scene_properties.R02(k+1)
                rho_k1 = rho(k+1)

                # Calculate Drag Flux
                rvel_upper = -rho_k1 * species.beta * sqrt(scene_properties.mu * scene_properties.RO(k+1)) * (24 * 3600* 365.25)
        
        # Take the current shell and then calculate...
        rho_current_shell_k = rho(k)
        rvel_current = -rho_current_shell_k * species.beta * sqrt(scene_properties.mu * scene_properties.RO(k)) * (24 * 3600* 365.25)
        #Fdot(k, 1) = +n0*rvel_upper/scene_properties.Dhu + rvel_current/scene_properties.Dhl * species.sym(k)


