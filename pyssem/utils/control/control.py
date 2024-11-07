
from sympy import zeros, symbols

def control_none(t, h, species_properties, scen_properties):
    return zeros(scen_properties.n_shells, 1)

def control_launch_sym(t, h, species_properties, scen_properties):
    U = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        U[k, 0] = symbols(f"u_l_{species_properties.sym_name}{k+1}") 

    return U

def control_adr_sym(t, h, species_properties, scen_properties):
    U = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        U[k, 0] = symbols(f"u_r_{species_properties.sym_name}{k+1}") 

    return U