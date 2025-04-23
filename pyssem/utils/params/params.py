
from sympy import zeros, symbols

def params_fix(t, h, species_properties, scen_properties):
    # Initialize Cpmddot as a symbolic zero matrix
    Cpmddot = zeros(scen_properties.n_shells, 1)
    
    # Iterate over each shell and calculate the PMD rate
    for k in range(scen_properties.n_shells):
        Cpmddot[k, 0] = (-1 / species_properties.deltat) * species_properties.sym[k]
    
    return Cpmddot

def params_sym(t, h, species_properties, scen_properties):
    U = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shells):
        U[k, 0] = symbols(f"u_l_{species_properties.sym_name}{k+1}") 

    return U