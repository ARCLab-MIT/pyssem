from sympy import zeros, Matrix, symbols

def launch_func_null(t, h, species_properties, scen_properties):
    """
    No launch function for species without a launch function.

    Args:
        t (float): Time from scenario start in years
        h (array_like): The set of altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): A dictionary with properties for the species
        scen_properties (dict): A dictionary with properties for the scenario

    Returns:
        numpy.ndarray: The rate of change in the species in each shell at the specified time due to launch.
                       If only one value is applied, it is assumed to be true for all shells.
    """

    Lambdadot = zeros(scen_properties.n_shells, 1)

    for k in range(scen_properties.n_shell):
        Lambdadot[k, 0] = 0 * species_properties.sym[k]

    Lambdadot_list = [Lambdadot[k, 0] for k in range(scen_properties.n_shell)]

    return Lambdadot_list

def launch_func_constant(t, h, species_properties, scen_properties):
    """
    Adds a constant launch rate from species_properties.lambda_constant.

    Args:
        t (float): Time from scenario start in years.
        h (list or numpy.ndarray): Altitudes of the scenario above ellipsoid in km of shell lower edges.
        species_properties (dict): Properties for the species, including 'lambda_constant'.
        scen_properties (dict): Properties for the scenario, including 'N_shell'.

    Returns:
        list: Lambdadot, a list of symbolic expressions representing the rate of change in the species in each shell due to launch.
    """
    if len(h) != scen_properties.n_shells:
        raise ValueError("Constant launch rate must be specified per altitude shell.")

    # Create a symbolic variable for the launch rate
    lambda_constant = symbols('lambda_constant')

    # Assign the constant launch rate to each shell
    Lambdadot = Matrix(scen_properties.n_shells, 1, lambda i, j: lambda_constant)

    # Convert the Matrix of symbolic expressions to a list
    Lambdadot_list = [Lambdadot[i] for i in range(scen_properties.n_shells)]

    return Lambdadot_list

    # def launch_func_constant(self):
    #     """
    #     Adds constant launch rate from species_properties.lambda_constant

    #     Args:
    #         t (float): Time from scenario start in years
    #         h (array_like): The set of altitudes of the scenario above ellipsoid in km of shell lower edges.
    #         species_properties (dict): A dictionary with properties for the species
    #         scen_properties (dict): A dictionary with properties for the scenario

    #     Returns:
    #         numpy.ndarray: The rate of change in the species in each shell at the specified time due to launch.
    #                     If only one value is applied, it is assumed to be true for all shells.
    #     """

    #     lambda_constant = [500 * random.random() for i in range(self.n_shells)]

    #     # Generate symbolic variables and multiply each by the corresponding lambda_constant value
    #     Lambdadot_symbols = symbols('Lambdadot_1:%d' % (self.n_shells + 1))  # Create n shells symbolic variables
    #     Lambdadot = Matrix(self.n_shells, 1, lambda i, j: Lambdadot_symbols[i] * lambda_constant[i])
        
    #     return Lambdadot