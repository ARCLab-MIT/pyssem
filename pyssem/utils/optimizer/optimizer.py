from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sympy as sp

def PrfAll(x, scenario_properties):
  """
  Python version of the PrfAll function.

  Args:
    x: Input array, where the first N_shell elements represent S_all.
    scenario_properties: Dictionary containing data, including 'N_shell'.

  Returns:
    f: Objective function.
  """ 

  # define which species that you want
  # log helps to norm the values
  # github - should be more hands on/tuning for the optimizer
  
  N_shell = scenario_properties.n_shells
  S_all = x[:N_shell]  # Extract S_all from x

  # HIGH CAPACITY
  # f = -np.sum(np.log(S_all)**1)
  f = -np.sum(np.log(S_all)**2) 
  # f = -np.sum(np.log(S_all)**3) 
  # f = 1/np.sum(S_all) 

  # NOT HIGH CAPACITY
  # f = - np.sum(np.log(S_all[5:8])) # shells at 500-600 km (Python index starts at 0)
  # f = - np.sum(S_all[6]) # shell at 550 km (Python index starts at 0)

  return f

def PrCeqAll(x, fun_ceq):
    """
    Args:
      x: The optimization variables.
      scenario_properties: Your scenario_properties object containing symbolic variables.

    Returns:
      An array containing the results of all equality constraint function evaluations.
    """

    c_eq = np.array(fun_ceq(x)) #.flatten()

    return c_eq

def PrIneqAll(x,scenario_properties,failure_rate_U):
    """
    Args:
      x: The optimization variables.
      scenario_properties: Your scenario_properties object containing symbolic variables.

    Returns:
      An array containing the results of all inequality constraint function evaluations.
    """

    # need to fix this - user defines what species they want to use
    # usually is greater than 0, need to flip the sign if you want less than. 

    N_shell = scenario_properties.n_shells
    S_all = x[:N_shell]
    lambda_all = x[3*N_shell:4*N_shell] # should be the end of the x array
    deltat = 8 # not able to access deltat from the scenario_properties object - fix 
    y_fail_u = []
    for i1 in range(N_shell):
        y_fail_u.append( -(lambda_all[i1] * deltat * (1 - failure_rate_U) - S_all[i1]) ) # flip minus sign for less than, failure rate upper is set to 100, this can be defined by the user. 
    c_ineq = y_fail_u

    return c_ineq


def run_optimizer(scenario_properties):
  """ 
  This occurs after the model simulation has been run, 
  it will take a full SimulationClass, extract the equations, and re-run an optimization. 
  """

  # solver
  equations_flattened = [scenario_properties.equations[i, j] for j in range(scenario_properties.equations.cols) for i in range(scenario_properties.equations.rows)]

  full_lambda = sp.Matrix(sp.symbols([f'lambda_{i+1}' for i in range(scenario_properties.n_shells)]))
  full_lambda_flattened = [full_lambda[i, j] for j in range(full_lambda.cols) for i in range(full_lambda.rows)]

  for i1 in range(scenario_properties.n_shells):
      equations_flattened[i1] = equations_flattened[i1]+full_lambda_flattened[i1]

  scenario_properties.all_symbolic_vars = scenario_properties.all_symbolic_vars + full_lambda_flattened

  N_shell = scenario_properties.n_shells
  print(scenario_properties.species_names)

  # Number of species
  num_species = len(scenario_properties.species_names)

  # Initialize lists to store initial guesses and lower bounds
  species_list = []
  lb_species_list = []

  # Loop through species names and generate initial guesses and lower bounds dynamically
  for i, var in enumerate(scenario_properties.species_names):
      var_name = var + '_0'  # For naming purposes
      # Dynamically calculate the number of variables for each species based on N_shell
      var_initial_guess = np.ones(len(scenario_properties.all_symbolic_vars[i * N_shell:(i + 1) * N_shell]))  # Initial guess for each species
      lb_initial = np.ones(len(scenario_properties.all_symbolic_vars[i * N_shell:(i + 1) * N_shell])) * 1  # Lower bound for each species
      
      species_list.append(var_initial_guess)
      lb_species_list.append(lb_initial)

  # Concatenate all species initial guesses
  x0_species = np.concatenate(species_list)

  # Concatenate all species lower bounds
  lb_species = np.concatenate(lb_species_list)

  # Dynamically handle the lambda variables based on how many species there are
  start_lambda_idx = num_species * N_shell
  lam_0 = np.ones(len(scenario_properties.all_symbolic_vars[start_lambda_idx:]))  # Handling lambda dynamically

  # Concatenate species initial guesses with lam_0 to form the final x0
  x0 = np.concatenate([x0_species, lam_0])

  # Handle the lambda variables' lower bounds similarly
  lb_lam = np.ones(len(scenario_properties.all_symbolic_vars[start_lambda_idx:])) * 1

  # Concatenate species lower bounds with lambda lower bounds to form the final lb array
  lb = np.concatenate([lb_species, lb_lam])
  ub = np.inf

  ## Farilure rate, % = fail_rate / 100
  # make this user define. This is a %. 
  failure_rate_U = 100
  failure_rate_U = failure_rate_U/100

  ## Objective function
  objective = PrfAll

  ## Constraints for SLSQP
  f3 = equations_flattened
  var_c = scenario_properties.all_symbolic_vars
  fun_ceq = sp.lambdify((var_c,), f3, 'numpy')
  con1 = {'type': 'eq', 'fun': lambda x: PrCeqAll(x, fun_ceq)}
  con2 = {'type': 'ineq', 'fun': lambda x: PrIneqAll(x,scenario_properties,failure_rate_U)}
  nonlcon = [con1, con2]

  ## Options for SLSQP
  options = {
      'disp': True,   
      'maxiter': 5e5,
      'ftol': 1e-6,     
      'eps': 1e-18,
      # 'finite_diff_rel_step': 1e-12,     
  }

  ## Perform the optimization
  result = minimize(objective, x0, args=(scenario_properties), method='SLSQP', 
                    jac='cs', hess=None, hessp=None, 
                    bounds = Bounds(lb=lb, ub=ub, keep_feasible=False), 
                    constraints = nonlcon,
                    options = options)

  ## Print the results
  print("Optimal found at:", result.x)
  print("Function value at optimal:", -result.fun)
  print("Equality constraints:", np.array([sp.lambdify(scenario_properties.all_symbolic_vars, eq, 'numpy')(*result.x) for eq in equations_flattened]))

  xopt = result.x
  R02 = scenario_properties.R0_km

  # Initialize a list to store the optimal values for all species
  species_opt_list = []

  # Dynamically handle the extraction of variables for all species (no assumptions on names like S, N, D)
  for i in range(num_species):
      start_idx = i * N_shell
      end_idx = (i + 1) * N_shell
      species_opt_list.append(xopt[start_idx:end_idx])

  # Concatenate species-related optimal values into a single array
  species_opt = np.concatenate(species_opt_list)

  # Dynamically handle lambda variables (lambda always comes after all species-related variables)
  start_lambda_idx = num_species * N_shell
  lam_opt = xopt[start_lambda_idx:]

  # Handle constraints and equilibrium values dynamically
  c_eq = np.array([sp.lambdify(scenario_properties.all_symbolic_vars, eq, 'numpy')(*xopt) for eq in equations_flattened])

  # Initialize a list to store equilibrium values for all species
  species_eq_list = []

  # Extract equilibrium values for each species dynamically
  for i in range(num_species):
      start_idx = i * N_shell
      end_idx = (i + 1) * N_shell
      species_eq_list.append(c_eq[start_idx:end_idx])

  # Concatenate equilibrium values for species into a single array
  species_eq = np.concatenate(species_eq_list)

  # Handle dynamic variable and lambda splitting for ODE solving
  var = scenario_properties.all_symbolic_vars[:num_species * N_shell]
  lam = scenario_properties.all_symbolic_vars[num_species * N_shell:]

  # Dynamically handle the ODE system function for solve_ivp
  f3 = equations_flattened
  fun3 = sp.lambdify((var, lam), f3, 'numpy')

  # Define the ODE system function to be used in solve_ivp
  def func(t, x, fun3, lam_opt):
      return np.array(fun3(x, np.array(lam_opt))).flatten()

  # Time span for the solution
  tf_ss = 100
  tspan1 = np.linspace(0, tf_ss, 100)

  # Solve the ODE using solve_ivp
  sol = solve_ivp(func, (0, tf_ss), xopt[:num_species * N_shell], 
                  method=scenario_properties.integrator,
                  t_eval=tspan1, 
                  args=(fun3, lam_opt),
                  rtol=1e-8, atol=1e-8)

  # Extract time and solution properties
  t_prop = sol.t
  x_prop = sol.y

  # Dynamically extract the properties of all species from the solution
  species_prop_list = []

  for i in range(num_species):
      start_idx = i * N_shell
      end_idx = (i + 1) * N_shell
      species_prop_list.append(x_prop[start_idx:end_idx])

  # Concatenate species properties into a single array for further processing
  species_prop = np.concatenate(species_prop_list)

  # Calculate total population (or other metric) for each species
  N_tot = np.sum(species_prop, axis=0)

  # Set plot settings
  sel_LineWidth = 2
  sel_MarkerWidth = 10
  sel_LineWidthAxis = 1
  sel_FontSize = 14
  deltat = 8

  # Ensure the figures and optimizer directories exist
  import os
  if not os.path.exists("figures"):
      os.makedirs("figures")
  if not os.path.exists("figures/optimizer"):
      os.makedirs("figures/optimizer")

  # Create a colormap to generate colors dynamically based on the number of species
  num_species = len(scenario_properties.species_names)
  colormap = cm.get_cmap('tab10', num_species)  # Using 'tab10' color map, adjust for more species

  # 1. Plot the total population and individual species
  plt.figure(facecolor='w')
  plt.grid(True)

  # Initialize N_tot_sum for total population across species
  N_tot_sum = np.zeros_like(species_prop_list[0].sum(axis=0))  # Shape based on the first species

  # Dynamically sum the populations for all species to get the total population
  for species_data in species_prop_list:
      N_tot_sum += species_data.sum(axis=0)  # Summing across time (or the dimension of interest)

  # Plot total population (summed across species)
  plt.plot(t_prop, N_tot_sum, color='black', linewidth=sel_LineWidth)

  # Dynamically plot all species with dynamically generated colors
  for i, species_name in enumerate(scenario_properties.species_names):
      plt.plot(t_prop, species_prop_list[i].sum(axis=0), color=colormap(i), linewidth=sel_LineWidth)

  # Update labels, title, and legend dynamically
  plt.title("Population")
  plt.xlabel("Years")
  plt.ylabel("Count")
  legend_labels = ["Total"] + scenario_properties.species_names  # Add species names dynamically
  plt.legend(legend_labels, loc="best")
  plt.gca().tick_params(axis='both', which='major', labelsize=sel_FontSize)
  plt.gca().spines['bottom'].set_linewidth(sel_LineWidthAxis)
  plt.gca().spines['left'].set_linewidth(sel_LineWidthAxis)
  plt.savefig("figures/optimizer/so_variation_no_fail.png", dpi=300)

  # 2. Plot population variation
  plt.figure(facecolor='w')
  plt.grid(True)

  # Plot total population variation
  plt.plot(t_prop, N_tot_sum[-1] - N_tot_sum, color='black', linewidth=sel_LineWidth)

  # Dynamically plot species population variation with dynamically generated colors
  for i, species_name in enumerate(scenario_properties.species_names):
      species_variation = species_prop_list[i].sum(axis=0)  # Summing over the shells dimension
      plt.plot(t_prop, species_variation[-1] - species_variation, color=colormap(i), linewidth=sel_LineWidth)

  # Update labels, title, and legend dynamically
  plt.title("Population variation")
  plt.xlabel("Years")
  plt.ylabel("Count")
  legend_labels = ["Total"] + scenario_properties.species_names  # Add species names dynamically
  plt.legend(legend_labels, loc="best")
  plt.gca().tick_params(axis='both', which='major', labelsize=sel_FontSize)
  plt.gca().spines['bottom'].set_linewidth(sel_LineWidthAxis)
  plt.gca().spines['left'].set_linewidth(sel_LineWidthAxis)
  plt.savefig("figures/optimizer/population_variation.png", dpi=300)

  # 3. Plot failure rate constraint
  plt.figure(facecolor='w')
  plt.grid(True)

  # Plot the max failure rate as a dashed line
  plt.plot(R02[1:] - 25, 100 * failure_rate_U * np.ones_like(R02[1:]), '--', color='black', linewidth=sel_LineWidth)

  # Filter species that begin with 'S'
  s_species_indices = [i for i, species_name in enumerate(scenario_properties.species_names) if species_name.startswith('S')]

  # Dynamically plot the failure rate only for species starting with 'S'
  for i in s_species_indices:
      plt.plot(R02[1:] - 25, 100 * (deltat * lam_opt - species_opt_list[i]) / (deltat * lam_opt),
              '-o', color=colormap(i), linewidth=sel_LineWidth, markersize=sel_MarkerWidth)

  # Update labels, title, and legend dynamically
  s_species_names = [scenario_properties.species_names[i] for i in s_species_indices]
  plt.title("Failure rate constraint")
  plt.xlabel("Altitude (km)")
  plt.ylabel("Failure rate (%)")
  plt.legend(["$\chi_{max}$"] + s_species_names, loc="best")
  plt.gca().tick_params(axis='both', which='major', labelsize=sel_FontSize)
  plt.gca().spines['bottom'].set_linewidth(sel_LineWidthAxis)
  plt.gca().spines['left'].set_linewidth(sel_LineWidthAxis)
  plt.savefig("figures/optimizer/fail_rate_no_fail.png", dpi=300)

  # 4. Plot the optimal solution dynamically for all species
  plt.figure(facecolor='w')
  plt.grid(True)

  # Plot optimal solutions for each species dynamically using a loop
  for i, species_name in enumerate(scenario_properties.species_names):
      plt.semilogy(R02[1:] - 25, species_opt_list[i], '-o', color=colormap(i), linewidth=sel_LineWidth, markersize=sel_MarkerWidth)

  # Always plot lambda as the last line
  plt.semilogy(R02[1:] - 25, lam_opt, '-v', color='black', linewidth=sel_LineWidth, markersize=sel_MarkerWidth)

  # Update labels, title, and legend dynamically
  plt.title("Optimal solution")
  plt.xlabel("Altitude (km)")
  plt.ylabel("Count")
  plt.legend(scenario_properties.species_names + ["$\lambda$"], loc="best")
  plt.gca().tick_params(axis='both', which='major', labelsize=sel_FontSize)
  plt.gca().spines['bottom'].set_linewidth(sel_LineWidthAxis)
  plt.gca().spines['left'].set_linewidth(sel_LineWidthAxis)
  plt.savefig("figures/optimizer/max_capacity_no_fail.png", dpi=300)

  # 5. Plot equilibrium constraint dynamically for all species
  plt.figure(facecolor='w')
  plt.grid(True)

  # Dynamically plot equilibrium values for each species with dynamically generated colors
  for i, species_name in enumerate(scenario_properties.species_names):
      plt.plot(R02[1:] - 25, species_eq_list[i], color=colormap(i), linewidth=sel_LineWidth)

  # Update labels, title, and legend dynamically
  plt.title("Equilibrium constraint")
  plt.xlabel("Altitude (km)")
  plt.ylabel("Count")
  plt.legend(scenario_properties.species_names, loc="best")
  plt.gca().tick_params(axis='both', which='major', labelsize=sel_FontSize)
  plt.gca().spines['bottom'].set_linewidth(sel_LineWidthAxis)
  plt.gca().spines['left'].set_linewidth(sel_LineWidthAxis)
  plt.savefig("figures/optimizer/equil_constr_no_fail.png", dpi=300)