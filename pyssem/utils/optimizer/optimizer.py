from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.integrate import solve_ivp
import numpy as np


def PrfAll(x, baseline):
  """
  Python version of the PrfAll function.

  Args:
    x: Input array, where the first N_shell elements represent S_all.
    baseline: Dictionary containing data, including 'N_shell'.

  Returns:
    f: Objective function.
  """

  N_shell = baseline.n_shells
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
      baseline: Your baseline object containing symbolic variables.

    Returns:
      An array containing the results of all equality constraint function evaluations.
    """

    c_eq = np.array(fun_ceq(x)) #.flatten()

    return c_eq

def PrIneqAll(x,baseline,failure_rate_U):
    """
    Args:
      x: The optimization variables.
      baseline: Your baseline object containing symbolic variables.

    Returns:
      An array containing the results of all inequality constraint function evaluations.
    """

    N_shell = baseline.n_shells
    S_all = x[:N_shell]
    lambda_all = x[3*N_shell:4*N_shell]
    deltat = 8
    y_fail_u = []
    for i1 in range(N_shell):
        y_fail_u.append( -(lambda_all[i1] * deltat * (1 - failure_rate_U) - S_all[i1]) ) 
    c_ineq = y_fail_u

    return c_ineq