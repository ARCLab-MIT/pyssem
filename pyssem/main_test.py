import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.integrate import solve_ivp

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

  # Uncomment for NOT HIGH CAPACITY scenarios (as needed)
  # f = - np.sum(np.log(S_all[5:8])) # shells at 500-600 km (Python index starts at 0)
  # f = - np.sum(S_all[6]) # shell at 550 km (Python index starts at 0)

  return f

def PrcAll(x, baseline):
    """
    Evaluates all constraints at once.

    Args:
      x: The optimization variables.
      baseline: Your baseline object containing symbolic variables.

    Returns:
      An array containing the results of all constraint function evaluations.
    """

    c_eq = np.array([sp.lambdify(baseline.all_symbolic_vars, eq, 'numpy')(*x) for eq in equations_flattened])
    
    # c_ineq 
    # y_fail_l(k) = -(lambda * Dt * (1 - failure_rate_L) - S);
    # y_fail_u(k) = lambda * Dt * (1 - failure_rate_U) - S; 

    return c_eq

# import pickle file 
with open('scenario-properties-test.pkl', 'rb') as f:
    baseline = pickle.load(f)
print("Loaded scenario properties from pickle file.")

# Equations
baseline.equations

# x0 initial population
baseline.x0

# baseline.
baseline.all_symbolic_vars

# solver
equations_flattened = [baseline.equations[i, j] for j in range(baseline.equations.cols) for i in range(baseline.equations.rows)]

full_lambda = sp.Matrix(sp.symbols([f'lambda_{i+1}' for i in range(baseline.n_shells)]))
full_lambda_flattened = [full_lambda[i, j] for j in range(full_lambda.cols) for i in range(full_lambda.rows)]

for i1 in range(baseline.n_shells):
    equations_flattened[i1] = equations_flattened[i1]+full_lambda_flattened[i1]

baseline.all_symbolic_vars = baseline.all_symbolic_vars + full_lambda_flattened

equations = [sp.lambdify(baseline.all_symbolic_vars, eq, 'numpy') for eq in equations_flattened]

## Initial guess
x0 = np.ones(len(baseline.all_symbolic_vars)) * 0

## Bounds
lb = np.ones(len(baseline.all_symbolic_vars)) * 1
ub = np.inf

## Objective function
objective = PrfAll

## Constraints for SLSQP
con1 = {'type': 'eq', 'fun': lambda x: PrcAll(x, baseline)}
# con2 = {'type': 'ineq', 'fun': constraint2}
nonlcon = con1

## Options for SLSQP
options = {
    'disp': True,   
    # 'maxiter': 5e5,
    'ftol': 1e-6,     
    'eps': 1e-18,     
}

print("Running optimization...")
start_time = time.time()

## Perform the optimization
result = minimize(objective, x0, args=(baseline), method='SLSQP',
                  jac=None, hess=None, hessp=None, 
                  bounds = Bounds(lb=lb, ub=ub, keep_feasible=False), 
                  constraints = nonlcon,
                  options = options)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")

## Print the results
print("Optimal found at:", result.x)
print("Function value at optimal:", -result.fun)
print("Equality constraints:", np.array([sp.lambdify(baseline.all_symbolic_vars, eq, 'numpy')(*result.x) for eq in equations_flattened]))
result