import numpy as np
import sympy as sp
from collections import defaultdict
from ..drag.drag import densityexp

class SymbolicCollisionTerm:
    def __init__(self, s1_idx, s2_idx, eqs_sources, eqs_sinks, fragment_spread_totals):
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.eqs_sources = eqs_sources     # list of sympy expressions
        self.eqs_sinks = eqs_sinks         # list of sympy expressions

        # Optionally lambdify now or later
        self.lambdified_sources = None
        self.lambdified_sinks = None

        # This is for the distribution of the fragments across a, e
        self.fragment_spread_totals = fragment_spread_totals

class StepFunction:
    """
    A callable object that acts as a fast, piecewise constant step function
    for evenly spaced time series data.
    """
    def __init__(self, start_time, time_step_duration, rate_values):
        self.start_time = start_time
        self.time_step_duration = time_step_duration
        self.rate_values = np.array(rate_values)
        self.num_steps = len(rate_values)

    def __call__(self, t):
        """
        This makes the object callable, e.g., func(t).
        It finds the correct index for time 't' and returns the corresponding rate.
        """
        # If t is outside the defined time range, return 0
        if t < self.start_time or t >= self.start_time + self.num_steps * self.time_step_duration:
            return 0.0

        # Calculate the index for the time step
        # This is extremely fast because the steps are uniform.
        index = int((t - self.start_time) / self.time_step_duration)
        
        # Clamp the index to be within the valid range of the array
        index = min(index, self.num_steps - 1)
        
        return self.rate_values[index]

def process_species_terms(scenario_props):
    """
    Process species terms (launch, PMD, drag) for all species in the scenario.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        tuple: (full_lambda, full_Cdot_PMD, drag_term_upper, drag_term_cur)
    """
    t = sp.symbols('t')
    species_list = [species for group in scenario_props.species.values() for species in group]
    
    full_Cdot_PMD = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
    full_lambda = []
    drag_term_upper = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
    drag_term_cur = sp.zeros(scenario_props.n_shells, scenario_props.species_length)

    # Process each species
    for i, species in enumerate(species_list):
        # Launch
        lambda_expr = species.launch_func(scenario_props.scen_times, scenario_props.HMid, species, scenario_props)
        full_lambda.append(lambda_expr)

        # Post mission Disposal
        Cdot_PMD = species.pmd_func(t, scenario_props.HMid, species, scenario_props)
        full_Cdot_PMD[:, i] = Cdot_PMD

        # Drag
        try:
            # Use different drag function based on time-dependent density setting
            if scenario_props.time_dep_density:
                from ..drag.drag import drag_func_exp_time_dep
                [upper_term, current_term] = drag_func_exp_time_dep(t, scenario_props.HMid, species, scenario_props)
            else:
                [upper_term, current_term] = species.drag_func(t, scenario_props.HMid, species, scenario_props)
            drag_term_upper[:, i] = upper_term
            drag_term_cur[:, i] = current_term
        except:
            continue
    
    return full_lambda, full_Cdot_PMD, drag_term_upper, drag_term_cur

def process_collision_terms(scenario_props):
    """
    Process collision terms for both elliptical and circular scenarios.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        tuple: (equations, collision_terms, full_coll_sink, full_coll_source)
    """
    if scenario_props.elliptical:
        return process_elliptical_collisions(scenario_props)
    else:
        return process_circular_collisions(scenario_props)

def process_elliptical_collisions(scenario_props):
    """
    Process collision terms for elliptical scenarios.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        tuple: (equations, collision_terms, full_coll_sink, full_coll_source)
    """
    collision_terms = []
    full_coll_sink = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
    full_coll_source = sp.zeros(scenario_props.n_shells, scenario_props.species_length)

    # Determine debris insertion range
    debris_species = [spc for spc in scenario_props.species['debris']]
    if len(debris_species) > 0:
        first_deb_name = debris_species[0].sym_name
        deb_start_idx = next((j for j, spc in enumerate([spc for grp in scenario_props.species.values() for spc in grp])
                              if spc.sym_name == first_deb_name), None)
        deb_len = len(debris_species)
    else:
        deb_start_idx, deb_len = None, 0

    for cp in scenario_props.collision_pairs:
        # indices of the two active species
        s1_idx = scenario_props.species_names.index(cp.species1.sym_name)
        s2_idx = scenario_props.species_names.index(cp.species2.sym_name)

        # cp.eqs is an (n_shells x species_length) matrix
        eqs = cp.eqs

        # Build sinks matrix with contributions only in active species columns
        sinks = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
        sinks[:, s1_idx] = eqs[:, s1_idx]
        sinks[:, s2_idx] = eqs[:, s2_idx]

        # Build sources matrix in debris columns
        sources = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
        if deb_len > 0 and deb_start_idx is not None:
            sources[:, deb_start_idx:deb_start_idx + deb_len] = eqs[:, deb_start_idx:deb_start_idx + deb_len]

        # Accumulate
        full_coll_sink = full_coll_sink + sinks
        full_coll_source = full_coll_source + sources

        # Store term for RHS use
        term = SymbolicCollisionTerm(
            s1_idx=s1_idx,
            s2_idx=s2_idx,
            eqs_sources=sources,
            eqs_sinks=sinks,
            fragment_spread_totals=getattr(cp, 'fragments_sd', None)
        )
        collision_terms.append(term)

    equations = scenario_props.full_Cdot_PMD
    return equations, collision_terms, full_coll_sink, full_coll_source

def process_circular_collisions(scenario_props):
    """
    Process collision terms for circular scenarios.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        tuple: (equations, collision_terms, full_coll_sink, full_coll_source)
    """
    full_coll = sp.zeros(scenario_props.n_shells, scenario_props.species_length)
    
    for i in scenario_props.collision_pairs:
        full_coll += i.eqs

    equations = sp.zeros(scenario_props.n_shells, scenario_props.species_length)      
    equations = scenario_props.full_Cdot_PMD + full_coll
    
    return equations, None, None, None

def process_drag_and_density(scenario_props, drag_term_upper, drag_term_cur):
    """
    Process drag terms with density considerations.
    
    Args:
        scenario_props: ScenarioProperties instance
        drag_term_upper: Upper drag terms matrix
        drag_term_cur: Current drag terms matrix
        
    Returns:
        tuple: (full_drag, sym_drag)
    """
    if not scenario_props.time_dep_density: 
        # Take the shell altitudes, this will be n_shells + 1
        rho = scenario_props.density_model(0, scenario_props.R0_km, scenario_props.species, scenario_props)
        rho_reshape = rho.reshape(-1, 1) # Convert to column vector
        rho_mat = np.tile(rho_reshape, (1, scenario_props.species_length)) 
        rho_mat = sp.Matrix(rho_mat)
        
        # Second to last row
        upper_rho = rho_mat[1:, :]
        
        # First to penultimate row (mimics rho_mat(1:end-1, :))
        current_rho = rho_mat[:-1, :]

        drag_upper_with_density = drag_term_upper.multiply_elementwise(upper_rho)
        drag_cur_with_density = drag_term_cur.multiply_elementwise(current_rho)
        full_drag = drag_upper_with_density + drag_cur_with_density
        sym_drag = True
    else:
        # Don't add drag if time dependent density, this will be added during integration due to time dependent density
        full_drag = drag_term_upper + drag_term_cur
        sym_drag = False
    
    return full_drag, sym_drag

def process_integrated_indicators(scenario_props):
    """
    Process integrated indicator variables.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        tuple: (num_integrated_indicator_vars, updated_full_lambda)
    """
    if not hasattr(scenario_props, 'integrated_indicator_var_list'):
        return 0, scenario_props.full_lambda
    
    integrated_indicator_var_list = scenario_props.integrated_indicator_var_list
    for ind_var in integrated_indicator_var_list:
        if not ind_var.eqs:
            ind_var = scenario_props.make_indicator_eqs(ind_var)

    num_integrated_indicator_vars = 0
    end_indicator_idxs = len(scenario_props.xdot_eqs)

    for ind_var in integrated_indicator_var_list:
        num_add_indicator_vars = len(ind_var.eqs)
        num_integrated_indicator_vars += num_add_indicator_vars

        start_indicator_idxs = end_indicator_idxs + 1
        end_indicator_idxs = start_indicator_idxs + num_add_indicator_vars - 1
        ind_var.indicator_idxs = list(range(start_indicator_idxs, end_indicator_idxs + 1))

        scenario_props.xdot_eqs = sp.Matrix.vstack(scenario_props.xdot_eqs, sp.Matrix(ind_var.eqs))

    # Update full_lambda if needed
    updated_full_lambda = scenario_props.full_lambda.copy()
    if not scenario_props.sym_lambda:
        indicator_pad = [lambda x, t: 0] * num_integrated_indicator_vars
        updated_full_lambda.extend(indicator_pad)
    
    return num_integrated_indicator_vars, updated_full_lambda

def prepare_launch_functions(scenario_props):
    """
    Prepare launch rate functions for integration.
    
    Args:
        scenario_props: ScenarioProperties instance
        
    Returns:
        list: List of launch rate functions
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    launch_rate_functions = []

    if not scenario_props.baseline:
        for rate_array in scenario_props.full_lambda_flattened:
            try: 
                if rate_array is not None:
                    clean_rate_array = np.array(rate_array)
                    clean_rate_array[np.isnan(clean_rate_array)] = 0  # Replace any NaN values with 0
                    clean_rate_array[np.isinf(clean_rate_array)] = 0  # Replace any infinity values with 0

                    # Use interpolation
                    interp_func = interp1d(scenario_props.scen_times, clean_rate_array, 
                                        kind='cubic',  # 'linear', 'cubic'
                                        bounds_error=False, 
                                        fill_value=0)
                    launch_rate_functions.append(interp_func)
                else:
                    # If there are no launches, create a simple lambda that always returns 0
                    launch_rate_functions.append(lambda t: 0.0)
            except:
                launch_rate_functions.append(lambda t: 0.0)
    
    return launch_rate_functions

# Constants
hours = 3600.0
days = 24.0 * hours
years = 365.25 * days

def get_dadt(a_current, e_current, p):
        re   = p['req']
        mu   = p['mu']
        n0   = np.sqrt(mu) * a_current ** -1.5
        a_minus_re = a_current - re
        rho_0 = densityexp(a_minus_re) * 1e9  # kg/km^3
        # C_0 = max((param['Bstar']/(1e6*0.157))*rho_0,1e-20)
        C_0   = max(0.5 * p['Bstar'] * rho_0, 1e-20)
        
        beta = (np.sqrt(3)/2)*e_current
        ang  = np.arctan(beta)
        sec2 = 1.0 / np.cos(ang) ** 2
        return -(4 / np.sqrt(3)) * (a_current**2 * n0 * C_0 / e_current) * np.tan(ang) * sec2

def get_dedt(a_current, e_current, p):
    re   = p['req']
    mu   = p['mu']
    n0   = np.sqrt(mu) * a_current ** -1.5
    beta = (np.sqrt(3)/2) * e_current
    a_minus_re = a_current - re
    rho_0 = densityexp(a_minus_re) * 1e9  # kg/km^3
    # C_0 = max((param['Bstar']/(1e6*0.157))*rho_0,1e-20)
    C_0   = max(0.5 * p['Bstar'] * rho_0, 1e-20)

    sec2 = 1.0 / np.cos(np.arctan(beta)) ** 2
    return -e_current * n0 * a_current * C_0 * sec2