import pickle
import cProfile, pstats
from utils.simulation.scen_properties import ScenarioProperties
from utils.collisions.collisions_elliptical import create_elliptical_collision_pairs

# import the scenario-properties-baseline.pkl file
with open('../scenario-properties-baseline.pkl', 'rb') as f:
    scenario_properties = pickle.load(f)

# --- run under the profiler ---
pr = cProfile.Profile()
pr.enable()
print(scenario_properties.n_shells)
scenario_properties.add_collision_pairs(create_elliptical_collision_pairs(scenario_properties))
pr.disable()

# --- dump the top 30 callers by cumulative time ---
stats = pstats.Stats(pr).sort_stats('cumtime')
stats.print_stats(30)