.. _SEPs:

===========================
Space Environment Pathways
===========================

The Space Environment Pathways (SEPs) are predefined launch scenarios used in pySSEM to model the evolution of the space environment under different assumptions about future space activities. These scenarios are based on Aerospace Corporation research and provide a framework for simulating the impact of various launch behaviors on the orbital environment.

Overview
--------

The SEPs are designed to represent a range of possible futures for space activities, from minimal launches to highly intensive space demand. Each SEP defines a specific launch behavior, which is used to generate the initial population and future launch model for the simulation.

Available SEPs
--------------

The following SEPs are supported in pySSEM:

- **SEP 1**: No Future Launch

  Represents a scenario where no additional launches occur after the initial population. Used primarily for model comparison.

- **SEP 2**: Continuing Current Behaviors

  Assumes that current launch behaviors (not trends) continue without significant changes.

- **SEP 3 M**: Space Winter (Medium Sustainability Effort)

  Models a scenario where demand for space activity crashes after an initial period of rapid growth, with moderate efforts to improve sustainability in space activities.

- **SEP 3 H**: Space Winter (High Sustainability Effort)

  Represents the same Space Winter scenario, with high sustainability efforts.

- **SEP 4**: Strategic Rivalry

  International tensions lead to significant government-backed non-market demand, with backsliding on sustainability effort. Predicted increased commercial demand for satellite services fails to materialize.

- **SEP 5 M**: Commercial-driven Development (Medium Sustainability Effort)

  Significant commercial demands drive expansion of space traffic.

- **SEP 5 H**: Commercial-driven Development (High Sustainability Effort)

  Models the same scenario of commercial growth and high sustainability measures.

- **SEP 6 M**: Intensive Space Demand (Medium Sustainability Effort)

  A combination of international tensions and validation of commercial business cases leads to doubly intensive space demand.

- **SEP 6 H**: Intensive Space Demand (High Sustainability Effort)
  
  Simulates the same high-demand scenario with significant sustainability measures.

+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **Scenario**         | **Non-Market Demand for**  | **Market Demand for**       | **Level of Sustainability** |
|                      | **Space Services**         | **Space Services**          | **Effort**                  |
+======================+=============================+=============================+=============================+
| **SEP 1**:           | None                       | None                        | Current                     |
| No Future Launch     |                             |                             |                             |
+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **SEP 2**:           | Current                    | Current                     | Current                     |
| Continuing Current   |                             |                             |                             |
| Behaviors            |                             |                             |                             |
+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **SEP 3 M/H**:       | Low                        | Low                         | Med (primary) / High        |
| Space Winter         |                             |                             | (secondary)                 |
+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **SEP 4**:           | High                       | Low                         | Low                         |
| Strategic Rivalry    |                             |                             |                             |
+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **SEP 5 M/H**:       | Low                        | High                        | Med (primary) / High        |
| Commercial-driven    |                             |                             | (secondary)                 |
| Development          |                             |                             |                             |
+----------------------+-----------------------------+-----------------------------+-----------------------------+
| **SEP 6 M/H**:       | High                       | High                        | Med (secondary) / High      |
| Intensive Space      |                             |                             | (primary)                   |
| Demand               |                             |                             |                             |
+----------------------+-----------------------------+-----------------------------+-----------------------------+

How SEPs Work in pySSEM
-----------------------

The SEPs are implemented in the `initial_pop_and_launch` and `initial_pop_and_launch2` methods of the :class:`ScenarioProperties` class. These methods determine the initial population and future launch model based on the selected SEP.

1. **Initial Population**:  
   The initial population is generated using data from the selected SEP file. This data includes information about the objects' orbital parameters, species classification, and other properties.

2. **Future Launch Model (FLM)**:  
   The future launch model defines the launch rate and distribution of objects over time. It is calculated based on the selected SEP and the simulation parameters.

3. **Integration with Simulation**:  
   The SEP data is integrated into the simulation through the :func:`SEP_traffic_model` or :func:`ADEPT_traffic_model` functions, which process the SEP file and generate the required inputs for the simulation.

Using SEPs in pySSEM
--------------------

To use an SEP in your simulation, specify the desired SEP in the `launch_scenario` parameter of the `scenario_properties` section in your JSON configuration file. The launch function is the last parameter in the `scenario_properties` section:

.. code-block:: json

    {
        "scenario_properties": {
            "start_date": "01/03/2022",
            "simulation_duration": 100,
            "steps": 200,
            "min_altitude": 200,
            "max_altitude": 1400,
            "n_shells": 40,
            "launch_function": "Constant",
            "integrator": "BDF",
            "density_model": "static_exp_dens_func",
            "LC": 0.1,
            "v_imp": 10.0,
            "launch_scenario": "SEP 3 M"
        }
    }

The corresponding SEP file will be loaded, and the simulation will proceed based on the specified scenario.

References
----------

For more details on the SEPs and their development, refer to the research paper:  
`Development of Reference Scenarios and Supporting Inputs for Space Environment Modeling <https://www.researchgate.net/publication/385299836_Development_of_Reference_Scenarios_and_Supporting_Inputs_for_Space_Environment_Modeling>`_
