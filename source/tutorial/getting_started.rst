Getting Started
===============

This tutorial walks you through the basics of setting up `pySSEM`.

1. Install the package:
   .. code-block:: bash

      pip install pyssem

2. Configure your environment:
   - Ensure Python version 3.8 or higher.
   - Set up a JSON configuration file (e.g., `example-sim.json`).

3. Run the simulation:
   .. code-block:: python

      from pyssem.model import Model

      # Example simulation setup
      model = Model(start_date="01/03/2022", ...)
      model.run_model()