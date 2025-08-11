from utils.simulation.scen_properties import ScenarioProperties

class EllipticalOuputsToAltitudeBins:
    """
            For simulations where elliptical soecies exist, the X0 and the outputs will be in the form, n_sma, n_species and n_ecc_bins. 

            This function will convert from this format into altitude bins using the time in shell matrix.
    """

    def __init__(self, scenario_properties: ScenarioProperties):
        self.scenario_properties = scenario_properties

        self.initial_popula