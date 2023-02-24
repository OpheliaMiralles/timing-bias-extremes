import os

import numpy as np

from conditioning_methods import ConditioningMethod
from pykelihood_distributions import GEV_reparametrized
from simulations import SimulationWithStoppingRuleAndConditioningForConditionedObservations, SimulationWithStoppingRuleAndConditioning, excel_from_dict
from timing_bias import StoppingRule

SIM_PATH = os.getenv('SIM_PATH', None)

if __name__ == '__main__':
    n_iter = 1000
    r = (1 / 0.2) * ((-np.log(1 - 1 / 200)) ** (-0.2) - 1)
    ref_distri = GEV_reparametrized(r=r, shape=0.2)
    rr = ref_distri.isf(1 / 200)
    nb_years = np.logspace(np.log10(20), np.log10(2000), 8).astype(int).tolist()
    conditioning_rules = [
        ('Standard', ConditioningMethod.no_conditioning),
        ("Excluding Extreme", ConditioningMethod.excluding_extreme),
        ('Cond. Including Extreme', ConditioningMethod.full_conditioning_including_extreme),
        ('Cond. Excluding Extreme', ConditioningMethod.conditioning_excluding_extreme)
    ]
    historical_sample_size = 50
    historical_sample = ref_distri.rvs(historical_sample_size)

    # # Fixed SR
    name = f'{n_iter}_iter_fixed_reparam'
    sim = SimulationWithStoppingRuleAndConditioning(reference=ref_distri, historical_sample=historical_sample, n_iter=n_iter,
                                                    stopping_rule_func=StoppingRule.fixed_to_k,
                                                    conditioning_rules=conditioning_rules,
                                                    return_periods_for_threshold=nb_years)

    excel_from_dict(sim.rl_estimates, sim.CI, sim.true_return_level, name)

    # Conditioned sampling
    name = f'{n_iter}_iter_conditioned_sample225_nh100_reparam'
    nb_years = np.logspace(np.log10(200), np.log10(20000), 8).astype(int).tolist()
    historical_sample_size = 100
    sim = SimulationWithStoppingRuleAndConditioningForConditionedObservations(reference=ref_distri,
                                                                              historical_sample=historical_sample,
                                                                              n_iter=n_iter,
                                                                              conditioning_rules=conditioning_rules,
                                                                              return_periods_for_threshold=nb_years)

    excel_from_dict(sim.rl_estimates, sim.CI, sim.true_return_level, name)
