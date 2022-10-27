import gc
import json
import os

import matplotlib
import numpy as np
import pandas as pd

from conditioning_methods import ConditioningMethod
from pykelihood.distributions import GEV
from simulations import SimulationWithStoppingRuleAndConditioningForConditionedObservations, SimulationWithStoppingRuleAndConditioning
from timing_bias import StoppingRule

matplotlib.rcParams['text.usetex'] = True

SIM_PATH = os.getenv('SIM_PATH', None)

if __name__ == '__main__':
    n_iter = 1000
    ref_distri = GEV(shape=0.2)
    nb_years = np.logspace(np.log10(100), np.log10(20000), 10).astype(int).tolist()
    conditioning_rules = [('Standard', ConditioningMethod.no_conditioning),
                          ("Excluding Extreme", ConditioningMethod.excluding_extreme),
                          ('Cond. Including Extreme', ConditioningMethod.full_conditioning_including_extreme),
                          ('Cond. Excluding Extreme', ConditioningMethod.conditioning_excluding_extreme)]
    historical_sample_size = 100
    historical_sample = ref_distri.rvs(historical_sample_size)

    # Fixed SR
    name = f'{n_iter}_iter_conditioned_fixed'
    sim = SimulationWithStoppingRuleAndConditioning(reference=ref_distri, historical_sample=historical_sample, n_iter=n_iter,
                                                    stopping_rule_func=StoppingRule.fixed_to_k,
                                                    conditioning_rules=conditioning_rules,
                                                    return_periods_for_threshold=nb_years)

    cic = pd.DataFrame.from_dict(sim.CI_Coverage(), orient='columns')
    ciw = pd.DataFrame.from_dict(sim.CI_Width(), orient='columns')
    relbias = pd.DataFrame.from_dict(sim.RelBias(), orient='columns')
    rrmse = pd.DataFrame.from_dict(sim.RRMSE(), orient='columns')
    excel = pd.ExcelWriter(f'{SIM_PATH}/{name}.xlsx')
    for n, d in zip(['CIC', 'CIW', 'RelBias', 'RRMSE'], [cic, ciw, relbias, rrmse]):
        d.to_excel(excel, sheet_name=n)
    excel.save()

    # Conditioned sampling
    name = f'{n_iter}_iter_conditioned_sample275_nh100'
    sim = SimulationWithStoppingRuleAndConditioningForConditionedObservations(reference=ref_distri, historical_sample=historical_sample, n_iter=n_iter,
                                                                              conditioning_rules=conditioning_rules,
                                                                              return_periods_for_threshold=nb_years)

    cic = pd.DataFrame.from_dict(sim.CI_Coverage(), orient='columns')
    ciw = pd.DataFrame.from_dict(sim.CI_Width(), orient='columns')
    relbias = pd.DataFrame.from_dict(sim.RelBias(), orient='columns')
    rrmse = pd.DataFrame.from_dict(sim.RRMSE(), orient='columns')
    excel = pd.ExcelWriter(f'{SIM_PATH}/{name}.xlsx')
    for n, d in zip(['CIC', 'CIW', 'RelBias', 'RRMSE'], [cic, ciw, relbias, rrmse]):
        d.to_excel(excel, sheet_name=n)
    excel.save()
    json.dump(sim.rl_estimates, open(f'{SIM_PATH}/{name}_RLE.json', 'w'))
    json.dump(sim.CI, open(f'{SIM_PATH}/{name}_CI.json', 'w'))
    gc.collect()
