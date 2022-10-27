import gc
import json
import os

import numpy as np
import pandas as pd
from copulae import GumbelCopula

from conditioning_methods import ConditioningMethod
from pykelihood.distributions import GEV
from simulations import SimulationForTwoCorrelatedVariables
from timing_bias import StoppingRule

SIM_PATH = os.getenv('SIM_PATH', None)

n_iter = 1000
historical_sample_size = 50
nb_years = np.logspace(np.log10(20), np.log10(1000), 8).astype(int).tolist()
if __name__ == '__main__':
    theta = 2
    name = f'{n_iter}_iter_biv_theta{theta}_df5_newdensity_trial2'
    conditioning_rules = [('Independant', ConditioningMethod.no_conditioning),
                          ('Including Extreme', ConditioningMethod.including_extreme_using_correlated_distribution),
                          ("Excluding Extreme", ConditioningMethod.excluding_extreme_using_correlated_distribution),
                          ('Cond. Including Extreme', ConditioningMethod.full_conditioning_using_correlated_distribution),
                          ('Cond. Excluding Extreme', ConditioningMethod.conditioning_excluding_extreme_using_correlated_distribution)]
    marg1 = GEV(shape=0.2)
    marg2 = GEV(shape=0.2)
    joint = GumbelCopula(theta=theta)
    historical_sample = joint.random(historical_sample_size)
    u1, u2 = historical_sample[:, 0], historical_sample[:, 1]
    m1, m2 = marg1.inverse_cdf(u1), marg2.inverse_cdf(u2)
    historical_sample = np.array(list(zip(m1, m2)))
    sim = SimulationForTwoCorrelatedVariables(marg1, marg2, joint, historical_sample, n_iter,
                                              stopping_rule_func=StoppingRule.fixed_to_k,
                                              conditioning_rules=conditioning_rules, return_periods_for_threshold=nb_years)
    relbias = pd.DataFrame.from_dict(sim.RelBias(), orient='columns')
    rrmse = pd.DataFrame.from_dict(sim.RRMSE(), orient='columns')
    cic = pd.DataFrame.from_dict(sim.CI_Coverage(), orient='columns')
    ciw = pd.DataFrame.from_dict(sim.CI_Width(), orient='columns')
    excel = pd.ExcelWriter(f'{SIM_PATH}/{name}.xlsx')
    for n, d in zip(['CIC', 'CIW', 'RelBias', 'RRMSE'], [cic, ciw, relbias, rrmse]):
        d.to_excel(excel, sheet_name=n)
    excel.save()
    gc.collect()
    json.dump(sim.rl_estimates, open(f'{SIM_PATH}/{name}_RLE.json', 'w'))
    json.dump(sim.CI, open(f'{SIM_PATH}/{name}_CI.json', 'w'))
    json.dump(sim.above_threshold, open(f'{SIM_PATH}/{name}_AT.json', 'w'))
