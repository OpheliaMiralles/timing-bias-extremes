import os

import numpy as np
from copulae import GumbelCopula

from conditioning_methods import ConditioningMethod
from pykelihood.multivariate import Copula, Multivariate
from pykelihood_distributions import GEV_reparametrized
from simulations import SimulationForTwoCorrelatedVariables, excel_from_dict
from timing_bias import StoppingRule

SIM_PATH = os.getenv('SIM_PATH', None)

n_iter = 1000
historical_sample_size = 50
nb_years = np.logspace(np.log10(20), np.log10(2000), 8).astype(int).tolist()
if __name__ == '__main__':
    theta = 2
    r = (1 / 0.2) * ((-np.log(1 - 1 / 200)) ** (-0.2) - 1)
    marg1 = GEV_reparametrized(r=r, shape=0.2)
    marg2 = GEV_reparametrized(r=r, shape=0.2)
    copula = Copula(GumbelCopula, theta=theta)
    mv = Multivariate(marg1, marg2, copula=copula)
    name = f'{n_iter}_iter_biv_theta{theta}_pos'
    conditioning_rules = [('Independant', ConditioningMethod.no_conditioning),
                          ('Including Extreme', ConditioningMethod.including_extreme_using_correlated_distribution),
                          ("Excluding Extreme", ConditioningMethod.excluding_extreme_using_correlated_distribution),
                          ('Cond. Including Extreme', ConditioningMethod.full_conditioning_using_correlated_distribution),
                          ('Cond. Excluding Extreme', ConditioningMethod.conditioning_excluding_extreme_using_correlated_distribution)]

    joint = GumbelCopula(theta=theta)
    historical_sample = joint.random(historical_sample_size)
    u1, u2 = historical_sample[:, 0], historical_sample[:, 1]
    m1, m2 = marg1.inverse_cdf(u1), marg2.inverse_cdf(u2)
    historical_sample = np.array(list(zip(m1, m2)))
    sim = SimulationForTwoCorrelatedVariables(marg1, marg2, joint, historical_sample, n_iter,
                                              stopping_rule_func=StoppingRule.fixed_to_k,
                                              conditioning_rules=conditioning_rules, return_periods_for_threshold=nb_years)
    excel_from_dict(sim.rl_estimates, sim.CI, sim.true_return_level, name, sim.above_threshold)
