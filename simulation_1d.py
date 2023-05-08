import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import Generator, MT19937, SeedSequence

from conditioning_methods import ConditioningMethod
from pykelihood_distributions import GEV_reparametrized
from simulations import SimulationWithStoppingRuleAndConditioningForConditionedObservations, \
    RelBias_from_dict

SIM_PATH = os.getenv('SIM_PATH', None)
from scipy.stats.distributions import genextreme


def cas_simple(n_iter, random_state):
    hss = 20
    hs = genextreme.rvs(c=-0.2, size=hss, random_state=random_state)
    nb_years = np.logspace(np.log10(20), np.log10(2000), 8).astype(int).tolist()
    rls = defaultdict(list)
    for _ in tqdm(range(n_iter)):
        data = np.concatenate([hs,
                               genextreme.rvs(size=790, c=-0.2, random_state=random_state)])
        for x in nb_years:
            threshold = genextreme.isf(1 / x, c=-0.2)
            first_index_above_threshold = next((i for i, d in enumerate(data[10:]) if d >= threshold), None)
            N = hss + first_index_above_threshold + 1 if first_index_above_threshold is not None else len(data)
            data_stopped = data[:N]
            params = genextreme.fit(data_stopped, -0.2, loc=0, scale=1)
            rl = genextreme.isf(1 / 200, *params)
            rls[x].append(rl)
    return rls


def get_r(xi, period=200):
    if xi == 0:
        return -np.log(-np.log(1 - 1 / period))
    else:
        return (1 / xi) * ((-np.log(1 - 1 / period)) ** (-xi) - 1)


if __name__ == '__main__':
    n_iter = 1000
    xi = 0.2
    rs = np.random.default_rng(19)
    r = get_r(xi)
    ref_distri = GEV_reparametrized(r=r, shape=xi)
    rr = ref_distri.isf(1 / 200)
    nb_years = np.logspace(np.log10(20), np.log10(2000), 8).astype(int).tolist()
    conditioning_rules = [
        ('Standard', ConditioningMethod.no_conditioning),
        ("Excluding Extreme", ConditioningMethod.excluding_extreme),
        ('Cond. Including Extreme', ConditioningMethod.full_conditioning_including_extreme),
        ('Cond. Excluding Extreme', ConditioningMethod.conditioning_excluding_extreme)
    ]
    historical_sample_size = 10
    historical_sample = ref_distri.rvs(historical_sample_size, random_state=rs)

    # Fixed SR
    name = f'{n_iter}_iter_fixed_negxi'
    sim = SimulationWithStoppingRuleAndConditioning(reference=ref_distri, historical_sample=historical_sample, n_iter=n_iter,
                                                    stopping_rule_func=StoppingRule.fixed_to_k,
                                                    conditioning_rules=conditioning_rules,
                                                    return_periods_for_threshold=nb_years)

    excel_from_dict(sim.rl_estimates, sim.CI, sim.true_return_level, name)

    # Conditioned sampling
    name = f'{n_iter}_iter_conditioned_sample225_nh100_nullxi'
    nb_years = np.logspace(np.log10(200), np.log10(20000), 8).astype(int).tolist()
    historical_sample_size = 100
    sim = SimulationWithStoppingRuleAndConditioningForConditionedObservations(reference=ref_distri,
                                                                              historical_sample=historical_sample,
                                                                              n_iter=n_iter,
                                                                              conditioning_rules=conditioning_rules,
                                                                              return_periods_for_threshold=nb_years)

    excel_from_dict(sim.rl_estimates, sim.CI, sim.true_return_level, name)
