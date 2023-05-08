import resource
import time
import warnings
from functools import partial
from multiprocessing import Pool, Lock
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from conditioning_methods import ConditioningMethod
from pykelihood.distributions import Distribution, Uniform
from pykelihood.profiler import Profiler
from timing_bias import StoppingRule

warnings.filterwarnings('ignore')
SIM_PATH = "/Users/Boubou/Documents/GitHub/Venezuela-Data/Timing Bias/new95/"

USE_POOL = True

lock = Lock()


class SimulationEVD(object):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 return_period: int = 200,
                 sample_size: int = 790):
        self.ref_distri = reference
        self.n_iter = n_iter
        self.return_period = return_period
        self.true_return_level = self.ref_distri.isf(1 / return_period)
        self.sample_size = sample_size
        self.historical_sample = historical_sample
        self.rl_estimates = {}
        self.CI = {}

    def __call__(self, *args, **kwargs):
        pass

    def RRMSE(self):
        rle = self.rl_estimates
        true_rl = self.true_return_level
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        RRMSE = RRMSE_from_dict(rle, true_rl, above_threshold)
        return RRMSE

    def RelBias(self):
        rle = self.rl_estimates
        true_rl = self.true_return_level
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        RelBias = RelBias_from_dict(rle, true_rl, above_threshold)
        return RelBias

    def CI_Coverage(self):
        CI = self.CI
        true_rl = self.true_return_level
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        CIC = CI_Coverage_from_dict(CI, true_rl, above_threshold)
        return CIC

    def UB_Cerror(self):
        CI = self.CI
        true_rl = self.true_return_level
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        UBCE = UB_Cerror_from_dict(CI, true_rl, above_threshold)
        return UBCE

    def LB_Cerror(self):
        CI = self.CI
        true_rl = self.true_return_level
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        LBCE = LB_Cerror_from_dict(CI, true_rl, above_threshold)
        return LBCE

    def CI_Width(self):
        CI = self.CI
        if hasattr(self, 'above_threshold'):
            above_threshold = self.above_threshold
        else:
            above_threshold = None
        CIW = CI_Width_from_dict(CI, above_threshold)
        return CIW


class SimulationWithStoppingRuleAndConditioning(SimulationEVD):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 stopping_rule_func: Callable,
                 conditioning_rules: Sequence[Tuple[str, Callable]],
                 return_periods_for_threshold,
                 return_period: int = 200,
                 sample_size: int = 790):
        self.stopping_rule_func = stopping_rule_func
        self.conditioning_rules = conditioning_rules
        self.return_periods_for_threshold = return_periods_for_threshold
        self.historical_sample_size = len(historical_sample)
        super(SimulationWithStoppingRuleAndConditioning, self).__init__(reference,
                                                                        historical_sample,
                                                                        n_iter,
                                                                        return_period,
                                                                        sample_size)
        rl_estimates = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        CI = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        rs = np.random.default_rng(19)
        datasets = (pd.Series(np.concatenate([self.historical_sample,
                                              self.ref_distri.rvs(self.sample_size,
                                                                  random_state=rs)]))
                    for _ in range(self.n_iter))
        results = [self.run(ds) for ds in tqdm(datasets, total=self.n_iter)]
        for res in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k].append(value[0])
                    CI[name][k].append(value[1])
        self.rl_estimates = rl_estimates
        self.CI = CI

    def run(self, data):
        res = {name: {} for name, _ in self.conditioning_rules}
        for x in self.return_periods_for_threshold:
            if self.stopping_rule_func == StoppingRule.fixed_to_k:
                k = self.ref_distri.isf(1 / x)
            else:
                k = x
            stopping_rule = StoppingRule(data, self.ref_distri,
                                         k=k, historical_sample_size=self.historical_sample_size,
                                         func=self.stopping_rule_func)
            data_stopped = stopping_rule.stopped_data()
            std_conditioning = ["Standard", "Excluding Extreme"]
            crules = [(name, partial(crule, threshold=stopping_rule.threshold(), historical_sample_size=self.historical_sample_size)) \
                      for (name, crule) in self.conditioning_rules if name
                      not in std_conditioning] \
                     + [(name, crule) for (name, crule) in self.conditioning_rules if name in std_conditioning]
            for name, c in crules:
                res[name][x] = estimate_return_level(data_stopped, self.ref_distri, self.return_period, (name, c))
        return res


class SimulationWithStoppingRuleAndConditioningForConditionedObservations(SimulationEVD):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 conditioning_rules: Sequence[Tuple[str, Callable]],
                 return_periods_for_threshold,
                 return_period: int = 200,
                 sample_size: int = 225,
                 rg=np.random.default_rng(seed=19)):
        self.stopping_rule_func = StoppingRule.fixed_to_k
        self.conditioning_rules = conditioning_rules
        self.rs = rg
        self.return_periods_for_threshold = return_periods_for_threshold
        super(SimulationWithStoppingRuleAndConditioningForConditionedObservations, self).__init__(reference,
                                                                                                  historical_sample,
                                                                                                  n_iter,
                                                                                                  return_period,
                                                                                                  sample_size)
        rl_estimates = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        CI = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        if USE_POOL:
            pool = Pool(4)
            results = pool.map(self.__call__, self.return_periods_for_threshold)
            pool.close()
        else:
            results = [self(x) for x in self.return_periods_for_threshold]
        for res in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k] = list([p[0] for p in value])
                    CI[name][k] = list([p[1] for p in value])
        self.rl_estimates = rl_estimates
        self.CI = CI

    def __call__(self, x):
        time_start = time.time()
        res = {name: {x: []} for name, _ in self.conditioning_rules}
        k = self.ref_distri.isf(1 / x)
        for _ in range(self.n_iter):
            data = self.historical_sample
            sf_quantile = self.ref_distri.sf(k)
            f_quantile = self.ref_distri.cdf(k)
            u = Uniform()
            random_below_thresh = self.ref_distri.inverse_cdf(f_quantile * (u.rvs(self.sample_size - len(self.historical_sample) - 1, random_state=self.rs)))
            random_below_thresh = random_below_thresh[random_below_thresh < k]
            data = np.concatenate([data, random_below_thresh])
            data = pd.Series(np.concatenate([data, self.ref_distri.inverse_cdf(sf_quantile * u.rvs(1, random_state=self.rs) + f_quantile)]))
            stopping_rule = StoppingRule(data, self.ref_distri,
                                         k=k, historical_sample_size=len(self.historical_sample),
                                         func=self.stopping_rule_func)
            data_stopped = stopping_rule.stopped_data()
            std_conditioning = ["Standard", "Excluding Extreme"]
            crules = ([(name, partial(crule, threshold=stopping_rule.threshold()))
                       for (name, crule) in self.conditioning_rules if name
                       not in std_conditioning]
                      + [(name, crule) for (name, crule) in self.conditioning_rules if name in std_conditioning])
            f = partial(estimate_return_level,
                        *[data_stopped, self.ref_distri, self.return_period])
            for name, c in crules:
                res[name][x].append(f((name, c)))
                print(x, name, _)
        time_elapsed = time.time() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        return res


class SimulationForTwoCorrelatedVariables(SimulationEVD):
    def __init__(self, reference_lag: Distribution,
                 reference_lead: Distribution,
                 joint_distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 stopping_rule_func: Callable,
                 conditioning_rules: Sequence[Tuple[str, Callable]],
                 return_periods_for_threshold,
                 return_period: int = 200,
                 sample_size: int = 790):
        self.stopping_rule_func = stopping_rule_func
        self.conditioning_rules = conditioning_rules
        self.return_periods_for_threshold = return_periods_for_threshold
        self.historical_sample_size = len(historical_sample)
        super(SimulationForTwoCorrelatedVariables, self).__init__(reference_lag,
                                                                  historical_sample,
                                                                  n_iter,
                                                                  return_period,
                                                                  sample_size)
        tol = 1e-10
        self.stopping_distri = reference_lead
        self.joint_distri = joint_distribution
        rl_estimates = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        CI = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        h1, h2 = historical_sample[:, 0], historical_sample[:, 1]
        datasets_ref = []
        datasets_stop = []
        self.above_threshold = {x: [None] * self.n_iter for x in return_periods_for_threshold}
        for _ in range(self.n_iter):
            joint_rvs = self.joint_distri.random(self.sample_size)
            u1, u2 = joint_rvs[:, 0], joint_rvs[:, 1]
            u1 = u1[u1 < 1 - tol]
            u2 = u2[u2 < 1 - tol]
            m1, m2 = self.ref_distri.inverse_cdf(u1), self.stopping_distri.inverse_cdf(u2)
            datasets_ref.append(pd.Series(np.concatenate([h1, m1]), ))
            datasets_stop.append(pd.Series(np.concatenate([h2, m2]), ))
        self.datasets_ref = datasets_ref
        if USE_POOL:
            pool = Pool(4)
            results = pool.map(self.__call__, zip(range(self.n_iter), datasets_ref, datasets_stop))
            pool.close()
        else:
            results = [self((i, ds1, ds2)) for i, ds1, ds2 in zip(range(self.n_iter), datasets_ref, datasets_stop)]
        for res, _ in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k].append(value[0])
                    CI[name][k].append(value[1])
        for _, at in results:
            for k, by_idx in at.items():
                for idx, value in by_idx.items():
                    self.above_threshold[k][idx] = value
        self.rl_estimates = rl_estimates
        self.CI = CI

    def __call__(self, data):
        idx, ref_data, stopping_data = data
        time_start = time.time()
        res = {name: {} for name, _ in self.conditioning_rules}
        at = {x: {} for x in self.return_periods_for_threshold}
        for x in self.return_periods_for_threshold:
            if self.stopping_rule_func == StoppingRule.fixed_to_k:
                k = self.ref_distri.isf(1 / x)
            else:
                k = x
            stopping_rule = StoppingRule(stopping_data, self.stopping_distri,
                                         k=k, historical_sample_size=self.historical_sample_size,
                                         func=self.stopping_rule_func)
            data_stopped = ref_data.iloc[:stopping_rule.last_index()]
            above_threshold = data_stopped.iloc[-1] >= k
            at[x][idx] = int(above_threshold)
            stopping_data_stopped = stopping_data.iloc[:stopping_rule.last_index()]
            threshold = stopping_rule.threshold()
            margins = [self.stopping_distri.fit_instance(stopping_data_stopped),
                       self.stopping_distri.fit_instance(stopping_data_stopped),
                       self.stopping_distri.fit_instance(stopping_data_stopped.iloc[:-1]),
                       self.stopping_distri.fit_instance(stopping_data_stopped,
                                                         score=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                                       historical_sample_size=self.historical_sample_size, threshold=threshold)),
                       self.stopping_distri.fit_instance(stopping_data_stopped.iloc[:-1],
                                                         score=partial(ConditioningMethod.full_conditioning_excluding_extreme, historical_sample_size=self.historical_sample_size,
                                                                       threshold=threshold[:-1])),
                       ]
            std_conditioning = ["Independant"]
            crules = [(name, partial(crule, joint_structure=self.joint_distri,
                                     correlated_margin=m,
                                     threshold=threshold, stopping_data=stopping_data_stopped,
                                     historical_sample_size=self.historical_sample_size)) \
                      for (name, crule), m in zip(self.conditioning_rules, margins) if name
                      not in std_conditioning] \
                     + [(name, crule) for (name, crule) in self.conditioning_rules if name in std_conditioning]
            for name, c in crules:
                res[name][x] = estimate_return_level(data_stopped, self.ref_distri, self.return_period, (name, c))
        time_elapsed = time.time() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        return res, at


def RRMSE_from_dict(rle, true_return_level, above_threshold=None):
    print(f"Computing the mean RRMSE")
    RRMSE = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            RRMSE[case] = {}
            for name in rle.keys():
                RRMSE[case][name] = {}
                for k in rle[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [j for j in np.array(rle[name][k])[indices] if j is not None]
                    non_null = np.array(non_null)
                    sqrt = np.sqrt(np.mean((np.array(non_null) - true_return_level) ** 2))
                    RRMSE[case][name][k] = (1 / true_return_level) * sqrt
    else:
        for name in rle.keys():
            RRMSE[name] = {}
            for k in rle[name].keys():
                non_null = [j for j in np.array(rle[name][k]) if j is not None]
                non_null = np.array(non_null)
                sqrt = np.sqrt(np.mean((np.array(non_null) - true_return_level) ** 2))
                RRMSE[name][k] = (1 / true_return_level) * sqrt
    return RRMSE


def RelBias_from_dict(rle, true_return_level, above_threshold=None, metric=lambda x: np.mean(x)):
    print(f"Computing the mean relative bias")
    RelBias = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            RelBias[case] = {}
            for name in rle.keys():
                RelBias[case][name] = {}
                for k in rle[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [j for j in np.array(rle[name][k])[indices] if j is not None]
                    non_null = np.array(non_null)
                    relbias = (1 / true_return_level) * metric(non_null) - 1
                    RelBias[case][name][k] = relbias
    else:
        for name in rle.keys():
            RelBias[name] = {}
            for k in rle[name].keys():
                non_null = [j for j in np.array(rle[name][k]) if j is not None]
                non_null = np.array(non_null)
                relbias = (1 / true_return_level) * metric(non_null) - 1
                RelBias[name][k] = relbias
    return RelBias


def CI_Coverage_from_dict(CI, true_return_level, above_threshold=None):
    print(f"Computing the mean CI Coverage")
    CIC = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            CIC[case] = {}
            for name in CI.keys():
                CIC[case][name] = {}
                for k in CI[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [(lb, ub) for (lb, ub) in np.array(CI[name][k])[indices] if
                                lb is not None and ub is not None]
                    bools = [lb <= true_return_level <= ub
                             for lb, ub in non_null]
                    CIC[case][name][k] = sum(bools) / len(bools) if bools else 0.
    else:
        for name in CI.keys():
            CIC[name] = {}
            for k in CI[name].keys():
                non_null = [(lb, ub) for (lb, ub) in CI[name][k] if
                            lb is not None and ub is not None]
                bools = [lb <= true_return_level <= ub
                         for lb, ub in non_null]
                CIC[name][k] = sum(bools) / len(bools) if bools else 0.
    return CIC


def UB_Cerror_from_dict(CI, true_return_level, above_threshold=None):
    print(f"Computing the mean UB Coverage error")
    UBCE = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            UBCE[case] = {}
            for name in CI.keys():
                UBCE[case][name] = {}
                for k in CI[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [(lb, ub) for (lb, ub) in np.array(CI[name][k])[indices] if
                                lb is not None and ub is not None]
                    bools = [true_return_level >= ub
                             for lb, ub in non_null]
                    UBCE[case][name][k] = sum(bools) / len(bools) if bools else 1.
    else:
        for name in CI.keys():
            UBCE[name] = {}
            for k in CI[name].keys():
                non_null = [(lb, ub) for (lb, ub) in CI[name][k] if
                            lb is not None and ub is not None]
                bools = [true_return_level >= ub
                         for lb, ub in non_null]
                UBCE[name][k] = sum(bools) / len(bools) if bools else 1.
                UBCE[name][k] = UBCE[name][k]
    return UBCE


def LB_Cerror_from_dict(CI, true_return_level, above_threshold=None):
    print(f"Computing the mean LB Coverage error")
    LBCE = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            LBCE[case] = {}
            for name in CI.keys():
                LBCE[case][name] = {}
                for k in CI[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [(lb, ub) for (lb, ub) in np.array(CI[name][k])[indices] if
                                lb is not None and ub is not None]
                    bools = [true_return_level <= lb
                             for lb, ub in non_null]
                    LBCE[case][name][k] = sum(bools) / len(bools) if bools else 1.
    else:
        for name in CI.keys():
            LBCE[name] = {}
            for k in CI[name].keys():
                non_null = [(lb, ub) for (lb, ub) in CI[name][k] if
                            lb is not None and ub is not None]
                bools = [true_return_level <= lb
                         for lb, ub in non_null]
                LBCE[name][k] = sum(bools) / len(bools) if bools else 1.
                LBCE[name][k] = LBCE[name][k]
    return LBCE


def CI_Width_from_dict(CI, above_threshold=None):
    print(f"Computing the mean CI Width")
    CIW = {}
    if above_threshold:
        for i, case in zip([0, 1], ['A', 'B']):
            CIW[case] = {}
            for name in CI.keys():
                CIW[case][name] = {}
                for k in CI[name].keys():
                    indices = np.where(np.array(above_threshold[k]) == i)
                    non_null = [(lb, ub) for (lb, ub) in np.array(CI[name][k])[indices] if
                                lb is not None and ub is not None and ub - lb < 1000]
                    CIW[case][name][k] = np.mean(non_null)
    else:
        for name in CI.keys():
            CIW[name] = {}
            for k in CI[name].keys():
                non_null = [ub - lb for (lb, ub) in CI[name][k] if
                            lb is not None and ub is not None and ub - lb < 1000]
                CIW[name][k] = np.mean(non_null)
    return CIW


def excel_from_dict(rle, CI, true_return_level, filename, above_threshold=None):
    RB = RelBias_from_dict(rle, true_return_level, above_threshold)
    RMSE = RRMSE_from_dict(rle, true_return_level, above_threshold)
    CI_C = CI_Coverage_from_dict(CI, true_return_level, above_threshold)
    UB_C = UB_Cerror_from_dict(CI, true_return_level, above_threshold)
    LB_C = LB_Cerror_from_dict(CI, true_return_level, above_threshold)
    CI_W = CI_Width_from_dict(CI, above_threshold)
    if above_threshold:
        relbiasdic = {k: pd.DataFrame.from_dict(RB[k], orient='columns') for k in RB}
        rrmsedic = {k: pd.DataFrame.from_dict(RMSE[k], orient='columns') for k in RMSE}
        cicdic = {k: pd.DataFrame.from_dict(CI_C[k], orient='columns') for k in CI_C}
        ciwdic = {k: pd.DataFrame.from_dict(CI_W[k], orient='columns') for k in CI_W}
        updic = {k: pd.DataFrame.from_dict(UB_C[k], orient='columns') for k in UB_C}
        lowdic = {k: pd.DataFrame.from_dict(LB_C[k], orient='columns') for k in LB_C}
        for case in relbiasdic.keys():
            excel = pd.ExcelWriter(f'{SIM_PATH}/{filename}{case}.xlsx')
            cic = cicdic[case]
            up = updic[case]
            low = lowdic[case]
            relbias = relbiasdic[case]
            rrmse = rrmsedic[case]
            ciw = ciwdic[case]
            for n, d in zip(['CIC', 'UPC', 'LPC', 'CIW', 'RelBias', 'RRMSE'], [cic, up, low, ciw, relbias, rrmse]):
                d.to_excel(excel, sheet_name=n)
            excel.save()
    else:
        excel = pd.ExcelWriter(f'{SIM_PATH}/{filename}.xlsx')
        relbias = pd.DataFrame.from_dict(RB, orient='columns')
        rrmse = pd.DataFrame.from_dict(RMSE, orient='columns')
        cic = pd.DataFrame.from_dict(CI_C, orient='columns')
        up = pd.DataFrame.from_dict(UB_C, orient='columns')
        low = pd.DataFrame.from_dict(LB_C, orient='columns')
        ciw = pd.DataFrame.from_dict(CI_W, orient='columns')
        for n, d in zip(['CIC', 'UPC', 'LPC', 'CIW', 'RelBias', 'RRMSE'], [cic, up, low, ciw, relbias, rrmse]):
            d.to_excel(excel, sheet_name=n)
        excel.save()


def estimate_return_level(data: pd.Series,
                          distribution: Distribution,
                          return_period: int,
                          conditioning_rule: Tuple[str, Callable]):
    name, cr = conditioning_rule

    def return_level(distribution):
        return distribution.isf(1 / return_period)

    try:
        likelihood = Profiler(data=data,
                              distribution=distribution,
                              name=name,
                              score_function=cr,
                              single_profiling_param='r',
                              inference_confidence=0.95)
        rle = return_level(likelihood.optimum[0])
        rci = [None, None]#likelihood.confidence_interval_bs('r', precision=1e-2)
    except Exception:
        rle = return_level(distribution.fit_instance(data, score=cr))
        rci = [None, None]
    return rle, rci
