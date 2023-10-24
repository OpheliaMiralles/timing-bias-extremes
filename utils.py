import numpy as np
import pandas as pd
from scipy.stats import genextreme

from pykelihood.profiler import Profiler

logrange = np.logspace(np.log10(1 + 1e-2), np.log10(10000), 100)


def bootstrap_confidence_interval(y: pd.Series, evt_distri=genextreme, confidence=0.05,
                                  metric=lambda x: x.ppf(1 - 1 / logrange)):
    n_points = len(y)
    shuffled_samples = np.array([y.sample(n_points, replace=True).values.flatten() for _ in range(500)])
    met_values = np.array([metric(evt_distri(*evt_distri.fit(n))) for n in shuffled_samples])
    lower_bound = np.quantile(met_values, confidence, axis=0)
    upper_bound = np.quantile(met_values, 1 - confidence, axis=0)
    return lower_bound, upper_bound


def parametric_confidence_interval(pykelihood_distribution_type, y, range_x, string_metric, x0):
    string_ci = 'r' if string_metric == 'p' else 'p'  # the one that we fix is not the one we compute CI for
    print("Building distributions")
    distributions = [pykelihood_distribution_type.fit_instance(y, x0=tuple(x), **{string_metric: n}) for n, x in
                     zip(range_x, x0)]
    valid_distributions_and_indices = np.array(
        [[1 / i, d] for i, d in zip(range_x, distributions) if np.isfinite(d.logpdf(y).sum())])
    profilers = [Profiler(d, y, inference_confidence=0.95) for d in valid_distributions_and_indices[:, 1]]
    print("Searching for interval bounds")
    cis = []
    for i, p in enumerate(profilers):
        print(f"Confidence interval {i + 1}/{len(profilers)}")
        cis.append(p.confidence_interval_bs(param=string_ci))
    cis = np.array(cis)
    lower, upper = cis[:, 0], cis[:, 1]
    indices = valid_distributions_and_indices[:, 0]
    return pd.Series(lower, index=indices).sort_values(), pd.Series(upper, index=indices).sort_values()
