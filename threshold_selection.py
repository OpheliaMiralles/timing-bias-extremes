from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from functools import partial
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from pykelihood.distributions import GPD, Exponential
from pykelihood.metrics import (
    AIC,
    BIC,
    Brier_score,
    bootstrap,
    crps,
    opposite_log_likelihood,
    qq_l1_distance,
    quantile_score,
)
from pykelihood.parameters import ParametrizedFunction


def threshold_selection_gpd_NorthorpColeman(
        data: Union[pd.Series, np.ndarray],
        thresholds: Union[Sequence, np.ndarray],
        plot=False,
):
    """
    Method based on a multiple threshold penultimate model,
    introduced by Northorp and Coleman in 2013 for threshold selection in extreme value analysis.
    Returns: table with likelihood computed using hypothesis of constant parameters h0, ll with h1, p_value of test, # obs
    and figure to plot as an option
    """
    if isinstance(data, pd.Series):
        data = data.rename("realized")
    elif isinstance(data, np.ndarray):
        data = pd.Series(data, name="realized")
    else:
        return TypeError("Observations should be in array or pandas series format.")
    if isinstance(thresholds, Sequence):
        thresholds = np.array(thresholds)

    fits = {}
    nll_ref = {}
    for u in thresholds:
        d = data[data > u]
        fits[u] = GPD.fit(d, loc=u)
        nll_ref[u] = opposite_log_likelihood(fits[u], d)

    def negated_ll(x: np.array, ref_threshold: float):
        tol = 1e-10
        sigma_init = x[0]
        if sigma_init <= tol:
            return 10 ** 10
        xi_init = x[1:]
        # It could be interesting to consider this parameter stability condition
        if len(xi_init[np.abs(xi_init) >= 1.0]):
            return 10 ** 10
        thresh = [u for u in thresholds if u >= ref_threshold]
        thresh_diff = pd.Series(
            np.concatenate([np.diff(thresh), [np.nan]]), index=thresh, name="w"
        )
        xi = pd.Series(xi_init, index=thresh, name="xi")
        sigma = pd.Series(
            sigma_init
            + np.cumsum(np.concatenate([[0], xi.iloc[:-1] * thresh_diff.iloc[:-1]])),
            index=thresh,
            name="sigma",
        )
        params_and_conditions = (
            pd.concat([sigma, xi, thresh_diff], axis=1)
            .assign(
                positivity_p_condition=lambda x: (1 + (x["xi"] * x["w"]) / x["sigma"])
                .shift(1)
                .fillna(1.0)
            )
            .assign(
                logp=lambda x: np.cumsum(
                    (1 / x["xi"]) * np.log(1 + x["xi"] * x["w"] / x["sigma"])
                )
                .shift(1)
                .fillna(0.0)
            )
            .reset_index()
            .rename(columns={"index": "lb"})
        )
        if np.any(params_and_conditions["positivity_p_condition"] <= tol):
            return 1 / tol
        thresh_for_pairs = np.concatenate([thresh, [data.max() + 1]])
        y_cut = (
            pd.concat(
                [
                    data,
                    pd.Series(
                        np.array(
                            pd.cut(
                                data, thresh_for_pairs, right=True, include_lowest=False
                            ).apply(lambda x: x.left)
                        ),
                        name="lb",
                        index=data.index,
                    ),
                ],
                axis=1,
            )
            .dropna()
            .merge(
                params_and_conditions.drop(columns=["w", "positivity_p_condition"]),
                on="lb",
                how="left",
            )
        )
        if (
                1 + y_cut["xi"] * (y_cut["realized"] - y_cut["lb"]) / y_cut["sigma"] <= tol
        ).any():
            return 1 / tol
        y_cut = y_cut.assign(
            nlogl=lambda x: x["logp"]
                            + np.log(x["sigma"])
                            + (1 + 1 / x["xi"])
                            * np.log(1 + x["xi"] * (x["realized"] - x["lb"]) / x["sigma"])
        )
        logl_per_interval = y_cut.groupby("lb").agg({"nlogl": "sum"})
        return logl_per_interval[np.isfinite(logl_per_interval)].sum()[0]

    par = {}
    results_dic = {}
    u_test = [thresholds[-1] + i for i in [5, 10, 20]]
    for u in thresholds:
        sigma_init, xi_1 = fits[u].optimisation_params
        xi_init = np.array([xi_1.value] * len(thresholds[thresholds >= u]))
        to_minimize = partial(negated_ll, ref_threshold=u)
        x0 = np.concatenate([[sigma_init.value], xi_init])
        params = minimize(
            to_minimize,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "fatol": 0.05},
        )
        print(f"Threshold {u}: ", params.message)
        mle = params.x
        nll = params.fun
        nll_h0 = to_minimize(x0)
        par[u] = mle
        delta = 2 * (nll_h0 - nll)
        df = len(thresholds[thresholds >= u]) - 1
        p_value = chi2.sf(delta, df=df)
        aic = AIC(fits[u], data[data > u])
        bic = BIC(fits[u], data[data > u])
        crpss = crps(fits[u], data[data > u])
        results_dic[u] = {
            "nobs": len(data[data > u]),
            "nll_h0": nll_h0,
            "nll_h1": nll,
            "pvalue": p_value,
            "aic": aic,
            "bic": bic,
            "crps": crpss,
        }
        for t in u_test:
            results_dic[u][f"bs_{int(t)}"] = Brier_score(
                fits[u], data[data > u], threshold=t
            )
        for q in [0.9, 0.95]:
            results_dic[u][f"qs_{int(q * 100)}"] = quantile_score(
                fits[u], np.quantile(data[data > u], q), quantile=q
            )
    results = pd.DataFrame.from_dict(results_dic, orient="index")
    if not plot:
        return results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(results.index, results["pvalue"], color="navy", label="p-value")
    ax.scatter(results.index, results["pvalue"], marker="x", s=8, color="navy")
    ax.legend(loc="upper center", title="LR test")
    ax2 = ax.twinx()
    if results["crps"].any():
        results = results.assign(
            rescaled_crps=lambda x: x["crps"] * (x["qs_90"].mean() / x["crps"].mean())
        )
        ax2.plot(
            results.dropna(subset=["pvalue"])["rescaled_crps"],
            c="royalblue",
            label="CRPS",
        )
    for f, c in zip([90, 95], ["salmon", "goldenrod"]):
        ax2.plot(
            results.dropna(subset=["pvalue"])[f"qs_{f}"],
            c=c,
            label=r"${}\%$ Quantile Score".format(f),
        )
    ax2.set_ylabel("GoF scores")
    ax2.legend(title="GoF scores")
    ax.set_title("Threshold Selection plot based on LR test")
    ax.set_xlabel("threshold")
    ax.set_ylabel("p-value")
    fig.show()
    return results, fig


def threshold_selection_GoF(
        data: Union[pd.Series, np.ndarray, pd.DataFrame],
        min_threshold: float,
        max_threshold: float,
        metric=qq_l1_distance,
        bootstrap_method=None,
        GPD_instance=None,
        data_column=None,
        plot=False,
):
    """
    Threshold selection method based on goodness of fit maximisation.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process. arXiv preprint arXiv:2102.00884.
    :return: table with results and plot (if selected)
    """
    from copy import copy

    if isinstance(data, pd.DataFrame):
        data_series = data[data_column]
        if data_column is None:
            raise AttributeError(
                "The data column is needed to perform the threshold selection on a DataFrame."
            )
    else:
        data_series = data

    if GPD_instance is None:
        GPD_instance = GPD()

    def update_covariates(GPD_instance, threshold):
        count = 0
        temp_gpd = copy(GPD_instance)
        for param in GPD_instance.param_dict:
            if isinstance(GPD_instance.__getattr__(param), ParametrizedFunction):
                # tracking number of functional parameters
                count += 1
                parametrized_param = getattr(GPD_instance, param)
                covariate = parametrized_param.x
                new_covariate = data[data[data_column] > threshold][covariate.name]
                temp_gpd = temp_gpd.with_params(
                    **{param: GPD_instance.loc.with_covariate(new_covariate)}
                )
        return temp_gpd, count

    def to_minimize(x):
        threshold = x[0]
        unit_exp = Exponential()
        if bootstrap_method is None:
            print("Performing threshold selection without bootstrap...")
            new_data = data_series[data_series > threshold]
            temp_gpd, count = update_covariates(GPD_instance, threshold)
            gpd_fit = temp_gpd.fit_instance(
                new_data, x0=GPD_instance.flattened_params[1:], loc=threshold
            )
            if count > 0:
                return metric(
                    distribution=unit_exp,
                    data=unit_exp.inverse_cdf(gpd_fit.cdf(new_data)),
                )
            else:
                return metric(distribution=gpd_fit, data=new_data)
        else:
            bootstrap_func = partial(bootstrap_method, threshold=threshold)
            new_metric = bootstrap(metric, bootstrap_func)
            return new_metric(unit_exp, data)

    threshold_sequence = np.linspace(min_threshold, max_threshold, 30)
    func_eval = np.array([to_minimize([t]) for t in threshold_sequence])
    nans_inf = np.isnan(func_eval) | ~np.isfinite(func_eval)
    func_eval = func_eval[~nans_inf]
    threshold_sequence = threshold_sequence[~nans_inf]

    optimal_thresh = threshold_sequence[np.where(func_eval == np.min(func_eval))[0]][0]
    func = np.min(func_eval)
    res = [optimal_thresh, func]

    if not plot:
        return res
    import matplotlib.pyplot as plt

    data_to_fit = data_series[data_series > optimal_thresh]
    optimal_gpd, count = update_covariates(GPD_instance, optimal_thresh)
    gpd_fit = optimal_gpd.fit_instance(data_to_fit)
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax = []
    for i in range(1):
        for j in range(2):
            axe = fig.add_subplot(gs[i, j])
            ax.append(axe)
    plt.subplots_adjust(wspace=0.2)
    ax0, ax1 = ax
    ax0.plot(threshold_sequence, func_eval, color="navy")
    ax0.scatter(threshold_sequence, func_eval, marker="x", s=8, color="navy")
    ax0.vlines(
        optimal_thresh,
        func,
        np.max(func_eval),
        color="royalblue",
        label="Optimal threshold",
    )
    ax0.legend()
    ax0.set_xlabel("threshold")
    ax0.set_ylabel(metric.__name__.replace("_", " "))

    if metric.__name__.startswith("qq"):
        if count > 0 or bootstrap_method is not None:
            from pykelihood.visualisation.utils import qq_plot_exponential_scale

            qq_plot_exponential_scale(gpd_fit, data_to_fit, ax=ax1)
        else:
            from pykelihood.visualisation.utils import qq_plot

            qq_plot(gpd_fit, data_to_fit, ax=ax1)
    elif metric.__name__.startswith("pp"):
        from pykelihood.visualisation.utils import pp_plot

        pp_plot(gpd_fit, data_to_fit, ax=ax1)
    fig.show()
    return res, fig
