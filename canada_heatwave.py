import os
from functools import partial

import matplotlib
import pykelihood.distributions
from pykelihood.distributions import GEV
from pykelihood.kernels import linear
from pykelihood.parameters import ConstantParameter
from pykelihood.profiler import Profiler

import data
from conditioning_methods import ConditioningMethod
from data import portland, seattle
from pykelihood_distributions import GEV_reparametrized_loc, GEV_reparametrized_p
from timing_bias import StoppingRule
from utils import parametric_confidence_interval, bootstrap_confidence_interval

matplotlib.rcParams['text.usetex'] = True
path_to_directory = os.getenv("CANADA_DATA") or "./data"
import matplotlib.pyplot as plt

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

warnings.filterwarnings('ignore')
optim = 'BFGS'

stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None,
                         names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
stations = stations[((stations.LATITUDE == seattle[0]) & (stations.LONGITUDE == seattle[-1]))
                    | ((stations.LATITUDE == portland[0]) & (stations.LONGITUDE == portland[-1]))]


def fit_gev_Tx_with_trend(x, y, rl=None):
    reparam = True if rl is not None else False
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = -0.5
    gev = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    if not reparam:
        return gev
    fit = GEV_reparametrized_loc(p=1 / rl, r=linear(x=x),
                                 scale=gev.scale()).fit_instance(y, x0=(
        gev.isf(1 / rl).mean(), alpha_init, gev.scale(), gev.shape()))
    return fit


def fit_gev_Tx_with_trend_p(x, y, r=None):
    reparam = True if r is not None else False
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = -0.5
    gev = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    if not reparam:
        return gev
    fit = GEV_reparametrized_p(loc=gev.loc, p=0, shape=gev.shape(), r=r).fit_instance(y, x0=(
        mu0_init, alpha_init, gev.sf(r).mean(), gev.shape()))
    return fit


def fit_gev_Tx_without_trend(y, rl=None):
    reparam = True if rl is not None else False
    gev = GEV.fit(y, x0=(y.mean(), y.std(), 0.))
    if reparam and (rl is not None):
        r = gev.isf(1 / rl)
        return GEV_reparametrized_loc(p=1 / rl, shape=gev.shape(), scale=gev.scale(), r=r).fit_instance(y)
    else:
        return gev


def plot_loc_and_rl(x, y, ci_type='parametric'):
    level = 46.7
    logrange = np.logspace(np.log10(1 + 1e-2), np.log10(10000), 50)
    year_used_for_comparison = 1951
    gev_fit = fit_gev_Tx_with_trend(x, y)
    sorted_y = y.sort_values()
    y_ref = y - gev_fit.loc() + gev_fit.loc().loc[year_used_for_comparison]
    y_now = y - gev_fit.loc() + gev_fit.loc().iloc[-1]
    gev_past = GEV(loc=gev_fit.loc().loc[year_used_for_comparison], scale=gev_fit.scale(), shape=gev_fit.shape())
    gev_now = GEV(loc=gev_fit.loc().iloc[-1], scale=gev_fit.scale(), shape=gev_fit.shape())
    loc_ci_past = Profiler(gev_past, y_ref, inference_confidence=0.95).confidence_interval_bs('loc')
    loc_ci_now = Profiler(gev_now, y_now, inference_confidence=0.95).confidence_interval_bs('loc')
    theo_past = gev_past.isf(1 / logrange)
    theo_now = gev_now.isf(1 / logrange)
    if ci_type == 'bootstrap':
        lower_ref, upper_ref = bootstrap_confidence_interval(y_ref)
        lower_now, upper_now = bootstrap_confidence_interval(y_now)
    elif ci_type == "parametric":
        lower_ref, upper_ref = parametric_confidence_interval(GEV_reparametrized_loc(), y=y_ref,
                                                              range_x=1 / logrange,
                                                              string_metric='p',
                                                              x0=np.array(
                                                                  [[r, gev_fit.scale(), gev_fit.shape()] for r in
                                                                   theo_past]))
        lower_now, upper_now = parametric_confidence_interval(GEV_reparametrized_loc(), y=y_now,
                                                              range_x=1 / logrange,
                                                              string_metric='p',
                                                              x0=np.array(
                                                                  [[r, gev_fit.scale(), gev_fit.shape()] for r in
                                                                   theo_now]))
    else:
        raise ValueError("ci_type must be either bootstrap or parametric")
    # plot
    l1, l2 = ('a', "b")
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8))
    ax1.plot(x, gev_fit.loc(), color="r", label="GEV location parameter")
    ax1.plot(x, gev_fit.loc() + gev_fit.scale(), color="r", linewidth=0.8)
    ax1.plot(x, gev_fit.loc() + 2 * gev_fit.scale(), color="r", linewidth=0.8)
    ax1.vlines(x.min(), *loc_ci_past, color="r")
    ax1.vlines(x.max(), *loc_ci_now, color="r")
    ax1.scatter(x, y, marker='x', color="k", s=10)
    ax1.set_title(f"{l1}) TXx Portland 1950-2023")
    ax1.set_xlabel('Global mean surface temperature (smoothed)')
    ax1.set_xlim(-0.4, 1)
    for ax in (ax1, ax2):
        ax.set_ylabel('TXx ($^\circ$C)')
    n = sorted_y.size
    real_rt = 1 / (1 - np.arange(0, n) / n)
    ax2.set_title(f"{l2}) Estimated return levels in Portland")
    ax2.scatter(real_rt, y_ref.sort_values(), s=15, marker='x', color='royalblue')
    ax2.scatter(real_rt, y_now.sort_values(), s=15, marker='x', color='r')
    rp_2021_in_2021 = 1 / gev_now.sf(level)
    ax2.plot(logrange, theo_past, color="royalblue", label=f"GEV fit {year_used_for_comparison}")
    ax2.plot(lower_ref, color="royalblue", ls='-.')
    ax2.plot(upper_ref, color="royalblue", ls='-.')
    ax2.plot(logrange, theo_now, color="r", label='GEV fit 2021')
    ax2.plot(lower_now, color="r", ls='-.')
    ax2.plot(upper_now, color="r", ls='-.')
    ax2.annotate(int(rp_2021_in_2021), xy=(rp_2021_in_2021, ax2.get_ylim()[0]),
                 xytext=(-3, 0), textcoords="offset points",
                 horizontalalignment="right", fontsize=16,
                 verticalalignment="bottom", color='r')
    ax2.hlines(level, ax2.get_xlim()[0], ax2.get_xlim()[-1], color='goldenrod', linewidth=0.8, label='Heatwave 2023')
    ylims = ax2.get_ylim()
    ax2.vlines(rp_2021_in_2021, ylims[0], level, color='r', linewidth=0.8, linestyle='--')
    ax2.set_xlim(logrange[0], logrange[-1])
    ax2.set_xlabel('return period (years)')
    ax2.legend(loc='best')
    ax1.legend(loc='best')
    ax2.set_xscale('log')
    ax2.set_ylim(*ylims)
    fig.show()
    return fig


def get_std_cond_fits_for_year_and_event(x, y, year, event=41.7, including=True):
    cov = x if including else x.loc[:2020]
    tx = y if including else y.loc[:2020]
    cm = ConditioningMethod.full_conditioning_including_extreme if including else ConditioningMethod.full_conditioning_excluding_extreme
    gev_fit = fit_gev_Tx_with_trend_p(cov, tx, r=event)
    location = gev_fit.loc(x).loc[year]
    y_ref = tx - gev_fit.loc() + location
    dist = GEV_reparametrized_p(
        loc=location, p=gev_fit.p(), shape=gev_fit.shape(),
        r=event)
    # fit with Nelder-Mead first to get a decent start point for the fit with BFGS
    gev_fit_year = dist.fit_instance(y_ref)
    # shape should not change with optimization method
    std_fit_year = gev_fit_year.fit_instance(y_ref, shape=gev_fit_year.shape(), scipy_args={"method": optim})
    historical_sample_size = len([i for i in y.index if i <= 2010])
    sr = StoppingRule(data=tx, k=event, distribution=gev_fit,
                      func=StoppingRule.fixed_to_k,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr.c, sr.N
    len_extreme_event = 1
    score = partial(cm,
                    historical_sample_size=historical_sample_size,
                    length_extreme_event=len_extreme_event,
                    threshold=thresh)
    cond_year = gev_fit_year.fit_instance(y_ref, score)
    gev_fit_cond_year = cond_year.fit_instance(y_ref, score, shape=cond_year.shape(), scipy_args={"method": optim})
    return std_fit_year, gev_fit_cond_year


def get_CI_for_rr_from_hessian(y, p1, p0, hess1, hess0, conf=0.95):
    hess_term1 = hess1[1, 1]
    hess_term0 = hess0[1, 1]
    var1 = hess_term1 / (len(y)) * 1 / p1 ** 2
    var0 = hess_term0 / (len(y)) * 1 / p0 ** 2
    total_var = var0 + var1
    up = pykelihood.distributions.Normal().ppf(1 - conf / 2)
    down = pykelihood.distributions.Normal().ppf(conf / 2)
    log_lb = np.log(p1 / p0) + down * np.sqrt(total_var)
    log_ub = np.log(p1 / p0) + up * np.sqrt(total_var)
    return [np.exp(log_lb), np.exp(log_ub)]


def get_risk_ratio_and_CI(x, y, event=41.7, year_past=1951, year_now=2021):
    dic = {}
    std_hist, cond_hist = get_std_cond_fits_for_year_and_event(x, y, year_past, event=event)
    ex_now, condex_now = get_std_cond_fits_for_year_and_event(x, y, year_now, event=event, including=False)
    ex_hist, condex_hist = get_std_cond_fits_for_year_and_event(x, y, year_past, event=event, including=False)
    std_now, cond_now = get_std_cond_fits_for_year_and_event(x, y, year_now, event=event)
    for (fit_hist, fit_now), name in zip(
            [(std_hist, std_now), (cond_hist, cond_now), (ex_hist, ex_now), (condex_hist, condex_now)],
            ["Standard", "COND", "Excluding", "CONDEX"]):
        p1 = fit_now.p()
        p0 = fit_hist.p()
        print(name, p0, p1)
        dic[name] = [p1 / p0] + get_CI_for_rr_from_hessian(y, p1, p0, fit_now.fit_result.hess_inv,
                                                           fit_hist.fit_result.hess_inv)
    return dic


if __name__ == '__main__':
    data = data.get_ghcn_daily_canada_annualmax().loc[:2021]
    portland = stations[stations.NAME == 'PORTLAND'].STATION.values[0]
    x = data.TEMPANOMALY_GLOB
    y = data[portland]
    plot_loc_and_rl(x, y)
    dic = get_risk_ratio_and_CI(x, y)
    print(dic)
