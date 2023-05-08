import os
from functools import partial

import matplotlib

import pykelihood.distributions
from conditioning_methods import ConditioningMethod
from data import portland, seattle, get_ghcn_daily_canada_annualmax
from pykelihood.distributions import GEV
from pykelihood.kernels import linear
from pykelihood.parameters import ConstantParameter
from pykelihood.profiler import Profiler
from pykelihood_distributions import GEV_reparametrized_loc, GEV_reparametrized_p, GEV_reparametrized_rp
from timing_bias import StoppingRule

matplotlib.rcParams['text.usetex'] = True

path_to_directory = os.getenv("CANADA_DATA")
import matplotlib.pyplot as plt

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

warnings.filterwarnings('ignore')
stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
stations = stations[((stations.LATITUDE == seattle[0]) & (stations.LONGITUDE == seattle[-1]))
                    | ((stations.LATITUDE == portland[0]) & (stations.LONGITUDE == portland[-1]))]
optim = 'BFGS'


def fit_gev_Tx_with_trend(x, y, rl=None):
    reparam = True if rl is not None else False
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = -0.5
    gev = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    if not reparam:
        return gev
    fit = GEV_reparametrized_loc(p=1 / rl, r=linear(x=x),
                                 scale=gev.scale()).fit_instance(y, x0=(gev.isf(1 / rl).mean(), alpha_init, gev.scale(), gev.shape()))
    return fit


def fit_gev_Tx_with_trend_p(x, y, r=None):
    reparam = True if r is not None else False
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = -0.5
    gev = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    if not reparam:
        return gev
    fit = GEV_reparametrized_p(loc=gev.loc, p=0, shape=gev.shape(), r=r).fit_instance(y, x0=(mu0_init, alpha_init, gev.sf(r).mean(), gev.shape()))
    return fit


def fit_gev_Tx_with_trend_rp(x, y, r=None):
    reparam = True if r is not None else False
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = -0.5
    gev = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    if not reparam:
        return gev
    fit = GEV_reparametrized_rp(loc=gev.loc, m=0, shape=gev.shape(), r=r).fit_instance(y, x0=(mu0_init, alpha_init, 1 / gev.sf(r).mean(), gev.shape()))
    return fit


def fit_gev_Tx_without_trend(y, rl=None):
    reparam = True if rl is not None else False
    gev = GEV.fit(y, x0=(y.mean(), y.std(), 0.))
    if reparam and (rl is not None):
        r = gev.isf(1 / rl)
        return GEV_reparametrized_loc(p=1 / rl, shape=gev.shape(), scale=gev.scale(), r=r).fit_instance(y)
    else:
        return gev


def compute_alternative_profiles(fit, y, x, infconf=0.95, return_period=100):
    historical_sample_size = len([i for i in y.index if i <= 2010])
    sr = StoppingRule(data=y, k=200, distribution=fit, func=StoppingRule.variable_in_the_estimated_rl_with_trend_in_r,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr.c, sr.N
    len_extreme_event = 1
    if y.index.max() == 2021:
        full_cond = Profiler(distribution=fit, data=y, inference_confidence=infconf,
                             score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                    historical_sample_size=historical_sample_size,
                                                    length_extreme_event=len_extreme_event,
                                                    threshold=thresh),
                             name='Conditioning including extreme event')
        ex_fit = fit_gev_Tx_with_trend(x.loc[:2020], y.loc[:2020], rl=return_period)
        excluding = Profiler(distribution=ex_fit, data=y.loc[:2020],
                             name='Excluding extreme event', inference_confidence=infconf)
        thresh_ex = thresh[:-len_extreme_event]
        excluding_cond = Profiler(distribution=ex_fit, data=y.loc[:2020],
                                  score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                         historical_sample_size=historical_sample_size,
                                                         length_extreme_event=len_extreme_event,
                                                         threshold=thresh_ex),
                                  name='Conditioning excluding extreme event', inference_confidence=infconf)
    else:
        full_cond = None
        excluding = None
        ex_fit = fit
        thresh_ex = thresh
        excluding_cond = Profiler(distribution=ex_fit, data=y,
                                  score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                         historical_sample_size=historical_sample_size,
                                                         length_extreme_event=len_extreme_event,
                                                         threshold=thresh_ex),
                                  name='Conditioning excluding extreme event')
    return [excluding, excluding_cond, full_cond]


def compute_timevarying_profile_pairs(year, x, y, infconf=0.95, return_period=100):
    historical_sample_size = len([i for i in y.index if i <= 2010])
    fit = fit_gev_Tx_without_trend(y.loc[:max(year, 2021)], rl=return_period)
    sr = StoppingRule(data=y, k=200, distribution=fit, func=StoppingRule.variable_in_the_estimated_rl_with_trend_in_r,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr.c, sr.N
    fit_x = fit_gev_Tx_with_trend(x.loc[:year], y.loc[:year], rl=return_period)
    fit_fixed_trend = GEV_reparametrized_loc(p=1 / return_period, r=linear(a=fit_x.r.a, b=ConstantParameter(fit_x.r.b()), x=x.loc[:year]),
                                             scale=fit_x.scale(), shape=fit_x.shape()) if return_period is not None else fit_x
    if year >= 2021:
        crule_cond = partial(ConditioningMethod.full_conditioning_including_extreme,
                             threshold=thresh.copy(),
                             historical_sample_size=historical_sample_size)
    else:
        thresh_ex = thresh[:-(len(y.loc[:2021]) - len(y.loc[:year]))]
        crule_cond = partial(ConditioningMethod.full_conditioning_excluding_extreme,
                             threshold=thresh_ex,
                             historical_sample_size=historical_sample_size)
    # the conditional fit only uses years in common for Jodphur and Bikaner as well as complete info about values above and below the threshold in the Jodhpur series
    std = Profiler(distribution=fit_fixed_trend, data=y.loc[:year], inference_confidence=infconf,
                   name=f'Standard fit {year}', single_profiling_param='r_a')
    cond = Profiler(distribution=fit_fixed_trend, data=y.loc[:year], inference_confidence=infconf,
                    score_function=crule_cond,
                    name=f'Conditioned fit {year}', single_profiling_param='r_a')
    return [std, cond]


def get_std_cond_profiles_for_year_and_event(event, year, var='p'):
    if var=='p':
        fit = fit_gev_Tx_with_trend_p(x, y, r=event)
        fit_year = GEV_reparametrized_p(loc=ConstantParameter(fit.loc().loc[year]), p=fit.p(), shape=fit.shape(), r=event).fit_instance(y)
        fit_cond = GEV_reparametrized_p(loc=ConstantParameter(fit.loc().loc[year]), p=fit.p(), shape=fit.shape(), r=event).fit_instance(y, method=optim)

    else:
        fit = fit_gev_Tx_with_trend_rp(x, y, r=event)
        fit_year = GEV_reparametrized_rp(loc=ConstantParameter(fit.loc().loc[year]), m=fit.m(), shape=fit.shape(), r=event).fit_instance(y)
        fit_cond = GEV_reparametrized_rp(loc=ConstantParameter(fit.loc().loc[year]), m=fit.m(), shape=fit.shape(), r=event).fit_instance(y, method=optim)
    standard_profile = Profiler(distribution=fit_year, data=y, inference_confidence=0.66, optimization_method=optim)
    historical_sample_size = len([i for i in y.index if i <= 2010])
    sr = StoppingRule(data=y, k=200, distribution=fit, func=StoppingRule.variable_in_the_estimated_rl_with_trend_in_loc,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr.c, sr.N
    len_extreme_event = 1
    full = Profiler(distribution=fit_cond, data=y, inference_confidence=0.66, optimization_method=optim,
                    score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                           historical_sample_size=historical_sample_size,
                                           length_extreme_event=len_extreme_event,
                                           threshold=thresh),
                    name='Conditioning including extreme event')
    return standard_profile, full


def get_std_cond_ex_profiles_for_year_and_event(event, year, var='p'):
    if var=='p':
        fit = fit_gev_Tx_with_trend_p(x.loc[:2020], y.loc[:2020], r=event)
        fit_year = GEV_reparametrized_p(loc=ConstantParameter(fit.loc(x).loc[year]), p=fit.p(), shape=fit.shape(), r=event).fit_instance(y.loc[:2020])
        fit_cond = GEV_reparametrized_p(loc=ConstantParameter(fit.loc(x).loc[year]), p=fit.p(),
                                         shape=fit.shape(), r=event).fit_instance(y, method=optim)
    else:
        fit = fit_gev_Tx_with_trend_rp(x.loc[:2020], y.loc[:2020], r=event)
        fit_year = GEV_reparametrized_rp(loc=ConstantParameter(fit.loc(x).loc[year]), m=fit.m(), shape=fit.shape(), r=event).fit_instance(y.loc[:2020], method=optim)
        fit_cond = GEV_reparametrized_rp(loc=ConstantParameter(fit.loc(x).loc[year]), m=fit.m(),
                                                       shape=fit.shape(), r=event).fit_instance(y, method=optim)
    standard_profile = Profiler(distribution=fit_year, data=y.loc[:2020], inference_confidence=0.66)
    historical_sample_size = len([i for i in y.loc[:2020].index if i <= 2010])
    sr = StoppingRule(data=y.loc[:2020], k=200, distribution=fit, func=StoppingRule.variable_in_the_estimated_rl_with_trend_in_loc,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr.c, sr.N
    len_extreme_event = 1
    full = Profiler(distribution=fit_cond, optimization_method=optim,
                    data=y.loc[:2020], inference_confidence=0.66,
                    score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                           historical_sample_size=historical_sample_size,
                                           length_extreme_event=len_extreme_event,
                                           threshold=thresh),
                    name='Conditioning including extreme event')
    return standard_profile, full


def plot_loc_vs_anomaly(x, y):
    fit_non_reparam = fit_gev_Tx_with_trend(x, y)
    profile = Profiler(distribution=fit_non_reparam,
                       data=y, inference_confidence=0.95)
    opt = profile.optimum[0]
    locs = [opt.loc.with_covariate(np.array([i]))()[0] for i in x.sort_values()]
    scales = [opt.scale()] * len(locs)
    loc_scale = [loc + scale for (loc, scale) in zip(locs, scales)]
    loc_2scale = [loc + 2 * scale for (loc, scale) in zip(locs, scales)]
    df = pd.DataFrame([x.sort_values().values, locs, loc_scale, loc_2scale],
                      columns=x.index,
                      index=['tempanomaly', 'loc_par', 'loc_plus_sigma', 'loc_plus_2sigma']) \
        .T
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.set_xlim(-0.4, 1)
    ax.scatter(x, y, s=16, marker='x', color='black')
    ax.set_xlabel('Global mean surface temperature (smoothed)')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    ax.plot(df.tempanomaly, df.loc_par, color='red')
    ax.plot(df.tempanomaly, df.loc_plus_sigma, linewidth=0.2, color='red')
    ax.plot(df.tempanomaly, df.loc_plus_2sigma, linewidth=0.2, color='red')
    for xx, idx_y, i in zip([x.min(), x.max()], [x.argmin(), x.argmax()], [0, -1]):
        metric = lambda x: x.loc().iloc[idx_y]
        ci = profile.confidence_interval(metric)
        ax.vlines(xx, ci[0], ci[-1], color='red')
    fig.show()
    return fig


def plot_return_levels(x, y, level=46.7, condition=False):
    logrange = np.logspace(np.log10(1 + 1e-2), np.log10(10000), 20)
    sorted_y = y.sort_values()
    n = sorted_y.size
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    levs = []
    for color, xx, idx_y, i in zip(['royalblue', 'red'], [x.min(), x.loc[2021]], [x.argmin(), -1], [0, -1]):
        name = f'GEV shift fit {x.index[idx_y]}'
        cis = []
        rls = []
        for k in logrange:
            fit = fit_gev_Tx_with_trend(x, y, rl=k)
            fit_fixed_trend = GEV_reparametrized_loc(p=1 / k, r=linear(a=fit.r.a, b=ConstantParameter(fit.r.b()), x=x),
                                                     scale=fit.scale(), shape=fit.shape())
            if condition:
                historical_sample_size = len([i for i in y.index if i <= 2010])
                sr = StoppingRule(data=y, k=200, distribution=fit, func=StoppingRule.variable_in_the_estimated_rl_with_trend_in_r,
                                  historical_sample_size=historical_sample_size)
                thresh, N = sr.c, sr.N
                len_extreme_event = 1
                p = Profiler(distribution=fit_fixed_trend, data=y, inference_confidence=0.95,
                             score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                    historical_sample_size=historical_sample_size,
                                                    length_extreme_event=len_extreme_event,
                                                    threshold=thresh),
                             name='Conditioning including extreme event')
                fit_p = p.optimum[0]
            else:
                p = Profiler(fit_fixed_trend, y, inference_confidence=0.95)
                fit_p = fit
            rls.append(fit.isf(1 / k)[idx_y])
            print(f'Computing CI for k={k}')
            cis.append(p.confidence_interval(lambda x: x.isf(1 / k)[idx_y]))
        # return period for 2021 event
        std, cond = get_std_cond_profiles_for_year_and_event(46.7, x.index[idx_y])
        proba = cond.optimum[0].m() if condition else std.optimum[0].m()
        canada_level = proba
        levs.append(canada_level)
        lbs = [lb for (lb, ub) in cis]
        ubs = [ub for (lb, ub) in cis]
        try:
            ax.vlines(canada_level, 0, level, color=color, linewidth=0.6, linestyle='--')
            ax.annotate(int(canada_level), xy=(canada_level, 25),
                        xytext=(-3, 0), textcoords="offset points",
                        horizontalalignment="right",
                        verticalalignment="bottom", color=color)
            fig.show()
        except:
            pass
        ax.plot(logrange, rls, color=color, label=name, linewidth=0.6)
        ax.plot(logrange[3:], lbs[3:], color=color, linewidth=0.6, linestyle='--')
        ax.plot(logrange[3:], ubs[3:], color=color, linewidth=0.6, linestyle='--')
        real_rt = 1 / (1 - np.arange(0, n) / n)
        scaled_y = (y - fit_p.flattened_param_dict['r_b'].value * (x - x.iloc[idx_y])).sort_values()
        ax.scatter(real_rt, scaled_y, s=10, marker='x', color=color)
    ax.hlines(level, logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Observed 2021')
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    ax.set_xscale('log')
    ax.set_xlim(logrange[0], logrange[-1])
    ax.set_ylim(25, 55)
    return fig


def segment_plot(profile_dic, y, state, risk_ratio=False):
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    infconf = profile_dic[150][0].inference_confidence
    logrange = list(profile_dic.keys())
    refyear = 1980
    for k in logrange:
        i = 0
        profiles = profile_dic[k]
        for profile, color in zip(profiles,
                                  ['salmon', 'pink', 'navy', 'royalblue']):
            ci1, ci2 = profile.confidence_interval(lambda x: x.r().iloc[-1] / x.r().loc[refyear]) if risk_ratio else profile.confidence_interval(lambda x: x.r().iloc[-1])
            current_rl = profile.optimum[0].r().iloc[-1]
            rl = current_rl / profile.optimum[0].r().loc[refyear] if risk_ratio else current_rl
            if k == 150:
                ax.vlines(k + i, ci1, ci2, color=color, label=profile.name)
            else:
                ax.vlines(k + i, ci1, ci2, color=color)
            ax.scatter(k + i, rl, marker='x', color=color)
            i += 20
        fig.show()
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    leg = 'risk ratio $p_{2021}/p_{2000}$' if risk_ratio else 'daily max temperature ($^\circ$C)'
    title = 'Risk ratio' if risk_ratio else 'TXx'
    ax.set_ylabel(leg)
    fig.suptitle(f'{title} {state} {y.index.min()}-{y.index.max()} ({int(100 * infconf)}\% CI)')
    return fig


def fig_std_cond_comparison(dic, state, level):
    years = list(dic.keys())
    logrange = [100, 1000, 10000]
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)
    infconf = dic[years[0]][logrange[0]][0].inference_confidence

    def get_ci_from_trend_params_bounds(profile, year):
        lb, ub = profile.confidence_interval(lambda x: x.r().loc[year])
        rl = profile.optimum[0].r().loc[year]
        return rl, (lb, ub)

    for k, ax in zip(logrange, axes.flatten()):
        std_rls = []
        cond_rls = []
        std_lbs = []
        std_ubs = []
        cond_lbs = []
        cond_ubs = []
        for year in years:
            print(f'Plotting std cond comparison fig for {state} for year {year} and RL {k}')
            std_prof = dic[year][k][0]
            cond_prof = dic[year][k][1]
            for prof, rl, lb_list, ub_list in zip([std_prof, cond_prof],
                                                  [std_rls, cond_rls],
                                                  [std_lbs, cond_lbs],
                                                  [std_ubs, cond_ubs]):
                r, ci = get_ci_from_trend_params_bounds(prof, year)
                rl.append(r)
                lb_list.append(ci[0])
                ub_list.append(ci[-1])
        for i in [0, 5, -1]:
            if i == -1:
                j = years[i] - 0.4
            else:
                j = years[i]
            ax.vlines(j, std_lbs[i], std_ubs[i], color='salmon', linewidth=0.5)
            ax.hlines(std_ubs[i], j - 0.2, j + 0.2, color='salmon', linewidth=0.5)
            ax.hlines(std_lbs[i], j - 0.2, j + 0.2, color='salmon', linewidth=0.5)
        for i in [0, 5, -1]:
            if i == -1:
                j = years[i] - 0.2
            else:
                j = years[i] + 0.2
            ax.vlines(j, cond_lbs[i], cond_ubs[i], color='navy', linewidth=0.5)
            ax.hlines(cond_ubs[i], j - 0.2, j + 0.2, color='navy', linewidth=0.5)
            ax.hlines(cond_lbs[i], j - 0.2, j + 0.2, color='navy', linewidth=0.5)
        ax.plot(years, std_rls, color='salmon', label='Standard fit', linewidth=0.7)
        ax.scatter(years, std_rls, color='salmon', s=10, marker='x')
        ax.plot(years, cond_rls, color='navy', label='Conditioned fit', linewidth=0.7)
        ax.scatter(years, cond_rls, color='navy', s=10, marker='x')
        ax.set_xlabel('time (years)')
        ax.hlines(level, years[0], years[-1], color='black', linewidth=0.6, label=f'Observed 2021')
        ax.legend(loc='best')
        ax.set_ylabel('daily max temperature ($^\circ$C)')
        ax.set_title(f'{k}-year return level')
        fig.show()
    fig.suptitle(f'TXx {state} May-June {np.min(years)}-{np.max(years)} ({int(100 * infconf)}\% CI)')
    return fig

def get_risk_ratio_and_CI(event=41.7, year_past = 1951, year_now=2021):
    dic = {}
    std_prev, cond_prev = get_std_cond_profiles_for_year_and_event(event, year_now)
    std_hist, cond_hist = get_std_cond_profiles_for_year_and_event(event, year_past)
    ex_prev, condex_prev = get_std_cond_ex_profiles_for_year_and_event(event, year_now)
    ex_hist, condex_hist = get_std_cond_ex_profiles_for_year_and_event(event, year_past)

    def get_CI_for_rr_from_hessian(p1, p0, hess1, hess0, conf=0.95):
        hess_term1 = hess1[0, 0]
        hess_term0 = hess0[0, 0]
        var1 = hess_term1 / (len(y)) * 1 / p1 ** 2
        var0 = hess_term0 / (len(y)) * 1 / p0 ** 2
        total_var = var0 + var1
        up = pykelihood.distributions.Normal().ppf(1 - conf / 2)
        down = pykelihood.distributions.Normal().ppf(conf / 2)
        log_lb = np.log(p1 / p0) + down * np.sqrt(total_var)
        log_ub = np.log(p1 / p0) + up * np.sqrt(total_var)
        return [np.exp(log_lb), np.exp(log_ub)]

    p1 = std_prev.optimum[0].p()
    p0 = std_hist.optimum[0].p()
    dic['Standard'] = [p1 / p0] + get_CI_for_rr_from_hessian(std_prev.optimum[0].p(), std_hist.optimum[0].p(), std_prev.optimum[0].hessinv, std_hist.optimum[0].hessinv)
    p1 = cond_prev.optimum[0].p()
    p0 = cond_hist.optimum[0].p()
    dic['COND'] = [p1 / p0]+  get_CI_for_rr_from_hessian(cond_prev.optimum[0].p(), cond_hist.optimum[0].p(), cond_prev.optimum[0].hessinv, cond_hist.optimum[0].hessinv)
    p1 = ex_prev.optimum[0].p()
    p0 = ex_hist.optimum[0].p()
    dic['Excluding'] = [p1 / p0]+ get_CI_for_rr_from_hessian(ex_prev.optimum[0].p(), ex_hist.optimum[0].p(), condex_prev.optimum[0].hessinv, condex_hist.optimum[0].hessinv)
    p1 = condex_prev.optimum[0].p()
    p0 = condex_hist.optimum[0].p()
    dic['CONDEX'] = [p1 / p0] + get_CI_for_rr_from_hessian(condex_prev.optimum[0].p(), condex_hist.optimum[0].p(), condex_prev.optimum[0].hessinv, condex_hist.optimum[0].hessinv)
    return dic
