import os
from functools import partial

import matplotlib
from copulae import GumbelCopula

from conditioning_methods import ConditioningMethod, shape_stability_conditioning
from data import jodhpur_coords, bikaner_coords
from parameters import ConstantParameter
from pykelihood.distributions import GEV
from pykelihood.kernels import linear
from pykelihood.profiler import Profiler
from timing_bias import StoppingRule

matplotlib.rcParams['text.usetex'] = True

path_to_directory = os.getenv("INDIA_DATA")
import matplotlib.pyplot as plt

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

warnings.filterwarnings('ignore')

path_to_directory = os.getenv("INDIA_DATA")
stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
stations = stations[((stations.LATITUDE == bikaner_coords[0]) & (stations.LONGITUDE == bikaner_coords[-1]))
                    | ((stations.LATITUDE == jodhpur_coords[0]) & (stations.LONGITUDE == jodhpur_coords[-1]))]


def fit_gev_Tx_with_trend(x, y):
    mu0_init, sigma_init, shape_init = fit_gev_Tx_without_trend(y).flattened_params
    alpha_init = 0.
    if y.name == 'IN019180500':
        fit = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init])
    else:
        fit = GEV.fit(y, loc=linear(x=x), x0=[mu0_init, alpha_init, sigma_init, shape_init], score=shape_stability_conditioning)
    return fit


def fit_gev_Tx_without_trend(y):
    if y.name == 'IN019180500':
        return GEV.fit(y, x0=(y.mean(), y.std(), 0.))
    else:
        return GEV.fit(y, x0=(y.mean(), y.std(), 0.), score=shape_stability_conditioning)


def compute_alternative_profiles(fit, y, x=None, trend=False, infconf=0.95):
    historical_sample_size = len([i for i in y.index if i <= 2010])
    sr = StoppingRule(data=y, k=30, distribution=fit, func=StoppingRule.fixed_to_1981_2010_average,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr()
    len_extreme_event = 1
    if y.index.max() == 2016:
        full_cond = Profiler(distribution=fit, data=y, inference_confidence=infconf,
                             score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                    historical_sample_size=historical_sample_size,
                                                    length_extreme_event=len_extreme_event,
                                                    threshold=thresh),
                             name='Conditioning including extreme event', single_profiling_param='shape')
        ex_fit = fit_gev_Tx_without_trend(y.loc[:2015]) if not trend else fit_gev_Tx_with_trend(x.loc[:2015], y.loc[:2015])
        excluding = Profiler(distribution=ex_fit, data=y.loc[:2015],
                             name='Excluding extreme event', inference_confidence=infconf, single_profiling_param='shape')
        thresh_ex = thresh[:-len_extreme_event]
        excluding_cond = Profiler(distribution=fit, data=y.loc[:2015],
                                  score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                         historical_sample_size=historical_sample_size,
                                                         length_extreme_event=len_extreme_event,
                                                         threshold=thresh_ex),
                                  name='Conditioning excluding extreme event', inference_confidence=infconf, single_profiling_param='shape')
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
                                  name='Conditioning excluding extreme event', single_profiling_param='shape')
    return [excluding, excluding_cond, full_cond]


def compute_timevarying_profile_pairs(year, y, infconf=0.95):
    historical_sample_size = len([i for i in y.index if i <= 2010])
    obs = y.loc[:year]
    fit = fit_gev_Tx_without_trend(y.loc[:max(year, 2016)])
    sr = StoppingRule(data=y, k=30, distribution=fit, func=StoppingRule.fixed_to_1981_2010_average,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr()
    len_extreme_event = 1
    std_prof = Profiler(distribution=fit_gev_Tx_without_trend(obs), data=obs, name=f'Standard fit {year}',
                        inference_confidence=infconf)
    if year >= 2016:
        cond = Profiler(distribution=fit, data=obs, inference_confidence=infconf,
                        score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                               historical_sample_size=historical_sample_size,
                                               length_extreme_event=len_extreme_event,
                                               threshold=thresh),
                        name=f'Conditioned fit {year}')
    else:
        thresh_ex = thresh[:-(len(y.loc[:2016]) - len(obs))]
        cond = Profiler(distribution=fit, data=obs, inference_confidence=infconf,
                        score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                               historical_sample_size=historical_sample_size,
                                               length_extreme_event=len_extreme_event,
                                               threshold=thresh_ex),
                        name=f'Conditioned fit {year}')
    return [std_prof, cond]


def plot_loc_vs_anomaly(y, profile):
    opt = profile.optimum[0]
    first_year = np.unique(y._get_label_or_level_values('YEAR'))[0]
    last_year = np.unique(y._get_label_or_level_values('YEAR'))[-1]
    years = np.arange(first_year, last_year + 1, 1)
    scaled_years = ((years - y.index.min()) / (y.index.max() - y.index.min()))
    x = pd.Series(scaled_years, index=years)
    locs = [opt.loc.with_covariate(np.array([i]))()[0] for i in x]
    df = pd.DataFrame([x.values, locs, locs + opt.scale(), locs + 2 * opt.scale()],
                      columns=x.index,
                      index=['time', 'loc_par', 'loc_plus_sigma', 'loc_plus_2sigma']) \
        .T
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(y.index, y, s=16, marker='x', color='royalblue')
    ax.set_xlim(int(y.index.min() / 10) * 10, 2020)
    ax.set_ylim(43, 51)
    ax.set_xlabel('time (years)')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    ax.plot(df.index, df.loc_par, color='red')
    ax.plot(df.index, df.loc_plus_sigma, linewidth=0.2, color='red')
    ax.plot(df.index, df.loc_plus_2sigma, linewidth=0.2, color='red')
    for year, i in zip([1973, last_year], [0, -1]):
        if 'xcluding' in profile.name and year == 2016:
            year_dis = year - 1
            i = i - 1
        else:
            year_dis = year
        idx_y = [i for i in y.index if int(i / 10) * 10 == int(year / 10) * 10][i]
        metric = lambda x: x.loc().loc[idx_y]
        ci = profile.confidence_interval(metric)
        ax.vlines(year_dis, ci[0], ci[-1], color='red')
    return fig


def plot_return_levels(y, profile, level):
    opt = profile.optimum[0]
    logrange = np.logspace(np.log10(1 + 1e-2), np.log10(10000), 100)
    first_year = 1973
    last_year = np.unique(y._get_label_or_level_values('YEAR'))[-1]
    sorted_y = y.sort_values()
    n = sorted_y.size
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    levs = []
    for color, year, i in zip(['royalblue', 'red'], [first_year, last_year], [0, -1]):
        name = f'GEV shift fit {year}'
        cis = []
        if 'xcluding' in profile.name and year == 2016:
            i = i - 1
        idx_y = [i for i in y.index if int(i / 10) * 10 == int(year / 10) * 10][i]
        idx = y.reset_index()[y.reset_index()['YEAR'] == idx_y].index[0]
        rls = [opt.isf(1 / k)[idx] for k in logrange]
        for k in logrange:
            metric = lambda x: x.isf(1 / k)[idx]
            cis.append(profile.confidence_interval(metric))
        phalodi_level = 1 / opt.sf(level)[idx]
        levs.append(phalodi_level)
        lbs = [lb for (lb, ub) in cis]
        ubs = [ub for (lb, ub) in cis]
        try:
            ax.vlines(phalodi_level, 0, level, color=color, linewidth=0.6, linestyle='--')
            ax.annotate(int(phalodi_level), xy=(phalodi_level, 43),
                        xytext=(-3, 0), textcoords="offset points",
                        horizontalalignment="right",
                        verticalalignment="bottom", color=color)
        except:
            pass
        ax.plot(logrange, rls, color=color, label=name, linewidth=0.6)
        ax.plot(logrange[25:], lbs[25:], color=color, linewidth=0.6, linestyle='--')
        ax.plot(logrange[25:], ubs[25:], color=color, linewidth=0.6, linestyle='--')
        real_rt = 1 / (1 - np.arange(0, n) / n)
        if i == 0:
            scaled_y = sorted_y - (opt.loc().loc[idx_y] - opt.flattened_param_dict['loc_a'].value)
        else:
            scaled_y = sorted_y
        ax.scatter(real_rt, scaled_y, s=10, marker='x', color=color)
    ax.annotate(f'RR={np.round(levs[0] / levs[-1], 1)}', xy=(logrange[0] + 1, 43),
                xytext=(-3, 0), textcoords="offset points",
                horizontalalignment="left",
                verticalalignment="bottom", color='goldenrod')
    ax.hlines(level, logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Observed 2016')
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    ax.set_xscale('log')
    ax.set_xlim(logrange[0], logrange[-1])
    ax.set_ylim(43, 51)
    return fig


def segment_plot(profiles, y, state):
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    logrange = np.linspace(150, 2000, 10).astype(int)
    infconf = profiles[0].inference_confidence
    for k in logrange:
        i = 0
        for profile, color in zip(profiles,
                                  ['salmon', 'pink', 'navy', 'royalblue']):
            metric = lambda x: x.isf(1 / k)
            ci1, ci2 = profile.confidence_interval(metric)
            rl = profile.optimum[0].isf(1 / k)
            if k == 150:
                ax.vlines(k + i, ci1, ci2, color=color, label=profile.name)
            else:
                ax.vlines(k + i, ci1, ci2, color=color)
            ax.scatter(k + i, rl, marker='x', color=color)
            i += 20
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    fig.suptitle(f'TXx {state} May-June {y.index.min()}-{y.index.max()} ({int(100 * infconf)}\% CI)')
    return fig


def rl_plot_mutiprofile(profiles, y, state, level):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    logrange = np.logspace(np.log10(1 + 1e-4), np.log10(10000), 100)
    sorted_y = y.sort_values()
    n = sorted_y.size
    for profile, color in zip(profiles,
                              ['salmon', 'pink', 'navy', 'royalblue']):
        rls = [profile.optimum[0].isf(1 / k) for k in logrange]
        ax.plot(logrange, rls, color=color, label=profile.name)
        cis = []
        for k in logrange:
            metric = lambda x: x.isf(1 / k)
            cis.append(profile.confidence_interval(metric))
        lbs = [lb for (lb, ub) in cis]
        ubs = [ub for (lb, ub) in cis]
        ax.fill_between(logrange, lbs, ubs, color=color, label=profile.name, alpha=0.2)
    real_rt = 1 / (1 - np.arange(0, n) / n)
    ax.scatter(real_rt, sorted_y, s=10, marker='x', color='black', label='Observed RLs')
    ax.hlines(level, logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Observed 2016')
    ax.set_xlabel('Return period (year)')
    ax.legend(loc='best')
    ax.set_ylabel('daily max temperature ($^\circ$C)')
    ax.set_xscale('log')
    ax.set_xlim(logrange[0], logrange[-1])
    ax.set_ylim(43, 51)
    fig.suptitle(f'TXx {state} May-June {y.index.min()}-{y.index.max()} (95\% CI)')
    return fig


def fig_std_cond_comparison(dic, state, level):
    years = list(dic.keys())
    logrange = [100, 1000, 10000]
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)
    infconf = dic[years[0]][0].inference_confidence
    for k, ax in zip(logrange, axes.flatten()):
        std_rls = []
        std_lbs = []
        std_ubs = []
        cond_rls = []
        cond_lbs = []
        cond_ubs = []
        for year in dic:
            std_prof = dic[year][0]
            cond_prof = dic[year][-1]
            std_rls.append(std_prof.optimum[0].isf(1 / k))
            cond_rls.append(cond_prof.optimum[0].isf(1 / k))
            metric = lambda x: x.isf(1 / k)
            lb, ub = std_prof.confidence_interval(metric)
            std_lbs.append(lb)
            std_ubs.append(ub)
            lb, ub = cond_prof.confidence_interval(metric)
            cond_lbs.append(lb)
            cond_ubs.append(ub)
        ax.plot(years, std_rls, color='red', label='Standard fit', linewidth=0.7)
        ax.scatter(years, std_rls, color='salmon', s=10, marker='x')
        for i in [0, 5, -1]:
            if i == -1:
                y = years[i] - 0.2
            else:
                y = years[i]
            ax.vlines(y, std_lbs[i], std_ubs[i], color='salmon', linewidth=0.5)
            ax.hlines(std_ubs[i], y - 0.2, y + 0.2, color='salmon', linewidth=0.5)
            ax.hlines(std_lbs[i], y - 0.2, y + 0.2, color='salmon', linewidth=0.5)
        ax.plot(years, cond_rls, color='navy', label='Conditioned fit', linewidth=0.7)
        ax.scatter(years, cond_rls, color='navy', s=10, marker='x')
        for i in [0, 5, -1]:
            if i == -1:
                y = years[i]
            else:
                y = years[i] + 0.2
            ax.vlines(y, cond_lbs[i], cond_ubs[i], color='navy', linewidth=0.5)
            ax.hlines(cond_ubs[i], y - 0.2, y + 0.2, color='navy', linewidth=0.5)
            ax.hlines(cond_lbs[i], y - 0.2, y + 0.2, color='navy', linewidth=0.5)
        ax.hlines(level, years[0], years[-1], color='black', linewidth=0.6, label=f'Observed 2016')
        ax.set_xlabel('time (years)')
        ax.legend(loc='best')
        ax.set_ylabel('daily max temperature ($^\circ$C)')
        ax.set_title(f'{k}-year return level')
    fig.suptitle(f'TXx {state} May-June {np.min(years)}-{np.max(years)} ({int(100 * infconf)}\% CI)')
    return fig


# Bivarate
def cdf_levels(u2, l, theta):
    a = np.log(1 / l) ** theta
    b = np.log(1 / u2) ** theta
    c = (a - b) ** (1 / theta)
    return np.exp(-c)


def isf_levels(u2, l, theta):
    return 1 - cdf_levels(u2, l, theta)


def fit_gumbel_biv(data):
    joint_dataset = data.copy().dropna()
    margs = []
    u = []
    stationlist = joint_dataset.columns
    for s in stationlist:
        state = stations[stations["STATION"] == s]["STATE"].unique()[0]
        joint_dataset = joint_dataset.rename(columns={s: state})
        y = joint_dataset[state]
        fit = fit_gev_Tx_without_trend(y)
        margs.append(fit)
        u.append(fit.cdf(y))
    joint = GumbelCopula(dim=2).fit(np.array(u).T)
    return joint, margs


def compute_timevarying_profile_pairs_using_bivariate_distribution(year, biv_data, infconf=0.95):
    y_jod = biv_data[biv_data.columns[-1]].dropna()
    y_bik = biv_data[biv_data.columns[0]].dropna()  # common_data[common_data.columns[0]]
    x_bik = pd.Series((y_bik.loc[:year].index - y_bik.loc[:year].index.min()) / (y_bik.loc[:year].index.max() - y_bik.loc[:year].index.min()), index=y_bik.loc[:year].index).rename('time')
    x_jod = pd.Series((y_jod.loc[:year].index - y_jod.loc[:year].index.min()) / (y_jod.loc[:year].index.max() - y_jod.loc[:year].index.min()), index=y_jod.loc[:year].index).rename('time')
    historical_sample_size = len([i for i in y_jod.index if i <= 2010])
    joint, margs = fit_gumbel_biv(biv_data)
    # The dataset from Jodhpur is the one that stops when the threshold is reached
    fit = fit_gev_Tx_without_trend(y_jod.loc[:max(year, 2016)])
    sr = StoppingRule(data=y_jod, k=30, distribution=fit, func=StoppingRule.fixed_to_1981_2010_average,
                      historical_sample_size=historical_sample_size)
    thresh, N = sr()
    fit_jod = fit_gev_Tx_with_trend(x_jod.loc[:year], y_jod.loc[:year])
    fit_bik = fit_gev_Tx_with_trend(x_bik.loc[:year], y_bik.loc[:year])  # best fit for bikaner with all of the available data
    fit_bik_fixed_trend = GEV(loc=linear(a=fit_bik.loc.a, b=ConstantParameter(fit_bik.loc.b.value), x=x_bik),
                              scale=fit_bik.scale(), shape=fit_bik.shape())
    fit_jod_fixed_trend = GEV(loc=linear(a=fit_jod.loc.a, b=ConstantParameter(fit_jod.loc.b.value), x=x_jod),
                              scale=fit_jod.scale(), shape=fit_jod.shape())
    # std Independant fit for Bikaner
    ind = Profiler(distribution=fit_bik, data=y_bik.loc[:year], name=f'Standard fit {year}',
                   inference_confidence=infconf)
    if year >= 2016:
        margin = fit_jod_fixed_trend.fit_instance(y_jod.loc[:year],
                                                  score=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                                historical_sample_size=historical_sample_size,
                                                                threshold=thresh.copy()))
        crule_std = partial(ConditioningMethod.including_extreme_using_correlated_distribution, joint_structure=joint,
                            stopping_data=y_jod.loc[:year],
                            correlated_margin=fit_jod,
                            threshold=thresh.copy(),
                            historical_sample_size=historical_sample_size)
        crule_cond = partial(ConditioningMethod.full_conditioning_using_correlated_distribution, joint_structure=joint,
                             stopping_data=y_jod.loc[:year],
                             correlated_margin=margin,
                             threshold=thresh.copy(),
                             historical_sample_size=historical_sample_size)
    else:
        thresh_ex = thresh[:-(len(y_jod.loc[:2016]) - len(y_jod.loc[:year]))]
        margin = fit_jod_fixed_trend.fit_instance(y_jod.loc[:year],
                                                  score=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                                historical_sample_size=historical_sample_size,
                                                                threshold=thresh_ex))
        crule_std = partial(ConditioningMethod.excluding_extreme_using_correlated_distribution_spec, joint_structure=joint,
                            stopping_data=y_jod.loc[:year],
                            correlated_margin=fit_jod,
                            threshold=thresh_ex,
                            historical_sample_size=historical_sample_size)
        crule_cond = partial(ConditioningMethod.full_conditioning_excluding_extreme_using_correlated_distribution,
                             joint_structure=joint,
                             stopping_data=y_jod.loc[:year],
                             correlated_margin=margin,
                             threshold=thresh_ex,
                             historical_sample_size=historical_sample_size)
    # the conditional fit only uses years in common for Jodphur and Bikaner as well as complete info about values above and below the threshold in the Jodhpur series
    std = Profiler(distribution=fit_bik_fixed_trend, data=y_bik.loc[:year], inference_confidence=infconf,
                   score_function=crule_std,
                   name=f'Conditioned fit {year}')
    cond = Profiler(distribution=fit_bik_fixed_trend, data=y_bik.loc[:year], inference_confidence=infconf,
                    score_function=crule_cond,
                    name=f'Conditioned fit {year}')
    return [ind, std, cond]


def fig_std_cond_comparison_biv(dic, state, level):
    years = list(dic.keys())
    logrange = [100, 1000, 10000]
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)
    infconf = dic[years[0]][0].inference_confidence
    for k, ax in zip(logrange, axes.flatten()):
        std_rls = []
        ind_rls = []
        cond_rls = []
        for year in dic:
            ind_prof = dic[year][0]
            idx_year = list(np.where(ind_prof.data.index <= year))[0][-1]
            std_prof = dic[year][1]
            cond_prof = dic[year][-1]
            ind_rls.append(float(ind_prof.optimum[0].isf(1 / k)[idx_year]))
            std_rls.append(float(std_prof.optimum[0].isf(1 / k)[idx_year]))
            cond_rls.append(float(cond_prof.optimum[0].isf(1 / k)[idx_year]))
        ax.plot(years, std_rls, color='salmon', label='Standard fit', linewidth=0.7)
        ax.scatter(years, std_rls, color='salmon', s=10, marker='x')
        ax.plot(years, cond_rls, color='navy', label='Conditioned fit', linewidth=0.7)
        ax.scatter(years, cond_rls, color='navy', s=10, marker='x')
        ax.plot(years, ind_rls, color='goldenrod', label='Independent fit', linewidth=0.7)
        ax.scatter(years, ind_rls, color='goldenrod', s=10, marker='x')
        ax.set_xlabel('time (years)')
        ax.legend(loc='best')
        ax.set_ylabel('daily max temperature ($^\circ$C)')
        ax.set_title(f'{k}-year return level')
    fig.suptitle(f'TXx {state} May-June {np.min(years)}-{np.max(years)} ({int(100 * infconf)}\% CI)')
    return fig


def plot_joint_distribution(biv_data):
    joint, margs = fit_gumbel_biv(biv_data)
    theta = joint.params
    dic_theo = {}
    for level in np.linspace(0.4, 0.9, 5):
        level = np.round(level, 1)
        dic_theo[level] = []
        for w in np.linspace(0., 1., 200):
            dic_theo[level].append([w, cdf_levels(w, level, theta)])

    dic_indep = {}
    for level in np.linspace(0.4, 0.9, 5):
        level = np.round(level, 1)
        dic_indep[level] = [[i, float(margs[0].inverse_cdf(level / margs[1].cdf(i)))] for i in np.linspace(43, 50, 100)]
    bik_data = biv_data[biv_data.columns[0]]
    jod_data = biv_data[biv_data.columns[-1]]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 6), constrained_layout=True)
    plot_lines = []
    cmap = matplotlib.cm.get_cmap('BuPu')
    for level in dic_theo.keys():
        color = cmap(level)
        bik = margs[0].inverse_cdf([d[0] for d in dic_theo[level]])
        jod = margs[1].inverse_cdf([d[1] for d in dic_theo[level]])
        ax2.set_xlabel('daily max temperature ($^\circ$C) in Bikaner')
        ax2.set_ylabel('daily max temperature ($^\circ$C) in Jodhpur')
        l2, = ax2.plot(bik, jod, label=f'{level}', color=color)
        bik = [d[0] for d in dic_indep[level]]
        jod = [d[1] for d in dic_indep[level]]
        l1, = ax2.plot(bik, jod, color=color, linestyle='--')
        plot_lines.append([l2, l1])
    ax2.set_title('b) Quantiles from the joint cumulative distribution function')
    ax2.set_xlim(None, 49.)
    legend1 = ax2.legend([l[0] for l in plot_lines], dic_theo.keys(), loc='upper left')
    legend2 = ax2.legend(plot_lines[1], [r"Corr. with $\alpha$" + f"={np.round(1 / theta, 2)}", "Indep."], loc='lower right')
    ax2.add_artist(legend1)
    ax2.add_artist(legend2)
    ax2.scatter(bik_data, jod_data, color='black', s=6, marker='x', label='Observed')
    ax1 = plot_joint_pdf(joint, margs, ax=ax1)
    ax1.scatter(bik_data, jod_data, color='black', s=6, marker='x', label='Observed')
    ax1.legend()
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_ylim(ax2.get_ylim())
    return fig


def plot_joint_pdf(joint, margs, ax=None, ticks_nbr=25):
    ax = ax or plt.gca()
    n_samples = 100
    eps = 1e-4
    uu, vv = np.meshgrid(np.linspace(eps, 1 - eps, n_samples),
                         np.linspace(eps, 1 - eps, n_samples))
    xx, yy = margs[0].inverse_cdf(uu), margs[1].inverse_cdf(vv)
    points = np.vstack([uu.ravel(), vv.ravel()]).T

    data = joint.pdf(points).T.reshape(uu.shape)
    min_ = np.nanpercentile(data, 100 * 1e-4)
    max_ = np.nanpercentile(data, (1 - 1e-4) * 100)
    vticks = np.logspace(np.log10(min_), np.log10(max_), ticks_nbr)
    range_cbar = [min_, max_]
    cs = ax.contourf(xx, yy, data, vticks,
                     antialiased=True, vmin=range_cbar[0],
                     vmax=range_cbar[1],
                     cmap='BuPu',
                     norm=matplotlib.colors.LogNorm(), alpha=0.8)
    ax.set_xlabel("Bikaner")
    ax.set_ylabel("Jodhpur")
    ax.set_aspect('equal')
    ax.set_xlabel('daily max temperature ($^\circ$C) in Bikaner')
    ax.set_ylabel('daily max temperature ($^\circ$C) in Jodhpur')
    ax.set_title('a) Joint density contour')
    plt.colorbar(cs, ticks=[0.01, 0.1, 1, 10, 100], ax=ax, location='left')
    return ax
