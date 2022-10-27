import os
from functools import partial

import pykelihood as pkl
from conditioning_methods import ConditioningMethod, shape_stability_conditioning
from pykelihood.distributions import GEV
from pykelihood.kernels import expo_ratio
from pykelihood.profiler import Profiler

path_to_directory = os.getenv("LOUISIANA_DATA")
import matplotlib.pyplot as plt

import warnings
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pykelihood.distributions import Distribution

warnings.filterwarnings('ignore')


class StoppingRuleLouisiana(object):
    def __init__(self, x: pd.Series, y: pd.Series,
                 k: int,
                 n: int,
                 historical_sample_yearsize: int,
                 func: Callable[[pd.Series, int, "Distribution", int], int]):
        """

        :param data: database that is the base of the analysis once we stop
        :param historical_sample_size: number of years necessary to provide a reliable first estimate
        :param func: stopping rule (it can be fixed or variable as the static methods detailed in the class)
        """
        self.x = x
        self.data = y
        self.k = k
        self.n = n
        self.stopping_rule = func
        self.historical_sample_yearsize = historical_sample_yearsize

    def __call__(self):
        return self.stopping_rule(self.x, self.data, self.historical_sample_yearsize, self.k, self.n)

    def stopped_data(self):
        c, N = self.__call__()
        return self.data.iloc[:N]

    def threshold(self):
        c, N = self.__call__()
        return c

    def last_year(self):
        c, N = self.__call__()
        return N

    @staticmethod
    def variable_in_the_estimated_return_level(covariate: pd.Series, data: pd.Series, historical_sample_size: int, k: int, n: int = 1):
        historical_sample_yearsize = historical_sample_size
        year = np.unique(data._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        cov_stopped = covariate.loc[:year]
        data_stopped = data.loc[:year]
        fit = fit_gev_cpc(cov_stopped, data_stopped)
        # we stop when at least n stations has exceeded the average return level in the relevant area for previous years
        return_level_estimate = np.mean(fit.isf(1 / k))
        cov_stopped = covariate.loc[:year + 1]
        data_stopped = data.loc[:year + 1]
        year += 1
        return_level_estimates = [float(return_level_estimate)] * len(data_stopped)
        while len(data_stopped.loc[year][data_stopped.loc[year] >= return_level_estimate]) < n and len(data_stopped) < len(data):
            fit = fit_gev_cpc(cov_stopped, data_stopped)
            return_level_estimate = np.mean(fit.isf(1 / k))
            cov_stopped = covariate.loc[:year + 1]
            data_stopped = data.loc[:year + 1]
            year += 1
            if not isinstance(data_stopped.loc[year], float):
                return_level_estimates.extend([float(return_level_estimate)] * len(data_stopped.loc[year]))
            else:
                return_level_estimates.append(float(return_level_estimate))
        N = year
        return return_level_estimates, N

    @staticmethod
    def fixed_to_k(covariate: pd.Series, data: pd.Series, historical_sample_size: int, k: int, n: int = 1):
        historical_sample_yearsize = historical_sample_size
        year = np.unique(y._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        data_stopped = data.loc[:year]
        return_level_estimates = [float(k)] * len(data_stopped)
        while len(data_stopped.loc[year][data_stopped.loc[year] >= k]) < n and len(data_stopped) < len(data):
            data_stopped = data.loc[:year + 1]
            year += 1
            return_level_estimates.extend([float(k)] * len(data_stopped.loc[year]))
        N = year
        return return_level_estimates, N

    @staticmethod
    def variable_in_the_historical_annualmax(covariate: pd.Series, data: pd.Series, historical_sample_size: int, k: int, n: int = 1):
        factor = 3.
        historical_sample_yearsize = historical_sample_size
        year = np.unique(y._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        data_stopped = data.loc[:year]
        # we stop when at least 1 stations has exceeded 3 times the annual maximum in the relevant area for previous years
        mean_annualmax = data_stopped.unstack(level='STATION').loc[year - historical_sample_size:year].mean()
        max_year = data_stopped.unstack(level='STATION').loc[year]
        data_stopped = data.loc[:year + 1]
        year += 1
        thresh = list(data_stopped.to_frame().merge(factor * mean_annualmax.to_frame(), left_index=True, right_index=True)[0])
        while not (max_year > factor * mean_annualmax).any() and len(data_stopped) < len(data):
            mean_annualmax = data_stopped.unstack(level='STATION').loc[year - historical_sample_size:year].mean()
            max_year = data_stopped.unstack(level='STATION').loc[year]
            data_stopped = data.loc[:year + 1]
            year += 1
            thresh.extend(np.array(data_stopped.loc[year].to_frame().merge(factor * mean_annualmax.to_frame(), left_index=True, right_index=True)[0]))
        N = year
        return thresh, N


def fit_gev_cpc(x, y):
    mu0_init, sigma0_init, shape_init = GEV.fit(y).flattened_params
    alpha0_init = 0.
    alpha = pkl.parameters.Parameter(alpha0_init)
    mu0 = pkl.parameters.Parameter(mu0_init)
    fit = GEV.fit(y, loc=expo_ratio(x=x, a=alpha, b=mu0, c=mu0), scale=expo_ratio(x=x, a=alpha, b=mu0),
                  score=shape_stability_conditioning, x0=[alpha0_init, mu0_init, sigma0_init, shape_init])
    return fit


def compute_alternative_profiles(fit, x, y):
    historical_sample_yearsize = 50
    sr = StoppingRuleLouisiana(x=x, y=y, k=140, func=StoppingRuleLouisiana.fixed_to_k,
                               historical_sample_yearsize=historical_sample_yearsize, n=6)
    historical_sample_size = len(y.loc[:np.unique(y._get_label_or_level_values('YEAR'))[sr.historical_sample_yearsize - 1]])
    thresh, N = sr()
    sd = y.loc[:N]
    len_extreme_event = len(y.loc[2016])
    ex_fit = fit_gev_cpc(x.loc[:2015], y.loc[:2015])
    full_cond = Profiler(distribution=fit, data=sd,
                         score_function=partial(ConditioningMethod.full_conditioning_including_spatial_extremes,
                                                historical_sample_size=historical_sample_size,
                                                length_extreme_event=len_extreme_event,
                                                threshold=thresh,
                                                number_spatial_points_above=6),
                         name='Conditioning including extreme event')
    excluding = Profiler(distribution=ex_fit, data=y.loc[:2015],
                         name='Excluding extreme event')
    excluding_cond = Profiler(distribution=ex_fit, data=y.loc[:2015],
                              score_function=partial(ConditioningMethod.full_conditioning_excluding_spatial_extremes,
                                                     historical_sample_size=historical_sample_size,
                                                     length_extreme_event=len_extreme_event,
                                                     threshold=thresh[:-len_extreme_event],
                                                     number_spatial_points_above=6),
                              name='Conditioning excluding extreme event')
    return [excluding, excluding_cond, full_cond]


def compute_timevarying_profile_pairs(year, x, y):
    inf_conf = 0.95
    obs = y.loc[:year]
    cov = x.loc[:year]
    fit = fit_gev_cpc(cov, obs)
    historical_sample_yearsize = 50
    # 140 is the 100-year return level in the area
    sr = StoppingRuleLouisiana(x=x, y=y, k=140, func=StoppingRuleLouisiana.fixed_to_k,
                               historical_sample_yearsize=historical_sample_yearsize, n=6)
    historical_sample_size = len(y.loc[:np.unique(y._get_label_or_level_values('YEAR'))[sr.historical_sample_yearsize - 1]])
    thresh, N = sr()
    len_extreme_event = max(len(y.loc[2016]), 1)
    std_prof = Profiler(distribution=fit_gev_cpc(cov, obs), data=obs, name=f'Standard fit {year}',
                        inference_confidence=inf_conf, single_profiling_param='shape')
    if year >= 2016:
        cond = Profiler(distribution=fit, data=obs, inference_confidence=inf_conf,
                        score_function=partial(ConditioningMethod.full_conditioning_including_spatial_extremes,
                                               historical_sample_size=historical_sample_size,
                                               length_extreme_event=len_extreme_event,
                                               threshold=thresh,
                                               number_spatial_points_above=6),
                        name='Conditioning including extreme event', single_profiling_param='shape')
    else:
        thresh_ex = thresh[:-(len(y.loc[:2016]) - len(obs))]
        cond = Profiler(distribution=fit, data=obs, inference_confidence=inf_conf,
                        score_function=partial(ConditioningMethod.full_conditioning_excluding_spatial_extremes,
                                               historical_sample_size=historical_sample_size,
                                               length_extreme_event=len_extreme_event,
                                               threshold=thresh_ex,
                                               number_spatial_points_above=6),
                        name='Conditioning excluding extreme event', single_profiling_param='shape')
    return [std_prof, cond]


def plot_loc_vs_anomaly(x, y, profile):
    opt = profile.optimum[0]
    first_year = np.unique(y._get_label_or_level_values('YEAR'))[10]
    if 'xcluding' in profile.name:
        endyear = 2015
    else:
        endyear = 2016
    df = pd.DataFrame([x.loc[:endyear], opt.loc(), opt.loc() + opt.scale(), opt.loc() + 2 * opt.scale()],
                      columns=x.loc[:endyear].index,
                      index=['TEMPANOMALY_GLOB', 'loc_par', 'loc_plus_sigma', 'loc_plus_2sigma']) \
        .T.groupby('TEMPANOMALY_GLOB', as_index=False).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=16, marker='x', color='royalblue')
    ax.set_xlim(-0.6, 1)
    ax.set_ylim(0, 350)
    ax.set_xlabel('Smoothed surface temperature anomaly (K)')
    ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
    ax.plot(df.TEMPANOMALY_GLOB, df.loc_par, color='red')
    ax.plot(df.TEMPANOMALY_GLOB, df.loc_plus_sigma, linewidth=0.2, color='red')
    ax.plot(df.TEMPANOMALY_GLOB, df.loc_plus_2sigma, linewidth=0.2, color='red')
    for year in [first_year, 2016]:
        if 'xcluding' in profile.name and year == 2016:
            year_dis = year - 1
        else:
            year_dis = year
        metric = lambda x: x.loc().loc[year_dis]
        ci = profile.confidence_interval(metric)
        ax.vlines(x.loc[year_dis], ci[0], ci[-1], color='red')
    return fig


def plot_return_levels(x, y, profile):
    logrange = np.logspace(np.log10(1 + 1e-4), np.log10(10000), 100)
    first_year = np.unique(y._get_label_or_level_values('YEAR'))[10]
    sorted_y = y.sort_values()
    n = sorted_y.size
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    levs = []
    for color, year in zip(['royalblue', 'red'], [first_year, 2016]):
        if 'xcluding' in profile.name and year == 2016:
            year_dis = year - 1
        else:
            year_dis = year
        idx = x.reset_index()[x.reset_index()['YEAR'] == year_dis].index[0]
        name = f'GEV scale fit {year_dis}'
        rls = [profile.optimum[0].isf(1 / k)[idx] for k in logrange]
        cis = []
        for k in logrange:
            metric = lambda x: x.isf(1 / k)[idx]
            cis.append(profile.confidence_interval(metric))
        louisiana_level = 1 / profile.optimum[0].sf(216.1)[idx]
        levs.append(louisiana_level)
        lbs = [lb for (lb, ub) in cis]
        ubs = [ub for (lb, ub) in cis]
        ax.vlines(louisiana_level, 0, 216.1, color=color, linewidth=0.6, linestyle='--')
        ax.annotate(int(louisiana_level), xy=(louisiana_level, 0),
                    xytext=(-3, 0), textcoords="offset points",
                    horizontalalignment="right",
                    verticalalignment="bottom", color=color)
        ax.plot(logrange, rls, color=color, label=name, linewidth=0.6)
        ax.plot(logrange, lbs, color=color, linewidth=0.6)
        ax.plot(logrange, ubs, color=color, linewidth=0.6)
        real_rt = 1 / (1 - np.arange(0, n) / n)
        scaled_y = sorted_y * (profile.optimum[0].loc().loc[year_dis] / profile.optimum[0].flattened_param_dict['loc_b'].value)
        ax.scatter(real_rt, scaled_y, s=10, marker='x', color=color)
    ax.annotate(f'RR={np.round(levs[0] / levs[-1], 1)}', xy=(logrange[0] + 1, 0),
                xytext=(-3, 0), textcoords="offset points",
                horizontalalignment="left",
                verticalalignment="bottom", color='goldenrod')
    ax.hlines(216.1, logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Louisiana 2016')
    ax.set_xlabel('Return period (year)')
    ax.legend(loc='upper left')
    ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
    ax.set_xscale('log')
    ax.set_xlim(logrange[0], logrange[-1])
    ax.set_ylim(0, None)
    fig.suptitle("GHCN-D all stations: return period")
    return fig


def segment_plot(profiles, year):
    idx = x.reset_index()[x.reset_index()['YEAR'] == year].index[0]
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    logrange = np.linspace(150, 2000, 10).astype(int)
    for k in logrange:
        i = 0
        for profile, color in zip(profiles,
                                  ['salmon', 'pink', 'navy', 'royalblue']):
            if 'xcluding' in profile.name and year == 2016:
                idx = x.reset_index()[x.reset_index()['YEAR'] == year - 1].index[0]
            metric = lambda x: x.isf(1 / k)[idx]
            ci1, ci2 = profile.confidence_interval(metric)
            rl = profile.optimum[0].isf(1 / k)[idx]
            if k == 150:
                ax.vlines(k + i, ci1, ci2, color=color, label=profile.name)
            else:
                ax.vlines(k + i, ci1, ci2, color=color)
            ax.scatter(k + i, rl, marker='x', color=color)
            i += 20
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
    return fig


def rl_plot_mutiprofile(profiles, year):
    idx = x.reset_index()[x.reset_index()['YEAR'] == year].index[0]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    logrange = np.logspace(np.log10(1 + 1e-4), np.log10(10000), 100)
    sorted_y = y.sort_values()
    n = sorted_y.size
    for profile, color in zip(profiles,
                              ['salmon', 'pink', 'navy', 'royalblue']):
        if 'xcluding' in profile.name and year == 2016:
            year_dis = year - 1
            idx = x.reset_index()[x.reset_index()['YEAR'] == year_dis].index[0]
        rls = [profile.optimum[0].isf(1 / k)[idx] for k in logrange]
        ax.plot(logrange, rls, color=color, label=profile.name)
        cis = []
        for k in logrange:
            metric = lambda x: x.isf(1 / k)[idx]
            cis.append(profile.confidence_interval(metric))
        lbs = [lb for (lb, ub) in cis]
        ubs = [ub for (lb, ub) in cis]
        ax.fill_between(logrange, lbs, ubs, color=color, label=profile.name, alpha=0.2)
    real_rt = 1 / (1 - np.arange(0, n) / n)
    ax.scatter(real_rt, sorted_y, s=10, marker='x', color='black', label='Observed RLs')
    ax.hlines(216.1, logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Louisiana 2016')
    ax.set_xlabel('Return period (year)')
    ax.legend(loc='upper left')
    ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
    ax.set_xscale('log')
    ax.set_xlim(logrange[0], logrange[-1])
    ax.set_ylim(0, None)
    fig.suptitle("GHCN-D all stations: return period")
    return fig


def fig_std_cond_comparison(dic):
    years = np.arange(2011, 2022, 1)
    logrange = [50, 100, 200, 500, 1000, 10000]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), constrained_layout=True)
    for k, ax in zip(logrange, axes.flatten()):
        std_rls = []
        cond_rls = []
        for year in dic:
            std_prof = dic[year][0]
            cond_prof = dic[year][-1]
            std_rls.append(np.mean(std_prof.optimum[0].isf(1 / k)))
            cond_rls.append(np.mean(cond_prof.optimum[0].isf(1 / k)))
        ax.plot(years, std_rls, color='red', label='Standard fit', linewidth=0.6)
        ax.scatter(years, std_rls, color='red', s=10, marker='x')
        ax.plot(years, cond_rls, color='b', label='Conditioned fit', linewidth=0.6)
        ax.scatter(years, cond_rls, color='b', s=10, marker='x')
        ax.set_xlabel('Time (year)')
        ax.legend(loc='best')
        ax.set_ylabel('daily max temperature ($^\circ$C)')
        ax.set_title(f'{k}-year return level')
    fig.suptitle("GHCN-D all stations: return period")
    return fig
