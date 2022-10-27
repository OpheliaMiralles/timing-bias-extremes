import os

import cv2
import matplotlib

import threshold_selection
from data import get_ghcn_daily_india_annualmax
from india_heatwave import fit_gev_Tx_with_trend, plot_loc_vs_anomaly, plot_return_levels, compute_timevarying_profile_pairs, fig_std_cond_comparison, compute_alternative_profiles, segment_plot, \
    fit_gev_Tx_without_trend, compute_timevarying_profile_pairs_using_bivariate_distribution, fig_std_cond_comparison_biv, plot_joint_distribution
from pykelihood.profiler import Profiler
from return_levels_vargas import get_gpd_profiles, get_gev_profiles

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
PLOT_PATH = os.getenv('PLOT_PATH', None)
SRC_PATH = os.getenv('SRC_PATH', None)
SIM_PATH = os.getenv('SIM_PATH', None)


# Simulations
def fig_univ_fixed_sr():
    p = f'{SIM_PATH}/500_iter_fixed_largeci.xlsx'
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10), constrained_layout=True)
    for sheet_name, ax, letter in zip(['RelBias', 'RRMSE', 'CIC', 'CIW'], axes.flatten(), ['a', 'b', 'c', 'd']):
        df = pd.read_excel(p, engine='openpyxl', sheet_name=sheet_name).rename(columns={'Unnamed: 0': 'Return level'}).set_index('Return level')
        df.plot(ax=ax, color=['salmon', 'pink', 'navy', 'royalblue'], marker='x')
        ax.grid()
        if sheet_name == 'CIC':
            ax.set_ylim(0.5, None)
        ax.set_title(f"{letter}) {sheet_name.replace('RelBias', 'RELATIVE BIAS')}")
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$')
    fig.show()
    fig.savefig(f'{PLOT_PATH}/univ_fixed_sr.png', DPI=200)


def fig_univ_cond_fixed_sr():
    p = f'{SIM_PATH}/1000_iter_conditioned_sample275_nh100.xlsx'
    for particularity in [None, '_nostandard']:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10), constrained_layout=True)
        for sheet_name, ax, letter in zip(['RelBias', 'RRMSE', 'CIC', 'CIW'], axes.flatten(), ['a', 'b', 'c', 'd']):
            df = pd.read_excel(p, engine='openpyxl', sheet_name=sheet_name).rename(columns={'Unnamed: 0': 'Return level'}).set_index('Return level')
            if particularity:
                toplot = df.drop(columns=['Standard'])
                toplot = toplot.loc[500:]
                colors=['pink', 'navy', 'royalblue']
            else:
                toplot = df
                colors=['salmon', 'pink', 'navy', 'royalblue']
                particularity = ''
            toplot.plot(ax=ax, color=colors, marker='x')
            ax.grid()
            if sheet_name == 'CIC':
                ax.set_ylim(0.5, None)
            ax.set_title(f"{letter}) {sheet_name.replace('RelBias', 'RELATIVE BIAS')}")
            ax.set_xscale('log')
            ax.set_xlabel(r'$\tau$')
        fig.show()
        fig.savefig(f'{PLOT_PATH}/univ_cond_fixed_sr{particularity}.png', DPI=200)


def fig_biv_fixed_sr():
    pA = f'{SIM_PATH}/1000_iter_biv_theta2_df5_newdensity_statmodels.xlsx'
    pB = f'{SIM_PATH}/500_iter_biv_theta2_df5_newdensity.xlsx'
    for p, case in zip([pA, pB], ['A', 'B']):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10), constrained_layout=True)
        for sheet_name, ax, letter in zip(['RelBias', 'RRMSE', 'CIC', 'CIW'], axes.flatten(), ['a', 'b', 'c', 'd']):
            df = pd.read_excel(p, engine='openpyxl', sheet_name=sheet_name).rename(columns={'Unnamed: 0': 'Return level'}).set_index('Return level')
            df.plot(ax=ax, color=['salmon', 'pink', 'navy', 'royalblue'], marker='x')
            ax.grid()
            if sheet_name == 'CIC':
                ax.set_ylim(0.5, None)
            ax.set_title(f"{letter}) {sheet_name.replace('RelBias', 'RELATIVE BIAS')}")
            ax.set_xscale('log')
            ax.set_xlabel(r'$\tau$')
        fig.show()
        fig.savefig(f'{PLOT_PATH}/biv_fixed_sr_{case}.png', DPI=200)


# Vargas example
inf_conf = 0.95
data = pd.read_csv(f'{SRC_PATH}/data/venezuela_data.csv')


def fig_threshold_selection():
    results = threshold_selection.threshold_selection_gpd_NorthorpColeman(data['data'], thresholds=np.linspace(5, 26, 8))
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
    ax.plot(results.index, results["pvalue"], color="navy", label="p-value")
    ax.scatter(results.index, results["pvalue"], marker="x", s=8, color="navy")
    ax.set_xlabel("threshold")
    ax.set_ylabel("p-value")
    fig.savefig(f"{PLOT_PATH}/threshold_sel_northorp.png", DPI=200)
    results, fig2 = threshold_selection.threshold_selection_GoF(data['data'], min_threshold=10, max_threshold=40, plot=True)
    fig2.savefig(f"{PLOT_PATH}/vargas_threshold_varty.png", DPI=200)
    im1 = cv2.imread(f"{PLOT_PATH}/threshold_sel_northorp.png")
    im2 = cv2.imread(f"{PLOT_PATH}/vargas_threshold_varty.png")
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True, figsize=(18, 5))
    for ax, im in zip((ax1, ax2), [im1, im2]):
        ax.imshow(im)
        ax.axis('off')
    ax1.set_title('a) Likelihood-ratio test based threshold selection')
    ax2.set_title('b) Goodness-of-fit based threshold selection')
    fig.show()
    fig.savefig(f"{PLOT_PATH}/threshold_sel.png", DPI=200)


pf_dic = {}
# GPD profiles
hss = 160
threshold = 12
above_thresh = data[data['data'] >= threshold]
theta = 0.73  # indicates time-correlated data
lambd = above_thresh.groupby('year')['data'].count().mean() * theta
pf_dic['GPD'] = get_gpd_profiles(data, hss, inf_conf)


def fig_segments_gpd():
    std, ex, fc, fcex = pf_dic['GPD']
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    metric = lambda x: x.isf(1 / (lambd * k))
    for k in range(50, 250, 25):
        i = 0
        for fit, color in zip([std, ex, fc, fcex],
                              ['salmon', 'pink', 'navy', 'royalblue']):
            ci1, ci2 = fit.confidence_interval(metric)
            rl = fit.optimum[0].isf(1 / (lambd * k))
            if k == 50:
                ax.vlines(k + i, ci1, ci2, color=color, label=fit.name)
            else:
                ax.vlines(k + i, ci1, ci2, color=color)
            ax.scatter(k + i, rl, marker='x', color=color)
            i += 2
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('precipitation (mm day$^{-1}$)')
    fig.suptitle(f"Return levels and {int(inf_conf * 100)}\% likelihood-based confidence intervals using a GPD fit")
    fig.show()
    fig.savefig(f'{PLOT_PATH}/venezuela_gpd_return_levels_CI.png')


def fig_rl_gpd():
    std, ex, fc, fcex = pf_dic['GPD']
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5), constrained_layout=True)
    sorted_y = above_thresh['data'].sort_values()
    n = sorted_y.size
    logrange = np.logspace(np.log10(1 + 1e-4), np.log10(1000), 100)
    metric = lambda x: x.isf(1 / (lambd * k))
    levs = []
    for fits, colors, ax, letter in zip([(std, fc), (ex, fcex)],
                                        [('salmon', 'navy'), ('pink', 'royalblue')], axes, ['a', 'b']):
        for fit, color, i in zip(fits, colors, [50, 0]):
            if 'xcluding' not in fit.name:
                i = 0
            lb_arr = []
            ub_arr = []
            rls = []
            for k in logrange:
                ci1, ci2 = fit.confidence_interval(metric)
                lb_arr.append(ci1)
                ub_arr.append(ci2)
                rl = fit.optimum[0].isf(1 / (lambd * k))
                rls.append(rl)
            ax.plot(logrange, rls, linewidth=0.7, color=color, label=fit.name)
            ax.plot(logrange, lb_arr, color=color, linewidth=0.7, linestyle='--')
            ax.plot(logrange, ub_arr, linewidth=0.7, color=color, linestyle='--')
            level = 1 / (lambd * fit.optimum[0].sf(above_thresh['data'].iloc[-1]))
            levs.append(level)
            l = min(level, np.max(logrange) - 50)
            if l == level:
                text = int(level)
            else:
                text = f'({int(level)})'
            ax.vlines(l, 0, above_thresh['data'].iloc[-1], color=color, linewidth=0.6, linestyle='--')
            ax.annotate(text, xy=(l, i),
                        xytext=(-3, 0), textcoords="offset points",
                        horizontalalignment="right",
                        verticalalignment="bottom", color=color)
        real_rt = 1 / ((1 - np.arange(0, n) / n))
        t = 'Including' if letter == 'a' else 'Excluding'
        ax.set_title(f'{letter}) {t} the extreme event')
        ax.scatter(real_rt / lambd, sorted_y, s=10, marker='x', color='black')
        ax.hlines(above_thresh['data'].iloc[-1], logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Vargas 1999')
        ax.set_xlabel('return period (years)')
        ax.legend(loc='best')
        ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(logrange[0], logrange[-1])
        ax.set_ylim(0, None)
    fig.suptitle(f"Return levels using a GPD fit ({int(inf_conf * 100)}\% CI)")
    fig.savefig(f'{PLOT_PATH}/venezuela_gpd_return_levels.png')


# GEV
annual_maxima = data.groupby('year').agg({'data': 'max'})['data']
hss = 20
pf_dic['GEV'] = get_gev_profiles(data, hss, inf_conf)


def fig_segments_gev():
    std, ex, fc, fcex = pf_dic['GEV']
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    metric = lambda x: x.isf(1 / k)
    for k in range(50, 250, 25):
        i = 0
        for fit, color in zip([std, ex, fc, fcex],
                              ['salmon', 'pink', 'navy', 'royalblue']):
            ci1, ci2 = fit.confidence_interval(metric)
            rl = fit.optimum[0].isf(1 / k)
            if k == 50:
                ax.vlines(k + i, ci1, ci2, color=color, label=fit.name)
            else:
                ax.vlines(k + i, ci1, ci2, color=color)
            ax.scatter(k + i, rl, marker='x', color=color)
            i += 2
    ax.set_xlabel('return period (years)')
    ax.legend(loc='upper left')
    ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
    fig.suptitle(f"Return levels and {int(inf_conf * 100)}\% likelihood-based confidence intervals using a GEV fit")
    fig.show()
    fig.savefig(f'{PLOT_PATH}/venezuela_gev_return_levels_CI.png')


def fig_rl_gev():
    std, ex, fc, fcex = pf_dic['GEV']
    sorted_y = annual_maxima.sort_values()
    n = sorted_y.size
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5), constrained_layout=True)
    logrange = np.logspace(np.log10(1 + 1e-4), np.log10(1000), 100)
    metric = lambda x: x.isf(1 / k)
    levs = []
    for fits, colors, ax, letter in zip([(std, fc), (ex, fcex)],
                                        [('salmon', 'navy'), ('pink', 'royalblue')], axes, ['a', 'b']):
        for fit, color, i in zip(fits, colors, [0, 60]):
            if 'xcluding' not in fit.name:
                i = 0
            lb_arr = []
            ub_arr = []
            rls = []
            for k in logrange:
                ci1, ci2 = fit.confidence_interval(metric)
                lb_arr.append(ci1)
                ub_arr.append(ci2)
                rl = fit.optimum[0].isf(1 / k)
                rls.append(rl)
            ax.plot(logrange, rls, linewidth=0.7, color=color, label=fit.name)
            ax.plot(logrange, lb_arr, color=color, linewidth=0.7, linestyle='--')
            ax.plot(logrange, ub_arr, linewidth=0.7, color=color, linestyle='--')
            level = 1 / fit.optimum[0].sf(annual_maxima.iloc[-1])
            levs.append(level)
            l = min(level, np.max(logrange) - 50)
            if l == level:
                text = int(level)
            else:
                text = f'({int(level)})'
            ax.vlines(l, 0, annual_maxima.iloc[-1], color=color, linewidth=0.6, linestyle='--')
            ax.annotate(text, xy=(l, i),
                        xytext=(-3, 0), textcoords="offset points",
                        horizontalalignment="right",
                        verticalalignment="bottom", color=color)
        real_rt = 1 / (1 - np.arange(0, n) / n)
        ax.scatter(real_rt, sorted_y, s=10, marker='x', color='black')
        ax.hlines(annual_maxima.iloc[-1], logrange[0], logrange[-1], color='goldenrod', linewidth=0.6, label='Vargas 1999')
        ax.set_xlabel('return period (years)')
        ax.legend(loc='best')
        ax.set_ylabel('3-day precipitation (mm day$^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(logrange[0], logrange[-1])
        ax.set_ylim(0, None)
        t = 'Including' if letter == 'a' else 'Excluding'
        ax.set_title(f'{letter}) {t} the extreme event')
    fig.suptitle(f"Return levels using a GEV fit ({int(inf_conf * 100)}\% CI)")
    fig.savefig(f'{PLOT_PATH}/venezuela_gev_return_levels.png')


# Heatwave example
temp_annualmax = get_ghcn_daily_india_annualmax()


def fig_loc_rl_std():
    s = temp_annualmax.columns[-1]
    y = temp_annualmax[s].dropna().loc[:2016]
    level = y.loc[2016]
    x = pd.Series((y.index - y.index.min()) / (y.index.max() - y.index.min()), index=y.index).rename('time')
    fit = fit_gev_Tx_with_trend(x, y)
    standard_profile = Profiler(distribution=fit, data=y, inference_confidence=0.95)
    fig1 = plot_loc_vs_anomaly(y, standard_profile)
    fig2 = plot_return_levels(y, standard_profile, level)
    fig1.savefig(f"{PLOT_PATH}/loc_jod.png", DPI=200)
    fig2.savefig(f"{PLOT_PATH}/rl_jod.png", DPI=200)
    im1 = cv2.imread(f"{PLOT_PATH}/loc_jod.png", )
    im2 = cv2.imread(f"{PLOT_PATH}/rl_jod.png")
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True, figsize=(14, 5))
    for ax, im in zip((ax1, ax2), [im1, im2]):
        ax.imshow(im)
        ax.axis('off')
    ax1.set_title('a) Estimated location parameter')
    ax2.set_title('b) Estimated return levels')
    fig.show()
    fig.savefig(f"{PLOT_PATH}/jod_rlsloc_std.png", DPI=200)


def segmentplot_jod():
    s = temp_annualmax.columns[-1]
    y = temp_annualmax[s].dropna().loc[:2016]
    fit = fit_gev_Tx_without_trend(y)
    standard_profile = Profiler(distribution=fit, data=y, inference_confidence=0.66, single_profiling_param='shape')
    ex, full_ex, full_inc = compute_alternative_profiles(fit, y, infconf=0.66)
    profiles = [p for p in [standard_profile, ex, full_inc, full_ex] if p is not None]
    fig = segment_plot(profiles, y, state='JODHPUR')
    fig.show()
    fig.savefig(f"{PLOT_PATH}/jod_segmentplot_india.png", DPI=200)


def fig_stdcond_comparison_jod():
    s = temp_annualmax.columns[-1]
    y = temp_annualmax[s].dropna()
    level = y.loc[2016]
    years = np.arange(2011, 2022)
    dic = {}
    for ye in years:
        dic[ye] = compute_timevarying_profile_pairs(ye, y, infconf=0.66)
    figjod = fig_std_cond_comparison(dic, 'JODHPUR', level=level)
    figjod.show()
    figjod.savefig(f"{PLOT_PATH}/jod_stdcond_comparison.png", DPI=200)


def fig_biv_distribution():
    fig = plot_joint_distribution(temp_annualmax)
    fig.savefig(f"{PLOT_PATH}/bivariate_distri_india.png", DPI=200)


def fig_stdcond_comparison_bik():
    phalodi = 49.5
    years = np.arange(2011, 2022)
    dic = {}
    for y in years:
        dic[y] = compute_timevarying_profile_pairs_using_bivariate_distribution(y, temp_annualmax, infconf=0.66)
    fig = fig_std_cond_comparison_biv(dic, 'BIKANER', level=phalodi)
    fig.show()
    fig.savefig(f"{PLOT_PATH}/bikaner_rls_comp.png", DPI=200)
