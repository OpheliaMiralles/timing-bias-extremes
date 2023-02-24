from functools import partial

from conditioning_methods import ConditioningMethod
from pykelihood.distributions import GPD, GEV
from pykelihood.profiler import Profiler
from pykelihood_distributions import GPD_reparametrized, GEV_reparametrized
from timing_bias import StoppingRule

threshold = 12


def gpd_stopping_thresh_stopped_data(data, hss):
    above_thresh = data[data['data'] >= threshold]
    standard_fit = GPD.fit(above_thresh['data'], loc=threshold)
    sr = StoppingRule(above_thresh['data'], standard_fit,
                      k=2200, func=StoppingRule.variable_in_the_estimated_return_level,
                      historical_sample_size=hss)
    sd = sr.stopped_data()
    thresh = sr.threshold()
    return sd, thresh


def get_gpd_profiles(data, hss, inf_conf, rl, sd, thresh):
    above_thresh = data[data['data'] >= threshold]
    fit = GPD.fit(above_thresh['data'], loc=threshold)
    standard_fit = GPD_reparametrized(loc=threshold, p=1 / rl, r=fit.isf(1 / rl), shape=fit.shape()).fit_instance(sd)
    ex_fit = GPD_reparametrized(loc=threshold, p=1 / rl, r=standard_fit.isf(1 / rl), shape=standard_fit.shape()).fit_instance(sd.iloc[:-1])
    full_cond = Profiler(distribution=standard_fit, data=sd, inference_confidence=inf_conf,
                         score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                historical_sample_size=hss,
                                                threshold=thresh),
                         name='Conditioning including extreme event', single_profiling_param='r')
    standard_profile = Profiler(distribution=standard_fit, data=sd, name='Standard', inference_confidence=inf_conf, single_profiling_param='r')
    excluding = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                         name='Excluding extreme event', single_profiling_param='r')
    excluding_cond = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                              score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                     threshold=thresh[:-1],
                                                     historical_sample_size=hss),
                              name='Conditioning excluding extreme event', single_profiling_param='r')
    return standard_profile, excluding, full_cond, excluding_cond


def gev_stopping_thresh_stopped_data(data, hss):
    k = 350
    standard_fit = GEV.fit(data)
    sr = StoppingRule(data, standard_fit,
                      k=k, func=StoppingRule.variable_in_the_estimated_return_level,
                      historical_sample_size=hss)
    sd = sr.stopped_data()
    thresh = sr.threshold()
    return sd, thresh


def get_gev_profiles(data, hss, inf_conf, rl, sd, thresh):
    fit = GEV.fit(sd)
    standard_fit = GEV_reparametrized(p=1 / rl, r=fit.isf(1 / rl), loc=fit.loc(), shape=fit.shape()).fit_instance(sd)
    ex_fit = GEV_reparametrized(p=1 / rl, r=fit.isf(1 / rl), loc=fit.loc(), shape=fit.shape()).fit_instance(sd.iloc[:-1])
    full_cond = Profiler(distribution=standard_fit, data=sd, inference_confidence=inf_conf, single_profiling_param='r',
                         score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                historical_sample_size=hss,
                                                threshold=thresh),
                         name='Conditioning including extreme event')
    standard_profile = Profiler(distribution=standard_fit, data=sd, name='Standard', inference_confidence=inf_conf, single_profiling_param='r')
    excluding = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                         name='Excluding extreme event', single_profiling_param='r')
    excluding_cond = Profiler(distribution=standard_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                              score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                     threshold=thresh[:-1],
                                                     historical_sample_size=hss),
                              name='Conditioning excluding extreme event', single_profiling_param='r')
    return standard_profile, excluding, full_cond, excluding_cond
