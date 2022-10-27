from functools import partial

from conditioning_methods import ConditioningMethod
from pykelihood.distributions import GPD, GEV
from pykelihood.profiler import Profiler
from timing_bias import StoppingRule


def get_gpd_profiles(data, hss, inf_conf):
    threshold = 12
    above_thresh = data[data['data'] >= threshold]
    standard_fit = GPD.fit(above_thresh['data'], loc=threshold)
    sr = StoppingRule(above_thresh['data'], standard_fit,
                      k=2200, func=StoppingRule.variable_in_the_estimated_return_level,
                      historical_sample_size=hss)
    sd = sr.stopped_data()
    thresh = sr.threshold()
    standard_fit = GPD.fit(sd, loc=threshold)
    ex_fit = GPD.fit(sd.iloc[:-1], loc=threshold)
    full_cond = Profiler(distribution=standard_fit, data=sd, inference_confidence=inf_conf,
                         score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                historical_sample_size=hss,
                                                threshold=thresh),
                         name='Conditioning including extreme event')
    standard_profile = Profiler(distribution=standard_fit, data=sd, name='Standard', inference_confidence=inf_conf)
    excluding = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                         name='Excluding extreme event')
    excluding_cond = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                              score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                     threshold=thresh[:-1],
                                                     historical_sample_size=hss),
                              name='Conditioning excluding extreme event')
    return standard_profile, excluding, full_cond, excluding_cond


def get_gev_profiles(data, hss, inf_conf):
    k = 350
    annual_maxima = data.groupby('year').agg({'data': 'max'})['data']
    standard_fit = GEV.fit(annual_maxima)
    sr = StoppingRule(annual_maxima, standard_fit,
                      k=k, func=StoppingRule.variable_in_the_estimated_return_level,
                      historical_sample_size=hss)
    sd = sr.stopped_data()
    thresh = sr.threshold()
    standard_fit = GEV.fit(sd)
    ex_fit = GEV.fit(sd.iloc[:-1])
    full_cond = Profiler(distribution=standard_fit, data=sd, inference_confidence=inf_conf,
                         score_function=partial(ConditioningMethod.full_conditioning_including_extreme,
                                                historical_sample_size=hss,
                                                threshold=thresh),
                         name='Conditioning including extreme event')
    standard_profile = Profiler(distribution=standard_fit, data=sd, name='Standard', inference_confidence=inf_conf)
    excluding = Profiler(distribution=ex_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                         name='Excluding extreme event')
    excluding_cond = Profiler(distribution=standard_fit, data=sd.iloc[:-1], inference_confidence=inf_conf,
                              score_function=partial(ConditioningMethod.full_conditioning_excluding_extreme,
                                                     threshold=thresh[:-1],
                                                     historical_sample_size=hss),
                              name='Conditioning excluding extreme event')
    return standard_profile, excluding, full_cond, excluding_cond
