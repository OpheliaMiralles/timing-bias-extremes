import numpy as np

from pykelihood.distributions import Normal
from pykelihood.metrics import opposite_log_likelihood


def shape_stability_conditioning(distribution, data):
    return opposite_log_likelihood(distribution, data) - np.sum(Normal(0., 0.25).logpdf(distribution.shape()))


class ConditioningMethod(object):
    @staticmethod
    def no_conditioning(distribution, data):
        return opposite_log_likelihood(distribution, data)

    @staticmethod
    def excluding_extreme(distribution, data, length_extreme_event: int = 1):
        return ConditioningMethod.no_conditioning(
            distribution, data
        ) + np.sum(distribution.logpdf(data)[-length_extreme_event:])

    @staticmethod
    def conditioning_excluding_extreme(
            distribution, data, threshold=None, historical_sample_size: int = 0, length_extreme_event: int = 1
    ):
        # when the inputs are the full dataset and threshold
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return ConditioningMethod.no_conditioning(distribution, data) + np.sum(distribution.logcdf(threshold)[historical_sample_size:-length_extreme_event]) + np.sum(
            distribution.logpdf(data)[-length_extreme_event:])

    @staticmethod
    def full_conditioning_including_extreme(
            distribution, data, threshold=None, historical_sample_size: int = 0, length_extreme_event: int = 1,
    ):
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return (ConditioningMethod.no_conditioning(distribution, data) + np.log(np.sum(distribution.sf(threshold)[-length_extreme_event:])) + np.sum(
            distribution.logcdf(threshold)[historical_sample_size:-length_extreme_event]))

    @staticmethod
    def full_conditioning_excluding_extreme(
            distribution, data, threshold=None, historical_sample_size: int = 0, length_extreme_event: int = 1
    ):
        # when the input are the dataset and threshold stopped before the extreme event
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return (ConditioningMethod.no_conditioning(distribution, data) + np.sum(distribution.logcdf(threshold)[historical_sample_size:]))

    ### BIVARIATE ###
    @staticmethod
    def full_conditioning_using_correlated_distribution(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        elif len(threshold) < len(stopping_data):
            t = threshold.copy()
            excess = len(stopping_data) - len(threshold)
            t.extend([0] * (len(stopping_data) - len(threshold)))
        else:
            excess = 0
            t = threshold.copy()
        d1_ind, d2_ind = ConditioningMethod.common_indices(data, stopping_data)
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[d1_ind], correlated_margin.cdf(stopping_data)[d2_ind]))), log=True)) - np.sum(distribution.logpdf(data)) - np.sum(
            correlated_margin.logpdf(stopping_data)) + np.sum(correlated_margin.logcdf(t)[historical_sample_size:-length_extreme_event - excess]) + np.sum(
            correlated_margin.logsf(t)[-length_extreme_event - excess:- excess])

    @staticmethod
    def full_conditioning_excluding_extreme_using_correlated_distribution(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        # when the inputs are the dataset and threshold stopped before the extreme event
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        d1_ind, d2_ind = ConditioningMethod.common_indices(data, stopping_data)
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[d1_ind],
                                                             correlated_margin.cdf(stopping_data)[d2_ind]))), log=True)) - np.sum(distribution.logpdf(data)) - np.sum(
            correlated_margin.logpdf(stopping_data)) + np.sum(correlated_margin.logcdf(threshold)[historical_sample_size:])

    @staticmethod
    def conditioning_excluding_extreme_using_correlated_distribution(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        # when the inputs are the full dataset and threshold
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[:-length_extreme_event],
                                                             correlated_margin.cdf(stopping_data)[:-length_extreme_event]))), log=True)) - np.sum(
            distribution.logpdf(data)[:-length_extreme_event]) - np.sum(correlated_margin.logpdf(stopping_data)[:-length_extreme_event]) + np.sum(
            correlated_margin.logcdf(threshold[historical_sample_size:-length_extreme_event]))

    @staticmethod
    def excluding_extreme_using_correlated_distribution(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        d1_ind, d2_ind = ConditioningMethod.common_indices(data, stopping_data)
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[d1_ind][:-length_extreme_event],
                                                             correlated_margin.cdf(stopping_data)[d2_ind][:-length_extreme_event]))), log=True)) - np.sum(
            distribution.logpdf(data)[:-length_extreme_event]) - np.sum(correlated_margin.logpdf(stopping_data)[:-length_extreme_event])

    @staticmethod
    def including_extreme_using_correlated_distribution(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        d1_ind, d2_ind = ConditioningMethod.common_indices(data, stopping_data)
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[d1_ind],
                                                             correlated_margin.cdf(stopping_data)[d2_ind]))), log=True)) - np.sum(distribution.logpdf(data)) - np.sum(
            correlated_margin.logpdf(stopping_data))

    @staticmethod
    def excluding_extreme_using_correlated_distribution_spec(
            distribution,
            data,
            joint_structure: object,
            stopping_data,
            correlated_margin,
            threshold=None,
            historical_sample_size: int = 0,
            length_extreme_event: int = 1
    ):
        d1_ind, d2_ind = ConditioningMethod.common_indices(data, stopping_data)
        return -np.sum(joint_structure.pdf(np.array(list(zip(distribution.cdf(data)[d1_ind],
                                                             correlated_margin.cdf(stopping_data)[d2_ind]))), log=True)) - np.sum(distribution.logpdf(data)) - np.sum(
            correlated_margin.logpdf(stopping_data))

    @staticmethod
    def common_indices(d1, d2):
        common_indices = np.intersect1d(list(d1.index.unique()), list(d2.index.unique()))
        d1_ind = np.where(d1.index.isin(common_indices))
        d2_ind = np.where(d2.index.isin(common_indices))
        return d1_ind, d2_ind

