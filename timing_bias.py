import warnings
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pykelihood.distributions import Distribution

warnings.filterwarnings('ignore')


class StoppingRule(object):
    def __init__(self, data: pd.Series,
                 distribution: "Distribution",
                 k: int,
                 historical_sample_size: int,
                 func: Callable[[pd.Series, int, "Distribution", int], int]):
        """

        :param data: database that is the base of the analysis once we stop
        :param historical_sample_size: number of data points necessary to provide a reliable first estimate
        :param func: stopping rule (it can be fixed or variable as the static methods detailed in the class)
        """
        self.data = data
        self.k = k
        self.distribution = distribution
        self.stopping_rule = func
        self.historical_sample_size = historical_sample_size
        self.c, self.N = self.stopping_rule(self.data, self.historical_sample_size,
                                            self.distribution, self.k)

    def stopped_data(self):
        return self.data.iloc[:self.N]

    def threshold(self):
        return self.c

    def last_index(self):
        return self.N

    @staticmethod
    def fixed_to_sample_middle_value(data: pd.Series, historical_sample_size: int,
                                     distribution: "Distribution", k: int):
        max = data.iloc[historical_sample_size:].max()
        min = data.iloc[historical_sample_size:].min()
        c = (max - min) / 2
        first_index_above_threshold = np.argmax(data.iloc[historical_sample_size:] >= c)
        if first_index_above_threshold == 0 and sum(data.iloc[historical_sample_size:] >= c) == 0:
            N = len(data)
        else:
            N = historical_sample_size + first_index_above_threshold + 1
        number_of_tests_threshold = N - historical_sample_size
        return [c] * number_of_tests_threshold, N

    @staticmethod
    def fixed_to_k(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        c = k
        first_index_above_threshold = np.argmax(data.iloc[historical_sample_size:] >= c)
        if first_index_above_threshold == 0 and sum(data.iloc[historical_sample_size:] >= c) == 0:
            N = len(data)
        else:
            N = historical_sample_size + first_index_above_threshold + 1
        number_of_tests_threshold = N - historical_sample_size
        return [c] * number_of_tests_threshold, N

    @staticmethod
    def variable_in_the_estimated_return_level(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        j = historical_sample_size
        fit = distribution.fit_instance(data.iloc[:j])
        return_level_estimate = fit.isf(1 / k)
        return_level_estimates = [float(return_level_estimate)] * len(data.iloc[:j + 1])
        data_stopped = data.iloc[:j + 1]
        j += 1
        while data_stopped.iloc[-1] < return_level_estimate and len(data_stopped) < len(data):
            fit = distribution.fit_instance(data_stopped)
            return_level_estimate = fit.isf(1 / k)
            return_level_estimates.append(float(return_level_estimate))
            data_stopped = data.iloc[:j + 1]
            j += 1
        N = j
        return return_level_estimates, N

    @staticmethod
    def fixed_to_1981_2010_average(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        j = historical_sample_size
        deviation = 2.2
        mean = data.loc[1981:2010].mean()
        thresh = [float(mean + deviation)] * len(data.iloc[:j + 1])
        data_stopped = data.iloc[:j + 1]
        j += 1
        while data_stopped.iloc[-1] < mean + deviation and len(data_stopped) < len(data):
            thresh.append(float(mean + deviation))
            data_stopped = data.iloc[:j + 1]
            j += 1
        N = j
        return thresh, N

    @staticmethod
    def variable_in_the_estimated_record(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        historical_sample_yearsize = historical_sample_size
        year = np.unique(data._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        data_stopped = data.loc[:year]
        thresh = np.max(data_stopped) + 2
        data_stopped = data.loc[:year + 1]
        year += 1
        thresholds = [float(thresh)] * len(data_stopped)
        while data_stopped.loc[year] < thresh and len(data_stopped) < len(data):
            thresh = np.max(data_stopped) + 2
            data_stopped = data.loc[:year + 1]
            year += 1
            thresholds.append(float(thresh))
        N = year
        return thresholds, N

    @staticmethod
    def variable_in_the_estimated_rl_with_trend_in_r(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        historical_sample_yearsize = historical_sample_size
        year = np.unique(data._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        rls = pd.DataFrame(distribution.isf(1/k), index=distribution.r().index)
        thresh = float(rls.loc[year])
        data_stopped = data.loc[:year + 1]
        year += 1
        thresholds = [thresh] * len(data_stopped)
        while data_stopped.loc[year] < thresh and len(data_stopped) < len(data):
            thresh = float(rls.loc[year])
            data_stopped = data.loc[:year + 1]
            year += 1
            thresholds.append(thresh)
        N = year
        return thresholds, N

    @staticmethod
    def variable_in_the_estimated_rl_with_trend_in_loc(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        historical_sample_yearsize = historical_sample_size
        year = np.unique(data._get_label_or_level_values('YEAR'))[historical_sample_yearsize - 1]
        rls = pd.DataFrame(distribution.isf(1/k), index=distribution.loc().index)
        thresh = float(rls.loc[year])
        data_stopped = data.loc[:year + 1]
        year += 1
        thresholds = [thresh] * len(data_stopped)
        while data_stopped.loc[year] < thresh and len(data_stopped) < len(data):
            thresh = float(rls.loc[year])
            data_stopped = data.loc[:year + 1]
            year += 1
            thresholds.append(thresh)
        N = year
        return thresholds, N