from enum import Enum
from typing import Tuple, Union
import numpy as np
from statsmodels.stats.diagnostic import kstest_normal
from statsmodels.tsa.stattools import adfuller, kpss

class StationaryType(Enum):
    NON_STATIONARY = 0
    STATIONARY = 1
    TREND_STATIONARY = 2
    DIFF_STATIONARY = 3

# time series normality test
def check_normality(data: np.array) -> bool:
    t_test, p_value = kstest_normal(data)
    return False if p_value < 0.05 else True

# check if time series is stationary
def check_stationarity(
    data: np.ndarray,
    nlags: Union[str, int] = 'auto'
) -> Tuple[bool, StationaryType]:
    # adf test
    adf_output = adfuller(data)
    adf_test_stat, adf_pvalue, adf_critical_values = adf_output[0], adf_output[1], adf_output[-2]
    is_stationary_adf = adf_pvalue <= 0.05 # and adf_test_stat < adf_critical_values['5%']
    # kpss test
    kpss_output = kpss(data, nlags = nlags)
    kpss_test_stat, kpss_pvalue, kpss_critical_values = kpss_output[0], kpss_output[1], kpss_output[-1]
    is_stationary_kpss = kpss_pvalue > 0.05 # and kpss_test_stat < kpss_critical_values['5%']
    
    if is_stationary_adf and is_stationary_kpss:
        return True, StationaryType.STATIONARY
    if is_stationary_adf and not is_stationary_kpss:
        return True, StationaryType.DIFF_STATIONARY
    if not is_stationary_adf and is_stationary_kpss:
        return True, StationaryType.TREND_STATIONARY
    return False, StationaryType.NON_STATIONARY
