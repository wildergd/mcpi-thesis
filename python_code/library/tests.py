from enum import Enum
from typing import Tuple, Union
import numpy as np
from statsmodels.stats.diagnostic import kstest_normal
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')

class StationaryType(Enum):
    NON_STATIONARY = 0
    STATIONARY = 1
    TREND_STATIONARY = 2
    DIFF_STATIONARY = 3

# time series normality test
def check_normality(data: np.array) -> bool:
    _, p_value = kstest_normal(data)
    return p_value >= 0.05

# check if time series is stationary
def check_stationarity(
    data: np.ndarray,
    nlags: Union[str, int] = 'auto'
) -> Tuple[bool, StationaryType]:
    # adf test
    adf_output = adfuller(data)
    adf_pvalue = adf_output[1]
    is_stationary_adf = adf_pvalue <= 0.05
    # kpss test
    kpss_output = kpss(data, nlags = nlags)
    kpss_pvalue = kpss_output[1]
    is_stationary_kpss = kpss_pvalue > 0.05
    if is_stationary_adf and is_stationary_kpss:
        return StationaryType.STATIONARY
    if is_stationary_adf and not is_stationary_kpss:
        return StationaryType.DIFF_STATIONARY
    if not is_stationary_adf and is_stationary_kpss:
        return StationaryType.TREND_STATIONARY
    return StationaryType.NON_STATIONARY
