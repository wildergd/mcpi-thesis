from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm, gzscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import NewType

class StandarizeMethod(Enum):
    DEFAULT = 0
    ROBUST = 1
    
#
def get_dataframe_summarized(
    df: pd.DataFrame,
    group_frequency: str,
    summarize_method: str
) -> pd.DataFrame:
    return getattr(df.groupby(pd.Grouper(freq=group_frequency)).activity, summarize_method)().reset_index()

#
def standarize(
    values: np.ndarray,
    method: int = StandarizeMethod.DEFAULT,
    degree: float = 3,
    remove_outliers: bool = False
) -> np.ndarray:
    if method == StandarizeMethod.ROBUST:
        s = norm.ppf(0.75)
        numerator = s * (values - np.median(values))
        MAD = np.median(np.abs(values - np.median(values)))
        transformed = numerator/MAD
    else:
        transformed = StandardScaler().fit_transform(values.reshape((len(values), 1))).reshape((len(values),))
        
    if remove_outliers:
        transformed[transformed > degree] = degree
        transformed[transformed < -degree] = -degree

    return transformed
#
def normalize(values: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(values.reshape((len(values), 1))).reshape((len(values),))
