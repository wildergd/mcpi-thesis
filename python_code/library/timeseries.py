from enum import Enum
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

class ComputeFeatures(Enum):
    ALL = 0
    MINIMAL = 1
    
# extract features
def extract_ts_features(df, compute_features, **kwargs):
    extracted_features = extract_features(
        df,
        default_fc_parameters = None if compute_features == ComputeFeatures.ALL else MinimalFCParameters(),
        **kwargs
    )
    impute(extracted_features)
    return extracted_features
