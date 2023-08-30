from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

def extract_ts_features(df, **kwargs):
    extracted_features = extract_features(df, **kwargs)
    impute(extracted_features)
    return extracted_features