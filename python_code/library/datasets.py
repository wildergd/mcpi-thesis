from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm, kstest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class StandarizeMethod(Enum):
    DEFAULT = 0
    ROBUST = 1
    NONE = 2
    
def get_bad_features(df_train, df_test):
    bad_features = []
    for feature in df_test.columns:
        statistic, p_value = kstest(df_train[feature], df_test[feature])
        if statistic > 0.1 and p_value < 0.05:
            bad_features.append(feature)
            
    return bad_features

def get_important_features(model, df):
    return df.columns.values[np.argmax(model.feature_importances_)]

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
    if method == StandarizeMethod.NONE:
        return values
    
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

#
def split_dataset_random(
    df: pd.DataFrame,
    predict_column: str,
    test_size: float = 0.2,
    random_state: int = None,
    **kwargs
):
    features = df.drop(predict_column, axis = 1)
    target = df[predict_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size = test_size,
        random_state = random_state,
        **kwargs
    )
    
    x_train[predict_column] = y_train
    x_test[predict_column] = y_test
    
    return x_train, x_test, []

#
def split_dataset_adversarial_validation(
    df: pd.DataFrame,
    predict_column: str,
    test_size: float = 0.2,
    random_state: int = None,
    **kwargs
):
    x_train, x_test, _, _ = train_test_split(
        df.drop(predict_column, axis = 1),
        df[predict_column],
        test_size = test_size,
        random_state = random_state,
        **kwargs
    )
    
    tmp_df = pd.concat([x_train, x_test])
    tmp_df['is_test'] = [0] * len(x_train) + [1] * len(x_test)
    tmp_df = tmp_df.sample(frac = 1, random_state = random_state)
    
    features = tmp_df.drop(['is_test'], axis = 1)
    target = tmp_df['is_test']
    
    model = RandomForestClassifier(max_features = 10)
    
    noisy_features = []
    max_iters = 10
    iter = 0
    while True:        
        if iter == max_iters:
            break
        
        model.fit(features, target)
        cv_preds = cross_val_predict(
            model,
            features,
            target,
            cv = 5,
            n_jobs = -1,
            method='predict_proba'
        )
        auc_score = roc_auc_score(y_true = target, y_score = cv_preds[:,1])
        
        print(f'AUC Score: {auc_score}')
        
        if abs(auc_score - 0.5) <= 0.1:
            break
        
        if np.max(model.feature_importances_) - np.mean(model.feature_importances_) < 0.1:
            break
        
        bad_features = get_important_features(model, features)

        print('Drop features: {}'.format(bad_features.join(', ')))

        features = features.drop(bad_features, axis = 1)
        noisy_features.extend(bad_features)
    
        iter += 1
    
    features_new = tmp_df.drop('is_test', axis = 1)
    features_new['proba'] = model.predict_proba(features)[:,1]
    features_new['target'] = target
            
    nrows = features_new.shape[0]
    features_new = features_new.sort_values(by='proba',ascending=False)
    test_data = features_new[:int(nrows * test_size)]
    train_data = features_new[int(nrows * test_size):]
    
    train_data = train_data.drop(['proba', 'target'], axis = 1)
    train_data['condition'] = train_data.index.map(lambda value: int(value.startswith('condition')))
    
    test_data = test_data.drop(['proba', 'target'], axis = 1)
    test_data['condition'] = test_data.index.map(lambda value: int(value.startswith('condition')))
    
    return train_data, test_data, noisy_features

#
def split_dataset(
    df: pd.DataFrame,
    predict_column: str,
    test_size: float = 0.2,
    adversarial_validation: bool = False,
    random_state: int = None,
    **kwargs
):
    if adversarial_validation:
        return split_dataset_adversarial_validation(
            df = df,
            predict_column = predict_column,
            test_size = test_size,
            random_state = random_state,
            **kwargs
        )

    return split_dataset_random(
        df = df,
        predict_column = predict_column,
        test_size = test_size,
        random_state = random_state,
        **kwargs
    )

__all__ = [
    'get_dataframe_summarized',
    'standarize',
    'normalize',
    'split_dataset',
]