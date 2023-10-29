import numpy as np
from darts.metrics import mape, rmse, mse
from darts.models.forecasting.baselines import NaiveSeasonal
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.xgboost import XGBModel
from darts.timeseries import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode
from datetime import timedelta
from sklearn.model_selection import ParameterGrid
from typing import Any, Tuple, Union, Literal, Callable

#
def find_best_num_test_days_for_cv(
    data: TimeSeries,
    n_splits: int = 3,
    test_days: int = 1
):
    if n_splits == 1 or test_days == 1:
        return test_days

    n_days = (data.end_time() - timedelta(days=test_days, hours = -1) - data.start_time()).days
    k_fold_size = n_days // n_splits
    
    if k_fold_size >= test_days:
        return test_days
    
    return find_best_num_test_days_for_cv(
        data = data,
        n_splits = n_splits,
        test_days = test_days - 1
    )

#
def split_time_series(
    data: TimeSeries,
    days: int = 1
):
    split_index = data.end_time() - timedelta(days = days, hours = -1)
    return data.split_before(split_index)

#
def split_time_series_bcv(
    data: TimeSeries,
    n_splits: int = 3,
    test_days: int = 1
):
    if n_splits == 1:
        return split_time_series(data, test_days)
    
    test_days_for_cv = find_best_num_test_days_for_cv(
        data = data,
        n_splits = n_splits,
        test_days = test_days
    )
    
    n_days = (data.end_time() - timedelta(days=test_days_for_cv, hours = -1) - data.start_time()).days
    k_fold_size = n_days // n_splits
    
    ts = data
    for _ in range(n_splits):        
        split_index = ts.start_time() + timedelta(days = k_fold_size)
        train, rest = ts.split_before(split_index)
        test, _ = rest.split_after(rest.start_time() + timedelta(days = test_days_for_cv, hours = -1))
        ts = rest
        yield train, test

#    
def extract_metric(scores, metric) -> float:
    return scores[metric]

# 
def evaluate_model_bcv(
    model: Any,
    data: TimeSeries,
    n_splits: int = 3,
    test_days: int = 1,
    debug: bool = False,
    **kwargs
) -> Tuple[dict, Any]:
    score_rmse = []
    score_mse = []
    score_mape = []
    for train, test in split_time_series_bcv(data, n_splits = n_splits, test_days = test_days):
        model.fit(train, **kwargs)
        forecast = model.predict(len(test))
        score_rmse.append(rmse(test, forecast))
        score_mse.append(mse(test, forecast))
        score_mape.append(mape(test, forecast))

    scores = {
        'RMSE': np.mean(score_rmse),
        'MSE': np.mean(score_mse),
        'MAPE': np.mean(score_mape)
    }

    if debug:
        print('Model {} obtained scores: {}'.format(model.__class__.__name__, scores))
    return scores, model

#
def evaluate_model(
    model: Any,
    data: TimeSeries,
    test_days: int = 1,
    debug: bool = False,
    cv: int = None,
    **kwargs
) -> Tuple[dict, Any]:
    if isinstance(cv, int):
        return evaluate_model_bcv(
            model = model,
            data = data,
            n_splits = cv,
            test_days = test_days,
            debug = debug,
            **kwargs
        )
    
    train, test = split_time_series(data, days = test_days)
    model.fit(train, **kwargs)
    forecast = model.predict(len(test))
    
    scores = {
        'RMSE': rmse(test, forecast),
        'MSE': mse(test, forecast),
        'MAPE': mape(test, forecast)
    }

    if debug:
        print('Model {} obtained scores: {}'.format(model.__class__.__name__, scores))
    return scores, model

#
def tune_theta_model(
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    thetas = 2 - np.linspace(-10, 10, 50)
    best_score = float('inf') if criteria == min else float('-inf')
    best_theta = 0

    for theta in thetas:
        model = Theta(theta, **kwargs)
        
        scores, _ = evaluate_model(
            model = model,
            data = data,
            cv = cv,
            test_days = test_days
        )

        model_score = extract_metric(scores, metric)
        if criteria(model_score, best_score) == model_score:
            best_score = model_score
            best_theta = theta
            
    best_theta_model = Theta(best_theta, **kwargs)
    
    return best_theta_model, best_theta

#
def tune_phophet_model(
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    n_days = (data.end_time() - timedelta(hours = -1) - data.start_time()).days
    params_grid = {
        'seasonality_mode': ('multiplicative','additive'),
        'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
        'n_changepoints': np.arange(1, n_days + 1) * 24
    }
    grid = ParameterGrid(params_grid)
    best_score = float('inf') if criteria == min else float('-inf')
    best_params = {}
    for params in grid:
        prophet_model = Prophet(
            seasonality_mode = params['seasonality_mode'],
            changepoint_prior_scale = params['changepoint_prior_scale'],
            n_changepoints = params['n_changepoints'],
            daily_seasonality = True,
            weekly_seasonality = True,
            yearly_seasonality = False,
            interval_width = 0.95,
            **kwargs
        )
        
        scores, _ = evaluate_model(
            model = prophet_model,
            data = data,
            cv = cv,
            test_days = test_days,
        )
        
        model_score = extract_metric(scores, metric)
        if criteria(model_score, best_score) == model_score:
            best_score = model_score
            best_params = params

    return Prophet(**best_params, **kwargs), best_params

#
def tune_naive_seasonal_model(
    model: NaiveSeasonal,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    n_days = (data.end_time() - timedelta(hours = -1) - data.start_time()).days
    best_score = float('inf') if criteria == min else float('-inf')
    best_k = 24

    for k in np.arange(1, min(n_days, 7) + 1) * 24:
        model = NaiveSeasonal(K = k)

        scores, _ = evaluate_model(
            model = model,
            data = data,
            cv = cv,
            test_days = test_days
        )

        model_score = extract_metric(scores, metric)
        if criteria(model_score, best_score) == model_score:
            best_score = model_score
            best_k = k
            
    best_naive_seasonal_model = NaiveSeasonal(K = best_k)
    
    return best_naive_seasonal_model, best_k

#
def tune_exponential_smoothing_model(
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    params_grid = {
        'trend': (ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE),
        'damped': [True, False],
        'seasonality': (SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE),
        'smoothing_level': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'smoothing_trend': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'smoothing_seasonal': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    grid = ParameterGrid(params_grid)
    best_score = float('inf') if criteria == min else float('-inf')
    best_params = {}
    for params in grid:
        try:
            model = ExponentialSmoothing(
                trend = params['trend'],
                damped = params['damped'] if params['trend'] != ModelMode.NONE else False,
                seasonal = params['seasonality'],
                smoothing_level = params['smoothing_level'],
                smoothing_trend = params['smoothing_trend'] if params['trend'] != ModelMode.NONE else None,
                smoothing_seasonal = params['smoothing_seasonal'] if params['seasonality'] != SeasonalityMode.NONE else None,
                **kwargs
            )
            
            scores, _ = evaluate_model(
                model = model,
                data = data,
                cv = cv,
                test_days = test_days,
            )
            
            model_score = extract_metric(scores, metric)
            if criteria(model_score, best_score) == model_score:
                best_score = model_score
                best_params = params
        except:
            continue
        
    best_model = ExponentialSmoothing(
        trend = best_params['trend'],
        damped = best_params['damped'] if best_params['trend'] != ModelMode.NONE else False,
        seasonal = best_params['seasonality'],
        smoothing_level = best_params['smoothing_level'],
        smoothing_trend = best_params['smoothing_trend'] if best_params['trend'] != ModelMode.NONE else None,
        smoothing_seasonal = best_params['smoothing_seasonal'] if best_params['seasonality'] != SeasonalityMode.NONE else None,
        **kwargs
    )
    
    return best_model, best_params

#
def tune_fft_model(
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    n_days = (data.end_time() - timedelta(hours = -1) - data.start_time()).days
    freqs_to_keep = np.arange(1, n_days * 24)
    best_score = float('inf') if criteria == min else float('-inf')
    best_freq_keep = 0

    for freqs in freqs_to_keep:
        model = FFT(
            nr_freqs_to_keep = freqs,
            **kwargs
        )
        
        scores, _ = evaluate_model(
            model = model,
            data = data,
            cv = cv,
            test_days = test_days
        )

        model_score = extract_metric(scores, metric)
        if criteria(model_score, best_score) == model_score:
            best_score = model_score
            best_freq_keep = freqs
            
    best_fft_model = FFT(best_freq_keep, **kwargs)
    
    return best_fft_model, best_freq_keep

#
def tune_xgboost_model(
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    **kwargs
) -> Tuple:
    params_grid = {
        'lags': list(range(24, 192, 24)),
    }
    grid = ParameterGrid(params_grid)
    best_score = float('inf') if criteria == min else float('-inf')
    best_params = {}
    for params in grid:
        try:
            model = XGBModel(
                **params,
                **kwargs
            )
            
            scores, _ = evaluate_model(
                model = model,
                data = data,
                cv = cv,
                test_days = test_days
            )
            
            model_score = extract_metric(scores, metric)
            if criteria(model_score, best_score) == model_score:
                best_score = model_score
                best_params = params
        except:
            continue

    best_model = XGBModel(**best_params, **kwargs)
    
    return best_model, best_params

# 
def tune_model(
    model: Any,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    random_state: int = None,
    **kwargs
) -> Tuple:    
    if isinstance(model, NaiveSeasonal):
        return tune_naive_seasonal_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, ExponentialSmoothing):
        return tune_exponential_smoothing_model(
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            random_state = random_state,
            **kwargs
        )
    
    if isinstance(model, FFT):
        return tune_fft_model(
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, Prophet):
        return tune_phophet_model(
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, Theta):
        return tune_theta_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, XGBModel):
        return tune_xgboost_model(
            data = data,
            metric = metric,
            criteria = criteria,
            cv = cv,
            test_days = test_days,
            random_state = random_state,
            **kwargs
        )
    
    return model, 

# 
def pick_best_forecast_model_for_ts(
    models: Union[Any, list[Any]],
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    cv: int = None,
    random_state: int = None,
    debug: bool = False,
    **kwargs
):
    models = models if isinstance(models, list) else [models]
    scores = [
        extract_metric(
            evaluate_model(
                model,
                data,
                cv = cv,
                test_days = test_days,
                debug = debug
            )[0],
            metric
        )
        for model in models
    ]
    best_score = criteria(scores)
    best_score_index = scores.index(best_score)
    best_model = models[best_score_index]
    
    best_model_tuned = tune_model(
        best_model,
        data,
        metric = metric,
        criteria = criteria,
        test_days = test_days,
        random_state = random_state,
        **kwargs
    )[0]
    
    best_model_tuned.fit(data)
    
    if debug:
        print('Best model is {} with {} = {}'.format(best_model_tuned.__class__.__name__, metric, best_score))
    
    return best_model_tuned, best_score