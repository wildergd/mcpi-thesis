import numpy as np
from darts.metrics import mape, rmse, mse
from darts.models.forecasting.baselines import NaiveSeasonal
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.xgboost import XGBModel
from darts.timeseries import TimeSeries
from datetime import timedelta
from typing import Any, Tuple, Union, Literal, Callable

def split_time_series(
    data: TimeSeries,
    days: int = 1
):
    split_index = data.end_time() - timedelta(days = days, hours = -1)
    return data.split_before(split_index)

def extract_metric(scores, metric) -> float:
    return scores[metric]

#
def evaluate_model(
    model: Any,
    data: TimeSeries,
    test_days: int = 1,
    debug: bool = False
) -> Tuple[dict, Any]:
    train, test = split_time_series(data, days = test_days)
    model.fit(train)
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
    model: Prophet,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:
    return model,

#
def tune_naive_seasonal_model(
    model: NaiveSeasonal,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:
    return model,

#
def tune_exponential_smoothing_model(
    model: ExponentialSmoothing,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:
    return model,

#
def tune_fft_model(
    model: FFT,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:
    return model,

#
def tune_xgboost_model(
    model: Prophet,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:
    return model,

# 
def tune_model(
    model: Any,
    data: TimeSeries,
    metric: Literal['RMSE', 'MSE', 'MAPE'] = 'MAPE',
    criteria: Callable = min,
    test_days: int = 1,
    **kwargs
) -> Tuple:    
    if isinstance(model, NaiveSeasonal):
        return tune_naive_seasonal_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, ExponentialSmoothing):
        return tune_exponential_smoothing_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, FFT):
        return tune_fft_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, Prophet):
        return tune_phophet_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, Theta):
        return tune_theta_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
            **kwargs
        )
    
    if isinstance(model, XGBModel):
        return tune_xgboost_model(
            model = model,
            data = data,
            metric = metric,
            criteria = criteria,
            test_days = test_days,
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
    debug: bool = False,
    **kwargs
):
    models = models if isinstance(models, list) else [models]
    scores = [
        extract_metric(
            evaluate_model(
                model,
                data,
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
        **kwargs
    )[0]
    
    best_model_tuned.fit(data)
    
    if debug:
        print('Best model is {} with {} = {}'.format(best_model_tuned.__class__.__name__, metric, best_score))
    
    return best_model_tuned, best_score