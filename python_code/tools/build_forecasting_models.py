#!/usr/bin/env python

import sys
from os import path, makedirs, sep

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
import logging
import numpy as np
import pandas as pd
import random
import warnings

from darts.models.forecasting.baselines import NaiveSeasonal
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.xgboost import XGBModel
from darts.timeseries import TimeSeries

from library.datasets import get_dataframe_summarized, standarize, StandarizeMethod
from library.depresjon import get_measured_days, read_activity_dataset, read_scores_dataset
from library.forecasting import pick_best_forecast_model_for_ts

pd.options.display.precision = 4

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-sm', '--score-metric',
    help='Score metric. One of: mape, mse, rmse.',
    default = 'mape',
    choices = ['mape', 'mse', 'rmse']
)
parser.add_argument(
    '-c', '--criteria',
    help='Score comparission criteria. One of: min, max.',
    default = 'min',
    choices = ['min', 'max']
)
parser.add_argument('-nsm', '--no-save-model', default=False, type=bool, action=BooleanOptionalAction, help='Don\'t save model')
parser.add_argument('-nsr', '--no-save-results', default=False, type=bool, action=BooleanOptionalAction, help='Don\'t save results')
args = vars(parser.parse_args())

# helper functions
def get_epsilon():
    return sys.float_info.epsilon

def transform_dataset(
    df: pd.DataFrame,
    frequency: str = '1T',
    summarize_method: str = 'sum',
    standarize_method: int = StandarizeMethod.DEFAULT,
    remove_outliers: bool = False
) -> pd.DataFrame:
    df_grouped = get_dataframe_summarized(df, frequency, summarize_method)
    df_grouped = df_grouped.rename(columns={'index': 'timestamp'})
    df_grouped['activity'] = standarize(
        df_grouped.activity.to_numpy(),
        method = standarize_method,
        remove_outliers = remove_outliers
    )
    return df_grouped.set_index('timestamp')

# Set up parameters
score_metric = args['score_metric'].upper()
criteria = args['criteria'].lower()
save_model_file = not args['no_save_model']
save_results = not args['no_save_results']
seed = 90

if __name__ == '__main__':
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # output paths
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    RESULTS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'results', 'forecasting'))
    results_output_folder = f'{RESULTS_PATH}/scores'
    models_output_folder = f'{RESULTS_PATH}/models'

    # scores dataset
    df_scores = read_scores_dataset(f'{DATASETS_PATH}/original/scores.csv')

    # models to try
    available_models = [
        NaiveSeasonal(K = 72),
        XGBModel(lags = 24, random_state = seed),
        ExponentialSmoothing(random_state = seed),
        Prophet(floor = 0),
        FFT(nr_freqs_to_keep = 24)
    ]    
    
    # find best model for each subject

    df_models_scores = pd.DataFrame(columns=[
        'number',
        'score_metric',
        'criteria',
        'best_model_mean',
        'score_best_model_mean',
        'model_file_mean',
        'best_model_median',
        'score_best_model_median',
        'model_file_median'
    ])

    for file, row in df_scores.iterrows():
        folder = file.split('_')[0]
        
        print('='*100)
        print(f' DATASET: {file}')
        print('='*100)
        print()

        print(f'Reading activity dataset...')
        days = get_measured_days(df_scores, file)
        df = read_activity_dataset(f'{DATASETS_PATH}/original/{folder}/{file}.csv', days)
        
        # hourly activity sumarized using mean
        hourly_activity_dataset_mean = transform_dataset(
            df,
            frequency = '1H',
            summarize_method = 'mean',
            standarize_method = StandarizeMethod.ROBUST,
            remove_outliers = True
        )
        
        ts_hourly_mean = TimeSeries.from_series(hourly_activity_dataset_mean.activity + get_epsilon())

        print('Finding best forecast model for hourly activity sumarized using "MEAN"...')
        best_model_mean, score_best_model_mean = pick_best_forecast_model_for_ts(
            models = available_models,
            data = ts_hourly_mean,
            cv = 3,
            test_days = 3,
            random_state = seed,
            metric = score_metric,
            criteria = min if criteria == 'min' else max
        )
        
        # hourly activity sumarized using median
        hourly_activity_dataset_median = transform_dataset(
            df,
            frequency = '1H',
            summarize_method = 'median',
            standarize_method = StandarizeMethod.DEFAULT,
            remove_outliers = True
        )
        
        ts_hourly_median = TimeSeries.from_series(hourly_activity_dataset_median.activity + get_epsilon())

        print('Finding best forecast model for hourly activity sumarized using "MEDIAN"...')
        best_model_median, score_best_model_median = pick_best_forecast_model_for_ts(
            models = available_models,
            data = ts_hourly_median,
            cv = 3,
            test_days = 3,
            random_state = seed,
            metric = score_metric,
            criteria = min if criteria == 'min' else max
        )

        # generatings model scores dataset
        df_models_scores.loc[len(df_models_scores)] = {
            'number': file,
            'score_metric': score_metric,
            'criteria': criteria,
            'best_model_mean': best_model_mean.__class__.__name__,
            'score_best_model_mean': score_best_model_mean,
            'model_file_mean': f'{file}__hourly_mean__{best_model_mean.__class__.__name__}__{score_metric}_{criteria}.pkl',
            'best_model_median': best_model_median.__class__.__name__,
            'score_best_model_median': score_best_model_median,
            'model_file_median': f'{file}__hourly_median__{best_model_median.__class__.__name__}__{score_metric}_{criteria}.pkl'
        }        

        # save models
        if save_model_file:
            print('Saving models...')
            if not path.exists(models_output_folder) or not path.isdir(models_output_folder):
                makedirs(models_output_folder)

            best_model_mean.save(path.abspath(f'{models_output_folder}/{file}__hourly_mean__{best_model_mean.__class__.__name__}__{score_metric}_{criteria}.pkl'))
            best_model_median.save(path.abspath(f'{models_output_folder}/{file}__hourly_median__{best_model_median.__class__.__name__}__{score_metric}_{criteria}.pkl'))
            
        print()

    if save_results:
        print()
        print('Saving scores...')
        # write report to file
        if not path.exists(results_output_folder) or not path.isdir(results_output_folder):
            makedirs(results_output_folder)
            
        df_models_scores.to_csv(
            path.abspath(f'{results_output_folder}/scores_forecasting_models__{score_metric}_{criteria}.csv'),
            index = False,
            float_format = '%.4f'
        )

    print()
