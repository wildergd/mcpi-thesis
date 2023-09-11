#!/usr/bin/env python

import sys
from os import path, makedirs

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings

warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
from library.datasets import get_dataframe_summarized, standarize, StandarizeMethod
from library.depresjon import get_measured_days, read_activity_dataset, read_scores_dataset
from library.timeseries import extract_ts_features, ComputeFeatures

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--frequency', default='T', help='time frequency')
parser.add_argument('-sm', '--summarize-method', default='sum', help='sumarize method. One of: mean, median, sum, max, min')
parser.add_argument('-st', '--standarize-method', default='zscore', help='standarize method. One of: zscore, zscore-robust')
parser.add_argument('-sto', '--standarize-remove-outliers', default='no', help='standarize method should remove ouliers. One of: yes, no')
parser.add_argument('-cf', '--compute-features', default='all', help='computed fetures. One of: all, minimal')
args = vars(parser.parse_args())

# Set up parameters
frequency = args['frequency']
summarize_method = args['summarize_method']
standarize_method = args['standarize_method']
remove_outliers = args['standarize_remove_outliers']
compute_features = ComputeFeatures.MINIMAL if args['compute_features'].lower() == 'minimal' else ComputeFeatures.ALL
seed = 123

if __name__ == '__main__':
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    output_folder = f'{DATASETS_PATH}/transformed/classification'

    # scores dataset
    df_scores = read_scores_dataset(f'{DATASETS_PATH}/original/scores.csv')

    # predict column
    y = df_scores.index.map(lambda value: int(value.startswith('condition')))
    
    # read and process ativity datasets
    features_dataset = pd.DataFrame()
    for file, row in df_scores.iterrows():
        folder = file.split('_')[0]
        days = get_measured_days(df_scores, file)
        df = read_activity_dataset(f'{DATASETS_PATH}/original/{folder}/{file}.csv', days)
        df_grouped = get_dataframe_summarized(df, frequency, summarize_method)
        df_grouped = df_grouped.rename(columns={'index': 'timestamp'})
        df_grouped['id'] = file
        df_grouped['activity'] = standarize(
            df_grouped.activity.to_numpy(),
            method = StandarizeMethod.ROBUST if standarize_method == 'zscore-robust' else StandarizeMethod.DEFAULT,
            remove_outliers = True if remove_outliers == 'yes' else False
        )
        
        # extract features
        extracted_features = extract_ts_features(
            df_grouped,
            compute_features,
            column_id = 'id',
            column_sort = 'timestamp',
            column_value = 'activity',
        )
                
        features_dataset = pd.concat([
            features_dataset,
            extracted_features,
        ])

    # add condition column
    features_dataset['condition'] = features_dataset.index.map(lambda value: int(value.startswith('condition')))
    
    # remove columns with unique value
    features_dataset = features_dataset[[c for c in list(features_dataset) if len(features_dataset[c].unique()) > 1]]
    
    if not path.exists(output_folder):
        makedirs(output_folder)
    
    features_dataset.to_csv(
        f'{output_folder}/features_{args["compute_features"].lower()}_{frequency}_{summarize_method}_standarized_{standarize_method}_outliers_{remove_outliers}.csv',
        index=True,
        index_label='number'
    )