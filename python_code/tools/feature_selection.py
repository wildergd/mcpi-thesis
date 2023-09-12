#!/usr/bin/env python

import sys
from os import path, makedirs, cpu_count

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from genetic_selection import GeneticSelectionCV
from library.classifiers import get_estimator
import numpy as np
import pandas as pd
import random
from re import match
from sklearn.model_selection import LeaveOneOut
import warnings

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
def cv_arg_check(arg_value):
    try:
        value = int(arg_value)
        if (value > 3):
            return value
    except ValueError:
        pass
    if arg_value == 'loo':
        return arg_value
    raise ArgumentTypeError("param must be an int > 3 or 'loo'")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', help='Path to dataset file (CSV format)', required = True)
parser.add_argument(
    '-m', '--method',
    help='ML method. One of: rlo, rf, svm, sgd, nearcent, adaboost.',
    default = 'nearcent',
    choices = ['rlo', 'rf', 'svm', 'sgd', 'nearcent', 'adaboost']
)
parser.add_argument('-mf', '--max-features', default='20', help='Number of features to be selected. Defaults to 20')
parser.add_argument('-to', '--train-only', default='no', help='Use only the train dataset', choices = ['yes', 'no'])
parser.add_argument(
    '-cv', '--cross-validation',
    help='Cross-Validation method',
    default = 5,
    type=cv_arg_check
)
args = vars(parser.parse_args())

# Set up parameters
csv_file = args['file']
classification_method = args['method']
max_features = int(args['max_features']) if args['max_features'].isdigit() else 20
train_only = args['train_only'] == 'yes'
cv_method = args['cross_validation'] if isinstance(args['cross_validation'], int) else LeaveOneOut()
seed = 90
n_cpus = cpu_count()
n_jobs = max(n_cpus // 3, 1) 

def extract_cv_split(file_path: str):
    split_part = path.split(file_path)[1]
    if match(r'^\d{2}-\d{2}(_av)?$', split_part):
        return split_part 
    return None

if __name__ == '__main__':
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    
    # get some general info
    dataset_name = path.basename(csv_file).split('.')[0]
    
    print('='*100)
    print(f' DATASET: {dataset_name}')
    print(f' MODEL: {classification_method}')
    print(f' MAX FEATURES: {max_features}')
    print(
        ' CV-METHOD: {}'
            .format(
                'k-fold ({})'.format(cv_method) if isinstance(args['cross_validation'], int) else args['cross_validation']
            )
    )
    if train_only:
        train_split = extract_cv_split(path.dirname(csv_file))   
        print(f' SPLIT_SET: {train_split}')
    else:
        print(f' SPLIT_SET: ALL DATA')
    print(f' Parallel Jobs: {n_jobs}')
    print('='*100)
    print()
    
    # read full dataset
    df = pd.read_csv(path.abspath(csv_file))

    # get estimator
    estimator = get_estimator(
        method = classification_method,
        random_state = seed,
        max_features = max_features,
        cv_method = cv_method
    )

    # drop index column    
    df = df.drop('number', axis = 1)
    
    target = df['condition']
    features = df.drop('condition', axis = 1)
    
    # find features using genetic algorithms
    selector = GeneticSelectionCV(
        estimator,
        cv = 5,
        verbose = 1,
        scoring = 'roc_auc',
        max_features = max_features,
        crossover_proba = 0.5,
        mutation_proba = 0.2,
        crossover_independent_proba = 0.5,
        mutation_independent_proba = 0.05,
        caching = True,
        n_jobs = n_jobs
    )
    
    selector = selector.fit(features, target)

    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    output_folder = f'{DATASETS_PATH}/features/all/{max_features}-features/{classification_method}'
    if train_only:
        train_split = extract_cv_split(path.dirname(csv_file))   
        output_folder = f'{DATASETS_PATH}/features/train/{train_split}/{max_features}-features/{classification_method}'
    file_name, _ = path.splitext(path.basename(path.abspath(csv_file)))
    
    if not path.exists(output_folder):
        makedirs(output_folder)
    
    f = open(f'{output_folder}/{file_name}.txt', 'w')
    f.write('\n'.join(features.loc[:, selector.support_].columns.to_list()))
    f.close()

    print()
    print()

