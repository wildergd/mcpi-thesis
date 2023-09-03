#!/usr/bin/env python

import sys
from os import path, makedirs

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from genetic_selection import GeneticSelectionCV

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', help='Path to dataset file (CSV format)', required = True)
parser.add_argument('-m', '--method', help='ML method. One of: rlo, ranforest, svm.')
parser.add_argument('-mf', '--max-features', default='20', help='Number of features to be selected. Defaults to 20')
args = vars(parser.parse_args())

# Set up parameters
csv_file = args['file']
classification_method = args['method']
max_feaures = int(args['max_features']) if args['max_features'].isdigit() else 20
seed = 90

def get_estimator(
    method: str,
    random_state: int = None,
    max_features: int = 20
):
    if method == 'rlo':
        return LogisticRegressionCV(
            solver='lbfgs',
            cv = 5,
            random_state = random_state
        )

    if method == 'rf':
        return RandomForestClassifier(
            bootstrap = False,
            max_features = max_feaures,
            n_jobs = -1,
            random_state = random_state
        )
        
    return None

if __name__ == '__main__':
    if csv_file is None or classification_method is None:
        print()
        parser.print_usage()
        print()
        quit()

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    
    # read full dataset
    df = pd.read_csv(path.abspath(csv_file))

    # get estimator
    estimator = get_estimator(
        method = classification_method,
        random_state = seed 
    )
    
    df = df.drop('number', axis = 1)

    target = df['condition']
    features = df.drop('condition', axis = 1)
    
    # find features using genetic algorithms
    selector = GeneticSelectionCV(
        estimator,
        cv = 5,
        verbose = 1,
        scoring = 'roc_auc',
        max_features = max_feaures,
        crossover_proba = 0.5,
        mutation_proba = 0.2,
        crossover_independent_proba = 0.5,
        mutation_independent_proba = 0.05,
        caching = True,
        n_jobs = -1
    )
    
    selector = selector.fit(features, target)

    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    output_folder = f'{DATASETS_PATH}/features/{max_feaures}-features/{classification_method}'
    file_name, _ = path.splitext(path.basename(path.abspath(csv_file)))
    
    if not path.exists(output_folder):
        makedirs(output_folder)
    
    f = open(f'{output_folder}/{file_name}.txt', 'w')
    f.write('\n'.join(features.loc[:, selector.support_].columns.to_list()))
    f.close()


