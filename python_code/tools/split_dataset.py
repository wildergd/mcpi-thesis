#!/usr/bin/env python

import sys
from os import path, makedirs

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import pandas as pd
from library.datasets import split_dataset, split_dataset_adversarial_validation

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', help='Path to dataset file (CSV format)', required = True)
parser.add_argument('-s', '--size', default='80:20', help='split ration (train:test). Defaults to 80:20')
parser.add_argument('-av', '--use-adversarial-validation', default='no', help='Use adversarial validation (yes|no). Default: no')
args = vars(parser.parse_args())

def get_test_size(value: str) -> float:
    _, test = [ratio.strip() for ratio in value.split(':')]
    try:
        float_val = float(test)
    except ValueError:
        return 0.2
    else:
        return float_val if float_val < 1 else float_val / 100

# Set up parameters
csv_file = args['file']
use_adversarial_validation_method = args['use_adversarial_validation']
split_test_size = get_test_size(args['size'])
seed = 1100

if __name__ == '__main__':
    if csv_file is None:
        print()
        parser.print_usage()
        print()
        quit()
        
    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    
    # read full dataset
    df = pd.read_csv(path.abspath(csv_file), index_col='number')

    # split
    train, test, noisy_features = split_dataset(
        df,
        predict_column = 'condition',
        test_size = split_test_size,
        adversarial_validation = use_adversarial_validation_method == 'yes',
        random_state = seed
    )
    
    # export datasets
    test_size = int(split_test_size * 100)
    train_size = 100 - test_size
    split_folder_info = f'{train_size}-{test_size}'
    adversatial_validation_info = '_av' if use_adversarial_validation_method == 'yes' else ''
    
    train_output_path = f'{DATASETS_PATH}/transformed/classification/train/{split_folder_info}{adversatial_validation_info}'
    test_output_path = f'{DATASETS_PATH}/transformed/classification/test/{split_folder_info}{adversatial_validation_info}'
    
    if not path.exists(train_output_path):
        makedirs(train_output_path)
    
    if not path.exists(test_output_path):
        makedirs(test_output_path)
    
    train.to_csv(
        f'{train_output_path}/{path.basename(path.abspath(csv_file))}',
        index=True,
        index_label='number'
    )
    test.to_csv(
        f'{test_output_path}/{path.basename(path.abspath(csv_file))}',
        index=True,
        index_label='number'
    )
